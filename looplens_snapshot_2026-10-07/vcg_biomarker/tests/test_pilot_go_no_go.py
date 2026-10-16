from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.marker_freeze import freeze_marker_inventory
from vcg_biomarker.mimic.pilot_go_no_go import (
    _pilot_performance_summary,
    _runtime_summary,
    build_pilot_go_no_go_report,
)
from vcg_biomarker.mimic.presets import get_analysis_preset


def _write_manifest(path, output, *, row_counts=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("artifact", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "output_paths": [str(output.resolve())],
                "row_counts": row_counts or {"rows": 1},
            }
        ),
        encoding="utf-8",
    )


def test_runtime_summary_does_not_double_count_profiled_queries(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    events = (
        {
            "kind": "stage",
            "event": "end",
            "stage": "build-evidence",
            "duration_seconds": 12.0,
            "rss_gb": 3.0,
            "available_gb": 20.0,
        },
        {
            "kind": "query",
            "event": "end",
            "stage": "build-evidence",
            "label": "evidence:labs.parquet",
            "duration_seconds": 9.0,
            "rows": 100,
            "rss_before_gb": 2.0,
            "rss_after_gb": 3.0,
        },
    )
    path.write_text(
        "\n".join(json.dumps(item) for item in events) + "\n",
        encoding="utf-8",
    )

    summary = _runtime_summary(path)

    assert summary["stage_duration_seconds"] == {"build-evidence": 12.0}
    assert summary["query_duration_seconds"] == 9.0
    assert summary["query_count"] == 1
    assert summary["slowest_queries"][0]["rows"] == 100


def test_scaling_ratio_is_reported_without_an_arbitrary_runtime_cutoff() -> None:
    stages = {
        "extract-features-pilot-1000": 10.0,
        "validate-pilot-1000": 2.0,
        # Deliberately much slower per record. This is visible for human
        # review; the automatic gates only assert that it was measured.
        "extract-features-pilot-10000": 10_000.0,
        "validate-pilot-10000": 3.0,
    }
    policies = {}
    available = {}
    queries = {}
    for stage in stages:
        policies[stage] = {
            "policy_version": "test",
            "workload_class": (
                "waveform"
                if stage.startswith("extract-features")
                else "analysis"
            ),
            "requested_memory_gb": 12.0,
            "duckdb_memory_limit_gb": 8.0,
            "non_duckdb_headroom_gb": 4.0,
            "critical_available_gb": 3.0,
        }
        available[stage] = 10.0
        if stage.startswith("validate"):
            queries[stage] = 1
    runtime = {
        "stage_duration_seconds": stages,
        "stage_nonreuse_reported_attempt_seconds": {key: [value] for key, value in stages.items()},
        "stage_resource_policies": policies,
        "stage_minimum_available_ram_gb": available,
        "completed_query_count_by_stage": queries,
        "record_count": 10,
        "stage_failure_count": 0,
        "query_missing_row_count": 0,
        "query_failure_count": 0,
    }
    details = {
        f"pilot_{size}": {
            "features": {
                "row_counts": {
                    "selected_ecgs": size,
                    "feature_count": 320,
                }
            }
        }
        for size in (1_000, 10_000)
    }

    report, gates = _pilot_performance_summary(runtime, details)

    assert report["feature_seconds_per_record_inflation_10k_vs_1k"] is None
    assert report["unmatched_pilot_feature_throughput_ratio_10k_vs_1k"] == 100.0
    assert not report["matched_feature_scaling_established"]
    assert gates["pilot_feature_throughput_ratio_measured"] is True
    assert all(gates.values())


def test_go_no_go_holds_full_run_and_requires_all_gates(tmp_path) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    run_id = "pilot-review"
    run_root = config.runs_root / run_id
    freeze_marker_inventory(config, run_id=run_id, finalize_inputs=True)
    panels = tuple(get_analysis_preset("exhaustive_research").panel_ids)
    for size, panel_ids in (
        (1_000, panels),
        (10_000, panels),
    ):
        feature_root = (
            config.features_root
            / "_".join(panel_ids)
            / f"smoke-development-{size}"
        )
        _write_manifest(
            feature_root / "features.manifest.json",
            feature_root / "part-000.parquet",
            row_counts={
                "selected_ecgs": size,
                "successful_ecgs": size,
                "feature_count": 320,
            },
        )
        validation_root = config.runs_root / f"{run_id}-pilot-{size}"
        _write_manifest(
            validation_root / "validate.manifest.json",
            validation_root / "screening_results.parquet",
        )
    (run_root / "software_preflight_gate.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "unexpected_skips": 0,
                "uncaptured_warnings": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "gui_review_manifest.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (run_root / "guided_workflow.manifest.json").write_text(
        json.dumps({"locked_test_read": False}),
        encoding="utf-8",
    )
    runtime_events = []
    for stage, duration in (
        ("extract-features-pilot-1000", 100.0),
        ("validate-pilot-1000", 50.0),
        ("extract-features-pilot-10000", 900.0),
        ("validate-pilot-10000", 100.0),
    ):
        runtime_events.extend([
            {"kind": "stage", "event": "begin", "stage": stage},
            {"kind": "reuse", "stage": stage, "reused": False},
        ])
        workload_class = (
            "waveform" if stage.startswith("extract-features") else "analysis"
        )
        runtime_events.append(
            {
                "kind": "stage",
                "event": "policy",
                "stage": stage,
                "memory_policy": {
                    "policy_version": "test",
                    "workload_class": workload_class,
                    "requested_memory_gb": 12.0,
                    "duckdb_memory_limit_gb": 8.0,
                    "non_duckdb_headroom_gb": 4.0,
                    "critical_available_gb": 3.0,
                },
            }
        )
        runtime_events.append(
            {
                "kind": "stage",
                "event": "end",
                "stage": stage,
                "duration_seconds": duration,
                "rss_gb": 4.0,
                "available_gb": 20.0,
            }
        )
    for size in (1_000, 10_000):
        runtime_events.append(
            {
                "kind": "query",
                "event": "end",
                "stage": f"validate-pilot-{size}",
                "label": "validation:endpoint-frame:test",
                "duration_seconds": 1.0,
                "rows": size,
                "rss_before_gb": 3.0,
                "rss_after_gb": 4.0,
                "available_gb": 20.0,
            }
        )
    (run_root / "run.jsonl").write_text(
        "\n".join(json.dumps(item) for item in runtime_events) + "\n",
        encoding="utf-8",
    )

    result = build_pilot_go_no_go_report(
        config,
        run_id=run_id,
        preset_id="exhaustive_research",
    )

    assert result.status == "complete"
    assert result.decision == "TECHNICAL_GO_SCIENTIFIC_REVIEW_REQUIRED", result.failed_gates
    payload = json.loads(
        (run_root / "go_no_go" / "go_no_go.json").read_text(encoding="utf-8")
    )
    assert payload["full_run_started"] is False
    assert all(payload["gates"].values())
    assert payload["details"]["performance"][
        "unmatched_pilot_feature_throughput_ratio_10k_vs_1k"
    ] == pytest.approx(0.9)
