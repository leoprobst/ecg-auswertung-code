"""Machine-readable release decision after the 1k/10k development pilots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
from typing import Any, cast

from .config import MimicProjectConfig
from .marker_freeze import assert_marker_freeze_current
from .presets import get_analysis_preset, resolve_preset_id
from .runtime_calibration import nonreuse_attempt_medians
from .storage import atomic_write_json


PILOT_GO_NO_GO_VERSION = "1.7.0"

# Gates about how well the pilots were *measured*, not whether they were
# *right*: telemetry, timing, throughput and memory bookkeeping.  A failure is
# reported as a form finding and keeps the decision.  Two pilot repeats
# (A150, A151) were lost to exactly these: a recovered 3.8 s stage failure in
# the log, and a fresh extraction whose reuse verdict was never written.
# Every gate not listed here -- marker freeze, complete features and
# validation, software suite, skips, warnings, GUI review, locked test --
# still decides the outcome on its own.
FORM_GATES: frozenset[str] = frozenset(
    {
        "runtime_query_telemetry_well_formed",
        "pilot_runtime_profiles_complete",
        "pilot_feature_throughput_measured",
        "pilot_feature_throughput_ratio_measured",
        "pilot_resource_profiles_complete",
        "pilot_policy_headroom_preserved",
        "pilot_validation_query_profiles_present",
    }
)


def pilot_technical_go(payload: dict[str, Any]) -> bool:
    """Technical GO of a go_no_go.json, for every schema version.

    Reports before 1.7.0 carry no ``blocking_gates``; there every gate binds.
    """

    gates = dict(payload.get("gates") or {})
    blocking = payload.get("blocking_gates")
    names = list(blocking) if isinstance(blocking, list) else list(gates)
    return (
        str(payload.get("decision", "")).startswith("TECHNICAL_GO")
        and bool(names)
        and all(gates.get(name) is True for name in names)
    )


@dataclass(frozen=True)
class PilotGoNoGoResult:
    status: str
    decision: str
    output_paths: tuple[str, ...]
    row_counts: dict[str, int]
    failed_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _manifest_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    outputs = [str(item) for item in payload.get("output_paths", ())]
    timing_path = path.parent / "feature_timing_contract.json"
    timing_notes = [str(note).split("=", 1)[1] for note in payload.get("notes", ())
                    if str(note).startswith("feature_timing_contract_sha256=")]
    timing_contract = None
    if len(timing_notes) == 1:
        try:
            timing_bytes = timing_path.read_bytes()
            if hashlib.sha256(timing_bytes).hexdigest() == timing_notes[0]:
                candidate = json.loads(timing_bytes)
                if isinstance(candidate, dict):
                    timing_contract = candidate
        except (OSError, ValueError):
            timing_contract = None
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "status": str(payload.get("status", "missing")),
        "row_counts": dict(payload.get("row_counts", {}) or {}),
        "outputs_present": bool(outputs) and all(Path(item).exists() for item in outputs),
        "notes": list(payload.get("notes", ()) or ()),
        "feature_runtime_contract": timing_contract,
    }


def _runtime_summary(path: Path) -> dict[str, Any]:
    durations: dict[str, float] = {}
    successful_durations: dict[str, list[float]] = {}
    failed_durations: dict[str, list[float]] = {}
    nonreuse_durations: dict[str, list[float]] = {}
    reuse_durations: dict[str, list[float]] = {}
    unknown_cache_durations: dict[str, list[float]] = {}
    active_reuse: dict[tuple[str, str, str], list[bool]] = {}
    stage_minimum_available: dict[str, float] = {}
    stage_resource_policies: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    record_count = 0
    stage_failure_count = 0
    peak_rss = 0.0
    minimum_available = float("inf")
    if not path.is_file():
        return {
            "log_path": str(path.resolve()),
            "stage_duration_seconds": durations,
            "stage_successful_attempt_seconds": {},
            "stage_failed_attempt_seconds": {},
            "stage_nonreuse_reported_attempt_seconds": {},
            "stage_reuse_reported_attempt_seconds": {},
            "stage_unknown_cache_attempt_seconds": {},
            "duration_semantics": "total_recorded_effort_not_single_execution_calibration",
            "stage_minimum_available_ram_gb": {},
            "stage_resource_policies": {},
            "query_duration_seconds": 0.0,
            "query_count": 0,
            "completed_query_count_by_stage": {},
            "query_failure_count": 0,
            "query_missing_row_count": 0,
            "record_count": 0,
            "stage_failure_count": 0,
            "slowest_queries": [],
            "peak_process_tree_rss_gb": None,
            "minimum_available_ram_gb": None,
        }
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        record_count += 1
        stage = str(event.get("stage", ""))
        key = (str(event.get("job_id", "")), str(event.get("attempt_id", "")), stage)
        if event.get("kind") == "stage" and event.get("event") == "begin":
            active_reuse[key] = []
        if event.get("kind") == "reuse" and key in active_reuse:
            reused = event.get("reused")
            if isinstance(reused, bool):
                active_reuse[key].append(reused)
        duration = event.get("duration_seconds")
        # Query durations sit inside their stage duration. Summing both made
        # the report grow as soon as real query profiling was enabled.
        if (
            event.get("kind") == "stage"
            and event.get("event") in {"end", "failed"}
            and stage
            and isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and duration >= 0
        ):
            durations[stage] = durations.get(stage, 0.0) + float(duration)
            attempts = failed_durations if event.get("event") == "failed" else successful_durations
            attempts.setdefault(stage, []).append(float(duration))
            if event.get("event") == "failed":
                stage_failure_count += 1
            else:
                decisions = active_reuse.get(key, [])
                bucket = (reuse_durations if any(decisions) else
                          nonreuse_durations if decisions else unknown_cache_durations)
                bucket.setdefault(stage, []).append(float(duration))
        if event.get("kind") == "stage" and event.get("event") in {"end", "failed"}:
            active_reuse.pop(key, None)
        if event.get("kind") == "query" and isinstance(duration, (int, float)):
            queries.append(event)
        if (
            event.get("kind") == "stage"
            and event.get("event") == "policy"
            and stage
            and isinstance(event.get("memory_policy"), dict)
        ):
            stage_resource_policies[stage] = dict(event["memory_policy"])
        rss_values = (
            event.get("rss_gb"),
            event.get("rss_before_gb"),
            event.get("rss_after_gb"),
        )
        available = event.get("available_gb")
        for rss in rss_values:
            if isinstance(rss, (int, float)):
                peak_rss = max(peak_rss, float(rss))
        if isinstance(available, (int, float)):
            minimum_available = min(minimum_available, float(available))
            if stage:
                stage_minimum_available[stage] = min(
                    stage_minimum_available.get(stage, float("inf")),
                    float(available),
                )
    return {
        "log_path": str(path.resolve()),
        "stage_duration_seconds": durations,
        "stage_successful_attempt_seconds": successful_durations,
        "stage_failed_attempt_seconds": failed_durations,
        "stage_nonreuse_reported_attempt_seconds": nonreuse_durations,
        "stage_reuse_reported_attempt_seconds": reuse_durations,
        "stage_unknown_cache_attempt_seconds": unknown_cache_durations,
        "cache_classification_scope": "logged_artifact_decisions_not_proof_of_no_internal_checkpoint_reuse",
        "duration_semantics": "total_recorded_effort_not_single_execution_calibration",
        "stage_minimum_available_ram_gb": stage_minimum_available,
        "stage_resource_policies": stage_resource_policies,
        "query_duration_seconds": sum(
            float(item["duration_seconds"]) for item in queries
        ),
        "query_count": len(queries),
        "completed_query_count_by_stage": {
            stage_id: sum(
                item.get("event") == "end"
                and str(item.get("stage", "")) == stage_id
                and item.get("rows") is not None
                for item in queries
            )
            for stage_id in sorted(
                {
                    str(item.get("stage", ""))
                    for item in queries
                    if item.get("stage")
                }
            )
        },
        "query_failure_count": sum(
            item.get("event") == "failed" for item in queries
        ),
        "query_missing_row_count": sum(
            item.get("event") == "end" and item.get("rows") is None
            for item in queries
        ),
        "record_count": record_count,
        "stage_failure_count": stage_failure_count,
        "slowest_queries": sorted(
            (
                {
                    "stage": str(item.get("stage", "")),
                    "label": str(item.get("label", "")),
                    "duration_seconds": float(item["duration_seconds"]),
                    "rows": item.get("rows"),
                    "rss_before_gb": item.get("rss_before_gb"),
                    "rss_after_gb": item.get("rss_after_gb"),
                }
                for item in queries
            ),
            key=lambda item: (
                float(cast(float, item["duration_seconds"]))
                if isinstance(item.get("duration_seconds"), (int, float))
                else float("-inf")
            ),
            reverse=True,
        )[:20],
        "peak_process_tree_rss_gb": peak_rss or None,
        "minimum_available_ram_gb": (
            minimum_available if minimum_available != float("inf") else None
        ),
    }


def _pilot_performance_summary(
    runtime: dict[str, Any],
    details: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Measured pilot scaling and objective telemetry release gates."""

    recorded_effort = dict(runtime.get("stage_duration_seconds", {}) or {})
    durations = nonreuse_attempt_medians(runtime)
    pilots: dict[str, Any] = {}
    feature_seconds_per_record: dict[int, float] = {}
    resource_policies = dict(runtime.get("stage_resource_policies", {}) or {})
    stage_available = dict(
        runtime.get("stage_minimum_available_ram_gb", {}) or {}
    )
    profiles_complete = True
    throughput_complete = True
    resource_profiles_complete = True
    policy_headroom_preserved = True
    pilot_query_profiles_present = True
    query_counts = dict(
        runtime.get("completed_query_count_by_stage", {}) or {}
    )
    for size in (1_000, 10_000):
        feature_duration = durations.get(f"extract-features-pilot-{size}")
        validation_duration = durations.get(f"validate-pilot-{size}")
        feature_counts = dict(
            details.get(f"pilot_{size}", {})
            .get("features", {})
            .get("row_counts", {})
            or {}
        )
        selected = int(feature_counts.get("selected_ecgs", 0))
        feature_count = int(feature_counts.get("feature_count", 0))
        feature_duration_valid = (
            isinstance(feature_duration, (int, float))
            and float(feature_duration) > 0
        )
        validation_duration_valid = (
            isinstance(validation_duration, (int, float))
            and float(validation_duration) > 0
        )
        duration_valid = feature_duration_valid and validation_duration_valid
        throughput_valid = (
            feature_duration_valid
            and selected == size
            and feature_count > 0
        )
        profiles_complete = profiles_complete and duration_valid
        throughput_complete = throughput_complete and throughput_valid
        seconds_per_record = (
            float(feature_duration if feature_duration is not None else 0.0)
            / selected
            if throughput_valid
            else None
        )
        if seconds_per_record is not None:
            feature_seconds_per_record[size] = seconds_per_record
        stage_ids = (
            f"extract-features-pilot-{size}",
            f"validate-pilot-{size}",
        )
        pilot_resource_profiles: dict[str, Any] = {}
        for stage_id in stage_ids:
            policy = dict(resource_policies.get(stage_id, {}) or {})
            observed_available = stage_available.get(stage_id)
            requested = policy.get("requested_memory_gb")
            duckdb_limit = policy.get("duckdb_memory_limit_gb")
            non_duckdb = policy.get("non_duckdb_headroom_gb")
            critical = policy.get("critical_available_gb")
            expected_workload = (
                "waveform"
                if stage_id.startswith("extract-features")
                else "analysis"
            )
            policy_valid = (
                bool(policy.get("policy_version"))
                and policy.get("workload_class") == expected_workload
                and isinstance(requested, (int, float))
                and float(requested) > 0
                and isinstance(duckdb_limit, (int, float))
                and float(duckdb_limit) > 0
                and isinstance(non_duckdb, (int, float))
                and float(non_duckdb) >= 0
                and abs(
                    float(requested)
                    - float(duckdb_limit)
                    - float(non_duckdb)
                )
                < 1e-6
                and isinstance(critical, (int, float))
                and float(critical) > 0
            )
            headroom_ok = (
                policy_valid
                and isinstance(observed_available, (int, float))
                and float(observed_available)
                >= float(critical if critical is not None else 0.0)
            )
            resource_profiles_complete = (
                resource_profiles_complete and policy_valid
            )
            policy_headroom_preserved = (
                policy_headroom_preserved and headroom_ok
            )
            pilot_resource_profiles[stage_id] = {
                "policy": policy,
                "observed_minimum_available_ram_gb": observed_available,
                "policy_valid": policy_valid,
                "critical_headroom_preserved": headroom_ok,
            }
        pilots[str(size)] = {
            "selected_ecgs": selected,
            "feature_count": feature_count,
            "feature_duration_seconds": feature_duration,
            "validation_duration_seconds": validation_duration,
            "feature_total_recorded_effort_seconds": recorded_effort.get(f"extract-features-pilot-{size}"),
            "validation_total_recorded_effort_seconds": recorded_effort.get(f"validate-pilot-{size}"),
            "duration_basis": "median_successful_explicit_nonreuse_attempts",
            "duration_workload_matched_to_future_run": False,
            "feature_runtime_contract": details.get(f"pilot_{size}", {}).get("features", {}).get("feature_runtime_contract"),
            "feature_records_per_second": (
                selected
                / float(feature_duration if feature_duration is not None else 0.0)
                if throughput_valid
                else None
            ),
            "feature_seconds_per_record": seconds_per_record,
            "resource_profiles": pilot_resource_profiles,
            "completed_validation_query_profiles": int(
                query_counts.get(f"validate-pilot-{size}", 0)
            ),
        }
        pilot_query_profiles_present = (
            pilot_query_profiles_present
            and int(query_counts.get(f"validate-pilot-{size}", 0)) > 0
        )

    inflation = None
    scaling_measured = False
    if set(feature_seconds_per_record) == {1_000, 10_000}:
        inflation = (
            feature_seconds_per_record[10_000]
            / feature_seconds_per_record[1_000]
        )
        scaling_measured = True
    telemetry_ok = (
        int(runtime.get("record_count", 0)) > 0
        and int(runtime.get("stage_failure_count", 0)) == 0
        and int(runtime.get("query_missing_row_count", 0)) == 0
    )
    summary = {
        "pilots": pilots,
        "feature_seconds_per_record_inflation_10k_vs_1k": None,
        "unmatched_pilot_feature_throughput_ratio_10k_vs_1k": inflation,
        "matched_feature_scaling_established": False,
        "scaling_interpretation": (
            "descriptive_ratio_only_pilots_use_different_panel_and_sampling_contracts"
        ),
        "query_failure_count_including_recovered_oom": int(
            runtime.get("query_failure_count", 0)
        ),
        "validation_scaling_not_compared": (
            "1k and 10k use different bootstrap/search contracts"
        ),
    }
    gates = {
        "pilot_runtime_profiles_complete": profiles_complete,
        "pilot_feature_throughput_measured": throughput_complete,
        "pilot_feature_throughput_ratio_measured": scaling_measured,
        "pilot_resource_profiles_complete": resource_profiles_complete,
        "pilot_policy_headroom_preserved": policy_headroom_preserved,
        "pilot_validation_query_profiles_present": (
            pilot_query_profiles_present
        ),
        "runtime_query_telemetry_well_formed": telemetry_ok,
    }
    return summary, gates


def build_pilot_go_no_go_report(
    config: MimicProjectConfig,
    *,
    run_id: str,
    preset_id: str,
) -> PilotGoNoGoResult:
    """Summarise release gates without ever starting a full analysis."""

    preset_id = resolve_preset_id(preset_id)
    panel_ids = tuple(get_analysis_preset(preset_id).panel_ids)
    run_root = config.runs_root / run_id
    report_root = run_root / "go_no_go"
    report_root.mkdir(parents=True, exist_ok=True)
    gates: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        details["marker_freeze"] = assert_marker_freeze_current(
            config,
            run_id=run_id,
        )
        gates["marker_freeze_current"] = True
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        details["marker_freeze"] = {"error": str(exc)}
        gates["marker_freeze_current"] = False

    for size, panels in (
        (1_000, panel_ids),
        (10_000, panel_ids),
    ):
        scope = config.features_root / "_".join(panels) / f"smoke-development-{size}"
        feature = _manifest_summary(scope / "features.manifest.json")
        validation = _manifest_summary(
            config.runs_root / f"{run_id}-pilot-{size}" / "validate.manifest.json"
        )
        details[f"pilot_{size}"] = {
            "features": feature,
            "validation": validation,
        }
        gates[f"pilot_{size}_features_complete"] = bool(
            feature["status"] == "complete" and feature["outputs_present"]
        )
        gates[f"pilot_{size}_validation_complete"] = bool(
            validation["status"] == "complete" and validation["outputs_present"]
        )

    software_gate = _read_json(run_root / "software_preflight_gate.json")
    gui_gate = _read_json(run_root / "gui_review_manifest.json")
    details["software_preflight"] = software_gate
    details["gui_review"] = gui_gate
    gates["software_preflight_passed"] = software_gate.get("status") == "passed"
    gates["no_unexpected_skips"] = int(software_gate.get("unexpected_skips", -1)) == 0
    gates["no_uncaptured_warnings"] = int(
        software_gate.get("uncaptured_warnings", -1)
    ) == 0
    gates["gui_contract_and_visual_review_passed"] = gui_gate.get("status") == "passed"
    gates["locked_test_unread"] = not bool(
        _read_json(run_root / "guided_workflow.manifest.json").get(
            "locked_test_read",
            False,
        )
    )
    runtime = _runtime_summary(run_root / "run.jsonl")
    performance, performance_gates = _pilot_performance_summary(
        runtime,
        details,
    )
    details["performance"] = performance
    gates.update(performance_gates)
    blocking = [name for name in gates if name not in FORM_GATES]
    failed = tuple(name for name in blocking if not gates[name])
    form_findings = [
        name for name in gates if name in FORM_GATES and not gates[name]
    ]
    decision = "TECHNICAL_GO_SCIENTIFIC_REVIEW_REQUIRED" if not failed else "NO_GO"
    limitations = [
        "Pilot results are pipeline QC and must not be used for post-hoc marker selection.",
        "MIMIC reference distributions are hospital-based and are not healthy-population normal values.",
        "Technical GO does not authorise full execution; a separate source-bound scientific release is required. Locked test remains closed.",
    ]
    payload = {
        "schema_version": PILOT_GO_NO_GO_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "preset_id": preset_id,
        "decision": decision,
        "full_run_started": False,
        "scientific_release_required": True,
        "clinical_approval": False,
        "gates": gates,
        "failed_gates": list(failed),
        "blocking_gates": blocking,
        "form_findings": form_findings,
        "form_findings_policy": (
            "Measurement and telemetry gaps are reported, not blocking; "
            "a missing throughput leaves the full-run ETA unestimated."
        ),
        "details": details,
        "runtime_and_memory": runtime,
        "limitations": limitations,
    }
    json_path = report_root / "go_no_go.json"
    markdown_path = report_root / "GO_NO_GO.md"
    atomic_write_json(json_path, payload)
    markdown_path.write_text(
        "# Pilot Go/No-Go\n\n"
        f"- Run: `{run_id}`\n"
        f"- Entscheidung: **{decision}**\n"
        "- Voll-Lauf automatisch gestartet: **nein**\n\n"
        "## Gates\n\n"
        + "\n".join(
            f"- {'PASS' if passed else ('FORM' if name in FORM_GATES else 'FAIL')}: `{name}`"
            for name, passed in gates.items()
        )
        + (
            "\n\n## Formbefunde (nicht blockierend)\n\n"
            + "\n".join(f"- `{name}`" for name in form_findings)
            if form_findings
            else ""
        )
        + "\n\n## Grenzen\n\n"
        + "\n".join(f"- {item}" for item in limitations)
        + "\n",
        encoding="utf-8",
    )
    counts = {
        "gates": len(gates),
        "passed_gates": sum(bool(value) for value in gates.values()),
        "failed_gates": len(failed),
    }
    return PilotGoNoGoResult(
        status="complete",
        decision=decision,
        output_paths=(str(json_path.resolve()), str(markdown_path.resolve())),
        row_counts=counts,
        failed_gates=failed,
    )


__all__ = [
    "FORM_GATES",
    "PILOT_GO_NO_GO_VERSION",
    "pilot_technical_go",
    "PilotGoNoGoResult",
    "build_pilot_go_no_go_report",
]
