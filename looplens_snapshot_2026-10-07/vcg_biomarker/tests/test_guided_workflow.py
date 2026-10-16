import json
import os
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic.workflow import (
    _assert_generated_run_is_current,
    _assert_run_preflight_gates,
    _assert_pilot_gate,
    _pilot_size,
    _service_stage_id,
    _stage_fraction_high_water,
    _stage_arguments,
    guided_study_spec,
    run_guided_study,
)
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.presets import get_analysis_preset


def test_a_stale_generated_run_is_rejected_by_the_backend(
    tmp_path,
    monkeypatch,
) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.workflow.resolve_current_analysis",
        lambda _config: (SimpleNamespace(run_id="A013-2026-09-03-fedcba"), None),
    )

    with pytest.raises(RuntimeError, match="no longer compatible"):
        _assert_generated_run_is_current(
            config,
            "A012-2026-09-03-abcdef",
        )


def test_the_current_generated_run_and_legacy_names_are_accepted(
    tmp_path,
    monkeypatch,
) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.workflow.resolve_current_analysis",
        lambda _config: (
            SimpleNamespace(run_id="A012-2026-09-03-abcdef"),
            SimpleNamespace(fingerprint="fingerprint"),
        ),
    )

    assert (
        _assert_generated_run_is_current(
            config,
            "A012-2026-09-03-abcdef",
        )
        == "fingerprint"
    )
    assert _assert_generated_run_is_current(config, "mimic-development") is None


def test_generated_run_requires_source_bound_software_and_gui_gates(
    tmp_path,
) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    run_id = "A012-2026-09-03-abcdef"
    run_root = config.runs_root / run_id
    run_root.mkdir(parents=True)
    common = {"status": "passed", "run_id": run_id, "run_fingerprint": "fp"}
    (run_root / "software_preflight_gate.json").write_text(
        json.dumps(
            {
                **common,
                "unexpected_skips": 0,
                "uncaptured_warnings": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "gui_review_manifest.json").write_text(
        json.dumps(common),
        encoding="utf-8",
    )

    _assert_run_preflight_gates(config, run_id, "fp")
    with pytest.raises(RuntimeError, match="failed or stale"):
        _assert_run_preflight_gates(config, run_id, "different")


@pytest.mark.parametrize("run_id", ["manual-full", "mimic-development-guided", "old-pilot"])
def test_manual_full_run_cannot_bypass_scientific_release(tmp_path, run_id):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    with pytest.raises(RuntimeError, match="current generated run identity"):
        run_guided_study(config, {"run_id": run_id, "preset_id": "exhaustive_research"})
    assert not (config.runs_root / run_id).exists()


@pytest.mark.parametrize("arguments", [
    {"preset_id": "fast_preview"},
    {"preset_id": "exhaustive_research", "stop_after_pilots": True},
])
def test_bounded_manual_runs_reach_specification_without_full_release(tmp_path, monkeypatch, arguments):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    class ReachedSpecification(Exception):
        pass
    def stop_at_specification(**kwargs):
        raise ReachedSpecification
    monkeypatch.setattr("vcg_biomarker.mimic.workflow.guided_study_spec", stop_at_specification)
    with pytest.raises(ReachedSpecification):
        run_guided_study(config, {"run_id": "bounded-manual", **arguments})
    assert not (config.runs_root / "bounded-manual").exists()


def test_generated_full_run_still_requires_scientific_release(tmp_path, monkeypatch):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    monkeypatch.setattr("vcg_biomarker.mimic.workflow._assert_generated_run_is_current", lambda *_: "current-fp")
    monkeypatch.setattr("vcg_biomarker.mimic.workflow._assert_run_preflight_gates", lambda *_: None)
    with pytest.raises(RuntimeError, match="scientific release missing or stale"):
        run_guided_study(config, {"run_id": "A099-2026-09-10-abcdef", "preset_id": "exhaustive_research"})


def test_subtask_progress_cannot_move_the_workflow_backwards() -> None:
    discharge = _stage_fraction_high_water(
        0.0,
        "Entlassungsbriefe 300,000/331,793",
    )
    radiology = _stage_fraction_high_water(
        discharge,
        "radiology_chest_xray 350,000/532,566",
    )
    assert discharge > 0.90
    assert radiology == discharge


def test_config_discovery_detects_optional_mimic_note_root(tmp_path) -> None:
    database_root = tmp_path / "database"
    note_root = database_root / "mimic-iv-note"
    note_root.mkdir(parents=True)

    config = MimicProjectConfig.discover(
        database_root=database_root,
        work_root=tmp_path / "work",
    )

    assert config.mimic_note_root == note_root


def test_guided_presets_insert_pilots_before_any_full_analysis() -> None:
    """A month of arithmetic never starts before the pilots have passed.

    The middle arm this test used to check -- ``recommended_clinical``, which
    ran the 1.000-record pilot but not the 10.000-record one -- no longer
    exists: the preset set collapsed to one study configuration plus a
    development preview. The ordering guarantee is the point of the test and
    is unchanged.
    """

    preview = guided_study_spec(preset_id="fast_preview", run_id="preview")
    study = guided_study_spec(preset_id="exhaustive_research", run_id="deep")

    # The preview exists to reach every stage quickly; it must never run the
    # full extraction, and it has nothing to scale up to.
    assert "extract-features-pilot-1000" in preview.stage_ids
    assert "extract-features" not in preview.stage_ids
    assert "extract-features-pilot-10000" not in preview.stage_ids

    # The study climbs 1.000 -> 10.000 -> everything, and each step is gated
    # by the validation of the one before it.
    assert study.stage_ids.index("validate-pilot-1000") < study.stage_ids.index(
        "extract-features-pilot-10000"
    )
    assert study.stage_ids.index("validate-pilot-10000") < study.stage_ids.index(
        "extract-features"
    )
    assert study.development_only is True
    assert len(study.stage_ids) == len(set(study.stage_ids))
    assert study.stage_ids.index("score-discovery") < study.stage_ids.index(
        "reporting-guideline"
    ) < study.stage_ids.index("report") < study.stage_ids.index("audit")
    assert "score-discovery" not in preview.stage_ids

    held = guided_study_spec(
        preset_id="exhaustive_research",
        run_id="held",
        stop_after_pilots=True,
    )
    assert held.stage_ids[-1] == "pilot-go-no-go"
    assert "extract-features" not in held.stage_ids
    assert "validate-pilot-10000" in held.stage_ids


def test_pilot_aliases_never_become_new_public_service_stages() -> None:
    assert _pilot_size("extract-features-pilot-1000") == 1_000
    assert _pilot_size("validate-pilot-10000") == 10_000
    assert _pilot_size("extract-features") is None
    assert _service_stage_id("extract-features-pilot-1000") == "extract-features"
    assert _service_stage_id("validate-pilot-10000") == "validate"


def test_pilot_gate_requires_complete_rows_and_technical_success() -> None:
    _assert_pilot_gate(
        "extract-features-pilot-1000",
        SimpleNamespace(
            status="complete",
            row_counts={"selected_ecgs": 1_000, "successful_ecgs": 800},
        ),
    )
    with pytest.raises(RuntimeError, match="80 %"):
        _assert_pilot_gate(
            "extract-features-pilot-1000",
            SimpleNamespace(
                status="complete",
                row_counts={"selected_ecgs": 1_000, "successful_ecgs": 799},
            ),
        )
    with pytest.raises(RuntimeError, match="keine auswertbaren"):
        _assert_pilot_gate(
            "validate-pilot-10000",
            SimpleNamespace(status="complete", row_counts={}),
        )


def test_large_pilot_scales_all_singles_without_repeating_bootstraps(
    tmp_path,
) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    small = _stage_arguments(
        config,
        "validate-pilot-1000",
        preset_id="exhaustive_research",
        run_id="deep",
    )
    scaling = _stage_arguments(
        config,
        "validate-pilot-10000",
        preset_id="exhaustive_research",
        run_id="deep",
    )

    assert small["bootstrap_iterations"] == 200
    assert small["grouped_score_selected_triples"] > 0
    assert scaling["bootstrap_iterations"] == 0
    assert scaling["panel_ids"] == tuple(
        get_analysis_preset("exhaustive_research").panel_ids
    )
    assert scaling["grouped_score_selected_pairs"] == 1
    assert scaling["grouped_score_selected_triples"] == 0
    assert scaling["maximum_criterion_combination_candidates"] == 0


def test_guided_retry_skips_manifested_completed_stages(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import services

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    calls: list[tuple[str, int | None]] = []

    def fake_stage(_config, stage, arguments, **_kwargs):
        maximum = arguments.get("max_records")
        calls.append((stage, maximum))
        if stage == "extract-features" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={
                    "selected_ecgs": int(maximum),
                    "successful_ecgs": int(maximum),
                },
            )
        if stage == "validate" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={"screening": 1},
            )
        return SimpleNamespace(status="complete", row_counts={"rows": 1})

    monkeypatch.setattr(services, "run_service_stage", fake_stage)
    arguments = {
        "preset_id": "fast_preview",
        "run_id": "resume-test",
        "include_locked_test": False,
    }
    first = run_guided_study(config, arguments)
    assert calls
    assert first.completed_stages == guided_study_spec(
        preset_id="fast_preview",
        run_id="resume-test",
    ).stage_ids
    calls.clear()
    second = run_guided_study(config, arguments)
    assert second.completed_stages == first.completed_stages
    assert calls == []


def test_a_changed_source_tree_recomputes_instead_of_ending_the_run(
    tmp_path, monkeypatch
) -> None:
    """The safety concern is real; refusing the run was the wrong instrument.

    Artifacts from two versions of the code must never be mixed. But the
    workflow hash covers the whole package -- every module, the interface, the
    test suite -- so editing a test file or a stage that has not run yet
    changes it. On 2026-08-20 a run four hours into attest-event-evidence was
    restarted by the supervisor after a memory dip and refused on the way back
    in, because unrelated modules had been edited while it worked.

    Nothing is mixed now either: the stale completed list is *discarded*, so
    every stage is offered to its own reuse key -- which is precise where the
    workflow hash is coarse. A stage still valid skips in seconds; one that is
    not recomputes.
    """

    from vcg_biomarker.mimic import services

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    run_root = config.runs_root / "stale-run"
    run_root.mkdir(parents=True)
    manifest_path = run_root / "guided_workflow.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "workflow_hash": "historical-contract-hash",
                "completed_stages": ["inventory", "ingest"],
                "locked_test_read": False,
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_stage(_config, stage, arguments, **_kwargs):
        calls.append(stage)
        maximum = arguments.get("max_records")
        if stage == "extract-features" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={
                    "selected_ecgs": int(maximum),
                    "successful_ecgs": int(maximum),
                },
            )
        if stage == "validate" and maximum:
            return SimpleNamespace(status="complete", row_counts={"screening": 1})
        return SimpleNamespace(status="complete", row_counts={"rows": 1})

    monkeypatch.setattr(services, "run_service_stage", fake_stage)

    result = run_guided_study(
        config,
        {
            "preset_id": "fast_preview",
            "run_id": "stale-run",
            "include_locked_test": False,
        },
    )

    # The stages the stale manifest claimed were done are offered again rather
    # than inherited: their own reuse keys, not this one, decide.
    assert "inventory" in calls
    assert "ingest" in calls
    assert result.completed_stages

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["workflow_hash"] != "historical-contract-hash"


def test_an_unchanged_source_tree_still_skips_completed_stages(
    tmp_path, monkeypatch
) -> None:
    """The cheap path must stay cheap: a real resume recomputes nothing."""

    from vcg_biomarker.mimic import services

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    calls: list[str] = []

    def fake_stage(_config, stage, arguments, **_kwargs):
        calls.append(stage)
        maximum = arguments.get("max_records")
        if stage == "extract-features" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={
                    "selected_ecgs": int(maximum),
                    "successful_ecgs": int(maximum),
                },
            )
        if stage == "validate" and maximum:
            return SimpleNamespace(status="complete", row_counts={"screening": 1})
        return SimpleNamespace(status="complete", row_counts={"rows": 1})

    monkeypatch.setattr(services, "run_service_stage", fake_stage)
    arguments = {
        "preset_id": "fast_preview",
        "run_id": "warm-run",
        "include_locked_test": False,
    }
    run_guided_study(config, arguments)
    assert calls
    calls.clear()
    run_guided_study(config, arguments)
    assert calls == []


def test_the_guided_run_leaves_a_readable_record(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What was missing when a stage stalled for 26 hours.

    The run produced 31,414 events, 99.1 % of them free-text progress lines
    that overwrite each other and vanish with the process. Nothing said how
    long a stage took, how much memory it held, or why it recomputed. This is
    the end-to-end proof that the replacement is written, survives the run,
    and reads back.
    """

    from vcg_biomarker.mimic import services
    from vcg_biomarker.mimic.run_log import (
        RUN_LOG_FILENAME,
        read_records,
        render_summary,
        summarize,
    )

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()

    def fake_stage(_config, stage, arguments, **_kwargs):
        maximum = arguments.get("max_records")
        if stage == "extract-features" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={
                    "selected_ecgs": int(maximum),
                    "successful_ecgs": int(maximum),
                },
            )
        if stage == "validate" and maximum:
            return SimpleNamespace(status="complete", row_counts={"screening": 1})
        return SimpleNamespace(status="complete", row_counts={"rows": 1})

    monkeypatch.setattr(services, "run_service_stage", fake_stage)
    arguments = {
        "preset_id": "fast_preview",
        "run_id": "protokoll-test",
        "include_locked_test": False,
    }
    result = run_guided_study(config, arguments)

    log_path = config.runs_root / "protokoll-test" / RUN_LOG_FILENAME
    assert log_path.is_file(), "der Lauf hat kein Protokoll hinterlassen"
    records = read_records(log_path)

    opened = [
        item
        for item in records
        if item.get("kind") == "workflow" and item.get("event") == "begin"
    ]
    assert opened and opened[0]["run_id"] == "protokoll-test"
    closed = [
        item
        for item in records
        if item.get("kind") == "workflow" and item.get("event") == "end"
    ]
    assert closed, "das Ende des Laufs steht nicht im Protokoll"

    summary = summarize(records)
    logged = {entry["stage"] for entry in summary["stages"]}
    for stage in result.completed_stages:
        assert stage in logged, stage

    # Pilot sizes are separate performance experiments even though they call
    # the same service. Collapsing both to "extract-features"/"validate"
    # made their throughput impossible to compare in the Go/No-Go report.
    assert "extract-features-pilot-1000" in logged
    assert "validate-pilot-1000" in logged

    for entry in summary["stages"]:
        assert entry["duration_seconds"] is not None, entry["stage"]

    text = render_summary(summary)
    assert "Stufe" in text
    for stage in logged:
        assert stage in text


def test_the_second_run_records_that_it_skipped(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stage that finishes in no time must be legible as a skip."""

    from vcg_biomarker.mimic import services
    from vcg_biomarker.mimic.run_log import RUN_LOG_FILENAME, read_records

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    def fake_stage(_config, stage, arguments, **_kwargs):
        maximum = arguments.get("max_records")
        if stage == "extract-features" and maximum:
            return SimpleNamespace(
                status="complete",
                row_counts={
                    "selected_ecgs": int(maximum),
                    "successful_ecgs": int(maximum),
                },
            )
        if stage == "validate" and maximum:
            return SimpleNamespace(status="complete", row_counts={"screening": 1})
        return SimpleNamespace(status="complete", row_counts={"rows": 1})

    monkeypatch.setattr(services, "run_service_stage", fake_stage)
    arguments = {
        "preset_id": "fast_preview",
        "run_id": "skip-test",
        "include_locked_test": False,
    }
    run_guided_study(config, arguments)
    log_path = config.runs_root / "skip-test" / RUN_LOG_FILENAME
    before = len(read_records(log_path))

    run_guided_study(config, arguments)
    new = read_records(log_path)[before:]
    skipped = [
        item
        for item in new
        if item.get("kind") == "stage" and item.get("event") == "skipped"
    ]
    assert skipped, "der zweite Lauf sagt nicht, dass er uebersprungen hat"
    assert all(item.get("reason") for item in skipped)


def test_production_path_preview_runs_every_post_pilot_stage_once(tmp_path) -> None:
    """The preview walks the study's own path after the pilots, on a bounded cohort.

    It exists because A117 ended after ``validate``: every stage after it had
    only ever seen synthetic fixtures. The preview must therefore contain the
    full-analysis and final stages in the study's order, no pilot stages, and
    no reduced-stage shortcut like the fast preview.
    """

    preview = guided_study_spec(preset_id="production_path_preview", run_id="named-preview")
    study = guided_study_spec(preset_id="exhaustive_research", run_id="deep")

    assert not any("pilot" in stage for stage in preview.stage_ids)
    assert "extract-features" in preview.stage_ids
    for stage in ("marker-redundancy", "therapy-response", "physiological-gap-markers",
                  "repeatability", "validate", "score-discovery", "reporting-guideline",
                  "report", "audit"):
        assert stage in preview.stage_ids, stage
    assert preview.stage_ids.index("score-discovery") < preview.stage_ids.index(
        "reporting-guideline"
    ) < preview.stage_ids.index("report") < preview.stage_ids.index("audit")
    assert preview.include_phenome_scan is False
    assert preview.development_only is True
    # Same order as the study once the pilots and the phenome scan are removed.
    assert "discover-phenome" not in preview.stage_ids
    study_order = [
        s for s in study.stage_ids if "pilot" not in s and s != "discover-phenome"
    ]
    assert [s for s in preview.stage_ids if s in study_order] == study_order

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    arguments = _stage_arguments(
        config, "validate", preset_id="production_path_preview", run_id="named-preview",
    )
    assert arguments["run_id"] == "named-preview"
    # The screen refuses capped artifacts without its preview flag; the flag
    # is set for that guard alone, the bootstrap stays at the preset's value.
    assert arguments["preview"] is True
    assert arguments["bootstrap_iterations"] == get_analysis_preset(
        "production_path_preview"
    ).policy.bootstrap_iterations
    assert arguments["include_all_ecg_secondary"] is True
    # Score discovery has the same guard: a probability score cannot be
    # trained on an endpoint-enriched sample, so it records that instead.
    assert _stage_arguments(
        config, "score-discovery", preset_id="production_path_preview", run_id="named-preview",
    )["preview"] is True
    assert _stage_arguments(
        config, "score-discovery", preset_id="exhaustive_research", run_id="deep",
    )["preview"] is False
    assert _stage_arguments(
        config, "audit", preset_id="production_path_preview", run_id="named-preview",
    )["strict"] is False
    assert _stage_arguments(
        config, "audit", preset_id="exhaustive_research", run_id="deep",
    )["strict"] is True
    assert arguments["panel_ids"] == tuple(
        get_analysis_preset("exhaustive_research").panel_ids
    )
    assert "smoke-development-10000" in str(arguments["feature_glob"])
    cohorts = _stage_arguments(
        config, "population-cohorts", preset_id="production_path_preview", run_id="named-preview",
    )
    assert cohorts["max_records_per_cohort"] == 20_000
    assert cohorts["verify_source_sha256"] is True
    assert _stage_arguments(
        config, "population-cohorts", preset_id="exhaustive_research", run_id="deep",
    )["max_records_per_cohort"] is None


def test_production_path_preview_refuses_a_generated_run_identity(tmp_path, monkeypatch) -> None:
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    monkeypatch.setattr(
        "vcg_biomarker.mimic.workflow._assert_generated_run_is_current",
        lambda *_: "current-fp",
    )
    with pytest.raises(RuntimeError, match="generated development run identity"):
        run_guided_study(
            config,
            {"run_id": "A118-2026-09-18-abcdef", "preset_id": "production_path_preview"},
        )


@pytest.mark.parametrize("preset_id", ["exhaustive_research", "production_path_preview"])
def test_every_final_stage_receives_the_run_identifier(tmp_path, preset_id) -> None:
    """The control panel and everything after it read this run's results.

    They used to receive no ``run_id`` and fell back to the service default,
    so a generated run would have ended at the control panel with missing
    screening results -- after the complete extraction and screen.
    """
    from vcg_biomarker.mimic.workflow import GUIDED_FINAL_STAGES

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work",
    )
    scope = "smoke-development-10000" if preset_id == "production_path_preview" else "development"
    for stage in (*GUIDED_FINAL_STAGES, "score-discovery"):
        arguments = _stage_arguments(config, stage, preset_id=preset_id, run_id="A118-2026-09-19-abcdef")
        assert arguments["run_id"] == "A118-2026-09-19-abcdef", stage
        # The stages reading features get this run's partitions, never the
        # service default: a bounded preview has no complete extraction.
        if "feature_glob" in arguments:
            assert f"{scope}{os.sep}split=*" in str(arguments["feature_glob"]), stage
