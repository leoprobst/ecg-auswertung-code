import pytest

from vcg_biomarker.mimic.runtime_calibration import nonreuse_attempt_medians
from vcg_biomarker.mimic.pilot_go_no_go import _pilot_performance_summary


def test_only_explicit_successful_nonreuse_times_contribute():
    runtime = {"stage_nonreuse_reported_attempt_seconds": {"x": [10, 30]},
               "stage_duration_seconds": {"x": 2000},
               "stage_failed_attempt_seconds": {"x": [1800]},
               "stage_unknown_cache_attempt_seconds": {"x": [2]},
               "stage_reuse_reported_attempt_seconds": {"x": [1]}}
    assert nonreuse_attempt_medians(runtime) == {"x": 20}


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), "12", None])
def test_invalid_attempts_do_not_create_prior(value):
    assert nonreuse_attempt_medians({"stage_nonreuse_reported_attempt_seconds": {"x": [value]}}) == {}


def test_legacy_aggregate_cost_does_not_prove_feature_throughput():
    runtime = {"stage_duration_seconds": {"extract-features-pilot-1000": 300, "validate-pilot-1000": 20}}
    details = {"pilot_1000": {"features": {"row_counts": {"selected_ecgs": 1000, "feature_count": 30}}}}
    report, gates = _pilot_performance_summary(runtime, details)
    pilot = report["pilots"]["1000"]
    assert pilot["feature_total_recorded_effort_seconds"] == 300
    assert pilot["feature_seconds_per_record"] is None
    assert not gates["pilot_feature_throughput_measured"]


def test_failed_and_reused_effort_does_not_inflate_pilot_throughput():
    runtime = {"stage_duration_seconds": {"extract-features-pilot-1000": 2000},
               "stage_nonreuse_reported_attempt_seconds": {
                   "extract-features-pilot-1000": [100, 200], "validate-pilot-1000": [20]}}
    details = {"pilot_1000": {"features": {"row_counts": {"selected_ecgs": 1000, "feature_count": 30}}}}
    report, _ = _pilot_performance_summary(runtime, details)
    pilot = report["pilots"]["1000"]
    assert pilot["feature_duration_seconds"] == 150
    assert pilot["feature_seconds_per_record"] == .15
    assert pilot["feature_total_recorded_effort_seconds"] == 2000


def test_unknown_validation_cache_does_not_erase_measured_extraction_rate():
    runtime = {"stage_nonreuse_reported_attempt_seconds": {"extract-features-pilot-1000": [100]}}
    details = {"pilot_1000": {"features": {"row_counts": {"selected_ecgs": 1000, "feature_count": 30}}}}
    report, gates = _pilot_performance_summary(runtime, details)
    assert report["pilots"]["1000"]["feature_seconds_per_record"] == .1
    assert report["pilots"]["1000"]["validation_duration_seconds"] is None
    assert not gates["pilot_runtime_profiles_complete"]


def timing_contract():
    from vcg_biomarker.mimic.runtime_calibration import FEATURE_TIMING_HASH_FIELDS
    return {"schema_version": "1.0.0", "stage": "extract-features",
            **{key: "a"*64 for key in FEATURE_TIMING_HASH_FIELDS}}


def stage_inputs():
    return {"feature_code_hash": "a"*64, "panel_hash": "b"*64, "cohort_selection": "c"*64,
            "locked_test_only": False, "max_records": 1000, "capped_sampling_version": "synthetic",
            "partition_size": 100, "sensitivity_vcg_methods": ["kors", "inverse_dower"]}


def test_producer_reuses_selected_cohort_identity_and_rejects_different_workload():
    from vcg_biomarker.mimic.runtime_calibration import build_feature_timing_contract, feature_timing_contracts_match
    def build(inputs):
        return build_feature_timing_contract(inputs, scientific_config_sha256="d"*64,
                                            scheduler_sha256="e"*64, resource_payload={"workers": 4})
    reference = build(stage_inputs())
    assert reference["input_population_sha256"] == "c"*64
    assert feature_timing_contracts_match(reference, build(stage_inputs()))
    for key, changed in [("cohort_selection", "f"*64), ("max_records", None), ("panel_hash", "f"*64),
                         ("partition_size", 200), ("sensitivity_vcg_methods", ["kors"])]:
        assert not feature_timing_contracts_match(reference, build({**stage_inputs(), key: changed}))
    with pytest.raises(ValueError):
        build({**stage_inputs(), "locked_test_only": True})
    with pytest.raises(ValueError):
        build({**stage_inputs(), "feature_code_hash": "unknown"})


def test_manifest_only_loads_content_bound_timing_contract(tmp_path):
    import json
    import hashlib
    from vcg_biomarker.mimic.pilot_go_no_go import _manifest_summary
    path = tmp_path / "features.manifest.json"
    timing_path = tmp_path / "feature_timing_contract.json"
    data = json.dumps(timing_contract()).encode()
    timing_path.write_bytes(data)
    path.write_text(json.dumps({"notes": ["feature_timing_contract_sha256="+hashlib.sha256(data).hexdigest()]}))
    assert _manifest_summary(path)["feature_runtime_contract"] == timing_contract()
    timing_path.write_text("{}")
    assert _manifest_summary(path)["feature_runtime_contract"] is None


def test_live_progress_roundtrips_prospective_feature_contract():
    from vcg_biomarker.mimic.live_progress import RunProgress
    progress = RunProgress("synthetic", ["extract-features"], feature_runtime_contract=timing_contract())
    assert progress.to_dict()["feature_runtime_contract"] == timing_contract()


def test_production_extractor_publishes_current_contract_before_reuse_without_rewriting_old_artifact(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from vcg_biomarker.mimic import feature_extraction as module
    from vcg_biomarker.mimic.run_log import RunLog, active
    config = SimpleNamespace(labels_root=tmp_path / "labels", standardized_root=tmp_path / "standardized",
                             features_root=tmp_path / "features", sensitivity_vcg_methods=("kors",))
    config.labels_root.mkdir()
    (config.labels_root / "clinical_label_evidence.parquet").write_bytes(b"synthetic existence sentinel")
    output = config.features_root / "synthetic" / "development"
    output.mkdir(parents=True)
    (output / "features.manifest.json").write_text("{}")
    timing = output / "feature_timing_contract.json"
    timing.write_bytes(b"historical artifact must remain unchanged on reuse")
    monkeypatch.setattr(module, "resolve_feature_panel_selection", lambda *a: {
        "panel_payload": {"panel_hash": "a"*64}, "feature_ids": ("synthetic",),
        "vcg_feature_ids": (), "classical_feature_ids": (), "sensitivity_feature_ids": ()})
    monkeypatch.setattr(module, "feature_implementation_hash", lambda: "b"*64)
    monkeypatch.setattr(module, "scientific_config_hash", lambda _: "c"*64)
    monkeypatch.setattr(module, "operational_config_hash", lambda _: "d"*64)
    monkeypatch.setattr(module, "_load_feature_cohort", lambda *a, **k: pd.DataFrame({"study_id": [123]}))
    monkeypatch.setattr(module, "_cohort_record_payloads", lambda _: {123: '{"study_id":123}'})
    previous = SimpleNamespace(matches=lambda *a: True)
    monkeypatch.setattr(module.StageManifest, "load", lambda _: previous)
    log = RunLog(tmp_path / "run.jsonl")
    messages = []
    with active(log):
        result = module.extract_mimic_features(config, module.FeatureExtractionOptions(panel_ids=("synthetic",)),
                                               progress=messages.append)
    assert result is previous
    assert log.feature_runtime_contract["stage"] == "extract-features"
    assert "observed_reused_records" not in log.feature_runtime_contract
    assert any("Laufzeitvertrag" in message for message in messages)
    assert timing.read_bytes() == b"historical artifact must remain unchanged on reuse"


def test_gui_does_not_extrapolate_legacy_pilot_throughput():
    from vcg_biomarker.gui.run_progress import _feature_full_seconds
    report = {"details": {"performance": {"pilots": {"1000": {"feature_seconds_per_record": .2}}}}}
    assert _feature_full_seconds(report, 900000) is None


def test_gui_extrapolation_requires_matching_prospective_contract_and_attempts():
    from vcg_biomarker.gui.run_progress import _feature_full_seconds
    pilot = {"feature_seconds_per_record": .2, "feature_runtime_contract": {**timing_contract(), "observed_reused_records": 0},
             "duration_basis": "median_successful_explicit_nonreuse_attempts"}
    report = {"details": {"performance": {"pilots": {"1000": pilot}}}}
    assert _feature_full_seconds(report, 900000, timing_contract()) == 180000
    assert _feature_full_seconds(report, 900000) is None
    altered = {**timing_contract(), "panel_sha256": "b"*64}
    assert _feature_full_seconds(report, 900000, altered) is None
    pilot["duration_basis"] = "total_recorded_effort"
    assert _feature_full_seconds(report, 900000, timing_contract()) is None


@pytest.mark.parametrize("reused", [None, True, False, 1, -1, "0"])
def test_gui_extrapolation_rejects_internal_reuse_or_unknown_reuse(reused):
    from vcg_biomarker.gui.run_progress import _feature_full_seconds
    pilot = {"feature_seconds_per_record": .2,
             "feature_runtime_contract": {**timing_contract(), "observed_reused_records": reused},
             "duration_basis": "median_successful_explicit_nonreuse_attempts"}
    report = {"details": {"performance": {"pilots": {"1000": pilot}}}}
    assert _feature_full_seconds(report, 900000, timing_contract()) is None


@pytest.mark.parametrize("field", ["implementation_sha256", "scientific_config_sha256", "panel_sha256",
                                  "input_population_sha256", "sampling_design_sha256", "resources_sha256"])
def test_timing_contract_rejects_missing_or_mismatched_identity(field):
    from vcg_biomarker.mimic.runtime_calibration import feature_timing_contracts_match
    assert not feature_timing_contracts_match(timing_contract(), {**timing_contract(), field: "b"*64})
    assert not feature_timing_contracts_match(timing_contract(), {k: v for k, v in timing_contract().items() if k != field})


def test_fresh_extraction_without_manifest_is_logged_as_not_reused(tmp_path, monkeypatch):
    """A150 pilots (01.10.2026): a genuinely fresh extraction left no reuse
    record, so the go/no-go gate filed its measured duration as cache-unknown."""
    from types import SimpleNamespace
    import pandas as pd
    from vcg_biomarker.mimic import feature_extraction as module
    from vcg_biomarker.mimic import run_log as run_log_module
    from vcg_biomarker.mimic.run_log import RunLog, active, read_records

    class _Stop(Exception):
        pass

    config = SimpleNamespace(labels_root=tmp_path / "labels", standardized_root=tmp_path / "standardized",
                             features_root=tmp_path / "features", sensitivity_vcg_methods=("kors",))
    config.labels_root.mkdir()
    (config.labels_root / "clinical_label_evidence.parquet").write_bytes(b"synthetic existence sentinel")
    monkeypatch.setattr(module, "resolve_feature_panel_selection", lambda *a: {
        "panel_payload": {"panel_hash": "a"*64}, "feature_ids": ("synthetic",),
        "vcg_feature_ids": (), "classical_feature_ids": (), "sensitivity_feature_ids": ()})
    monkeypatch.setattr(module, "feature_implementation_hash", lambda: "b"*64)
    monkeypatch.setattr(module, "scientific_config_hash", lambda _: "c"*64)
    monkeypatch.setattr(module, "operational_config_hash", lambda _: "d"*64)
    monkeypatch.setattr(module, "_load_feature_cohort", lambda *a, **k: pd.DataFrame({"study_id": [123]}))
    monkeypatch.setattr(module, "_cohort_record_payloads", lambda _: {123: '{"study_id":123}'})
    original = run_log_module.note_reuse

    def note_then_stop(*args, **kwargs):
        original(*args, **kwargs)
        raise _Stop

    monkeypatch.setattr(run_log_module, "note_reuse", note_then_stop)
    log = RunLog(tmp_path / "run.jsonl")
    with active(log), log.stage_scope("extract-features-pilot-1000"):
        with pytest.raises(_Stop):
            module.extract_mimic_features(config, module.FeatureExtractionOptions(panel_ids=("synthetic",)))
    reuse = [r for r in read_records(tmp_path / "run.jsonl") if r.get("kind") == "reuse"]
    assert len(reuse) == 1
    assert reuse[0]["reused"] is False
    assert reuse[0]["reason"] == "no_previous_manifest"
    assert reuse[0]["decided_stage"] == "extract-features-pilot-1000"
