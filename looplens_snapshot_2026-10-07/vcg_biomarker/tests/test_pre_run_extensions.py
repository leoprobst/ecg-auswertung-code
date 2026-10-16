import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.core.delineation.base import BeatDelineation, MarkerPoint
from vcg_biomarker.core.measurement_capsule import measurement_capsule, decode_template
from vcg_biomarker.mimic.pre_run_extensions import (
    ExtensionPolicy,
    activation_residuals,
    personal_deviations,
    sensitivity_summary,
    long_window_summary,
    clinical_association,
    competing_diagnosis_frame,
    cohort_information,
    reference_timing_summary,
)


def frame(n=100):
    rng = np.random.default_rng(45)
    x = rng.normal(size=n)
    return pd.DataFrame(
        dict(
            source="mimic",
            subject_id=np.arange(n),
            study_id=np.arange(n),
            split=["train"] * (n * 4 // 5) + ["validation"] * (n - n * 4 // 5),
            activation=x,
            target=2 * x + rng.normal(0, 0.05, n),
            reference_verified=True,
        )
    )


def test_capsule_roundtrip_and_cluster_isolation():
    signal = np.random.default_rng(1).normal(size=(2000, 12))
    beats = [
        BeatDelineation(
            r_peak=MarkerPoint(r, 0.9, "test"),
            r_onset=MarkerPoint(r - 20, 0.9, "test"),
            r_offset=MarkerPoint(r + 20, 0.9, "test"),
            morphology_cluster_id=cluster,
        )
        for r, cluster in ((400, 1), (900, 2), (1400, 1))
    ]
    capsule = measurement_capsule(
        signal,
        SimpleNamespace(beats=beats),
        500,
        source="test",
        record_id="1",
        raw_reference="raw",
        processing={},
    )
    json.dumps(capsule, allow_nan=False)
    assert len(capsule["templates"]) == 2
    for template in capsule["templates"]:
        start = template["start_sample"]
        np.testing.assert_array_equal(
            decode_template(template),
            signal[start : start + template["shape"][0]].astype("<f4"),
        )
    template["sha256"] = "bad"
    with pytest.raises(ValueError, match="integrity"):
        decode_template(template)


def test_capsule_unknown_cluster_never_assumed_sinus():
    beat = BeatDelineation(r_peak=MarkerPoint(100, 0.9, "test"))
    capsule = measurement_capsule(
        np.ones((500, 12)),
        SimpleNamespace(beats=[beat]),
        500,
        source="test",
        record_id=1,
        raw_reference="raw",
        processing={},
    )
    assert capsule["status"] == "metadata_only"
    assert capsule["beats"][0]["points"]["p_onset"] is None


def test_personal_history_no_future_or_tie_leakage():
    data = frame(8).assign(
        subject_id=1,
        split="train",
        rhythm_context="regular",
        transform="kors",
        ecg_time=pd.date_range("2020-01-01", periods=8, tz="UTC"),
        activation=np.arange(8, dtype=float),
    )
    before = personal_deviations(data, ["activation"])
    data.loc[7, "activation"] = 1e9
    after = personal_deviations(data, ["activation"])
    pd.testing.assert_frame_equal(before.iloc[:7], after.iloc[:7])
    assert before.iloc[0].status == "non_estimable"
    assert before.iloc[6].delta_previous == 1
    data.loc[7, "ecg_time"] = data.loc[6, "ecg_time"]
    assert len(personal_deviations(data, ["activation"])) == 7


@pytest.mark.parametrize("split", ["locked_test", "test", "unknown"])
def test_no_locked_test(split):
    data = frame()
    data.loc[0, "split"] = split
    with pytest.raises(ValueError, match="nondevelopment"):
        activation_residuals(data, ["activation"], ["target"])


def test_activation_grouped_oof_frozen_validation():
    data = frame()
    out, models = activation_residuals(data, ["activation"], ["target"])
    assert out.residual.abs().mean() < 0.1
    assert set(out.evaluation_role) == {"out_of_fold", "frozen_validation"}
    data.loc[data.split.eq("validation"), "target"] = 1e5
    altered, altered_models = activation_residuals(data, ["activation"], ["target"])
    np.testing.assert_array_equal(out.predicted, altered.predicted)
    assert models == altered_models


def test_sensitivity_missing_variants_not_zero_or_ci():
    out = sensitivity_summary(
        2, {"boundary": [1, 3, np.nan]}, threshold=2, threshold_frozen=True
    )
    assert out.iloc[0].n_attempted == 3
    assert out.iloc[0].n_valid == 2
    assert out.iloc[0].classification_flip_fraction == 0.5
    with pytest.raises(ValueError, match="frozen"):
        sensitivity_summary(2, {}, threshold=2)


def test_long_windows_and_patient_denominator():
    data = pd.DataFrame(
        dict(
            source="incart",
            subject_id=1,
            record_id=1,
            split="train",
            start_s=np.arange(8) * 10,
            duration_s=10,
            quality=0.9,
            marker=np.arange(8),
        )
    )
    out = long_window_summary(data, ["marker"])
    assert out.iloc[0].independent_patients == 1
    assert out.iloc[0].n_windows_total == 8
    data.loc[1, "start_s"] = 9
    with pytest.raises(ValueError, match="overlapping"):
        long_window_summary(data, ["marker"])


def test_continuous_association_and_separation_abstention():
    data = frame(200)
    result = clinical_association(
        data, marker="activation", outcome="target", covariates=()
    )
    assert result.iloc[0].status == "estimated"
    assert 0.99 < result.iloc[0].coefficient <= 1.01
    assert result.iloc[0].effect_scale == "standardized_beta"
    data["target"] = data.activation.gt(0).astype(int)
    result = clinical_association(
        data, marker="activation", outcome="target", binary=True, covariates=()
    )
    assert result.iloc[0].status == "non_estimable"


def test_specificity_unknown_not_control():
    data = frame(4).assign(
        a=[1, 0, 1, 1], b=[0, 1, np.nan, 1], a_verified=True, b_verified=True
    )
    selected, counts = competing_diagnosis_frame(data, target="a", competitor="b")
    assert len(selected) == 2
    assert counts["n_overlap"] == 1
    assert counts["n_known_both"] == 3


def test_availability_before_filtering():
    data = frame(10)
    data.loc[0, "activation"] = np.nan
    out = cohort_information(data, ["activation"])
    assert out.iloc[0].n_records == 8
    assert out.iloc[0].available_records == 7
    data = data.assign(
        reference_kind="lab",
        reference_value=1,
        reference_delta_hours=np.nan,
        setting="icu",
    )
    out = reference_timing_summary(data, marker="activation", endpoint="test")
    assert set(out.time_stratum) == {"unknown"}
    assert out.both_available_n.sum() == 9


def test_residual_small_sample_abstains():
    out, models = activation_residuals(frame(10), ["activation"], ["target"])
    assert out.predicted.isna().all()
    assert models[0]["status"] == "non_estimable"
    with pytest.raises(ValueError, match="overlap"):
        activation_residuals(frame(), ["target"], ["target"])


def test_unknown_rhythm_and_zero_scale_abstain():
    data = frame(8).assign(
        subject_id=1,
        split="train",
        rhythm_context="unknown",
        transform="kors",
        ecg_time=pd.date_range("2020-01-01", periods=8),
        activation=1,
    )
    out = personal_deviations(data, ["activation"])
    assert out.standardized_deviation.isna().all()
    assert out.status.eq("non_estimable").all()
    data["rhythm_context"] = "regular"
    out = personal_deviations(
        data, ["activation"], replace(ExtensionPolicy(), minimum_history=3)
    )
    assert out.standardized_deviation.isna().all()


def test_streamed_history_and_extension_job(tmp_path):
    import duckdb
    from vcg_biomarker.mimic.pre_run_extension_stage import (
        stream_personal_deviations,
        run_extension_job,
    )
    from vcg_biomarker.mimic.storage import file_sha256

    data = frame(8).assign(
        subject_id=1,
        split="train",
        rhythm_context="regular",
        transform="kors",
        ecg_time=pd.date_range("2020-01-01", periods=8),
        qrs_duration_ms=np.arange(8),
        extraction_status="ok",
    )
    with duckdb.connect() as connection:
        connection.register("features", data)
        paths, summary = stream_personal_deviations(
            connection, "features", set(data), tmp_path
        )
    assert summary["n_results"] == 8
    assert pd.read_parquet(paths[0]).iloc[-1].delta_previous == 1
    source = tmp_path / "input.parquet"
    frame().to_parquet(source)
    manifest = tmp_path / "job.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "input": source.name,
                "input_sha256": file_sha256(source),
                "tasks": [
                    {
                        "kind": "clinical_association",
                        "options": {
                            "marker": "activation",
                            "outcome": "target",
                            "covariates": [],
                        },
                    }
                ],
            }
        )
    )
    report = run_extension_job(manifest, tmp_path / "results")
    assert report["status"] == "complete"
    result = pd.read_parquet(tmp_path / "results" / "task-000.parquet")
    assert "q_global" in result
    with pytest.raises(FileExistsError):
        run_extension_job(manifest, tmp_path / "results")


def test_signal_sensitivity_uses_same_kernel_and_no_input_mutation():
    from vcg_biomarker.core.measurement_robustness import angle_sensitivity

    x = np.zeros((1000, 12))
    x[300:350, 0] = 1
    x[400:600, 1] = 0.4
    x[:, 2] = x[:, 1] - x[:, 0]
    beat = BeatDelineation(
        r_peak=MarkerPoint(325, 0.9, "test"),
        p_offset=MarkerPoint(250, 0.9, "test"),
        r_onset=MarkerPoint(300, 0.9, "test"),
        r_offset=MarkerPoint(350, 0.9, "test"),
        t_onset=MarkerPoint(400, 0.9, "test"),
        t_offset=MarkerPoint(600, 0.9, "test"),
        morphology_cluster_id=0,
    )
    before = x.copy()
    result = angle_sensitivity(
        x, SimpleNamespace(beats=[beat]), 500, source="test", record_id=1, force=True
    )
    assert result["status"] == "available"
    assert len(result["transform_comparison"]) == 4
    assert len(result["variants"]) == 10
    np.testing.assert_array_equal(x, before)
    assert result["baseline"]["area_angle_deg"] == pytest.approx(90)


def test_window_adapter_parallel_equality(monkeypatch):
    from vcg_biomarker.mimic import long_window_extraction as module

    def analyze(record, *args, **kwargs):
        return {
            "marker": float(record.signal_millivolts.mean()),
            "signal_quality_score": 0.9,
            "n_beats": 12,
            "extraction_status": "ok",
        }

    monkeypatch.setattr(module, "_analyze_record", analyze)
    signal = np.random.default_rng(3).normal(size=(30000, 12))
    kwargs = dict(
        source="incart",
        subject_id=1,
        record_id=1,
        split="train",
        markers=["marker"],
        powerline_hz=50,
    )
    a, _, _ = module.fixed_windows(signal, 500, workers=1, **kwargs)
    b, _, _ = module.fixed_windows(signal, 500, workers=2, **kwargs)
    pd.testing.assert_frame_equal(a, b)


def test_rank_stability_counts_patients_not_windows():
    from vcg_biomarker.mimic.pre_run_extensions import window_rank_stability

    rows = [
        dict(
            source="incart",
            subject_id=patient,
            record_id=patient,
            split="train",
            start_s=window * 10,
            duration_s=10,
            quality=0.9,
            marker=patient + window * 0.01,
        )
        for patient in range(32)
        for window in range(8)
    ]
    result = window_rank_stability(pd.DataFrame(rows), ["marker"])
    assert result.iloc[0].n_patients_complete == 32
    assert result.iloc[0].spearman_rho == pytest.approx(1)


def test_signed_time_sensitivity_and_missing_covariates():
    from vcg_biomarker.mimic.pre_run_extensions import clinical_time_sensitivity

    data = frame(200).assign(reference_delta_hours=-2.0)
    result = clinical_time_sensitivity(
        data, marker="activation", outcome="target", covariates=()
    )
    assert (
        result.loc[result.status.eq("estimated"), "time_stratum"]
        .eq("before_0_6h")
        .all()
    )
    assert (
        result.loc[result.time_stratum.eq("after_0_6h"), "status"]
        .eq("non_estimable")
        .all()
    )
    with pytest.raises(ValueError, match="missing_columns"):
        clinical_association(data, marker="activation", outcome="target")


def test_infinite_reference_time_is_unknown():
    data = frame(10).assign(
        reference_kind="lab",
        reference_value=1,
        reference_delta_hours=np.inf,
        setting="icu",
    )
    result = reference_timing_summary(data, marker="activation", endpoint="test")
    assert set(result.time_stratum) == {"unknown"}


def test_population_model_reader_excludes_archive_payloads(tmp_path):
    from vcg_biomarker.mimic.population_cohorts import _read_feature_parts
    path = tmp_path / "features.parquet"
    pd.DataFrame({"marker": [1.], "context_measurement_capsule_json": ["large archive"],
                  "context_measurement_sensitivity_json": ["large sensitivity"],
                  "context_temporal_research_json": ["needed temporal descriptor"]}).to_parquet(path)
    table = _read_feature_parts([path])
    assert list(table) == ["marker", "context_temporal_research_json"]
    assert "context_measurement_capsule_json" in pd.read_parquet(path)


def test_external_archive_is_separate_and_reconstructable(tmp_path, monkeypatch):
    from vcg_biomarker.mimic.external_validation import extract_external_features, ExternalExtractionOptions
    phase = (np.arange(5000) / 500) % 1
    signal = (np.exp(-((phase-.4)/.018)**2)[:, None] * np.linspace(.2, 1, 12)
              + np.exp(-((phase-.68)/.07)**2)[:, None] * np.linspace(.1, .4, 12)
              + np.exp(-((phase-.2)/.03)**2)[:, None] * .1)
    signal += np.random.default_rng(19).normal(0, .001, signal.shape)
    monkeypatch.setattr("vcg_biomarker.core.external_loader.load_external_ecg_result",
        lambda *args: SimpleNamespace(signal_millivolts=signal, sampling_rate_hz=500))
    inventory = pd.DataFrame({"source_id": ["georgia"], "record_id": ["synthetic"],
        "source_role": ["independent_external"], "header_path": ["synthetic.hea"],
        "sampling_rate_hz": [500], "eligible_for_diagnostic_validation": [True]})
    table = extract_external_features(inventory, ExternalExtractionOptions(marker_ids=("qrs_duration_ms",)),
                                      measurement_archive_root=tmp_path)
    assert table.extraction_status.iloc[0] == "ok"
    assert "context_measurement_capsule_json" not in table
    paths = table.attrs["measurement_archive_paths"]
    assert len(paths) == 1
    archive = pd.read_parquet(paths[0])
    assert archive.record_id.iloc[0] == "synthetic"
    capsule = json.loads(archive.context_measurement_capsule_json.iloc[0])
    assert len(capsule["beats"]) > 0


def test_capsule_interns_fiducial_strings_and_resolves_them_back():
    from vcg_biomarker.core.measurement_capsule import resolve_point

    beats = [
        BeatDelineation(
            r_peak=MarkerPoint(r, 0.83772455, "fusion_consensus:neurokit+multilead+vcg"),
            r_onset=MarkerPoint(r - 20, 0.80666667, "fusion_consensus:multilead+vcg"),
            r_offset=MarkerPoint(r + 20, 0.66, "fusion_consensus:neurokit+vcg+multilead"),
            morphology_cluster_id=0,
        )
        for r in (400, 900, 1400)
    ]
    capsule = measurement_capsule(
        np.ones((2000, 12)), SimpleNamespace(beats=beats), 500,
        source="test", record_id="1", raw_reference="raw", processing={},
    )
    encoded = json.dumps(capsule, allow_nan=False, separators=(",", ":"))
    # Each distinct source string appears exactly once, in the vocabulary.
    assert encoded.count("fusion_consensus:neurokit+multilead+vcg") == 1
    assert len(capsule["fiducial_vocabulary"]["sources"]) == 3
    point = capsule["beats"][2]["points"]["r_peak"]
    assert capsule["point_fields"][0] == "sample" and len(point) == len(capsule["point_fields"])
    assert isinstance(point[4], int)
    assert point[1] == 0.8377
    resolved = resolve_point(capsule, point)
    assert resolved["confidence"] == 0.8377
    # The 1.0.0 dict form stays readable.
    legacy = {"sample": 5, "confidence": 1.0, "source": "x", "evidence_sources": ["a"]}
    assert resolve_point(capsule, legacy)["source"] == "x"
    assert resolved["source"] == "fusion_consensus:neurokit+multilead+vcg"
    assert resolved["sample"] == 1400
    assert resolve_point(capsule, None) is None
