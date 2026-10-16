from __future__ import annotations

from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.evidence import _ecg_medication_exposure_pair_query
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.mappings import AV_NODAL_BLOCKING_MEDICATION_CLASSES
from vcg_biomarker.mimic.therapy_response import (
    THERAPY_RESPONSE_VERSION,
    _therapy_response_pair_query,
    run_therapy_response_analysis,
    therapy_response_tachycardia_rate_change,
)


def test_adenosine_is_registered_as_av_nodal_blocking() -> None:
    assert "adenosine" in AV_NODAL_BLOCKING_MEDICATION_CLASSES


def test_ecg_medication_exposure_pair_query_selects_nearest_within_window(
    tmp_path: Path,
) -> None:
    master = pd.DataFrame(
        [
            # subject 1: clean before/after pair around a 09:00 exposure.
            {"subject_id": 1, "study_id": 101, "ecg_time": pd.Timestamp("2100-01-01 08:00")},
            {"subject_id": 1, "study_id": 102, "ecg_time": pd.Timestamp("2100-01-01 09:30")},
            {"subject_id": 1, "study_id": 103, "ecg_time": pd.Timestamp("2100-01-01 13:00")},
            # subject 2: only a before-ECG in window -- must be dropped.
            {"subject_id": 2, "study_id": 201, "ecg_time": pd.Timestamp("2100-01-01 08:00")},
            # subject 3: one ECG exactly at the exposure instant plus one
            # further ECG on each side.  The simultaneous ECG is neither a
            # baseline nor a post-exposure response, so the strict query must
            # retain the genuine side pair.
            {"subject_id": 3, "study_id": 301, "ecg_time": pd.Timestamp("2100-01-01 10:00")},
            {"subject_id": 3, "study_id": 302, "ecg_time": pd.Timestamp("2100-01-01 09:00")},
            {"subject_id": 3, "study_id": 303, "ecg_time": pd.Timestamp("2100-01-01 11:00")},
            # subject 4: no ECG anywhere near the exposure.
            {"subject_id": 4, "study_id": 401, "ecg_time": pd.Timestamp("2100-01-01 20:00")},
        ]
    )
    medications = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "medication_class": "adenosine",
                "event_time": pd.Timestamp("2100-01-01 09:00"),
                "source_table": "emar",
            },
            {
                "subject_id": 2,
                "medication_class": "adenosine",
                "event_time": pd.Timestamp("2100-01-01 09:00"),
                "source_table": "emar",
            },
            {
                "subject_id": 3,
                "medication_class": "adenosine",
                "event_time": pd.Timestamp("2100-01-01 10:00"),
                "source_table": "emar",
            },
            {
                "subject_id": 4,
                "medication_class": "adenosine",
                "event_time": pd.Timestamp("2100-01-01 12:00"),
                "source_table": "emar",
            },
            # non-qualifying source_table (multi-day active order, no
            # discrete administration instant) -- must not match.
            {
                "subject_id": 1,
                "medication_class": "adenosine",
                "event_time": pd.Timestamp("2100-01-01 09:00"),
                "source_table": "prescriptions",
            },
        ]
    )
    master_path = tmp_path / "master.parquet"
    medications_path = tmp_path / "medications.parquet"
    master.to_parquet(master_path, index=False)
    medications.to_parquet(medications_path, index=False)

    connection = duckdb.connect()
    query = _ecg_medication_exposure_pair_query(
        str(master_path), str(medications_path)
    )
    result = connection.execute(query).fetchdf()

    assert len(result) == 2
    row = result.loc[result["subject_id"].eq(1)].iloc[0]
    assert int(row["subject_id"]) == 1
    assert int(row["study_id_before"]) == 101
    assert int(row["study_id_after"]) == 102
    assert row["hours_before_to_exposure"] == pytest.approx(1.0)
    assert row["hours_exposure_to_after"] == pytest.approx(0.5)
    simultaneous = result.loc[result["subject_id"].eq(3)].iloc[0]
    assert int(simultaneous["study_id_before"]) == 302
    assert int(simultaneous["study_id_after"]) == 303
    assert simultaneous["hours_before_to_exposure"] == pytest.approx(1.0)
    assert simultaneous["hours_exposure_to_after"] == pytest.approx(1.0)


def test_therapy_response_pair_query_computes_deltas_and_normalization_flag(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "medication_class": "adenosine",
                "exposure_event_time": pd.Timestamp("2100-01-01 09:00"),
                "study_id_before": 101,
                "study_id_after": 102,
                "hours_before_to_exposure": 1.0,
                "hours_exposure_to_after": 0.5,
            },
            {
                "subject_id": 2,
                "medication_class": "beta_blocker",
                "exposure_event_time": pd.Timestamp("2100-01-01 10:00"),
                "study_id_before": 201,
                "study_id_after": 202,
                "hours_before_to_exposure": 0.5,
                "hours_exposure_to_after": 1.0,
            },
            {
                "subject_id": 3,
                "medication_class": "digoxin",
                "exposure_event_time": pd.Timestamp("2100-01-01 11:00"),
                "study_id_before": 301,
                "study_id_after": 302,
                "hours_before_to_exposure": 0.2,
                "hours_exposure_to_after": 0.2,
            },
        ]
    )
    features = pd.DataFrame(
        [
            # subject 1: wide-complex tachycardia before -> regular,
            # non-tachycardic after -> should normalize.
            {
                "subject_id": 1, "study_id": 101, "split": "train",
                "extraction_status": "ok", "rhythm_context": "wide_qrs_pattern",
                "heart_rate_bpm": 180.0,
                "narrow_complex_tachycardia_pattern_flag": 0.0,
                "wide_complex_tachycardia_pattern_flag": 1.0,
            },
            {
                "subject_id": 1, "study_id": 102, "split": "train",
                "extraction_status": "ok", "rhythm_context": "regular",
                "heart_rate_bpm": 75.0,
                "narrow_complex_tachycardia_pattern_flag": 0.0,
                "wide_complex_tachycardia_pattern_flag": 0.0,
            },
            # subject 2: narrow-complex tachycardia before AND after ->
            # rhythm_context stays "regular" throughout, but the
            # tachycardia pattern flag must still catch that it did not
            # normalize.
            {
                "subject_id": 2, "study_id": 201, "split": "validation",
                "extraction_status": "ok", "rhythm_context": "regular",
                "heart_rate_bpm": 170.0,
                "narrow_complex_tachycardia_pattern_flag": 1.0,
                "wide_complex_tachycardia_pattern_flag": 0.0,
            },
            {
                "subject_id": 2, "study_id": 202, "split": "validation",
                "extraction_status": "ok", "rhythm_context": "regular",
                "heart_rate_bpm": 160.0,
                "narrow_complex_tachycardia_pattern_flag": 1.0,
                "wide_complex_tachycardia_pattern_flag": 0.0,
            },
            # subject 3: never tachycardic.
            {
                "subject_id": 3, "study_id": 301, "split": "train",
                "extraction_status": "ok", "rhythm_context": "regular",
                "heart_rate_bpm": 80.0,
                "narrow_complex_tachycardia_pattern_flag": 0.0,
                "wide_complex_tachycardia_pattern_flag": 0.0,
            },
            {
                "subject_id": 3, "study_id": 302, "split": "train",
                "extraction_status": "ok", "rhythm_context": "regular",
                "heart_rate_bpm": 78.0,
                "narrow_complex_tachycardia_pattern_flag": 0.0,
                "wide_complex_tachycardia_pattern_flag": 0.0,
            },
            # decoy: failed extraction sharing an unrelated subject/study
            # id space -- must never leak into a join match.
            {
                "subject_id": 4, "study_id": 401, "split": "train",
                "extraction_status": "failed", "rhythm_context": None,
                "heart_rate_bpm": None,
                "narrow_complex_tachycardia_pattern_flag": None,
                "wide_complex_tachycardia_pattern_flag": None,
            },
        ]
    )
    pairs_path = tmp_path / "pairs.parquet"
    features_path = tmp_path / "features.parquet"
    pairs.to_parquet(pairs_path, index=False)
    features.to_parquet(features_path, index=False)

    connection = duckdb.connect()
    query = _therapy_response_pair_query(str(pairs_path), str(features_path))
    result = connection.execute(query).fetchdf().set_index("subject_id")

    assert bool(result.loc[1, "tachycardia_pattern_before"]) is True
    assert bool(result.loc[1, "tachycardia_pattern_after"]) is False
    assert bool(result.loc[1, "rhythm_normalized_flag"]) is True
    assert result.loc[1, "heart_rate_delta_bpm"] == pytest.approx(-105.0)

    assert bool(result.loc[2, "tachycardia_pattern_before"]) is True
    assert bool(result.loc[2, "tachycardia_pattern_after"]) is True
    assert bool(result.loc[2, "rhythm_normalized_flag"]) is False

    assert bool(result.loc[3, "tachycardia_pattern_before"]) is False
    assert bool(result.loc[3, "rhythm_normalized_flag"]) is False


def test_therapy_response_tachycardia_rate_change_bootstrap() -> None:
    pairs = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5, 6],
            "tachycardia_pattern_before": [True, True, True, True, True, True],
            "tachycardia_pattern_after": [
                False, False, False, False, True, True,
            ],
        }
    )
    result = therapy_response_tachycardia_rate_change(
        pairs, iterations=500, seed=11
    )
    assert result["n_pairs"] == 6
    assert result["n_subjects"] == 6
    assert result["rate_before"] == pytest.approx(1.0)
    assert result["rate_after"] == pytest.approx(2.0 / 6.0)
    assert result["rate_difference"] == pytest.approx(-4.0 / 6.0)
    assert np.isfinite(result["ci_low"])
    assert np.isfinite(result["ci_high"])
    assert result["ci_low"] <= result["rate_difference"] <= result["ci_high"]


def test_therapy_response_tachycardia_rate_change_handles_empty_input() -> None:
    empty = pd.DataFrame(
        {
            "subject_id": pd.Series(dtype=int),
            "tachycardia_pattern_before": pd.Series(dtype=float),
            "tachycardia_pattern_after": pd.Series(dtype=float),
        }
    )
    result = therapy_response_tachycardia_rate_change(empty)
    assert result["n_pairs"] == 0
    assert np.isnan(result["rate_difference"])
    assert np.isnan(result["ci_low"])


def test_therapy_response_stage_is_run_local_cached_and_scaling_pilot_safe(
    tmp_path: Path,
) -> None:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
        workers=1,
        duckdb_memory_limit_gb=2,
    )
    config.ensure_directories()
    pairs = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "medication_class": ["adenosine", "beta_blocker"],
            "exposure_event_time": pd.to_datetime(
                ["2100-01-01 09:00", "2100-01-01 10:00"]
            ),
            "study_id_before": [101, 201],
            "study_id_after": [102, 202],
            "hours_before_to_exposure": [1.0, 1.0],
            "hours_exposure_to_after": [0.5, 0.5],
        }
    )
    pairs.to_parquet(
        config.evidence_root / "ecg_medication_exposure_pairs.parquet",
        index=False,
    )
    (config.evidence_root / "evidence.manifest.json").write_text(
        '{"status":"complete","input_hash":"evidence-v1"}',
        encoding="utf-8",
    )
    feature_root = tmp_path / "features" / "development"
    partition = feature_root / "split=train"
    partition.mkdir(parents=True)
    features = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "study_id": [101, 102, 201, 202],
            "split": ["train"] * 4,
            "extraction_status": ["ok"] * 4,
            "rhythm_context": ["regular"] * 4,
            "heart_rate_bpm": [180.0, 80.0, 160.0, 150.0],
            "narrow_complex_tachycardia_pattern_flag": [1.0, 0.0, 1.0, 1.0],
            "wide_complex_tachycardia_pattern_flag": [0.0] * 4,
        }
    )
    from vcg_biomarker.core.temporal_research import VERSION as temporal_version
    temporal_descriptor = {
        "version": temporal_version, "source": "mimic", "subject_id": "1",
        "transform": "kors", "signal_path": "record_morphology",
        "qrs": {"arc_clock": np.linspace(0, 1, 33).tolist(), "duration_ms": 100},
    }
    features["context_temporal_research_json"] = [json.dumps(temporal_descriptor)] * 2 + [None] * 2
    features.to_parquet(partition / "part-000.parquet", index=False)
    (feature_root / "features.manifest.json").write_text(
        '{"status":"complete","input_hash":"features-v1"}',
        encoding="utf-8",
    )
    feature_glob = str(feature_root / "split=*" / "part-*.parquet")

    first = run_therapy_response_analysis(
        config,
        run_id="R001",
        feature_glob=feature_glob,
        iterations=0,
        seed=7,
    )
    output_root = config.runs_root / "R001" / "therapy_response"
    results = pd.read_parquet(output_root / "therapy_response_results.parquet")
    temporal = pd.read_parquet(output_root / "temporal_response_pairs.parquet")
    assert len(temporal) == 2
    assert temporal.loc[temporal.subject_id.eq(1), "status"].iloc[0] == "descriptive"
    assert temporal.loc[temporal.subject_id.eq(2), "status"].iloc[0] == "non_estimable"
    assert not temporal.clinical_claim_allowed.any()
    assert first.stage_version == THERAPY_RESPONSE_VERSION
    treated_rows = results.loc[results["scope"].ne("untreated_comparison")]
    assert set(treated_rows["uncertainty_method"]) == {"not_computed_scaling_pilot"}
    # Audit X7: without a master index there are no untreated pairs, and the
    # comparison says so instead of reporting an effect.
    untreated_rows = results.loc[results["scope"].eq("untreated_comparison")]
    assert set(untreated_rows["abstention_reason"]) == {"untreated_control_sources_unavailable"}
    assert set(results["run_id"]) == {"R001"}
    assert not (config.evidence_root / "therapy_response_summary.json").exists()
    assert not (config.evidence_root / "therapy_response_pairs.parquet").exists()

    cached = run_therapy_response_analysis(
        config,
        run_id="R001",
        feature_glob=feature_glob,
        iterations=0,
        seed=7,
    )
    assert cached.created_at == first.created_at

    # A cache entry is not trusted merely because all paths still exist.
    summary_path = output_root / "therapy_response_summary.json"
    summary_path.write_text("tampered", encoding="utf-8")
    repaired = run_therapy_response_analysis(
        config,
        run_id="R001",
        feature_glob=feature_glob,
        iterations=0,
        seed=7,
    )
    assert json.loads(summary_path.read_text(encoding="utf-8"))["run_id"] == "R001"
    assert repaired.output_integrity
