from types import SimpleNamespace
from pathlib import Path
import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.score_stage import load_score_cohort, run_score_discovery
from vcg_biomarker.mimic.score_stage import classical_negative_mask, NORMALITY_FLAGS
from vcg_biomarker.mimic.score_stage import score_modality, DIRECT_ECG_GEOMETRY
from vcg_biomarker.mimic.score_research import SearchPolicy
from vcg_biomarker.mimic.workflow import guided_study_spec


def test_only_new_full_specs_include_bounded_followup():
    full = guided_study_spec(preset_id="recommended_clinical", run_id="test")
    assert full.stage_ids[-1] == "audit"
    assert full.stage_ids.index("score-discovery") < full.stage_ids.index("reporting-guideline")
    assert "deep-combination-search" not in full.stage_ids
    pilot = guided_study_spec(
        preset_id="recommended_clinical", run_id="test", stop_after_pilots=True
    )
    assert "score-discovery" not in pilot.stage_ids


def test_direct_eight_lead_geometry_is_not_attributed_to_vcg_transformation():
    assert all(
        score_modality(name) == "direct_ecg_geometry" for name in DIRECT_ECG_GEOMETRY
    )
    assert score_modality("classical_cornell_voltage") == "classical_ecg"


def test_index_reference_missingness_and_patient_boundary(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # The unknown index of patient 2 must not be replaced by a later verified ECG.
    cohort = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [11, 21, 31, 41],
            "label_id": ["ef"] * 4,
            "split": ["train", "train", "train", "validation"],
            "label_state": ["positive", "unavailable", "negative", "positive"],
            "eligibility_state": ["eligible"] * 4,
            "source_tables": ["labevents"] * 4,
            "evidence_ids_json": ['["x1"]', "[]", '["x3"]', '["x4"]'],
            "time_delta_hours": [0.0] * 4,
            "time_precision": ["timestamp"] * 4,
        }
    )
    cohort.to_parquet(labels / "endpoint_index_cohorts.parquet")
    features = tmp_path / "features" / "split=train"
    features.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "study_id": [11, 22],
            "split": ["train"] * 2,
            "extraction_status": ["ok"] * 2,
            "qrs_area": [1.0, 999.0],
        }
    ).to_parquet(features / "part-0.parquet")
    config = SimpleNamespace(
        labels_root=labels,
        catalog_path=tmp_path / "catalog.duckdb",
        workers=1,
        duckdb_memory_limit_gb=1,
    )
    frame, counts = load_score_cohort(
        config, str(features / "*.parquet"), "ef", ["qrs_area"]
    )
    assert counts["n_index"] == 3
    assert counts["n_verified_reference"] == 2
    assert counts["n_extraction_ok"] == 1
    assert frame.subject_id.tolist() == [1, 3]
    assert frame.qrs_area.isna().sum() == 1
    assert counts["marker_availability_before_filter"]["qrs_area"] == 0.5

    # Only demographics and index-time care links, never subsequent labs.
    master = tmp_path / "master"
    master.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 3],
            "study_id": [11, 31],
            "split": ["train"] * 2,
            "age": [60.0, 70.0],
            "sex": ["F", "M"],
            "machine_heart_rate_bpm": [70.0, 80.0],
            "icustay_id": [1.0, None],
            "ed_stay_id": [None, 3.0],
            "hadm_id": [1.0, 2.0],
        }
    ).to_parquet(master / "ecg_master_index.parquet")
    features_frame = pd.read_parquet(features / "part-0.parquet")
    features_frame["signal_quality_score"] = 0.9
    features_frame.to_parquet(features / "part-0.parquet")
    config.master_root = master
    clinical, counts = load_score_cohort(
        config, str(features / "*.parquet"), "ef", ["qrs_area"]
    )
    assert counts["clinical_context_available"]
    assert clinical.context_age.tolist() == [60.0, 70.0]
    assert clinical.context_sex_male.tolist() == [0.0, 1.0]
    assert clinical.context_icu.tolist() == [1.0, 0.0]
    # Additional residual-model projections obey the same extraction-QC gate
    # even when these columns were not requested by score selection.
    features_frame["qrs_duration_ms"] = 120.
    features_frame["t_loop_width_major"] = .2
    features_frame["context_heart_rate_bpm"] = 70.
    features_frame["rhythm_context"] = "regular"
    features_frame["qrs_t_normalized_discrete_frechet"] = .3
    features_frame["extraction_status"] = "failed"
    features_frame.to_parquet(features / "part-0.parquet")
    failed, _ = load_score_cohort(config, str(features / "*.parquet"), "ef", ["qrs_area"])
    assert failed[["qrs_duration_ms", "t_loop_width_major", "context_heart_rate_bpm", "rhythm_context",
                   "qrs_t_normalized_discrete_frechet"]].isna().all().all()


def test_missing_inputs_write_honest_nonclinical_report(tmp_path):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "db", work_root=tmp_path / "work"
    )
    result = run_score_discovery(config, run_id="synthetic", max_endpoints=1)
    assert result.status == "complete_without_eligible_inputs"
    report = next(
        p for p in result.output_paths if p.endswith("score_research_report.json")
    )
    content = json.loads(Path(report).read_text(encoding="utf-8"))
    assert not content["clinical_claim_allowed"]
    assert content["localization"]["status"] == "missing_mapping_reference"
    assert all(row["status"] == "preflight_abstention" for row in content["results"])


def test_locked_paths_rejected_before_any_parquet_read(tmp_path):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "db", work_root=tmp_path / "work"
    )
    path = tmp_path / "split=locked_test"
    path.mkdir()
    (path / "part-0.parquet").write_bytes(b"not parquet; must not be opened")
    with pytest.raises(ValueError, match="locked_feature_paths"):
        run_score_discovery(
            config, run_id="synthetic", feature_glob=str(path / "*.parquet")
        )


def test_classical_negative_requires_every_prespecified_measurement():
    frame = pd.DataFrame({name: [0.0, 0.0] for name in NORMALITY_FLAGS})
    frame.loc[1, NORMALITY_FLAGS[0]] = float("nan")
    mask, rule = classical_negative_mask(frame)
    assert mask.tolist() == [True, False]
    assert rule["n_assessable"] == 1
    assert not classical_negative_mask(frame.drop(columns=[NORMALITY_FLAGS[0]]))[
        0
    ].any()


def test_stage_writes_frozen_formulas_and_reuses_verified_checkpoint(
    tmp_path, monkeypatch
):
    import numpy as np
    import vcg_biomarker.mimic.score_stage as module
    import vcg_biomarker.mimic.validation as validation

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "db", work_root=tmp_path / "work"
    )
    config.labels_root.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_parquet(
        config.labels_root / "endpoint_index_cohorts.parquet"
    )
    features = tmp_path / "features" / "split=train"
    features.mkdir(parents=True)
    pd.DataFrame({"placeholder": [1]}).to_parquet(features / "part-0.parquet")
    rng = np.random.default_rng(8)
    x = rng.normal(size=100)
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(100),
            "study_id": np.arange(100),
            "label_state": np.where(
                x + rng.normal(size=100) > 0, "positive", "negative"
            ),
            "qrs_area": x,
            "classical_cornell_voltage": x + rng.normal(size=100),
        }
    )
    monkeypatch.setattr(
        validation,
        "verify_feature_artifacts",
        lambda paths: {"artifact_hash": "synthetic"},
    )
    monkeypatch.setattr(
        validation, "_verify_current_development_feature_inputs", lambda *a, **k: None
    )
    monkeypatch.setattr(
        validation, "_resolved_feature_scope", lambda *a: "development_only"
    )
    monkeypatch.setattr(
        validation, "verify_development_label_attestation", lambda c: {"verified": True}
    )
    monkeypatch.setattr(
        module, "load_score_cohort", lambda *a, **k: (frame, {"n_index": 100})
    )
    monkeypatch.setattr(
        module,
        "default_feature_panels",
        lambda: [
            SimpleNamespace(
                panel_id="primary_core",
                feature_ids=("qrs_area", "classical_cornell_voltage"),
            )
        ],
    )
    kwargs = dict(
        run_id="synthetic",
        panel_ids=("primary_core",),
        feature_glob=str(features / "*.parquet"),
        endpoint_ids=("reduced_lvef_strict",),
        policy=SearchPolicy(
            outer_folds=2,
            inner_folds=2,
            max_components=1,
            minimum_class_patients=10,
            bootstrap_iterations=2,
            benchmark=False,
        ),
    )
    first = module.run_score_discovery(config, **kwargs)
    assert first.status == "complete"
    report = json.loads(
        Path(
            next(
                p
                for p in first.output_paths
                if p.endswith("score_research_report.json")
            )
        ).read_text()
    )
    assert "score_spec" in report["results"][0]["models"]["ecg"]
    assert "paired_comparisons" in report["results"][0]
    monkeypatch.setattr(
        module,
        "load_score_cohort",
        lambda *a, **k: pytest.fail("cached result recomputed"),
    )
    second = module.run_score_discovery(config, **kwargs)
    assert first.input_hash == second.input_hash
    assert first.created_at == second.created_at
