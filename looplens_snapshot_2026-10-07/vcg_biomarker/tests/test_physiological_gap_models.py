from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.physiological_gap_models import (
    GapMarkerModel,
    MINIMUM_FEATURE_COVERAGE,
    SexDiscordanceModel,
    apply_gap_model,
    apply_sex_discordance_model,
    fit_gap_marker_model,
    fit_sex_discordance_model,
    load_physiological_gap_models,
    load_sex_discordance_model,
    _fetch_gap_population_in_blocks,
    run_physiological_gap_discovery,
    target_column_for,
)


class _CountingDuckDbConnection:
    def __init__(self) -> None:
        self.connection = duckdb.connect(":memory:")
        self.queries: list[str] = []

    def execute(self, query: str):
        self.queries.append(query)
        return self.connection.execute(query)

    def close(self) -> None:
        self.connection.close()


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def test_gap_population_single_block_uses_one_feature_scan(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "subject_id": [2, 1, 3],
            "study_id": [20, 10, 30],
            "split": ["validation", "train", "locked_test"],
            "extraction_status": ["ok", "ok", "ok"],
            "feature_a": [2.0, 1.0, 3.0],
            "feature_b": [20.0, 10.0, 30.0],
        }
    )
    context = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [10, 20, 30],
            "split": ["train", "validation", "locked_test"],
            "age": [40.0, 50.0, 60.0],
            "bmi_before_ecg": [20.0, 25.0, 30.0],
            "sex": ["F", "M", "F"],
        }
    )
    feature_path = tmp_path / "features.parquet"
    context_path = tmp_path / "context.parquet"
    features.to_parquet(feature_path, index=False)
    context.to_parquet(context_path, index=False)
    connection = _CountingDuckDbConnection()
    try:
        result = _fetch_gap_population_in_blocks(
            connection,
            feature_sql=str(feature_path).replace("\\", "/"),
            context_sql=str(context_path).replace("\\", "/"),
            selected=("feature_a", "feature_b"),
        )
    finally:
        connection.close()

    assert len(connection.queries) == 1
    assert result["subject_id"].tolist() == [1, 2]
    assert result["feature_a"].tolist() == [1.0, 2.0]
    assert result["feature_b"].tolist() == [10.0, 20.0]


def _true_signal_frame(
    n_subjects: int,
    *,
    seed: int,
    missing_fraction: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feature_a = rng.normal(0.0, 1.0, n_subjects)
    feature_b = rng.normal(0.0, 1.0, n_subjects)
    noise_feature = rng.normal(0.0, 1.0, n_subjects)
    age = 50.0 + 10.0 * feature_a + 3.0 * feature_b + rng.normal(
        0.0, 2.0, n_subjects
    )
    frame = pd.DataFrame(
        {
            "subject_id": np.arange(n_subjects),
            "study_id": np.arange(n_subjects),
            "age": age,
            "bmi_before_ecg": age / 2.0,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_noise": noise_feature,
        }
    )
    if missing_fraction > 0.0:
        mask = rng.random(n_subjects) < missing_fraction
        frame.loc[mask, "feature_a"] = np.nan
    return frame


def test_gap_model_recovers_true_linear_signal_on_validation() -> None:
    train = _true_signal_frame(600, seed=1)
    validation = _true_signal_frame(300, seed=2)
    model, train_rows, validation_rows = fit_gap_marker_model(
        train,
        validation,
        target_id="age_gap",
        feature_ids=("feature_a", "feature_b", "feature_noise"),
        minimum_train_subjects=50,
        random_seed=20260725,
    )
    assert model.validation_r2 > 0.5
    assert model.oof_method == "nested_grouped_cv"
    assert np.isfinite(model.validation_mae)
    assert validation_rows["evaluable"].all()
    # The model should lean on feature_a/feature_b (true signal) more than
    # the pure-noise feature; a fully collapsed (all-zero) model would mean
    # the ElasticNet penalty ate the real signal too.
    coefficients = dict(model.nonzero_features)
    assert "feature_a" in coefficients or "feature_b" in coefficients


def test_train_gaps_are_out_of_fold_not_in_sample_leakage() -> None:
    """Rows sharing identical, subject-duplicated features with a purely
    per-subject-random (feature-independent) target must NOT be predicted
    near-perfectly: that would only be possible if a subject's own rows
    leaked across the train/test boundary of a CV fold."""
    rng = np.random.default_rng(7)
    n_subjects = 40
    rows_per_subject = 4
    subject_ids = np.repeat(np.arange(n_subjects), rows_per_subject)
    feature_a = np.repeat(rng.normal(0.0, 1.0, n_subjects), rows_per_subject)
    feature_b = np.repeat(rng.normal(0.0, 1.0, n_subjects), rows_per_subject)
    # Target has no relationship whatsoever to the features.
    target = np.repeat(rng.normal(50.0, 10.0, n_subjects), rows_per_subject)
    train = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "study_id": np.arange(len(subject_ids)),
            "age": target,
            "feature_a": feature_a,
            "feature_b": feature_b,
        }
    )
    validation = _true_signal_frame(50, seed=3)
    validation["feature_noise"] = 0.0
    model, train_rows, _ = fit_gap_marker_model(
        train,
        validation.rename(columns={"feature_noise": "unused"}),
        target_id="age_gap",
        feature_ids=("feature_a", "feature_b"),
        minimum_train_subjects=20,
        n_folds=5,
        random_seed=20260725,
    )
    # With correct subject-grouped CV, features carrying zero true signal
    # cannot produce a strong out-of-fold fit.
    assert model.train_out_of_fold_r2 < 0.3
    assert model.oof_method == "nested_grouped_cv"


def test_rows_with_insufficient_feature_coverage_are_not_evaluable() -> None:
    train = _true_signal_frame(400, seed=4, missing_fraction=0.0)
    validation = _true_signal_frame(200, seed=5)
    # Blank out two of three features for a subset of validation rows so
    # coverage drops below the MINIMUM_FEATURE_COVERAGE threshold.
    sparse_rows = validation.index[:20]
    validation.loc[sparse_rows, "feature_a"] = np.nan
    validation.loc[sparse_rows, "feature_b"] = np.nan
    model, _, validation_rows = fit_gap_marker_model(
        train,
        validation,
        target_id="age_gap",
        feature_ids=("feature_a", "feature_b", "feature_noise"),
        minimum_train_subjects=50,
    )
    sparse_subjects = set(validation.loc[sparse_rows, "subject_id"])
    sparse_output = validation_rows[
        validation_rows["subject_id"].isin(sparse_subjects)
    ]
    assert not sparse_output["evaluable"].any()
    assert sparse_output["gap_value"].isna().all()
    assert MINIMUM_FEATURE_COVERAGE == pytest.approx(0.7)


def test_too_few_train_subjects_raises() -> None:
    train = _true_signal_frame(10, seed=6)
    validation = _true_signal_frame(10, seed=7)
    with pytest.raises(ValueError, match="usable train subjects"):
        fit_gap_marker_model(
            train,
            validation,
            target_id="age_gap",
            feature_ids=("feature_a", "feature_b"),
            minimum_train_subjects=200,
        )


def test_apply_gap_model_scores_a_single_case_without_refitting() -> None:
    train = _true_signal_frame(500, seed=8)
    validation = _true_signal_frame(200, seed=9)
    model, _, _ = fit_gap_marker_model(
        train,
        validation,
        target_id="age_gap",
        feature_ids=("feature_a", "feature_b", "feature_noise"),
        minimum_train_subjects=50,
    )
    single_case = pd.DataFrame(
        {
            "subject_id": [999999],
            "study_id": [999999],
            "age": [61.0],
            "feature_a": [1.2],
            "feature_b": [-0.4],
            "feature_noise": [0.1],
        }
    )
    scored = apply_gap_model(single_case, model)
    assert len(scored) == 1
    assert bool(scored["evaluable"].iloc[0])
    assert np.isfinite(scored["gap_value"].iloc[0])
    scored_again = apply_gap_model(single_case, model)
    assert scored["predicted_value"].iloc[0] == pytest.approx(
        scored_again["predicted_value"].iloc[0]
    )


def test_gap_marker_model_round_trips_through_dict() -> None:
    train = _true_signal_frame(300, seed=10)
    validation = _true_signal_frame(150, seed=11)
    model, _, _ = fit_gap_marker_model(
        train,
        validation,
        target_id="bmi_gap",
        feature_ids=("feature_a", "feature_b"),
        minimum_train_subjects=50,
    )
    restored = GapMarkerModel.from_dict(model.to_dict())
    assert restored == model


def test_target_column_mapping_is_explicit() -> None:
    assert target_column_for("age_gap") == "age"
    assert target_column_for("bmi_gap") == "bmi_before_ecg"
    assert target_column_for("something_else") == "something_else"


def test_locked_test_split_is_never_referenced_in_module() -> None:
    import inspect

    from vcg_biomarker.mimic import physiological_gap_models as module

    source = inspect.getsource(module)
    assert "locked_test" not in source.casefold()


def _write_gap_discovery_project(tmp_path: Path) -> tuple[MimicProjectConfig, str]:
    config = _config(tmp_path)
    rng = np.random.default_rng(42)

    def _split_frame(n: int, split: str, offset: int) -> pd.DataFrame:
        feature_a = rng.normal(0.0, 1.0, n)
        feature_b = rng.normal(0.0, 1.0, n)
        age = 50.0 + 10.0 * feature_a + rng.normal(0.0, 2.0, n)
        subject_id = np.arange(offset, offset + n)
        sex_male = rng.integers(0, 2, n)
        return pd.DataFrame(
            {
                "subject_id": subject_id,
                "study_id": subject_id,
                "split": split,
                "age": age,
                "bmi_before_ecg": 25.0 + feature_b + rng.normal(0.0, 1.0, n),
                "sex": np.where(sex_male == 1, "M", "F"),
            }
        ), feature_a, feature_b

    context_rows = []
    feature_rows = []
    offset = 0
    for split, n in (("train", 400), ("validation", 150), ("locked_test", 20)):
        context_chunk, feature_a, feature_b = _split_frame(n, split, offset)
        context_rows.append(context_chunk)
        feature_rows.append(
            pd.DataFrame(
                {
                    "subject_id": context_chunk["subject_id"],
                    "study_id": context_chunk["study_id"],
                    "split": split,
                    "extraction_status": "ok",
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                }
            )
        )
        offset += n

    context_path = config.evidence_root / "clinical_context_features.parquet"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(context_rows, ignore_index=True).to_parquet(
        context_path, index=False
    )

    feature_dir = config.features_root / "primary_core" / "development"
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_path = feature_dir / "part-000000.parquet"
    pd.concat(feature_rows, ignore_index=True).to_parquet(
        feature_path, index=False
    )
    feature_glob = str(feature_dir / "part-*.parquet")
    return config, feature_glob


def test_gap_discovery_stage_persists_artifacts_and_skips_locked_test(
    tmp_path: Path,
) -> None:
    config, feature_glob = _write_gap_discovery_project(tmp_path)
    output_root = config.runs_root / "gap-test"

    manifest = run_physiological_gap_discovery(
        config,
        feature_glob=feature_glob,
        feature_ids=("feature_a", "feature_b"),
        output_root=output_root,
    )
    assert manifest.status == "complete"
    markers = pd.read_parquet(output_root / "physiological_gap_markers.parquet")
    assert set(markers["split"]) == {"train", "validation"}
    assert set(markers["target_id"]) == {"age_gap", "bmi_gap"}
    assert markers["subject_id"].max() < 550  # never reaches locked_test ids

    models = load_physiological_gap_models(output_root)
    assert set(models) == {"age_gap", "bmi_gap"}
    assert models["age_gap"].validation_r2 > 0.3

    sex_markers = pd.read_parquet(
        output_root / "sex_discordance_markers.parquet"
    )
    assert set(sex_markers["split"]) == {"train", "validation"}
    assert sex_markers["subject_id"].max() < 550

    sex_model = load_sex_discordance_model(output_root)
    assert sex_model is not None

    # Re-running with unchanged inputs must reuse the manifest, not
    # recompute (idempotent, hash-tracked like every other stage).
    second_manifest = run_physiological_gap_discovery(
        config,
        feature_glob=feature_glob,
        feature_ids=("feature_a", "feature_b"),
        output_root=output_root,
    )
    assert second_manifest.created_at == manifest.created_at


def _sex_signal_frame(
    n_subjects: int,
    *,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex_male = rng.integers(0, 2, n_subjects)
    # feature_a carries real signal about sex; feature_noise carries none.
    feature_a = np.where(
        sex_male == 1,
        rng.normal(1.0, 1.0, n_subjects),
        rng.normal(-1.0, 1.0, n_subjects),
    )
    feature_noise = rng.normal(0.0, 1.0, n_subjects)
    return pd.DataFrame(
        {
            "subject_id": np.arange(n_subjects),
            "study_id": np.arange(n_subjects),
            "sex": np.where(sex_male == 1, "M", "F"),
            "feature_a": feature_a,
            "feature_noise": feature_noise,
        }
    )


def test_sex_discordance_model_recovers_true_signal_on_validation() -> None:
    train = _sex_signal_frame(600, seed=20)
    validation = _sex_signal_frame(300, seed=21)
    model, train_rows, validation_rows = fit_sex_discordance_model(
        train,
        validation,
        feature_ids=("feature_a", "feature_noise"),
        minimum_train_subjects=50,
    )
    assert model.validation_auc > 0.7
    assert validation_rows["evaluable"].all()
    assert "discordant" in validation_rows
    # Most validation cases should NOT be discordant given real signal.
    assert validation_rows["discordant"].astype(bool).mean() < 0.5


def test_sex_discordance_out_of_fold_predictions_do_not_leak() -> None:
    rng = np.random.default_rng(22)
    n_subjects = 40
    rows_per_subject = 4
    subject_ids = np.repeat(np.arange(n_subjects), rows_per_subject)
    feature_a = np.repeat(rng.normal(0.0, 1.0, n_subjects), rows_per_subject)
    sex_male = np.repeat(rng.integers(0, 2, n_subjects), rows_per_subject)
    train = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "study_id": np.arange(len(subject_ids)),
            "sex": np.where(sex_male == 1, "M", "F"),
            "feature_a": feature_a,
        }
    )
    validation = _sex_signal_frame(50, seed=23)
    model, train_rows, _ = fit_sex_discordance_model(
        train,
        validation,
        feature_ids=("feature_a",),
        minimum_train_subjects=20,
        n_folds=5,
    )
    # feature_a carries zero true relationship to this synthetic sex
    # labeling (sex_male was drawn independently), so grouped, leakage-free
    # CV should not achieve a strong out-of-fold AUC.
    assert model.train_out_of_fold_auc < 0.75


def test_apply_sex_discordance_model_scores_single_case() -> None:
    train = _sex_signal_frame(500, seed=24)
    validation = _sex_signal_frame(200, seed=25)
    model, _, _ = fit_sex_discordance_model(
        train,
        validation,
        feature_ids=("feature_a", "feature_noise"),
        minimum_train_subjects=50,
    )
    single_case = pd.DataFrame(
        {
            "subject_id": [777],
            "study_id": [777],
            "sex": ["M"],
            "feature_a": [1.5],
            "feature_noise": [0.0],
        }
    )
    scored = apply_sex_discordance_model(single_case, model)
    assert len(scored) == 1
    assert bool(scored["evaluable"].iloc[0])
    assert 0.0 <= scored["predicted_probability_male"].iloc[0] <= 1.0


def test_sex_discordance_model_round_trips_through_dict() -> None:
    train = _sex_signal_frame(400, seed=26)
    validation = _sex_signal_frame(150, seed=27)
    model, _, _ = fit_sex_discordance_model(
        train,
        validation,
        feature_ids=("feature_a",),
        minimum_train_subjects=50,
    )
    restored = SexDiscordanceModel.from_dict(model.to_dict())
    assert restored == model
