import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from vcg_biomarker.mimic.lead_comparison import compare_leads, paired_auc_samples
from vcg_biomarker.mimic.lead_measurements import empty_lead_measurements


def sample():
    rng = np.random.default_rng(14)
    n = 1800
    latent = rng.normal(size=n)
    outcome = rng.random(n) < 1/(1+np.exp(-latent))
    frame = pd.DataFrame({"subject_id": np.arange(n), "split": ["train"]*1100+["validation"]*700,
        "label_state": np.where(outcome, "positive", "negative"), "reference_eligible": True,
        "extraction_status": "ok", "age": rng.normal(60, 10, n), "sex_encoded": rng.integers(0, 2, n)})
    measures = {name: rng.normal(size=n) for name in empty_lead_measurements() if "_qrs_" in name}
    measures["lead_measurement_I_qrs_r_height_mv"] = latent
    return pd.concat([frame, pd.DataFrame(measures)], axis=1)


def run(frame, iterations=0):
    return compare_leads(frame, endpoint_id="synthetic", cohort_id="synthetic", reference_kind="simulation",
        family="qrs", covariates=("age", "sex_encoded"), iterations=iterations)


def test_shared_patients_train_only_anchor_and_validation_effect():
    frame = sample()
    result = run(frame, iterations=100)
    assert result.results.status.eq("estimable").all()
    assert result.results.n_validation_common.eq(700).all()
    row = result.results.set_index("lead").loc["I"]
    assert row.validation_auc > .7
    assert row.delta_auc_clinical_baseline > .1
    assert row.validation_auc_ci_low < row.validation_auc_ci_high
    assert row.anchor_selected_in_train == "I"
    assert row.delta_auc_train_anchor == 0.0
    assert row.delta_auc_train_anchor_ci_low == 0.0
    assert row.delta_auc_train_anchor_ci_high == 0.0
    expected_delta = result.results.validation_auc - row.validation_auc
    np.testing.assert_allclose(result.results.delta_auc_train_anchor, expected_delta)
    reversed_validation = frame.copy()
    mask = frame.split.eq("validation")
    reversed_validation.loc[mask, "label_state"] = frame.loc[mask, "label_state"].map({"positive": "negative", "negative": "positive"})
    second = run(reversed_validation)
    assert second.results.anchor_selected_in_train.eq("I").all()
    assert second.results.delta_auc_train_anchor_ci_low.isna().all()
    assert second.results.delta_auc_train_anchor_ci_high.isna().all()
    assert not result.results.monitoring_recommendation_allowed.any()


def test_missingness_reduces_every_lead_to_the_same_validation_patients():
    frame = sample()
    frame.loc[1100:1149, "lead_measurement_V6_qrs_r_height_mv"] = np.nan
    result = run(frame)
    assert result.results.n_validation_common.eq(650).all()
    assert result.predictions.subject_id.nunique() == 650


def test_cached_model_arrays_match_original_dataframe_selection(monkeypatch):
    import vcg_biomarker.mimic.lead_comparison as module
    frame = sample()
    # Nontrivial index ensures caching is positional only after patient masks.
    frame.index = np.arange(len(frame)) * 3 + 7
    original = module._fit
    seen = []

    def checked_fit(train, labels, validation):
        seen.append((train.copy(), labels.copy(), validation.copy()))
        return original(train, labels, validation)

    monkeypatch.setattr(module, "_fit", checked_fit)
    result = run(frame)
    for (name, model), (tr, y, va) in zip(result.models.items(), seen, strict=True):
        columns = model["columns"]
        np.testing.assert_array_equal(tr, frame.loc[frame.split.eq("train"), columns].to_numpy())
        np.testing.assert_array_equal(va, frame.loc[frame.split.eq("validation"), columns].to_numpy())
        np.testing.assert_array_equal(y, frame.loc[frame.split.eq("train"), "label_state"].eq("positive"))


def test_missingness_funnel_counts_overlapping_failures_only_once():
    frame = sample()
    frame.loc[0:3, "lead_measurement_V6_qrs_r_height_mv"] = np.nan
    frame.loc[0:2, "age"] = np.nan
    frame.loc[0:1, "extraction_status"] = "failed"
    frame.loc[0, "reference_eligible"] = False
    result = run(frame).results
    fields = ["n_excluded_reference", "n_excluded_extraction_after_reference",
              "n_excluded_covariates_after_extraction", "n_excluded_leads_after_covariates"]
    assert result[fields].eq(1).all().all()
    assert result.n_common_complete.eq(len(frame)-4).all()
    assert (result[fields].sum(axis=1)+result.n_common_complete).eq(result.n_index).all()


def test_bootstrap_matches_weighted_reference_and_preserves_ties():
    y = np.array([0, 1, 1, 0, 1, 0])
    scores = np.array([[.1, .6], [.2, .4], [.2, .4], [.4, .2], [.7, .2], [.7, .1]])
    actual = paired_auc_samples(y, scores, iterations=20, seed=71)
    weights = np.random.default_rng(71).multinomial(len(y), np.full(len(y), 1/len(y)), size=20)
    for index, weight in enumerate(weights):
        if weight[y == 0].sum() and weight[y == 1].sum():
            for column in range(2):
                assert actual[index, column] == pytest.approx(roc_auc_score(y, scores[:, column], sample_weight=weight))
        else:
            assert np.isnan(actual[index]).all()


def test_zero_iteration_lead_bootstrap_returns_without_score_preparation(
    monkeypatch,
):
    import vcg_biomarker.mimic.lead_comparison as module

    monkeypatch.setattr(
        module,
        "_prepare_weighted_bootstrap_score_order",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not sort")),
    )
    result = paired_auc_samples(
        np.array([0, 1]),
        np.array([[0.1, 0.2], [0.8, 0.9]]),
        iterations=0,
        seed=71,
    )
    assert result.shape == (0, 2)


def test_paired_identical_scores_have_exactly_zero_difference_in_every_draw():
    y = np.array([0, 1, 1, 0, 1, 0])
    score = np.array([.1, .2, .2, .4, .7, .7])
    draws = paired_auc_samples(y, np.column_stack([score, score]), iterations=150, seed=71)
    valid = np.isfinite(draws).all(axis=1)
    assert valid.sum() >= 100
    np.testing.assert_array_equal(draws[valid, 0] - draws[valid, 1], 0.0)


def test_locked_test_and_patient_overlap_are_rejected():
    frame = sample()
    frame.loc[0, "split"] = "locked_test"
    with pytest.raises(PermissionError): run(frame)
    frame.loc[0, "split"] = "train"
    frame.loc[1100, "subject_id"] = 0
    with pytest.raises(ValueError, match="one known patient"): run(frame)


@pytest.mark.parametrize("unknown", ["unknown", " NULL ", "nan", "n/a", "<NA>", True, False, np.inf])
def test_unknown_patient_tokens_are_rejected_before_model_fitting(unknown):
    frame = sample()
    frame["subject_id"] = frame.subject_id.astype(object)
    frame.loc[0, "subject_id"] = unknown
    with pytest.raises(ValueError, match="one known patient"):
        run(frame)


def test_separation_is_an_abstention_not_an_extreme_lead_claim():
    frame = sample()
    frame["lead_measurement_I_qrs_r_height_mv"] = frame.label_state.eq("positive").astype(float)
    row = run(frame).results.set_index("lead").loc["I"]
    assert row.status == "non_estimable"


def test_nullable_dtype_columns_do_not_break_the_fit():
    """A091's validate-pilot-1000 failed here: one covariate arrived as a
    nullable extension dtype, the split matrices became object arrays and
    ``std`` had no square root. Complete rows must fit regardless of the
    dtype the frame carries."""
    frame = sample()
    frame["sex_encoded"] = pd.array(frame["sex_encoded"].to_numpy(), dtype="Int64")
    frame["lead_measurement_V2_qrs_r_height_mv"] = pd.array(
        frame["lead_measurement_V2_qrs_r_height_mv"].to_numpy(), dtype="Float64")
    frame.loc[0:9, "lead_measurement_V2_qrs_r_height_mv"] = pd.NA
    result = run(frame)
    assert result.results.status.eq("estimable").all()
    assert result.results.n_validation_common.eq(700).all()
