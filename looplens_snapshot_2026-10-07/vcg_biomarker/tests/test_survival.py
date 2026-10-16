"""A Cox fit written out by hand has to be checked against the definitions.

Checks combine brute-force definitions, simulations with known coefficients,
and an independent Statsmodels PHReg reference for the Efron Cox fit. The
reference compares coefficients, model-based covariance and log likelihood;
it does not validate every cluster-robust or clinical sampling assumption.

One defect found this way and pinned below: the concordance index counted the
wrong half of each pair. It returned 0,471 for a covariate simulated with
HR = 2, which is exactly ``1 - 0,529`` and looks like nothing more than a weak
marker -- the kind of wrong answer that never announces itself.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.survival import (
    ConvergenceFailure,
    CoxFit,
    DEFAULT_HORIZON_DAYS,
    concordance_index,
    fit_cox,
    follow_up,
    survival_frame_query,
)


def _simulate(n=4000, beta=0.7, seed=0, ties=None):
    """Exponential survival with a known coefficient."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    hazard = 0.02 * np.exp(beta * x)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.exponential(60.0, n)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    if ties is not None:
        time = np.ceil(time / ties) * ties
    return pd.DataFrame(
        {
            "x": x,
            "time_days": np.maximum(time, 1e-6),
            "event": event,
            "subject_id": np.arange(n),
        }
    )


# --- The partial likelihood against its own definition ------------------------


@pytest.mark.parametrize("event", [2., -1., .5, np.inf, "dead", "1"])
def test_cox_rejects_nonbinary_events_before_complete_case_filter(event):
    frame = _simulate(n=200)
    frame["event"] = frame.event.astype(object)
    frame.loc[0, "event"] = event
    frame.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="binary"):
        fit_cox(frame, ("x",))


def test_missing_event_is_excluded_not_converted_to_survival():
    frame = _simulate(n=200)
    frame.loc[0, "event"] = np.nan
    actual = fit_cox(frame, ("x",))
    expected = fit_cox(frame.iloc[1:], ("x",))
    assert actual.n_observations == 199
    assert actual.n_events == expected.n_events
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)


def test_cox_efron_matches_independent_statsmodels_reference():
    from statsmodels.duration.hazard_regression import PHReg
    frame = _simulate(n=600, seed=762, ties=3.)
    actual = fit_cox(frame, ("x",), robust=False)
    reference = PHReg(frame.time_days.to_numpy(), frame[["x"]].to_numpy(),
                      status=frame.event.to_numpy(), ties="efron").fit()
    np.testing.assert_allclose(actual.coefficients, reference.params, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(actual.variance, reference.cov_params(), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(actual.log_likelihood, reference.llf, rtol=1e-8)


def _brute_force_efron(X, time, event, beta):
    """Efron's log partial likelihood, written straight from the formula."""

    eta = X @ beta
    weight = np.exp(eta)
    total = 0.0
    for moment in np.unique(time[event > 0]):
        at_risk = time >= moment
        tied = (time == moment) & (event > 0)
        d = int(tied.sum())
        s_risk = float(weight[at_risk].sum())
        s_tied = float(weight[tied].sum())
        total += float(eta[tied].sum())
        for k in range(d):
            total -= np.log(s_risk - (k / d) * s_tied)
    return float(total)


def test_the_partial_likelihood_matches_the_formula_with_no_ties():
    from vcg_biomarker.mimic.survival import _partial_likelihood, _risk_set_blocks

    frame = _simulate(n=200, seed=1)
    X = frame[["x"]].to_numpy()
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    beta = np.array([0.35])

    order, sorted_time, sorted_event, distinct, starts = _risk_set_blocks(time, event)
    value, _, _ = _partial_likelihood(
        X[order], sorted_time, sorted_event, distinct, starts, beta
    )
    # The implementation shifts eta before exponentiating, to keep exp() from
    # overflowing on a wide linear predictor. The shift cancels exactly: it
    # removes ``shift * D`` from the linear term and returns the same amount
    # through the D denominator terms, so the value is comparable as it is.
    expected = _brute_force_efron(X, time, event, beta)
    assert value == pytest.approx(expected, rel=1e-9)


def test_the_partial_likelihood_matches_the_formula_with_heavy_ties():
    """Follow-up is in whole days, so hundreds of deaths share a time."""

    from vcg_biomarker.mimic.survival import _partial_likelihood, _risk_set_blocks

    frame = _simulate(n=600, seed=2, ties=5.0)
    X = frame[["x"]].to_numpy()
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    assert len(np.unique(time[event > 0])) < int(event.sum()), "der Test braucht Bindungen"

    beta = np.array([0.4])
    order, sorted_time, sorted_event, distinct, starts = _risk_set_blocks(time, event)
    value, _, _ = _partial_likelihood(
        X[order], sorted_time, sorted_event, distinct, starts, beta
    )
    expected = _brute_force_efron(X, time, event, beta)
    assert value == pytest.approx(expected, rel=1e-9)


def test_the_overflow_guard_does_not_move_the_likelihood():
    """The eta shift must be exactly neutral, not approximately."""

    from vcg_biomarker.mimic.survival import _partial_likelihood, _risk_set_blocks

    frame = _simulate(n=400, seed=13, ties=4.0)
    X = frame[["x"]].to_numpy()
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    order, sorted_time, sorted_event, distinct, starts = _risk_set_blocks(time, event)

    # A large constant added to every covariate shifts eta wholesale; the
    # partial likelihood is invariant to it, and so must the implementation be.
    plain, _, _ = _partial_likelihood(
        X[order], sorted_time, sorted_event, distinct, starts, np.array([0.5])
    )
    shifted, _, _ = _partial_likelihood(
        (X + 1000.0)[order], sorted_time, sorted_event, distinct, starts,
        np.array([0.5]),
    )
    assert shifted == pytest.approx(plain, rel=1e-9)


def test_the_gradient_is_the_derivative_of_the_likelihood():
    """A wrong gradient still converges -- to the wrong place."""

    from vcg_biomarker.mimic.survival import _partial_likelihood, _risk_set_blocks

    frame = _simulate(n=300, seed=3, ties=3.0)
    frame["y"] = np.random.default_rng(9).normal(size=len(frame))
    X = frame[["x", "y"]].to_numpy()
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    order, sorted_time, sorted_event, distinct, starts = _risk_set_blocks(time, event)
    Xs = X[order]

    beta = np.array([0.3, -0.2])
    _, gradient, _ = _partial_likelihood(
        Xs, sorted_time, sorted_event, distinct, starts, beta
    )
    step = 1e-6
    for index in range(len(beta)):
        up, down = beta.copy(), beta.copy()
        up[index] += step
        down[index] -= step
        high, _, _ = _partial_likelihood(
            Xs, sorted_time, sorted_event, distinct, starts, up
        )
        low, _, _ = _partial_likelihood(
            Xs, sorted_time, sorted_event, distinct, starts, down
        )
        assert gradient[index] == pytest.approx((high - low) / (2 * step), rel=1e-4)


# --- The estimate ------------------------------------------------------------


def test_a_known_coefficient_is_recovered():
    fit = fit_cox(_simulate(beta=0.7), ["x"], robust=False)
    assert fit.coefficients[0] == pytest.approx(0.7, abs=3 * fit.standard_errors[0])
    assert fit.hazard_ratios[0] == pytest.approx(2.0, rel=0.1)


def test_a_null_coefficient_is_recovered_as_null():
    fit = fit_cox(_simulate(beta=0.0, seed=7), ["x"], robust=False)
    assert abs(fit.z_scores[0]) < 3
    assert fit.p_values[0] > 0.01


def test_the_estimate_survives_heavy_ties():
    """Efron exists for exactly this case; a biased fit would show here."""

    fit = fit_cox(_simulate(beta=0.7, seed=4, ties=10.0), ["x"], robust=False)
    assert fit.coefficients[0] == pytest.approx(0.7, abs=4 * fit.standard_errors[0])


def test_convergence_takes_a_handful_of_steps():
    fit = fit_cox(_simulate(), ["x"], robust=False)
    assert fit.steps <= 8, "eine konkave Likelihood braucht nicht mehr"


def test_a_duplicated_covariate_is_refused_loudly():
    frame = _simulate(n=500)
    frame["x_copy"] = frame["x"]
    with pytest.raises(ConvergenceFailure, match="singulaer"):
        fit_cox(frame, ["x", "x_copy"], robust=False)


def test_a_death_before_the_ecg_is_refused():
    """A negative follow-up is a linkage error, not a fast event."""

    frame = _simulate(n=300)
    frame.loc[0, "time_days"] = -5.0
    with pytest.raises(ValueError, match="Nachbeobachtungszeit"):
        fit_cox(frame, ["x"], robust=False)


def test_a_cohort_without_events_is_refused():
    frame = _simulate(n=300)
    frame["event"] = 0.0
    with pytest.raises(ValueError, match="Keine Ereignisse"):
        fit_cox(frame, ["x"], robust=False)


def test_no_covariates_is_refused():
    with pytest.raises(ValueError, match="Kovariaten"):
        fit_cox(_simulate(n=200), [], robust=False)


# --- Repeated measures -------------------------------------------------------


def test_repeating_every_patient_does_not_narrow_the_robust_interval():
    """The defect this guards against: 800.035 rows from 161.352 patients.

    Duplicating each row adds no information at all. A naive fit halves the
    standard error anyway; the sandwich must not.
    """

    frame = _simulate(n=1500, seed=5)
    doubled = pd.concat([frame, frame], ignore_index=True)
    single = fit_cox(frame, ["x"], robust=False)
    naive = fit_cox(doubled, ["x"], robust=False, cluster_column=None)
    robust = fit_cox(doubled, ["x"], robust=True)

    assert naive.standard_errors[0] < 0.8 * single.standard_errors[0], (
        "der naive Fit muss hier zu optimistisch sein, sonst prueft der Test "
        "nichts"
    )
    assert robust.standard_errors[0] == pytest.approx(
        single.standard_errors[0], rel=0.15
    )


def test_the_point_estimate_does_not_depend_on_the_variance_choice():
    frame = _simulate(n=1200, seed=6)
    naive = fit_cox(frame, ["x"], robust=False)
    robust = fit_cox(frame, ["x"], robust=True, cluster_column="subject_id")
    assert robust.coefficients[0] == pytest.approx(naive.coefficients[0], rel=1e-9)


def test_clustering_is_switched_on_by_itself_when_patients_repeat():
    frame = _simulate(n=800, seed=8)
    frame["subject_id"] = frame.index // 4
    fit = fit_cox(frame, ["x"])
    assert fit.robust is True
    assert fit.n_clusters == 200


def test_one_row_per_patient_needs_no_correction():
    fit = fit_cox(_simulate(n=800), ["x"])
    assert fit.robust is False
    assert fit.n_clusters == fit.n_observations


# --- Concordance -------------------------------------------------------------


def _brute_force_concordance(time, event, risk):
    concordant = 0.0
    comparable = 0
    for i, j in itertools.combinations(range(len(time)), 2):
        if time[i] == time[j]:
            continue
        first, second = (i, j) if time[i] < time[j] else (j, i)
        if event[first] <= 0:
            continue
        comparable += 1
        if risk[first] > risk[second]:
            concordant += 1.0
        elif risk[first] == risk[second]:
            concordant += 0.5
    return concordant / comparable, comparable


def test_concordance_matches_its_definition_exactly():
    frame = _simulate(n=300, seed=1)
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    risk = frame["x"].to_numpy()
    expected, expected_pairs = _brute_force_concordance(time, event, risk)
    measured, pairs = concordance_index(time, event, risk)
    assert measured == pytest.approx(expected, abs=1e-12)
    assert pairs == expected_pairs


def test_concordance_matches_its_definition_with_ties_in_both_axes():
    frame = _simulate(n=250, seed=2, ties=8.0)
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    risk = np.round(frame["x"].to_numpy(), 1)  # deliberate ties in the risk
    expected, expected_pairs = _brute_force_concordance(time, event, risk)
    measured, pairs = concordance_index(time, event, risk)
    assert measured == pytest.approx(expected, abs=1e-12)
    assert pairs == expected_pairs


def test_a_harmful_marker_scores_above_a_half():
    """The defect: this returned 0,471 for a covariate simulated at HR = 2."""

    frame = _simulate(beta=0.7)
    measured, _ = concordance_index(
        frame["time_days"].to_numpy(),
        frame["event"].to_numpy(),
        frame["x"].to_numpy(),
    )
    assert measured > 0.62, f"C = {measured:.3f} bei HR = 2 waere die falsche Haelfte"


def test_reversing_the_marker_reverses_the_index():
    frame = _simulate(beta=0.7)
    time = frame["time_days"].to_numpy()
    event = frame["event"].to_numpy()
    forward, _ = concordance_index(time, event, frame["x"].to_numpy())
    backward, _ = concordance_index(time, event, -frame["x"].to_numpy())
    assert forward + backward == pytest.approx(1.0, abs=1e-12)


def test_a_useless_marker_scores_a_half():
    frame = _simulate(beta=0.0, seed=11)
    measured, _ = concordance_index(
        frame["time_days"].to_numpy(),
        frame["event"].to_numpy(),
        np.random.default_rng(12).normal(size=len(frame)),
    )
    assert measured == pytest.approx(0.5, abs=0.02)


def test_concordance_without_events_is_not_a_number():
    value, pairs = concordance_index(
        np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, 2.0])
    )
    assert np.isnan(value) and pairs == 0


# --- Follow-up ---------------------------------------------------------------


def test_a_patient_without_a_death_date_is_censored_not_immortal():
    """Audit T1: survival is ascertained through the registry window.

    The registry covers a year after the last discharge, so a survivor
    discharged 40 days after the ECG is known alive to day 405 and is
    censored at the horizon, not at day 40. Under the last-contact
    sensitivity rule the censoring stays at the contact.
    """
    frame = pd.DataFrame(
        {"died": [0], "days_to_death": [np.nan], "days_to_last_seen": [40.0]}
    )
    result = follow_up(frame, horizon_days=365.0)
    assert int(result["event"].iloc[0]) == 0
    assert float(result["time_days"].iloc[0]) == 365.0
    contact = follow_up(frame, horizon_days=365.0, censoring="last_contact")
    assert float(contact["time_days"].iloc[0]) == 40.0
    shorter = follow_up(frame, horizon_days=365.0 * 2)
    assert float(shorter["time_days"].iloc[0]) == 405.0


def test_t1_both_outcomes_are_selected_by_the_same_rule():
    """Never admitted: excluded whether dead or alive (no registry coverage).
    Last admission before the ECG: kept while its registry window reaches past
    the ECG -- for survivors and deaths alike."""
    frame = pd.DataFrame({
        "died":              [0,      1,      0,     1],
        "days_to_death":     [np.nan, 50.0,   np.nan, 50.0],
        "days_to_last_seen": [np.nan, np.nan, -100.0, -100.0],
        "ever_admitted":     [0,      0,      1,     1],
    })
    result = follow_up(frame, horizon_days=365.0)
    assert result.index.tolist() == [2, 3]
    assert result["time_days"].tolist() == [265.0, 50.0]
    assert result["event"].tolist() == [0, 1]


def test_t1_last_contact_rule_drops_deaths_after_the_contact():
    frame = pd.DataFrame({
        "died": [1, 1], "days_to_death": [30.0, 200.0],
        "days_to_last_seen": [10.0, 10.0], "days_to_last_contact": [40.0, 40.0],
    })
    result = follow_up(frame, horizon_days=365.0, censoring="last_contact")
    assert result["event"].tolist() == [1, 0]
    assert result["time_days"].tolist() == [30.0, 40.0]


def test_a_death_past_the_horizon_is_censored_at_the_horizon():
    frame = pd.DataFrame(
        {"died": [1], "days_to_death": [400.0], "days_to_last_seen": [400.0]}
    )
    result = follow_up(frame, horizon_days=365.0)
    assert int(result["event"].iloc[0]) == 0
    assert float(result["time_days"].iloc[0]) == 365.0


def test_a_death_inside_the_horizon_is_an_event():
    frame = pd.DataFrame(
        {"died": [1], "days_to_death": [30.0], "days_to_last_seen": [10.0]}
    )
    result = follow_up(frame, horizon_days=365.0)
    assert int(result["event"].iloc[0]) == 1
    assert float(result["time_days"].iloc[0]) == 30.0


def test_a_death_recorded_before_the_ecg_is_dropped_not_clamped():
    """Clamping would turn a linkage error into an immediate event."""

    frame = pd.DataFrame(
        {"died": [1, 1], "days_to_death": [-5.0, 30.0],
         "days_to_last_seen": [-5.0, 30.0]}
    )
    result = follow_up(frame, horizon_days=365.0)
    assert len(result) == 1
    assert float(result["time_days"].iloc[0]) == 30.0


def test_a_death_before_the_ecg_is_dropped_even_with_a_later_contact():
    """The case the first test missed, and the defect it hid.

    ``observed`` required ``death >= 0``, so a negative death offset fell
    through to the censoring branch: the patient survived as censored at 300
    days, contributing person-time on the strength of a date known to be
    wrong. The original test set the contact to -5 as well, so the row was
    removed by the positive-time filter and the defect never showed.
    """

    frame = pd.DataFrame(
        {"died": [1], "days_to_death": [-5.0], "days_to_last_seen": [300.0]}
    )
    assert follow_up(frame, horizon_days=365.0).empty


def test_a_patient_who_never_died_keeps_their_follow_up():
    """The fix must not remove anyone whose death date is simply absent."""

    frame = pd.DataFrame(
        {"died": [0], "days_to_death": [np.nan], "days_to_last_seen": [120.0]}
    )
    result = follow_up(frame, horizon_days=365.0)
    assert len(result) == 1
    assert int(result["event"].iloc[0]) == 0


def test_the_default_horizon_is_one_year():
    assert DEFAULT_HORIZON_DAYS == 365.0


# --- The query ---------------------------------------------------------------


def test_the_query_censors_at_the_last_hospital_contact():
    sql = survival_frame_query(
        "master.parquet", "patients.parquet", "admissions.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "max(dischtime)" in sql
    assert "days_to_last_seen" in sql


def test_the_query_ranks_the_ecgs_per_patient():
    sql = survival_frame_query(
        "master.parquet", "patients.parquet", "admissions.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "ecg_rank" in sql
    assert "PARTITION BY b.subject_id ORDER BY b.ecg_time" in sql
    assert "b.ecg_time, b.study_id" in sql


def test_missing_sex_is_not_silently_encoded_as_male():
    sql = survival_frame_query(
        "master.parquet", "patients.parquet", "admissions.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "WHEN lower(m.sex) LIKE 'm%' THEN 0" in sql
    assert "ELSE NULL" in sql


def test_the_query_only_reads_successfully_extracted_ecgs():
    sql = survival_frame_query(
        "master.parquet", "patients.parquet", "admissions.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "extraction_status = 'ok'" in sql


def test_the_locked_test_split_is_never_read():
    sql = survival_frame_query(
        "master.parquet", "patients.parquet", "admissions.parquet",
        "read_parquet('f/*.parquet')", marker_columns="f.svg_magnitude",
    )
    assert "'train', 'validation'" in sql
    assert "locked_test" not in sql


# --- Reporting ---------------------------------------------------------------


def test_the_fit_reports_what_it_did():
    payload = fit_cox(_simulate(n=600), ["x"]).to_dict()
    for key in ("n_observations", "n_events", "n_clusters", "robust_variance",
                "ties", "version"):
        assert key in payload
    assert payload["ties"] == "efron"


def test_the_frame_carries_intervals_that_bracket_the_estimate():
    fit = fit_cox(_simulate(n=1500), ["x"])
    row = fit.to_frame().iloc[0]
    assert row["hr_lower"] < row["hazard_ratio"] < row["hr_upper"]
