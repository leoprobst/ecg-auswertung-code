"""Audit 2026-09-23, section 5.2: new targets and endpoints."""
from __future__ import annotations

import duckdb
import pandas as pd

from vcg_biomarker.mimic import association_domains, evidence


def _write(path, frame):
    frame.to_parquet(path, index=False)
    return str(path).replace("\\", "/")


def test_hemodynamics_take_the_nearest_plausible_chart_value_within_six_hours(tmp_path):
    ecg_time = pd.Timestamp("2150-01-01 12:00")
    master = _write(tmp_path / "master.parquet", pd.DataFrame({
        "subject_id": [1, 1, 2],
        "study_id": [10, 11, 20],
        "ecg_time": [ecg_time, ecg_time + pd.Timedelta(days=3), ecg_time],
    }))
    chart = _write(tmp_path / "chart.parquet", pd.DataFrame({
        "subject_id": [1, 1, 1, 1, 1, 2],
        "itemid": [220074, 220074, 220074, 220088, 220061, 220074],
        "charttime": [
            ecg_time - pd.Timedelta(hours=2),    # CVP, 2 h before
            ecg_time + pd.Timedelta(hours=1),    # CVP, 1 h after -> nearest
            ecg_time + pd.Timedelta(minutes=30),  # CVP 999: artefact, dropped
            ecg_time + pd.Timedelta(hours=7),    # CO outside the window
            ecg_time - pd.Timedelta(hours=5),    # PAP mean, 5 h before
            ecg_time + pd.Timedelta(hours=3),
        ],
        "valuenum": [8.0, 12.0, 999.0, 5.0, 30.0, 4.0],
    }))
    query = evidence._icu_hemodynamics_query(master, f"read_parquet('{chart}')")
    frame = duckdb.connect().execute(query).df().set_index("study_id")
    assert frame.loc[10, "icu_cvp_nearest"] == 12.0
    assert frame.loc[10, "icu_cvp_delta_hours"] == 1.0
    assert frame.loc[10, "icu_pap_mean_nearest"] == 30.0
    assert pd.isna(frame.loc[10, "icu_cardiac_output_nearest"])
    assert 11 not in frame.index            # no chart value within 6 h
    assert frame.loc[20, "icu_cvp_nearest"] == 4.0

    empty = duckdb.connect().execute(evidence._icu_hemodynamics_query(master, None)).df()
    assert empty.empty and "icu_svo2_nearest" in empty.columns


def test_continuous_echo_and_haemodynamic_targets_are_registered():
    definitions = {
        item.association_id: item
        for item in association_domains.default_context_association_definitions()
    }
    for identifier in ("echo_lvef", "echo_la_volume_index", "echo_e_e_prime_ratio",
                       "echo_tapse", "echo_rvsp_pasp", "echo_lv_mass_index_devereux"):
        item = definitions[identifier]
        assert item.target_kind == "continuous"
        assert "has_tte_30d" in item.eligibility_sql
    for name in ("cvp", "pap_mean", "pcwp", "cardiac_output", "svo2"):
        item = definitions[f"icu_hemodynamics_{name}"]
        assert item.domain == "icu_invasive_hemodynamics"
        assert item.value_sql == f"TRY_CAST(icu_{name}_nearest AS DOUBLE)"


def test_conduction_code_groups_match_the_verified_codes():
    from vcg_biomarker.mimic.mappings import ICD_GROUPS, PROCEDURE_GROUPS

    groups = {group.group_id: group for group in ICD_GROUPS + PROCEDURE_GROUPS}
    avb, lbbb = groups["advanced_av_block"], groups["left_bundle_branch_block"]
    pacer = groups["permanent_pacemaker_implant"]
    assert avb.matches("426.0", 9) and avb.matches("42612", 9) and avb.matches("I442", 10)
    assert not avb.matches("42611", 9) and not avb.matches("I440", 10)  # first degree
    assert lbbb.matches("4263", 9) and lbbb.matches("I447", 10)
    assert not lbbb.matches("4264", 9)  # RBBB
    assert pacer.matches("3783", 9) and pacer.matches("0050", 9)
    assert not pacer.matches("3785", 9) and not pacer.matches("3778", 9)  # replacement, temporary
    assert pacer.matches("0JH606Z", 10) and pacer.matches("0JH837Z", 10)
    assert not pacer.matches("0JH608Z", 10)  # defibrillator generator
    assert not pacer.matches("02HV33Z", 10)  # central line into the SVC
    assert len(pacer.icd10_prefixes) == 16


def test_conduction_timing_separates_prior_untimed_and_next_evidence(tmp_path):
    t = pd.Timestamp("2150-03-10 10:00")
    master = _write(tmp_path / "m.parquet", pd.DataFrame({
        "subject_id": [1, 2, 3, 4], "study_id": [10, 20, 30, 40], "hadm_id": [100, 200, 300, 400],
        "ecg_time": [t, t, t, t]}))
    admissions = _write(tmp_path / "a.parquet", pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4], "hadm_id": [100, 101, 200, 300, 401],
        "admittime": [t - pd.Timedelta(days=1), t + pd.Timedelta(days=40),
                      t - pd.Timedelta(days=1), t - pd.Timedelta(days=1),
                      t + pd.Timedelta(days=90)],
        "dischtime": [t + pd.Timedelta(days=2), t + pd.Timedelta(days=45),
                      t + pd.Timedelta(days=2), t + pd.Timedelta(days=2),
                      t + pd.Timedelta(days=92)]}))
    diagnoses = _write(tmp_path / "d.parquet", pd.DataFrame({
        "subject_id": [1, 2, 4], "hadm_id": [101, 200, 401],
        "code_group": ["left_bundle_branch_block", "advanced_av_block", "cardiac_device_status"]}))
    procedures = _write(tmp_path / "p.parquet", pd.DataFrame({
        "subject_id": [3, 3], "hadm_id": [300, 300],
        "chartdate": [t.normalize(), t.normalize() + pd.Timedelta(days=2)],
        "code_group": ["permanent_pacemaker_implant", "permanent_pacemaker_implant"]}))
    frame = duckdb.connect().execute(
        evidence._conduction_timing_query(master, diagnoses, admissions, procedures)
    ).df().set_index("subject_id")
    # 1: LBBB coded in a later admission -> next event 40 days after the ECG.
    assert not frame.loc[1, "conduction_prior_evidence"]
    assert frame.loc[1, "next_conduction_dx_admission_time"] == t + pd.Timedelta(days=40)
    # 2: AV block coded in the index admission -> untimed, not prior.
    assert frame.loc[2, "conduction_index_untimed"] and not frame.loc[2, "conduction_prior_evidence"]
    # 3: pacemaker on the ECG date is untimed; the one two days later is the next event.
    assert frame.loc[3, "conduction_index_untimed"]
    assert frame.loc[3, "next_pacemaker_date"] == t.normalize() + pd.Timedelta(days=2)
    # 4: device-in-situ code in a later admission -> untimed later device.
    assert frame.loc[4, "next_device_status_admission_time"] == t + pd.Timedelta(days=90)


def test_conduction_progression_rule_states():
    from vcg_biomarker.mimic.labels import _label_rules

    rule = _label_rules()["conduction_progression_1y"]
    frame = pd.DataFrame({
        "current_report_lbbb_state": [None, "positive", None, None, None, None],
        "current_report_pacemaker_state": [None] * 6,
        "conduction_prior_evidence": [False, False, False, False, False, False],
        "conduction_index_untimed": [False, False, True, False, False, False],
        "next_conduction_event_delta_hours": [24 * 40, None, None, None, None, None],
        "next_device_status_delta_hours": [None, None, None, 24 * 90, None, None],
        "death_delta_days": [None, None, None, None, 100, None],
        "followup_hours": [24 * 400, 24 * 400, 24 * 400, 24 * 400, 24 * 400, 24 * 100],
    })
    with duckdb.connect() as connection:
        connection.register("f", frame)
        states = [row[0] for row in connection.execute(f"SELECT {rule.state} FROM f").fetchall()]
    assert states == ["positive", "indeterminate", "indeterminate", "indeterminate",
                      "indeterminate", "indeterminate"]
    frame.loc[0, "next_conduction_event_delta_hours"] = None
    with duckdb.connect() as connection:
        connection.register("f", frame)
        first = connection.execute(f"SELECT {rule.state} FROM f").fetchall()[0][0]
    assert first == "negative"


def _spec():
    from types import SimpleNamespace

    # The attributes the subgroup functions read, at ValidationSpec's defaults.
    return SimpleNamespace(min_class_inference=30, min_class_primary=100, random_seed=20260725)


def _era_frame(n, era, shift, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    return pd.DataFrame({
        "subject_id": np.arange(seed * 100_000, seed * 100_000 + n),
        "label": labels.astype("int8"),
        "marker": rng.normal(0, 1, n) + shift * labels,
        "anchor_year_group": era,
        "age": rng.uniform(30, 90, n),
        "sex": rng.choice(["F", "M"], n),
        "race": rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN"], n),
        "machine_qrs_duration_ms": 100.0,
        "bmi_before_ecg": 27.0,
        "creatinine_nearest": 1.0,
        "ed_stay_id": None,
        "hadm_id": 1,
    })


def test_temporal_validation_fits_early_and_reports_drift():
    from vcg_biomarker.mimic import validation

    train = pd.concat([_era_frame(600, "2008 - 2010", 1.5, 1), _era_frame(600, "2017 - 2019", 0.2, 2)])
    held_out = pd.concat([_era_frame(600, "2011 - 2013", 1.5, 3), _era_frame(600, "2014 - 2016", 0.2, 4)])
    spec = _spec()
    rows = validation._temporal_validation_records(
        "endpoint", "marker", train, held_out, spec, bootstrap_iterations=50)
    by_level = {row["subgroup_level"]: row for row in rows}
    assert set(by_level) == {"validation_early_eras", "validation_late_eras"}
    assert all(row["threshold_source"] == "train_early_eras_only" for row in rows)
    assert all(row["n_train_early"] == 600 for row in rows)
    early, late = by_level["validation_early_eras"], by_level["validation_late_eras"]
    assert early["roc_auc"] > 0.8 and late["roc_auc"] < 0.65      # drift is visible
    assert early["roc_auc_ci_low"] < early["roc_auc"] < early["roc_auc_ci_high"]


def test_fairness_subgroups_carry_intervals_and_technical_ones_do_not():
    from vcg_biomarker.mimic import validation
    from vcg_biomarker.mimic.statistics import fit_marker_model

    frame = _era_frame(1200, "2014 - 2016", 1.0, 5)
    model = fit_marker_model(frame["marker"].to_numpy(), frame["label"].to_numpy(), "marker")
    rows = validation._subgroup_stability_records(
        "endpoint", "marker", model, frame, _spec(), bootstrap_iterations=50)
    by_variable = {}
    for row in rows:
        by_variable.setdefault(row["subgroup_variable"], []).append(row)
    for variable in ("sex", "age", "race_performance_audit"):
        assert all(row["interval_scope"] == "fairness_subgroup_subject_cluster_bootstrap"
                   for row in by_variable[variable]), variable
        assert all(row["roc_auc_ci_low"] < row["roc_auc_ci_high"] for row in by_variable[variable])
    assert all(row["interval_scope"] == "point_estimate_only" for row in by_variable["qrs_duration_ms"])


def _km_censoring_before(time, event, t):
    """Textbook product-limit of the censoring distribution, G(t-)."""
    import numpy as np

    value = 1.0
    for u in np.unique(time):
        if u >= t:
            break
        at_risk = np.sum(time >= u)
        censored = np.sum((time == u) & (event == 0))
        value *= 1.0 - censored / at_risk
    return value


def test_uno_c_and_time_dependent_auc_match_the_pairwise_definitions():
    import numpy as np

    from vcg_biomarker.mimic.survival import cumulative_dynamic_auc, uno_concordance

    rng = np.random.default_rng(7)
    n = 120
    risk = np.round(rng.normal(0, 1, n), 1)                     # ties in the marker
    time = np.round(rng.exponential(200 / np.exp(risk)), 0) + 1  # ties in time
    event = (rng.uniform(size=n) < 0.6).astype(float)
    tau = 250.0
    num = den = 0.0
    for i in range(n):
        if event[i] == 0 or time[i] >= tau:
            continue
        w = 1.0 / _km_censoring_before(time, event, time[i]) ** 2
        for j in range(n):
            if time[j] > time[i]:
                den += w
                num += w * (1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0)
    uno, used = uno_concordance(time, event, risk, tau=tau)
    assert abs(uno - num / den) < 1e-12 and used > 0

    horizon = 150.0
    cases = [i for i in range(n) if event[i] > 0 and time[i] <= horizon]
    controls = [j for j in range(n) if time[j] > horizon]
    num = sum((1.0 / _km_censoring_before(time, event, time[i])) *
              sum(1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0 for j in controls)
              for i in cases)
    den = sum(1.0 / _km_censoring_before(time, event, time[i]) for i in cases) * len(controls)
    assert abs(cumulative_dynamic_auc(time, event, risk, horizon=horizon) - num / den) < 1e-12


def test_landmark_restarts_the_clock_at_discharge_and_drops_in_hospital_deaths():
    from vcg_biomarker.mimic.survival import landmark_from_index_discharge

    frame = pd.DataFrame({
        "time_days": [100.0, 3.0, 50.0, 20.0],
        "event": [0, 1, 1, 0],
        "days_to_index_discharge": [5.0, 5.0, None, 20.0],
    })
    result = landmark_from_index_discharge(frame)
    assert result["time_days"].tolist() == [95.0]   # died in hospital, no admission, censored at discharge


def test_covariate_adjusted_records_cover_both_covariate_sets():
    from vcg_biomarker.mimic import validation

    train = _era_frame(1500, "2014 - 2016", 1.0, 6).assign(machine_heart_rate_bpm=75.0)
    held_out = _era_frame(1500, "2014 - 2016", 1.0, 7).assign(machine_heart_rate_bpm=75.0)
    rows = validation._covariate_adjusted_records("no_such_endpoint", "marker", 1, train, held_out)
    assert {row["covariate_set"] for row in rows} == {"age_sex", "age_sex_heart_rate"}
    for row in rows:
        assert row["analysis_type"] == "covariate_adjusted_auc"
        assert abs(row["covariate_adjusted_auc"] - row["unadjusted_auc_same_rows"]) < 0.05


def test_label_measurement_sensitivity_counts_development_only(tmp_path):
    from types import SimpleNamespace

    from vcg_biomarker.mimic.labels import measurement_sensitivity_summary

    master_root, evidence_root = tmp_path / "master", tmp_path / "evidence"
    master_root.mkdir(); evidence_root.mkdir()
    t = pd.Timestamp("2150-01-01")
    pd.DataFrame({"subject_id": [1, 1, 2, 3], "study_id": [10, 11, 20, 30],
                  "split": ["train", "train", "validation", "test"],
                  "ecg_time": [t, t, t, t]}).to_parquet(master_root / "ecg_master_index.parquet")
    pd.DataFrame({"subject_id": [1, 2, 3], "lab_group": ["potassium"] * 3,
                  "value_status": ["hemolysis_reported"] * 3}).to_parquet(evidence_root / "labs.parquet")
    pd.DataFrame({"subject_id": [1, 3], "study_id": [10, 30],
                  "serial_interval_state": ["positive", "negative"],
                  "undetectable_rise_state": ["rise_from_undetectable", "not_applicable"],
                  }).to_parquet(evidence_root / "ecg_troponin_serial.parquet")
    config = SimpleNamespace(master_root=master_root, evidence_root=evidence_root)
    summary = measurement_sensitivity_summary(config)
    assert summary["hemolysis_excluded_results_by_lab_group"] == [{"lab_group": "potassium", "results": 2}]
    assert summary["troponin_interval_states"] == [
        {"serial_interval_state": "positive", "undetectable_rise_state": "rise_from_undetectable", "ecgs": 1}]
    assert summary["same_subject_same_time_ecg_groups"] == [{"groups": 1, "ecgs": 2}]
    assert summary["vital_status_followup_below_30_days_without_death"]["status"] == "source_unavailable"
