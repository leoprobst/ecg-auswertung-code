"""Audit 5.3.3: empirical calibration against negative controls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from vcg_biomarker.mimic import empirical_calibration as ec


def test_hanley_mcneil_standard_error():
    # Hanley & McNeil 1982, worked example: AUC 0.5, 50/50 -> sqrt of 1/(12*50*50/101)
    se = ec.auc_standard_error(np.array([0.5]), np.array([50]), np.array([50]))[0]
    assert abs(se - np.sqrt((0.25 + 49 * (1 / 3 - 0.25) + 49 * (1 / 3 - 0.25)) / 2500)) < 1e-12


def test_fit_recovers_systematic_error_and_calibrates_the_null():
    rng = np.random.default_rng(3)
    mu, sigma = 0.03, 0.02
    se = rng.uniform(0.005, 0.02, 4000)
    estimates = rng.normal(mu, np.sqrt(sigma ** 2 + se ** 2))
    fitted_mu, fitted_sigma = ec.fit_systematic_error(estimates, se)
    assert abs(fitted_mu - mu) < 0.003 and abs(fitted_sigma - sigma) < 0.003
    raw_p = 2 * __import__("scipy").stats.norm.sf(np.abs(estimates / se))
    calibrated = ec.calibrated_p_values(estimates, se, fitted_mu, fitted_sigma)
    assert np.mean(raw_p < 0.05) > 0.5          # confounding looks like signal
    assert abs(np.mean(calibrated < 0.05) - 0.05) < 0.015


def test_calibrate_screen_uses_per_marker_or_pooled_null():
    rng = np.random.default_rng(4)
    names = [f"control_{i}" for i in range(8)]
    rows = []
    for feature, n_controls in (("a", 8), ("b", 2)):
        for i, name in enumerate(names[:n_controls] + ["Heart failure"]):
            rows.append({"feature_id": feature, "phenotype_name": name,
                         "raw_auc": 0.5 + rng.normal(0.02, 0.01), "n_case": 500,
                         "n_control": 5000, "p_value": 0.01})
    frame = pd.DataFrame(rows)
    output, nulls, summary = ec.calibrate_screen(frame, names)
    scopes = nulls.set_index("feature_id")["scope"].to_dict()
    assert scopes == {"a": "per_marker", "b": "pooled_all_markers"}
    assert output["empirical_calibrated_p_value"].notna().all()
    assert summary["negative_control_rows"] == 10
    # The screen's own columns are untouched.
    pd.testing.assert_frame_equal(output[frame.columns], frame)


def test_disattenuated_auc_follows_the_binormal_model():
    from scipy import stats

    from vcg_biomarker.mimic.reporting_guideline import disattenuated_auc

    d_true = 1.2
    icc = 0.64
    observed = stats.norm.cdf(d_true * np.sqrt(icc) / np.sqrt(2))
    assert abs(disattenuated_auc(observed, icc) - stats.norm.cdf(d_true / np.sqrt(2))) < 1e-12
    assert disattenuated_auc(0.7, 1.0) == 0.7
    assert np.isnan(disattenuated_auc(0.7, 0.0)) and np.isnan(disattenuated_auc(0.7, np.nan))


def test_lab_reads_change_limits_from_the_run(tmp_path):
    from vcg_biomarker.mimic.ecg_lab import _load_marker_change_limits

    pd.DataFrame({"feature_id": ["qt_interval_ms", "other"], "icc": [0.9, 0.5],
                  "smallest_detectable_change": [18.0, 3.0]}).to_parquet(
        tmp_path / "marker_repeatability.parquet")
    limits = _load_marker_change_limits(tmp_path, ("qt_interval_ms",))
    assert limits == {"qt_interval_ms": {"icc": 0.9, "smallest_detectable_change": 18.0,
                                         "source_run": tmp_path.name}}
    assert _load_marker_change_limits(None, ("qt_interval_ms",)) == {}


def test_covariate_adjusted_auc_removes_a_pure_rate_effect():
    from vcg_biomarker.mimic.covariate_adjusted_roc import covariate_adjusted_auc

    rng = np.random.default_rng(9)

    def sample(n):
        rate = rng.normal(80, 15, n)
        label = (rng.uniform(size=n) < 1 / (1 + np.exp(-(rate - 90) / 8))).astype(int)
        rate_marker = rate + rng.normal(0, 3, n)                 # tracks rate only
        real_marker = rng.normal(0, 1, n) + 1.0 * label          # independent of rate
        return rate, label, rate_marker, real_marker

    rate_t, label_t, rm_t, real_t = sample(8000)
    rate_v, label_v, rm_v, real_v = sample(8000)
    confounded = covariate_adjusted_auc(rm_t, label_t, rate_t[:, None], rm_v, label_v, rate_v[:, None])
    assert confounded["unadjusted_auc_same_rows"] > 0.75
    assert abs(confounded["covariate_adjusted_auc"] - 0.5) < 0.03
    genuine = covariate_adjusted_auc(real_t, label_t, rate_t[:, None], real_v, label_v, rate_v[:, None])
    assert abs(genuine["covariate_adjusted_auc"] - genuine["unadjusted_auc_same_rows"]) < 0.03


def test_reporting_checklists_mark_present_missing_and_author_items(tmp_path):
    from vcg_biomarker.mimic.reporting_checklists import AUTHOR, build_reporting_checklists

    run = tmp_path / "run"
    labels = tmp_path / "labels"
    run.mkdir(); labels.mkdir()
    (run / "screening_results.parquet").write_bytes(b"x")
    (labels / "endpoint_index_cohorts.parquet").write_bytes(b"x")
    result = build_reporting_checklists((run, labels, tmp_path / "absent"))
    stard = {row["item"]: row for row in result["checklists"]["STARD_2015"]["items"]}
    assert stard["24"]["status"] == "artifact_present"
    assert stard["19"]["status"] == "artifact_present"
    assert stard["23"]["status"] == "artifact_missing"
    assert stard["30"]["status"] == AUTHOR
    assert result["checklists"]["RECORD"]["n_items"] == 13
    tripod_items = [row["item"] for row in result["checklists"]["TRIPOD_AI"]["items"]]
    assert len(set(tripod_items)) == len(tripod_items) and tripod_items[-1] == "27c"
