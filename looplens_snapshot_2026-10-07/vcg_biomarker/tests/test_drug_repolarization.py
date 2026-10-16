"""Audit 5.2.6: repolarisation after starting a QT-prolonging drug."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from vcg_biomarker.mimic import drug_repolarization as dr


def _write(path, frame):
    frame.to_parquet(path, index=False)
    return str(path).replace("\\", "/")


def test_new_starts_pair_a_baseline_and_an_on_drug_ecg(tmp_path):
    t = pd.Timestamp("2150-05-01 08:00")
    hours = lambda h: t + pd.Timedelta(hours=h)  # noqa: E731
    master = _write(tmp_path / "m.parquet", pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "study_id": [11, 12, 13, 21, 22, 31, 32, 33],
        "hadm_id": [100, 100, 100, 200, 200, 300, 300, 300],
        "ecg_time": [hours(-30), hours(-1), hours(20), hours(-5), hours(30),
                     hours(0), hours(10), hours(200)],
    }))
    meds = _write(tmp_path / "e.parquet", pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "event_time": [hours(0), hours(12), hours(0), hours(-24 * 3)],
        "medication_raw": ["Amiodarone 150 mg", "Amiodarone", "Haloperidol", "Ondansetron"],
        "medication_class": ["qt_prolonging_drug"] * 4,
        "source_table": ["emar"] * 4,
    }))
    con = duckdb.connect()
    treated = con.execute(dr.qt_drug_treated_pair_query(master, meds)).df()
    # Subject 1: start at 0 h, baseline = last ECG before (-1 h), on-drug = 20 h.
    row = treated.set_index("subject_id").loc[1]
    assert (row["drug"], row["study_id_before"], row["study_id_after"]) == ("amiodarone", 12, 13)
    assert row["ecg_gap_hours"] == 21.0
    # Subject 2: ondansetron three days earlier -> haloperidol is no new start.
    assert 2 not in set(treated["subject_id"])
    controls = con.execute(dr.qt_drug_control_pair_query(master, meds)).df()
    # Subject 1: -30 -> -1 h lies wholly before the start: untreated in the
    # interval, a valid control. 12 -> 13 spans the start and is excluded.
    # Subject 2: ondansetron within 7 days before the first ECG -> excluded.
    # Subject 3: 0 -> 10 h is a control pair; 10 -> 200 h is outside 120 h.
    assert sorted(zip(controls["subject_id"], controls["study_id_before"],
                      controls["study_id_after"])) == [(1, 11, 12), (3, 31, 32)]


def test_comparison_removes_regression_to_the_mean_and_finds_a_true_shift():
    rng = np.random.default_rng(0)

    def pairs(n, kind, shift, start):
        before = rng.normal(430, 30, n)
        # Regression to the mean in both groups: extreme baselines move back.
        after = 430 + 0.6 * (before - 430) + rng.normal(0, 10, n) + shift
        return pd.DataFrame({
            "subject_id": np.arange(start, start + n), "pair_kind": kind,
            "drug": "amiodarone" if kind == "treated" else None,
            "ecg_gap_hours": rng.uniform(3, 60, n),
            "qtc_fridericia_ms_before": before, "qtc_fridericia_ms_after": after,
        })

    frame = pd.concat([pairs(400, "treated", 15.0, 0), pairs(3000, "untreated", 0.0, 10_000)])
    result = dr.qt_drug_results(frame, ("qtc_fridericia_ms",), iterations=200, seed=1,
                                confidence=0.95)
    overall = result.set_index("scope").loc["all_qt_prolonging_drugs"]
    assert overall["status"] == "estimable"
    assert abs(overall["difference"] - 15.0) < 3.0
    assert overall["ci_low"] < 15.0 < overall["ci_high"]
    # Per drug only with enough pairs.
    assert "amiodarone" in set(result["scope"])


def test_comparison_abstains_without_untreated_pairs():
    frame = pd.DataFrame({
        "subject_id": [1, 2, 3], "pair_kind": "treated", "drug": "sotalol",
        "ecg_gap_hours": [5.0, 6.0, 7.0],
        "qtc_fridericia_ms_before": [420.0, 430.0, 440.0],
        "qtc_fridericia_ms_after": [450.0, 455.0, 470.0],
    })
    result = dr.repolarization_comparison(frame, frame.iloc[0:0], "qtc_fridericia_ms")
    assert np.isnan(result["difference"]) and result["n_treated_pairs"] == 3
