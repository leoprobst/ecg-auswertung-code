import duckdb
import pandas as pd

from vcg_biomarker.mimic.recovery_loader import dialysis_recovery


def test_real_sql_loader_through_reference_and_recovery():
    start = pd.Timestamp("2020-01-02")
    master = pd.DataFrame(dict(subject_id=[1]*4+[2], hadm_id=[10]*4+[20], study_id=range(5),
        split=["train"]*4+["locked_test"],
        ecg_time=[start+pd.Timedelta(hours=h) for h in (-2,1,3,7,1)]))
    features = master[["subject_id","study_id"]].assign(extraction_status="ok",signal_quality_score=.9,marker=[0.,10.,8.,4.,99.])
    procedures = pd.DataFrame(dict(subject_id=[1,2],hadm_id=[10,20],starttime=start,itemid=225441))
    labs = pd.DataFrame(dict(subject_id=1,hadm_id=10,labevent_id=[1,2],itemid=50971,lab_group="potassium",
        charttime=[start-pd.Timedelta(hours=1),start+pd.Timedelta(hours=2)],
        storetime=[start,start+pd.Timedelta(hours=3)],value_numeric=[6.,4.],unit="mEq/L",
        reference_lower=3.5,reference_upper=5.,value_status="usable"))
    with duckdb.connect() as con:
        for name, frame in (("master",master),("features",features),("procedures",procedures),("labs",labs)):
            con.register(name, frame)
        results, summary = dialysis_recovery(con, master_relation="master",procedure_relation="procedures",
            feature_relation="features",lab_relation="labs",marker_ids=("marker",),dialysis_itemids=(225441,))
        assert len(results) == 1
        row = results.iloc[0]
        assert row.subject_id == 1
        assert row.normalization_residual == 8.
        assert row.magnesium__normalization_status == "not_evaluable"
        assert row.reference_magnesium_normalization_reason == "no_baseline_lab_in_window"
        assert row.observed_half_peak_crossing_hours == 6.
        assert summary["n_selected_ecgs"] == 4
        assert summary["control_comparison_status"] == "not_computed"
        assert summary["normalization_references"]["magnesium"]["n_labs"] == 0
        assert not any(name.startswith("recovery_") for (name,) in con.execute("SHOW TABLES").fetchall())
        # Missing selected features are not counted as marker missingness in a
        # pilot: only records actually in its feature artifact enter the cohort.
        con.unregister("features")
        con.register("features",features.iloc[:3])
        partial, summary = dialysis_recovery(con, master_relation="master",procedure_relation="procedures",
            feature_relation="features",lab_relation="labs",marker_ids=("marker",),dialysis_itemids=(225441,))
        assert summary["n_selected_ecgs"] == 3
        assert partial.status.eq("not_evaluable").all()


def test_multiple_analytes_share_kinetics_but_not_normalization(monkeypatch):
    from vcg_biomarker.mimic import recovery_episodes
    from vcg_biomarker.tests.test_recovery_episodes import fixture
    ecgs, events = fixture()
    features = ecgs[["subject_id", "study_id", "extraction_status", "signal_quality_score", "marker"]]
    master = ecgs[["subject_id", "hadm_id", "study_id", "split", "ecg_time"]]
    start = events.landmark_time.iloc[0]
    procedures = pd.DataFrame(dict(subject_id=[1], hadm_id=[10], starttime=[start], itemid=[225441]))
    rows = []
    for group, itemid, low, high, values, norm_hour, unit in (
        ("potassium", 50971, 3.5, 5., (6., 4.), 2, "mEq/L"),
        ("magnesium", 50960, 1.5, 2.5, (1., 2.), 4, "mg/dL"),
        ("total_calcium", 50893, 8., 10., (7., 9.), 2, "mg/dL"),
        ("ionized_calcium", 50808, 1., 1.3, (.8, 1.2), 4, "mmol/L"),
        ("blood_gas_ph", 50820, 7.35, 7.45, (7.2, 7.4), 2, "units"),
    ):
        for hour, value in zip((-1, norm_hour), values):
            rows.append(dict(subject_id=1, hadm_id=10, labevent_id=len(rows)+1,
                itemid=itemid, lab_group=group, charttime=start+pd.Timedelta(hours=hour),
                storetime=start+pd.Timedelta(hours=hour+1), value_numeric=value,
                reference_lower=low, reference_upper=high, unit=unit, value_status="usable"))
    original = recovery_episodes.recovery_kinetics
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs["marker_column"])
        return original(*args, **kwargs)
    monkeypatch.setattr(recovery_episodes, "recovery_kinetics", counted)
    with duckdb.connect() as con:
        for name, frame in (("master",master),("features",features),("procedures",procedures),("labs",pd.DataFrame(rows))):
            con.register(name, frame)
        results, summary = dialysis_recovery(con, master_relation="master", procedure_relation="procedures",
            feature_relation="features", lab_relation="labs", marker_ids=("marker",), dialysis_itemids=(225441,))
    assert calls == ["marker"]  # no five-fold baseline/kinetics recomputation
    assert len(results) == 1  # analytes do not inflate independent episode count
    row = results.iloc[0]
    assert row.normalization_residual == 8.
    assert row.total_calcium__normalization_residual == 8.
    assert row.ionized_calcium__normalization_residual == 4.
    assert row.magnesium__normalization_residual == 4.
    assert row.blood_gas_ph__normalization_residual == 8.
    assert row.reference_total_calcium_unit != row.reference_ionized_calcium_unit
    assert all(item["n_observed_normalizations"] == 1 for item in summary["normalization_references"].values())
