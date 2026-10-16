import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import (
    DIALYSIS_WINDOW, DRUG_WINDOW, DIALYSIS_ITEMIDS, intervention_pair_query,
    dose_pair_query, dose_pairs_all_classes_query,
)


@pytest.mark.parametrize("design", ["dialysis", "dose_single", "dose_combined"])
@pytest.mark.parametrize("canonical_failed", [False, True])
@pytest.mark.parametrize("quality,quality_ok", [(.9, True), (.5, True), (1., True),
    (.49, False), (None, False), (float("inf"), False), (float("nan"), False), (1.1, False)])
def test_clock_representative_is_order_invariant_and_chosen_before_qc(tmp_path, design, canonical_failed, quality, quality_ok):
    origin = pd.Timestamp("2020-01-01 10:00")
    master = pd.DataFrame(dict(subject_id=1, hadm_id=10, study_id=[99, 100, 101, 102], split="train",
        ecg_time=[origin-pd.Timedelta(hours=2), origin, origin, origin+pd.Timedelta(hours=2)]))
    features = master[["subject_id", "study_id"]].assign(marker=[1., 2., 100., 3.],
        extraction_status=["ok", "failed" if canonical_failed else "ok", "ok", "ok"], signal_quality_score=.9)
    features.loc[features.study_id.eq(100), "signal_quality_score"] = quality
    times = [origin-pd.Timedelta(hours=1), origin+pd.Timedelta(hours=1)]
    procedures = pd.DataFrame(dict(subject_id=1, hadm_id=10, itemid=DIALYSIS_ITEMIDS[0], starttime=times))
    inputs = pd.DataFrame(dict(subject_id=1, hadm_id=10, itemid=1, starttime=times,
        endtime=[value+pd.Timedelta(minutes=15) for value in times], amount=5., amountuom="mg",
        orderid=[501, 502], linkorderid=[501, 502], statusdescription="FinishedRunning"))
    items = pd.DataFrame(dict(itemid=[1], label=["Metoprolol"]))
    frames = dict(master=master, features=features, procedures=procedures, inputs=inputs, items=items)
    paths = {name: (tmp_path/f"{name}.parquet").as_posix() for name in frames}
    feature_clause = f"read_parquet('{paths['features']}')"
    if design == "dialysis":
        query = intervention_pair_query(paths["master"], paths["procedures"], feature_clause,
            window=DIALYSIS_WINDOW, marker_columns="b.marker-a.marker AS delta_marker")
    elif design == "dose_single":
        query = dose_pair_query(paths["master"], paths["inputs"], paths["items"], feature_clause,
            window=DRUG_WINDOW, marker_columns="b.marker-a.marker AS delta_marker", label_pattern="metoprolol")
    else:
        query = dose_pairs_all_classes_query(paths["master"], paths["inputs"], paths["items"], feature_clause,
            window=DRUG_WINDOW, marker_ids=("marker",), class_patterns={"beta": "metoprolol"})
    results = []
    for seed in (12, 97, 253):
        for name, frame in frames.items():
            frame.sample(frac=1, random_state=seed).to_parquet(paths[name], index=False)
        with duckdb.connect() as con:
            result = con.execute(query).df().sort_values("first_study_id").reset_index(drop=True)
        results.append(result)
        if canonical_failed or not quality_ok:
            assert result.empty  # no rescue by selecting study 101 or bridging 99 -> 102
        else:
            assert list(zip(result.first_study_id, result.second_study_id)) == [(99, 100), (100, 102)]
            assert result.delta_marker.tolist() == [1., 1.]
            assert result.first_clock_tie_count.tolist() == [1, 2]
            assert result.second_clock_tie_count.tolist() == [2, 1]
    for result in results[1:]:
        pd.testing.assert_frame_equal(result, results[0])
