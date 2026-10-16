"""Incomplete exposure must not become a seemingly complete positive sum."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.within_subject import (
    DRUG_WINDOW,
    _dose_aggregate_columns,
    dose_pair_query,
    dose_pairs_all_classes_query,
)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("amount", [None, 0., -1., np.inf, -np.inf, np.nan])
def test_bad_administration_is_not_dropped_before_pair_sum(tmp_path, combined, amount):
    origin = pd.Timestamp("2020-01-01")
    master = pd.DataFrame(dict(subject_id=[1, 1], hadm_id=[10, 10],
        study_id=[100, 101], split="train",
        ecg_time=[origin, origin + pd.Timedelta(hours=2)]))
    inputs = pd.DataFrame(dict(subject_id=[1, 1], hadm_id=[10, 10],
        orderid=[20, 21], linkorderid=[20, 21], itemid=1,
        starttime=origin + pd.Timedelta(hours=1),
        endtime=origin + pd.Timedelta(hours=1.5), amount=[5., amount],
        amountuom="mg", statusdescription="FinishedRunning"))
    frames = {"master": master, "inputs": inputs,
        "items": pd.DataFrame(dict(itemid=[1], label=["Metoprolol"])),
        "features": master[["subject_id", "study_id"]].assign(
            extraction_status="ok", marker=1., signal_quality_score=.9)}
    paths = {}
    for name, frame in frames.items():
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path.as_posix()
    args = (paths["master"], paths["inputs"], paths["items"],
            f"read_parquet('{paths['features']}')")
    if combined:
        query = dose_pairs_all_classes_query(*args, window=DRUG_WINDOW,
            marker_ids=("marker",), class_patterns={"beta": "metoprolol"})
    else:
        query = dose_pair_query(*args, window=DRUG_WINDOW,
            marker_columns="b.marker-a.marker AS delta_marker", label_pattern="metoprolol")
    with duckdb.connect() as con:
        result = con.execute(query).df()
    assert len(result) == 1
    row = result.iloc[0]
    assert row.n_administrations == row.dose_source_rows == 2
    assert pd.isna(row.dose_amount)
    assert pd.isna(row.dose_itemid)
    assert row.dose_aggregation_status == "invalid_or_missing_dose_amount"


def test_finite_individual_amounts_cannot_produce_infinite_sum():
    doses = pd.DataFrame(dict(itemid=[1, 1], amountuom="mg",
        amount=[1e308, 1e308], label="drug"))
    with duckdb.connect() as con:
        con.register("doses", doses)
        row = con.execute("SELECT " + _dose_aggregate_columns() + " FROM doses x").df().iloc[0]
    assert pd.isna(row.dose_amount)
    assert row.dose_aggregation_status == "invalid_or_missing_dose_amount"
