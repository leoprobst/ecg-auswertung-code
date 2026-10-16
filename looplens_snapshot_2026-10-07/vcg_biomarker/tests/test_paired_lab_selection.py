from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic import within_subject as stage


def fixture_tables(tmp_path):
    origin = pd.Timestamp("2020-01-01")
    master = pd.DataFrame(dict(subject_id=[1, 1, 2, 2, 3, 3], hadm_id=[10, 10, 20, 20, 30, 30],
        study_id=[100, 101, 200, 201, 300, 301], split=["train"]*4+["locked_test"]*2,
        ecg_time=[origin, origin+pd.Timedelta(hours=2)]*3))
    labs = master[["subject_id", "hadm_id", "ecg_time"]].rename(columns={"ecg_time": "charttime"}).assign(
        itemid=50971, valueuom="mmol/L", valuenum=4., comments=None)
    paths = {}
    for name, frame in (("master", master), ("labs", labs)):
        path = tmp_path/f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    return master, paths


def test_selected_semijoin_preserves_references_without_row_multiplication(tmp_path):
    _, paths = fixture_tables(tmp_path)
    args = (paths["master"].as_posix(), paths["labs"].as_posix())
    selected = pd.DataFrame(dict(subject_id=[1, 1, 1, 999, 3], study_id=[100, 100, 101, 200, 300]))
    with duckdb.connect() as con:
        con.register("selected", selected)
        all_rows = con.execute(stage.nearest_labs_query(*args)).df()
        bounded = con.execute(stage.nearest_labs_query(*args, selected_ecgs_relation="selected")).df()
        con.unregister("selected")
        con.register("selected", selected.iloc[:0])
        assert con.execute(stage.nearest_labs_query(*args, selected_ecgs_relation="selected")).df().empty
    expected = all_rows.loc[all_rows.subject_id.eq(1)]
    pd.testing.assert_frame_equal(bounded.sort_values("study_id").reset_index(drop=True),
                                  expected.sort_values("study_id").reset_index(drop=True))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("wrong_admission", [False, True])
def test_actual_pair_builder_selects_only_needed_ecgs(tmp_path, monkeypatch, empty, wrong_admission):
    master, paths = fixture_tables(tmp_path)
    master_root = tmp_path/"master_index"
    master_root.mkdir()
    master.to_parquet(master_root/"ecg_master_index.parquet", index=False)
    features = master[["subject_id", "study_id"]].assign(marker=1., extraction_status="ok", signal_quality_score=.9)
    feature_path = tmp_path/"features.parquet"
    features.to_parquet(feature_path, index=False)
    procedures = pd.DataFrame(dict(subject_id=[99 if empty else 1], hadm_id=[20 if wrong_admission else 10],
        itemid=[stage.DIALYSIS_ITEMIDS[0]], starttime=[pd.Timestamp("2020-01-01 01:00")]))
    procedure_path = tmp_path/"procedures.parquet"
    procedures.to_parquet(procedure_path, index=False)
    monkeypatch.setattr(stage, "_events_glob", lambda _, table: (
        procedure_path if table == "procedureevents" else paths["labs"]).as_posix())
    config = SimpleNamespace(master_root=master_root, catalog_path=tmp_path/"catalog.duckdb",
        duckdb_memory_limit_gb=1, workers=1)
    if empty or wrong_admission:
        monkeypatch.setattr(stage, "nearest_labs_query", lambda *a, **k: pytest.fail("empty pairs must not scan labs"))
    pairs, labs = stage.build_dialysis_lab_pairs(config, feature_glob=feature_path.as_posix(), marker_ids=("marker",))
    if empty or wrong_admission:
        assert pairs.empty and labs.empty and "lab_reference_status" in labs
    else:
        assert pairs.subject_id.tolist() == [1]
        assert set(labs.study_id) == {100, 101}
        assert labs.lab_reference_status.eq("eligible").all()
