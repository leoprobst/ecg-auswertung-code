"""Real stage persistence; unrelated paired/dose designs are stubbed explicitly."""
from types import SimpleNamespace
import json

import pandas as pd

from vcg_biomarker.mimic import within_subject as stage


def test_empty_episode_stage_writes_manifest_and_reloads(tmp_path, monkeypatch):
    config = SimpleNamespace(catalog_path=tmp_path/"catalog.duckdb",
        runs_root=tmp_path/"runs", master_root=tmp_path/"master",
        standardized_root=tmp_path/"standardized", evidence_root=tmp_path/"evidence",
        workers=1, duckdb_memory_limit_gb=2, to_dict=lambda: {"test_workspace": str(tmp_path)})
    config.master_root.mkdir()
    config.evidence_root.mkdir()
    master = pd.DataFrame(dict(subject_id=[1], hadm_id=[10], study_id=[1],split=["train"],
                               ecg_time=[pd.Timestamp("2020-01-01")]))
    master.to_parquet(config.master_root/"ecg_master_index.parquet")
    features = master[["subject_id","study_id"]].assign(extraction_status="ok",signal_quality_score=.9,marker=1.)
    feature_path = tmp_path/"features.parquet"
    features.to_parquet(feature_path)
    procedures = pd.DataFrame({"subject_id": pd.Series(dtype="int64"), "hadm_id":pd.Series(dtype="int64"),
        "itemid":pd.Series(dtype="int64"), "starttime":pd.Series(dtype="datetime64[ns]")})
    procedure_path = tmp_path/"procedures.parquet"
    procedures.to_parquet(procedure_path)
    pd.DataFrame({"subject_id":pd.Series(dtype="int64"),"hadm_id":pd.Series(dtype="int64"),
        "charttime":pd.Series(dtype="datetime64[ns]"),"lab_group":pd.Series(dtype="str")}).to_parquet(config.evidence_root/"labs.parquet")
    monkeypatch.setattr(stage,"_events_glob",lambda *_: str(procedure_path).replace("\\","/"))
    monkeypatch.setattr(stage,"build_dialysis_lab_pairs",lambda *a,**k: (pd.DataFrame(),pd.DataFrame(columns=["analyte_id"])))
    monkeypatch.setattr(stage,"dose_pairs_all_classes_query",lambda *a,**k: "SELECT ''::VARCHAR AS drug_class WHERE FALSE")
    manifest = stage.run_within_subject_change(config,run_id="synthetic",feature_glob=str(feature_path),markers=("marker",))
    output = config.runs_root/"synthetic"/"within_subject"
    recovery = output/"recovery_episode_results.parquet"
    assert recovery.is_file()
    assert pd.read_parquet(recovery).empty
    assert str(recovery.resolve()) in manifest.output_paths
    for name in ("echo_pairs", "echo_models", "electromechanical_residuals", "pr_geometry_models"):
        path = output / (name + ".parquet")
        assert path.is_file() and str(path.resolve()) in manifest.output_paths
    assert manifest.row_counts["recovery_results"] == 0
    summary = json.loads((output/"within_subject_summary.json").read_text())
    assert summary["recovery"]["status"] == "no_eligible_episodes"
    monkeypatch.setattr(stage,"build_dialysis_lab_pairs",lambda *a,**k: (_ for _ in ()).throw(AssertionError("cache not reused")))
    again = stage.run_within_subject_change(config,run_id="synthetic",feature_glob=str(feature_path),markers=("marker",))
    assert again.input_hash == manifest.input_hash
