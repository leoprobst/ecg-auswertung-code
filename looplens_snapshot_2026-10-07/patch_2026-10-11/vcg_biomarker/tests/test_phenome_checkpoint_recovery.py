from pathlib import Path
import ast
import hashlib

import pandas as pd
import pytest

from vcg_biomarker.mimic import phenome
from vcg_biomarker.tests.test_performance_hardening import _config, _phenome_stream_fixture


def setup_case(tmp_path, monkeypatch):
    full, current, definitions = _phenome_stream_fixture()
    config = _config(tmp_path, workers=1)

    def load_block(*args, feature_ids, splits=("train", "validation"), **kwargs):
        selected = full.loc[full["split"].isin(splits)]
        return selected[["subject_id", "study_id", "split", *feature_ids]].reset_index(drop=True)

    monkeypatch.setattr(phenome, "_load_phenome_feature_block", load_block)
    monkeypatch.setattr(phenome, "_feature_blocks", lambda ids, size=32: tuple((x,) for x in ids))
    common = dict(config=config, feature_glob="unused", index_path=tmp_path / "index",
                  input_hash="same-scientific-input", feature_index=full.drop(columns=["f1", "f2"]),
                  history_body=pd.DataFrame(), definition_lookup=definitions,
                  min_inference=1, min_formal=1, progress=None)
    train = dict(common, current=current.loc[current["split"].eq("train")],
                 phenotype_ids=("p1", "p2"), feature_ids=("f1", "f2"),
                 panel_map={"f1": "panel", "f2": "panel"})
    hits = pd.DataFrame(dict(phenotype_id=["p1", "p2"], phenotype_name=["P1", "P2"],
                             body_system=["A", "B"], panel_id=["panel", "panel"],
                             feature_id=["f1", "f2"], q_value=[.01, .02], aligned_auc=[.8, .8],
                             direction=[1, 1], replication_seed_index=[1, 2]))
    validation = dict(common, current=current, train_hits=hits, bootstrap_iterations=0,
                      random_seed=123, worker_count=1, checkpoint_namespace="validation",
                      progress_label="Validation")
    return train, validation


@pytest.mark.parametrize("stage", ["train", "validation"])
def test_completed_blocks_resume_identically_without_loading_features(tmp_path, monkeypatch, stage):
    train, validation = setup_case(tmp_path, monkeypatch)
    options = train if stage == "train" else validation
    function = (phenome._checkpointed_train_screen_streaming if stage == "train"
                else phenome._checkpointed_validation_replication_streaming)
    root = tmp_path / "chunks"
    cold = function(root, **options)
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("part-*")}

    def fail(*args, **kwargs):
        raise AssertionError("Completed block loaded raw data or built model arrays")

    monkeypatch.setattr(phenome, "_load_phenome_feature_block", fail)
    monkeypatch.setattr(phenome, "_prepare_screen_block", fail)
    monkeypatch.setattr(phenome, "_prepare_phenome_replication_cache", fail)
    resumed = function(root, **options)
    if stage == "train":
        assert resumed[1] == cold[1]
        pd.testing.assert_frame_equal(resumed[0], cold[0], check_exact=True)
    else:
        pd.testing.assert_frame_equal(resumed, cold, check_exact=True)
    assert original == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("part-*")}


@pytest.mark.parametrize("stage", ["train", "validation"])
def test_incomplete_block_computes_only_missing_work(tmp_path, monkeypatch, stage):
    train, validation = setup_case(tmp_path, monkeypatch)
    options = train if stage == "train" else validation
    function = (phenome._checkpointed_train_screen_streaming if stage == "train"
                else phenome._checkpointed_validation_replication_streaming)
    root = tmp_path / "chunks"
    cold = function(root, **options)
    manifests = sorted(root.rglob("part-*.json"))
    missing = manifests[-1]
    missing.unlink()
    reused = manifests[0].read_bytes()
    resumed = function(root, **options)
    assert missing.is_file() and manifests[0].read_bytes() == reused
    pd.testing.assert_frame_equal(resumed[0] if stage == "train" else resumed,
                                  cold[0] if stage == "train" else cold, check_exact=True)


@pytest.mark.parametrize("stage", ["train", "validation"])
def test_damaged_or_stale_blocks_are_rejected(tmp_path, monkeypatch, stage):
    train, validation = setup_case(tmp_path, monkeypatch)
    options = train if stage == "train" else validation
    function = (phenome._checkpointed_train_screen_streaming if stage == "train"
                else phenome._checkpointed_validation_replication_streaming)
    root = tmp_path / "chunks"
    function(root, **options)
    with pytest.raises(RuntimeError, match="stale"):
        function(root, **dict(options, input_hash="different-input"))
    data = next(root.rglob("part-*.parquet"))
    data.write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        function(root, **options)


def test_changed_feature_selection_falls_back_without_inventing_available_markers(tmp_path, monkeypatch):
    train, _ = setup_case(tmp_path, monkeypatch)
    monkeypatch.setattr(phenome, "_feature_blocks", lambda ids, size=32: (tuple(ids),))
    loader = phenome._load_phenome_feature_block

    def with_missing(*args, **kwargs):
        data = loader(*args, **kwargs)
        data["f2"] = float("nan")
        return data

    monkeypatch.setattr(phenome, "_load_phenome_feature_block", with_missing)
    root = tmp_path / "chunks"
    cold, available = phenome._checkpointed_train_screen_streaming(root, **train)
    resumed, reused_available = phenome._checkpointed_train_screen_streaming(root, **train)
    assert available == reused_available == ("f1",)
    pd.testing.assert_frame_equal(cold, resumed, check_exact=True)


@pytest.mark.parametrize("stage,bootstrap", [("train", 0), ("validation", 0), ("validation", 7)])
def test_exact_equivalence_to_bound_streaming_functions(tmp_path, monkeypatch, stage, bootstrap):
    train, validation = setup_case(tmp_path, monkeypatch)
    reference_path = Path(__file__).parent / "fixtures/phenome_bound_recovery_reference.py"
    content = reference_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == "3fc8cdd1d3454adc30a652031a515011c260104e9c1df20574103bfcd0cba5e8"
    name = ("_checkpointed_train_screen_streaming" if stage == "train"
            else "_checkpointed_validation_replication_streaming")
    tree = ast.parse(content.decode("utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(phenome.__dict__)
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(reference_path), "exec"), namespace)
    options = train if stage == "train" else dict(validation, bootstrap_iterations=bootstrap)
    reference = namespace[name](tmp_path / "bound", **options)
    function = getattr(phenome, name)
    cold = function(tmp_path / "repair", **options)
    resumed = function(tmp_path / "repair", **options)
    for actual in (cold, resumed):
        if stage == "train":
            assert actual[1] == reference[1]
            pd.testing.assert_frame_equal(actual[0], reference[0], check_exact=True)
        else:
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
