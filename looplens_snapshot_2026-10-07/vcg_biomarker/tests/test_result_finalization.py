import weakref

import pandas as pd
import pytest

from vcg_biomarker.mimic.result_finalization import finalize_result_tables


def test_factories_release_previous_table_before_building_next(tmp_path):
    references = []
    def factory(value):
        def make():
            assert all(reference() is None for reference in references)
            frame = pd.DataFrame({"value": [value]})
            references.append(weakref.ref(frame))
            return frame
        return make
    sources = {"a": factory(1), "b": factory(2)}
    retained, counts, paths = finalize_result_tables(sources, tmp_path, lambda name, frame: frame)
    assert sources == {} and retained == {}
    assert counts == {"a": 1, "b": 1}
    assert len(paths) == 2
    assert all(reference() is None for reference in references)


def test_prevalence_sanitizing_and_registry_retention_match_legacy(tmp_path):
    from vcg_biomarker.mimic.validation import _suppress_prevalence_dependent_metrics
    original = {"criterion_candidates": pd.DataFrame({"roc_auc": [.8], "brier": [.1]}),
                "performance_curve_points": pd.DataFrame({"curve_type": ["roc", "precision_recall"], "x": [.1, .2]})}
    expected = _suppress_prevalence_dependent_metrics(original)
    retained, counts, _ = finalize_result_tables(dict(original), tmp_path,
        lambda name, frame: _suppress_prevalence_dependent_metrics({name: frame})[name],
        retain=("criterion_candidates",))
    for name, frame in expected.items():
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / f"{name}.parquet"), frame)
        assert counts[name] == len(frame)
    assert list(retained) == ["criterion_candidates"]


def test_failed_export_does_not_continue_to_other_factories(tmp_path, monkeypatch):
    import vcg_biomarker.mimic.result_finalization as module
    def failed(*args):
        raise OSError("simulated full disk")
    monkeypatch.setattr(module, "atomic_write_parquet", failed)
    sources = {"a": lambda: pd.DataFrame({"x": [1]}), "b": lambda: pytest.fail("must not run")}
    with pytest.raises(OSError, match="full disk"):
        finalize_result_tables(sources, tmp_path, lambda name, frame: frame)
    assert "b" in sources


def test_criterion_finalization_never_materializes_unselected_marker_dicts(monkeypatch):
    from vcg_biomarker.mimic.criteria import finalize_criterion_catalog_evidence
    candidate = pd.DataFrame([{"endpoint_id": "e", "panel_id": "p", "feature_id": "chosen",
                               "stage": "hypothesis", "core_replication_evidence_passed": False}])
    selected = pd.DataFrame([{"endpoint_id": "e", "panel_id": "p", "feature_id": "chosen",
                             "evaluation_split": split} for split in ("train", "validation")])
    expected = finalize_criterion_catalog_evidence(candidate, selected)
    irrelevant = pd.DataFrame({"endpoint_id": "e", "panel_id": "p",
        "feature_id": [f"not_selected_{i}" for i in range(10000)], "evaluation_split": "validation"})
    large = pd.concat([selected, irrelevant], ignore_index=True)
    original = pd.DataFrame.to_dict
    def checked(frame, *args, **kwargs):
        assert len(frame) <= 2, "Unselected full-table dictionaries were materialized"
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "to_dict", checked)
    actual = finalize_criterion_catalog_evidence(candidate, large)
    pd.testing.assert_frame_equal(actual, expected)
