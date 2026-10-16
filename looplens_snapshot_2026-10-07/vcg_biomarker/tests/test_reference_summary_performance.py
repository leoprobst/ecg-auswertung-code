import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.validation import _observed_reference_kind


@pytest.mark.parametrize("values", [None, [None, None, None],
    ["timed_quantitative"]*3, ["timed_quantitative", "annotation", None], [1, "1", None]])
@pytest.mark.parametrize("mask", [[True, True, True], [True, False, False], [False]*3])
def test_optimized_summary_matches_original_contract(values, mask):
    frame = pd.DataFrame(index=[4, 7, 10])
    if values is not None:
        frame["reference_kind"] = values
    valid = np.asarray(mask)
    original = frame.get("reference_kind", pd.Series("unverified_or_unknown", index=frame.index))
    kinds = set(original.loc[valid].dropna().astype(str))
    expected = next(iter(kinds)) if len(kinds) == 1 else "mixed_or_unknown"
    assert _observed_reference_kind(frame, valid) == expected


def test_verification_bias_does_not_copy_unrelated_feature_columns(monkeypatch):
    from vcg_biomarker.mimic import validation as module
    from vcg_biomarker.mimic.statistics import fit_marker_model

    frame = pd.DataFrame({"split": ["train"]*4+["validation"]*4,
        "label_state": ["positive", "negative", "unknown", "positive"]*2,
        "marker": [2., -1., np.nan, 1.]*2,
        **{f"unrelated_{i}": np.arange(8) for i in range(128)}})
    model = fit_marker_model(np.array([1., 2., -1., -2.]), np.array([1, 1, 0, 0]), "marker")
    arguments = dict(endpoint_id="synthetic", panel_id="primary_core", feature_id="marker", model=model)
    expected = module._marker_verification_bias_records(
        endpoint_universe=frame[["split", "label_state", "marker"]], **arguments)
    original = module._numeric_feature
    inspected = []
    def bounded_numeric(scoped, name):
        inspected.append(len(scoped.columns))
        assert set(scoped.columns) == {"label_state", "marker"}
        return original(scoped, name)
    monkeypatch.setattr(module, "_numeric_feature", bounded_numeric)
    actual = module._marker_verification_bias_records(endpoint_universe=frame, **arguments)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), check_exact=True)
    assert inspected == [2, 2]
