import numpy as np
import pandas as pd
import pytest
from vcg_biomarker.mimic.statistics import apply_fdr_by_family, _apply_fdr_by_family_narrow


@pytest.mark.parametrize("sensitivity", [True, False])
@pytest.mark.parametrize("empty", [True, False])
def test_narrow_fdr_exactly_matches_previous_full_width_procedure(sensitivity, empty):
    rng = np.random.default_rng(108)
    n = 0 if empty else 300
    frame = pd.DataFrame({"endpoint_id": np.resize(["a", "b", "c"], n),
        "panel_id": np.resize(["primary_core", "sentinel"], n),
        "evaluation_split": np.resize(["train", "validation"], n),
        "p_value": rng.uniform(0., .15, n),
        "inference_status": np.resize(["formal_inference", "descriptive"], n),
        "confirmatory_assessability_gate_passed": np.resize([True, True, False], n),
        **{f"effect_{i}": rng.normal(size=n) for i in range(40)}})
    original = frame.copy(deep=True)
    expected = _apply_fdr_by_family_narrow(frame, multiplicity_sensitivity=sensitivity)
    actual = apply_fdr_by_family(frame, multiplicity_sensitivity=sensitivity)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    if n:
        actual.loc[0, "effect_0"] = 999.
        pd.testing.assert_frame_equal(frame, original, check_exact=True)


def test_object_nullable_gates_are_false_for_missing_without_downcast_warning():
    frame = pd.DataFrame({"endpoint_id": ["endpoint"] * 4,
                          "panel_id": ["primary_core"] * 4,
                          "evaluation_split": ["validation"] * 4,
                          "p_value": [.001] * 4,
                          "inference_status": ["formal_inference"] * 4,
                          "confirmatory_assessability_gate_passed": pd.Series(
                              [True, False, None, pd.NA], dtype=object)})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = apply_fdr_by_family(frame)
    assert result.confirmatory_fdr_significant.tolist() == [True, False, False, False]
