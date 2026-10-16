from types import SimpleNamespace
import pandas as pd
import pytest
from vcg_biomarker.mimic.reference_clock_stress import clock_stress_counts, SHIFTS_HOURS


def test_clock_stress_sign_direction_denominator_and_no_reindexing():
    frame = pd.DataFrame({"subject_id": [1, 2, 3], "split": ["train"] * 3,
        "label_state": ["positive"] * 3, "reference_evidence_available": [True, True, False],
        "time_precision": ["timestamp"] * 3, "time_delta_hours": [-.9, .9, 0.]})
    original = frame.copy(deep=True)
    definition = SimpleNamespace(label_id="potassium", window_before_hours=-1., window_after_hours=1.)
    result = pd.DataFrame(clock_stress_counts(frame, definition))
    assert set(result.assumed_ecg_clock_shift_hours) == set(SHIFTS_HOURS)
    positive = result[result.label_state.eq("positive")].set_index("assumed_ecg_clock_shift_hours")
    assert positive.loc[0., "n_within_shifted_window"] == 2
    assert positive.loc[.25, "n_lost_vs_zero_shift"] == 1
    assert positive.loc[-.25, "n_lost_vs_zero_shift"] == 1
    assert positive.n_observed_timed_reference.eq(2).all()
    assert not result.optimal_shift_selection.any()
    pd.testing.assert_frame_equal(frame, original)
    frame.loc[0, "split"] = "locked_test"
    with pytest.raises(ValueError, match="development-only"):
        clock_stress_counts(frame, definition)


def test_every_production_definition_uses_signed_window_contract():
    from vcg_biomarker.mimic.labels import default_label_definitions
    frame = pd.DataFrame({"split": ["train"], "label_state": ["positive"],
                          "reference_evidence_available": [True],
                          "time_precision": ["timestamp"], "time_delta_hours": [0.]})
    for definition in default_label_definitions():
        rows = clock_stress_counts(frame, definition)
        assert rows
        if definition.window_before_hours is None and definition.window_after_hours is None:
            assert rows[0]["status"] == "not_applicable_no_finite_window"
        else:
            zero = next(row for row in rows if row["label_state"] == "positive"
                        and row["assumed_ecg_clock_shift_hours"] == 0)
            lower = definition.window_before_hours
            upper = definition.window_after_hours
            assert zero["n_within_shifted_window"] == int(
                (lower is None or lower <= 0) and (upper is None or upper >= 0))


@pytest.mark.parametrize("lower,upper", [(-6., 12.), (1., 12.), (-12., -1.),
                                         (None, 0.), (0., None)])
def test_signed_one_sided_and_future_only_windows(lower, upper):
    times = pd.Series([-13., -6., -1., 0., 1., 12., 13.])
    frame = pd.DataFrame({"split": "train", "label_state": "positive",
                          "reference_evidence_available": True, "time_precision": "timestamp",
                          "time_delta_hours": times})
    definition = SimpleNamespace(label_id="test", window_before_hours=lower,
                                 window_after_hours=upper)
    result = clock_stress_counts(frame, definition)
    for row in result:
        if row["label_state"] != "positive":
            continue
        shifted = times - row["assumed_ecg_clock_shift_hours"]
        expected = (True if lower is None else shifted >= lower) & (
                    True if upper is None else shifted <= upper)
        assert row["n_within_shifted_window"] == int(expected.sum())


@pytest.mark.parametrize("lower,upper", [(2., 1.), (float("nan"), 1.),
                                         (-1., float("inf"))])
def test_invalid_signed_windows_are_rejected(lower, upper):
    definition = SimpleNamespace(label_id="test", window_before_hours=lower,
                                 window_after_hours=upper)
    with pytest.raises(ValueError, match="Invalid prespecified"):
        clock_stress_counts(pd.DataFrame(), definition)
