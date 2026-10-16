import numpy as np
import pandas as pd

from vcg_biomarker.gui.paired_scatter import prepare_paired_scatter


def test_finite_reference_gates_and_patient_counts_are_not_display_sample_counts():
    frame = pd.DataFrame(dict(delta_lab=[1, 2, 3, np.inf, 5, 6], delta_marker=[1, 2, 3, 4, "bad", 6],
        subject_id=[1, 1, None, 2, 3, 4], analyte_id="potassium", analyte_unit="mmol/l",
        lab_pair_status=["eligible"]*5+["different_assay_or_unit"]))
    original = frame.copy(deep=True)
    sample, info = prepare_paired_scatter(frame, "marker", "potassium", limit=2)
    assert info["n_total"] == 6 and info["n_valid"] == 3 and info["n_shown"] == 2
    assert info["n_known_patients"] == 1 and info["n_missing_patient_rows"] == 1
    assert info["sampling"] == "uniform_pairs_fixed_seed_0_display_only"
    assert np.isfinite(sample.to_numpy(float)).all()
    again, _ = prepare_paired_scatter(frame, "marker", "potassium", limit=2)
    pd.testing.assert_frame_equal(sample, again)
    pd.testing.assert_frame_equal(frame, original)


def test_selection_does_not_mix_analytes_or_units_and_legacy_is_explicit():
    frame = pd.DataFrame(dict(delta_lab=[1., 2.], delta_marker=[3., 4.],
        analyte_id=["potassium", "ph"], analyte_unit=["mmol/l", "units"]))
    sample, info = prepare_paired_scatter(frame, "marker", None)
    assert sample.empty and info["status"] == "ambiguous_analyte_or_unit"
    sample, info = prepare_paired_scatter(frame, "marker", "ph")
    assert len(sample) == 1 and info["unit"] == "units"
    assert info["legacy_reference"] is True
    assert info["n_missing_patient_rows"] == 1


def test_empty_and_unknown_marker_do_not_retain_old_counts():
    for frame in (None, pd.DataFrame(), pd.DataFrame(dict(delta_lab=[1.]))):
        sample, info = prepare_paired_scatter(frame, "missing", None)
        assert sample.empty and info["n_shown"] == 0


def test_bad_reference_status_blocks_even_finite_values():
    frame = pd.DataFrame(dict(delta_lab=[1., 2.], delta_marker=[3., 4.], lab_pair_status=[pd.NA, "rejected"]))
    sample, info = prepare_paired_scatter(frame, "marker", None)
    assert sample.empty and info["n_valid"] == 0
