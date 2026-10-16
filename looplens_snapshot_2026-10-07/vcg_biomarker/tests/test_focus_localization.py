import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.focus_localization import (
    validate_mapping,
    regional_result,
    mapping_inventory,
    evaluate_regions,
    beat_activation_features,
)


def mapping():
    return pd.DataFrame(
        {
            "cohort_id": ["a"] * 40,
            "patient_id": np.arange(40),
            "episode_id": np.arange(40),
            "morphology_id": ["pvc"] * 40,
            "signal_id": np.arange(40),
            "rhythm": ["pvc"] * 40,
            "reference_type": ["focal_origin"] * 40,
            "region": ["rv_outflow", "lv_other"] * 20,
            "evidence_id": np.arange(40),
            "reference_verified": [True] * 40,
            "reference_uncertainty": ["regional"] * 40,
            "signal_quality_pass": [True] * 40,
            "split": ["train"] * 40,
            "center_id": ["a"] * 40,
            "transform": ["kors"] * 40,
            "axis_convention": ["xyz_declared"] * 40,
            "initial_force": [1.0, -1.0] * 20,
        }
    )


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("rhythm", "af", "rhythm"),
        ("region", "octant_3", "anatomical"),
        ("reference_type", "successful_ablation", "reference"),
        ("reference_verified", False, "unverified"),
        ("split", "locked_test", "locked"),
    ],
)
def test_mapping_rejects_surrogates(column, value, reason):
    frame = mapping()
    frame[column] = value
    with pytest.raises(ValueError, match=reason):
        validate_mapping(frame, "ventricular_focus")


def test_regional_uncertainty_is_not_a_three_dimensional_position():
    result = regional_result(
        "ventricular_focus",
        "focal_origin",
        ["rv_outflow", "lv_other"],
        [0.51, 0.49],
        quality_pass=True,
    )
    assert result.status == "abstained"
    assert result.coordinates is None
    assert not result.clinical_claim_allowed


def test_no_mapping_files_does_not_claim_validation(tmp_path):
    assert mapping_inventory(tmp_path)["status"] == "missing_mapping_reference"


def test_regional_training_uses_patient_groups():
    result = evaluate_regions(
        mapping(), ["initial_force"], task="ventricular_focus", folds=2
    )
    assert result["macro_f1"] == 1
    assert result["n_patients"] == 40
    assert result["coordinates"] is None


def test_mapping_outcome_never_a_predictor():
    with pytest.raises(ValueError, match="not_a_predictor"):
        evaluate_regions(mapping(), ["region"], task="ventricular_focus")


def test_early_activation_integral_retains_axis_and_baseline_invariance():
    xyz = np.tile([1.0, 0.0, -2.0], (100, 1))
    kwargs = dict(
        sampling_rate=1000.0,
        onset=0,
        offset=80,
        segment="qrs",
        quality_pass=True,
        morphology_id="pvc1",
    )
    first = beat_activation_features(xyz, baseline_xyz=[0.0, 0.0, 0.0], **kwargs)
    second = beat_activation_features(xyz + 3, baseline_xyz=[3.0, 3.0, 3.0], **kwargs)
    assert first == second
    assert first["10ms"]["integral_xyz_mv_s"] == pytest.approx([0.010, 0.0, -0.020])
    assert first["axis_preserved"]
    with pytest.raises(ValueError, match="quality"):
        beat_activation_features(
            xyz, baseline_xyz=[0.0, 0.0, 0.0], **{**kwargs, "quality_pass": False}
        )
