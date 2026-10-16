import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.survival import tertile_curves
from vcg_biomarker.tests.test_evaluation_views import _survival_frame


@pytest.mark.parametrize("split", ["train", "validation"])
def test_no_fallback_to_single_split(split):
    frame = _survival_frame()
    frame["split"] = split
    assert tertile_curves(frame, "marker", minimum_per_group=50).empty


@pytest.mark.parametrize("split", ["locked_test", "test", "unknown", None])
def test_forbidden_or_unknown_split_rejected(split):
    frame = _survival_frame()
    frame.loc[0, "split"] = split
    with pytest.raises(PermissionError):
        tertile_curves(frame, "marker")


def test_split_contract_and_training_orientation_required():
    frame = _survival_frame()
    with pytest.raises(ValueError, match="explicit"):
        tertile_curves(frame.drop(columns="split"), "marker")
    with pytest.raises(ValueError, match="training"):
        tertile_curves(frame, "marker", boundary_split="validation")


def test_no_validation_derived_boundaries_when_training_marker_missing():
    frame = _survival_frame()
    frame.loc[frame.split.eq("train"), "marker"] = np.inf
    assert tertile_curves(frame, "marker", minimum_per_group=50).empty


def test_patient_overlap_and_duplicate_rank_one_rejected():
    frame = _survival_frame()
    frame["subject_id"] = np.arange(len(frame))
    frame["ecg_rank"] = 1
    duplicate = frame.iloc[[0]].copy()
    duplicate["split"] = "validation" if frame.iloc[0].split == "train" else "train"
    with pytest.raises(ValueError, match="overlap"):
        tertile_curves(pd.concat([frame, duplicate]), "marker")
    with pytest.raises(ValueError, match="multiple index"):
        tertile_curves(pd.concat([frame, frame.iloc[[0]]]), "marker")
