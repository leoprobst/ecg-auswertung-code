import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.pilot_viewmodel import PilotViewModel
from vcg_biomarker.mimic.storage import file_sha256


def archived_model(tmp_path, frame):
    relative = "original/master/ecg_master_index.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    frame.to_parquet(path)
    (tmp_path / "workspace.json").write_text(json.dumps(dict(
        version="1.0.0", production_resume_allowed=False, locked_test_read=False,
        run_id="fixture", pilot_sizes=[1000],
        files=[dict(path=relative, sha256=file_sha256(path))])), encoding="utf-8")
    model = PilotViewModel(tmp_path)
    return model, path


def test_pilot_uses_archived_positive_pacing_and_bandwidth(tmp_path):
    model, _ = archived_model(tmp_path, pd.DataFrame(dict(study_id=[12],
        machine_report=["Ventricular paced rhythm"], machine_bandwidth=["0.05-150 Hz"],
        original_sampling_rate=[250.])))
    result = model.acquisition(12)
    assert result["pacing_present"] is True
    assert result["upper_bandwidth_hz"] == 150
    assert result["original_sampling_rate_hz"] == 250


def test_missing_pacing_is_unknown_not_absent(tmp_path):
    model, _ = archived_model(tmp_path, pd.DataFrame({"study_id": [12]}))
    assert model.acquisition(12)["pacing_present"] is None
    assert model.acquisition(12)["upper_bandwidth_hz"] is None


def test_duplicate_identity_or_modified_archive_rejected(tmp_path):
    model, path = archived_model(tmp_path, pd.DataFrame({"study_id": [12, 12]}))
    with pytest.raises(ValueError, match="Ambiguous"):
        model.acquisition(12)
    pd.DataFrame({"study_id": [12], "changed": [True]}).to_parquet(path)
    with pytest.raises(ValueError, match="integrity"):
        model.acquisition(12)
