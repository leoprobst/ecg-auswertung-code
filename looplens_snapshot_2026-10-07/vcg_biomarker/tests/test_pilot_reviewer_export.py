import json
import numpy as np
import pandas as pd
import pytest
import wfdb

from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.mimic.pilot_reviewer_export import export_pilot_review
from vcg_biomarker.mimic.pilot_workspace import WORKSPACE_VERSION
from vcg_biomarker.mimic.storage import file_sha256


def archive(tmp_path):
    original = tmp_path/"original"
    original.mkdir()
    wfdb.wrsamp("12",fs=250,units=["mV"]*12,sig_name=list(LEAD_NAMES_12),
                 p_signal=np.tile(np.sin(np.arange(100)/10)[:,None],(1,12)),fmt=["16"]*12,
                 write_dir=str(original))
    pd.DataFrame(dict(subject_id=[1],study_id=[12],waveform=["original/12"])).to_parquet(original/"case_index.parquet")
    files = [{"path":p.relative_to(tmp_path).as_posix(),"sha256":file_sha256(p)} for p in original.iterdir()]
    (tmp_path/"workspace.json").write_text(json.dumps(dict(version=WORKSPACE_VERSION,run_id="synthetic",
        production_resume_allowed=False,locked_test_read=False,files=files)))
    return pd.DataFrame(dict(subject_id=1,study_id=12,review_case_id="private",reviewer_slot=[1,2],
        endpoint_id="synthetic",selection_probability=1.,review_stratum="discordant"))


def test_verified_native_signal_export_is_overlay_only(tmp_path):
    selection = archive(tmp_path)
    before = {p.name:file_sha256(p) for p in (tmp_path/"original").iterdir()}
    manifest = export_pilot_review(tmp_path,selection,export_name="review-1")
    assert manifest["n_cases"] == 1
    packet = tmp_path/"overlay"/"review_exports"/"review-1"/"reviewer_packets"/"slot-1"
    with np.load(next(packet.glob("*.npz")),allow_pickle=False) as data:
        assert data["sampling_rate_hz"] == 250  # not resampled to analytical 500
        assert data["signal_mv"].shape == (100,12)
    assert before == {p.name:file_sha256(p) for p in (tmp_path/"original").iterdir()}


def test_wrong_identity_and_path_escape_rejected(tmp_path):
    selection = archive(tmp_path)
    with pytest.raises(ValueError,match="identity"):
        export_pilot_review(tmp_path,selection.assign(subject_id=2),export_name="wrong")
    with pytest.raises(ValueError,match="escapes"):
        export_pilot_review(tmp_path,selection,export_name="../../../outside")
    assert not (tmp_path/"overlay").exists()


def test_modified_signal_is_rejected_before_export(tmp_path):
    selection = archive(tmp_path)
    (tmp_path/"original"/"12.dat").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="integrity"):
        export_pilot_review(tmp_path,selection,export_name="corrupt")
    assert not (tmp_path/"overlay").exists()
