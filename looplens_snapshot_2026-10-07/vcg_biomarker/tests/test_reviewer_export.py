import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.reviewer_export import export_reviewer_packets


def selection():
    return pd.DataFrame(dict(review_case_id=["internal-a"]*2+["internal-b"]*2,
        reviewer_slot=[1,2,1,2], endpoint_id="synthetic_endpoint", selection_probability=.5,
        review_stratum=["discordant"]*2+["negative"]*2, subject_id=[123,123,456,456],
        study_id=[111,111,222,222]))


def test_export_only_curated_signals_and_blinded_columns(tmp_path):
    calls = []
    def loader(row):
        calls.append(row.study_id)
        return np.arange(120, dtype=float).reshape(10,12), 500.
    root = tmp_path/"export"
    manifest = export_reviewer_packets(selection(),root,signal_loader=loader)
    assert calls == [111,222]  # loaded once, not once per reviewer
    assert manifest["n_reviewers"] == 2
    assert (root/"coordinator_private"/"selection_and_linkage.parquet").is_file()
    first = root/"reviewer_packets"/"slot-1"
    second = root/"reviewer_packets"/"slot-2"
    table = pd.read_csv(first/"cases.csv")
    assert set(table) == {"review_case_id","reviewer_slot","endpoint_id","blinded","pipeline_label_withheld"}
    assert not table.review_case_id.str.contains("internal").any()
    packet = json.loads((first/"packet_complete.json").read_text())
    for name,digest in packet["files_sha256"].items():
        assert hashlib.sha256((first/name).read_bytes()).hexdigest() == digest
    for path in first.glob("*.npz"):
        assert path.read_bytes() == (second/path.name).read_bytes()
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files) == {"signal_mv","sampling_rate_hz","lead_names"}
            assert arrays["signal_mv"].shape == (10,12)


def test_incomplete_signal_export_is_not_marked_complete(tmp_path):
    root = tmp_path/"incomplete"
    with pytest.raises(ValueError,match="Invalid twelve"):
        export_reviewer_packets(selection(),root,signal_loader=lambda _: (np.zeros((10,3)),500))
    assert not (root/"export_complete.json").exists()
    assert not list(root.rglob("packet_complete.json"))
    with pytest.raises(FileExistsError):
        export_reviewer_packets(selection(),root,signal_loader=lambda _: (np.zeros((10,12)),500))


def test_assignments_and_probability_validated_before_writing(tmp_path):
    for frame in (selection().iloc[:3],selection().assign(selection_probability=0),
                  pd.concat([selection(),selection().iloc[:1]])):
        with pytest.raises(ValueError):
            export_reviewer_packets(frame,tmp_path/"never-created",signal_loader=lambda _: None)
        assert not (tmp_path/"never-created").exists()
