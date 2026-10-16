import hashlib
import json
import numpy as np
import pytest

from vcg_biomarker.mimic.reviewer_packet import ReviewerPacket
from vcg_biomarker.mimic.reviewer_export import export_reviewer_packets
from vcg_biomarker.tests.test_reviewer_export import selection


def packet(tmp_path):
    root = tmp_path/"export"
    export_reviewer_packets(selection(),root,signal_loader=lambda _: (np.ones((100,12)),250.))
    return root/"reviewer_packets"/"slot-1"


def test_real_export_loads_exact_signal_and_rechecks_on_each_access(tmp_path):
    folder = packet(tmp_path)
    model = ReviewerPacket(folder)
    case = model.cases.review_case_id.iloc[0]
    signal, rate = model.signal(case)
    assert rate == 250.
    np.testing.assert_array_equal(signal,np.ones((100,12)))
    (folder/f"{case}.npz").write_bytes(b"changed after packet opening")
    with pytest.raises(ValueError,match="integrity"):
        model.signal(case)


def test_coordinator_or_unlisted_file_is_rejected(tmp_path):
    folder = packet(tmp_path)
    (folder/"private_labels.csv").write_text("not for reviewers")
    with pytest.raises(ValueError,match="Unlisted"):
        ReviewerPacket(folder)


def test_path_escape_even_in_manifest_is_rejected(tmp_path):
    folder = packet(tmp_path)
    path = folder/"packet_complete.json"
    manifest = json.loads(path.read_text())
    manifest["files_sha256"]["../private.npz"] = "0"*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Unapproved"):
        ReviewerPacket(folder)


def test_rehashed_invalid_signal_still_fails_contract(tmp_path):
    folder = packet(tmp_path)
    signal_path = next(folder.glob("*.npz"))
    np.savez_compressed(signal_path,signal_mv=np.ones((10,3)),sampling_rate_hz=250.,lead_names=["I","II","III"])
    manifest_path = folder/"packet_complete.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][signal_path.name] = hashlib.sha256(signal_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    model = ReviewerPacket(folder)
    with pytest.raises(ValueError,match="twelve-lead"):
        model.signal(signal_path.stem)
    with pytest.raises(ValueError,match="not assigned"):
        model.signal("../private")
