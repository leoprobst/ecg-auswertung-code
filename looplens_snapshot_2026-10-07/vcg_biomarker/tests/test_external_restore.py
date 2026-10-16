import hashlib
import io

import pytest

from vcg_biomarker.mimic.external_restore import missing_training_files, restore_one


def test_existing_files_are_never_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "training/a/part.mat"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    monkeypatch.setattr("vcg_biomarker.mimic.external_restore.urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    assert restore_one(tmp_path, {"relative_path": "training/a/part.mat", "sha256": "0"*64})["status"] == "already_present_not_modified"
    assert path.read_bytes() == b"original"


def test_wrong_download_hash_never_creates_target(tmp_path, monkeypatch):
    monkeypatch.setattr("vcg_biomarker.mimic.external_restore.urlopen", lambda *a, **kw: io.BytesIO(b"wrong"))
    row = restore_one(tmp_path, {"relative_path": "training/a/part.mat", "sha256": "0"*64})
    assert row["status"] == "failed"
    assert not (tmp_path / "training/a/part.mat").exists()


def test_valid_download_is_verified_and_created_exclusively(tmp_path, monkeypatch):
    payload = b"test signal"
    monkeypatch.setattr("vcg_biomarker.mimic.external_restore.urlopen", lambda *a, **kw: io.BytesIO(payload))
    row = restore_one(tmp_path, {"relative_path": "training/a/part.mat", "sha256": hashlib.sha256(payload).hexdigest()})
    assert row["status"] == "restored_verified"
    assert (tmp_path / "training/a/part.mat").read_bytes() == payload


def test_traversal_in_checksum_manifest_rejected(tmp_path):
    checksums = tmp_path / "SHA256SUMS.txt"
    checksums.write_text("0"*64 + " training/../../outside.mat\n")
    with pytest.raises(ValueError): missing_training_files(tmp_path, checksums)
