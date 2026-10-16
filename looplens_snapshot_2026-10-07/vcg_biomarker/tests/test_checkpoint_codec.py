import hashlib

import pytest

from vcg_biomarker.mimic.checkpoint_codec import read_group, write_group, MAX_GROUP_BYTES


def test_compressed_checkpoint_roundtrips_nested_results_deterministically(tmp_path):
    payload = {"records": [{"marker": "QRS–T", "values": [None, .125, True, {"n": 150}]}]*50}
    path = tmp_path / "group.json.gz"
    metadata = write_group(path, payload)
    assert metadata["encoding"] == "gzip-json-v1"
    assert read_group(path, sha256=metadata["sha256"], uncompressed_bytes=metadata["uncompressed_bytes"]) == payload
    assert write_group(tmp_path / "other.json.gz", payload) == metadata


def test_tampered_group_rejected_before_decoding(tmp_path):
    path = tmp_path / "group.json.gz"
    metadata = write_group(path, {"records": [1]})
    path.write_bytes(path.read_bytes()+b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        read_group(path, sha256=metadata["sha256"], uncompressed_bytes=metadata["uncompressed_bytes"])


@pytest.mark.parametrize("length", [-1, True, MAX_GROUP_BYTES+1, "12"])
def test_invalid_decode_bound_rejected_without_reading_file(tmp_path, length):
    with pytest.raises(RuntimeError, match="decompression size"):
        read_group(tmp_path / "absent", sha256="a"*64, uncompressed_bytes=length)


def test_declared_size_mismatch_and_invalid_gzip_rejected(tmp_path):
    path = tmp_path / "group.json.gz"
    metadata = write_group(path, {"records": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="size mismatch"):
        read_group(path, sha256=metadata["sha256"], uncompressed_bytes=1)
    path.write_bytes(b"invalid gzip")
    with pytest.raises(RuntimeError, match="cannot be decoded"):
        read_group(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), uncompressed_bytes=12)


def test_atomic_failure_does_not_replace_prior_checkpoint(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import checkpoint_codec
    path = tmp_path / "group.json.gz"
    old = write_group(path, {"old": True})
    def fail(*args):
        raise OSError("synthetic replacement failure")
    monkeypatch.setattr(checkpoint_codec, "atomic_replace", fail)
    with pytest.raises(OSError):
        write_group(path, {"new": True})
    assert read_group(path, sha256=old["sha256"], uncompressed_bytes=old["uncompressed_bytes"]) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_encoded_input_bound_checked_even_with_plausible_declared_size(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import checkpoint_codec
    path = tmp_path / "oversized.json.gz"
    path.write_bytes(b"x"*20)
    monkeypatch.setattr(checkpoint_codec, "MAX_ENCODED_BYTES", 10)
    with pytest.raises(RuntimeError, match="compressed input exceeds"):
        read_group(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), uncompressed_bytes=1)


@pytest.mark.parametrize("digest", [None, "", "g"*64, True])
def test_invalid_digest_rejected_before_file_read(tmp_path, digest):
    with pytest.raises(RuntimeError, match="compressed digest"):
        read_group(tmp_path / "absent", sha256=digest, uncompressed_bytes=1)
