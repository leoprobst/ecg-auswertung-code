from __future__ import annotations

import json

from vcg_biomarker.core.delineation.benchmark_suite import (
    DELINEATION_RELEASE_POLICY_VERSION,
    DELINEATION_SUITE_VERSION,
    delineation_implementation_hash,
)
from vcg_biomarker.mimic.services import (
    _directory_stat_fingerprint,
    _load_cached_delineation_suite,
    _write_delineation_cache_attestation,
)


def _request() -> dict:
    return {
        "cache_version": "1.0.0",
        "implementation_hash": delineation_implementation_hash(),
    }


def _result() -> dict:
    return {
        "status": "complete",
        "implementation_hash": delineation_implementation_hash(),
        "suite_version": DELINEATION_SUITE_VERSION,
        "release_policy_version": DELINEATION_RELEASE_POLICY_VERSION,
    }


def test_delineation_cache_requires_matching_request_and_output(tmp_path) -> None:
    output = tmp_path / "benchmark_suite.json"
    output.write_text(json.dumps(_result()), encoding="utf-8")
    request = _request()
    _write_delineation_cache_attestation(output, request)

    assert _load_cached_delineation_suite(output, request) == _result()
    assert _load_cached_delineation_suite(output, {**request, "strict": True}) is None

    output.write_text(json.dumps({**_result(), "unexpected": True}), encoding="utf-8")
    assert _load_cached_delineation_suite(output, request) is None


def test_directory_stat_fingerprint_detects_dataset_change(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    record = source / "record.dat"
    record.write_bytes(b"first")
    first = _directory_stat_fingerprint(source)

    record.write_bytes(b"changed-size")
    second = _directory_stat_fingerprint(source)

    assert first["file_count"] == second["file_count"] == 1
    assert first["content_stat_hash"] != second["content_stat_hash"]
