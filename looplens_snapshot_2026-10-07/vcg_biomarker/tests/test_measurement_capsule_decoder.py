from copy import deepcopy

import pytest

from vcg_biomarker.core.measurement_capsule import POINT_FIELDS, resolve_point


def capsule():
    return {
        "version": "1.1.0", "point_fields": list(POINT_FIELDS), "signal_samples": 100,
        "fiducial_vocabulary": {"sources": ["fusion"], "evidence_sources": ["lead"]},
    }


def test_decoder_obeys_declared_field_order_and_preserves_legacy():
    data = capsule()
    point = [12, 0.8, 2.0, 1.0, 0, [0]]
    expected = resolve_point(data, point)
    data["point_fields"] = list(reversed(POINT_FIELDS))
    assert resolve_point(data, list(reversed(point))) == expected
    assert resolve_point({"version": "1.0.0"}, expected) == expected


@pytest.mark.parametrize("index", [-1, 1, True])
def test_decoder_rejects_bad_vocabulary_indices(index):
    with pytest.raises(ValueError, match="vocabulary_index"):
        resolve_point(capsule(), [12, 0.8, 2.0, 1.0, index, [0]])


@pytest.mark.parametrize("sample", [-1, 100, 1.5, True, None])
def test_decoder_rejects_invalid_sample(sample):
    with pytest.raises(ValueError, match="sample"):
        resolve_point(capsule(), [sample, 0.8, 2.0, 1.0, 0, [0]])


def test_unknown_version_and_truncated_or_duplicate_schema_fail_closed():
    data = capsule()
    data["version"] = "2.0.0"
    with pytest.raises(ValueError, match="version"):
        resolve_point(data, None)
    data = capsule()
    with pytest.raises(ValueError, match="schema"):
        resolve_point(data, [1, 0.8])
    damaged = deepcopy(data)
    damaged["point_fields"][1] = "sample"
    with pytest.raises(ValueError, match="schema"):
        resolve_point(damaged, [12, 0.8, 2.0, 1.0, 0, [0]])
