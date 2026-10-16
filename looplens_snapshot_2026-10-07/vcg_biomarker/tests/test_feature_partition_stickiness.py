"""Feature partitions must survive the cohort growing.

Partitions used to be positional chunks of a cohort ordered by
``(split, study_id)``, and the partition key carried the whole stage input
hash -- which in turn carried ``file_fingerprint`` of
``clinical_label_evidence.parquet``.  That fingerprint defaults to size plus
mtime, so *rebuilding* the labels invalidated every partition even when the
produced bytes were identical, and changing a label rule did so with certainty.

The consequence was measured on the real project: a label-rule fix would have
re-extracted all 630,044 recordings -- roughly nine days of compute -- although
no marker depends on a label.  Markers depend on the cohort only through the
records it selects.

These tests pin the three properties that make a label change cheap again:
the key is built from the selected records rather than from file timestamps,
an ECG keeps the partition it was first put into, and a partition's key does
not move when unrelated ECGs join the cohort.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vcg_biomarker.mimic.feature_extraction import (
    _assign_cohort_partitions,
    _cohort_record_payloads,
    _cohort_selection_hash,
    _load_partition_assignment,
)


def _cohort(study_ids, *, split: str = "train", age: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1000 + int(value) for value in study_ids],
            "study_id": [int(value) for value in study_ids],
            "split": [split] * len(study_ids),
            "ecg_time": pd.to_datetime(["2180-01-01 10:00:00"] * len(study_ids)),
            "waveform_path": [f"files/p{value}.dat" for value in study_ids],
            "age": [age] * len(study_ids),
            "sex": ["F"] * len(study_ids),
        }
    )


def _assign(cohort: pd.DataFrame, root, size: int = 2):
    return _assign_cohort_partitions(
        cohort,
        output_root=root,
        partition_size=size,
        force=False,
    )


def test_the_selection_hash_ignores_row_order_but_not_content():
    forward = _cohort_record_payloads(_cohort([3, 1, 2]))
    backward = _cohort_record_payloads(_cohort([2, 3, 1]))
    assert _cohort_selection_hash(forward) == _cohort_selection_hash(backward)

    changed = _cohort_record_payloads(_cohort([3, 1, 2], age=61))
    assert _cohort_selection_hash(changed) != _cohort_selection_hash(forward)


def test_an_added_ecg_leaves_every_existing_partition_untouched(tmp_path):
    before = {
        (split, index): tuple(int(v) for v in frame["study_id"])
        for split, index, frame in _assign(_cohort([10, 20, 30, 40]), tmp_path)
    }
    assert sorted(before.values()) == [(10, 20), (30, 40)]

    # 25 sorts between 20 and 30: positional chunking would have moved 30 and 40
    # into different partitions and invalidated both.
    after = {
        (split, index): tuple(int(v) for v in frame["study_id"])
        for split, index, frame in _assign(
            _cohort([10, 20, 25, 30, 40]), tmp_path
        )
    }
    for key, members in before.items():
        assert after[key] == members, f"partition {key} lost its membership"
    assert (25,) in set(after.values())


def test_an_ecg_that_leaves_the_cohort_only_shrinks_its_own_partition(tmp_path):
    _assign(_cohort([10, 20, 30, 40]), tmp_path)
    after = {
        (split, index): tuple(int(v) for v in frame["study_id"])
        for split, index, frame in _assign(_cohort([10, 30, 40]), tmp_path)
    }
    assert ("train", 0) in after and after[("train", 0)] == (10,)
    assert after[("train", 1)] == (30, 40)


def test_the_assignment_is_persisted_and_reused(tmp_path):
    _assign(_cohort([10, 20, 30]), tmp_path)
    stored = _load_partition_assignment(tmp_path)
    assert stored[10] == ("train", 0)
    assert stored[30] == ("train", 1)

    # A second call must not renumber anything.
    _assign(_cohort([10, 20, 30]), tmp_path)
    assert _load_partition_assignment(tmp_path) == stored


def test_splits_do_not_share_a_partition(tmp_path):
    cohort = pd.concat(
        [_cohort([10, 20]), _cohort([30, 40], split="validation")],
        ignore_index=True,
    )
    assigned = _assign(cohort, tmp_path, size=4)
    splits = {split for split, _, _ in assigned}
    assert splits == {"train", "validation"}
    for split, _, frame in assigned:
        assert set(frame["split"]) == {split}


@pytest.mark.parametrize("size", [1, 2, 3, 250])
def test_no_partition_exceeds_the_requested_size(tmp_path, size):
    assigned = _assign(_cohort(list(range(10, 30))), tmp_path, size=size)
    assert assigned
    for _, _, frame in assigned:
        assert len(frame) <= size
