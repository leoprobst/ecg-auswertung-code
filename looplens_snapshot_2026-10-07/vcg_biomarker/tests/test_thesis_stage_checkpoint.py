"""A late interruption must not recompute verified earlier theses."""
import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.thesis_statistics import ThesisResult
from vcg_biomarker.mimic.thesis_tests import _checkpointed_thesis_rows


def test_verified_thesis_block_resumes_with_identical_rows(tmp_path):
    calls = []

    def calculate():
        calls.append(1)
        return [ThesisResult("T1", "effect", "positive", estimate=2.5,
                             n_units=120, extra={"diagnostic": "stable"})]

    first = _checkpointed_thesis_rows(tmp_path, thesis_id="T1", input_hash="abc",
                                     calculate=calculate)
    resumed = _checkpointed_thesis_rows(tmp_path, thesis_id="T1", input_hash="abc",
                                       calculate=calculate)
    assert calls == [1]
    pd.testing.assert_frame_equal(pd.DataFrame(first), pd.DataFrame(resumed),
                                  check_dtype=False)
    assert resumed[0]["estimate"] == 2.5


def test_changed_inputs_and_damaged_checkpoint_do_not_reuse_rows(tmp_path):
    calls = []

    def calculate():
        calls.append(1)
        return [ThesisResult("T2", "effect", "positive", estimate=float("nan"))]

    _checkpointed_thesis_rows(tmp_path, thesis_id="T2", input_hash="first",
                             calculate=calculate)
    _checkpointed_thesis_rows(tmp_path, thesis_id="T2", input_hash="second",
                             calculate=calculate)
    assert len(calls) == 2
    checkpoint = tmp_path / ".thesis_blocks" / "second" / "T2.checkpoint.json"
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    record["output_integrity"] = []
    checkpoint.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Damaged or stale"):
        _checkpointed_thesis_rows(tmp_path, thesis_id="T2", input_hash="second",
                                 calculate=calculate)
    assert len(calls) == 2


def test_force_recomputes_and_replaces_block(tmp_path):
    calls = []

    def calculate():
        calls.append(1)
        return [ThesisResult("T3", "effect", "positive", estimate=float(len(calls)))]

    _checkpointed_thesis_rows(tmp_path, thesis_id="T3", input_hash="same",
                             calculate=calculate)
    second = _checkpointed_thesis_rows(tmp_path, thesis_id="T3", input_hash="same",
                                      calculate=calculate, force=True)
    assert len(calls) == 2
    assert second[0]["estimate"] == 2.0
