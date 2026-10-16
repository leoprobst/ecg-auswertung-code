"""The evidence resume checkpoint must not outlive the code that wrote it.

Two mechanisms decide whether the evidence stage recomputes: the manifest,
which is rejected when the stage version changes, and the per-group checkpoint,
which lets an interrupted build resume where it stopped. They have to agree.

If only the manifest carries the stage version, bumping it produces the worst
possible outcome -- worse than either extreme. The manifest is rejected, so the
stage runs; the checkpoint is accepted, so every group reports itself already
finished; and the stage writes a fresh manifest claiming the new version over
artifacts still built by the old code. The change silently never reaches the
data, and the manifest now asserts that it did.

Rejecting a checkpoint means *starting over*, not stopping. A checkpoint only
ever saves time, and everything it describes can be computed again; ending a
multi-day run over one would trade a few hours for several days.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.evidence import (
    EVIDENCE_CHECKPOINT_VERSION,
    EVIDENCE_STAGE_VERSION,
    _load_evidence_checkpoint,
)


def _checkpoint(tmp_path, **overrides):
    payload = {
        "checkpoint_version": EVIDENCE_CHECKPOINT_VERSION,
        "stage_version": EVIDENCE_STAGE_VERSION,
        "input_hash": "input-1",
        "config_hash": "config-1",
        "groups": {"laboratory": {"artifacts": []}},
    }
    payload.update(overrides)
    path = tmp_path / "evidence-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_matching_checkpoint_is_resumed(tmp_path):
    path = _checkpoint(tmp_path)
    payload = _load_evidence_checkpoint(
        path, input_hash="input-1", config_hash="config-1"
    )
    assert "laboratory" in payload["groups"]


def _assert_discarded(path):
    """A rejected checkpoint comes back empty, so the stage starts over."""

    payload = _load_evidence_checkpoint(
        path, input_hash="input-1", config_hash="config-1"
    )
    assert payload["groups"] == {}
    assert payload["stage_version"] == EVIDENCE_STAGE_VERSION
    assert payload["input_hash"] == "input-1"
    return payload


def test_a_checkpoint_from_another_stage_version_is_discarded(tmp_path):
    """The one this module exists for."""

    _assert_discarded(_checkpoint(tmp_path, stage_version="0.0.1-ancient"))


def test_a_checkpoint_predating_the_stage_version_field_is_discarded(tmp_path):
    """Files written before the field existed carry no claim about the code."""

    payload = json.loads(_checkpoint(tmp_path).read_text(encoding="utf-8"))
    del payload["stage_version"]
    path = tmp_path / "evidence-checkpoint.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _assert_discarded(path)


@pytest.mark.parametrize(
    "field, value",
    [
        ("checkpoint_version", "0"),
        ("input_hash", "input-2"),
        ("config_hash", "config-2"),
    ],
)
def test_the_existing_staleness_checks_still_hold(tmp_path, field, value):
    _assert_discarded(_checkpoint(tmp_path, **{field: value}))


def test_a_truncated_checkpoint_is_discarded_rather_than_fatal(tmp_path):
    """Killing the process mid-write must not cost the run."""

    path = tmp_path / "evidence-checkpoint.json"
    path.write_text('{"checkpoint_version": "1", "grou', encoding="utf-8")
    _assert_discarded(path)


def test_a_checkpoint_holding_a_json_scalar_is_discarded(tmp_path):
    path = tmp_path / "evidence-checkpoint.json"
    path.write_text("42", encoding="utf-8")
    _assert_discarded(path)


def test_a_fresh_checkpoint_records_the_stage_version(tmp_path):
    """Otherwise the next run cannot tell what produced it."""

    path = tmp_path / "evidence-checkpoint.json"
    _load_evidence_checkpoint(
        path, input_hash="input-1", config_hash="config-1"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage_version"] == EVIDENCE_STAGE_VERSION
    assert payload["groups"] == {}
