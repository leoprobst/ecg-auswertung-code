"""The equivalence shadow build must not touch the canonical tree.

The attestation proves the evidence can be rebuilt from the event registry by
building a second copy and comparing it. That only works if the second copy is
genuinely separate. ``_ShadowEvidenceConfig`` redirects the roots it knows
about and forwards everything else to the real project config, so a write root
that nobody remembered to redirect silently lands in the canonical directory.

That happened: the shadow build wrote its quarantine artifacts over the real
ones. The canonical evidence checkpoint records their fingerprints, so the next
run found its own finished work invalidated by the attestation that was
supposed to leave it untouched -- and rebuilt, and was invalidated again.

This test pins the property rather than the list: every root the evidence stage
writes to has to resolve inside the shadow directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.event_equivalence import _ShadowEvidenceConfig


WRITTEN_ROOTS = ("evidence_root", "quarantine_root", "labels_root")
READ_ONLY_ROOTS = ("master_root", "standardized_root", "mimic_iv_root")


@pytest.fixture
def project(tmp_path) -> MimicProjectConfig:
    return MimicProjectConfig(
        work_root=tmp_path / "work",
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "mimic-ed",
        mimic_ecg_root=tmp_path / "mimic-ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-studies.csv",
        echo_record_list_path=tmp_path / "echo-records.csv",
    )


@pytest.fixture
def shadow(project, tmp_path) -> _ShadowEvidenceConfig:
    return _ShadowEvidenceConfig(project, tmp_path / "shadow-root")


def test_every_written_root_resolves_inside_the_shadow(shadow, tmp_path):
    shadow_root = (tmp_path / "shadow-root").resolve()
    for name in WRITTEN_ROOTS:
        resolved = Path(getattr(shadow, name)).resolve()
        assert shadow_root in resolved.parents or resolved == shadow_root, (
            f"{name} escapes the shadow directory: {resolved}"
        )


def test_the_catalog_and_feature_root_are_redirected_too(shadow, tmp_path):
    shadow_root = (tmp_path / "shadow-root").resolve()
    for name in ("catalog_path", "features_root"):
        resolved = Path(getattr(shadow, name)).resolve()
        assert shadow_root in resolved.parents


def test_read_only_roots_stay_shared(shadow, project):
    """Redirecting these would make the shadow read different source data."""

    for name in READ_ONLY_ROOTS:
        assert Path(getattr(shadow, name)) == Path(getattr(project, name))


def test_no_write_root_of_the_evidence_stage_is_left_unredirected(shadow, tmp_path):
    """Read the stage's own source rather than trusting a hand-kept list.

    Any ``config.<name>_root`` that the evidence builder writes to -- detected
    by it appearing on the left of a path join that is then written -- must be
    redirected. Rather than parse that precisely, this checks every root the
    module mentions that is not a known input, which errs towards flagging
    something new for a human to classify.
    """

    source = Path("vcg_biomarker/mimic/evidence.py").read_text(encoding="utf-8")
    mentioned = set(re.findall(r"config\.(\w+_root)\b", source))
    inputs = {
        "master_root",
        "standardized_root",
        "mimic_iv_root",
        "mimic_ed_root",
        "mimic_note_root",
        "mimic_echo_root",
        "external_ecg_root",
        "work_root",
    }
    shadow_root = (tmp_path / "shadow-root").resolve()
    unredirected = []
    for name in sorted(mentioned - inputs):
        value = getattr(shadow, name, None)
        if value is None:
            continue
        resolved = Path(value).resolve()
        if shadow_root not in resolved.parents and resolved != shadow_root:
            unredirected.append(name)
    assert not unredirected, (
        "the evidence stage writes to roots the shadow does not redirect: "
        f"{unredirected}"
    )
