"""A pilot need not re-prove what the run it guarded has since shown.

Stages 14-17 exist to check that the pipeline runs and scales *before* days
are committed to the full extraction. Measured on 2026-08-28, re-running them
costs 23.9 hours (validate-pilot-1000) plus 1.9 (validate-pilot-10000). Their
reuse key covers ``mimic/storage.py``, so any edit there -- a docstring
counts -- buys those 26 hours again.

That guard is doing its job; what is wrong is re-proving a superseded claim.
Once the full extraction has completed on 630,044 recordings, the pilots'
question is answered by the thing they were guarding, and nothing downstream
reads their artifacts: the report and audit stages take the main run id.

The dangerous direction is the other one -- skipping a pilot when the
extraction has *not* finished would remove the only check standing between a
broken pipeline and several days of it. Most of this file tests that.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.presets import STUDY_PRESET_ID
from vcg_biomarker.mimic.feature_extraction import FEATURE_STAGE_VERSION
from vcg_biomarker.mimic.storage import StageManifest
from vcg_biomarker.mimic.workflow import (
    _pilot_purpose_already_served,
    _pilot_size,
    guided_study_spec,
    policy_panel_ids,
)


PILOT_STAGES = (
    "extract-features-pilot-1000",
    "validate-pilot-1000",
    "extract-features-pilot-10000",
    "validate-pilot-10000",
)


class _Config:
    def __init__(self, features_root):
        self.features_root = features_root


@pytest.fixture()
def project(tmp_path):
    panel_key = "_".join(policy_panel_ids(STUDY_PRESET_ID))
    root = tmp_path / "features" / panel_key / "development"
    root.mkdir(parents=True)
    return _Config(tmp_path / "features"), root


def _write_manifest(
    root,
    *,
    status="complete",
    outputs=("part-0.parquet",),
    stage_version=FEATURE_STAGE_VERSION,
):
    for name in outputs:
        (root / name).write_bytes(b"x")
    manifest = StageManifest(
        stage="extract-features",
        stage_version=stage_version,
        input_hash="in",
        config_hash="cfg",
        output_paths=tuple(str(root / name) for name in outputs),
        row_counts={"successful_ecgs": 630044},
        created_at="2026-08-26T03:51:26+00:00",
        status=status,
    )
    (root / "features.manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    return manifest


def test_without_a_finished_extraction_every_pilot_still_runs(project):
    config, _root = project
    for stage in PILOT_STAGES:
        assert _pilot_purpose_already_served(config, stage) is None


def test_a_finished_extraction_retires_the_pilots(project):
    config, root = project
    _write_manifest(root)
    for stage in PILOT_STAGES:
        reason = _pilot_purpose_already_served(config, stage)
        assert reason is not None
        assert "630,044" in reason


def test_an_unfinished_extraction_does_not_retire_them(project):
    """Half a run is not evidence; this is the direction that costs days."""

    config, root = project
    _write_manifest(root, status="source_missing")
    for stage in PILOT_STAGES:
        assert _pilot_purpose_already_served(config, stage) is None


def test_an_old_feature_version_never_retires_current_pilots(project):
    config, root = project
    _write_manifest(root, stage_version="3.0.0")
    for stage in PILOT_STAGES:
        assert _pilot_purpose_already_served(config, stage) is None


def test_a_deleted_output_sends_the_pilots_back_to_work(project):
    config, root = project
    _write_manifest(root, outputs=("part-0.parquet", "part-1.parquet"))
    (root / "part-1.parquet").unlink()
    for stage in PILOT_STAGES:
        assert _pilot_purpose_already_served(config, stage) is None


def test_an_unreadable_manifest_sends_the_pilots_back_to_work(project):
    config, root = project
    (root / "features.manifest.json").write_text("{ kaputt", encoding="utf-8")
    assert _pilot_purpose_already_served(config, PILOT_STAGES[0]) is None


def test_no_real_stage_is_ever_retired_by_this_rule(project):
    """The rule must reach pilots and nothing else."""

    config, root = project
    _write_manifest(root)
    spec = guided_study_spec(
        preset_id=STUDY_PRESET_ID,
        run_id="pilot-rule",
        study_id="pilot-rule",
        include_external_validation=False,
    )
    retired = {
        stage
        for stage in spec.stage_ids
        if _pilot_purpose_already_served(config, stage) is not None
    }
    assert retired == set(PILOT_STAGES)
    for stage in retired:
        assert _pilot_size(stage) is not None


def test_the_expensive_stages_are_untouched(project):
    config, root = project
    _write_manifest(root)
    for stage in ("extract-features", "validate", "therapy-response", "report"):
        assert _pilot_purpose_already_served(config, stage) is None


def test_the_panel_key_follows_the_run_not_the_study_preset(project):
    """A preview run writes a different feature set and has no full sweep.

    The first version always read the study preset's panel key, so a preview
    run would have consulted a directory it never writes -- and if the two
    ever shared a name, it would have retired a preview's pilots on the
    strength of a study extraction. Neither is defensible.
    """

    config, root = project
    _write_manifest(root)

    from vcg_biomarker.mimic.presets import PREVIEW_PRESET_ID

    study_key = "_".join(policy_panel_ids(STUDY_PRESET_ID))
    preview_key = "_".join(policy_panel_ids(PREVIEW_PRESET_ID))

    for stage in PILOT_STAGES:
        assert _pilot_purpose_already_served(
            config, stage, preset_id=STUDY_PRESET_ID
        ) is not None
        if preview_key != study_key:
            assert _pilot_purpose_already_served(
                config, stage, preset_id=PREVIEW_PRESET_ID
            ) is None
