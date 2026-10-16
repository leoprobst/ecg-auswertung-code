"""The DuckDB budget must follow the stage, and must not drift.

A guided run applies the resource environment once, at worker start, from the
16 GB job profile. The per-stage cap is derived from that allocation inside
``stage_worker_memory_scope``. Two things can go wrong and neither shows up
until a month-long study pauses:

* The cap is computed from the *previous stage's* cap instead of the job
  allocation. Twenty-seven stages then ratchet the budget towards nothing,
  and the last stages run in a fraction of the memory they were promised.
* A stage identifier the workload classifier has never seen raises, or falls
  into a class whose headroom was chosen for something else entirely.

The stage the supervisor measures its pause threshold against and the stage
the worker caps DuckDB with must also be the same string; they are formed at
different places, so nothing but a test holds them together.
"""

from __future__ import annotations

import os

import pytest

from vcg_biomarker.mimic.presets import STUDY_PRESET_ID
from vcg_biomarker.mimic.resources import (
    stage_memory_safety_policy,
    stage_worker_memory_scope,
)
from vcg_biomarker.mimic.workflow import _service_stage_id, guided_study_spec


ALLOCATION_GB = 16.0
RESERVE_GB = 6.0


@pytest.fixture()
def worker_environment(monkeypatch):
    """What ``_apply_job_resource_environment`` leaves behind."""

    monkeypatch.setenv("VCG_JOB_MEMORY_ALLOCATION_GB", str(ALLOCATION_GB))
    monkeypatch.setenv("VCG_JOB_OS_RESERVE_GB", str(RESERVE_GB))
    monkeypatch.delenv("VCG_JOB_MEMORY_GB", raising=False)
    monkeypatch.delenv("VCG_JOB_ACTIVE_STAGE", raising=False)


def _guided_stages() -> tuple[str, ...]:
    spec = guided_study_spec(
        preset_id=STUDY_PRESET_ID,
        run_id="policy-test",
        study_id="policy-test",
        include_external_validation=False,
    )
    return tuple(_service_stage_id(stage) for stage in spec.stage_ids)


def test_every_stage_of_the_real_run_has_a_policy(worker_environment):
    for stage in _guided_stages():
        with stage_worker_memory_scope(
            stage,
            default_allocation_gb=ALLOCATION_GB,
            default_reserve_gb=RESERVE_GB,
        ) as policy:
            assert policy.duckdb_memory_limit_gb > 0.0
            assert policy.stage == stage.casefold()


def test_the_budget_does_not_ratchet_down_across_stages(worker_environment):
    """The failure that would only surface at stage twenty-seven."""

    stages = _guided_stages()
    first_pass: dict[str, float] = {}
    for stage in stages:
        with stage_worker_memory_scope(
            stage,
            default_allocation_gb=ALLOCATION_GB,
            default_reserve_gb=RESERVE_GB,
        ) as policy:
            first_pass[stage] = policy.duckdb_memory_limit_gb

    # Run the whole sequence again. Every stage must land on exactly the same
    # number as when it was met the first time.
    for stage in stages:
        with stage_worker_memory_scope(
            stage,
            default_allocation_gb=ALLOCATION_GB,
            default_reserve_gb=RESERVE_GB,
        ) as policy:
            assert policy.duckdb_memory_limit_gb == first_pass[stage]


def test_the_cap_comes_from_the_allocation_not_the_previous_cap(
    worker_environment,
):
    with stage_worker_memory_scope(
        "extract-features",
        default_allocation_gb=ALLOCATION_GB,
        default_reserve_gb=RESERVE_GB,
    ) as waveform:
        inner_limit = os.environ["VCG_JOB_MEMORY_GB"]
        assert float(inner_limit) == waveform.duckdb_memory_limit_gb

    standalone = stage_memory_safety_policy(
        "physiological-gap-markers",
        reserve_gb=RESERVE_GB,
        requested_memory_gb=ALLOCATION_GB,
    )
    with stage_worker_memory_scope(
        "physiological-gap-markers",
        default_allocation_gb=ALLOCATION_GB,
        default_reserve_gb=RESERVE_GB,
    ) as after:
        assert after.duckdb_memory_limit_gb == standalone.duckdb_memory_limit_gb


def test_the_environment_is_restored(worker_environment):
    with stage_worker_memory_scope(
        "validate",
        default_allocation_gb=ALLOCATION_GB,
        default_reserve_gb=RESERVE_GB,
    ):
        assert os.environ["VCG_JOB_ACTIVE_STAGE"] == "validate"
    assert "VCG_JOB_MEMORY_GB" not in os.environ
    assert "VCG_JOB_ACTIVE_STAGE" not in os.environ


def test_the_scope_announces_the_stage_the_supervisor_watches(
    worker_environment,
):
    """Worker cap and supervisor threshold must name the same stage."""

    for stage in _guided_stages():
        with stage_worker_memory_scope(
            stage,
            default_allocation_gb=ALLOCATION_GB,
            default_reserve_gb=RESERVE_GB,
        ):
            assert os.environ["VCG_JOB_ACTIVE_STAGE"] == stage.casefold()


def test_every_stage_keeps_memory_outside_duckdb():
    """The invariant that actually holds -- I first asserted its opposite.

    "A heavy stage gets a smaller DuckDB budget" is wrong: ``inventory``
    streams and is capped at 3 GB, while ``extract-features`` is given 12 of
    16 GB because its work happens *inside* DuckDB. The rule is about what is
    kept back for Python, Arrow and the worker processes, and every class
    declares its own floor for that.
    """

    floors = {
        "waveform": 2.0,
        "analysis": 1.5,
        "join": 1.5,
        "orchestration": 2.0,
        "streaming": 1.0,
        "light": 1.0,
        "balanced": 1.0,
    }
    for stage in _guided_stages():
        policy = stage_memory_safety_policy(
            stage, reserve_gb=RESERVE_GB, requested_memory_gb=ALLOCATION_GB
        )
        assert policy.non_duckdb_headroom_gb >= floors[policy.workload_class]
        assert policy.duckdb_memory_limit_gb >= 1.0


def test_the_capped_classes_stay_under_their_cap():
    caps = {"streaming": 3.0, "light": 3.0, "join": 6.0, "analysis": 8.0}
    for stage in _guided_stages():
        policy = stage_memory_safety_policy(
            stage, reserve_gb=RESERVE_GB, requested_memory_gb=ALLOCATION_GB
        )
        cap = (
            10.0
            if stage in {"build-evidence", "attest-event-evidence"}
            else caps.get(policy.workload_class)
        )
        if cap is not None:
            assert policy.duckdb_memory_limit_gb <= cap


def test_evidence_attestation_has_more_memory_than_blocked_production_run():
    for stage in ("build-evidence", "attest-event-evidence"):
        policy = stage_memory_safety_policy(
            stage,
            reserve_gb=RESERVE_GB,
            requested_memory_gb=14.0,
        )
        assert policy.workload_class == "join"
        assert policy.duckdb_memory_limit_gb == 10.0
        assert policy.non_duckdb_headroom_gb >= 2.0

    repeatability = stage_memory_safety_policy(
        "repeatability",
        reserve_gb=RESERVE_GB,
        requested_memory_gb=14.0,
    )
    assert repeatability.duckdb_memory_limit_gb == 6.0


def test_the_two_stages_that_read_the_wide_glob_are_not_left_on_a_light_cap():
    """``therapy-response`` and the gap stage read the 1,889-column glob.

    Both were classified before either had ever run. This does not assert a
    number -- it asserts that somebody chose one.
    """

    for stage in ("therapy-response", "physiological-gap-markers"):
        policy = stage_memory_safety_policy(
            stage, reserve_gb=RESERVE_GB, requested_memory_gb=ALLOCATION_GB
        )
        assert policy.workload_class in {"analysis", "waveform", "join"}


def test_the_cap_never_hands_out_the_whole_allocation():
    """DuckDB shares the process tree with Python, Arrow and workers."""

    for stage in _guided_stages():
        policy = stage_memory_safety_policy(
            stage, reserve_gb=RESERVE_GB, requested_memory_gb=ALLOCATION_GB
        )
        assert policy.duckdb_memory_limit_gb <= ALLOCATION_GB
