"""Which stages a person can meaningfully run on their own.

The GUI kept its own tuple of stage names, hand-maintained, and it had drifted:
twelve stages the backend could run were absent from the dropdown, including
``marker-redundancy``, which had been wired months earlier. Nothing failed --
the contract check only verifies that every *offered* stage is runnable, not
that every runnable stage is offered -- so the drift was silent and one-way.

``StageDescriptor`` already carries a ``technical_only`` field for exactly this
purpose. It was declared and never read by anything, which is why it could not
prevent the drift.

**The rule.** Every stage the service layer accepts belongs to exactly one of
three kinds, and a stage that belongs to none is a failure rather than a
default. That is the whole mechanism: adding a stage to the backend without
saying what it is now breaks a test, instead of quietly not appearing.

``individual``
    Meaningful to run alone: it reads declared inputs, writes declared
    outputs, and a person can decide to run just this. Everything the guided
    study runs is of this kind, plus a few extras that exist for repair work.

``orchestration``
    Runs other stages. Offering it next to its own members would let someone
    start a full study from a menu of single steps.

``internal``
    A supporting operation with no scientific result of its own -- freezing a
    run, taking a snapshot, rendering a report someone else configured. These
    are reachable through the interfaces that own them, not from a list of
    analysis steps.
"""

from __future__ import annotations

from typing import Iterable

from .services import MANAGED_SERVICE_STAGE_IDS


STAGE_EXPOSURE_VERSION = "1.0.0"

KIND_INDIVIDUAL = "individual"
KIND_ORCHESTRATION = "orchestration"
KIND_INTERNAL = "internal"

# Runs the others. Never offered beside its own members.
ORCHESTRATION_STAGES: frozenset[str] = frozenset({"study-run"})

# Supporting operations. Each is reachable where it belongs -- the freeze and
# snapshot buttons, the report configuration dialog, the utility dossier -- and
# none of them produces a scientific result that stands on its own.
INTERNAL_STAGES: frozenset[str] = frozenset(
    {
        "freeze",
        "snapshot",
        "report-build",
        "assess-clinical-utility",
    }
)


def stage_kind(stage_id: object) -> str:
    """What kind of thing this stage is."""

    name = str(stage_id)
    if name in ORCHESTRATION_STAGES:
        return KIND_ORCHESTRATION
    if name in INTERNAL_STAGES:
        return KIND_INTERNAL
    return KIND_INDIVIDUAL


def individually_runnable_stage_ids(
    managed: Iterable[str] = MANAGED_SERVICE_STAGE_IDS,
) -> tuple[str, ...]:
    """Every stage a person can choose to run by itself.

    Derived from what the service layer actually accepts, so a stage cannot be
    added to the backend and stay invisible in the interface. The order is the
    managed order, which follows the pipeline.
    """

    return tuple(
        name
        for name in dict.fromkeys(str(item) for item in managed)
        if stage_kind(name) == KIND_INDIVIDUAL
    )


def unclassified_stage_ids(
    managed: Iterable[str] = MANAGED_SERVICE_STAGE_IDS,
) -> tuple[str, ...]:
    """Stages the interface would offer without knowing what they are.

    Every managed stage falls into ``individual`` by default, which is the
    permissive direction: a new analysis stage appears in the interface rather
    than disappearing from it. This reports the ones with no descriptor, since
    those would be offered with a fallback label and no explanation.
    """

    from .user_guidance import STAGE_DESCRIPTORS

    described = {
        str(getattr(descriptor, "stage_id", ""))
        for descriptor in STAGE_DESCRIPTORS
    }
    return tuple(
        name
        for name in individually_runnable_stage_ids(managed)
        if name not in described
    )


__all__ = [
    "INTERNAL_STAGES",
    "KIND_INDIVIDUAL",
    "KIND_INTERNAL",
    "KIND_ORCHESTRATION",
    "ORCHESTRATION_STAGES",
    "STAGE_EXPOSURE_VERSION",
    "individually_runnable_stage_ids",
    "stage_kind",
    "unclassified_stage_ids",
]
