"""A stage the backend can run but the interface never offers is invisible work.

The GUI kept a hand-written tuple of stage names. It had drifted in both
directions at once, and neither showed up as a failure:

* twelve runnable stages were missing from the dropdown, ``marker-redundancy``
  among them -- wired into the pipeline and unreachable by hand;
* four more were offered under the fallback label "Technische Aufgabe --
  Interner, nicht naeher beschriebener Verarbeitungsschritt", including
  ``screen-broad-axes``, which is the largest analysis in the project.

The existing contract check could not catch either, because it asks whether
every *offered* stage is runnable -- never whether every runnable stage is
offered. That direction is the one that goes wrong silently, and it is the one
tested here.

``StageDescriptor.technical_only`` had been declared for this purpose and was
read by nothing.
"""

from __future__ import annotations

import pytest

from vcg_biomarker.mimic.services import MANAGED_SERVICE_STAGE_IDS
from vcg_biomarker.mimic.stage_exposure import (
    INTERNAL_STAGES,
    KIND_INDIVIDUAL,
    KIND_INTERNAL,
    KIND_ORCHESTRATION,
    ORCHESTRATION_STAGES,
    individually_runnable_stage_ids,
    stage_kind,
    unclassified_stage_ids,
)
from vcg_biomarker.mimic.user_guidance import (
    STAGE_GROUP_NAMES_DE,
    stage_descriptor,
)


# --- The direction that went wrong silently ----------------------------------


def test_every_runnable_stage_is_either_offered_or_declared_not_to_be():
    """The check the old contract test was missing."""

    offered = set(individually_runnable_stage_ids())
    declared = ORCHESTRATION_STAGES | INTERNAL_STAGES
    unaccounted = set(MANAGED_SERVICE_STAGE_IDS) - offered - declared
    assert not unaccounted, f"weder angeboten noch eingestuft: {sorted(unaccounted)}"


def test_the_interface_list_is_derived_rather_than_written_out():
    """A second hand-maintained copy would drift again."""

    from vcg_biomarker.gui.widgets import mimic_project_widget

    assert tuple(mimic_project_widget.TECHNICAL_STAGE_IDS) == (
        individually_runnable_stage_ids()
    )


@pytest.mark.parametrize(
    "stage",
    (
        "marker-redundancy",
        "within-subject-change",
        "voltage-mass-discordance",
        "time-to-event",
        "occlusion-yield",
        "group-vcg",
        "refresh-phenome-promotions",
    ),
)
def test_the_stages_that_were_invisible_are_offered(stage):
    assert stage in individually_runnable_stage_ids()


# --- No stage may appear without an explanation -------------------------------


def test_no_offered_stage_falls_back_to_the_unnamed_label():
    """``screen-broad-axes`` was offered as an unnamed internal step."""

    assert unclassified_stage_ids() == ()


@pytest.mark.parametrize("stage", individually_runnable_stage_ids())
def test_every_offered_stage_names_itself_and_its_result(stage):
    descriptor = stage_descriptor(stage)
    assert descriptor.display_name_de != "Technische Aufgabe"
    assert descriptor.purpose_de.strip()
    assert descriptor.result_de.strip()


@pytest.mark.parametrize("stage", individually_runnable_stage_ids())
def test_every_offered_stage_lands_in_a_named_group(stage):
    """An unknown group id shows the stage under 'Technik' in the menu."""

    descriptor = stage_descriptor(stage)
    assert descriptor.group_id in STAGE_GROUP_NAMES_DE, (
        f"{stage} traegt die Gruppe {descriptor.group_id!r}, die es nicht gibt"
    )


def test_the_german_spelling_of_the_analysis_group_is_not_a_valid_id():
    """The four new stages first shipped with group 'analyse', which silently
    fell back to 'Technik'."""

    assert "analyse" not in STAGE_GROUP_NAMES_DE
    assert "analyze" in STAGE_GROUP_NAMES_DE


# --- The classification itself ------------------------------------------------


def test_the_guided_run_is_never_offered_beside_its_own_steps():
    assert stage_kind("study-run") == KIND_ORCHESTRATION
    assert "study-run" not in individually_runnable_stage_ids()


@pytest.mark.parametrize(
    "stage", ("freeze", "snapshot", "report-build", "assess-clinical-utility")
)
def test_supporting_operations_are_not_offered_as_analyses(stage):
    assert stage_kind(stage) == KIND_INTERNAL
    assert stage not in individually_runnable_stage_ids()


def test_an_unknown_stage_defaults_to_being_offered():
    """The permissive direction: a new analysis appears rather than vanishes."""

    assert stage_kind("some-new-analysis") == KIND_INDIVIDUAL


def test_the_order_follows_the_pipeline_not_the_alphabet():
    offered = individually_runnable_stage_ids()
    managed = [s for s in MANAGED_SERVICE_STAGE_IDS if s in set(offered)]
    assert list(offered) == managed


def test_a_stage_belongs_to_exactly_one_kind():
    assert not (ORCHESTRATION_STAGES & INTERNAL_STAGES)
