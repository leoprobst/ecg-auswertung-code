"""A stage that is registered but not reachable is worse than no stage.

Four designs were added to the pipeline at once -- the paired within-subject
change, the voltage/mass discordance, the time-to-event arm, and the occlusion
yield. Each has to be listed, described, weighted, dispatchable and runnable,
and a gap in any one of those is invisible until a run reaches it hours in.

Two defects found by running them rather than by reading them, both pinned
below:

* the survival query said ``read_parquet`` while the runner handed it
  ``patients.csv.gz``, and the standardized copy renames ``dod`` to
  ``date_of_death``;
* the dose-response arm asked for ``heart_rate_bpm``, which no feature panel
  contains -- so the one marker whose direction is known in advance was
  silently missing from every result.
"""

from __future__ import annotations

import inspect

import pytest

from vcg_biomarker.mimic import services, workflow
from vcg_biomarker.mimic.live_progress import STAGE_WEIGHTS
from vcg_biomarker.mimic.user_guidance import STAGE_DESCRIPTORS


TARGETED = (
    "control-panel",
    "ext-icd-crosscheck",
    "two-stage-selection",
    "within-subject-change",
    "voltage-mass-discordance",
    "time-to-event",
    "occlusion-yield",
)

# Everything after the screens. The two that read ``screening_results`` run
# before them, so the ordering test below has to distinguish the two families.
AFTER_THE_SCREENS = (
    "within-subject-change",
    "voltage-mass-discordance",
    "time-to-event",
    "occlusion-yield",
)


def _descriptor_ids() -> set[str]:
    ids = set()
    for descriptor in STAGE_DESCRIPTORS:
        name = getattr(descriptor, "stage_id", None)
        if name is None:
            name = getattr(descriptor, "stage", None)
        if name is None and isinstance(descriptor, tuple):
            name = descriptor[0]
        if name is not None:
            ids.add(str(name))
    return ids


@pytest.mark.parametrize("stage", TARGETED)
def test_the_stage_is_part_of_the_guided_run(stage):
    assert stage in workflow.GUIDED_FINAL_STAGES


@pytest.mark.parametrize("stage", TARGETED)
def test_the_stage_can_be_dispatched(stage):
    """Registered in the run but absent from the dispatch would fail late."""

    source = inspect.getsource(services.run_service_stage)
    assert f'stage == "{stage}"' in source


@pytest.mark.parametrize("stage", TARGETED)
def test_the_stage_has_a_weight_and_a_description(stage):
    assert stage in STAGE_WEIGHTS, "ohne Gewicht verfaelscht die Stufe jede ETA"
    assert stage in _descriptor_ids(), "ohne Beschreibung bleibt sie im Ablauf stumm"


@pytest.mark.parametrize("stage", TARGETED)
def test_the_stage_runs_exactly_once_in_a_guided_run(stage):
    """Listed in two tuples, it would compute everything twice."""

    everything = (
        workflow.GUIDED_PREPARATION_STAGES
        + workflow.GUIDED_FULL_ANALYSIS_STAGES
        + workflow.GUIDED_FINAL_STAGES
    )
    assert list(everything).count(stage) == 1


def test_the_designs_run_after_the_screens_they_do_not_feed():
    """They read the feature partitions; nothing below reads them."""

    final = list(workflow.GUIDED_FINAL_STAGES)
    for stage in AFTER_THE_SCREENS:
        assert final.index(stage) > final.index("screen-broad-axes")
        assert final.index(stage) < final.index("report")


def test_the_control_panel_runs_before_the_open_ended_screens():
    """It answers whether the finished run is worth reading further, which is
    the question to settle before spending hours on the broad axes."""

    final = list(workflow.GUIDED_FINAL_STAGES)
    for stage in ("control-panel", "ext-icd-crosscheck", "two-stage-selection"):
        assert final.index(stage) < final.index("screen-broad-axes")


def test_the_selection_never_reads_the_validation_split():
    """The whole inferential argument is this one property."""

    from vcg_biomarker.mimic import two_stage_selection

    source = inspect.getsource(two_stage_selection.run_two_stage_selection)
    assert '"train"' in source
    assert "evaluation_split" in source


def test_the_control_panel_reads_the_same_table_everyone_else_reads():
    """A private copy of the numbers would make the check meaningless."""

    from vcg_biomarker.mimic import control_panel

    source = inspect.getsource(control_panel.run_control_panel)
    assert "screening_results.parquet" in source


def test_the_control_panel_reports_the_keys_its_verdict_actually_has():
    """The first version printed 0/0 for a panel that recovered 4 of 5."""

    from vcg_biomarker.mimic import control_panel

    source = inspect.getsource(control_panel.run_control_panel)
    for key in ("n_positive_recovered", "n_positive_tested"):
        assert key in source
    for absent in ("n_positive_passed", "n_positive_evaluated"):
        assert absent not in source


def test_the_ext_icd_check_compares_both_code_scopes():
    """One scope makes a correct pipeline look broken.

    Compared against the encounter codes alone, this project marked
    ``heart_failure_icd`` positive 21.458 times with no code present -- a 28 %
    apparent over-call that is entirely the window: the endpoint asks about the
    patient, the column answers about the admission. Against the history scope
    the same comparison leaves 32 disagreements out of 97.999.
    """

    from vcg_biomarker.mimic.ext_icd import ExtIcdPolicy, ext_icd_query

    policy = ExtIcdPolicy()
    assert policy.diagnosis_column != policy.wide_column
    sql = ext_icd_query("ext.csv", "master.parquet", policy=policy)
    assert policy.diagnosis_column in sql
    assert policy.wide_column in sql


def test_the_ext_icd_endpoints_all_exist_in_the_label_registry():
    """Six of the first eight names were invented and matched nothing."""

    from vcg_biomarker.mimic.ext_icd import CROSS_CHECK_PREFIXES
    from vcg_biomarker.mimic.label_registry import definitions_by_id

    known = set(definitions_by_id())
    missing = sorted(set(CROSS_CHECK_PREFIXES) - known)
    assert not missing, f"nicht im Endpunktregister: {missing}"


def test_the_ext_icd_folds_are_reported_but_never_adopted():
    """Swapping partitions would invalidate every threshold in the run."""

    import inspect

    from vcg_biomarker.mimic import ext_icd

    source = inspect.getsource(ext_icd.run_ext_icd_crosscheck)
    assert "folds_are_reported_not_adopted" in source
    fold_source = inspect.getsource(ext_icd.fold_overlap)
    assert "split" in fold_source


def test_the_ext_icd_summary_says_it_is_not_an_external_cohort():
    import inspect

    from vcg_biomarker.mimic import ext_icd

    source = inspect.getsource(ext_icd.run_ext_icd_crosscheck)
    assert "same_population" in source


# --- The two defects the pipeline run exposed --------------------------------


def test_the_survival_stage_reads_the_standardized_tables():
    """It said read_parquet and was handed csv.gz."""

    from vcg_biomarker.mimic import survival

    source = inspect.getsource(survival.run_time_to_event)
    assert "standardized_root" in source
    assert ".csv.gz" not in source


def test_the_survival_query_does_not_hard_code_the_death_column():
    """MIMIC calls it dod; the standardized copy calls it date_of_death."""

    from vcg_biomarker.mimic.survival import survival_frame_query

    signature = inspect.signature(survival_frame_query)
    assert "death_column" in signature.parameters
    sql = survival_frame_query(
        "m.parquet", "p.parquet", "a.parquet", "read_parquet('f/*.parquet')",
        marker_columns="f.x", death_column="date_of_death",
    )
    assert "date_of_death AS dod" in sql


def test_every_within_subject_marker_could_exist_in_a_panel():
    """``heart_rate_bpm`` was requested and silently absent from every result."""

    from vcg_biomarker.mimic.panels import default_feature_panels
    from vcg_biomarker.mimic.within_subject import WITHIN_SUBJECT_MARKERS

    known = {
        str(feature_id)
        for panel in default_feature_panels()
        for feature_id in panel.feature_ids
    }
    missing = sorted(set(WITHIN_SUBJECT_MARKERS) - known)
    assert not missing, f"nicht im Katalog: {missing}"


def test_the_dose_control_is_deliberately_outside_the_panels():
    """It is a context comparator, so it must come from the index instead."""

    from vcg_biomarker.mimic.feature_roles import is_candidate_biomarker
    from vcg_biomarker.mimic.within_subject import DOSE_CONTEXT_COLUMNS

    for name in DOSE_CONTEXT_COLUMNS:
        assert not is_candidate_biomarker(name.replace("machine_", ""))


def test_the_occlusion_stage_reads_the_label_table():
    from vcg_biomarker.mimic import occlusion

    source = inspect.getsource(occlusion.run_occlusion_yield)
    assert "clinical_labels_wide.parquet" in source


def test_the_voltage_mass_stage_reuses_the_existing_echo_pairing():
    from vcg_biomarker.mimic import voltage_mass

    source = inspect.getsource(voltage_mass.run_voltage_mass_discordance)
    assert "ecg_echo_matches.parquet" in source


def test_the_discordance_endpoints_are_not_made_of_the_denominator():
    """``lv_wall_thickening`` is the geometry the mass is computed from."""

    from vcg_biomarker.mimic.voltage_mass import DISCORDANCE_ENDPOINTS

    assert "lv_wall_thickening" not in DISCORDANCE_ENDPOINTS
    assert "echo_lvh_strict" not in DISCORDANCE_ENDPOINTS


@pytest.mark.parametrize("stage", TARGETED)
def test_every_stage_writes_a_manifest_it_can_be_resumed_from(stage):
    module_for = {
        "control-panel": ("control_panel", "run_control_panel"),
        "ext-icd-crosscheck": ("ext_icd", "run_ext_icd_crosscheck"),
        "two-stage-selection": ("two_stage_selection", "run_two_stage_selection"),
        "within-subject-change": ("within_subject", "run_within_subject_change"),
        "voltage-mass-discordance": ("voltage_mass", "run_voltage_mass_discordance"),
        "time-to-event": ("survival", "run_time_to_event"),
        "occlusion-yield": ("occlusion", "run_occlusion_yield"),
    }
    import importlib

    name, function = module_for[stage]
    module = importlib.import_module(f"vcg_biomarker.mimic.{name}")
    source = inspect.getsource(getattr(module, function))
    assert "StageManifest.load" in source, "ohne Wiederaufnahme rechnet sie immer neu"
    assert "previous.matches" in source
    assert "StageManifest.complete" in source
