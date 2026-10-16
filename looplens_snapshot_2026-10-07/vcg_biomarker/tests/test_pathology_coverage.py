"""What the endpoint list covers, and the two gaps that were measured shut.

The list holds *clinical references* -- echocardiography, laboratory values,
coded diagnoses, outcomes -- against which ECG markers are tested. It is
deliberately not a list of ECG readings: a reading of the same tracing is a
circular reference, and this project has measured the size of that
circularity (PTB-XL cardiologist labels reached AUC 0,891 against 0,68 for an
independent echocardiographic reference). So the 3.120 distinct statements the
machine reports carry are correctly absent from it.

Two things were genuinely missing, and both were found by measuring rather
than by reading:

* The machine report's pacemaker pattern matched 28.858 of 46.419 pacing
  mentions across all 800.035 reports and missed 17.561 -- "demand pacing"
  (6.608), "atrial pacing" (4.691), "demand atrial pacing" (2.695). That label
  is what excludes paced recordings from the voltage, Q-wave and ischaemia
  criteria, so roughly 14.000 paced recordings were being read by rules that do
  not apply to them.
* Five conditions the cohort carries had no endpoint at all, among them
  prevalent atrial fibrillation: 26.446 subjects with an ECG and 282.623
  recordings, where the list held only the three prospective new-onset
  horizons and prevalent AF was an exclusion criterion rather than a target.
"""

from __future__ import annotations

import inspect

import pytest

from vcg_biomarker.mimic import labels as labels_module
from vcg_biomarker.mimic.ecg_derived_targets import classify_target
from vcg_biomarker.mimic.label_registry import (
    LABEL_RULE_VERSION,
    default_label_definitions,
)
from vcg_biomarker.mimic.mappings import ICD_GROUPS
from vcg_biomarker.mimic.report_parser import (
    REPORT_PATTERNS,
    MachineReportParser,
    ReportMentionState,
)


NEW_ENDPOINTS = (
    "pacemaker_carrier",
    "pericardial_effusion",
    "cardiac_amyloidosis",
    "hypertrophic_cardiomyopathy_icd",
    "copd_icd",
    "prevalent_atrial_fibrillation",
)


# --- the pacemaker pattern ----------------------------------------------------


@pytest.mark.parametrize(
    "report",
    [
        "Sinus rhythm | Demand pacing",
        "Atrial pacing | Left axis deviation",
        "Demand atrial pacing",
        "A-V sequential pacemaker",
        "Ventricular-paced complexes",
        "Atrial-sensed ventricular-paced complexes",
        "Pacemaker rhythm - no further analysis",
        "Ventricular pacing",
        "AV sequential pacing",
    ],
)
def test_every_measured_pacing_phrase_is_recognised(report):
    """The nine phrasings that carry the 46.419 pacing mentions.

    The three literal phrases this used to match left "demand pacing",
    "atrial pacing" and the hyphenated "-paced complexes" forms outside the
    label, and with them the exclusion that keeps paced recordings out of the
    voltage and Q-wave criteria.
    """

    parsed = MachineReportParser().parse(report)
    assert parsed["pacemaker"].state == ReportMentionState.POSITIVE


def test_a_failed_pacer_detection_is_not_a_paced_rhythm():
    """1.292 reports say the pacer *detection* was suspended by noise. That
    reports a failure to detect a pacemaker, not the presence of one, so
    "pacer" is deliberately not among the stems."""

    parsed = MachineReportParser().parse(
        "Pacer detection suspended due to external noise-review advised"
    )
    assert parsed["pacemaker"].state == ReportMentionState.NOT_MENTIONED


def test_an_unpaced_report_stays_negative():
    parsed = MachineReportParser().parse("Sinus rhythm | Normal ECG")
    assert parsed["pacemaker"].state == ReportMentionState.NOT_MENTIONED


def test_the_pacing_stems_do_not_reach_beyond_pacing():
    """Every stem has to contain "pac"; a looser one would silently mark
    unrelated findings as paced and drop them from the classical criteria."""

    assert all("pac" in pattern for pattern in REPORT_PATTERNS["pacemaker"])


# --- persisting what the parser reads ----------------------------------------


def test_all_fourteen_findings_are_stored_by_the_parse_reports_stage():
    """They are stored, and a first look that only checked the feature table
    concluded the opposite.

    ``build_machine_report_labels`` writes every parsed finding as
    ``report_<id>_state`` into ``machine_report_labels.parquet``, and
    ``validation.py`` joins that table beside the features. The findings were
    therefore always available to the analysis; what was missing was a stage
    that used them.
    """

    from vcg_biomarker.mimic.feature_extraction import build_machine_report_labels

    source = inspect.getsource(build_machine_report_labels)
    assert 'output[f"report_{label_id}_state"] = label.state.value' in source
    assert len(REPORT_PATTERNS) == 14


def test_the_feature_record_does_not_keep_a_second_copy():
    """One fact, one producer. A duplicate computed in a different stage sits
    behind a different reuse key, and two copies of the same finding are how
    they drift apart."""

    from vcg_biomarker.mimic import feature_record

    source = inspect.getsource(feature_record)
    assert 'base[f"report_{label_id}"]' not in source


def test_the_stored_state_keeps_all_four_values():
    """"not_mentioned" is not "negative": a report that says nothing about a
    finding has not ruled it out, and collapsing the two would invent
    negatives for every silent report."""

    assert len(set(ReportMentionState)) == 4


# --- the five new endpoints ---------------------------------------------------


def test_endpoint_registry_includes_separate_rwma_documentation_proxy():
    definitions = default_label_definitions()
    # 80 + concentric and eccentric LVH by ASE geometry (audit D5)
    # + troponin rise from undetectable (audit 5.2.3)
    # + conduction progression within one year (audit 5.2.5).
    assert len(definitions) == 84
    assert len({item.label_id for item in definitions}) == len(definitions)
    assert "regional_wall_motion_documentation_proxy" in {item.label_id for item in definitions}
    assert all(item.rule_version == LABEL_RULE_VERSION for item in definitions)


def test_each_new_endpoint_is_defined_and_has_a_rule():
    ids = {item.label_id for item in default_label_definitions()}
    source = inspect.getsource(labels_module)
    for label_id in NEW_ENDPOINTS:
        assert label_id in ids, label_id
        assert f'"{label_id}"' in source, label_id


def test_prevalent_af_reads_the_diagnosis_and_never_the_machine_report():
    """The decisive line of the whole addition.

    ``first_af_event_time`` is the earlier of the coded diagnosis and the
    machine report. Building the endpoint on it would mean testing markers
    computed from a tracing against a reading of that same tracing, which is
    the circularity this project has already quantified. ``first_af_dx_time``
    is the ICD path alone.
    """

    source = inspect.getsource(labels_module)
    start = source.index('"prevalent_atrial_fibrillation": _history_rule(')
    rule = source[start : start + 400]
    assert "first_af_dx_time" in rule
    assert "first_af_event_time" not in rule
    assert "current_report_af_state" not in rule
    assert "first_af_report_time" not in rule


def test_prevalent_af_fills_a_gap_the_prospective_endpoints_left():
    """The three new-onset horizons exclude prevalent AF by construction --
    it was an exclusion criterion and never a target."""

    ids = {item.label_id for item in default_label_definitions()}
    assert {"new_onset_af_30d", "new_onset_af_90d", "new_onset_af_1y"} <= ids
    assert "prevalent_atrial_fibrillation" in ids


# --- the ICD groups behind them ----------------------------------------------


def test_the_new_icd_groups_carry_the_codes_they_claim():
    groups = {group.group_id: group for group in ICD_GROUPS}
    assert groups["pericardial_effusion"].icd9_prefixes == ("4233",)
    assert groups["pericardial_effusion"].icd10_prefixes == ("I313",)
    assert "J44" in groups["copd"].icd10_prefixes
    assert "496" in groups["copd"].icd9_prefixes


def test_unspecified_pericardial_disease_is_left_out():
    """423.9 would add 886 subjects, but the endpoint exists to name fluid and
    an unspecified pericardial disease is not evidence of any."""

    groups = {group.group_id: group for group in ICD_GROUPS}
    assert "4239" not in groups["pericardial_effusion"].icd9_prefixes


def test_effusion_is_separate_from_pericarditis():
    """Low voltage and electrical alternans follow the fluid, not the
    inflammation. ``qrs_alternans_vector`` had only ``pericarditis_icd`` as a
    target, which is a different condition."""

    groups = {group.group_id: group for group in ICD_GROUPS}
    assert groups["pericarditis"].group_id != groups["pericardial_effusion"].group_id
    assert "I313" not in groups["pericarditis"].icd10_prefixes


def test_amyloidosis_and_hcm_reuse_the_groups_that_already_existed():
    """Both were already extracted as ICD groups and simply never became
    endpoints -- the codes were there the whole time."""

    ids = {group.group_id for group in ICD_GROUPS}
    assert {"systemic_amyloidosis", "hypertrophic_cardiomyopathy"} <= ids


def test_the_pacemaker_endpoint_never_asserts_a_case_from_the_tracing():
    """The asymmetry that makes it usable.

    Positives come from the structured record alone -- a device status code or
    an implantation procedure -- so the endpoint is not read off the tracing
    the markers are computed from. But 19.372 recordings the machine report
    calls paced carry no device code in the index encounter, because the
    device went in years earlier; calling those negative would put 19.372
    paced tracings into the control group of an endpoint about pacing. They
    are withheld instead: the report may remove a control, never create a
    case.
    """

    source = inspect.getsource(labels_module)
    start = source.index('"pacemaker_carrier": SQLLabelRule(')
    rule = source[start : start + 1800]
    # Everything up to the positive branch is what can make a case.
    positive = rule[: rule.index("THEN 'positive'")]
    assert "dx_cardiac_device_status" in positive
    assert "proc_pacemaker_or_icd" in positive
    assert "current_report_pacemaker_state" not in positive
    assert "current_report_pacemaker_state = 'positive'" in rule
    assert "'indeterminate'" in rule


def test_the_device_status_group_covers_a_device_implanted_earlier():
    """The procedure code only appears in the encounter that implanted the
    device. The status codes are what carry a pacemaker put in years ago."""

    groups = {group.group_id: group for group in ICD_GROUPS}
    assert "Z950" in groups["cardiac_device_status"].icd10_prefixes
    assert "V450" in groups["cardiac_device_status"].icd9_prefixes


def test_the_pacing_spike_marker_says_it_does_not_detect_pacing():
    """It computes on 94,3 % of paced recordings and separates them at AUC
    0,523 -- the 0,5-ms spike does not survive 500 Hz behind a 150 Hz analogue
    corner. Anyone reading only the name would conclude pacing detection is
    covered."""

    from vcg_biomarker.core.biomarkers.electrophysiology_markers import (
        _pacing_spike_energy_ratio,
    )

    doc = _pacing_spike_energy_ratio.__doc__ or ""
    assert "0,523" in doc
    assert "does not detect pacing" in doc
    assert "pacemaker_carrier" in doc


# --- the circularity guard ----------------------------------------------------


@pytest.mark.parametrize(
    "label_id",
    [
        "pericardial_effusion",
        "cardiac_amyloidosis",
        "hypertrophic_cardiomyopathy_icd",
        "copd_icd",
    ],
)
def test_the_four_structural_endpoints_are_independent_of_the_ecg(label_id):
    assert classify_target(target_id=label_id).classification == "independent_of_ecg"


def test_prevalent_af_is_still_flagged_as_ecg_defined_and_that_is_correct():
    """Reading it from ICD alone keeps it off *this* tracing, but a rhythm
    diagnosis is still made from some recording of the rhythm. The guard
    therefore flags it, as it flags the three new-onset horizons, and the claim
    ceiling that follows from that is the honest one."""

    verdict = classify_target(target_id="prevalent_atrial_fibrillation")
    assert verdict.classification == "ecg_defined_diagnosis"
    assert verdict.rule_id == "atrial_fibrillation_flutter"


# --- what deliberately did not change ----------------------------------------


def test_no_machine_report_statement_became_an_endpoint():
    """3.120 distinct statements appear on the recordings -- "sinus rhythm" on
    49 % of them, "left axis deviation" on 9,5 %. None is an endpoint, and
    none should be: an endpoint read from the tracing measures agreement
    between two readings rather than a finding."""

    ids = {item.label_id for item in default_label_definitions()}
    for reading in (
        "sinus_rhythm",
        "left_axis_deviation",
        "poor_r_wave_progression",
        "nonspecific_st_t_changes",
        "low_qrs_voltage",
        "lvh_by_report",
    ):
        assert reading not in ids
