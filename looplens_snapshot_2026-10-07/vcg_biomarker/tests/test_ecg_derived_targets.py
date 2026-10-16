"""The screen must say which of its best hits are the ECG reading itself.

Measured on the last development run: of the 25 phenotypes with the highest
replicated validation AUC, 15 are diagnoses made by reading an electrocardiogram
-- including eight of the top nine.  Right bundle branch block at AUC 0.924 is
not a discovery about bundle branch block; it is agreement with the person who
read the same trace.

Two failure modes matter and both are tested here: missing a circular target
(the finding is then reported as diagnostic accuracy), and flagging an
independent one (a real finding is then discounted).  The second is the reason
the patterns use word boundaries -- "nerve block", "arterial blockage" and
"beta blocker" must not match the conduction rules.
"""

from __future__ import annotations

import pytest

from vcg_biomarker.mimic.ecg_derived_targets import (
    CLASS_ECG_DEFINED,
    CLASS_ECG_VISIBLE_DEVICE,
    CLASS_INDEPENDENT,
    annotate_targets,
    classify_target,
    is_ecg_derived,
)


@pytest.mark.parametrize(
    "name",
    [
        # The seven strongest hits of the last run, in order.
        "Right bundle branch block",
        "Left bundle branch block",
        "Heart block",
        "Chronic atrial fibrillation*",
        "Atrioventricular block, complete",
        "Ventricular fibrillation and flutter",
        "Persistent atrial fibrillation*",
        # Others the registry must not miss.
        "Atrial flutter",
        "Ventricular tachycardia",
        "Supraventricular tachycardia",
        "Long QT syndrome",
        "Sinoatrial node dysfunction",
        "Premature depolarization [Premature beats]",
        "Abnormal electrocardiogram [ECG] [EKG]",
        "Left anterior fascicular block",
        "Wolff-Parkinson-White syndrome",
        "Brugada syndrome",
    ],
)
def test_an_ecg_reading_is_recognised_as_one(name: str) -> None:
    verdict = classify_target(target_name=name)
    assert verdict.classification == CLASS_ECG_DEFINED, name
    assert is_ecg_derived(verdict.classification)
    assert verdict.claim_ceiling == (
        "analytic_concordance_not_diagnostic_accuracy"
    )


@pytest.mark.parametrize(
    "name",
    [
        "Presence of cardiac pacemaker",
        "Presence of cardiac defibrillator",
        "Presence of cardiac device",
        "Cardiac resynchronization therapy",
    ],
)
def test_a_visible_device_is_separated_from_a_diagnosis(name: str) -> None:
    verdict = classify_target(target_name=name)
    assert verdict.classification == CLASS_ECG_VISIBLE_DEVICE, name
    assert is_ecg_derived(verdict.classification)


@pytest.mark.parametrize(
    "name",
    [
        # Cardiac, but not diagnosed by reading the trace.
        "Heart failure",
        "Combined systolic and diastolic heart failure",
        "Ischemic cardiomyopathy*",
        "Cardiomyopathy",
        "Acute myocardial infarction",
        "Aortic valve stenosis",
        "Tricuspid valve disorders",
        "Cardiac arrest",
        "Heart valve replaced",
        # Non-cardiac, and containing words the patterns must not grab.
        "Nerve block procedure",
        "Arterial blockage",
        "Beta blocker adverse effect",
        "Intestinal blockage",
        "Chronic kidney disease",
        "Sepsis",
        "Diabetic retinopathy",
        "Ascites",
        "Major depressive disorder",
        "Pressure ulcer",
        "Myelodysplastic syndrome",
    ],
)
def test_an_independent_target_is_not_discounted(name: str) -> None:
    verdict = classify_target(target_name=name)
    assert verdict.classification == CLASS_INDEPENDENT, name
    assert not is_ecg_derived(verdict.classification)


def test_the_identifier_is_read_when_the_name_is_missing() -> None:
    verdict = classify_target(target_id="right_bundle_branch_block")
    assert verdict.classification == CLASS_ECG_DEFINED
    assert verdict.rule_id == "bundle_branch_block"


def test_an_empty_target_is_treated_as_independent() -> None:
    assert classify_target().classification == CLASS_INDEPENDENT
    assert classify_target(target_id="", target_name=None).classification == (
        CLASS_INDEPENDENT
    )


def test_the_frame_helper_returns_one_entry_per_target() -> None:
    columns = annotate_targets(
        ["a", "b", "c"],
        ["Left bundle branch block", "Sepsis", "Presence of cardiac pacemaker"],
    )
    assert columns["ecg_derived_class"] == [
        CLASS_ECG_DEFINED,
        CLASS_INDEPENDENT,
        CLASS_ECG_VISIBLE_DEVICE,
    ]
    assert columns["ecg_derived_rule"][1] is None
    assert len(columns["ecg_circularity_claim_ceiling"]) == 3
