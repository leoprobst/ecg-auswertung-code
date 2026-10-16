"""Which screen targets are read off the ECG itself.

The broad phenotype screen tests every marker against every PheCodeX phenotype.
Some of those phenotypes are not independent of the ECG: they *are* an ECG
reading, written down as a diagnosis code.  A bundle branch block is diagnosed
by looking at a QRS complex; a pacemaker is visible in the trace; "long QT" is
a measured interval.  A marker that recovers them has demonstrated **analytic
concordance** with the cardiologist who read the same recording -- not
diagnostic accuracy against an independent standard.

This is not a hypothetical concern.  In the last development run the highest
replicated hits of the whole screen were, in order: right bundle branch block
(validation AUC 0.924), left bundle branch block (0.906), presence of a cardiac
pacemaker (0.868), presence of a defibrillator (0.862), heart block (0.846) and
chronic atrial fibrillation (0.843).  Every one of them is an ECG reading.
Presented beside a genuine finding they look like the strongest results in the
study, and they are the least informative ones in it.

The registry below is deliberately conservative in the other direction: a
phenotype is flagged only when the ECG is a *defining* part of how the
diagnosis is made, not merely a common finding.  Heart failure stays unflagged
-- an ECG contributes to the suspicion but the diagnosis rests on imaging,
peptides and clinical course.  Atrial fibrillation is flagged, because a rhythm
diagnosis without a rhythm strip is not made.

Flagged targets keep being screened and reported.  What changes is the claim
they may carry, and that they are separated in the report instead of ranked
against independent endpoints.

**Two vocabularies, one registry.**  The phenotype screen supplies written-out
names ("right bundle branch block"); the study's own endpoints supply
identifiers (``qtc_gt_480``, ``vt_vf_icd_or_event``).  The first version
matched only the first vocabulary, so every one of the 63 clinical endpoints
came back independent -- including the two QTc endpoints, whose label is a
threshold on the interval the recorder measured from the same tracing.  The
identifier rules were added for that and change no policy: each names a
phenotype the phrase rules already flag when it is spelled out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

ECG_DERIVED_REGISTRY_VERSION = "1.2.0"

#: The target is defined by reading the electrocardiogram.
CLASS_ECG_DEFINED = "ecg_defined_diagnosis"
#: The device that produces the pacing artefact is visible in the trace.
CLASS_ECG_VISIBLE_DEVICE = "ecg_visible_device"
#: No part of the diagnosis requires the ECG.
CLASS_INDEPENDENT = "independent_of_ecg"

#: What a hit against such a target is allowed to be called.
CLAIM_CEILING: Mapping[str, str] = {
    CLASS_ECG_DEFINED: "analytic_concordance_not_diagnostic_accuracy",
    CLASS_ECG_VISIBLE_DEVICE: "analytic_concordance_not_diagnostic_accuracy",
    CLASS_INDEPENDENT: "unrestricted_by_ecg_circularity",
}


@dataclass(frozen=True)
class EcgDerivedRule:
    rule_id: str
    pattern: str
    classification: str
    reason: str


# Matched case-insensitively against the phenotype name and identifier.  Written
# as word-boundary patterns so that "block" alone never matches -- "blockage",
# "nerve block" and "beta blocker" are not conduction disease.
_RULES: tuple[EcgDerivedRule, ...] = (
    EcgDerivedRule(
        "bundle_branch_block",
        r"\bbundle[- ]branch\b|\bbbb\b|\bfascicular block\b|\bhemiblock\b",
        CLASS_ECG_DEFINED,
        "Conduction block is defined by QRS morphology and duration.",
    ),
    EcgDerivedRule(
        "av_block",
        r"\batrioventricular block\b|\bav block\b|\bheart block\b"
        r"|\bwenckebach\b|\bmobitz\b|\btrifascicular\b|\bbifascicular\b",
        CLASS_ECG_DEFINED,
        "Degree of AV block is read from the P-QRS relationship.",
    ),
    EcgDerivedRule(
        "conduction_disorder_generic",
        r"\bconduction disorder\b|\bintraventricular conduction\b"
        r"|\bivcd\b|\bpre-?excitation\b|\bwolff\b|\bwpw\b",
        CLASS_ECG_DEFINED,
        "Conduction phenotypes are ECG descriptions.",
    ),
    EcgDerivedRule(
        "atrial_fibrillation_flutter",
        r"\batrial fibrillation\b|\batrial flutter\b|\bafib\b|\baflutter\b",
        CLASS_ECG_DEFINED,
        "A rhythm diagnosis requires a recording of the rhythm.",
    ),
    EcgDerivedRule(
        "other_tachyarrhythmia",
        r"\bsupraventricular tachycardia\b|\bventricular tachycardia\b"
        r"|\bventricular fibrillation\b|\bparoxysmal tachycardia\b"
        r"|\btorsade\b|\bre-?entry\b|\bectopic beats?\b|\bpremature beats?\b"
        r"|\bpremature (?:atrial|ventricular) (?:contraction|depolarization)\b",
        CLASS_ECG_DEFINED,
        "Arrhythmia classification is an ECG reading.",
    ),
    EcgDerivedRule(
        "bradyarrhythmia",
        r"\bsick sinus\b|\bsinus bradycardia\b|\bsinus tachycardia\b"
        r"|\bsinoatrial\b|\bsinus arrest\b|\bsinus node dysfunction\b",
        CLASS_ECG_DEFINED,
        "Sinus node phenotypes are defined by rate and P-wave behaviour.",
    ),
    EcgDerivedRule(
        "repolarization_syndrome",
        r"\blong qt\b|\bshort qt\b|\bqt prolongation\b|\bbrugada\b"
        r"|\bearly repolari[sz]ation\b|\bst[- ]segment\b|\bt[- ]wave\b",
        CLASS_ECG_DEFINED,
        "Repolarization syndromes are defined by interval or ST-T shape.",
    ),
    EcgDerivedRule(
        "axis_and_hypertrophy_pattern",
        r"\baxis deviation\b|\belectrocardiographic\b|\becg\b|\bekg\b"
        r"|\babnormal electrocardiogram\b",
        CLASS_ECG_DEFINED,
        "The phenotype names the electrocardiogram as its source.",
    ),
    EcgDerivedRule(
        "cardiac_device",
        r"\bpacemaker\b|\bdefibrillator\b|\bcardiac device\b"
        r"|\bcardiac resynchron\w*|\bpacing\b|\bpaced\b",
        CLASS_ECG_VISIBLE_DEVICE,
        "A paced beat is visible in the trace; the device is not a disease.",
    ),
    # The rules above were written against PheCodeX phenotype *names*, which
    # are written out in full. The study's own clinical endpoints are
    # identifiers and use abbreviations, so none of them matched: measured,
    # ``classify_target`` returned "independent" for every one of the 63
    # endpoints -- including ``qtc_gt_480``, whose label is cut from the
    # recorder's own QT measurement of the same tracing, which is circularity
    # of the most direct kind there is.
    #
    # These four add the identifier vocabulary. They change no policy: each
    # one names a phenotype the phrase rules above already flag when it is
    # spelled out.
    EcgDerivedRule(
        "measured_interval_endpoint",
        r"\bqtc[_ ]?(?:gt|ge|over|above)?[_ ]?\d*\b|\bqtc\b|\bjtc\b"
        r"|\bqt[_ ]prolong\w*|\bpr[_ ]prolong\w*"
        r"|\bqrs[_ ]widening\b|\bfirst[_ ]degree[_ ]av\b",
        CLASS_ECG_DEFINED,
        "The endpoint is a threshold on an interval the recorder measured "
        "from the same tracing.",
    ),
    EcgDerivedRule(
        "rhythm_endpoint_abbreviation",
        r"\baf\b|\bafl\b|\bnew[_ ]onset[_ ]af\b|\bchronic[_ ]af\b",
        CLASS_ECG_DEFINED,
        "As for atrial fibrillation spelled out: a rhythm diagnosis needs a "
        "rhythm recording.",
    ),
    EcgDerivedRule(
        "ventricular_arrhythmia_abbreviation",
        r"\bvt\b|\bvf\b|\bvt[_ ]vf\b|\bnsvt\b",
        CLASS_ECG_DEFINED,
        "As for ventricular tachycardia and fibrillation spelled out.",
    ),
    EcgDerivedRule(
        "conduction_endpoint_abbreviation",
        r"\brbbb\b|\blbbb\b|\blafb\b|\blpfb\b|\bifb\b",
        CLASS_ECG_DEFINED,
        "As for bundle branch block spelled out.",
    ),
)

# Checked *before* the rules above, because two kinds of identifier name the
# electrocardiogram without being one.
#
# Both were false positives of the first identifier version, and both are the
# damaging direction: they would have downgraded the study's *best* independent
# endpoints to analytic concordance.
#
# ``..._near_ecg`` is a laboratory value measured close in time to the
# recording -- ``hyperkalemia_near_ecg`` is a potassium result, which is
# exactly the independent reference standard this project wants, and the
# opposite of an ECG reading. The generic ``\becg\b`` rule was written for
# PheCodeX names such as "abnormal electrocardiogram", where naming the ECG
# does mean the phenotype is one.
#
# ``qt_prolonging_drug_exposure`` is a medication record. A prescription is not
# an interval measurement, however much the drug moves the interval.
_INDEPENDENT_GUARDS: tuple[EcgDerivedRule, ...] = (
    EcgDerivedRule(
        "laboratory_value_timed_to_the_ecg",
        r"near[_ ]ecg\b|near[_ ]ekg\b",
        CLASS_INDEPENDENT,
        "A laboratory value measured near the recording is an independent "
        "reference, not an electrocardiographic reading.",
    ),
    EcgDerivedRule(
        "medication_exposure",
        r"\b(?:drug|medication|therapy)[_ ]exposure\b|\bexposure[_ ]near\b"
        r"|\b\w+[_ ]exposure\b",
        CLASS_INDEPENDENT,
        "A prescription record is independent of the tracing, however much "
        "the drug moves it.",
    ),
)

_COMPILED_GUARDS = tuple(
    (rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in _INDEPENDENT_GUARDS
)

_COMPILED = tuple(
    (rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in _RULES
)


@dataclass(frozen=True)
class EcgDerivedVerdict:
    classification: str
    rule_id: Optional[str]
    reason: str
    claim_ceiling: str


_INDEPENDENT = EcgDerivedVerdict(
    CLASS_INDEPENDENT,
    None,
    "No registry rule matches; the target is treated as independent.",
    CLAIM_CEILING[CLASS_INDEPENDENT],
)


def classify_target(
    *,
    target_id: object = None,
    target_name: object = None,
) -> EcgDerivedVerdict:
    """Classify one screen target by its identifier and human-readable name."""

    haystack = " ".join(
        str(part).replace("_", " ")
        for part in (target_id, target_name)
        if part is not None and str(part).strip()
    )
    if not haystack:
        return _INDEPENDENT
    for rule, pattern in _COMPILED_GUARDS:
        if pattern.search(haystack):
            return EcgDerivedVerdict(
                CLASS_INDEPENDENT,
                rule.rule_id,
                rule.reason,
                CLAIM_CEILING[CLASS_INDEPENDENT],
            )
    for rule, pattern in _COMPILED:
        if pattern.search(haystack):
            return EcgDerivedVerdict(
                rule.classification,
                rule.rule_id,
                rule.reason,
                CLAIM_CEILING[rule.classification],
            )
    return _INDEPENDENT


def annotate_targets(
    target_ids: Iterable[object],
    target_names: Iterable[object],
) -> dict[str, list]:
    """Vectorised form for attaching the verdict to a results frame."""

    classifications: list[str] = []
    rule_ids: list[Optional[str]] = []
    ceilings: list[str] = []
    for target_id, target_name in zip(target_ids, target_names):
        verdict = classify_target(target_id=target_id, target_name=target_name)
        classifications.append(verdict.classification)
        rule_ids.append(verdict.rule_id)
        ceilings.append(verdict.claim_ceiling)
    return {
        "ecg_derived_class": classifications,
        "ecg_derived_rule": rule_ids,
        "ecg_circularity_claim_ceiling": ceilings,
    }


def is_ecg_derived(classification: object) -> bool:
    return str(classification) in {CLASS_ECG_DEFINED, CLASS_ECG_VISIBLE_DEVICE}
