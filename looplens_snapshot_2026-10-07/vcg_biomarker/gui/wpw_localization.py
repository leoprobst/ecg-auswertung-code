"""Review-gated EASY-WPW lookup for a single manifest accessory pathway.

The published decision tree is reproduced; automatic waveform measurements
below are merely *proposals* and have not been independently validated as an
implementation of the paper's human-read ECG inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SOURCE = "El Hamriti et al., Europace 2023; doi:10.1093/europace/euac216"
SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9935024/"
DELTA_LEADS = ("II", "III", "aVR", "aVL")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class EASYWPWInputs:
    manifest_preexcitation_confirmed: bool
    single_pathway_assumed: bool
    structural_disease_excluded: bool
    prior_ablation_excluded: bool
    v1_polarity: Literal["positive", "negative", "isoelectric", "unknown"]
    most_positive_delta_lead: str
    transition: Literal["at_or_before_v3", "after_v3", "unknown"] = "unknown"


@dataclass(frozen=True)
class EASYWPWResult:
    status: Literal["candidate", "abstained"]
    region: str | None
    reason: str
    decision_path: tuple[str, ...]
    source: str = SOURCE
    source_url: str = SOURCE_URL
    probability: None = None


def easy_wpw_localize(inputs: EASYWPWInputs) -> EASYWPWResult:
    """Seven-region paper flowchart; never diagnose WPW from inputs alone."""
    gates = (
        (inputs.manifest_preexcitation_confirmed, "Manifeste Präexzitation nicht bestätigt"),
        (inputs.single_pathway_assumed, "Ein einzelner akzessorischer Leitungsweg nicht belegt"),
        (inputs.structural_disease_excluded, "Strukturelle Herzerkrankung nicht ausgeschlossen"),
        (inputs.prior_ablation_excluded, "Frühere Ablation/Narbe nicht ausgeschlossen"),
    )
    for passed, reason in gates:
        if not passed:
            return EASYWPWResult("abstained", None, reason, ())
    lead = inputs.most_positive_delta_lead
    if lead not in DELTA_LEADS:
        return EASYWPWResult("abstained", None, "Meistpositive Delta-Ableitung fehlt oder ist ungültig", ())
    if inputs.v1_polarity == "positive":
        region = {"aVL": "Mitralring posteroseptal", "II": "Mitralring posterolateral",
                  "aVR": "Mitralring posterolateral", "III": "Mitralring anterolateral"}[lead]
        path = ("V1 positiv → Mitralring", f"Delta am positivsten in {lead} → {region}")
    elif inputs.v1_polarity in {"negative", "isoelectric"}:
        if inputs.transition == "unknown":
            return EASYWPWResult("abstained", None, "Präkordiale QRS-Transition fehlt", ())
        if inputs.transition == "at_or_before_v3":
            region = "Trikuspidalring anteroseptal" if lead in {"II", "III"} else "Trikuspidalring posteroseptal"
        elif inputs.transition == "after_v3":
            region = {"aVL": "Trikuspidalring posterolateral", "II": "Trikuspidalring anterolateral",
                      "III": "Trikuspidalring anteroseptal", "aVR": "Trikuspidalring posteroseptal"}[lead]
        else:
            return EASYWPWResult("abstained", None, "Ungültige QRS-Transition", ())
        path = ("V1 negativ/isolelektrisch → Trikuspidalring",
                f"Transition {inputs.transition}", f"Delta am positivsten in {lead} → {region}")
    else:
        return EASYWPWResult("abstained", None, "V1-Polarität fehlt oder ist ungültig", ())
    return EASYWPWResult(
        "candidate", region,
        "Orientierende präinterventionelle Region, keine bestätigte Bündelposition; "
        "Mapping/Ablation bleibt Referenz. Keine fallbezogene Wahrscheinlichkeit.", path,
    )


def propose_easy_wpw_inputs(
    signal: np.ndarray, fs: float, qrs_onset: int, qrs_offset: int,
) -> dict[str, object]:
    """Conservative *measurement preview* for clinician review, not classification.

    Requires a manually accepted or otherwise verified QRS onset/offset.
    Pre-QRS baseline is intentionally not inferred when the window is absent.
    """
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 12 or not np.isfinite(samples).all():
        raise ValueError("Zwölf endliche Standardableitungen erforderlich")
    if not np.isfinite(fs) or fs < 250:
        raise ValueError("Für 20–40-ms-Delta-Vorschläge sind mindestens 250 Hz erforderlich")
    q0, q1 = int(qrs_onset), int(qrs_offset)
    if not (round(.05 * fs) <= q0 < q1 <= len(samples) and .06 <= (q1 - q0) / fs <= .24):
        raise ValueError("Geprüfte QRS-Grenzen und mindestens 50 ms Vorlauf erforderlich")
    baseline = samples[q0-round(.05*fs):q0-round(.025*fs)]
    early = samples[q0+round(.02*fs):q0+round(.04*fs)]
    if len(baseline) < 5 or len(early) < 5:
        raise ValueError("Zu wenige Samples für Baseline oder frühe Delta-Welle")
    reference = np.median(baseline, axis=0)
    noise = 1.4826 * np.median(np.abs(baseline - reference), axis=0)
    if np.any(noise > .04):
        raise ValueError("Instabile Pre-QRS-Baseline; Delta-Polarität manuell beurteilen")
    delta = np.median(early, axis=0) - reference
    ranked = sorted(((float(delta[LEADS.index(lead)]), lead) for lead in DELTA_LEADS), reverse=True)
    winner = ranked[0][1] if ranked[0][0] - ranked[1][0] >= .02 else None
    qrs = samples[q0:q1] - reference
    r = np.max(qrs, axis=0)
    s = -np.min(qrs, axis=0)
    v1_index = LEADS.index("V1")
    v1 = "isoelectric" if abs(r[v1_index]-s[v1_index]) < .05 else (
        "positive" if r[v1_index] > s[v1_index] else "negative"
    )
    first_transition = next((lead for lead in ("V1", "V2", "V3", "V4", "V5", "V6")
                             if r[LEADS.index(lead)] >= s[LEADS.index(lead)]
                             and r[LEADS.index(lead)] >= .1), None)
    transition = ("at_or_before_v3" if first_transition in {"V1", "V2", "V3"} else
                  "after_v3" if first_transition in {"V4", "V5", "V6"} else "unknown")
    return {
        "v1_polarity_proposal": v1,
        "transition_proposal": transition,
        "most_positive_delta_lead_proposal": winner,
        "delta_signed_mv": {lead: float(delta[LEADS.index(lead)]) for lead in DELTA_LEADS},
        "baseline_noise_mv_max": float(np.max(noise)),
        "status": "unvalidated_measurement_proposals_require_visual_confirmation",
    }
