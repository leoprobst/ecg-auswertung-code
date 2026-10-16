"""Clinician-gated, source-linked idiopathic outflow-tract PVC worksheet.

These published ECG indices are regional *research clues*, not a substitute
for electroanatomic activation/pace mapping.  No automatic PVC diagnosis or
patient-specific probability is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


BETENSKY_URL = "https://pubmed.ncbi.nlm.nih.gov/21616286/"
YOSHIDA_URL = "https://pubmed.ncbi.nlm.nih.gov/24612087/"


@dataclass(frozen=True)
class OutflowPVCInputs:
    pvc_confirmed: bool
    inferior_axis_confirmed: bool
    lbbb_pattern_confirmed: bool
    structurally_normal: bool
    sinus_no_bundle_block: bool
    same_morphology: bool
    pvc_transition: Literal["v3", "other", "unknown"]
    sinus_transition: Literal["earlier", "same", "later", "unknown"]
    pvc_v2_r_mv: float | None
    pvc_v2_s_mv: float | None
    pvc_v3_r_mv: float | None
    sinus_v2_r_mv: float | None
    sinus_v2_s_mv: float | None


@dataclass(frozen=True)
class OutflowPVCResult:
    status: Literal["candidate", "discordant", "abstained"]
    region: str | None
    reason: str
    v2_transition_ratio: float | None
    v2s_v3r_index: float | None
    criteria: tuple[str, ...]
    sources: tuple[str, str] = (BETENSKY_URL, YOSHIDA_URL)
    probability: None = None


def _amplitude(value: float | None) -> bool:
    return value is not None and isfinite(value) and value >= 0


def localize_outflow_pvc(inputs: OutflowPVCInputs) -> OutflowPVCResult:
    """Compute published indices, abstaining outside the intended population.

    V2 transition ratio = [R/(R+S)]PVC / [R/(R+S)]sinus (Betensky).
    V2S/V3R = S amplitude in PVC V2 / R amplitude in PVC V3 (Yoshida).
    Both amplitudes refer to a *single, morphologically matched PVC*, not a
    median beat which mixes sinus and ectopic complexes.
    """
    gates = (
        (inputs.pvc_confirmed, "Ventrikulärer ektoper Schlag nicht bestätigt"),
        (inputs.inferior_axis_confirmed, "Inferiore PVC-Achse nicht bestätigt"),
        (inputs.lbbb_pattern_confirmed, "LSB-artige PVC-Morphologie nicht bestätigt"),
        (inputs.structurally_normal, "Strukturelle Herzerkrankung nicht ausgeschlossen"),
        (inputs.sinus_no_bundle_block, "Sinus-QRS mit Schenkelblock oder nicht geprüft"),
        (inputs.same_morphology, "Keine einzelne homogene PVC-Morphologie bestätigt"),
    )
    for passed, reason in gates:
        if not passed:
            return OutflowPVCResult("abstained", None, reason, None, None, ())
    pvc_r, pvc_s, v3_r = inputs.pvc_v2_r_mv, inputs.pvc_v2_s_mv, inputs.pvc_v3_r_mv
    sinus_r, sinus_s = inputs.sinus_v2_r_mv, inputs.sinus_v2_s_mv
    if not all(_amplitude(value) for value in (pvc_s, v3_r)):
        return OutflowPVCResult("abstained", None, "Endliche V2-S- und V3-R-Amplituden des PVC fehlen", None, None, ())
    assert pvc_s is not None and v3_r is not None
    if v3_r <= 0:
        return OutflowPVCResult("abstained", None, "V3-R-Amplitude fehlt: V2S/V3R nicht definiert", None, None, ())
    index = pvc_s / v3_r
    index_side = "LVOT" if index <= 1.5 else "RVOT"
    criteria = [f"Yoshida V2S/V3R = {index:.3f} → {index_side} (Grenze 1,5)"]

    ratio = None
    ratio_side = None
    if inputs.pvc_transition == "v3":
        if all(_amplitude(value) for value in (pvc_r, sinus_r, sinus_s)) and (
            pvc_r is not None and sinus_r is not None and sinus_s is not None
            and pvc_r + pvc_s > 0 and sinus_r > 0 and sinus_r + sinus_s > 0
        ):
            ratio = (pvc_r / (pvc_r + pvc_s)) / (sinus_r / (sinus_r + sinus_s))
            ratio_side = "LVOT" if ratio >= 0.60 else "RVOT"
            criteria.append(f"Betensky V2-Transitionsverhältnis = {ratio:.3f} → {ratio_side} (Grenze 0,60)")
        else:
            criteria.append("Betensky nicht berechenbar: V2-R/S in Sinus oder PVC fehlt/ist degeneriert")
    else:
        criteria.append("Betensky-V2-Grenze nicht angewandt: nur für V3-Transition untersucht")

    if inputs.sinus_transition == "earlier":
        criteria.append("PVC-Transition später als Sinus: in der Betensky-Kohorte gegen LVOT")
        if index_side == "LVOT" or ratio_side == "LVOT":
            return OutflowPVCResult("discordant", None, "Übergang und Amplitudenregeln widersprechen sich", ratio, index, tuple(criteria))
    if ratio_side is not None and ratio_side != index_side:
        return OutflowPVCResult("discordant", None, "Publizierte Indizes widersprechen sich", ratio, index, tuple(criteria))
    return OutflowPVCResult(
        "candidate", index_side,
        "Nur orientierende Ausflusstrakt-Region; EKG-Oberflächenmorphologie ersetzt kein invasives Mapping. "
        "Publizierte Genauigkeiten sind keine fallbezogenen Wahrscheinlichkeiten.",
        ratio, index, tuple(criteria),
    )
