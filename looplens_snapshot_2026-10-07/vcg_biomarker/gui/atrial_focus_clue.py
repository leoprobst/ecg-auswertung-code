"""Conservative V1 P-wave morphology clue for *confirmed focal* atrial tachycardia.

Not the full Kistler decision tree: this deliberately stops at broad regions,
because paraseptal sites overlap right/left atrial P-wave signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SOURCES = (
    "https://pubmed.ncbi.nlm.nih.gov/16949495/",
    "https://pubmed.ncbi.nlm.nih.gov/34217661/",
)
V1Pattern = Literal["negative", "positive_negative", "positive", "negative_positive", "isoelectric_positive", "isoelectric", "unclear"]


@dataclass(frozen=True)
class AtrialFocusClue:
    status: Literal["clue", "abstained"]
    region_hint: str | None
    explanation: str
    sources: tuple[str, str] = SOURCES
    probability: None = None


def focal_at_v1_clue(*, focal_at_confirmed: bool, p_wave_unobscured: bool,
                     p_quality_accepted: bool, v1_pattern: V1Pattern) -> AtrialFocusClue:
    if not focal_at_confirmed:
        return AtrialFocusClue("abstained", None, "Fokale atriale Tachykardie nicht bestätigt")
    if not p_wave_unobscured or not p_quality_accepted:
        return AtrialFocusClue("abstained", None, "Ektopische P-Welle in V1 nicht frei und ausreichend sicher messbar")
    if v1_pattern in {"negative_positive", "isoelectric_positive", "isoelectric"}:
        return AtrialFocusClue(
            "clue", "paraseptale Region mitbedenken",
            "Die aktualisierte Mappingkohorte fand diese V1-Muster bei vielen paraseptalen Foki. "
            "Weder Spezifität für paraseptal noch eine einzelne anatomische Stelle ist damit belegt.",
        )
    if v1_pattern in {"negative", "positive_negative"}:
        return AtrialFocusClue(
            "clue", "rechtsatrial vereinbar",
            "Älterer P-Wellen-Algorithmus: negative oder positiv/negative V1-P-Welle sprach für rechtsatrial. "
            "Septale Überlappung und Elektrodenlage beachten.",
        )
    if v1_pattern == "positive":
        return AtrialFocusClue(
            "clue", "linksatrial vereinbar",
            "Positive V1-P-Welle war in der älteren Kohorte mit linksatrialen Foki vereinbar. "
            "Kein Ausschluss anderer Regionen und keine Mappingbestätigung.",
        )
    return AtrialFocusClue("abstained", None, "V1-Morphologie unklar oder außerhalb der konservativen Regel")
