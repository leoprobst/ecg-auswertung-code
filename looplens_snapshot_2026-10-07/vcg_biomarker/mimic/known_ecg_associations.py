"""Curated cross-reference of already-published ECG/VCG marker-family ×
clinical-target-family associations.

This is deliberately small and will never be complete — its only job is to
let the broad discovery scan flag, for each replicated hit, whether it
plausibly matches a *known* published relationship (so a reviewer can
prioritize truly novel-looking hits) or has not been matched against
anything in this starter list yet. "Not yet described" is not a claim of
genuine novelty — only that this curated registry does not (yet) recognize
it; the registry is expected to grow over time as it is reviewed.

Entries were seeded from literature reviewed for this project (spatial
QRS-T angle / mortality, T-wave vector magnitude / mortality, P-terminal
force and interatrial block / atrial cardiomyopathy and AF, AI-ECG
age/BMI "gap" markers / mortality and cardiometabolic risk, classical LVH
voltage criteria, electrolyte and drug ECG effects, structural heart
disease screening). Matching is a simple, transparent, inspectable
substring match on the feature id and the clinical target id/name/domain
— not a claim of exhaustive literature coverage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownAssociation:
    association_id: str
    marker_family_tokens: tuple[str, ...]
    target_family_tokens: tuple[str, ...]
    summary_de: str
    source: str


KNOWN_ECG_ASSOCIATIONS: tuple[KnownAssociation, ...] = (
    KnownAssociation(
        "qrs_t_angle_mortality",
        ("qrs_t_angle", "spatial_qrs_t", "qrs_t_cosine", "qrs_t_mirror"),
        ("mortality", "death", "sudden_cardiac"),
        "Spatialer QRS-T-Winkel als etablierter Mortalitäts-/"
        "Risikoprädiktor.",
        "z. B. Kardar et al., Europace 2012; Yamazaki et al., "
        "Heart Rhythm 2005",
    ),
    KnownAssociation(
        "p_terminal_force_atrial_cardiomyopathy",
        ("p_terminal", "p_loop", "p_wave", "interatrial"),
        (
            "atrial_fibrillation",
            "af_",
            "atrial_cardiomyopathy",
            "stroke",
            "interatrial_block",
        ),
        "P-terminal force V1 / P-Wellen-Indizes als etablierte Marker "
        "für Vorhofkardiomyopathie und AF-/Schlaganfallrisiko.",
        "z. B. Kamel et al., Stroke 2015; Hisayama-Studie 2025",
    ),
    KnownAssociation(
        "voltage_lvh",
        (
            "cornell",
            "sokolow",
            "peguero",
            "qrs_amplitude",
            "r_wave_amplitude",
        ),
        ("lvh", "hypertrophy", "wall_thickness", "lv_mass"),
        "Klassische Voltage-Kriterien (Cornell, Sokolow-Lyon) für LVH "
        "sind seit Jahrzehnten etabliert.",
        "z. B. Casale et al., Circulation 1987",
    ),
    KnownAssociation(
        "low_voltage_amyloid_discordance",
        ("voltage", "amplitude"),
        ("amyloid", "wall_thickness", "lv_mass"),
        "Diskordanz aus niedriger QRS-Voltage trotz echokardiographischer "
        "Hypertrophie ist ein etabliertes Amyloidose-Red-Flag.",
        "z. B. Rapezzi et al., Circulation 2009",
    ),
    KnownAssociation(
        "electrolyte_ecg",
        ("t_wave", "qt_", "qrs_duration", "peaked"),
        ("potassium", "hyperkalemia", "electrolyte", "calcium", "magnesium"),
        "Elektrolytstörungen (v. a. Kalium) verändern T-Welle, QRS-Dauer "
        "und QT-Zeit auf etablierte Weise.",
        "z. B. Galloway et al., JAMA Cardiol 2019 (AI-ECG-Hyperkaliämie)",
    ),
    KnownAssociation(
        "age_bmi_gap_mortality",
        ("age_gap", "bmi_gap"),
        ("mortality", "death", "cardiovascular"),
        "KI-geschätztes ECG-Alter/-BMI minus tatsächlichem Wert ('Gap') "
        "ist ein etablierter, aufstrebender Mortalitäts-/Risikomarker.",
        "z. B. Ladejobi et al., EHJ Digital Health 2021/2024",
    ),
    KnownAssociation(
        "structural_heart_disease_screening",
        (
            "qrs_loop",
            "svg",
            "ejection",
            "spatial_ventricular_gradient",
        ),
        ("reduced_lvef", "heart_failure", "cardiomyopathy", "valve"),
        "ECG-/VCG-basierte Marker zur Erkennung struktureller "
        "Herzerkrankung (reduzierte EF, Klappenerkrankung) sind ein "
        "aktiver, gut belegter Forschungsbereich.",
        "z. B. Attia et al., Nat Med 2019; diverse AI-ECG-Reviews 2025",
    ),
)


def classify_novelty(
    *,
    feature_id: str,
    target_id: str,
    target_name: str = "",
    discovery_domain: str = "",
) -> str:
    """Return "known_literature_association" or "not_yet_described".

    Simple, transparent substring matching against the small curated
    registry above — not a claim that "not_yet_described" hits are
    actually novel, only that this starter registry does not recognize
    them yet.
    """
    feature_lower = str(feature_id or "").casefold()
    target_text = " ".join(
        str(value or "")
        for value in (target_id, target_name, discovery_domain)
    ).casefold()
    for entry in KNOWN_ECG_ASSOCIATIONS:
        marker_match = any(
            token in feature_lower for token in entry.marker_family_tokens
        )
        target_match = any(
            token in target_text for token in entry.target_family_tokens
        )
        if marker_match and target_match:
            return "known_literature_association"
    return "not_yet_described"


def known_association_registry_summary() -> tuple[dict, ...]:
    """Small, display-ready summary of the registry (for a future GUI
    "why interesting" tooltip or an about/methods panel)."""
    return tuple(
        {
            "association_id": entry.association_id,
            "summary_de": entry.summary_de,
            "source": entry.source,
        }
        for entry in KNOWN_ECG_ASSOCIATIONS
    )
