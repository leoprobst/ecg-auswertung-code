from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PathologyGroup:
    group_id: str
    display_name_de: str
    endpoint_ids: tuple[str, ...]
    description_de: str
    navigation_only: bool = True

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "display_name_de": self.display_name_de,
            "endpoint_ids": list(self.endpoint_ids),
            "description_de": self.description_de,
            "navigation_only": self.navigation_only,
            "pooling_policy": (
                "not_pooled"
                if self.navigation_only
                else "separately_predefined_composite"
            ),
        }


PATHOLOGY_GROUPS: tuple[PathologyGroup, ...] = (
    PathologyGroup(
        "acs_omi_stemi_equivalent",
        "ACS / OMI / STEMI-Äquivalente",
        (
            "acute_mi_high_specificity",
            "acute_mi_probable",
            "omi_revascularization_proxy",
        ),
        (
            "Akute koronare Ereignisproxies. Die Endpunkte bleiben einzeln; "
            "die Gruppe erleichtert nur den Vergleich mit ST- und "
            "Expertenmustern wie de Winter oder Wellens."
        ),
    ),
    PathologyGroup(
        "myocardial_injury_demand",
        "Myokardschädigung / Demand-Kontext",
        (
            "troponin_positive_any",
            "acute_troponin_change_proxy",
            "troponin_rise_from_undetectable",
            "myocardial_injury_demand_context",
        ),
        "Troponin- und Dynamikendpunkte ohne Gleichsetzung mit Typ-1-MI.",
    ),
    PathologyGroup(
        "lvh_remodeling",
        "LVH / Remodeling",
        (
            "echo_lvh_strict",
            "echo_lvh_sensitive",
            "lv_wall_thickening",
            "lv_dilatation_proxy",
            "concentric_remodeling_proxy",
        ),
        "Echo-basierte Strukturendpunkte und klar benannte Proxies.",
    ),
    PathologyGroup(
        "heart_failure_ef",
        "Herzinsuffizienz / Ejektionsfraktion",
        (
            "reduced_lvef_strict",
            "reduced_lvef",
            "preserved_lvef",
            "heart_failure_icd",
            "elevated_ntprobnp",
            "acute_decompensated_heart_failure_proxy",
        ),
        "Systolische Funktion, HF-Kontext und Dekompensationsproxy.",
    ),
    PathologyGroup(
        "scar_revascularization",
        "Infarktnarbe / Revaskularisation",
        (
            "regional_wall_motion_abnormality",
            "old_mi_icd",
            "mi_or_revascularization_history",
        ),
        "Chronische Infarkt- und Wandbewegungsreferenzen.",
    ),
    PathologyGroup(
        "right_heart_pe_ph",
        "Rechtsherz / PE / PH",
        (
            "echo_rv_dilation",
            "echo_rv_systolic_dysfunction",
            "elevated_rvsp_or_pasp",
            "echo_pulmonary_hypertension",
            "pulmonary_embolism_icd",
            "pulmonary_hypertension_icd",
            "right_heart_strain_proxy",
        ),
        "Struktur-, Druck- und akute Rechtsbelastungsendpunkte.",
    ),
    PathologyGroup(
        "atrial_af",
        "Vorhöfe / AF-Risiko",
        (
            "echo_lae",
            "echo_rae",
            "new_onset_af_30d",
            "new_onset_af_90d",
            "new_onset_af_1y",
        ),
        "Vorhofgröße und zensierte zukünftige AF-Ereignisse.",
    ),
    PathologyGroup(
        "repolarization_electrolyte_drugs",
        "Repolarisation / Elektrolyte / Medikamente",
        (
            "qtc_gt_480",
            "qtc_gt_500",
            "hypokalemia_near_ecg",
            "severe_hypokalemia_near_ecg",
            "hyperkalemia_near_ecg",
            "severe_hyperkalemia_near_ecg",
            "hypomagnesemia_near_ecg",
            "hypocalcemia_near_ecg",
            "qt_prolonging_drug_exposure",
            "antiarrhythmic_exposure_near_ecg",
            "digoxin_exposure_near_ecg",
        ),
        "Zeitnahe Labor- und Expositionsendpunkte; QTc bleibt ECG-basiert.",
    ),
    PathologyGroup(
        "valvular_cardiomyopathy",
        "Klappen / Kardiomyopathien",
        (
            "aortic_stenosis",
            "aortic_regurgitation",
            "mitral_stenosis",
            "mitral_regurgitation",
            "tricuspid_regurgitation",
            "significant_valvular_disease",
            "aortic_stenosis_severe",
            "aortic_regurgitation_severe",
            "mitral_stenosis_severe",
            "mitral_regurgitation_severe",
            "tricuspid_regurgitation_severe",
            "significant_valvular_disease_severe",
            "cardiomyopathy_icd",
        ),
        "Echo-Schweregrade und Kardiomyopathiekontext.",
    ),
    PathologyGroup(
        "conduction_progression",
        "Leitungsprogression",
        ("conduction_progression_1y",),
        (
            "Neuer höhergradiger AV-Block, neuer LSB oder erster "
            "Schrittmacher binnen eines Jahres; prädiktiv, nach dem "
            "Muster des neuen Vorhofflimmerns."
        ),
    ),
    PathologyGroup(
        "arrhythmia_outcomes",
        "Ventrikuläre Ereignisse / Mortalität",
        (
            "vt_vf_icd_or_event",
            "cardiac_arrest",
            "in_hospital_mortality",
            "mortality_30d",
        ),
        "Sekundäre prognostische Endpunkte, nicht diagnostische Hauptziele.",
    ),
    PathologyGroup(
        "inflammatory_infectious",
        "Entzündlich / infektiös",
        (
            "pericarditis_icd",
            "myocarditis_icd",
            "endocarditis_high_confidence_proxy",
        ),
        "Explorative kleine Gruppen mit getrennten Evidenzstufen.",
    ),
)


def pathology_group_for_endpoint(endpoint_id: object) -> PathologyGroup:
    normalized = str(endpoint_id or "")
    for group in PATHOLOGY_GROUPS:
        if normalized in group.endpoint_ids:
            return group
    return PathologyGroup(
        "other",
        "Weitere Endpunkte",
        (normalized,) if normalized else (),
        "Nicht separat gruppierter klinischer Referenzendpunkt.",
    )


def annotate_pathology_groups(records: Iterable[dict]) -> list[dict]:
    output: list[dict] = []
    for record in records:
        item = dict(record)
        group = pathology_group_for_endpoint(item.get("endpoint_id"))
        item["pathology_group_id"] = group.group_id
        item["pathology_group"] = group.display_name_de
        item["pathology_group_navigation_only"] = group.navigation_only
        output.append(item)
    return output


__all__ = [
    "PATHOLOGY_GROUPS",
    "PathologyGroup",
    "annotate_pathology_groups",
    "pathology_group_for_endpoint",
]
