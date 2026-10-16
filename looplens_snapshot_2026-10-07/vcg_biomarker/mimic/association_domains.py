from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import MimicProjectConfig
from .discovery_statistics import add_effect_replication_status
from .frame_checkpoints import (
    checkpointed_frame_chunk_path,
    materialize_frame_chunks,
)
from .note_labels import (
    NOTE_DEPENDENT_DOMAIN_PREFIX,
    read_note_coverage,
)
from .statistics import apply_fdr_by_family, inference_status
from .prespecified_controls import (
    EFFECT_RHO,
    control_summary,
    evaluate_negative_controls,
    evaluate_positive_controls,
)
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    feature_parquet_clause,
    parquet_columns,
)


CONTEXT_ASSOCIATION_VERSION = "1.15.0"  # audit 5.2.2 continuous echo targets
CONTEXT_INTERVAL_CHUNK_SIZE = 16
CONTEXT_FEATURE_BLOCK_SIZE = 32


@dataclass(frozen=True)
class ContextAssociationDefinition:
    association_id: str
    display_name: str
    domain: str
    target_kind: str
    value_sql: str
    eligibility_sql: str
    temporal_role: str
    index_policy: str
    distance_sql: str = "NULL::DOUBLE"
    unit: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)



# Lab groups added in the second wave. Every itemid behind them was resolved
# against this project's own d_labitems table by exact label match. They cover
# the electrolyte, metabolic, inflammatory, thyroid and coagulation axes that
# plausibly move the ECG and were previously not screened at all.
_SECOND_WAVE_LAB_ASSOCIATIONS: tuple[tuple[str, str, str], ...] = (
    ("sodium", "Natrium nahe ECG", "mmol/l"),
    ("bicarbonate", "Bicarbonat nahe ECG", "mmol/l"),
    ("phosphate", "Phosphat nahe ECG", "mg/dl"),
    ("glucose", "Glukose nahe ECG", "mg/dl"),
    ("hba1c", "HbA1c nahe ECG", "%"),
    ("urea_nitrogen", "Harnstoff-Stickstoff nahe ECG", "mg/dl"),
    ("c_reactive_protein", "CRP nahe ECG", "mg/l"),
    ("white_blood_cells", "Leukozyten nahe ECG", "K/uL"),
    ("tsh", "TSH nahe ECG", "uIU/ml"),
    ("free_t4", "Freies T4 nahe ECG", "ng/dl"),
    ("inr", "INR nahe ECG", "ratio"),
    ("prothrombin_time", "Prothrombinzeit nahe ECG", "s"),
    ("partial_thromboplastin_time", "PTT nahe ECG", "s"),
    ("d_dimer", "D-Dimer nahe ECG", "ng/ml"),
    ("albumin", "Albumin nahe ECG", "g/dl"),
)


def _second_wave_lab_associations() -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        (
            f"lab_{group}",
            display,
            f"{group}_nearest",
            f"{group}_count_6h > 0",
            f"ABS({group}_delta_hours)",
            unit,
        )
        for group, display, unit in _SECOND_WAVE_LAB_ASSOCIATIONS
    )


def default_context_association_definitions(
) -> tuple[ContextAssociationDefinition, ...]:
    continuous = (
        (
            "lab_troponin_url_ratio",
            "Troponin / zeilenspezifische URL",
            "troponin_max_url_ratio",
            "troponin_count_standard > 0",
            "ABS(troponin_max_url_delta_hours)",
            "ratio",
        ),
        (
            "lab_ntprobnp_url_ratio",
            "NT-proBNP / zeilenspezifische URL",
            "ntprobnp_max_url_ratio",
            "ntprobnp_max_url_ratio IS NOT NULL",
            "ABS(ntprobnp_max_url_delta_hours)",
            "ratio",
        ),
        (
            "lab_potassium",
            "Kalium nahe ECG",
            "potassium_nearest",
            "potassium_count_6h > 0",
            "ABS(potassium_delta_hours)",
            "mmol/l",
        ),
        (
            "lab_magnesium",
            "Magnesium nahe ECG",
            "magnesium_nearest",
            "magnesium_count_6h > 0",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_total_calcium",
            "Gesamtcalcium nahe ECG",
            "total_calcium_nearest",
            "total_calcium_count_6h > 0",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_ionized_calcium",
            "Ionisiertes Calcium nahe ECG",
            "ionized_calcium_nearest",
            "ionized_calcium_count_6h > 0",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_creatinine",
            "Kreatinin nahe ECG",
            "creatinine_nearest",
            "creatinine_nearest IS NOT NULL",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_hemoglobin",
            "Hämoglobin nahe ECG",
            "hemoglobin_nearest",
            "hemoglobin_nearest IS NOT NULL",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_lactate",
            "Laktat nahe ECG",
            "lactate_nearest",
            "lactate_nearest IS NOT NULL",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_ck_mb",
            "CK-MB nahe ECG",
            "ck_mb_nearest",
            "ck_mb_count_6h > 0",
            "NULL::DOUBLE",
            "source-normalized",
        ),
        (
            "lab_blood_gas_ph",
            "Blutgas-pH nahe ECG",
            "blood_gas_ph_nearest",
            "blood_gas_ph_count_6h > 0",
            "NULL::DOUBLE",
            "source-normalized",
        ),
    ) + _second_wave_lab_associations()
    definitions = [
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="laboratory",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=eligibility,
            distance_sql=distance,
            temporal_role="concurrent_clinical_measurement",
            index_policy=(
                "nearest assessable measurement ECG per subject; "
                "ties by ECG time and study_id"
            ),
            unit=unit,
            notes=(
                "Continuous association only. It is not converted into a "
                "diagnosis or optimized threshold."
            ),
        )
        for identifier, display, column, eligibility, distance, unit in continuous
    ]

    # Audit 5.2.2: the echocardiographic quantities as continuous targets.
    # The endpoints cut them at guideline thresholds; below the thresholds
    # their information was unused. Same TTE and window as the echo labels
    # (nearest transthoracic study, at most 30 days from the ECG).
    echo_continuous = (
        ("echo_lvef", "LVEF (Echo, bevorzugte Methode)", "preferred_lvef", "%"),
        ("echo_la_volume_index", "LA-Volumenindex (Echo)", "la_volume_index", "ml/m^2"),
        ("echo_e_e_prime_ratio", "E/e' (Echo)", "mitral_e_e_prime_ratio", "ratio"),
        ("echo_tapse", "TAPSE (Echo)", "tapse_cm", "cm"),
        ("echo_rvsp_pasp", "RVSP/PASP (Echo)", "derived_rvsp_or_pasp_mmhg", "mmHg"),
        (
            "echo_lv_mass_index_devereux",
            "LV-Masseindex nach Devereux (Echo)",
            "lv_mass_index_devereux_g_m2",
            "g/m^2",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="echocardiography_continuous",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=(
                f"COALESCE(has_tte_30d, FALSE) AND {column} IS NOT NULL"
            ),
            distance_sql="ABS(nearest_tte_delta_hours)",
            temporal_role="concurrent_clinical_measurement",
            index_policy=(
                "first ECG per subject with a transthoracic echo within 30 "
                "days; ties by ECG time and study_id"
            ),
            unit=unit,
            notes=(
                "Continuous rank association beside the thresholded echo "
                "endpoints; uses the nearest TTE within 30 days, the window "
                "of the echo labels."
            ),
        )
        for identifier, display, column, unit in echo_continuous
    )

    # Audit 5.2.9: invasive haemodynamics as surrogate references, from the
    # ICU chart value nearest to the ECG within 6 hours.
    hemodynamic_continuous = (
        ("cvp", "ZVD (invasiv, ICU)", "mmHg"),
        ("pap_systolic", "PA-Druck systolisch (invasiv, ICU)", "mmHg"),
        ("pap_diastolic", "PA-Druck diastolisch (invasiv, ICU)", "mmHg"),
        ("pap_mean", "PA-Mitteldruck (invasiv, ICU)", "mmHg"),
        ("pcwp", "Wedge-Druck (invasiv, ICU)", "mmHg"),
        ("cardiac_output", "Herzzeitvolumen (invasiv, ICU)", "L/min"),
        ("svo2", "Gemischtvenöse Sättigung (invasiv, ICU)", "%"),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=f"icu_hemodynamics_{name}",
            display_name=display,
            domain="icu_invasive_hemodynamics",
            target_kind="continuous",
            value_sql=f"TRY_CAST(icu_{name}_nearest AS DOUBLE)",
            eligibility_sql=f"icu_{name}_nearest IS NOT NULL",
            distance_sql=f"ABS(icu_{name}_delta_hours)",
            temporal_role="concurrent_clinical_measurement",
            index_policy=(
                "first ECG per subject with the chart value within 6 hours; "
                "ties by ECG time and study_id"
            ),
            unit=unit,
            notes=(
                "Surrogate validation against invasive haemodynamics; only "
                "patients with an arterial/central/PA line, a selected ICU "
                "population."
            ),
        )
        for name, display, unit in hemodynamic_continuous
    )

    demographic_continuous = (
        (
            "demographic_age",
            "Alter bei ECG",
            "age",
            "age IS NOT NULL",
            "years",
        ),
        (
            "demographic_bmi",
            "BMI vor ECG",
            "bmi_before_ecg",
            "bmi_before_ecg IS NOT NULL",
            "kg/m^2",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="demographic",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=eligibility,
            temporal_role="index_demographic_measurement",
            index_policy="first assessable encounter ECG per subject",
            unit=unit,
            notes=(
                "Demographic association target, independent from the "
                "confounder-adjustment use of the same field elsewhere."
            ),
        )
        for identifier, display, column, eligibility, unit in demographic_continuous
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="demographic_sex_male",
            display_name="Geschlecht (männlich)",
            domain="demographic",
            target_kind="binary",
            value_sql=(
                "CASE WHEN sex = 'M' THEN 1.0 "
                "WHEN sex = 'F' THEN 0.0 END"
            ),
            eligibility_sql="sex IN ('M', 'F')",
            temporal_role="index_demographic_measurement",
            index_policy="first assessable encounter ECG per subject",
            unit="binary",
            notes=(
                "Demographic association target, independent from the "
                "confounder-adjustment use of sex elsewhere."
            ),
        )
    )

    vitals_ed_continuous = (
        (
            "vitals_ed_heart_rate",
            "ED-Triage-Herzfrequenz",
            "triage_heart_rate",
            "triage_heart_rate IS NOT NULL",
            "bpm",
        ),
        (
            "vitals_ed_systolic_bp",
            "ED-Triage-systolischer Blutdruck",
            "triage_systolic_bp",
            "triage_systolic_bp IS NOT NULL",
            "mmHg",
        ),
        (
            "vitals_ed_diastolic_bp",
            "ED-Triage-diastolischer Blutdruck",
            "triage_diastolic_bp",
            "triage_diastolic_bp IS NOT NULL",
            "mmHg",
        ),
        (
            "vitals_ed_oxygen_saturation",
            "ED-Triage-Sauerstoffsättigung",
            "triage_oxygen_saturation",
            "triage_oxygen_saturation IS NOT NULL",
            "%",
        ),
        (
            "vitals_ed_temperature",
            "ED-Triage-Temperatur",
            "triage_temperature",
            "triage_temperature IS NOT NULL",
            "°F",
        ),
        (
            "vitals_ed_respiratory_rate",
            "ED-Triage-Atemfrequenz",
            "triage_respiratory_rate",
            "triage_respiratory_rate IS NOT NULL",
            "breaths/min",
        ),
        (
            "vitals_ed_pain_score",
            "ED-Triage-Schmerzscore",
            "triage_pain_score",
            "triage_pain_score IS NOT NULL",
            "0-10",
        ),
        (
            "vitals_ed_acuity",
            "ED-Triage-Akuitätsstufe (ESI)",
            "triage_acuity",
            "triage_acuity IS NOT NULL",
            "ESI 1-5",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="vitals_ed",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=eligibility,
            temporal_role="ed_triage_measurement",
            index_policy=(
                "first ECG within the ED stay carrying this triage "
                "measurement per subject"
            ),
            unit=unit,
            notes=(
                "Only assessable for ECGs matched to an ED stay with a "
                "recorded triage measurement; not a diagnosis."
            ),
        )
        for identifier, display, column, eligibility, unit in vitals_ed_continuous
    )

    medication_names = {
        "qt_prolonging_drug": "QT-verlängernde Medikation",
        "antiarrhythmic_class_i_iii": "Antiarrhythmikum Klasse I/III",
        "beta_blocker": "Betablocker",
        "calcium_channel_blocker": "Calciumkanalblocker",
        "digoxin": "Digoxin",
        "diuretic": "Diuretikum",
        "vasopressor": "Vasopressor",
        "anticoagulant": "Antikoagulans",
        "antiplatelet": "Thrombozytenaggregationshemmer",
        "heparin": "Heparin",
        "nitrate": "Nitrat",
        "electrolyte_replacement": "Elektrolytsubstitution",
        "endocarditis_antibiotic_context": "Endokarditis-Antibiotikakontext",
    }
    definitions.extend(
        ContextAssociationDefinition(
            association_id=f"medication_{name}",
            display_name=display,
            domain="medication",
            target_kind="binary",
            value_sql=f"CASE WHEN med_{name} THEN 1.0 ELSE 0.0 END",
            eligibility_sql="medication_assessed",
            temporal_role="retrospective_exposure_context",
            index_policy=(
                "earliest exposed ECG for exposed subjects; otherwise first "
                "assessed unexposed ECG"
            ),
            unit="binary exposure",
            notes=(
                "Exposure evidence distinguishes administration/order/ED/"
                "reconciliation upstream. This association is not a disease."
            ),
        )
        for name, display in medication_names.items()
    )

    procedure_names = {
        "pci": "PCI im Aufenthalt",
        "coronary_angiography": "Koronarangiographie im Aufenthalt",
        "cabg": "CABG im Aufenthalt",
        "pacemaker_or_icd": "Schrittmacher-/ICD-Prozedur",
        "cardioversion_or_defibrillation": "Kardioversion/Defibrillation",
        "valve_procedure": "Klappenprozedur",
    }
    definitions.extend(
        ContextAssociationDefinition(
            association_id=f"procedure_{name}",
            display_name=display,
            domain="procedure",
            target_kind="binary",
            value_sql=f"CASE WHEN proc_{name} THEN 1.0 ELSE 0.0 END",
            eligibility_sql="hadm_id IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy=(
                "first ECG in a qualifying procedure encounter for exposed "
                "subjects; otherwise first assessable encounter ECG"
            ),
            unit="binary encounter context",
            notes=(
                "ICD procedure dates may be day-precision. No minute-level "
                "temporal claim is made."
            ),
        )
        for name, display in procedure_names.items()
    )
    definitions.extend(
        (
            ContextAssociationDefinition(
                association_id="outcome_in_hospital_mortality",
                display_name="In-hospital mortality",
                domain="outcome",
                target_kind="binary",
                value_sql=(
                    "CASE WHEN hospital_expire_flag = 1 THEN 1.0 ELSE 0.0 END"
                ),
                eligibility_sql="hospital_expire_flag IN (0, 1)",
                temporal_role="prognostic_outcome",
                index_policy=(
                    "first ECG in a fatal encounter for event subjects; "
                    "otherwise first assessable nonfatal encounter ECG"
                ),
                unit="binary outcome",
                notes="Outcome association, not a diagnostic pathology label.",
            ),
            ContextAssociationDefinition(
                association_id="outcome_30d_mortality",
                display_name="30-day mortality",
                domain="outcome",
                target_kind="binary",
                value_sql=(
                    "CASE WHEN death_delta_days BETWEEN 0 AND 30 "
                    "THEN 1.0 ELSE 0.0 END"
                ),
                eligibility_sql=(
                    "death_delta_days BETWEEN 0 AND 30 "
                    "OR death_delta_days > 30 OR (death_delta_days IS NULL "
                    "AND vital_status_followup_hours >= 720)"
                ),
                temporal_role="censored_prognostic_outcome",
                index_policy=(
                    "first eligible ECG per subject after explicit 30-day "
                    "outcome assessability"
                ),
                unit="binary outcome",
                notes=(
                    "Subjects dying or lost before a documented 30-day "
                    "negative assessment are unavailable."
                ),
            ),
            ContextAssociationDefinition(
                association_id="outcome_90d_mortality",
                display_name="90-day mortality",
                domain="outcome",
                target_kind="binary",
                value_sql=(
                    "CASE WHEN death_delta_days BETWEEN 0 AND 90 "
                    "THEN 1.0 ELSE 0.0 END"
                ),
                eligibility_sql=(
                    "death_delta_days BETWEEN 0 AND 90 "
                    "OR death_delta_days > 90 OR (death_delta_days IS NULL "
                    "AND vital_status_followup_hours >= 2160)"
                ),
                temporal_role="censored_prognostic_outcome",
                index_policy=(
                    "first eligible ECG per subject after explicit 90-day "
                    "outcome assessability"
                ),
                unit="binary outcome",
                notes=(
                    "Subjects dying or lost before a documented 90-day "
                    "negative assessment are unavailable."
                ),
            ),
            ContextAssociationDefinition(
                association_id="outcome_1y_mortality",
                display_name="1-year mortality",
                domain="outcome",
                target_kind="binary",
                value_sql=(
                    "CASE WHEN death_delta_days BETWEEN 0 AND 365 "
                    "THEN 1.0 ELSE 0.0 END"
                ),
                eligibility_sql=(
                    "death_delta_days BETWEEN 0 AND 365 "
                    "OR death_delta_days > 365 OR (death_delta_days IS NULL "
                    "AND vital_status_followup_hours >= 8760)"
                ),
                temporal_role="censored_prognostic_outcome",
                index_policy=(
                    "first eligible ECG per subject after explicit 1-year "
                    "outcome assessability"
                ),
                unit="binary outcome",
                notes=(
                    "Subjects dying or lost before a documented 1-year "
                    "negative assessment are unavailable."
                ),
            ),
            ContextAssociationDefinition(
                association_id="outcome_length_of_stay",
                display_name="Verweildauer (Tage)",
                domain="outcome",
                target_kind="continuous",
                value_sql="TRY_CAST(length_of_stay_days AS DOUBLE)",
                eligibility_sql=(
                    "length_of_stay_days IS NOT NULL "
                    "AND length_of_stay_days >= 0"
                ),
                temporal_role="prognostic_outcome",
                index_policy="first ECG in the qualifying admission",
                unit="days",
                notes="Outcome association, not a diagnostic pathology label.",
            ),
            ContextAssociationDefinition(
                association_id="outcome_readmission_30d",
                display_name="30-Tage-Wiederaufnahme",
                domain="outcome",
                target_kind="binary",
                value_sql=(
                    "CASE WHEN readmission_30d_flag THEN 1.0 ELSE 0.0 END"
                ),
                eligibility_sql="hadm_id IS NOT NULL",
                temporal_role="censored_prognostic_outcome",
                index_policy="first ECG in the qualifying admission",
                unit="binary outcome",
                notes=(
                    "Readmission is undercounted for the last observed "
                    "admission of a subject (no later admission to compare "
                    "against, not necessarily true negative)."
                ),
            ),
        )
    )

    insurance_categories = {
        "medicare": "Medicare",
        "medicaid": "Medicaid",
        "private": "Private",
    }
    definitions.extend(
        ContextAssociationDefinition(
            association_id=f"socioeconomic_insurance_{name}",
            display_name=f"Versicherung: {label}",
            domain="socioeconomic",
            target_kind="binary",
            value_sql=f"CASE WHEN insurance = '{label}' THEN 1.0 ELSE 0.0 END",
            eligibility_sql="insurance IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes="Encounter-level insurance category, not a diagnosis.",
        )
        for name, label in insurance_categories.items()
    )
    marital_status_categories = {
        "married": "MARRIED",
        "single": "SINGLE",
        "widowed": "WIDOWED",
        "divorced": "DIVORCED",
    }
    definitions.extend(
        ContextAssociationDefinition(
            association_id=f"socioeconomic_marital_{name}",
            display_name=f"Familienstand: {label.title()}",
            domain="socioeconomic",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN marital_status = '{label}' THEN 1.0 ELSE 0.0 END"
            ),
            eligibility_sql="marital_status IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes="Encounter-level marital status category, not a diagnosis.",
        )
        for name, label in marital_status_categories.items()
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="socioeconomic_non_english_language",
            display_name="Bevorzugte Sprache ungleich Englisch",
            domain="socioeconomic",
            target_kind="binary",
            value_sql=(
                "CASE WHEN language IS NULL THEN NULL "
                "WHEN language = 'English' THEN 0.0 ELSE 1.0 END"
            ),
            eligibility_sql="language IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Language-preference proxy, not a race/ethnicity target. "
                "Race/ethnicity is used only as a subgroup-performance axis "
                "elsewhere, never as an association target."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="care_setting_icu",
            display_name="Intensivaufenthalt während dieser Aufnahme",
            domain="care_setting",
            target_kind="binary",
            value_sql=(
                "CASE WHEN ever_icu_this_admission THEN 1.0 "
                "WHEN NOT ever_icu_this_admission THEN 0.0 END"
            ),
            eligibility_sql="ever_icu_this_admission IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Whether the patient was ever transferred to an ICU/CCU "
                "care unit during the same hospitalization as the index "
                "ECG (transfers.parquet). Encounter-level exposure, not a "
                "diagnosis; acuity of the underlying illness is a likely "
                "confounder, not something this target adjusts for."
            ),
        )
    )
    vitals_ed_trend = (
        (
            "vitals_ed_hr_variability",
            "ED-Herzfrequenz-Variabilität über den Aufenthalt",
            "vitals_ed_hr_stddev",
            "vitals_ed_reading_count >= 2",
            "bpm",
        ),
        (
            "vitals_ed_hr_trend_delta",
            "ED-Herzfrequenz erste zu letzte Messung",
            "vitals_ed_hr_trend_delta",
            "vitals_ed_reading_count >= 2",
            "bpm",
        ),
        (
            "vitals_ed_sbp_variability",
            "ED-systolischer-Blutdruck-Variabilität über den Aufenthalt",
            "vitals_ed_sbp_stddev",
            "vitals_ed_reading_count >= 2",
            "mmHg",
        ),
        (
            "vitals_ed_spo2_min",
            "Niedrigste Sauerstoffsättigung im ED-Aufenthalt",
            "vitals_ed_spo2_min",
            "vitals_ed_reading_count >= 2",
            "%",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="vitals_ed_trend",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=eligibility,
            temporal_role="ed_stay_vitals_trend",
            index_policy=(
                "first ECG matched to an ED stay with at least two "
                "vital-sign readings during that stay per subject"
            ),
            unit=unit,
            notes=(
                "Reflects acute ED-course variability (vitalsign.parquet, "
                "serial readings), not a stable baseline physiological "
                "state -- distinct from the single-snapshot triage vitals "
                "in the vitals_ed domain above."
            ),
        )
        for identifier, display, column, eligibility, unit in vitals_ed_trend
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="infection_positive_culture",
            display_name="Positive Mikrobiologie-Kultur während dieser Aufnahme",
            domain="infection",
            target_kind="binary",
            value_sql=(
                "CASE WHEN any_positive_culture THEN 1.0 "
                "WHEN NOT any_positive_culture THEN 0.0 END"
            ),
            eligibility_sql="microbiology_specimen_count > 0",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Any organism identified in a microbiology specimen "
                "(microbiologyevents.parquet) during this hospitalization. "
                "Eligibility requires a specimen to have actually been "
                "drawn (microbiology_specimen_count > 0) so 'not tested' "
                "is never conflated with 'negative'. Testing itself is "
                "driven by clinical suspicion, a real selection bias -- "
                "this is not a population infection-prevalence estimate."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="care_setting_acute_admission",
            display_name="Nicht-elektive (akute/dringliche) Aufnahme",
            domain="care_setting",
            target_kind="binary",
            value_sql=(
                "CASE WHEN admission_type IS NULL THEN NULL "
                "WHEN admission_type = 'ELECTIVE' THEN 0.0 ELSE 1.0 END"
            ),
            eligibility_sql="admission_type IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Whether this hospitalization's admission_type "
                "(admissions.parquet) is anything other than 'ELECTIVE' "
                "(e.g. emergency, urgent, observation admits). Encounter-"
                "level acuity proxy, not a diagnosis."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="care_setting_transfer_from_facility",
            display_name="Aufnahme durch Verlegung von einer anderen Einrichtung",
            domain="care_setting",
            target_kind="binary",
            value_sql=(
                "CASE WHEN admission_location IS NULL THEN NULL "
                "WHEN admission_location ILIKE '%TRANSFER%' THEN 1.0 "
                "ELSE 0.0 END"
            ),
            eligibility_sql="admission_location IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Whether admission_location (admissions.parquet) indicates "
                "an inter-facility transfer (e.g. from another hospital or "
                "a skilled nursing facility), as opposed to a direct/"
                "self-/physician-referred admission. Case-mix proxy, not a "
                "diagnosis."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="outcome_discharge_not_home",
            display_name="Entlassung nicht nach Hause",
            domain="outcome",
            target_kind="binary",
            value_sql=(
                "CASE WHEN discharge_location IS NULL THEN NULL "
                "WHEN discharge_location = 'HOME' THEN 0.0 ELSE 1.0 END"
            ),
            eligibility_sql=(
                "discharge_location IS NOT NULL "
                "AND NOT hospital_expire_flag"
            ),
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Survivors only (hospital_expire_flag excluded, so this "
                "does not overlap with the mortality outcome targets "
                "above) discharged anywhere other than 'HOME' (e.g. "
                "rehab, skilled nursing, hospice, home health care) -- a "
                "common functional-outcome/frailty proxy, not a diagnosis."
            ),
        )
    )
    vitals_pre_ecg = (
        (
            "vitals_pre_ecg_systolic_bp",
            "Prä-ECG-systolischer Blutdruck (OMR)",
            "systolic_bp_before_ecg",
            "mmHg",
        ),
        (
            "vitals_pre_ecg_diastolic_bp",
            "Prä-ECG-diastolischer Blutdruck (OMR)",
            "diastolic_bp_before_ecg",
            "mmHg",
        ),
        (
            "vitals_pre_ecg_weight",
            "Prä-ECG-Körpergewicht (OMR)",
            "weight_kg_before_ecg",
            "kg",
        ),
        (
            "vitals_pre_ecg_height",
            "Prä-ECG-Körpergröße (OMR)",
            "height_cm_before_ecg",
            "cm",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="vitals_pre_ecg",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=f"{column} IS NOT NULL",
            temporal_role="retrospective_outpatient_baseline",
            index_policy=(
                "nearest OMR reading within 365 days before the index ECG "
                "per subject"
            ),
            unit=unit,
            notes=(
                "Outpatient Online Medical Record (omr.parquet) baseline, "
                "not a same-encounter vital sign -- distinct from the ED "
                "triage vitals in the vitals_ed domain and from the "
                "derived bmi_before_ecg already used in the demographic "
                "domain."
            ),
        )
        for identifier, display, column, unit in vitals_pre_ecg
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="ed_arrival_by_ambulance",
            display_name="ED-Ankunft per Rettungswagen",
            domain="ed_context",
            target_kind="binary",
            value_sql=(
                "CASE WHEN arrival_transport IS NULL THEN NULL "
                "WHEN arrival_transport = 'AMBULANCE' THEN 1.0 "
                "ELSE 0.0 END"
            ),
            eligibility_sql="arrival_transport IS NOT NULL",
            temporal_role="ed_stay_context",
            index_policy=(
                "first ECG matched to an ED stay per subject"
            ),
            unit="binary encounter context",
            notes=(
                "arrival_transport (edstays.parquet) -- a standard ED "
                "acuity proxy, not a diagnosis."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="ed_disposition_admitted",
            display_name="Stationäre Aufnahme aus dem ED",
            domain="ed_context",
            target_kind="binary",
            value_sql=(
                "CASE WHEN ed_disposition IS NULL THEN NULL "
                "WHEN ed_disposition = 'ADMITTED' THEN 1.0 ELSE 0.0 END"
            ),
            eligibility_sql="ed_disposition IS NOT NULL",
            temporal_role="ed_stay_context",
            index_policy=(
                "first ECG matched to an ED stay per subject"
            ),
            unit="binary encounter context",
            notes=(
                "disposition (edstays.parquet) -- whether the ED stay "
                "ended in hospital admission rather than discharge home/"
                "elopement/other. Encounter-level outcome, not a "
                "diagnosis."
            ),
        )
    )
    icu_vitals_trend = (
        (
            "icu_vitals_hr_variability",
            "ICU-Herzfrequenz-Variabilität, erste 24h",
            "icu_vitals_hr_stddev",
            "icu_vitals_reading_count >= 2",
            "bpm",
        ),
        (
            "icu_vitals_hr_trend_delta",
            "ICU-Herzfrequenz erste zu letzte Messung, erste 24h",
            "icu_vitals_hr_trend_delta",
            "icu_vitals_reading_count >= 2",
            "bpm",
        ),
        (
            "icu_vitals_sbp_variability",
            "ICU-systolischer-Blutdruck-Variabilität, erste 24h",
            "icu_vitals_sbp_stddev",
            "icu_vitals_reading_count >= 2",
            "mmHg",
        ),
        (
            "icu_vitals_spo2_min",
            "Niedrigste Sauerstoffsättigung in der ICU, erste 24h",
            "icu_vitals_spo2_min",
            "icu_vitals_reading_count >= 2",
            "%",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="icu_vitals_trend",
            target_kind="continuous",
            value_sql=f"TRY_CAST({column} AS DOUBLE)",
            eligibility_sql=eligibility,
            temporal_role="icu_stay_vitals_trend",
            index_policy=(
                "first ECG matched to an ICU stay with at least two "
                "vital-sign readings in the first 24h of that stay per "
                "subject"
            ),
            unit=unit,
            notes=(
                "Reflects the first 24h of ICU-course variability "
                "(chartevents.parquet, curated itemid subset), a more "
                "acute/monitored physiological-state window than the "
                "ed_vitals_trend domain's whole-ED-stay variability."
            ),
        )
        for identifier, display, column, eligibility, unit in icu_vitals_trend
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_ever_ventilated",
            display_name="Maschinelle Beatmung während des ICU-Aufenthalts",
            domain="icu_context",
            target_kind="binary",
            value_sql=(
                "CASE WHEN icu_ever_ventilated THEN 1.0 "
                "WHEN NOT icu_ever_ventilated THEN 0.0 END"
            ),
            eligibility_sql="icustay_id IS NOT NULL",
            temporal_role="icu_stay_context",
            index_policy="first ECG matched to an ICU stay per subject",
            unit="binary encounter context",
            notes=(
                "Derived from procedureevents.parquet intubation/"
                "extubation events for that ICU stay. Encounter-level "
                "exposure, not a diagnosis."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_ventilation_duration_hours",
            display_name="Beatmungsdauer (Erstintubation bis letzte Extubation)",
            domain="icu_context",
            target_kind="continuous",
            value_sql="TRY_CAST(icu_ventilation_duration_hours AS DOUBLE)",
            eligibility_sql="icu_ever_ventilated",
            temporal_role="icu_stay_context",
            index_policy="first ECG matched to an ICU stay per subject",
            unit="hours",
            notes=(
                "Envelope from first intubation to last extubation event "
                "in the stay -- an approximation for stays with multiple "
                "ventilation episodes, not a sum of individual episode "
                "durations. NULL if never extubated in the data."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_ever_vasopressor",
            display_name="Vasopressor-Gabe während des ICU-Aufenthalts",
            domain="icu_context",
            target_kind="binary",
            value_sql=(
                "CASE WHEN icu_ever_vasopressor THEN 1.0 "
                "WHEN NOT icu_ever_vasopressor THEN 0.0 END"
            ),
            eligibility_sql="icustay_id IS NOT NULL",
            temporal_role="icu_stay_context",
            index_policy="first ECG matched to an ICU stay per subject",
            unit="binary encounter context",
            notes=(
                "Derived from inputevents.parquet itemids for the same "
                "5 vasopressors as MEDICATION_PATTERNS['vasopressor'] -- "
                "sharpens that existing name-string flag with itemid-"
                "based precision. Encounter-level exposure, not a "
                "diagnosis."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_vasopressor_duration_hours",
            display_name="Vasopressor-Dauer (Hüllenzeitraum)",
            domain="icu_context",
            target_kind="continuous",
            value_sql="TRY_CAST(icu_vasopressor_duration_hours AS DOUBLE)",
            eligibility_sql="icu_ever_vasopressor",
            temporal_role="icu_stay_context",
            index_policy="first ECG matched to an ICU stay per subject",
            unit="hours",
            notes=(
                "MIN(starttime) to MAX(endtime) envelope across all "
                "vasopressor infusion rows in the stay -- an "
                "approximation, not a gap-free sum of individual "
                "infusion intervals."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_fluid_balance_24h_ml",
            display_name="Netto-Flüssigkeitsbilanz, erste 24h des ICU-Aufenthalts",
            domain="icu_context",
            target_kind="continuous",
            value_sql="TRY_CAST(icu_fluid_balance_24h_ml AS DOUBLE)",
            eligibility_sql="icu_fluid_event_count > 0",
            temporal_role="icu_stay_context",
            index_policy="first ECG matched to an ICU stay per subject",
            unit="mL",
            notes=(
                "SUM(inputevents.amount, mL) minus SUM(outputevents.value) "
                "over the first 24h of the ICU stay. Eligibility requires "
                "at least one recorded input or output event, so 'no "
                "documented fluid events' is never conflated with a "
                "genuine zero net balance."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="icu_gcs_admission_total",
            display_name="Glasgow Coma Scale bei ICU-Aufnahme",
            domain="icu_context",
            target_kind="continuous",
            value_sql="TRY_CAST(icu_gcs_admission_total AS DOUBLE)",
            eligibility_sql="icu_gcs_component_count = 3",
            temporal_role="icu_stay_context",
            index_policy=(
                "nearest chartevents GCS component readings within "
                "-6h/+24h of ICU admission per subject"
            ),
            unit="points (3-15)",
            notes=(
                "Sum of the nearest eye/verbal/motor component readings "
                "to ICU intime. Eligibility requires all three "
                "components to be present, so a partial sum is never "
                "reported as a full GCS score."
            ),
        )
    )
    note_radiology_findings = (
        (
            "note_cardiomegaly",
            "Kardiomegalie in Röntgen-Thorax-Notiz erwähnt",
            "cardiomegaly",
        ),
        (
            "note_pulmonary_edema",
            "Lungenödem/-stauung in Röntgen-Thorax-Notiz erwähnt",
            "pulmonary_edema",
        ),
        (
            "note_pleural_effusion",
            "Pleuraerguss in Röntgen-Thorax-Notiz erwähnt",
            "pleural_effusion",
        ),
        (
            "note_pneumothorax",
            "Pneumothorax in Röntgen-Thorax-Notiz erwähnt",
            "pneumothorax",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_radiology_findings",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention in the IMPRESSION section of a chest "
                "X-ray radiology report during this admission "
                "(radiology.parquet, exam_name ILIKE '%CHEST%'; falls back "
                "to FINDINGS when no separate IMPRESSION is present), "
                "parsed with negation/uncertainty handling (note_parser.py, "
                "curated against real MIMIC-IV-Notes samples). Eligibility "
                "requires at least one chest X-ray report to exist for "
                "this admission, so 'no chest X-ray performed' is never "
                "conflated with 'confirmed absent'. Radiologist narrative "
                "language, not a structured diagnosis, and not a "
                "substitute for the existing echo-derived measurements."
            ),
        )
        for identifier, display, label in note_radiology_findings
    )
    note_social_history_risk_factors = (
        (
            "note_smoking_history",
            "Raucheranamnese in Entlassungsbrief erwähnt",
            "smoking_history",
        ),
        (
            "note_alcohol_use",
            "Alkoholkonsum in Entlassungsbrief erwähnt",
            "alcohol_use",
        ),
        (
            "note_illicit_drug_use",
            "Illegaler Substanzkonsum in Entlassungsbrief erwähnt",
            "illicit_drug_use",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_risk_factors",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention in the Social History section of the "
                "discharge summary (discharge.parquet), parsed with "
                "negation/uncertainty handling. Eligibility requires a "
                "discharge summary to exist for this admission. Important "
                "caveat found during calibration against real MIMIC-IV-"
                "Notes data: Social History sections are frequently "
                "de-identification-redacted to a bare '___' placeholder "
                "(the large majority of a real sample) -- 'not mentioned' "
                "therefore often reflects redaction, not a confirmed "
                "absence of the risk factor, and positive rates in this "
                "domain substantially undercount true prevalence."
            ),
        )
        for identifier, display, label in note_social_history_risk_factors
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="note_family_history_cardiac",
            display_name="Kardiale Familienanamnese in Entlassungsbrief erwähnt",
            domain="note_risk_factors",
            target_kind="binary",
            value_sql=(
                "CASE WHEN note_family_history_cardiac_state = 'positive' "
                "THEN 1.0 "
                "WHEN note_family_history_cardiac_state = 'negative' "
                "THEN 0.0 END"
            ),
            eligibility_sql="note_family_history_cardiac_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention of heart disease, myocardial infarction, "
                "sudden cardiac death, cardiomyopathy, or coronary artery "
                "disease in the Family History section of the discharge "
                "summary (discharge.parquet), parsed with negation/"
                "uncertainty handling. Eligibility requires a discharge "
                "summary to exist for this admission. Family History "
                "sections are far less frequently redacted than Social "
                "History in this dataset, but this is still narrative "
                "text, not a structured pedigree."
            ),
        )
    )
    definitions.append(
        ContextAssociationDefinition(
            association_id="note_medication_nonadherence_mentioned",
            display_name="Medikamenten-Non-Adhärenz in Entlassungsbrief erwähnt",
            domain="note_risk_factors",
            target_kind="binary",
            value_sql=(
                "CASE WHEN note_medication_nonadherence_mentioned_state "
                "= 'positive' THEN 1.0 "
                "WHEN note_medication_nonadherence_mentioned_state "
                "= 'negative' THEN 0.0 END"
            ),
            eligibility_sql=(
                "note_medication_nonadherence_mentioned_state IS NOT NULL"
            ),
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention of medication non-adherence (missed "
                "doses, self-discontinuation, non-compliance) in the "
                "History of Present Illness, Brief Hospital Course, or "
                "Social History sections of the discharge summary "
                "(discharge.parquet), parsed with negation/uncertainty "
                "handling; a structured 'Stopped Medications:' list "
                "(clinician-directed discontinuation) is explicitly "
                "excluded from the pattern to avoid conflating it with "
                "patient non-adherence. Eligibility requires a discharge "
                "summary to exist. Unlike the Social-History-sourced "
                "smoking/alcohol/drug-use items above, History of Present "
                "Illness and Brief Hospital Course are not "
                "de-identification-redacted in this dataset, so this "
                "rate is not artificially suppressed the same way."
            ),
        )
    )
    note_diagnosis_mentions = (
        (
            "note_heart_failure_mentioned",
            "Herzinsuffizienz in Entlassungsbrief erwähnt",
            "heart_failure_mentioned",
        ),
        (
            "note_arrhythmia_mentioned",
            "Arrhythmie in Entlassungsbrief erwähnt",
            "arrhythmia_mentioned",
        ),
        (
            "note_mi_or_acs_mentioned",
            "Myokardinfarkt/akutes Koronarsyndrom in Entlassungsbrief erwähnt",
            "mi_or_acs_mentioned",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_diagnosis_mentions",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention in the Discharge Diagnosis and/or Brief "
                "Hospital Course sections of the discharge summary "
                "(discharge.parquet), parsed with negation/uncertainty "
                "handling. Eligibility requires a discharge summary to "
                "exist for this admission. This is clinician narrative "
                "language, not a coded ICD diagnosis -- see the existing "
                "dx_* diagnosis-flag domains for the coded equivalent -- "
                "and can both miss atypical phrasing and, more rarely, "
                "misclassify a historical or differential mention the "
                "negation heuristic did not catch."
            ),
        )
        for identifier, display, label in note_diagnosis_mentions
    )
    note_head_imaging_findings = (
        (
            "note_acute_stroke_or_infarct",
            "Akuter Schlaganfall/Infarkt in CT/MR-Kopf-Befund erwähnt",
            "acute_stroke_or_infarct",
        ),
        (
            "note_intracranial_hemorrhage",
            "Intrakranielle Blutung in CT/MR-Kopf-Befund erwähnt",
            "intracranial_hemorrhage",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_head_imaging_findings",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention in the IMPRESSION section of a CT or "
                "MR head report during this admission (radiology.parquet, "
                "exam_name ILIKE '%CT HEAD%' OR '%MR HEAD%'; falls back "
                "to FINDINGS when no separate IMPRESSION is present), "
                "parsed with negation/uncertainty handling. Eligibility "
                "requires at least one CT/MR head report to exist for "
                "this admission, so 'no head imaging performed' is never "
                "conflated with 'confirmed absent'. Included for its "
                "direct cardiology relevance (cardioembolic stroke risk "
                "in atrial fibrillation) -- radiologist narrative "
                "language, not a structured diagnosis."
            ),
        )
        for identifier, display, label in note_head_imaging_findings
    )
    note_cta_chest_findings = (
        (
            "note_pulmonary_embolism",
            "Lungenembolie in CTA-Thorax-Befund erwähnt",
            "pulmonary_embolism",
        ),
        (
            "note_aortic_dissection",
            "Aortendissektion/-syndrom in CTA-Thorax-Befund erwähnt",
            "aortic_dissection",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_cta_chest_findings",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Free-text mention in the IMPRESSION section of a CTA "
                "chest report during this admission (radiology.parquet, "
                "exam_name ILIKE '%CTA CHEST%'; falls back to FINDINGS "
                "when no separate IMPRESSION is present), parsed with "
                "negation/uncertainty handling. Eligibility requires at "
                "least one CTA chest report to exist for this admission. "
                "Both targets are classic acute-chest-pain differentials "
                "alongside ACS -- radiologist narrative language, not a "
                "structured diagnosis, and distinct from the plain-film "
                "chest X-ray findings above (a separate, non-overlapping "
                "report selection)."
            ),
        )
        for identifier, display, label in note_cta_chest_findings
    )
    note_functional_status = (
        (
            "note_functional_status_impaired",
            "Eingeschränkter Mobilitätsstatus bei Entlassung",
            "functional_status_impaired",
        ),
        (
            "note_discharge_mental_status_impaired",
            "Eingeschränkter mentaler Status bei Entlassung",
            "discharge_mental_status_impaired",
        ),
    )
    definitions.extend(
        ContextAssociationDefinition(
            association_id=identifier,
            display_name=display,
            domain="note_functional_status",
            target_kind="binary",
            value_sql=(
                f"CASE WHEN note_{label}_state = 'positive' THEN 1.0 "
                f"WHEN note_{label}_state = 'negative' THEN 0.0 END"
            ),
            eligibility_sql=f"note_{label}_state IS NOT NULL",
            temporal_role="retrospective_encounter_context",
            index_policy="first assessable encounter ECG per subject",
            unit="binary encounter context",
            notes=(
                "Extracted from the templated 'Activity Status:'/'Mental "
                "Status:' sub-fields nested inside the Discharge Condition "
                "section of the discharge summary (discharge.parquet) via "
                "a dedicated keyword classifier (not the general negation/"
                "uncertainty heuristic -- these are short fill-in-the-"
                "blank fields, not prose). Eligibility requires the "
                "sub-field itself to be present, which real-data "
                "calibration found in only ~76% of discharge summaries -- "
                "the remaining ~24% have a Discharge Condition section "
                "with a different or absent format and are excluded "
                "rather than guessed at."
            ),
        )
        for identifier, display, label in note_functional_status
    )
    identifiers = [item.association_id for item in definitions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate context association definition")
    return tuple(definitions)


def context_association_registry_hash() -> str:
    return content_hash(
        [item.to_dict() for item in default_context_association_definitions()]
    )


def run_context_association_discovery(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    output_root: Path,
    train_fdr_alpha: float,
    validation_fdr_alpha: float,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    checkpoint_root: Optional[Path] = None,
    checkpoint_input_hash: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Path]:
    context_path = config.evidence_root / "clinical_context_features.parquet"
    if not context_path.is_file():
        raise FileNotFoundError(context_path)
    definitions = default_context_association_definitions()
    definition_path = output_root / "association_domain_definitions.parquet"
    index_path = output_root / "association_domain_index_ecg.parquet"
    coverage_path = output_root / "association_domain_source_coverage.parquet"
    definitions, coverage_frame = _scope_definitions_to_available_sources(
        config,
        definitions,
    )
    atomic_write_parquet(coverage_frame, coverage_path)
    atomic_write_parquet(
        pd.DataFrame([item.to_dict() for item in definitions]),
        definition_path,
    )
    if progress:
        skipped = coverage_frame.loc[
            ~coverage_frame["source_available"].astype(bool),
            "association_id",
        ]
        if len(skipped):
            progress(
                f"Assoziationsdomänen: {len(skipped):,} Definitionen ohne "
                "verfügbare Quelle bleiben ungeprüft (nicht 'kein Befund')"
            )
    _build_association_index(
        config,
        feature_glob=feature_glob,
        context_path=context_path,
        definitions=definitions,
        output_path=index_path,
    )
    if progress:
        progress(
            "Assoziationsdomänen: Labor, Medikamente, Prozeduren und Outcomes "
            "im Train-Split screenen"
        )
    train = _checkpointed_association_screen(
        checkpoint_root,
        checkpoint_input_hash=checkpoint_input_hash,
        config=config,
        feature_glob=feature_glob,
        index_path=index_path,
        definitions=definitions,
        feature_ids=feature_ids,
        panel_map=panel_map,
        split="train",
        min_inference=min_inference,
        min_formal=min_formal,
        selected_pairs=None,
        progress=progress,
    )
    if not train.empty:
        train = apply_fdr_by_family(
            train,
            family_columns=("domain", "panel_id"),
        )
        train_hits = train.loc[
            train["q_value"].le(train_fdr_alpha)
            & train["inference_status"].eq("formal_inference")
        ].copy()
    else:
        train_hits = train.copy()
    selected_pairs = {
        (str(row.association_id), str(row.feature_id))
        for row in train_hits.itertuples(index=False)
    }
    if progress:
        progress(
            f"Assoziationsdomänen: {len(selected_pairs):,} Train-Treffer "
            "in Validation replizieren"
        )
    validation = _checkpointed_association_screen(
        checkpoint_root,
        checkpoint_input_hash=checkpoint_input_hash,
        config=config,
        feature_glob=feature_glob,
        index_path=index_path,
        definitions=definitions,
        feature_ids=feature_ids,
        panel_map=panel_map,
        split="validation",
        min_inference=min_inference,
        min_formal=min_formal,
        selected_pairs=selected_pairs,
        progress=progress,
    )
    if not validation.empty:
        train_lookup = train_hits.set_index(
            ["association_id", "feature_id"]
        )
        validation["train_effect"] = [
            float(train_lookup.loc[(association, feature), "effect_size"])
            for association, feature in zip(
                validation["association_id"],
                validation["feature_id"],
            )
        ]
        validation["train_q_value"] = [
            float(train_lookup.loc[(association, feature), "q_value"])
            for association, feature in zip(
                validation["association_id"],
                validation["feature_id"],
            )
        ]
        validation["direction_consistent"] = (
            np.sign(validation["effect_size"])
            == np.sign(validation["train_effect"])
        )
        validation = apply_fdr_by_family(
            validation,
            family_columns=("domain", "panel_id"),
        )
        validation["validation_q_value"] = validation["q_value"]
        validation["validation_effect"] = validation["effect_size"]
        validation = add_effect_replication_status(
            validation,
            validation_fdr_alpha=validation_fdr_alpha,
        )
        interval_rows = _add_validation_effect_intervals(
            validation.loc[validation["replicated"]].copy(),
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            definitions=definitions,
            iterations=bootstrap_iterations,
            random_seed=random_seed,
            checkpoint_root=checkpoint_root,
            checkpoint_input_hash=checkpoint_input_hash,
            progress=progress,
        )
        validation = _overlay_context_intervals(
            validation,
            interval_rows,
        )
    else:
        validation["replicated"] = pd.Series(dtype=bool)
    replicated = validation.loc[
        validation.get("replicated", pd.Series(False, index=validation.index))
    ].copy()

    paths = {
        "definitions": definition_path,
        "index": index_path,
        "train": output_root / "association_domain_train_screening.parquet",
        "validation": (
            output_root / "association_domain_validation_replication.parquet"
        ),
        "replicated": (
            output_root / "association_domain_replicated_hits.parquet"
        ),
    }
    paths["source_coverage"] = coverage_path
    atomic_write_parquet(train, paths["train"])
    atomic_write_parquet(validation, paths["validation"])
    atomic_write_parquet(replicated, paths["replicated"])

    # Run the prespecified controls on the validation half.  They are declared
    # in code before any result is seen, and their verdict decides whether this
    # screen's *negative* findings mean anything: an instrument that cannot
    # recover the calcium-QT or potassium-T-wave relationship is not entitled to
    # report absences.  The verdict is written next to the results rather than
    # left to be worked out later.
    positive_controls = evaluate_positive_controls(
        validation,
        target_column="association_id",
        effect_column="effect_size",
    )
    negative_controls = evaluate_negative_controls(
        validation,
        target_column="association_id",
        effect_column="effect_size",
        effect_kind=EFFECT_RHO,
    )
    paths["positive_controls"] = (
        output_root / "association_domain_positive_controls.parquet"
    )
    paths["negative_controls"] = (
        output_root / "association_domain_negative_controls.parquet"
    )
    paths["control_summary"] = (
        output_root / "association_domain_control_summary.json"
    )
    atomic_write_parquet(positive_controls, paths["positive_controls"])
    atomic_write_parquet(negative_controls, paths["negative_controls"])
    atomic_write_json(
        paths["control_summary"],
        control_summary(positive_controls, negative_controls),
    )
    return paths


def _scope_definitions_to_available_sources(
    config: MimicProjectConfig,
    definitions: tuple[ContextAssociationDefinition, ...],
) -> tuple[tuple[ContextAssociationDefinition, ...], pd.DataFrame]:
    """Drop definitions whose only evidence source was never ingested.

    A note-dependent domain screened against an absent source yields zero
    rows, which downstream is indistinguishable from "screened, nothing
    found". Such domains are therefore excluded from the screen and recorded
    as ``not_assessed`` with the concrete reason, so a report can state that
    they were never examined rather than implying an absence of association.
    """

    coverage = read_note_coverage(config)
    note_available = bool(coverage.get("assessed", False))
    reason = str(coverage.get("status", "unknown"))
    rows: list[dict] = []
    retained: list[ContextAssociationDefinition] = []
    for definition in definitions:
        note_dependent = definition.domain.startswith(
            NOTE_DEPENDENT_DOMAIN_PREFIX
        )
        available = note_available or not note_dependent
        rows.append(
            {
                "association_id": definition.association_id,
                "domain": definition.domain,
                "source_kind": (
                    "mimic_iv_note" if note_dependent else "mimic_iv_structured"
                ),
                "source_available": available,
                "assessment_state": (
                    "assessed" if available else "not_assessed"
                ),
                "unavailable_reason": (
                    "" if available else f"note_coverage_status={reason}"
                ),
            }
        )
        if available:
            retained.append(definition)
    return tuple(retained), pd.DataFrame(rows)


def _overlay_context_intervals(
    validation: pd.DataFrame,
    interval_rows: pd.DataFrame,
) -> pd.DataFrame:
    output = validation.copy()
    output["effect_ci_low"] = np.nan
    output["effect_ci_high"] = np.nan
    output["bootstrap_successful_iterations"] = 0
    output["bootstrap_unit"] = "subject_id"
    output["bootstrap_scope"] = np.where(
        output["replicated"].fillna(False),
        "subject_cluster_replicated_hit",
        "not_requested_not_replicated",
    )
    if interval_rows.empty:
        return output
    keys = ["association_id", "feature_id"]
    update_columns = [
        "effect_ci_low",
        "effect_ci_high",
        "bootstrap_successful_iterations",
        "bootstrap_unit",
    ]
    updates = interval_rows[
        keys + update_columns
    ].drop_duplicates(keys)
    merged = output.merge(
        updates.assign(_bootstrap_matched=True),
        on=keys,
        how="left",
        suffixes=("", "__bootstrap"),
        validate="one_to_one",
    )
    matched = merged["_bootstrap_matched"].eq(True)
    for column in update_columns:
        update_column = f"{column}__bootstrap"
        merged.loc[matched, column] = merged.loc[
            matched,
            update_column,
        ].to_numpy()
        merged = merged.drop(columns=update_column)
    return merged.drop(columns="_bootstrap_matched")


def _checkpointed_association_screen(
    checkpoint_root: Optional[Path],
    *,
    checkpoint_input_hash: Optional[str],
    config: MimicProjectConfig,
    feature_glob: str,
    index_path: Path,
    definitions: tuple[ContextAssociationDefinition, ...],
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    split: str,
    min_inference: int,
    min_formal: int,
    selected_pairs: Optional[set[tuple[str, str]]],
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if checkpoint_root is None or checkpoint_input_hash is None:
        return _screen_association_domains(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            definitions=definitions,
            feature_ids=feature_ids,
            panel_map=panel_map,
            split=split,
            min_inference=min_inference,
            min_formal=min_formal,
            selected_pairs=selected_pairs,
            progress=progress,
        )

    definition_lookup = {
        item.association_id: item for item in definitions
    }
    association_ids = (
        sorted(definition_lookup)
        if selected_pairs is None
        else sorted({association for association, _ in selected_pairs})
    )
    paths: list[Path] = []
    for chunk_index, association_id in enumerate(association_ids):
        selected_features = tuple(
            feature_id
            for feature_id in feature_ids
            if selected_pairs is None
            or (association_id, feature_id) in selected_pairs
        )
        pair_filter = {
            (association_id, feature_id)
            for feature_id in selected_features
        }
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace=f"{split}_screen",
            chunk_index=chunk_index,
            input_hash=checkpoint_input_hash,
            item_keys=(
                association_id,
                *selected_features,
            ),
            compute=lambda pairs=pair_filter: _screen_association_domains(
                config,
                feature_glob=feature_glob,
                index_path=index_path,
                definitions=definitions,
                feature_ids=feature_ids,
                panel_map=panel_map,
                split=split,
                min_inference=min_inference,
                min_formal=min_formal,
                selected_pairs=pairs,
                progress=None,
            ),
        )
        paths.append(path)
        if progress:
            progress(
                f"Assoziationsdomänen {split} "
                f"{chunk_index + 1:,}/{len(association_ids):,}: "
                f"{association_id} "
                f"({'fortgesetzt' if resumed else 'gesichert'})"
            )
    return materialize_frame_chunks(paths)


def _build_association_index(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    context_path: Path,
    definitions: tuple[ContextAssociationDefinition, ...],
    output_path: Path,
) -> None:
    context_sql = _sql_path(context_path)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    read_features = feature_parquet_clause(sql_path=feature_sql)
    branches = []
    for definition in definitions:
        branches.append(
            f"""
            SELECT * EXCLUDE(index_rank)
            FROM (
                SELECT
                    '{_literal(definition.association_id)}' AS association_id,
                    '{_literal(definition.display_name)}' AS association_name,
                    '{_literal(definition.domain)}' AS domain,
                    '{_literal(definition.target_kind)}' AS target_kind,
                    '{_literal(definition.temporal_role)}' AS temporal_role,
                    c.subject_id,
                    c.study_id,
                    c.split,
                    c.ecg_time,
                    TRY_CAST(({definition.value_sql}) AS DOUBLE) AS target_value,
                    TRY_CAST(({definition.distance_sql}) AS DOUBLE)
                        AS target_time_distance,
                    'first_eligible_ecg_target_blind'
                        AS index_selection_strategy,
                    -- Target-blind: the index ECG is the first eligible one
                    -- per subject, ranked without looking at the target at
                    -- all. Ordering by target_value (the previous binary
                    -- branch) picked the first *positive* ECG for exposed
                    -- subjects and the first ECG for everyone else, which is
                    -- outcome-dependent index selection. Ordering by the
                    -- measurement's time distance (the previous continuous
                    -- branch) is value-blind but still selects on
                    -- measurement timing; the acceptable window belongs in
                    -- eligibility_sql, and the realised distance is kept
                    -- below as target_time_distance so pairing quality stays
                    -- auditable instead of being optimised per subject.
                    ROW_NUMBER() OVER (
                        PARTITION BY c.subject_id
                        ORDER BY
                            c.ecg_time,
                            c.study_id
                    ) AS index_rank
                FROM read_parquet('{context_sql}') c
                JOIN available_features f
                  USING (subject_id, study_id, split)
                WHERE c.split IN ('train', 'validation')
                  AND ({definition.eligibility_sql})
                  AND TRY_CAST(({definition.value_sql}) AS DOUBLE) IS NOT NULL
            )
            WHERE index_rank = 1
            """
        )
    query = f"""
        COPY (
            WITH available_features AS (
                SELECT DISTINCT subject_id, study_id, split
                FROM {read_features}
                WHERE split IN ('train', 'validation')
                  AND extraction_status = 'ok'
            )
            {' UNION ALL '.join(branches)}
            ORDER BY association_id, split, subject_id
        ) TO '{_sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    temporary_sql = _sql_path(temporary)
    query = query.replace(_sql_path(output_path), temporary_sql)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        with profile_query(
            f"association-domains:{output_path.name}"
        ) as profile:
            copy_result = connection.execute(query).fetchone()
            if copy_result is None:
                raise RuntimeError("DuckDB COPY returned no row count")
            profile["rows"] = int(copy_result[0])
    atomic_replace(temporary, output_path)


def _screen_association_domains(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    definitions: tuple[ContextAssociationDefinition, ...],
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    split: str,
    min_inference: int,
    min_formal: int,
    selected_pairs: Optional[set[tuple[str, str]]],
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    definition_lookup = {
        item.association_id: item for item in definitions
    }
    selected_associations = {
        association for association, _ in selected_pairs
    } if selected_pairs is not None else set(definition_lookup)
    records: list[dict] = []
    ordered_associations = sorted(selected_associations)
    for association_index, association_id in enumerate(
        ordered_associations,
        start=1,
    ):
        definition = definition_lookup[association_id]
        selected_features = tuple(
            feature_id
            for feature_id in feature_ids
            if selected_pairs is None
            or (association_id, feature_id) in selected_pairs
        )
        if len(selected_features) > CONTEXT_FEATURE_BLOCK_SIZE:
            for start in range(
                0,
                len(selected_features),
                CONTEXT_FEATURE_BLOCK_SIZE,
            ):
                feature_block = selected_features[
                    start : start + CONTEXT_FEATURE_BLOCK_SIZE
                ]
                block = _screen_association_domains(
                    config,
                    feature_glob=feature_glob,
                    index_path=index_path,
                    definitions=definitions,
                    feature_ids=feature_block,
                    panel_map=panel_map,
                    split=split,
                    min_inference=min_inference,
                    min_formal=min_formal,
                    selected_pairs={
                        (association_id, feature_id)
                        for feature_id in feature_block
                    },
                    progress=None,
                )
                if not block.empty:
                    records.extend(block.to_dict(orient="records"))
            if progress:
                progress(
                    f"AssoziationsdomÃ¤nen {split} "
                    f"{association_index:,}/{len(ordered_associations):,}: "
                    f"{association_id}"
                )
            continue
        frame = _load_association_frame(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            association_id=association_id,
            split=split,
            feature_ids=selected_features,
        )
        if frame.empty:
            continue
        target = pd.to_numeric(
            frame["target_value"],
            errors="coerce",
        ).to_numpy(dtype=float)
        subjects = frame["subject_id"].to_numpy()
        # Pairing quality of the target-blind index. Previously the index was
        # ranked to minimise this distance (continuous) or to prefer positive
        # targets (binary); now it is simply observed and reported, so a poor
        # pairing shows up as a wide distance instead of being optimised away
        # per subject.
        distance = pd.to_numeric(
            frame.get(
                "target_time_distance",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).abs().to_numpy(dtype=float)
        distance_median = (
            float(np.nanmedian(distance))
            if np.isfinite(distance).any()
            else float("nan")
        )
        distance_p90 = (
            float(np.nanpercentile(distance[np.isfinite(distance)], 90))
            if np.isfinite(distance).any()
            else float("nan")
        )
        adjustment_rate = pd.to_numeric(
            frame.get(
                "adjustment_heart_rate",
                pd.Series(np.nan, index=frame.index),
            ),
            errors="coerce",
        ).to_numpy(dtype=float)
        for feature_id in selected_features:
            values = pd.to_numeric(
                frame[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(target)
            clean_values = values[valid]
            clean_target = target[valid]
            if definition.target_kind == "binary":
                labels = clean_target.astype(np.int8)
                positive = clean_values[labels == 1]
                negative = np.sort(clean_values[labels == 0])
                raw_auc, p_value = _auc_and_p_from_sorted_controls(
                    positive,
                    negative,
                )
                effect = 2.0 * raw_auc - 1.0
                n_positive = int(np.sum(labels == 1))
                n_negative = int(np.sum(labels == 0))
                status = inference_status(
                    n_positive,
                    n_negative,
                    minimum_inference=min_inference,
                    minimum_primary=min_formal,
                )
                metric_name = "rank_biserial"
                n_total = n_positive + n_negative
            else:
                effect, p_value = _spearman(clean_values, clean_target)
                n_positive = n_negative = 0
                n_total = len(clean_values)
                status = _continuous_inference_status(
                    n_total,
                    minimum_inference=min_inference,
                    minimum_formal=min_formal,
                )
                raw_auc = np.nan
                metric_name = "spearman_rho"
            adjusted_effect, adjusted_n = _partial_spearman(
                values, target, adjustment_rate
            )
            records.append(
                {
                    "association_id": association_id,
                    "association_name": definition.display_name,
                    "domain": definition.domain,
                    "target_kind": definition.target_kind,
                    "temporal_role": definition.temporal_role,
                    "index_policy": definition.index_policy,
                    "panel_id": panel_map[feature_id],
                    "feature_id": feature_id,
                    "evaluation_split": split,
                    "analysis_type": "non_pathology_association_domain",
                    "index_selection_strategy": (
                        "first_eligible_ecg_target_blind"
                    ),
                    "target_pairing_distance_median": distance_median,
                    "target_pairing_distance_p90": distance_p90,
                    "effect_metric": metric_name,
                    "effect_size": effect,
                    "effect_size_rate_adjusted": adjusted_effect,
                    "rate_adjustment_n": adjusted_n,
                    "rate_adjustment_status": (
                        "heart_rate_partial_rank_correlation"
                        if np.isfinite(adjusted_effect)
                        else "not_adjustable_rate_unavailable"
                    ),
                    "rate_adjustment_sign_flip": bool(
                        np.isfinite(effect)
                        and np.isfinite(adjusted_effect)
                        and np.sign(effect) != np.sign(adjusted_effect)
                    ),
                    "raw_auc": raw_auc,
                    "direction": (
                        1 if np.isfinite(effect) and effect >= 0 else -1
                    ),
                    "p_value": p_value,
                    "n_total": n_total,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "missingness": 1.0 - float(np.mean(valid)),
                    "inference_status": status,
                    "subject_count": len(np.unique(subjects[valid])),
                    "locked_test_scanned": False,
                }
            )
        if progress:
            progress(
                f"Assoziationsdomänen {split} "
                f"{association_index:,}/{len(ordered_associations):,}: "
                f"{association_id}"
            )
    return pd.DataFrame(records)


def _load_association_frame(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    association_id: str,
    split: str,
    feature_ids: tuple[str, ...],
) -> pd.DataFrame:
    if not feature_ids:
        return pd.DataFrame()
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    index_sql = _sql_path(index_path)
    columns = ", ".join(
        f"f.{_quote_identifier(item)}" for item in feature_ids
    )
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        available = parquet_columns(connection, feature_sql)
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
        selected = tuple(
            item for item in feature_ids if item in available
        )
        if not selected:
            return pd.DataFrame()
        columns = ", ".join(
            f"f.{_quote_identifier(item)}" for item in selected
        )
        # The heart rate is joined in only when the master index is there.
        # Without it the screen still runs and every row is reported as
        # ``not_adjustable_rate_unavailable`` -- an absent adjustment has to be
        # visible in the result, not an absent stage.
        master_path = config.master_root / "ecg_master_index.parquet"
        if master_path.is_file():
            master_sql = _sql_path(master_path)
            rate_select = "m.machine_heart_rate_bpm AS adjustment_heart_rate,"
            rate_join = (
                f"LEFT JOIN read_parquet('{master_sql}') m "
                "USING (subject_id, study_id)"
            )
        else:
            rate_select = "NULL::DOUBLE AS adjustment_heart_rate,"
            rate_join = ""
        frame = connection.execute(
            f"""
            SELECT
                i.subject_id,
                i.study_id,
                i.split,
                i.target_value,
                i.target_time_distance,
                {rate_select}
                {columns}
            FROM read_parquet('{index_sql}') i
            JOIN {read_features} f USING (subject_id, study_id, split)
            {rate_join}
            WHERE i.association_id = ?
              AND i.split = ?
              AND f.extraction_status = 'ok'
            ORDER BY i.subject_id
            """,
            [association_id, split],
        ).fetchdf()
    for feature_id in feature_ids:
        if feature_id not in frame:
            frame[feature_id] = np.nan
    return frame


def _add_validation_effect_intervals(
    validation: pd.DataFrame,
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    definitions: tuple[ContextAssociationDefinition, ...],
    iterations: int,
    random_seed: int,
    checkpoint_root: Optional[Path] = None,
    checkpoint_input_hash: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    if validation.empty:
        return validation
    if checkpoint_root is not None and checkpoint_input_hash is not None:
        paths: list[Path] = []
        completed = 0
        for chunk_index, start in enumerate(
            range(0, len(validation), CONTEXT_INTERVAL_CHUNK_SIZE)
        ):
            chunk = validation.iloc[
                start:start + CONTEXT_INTERVAL_CHUNK_SIZE
            ].copy()
            keys = tuple(
                (str(row.association_id), str(row.feature_id))
                for row in chunk.itertuples(index=False)
            )
            path, resumed = checkpointed_frame_chunk_path(
                checkpoint_root,
                namespace="validation_intervals",
                chunk_index=chunk_index,
                input_hash=checkpoint_input_hash,
                item_keys=keys,
                compute=lambda selected=chunk, offset=start: (
                    _compute_validation_effect_intervals(
                        selected,
                        config,
                        feature_glob=feature_glob,
                        index_path=index_path,
                        definitions=definitions,
                        iterations=iterations,
                        random_seed=random_seed,
                        row_offset=offset,
                    )
                ),
            )
            paths.append(path)
            completed += len(chunk)
            if progress:
                progress(
                    "Assoziationsdomänen Validation-CI "
                    f"{completed:,}/{len(validation):,}: "
                    f"{'fortgesetzt' if resumed else 'gesichert'}"
                )
        return materialize_frame_chunks(paths)
    return _compute_validation_effect_intervals(
        validation,
        config,
        feature_glob=feature_glob,
        index_path=index_path,
        definitions=definitions,
        iterations=iterations,
        random_seed=random_seed,
        row_offset=0,
    )


def _compute_validation_effect_intervals(
    validation: pd.DataFrame,
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    definitions: tuple[ContextAssociationDefinition, ...],
    iterations: int,
    random_seed: int,
    row_offset: int,
) -> pd.DataFrame:
    definition_lookup = {
        item.association_id: item for item in definitions
    }
    lows: np.ndarray = np.full(len(validation), np.nan, dtype=float)
    highs: np.ndarray = np.full(len(validation), np.nan, dtype=float)
    successes: np.ndarray = np.zeros(len(validation), dtype=int)
    rows = list(validation.itertuples(index=False))
    positions_by_association: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        positions_by_association.setdefault(
            str(row.association_id),
            [],
        ).append(position)
    for association_id, positions in positions_by_association.items():
        for start in range(0, len(positions), CONTEXT_FEATURE_BLOCK_SIZE):
            block_positions = positions[
                start : start + CONTEXT_FEATURE_BLOCK_SIZE
            ]
            selected_features = tuple(
                dict.fromkeys(
                    str(rows[position].feature_id)
                    for position in block_positions
                )
            )
            frame = _load_association_frame(
                config,
                feature_glob=feature_glob,
                index_path=index_path,
                association_id=association_id,
                split="validation",
                feature_ids=selected_features,
            )
            target = pd.to_numeric(
                frame["target_value"],
                errors="coerce",
            ).to_numpy(dtype=float)
            for position in block_positions:
                row = rows[position]
                feature_id = str(row.feature_id)
                values = pd.to_numeric(
                    frame[feature_id],
                    errors="coerce",
                ).to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(target)
                low, high, successful = _bootstrap_effect_interval(
                    values[valid],
                    target[valid],
                    target_kind=definition_lookup[
                        association_id
                    ].target_kind,
                    iterations=(
                        int(iterations)
                        if str(row.inference_status) != "descriptive_only"
                        else 0
                    ),
                    seed=_stable_seed(
                        random_seed,
                        association_id,
                        feature_id,
                    ),
                )
                lows[position] = low
                highs[position] = high
                successes[position] = successful
    result = validation.copy()
    result["effect_ci_low"] = lows.tolist()
    result["effect_ci_high"] = highs.tolist()
    result["bootstrap_successful_iterations"] = successes.tolist()
    result["bootstrap_unit"] = "subject_id"
    return result


def _bootstrap_effect_interval(
    values: np.ndarray,
    target: np.ndarray,
    *,
    target_kind: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, int]:
    if iterations <= 0 or len(values) < 2:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(iterations)):
        index = rng.integers(0, len(values), size=len(values))
        sampled_values = values[index]
        sampled_target = target[index]
        if target_kind == "binary":
            labels = sampled_target.astype(np.int8)
            if np.unique(labels).size < 2:
                continue
            positive = sampled_values[labels == 1]
            negative = np.sort(sampled_values[labels == 0])
            auc, _ = _auc_and_p_from_sorted_controls(positive, negative)
            estimate = 2.0 * auc - 1.0
        else:
            estimate, _ = _spearman(sampled_values, sampled_target)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(estimates),
    )


def _partial_spearman(
    values: np.ndarray,
    target: np.ndarray,
    covariate: np.ndarray,
) -> tuple[float, int]:
    """Rank correlation of marker and target with a covariate removed.

    This exists because of a measured inversion. The curated association axis
    reports unadjusted rank correlations, and the prespecified control for
    "calcium shortens the QT" came back with the opposite sign: rho = +0.128
    against ionised calcium, one of the most reliable relationships in clinical
    electrophysiology, reported backwards.

    Checked on the raw evidence over 25,909 recordings that carry both an
    ionised calcium and a measurable QT:

        QT vs ionised calcium, unadjusted   +0.030
        QT vs heart rate                    -0.667
        heart rate vs ionised calcium       -0.138

    Sicker patients have a lower calcium and a faster heart rate, and the rate
    shortens the QT by more than the low calcium lengthens it.  Inside every
    heart-rate band the sign is correct again -- -0.142, -0.128, -0.085,
    -0.090, -0.101, -0.081, -0.041 from the slowest band to the fastest.

    Most VCG markers are rate-dependent and the rate correlates with nearly
    every laboratory value in an intensive-care cohort, so this is not a
    peculiarity of calcium: any hit on this axis could carry the wrong sign.
    The adjusted value is reported next to the unadjusted one rather than
    replacing it, because the difference between them is itself the finding.
    """

    usable = (
        np.isfinite(values) & np.isfinite(target) & np.isfinite(covariate)
    )
    count = int(usable.sum())
    if count < 10:
        return float("nan"), count
    ranked = np.vstack(
        [
            scipy_stats.rankdata(values[usable]),
            scipy_stats.rankdata(target[usable]),
            scipy_stats.rankdata(covariate[usable]),
        ]
    )
    if np.any(np.ptp(ranked, axis=1) == 0):
        return float("nan"), count
    correlation = np.corrcoef(ranked)
    marker_target = float(correlation[0, 1])
    marker_rate = float(correlation[0, 2])
    target_rate = float(correlation[1, 2])
    denominator = np.sqrt(
        max(0.0, 1.0 - marker_rate**2) * max(0.0, 1.0 - target_rate**2)
    )
    if denominator <= 0:
        return float("nan"), count
    return float((marker_target - marker_rate * target_rate) / denominator), count


def _spearman(values: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    if (
        len(values) < 3
        or np.unique(values).size < 2
        or np.unique(target).size < 2
    ):
        return float("nan"), 1.0
    result = scipy_stats.spearmanr(values, target, nan_policy="omit")
    effect = float(result.statistic)
    p_value = float(result.pvalue)
    return (
        effect if np.isfinite(effect) else float("nan"),
        max(float(np.finfo(float).tiny), min(1.0, p_value))
        if np.isfinite(p_value)
        else 1.0,
    )


def _auc_and_p_from_sorted_controls(
    case_values: np.ndarray,
    sorted_control: np.ndarray,
) -> tuple[float, float]:
    n_case = len(case_values)
    n_control = len(sorted_control)
    if n_case == 0 or n_control == 0:
        return float("nan"), 1.0
    less = np.searchsorted(sorted_control, case_values, side="left")
    less_or_equal = np.searchsorted(
        sorted_control,
        case_values,
        side="right",
    )
    u_value = float(np.sum((less + less_or_equal) / 2.0))
    auc = u_value / (n_case * n_control)
    result = scipy_stats.mannwhitneyu(
        case_values,
        sorted_control,
        alternative="two-sided",
        method="asymptotic",
    )
    p_value = float(result.pvalue)
    return (
        float(auc),
        max(float(np.finfo(float).tiny), min(1.0, p_value))
        if np.isfinite(p_value)
        else 1.0,
    )


def _continuous_inference_status(
    n_total: int,
    *,
    minimum_inference: int,
    minimum_formal: int,
) -> str:
    if n_total < minimum_inference:
        return "descriptive_only"
    if n_total < minimum_formal:
        return "underpowered_descriptive"
    return "formal_inference"


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(item) for item in (seed, *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _literal(value: str) -> str:
    return str(value).replace("'", "''")


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
