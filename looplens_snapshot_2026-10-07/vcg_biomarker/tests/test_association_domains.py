from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.mappings import LAB_ITEM_GROUPS
from vcg_biomarker.mimic.association_domains import (
    default_context_association_definitions,
)


def _synthetic_context_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hadm_id": [101, 102],
            # Audit 5.2.2 / 5.2.9: continuous echo and invasive haemodynamics.
            "has_tte_30d": [True, None],
            "nearest_tte_delta_hours": [-30.0, None],
            "preferred_lvef": [55.0, None],
            "la_volume_index": [38.0, None],
            "mitral_e_e_prime_ratio": [9.5, None],
            "tapse_cm": [2.1, None],
            "derived_rvsp_or_pasp_mmhg": [32.0, None],
            "lv_mass_index_devereux_g_m2": [101.0, None],
            **{
                f"icu_{name}_{suffix}": [value, None]
                for name, value in (
                    ("cvp", 9.0), ("pap_systolic", 35.0), ("pap_diastolic", 14.0),
                    ("pap_mean", 22.0), ("pcwp", 12.0), ("cardiac_output", 4.8),
                    ("svo2", 68.0),
                )
                for suffix, value in (("nearest", value), ("delta_hours", 1.5))
            },
            "age": [55.0, 41.0],
            "bmi_before_ecg": [31.2, None],
            "sex": ["M", "F"],
            "hospital_expire_flag": [0, 1],
            "death_delta_days": [45.0, 0.0],
            "followup_hours": [100.0, 0.0],
            "vital_status_followup_hours": [9000.0, 0.0],
            "length_of_stay_days": [6, 2],
            "readmission_30d_flag": [True, False],
            "insurance": ["Medicare", "Private"],
            "marital_status": ["WIDOWED", "SINGLE"],
            "language": ["Spanish", "English"],
            "triage_heart_rate": [88.0, None],
            "triage_systolic_bp": [130.0, None],
            "triage_diastolic_bp": [80.0, None],
            "triage_oxygen_saturation": [97.0, None],
            "triage_temperature": [98.6, None],
            "triage_respiratory_rate": [18.0, None],
            "triage_pain_score": [4.0, None],
            "triage_acuity": [3.0, None],
            "ever_icu_this_admission": [True, False],
            "primary_service_this_admission": ["MED", "CARD"],
            "admission_type": ["EW EMER.", "ELECTIVE"],
            "admission_location": ["EMERGENCY ROOM", "TRANSFER FROM HOSPITAL"],
            "discharge_location": ["HOME", "REHAB"],
            "arrival_transport": ["AMBULANCE", None],
            "ed_disposition": ["ADMITTED", None],
            "systolic_bp_before_ecg": [128.0, None],
            "diastolic_bp_before_ecg": [76.0, None],
            "weight_kg_before_ecg": [82.5, None],
            "height_cm_before_ecg": [175.0, None],
            "vitals_ed_reading_count": [3, 0],
            "vitals_ed_hr_stddev": [8.5, None],
            "vitals_ed_hr_trend_delta": [12.0, None],
            "vitals_ed_sbp_stddev": [6.0, None],
            "vitals_ed_spo2_min": [91.0, None],
            "microbiology_specimen_count": [1, 0],
            "any_positive_culture": [True, False],
            "icustay_id": [5001, None],
            "icu_vitals_reading_count": [4, 0],
            "icu_vitals_hr_stddev": [7.2, None],
            "icu_vitals_hr_trend_delta": [-5.0, None],
            "icu_vitals_sbp_stddev": [9.1, None],
            "icu_vitals_spo2_min": [92.0, None],
            "icu_ever_ventilated": [True, False],
            "icu_ventilation_duration_hours": [26.5, None],
            "icu_ever_vasopressor": [True, False],
            "icu_vasopressor_duration_hours": [12.0, None],
            "icu_fluid_balance_24h_ml": [350.0, None],
            "icu_fluid_event_count": [6, 0],
            "icu_gcs_admission_total": [15.0, None],
            "icu_gcs_component_count": [3, 0],
            "note_cardiomegaly_state": ["positive", None],
            "note_pulmonary_edema_state": ["negative", None],
            "note_pleural_effusion_state": ["uncertain", None],
            "note_pneumothorax_state": ["not_mentioned", None],
            "note_smoking_history_state": ["positive", None],
            "note_alcohol_use_state": ["not_mentioned", None],
            "note_illicit_drug_use_state": ["negative", None],
            "note_family_history_cardiac_state": ["positive", None],
            "note_heart_failure_mentioned_state": ["positive", None],
            "note_arrhythmia_mentioned_state": ["not_mentioned", None],
            "note_mi_or_acs_mentioned_state": ["negative", None],
            "note_medication_nonadherence_mentioned_state": ["positive", None],
            "note_acute_stroke_or_infarct_state": ["negative", None],
            "note_intracranial_hemorrhage_state": ["uncertain", None],
            "note_pulmonary_embolism_state": ["positive", None],
            "note_aortic_dissection_state": ["not_mentioned", None],
            "note_functional_status_impaired_state": ["positive", None],
            "note_discharge_mental_status_impaired_state": ["negative", None],
            # existing lab/medication/procedure columns referenced by other
            # definitions, kept NULL-able so their SQL still parses/executes.
            "troponin_max_url_ratio": [None, None],
            "troponin_count_standard": [0, 0],
            "troponin_max_url_delta_hours": [None, None],
            "ntprobnp_max_url_ratio": [None, None],
            "ntprobnp_max_url_delta_hours": [None, None],
            "potassium_nearest": [None, None],
            "potassium_count_6h": [0, 0],
            "potassium_delta_hours": [None, None],
            "magnesium_nearest": [None, None],
            "magnesium_count_6h": [0, 0],
            "total_calcium_nearest": [None, None],
            "total_calcium_count_6h": [0, 0],
            "ionized_calcium_nearest": [None, None],
            "ionized_calcium_count_6h": [0, 0],
            "creatinine_nearest": [None, None],
            "hemoglobin_nearest": [None, None],
            "lactate_nearest": [None, None],
            "ck_mb_nearest": [None, None],
            "ck_mb_count_6h": [0, 0],
            "blood_gas_ph_nearest": [None, None],
            "blood_gas_ph_count_6h": [0, 0],
            # Derived from the mapping rather than listed by hand, so adding a
            # lab group cannot leave this fixture (and therefore the SQL
            # coverage check) silently behind.
            **{
                f"{group}{suffix}": default
                for group in LAB_ITEM_GROUPS
                for suffix, default in (
                    ("_nearest", [None, None]),
                    ("_delta_hours", [None, None]),
                    ("_count_6h", [0, 0]),
                    ("_reference_lower", [None, None]),
                    ("_reference_upper", [None, None]),
                )
            },
            "medication_assessed": [True, True],
            **{f"med_{name}": [False, False] for name in (
                "qt_prolonging_drug",
                "antiarrhythmic_class_i_iii",
                "beta_blocker",
                "calcium_channel_blocker",
                "digoxin",
                "diuretic",
                "vasopressor",
                "anticoagulant",
                "antiplatelet",
                "heparin",
                "nitrate",
                "electrolyte_replacement",
                "endocarditis_antibiotic_context",
            )},
            **{f"proc_{name}": [False, False] for name in (
                "pci",
                "coronary_angiography",
                "cabg",
                "pacemaker_or_icd",
                "cardioversion_or_defibrillation",
                "valve_procedure",
            )},
        }
    )


def test_every_registered_association_sql_executes_without_error() -> None:
    frame = _synthetic_context_frame()
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        for definition in default_context_association_definitions():
            connection.execute(
                "SELECT "
                f"({definition.value_sql}) AS value, "
                f"({definition.eligibility_sql}) AS eligible, "
                f"({definition.distance_sql}) AS distance "
                "FROM ctx"
            ).fetchdf()
    finally:
        connection.close()


def test_demographic_definitions_extract_expected_values() -> None:
    frame = _synthetic_context_frame()
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        result = connection.execute(
            "SELECT "
            "(age) AS age_value, "
            "(TRY_CAST(bmi_before_ecg AS DOUBLE)) AS bmi_value, "
            "(CASE WHEN sex = 'M' THEN 1.0 "
            " WHEN sex = 'F' THEN 0.0 END) AS sex_male_value "
            "FROM ctx"
        ).fetchdf()
    finally:
        connection.close()
    assert result["age_value"].tolist() == [55.0, 41.0]
    assert result["sex_male_value"].tolist() == [1.0, 0.0]
    assert result["bmi_value"].iloc[0] == pytest.approx(31.2)
    assert pd.isna(result["bmi_value"].iloc[1])


def test_care_setting_vitals_trend_infection_definitions_extract_expected_values() -> None:
    frame = _synthetic_context_frame()
    connection = duckdb.connect()
    definitions = {
        item.association_id: item
        for item in default_context_association_definitions()
    }
    try:
        connection.register("ctx", frame)
        for association_id in (
            "care_setting_icu",
            "vitals_ed_hr_variability",
            "vitals_ed_hr_trend_delta",
            "infection_positive_culture",
        ):
            definition = definitions[association_id]
            result = connection.execute(
                "SELECT "
                f"({definition.value_sql}) AS value, "
                f"({definition.eligibility_sql}) AS eligible "
                "FROM ctx"
            ).fetchdf()
            if association_id == "care_setting_icu":
                assert result["value"].tolist() == [1.0, 0.0]
                assert result["eligible"].tolist() == [True, True]
            elif association_id == "vitals_ed_hr_variability":
                assert result["value"].iloc[0] == pytest.approx(8.5)
                assert pd.isna(result["value"].iloc[1])
                # Second synthetic row has vitals_ed_reading_count == 0,
                # so it must be correctly excluded rather than treated as
                # "no variability" (a real, not just missing, measurement).
                assert result["eligible"].tolist() == [True, False]
            elif association_id == "vitals_ed_hr_trend_delta":
                assert result["value"].iloc[0] == pytest.approx(12.0)
            elif association_id == "infection_positive_culture":
                assert result["value"].tolist() == [1.0, 0.0]
                # Eligibility requires a specimen to actually have been
                # drawn -- "not tested" must never read as "negative".
                assert result["eligible"].tolist() == [True, False]
    finally:
        connection.close()


def test_extended_mortality_and_outcome_windows_match_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "death_delta_days": [45.0, 100.0, 400.0, None],
            "followup_hours": [200.0, 200.0, 200.0, 9000.0],
            "length_of_stay_days": [6, -1, None, 10],
            "readmission_30d_flag": [True, False, False, True],
            "hadm_id": [1, 2, 3, 4],
        }
    )
    definitions = {
        item.association_id: item
        for item in default_context_association_definitions()
    }
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        for association_id, expected in (
            ("outcome_90d_mortality", [1.0, 0.0, 0.0, 0.0]),
            ("outcome_1y_mortality", [1.0, 1.0, 0.0, 0.0]),
        ):
            definition = definitions[association_id]
            result = connection.execute(
                f"SELECT ({definition.value_sql}) AS value FROM ctx"
            ).fetchdf()
            assert result["value"].tolist() == expected, association_id

        los = definitions["outcome_length_of_stay"]
        eligible = connection.execute(
            f"SELECT ({los.eligibility_sql}) AS eligible FROM ctx"
        ).fetchdf()["eligible"].tolist()
        assert eligible == [True, False, False, True]

        readmission = definitions["outcome_readmission_30d"]
        readmission_values = connection.execute(
            f"SELECT ({readmission.value_sql}) AS value FROM ctx"
        ).fetchdf()["value"].tolist()
        assert readmission_values == [1.0, 0.0, 0.0, 1.0]
    finally:
        connection.close()


def test_socioeconomic_definitions_are_one_hot_and_exclusive() -> None:
    frame = _synthetic_context_frame()
    definitions = {
        item.association_id: item
        for item in default_context_association_definitions()
    }
    connection = duckdb.connect()
    try:
        connection.register("ctx", frame)
        medicare = definitions["socioeconomic_insurance_medicare"]
        private_flag = definitions["socioeconomic_insurance_private"]
        medicare_values = connection.execute(
            f"SELECT ({medicare.value_sql}) AS value FROM ctx"
        ).fetchdf()["value"].tolist()
        private_values = connection.execute(
            f"SELECT ({private_flag.value_sql}) AS value FROM ctx"
        ).fetchdf()["value"].tolist()
        assert medicare_values == [1.0, 0.0]
        assert private_values == [0.0, 1.0]

        language = definitions["socioeconomic_non_english_language"]
        language_values = connection.execute(
            f"SELECT ({language.value_sql}) AS value FROM ctx"
        ).fetchdf()["value"].tolist()
        assert language_values == [1.0, 0.0]
    finally:
        connection.close()


def test_no_association_domain_targets_race_or_ethnicity() -> None:
    for definition in default_context_association_definitions():
        assert "race" not in definition.value_sql.casefold()
        assert "ethnicity" not in definition.value_sql.casefold()
