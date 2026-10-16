from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceTableSpec:
    table_id: str
    relative_path: str
    required_columns: tuple[str, ...]
    required: bool = True
    source: str = "mimic_iv"
    alternative_relative_paths: tuple[str, ...] = ()


SOURCE_TABLE_SPECS: tuple[SourceTableSpec, ...] = (
    SourceTableSpec(
        "patients", "hosp/patients.csv.gz",
        ("subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"),
    ),
    SourceTableSpec(
        "admissions", "hosp/admissions.csv.gz",
        (
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "admission_type", "race", "edregtime", "edouttime",
            "hospital_expire_flag",
        ),
    ),
    SourceTableSpec(
        "transfers", "hosp/transfers.csv.gz",
        ("subject_id", "hadm_id", "transfer_id", "eventtype", "intime", "outtime"),
    ),
    SourceTableSpec(
        "diagnoses_icd", "hosp/diagnoses_icd.csv.gz",
        ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
    ),
    SourceTableSpec(
        "procedures_icd", "hosp/procedures_icd.csv.gz",
        ("subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"),
    ),
    SourceTableSpec(
        "d_labitems", "hosp/d_labitems.csv.gz",
        ("itemid", "label", "fluid", "category"),
    ),
    SourceTableSpec(
        "labevents", "hosp/labevents.csv.gz",
        (
            "labevent_id", "subject_id", "hadm_id", "itemid", "charttime",
            "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag",
        ),
    ),
    SourceTableSpec(
        "prescriptions", "hosp/prescriptions.csv.gz",
        ("subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"),
        required=False,
    ),
    SourceTableSpec(
        "pharmacy", "hosp/pharmacy.csv.gz",
        ("subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime", "medication"),
        required=False,
    ),
    SourceTableSpec(
        "emar", "hosp/emar.csv.gz",
        ("subject_id", "hadm_id", "emar_id", "charttime", "medication", "event_txt"),
        required=False,
    ),
    SourceTableSpec(
        "emar_detail", "hosp/emar_detail.csv.gz",
        ("subject_id", "emar_id", "emar_seq", "dose_given", "dose_given_unit", "route"),
        required=False,
    ),
    SourceTableSpec(
        "poe", "hosp/poe.csv.gz",
        (
            "poe_id", "poe_seq", "subject_id", "hadm_id", "ordertime",
            "order_type", "transaction_type", "order_status",
        ),
        required=False,
    ),
    SourceTableSpec(
        "poe_detail", "hosp/poe_detail.csv.gz",
        ("poe_id", "poe_seq", "subject_id", "field_name", "field_value"),
        required=False,
    ),
    SourceTableSpec(
        "services", "hosp/services.csv.gz",
        ("subject_id", "hadm_id", "transfertime", "curr_service"),
        required=False,
    ),
    SourceTableSpec(
        "omr", "hosp/omr.csv.gz",
        ("subject_id", "chartdate", "seq_num", "result_name", "result_value"),
        required=False,
    ),
    SourceTableSpec(
        "microbiologyevents", "hosp/microbiologyevents.csv.gz",
        ("microevent_id", "subject_id", "hadm_id", "chartdate", "charttime"),
        required=False,
    ),
    SourceTableSpec(
        "edstays", "ed/edstays.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "intime", "outtime", "race", "disposition"),
        source="mimic_ed",
    ),
    SourceTableSpec(
        "triage", "ed/triage.csv.gz",
        ("subject_id", "stay_id", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"),
        source="mimic_ed",
    ),
    SourceTableSpec(
        "vitalsign", "ed/vitalsign.csv.gz",
        ("subject_id", "stay_id", "charttime", "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "rhythm", "pain"),
        source="mimic_ed",
    ),
    SourceTableSpec(
        "ed_diagnosis", "ed/diagnosis.csv.gz",
        ("subject_id", "stay_id", "seq_num", "icd_code", "icd_version", "icd_title"),
        source="mimic_ed",
    ),
    SourceTableSpec(
        "medrecon", "ed/medrecon.csv.gz",
        ("subject_id", "stay_id", "charttime", "name", "gsn", "ndc"),
        required=False,
        source="mimic_ed",
    ),
    SourceTableSpec(
        "pyxis", "ed/pyxis.csv.gz",
        ("subject_id", "stay_id", "charttime", "med_rn", "name", "gsn_rn", "gsn"),
        required=False,
        source="mimic_ed",
    ),
    SourceTableSpec(
        "ecg_record_list", "record_list.csv",
        ("subject_id", "study_id", "file_name", "ecg_time", "path"),
        source="mimic_ecg",
    ),
    SourceTableSpec(
        "machine_measurements", "machine_measurements.csv",
        ("subject_id", "study_id", "ecg_time", "qrs_onset", "qrs_end", "t_end", "qrs_axis"),
        source="mimic_ecg",
    ),
    SourceTableSpec(
        "waveform_note_links", "waveform_note_links.csv",
        ("subject_id", "study_id", "waveform_path", "note_id", "charttime"),
        required=False,
        source="mimic_ecg",
    ),
    SourceTableSpec(
        "icustays", "icu/icustays.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit", "last_careunit", "los"),
        required=False,
    ),
    SourceTableSpec(
        "d_items", "icu/d_items.csv.gz",
        ("itemid", "label", "category", "unitname", "param_type", "linksto"),
        required=False,
    ),
    SourceTableSpec(
        "chartevents", "icu/chartevents.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valuenum", "valueuom"),
        required=False,
    ),
    SourceTableSpec(
        "inputevents", "icu/inputevents.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "amount", "amountuom"),
        required=False,
    ),
    SourceTableSpec(
        "outputevents", "icu/outputevents.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value", "valueuom"),
        required=False,
    ),
    SourceTableSpec(
        "procedureevents", "icu/procedureevents.csv.gz",
        ("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "value"),
        required=False,
    ),
    SourceTableSpec(
        "discharge", "discharge.csv.gz",
        ("note_id", "subject_id", "hadm_id", "note_type", "charttime", "text"),
        required=False,
        source="mimic_note",
        alternative_relative_paths=(
            "discharge.csv",
            "note/discharge.csv.gz",
            "note/discharge.csv",
        ),
    ),
    SourceTableSpec(
        "discharge_detail", "discharge_detail.csv.gz",
        ("note_id", "subject_id", "field_name", "field_value", "field_ordinal"),
        required=False,
        source="mimic_note",
        alternative_relative_paths=(
            "discharge_detail.csv",
            "note/discharge_detail.csv.gz",
            "note/discharge_detail.csv",
        ),
    ),
    SourceTableSpec(
        "radiology", "radiology.csv.gz",
        ("note_id", "subject_id", "hadm_id", "note_type", "charttime", "text"),
        required=False,
        source="mimic_note",
        alternative_relative_paths=(
            "radiology.csv",
            "note/radiology.csv.gz",
            "note/radiology.csv",
        ),
    ),
    SourceTableSpec(
        "radiology_detail", "radiology_detail.csv.gz",
        ("note_id", "subject_id", "field_name", "field_value", "field_ordinal"),
        required=False,
        source="mimic_note",
        alternative_relative_paths=(
            "radiology_detail.csv",
            "note/radiology_detail.csv.gz",
            "note/radiology_detail.csv",
        ),
    ),
)


ECHO_SPECS: tuple[SourceTableSpec, ...] = (
    SourceTableSpec(
        "echo_structured", "",
        (
            "subject_id", "measurement_id", "measurement_datetime", "test_type",
            "measurement", "measurement_description", "result", "unit",
        ),
        source="mimic_echo",
    ),
    SourceTableSpec(
        "echo_study_list", "",
        ("subject_id", "study_id", "measurement_id"),
        source="mimic_echo",
    ),
    SourceTableSpec(
        "echo_record_list", "",
        ("subject_id",),
        source="mimic_echo",
    ),
)


def read_csv_header(path: Path | str) -> tuple[str, ...]:
    source = Path(path)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return tuple(next(reader))
        except StopIteration:
            return ()


def validate_columns(
    path: Path | str,
    required_columns: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns = read_csv_header(path)
    available = set(columns)
    missing = tuple(column for column in required_columns if column not in available)
    return columns, missing
