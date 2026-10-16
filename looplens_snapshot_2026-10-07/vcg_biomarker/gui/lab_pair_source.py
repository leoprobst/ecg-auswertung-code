"""Read-only development identity and WFDB confinement checks before lab import."""
from pathlib import Path

import duckdb
import pandas as pd

from ..mimic.patient_identity import known_patient_mask


def resolve_lab_pair_source(config, pair, side):
    if side not in ("first", "second"):
        raise ValueError("Choose first or second ECG")
    keys = ("subject_id", "first_study_id", "second_study_id")
    if not isinstance(pair, dict) or not known_patient_mask(pd.Series([pair.get(key) for key in keys])).all():
        raise ValueError("Missing patient/ECG pair identity")
    if pair["first_study_id"] == pair["second_study_id"]:
        raise ValueError("Pair must contain different ECGs")
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as con:
        rows = con.execute('''SELECT study_id, ecg_time, hadm_id, relative_waveform_path, split
            FROM read_parquet(?) WHERE split IN ('train', 'validation')
            AND subject_id=? AND study_id IN (?, ?)''',
            [str(config.master_root/"ecg_master_index.parquet"), *[pair[key] for key in keys]]).df()
    if len(rows) != 2 or rows.study_id.nunique() != 2:
        raise ValueError("Pair has no unique matching development ECGs; no fallback or locked-test access")
    rows = rows.set_index("study_id")
    first, second = rows.loc[pair["first_study_id"]], rows.loc[pair["second_study_id"]]
    if first.split != second.split:
        raise ValueError("Patient pair crosses development splits")
    if pd.isna(first.ecg_time) or pd.isna(second.ecg_time) or first.ecg_time >= second.ecg_time:
        raise ValueError("ECG pair chronology is invalid")
    if pd.isna(first.hadm_id) or pd.isna(second.hadm_id) or first.hadm_id != second.hadm_id:
        raise ValueError("Dose pair does not belong to the same known admission")
    relative = rows.loc[pair[f"{side}_study_id"], "relative_waveform_path"]
    return _confined_wfdb_source(config, relative)


def resolve_review_source(config, row):
    """Resolve one review case against the development master, not a row path."""
    if not isinstance(row, dict):
        raise ValueError("No review case selected")
    identifiers = pd.Series([row.get("subject_id"), row.get("study_id")])
    if not known_patient_mask(identifiers).all():
        raise ValueError("Review row lacks a known patient and ECG identity")
    split = str(row.get("split") or "")
    if split not in {"train", "validation"}:
        raise ValueError("Only development review cases can be opened")
    with duckdb.connect(config={"threads": 1, "memory_limit": "256MB"}) as con:
        matched = con.execute('''SELECT subject_id, study_id, split, relative_waveform_path
            FROM read_parquet(?) WHERE subject_id=? AND study_id=?
            AND split IN ('train', 'validation')''', [
            str(config.master_root / "ecg_master_index.parquet"),
            row["subject_id"], row["study_id"],
        ]).df()
    if len(matched) != 1 or str(matched.iloc[0]["split"]) != split:
        raise ValueError("Review identity has no unique matching development ECG")
    relative = matched.iloc[0]["relative_waveform_path"]
    review_relative = row.get("waveform_path")
    if isinstance(review_relative, str) and review_relative.strip():
        if Path(review_relative).as_posix().removesuffix(".hea") != Path(str(relative)).as_posix().removesuffix(".hea"):
            raise ValueError("Review waveform path conflicts with master identity")
    return _confined_wfdb_source(config, relative)


def _confined_wfdb_source(config, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("No waveform path for selected ECG")
    relative = Path(relative)
    if relative.is_absolute() or relative.drive:
        raise ValueError("Expected confined relative waveform path")
    root = config.mimic_ecg_root.resolve()
    source = (root/relative).resolve()
    if source.suffix == "":
        source = source.with_suffix(".hea")
    if source.suffix.lower() != ".hea" or not source.is_relative_to(root):
        raise ValueError("Waveform path escapes ECG root or is not WFDB")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not lines or "/" in lines[0].split()[0]:
        raise ValueError("Only single-segment WFDB records are supported for this link")
    try:
        count = int(lines[0].split()[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid WFDB signal count") from exc
    if not 1 <= count <= 64 or len(lines) < count+1:
        raise ValueError("Incomplete WFDB signal header")
    for line in lines[1:count+1]:
        data = (source.parent/line.split()[0]).resolve()
        if not data.is_relative_to(source.parent) or not data.is_file():
            raise ValueError("Missing or unconfined WFDB signal file")
    return source
