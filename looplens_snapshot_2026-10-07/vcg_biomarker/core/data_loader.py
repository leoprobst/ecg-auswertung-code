from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb


LEAD_NAMES_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LEAD_MAP = {name: i for i, name in enumerate(LEAD_NAMES_12)}

PTBXL_DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


@dataclass
class ECGRecord:
    ecg_id: int
    patient_id: int
    age: Optional[int]
    sex: Optional[str]
    height: Optional[float]
    weight: Optional[float]
    recording_date: Optional[str]
    scp_codes: dict[str, int]
    report: str
    heart_axis: Optional[str]
    diagnostic_superclass: list[str]
    diagnostic_subclass: list[str]
    validated_by_human: bool
    strat_fold: int
    signal_quality: dict = field(default_factory=dict)
    ecg_signal: Optional[np.ndarray] = None
    sampling_rate: int = 500
    r_peaks: Optional[np.ndarray] = None

    @property
    def filename(self) -> str:
        return f"{self.ecg_id:05d}"

    @property
    def has_signal(self) -> bool:
        return self.ecg_signal is not None


def _parse_scp_codes(raw: str) -> dict[str, int]:
    if pd.isna(raw):
        return {}
    return ast.literal_eval(raw)


def _aggregate_diagnostic(scp_codes: dict[str, int], agg_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    superclasses = set()
    subclasses = set()
    for key in scp_codes:
        if key in agg_df.index:
            row = agg_df.loc[key]
            if pd.notna(row.get("diagnostic_class")):
                superclasses.add(row["diagnostic_class"])
            if pd.notna(row.get("diagnostic_subclass")):
                subclasses.add(row["diagnostic_subclass"])
    return sorted(superclasses), sorted(subclasses)


def _parse_signal_quality(row: pd.Series) -> dict:
    quality = {}
    for col in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats", "pacemaker"]:
        val = row.get(col)
        if pd.notna(val):
            quality[col] = val
    return quality


class PTBXLDataLoader:
    def __init__(self, ptbxl_path: str | Path, use_hr: bool = True):
        self.ptbxl_path = Path(ptbxl_path)
        self.use_hr = use_hr
        self._db: Optional[pd.DataFrame] = None
        self._agg_df: Optional[pd.DataFrame] = None

    def load_database(self) -> pd.DataFrame:
        db_path = self.ptbxl_path / "ptbxl_database.csv"
        self._db = pd.read_csv(db_path, index_col="ecg_id")
        self._db.scp_codes = self._db.scp_codes.apply(_parse_scp_codes)

        scp_path = self.ptbxl_path / "scp_statements.csv"
        scp_df = pd.read_csv(scp_path, index_col=0)
        self._agg_df = scp_df[scp_df.diagnostic == 1].copy()

        superclasses_list = []
        subclasses_list = []
        for idx, row in self._db.iterrows():
            sc, ssc = _aggregate_diagnostic(row.scp_codes, self._agg_df)
            superclasses_list.append(sc)
            subclasses_list.append(ssc)

        self._db["diagnostic_superclass"] = superclasses_list
        self._db["diagnostic_subclass"] = subclasses_list

        return self._db

    @property
    def database(self) -> pd.DataFrame:
        if self._db is None:
            self.load_database()
        return self._db

    def get_record_ids(
        self,
        superclasses: Optional[list[str]] = None,
        min_likelihood: int = 0,
        folds: Optional[list[int]] = None,
        validated_only: bool = False,
    ) -> list[int]:
        db = self.database
        mask = pd.Series(True, index=db.index)

        if superclasses is not None:
            mask = mask & db.diagnostic_superclass.apply(
                lambda x: any(s in x for s in superclasses)
            )

        if min_likelihood > 0:
            mask = mask & db.scp_codes.apply(
                lambda x: any(v >= min_likelihood for v in x.values())
            )

        if folds is not None:
            mask = mask & db.strat_fold.isin(folds)

        if validated_only:
            mask = mask & (db.validated_by_human == True)

        return db[mask].index.tolist()

    def load_signal(self, ecg_id: int) -> np.ndarray:
        db = self.database
        row = db.loc[ecg_id]

        if self.use_hr:
            filename = row.filename_hr
        else:
            filename = row.filename_lr

        record_path = self.ptbxl_path / filename
        record = wfdb.rdrecord(str(record_path))

        if record.p_signal is None:
            raise ValueError(f"WFDB record has no physical signal: {record_path}")
        signal = np.asarray(record.p_signal, dtype=float)
        if signal.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional ECG, got shape {signal.shape}"
            )

        normalized = [
            str(name).strip().replace(" ", "").casefold()
            for name in record.sig_name
        ]
        duplicates = {
            name for name in normalized if normalized.count(name) > 1
        }
        if duplicates:
            raise ValueError(
                f"Duplicate WFDB lead names: {sorted(duplicates)}"
            )
        required = [
            name.strip().replace(" ", "").casefold()
            for name in LEAD_NAMES_12
        ]
        missing = [
            name
            for name, normalized_name in zip(LEAD_NAMES_12, required)
            if normalized_name not in normalized
        ]
        if missing:
            raise ValueError(
                f"Missing required ECG leads: {', '.join(missing)}"
            )
        lead_indices = [normalized.index(name) for name in required]
        signal = signal[:, lead_indices]

        expected_fs = 500 if self.use_hr else 100
        if int(round(float(record.fs))) != expected_fs:
            raise ValueError(
                f"Unexpected PTB-XL sampling rate {record.fs}; "
                f"expected {expected_fs} Hz"
            )

        return signal

    def load_r_peaks(self, ecg_id: int) -> Optional[np.ndarray]:
        db = self.database
        row = db.loc[ecg_id]
        r_peaks_raw = row.get("r_peaks")
        if pd.isna(r_peaks_raw):
            return None
        try:
            return np.array(ast.literal_eval(str(r_peaks_raw)))
        except (ValueError, SyntaxError):
            return None

    def get_record(self, ecg_id: int, load_signal: bool = True) -> ECGRecord:
        db = self.database
        row = db.loc[ecg_id]

        signal = None
        signal_quality = _parse_signal_quality(row)
        sr = 500 if self.use_hr else 100
        if load_signal:
            try:
                signal = self.load_signal(ecg_id)
            except Exception as exc:
                signal_quality["load_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        r_peaks = None
        try:
            r_peaks = self.load_r_peaks(ecg_id)
        except Exception:
            pass

        return ECGRecord(
            ecg_id=ecg_id,
            patient_id=int(row.patient_id) if pd.notna(row.patient_id) else 0,
            age=int(row.age) if pd.notna(row.age) else None,
            sex=row.sex if pd.notna(row.sex) else None,
            height=float(row.height) if pd.notna(row.height) else None,
            weight=float(row.weight) if pd.notna(row.weight) else None,
            recording_date=str(row.recording_date) if pd.notna(row.recording_date) else None,
            scp_codes=row.scp_codes,
            report=str(row.report) if pd.notna(row.report) else "",
            heart_axis=row.heart_axis if pd.notna(row.heart_axis) else None,
            diagnostic_superclass=row.diagnostic_superclass,
            diagnostic_subclass=row.diagnostic_subclass,
            validated_by_human=bool(row.validated_by_human),
            strat_fold=int(row.strat_fold),
            signal_quality=signal_quality,
            ecg_signal=signal,
            sampling_rate=sr,
            r_peaks=r_peaks,
        )

    def get_records_batch(
        self,
        ecg_ids: list[int],
        load_signals: bool = True,
        progress_callback=None,
    ) -> list[ECGRecord]:
        records = []
        total = len(ecg_ids)
        for i, eid in enumerate(ecg_ids):
            try:
                rec = self.get_record(eid, load_signal=load_signals)
                records.append(rec)
            except Exception as e:
                print(f"Warning: Failed to load record {eid}: {e}")
            if progress_callback and (i + 1) % max(1, total // 100) == 0:
                progress_callback(i + 1, total)
        return records

    def get_diagnostic_distribution(self) -> pd.DataFrame:
        db = self.database
        counts = {}
        for sc in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            counts[sc] = db.diagnostic_superclass.apply(lambda x: sc in x).sum()
        return pd.DataFrame.from_dict(counts, orient="index", columns=["count"])

    def filter_by_superclass(self, superclass: str) -> list[int]:
        db = self.database
        mask = db.diagnostic_superclass.apply(lambda x: superclass in x)
        return db[mask].index.tolist()
