"""CODE-15 % and SaMi-Trop: the outpatient cohorts this project is missing.

Every reference value the project can currently produce comes from patients in
a tertiary hospital. Excluding the ones with a recorded cardiac diagnosis, a
cardiotropic drug, a device or an intensive-care episode leaves 76.977 people,
and they are still people who came to a hospital and had an electrocardiogram
taken. That is a defensible comparison group and it is not a normal range, and
no amount of care with the exclusion rules changes which population was
sampled.

**CODE-15 %** supplies a primary-care comparison population: 345.779
recordings from 233.770 patients of the Telehealth Network of Minas Gerais,
recorded in primary care between 2010 and 2016, with age, sex and mortality.
Not a verified healthy reference population. Zenodo, DOI 10.5281/zenodo.4916206.

**SaMi-Trop** is 1.631 recordings with serologically confirmed Chagas
infection in the same format. Infection does not establish cardiomyopathy in
every participant. It supports Chagas-associated morphology and prognosis
research, not an automatic diagnostic rule or a claim of marker novelty.

Both cohorts can be configured independently. This module is the reader, the
contract and the checks; ``cohort_status`` reports what is present. The checks matter more
than the reader, because four things about this format will silently produce
wrong numbers:

*400 Hz, not 500.* The resampling is explicit and recorded, and the original
rate travels as a column, because a rate difference otherwise enters every
marker that counts samples or measures a slope.

*Zero padding.* Each recording is 4.096 samples wide regardless of its true
length, padded with exact zeros at both ends. A delineator handed the padding
sees a flat lead and abstains, or worse, treats the step at the padding edge as
a deflection. The padding is trimmed and the trimmed length recorded.

*Lead order is fixed and different.* ``DI, DII, DIII, aVR, aVL, aVF, V1..V6``.
The project's own order is checked against it rather than assumed.

*The exam identifier is not the patient identifier.* CODE-15 carries both, and
a percentile computed per recording rather than per patient repeats the
pseudo-replication the MIMIC cohort had to be protected from.

Distribution shifts are diagnostic warnings, not loader acceptance tests.
Neither cohort must have smaller QRS-T angles than MIMIC. Lead order, units,
resampling and identity require direct checks, independent of whether the
observed physiology matches an expected cohort median.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.signal import resample_poly


CODE15_COHORT_VERSION = "1.4.0"

# The lead order the HDF5 tracings arrive in. Fixed by the publication, and
# checked rather than assumed on every file.
CODE15_LEAD_ORDER: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

CODE15_SAMPLING_RATE_HZ = 400.0
CODE15_SAMPLES_PER_RECORD = 4096

# A recording is padded with exact zeros to the fixed width. Only exact zeros
# across every lead count as padding: a genuinely isoelectric sample is never
# exactly zero in twelve leads at once after the recorder's own scaling.
PADDING_TOLERANCE = 0.0

# Below this the trimmed recording is too short for the eight-second rule the
# quality gate already applies.
MINIMUM_TRIMMED_SECONDS = 8.0

_COHORT_REQUIRED_METADATA: dict[str, tuple[str, ...]] = {
    "code15": (
        "exam_id",
        "patient_id",
        "age",
        "is_male",
        "death",
        "timey",
        "trace_file",
    ),
    "sami_trop": (
        "exam_id",
        "age",
        "is_male",
        "death",
        "timey",
    ),
}


@dataclass(frozen=True)
class Code15Record:
    exam_id: int
    patient_id: Optional[int]
    signal_millivolts: np.ndarray
    sampling_rate_hz: float
    original_sampling_rate_hz: float
    lead_names: tuple[str, ...]
    trimmed_samples: int
    padded_samples: int
    source: str

    def duration_seconds(self) -> float:
        return float(self.signal_millivolts.shape[0]) / float(self.sampling_rate_hz)


class TracingSource(Protocol):
    """The two things this module needs from an HDF5 file.

    Kept behind a protocol so the contract, the trimming and the resampling can
    be tested without the HDF5 library, and so a future format change touches
    one binding rather than the whole module.
    """

    def exam_ids(self) -> np.ndarray: ...

    def tracings(self) -> np.ndarray: ...

    def iter_batches(
        self,
        *,
        batch_size: int = 64,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


@dataclass
class ArrayTracingSource:
    """A source backed by arrays already in memory."""

    ids: np.ndarray
    values: np.ndarray

    def exam_ids(self) -> np.ndarray:
        return np.asarray(self.ids)

    def tracings(self) -> np.ndarray:
        return np.asarray(self.values)


class Hdf5TracingSource:
    """The real binding. ``h5py`` is imported here and nowhere else."""

    def __init__(
        self,
        path: Path | str,
        *,
        tracings_key: str = "tracings",
        exam_id_key: str = "exam_id",
        fallback_exam_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self._path = Path(path)
        self._tracings_key = tracings_key
        self._exam_id_key = exam_id_key
        self._fallback_exam_ids = (
            None
            if fallback_exam_ids is None
            else np.asarray(fallback_exam_ids, dtype=np.int64)
        )

    def _open(self):
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - environment specific
            raise RuntimeError(
                "CODE-15 and SaMi-Trop are distributed as HDF5. Install h5py "
                "(declared in requirements.txt) before reading them."
            ) from error
        return h5py.File(str(self._path), "r")

    def exam_ids(self) -> np.ndarray:
        with self._open() as handle:
            if self._exam_id_key in handle:
                return np.asarray(handle[self._exam_id_key][:])
            if self._fallback_exam_ids is None:
                raise KeyError(
                    f"{self._path.name} has no {self._exam_id_key!r} dataset "
                    "and no metadata-order fallback was supplied"
                )
            if len(self._fallback_exam_ids) != len(handle[self._tracings_key]):
                raise ValueError(
                    "Fallback exam IDs do not match the number of tracings"
                )
            return self._fallback_exam_ids.copy()

    def tracings(self) -> np.ndarray:
        with self._open() as handle:
            return np.asarray(handle[self._tracings_key][:])

    def iter_batches(
        self,
        *,
        batch_size: int = 64,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Stream bounded HDF5 batches; never materialise a multi-GB file."""

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        with self._open() as handle:
            tracing_dataset = handle[self._tracings_key]
            if self._exam_id_key in handle:
                exam_dataset = handle[self._exam_id_key]
                fallback = None
            else:
                exam_dataset = None
                fallback = self._fallback_exam_ids
                if fallback is None or len(fallback) != len(tracing_dataset):
                    raise ValueError(
                        "Metadata-order exam IDs are required and must match "
                        "a tracing file without its own exam_id dataset"
                    )
            assert exam_dataset is not None or fallback is not None
            for start in range(0, len(tracing_dataset), int(batch_size)):
                end = min(start + int(batch_size), len(tracing_dataset))
                exam_ids = (
                    np.asarray(exam_dataset[start:end])
                    if exam_dataset is not None
                    else np.asarray(fallback)[start:end]
                )
                yield exam_ids, np.asarray(tracing_dataset[start:end])

    def iter_selected_batches(
        self,
        selected_exam_ids: Collection[int],
        *,
        batch_size: int = 64,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Read only explicitly selected tracings from a large HDF5 release.

        A capped external-cohort pilot previously streamed every tracing and
        discarded almost all of them *after* the multi-gigabyte HDF5 read. A
        100-record smoke test therefore performed the same 68 GB signal read
        as the complete CODE-15 extraction. HDF5 supports increasing fancy
        indices, so capped runs can resolve the small ID vector first and read
        only the requested tracing slabs. Complete runs keep ``iter_batches``
        because contiguous streaming remains the fastest full-table access.
        """

        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        requested = {int(value) for value in selected_exam_ids}
        if not requested:
            return
        with self._open() as handle:
            tracing_dataset = handle[self._tracings_key]
            if self._exam_id_key in handle:
                all_exam_ids = np.asarray(handle[self._exam_id_key][:]).reshape(-1)
            else:
                fallback = self._fallback_exam_ids
                if fallback is None or len(fallback) != len(tracing_dataset):
                    raise ValueError(
                        "Metadata-order exam IDs are required and must match "
                        "a tracing file without its own exam_id dataset"
                    )
                all_exam_ids = np.asarray(fallback).reshape(-1)
            if len(all_exam_ids) != len(tracing_dataset):
                raise ValueError("Tracing and exam_id datasets must have equal length")
            positions = np.flatnonzero(
                np.fromiter(
                    (int(value) in requested for value in all_exam_ids),
                    dtype=bool,
                    count=len(all_exam_ids),
                )
            )
            for start in range(0, len(positions), int(batch_size)):
                selected_positions = positions[start : start + int(batch_size)]
                # h5py requires fancy indices to be strictly increasing;
                # ``flatnonzero`` already provides that order.
                yield (
                    np.asarray(all_exam_ids[selected_positions]),
                    np.asarray(tracing_dataset[selected_positions.tolist()]),
                )


def assert_lead_order(lead_names: Sequence[str]) -> None:
    """Fail loudly rather than compute twelve markers on shuffled leads."""

    observed = tuple(str(name).strip() for name in lead_names)
    if observed != CODE15_LEAD_ORDER:
        raise ValueError(
            "CODE-15 lead order does not match the published one. "
            f"Expected {CODE15_LEAD_ORDER}, got {observed}."
        )


def trim_zero_padding(
    tracing: np.ndarray,
    *,
    tolerance: float = PADDING_TOLERANCE,
) -> tuple[np.ndarray, int, int]:
    """Remove the exact-zero padding at both ends of one recording.

    Returns the trimmed signal, the number of samples kept and the number
    removed. A recording that is entirely padding returns an empty signal
    rather than raising: the caller decides what to do with a dead record, and
    a loader that raises on one file cannot read the other 345.778.
    """

    values = np.asarray(tracing, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"A CODE-15 tracing is two-dimensional, got shape {values.shape}"
        )
    alive = np.any(np.abs(values) > tolerance, axis=1)
    if not alive.any():
        return values[:0], 0, int(values.shape[0])
    first = int(np.argmax(alive))
    last = int(len(alive) - np.argmax(alive[::-1]))
    trimmed = values[first:last]
    return trimmed, int(trimmed.shape[0]), int(values.shape[0] - trimmed.shape[0])


def resample_to(
    signal: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
) -> np.ndarray:
    """Polyphase-resample the time axis while preserving duration.

    Even pure upsampling needs a reconstruction filter: piecewise-linear
    interpolation attenuates the upper ECG spectrum and can bias slopes and
    QRS morphology systematically by cohort. ``resample_poly`` performs the
    rational 4:5 conversion in compiled code with an anti-imaging FIR.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a two-dimensional signal, got {values.shape}")
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive")
    if float(source_rate_hz) == float(target_rate_hz) or values.shape[0] < 2:
        return values
    duration = values.shape[0] / float(source_rate_hz)
    target_count = int(round(duration * float(target_rate_hz)))
    if target_count < 2:
        return values[:0]
    from fractions import Fraction

    ratio = Fraction(float(target_rate_hz) / float(source_rate_hz)).limit_denominator(
        10_000
    )
    result = resample_poly(
        values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
        padtype="line",
    )
    if result.shape[0] > target_count:
        return result[:target_count]
    if result.shape[0] < target_count:
        return np.pad(result, ((0, target_count - result.shape[0]), (0, 0)), mode="edge")
    return result


def read_records(
    source: TracingSource,
    *,
    target_rate_hz: float = 500.0,
    source_rate_hz: float = CODE15_SAMPLING_RATE_HZ,
    patient_ids: Optional[Mapping[int, int]] = None,
    cohort: str = "code15",
    minimum_trimmed_seconds: float = MINIMUM_TRIMMED_SECONDS,
) -> tuple[list[Code15Record], pd.DataFrame]:
    """Read, trim, resample -- and report what was dropped and why."""

    ids = np.asarray(source.exam_ids()).reshape(-1)
    tracings = np.asarray(source.tracings())
    if tracings.ndim != 3 or tracings.shape[0] != ids.shape[0]:
        raise ValueError(
            "Tracings must be (records, samples, leads) and match the exam "
            f"ids: got {tracings.shape} against {ids.shape}"
        )
    if tracings.shape[2] != len(CODE15_LEAD_ORDER):
        raise ValueError(
            f"Expected {len(CODE15_LEAD_ORDER)} leads, got {tracings.shape[2]}"
        )

    records: list[Code15Record] = []
    rejected: list[dict] = []
    for index in range(tracings.shape[0]):
        exam_id = int(ids[index])
        trimmed, kept, padded = trim_zero_padding(tracings[index])
        seconds = kept / float(source_rate_hz)
        if kept == 0:
            rejected.append(
                {
                    "exam_id": exam_id,
                    "reason": "entirely_padding",
                    "trimmed_seconds": 0.0,
                }
            )
            continue
        if seconds < float(minimum_trimmed_seconds):
            rejected.append(
                {
                    "exam_id": exam_id,
                    "reason": "shorter_than_eight_seconds_after_trimming",
                    "trimmed_seconds": seconds,
                }
            )
            continue
        resampled = resample_to(trimmed, source_rate_hz, target_rate_hz)
        records.append(
            Code15Record(
                exam_id=exam_id,
                patient_id=(
                    None if patient_ids is None else patient_ids.get(exam_id)
                ),
                signal_millivolts=resampled,
                sampling_rate_hz=float(target_rate_hz),
                original_sampling_rate_hz=float(source_rate_hz),
                lead_names=CODE15_LEAD_ORDER,
                trimmed_samples=kept,
                padded_samples=padded,
                source=cohort,
            )
        )
    return records, pd.DataFrame(
        rejected, columns=["exam_id", "reason", "trimmed_seconds"]
    )


def read_exam_metadata(path: Path | str, *, cohort: str | None = None) -> pd.DataFrame:
    """The ``exams.csv`` companion, normalised to this project's names."""

    frame = pd.read_csv(path)
    if cohort not in {None, "code15", "sami_trop"}:
        raise ValueError("Unknown population cohort identity contract")
    if "exam_id" not in frame:
        raise ValueError("Missing exam_id")
    ids = pd.to_numeric(frame.exam_id, errors="coerce")
    if (not np.isfinite(ids).all() or (ids <= 0).any()
            or (ids % 1 != 0).any() or ids.duplicated().any()):
        raise ValueError("Exam identifiers must be unique positive integers")
    frame["exam_id"] = ids.astype("int64")
    rename = {
        "exam_id": "exam_id",
        "patient_id": "patient_id",
        "age": "age",
        "is_male": "is_male",
        "death": "death",
        "timey": "follow_up_years",
        "normal_ecg": "reported_normal_ecg",
        "trace_file": "trace_file",
        "1dAVb": "first_degree_av_block",
        "RBBB": "right_bundle_branch_block",
        "LBBB": "left_bundle_branch_block",
        "SB": "sinus_bradycardia",
        "ST": "sinus_tachycardia",
        "AF": "atrial_fibrillation",
        "nn_predicted_age": "ecg_predicted_age",
    }
    present = {key: value for key, value in rename.items() if key in frame.columns}
    frame = frame[list(present)].rename(columns=present)
    if "is_male" in frame.columns:
        # Missing sex is missing information, never female.  The previous
        # fillna(False) silently changed every redacted/unknown value into F
        # and therefore biased sex-stratified references and adjustment.
        male = pd.to_numeric(frame["is_male"], errors="coerce")
        sex = pd.Series(pd.NA, index=frame.index, dtype="string")
        sex.loc[male.eq(1)] = "M"
        sex.loc[male.eq(0)] = "F"
        frame["sex"] = sex
    if "patient_id" not in frame.columns:
        # SaMi-Trop has one baseline exam per participant and does not expose
        # a separate identifier. The derivation is explicit so it can never
        # be mistaken for a native patient key.
        if cohort != "sami_trop":
            raise ValueError("Missing patient_id: exam identity is not patient identity")
        frame["patient_id"] = frame["exam_id"]
        frame["patient_id_source"] = "derived_one_exam_per_participant"
    else:
        patients = pd.to_numeric(frame.patient_id, errors="coerce")
        if (not np.isfinite(patients).all() or (patients <= 0).any()
                or (patients % 1 != 0).any()):
            raise ValueError("Patient identifiers must be known positive integers")
        frame["patient_id"] = patients.astype("int64")
        frame["patient_id_source"] = "native"
    frame["patient_identity_namespace"] = cohort or "unspecified_native_source"
    if {"ecg_predicted_age", "age"}.issubset(frame.columns):
        frame["ecg_age_gap_years"] = pd.to_numeric(
            frame["ecg_predicted_age"], errors="coerce"
        ) - pd.to_numeric(frame["age"], errors="coerce")
    frame["cohort_version"] = CODE15_COHORT_VERSION
    return frame


@dataclass(frozen=True)
class ExternalCohortStatus:
    cohort: str
    root: Optional[Path]
    present: bool
    tracing_files: int
    metadata_present: bool
    purpose: str
    note: str

    def to_dict(self) -> dict:
        return {
            "cohort": self.cohort,
            "root": None if self.root is None else str(self.root),
            "present": bool(self.present),
            "tracing_files": int(self.tracing_files),
            "metadata_present": bool(self.metadata_present),
            "purpose": self.purpose,
            "note": self.note,
            "cohort_version": CODE15_COHORT_VERSION,
        }


_PURPOSES = {
    "code15": (
        "The outpatient reference cohort. Reference ranges computed on a "
        "hospital population are comparison values, not normal values; this "
        "is the population from which a normal value may be taken.",
    ),
    "sami_trop": (
        "A Chagas cardiomyopathy signature in loop terms. Right bundle "
        "branch block with left anterior fascicular block is the classical "
        "description and has never been expressed vectorcardiographically.",
    ),
}


def cohort_status(
    root: Optional[Path | str],
    cohort: str = "code15",
) -> ExternalCohortStatus:
    """Say plainly whether the data is here, without pretending either way."""

    purpose = _PURPOSES.get(cohort, ("Declared external cohort.",))[0]
    if root is None:
        return ExternalCohortStatus(
            cohort=cohort,
            root=None,
            present=False,
            tracing_files=0,
            metadata_present=False,
            purpose=purpose,
            note="No root configured for this cohort.",
        )
    base = Path(root)
    if not base.is_dir():
        return ExternalCohortStatus(
            cohort=cohort,
            root=base,
            present=False,
            tracing_files=0,
            metadata_present=False,
            purpose=purpose,
            note=f"Configured root does not exist: {base}",
        )
    tracings = sorted(base.glob("*.hdf5")) + sorted(base.glob("*.h5"))
    metadata = (base / "exams.csv").is_file()
    return ExternalCohortStatus(
        cohort=cohort,
        root=base,
        present=bool(tracings),
        tracing_files=len(tracings),
        metadata_present=metadata,
        purpose=purpose,
        note=(
            "Present."
            if tracings
            else "Root exists but holds no HDF5 tracing file."
        ),
    )


def audit_cohort_release(
    root: Optional[Path | str],
    cohort: str,
) -> dict[str, Any]:
    """Validate metadata-to-HDF coverage without loading waveform arrays."""

    name = str(cohort).strip().casefold()
    required_columns = _COHORT_REQUIRED_METADATA.get(name)
    if required_columns is None:
        raise ValueError(f"Unknown external cohort: {cohort}")
    empty = {
        "cohort": name,
        "metadata_rows": 0,
        "tracing_rows": 0,
        "columns": (),
        "missing_columns": required_columns,
        "source_paths": (),
    }
    if root is None:
        return {
            **empty,
            "status": "unconfigured",
            "diagnostic": "cohort root is not configured",
        }
    base = Path(root)
    metadata_path = base / "exams.csv"
    if not metadata_path.is_file():
        return {
            **empty,
            "status": "error",
            "diagnostic": f"metadata missing: {metadata_path}",
        }
    try:
        metadata = pd.read_csv(metadata_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return {
            **empty,
            "status": "error",
            "source_paths": (str(metadata_path.resolve()),),
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    columns = tuple(str(column) for column in metadata.columns)
    missing_columns = tuple(
        column for column in required_columns if column not in metadata
    )
    errors: list[str] = []
    if missing_columns:
        errors.append("missing columns: " + ", ".join(missing_columns))
    if "exam_id" in metadata:
        exam_ids = pd.to_numeric(metadata["exam_id"], errors="coerce")
        if exam_ids.isna().any():
            errors.append("metadata contains missing/non-numeric exam_id")
        if exam_ids.duplicated().any():
            errors.append("metadata contains duplicate exam_id")

    source_paths = [metadata_path.resolve()]
    tracing_rows = 0
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency gate
        errors.append(f"h5py unavailable: {exc}")
        h5py = None

    if h5py is not None and not missing_columns:
        if name == "code15":
            referenced = tuple(
                sorted(metadata["trace_file"].dropna().astype(str).unique())
            )
            if not referenced:
                errors.append("metadata references no tracing files")
            for file_name in referenced:
                path = (base / file_name).resolve()
                if not path.is_file():
                    errors.append(f"referenced tracing file missing: {file_name}")
                    continue
                source_paths.append(path)
                with h5py.File(path, "r") as handle:
                    if "tracings" not in handle or "exam_id" not in handle:
                        errors.append(f"{file_name}: required HDF datasets missing")
                        continue
                    shape = tuple(handle["tracings"].shape)
                    ids = np.asarray(handle["exam_id"][:]).reshape(-1)
                if len(shape) != 3 or shape[1:] != (
                    CODE15_SAMPLES_PER_RECORD,
                    len(CODE15_LEAD_ORDER),
                ):
                    errors.append(f"{file_name}: invalid tracing shape {shape}")
                    continue
                if shape[0] != len(ids):
                    errors.append(f"{file_name}: tracing/exam_id length mismatch")
                    continue
                expected = set(
                    pd.to_numeric(
                        metadata.loc[
                            metadata["trace_file"].astype(str).eq(file_name),
                            "exam_id",
                        ],
                        errors="coerce",
                    ).dropna().astype(np.int64)
                )
                observed = set(np.asarray(ids, dtype=np.int64))
                missing_ids = expected - observed
                unexpected_ids = observed - expected
                if missing_ids:
                    errors.append(
                        f"{file_name}: {len(missing_ids)} metadata exam IDs missing"
                    )
                # The published CODE files carry one zero-ID sentinel row.
                if unexpected_ids not in (set(), {0}):
                    errors.append(f"{file_name}: unexpected HDF exam IDs present")
                tracing_rows += len(expected)
        else:
            path = (base / "exams.hdf5").resolve()
            if not path.is_file():
                errors.append("referenced tracing file missing: exams.hdf5")
            else:
                source_paths.append(path)
                with h5py.File(path, "r") as handle:
                    if "tracings" not in handle:
                        errors.append("exams.hdf5: tracings dataset missing")
                        shape = ()
                    else:
                        shape = tuple(handle["tracings"].shape)
                expected_shape = (
                    len(metadata),
                    CODE15_SAMPLES_PER_RECORD,
                    len(CODE15_LEAD_ORDER),
                )
                if shape != expected_shape:
                    errors.append(
                        f"exams.hdf5: expected {expected_shape}, got {shape}"
                    )
                else:
                    tracing_rows = len(metadata)

    return {
        "cohort": name,
        "status": "error" if errors else "ok",
        "metadata_rows": int(len(metadata)),
        "tracing_rows": int(tracing_rows),
        "columns": columns,
        "missing_columns": missing_columns,
        "source_paths": tuple(str(path) for path in dict.fromkeys(source_paths)),
        "diagnostic": "; ".join(errors) if errors else (
            f"{len(metadata):,}/{len(metadata):,} metadata ECGs mapped to HDF5"
        ),
    }


# The two checks that decide whether the external path computes what it claims.
# Stated as data rather than as prose so a stage can evaluate them.
LOADER_ACCEPTANCE_CHECKS: tuple[dict, ...] = (
    {
        "check_id": "spatial_qrs_t_angle_lower_than_hospital",
        "marker_id": "spatial_qrs_t_angle_deg",
        "statistic": "median",
        "comparison": "less_than",
        "reference_value": 74.4,
        "minimum_records": 100,
        "reference_source": "MIMIC development median, Kors",
        "literature_expectation": "44-65 degrees in healthy adults with Kors",
        "reason": (
            "An outpatient cohort must show a markedly narrower spatial "
            "QRS-T angle than a hospital one. If it does not, the loader, "
            "the resampling or the lead order is wrong and no result from "
            "this cohort may be used."
        ),
    },
    {
        "check_id": "rate_qt_relationship_reproduced",
        "marker_id": "qt_interval_ms",
        "statistic": "spearman_with_heart_rate",
        "comparison": "close_to",
        "reference_value": -0.45,
        "absolute_tolerance": 0.30,
        "minimum_records": 100,
        "reference_source": "MIMIC development, same markers",
        "literature_expectation": "QT shortens as rate rises",
        "reason": (
            "The cheapest possible check that the external path computes at "
            "all: a relationship known since 1920 must reappear with the "
            "same sign and order of magnitude."
        ),
    },
)


__all__ = [
    "CODE15_COHORT_VERSION",
    "CODE15_LEAD_ORDER",
    "CODE15_SAMPLES_PER_RECORD",
    "CODE15_SAMPLING_RATE_HZ",
    "LOADER_ACCEPTANCE_CHECKS",
    "MINIMUM_TRIMMED_SECONDS",
    "ArrayTracingSource",
    "Code15Record",
    "ExternalCohortStatus",
    "Hdf5TracingSource",
    "TracingSource",
    "assert_lead_order",
    "audit_cohort_release",
    "cohort_status",
    "read_exam_metadata",
    "read_records",
    "resample_to",
    "trim_zero_padding",
]
