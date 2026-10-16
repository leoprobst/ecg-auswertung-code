"""What a vectorcardiographic measurement usually is, and for whom.

The stated purpose of this project is a clinical reporting guideline for
vectorcardiography. A guideline needs six things, and the audit of 2026-08-29
found the project had two of them: repeatability and declared limitations.
The missing one that everything else waits on is this -- a reference range per
marker, per age, per sex, per rate band, with the number of people behind it.
Without it a marker has an AUC and no value a clinician can read off a report.

``core/analysis/normative_ranges.py`` exists and is not this. It is
pickle-backed, has a minimum stratum size of ten, and is called from nowhere in
the MIMIC pipeline; it belongs to the single-recording desktop application. No
stage of the study run produces a reference range.

**The honest name for what this computes.**  These are *not* normal values.
The population is patients in a large tertiary hospital who happen to carry no
recorded cardiac diagnosis, no cardiotropic drug and no intensive-care episode
around the recording. They are still people who came to a hospital and had an
electrocardiogram taken, which is a reason in itself. Every table this stage
writes carries ``population = hospital_without_recorded_cardiac_disease`` in a
column, and the reporting layer is required to print it, because a percentile
quoted without its population is the single most misusable number a project
like this can emit.

The population that *can* carry normal values is an outpatient one. CODE-15 %
-- 345.779 recordings from a Brazilian telehealth network, with age, sex and
mortality -- is exactly that, and the loader for it is the reason Wave 5 exists.
When it is present this stage runs against it unchanged and the population
column changes; until then the ranges are labelled for what they are.

**Four decisions that are not obvious.**

*One recording per subject.* MIMIC holds 800.035 recordings from 161.352
patients, and a patient with forty recordings would otherwise contribute forty
times to a percentile. The first admissible recording per subject is taken.

*Train split only.* A reference range is a preprocessing decision, and fitting
it on validation would let the held-out half shape what "normal" means before
it is tested.

*Strata below 200 subjects are declared unestimable, not reported.* A 2,5th
percentile from sixty people is an order statistic with a confidence interval
wider than the range it is describing. The row is written with
``estimable = false`` and its reason, so the gap is visible rather than absent.

*The transform travels with the range.* Two thirds of the VCG markers change
when Kors is exchanged for inverse Dower -- for ``st_loop_area_best_fit_plane``
the two agree at rho = 0,70. A threshold without its transform is not a
threshold, so the transform is a column here and a required field in the
reporting set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import re

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .panels import STUDY_PANEL_IDS
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    development_feature_glob,
    parquet_columns,
    scientific_config_hash,
)


REFERENCE_RANGE_STAGE_VERSION = "1.1.0"
NORMATIVE_COHORT_VERSION = "hospital-without-recorded-cardiac-disease-1.1.0"

# The label that must accompany every number this stage produces.
POPULATION_LABEL = "hospital_without_recorded_cardiac_disease"

# Below this many subjects a percentile is an order statistic with a
# confidence interval wider than the interval it describes.
MINIMUM_STRATUM_SUBJECTS = 200

# Percentiles reported. The pair 2,5/97,5 is the conventional reference
# interval; the median is what a report prints beside a measured value.
REFERENCE_PERCENTILES: tuple[float, ...] = (2.5, 50.0, 97.5)

# Age decades. Below 18 the electrocardiogram is a different measurement and
# MIMIC-IV holds no paediatric records anyway; above 89 the age is censored by
# de-identification and is carried as one open band.
AGE_BAND_EDGES: tuple[float, ...] = (18.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)

# Rate bands. The QT interval alone changes by about a third across this
# range, so a reference interval that does not condition on rate is describing
# the rate distribution of the cohort as much as the marker.
RATE_BAND_EDGES: tuple[float, ...] = (30.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 200.0)

# Diagnoses that disqualify a recording from the reference cohort. Cardiac
# disease of any kind, plus the systemic states that move the
# electrocardiogram as reliably as cardiac disease does.
EXCLUDING_DIAGNOSIS_COLUMNS: tuple[str, ...] = (
    "dx_acute_mi",
    "dx_old_mi",
    "dx_ischemic_heart_disease",
    "dx_heart_failure",
    "dx_atrial_fibrillation_flutter",
    "dx_ventricular_tachycardia",
    "dx_ventricular_fibrillation",
    "dx_cardiac_arrest",
    "dx_cardiomyopathy",
    "dx_hypertrophic_cardiomyopathy",
    "dx_dilated_cardiomyopathy",
    "dx_pericarditis",
    "dx_myocarditis",
    "dx_endocarditis",
    "dx_aortic_valve_disease",
    "dx_mitral_valve_disease",
    "dx_tricuspid_valve_disease",
    "dx_pulmonary_hypertension",
    "dx_pulmonary_embolism",
    # Audit X9: hypertension and diabetes remodel the ventricle and move the
    # voltage and repolarisation markers; a reference cohort keeps them out.
    "dx_hypertensive_heart_disease",
    "dx_essential_hypertension",
    "dx_diabetes_mellitus",
    "dx_electrolyte_disorder",
    "dx_chronic_kidney_disease",
    "dx_acute_renal_failure",
    "dx_sepsis",
    "dx_shock",
    "dx_hypoxia_respiratory_failure",
)

# Drugs with a direct electrocardiographic effect. Anticoagulants and
# antiplatelets are deliberately not here: they mark cardiovascular risk
# without changing the tracing, and excluding them would narrow the cohort
# without making it more normal.
EXCLUDING_MEDICATION_COLUMNS: tuple[str, ...] = (
    "med_qt_prolonging_drug",
    "med_antiarrhythmic_class_i_iii",
    "med_beta_blocker",
    "med_calcium_channel_blocker",
    "med_digoxin",
    "med_vasopressor",
    "med_adenosine",
)

# A device paces or senses the tracing outright.
EXCLUDING_DEVICE_COLUMNS: tuple[str, ...] = ("proc_pacemaker_or_icd",)


@dataclass(frozen=True)
class NormativeCohortPolicy:
    """Who counts as a reference subject, declared before anything is read."""

    minimum_signal_quality: float = 0.80
    exclude_intensive_care: bool = True
    exclude_inpatient_admission: bool = False
    minimum_stratum_subjects: int = MINIMUM_STRATUM_SUBJECTS
    bootstrap_iterations: int = 500
    percentiles: tuple[float, ...] = REFERENCE_PERCENTILES
    age_band_edges: tuple[float, ...] = AGE_BAND_EDGES
    rate_band_edges: tuple[float, ...] = RATE_BAND_EDGES
    diagnosis_columns: tuple[str, ...] = EXCLUDING_DIAGNOSIS_COLUMNS
    medication_columns: tuple[str, ...] = EXCLUDING_MEDICATION_COLUMNS
    device_columns: tuple[str, ...] = EXCLUDING_DEVICE_COLUMNS
    random_seed: int = 20260831

    def to_dict(self) -> dict:
        return {
            "minimum_signal_quality": float(self.minimum_signal_quality),
            "exclude_intensive_care": bool(self.exclude_intensive_care),
            "exclude_inpatient_admission": bool(self.exclude_inpatient_admission),
            "minimum_stratum_subjects": int(self.minimum_stratum_subjects),
            "bootstrap_iterations": int(self.bootstrap_iterations),
            "percentiles": list(self.percentiles),
            "age_band_edges": list(self.age_band_edges),
            "rate_band_edges": list(self.rate_band_edges),
            "diagnosis_columns": list(self.diagnosis_columns),
            "medication_columns": list(self.medication_columns),
            "device_columns": list(self.device_columns),
            "random_seed": int(self.random_seed),
            "cohort_version": NORMATIVE_COHORT_VERSION,
        }


def band_label(value: float, edges: Sequence[float], unit: str) -> str:
    """A readable band name, and one open band at each end."""

    if value is None or not np.isfinite(value):
        return "unknown"
    edges = list(edges)
    if value < edges[0]:
        return f"<{edges[0]:g}{unit}"
    for lower, upper in zip(edges[:-1], edges[1:]):
        if lower <= value < upper:
            return f"{lower:g}-{upper:g}{unit}"
    return f">={edges[-1]:g}{unit}"


def assign_strata(
    frame: pd.DataFrame,
    policy: NormativeCohortPolicy,
) -> pd.DataFrame:
    """Add the age, sex and rate band each recording belongs to."""

    result = frame.copy()
    age = pd.to_numeric(result.get("age"), errors="coerce")
    rate = pd.to_numeric(result.get("heart_rate_bpm"), errors="coerce")
    result["age_band"] = [
        band_label(value, policy.age_band_edges, "y") for value in age
    ]
    result["rate_band"] = [
        band_label(value, policy.rate_band_edges, "bpm") for value in rate
    ]
    sex = result.get("sex")
    result["sex_band"] = (
        pd.Series(["unknown"] * len(result), index=result.index)
        if sex is None
        else sex.astype(str).str.upper().str[:1].replace({"": "unknown"})
    )
    return result


def _truthy(series: pd.Series) -> pd.Series:
    """Is this flag set, treating an unknown as not set.

    The evidence table stores the ``dx_`` and ``med_`` flags in pandas'
    nullable boolean dtype, where ``fillna(0)`` raises rather than filling --
    an integer is not a boolean. An absent flag means the diagnosis was not
    recorded, which is what the cohort rule already treats as "not present";
    the distinction between absent and false does not exist in this source.
    """

    if str(series.dtype) in {"boolean", "bool"}:
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & (numeric.fillna(0) > 0)


def apply_cohort_policy(
    context: pd.DataFrame,
    policy: NormativeCohortPolicy,
) -> pd.DataFrame:
    """Mark each recording as admissible for the reference cohort, or say why not.

    Exclusions are recorded rather than filtered, so the table can answer "how
    many were lost to which criterion" -- which is the first question anyone
    reviewing a reference range asks, and the one an already-filtered table
    cannot answer.
    """

    frame = context.copy()
    reasons: list[list[str]] = [[] for _ in range(len(frame))]

    def flagged(columns: Iterable[str]) -> pd.Series:
        present = [name for name in columns if name in frame.columns]
        if not present:
            return pd.Series(False, index=frame.index)
        result = pd.Series(False, index=frame.index)
        for name in present:
            result |= _truthy(frame[name])
        return result

    for label, columns in (
        ("cardiac_or_systemic_diagnosis", policy.diagnosis_columns),
        ("cardiotropic_medication", policy.medication_columns),
        ("implanted_device", policy.device_columns),
    ):
        mask = flagged(columns).to_numpy()
        for index in np.flatnonzero(mask):
            reasons[index].append(label)

    if policy.exclude_intensive_care and "icustay_id" in frame.columns:
        in_icu = frame["icustay_id"].notna().to_numpy()
        for index in np.flatnonzero(in_icu):
            reasons[index].append("intensive_care_episode")

    if policy.exclude_inpatient_admission and "hadm_id" in frame.columns:
        admitted = frame["hadm_id"].notna().to_numpy()
        for index in np.flatnonzero(admitted):
            reasons[index].append("inpatient_admission")

    age = pd.to_numeric(frame.get("age"), errors="coerce")
    too_young = (age.isna() | (age < policy.age_band_edges[0])).to_numpy()
    for index in np.flatnonzero(too_young):
        reasons[index].append("age_outside_bands")

    frame["exclusion_reasons"] = ["|".join(entry) for entry in reasons]
    frame["reference_eligible"] = [not entry for entry in reasons]
    frame["population"] = POPULATION_LABEL
    frame["cohort_version"] = NORMATIVE_COHORT_VERSION
    return frame


def apply_signal_quality_gate(
    frame: pd.DataFrame,
    policy: NormativeCohortPolicy,
) -> pd.DataFrame:
    """Drop recordings whose tracing is not good enough to define a normal.

    Applied after the feature join, not before, because that is where the
    number exists: ``signal_quality_score`` is null for all 477.493 training
    rows of the clinical context table and of the master index -- the score is
    computed during extraction and lives only in the feature partitions.
    Filtering on it in the context step removed the entire cohort while
    reporting a clean-looking hundred per cent exclusion.

    Measured on the store, the gate is mild: 98,0 % of successfully extracted
    recordings reach 0,80 and 90,8 % reach 0,90. It is there to keep a poor
    tracing from widening a reference interval, not to select a cohort.
    """

    if frame.empty or "signal_quality_score" not in frame.columns:
        result = frame.copy()
        result["signal_quality_excluded"] = False
        return result
    quality = pd.to_numeric(frame["signal_quality_score"], errors="coerce")
    poor = quality.isna() | (quality < float(policy.minimum_signal_quality))
    result = frame.copy()
    result["signal_quality_excluded"] = poor.to_numpy()
    return result


def one_recording_per_subject(frame: pd.DataFrame) -> pd.DataFrame:
    """The earliest admissible recording of each subject, and only that one.

    800.035 recordings come from 161.352 patients. A patient with forty
    recordings would otherwise contribute forty times to a percentile, and the
    patients with forty recordings are systematically the sicker ones -- so
    pseudo-replication here does not merely narrow the confidence interval, it
    moves the estimate.
    """

    if frame.empty:
        return frame
    ordered = frame.sort_values(["subject_id", "ecg_time", "study_id"])
    return ordered.drop_duplicates("subject_id", keep="first").reset_index(drop=True)


def _percentile_with_interval(
    values: np.ndarray,
    percentile: float,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = float(np.percentile(values, percentile))
    if iterations <= 0 or values.size < 3:
        return point, float("nan"), float("nan")
    draws = rng.integers(0, values.size, size=(iterations, values.size))
    samples = np.percentile(values[draws], percentile, axis=1)
    return (
        point,
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def reference_ranges_for_markers(
    frame: pd.DataFrame,
    marker_ids: Sequence[str],
    policy: NormativeCohortPolicy,
    *,
    transform: str,
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Percentiles per marker and stratum, with the number of people behind them.

    Every stratum a marker could be reported in gets a row, including the ones
    that are too small. A table of only the estimable strata cannot be told
    apart from a table where nobody looked.
    """

    columns = [
        "marker_id",
        "sex_band",
        "age_band",
        "rate_band",
        "subject_count",
        "observed_count",
        "estimable",
        "unestimable_reason",
        "percentile_2_5",
        "percentile_50",
        "percentile_97_5",
        "percentile_2_5_ci_low",
        "percentile_2_5_ci_high",
        "percentile_97_5_ci_low",
        "percentile_97_5_ci_high",
        "population",
        "vcg_transform",
        "cohort_version",
        "stage_version",
        "literature_comparable",
        "comparability_note",
    ]
    if frame.empty or not marker_ids:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(policy.random_seed)
    grouped = frame.groupby(["sex_band", "age_band", "rate_band"], sort=True)
    rows: list[dict] = []
    total = len(marker_ids) * grouped.ngroups
    done = 0
    for (sex, age_band, rate_band), block in grouped:
        subject_count = int(block["subject_id"].nunique())
        enough = subject_count >= policy.minimum_stratum_subjects
        for marker_id in marker_ids:
            done += 1
            if progress and done % 20000 == 0:
                progress(f"Referenzbereiche {done:,}/{total:,}")
            series = pd.to_numeric(block.get(marker_id), errors="coerce")
            values = series.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row = {
                "marker_id": marker_id,
                "sex_band": sex,
                "age_band": age_band,
                "rate_band": rate_band,
                "subject_count": subject_count,
                "observed_count": int(values.size),
                "population": POPULATION_LABEL,
                "vcg_transform": str(transform),
                "cohort_version": NORMATIVE_COHORT_VERSION,
                "stage_version": REFERENCE_RANGE_STAGE_VERSION,
                **reference_comparability(marker_id),
            }
            reasons = []
            if not enough:
                reasons.append("stratum_below_minimum_subjects")
            if values.size < policy.minimum_stratum_subjects:
                reasons.append("marker_missing_in_too_many_of_the_stratum")
            if reasons:
                row.update(
                    {
                        "estimable": False,
                        "unestimable_reason": "|".join(reasons),
                        "percentile_2_5": float("nan"),
                        "percentile_50": float("nan"),
                        "percentile_97_5": float("nan"),
                        "percentile_2_5_ci_low": float("nan"),
                        "percentile_2_5_ci_high": float("nan"),
                        "percentile_97_5_ci_low": float("nan"),
                        "percentile_97_5_ci_high": float("nan"),
                    }
                )
                rows.append(row)
                continue
            low, low_ci_low, low_ci_high = _percentile_with_interval(
                values, 2.5, iterations=policy.bootstrap_iterations, rng=rng
            )
            median = float(np.percentile(values, 50.0))
            high, high_ci_low, high_ci_high = _percentile_with_interval(
                values, 97.5, iterations=policy.bootstrap_iterations, rng=rng
            )
            row.update(
                {
                    "estimable": True,
                    "unestimable_reason": "",
                    "percentile_2_5": low,
                    "percentile_50": median,
                    "percentile_97_5": high,
                    "percentile_2_5_ci_low": low_ci_low,
                    "percentile_2_5_ci_high": low_ci_high,
                    "percentile_97_5_ci_low": high_ci_low,
                    "percentile_97_5_ci_high": high_ci_high,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


_INTERVAL_MARKER_PATTERN = re.compile(
    r"_ms$|_interval|duration|qtc|jtc|tpeak_tend|pr_prolong|^p_duration",
    re.IGNORECASE,
)


def reference_comparability(marker_id: str) -> dict:
    """Mandatory report field: what a range may be compared with (audit X9).

    Interval ranges inherit the delineation's boundary bias (LUDB against
    experts: P duration about -25 ms, QT about -25 ms, QRS -7.5 ms), so they
    are internally consistent but not comparable with published normal values.
    VCG ranges come from the Kors reconstruction, not from Frank leads, and
    are comparable only with other Kors-derived ranges.
    """
    if _INTERVAL_MARKER_PATTERN.search(str(marker_id)):
        return {
            "literature_comparable": False,
            "comparability_note": (
                "interval_from_own_delineation_ludb_bias_p_duration_minus_25ms_"
                "qt_minus_25ms_qrs_minus_7ms_not_comparable_with_published_norms"
            ),
        }
    return {
        "literature_comparable": False,
        "comparability_note": (
            "kors_reconstructed_vcg_or_amplitude_marker_comparable_only_with_"
            "kors_derived_ranges_not_frank_lead_norms"
        ),
    }


def cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """How many recordings each criterion removed, one row per criterion."""

    if frame.empty:
        return pd.DataFrame(
            columns=["criterion", "excluded_ecgs", "excluded_subjects", "share"]
        )
    total = len(frame)
    reasons = frame["exclusion_reasons"].fillna("")
    names: set[str] = set()
    for entry in reasons:
        names.update(part for part in str(entry).split("|") if part)
    rows = [
        {
            "criterion": "considered",
            "excluded_ecgs": total,
            "excluded_subjects": int(frame["subject_id"].nunique()),
            "share": 1.0,
        }
    ]
    for name in sorted(names):
        mask = reasons.str.contains(name, regex=False)
        rows.append(
            {
                "criterion": name,
                "excluded_ecgs": int(mask.sum()),
                "excluded_subjects": int(frame.loc[mask, "subject_id"].nunique()),
                "share": float(mask.mean()),
            }
        )
    eligible = frame["reference_eligible"].astype(bool)
    rows.append(
        {
            "criterion": "reference_eligible",
            "excluded_ecgs": int(eligible.sum()),
            "excluded_subjects": int(frame.loc[eligible, "subject_id"].nunique()),
            "share": float(eligible.mean()),
        }
    )
    return pd.DataFrame(rows)


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


# Markers read at a time when the percentiles are computed. 76.977 subjects
# times 1.802 markers is a gigabyte in float64 and two while pandas
# concatenates -- measured, as a MemoryError on the first full run. Two
# hundred columns of float32 is about sixty megabytes.
MARKER_BLOCK = 200


def _write_cohort_features(
    feature_glob: str,
    marker_ids: Sequence[str],
    study_ids: set[int],
    destination: Path,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[Path, tuple[str, ...], int]:
    """Stream the cohort's marker values into one compact parquet.

    The obvious implementations both fail on the real store. A DuckDB join of
    77.000 study ids against 16 GB of partitions materialises most of the
    store; accumulating the partitions in memory and concatenating them at the
    end raised MemoryError trying to allocate a 1016 MiB block, and would have
    needed twice that to finish.

    Written once as float32 and read back in column blocks, the peak is the
    block rather than the cohort. float32 keeps seven significant digits,
    which a percentile does not exhaust.
    """

    import glob as _glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    pattern = str(feature_glob).replace("\\", "/")
    paths = sorted(_glob.glob(pattern.replace("split=*", "split=train")))
    if not paths:
        paths = sorted(_glob.glob(pattern))
    if not paths:
        return destination, tuple(), 0

    available = set(pq.ParquetFile(paths[0]).schema_arrow.names)
    present = [name for name in marker_ids if name in available]
    extras = [
        name
        for name in ("signal_quality_score", "context_heart_rate_bpm")
        if name in available and name not in present
    ]
    columns = ["study_id", "extraction_status"] + present + extras
    columns = [name for name in dict.fromkeys(columns) if name in available]

    schema = pa.schema(
        [pa.field("study_id", pa.int64())]
        + [pa.field(name, pa.float32()) for name in present + extras]
    )
    written = 0
    writer = pq.ParquetWriter(str(destination), schema)
    try:
        for index, path in enumerate(paths):
            frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
            if "extraction_status" in frame.columns:
                frame = frame[frame["extraction_status"] == "ok"]
                frame = frame.drop(columns=["extraction_status"])
            frame = frame[frame["study_id"].astype("int64").isin(study_ids)]
            if frame.empty:
                continue
            payload = {"study_id": frame["study_id"].astype("int64")}
            for name in present + extras:
                payload[name] = pd.to_numeric(
                    frame.get(name), errors="coerce"
                ).astype("float32")
            writer.write_table(
                pa.Table.from_pandas(
                    pd.DataFrame(payload), schema=schema, preserve_index=False
                )
            )
            written += len(frame)
            if progress and (index + 1) % 300 == 0:
                progress(
                    f"Referenzkohorte: Partition {index + 1:,}/{len(paths):,}"
                )
    finally:
        writer.close()
    return destination, tuple(present), written


def _ranges_in_blocks(
    cohort_path: Path,
    strata: pd.DataFrame,
    marker_ids: Sequence[str],
    policy: NormativeCohortPolicy,
    *,
    transform: str,
    block_size: int = MARKER_BLOCK,
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Compute the percentiles a block of markers at a time."""

    import pyarrow.parquet as pq

    if not marker_ids or strata.empty:
        return reference_ranges_for_markers(
            pd.DataFrame(), [], policy, transform=transform
        )
    keys = strata[
        ["study_id", "subject_id", "sex_band", "age_band", "rate_band"]
    ].copy()
    keys["study_id"] = keys["study_id"].astype("int64")
    from .grant_parallel import grant_parallel_map

    del pq
    blocks = [
        list(marker_ids[start_index : start_index + block_size])
        for start_index in range(0, len(marker_ids), block_size)
    ]
    if progress:
        progress(f"Referenzbereiche: {len(blocks)} Markerbloecke")
    # Audit R15: every block seeds its own generator from the policy, so the
    # blocks are independent and come back in block order.
    tables = grant_parallel_map(
        _range_block,
        [(str(cohort_path), keys, block, policy, transform) for block in blocks],
    )
    return pd.concat(tables, ignore_index=True)


def _range_block(
    cohort_path: str,
    keys: pd.DataFrame,
    block: list[str],
    policy: NormativeCohortPolicy,
    transform: str,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    values = pq.read_table(cohort_path, columns=["study_id"] + block).to_pandas()
    values["study_id"] = values["study_id"].astype("int64")
    merged = keys.merge(values, on="study_id", how="inner")
    return reference_ranges_for_markers(merged, block, policy, transform=transform)


def run_reference_ranges(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    marker_ids: Optional[Sequence[str]] = None,
    policy: NormativeCohortPolicy = NormativeCohortPolicy(),
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Build the reference table the reporting guideline is missing."""

    output_root = config.runs_root / run_id / "reference_ranges"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "reference_ranges.manifest.json"

    from .panels import default_feature_panels

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    unknown = sorted(set(panel_ids) - set(panels))
    if unknown:
        raise ValueError(f"Unknown feature panels: {unknown}")
    if marker_ids is None:
        marker_ids = tuple(
            dict.fromkeys(
                str(feature_id)
                for panel_id in panel_ids
                for feature_id in panels[panel_id].feature_ids
                if not str(feature_id).endswith("_positive")
            )
        )
    marker_ids = tuple(marker_ids)

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": list(marker_ids),
            "policy": policy.to_dict(),
            "transform": str(config.primary_vcg_method),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash, config_hash, REFERENCE_RANGE_STAGE_VERSION
        ):
            return previous

    context_path = config.evidence_root / "clinical_context_features.parquet"
    if not context_path.is_file():
        raise RuntimeError(
            "The reference cohort needs the clinical context table: "
            f"{context_path}"
        )

    if progress:
        progress("Referenzkohorte: klinischen Kontext lesen")

    context_columns = [
        "subject_id",
        "study_id",
        "ecg_time",
        "split",
        "age",
        "sex",
        "race",
        "hadm_id",
        "icustay_id",
        "signal_quality_score",
        "machine_heart_rate_bpm",
        *policy.diagnosis_columns,
        *policy.medication_columns,
        *policy.device_columns,
    ]
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=min(8.0, float(config.duckdb_memory_limit_gb)),
        threads=max(1, int(config.workers)),
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        available = parquet_columns(connection, _sql_path(context_path))
        selected = [name for name in context_columns if name in available]
        context = connection.execute(
            "SELECT "
            + ", ".join(f'"{name}"' for name in selected)
            + f" FROM read_parquet('{_sql_path(context_path)}')"
            " WHERE split = 'train'"
        ).fetchdf()

    if progress:
        progress(f"Referenzkohorte: {len(context):,} Trainings-EKGs geprueft")

    marked = apply_cohort_policy(context, policy)
    summary = cohort_summary(marked)
    eligible = marked[marked["reference_eligible"].astype(bool)]
    eligible = one_recording_per_subject(eligible)

    if progress:
        progress(
            f"Referenzkohorte: {len(eligible):,} Subjekte, "
            f"{len(marker_ids):,} Marker"
        )

    cohort_path = output_root / "cohort_marker_values.parquet"
    written = 0
    if not eligible.empty:
        cohort_path, marker_ids, written = _write_cohort_features(
            resolved_glob,
            marker_ids,
            set(eligible["study_id"].astype("int64").tolist()),
            cohort_path,
            progress=progress,
        )
    features = pd.DataFrame()
    if written:
        import pyarrow.parquet as pq

        context_columns = [
            name
            for name in ("study_id", "signal_quality_score", "context_heart_rate_bpm")
            if name in set(pq.ParquetFile(cohort_path).schema_arrow.names)
        ]
        features = pq.read_table(
            str(cohort_path), columns=context_columns
        ).to_pandas()

    if features.empty:
        joined = eligible.assign(
            heart_rate_bpm=np.nan, signal_quality_excluded=False
        )
    else:
        joined = eligible.merge(
            features, on="study_id", how="inner", suffixes=("", "_feature")
        )
        # The record's own rate first: it is the rate at which this tracing
        # was taken, measured from the same beats the markers were. The
        # machine report is the fallback for a store extracted before the
        # context column existed.
        rate_source = (
            "context_heart_rate_bpm"
            if joined.get("context_heart_rate_bpm") is not None
            and joined["context_heart_rate_bpm"].notna().any()
            else "machine_heart_rate_bpm"
        )
        joined["heart_rate_bpm"] = pd.to_numeric(
            joined.get(rate_source), errors="coerce"
        )
        joined["rate_source"] = rate_source
        if "signal_quality_score_feature" in joined.columns:
            joined["signal_quality_score"] = joined[
                "signal_quality_score_feature"
            ]
        joined = apply_signal_quality_gate(joined, policy)
        quality_excluded = int(joined["signal_quality_excluded"].sum())
        joined = joined[~joined["signal_quality_excluded"].astype(bool)]
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        {
                            "criterion": "signal_quality_below_threshold",
                            "excluded_ecgs": quality_excluded,
                            "excluded_subjects": quality_excluded,
                            "share": (
                                quality_excluded / len(eligible)
                                if len(eligible)
                                else 0.0
                            ),
                        },
                        {
                            "criterion": "reference_subjects_final",
                            "excluded_ecgs": int(len(joined)),
                            "excluded_subjects": int(
                                joined["subject_id"].nunique()
                            ),
                            "share": (
                                len(joined) / len(eligible)
                                if len(eligible)
                                else 0.0
                            ),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
    joined = assign_strata(joined, policy)

    ranges = _ranges_in_blocks(
        cohort_path,
        joined,
        marker_ids,
        policy,
        transform=str(config.primary_vcg_method),
        progress=progress,
    )

    cohort_path = output_root / "normative_cohort.parquet"
    summary_path = output_root / "cohort_summary.parquet"
    ranges_path = output_root / "marker_reference_ranges.parquet"
    policy_path = output_root / "cohort_policy.json"
    cohort_columns = [
        name
        for name in (
            "subject_id",
            "study_id",
            "ecg_time",
            "split",
            "age",
            "sex",
            "race",
            "age_band",
            "sex_band",
            "rate_band",
            "heart_rate_bpm",
            "population",
            "cohort_version",
        )
        if name in joined.columns
    ]
    atomic_write_parquet(joined[cohort_columns], cohort_path)
    atomic_write_parquet(summary, summary_path)
    atomic_write_parquet(ranges, ranges_path)
    atomic_write_json(
        policy_path,
        {
            "policy": policy.to_dict(),
            "population": POPULATION_LABEL,
            "population_note": (
                "Not normal values. Hospital patients without a recorded "
                "cardiac diagnosis, cardiotropic drug, device or intensive "
                "care episode. An outpatient cohort is required before these "
                "may be called reference values."
            ),
            "vcg_transform": str(config.primary_vcg_method),
        },
    )

    estimable = (
        int(ranges["estimable"].astype(bool).sum()) if not ranges.empty else 0
    )
    manifest = StageManifest.complete(
        stage="reference-ranges",
        stage_version=REFERENCE_RANGE_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(cohort_path.resolve()),
            str(summary_path.resolve()),
            str(ranges_path.resolve()),
            str(policy_path.resolve()),
        ),
        row_counts={
            "considered_ecgs": int(len(marked)),
            "reference_subjects": int(len(joined)),
            "markers": int(len(marker_ids)),
            "stratum_marker_rows": int(len(ranges)),
            "estimable_rows": estimable,
        },
        notes=(
            f"population={POPULATION_LABEL}",
            f"cohort_version={NORMATIVE_COHORT_VERSION}",
            f"vcg_transform={config.primary_vcg_method}",
            "these_are_not_normal_values",
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "AGE_BAND_EDGES",
    "EXCLUDING_DEVICE_COLUMNS",
    "EXCLUDING_DIAGNOSIS_COLUMNS",
    "EXCLUDING_MEDICATION_COLUMNS",
    "MINIMUM_STRATUM_SUBJECTS",
    "NORMATIVE_COHORT_VERSION",
    "NormativeCohortPolicy",
    "POPULATION_LABEL",
    "RATE_BAND_EDGES",
    "REFERENCE_PERCENTILES",
    "REFERENCE_RANGE_STAGE_VERSION",
    "apply_cohort_policy",
    "apply_signal_quality_gate",
    "assign_strata",
    "band_label",
    "cohort_summary",
    "one_recording_per_subject",
    "reference_ranges_for_markers",
    "run_reference_ranges",
]
