"""CODE-15% and SaMi-Trop extraction and cohort-specific analyses.

The two cohorts are never pooled with MIMIC. CODE-15 supplies a large
primary-care ECG-reference distribution and independently read ECG labels;
SaMi-Trop supplies prospective mortality follow-up in seropositive Chagas
cardiomyopathy. Cross-cohort contrasts are explicitly site-confounded.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path
import threading
from typing import Callable, Iterable, Optional, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..core.biomarkers import BiomarkerRegistry
from ..core.delineation.base import ventricular_rr_seconds
from ..core.delineation.fusion_delineator import FusionDelineator
from ..core.preprocessing import (
    construct_robust_morphology_record,
    detect_r_peaks_multilead,
)
from ..core.vcg_transform import VCGTransformRegistry
from .code15_cohort import (
    ArrayTracingSource,
    CODE15_COHORT_VERSION,
    CODE15_SAMPLING_RATE_HZ,
    Hdf5TracingSource,
    LOADER_ACCEPTANCE_CHECKS,
    TracingSource,
    audit_cohort_release,
    read_exam_metadata,
    read_records,
)
from .classical import ClassicalECGCriteria
from .config import MimicProjectConfig
from .feature_extraction import feature_implementation_hash
from .panels import ALL_CLASSICAL_FEATURE_IDS, default_feature_panels
from .signals import (
    assess_signal_quality,
    compute_vcg_sensitivity_transforms,
    prepare_signal_paths,
)
from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    file_sha256,
    optional_file_content_fingerprint,
    output_file_integrity,
    verify_output_integrity,
)
from .survival import CoxFit, ConvergenceFailure, concordance_index, fit_cox


POPULATION_COHORT_STAGE_VERSION = "1.15.0"
POPULATION_ANALYSIS_SOURCES = (
    "population_cohorts.py", "survival.py", "absolute_risk.py",
    "population_multiplicity.py", "patient_identity.py", "code15_cohort.py",
)


def _population_analysis_source_hash() -> str:
    """Bind derived analyses separately from reusable raw feature partitions."""
    root = Path(__file__).resolve().parent
    return content_hash({name: file_sha256(root / name) for name in POPULATION_ANALYSIS_SOURCES})


def _population_extraction_source_hash() -> str:
    """Bind local extraction functions, imports and external signal code."""
    import ast
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_stable_fraction", "_thread_delineator", "_initialize_external_worker",
             "_analyze_external_worker_item", "_analyze_record", "_extract_cohort"}
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not names <= set(definitions):
        raise RuntimeError("External extraction source contract incomplete")
    # Follow statically referenced local helpers and callbacks transitively.
    # Adding a helper must not require remembering a second manual hash list.
    pending = list(names)
    while pending:
        node = definitions[pending.pop()]
        referenced = {child.id for child in ast.walk(node)
                      if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        for name in referenced & definitions.keys() - names:
            names.add(name)
            pending.append(name)
    functions = {name: ast.dump(definitions[name], include_attributes=False) for name in sorted(names)}
    imports = [ast.dump(node, include_attributes=False) for node in tree.body
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    return content_hash(dict(functions=functions, imports=imports,
        loader=file_sha256(Path(__file__).with_name("code15_cohort.py")),
        feature_code=feature_implementation_hash(),
        checkpoint_version=EXTERNAL_EXTRACTION_CHECKPOINT_VERSION,
        sampling_version=EXTERNAL_CAPPED_SAMPLING_VERSION,
        transform_angle_marker=TRANSFORM_ANGLE_MARKER,
        partition_size=EXTERNAL_FEATURE_PARTITION_SIZE))


def _population_extraction_input_hash(inventory, cohort, marker_ids, max_records, minimum_quality):
    sources = inventory.loc[inventory.cohort.eq(cohort),
        ["cohort", "path", "size", "mtime_ns", "sha256"]].to_dict(orient="records")
    return content_hash(dict(extraction_code=_population_extraction_source_hash(),
        cohort=cohort, marker_ids=list(marker_ids), sources=sources,
        max_records=max_records, minimum_signal_quality=minimum_quality))


SOURCE_HASH_CACHE_VERSION = "1.0.0"
EXTERNAL_EXTRACTION_CHECKPOINT_VERSION = "1.0.0"
EXTERNAL_CAPPED_SAMPLING_VERSION = "stable_exam_hash_v1"
EXTERNAL_FEATURE_PARTITION_SIZE = 1_000
_ANALYSIS_THREAD_LOCAL = threading.local()
_EXTERNAL_WORKER_MARKER_IDS: tuple[str, ...] = ()
_EXTERNAL_WORKER_REQUESTED_BIOMARKERS: tuple[str, ...] = ()
_EXTERNAL_WORKER_POWERLINE_HZ = 60.0
_EXTERNAL_WORKER_MINIMUM_SIGNAL_QUALITY = 0.0
TRANSFORM_ANGLE_MARKER = "spatial_qrs_t_angle_deg"
DIRECT_ANGLE_MARKERS: tuple[str, ...] = (
    "qrs_t_area_angle_direct_8lead_deg",
    "qrs_t_integral_angle_direct_8lead_deg",
)
DIAGNOSTIC_ENDPOINTS: tuple[str, ...] = (
    "first_degree_av_block",
    "right_bundle_branch_block",
    "left_bundle_branch_block",
    "sinus_bradycardia",
    "sinus_tachycardia",
    "atrial_fibrillation",
    "reported_normal_ecg",
)


@dataclass(frozen=True)
class PopulationCohortResult:
    status: str
    output_paths: tuple[str, ...]
    row_counts: dict[str, int]
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _panel_markers(panel_ids: Sequence[str]) -> tuple[str, ...]:
    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    unknown = sorted(set(panel_ids) - set(panels))
    if unknown:
        raise ValueError(f"Unknown external cohort panels: {unknown}")
    return tuple(
        dict.fromkeys(
            marker
            for panel_id in panel_ids
            for marker in panels[panel_id].feature_ids
        )
    )


def _transportable_validation_markers(
    config: MimicProjectConfig,
    run_id: str,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Outcome-blind external transport of MIMIC-replicated scalar markers.

    Extracting all discovery-reserve formulas over 347k external ECGs would
    spend most of the run on hypotheses that did not replicate in MIMIC.  The
    validation criterion catalog already supplies a prespecified, locked-test
    eligible shortlist before CODE-15 or SaMi-Trop outcomes are inspected.
    Transporting that shortlist in addition to Core/Sentinel captures the
    useful discovery markers without selecting on either external cohort.
    """

    path = config.runs_root / str(run_id) / "criterion_candidates.parquet"
    provenance: dict[str, object] = {
        "path": str(path.resolve()),
        "fingerprint": optional_file_content_fingerprint(path),
        "selection": (
            "validation_replicated_and_locked_test_eligible_before_external_read"
        ),
    }
    if not path.is_file():
        provenance["status"] = "criterion_catalog_not_available"
        return (), provenance
    frame = pd.read_parquet(path)
    required = {"feature_id", "stage", "locked_test_eligible"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "External marker transport requires criterion columns: "
            + ", ".join(missing)
        )
    eligible = (
        frame["stage"].astype(str).eq("validation_replicated")
        & frame["locked_test_eligible"].fillna(False).eq(True)
    )
    registered = set(BiomarkerRegistry.available())
    selected = tuple(
        sorted(
            {
                str(value)
                for value in frame.loc[eligible, "feature_id"].dropna()
                if str(value) in registered
            }
        )
    )
    provenance.update(
        {
            "status": "selected",
            "eligible_criterion_rows": int(eligible.sum()),
            "transported_scalar_marker_count": len(selected),
            "transported_scalar_markers": list(selected),
        }
    )
    return selected, provenance


def _stable_fraction(value: object) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _source_inventory(
    root: Path,
    cohort: str,
    *,
    verify_sha256: bool,
    hash_cache: Optional[dict[str, dict[str, object]]] = None,
) -> pd.DataFrame:
    rows = []
    metadata_path = root / "exams.csv"
    candidates = [metadata_path, *sorted(root.glob("*.hdf5")), *sorted(root.glob("*.h5"))]
    for path in candidates:
        if not path.is_file():
            continue
        stat = path.stat()
        cache_key = str(path.resolve())
        cached = (hash_cache or {}).get(cache_key, {})
        cache_hit = bool(
            verify_sha256
            and cached.get("size") == int(stat.st_size)
            and cached.get("mtime_ns") == int(stat.st_mtime_ns)
            and len(str(cached.get("sha256", ""))) == 64
        )
        digest = (
            str(cached["sha256"])
            if cache_hit
            else file_sha256(path)
            if verify_sha256
            else "not_computed_smoke_run"
        )
        if verify_sha256 and hash_cache is not None and not cache_hit:
            hash_cache[cache_key] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
            }
        rows.append(
            {
                "cohort": cohort,
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                # CSV is small and always hashed. Large HDF5 files are hashed
                # once into this inventory before any clinical result is used.
                "sha256": digest,
                "sha256_verified": bool(verify_sha256),
                "sha256_cache_hit": cache_hit,
            }
        )
    return pd.DataFrame(rows)


def _load_source_hash_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("cache_version") != SOURCE_HASH_CACHE_VERSION:
        return {}
    entries = payload.get("entries", {})
    return dict(entries) if isinstance(entries, dict) else {}


def _save_source_hash_cache(
    path: Path,
    entries: dict[str, dict[str, object]],
) -> None:
    atomic_write_json(
        path,
        {"cache_version": SOURCE_HASH_CACHE_VERSION, "entries": entries},
    )


def _thread_delineator() -> FusionDelineator:
    delineator = getattr(_ANALYSIS_THREAD_LOCAL, "delineator", None)
    if delineator is None:
        delineator = FusionDelineator()
        _ANALYSIS_THREAD_LOCAL.delineator = delineator
    return delineator


def _initialize_external_worker(
    marker_ids: tuple[str, ...],
    requested_biomarkers: tuple[str, ...],
    powerline_hz: float,
    minimum_signal_quality: float,
) -> None:
    """Install immutable cohort settings once in each spawned process."""

    global _EXTERNAL_WORKER_MARKER_IDS
    global _EXTERNAL_WORKER_REQUESTED_BIOMARKERS
    global _EXTERNAL_WORKER_POWERLINE_HZ
    global _EXTERNAL_WORKER_MINIMUM_SIGNAL_QUALITY
    from .resources import apply_worker_thread_limits

    apply_worker_thread_limits()
    _EXTERNAL_WORKER_MARKER_IDS = tuple(marker_ids)
    _EXTERNAL_WORKER_REQUESTED_BIOMARKERS = tuple(requested_biomarkers)
    _EXTERNAL_WORKER_POWERLINE_HZ = float(powerline_hz)
    _EXTERNAL_WORKER_MINIMUM_SIGNAL_QUALITY = float(
        minimum_signal_quality
    )


def _analyze_external_worker_item(item) -> dict[str, object]:
    record, metadata = item
    return _analyze_record(
        record,
        metadata,
        _EXTERNAL_WORKER_MARKER_IDS,
        _EXTERNAL_WORKER_REQUESTED_BIOMARKERS,
        powerline_hz=_EXTERNAL_WORKER_POWERLINE_HZ,
        minimum_signal_quality=(
            _EXTERNAL_WORKER_MINIMUM_SIGNAL_QUALITY
        ),
    )


def _analyze_record(
    record,
    metadata: dict,
    marker_ids: tuple[str, ...],
    requested_biomarkers: tuple[str, ...],
    *,
    powerline_hz: float,
    minimum_signal_quality: float,
) -> dict[str, object]:
    base: dict[str, object] = {
        **metadata,
        "exam_id": int(record.exam_id),
        "patient_id": record.patient_id,
        "cohort": record.source,
        "original_sampling_rate_hz": record.original_sampling_rate_hz,
        "analysis_sampling_rate_hz": record.sampling_rate_hz,
        "trimmed_samples": record.trimmed_samples,
        "padded_samples": record.padded_samples,
        "cohort_stage_version": POPULATION_COHORT_STAGE_VERSION,
        "context_temporal_research_json": "",
        "context_measurement_capsule_json": "",
        "context_measurement_sensitivity_json": "",
    }
    try:
        signal = np.asarray(record.signal_millivolts, dtype=float)
        fs = int(round(record.sampling_rate_hz))
        quality = assess_signal_quality(
            signal,
            fs,
            expected_powerline_hz=float(powerline_hz),
            minimum_duration_seconds=6.5,
        )
        base.update(
            {f"signal_quality_{key}": value for key, value in quality.to_dict().items()}
        )
        if not quality.usable or quality.score < float(minimum_signal_quality):
            base["extraction_status"] = "abstained_signal_qc"
            base["failure_type"] = "SignalQualityAbstention"
            base["failure_message"] = "|".join(quality.reasons)[:300]
            return base
        paths = prepare_signal_paths(signal, fs, powerline=float(powerline_hz))
        r_peaks = detect_r_peaks_multilead(paths["delineation"], fs)
        if len(r_peaks) < 2:
            raise ValueError(f"only {len(r_peaks)} R peak(s) detected")
        initial_vcg = VCGTransformRegistry.transform(paths["delineation"], "kors")
        delineation = _thread_delineator().delineate_record(
            ecg_12lead=paths["delineation"],
            r_peaks=r_peaks,
            fs=fs,
            ecg_id=int(record.exam_id),
            vcg_xyz=initial_vcg.xyz,
        )
        if not delineation.beats:
            raise ValueError("no beats could be delineated")
        robust, _ = construct_robust_morphology_record(
            paths["morphology"], delineation
        )
        vcg = VCGTransformRegistry.transform(robust, "kors")
        record_vcg = VCGTransformRegistry.transform(paths["morphology"], "kors")
        from ..core.temporal_research import record_descriptor
        base["context_temporal_research_json"] = json.dumps(record_descriptor(
            delineation, record_vcg.xyz, fs, source=record.source,
            subject_id=record.patient_id, transform="kors",
        ), allow_nan=False, separators=(",", ":"))
        from ..core.measurement_capsule import capsule_json
        from ..core.measurement_robustness import angle_sensitivity
        base["context_measurement_sensitivity_json"] = json.dumps(angle_sensitivity(
            paths["morphology"], delineation, fs, source=record.source, record_id=record.exam_id),
            allow_nan=False, separators=(",", ":"))
        base["context_measurement_capsule_json"] = capsule_json(
            paths["morphology"], delineation, fs, source=record.source,
            record_id=record.exam_id, raw_reference=f"{record.source}:{record.exam_id}",
            processing={"signal_path": "record_morphology", "powerline_hz": float(powerline_hz),
                        "original_sampling_rate_hz": float(record.original_sampling_rate_hz),
                        "baseline": "prepared_signal_path_no_additional_capsule_correction"},
        )
        results = BiomarkerRegistry.compute_all(
            delineation=delineation,
            ecg_12lead=robust,
            vcg_xyz=vcg.xyz,
            fs=fs,
            names=list(requested_biomarkers),
            record_ecg_12lead=paths["morphology"],
            record_raw_ecg_12lead=signal,
            record_acquisition={"original_sampling_rate_hz": float(record.original_sampling_rate_hz)},
            record_vcg_xyz=record_vcg.xyz,
            # The public release documents sampling rate, not validated
            # analogue bandwidth; high-frequency markers must abstain.
            record_vcg_wideband_xyz=None,
        )
        for result in results:
            base[result.name] = result.value
        # Keep an explicit, method-labelled angle for agreement analysis.
        # PCA aligned to Kors is the project's SVD sensitivity construction;
        # it remains sensitivity-only and is never treated as a reference.
        transform_vcgs = compute_vcg_sensitivity_transforms(
            robust,
            methods=("inverse_dower", "pca_kors_aligned"),
            kors_result=vcg,
        )
        kors_angle = base.get(TRANSFORM_ANGLE_MARKER)
        if TRANSFORM_ANGLE_MARKER not in base:
            angle_result = BiomarkerRegistry.compute_all(
                delineation=delineation,
                ecg_12lead=robust,
                vcg_xyz=vcg.xyz,
                fs=fs,
                names=[TRANSFORM_ANGLE_MARKER],
                record_ecg_12lead=paths["morphology"],
                record_raw_ecg_12lead=signal,
                record_acquisition={"original_sampling_rate_hz": float(record.original_sampling_rate_hz)},
                record_vcg_xyz=None,
                record_vcg_wideband_xyz=None,
            )
            kors_angle = angle_result[0].value if angle_result else np.nan
        base[f"transform_kors__{TRANSFORM_ANGLE_MARKER}"] = kors_angle
        for method, transformed in transform_vcgs.items():
            angle_result = BiomarkerRegistry.compute_all(
                delineation=delineation,
                ecg_12lead=robust,
                vcg_xyz=transformed.xyz,
                fs=fs,
                names=[TRANSFORM_ANGLE_MARKER],
                record_ecg_12lead=paths["morphology"],
                record_acquisition={"original_sampling_rate_hz": float(record.original_sampling_rate_hz)},
                record_vcg_xyz=None,
                record_vcg_wideband_xyz=None,
            )
            base[f"transform_{method}__{TRANSFORM_ANGLE_MARKER}"] = (
                angle_result[0].value if angle_result else np.nan
            )
        lead_measurements: dict[str, float] = {}
        classical = ClassicalECGCriteria().compute(
            paths["diagnostic"],
            delineation,
            age=(
                int(metadata["age"])
                if pd.notna(metadata.get("age"))
                else None
            ),
            sex=str(metadata["sex"]) if pd.notna(metadata.get("sex")) else "",
            machine_qrs_axis_deg=None,
            paced_or_lbbb=False,
            measurement_sink=lead_measurements,
        )
        from .lead_measurements import LEAD_MEASUREMENT_VERSION
        base.update(lead_measurements)
        base["lead_measurement_version"] = LEAD_MEASUREMENT_VERSION
        base["lead_measurement_scope"] = "diagnostic_12lead_shared_fiducials"
        for feature_id in marker_ids:
            if feature_id not in ALL_CLASSICAL_FEATURE_IDS:
                continue
            criterion = classical.get(feature_id.removeprefix("classical_"))
            base[feature_id] = (
                criterion.value
                if criterion is not None and criterion.evaluable
                else np.nan
            )
            base[f"{feature_id}_positive"] = (
                criterion.positive
                if criterion is not None and criterion.evaluable
                else None
            )
        rr_seconds = ventricular_rr_seconds(r_peaks, fs)
        base["heart_rate_bpm"] = (
            np.nan if rr_seconds is None else float(60.0 / rr_seconds)
        )
        base["n_beats"] = int(delineation.n_beats)
        base["delineation_quality_score"] = float(
            getattr(delineation, "quality_score", np.nan)
        )
        base["extraction_status"] = "ok"
    except Exception as exc:  # failure taxonomy is part of the artifact
        base["extraction_status"] = "failed"
        base["failure_type"] = type(exc).__name__
        base["failure_message"] = str(exc)[:300]
    return base


def _extract_cohort(
    root: Path,
    cohort: str,
    marker_ids: tuple[str, ...],
    output_root: Path,
    *,
    extraction_input_hash: str,
    max_records: Optional[int],
    workers: int,
    minimum_signal_quality: float,
    force: bool,
    progress: Optional[Callable[[str], None]],
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    full_metadata = read_exam_metadata(root / "exams.csv", cohort=cohort)
    metadata = full_metadata
    if max_records is not None:
        limit = max(0, int(max_records))
        metadata = (
            metadata.assign(
                _stable_sample_key=metadata["exam_id"].map(_stable_fraction)
            )
            .sort_values(
                ["_stable_sample_key", "exam_id"],
                kind="stable",
            )
            .head(limit)
            .drop(columns="_stable_sample_key")
            .copy()
        )
    metadata["cohort"] = cohort
    metadata_lookup = {
        int(row["exam_id"]): row
        for row in metadata.to_dict(orient="records")
    }
    selected_ids = set(metadata_lookup)
    rejected_frames: list[pd.DataFrame] = []
    cohort_output_root = output_root / "features" / extraction_input_hash[:20] / cohort
    checkpoint_path = cohort_output_root / "extraction.checkpoint.json"
    output_paths: list[Path] = []
    checkpoint_integrity: list[dict[str, object]] = []
    completed_ids: set[int] = set()
    if checkpoint_path.is_file() and not force:
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unreadable {cohort} extraction checkpoint: {checkpoint_path}"
            ) from exc
        recorded_integrity = tuple(checkpoint.get("output_integrity", ()))
        if (
            checkpoint.get("checkpoint_version")
            != EXTERNAL_EXTRACTION_CHECKPOINT_VERSION
            or checkpoint.get("extraction_input_hash") != extraction_input_hash
            or checkpoint.get("cohort") != cohort
            or not verify_output_integrity(recorded_integrity)
        ):
            raise RuntimeError(
                f"Stale or damaged {cohort} extraction checkpoint; rerun with force"
            )
        output_paths = [Path(str(item["path"])) for item in recorded_integrity]
        checkpoint_integrity = [dict(item) for item in recorded_integrity]
        for path in output_paths:
            completed_ids.update(
                int(value)
                for value in pd.read_parquet(path, columns=["exam_id"])[
                    "exam_id"
                ]
            )
        if not completed_ids.issubset(selected_ids):
            raise RuntimeError(
                f"{cohort} checkpoint contains exams outside the selected cohort"
            )
    elif force:
        atomic_write_json(
            checkpoint_path,
            {
                "checkpoint_version": EXTERNAL_EXTRACTION_CHECKPOINT_VERSION,
                "extraction_input_hash": extraction_input_hash,
                "cohort": cohort,
                "completed_record_count": 0,
                "output_integrity": [],
            },
        )
    part = len(output_paths)
    processed = len(completed_ids)
    output_buffer: list[dict[str, object]] = []
    registered = {
        definition.name for definition in BiomarkerRegistry.definitions()
    }
    requested_biomarkers = tuple(
        marker for marker in marker_ids if marker in registered
    )

    def flush_output(*, final: bool = False) -> None:
        nonlocal part
        while len(output_buffer) >= EXTERNAL_FEATURE_PARTITION_SIZE or (
            final and output_buffer
        ):
            take = min(EXTERNAL_FEATURE_PARTITION_SIZE, len(output_buffer))
            rows = output_buffer[:take]
            del output_buffer[:take]
            path = (
                cohort_output_root
                / f"part-{part:06d}.parquet"
            )
            atomic_write_parquet(pd.DataFrame(rows), path)
            output_paths.append(path)
            integrity = output_file_integrity(path)
            checkpoint_integrity.append(integrity)
            completed_ids.update(
                int(cast(int, row["exam_id"])) for row in rows
            )
            atomic_write_json(
                checkpoint_path,
                {
                    "checkpoint_version": EXTERNAL_EXTRACTION_CHECKPOINT_VERSION,
                    "extraction_input_hash": extraction_input_hash,
                    "cohort": cohort,
                    "completed_record_count": len(completed_ids),
                    "output_integrity": checkpoint_integrity,
                },
            )
            part += 1

    def compute_serial(item):
        record, row = item
        return _analyze_record(
            record,
            row,
            marker_ids,
            requested_biomarkers,
            powerline_hz=60.0,
            minimum_signal_quality=minimum_signal_quality,
        )

    if cohort == "code15":
        files = tuple(sorted(metadata["trace_file"].dropna().astype(str).unique()))
        sources = [
            Hdf5TracingSource(root / file_name)
            for file_name in files
        ]
    else:
        sources = [
            Hdf5TracingSource(
                root / "exams.hdf5",
                fallback_exam_ids=full_metadata["exam_id"].astype(int).to_numpy(),
            )
        ]
    # Audit R13: the pool ceiling follows the live grant instead of the fixed
    # ``workers`` setting; admission is re-read while the stage runs.
    from .resource_grants import worker_ceiling

    bounded = max(1, int(worker_ceiling(int(workers))))
    executor = (
        ProcessPoolExecutor(
            max_workers=bounded,
            # Explicit spawn keeps HDF5 handles and parent-side native-library
            # state out of workers on every platform and matches Windows.
            mp_context=get_context("spawn"),
            initializer=_initialize_external_worker,
            initargs=(
                marker_ids,
                requested_biomarkers,
                60.0,
                minimum_signal_quality,
            ),
        )
        if bounded > 1
        else None
    )
    def pending_items():
        for source in sources:
            # A capped pilot is an indexed read, not a full-release scan with
            # a late filter. CODE-15 is ~68 GB: reading all tracings for a
            # 100-record smoke test defeated the purpose of the cap. Full
            # extraction deliberately keeps contiguous streaming, which is
            # faster than HDF5 fancy indexing when nearly every row is needed.
            source_batches = (
                source.iter_selected_batches(
                    selected_ids - completed_ids,
                    batch_size=64,
                )
                if max_records is not None
                else source.iter_batches(batch_size=64)
            )
            for exam_ids, tracings in source_batches:
                keep = np.asarray(
                    [
                        int(item) in selected_ids and int(item) not in completed_ids
                        for item in exam_ids
                    ]
                )
                if not keep.any():
                    continue
                batch_ids = np.asarray(exam_ids)[keep]
                batch_tracings = np.asarray(tracings)[keep]
                records, rejected = read_records(
                    source=cast(
                        TracingSource,
                        ArrayTracingSource(batch_ids, batch_tracings),
                    ),
                    patient_ids={
                        exam_id: int(metadata_lookup[exam_id]["patient_id"])
                        for exam_id in map(int, batch_ids)
                    },
                    cohort=cohort,
                    minimum_trimmed_seconds=6.5,
                )
                if not rejected.empty:
                    rejected["cohort"] = cohort
                    rejected_frames.append(rejected)
                for record in records:
                    yield (record, metadata_lookup[int(record.exam_id)])

    def admission() -> int:
        from .resources import recommend_dynamic_workers

        return recommend_dynamic_workers(
            bounded,
            sample_interval=0.05,
            current_workers=bounded,
            reserve_gb=float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "6")),
        )

    try:
        # Audit R13: one stream across the 64-record read blocks. The old
        # per-block map waited for the slowest record of every block before
        # reading the next; results still arrive in input order, so the
        # partitions are byte-identical to the block-wise run.
        if executor is None:
            rows_iter = (compute_serial(item) for item in pending_items())
        else:
            from .bounded_executor import pipelined_ordered_map

            rows_iter = pipelined_ordered_map(
                executor,
                _analyze_external_worker_item,
                pending_items(),
                capacity_provider=admission,
            )
        for row in rows_iter:
            output_buffer.append(row)
            processed += 1
            if len(output_buffer) >= EXTERNAL_FEATURE_PARTITION_SIZE:
                flush_output()
            if progress is not None and processed % 64 == 0:
                progress(f"{cohort}: {processed:,}/{len(metadata):,} ECGs")
        if progress is not None and processed % 64:
            progress(f"{cohort}: {processed:,}/{len(metadata):,} ECGs")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    flush_output(final=True)
    rejected = (
        pd.concat(rejected_frames, ignore_index=True)
        if rejected_frames
        else pd.DataFrame(columns=["exam_id", "reason", "trimmed_seconds", "cohort"])
    )
    return output_paths, metadata, rejected


def _read_feature_parts(paths: Iterable[Path]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    paths = tuple(paths)
    archive_columns = {"context_measurement_capsule_json", "context_measurement_sensitivity_json"}
    return (
        pd.concat((pd.read_parquet(path, columns=[name for name in pq.read_schema(path).names
                                                  if name not in archive_columns]) for path in paths), ignore_index=True)
        if paths
        else pd.DataFrame()
    )


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _select_patient_index_exam(frame: pd.DataFrame) -> pd.DataFrame:
    """Select CODE's published first exam, with a deterministic fallback."""
    death = (
        frame["death"].notna()
        if "death" in frame
        else pd.Series(False, index=frame.index)
    )
    follow_up = (
        frame["follow_up_years"].notna()
        if "follow_up_years" in frame
        else pd.Series(False, index=frame.index)
    )
    return (
        frame.assign(_published_first_exam=death & follow_up)
        .sort_values(
            ["patient_id", "_published_first_exam", "exam_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("patient_id", keep="first")
        .drop(columns="_published_first_exam")
    )


def _coverage(features: pd.DataFrame, marker_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for cohort, group in features.groupby("cohort", sort=True):
        extraction_status = group["extraction_status"]
        ok = group.loc[extraction_status.eq("ok")]
        n_records = len(group)
        n_extracted = len(ok)
        signal_qc_abstention_rate = float(
            extraction_status.eq("abstained_signal_qc").mean()
        )
        pipeline_failure_rate = float(
            extraction_status.eq("failed").mean()
        )
        for marker in marker_ids:
            values = _numeric_column(ok, marker)
            n_measurable = int(values.notna().sum())
            rows.append(
                {
                    "cohort": cohort,
                    "marker_id": marker,
                    "n_records": n_records,
                    "n_extracted": n_extracted,
                    "n_measurable": n_measurable,
                    "availability": float(
                        n_measurable / max(n_records, 1)
                    ),
                    "signal_qc_abstention_rate": signal_qc_abstention_rate,
                    "pipeline_failure_rate": pipeline_failure_rate,
                }
            )
    return pd.DataFrame(rows)


def _loader_acceptance(features: pd.DataFrame) -> pd.DataFrame:
    """Execute the loader checks that previously existed only as prose."""

    code = features.loc[
        features["cohort"].eq("code15")
        & features["extraction_status"].eq("ok")
    ].copy()
    code = _select_patient_index_exam(code) if not code.empty else code
    rows: list[dict[str, object]] = []
    for check in LOADER_ACCEPTANCE_CHECKS:
        statistic = str(check["statistic"])
        marker_id = str(check["marker_id"])
        minimum = int(check.get("minimum_records", 100))
        if statistic == "median":
            values = _numeric_column(code, marker_id).dropna()
            n = int(len(values))
            observed = float(values.median()) if n else np.nan
        elif statistic == "spearman_with_heart_rate":
            marker = _numeric_column(code, marker_id)
            rate = _numeric_column(code, "heart_rate_bpm")
            valid = marker.notna() & rate.notna()
            n = int(valid.sum())
            observed = (
                float(stats.spearmanr(marker.loc[valid], rate.loc[valid]).statistic)
                if n >= 3
                else np.nan
            )
        else:
            raise ValueError(f"Unknown external loader statistic: {statistic}")
        reference = float(check["reference_value"])
        comparison = str(check["comparison"])
        if n < minimum or not np.isfinite(observed):
            status = "non_estimable"
            passed = False
            reason = f"requires_at_least_{minimum}_records"
        elif comparison == "less_than":
            passed = observed < reference
            status = "passed" if passed else "failed"
            reason = "" if passed else "observed_not_below_reference"
        elif comparison == "close_to":
            tolerance = float(check.get("absolute_tolerance", 0.30))
            passed = abs(observed - reference) <= tolerance
            status = "passed" if passed else "failed"
            reason = "" if passed else "observed_outside_absolute_tolerance"
        else:
            raise ValueError(f"Unknown external loader comparison: {comparison}")
        rows.append(
            {
                **check,
                "cohort": "code15",
                "n": n,
                "observed_value": observed,
                "status": status,
                "passed": bool(passed),
                "failure_reason": reason,
                "observational_unit": "one_published_index_ecg_per_patient",
            }
        )
    return pd.DataFrame(rows)


def _transform_agreement(features: pd.DataFrame) -> pd.DataFrame:
    """Bland-Altman agreement of direct angles with transformed VCG angles."""

    rows: list[dict[str, object]] = []
    methods = ("kors", "inverse_dower", "pca_kors_aligned")
    for cohort, group in features.groupby("cohort", sort=True):
        index_n = len(group)
        ok = group.loc[group["extraction_status"].eq("ok")]
        transformed_values = {
            method: _numeric_column(
                ok,
                f"transform_{method}__{TRANSFORM_ANGLE_MARKER}",
            )
            for method in methods
        }
        for direct_marker in DIRECT_ANGLE_MARKERS:
            direct = _numeric_column(ok, direct_marker)
            for method in methods:
                transformed_marker = (
                    f"transform_{method}__{TRANSFORM_ANGLE_MARKER}"
                )
                transformed = transformed_values[method]
                valid = direct.notna() & transformed.notna()
                difference = direct.loc[valid] - transformed.loc[valid]
                n = int(valid.sum())
                estimable = n >= 30
                bias = float(difference.mean()) if n else np.nan
                sd = float(difference.std(ddof=1)) if n > 1 else np.nan
                correlation = (
                    float(direct.loc[valid].corr(transformed.loc[valid]))
                    if n > 2
                    else np.nan
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "direct_marker_id": direct_marker,
                        "comparison_marker_id": transformed_marker,
                        "method": method,
                        "index_record_count": index_n,
                        "extraction_ok_count": len(ok),
                        "direct_available_count": int(direct.notna().sum()),
                        "comparison_available_count": int(transformed.notna().sum()),
                        "n": n,
                        "paired_availability_fraction": n / index_n if index_n else np.nan,
                        "bias_deg": bias,
                        "difference_sd_deg": sd,
                        "loa_low_deg": bias - 1.96 * sd if estimable and np.isfinite(sd) else np.nan,
                        "loa_high_deg": bias + 1.96 * sd if estimable and np.isfinite(sd) else np.nan,
                        "pearson_r": correlation,
                        "estimable": estimable,
                        "status": "measured" if estimable else "too_few_pairs",
                        "uncertainty_basis": "descriptive_limits_assuming_independent_index_patients" if estimable else "withheld_below_30_pairs",
                        "pca_interpretation": "record_local_pca_aligned_to_kors_sensitivity_not_anatomical_reference" if method == "pca_kors_aligned" else "not_applicable",
                        "claim_scope": "measurement_agreement_not_interchangeability",
                    }
                )
    return pd.DataFrame(rows)


def _quantile_ci(values: np.ndarray, probability: float) -> tuple[float, float, float]:
    ordered = np.sort(values[np.isfinite(values)])
    return _quantile_ci_from_ordered(ordered, probability)


def _quantile_ci_from_ordered(
    ordered: np.ndarray,
    probability: float,
) -> tuple[float, float, float]:
    """Distribution-free quantile interval for an already sorted sample."""

    n = len(ordered)
    estimate = float(np.quantile(ordered, probability))
    low = int(max(0, stats.binom.ppf(0.025, n, probability) - 1))
    high = int(min(n - 1, stats.binom.ppf(0.975, n, probability)))
    return estimate, float(ordered[low]), float(ordered[high])


def _reference_ranges(code: pd.DataFrame, marker_ids: Sequence[str]) -> pd.DataFrame:
    ok = code.loc[code["extraction_status"].eq("ok")].copy()
    labelled = code.loc[code["patient_id"].notna()].copy()
    labelled["reported_normal_ecg"] = labelled["reported_normal_ecg"].astype(
        "boolean"
    )
    # Eligibility is a property of every ECG belonging to the patient, not
    # only of the tracings whose feature extraction succeeded. Otherwise an
    # abnormal but technically failed ECG silently converts that person into
    # an all-normal reference subject.
    patient_all_normal = labelled.groupby("patient_id")[
        "reported_normal_ecg"
    ].agg(lambda values: bool(values.notna().all() and values.fillna(False).all()))
    eligible = set(patient_all_normal[patient_all_normal].index)
    reference = _select_patient_index_exam(
        ok.loc[ok["patient_id"].isin(eligible)]
    )
    age = pd.to_numeric(reference["age"], errors="coerce")
    reference["age_stratum"] = pd.cut(
        age,
        bins=[18, 40, 60, 80, np.inf],
        right=False,
        labels=["18-39", "40-59", "60-79", "80+"],
    ).astype("string")
    strata = [("all", reference)]
    strata.extend(
        (f"age={age_group}|sex={sex}", group)
        for (age_group, sex), group in reference.groupby(
            ["age_stratum", "sex"], observed=True, sort=True
        )
    )
    rows = []
    for stratum, group in strata:
        for marker in marker_ids:
            values = _numeric_column(group, marker).dropna().to_numpy()
            estimable = len(values) >= 120
            row = {
                "cohort": "code15",
                "population": "primary_care_ecg_normal_not_verified_healthy",
                "stratum": stratum,
                "marker_id": marker,
                "subject_count": int(len(group)),
                "measurable_count": int(len(values)),
                "estimable": estimable,
                "reference_claim": "ecg_reference_interval_not_healthy_normal_value",
            }
            # The three requested percentiles share the same empirical
            # sample. Sort it once, not independently for every percentile.
            ordered = np.sort(values[np.isfinite(values)]) if estimable else values
            for probability, label in ((0.025, "p2_5"), (0.5, "median"), (0.975, "p97_5")):
                if estimable:
                    value, low, high = _quantile_ci_from_ordered(
                        ordered,
                        probability,
                    )
                else:
                    value = low = high = np.nan
                row[label] = value
                row[f"{label}_ci_low"] = low
                row[f"{label}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def _diagnostic_associations(code: pd.DataFrame, marker_ids: Sequence[str]) -> pd.DataFrame:
    frame = _select_patient_index_exam(
        code.loc[code["extraction_status"].eq("ok")]
    ).copy()
    frame["evaluation_split"] = np.where(
        frame["patient_id"].map(_stable_fraction) < 0.70,
        "development",
        "validation",
    )
    rows = []
    for endpoint in DIAGNOSTIC_ENDPOINTS:
        if endpoint not in frame:
            continue
        labels_all = frame[endpoint].astype("boolean")
        for marker in marker_ids:
            if marker not in frame:
                continue
            direction = None
            for split in ("development", "validation"):
                scoped = frame.loc[frame["evaluation_split"].eq(split)]
                labels = labels_all.loc[scoped.index]
                values = pd.to_numeric(scoped[marker], errors="coerce")
                valid = labels.notna() & values.notna()
                y = labels.loc[valid].astype(int).to_numpy()
                x = values.loc[valid].to_numpy(dtype=float)
                positives = x[y == 1]
                negatives = x[y == 0]
                if len(positives) < 30 or len(negatives) < 30:
                    continue
                comparison = stats.mannwhitneyu(
                    positives,
                    negatives,
                    alternative="two-sided",
                )
                raw_auc = float(
                    comparison.statistic / (len(positives) * len(negatives))
                )
                if split == "development":
                    direction = 1 if raw_auc >= 0.5 else -1
                elif direction is None:
                    # Validation must never choose its own orientation merely
                    # because the development arm was too small.
                    continue
                aligned = raw_auc if direction == 1 else 1.0 - raw_auc
                p_value = float(comparison.pvalue)
                rows.append(
                    {
                        "cohort": "code15",
                        "endpoint_id": endpoint,
                        "marker_id": marker,
                        "evaluation_split": split,
                        "direction_from_development": direction,
                        "roc_auc_aligned": aligned,
                        "p_value": p_value,
                        "n": len(x),
                        "events": len(positives),
                        "non_events": len(negatives),
                        "label_scope": _diagnostic_label_scope(endpoint),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value_endpoint_split"] = np.nan
    for _, index in result.groupby(["endpoint_id", "evaluation_split"]).groups.items():
        result.loc[index, "q_value_endpoint_split"] = multipletests(
            result.loc[index, "p_value"].to_numpy(), method="fdr_bh"
        )[1]
    result["q_value_global"] = multipletests(
        result["p_value"].to_numpy(), method="fdr_bh"
    )[1]
    return result


def _diagnostic_label_scope(endpoint: str) -> str:
    if endpoint == "reported_normal_ecg":
        return "automatic_same_ecg_annotation_not_independent_reference"
    return "same_ecg_annotation_not_independent_clinical_reference"


def _survival_associations(
    features: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    incremental_marker_ids: Sequence[str] = (),
) -> pd.DataFrame:
    rows = []
    incremental_markers = {str(value) for value in incremental_marker_ids}
    for cohort, source in features.groupby("cohort", sort=True):
        # CODE-15 publishes mortality/follow-up on exactly one prespecified
        # exam row per patient; other exams deliberately contain NA.  Select
        # that outcome-bound index row first.  Choosing the smallest exam ID
        # first both discarded ~16.6% of followed patients in the real release
        # and could pair the outcome with a different ECG.
        outcome_bound = source.loc[
            source["death"].notna() & source["follow_up_years"].notna()
        ].copy()
        duplicate_outcomes = outcome_bound["patient_id"].duplicated(keep=False)
        if duplicate_outcomes.any():
            raise RuntimeError(
                f"{cohort} has multiple outcome-bound ECG rows for "
                f"{outcome_bound.loc[duplicate_outcomes, 'patient_id'].nunique()} patients"
            )
        frame = outcome_bound.loc[
            outcome_bound["extraction_status"].eq("ok")
        ].copy()
        frame["time_days"] = pd.to_numeric(frame["follow_up_years"], errors="coerce") * 365.25
        frame["event"] = frame["death"].astype("boolean").astype(float)
        normalized_sex = frame["sex"].astype("string").str.upper()
        frame["sex_male"] = normalized_sex.map({"M": 1.0, "F": 0.0})
        missing_sex = int(frame["sex_male"].isna().sum())
        age = pd.to_numeric(frame["age"], errors="coerce")
        frame["age_sd"] = (age - age.mean()) / age.std(ddof=1)
        if "reported_normal_ecg" in frame:
            normal = frame["reported_normal_ecg"].astype("boolean")
            frame["classical_normal_ecg"] = normal.astype("Float64")
        else:
            frame["classical_normal_ecg"] = np.nan
        age_gap = pd.to_numeric(frame.get("ecg_age_gap_years", pd.Series(np.nan, index=frame.index)), errors="coerce")
        gap_sd = age_gap.std(ddof=1)
        frame["ecg_age_gap_sd"] = (age_gap-age_gap.mean())/gap_sd if np.isfinite(gap_sd) and gap_sd > 0 else np.nan
        candidates = tuple(dict.fromkeys((*marker_ids, "ecg_age_gap_years")))
        model_base = frame[
            [
                "patient_id",
                "time_days",
                "event",
                "age_sd",
                "sex_male",
                "classical_normal_ecg",
                "ecg_age_gap_sd",
            ]
        ].copy()
        incremental_baseline_cache: dict[
            bytes, tuple[CoxFit, float, int]
        ] = {}
        age_gap_baseline_cache: dict[bytes, CoxFit] = {}
        for marker in candidates:
            unavailable = {
                "cohort": cohort, "marker_id": marker, "status": "non_estimable",
                "clinical_claim_allowed": False, "n_missing_sex": missing_sex,
                "incremental_evaluation_scope": "in_sample_nested_association_not_validated_prediction",
                "incremental_comparison_status": "non_estimable" if marker in incremental_markers else "not_requested",
                "ecg_age_comparison_status": "non_estimable" if marker in incremental_markers and marker != "ecg_age_gap_years" else "not_requested",
            }
            if marker not in frame:
                rows.append({**unavailable, "abstention_reason": "marker_not_available"})
                continue
            value = pd.to_numeric(frame[marker], errors="coerce")
            deviation = value.std(ddof=1)
            if not np.isfinite(deviation) or deviation <= 0:
                rows.append({**unavailable, "abstention_reason": "marker_constant_or_nonfinite"})
                continue
            # Reuse the compact six-column Cox frame and overwrite its one
            # scratch marker column. ``frame.assign`` used to duplicate the
            # entire ~320-marker external matrix once per marker; copying even
            # this compact frame is unnecessary because all fits are serial
            # and ``fit_cox`` materializes the columns it consumes.
            model_frame = model_base
            model_frame["marker_sd"] = (value - value.mean()) / deviation
            try:
                complete_fit = model_frame[["time_days", "event", "marker_sd", "age_sd", "sex_male"]].replace([np.inf, -np.inf], np.nan).dropna()
                if int(complete_fit["event"].sum()) < 30:
                    raise ValueError("insufficient_events: at least 30 events for marker+age+sex")
                fit = fit_cox(
                    model_frame,
                    ("marker_sd", "age_sd", "sex_male"),
                    cluster_column="patient_id",
                )
            except (ValueError, np.linalg.LinAlgError, ConvergenceFailure) as exc:
                rows.append(
                    {
                        "cohort": cohort,
                        "marker_id": marker,
                        "status": "non_estimable",
                        "abstention_reason": str(exc),
                        "n_missing_sex": missing_sex,
                    }
                )
                continue
            marker_row = fit.to_frame().iloc[0]
            row = {
                    "cohort": cohort,
                    "marker_id": marker,
                    "status": "estimable",
                    "hazard_ratio_per_sd": marker_row["hazard_ratio"],
                    "ci_low": marker_row["hr_lower"],
                    "ci_high": marker_row["hr_upper"],
                    "p_value": marker_row["p_value"],
                    "n": fit.n_observations,
                    "events": fit.n_events,
                    "adjustment": "age+sex",
                    "outcome": "all_cause_mortality",
                    "reference_kind": "independent_outcome",
                    "analysis_design": "cohort_specific_adjusted_association",
                    "clinical_claim_allowed": False,
                    "incremental_evaluation_scope": "in_sample_nested_association_not_validated_prediction",
                    "n_missing_sex": missing_sex,
                    "incremental_comparison_status": "not_requested",
                    "classical_comparator": "reported_normal_ecg",
                    "baseline_concordance": np.nan,
                    "candidate_concordance": np.nan,
                    "incremental_concordance": np.nan,
                    "incremental_likelihood_ratio": np.nan,
                    "incremental_p_value": np.nan,
                    "hazard_ratio_per_sd_plus_classical": np.nan,
                    "hr_plus_classical_lower": np.nan,
                    "hr_plus_classical_upper": np.nan,
                }
            if marker in incremental_markers:
                complete_columns = [
                    "time_days",
                    "event",
                    "age_sd",
                    "sex_male",
                    "classical_normal_ecg",
                    "marker_sd",
                ]
                complete = (
                    model_frame[complete_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .notna()
                    .all(axis=1)
                )
                scoped = model_frame.loc[complete]
                if (
                    len(scoped) >= 100
                    and int(scoped["event"].sum()) >= 40
                    and scoped["classical_normal_ecg"].nunique() >= 2
                ):
                    cache_key = np.packbits(
                        complete.to_numpy(dtype=np.uint8)
                    ).tobytes()
                    try:
                        cached = incremental_baseline_cache.get(cache_key)
                        if cached is None:
                            baseline_fit = fit_cox(
                                scoped,
                                (
                                    "age_sd",
                                    "sex_male",
                                    "classical_normal_ecg",
                                ),
                                cluster_column="patient_id",
                            )
                            baseline_risk = scoped[
                                [
                                    "age_sd",
                                    "sex_male",
                                    "classical_normal_ecg",
                                ]
                            ].to_numpy(dtype=float) @ baseline_fit.coefficients
                            baseline_c, comparable = concordance_index(
                                scoped["time_days"].to_numpy(dtype=float),
                                scoped["event"].to_numpy(dtype=float),
                                baseline_risk,
                            )
                            cached = (baseline_fit, baseline_c, comparable)
                            incremental_baseline_cache[cache_key] = cached
                        baseline_fit, baseline_c, comparable = cached
                        candidate_fit = fit_cox(
                            scoped,
                            (
                                "age_sd",
                                "sex_male",
                                "classical_normal_ecg",
                                "marker_sd",
                            ),
                            cluster_column="patient_id",
                        )
                        candidate_risk = scoped[
                            [
                                "age_sd",
                                "sex_male",
                                "classical_normal_ecg",
                                "marker_sd",
                            ]
                        ].to_numpy(dtype=float) @ candidate_fit.coefficients
                        candidate_c, _ = concordance_index(
                            scoped["time_days"].to_numpy(dtype=float),
                            scoped["event"].to_numpy(dtype=float),
                            candidate_risk,
                        )
                        likelihood_ratio = max(
                            0.0,
                            2.0
                            * (
                                candidate_fit.log_likelihood
                                - baseline_fit.log_likelihood
                            ),
                        )
                        interval = candidate_fit.confidence_interval()
                        row.update(
                            {
                                "incremental_comparison_status": "estimable",
                                "baseline_concordance": baseline_c,
                                "candidate_concordance": candidate_c,
                                "incremental_concordance": candidate_c
                                - baseline_c,
                                "concordance_pairs": int(comparable),
                                "incremental_likelihood_ratio": likelihood_ratio,
                                "incremental_p_value": float(
                                    stats.chi2.sf(likelihood_ratio, df=1)
                                ),
                                "hazard_ratio_per_sd_plus_classical": float(
                                    candidate_fit.hazard_ratios[-1]
                                ),
                                "hr_plus_classical_lower": float(
                                    interval[-1, 0]
                                ),
                                "hr_plus_classical_upper": float(
                                    interval[-1, 1]
                                ),
                            }
                        )
                    except (
                        ValueError,
                        np.linalg.LinAlgError,
                        ConvergenceFailure,
                    ) as exc:
                        row["incremental_comparison_status"] = (
                            "non_estimable_model_failure"
                        )
                        row["incremental_abstention_reason"] = str(exc)
                else:
                    row["incremental_comparison_status"] = (
                        "non_estimable_classical_comparator_or_events"
                    )
            row["ecg_age_comparison_status"] = "not_requested"
            if marker in incremental_markers and marker != "ecg_age_gap_years":
                baseline_columns = ("age_sd", "sex_male", "classical_normal_ecg", "ecg_age_gap_sd")
                columns = ("time_days", "event", *baseline_columns, "marker_sd")
                complete = np.isfinite(model_frame[list(columns)].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
                scoped = model_frame.loc[complete]
                row["ecg_age_comparison_n"] = len(scoped)
                row["ecg_age_comparison_events"] = int(scoped.event.sum())
                row["ecg_age_comparator"] = "age+sex+reported_normal_ecg+ecg_age_gap"
                try:
                    if len(scoped) < 100 or int(scoped.event.sum()) < 50:
                        raise ValueError("insufficient_complete_patients_or_events_for_five_parameters")
                    if (scoped[list(baseline_columns)].nunique() < 2).any():
                        raise ValueError("constant_or_missing_classical_or_ecg_age_comparator")
                    key = np.packbits(complete).tobytes()
                    baseline = age_gap_baseline_cache.get(key)
                    if baseline is None:
                        baseline = fit_cox(scoped, baseline_columns, cluster_column="patient_id")
                        age_gap_baseline_cache[key] = baseline
                    augmented = fit_cox(scoped, (*baseline_columns, "marker_sd"), cluster_column="patient_id")
                    lr = max(0., 2*(augmented.log_likelihood-baseline.log_likelihood))
                    interval = augmented.confidence_interval()
                    row.update(ecg_age_comparison_status="estimable", ecg_age_incremental_likelihood_ratio=lr,
                        ecg_age_incremental_p_value=float(stats.chi2.sf(lr, 1)),
                        hr_plus_classical_ecg_age=float(augmented.hazard_ratios[-1]),
                        hr_plus_classical_ecg_age_lower=float(interval[-1, 0]),
                        hr_plus_classical_ecg_age_upper=float(interval[-1, 1]))
                except (ValueError, np.linalg.LinAlgError, ConvergenceFailure) as exc:
                    row["ecg_age_comparison_status"] = "non_estimable"
                    row["ecg_age_abstention_reason"] = str(exc)
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # All-abstention cohorts are valid outputs, not missing-schema crashes.
    for name, default in (("incremental_comparison_status", "non_estimable"),
                          ("p_value", np.nan), ("incremental_p_value", np.nan)):
        if name not in result:
            result[name] = default
    from .population_multiplicity import adjust_population_survival
    return adjust_population_survival(result, marker_ids=marker_ids,
                                      incremental_marker_ids=incremental_marker_ids)


def _chagas_contrast(features: pd.DataFrame, marker_ids: Sequence[str]) -> pd.DataFrame:
    code = features.loc[
        features["cohort"].eq("code15")
    ].copy()
    sami = features.loc[
        features["cohort"].eq("sami_trop")
    ].copy()
    if code.empty or sami.empty:
        return pd.DataFrame()
    code = _select_patient_index_exam(code)
    sami = _select_patient_index_exam(sami)
    n_code_index, n_sami_index = len(code), len(sami)
    code = code.loc[code.extraction_status.eq("ok")].copy()
    sami = sami.loc[sami.extraction_status.eq("ok")].copy()
    n_code_qc, n_sami_qc = len(code), len(sami)
    for frame in (code, sami):
        frame["match_age"] = (pd.to_numeric(frame["age"], errors="coerce") // 5) * 5
    matched_code, matched_sami = [], []
    eligible_sami = sami.loc[np.isfinite(sami.match_age) & sami.sex.isin(["M", "F"])]
    for (age, sex), cases in eligible_sami.groupby(["match_age", "sex"]):
        controls = code.loc[code["match_age"].eq(age) & code["sex"].eq(sex)]
        controls = controls.sort_values("patient_id", key=lambda values: values.map(_stable_fraction))
        cases = cases.sort_values("patient_id", key=lambda values: values.map(_stable_fraction))
        count = min(len(cases), len(controls))
        matched_code.append(controls.head(count))
        matched_sami.append(cases.head(count))
    code = pd.concat(matched_code, ignore_index=True) if matched_code else code.iloc[:0].reset_index(drop=True)
    sami = pd.concat(matched_sami, ignore_index=True) if matched_sami else sami.iloc[:0].reset_index(drop=True)
    rows = []
    for marker in marker_ids:
        case_values = _numeric_column(sami, marker)
        control_values = _numeric_column(code, marker)
        # Pairing is fixed before marker availability is inspected. Never
        # replace a missing member with a more measurable patient; remove
        # both members so the age/sex strata remain balanced for each marker.
        complete_pairs = np.isfinite(case_values) & np.isfinite(control_values)
        case_values = case_values.loc[complete_pairs]
        control_values = control_values.loc[complete_pairs]
        if len(case_values) < 30 or len(control_values) < 30:
            continue
        comparison = stats.mannwhitneyu(
            case_values,
            control_values,
            alternative="two-sided",
        )
        raw_auc = float(
            comparison.statistic / (len(case_values) * len(control_values))
        )
        rows.append(
            {
                "marker_id": marker,
                "aligned_auc": max(raw_auc, 1.0 - raw_auc),
                "direction": "higher_in_chagas" if raw_auc >= 0.5 else "lower_in_chagas",
                "p_value": float(comparison.pvalue),
                "n_chagas": len(case_values),
                "n_code15_matched": len(control_values),
                "n_code15_index": n_code_index,
                "n_sami_index": n_sami_index,
                "n_code15_after_qc": n_code_qc,
                "n_sami_after_qc": n_sami_qc,
                "n_matched_pairs_before_marker": len(code),
                "n_pairs_excluded_marker": int((~complete_pairs).sum()),
                "matching_design": "one_to_one_age5_sex_before_marker_no_replacement",
                "p_value_scope": "unadjusted_marginal_distribution_not_conditional_disease_effect",
                "claim_scope": "exploratory_site_confounded_cross_cohort_contrast",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def run_population_cohort_analysis(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: Sequence[str] = ("primary_core", "sentinel"),
    additional_marker_ids: Sequence[str] = (),
    max_records_per_cohort: Optional[int] = None,
    minimum_signal_quality: float = 0.0,
    workers: Optional[int] = None,
    verify_source_sha256: bool = True,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> PopulationCohortResult:
    if config.code15_root is None or config.sami_trop_root is None:
        raise RuntimeError("CODE-15 and SaMi-Trop roots must both be configured")
    roots = {
        "code15": Path(config.code15_root),
        "sami_trop": Path(config.sami_trop_root),
    }
    for cohort, root in roots.items():
        if not (root / "exams.csv").is_file():
            raise FileNotFoundError(f"{cohort} metadata missing: {root / 'exams.csv'}")
    release_audits = {
        cohort: audit_cohort_release(root, cohort)
        for cohort, root in roots.items()
    }
    release_errors = {
        cohort: audit
        for cohort, audit in release_audits.items()
        if audit.get("status") != "ok"
    }
    if release_errors:
        raise RuntimeError(
            "External cohort release audit failed: "
            + "; ".join(
                f"{cohort}: {audit.get('diagnostic', 'unknown error')}"
                for cohort, audit in release_errors.items()
            )
        )
    panel_marker_ids = _panel_markers(panel_ids)
    transported_marker_ids, transport_provenance = (
        _transportable_validation_markers(config, run_id)
    )
    registered = set(BiomarkerRegistry.available())
    unknown_additional = sorted(
        {str(value) for value in additional_marker_ids} - registered
    )
    if unknown_additional:
        raise ValueError(
            "Unknown or non-computable external marker IDs: "
            + ", ".join(unknown_additional)
        )
    marker_ids = tuple(
        dict.fromkeys(
            (
                *panel_marker_ids,
                *transported_marker_ids,
                *(str(value) for value in additional_marker_ids),
            )
        )
    )
    feature_code_hash = feature_implementation_hash()
    analysis_code_hash = _population_analysis_source_hash()
    output_root = config.external_validation_root / run_id / "population_cohorts"
    manifest_path = output_root / "population_cohorts.manifest.json"
    hash_cache_path = (
        config.external_validation_root / "population_source_hash_cache.json"
    )
    hash_cache = _load_source_hash_cache(hash_cache_path)
    source_frames = [
        _source_inventory(
            root,
            cohort,
            verify_sha256=verify_source_sha256,
            hash_cache=hash_cache,
        )
        for cohort, root in roots.items()
    ]
    if verify_source_sha256:
        _save_source_hash_cache(hash_cache_path, hash_cache)
    inventory = pd.concat(source_frames, ignore_index=True)
    input_hash = content_hash(
        {
            "stage_version": POPULATION_COHORT_STAGE_VERSION,
            "analysis_implementation_hash": analysis_code_hash,
            "cohort_version": CODE15_COHORT_VERSION,
            "mortality_multiplicity_source_hash": content_hash(
                (Path(__file__).parent / "population_multiplicity.py").read_text(encoding="utf-8")),
            "feature_implementation_hash": feature_code_hash,
            "panel_ids": list(panel_ids),
            "marker_ids": list(marker_ids),
            "transport_provenance": transport_provenance,
            "max_records_per_cohort": max_records_per_cohort,
            "minimum_signal_quality": minimum_signal_quality,
            "verify_source_sha256": verify_source_sha256,
            "feature_partition_size": EXTERNAL_FEATURE_PARTITION_SIZE,
            "capped_sampling_version": EXTERNAL_CAPPED_SAMPLING_VERSION,
            "release_audits": release_audits,
            "sources": inventory[
                ["cohort", "path", "size", "mtime_ns", "sha256"]
            ].to_dict(orient="records"),
        }
    )
    if manifest_path.is_file() and not force:
        import json
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = tuple(str(item) for item in previous.get("output_paths", ()))
        output_integrity = tuple(previous.get("output_integrity", ()))
        if (
            previous.get("stage_version") == POPULATION_COHORT_STAGE_VERSION
            and previous.get("input_hash") == input_hash
            and outputs
            and output_integrity
            and verify_output_integrity(output_integrity)
        ):
            return PopulationCohortResult(
                status="complete",
                output_paths=outputs,
                row_counts=dict(previous.get("row_counts", {})),
                notes=("reused_exact_input_hash",),
            )
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "source_inventory.parquet"
    atomic_write_parquet(inventory, inventory_path)
    feature_paths: list[Path] = []
    metadata_frames = []
    rejected_frames = []
    for cohort, root in roots.items():
        extraction_input_hash = _population_extraction_input_hash(
            inventory, cohort, marker_ids, max_records_per_cohort, minimum_signal_quality)
        paths, metadata, rejected = _extract_cohort(
            root,
            cohort,
            marker_ids,
            output_root,
            extraction_input_hash=extraction_input_hash,
            max_records=max_records_per_cohort,
            workers=int(workers or config.workers),
            minimum_signal_quality=minimum_signal_quality,
            force=force,
            progress=progress,
        )
        feature_paths.extend(paths)
        metadata_frames.append(metadata)
        rejected_frames.append(rejected)
    features = _read_feature_parts(feature_paths)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    rejected = pd.concat(rejected_frames, ignore_index=True)
    loader_acceptance = _loader_acceptance(features)
    # Biological population differences cannot prove a loader error. Physical
    # shape, ID, units, lead order and padding checks still fail closed upstream.
    loader_acceptance["gate_kind"] = "population_distribution_warning"
    loader_acceptance["blocks_analysis"] = False
    tables = {
        "metadata": metadata,
        "rejected_records": rejected,
        "loader_acceptance": loader_acceptance,
        "coverage": _coverage(features, marker_ids),
        "direct_transform_agreement": _transform_agreement(features),
        "code15_reference_ranges": _reference_ranges(
            features.loc[features["cohort"].eq("code15")], marker_ids
        ),
        "code15_diagnostic_associations": _diagnostic_associations(
            features.loc[features["cohort"].eq("code15")], marker_ids
        ),
        "mortality_associations": _survival_associations(
            features,
            marker_ids,
            incremental_marker_ids=tuple(
                dict.fromkeys(
                    (
                        TRANSFORM_ANGLE_MARKER,
                        *DIRECT_ANGLE_MARKERS,
                        *transported_marker_ids,
                    )
                )
            ),
        ),
        "sami_trop_chagas_contrast": _chagas_contrast(features, marker_ids),
    }
    from .cohort_value import cohort_value_table
    tables["cohort_added_value"] = cohort_value_table(features, cohort_column="cohort",
        marker_ids=tuple(marker_ids), survival=tables["mortality_associations"])
    output_paths = [inventory_path, *feature_paths]
    row_counts = {"feature_rows": int(len(features))}
    for name, frame in tables.items():
        path = output_root / f"{name}.parquet"
        atomic_write_parquet(frame, path)
        output_paths.append(path)
        row_counts[name] = int(len(frame))
    resolved_outputs = [str(path.resolve()) for path in output_paths]
    output_integrity = capture_output_integrity(output_paths)
    payload = {
        "status": "complete",
        "stage_version": POPULATION_COHORT_STAGE_VERSION,
        "cohort_version": CODE15_COHORT_VERSION,
        "feature_implementation_hash": feature_code_hash,
        "input_hash": input_hash,
        "analysis_implementation_hash": analysis_code_hash,
        "run_id": run_id,
        "panel_ids": list(panel_ids),
        "marker_count": len(marker_ids),
        "base_panel_marker_count": len(panel_marker_ids),
        "transported_validation_marker_count": len(transported_marker_ids),
        "transported_validation_markers": list(transported_marker_ids),
        "transport_provenance": transport_provenance,
        "max_records_per_cohort": max_records_per_cohort,
        "row_counts": row_counts,
        "output_paths": resolved_outputs,
        "output_integrity": list(output_integrity),
        "locked_test_read": False,
        "source_sha256_verified": bool(verify_source_sha256),
        "release_audits": release_audits,
        "claim_limits": {
            "code15_reference": "primary_care_ecg_normal_not_verified_healthy",
            "code15_labels": (
                "same_ecg_annotations_not_independent_clinical_reference;"
                "normal_ecg_is_automatic_annotation"
            ),
            "sami_trop": "seropositive_chagas_prognosis",
            "cross_cohort": "site_confounded_exploratory_only",
        },
    }
    atomic_write_json(manifest_path, payload)
    return PopulationCohortResult(
        status="complete",
        output_paths=tuple(resolved_outputs),
        row_counts=row_counts,
        notes=(
            "full_run_not_started",
            "patient_level_index_for_cross_sectional_analyses",
            "external_cohorts_not_pooled_with_mimic",
            (
                "mimic_validation_markers_transported="
                f"{len(transported_marker_ids)}"
            ),
        ),
    )


__all__ = [
    "DIAGNOSTIC_ENDPOINTS",
    "POPULATION_COHORT_STAGE_VERSION",
    "PopulationCohortResult",
    "run_population_cohort_analysis",
]
