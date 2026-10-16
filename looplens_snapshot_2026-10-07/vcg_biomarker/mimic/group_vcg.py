from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..core.delineation.fusion_delineator import FusionDelineator
from ..core.preprocessing import detect_r_peaks_multilead
from ..core.vcg_transform import VCGTransformRegistry
from .config import MimicProjectConfig
from .event_equivalence import (
    verify_clinical_event_evidence_provenance,
)
from .feature_extraction import verify_feature_artifacts
from .label_registry import default_label_definitions
from .run_integrity import assert_development_run_mutable
from .signals import MimicWaveformLoader
from .snapshots import create_artifact_snapshot, verify_artifact_snapshot
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    atomic_replace,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
    file_sha256,
    feature_parquet_clause,
    source_tree_hash,
)


GROUP_VCG_STAGE_VERSION = "1.2.0"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class GroupVCGOptions:
    run_id: str = "mimic-development"
    endpoint_id: str = ""
    split: str = "validation"
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel")
    feature_glob: Optional[str] = None
    max_per_class: int = 40
    minimum_per_class: int = 10
    candidate_multiplier: int = 4
    quality_minimum: float = 0.75
    bootstrap_iterations: int = 500
    pre_r_seconds: float = 0.30
    post_r_seconds: float = 0.50
    force: bool = False
    execution_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise PermissionError(
                "Group VCG analysis is development-only; split must be "
                "'train' or 'validation'. locked_test remains sealed."
            )
        if not self.endpoint_id.strip():
            raise ValueError("endpoint_id is required")
        if not self.run_id.strip() or any(
            token in self.run_id for token in ("..", "/", "\\")
        ):
            raise ValueError("run_id must be a safe local identifier")
        if self.max_per_class < 2:
            raise ValueError("max_per_class must be at least two")
        if not 2 <= self.minimum_per_class <= self.max_per_class:
            raise ValueError(
                "minimum_per_class must be between two and max_per_class"
            )
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least one")
        if not 0.0 <= self.quality_minimum <= 1.0:
            raise ValueError("quality_minimum must be between zero and one")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if self.pre_r_seconds <= 0 or self.post_r_seconds <= 0:
            raise ValueError("R-aligned beat windows must be positive")


def group_vcg_artifact_dir(
    config: MimicProjectConfig,
    run_id: str,
    endpoint_id: str,
    split: str,
) -> Path:
    return (
        config.runs_root
        / str(run_id)
        / "group_vcg"
        / f"endpoint={_safe_component(endpoint_id)}"
        / f"split={_safe_component(split)}"
    )


def run_group_vcg_analysis(
    config: MimicProjectConfig,
    options: GroupVCGOptions,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    definitions = {
        definition.label_id for definition in default_label_definitions()
    }
    if options.endpoint_id not in definitions:
        raise ValueError(f"Unknown clinical endpoint: {options.endpoint_id}")

    feature_glob = options.feature_glob or str(
        config.features_root
        / "_".join(options.panel_ids)
        / "development"
        / "split=*"
        / "part-*.parquet"
    )
    feature_paths = tuple(
        sorted(Path(path) for path in glob.glob(feature_glob))
    )
    if not feature_paths:
        raise FileNotFoundError(
            f"No complete development features match {feature_glob!r}"
        )
    feature_integrity = verify_feature_artifacts(feature_paths)
    feature_root = Path(str(feature_integrity["root"]))
    if feature_root.name.startswith("smoke-"):
        raise RuntimeError(
            "Group VCG results require a complete development extraction; "
            "smoke/preview artifacts are descriptive only."
        )
    feature_snapshot = create_artifact_snapshot(
        config,
        artifact_kind="development_features",
        source_glob=feature_glob,
        snapshot_id=(
            f"{_safe_component(options.run_id)}-group-vcg-features-"
            f"{str(feature_integrity['artifact_hash'])[:16]}"
        ),
        require_complete=True,
        notes=(
            f"group_vcg_endpoint={options.endpoint_id}",
            f"split={options.split}",
            "locked_test_included=False",
        ),
    )
    feature_snapshot.assert_freezable()

    cohort_path = config.labels_root / "endpoint_index_cohorts.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    for path in (cohort_path, master_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required group VCG input is missing: {path}")
    event_evidence_provenance = (
        verify_clinical_event_evidence_provenance(config)
    )
    event_evidence_summary = {
        key: event_evidence_provenance.get(key)
        for key in (
            "status",
            "clinical_event_registry_input_hash",
            "attestation_hash",
        )
        if event_evidence_provenance.get(key) is not None
    }
    event_evidence_path = (
        config.evidence_root / "evidence_provenance.json"
        if event_evidence_provenance.get("status")
        == "direct_event_backed"
        else (
            config.work_root
            / "provenance"
            / "clinical_event_evidence_equivalence.json"
        )
    )

    output_root = group_vcg_artifact_dir(
        config,
        options.run_id,
        options.endpoint_id,
        options.split,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "group_vcg_summary.parquet"
    difference_path = output_root / "group_vcg_difference.parquet"
    audit_path = output_root / "group_vcg_case_audit.parquet"
    failure_path = output_root / "group_vcg_failure.json"
    provenance_path = output_root / "group_vcg_provenance.json"
    manifest_path = output_root / "group_vcg.manifest.json"

    implementation_hash = group_vcg_implementation_hash()
    input_hash = content_hash(
        {
            "stage_version": GROUP_VCG_STAGE_VERSION,
            "feature_snapshot_hash": feature_snapshot.snapshot_hash,
            "feature_artifact_hash": feature_integrity["artifact_hash"],
            "cohort": file_fingerprint(cohort_path, include_sha256=True),
            "master": file_fingerprint(master_path, include_sha256=True),
            "clinical_event_evidence_provenance": (
                file_fingerprint(
                    event_evidence_path,
                    include_sha256=True,
                )
            ),
            "options": {
                key: value
                for key, value in options.__dict__.items()
                if key not in {"force", "execution_id"}
            },
            "implementation_hash": implementation_hash,
        }
    )
    config_hash = scientific_config_hash(config)
    if not options.force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            GROUP_VCG_STAGE_VERSION,
        ):
            return previous
    assert_development_run_mutable(
        config.runs_root / options.run_id,
        operation="Group VCG analysis",
    )

    candidates = _load_candidates(
        config,
        options,
        feature_glob=feature_glob,
    )
    selected = _deterministic_candidate_pool(
        candidates,
        options,
        random_seed=config.random_seed,
    )
    checkpoint_path = output_root / ".group_vcg.resume.npz"
    (
        loops_by_state,
        audit_rows,
        completed_study_ids,
    ) = _load_group_checkpoint(
        checkpoint_path,
        input_hash=input_hash,
        execution_id=options.execution_id,
        force=options.force,
        expected_samples=int(
            round(
                (options.pre_r_seconds + options.post_r_seconds) * 500
            )
        ),
    )
    attempted = len(completed_study_ids)
    total_attempts = len(selected)
    if progress is not None and completed_study_ids:
        progress(
            f"Gruppen-VCG {attempted:,}/{total_attempts:,}: "
            "atomaren Checkpoint fortgesetzt"
        )
    for state in ("positive", "negative"):
        state_rows = selected.loc[
            selected["label_state"].eq(state)
        ]
        for row in state_rows.itertuples(index=False):
            study_id = int(row.study_id)
            if study_id in completed_study_ids:
                continue
            if len(loops_by_state[state]) >= options.max_per_class:
                break
            attempted += 1
            audit = {
                "subject_id": int(row.subject_id),
                "study_id": study_id,
                "endpoint_id": options.endpoint_id,
                "label_state": state,
                "split": options.split,
                "signal_quality_score": float(row.signal_quality_score),
                "status": "excluded",
                "error_type": "",
                "error_message": "",
                "accepted_beat_count": 0,
                "selected_rank": int(row.selected_rank),
            }
            try:
                loop, accepted_count = _load_representative_vcg(
                    row,
                    options,
                    powerline_hz=config.powerline_hz,
                )
                loops_by_state[state].append(loop)
                audit["status"] = "included"
                audit["accepted_beat_count"] = accepted_count
            except Exception as exc:
                audit["error_type"] = type(exc).__name__
                audit["error_message"] = str(exc)[:1000]
            audit_rows.append(audit)
            completed_study_ids.add(study_id)
            _atomic_write_group_checkpoint(
                checkpoint_path,
                input_hash=input_hash,
                execution_id=options.execution_id,
                loops_by_state=loops_by_state,
                audit_rows=audit_rows,
                completed_study_ids=completed_study_ids,
                expected_samples=int(
                    round(
                        (
                            options.pre_r_seconds
                            + options.post_r_seconds
                        )
                        * 500
                    )
                ),
            )
            if progress is not None:
                progress(
                    f"Gruppen-VCG {attempted:,}/{total_attempts:,}: "
                    f"{state}, {len(loops_by_state[state])}/"
                    f"{options.max_per_class} verwendbar"
                )

    counts = {
        state: len(loops) for state, loops in loops_by_state.items()
    }
    insufficient = {
        state: count
        for state, count in counts.items()
        if count < options.minimum_per_class
    }
    audit_frame = pd.DataFrame(audit_rows)
    if insufficient:
        atomic_write_parquet(audit_frame, audit_path)
        atomic_write_json(
            failure_path,
            {
                "status": "insufficient_usable_ecgs",
                "class_counts": counts,
                "minimum_per_class": options.minimum_per_class,
                "locked_test_read": False,
                "feature_snapshot_hash": feature_snapshot.snapshot_hash,
            },
        )
        raise RuntimeError(
            "Too few usable validation ECGs for a scientifically interpretable "
            f"group VCG: {insufficient}; required "
            f"{options.minimum_per_class} per class."
        )

    summary, difference = aggregate_group_vcg(
        loops_by_state,
        sampling_rate_hz=500,
        pre_r_seconds=options.pre_r_seconds,
        bootstrap_iterations=options.bootstrap_iterations,
        random_seed=config.random_seed,
    )
    for frame in (summary, difference):
        frame.insert(0, "split", options.split)
        frame.insert(0, "endpoint_id", options.endpoint_id)
        frame.insert(0, "run_id", options.run_id)
        frame["feature_snapshot_hash"] = feature_snapshot.snapshot_hash
        frame["artifact_completeness"] = "complete"
        frame["exploration_grade"] = "descriptive_development"

    snapshot_verification = verify_artifact_snapshot(feature_snapshot)
    if not bool(snapshot_verification["valid"]):
        raise RuntimeError(
            "Feature snapshot changed while group VCG analysis was running; "
            "formal outputs were not written."
        )
    atomic_write_parquet(summary, summary_path)
    atomic_write_parquet(difference, difference_path)
    atomic_write_parquet(audit_frame, audit_path)
    failure_path.unlink(missing_ok=True)
    checkpoint_path.unlink(missing_ok=True)
    snapshot_verification = verify_artifact_snapshot(feature_snapshot)
    if not bool(snapshot_verification["valid"]):
        for path in (summary_path, difference_path, audit_path):
            path.unlink(missing_ok=True)
        raise RuntimeError(
            "Feature snapshot changed while group VCG analysis was running; "
            "formal outputs were not accepted."
        )
    provenance = {
        "stage_version": GROUP_VCG_STAGE_VERSION,
        "implementation_hash": implementation_hash,
        "run_id": options.run_id,
        "endpoint_id": options.endpoint_id,
        "split": options.split,
        "locked_test_read": False,
        "feature_snapshot": feature_snapshot.to_dict(),
        "feature_snapshot_verification": snapshot_verification,
        "feature_integrity": feature_integrity,
        "clinical_event_evidence_provenance": event_evidence_summary,
        "selection": {
            "index_unit": "one endpoint-specific ECG per subject",
            "deterministic_rank": (
                "SHA-256(endpoint_id, subject_id, study_id, random_seed)"
            ),
            "quality_minimum": options.quality_minimum,
            "maximum_per_class": options.max_per_class,
            "minimum_per_class": options.minimum_per_class,
            "candidate_multiplier": options.candidate_multiplier,
        },
        "signal_method": {
            "sampling_rate_hz": 500,
            "powerline_hz": config.powerline_hz,
            "vcg_transform": "kors",
            "beat_alignment": "R peak",
            "beat_window_seconds": [
                -options.pre_r_seconds,
                options.post_r_seconds,
            ],
            "within_record_summary": (
                "pointwise median of physiologically plausible, "
                "delineation-consistent accepted beats"
            ),
            "between_subject_summary": "pointwise median",
            "confidence_interval": (
                "95% pointwise subject-bootstrap interval; simultaneous "
                "coverage is not claimed"
            ),
            "amplitude_normalization": "none; physical millivolts retained",
            "baseline_correction": "median pre-QRS baseline",
        },
        "class_counts": counts,
        "input_hash": input_hash,
    }
    atomic_write_json(provenance_path, provenance)
    manifest = StageManifest.complete(
        stage="group-vcg",
        stage_version=GROUP_VCG_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(summary_path.resolve()),
            str(difference_path.resolve()),
            str(audit_path.resolve()),
            str(provenance_path.resolve()),
            str(event_evidence_path.resolve()),
        ),
        row_counts={
            "candidate_ecgs": int(len(candidates)),
            "attempted_ecgs": int(attempted),
            "included_positive": counts["positive"],
            "included_negative": counts["negative"],
            "time_points": int(len(difference)),
            "bootstrap_iterations": options.bootstrap_iterations,
        },
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"feature_snapshot_hash={feature_snapshot.snapshot_hash}",
            "scope=development_only",
            "locked_test_read=False",
            "analysis_grade=descriptive_development",
            (
                "clinical_event_evidence_provenance="
                f"{content_hash(event_evidence_summary)}"
            ),
        ),
    )
    manifest.save(manifest_path)
    DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    ).register_stage(manifest, manifest_path)
    return manifest


def aggregate_group_vcg(
    loops_by_state: dict[str, list[np.ndarray]],
    *,
    sampling_rate_hz: int,
    pre_r_seconds: float,
    bootstrap_iterations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"positive", "negative"}
    if set(loops_by_state) != required:
        raise ValueError(
            "loops_by_state must contain exactly positive and negative"
        )
    rng = np.random.default_rng(random_seed)
    summaries: dict[str, dict[str, np.ndarray]] = {}
    bootstrap_medians: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    expected_shape: Optional[tuple[int, int]] = None
    for state in ("negative", "positive"):
        stack = np.asarray(loops_by_state[state], dtype=float)
        if stack.ndim != 3 or stack.shape[0] < 2 or stack.shape[2] != 3:
            raise ValueError(
                f"{state} group must contain at least two samples-by-3 loops"
            )
        if not np.all(np.isfinite(stack)):
            raise ValueError(f"{state} group contains non-finite VCG values")
        if expected_shape is None:
            expected_shape = tuple(stack.shape[1:])
        elif tuple(stack.shape[1:]) != expected_shape:
            raise ValueError("Positive and negative VCG loops differ in shape")
        median = np.median(stack, axis=0)
        boot = _bootstrap_group_medians(
            stack,
            bootstrap_iterations=bootstrap_iterations,
            rng=rng,
        )
        low = np.quantile(boot, 0.025, axis=0)
        high = np.quantile(boot, 0.975, axis=0)
        radius = np.quantile(
            np.linalg.norm(boot - median[None, :, :], axis=2),
            0.95,
            axis=0,
        )
        summaries[state] = {
            "median": median,
            "low": low,
            "high": high,
            "radius": radius,
        }
        bootstrap_medians[state] = boot
        for sample_index in range(stack.shape[1]):
            rows.append(
                {
                    "label_state": state,
                    "sample_index": sample_index,
                    "time_ms": (
                        sample_index / float(sampling_rate_hz)
                        - pre_r_seconds
                    )
                    * 1000.0,
                    "n_subjects": int(stack.shape[0]),
                    "median_x_mV": median[sample_index, 0],
                    "median_y_mV": median[sample_index, 1],
                    "median_z_mV": median[sample_index, 2],
                    "x_ci_low_mV": low[sample_index, 0],
                    "x_ci_high_mV": high[sample_index, 0],
                    "y_ci_low_mV": low[sample_index, 1],
                    "y_ci_high_mV": high[sample_index, 1],
                    "z_ci_low_mV": low[sample_index, 2],
                    "z_ci_high_mV": high[sample_index, 2],
                    "envelope_radius_mV": radius[sample_index],
                }
            )

    positive = summaries["positive"]["median"]
    negative = summaries["negative"]["median"]
    delta = positive - negative
    delta_boot = (
        bootstrap_medians["positive"]
        - bootstrap_medians["negative"]
    )
    delta_low = np.quantile(delta_boot, 0.025, axis=0)
    delta_high = np.quantile(delta_boot, 0.975, axis=0)
    delta_magnitude_boot = np.linalg.norm(delta_boot, axis=2)
    delta_magnitude = np.linalg.norm(delta, axis=1)
    magnitude_low = np.quantile(delta_magnitude_boot, 0.025, axis=0)
    magnitude_high = np.quantile(delta_magnitude_boot, 0.975, axis=0)
    difference_rows = []
    for sample_index in range(delta.shape[0]):
        difference_rows.append(
            {
                "sample_index": sample_index,
                "time_ms": (
                    sample_index / float(sampling_rate_hz)
                    - pre_r_seconds
                )
                * 1000.0,
                "positive_x_mV": positive[sample_index, 0],
                "positive_y_mV": positive[sample_index, 1],
                "positive_z_mV": positive[sample_index, 2],
                "negative_x_mV": negative[sample_index, 0],
                "negative_y_mV": negative[sample_index, 1],
                "negative_z_mV": negative[sample_index, 2],
                "delta_x_mV": delta[sample_index, 0],
                "delta_y_mV": delta[sample_index, 1],
                "delta_z_mV": delta[sample_index, 2],
                "delta_x_ci_low_mV": delta_low[sample_index, 0],
                "delta_x_ci_high_mV": delta_high[sample_index, 0],
                "delta_y_ci_low_mV": delta_low[sample_index, 1],
                "delta_y_ci_high_mV": delta_high[sample_index, 1],
                "delta_z_ci_low_mV": delta_low[sample_index, 2],
                "delta_z_ci_high_mV": delta_high[sample_index, 2],
                "delta_magnitude_mV": delta_magnitude[sample_index],
                "delta_magnitude_ci_low_mV": magnitude_low[sample_index],
                "delta_magnitude_ci_high_mV": magnitude_high[sample_index],
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(difference_rows)


def _load_candidates(
    config: MimicProjectConfig,
    options: GroupVCGOptions,
    *,
    feature_glob: str,
) -> pd.DataFrame:
    cohort = _sql_path(config.labels_root / "endpoint_index_cohorts.parquet")
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    features = _sql_path(feature_glob)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True, read_only=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=features,
        )
        frame = connection.execute(
            f"""
            WITH feature_quality AS (
                SELECT
                    subject_id,
                    study_id,
                    split,
                    extraction_status,
                    signal_quality_score,
                    signal_quality_usable
                FROM {read_features}
            ),
            eligible AS (
                SELECT
                    c.subject_id,
                    c.study_id,
                    c.label_state,
                    c.split,
                    m.waveform_path,
                    f.signal_quality_score
                FROM read_parquet('{cohort}') c
                JOIN read_parquet('{master}') m
                  USING (subject_id, study_id, split)
                JOIN feature_quality f
                  USING (subject_id, study_id, split)
                WHERE c.label_id = ?
                  AND c.split = ?
                  AND c.label_state IN ('positive', 'negative')
                  AND c.eligibility_state = 'eligible'
                  AND f.extraction_status = 'ok'
                  AND f.signal_quality_usable
                  AND f.signal_quality_score >= ?
            )
            SELECT *
            FROM eligible
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY subject_id
                ORDER BY study_id
            ) = 1
            """,
            [
                options.endpoint_id,
                options.split,
                options.quality_minimum,
            ],
        ).fetchdf()
    if frame.empty:
        raise RuntimeError(
            "No eligible high-quality development ECGs are available for "
            f"{options.endpoint_id!r} in {options.split!r}."
        )
    if set(frame["split"].astype(str)) != {options.split}:
        raise RuntimeError("Group VCG candidate query crossed split boundaries")
    if frame["subject_id"].duplicated().any():
        raise RuntimeError("Group VCG candidate query returned repeated subjects")
    return frame


def _deterministic_candidate_pool(
    candidates: pd.DataFrame,
    options: GroupVCGOptions,
    *,
    random_seed: int,
) -> pd.DataFrame:
    frame = candidates.copy()
    frame["_rank_hash"] = [
        hashlib.sha256(
            (
                f"{options.endpoint_id}|{int(subject_id)}|{int(study_id)}|"
                f"{options.split}|{random_seed}"
            ).encode("ascii")
        ).hexdigest()
        for subject_id, study_id in zip(
            frame["subject_id"],
            frame["study_id"],
        )
    ]
    pool_size = max(
        options.max_per_class * options.candidate_multiplier,
        options.max_per_class + 20,
    )
    selected_frames = []
    for state in ("positive", "negative"):
        group = (
            frame.loc[frame["label_state"].eq(state)]
            .sort_values(
                ["_rank_hash", "subject_id", "study_id"],
                kind="mergesort",
            )
            .head(pool_size)
            .copy()
        )
        group["selected_rank"] = np.arange(1, len(group) + 1)
        selected_frames.append(group)
    selected = pd.concat(selected_frames, ignore_index=True)
    return selected.drop(columns=["_rank_hash"])


def _load_representative_vcg(
    row,
    options: GroupVCGOptions,
    *,
    powerline_hz: float,
) -> tuple[np.ndarray, int]:
    loader = MimicWaveformLoader(target_sampling_rate=500)
    loaded = loader.load(
        row.waveform_path,
        expected_subject_id=int(row.subject_id),
        expected_study_id=int(row.study_id),
    )
    if (
        not loaded.quality.usable
        or loaded.quality.score < options.quality_minimum
    ):
        raise ValueError(
            f"Waveform quality {loaded.quality.score:.3f} is below the "
            f"required {options.quality_minimum:.3f}"
        )
    paths = loader.prepare_paths(
        loaded,
        powerline_hz=powerline_hz,
    )
    r_peaks = detect_r_peaks_multilead(
        paths["delineation"],
        loaded.sampling_rate,
    )
    if len(r_peaks) < 2:
        raise ValueError(f"Only {len(r_peaks)} R peak(s) detected")
    delineation_vcg = VCGTransformRegistry.transform(
        paths["delineation"],
        "kors",
    )
    delineation = FusionDelineator().delineate_record(
        ecg_12lead=paths["delineation"],
        r_peaks=r_peaks,
        fs=loaded.sampling_rate,
        ecg_id=int(row.study_id),
        vcg_xyz=delineation_vcg.xyz,
    )
    accepted_peaks = np.asarray(
        [
            beat.r_peak.sample
            for beat in delineation.accepted_beats
        ],
        dtype=int,
    )
    median_beat = _median_r_aligned_beat(
        paths["morphology"],
        accepted_peaks,
        fs=loaded.sampling_rate,
        pre_r_seconds=options.pre_r_seconds,
        post_r_seconds=options.post_r_seconds,
    )
    pre_samples = int(round(options.pre_r_seconds * loaded.sampling_rate))
    baseline_end = max(1, pre_samples - int(round(0.10 * loaded.sampling_rate)))
    baseline = np.median(median_beat[:baseline_end], axis=0)
    median_beat = median_beat - baseline
    vcg = VCGTransformRegistry.transform(median_beat, "kors").xyz
    if not np.all(np.isfinite(vcg)):
        raise ValueError("Representative VCG contains non-finite values")
    return vcg, int(len(accepted_peaks))


def _median_r_aligned_beat(
    ecg: np.ndarray,
    r_peaks: np.ndarray,
    *,
    fs: int,
    pre_r_seconds: float,
    post_r_seconds: float,
) -> np.ndarray:
    pre = int(round(pre_r_seconds * fs))
    post = int(round(post_r_seconds * fs))
    beats = []
    for peak in np.asarray(r_peaks, dtype=int):
        start = int(peak) - pre
        stop = int(peak) + post
        if start >= 0 and stop <= len(ecg):
            beats.append(np.asarray(ecg[start:stop], dtype=float))
    if len(beats) < 2:
        raise ValueError(
            "Fewer than two accepted beats fit the complete R-aligned window"
        )
    return np.median(np.stack(beats, axis=0), axis=0)


def _bootstrap_group_medians(
    stack: np.ndarray,
    *,
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty(
        (bootstrap_iterations, stack.shape[1], stack.shape[2]),
        dtype=np.float32,
    )
    for iteration in range(bootstrap_iterations):
        indices = rng.integers(0, stack.shape[0], size=stack.shape[0])
        output[iteration] = np.median(stack[indices], axis=0)
    return output


def _load_group_checkpoint(
    path: Path,
    *,
    input_hash: str,
    execution_id: Optional[str],
    force: bool,
    expected_samples: int,
) -> tuple[dict[str, list[np.ndarray]], list[dict], set[int]]:
    empty = ({"positive": [], "negative": []}, [], set())
    if not path.is_file():
        return empty
    if force and execution_id is None:
        path.unlink(missing_ok=True)
        return empty
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(
                bytes(payload["metadata_json"].tolist()).decode("utf-8")
            )
            if metadata.get("input_hash") != input_hash:
                path.unlink(missing_ok=True)
                return empty
            checkpoint_execution = metadata.get("execution_id")
            if checkpoint_execution != execution_id:
                path.unlink(missing_ok=True)
                return empty
            loops = {}
            for state in ("positive", "negative"):
                values = np.asarray(payload[state], dtype=float)
                if values.ndim != 3 or values.shape[1:] != (
                    expected_samples,
                    3,
                ):
                    raise ValueError(
                        f"Invalid {state} checkpoint shape: {values.shape}"
                    )
                loops[state] = [item for item in values]
            audit_rows = list(metadata.get("audit_rows", ()))
            completed = {
                int(item)
                for item in metadata.get("completed_study_ids", ())
            }
            if completed != {
                int(item["study_id"]) for item in audit_rows
            }:
                raise ValueError(
                    "Checkpoint study IDs differ from its case audit"
                )
            return loops, audit_rows, completed
    except Exception:
        path.unlink(missing_ok=True)
        return empty


def _atomic_write_group_checkpoint(
    path: Path,
    *,
    input_hash: str,
    execution_id: Optional[str],
    loops_by_state: dict[str, list[np.ndarray]],
    audit_rows: list[dict],
    completed_study_ids: set[int],
    expected_samples: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_shape = next(
        (
            loop.shape
            for state in ("positive", "negative")
            for loop in loops_by_state[state]
        ),
        (expected_samples, 3),
    )
    arrays = {
        state: (
            np.stack(loops_by_state[state], axis=0).astype(np.float32)
            if loops_by_state[state]
            else np.empty((0, *sample_shape), dtype=np.float32)
        )
        for state in ("positive", "negative")
    }
    metadata = {
        "input_hash": input_hash,
        "execution_id": execution_id,
        "audit_rows": audit_rows,
        "completed_study_ids": sorted(completed_study_ids),
    }
    metadata_bytes = np.frombuffer(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        dtype=np.uint8,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=metadata_bytes,
                positive=arrays["positive"],
                negative=arrays["negative"],
            )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _safe_component(value: str) -> str:
    safe = _SAFE_COMPONENT.sub("_", str(value).strip()).strip("._")
    if not safe:
        raise ValueError(f"Unsafe empty path component derived from {value!r}")
    return safe


def group_vcg_implementation_hash() -> str:
    package_root = Path(__file__).resolve().parents[1]
    core_root = package_root / "core"
    return content_hash(
        {
            "group_vcg": file_sha256(__file__),
            "signals": file_sha256(Path(__file__).with_name("signals.py")),
            "preprocessing": file_sha256(core_root / "preprocessing.py"),
            "vcg_transform": file_sha256(core_root / "vcg_transform.py"),
            "delineation": source_tree_hash(core_root / "delineation"),
        }
    )


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")
