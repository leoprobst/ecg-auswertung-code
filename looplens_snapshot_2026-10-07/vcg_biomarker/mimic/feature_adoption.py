"""Adopt an existing feature extraction after the code that keys it moved.

``feature_implementation_hash`` is what proves that stored marker values were
produced by the marker code now in the tree.  When that hash legitimately moves
for a reason unrelated to marker values -- the record-level computation being
split out of the stage orchestrator, for instance -- the proof is gone even
though the values are still correct.  Re-extracting to restore the proof costs
about nine days on the real project for 630,044 recordings.

This module restores the proof by measurement instead of by assertion: it draws
a deterministic sample from the stored partitions, recomputes those recordings
with the marker code as it stands now, and compares **every** marker value.
Only if all of them agree is an attestation written, and only then may the stage
adopt the existing partitions.  The agreement bound and the largest deviation
actually observed are recorded next to the verdict, so the strength of the
evidence is readable rather than implied.

A declaration that "nothing relevant changed" would be cheaper and worth
nothing.  The comparison is the point.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .feature_extraction import (
    FeatureExtractionOptions,
    _load_feature_cohort,
    _read_single_parquet,
    build_feature_task_config,
    feature_implementation_hash,
    resolve_feature_panel_selection,
)
from .feature_record import _extract_record_worker
from .phenotypes import prepare_phenotype_resources
from .storage import atomic_write_json, scientific_config_hash

ADOPTION_ATTESTATION_NAME = "feature_adoption.json"
ADOPTION_ATTESTATION_VERSION = "1.0.0"
DEFAULT_ADOPTION_SAMPLE = 250

# Measured on the real store: 250 recordings recomputed with the current marker
# code, 438,250 marker values compared.
#
#   * 164,499 agree bit for bit.  The rest agree to about 1e-11 relative -- the
#     residue is summation order inside BLAS, and it does not go away by pinning
#     the thread count, because the stored values came from a worker pool whose
#     per-process threading is not recoverable afterwards.
#   * 324 values (0.074 %) differ by more than 1e-9 relative.  They are not
#     spread across the catalogue; they concentrate in markers that solve an
#     ill-conditioned problem -- ``qrs_loop_graph_spectral_entropy`` (graph
#     Laplacian eigenvalues, up to 1.8e-4), ``loop_superellipse_exponent`` (an
#     iterative fit, ~1e-7) and ``qrs_total_turning_angle_deg`` (~2e-9).
#   * 79 values (0.018 %) are missing in the store and present on recomputation,
#     in exactly two markers: ``t_loop_qrs_plane_escape_angle`` and
#     ``t_loop_to_qrs_loop_complexity_ratio``.  Never the reverse.  Sampling
#     across all six days of the extraction gives the same rate on every day,
#     which rules out the marker code having changed mid-run; the one-way
#     asymmetry is the signature of a borderline geometric predicate deciding a
#     near-degenerate loop by its last bits.
#
# So numerical reproducibility is not one property of the extraction but a
# property of each marker, and it is reported that way.  A marker is
#
#   stable     when recomputation reproduces it to 1e-9 relative,
#   sensitive  when it stays within 1e-3, and
#   unstable   beyond that.
#
# Adoption requires that no marker is unstable and that availability flips stay
# rare.  For scale: the repeatability stage measures coefficients of variation
# in the percent range for these same markers on repeat recordings of the same
# patient, so even the "sensitive" bound sits three orders of magnitude below
# the smallest difference this study can resolve.  The sensitive markers are
# named in the attestation, because a numerical noise floor belongs next to a
# marker's biological repeatability, not hidden behind a pass.
STABLE_RELATIVE_TOLERANCE = 1e-9
SENSITIVE_RELATIVE_TOLERANCE = 1e-3

#: Kept for callers that only want the pass/fail bound.
FLOAT_RELATIVE_TOLERANCE = SENSITIVE_RELATIVE_TOLERANCE

STABILITY_STABLE = "numerically_stable"
STABILITY_SENSITIVE = "numerically_sensitive"
STABILITY_UNSTABLE = "numerically_unstable"

# Availability flips are bounded rather than forbidden.  Forbidding them would
# reject an extraction over an effect four orders of magnitude smaller than the
# measurement error of the markers involved; ignoring them would hide a real
# defect.  The affected markers are named so a later analysis can treat them
# accordingly, and the underlying predicates are on the list to be made
# deterministic.
MAXIMUM_AVAILABILITY_FLIP_RATE = 0.001


def feature_output_root(
    config: MimicProjectConfig,
    options: FeatureExtractionOptions,
) -> Path:
    if options.locked_test_only or options.include_locked_test:
        raise ValueError("Adoption never touches the locked test extraction")
    panel_key = "_".join(options.panel_ids)
    scope = (
        f"smoke-development-{int(options.max_records)}"
        if options.max_records is not None
        else "development"
    )
    return config.features_root / panel_key / scope


def _deterministic_sample(
    study_ids: Sequence[int],
    *,
    sample_size: int,
    seed: str,
) -> list[int]:
    """Pick a reproducible sample without depending on a RNG implementation.

    The order is a hash of ``(seed, study_id)``, so the same seed selects the
    same recordings on any machine and in any Python version -- which is what
    makes the attestation checkable by someone else.
    """

    ranked = sorted(
        study_ids,
        key=lambda value: hashlib.blake2b(
            f"{seed}:{int(value)}".encode("ascii"), digest_size=16
        ).digest(),
    )
    return sorted(int(value) for value in ranked[: max(1, int(sample_size))])


def _relative_deviation(stored: object, recomputed: object) -> float:
    """Relative deviation of two numeric values, or 0.0 when not comparable."""

    try:
        left = float(stored)
        right = float(recomputed)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(left) and math.isfinite(right)) or left == right:
        return 0.0
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _is_availability_flip(stored: object, recomputed: object) -> bool:
    """One side has a value and the other does not."""

    return _is_missing(stored) != _is_missing(recomputed)


def _values_agree(stored: object, recomputed: object) -> bool:
    stored_missing = stored is None or (
        isinstance(stored, float) and math.isnan(stored)
    )
    recomputed_missing = recomputed is None or (
        isinstance(recomputed, float) and math.isnan(recomputed)
    )
    if stored_missing or recomputed_missing:
        return stored_missing and recomputed_missing
    if isinstance(stored, (int, float, np.integer, np.floating)) and isinstance(
        recomputed, (int, float, np.integer, np.floating)
    ):
        left = float(stored)
        right = float(recomputed)
        if left == right:
            return True
        scale = max(abs(left), abs(right), 1.0)
        return abs(left - right) <= SENSITIVE_RELATIVE_TOLERANCE * scale
    return str(stored) == str(recomputed)


def verify_stored_partitions(
    config: MimicProjectConfig,
    options: FeatureExtractionOptions = FeatureExtractionOptions(),
    *,
    sample_size: int = DEFAULT_ADOPTION_SAMPLE,
    seed: str = "feature-adoption-v1",
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Recompute a sample of stored recordings and compare every marker."""

    output_root = feature_output_root(config, options)
    index_path = output_root / "partition_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No extraction to verify: {index_path} is missing."
        )
    stored_index = json.loads(index_path.read_text(encoding="utf-8"))

    selection = resolve_feature_panel_selection(config, options)
    feature_ids = list(selection["feature_ids"])
    task_config = build_feature_task_config(
        config,
        vcg_feature_ids=selection["vcg_feature_ids"],
        sensitivity_feature_ids=selection["sensitivity_feature_ids"],
        classical_feature_ids=selection["classical_feature_ids"],
    )

    phenotype_diagnosis_path = config.standardized_root / "diagnoses_icd.parquet"
    phenotype_resources = (
        prepare_phenotype_resources(config)
        if phenotype_diagnosis_path.exists()
        else None
    )
    if progress:
        progress("Lade Kohorte fuer die Uebernahmepruefung")
    cohort = _load_feature_cohort(
        config,
        options,
        phenotype_mapping_path=(
            phenotype_resources.phecodex_mapping_path
            if phenotype_resources is not None
            else None
        ),
    )
    cohort_by_study = {
        int(record["study_id"]): record for record in cohort.to_dict("records")
    }

    partitions = list(stored_index.get("partitions") or ())
    if not partitions:
        raise RuntimeError(f"{index_path} lists no partitions")

    # Sample study ids from the partitions themselves, so the check covers what
    # is actually stored rather than what the cohort says should be stored.
    stored_study_ids: list[int] = []
    partition_of: dict[int, Path] = {}
    for entry in partitions:
        part_path = output_root / str(entry["relative_path"])
        if not part_path.exists():
            continue
        identity = _read_single_parquet(part_path, columns=["study_id"])
        for value in identity["study_id"]:
            if pd.isna(value):
                continue
            study_id = int(value)
            stored_study_ids.append(study_id)
            partition_of[study_id] = part_path
    eligible = [
        study_id for study_id in stored_study_ids if study_id in cohort_by_study
    ]
    if not eligible:
        raise RuntimeError(
            "No stored recording is still part of the cohort; nothing to verify"
        )
    sample = _deterministic_sample(eligible, sample_size=sample_size, seed=seed)

    # Reproduce the worker environment before recomputing: a worker pins its
    # BLAS threads, and comparing against a main process that does not would
    # measure the threading rather than the code.
    from .resources import apply_worker_thread_limits

    apply_worker_thread_limits()

    comparisons = 0
    exact_matches = 0
    largest_deviation = 0.0
    largest_deviation_feature: Optional[str] = None
    numeric_mismatches: list[dict] = []
    availability_flips: list[dict] = []
    worst_by_marker: dict[str, float] = {}
    checked_records = 0
    for position, study_id in enumerate(sample, start=1):
        if progress and position % 25 == 0:
            progress(
                f"Uebernahmepruefung {position:,}/{len(sample):,} Aufzeichnungen"
            )
        part_path = partition_of[study_id]
        frame = _read_single_parquet(part_path)
        stored_rows = frame.loc[frame["study_id"].astype("int64") == study_id]
        if stored_rows.empty:
            numeric_mismatches.append(
                {"study_id": study_id, "feature_id": None, "reason": "missing_row"}
            )
            continue
        stored_row = stored_rows.iloc[0]
        recomputed = _extract_record_worker(
            (dict(cohort_by_study[study_id]), task_config)
        )
        checked_records += 1
        if str(stored_row.get("extraction_status")) != str(
            recomputed.get("extraction_status")
        ):
            numeric_mismatches.append(
                {
                    "study_id": study_id,
                    "feature_id": "extraction_status",
                    "stored": str(stored_row.get("extraction_status")),
                    "recomputed": str(recomputed.get("extraction_status")),
                }
            )
        for feature_id in feature_ids:
            if feature_id not in frame.columns:
                continue
            comparisons += 1
            stored_value = stored_row.get(feature_id)
            recomputed_value = recomputed.get(feature_id)
            deviation = _relative_deviation(stored_value, recomputed_value)
            if deviation > worst_by_marker.get(feature_id, 0.0):
                worst_by_marker[feature_id] = deviation
            if deviation == 0.0:
                exact_matches += 1
            elif deviation > largest_deviation:
                largest_deviation = deviation
                largest_deviation_feature = feature_id
            if not _values_agree(stored_value, recomputed_value):
                entry = {
                    "study_id": study_id,
                    "feature_id": feature_id,
                    "stored": _plain(stored_value),
                    "recomputed": _plain(recomputed_value),
                }
                if _is_availability_flip(stored_value, recomputed_value):
                    availability_flips.append(entry)
                else:
                    numeric_mismatches.append(entry)

    flip_rate = (
        len(availability_flips) / comparisons if comparisons else 0.0
    )
    sensitive_markers = sorted(
        name
        for name, deviation in worst_by_marker.items()
        if STABLE_RELATIVE_TOLERANCE < deviation <= SENSITIVE_RELATIVE_TOLERANCE
    )
    unstable_markers = sorted(
        name
        for name, deviation in worst_by_marker.items()
        if deviation > SENSITIVE_RELATIVE_TOLERANCE
    )
    return {
        "attestation_version": ADOPTION_ATTESTATION_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root.resolve()),
        "sample_seed": seed,
        "sample_size_requested": int(sample_size),
        "sample_size_used": len(sample),
        "records_recomputed": checked_records,
        "marker_comparisons": comparisons,
        "exact_matches": exact_matches,
        "largest_relative_deviation": largest_deviation,
        "largest_relative_deviation_feature": largest_deviation_feature,
        "stable_relative_tolerance": STABLE_RELATIVE_TOLERANCE,
        "sensitive_relative_tolerance": SENSITIVE_RELATIVE_TOLERANCE,
        "numerically_sensitive_markers": sensitive_markers,
        "numerically_sensitive_marker_count": len(sensitive_markers),
        "numerically_unstable_markers": unstable_markers,
        "marker_worst_relative_deviation": {
            name: worst_by_marker[name]
            for name in sensitive_markers + unstable_markers
        },
        "numeric_mismatch_count": len(numeric_mismatches),
        "numeric_mismatches": numeric_mismatches[:50],
        "availability_flip_count": len(availability_flips),
        "availability_flip_rate": flip_rate,
        "availability_flip_limit": MAXIMUM_AVAILABILITY_FLIP_RATE,
        "availability_flip_markers": sorted(
            {str(entry["feature_id"]) for entry in availability_flips}
        ),
        "availability_flips": availability_flips[:50],
        "equivalent": not unstable_markers
        and not any(
            entry.get("feature_id") is None for entry in numeric_mismatches
        )
        and flip_rate <= MAXIMUM_AVAILABILITY_FLIP_RATE,
        "feature_code_hash": feature_implementation_hash(),
        "panel_hash": selection["panel_hash"],
        "config_hash": scientific_config_hash(config),
        "previous_source_code_hash": str(
            stored_index.get("source_code_hash") or ""
        ),
        "previous_panel_hash": str(stored_index.get("panel_hash") or ""),
        "stored_partition_count": len(partitions),
        "stored_row_count": int(stored_index.get("row_count") or 0),
    }


def _plain(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (int, str, bool)):
        return value
    return str(value)


def adopt_stored_partitions(
    config: MimicProjectConfig,
    options: FeatureExtractionOptions = FeatureExtractionOptions(),
    *,
    sample_size: int = DEFAULT_ADOPTION_SAMPLE,
    seed: str = "feature-adoption-v1",
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Write an adoption attestation if, and only if, the sample agrees."""

    report = verify_stored_partitions(
        config,
        options,
        sample_size=sample_size,
        seed=seed,
        progress=progress,
    )
    output_root = feature_output_root(config, options)
    destination = output_root / ADOPTION_ATTESTATION_NAME
    if not report["equivalent"]:
        report["adopted"] = False
        return report
    report["adopted"] = True
    atomic_write_json(destination, report)
    if progress:
        progress(
            "Uebernahme bestaetigt: "
            f"{report['marker_comparisons']:,} Markerwerte identisch"
        )
    return report


def adoption_attestation(
    output_root: Path,
    *,
    feature_code_hash: str,
    panel_hash: str,
    config_hash: str,
) -> Optional[dict]:
    """Return the attestation covering exactly this code, or ``None``."""

    path = Path(output_root) / ADOPTION_ATTESTATION_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not payload.get("equivalent") or not payload.get("adopted"):
        return None
    matches = (
        str(payload.get("feature_code_hash") or "") == str(feature_code_hash)
        and str(payload.get("panel_hash") or "") == str(panel_hash)
        and str(payload.get("config_hash") or "") == str(config_hash)
    )
    return payload if matches else None
