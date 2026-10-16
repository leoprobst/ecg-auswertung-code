"""How far a VCG marker depends on the transform it was derived with.

A twelve-lead electrocardiogram is not a vectorcardiogram. The X, Y and Z
leads are reconstructed by a linear transform, and the two in common use do not
agree. The published spatial QRS-T angle of a healthy adult is 35-51 degrees
measured on real Frank leads, 44-65 with Kors and 66-81 with inverse Dower --
a systematic offset of roughly twenty degrees that comes from the arithmetic
and not from the heart.

The actual sampled record/patient counts must come from the current artifact,
not a historical store size. Bias and limits of agreement describe differences;
an exploratory regression is not a validated clinical conversion rule.

**Why this is a guideline component and not a curiosity.** A threshold is a
number on a scale. If the scale moves by twenty degrees between two accepted
derivations, then a threshold published without its transform cannot be
applied by anyone, including this project. Everything the reporting set emits
therefore carries the transform, and this table is what makes a value measured
under one transform readable under the other.

Two things this deliberately does not do. It does not pick a winner -- Kors is
closer to Frank in the literature and is this project's primary, but a
conversion table is useful precisely because other groups use the other one.
And it does not average the two: a mean of two derivations is a third scale
that nobody else can reproduce.

Four of the forty sensitivity markers agree exactly, because they never touch
the reconstructed vector: ``qt_interval_ms``, ``qtc_fridericia_ms``,
``tpeak_tend_ms`` and ``cornell_voltage_surrogate`` are read off the twelve
leads and the delineation. They are reported as ``transform_independent``
rather than as perfect agreement, which is a different statement -- one says
the transform does not matter, the other says it was never involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import hashlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .patient_identity import known_patient_mask
from .panels import STUDY_PANEL_IDS, TRANSFORM_INDEPENDENT_FEATURE_IDS
from .storage import (
    StageManifest,
    atomic_write_parquet,
    content_hash,
    development_feature_glob,
    scientific_config_hash,
)


TRANSFORM_AGREEMENT_VERSION = "1.3.0"


def _sample_training_partitions(pattern: str, sample_partitions: int) -> list[Path]:
    """Resolve only explicitly named training partitions; never open locked data."""
    if isinstance(sample_partitions, bool) or not isinstance(sample_partitions, int) or sample_partitions < 1:
        raise ValueError("sample_partitions must be a positive integer")
    training_pattern = str(pattern).replace("\\", "/").replace("split=*", "split=train")
    files = []
    for name in sorted(glob.glob(training_pattern)):
        path = Path(name).resolve()
        splits = [part for part in path.parts if part.startswith("split=")]
        if splits != ["split=train"] or not path.name.startswith("part-") or path.suffix != ".parquet":
            raise ValueError("Transform agreement requires explicit split=train feature partitions")
        files.append(path)
    if not files:
        raise RuntimeError("No training feature partitions found; no validation/locked fallback allowed")
    return files[::max(1, len(files) // sample_partitions)][:sample_partitions]


def _partition_content_identity(paths: Sequence[Path]) -> list[dict[str, str]]:
    identities = []
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        identities.append({"path": str(path), "sha256": digest})
    return identities

# Bland and Altman's interval: the range within which 95 % of the differences
# between two methods fall, which is what a reader needs in order to know
# whether the two are interchangeable for their purpose.
LIMITS_OF_AGREEMENT_Z = 1.96

# Below this many paired values a bias and a limit of agreement are not worth
# printing.
MINIMUM_PAIRS = 200

_COLUMNS = (
    "marker_id",
    "primary_transform",
    "comparison_transform",
    "paired_count",
    "index_record_count",
    "extraction_ok_count",
    "paired_availability_fraction",
    "transform_independent",
    "numerically_identical",
    "clinical_interchangeability_established",
    "conversion_validated",
    "agreement_claim_scope",
    "uncertainty_basis",
    "spearman_rho",
    "pearson_r",
    "mean_primary",
    "mean_comparison",
    "bias",
    "bias_ci_low",
    "bias_ci_high",
    "difference_sd",
    "limit_of_agreement_low",
    "limit_of_agreement_high",
    "relative_bias",
    "conversion_slope",
    "conversion_intercept",
    "agreement_class",
    "status",
    "agreement_version",
)


@dataclass(frozen=True)
class AgreementPolicy:
    minimum_pairs: int = MINIMUM_PAIRS
    # Descriptive rank-correlation bins only, never agreement acceptance gates.
    excellent_rho: float = 0.95
    usable_rho: float = 0.80

    def to_dict(self) -> dict:
        return {
            "minimum_pairs": int(self.minimum_pairs),
            "excellent_rho": float(self.excellent_rho),
            "usable_rho": float(self.usable_rho),
        }


def _classify(rho: float, independent: bool, policy: AgreementPolicy) -> str:
    if independent:
        return "transform_independent"
    if not np.isfinite(rho):
        return "not_assessable"
    if rho >= policy.excellent_rho:
        return "high_rank_correlation_agreement_unproven"
    if rho >= policy.usable_rho:
        return "moderate_rank_correlation_agreement_unproven"
    return "low_rank_correlation_agreement_unproven"


def agreement_for_pair(
    marker_id: str,
    primary: np.ndarray,
    comparison: np.ndarray,
    *,
    primary_transform: str,
    comparison_transform: str,
    policy: AgreementPolicy = AgreementPolicy(),
    independent_pairs: bool = False,
) -> dict:
    """Bias, limits of agreement and a conversion for one marker."""

    from scipy import stats

    primary = np.asarray(primary, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if primary.ndim != 1 or comparison.ndim != 1 or primary.shape != comparison.shape:
        raise ValueError("Agreement requires aligned one-dimensional paired measurements")
    both = np.isfinite(primary) & np.isfinite(comparison)
    primary = primary[both]
    comparison = comparison[both]
    row = {
        "marker_id": marker_id,
        "primary_transform": primary_transform,
        "comparison_transform": comparison_transform,
        "paired_count": int(primary.size),
        "transform_independent": False,
        "numerically_identical": False,
        "clinical_interchangeability_established": False,
        "conversion_validated": False,
        "agreement_claim_scope": "descriptive_pair_differences_no_prespecified_clinical_tolerance",
        "uncertainty_basis": "declared_independent_pairs" if independent_pairs else "independence_unverified_no_bias_ci",
        "agreement_version": TRANSFORM_AGREEMENT_VERSION,
    }
    if primary.size < policy.minimum_pairs:
        row.update(
            {
                "status": "too_few_pairs",
                "agreement_class": "not_assessable",
                "spearman_rho": float("nan"),
                "pearson_r": float("nan"),
                "mean_primary": float("nan"),
                "mean_comparison": float("nan"),
                "bias": float("nan"),
                "bias_ci_low": float("nan"),
                "bias_ci_high": float("nan"),
                "difference_sd": float("nan"),
                "limit_of_agreement_low": float("nan"),
                "limit_of_agreement_high": float("nan"),
                "relative_bias": float("nan"),
                "conversion_slope": float("nan"),
                "conversion_intercept": float("nan"),
            }
        )
        return row

    identical = bool(np.array_equal(primary, comparison))
    independent = identical and marker_id in TRANSFORM_INDEPENDENT_FEATURE_IDS
    difference = comparison - primary
    bias = float(np.mean(difference))
    difference_sd = float(np.std(difference, ddof=1)) if primary.size > 1 else 0.0
    standard_error = (
        difference_sd / np.sqrt(primary.size) if primary.size > 1 else 0.0
    )
    if np.std(primary) <= 0 or np.std(comparison) <= 0:
        rho = pearson = float("nan")
    elif identical:
        rho = pearson = 1.0
    else:
        rho = float(stats.spearmanr(primary, comparison).statistic)
        pearson = float(np.corrcoef(primary, comparison)[0, 1])
    if np.std(primary) <= 0:
        slope, intercept = float("nan"), float("nan")
    elif identical:
        slope, intercept = 1.0, 0.0
    else:
        slope, intercept = np.polyfit(primary, comparison, 1)
    scale = float(np.mean(np.abs(primary))) if primary.size else float("nan")
    row.update(
        {
            "transform_independent": independent,
            "numerically_identical": identical,
            "spearman_rho": rho,
            "pearson_r": pearson,
            "mean_primary": float(np.mean(primary)),
            "mean_comparison": float(np.mean(comparison)),
            "bias": bias,
            "bias_ci_low": bias - LIMITS_OF_AGREEMENT_Z * standard_error if independent_pairs else np.nan,
            "bias_ci_high": bias + LIMITS_OF_AGREEMENT_Z * standard_error if independent_pairs else np.nan,
            "difference_sd": difference_sd,
            "limit_of_agreement_low": bias - LIMITS_OF_AGREEMENT_Z * difference_sd,
            "limit_of_agreement_high": bias + LIMITS_OF_AGREEMENT_Z * difference_sd,
            "relative_bias": (
                bias / scale if np.isfinite(scale) and scale > 0 else float("nan")
            ),
            "conversion_slope": float(slope),
            "conversion_intercept": float(intercept),
            "agreement_class": _classify(rho, independent, policy),
            "status": "measured",
        }
    )
    return row


def agreement_table(
    frame: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    primary_transform: str,
    comparison_transform: str,
    policy: AgreementPolicy = AgreementPolicy(),
) -> pd.DataFrame:
    independent_pairs = False
    if {"subject_id", "study_id"} <= set(frame):
        if frame.duplicated(["subject_id", "study_id"]).any():
            raise ValueError("Ambiguous duplicate patient/study pairs in method comparison")
        valid_ids = known_patient_mask(frame.subject_id) & known_patient_mask(frame.study_id)
        if valid_ids.all():
            # Select before QC/marker missingness; never prefer the record
            # giving better method agreement. This is within the sampled files.
            frame = frame.sort_values(["subject_id", "study_id"], kind="stable").drop_duplicates("subject_id")
            independent_pairs = True
    index_record_count = len(frame)
    if "extraction_status" in frame:
        frame = frame.loc[frame.extraction_status.eq("ok")]
    rows = []
    for marker_id in marker_ids:
        twin = f"sensitivity_{comparison_transform}__{marker_id}"
        if marker_id not in frame.columns or twin not in frame.columns:
            continue
        row = agreement_for_pair(
                marker_id,
                pd.to_numeric(frame[marker_id], errors="coerce").to_numpy(float),
                pd.to_numeric(frame[twin], errors="coerce").to_numpy(float),
                primary_transform=primary_transform,
                comparison_transform=comparison_transform,
                policy=policy,
                independent_pairs=independent_pairs,
        )
        row.update(index_record_count=index_record_count,
            extraction_ok_count=len(frame) if "extraction_status" in frame else np.nan,
            paired_availability_fraction=row["paired_count"] / index_record_count if index_record_count else np.nan)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(_COLUMNS))
    table = pd.DataFrame(rows)
    for column in _COLUMNS:
        if column not in table.columns:
            table[column] = np.nan
    return table[list(_COLUMNS)].sort_values("marker_id").reset_index(drop=True)


def run_transform_agreement(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    feature_glob: Optional[str] = None,
    policy: AgreementPolicy = AgreementPolicy(),
    sample_partitions: int = 60,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Write descriptive method agreement; no validated clinical conversion."""

    import pyarrow.parquet as pq

    output_root = config.runs_root / run_id / "guideline"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "transform_agreement.manifest.json"

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    comparison_transforms = tuple(config.sensitivity_vcg_methods)
    chosen = _sample_training_partitions(resolved_glob, sample_partitions)
    input_identity = _partition_content_identity(chosen)
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "policy": policy.to_dict(),
            "sample_partitions": int(sample_partitions),
            "primary": str(config.primary_vcg_method),
            "comparisons": list(comparison_transforms),
            "selected_partition_content": input_identity,
            "implementation": _partition_content_identity([Path(__file__).resolve()]),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash, config_hash, TRANSFORM_AGREEMENT_VERSION
        ):
            return previous

    available = set(pq.ParquetFile(chosen[0]).schema_arrow.names)
    tables: list[pd.DataFrame] = []
    for comparison in comparison_transforms:
        prefix = f"sensitivity_{comparison}__"
        markers = sorted(
            name[len(prefix) :]
            for name in available
            if name.startswith(prefix)
        )
        markers = [name for name in markers if name in available]
        if not markers:
            continue
        columns = ["subject_id", "study_id", "extraction_status"] + [
            name
            for marker in markers
            for name in (marker, f"{prefix}{marker}")
        ]
        columns = [name for name in dict.fromkeys(columns) if name in available]
        if progress:
            progress(
                f"Transformvergleich {comparison}: {len(markers)} Marker, "
                f"{len(chosen)} Partitionen"
            )
        frames = [
            pq.ParquetFile(path).read(columns=columns).to_pandas()
            for path in chosen
        ]
        frame = pd.concat(frames, ignore_index=True)
        tables.append(
            agreement_table(
                frame,
                markers,
                primary_transform=str(config.primary_vcg_method),
                comparison_transform=str(comparison),
                policy=policy,
            )
        )

    table = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=list(_COLUMNS))
    )
    # Numerical equality cannot establish structural transform independence.
    if not table.empty:
        declared = table["marker_id"].isin(TRANSFORM_INDEPENDENT_FEATURE_IDS)
        table["declared_transform_independent"] = declared
        table["declaration_matches_measurement"] = (
            declared == table["numerically_identical"].astype(bool)
        )

    agreement_path = output_root / "transform_agreement.parquet"
    if _partition_content_identity(chosen) != input_identity:
        raise RuntimeError("Training feature partitions changed during transform agreement")
    atomic_write_parquet(table, agreement_path)

    measured = table[table["status"] == "measured"] if not table.empty else table
    manifest = StageManifest.complete(
        stage="transform-agreement",
        stage_version=TRANSFORM_AGREEMENT_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(str(agreement_path.resolve()),),
        row_counts={
            "marker_transform_pairs": int(len(table)),
            "measured_pairs": int(len(measured)),
            "transform_independent": (
                int(table["transform_independent"].astype(bool).sum())
                if not table.empty
                else 0
            ),
            "clinical_interchangeability_unproven": (
                int((~table["clinical_interchangeability_established"].astype(bool)).sum())
                if not table.empty
                else 0
            ),
            "declaration_mismatches": (
                int((~table["declaration_matches_measurement"]).sum())
                if not table.empty
                and "declaration_matches_measurement" in table.columns
                else 0
            ),
        },
        notes=(
            f"primary_transform={config.primary_vcg_method}",
            f"comparison_transforms={','.join(comparison_transforms)}",
            f"partitions_sampled={len(chosen)}",
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "AgreementPolicy",
    "LIMITS_OF_AGREEMENT_Z",
    "MINIMUM_PAIRS",
    "TRANSFORM_AGREEMENT_VERSION",
    "agreement_for_pair",
    "agreement_table",
    "run_transform_agreement",
]
