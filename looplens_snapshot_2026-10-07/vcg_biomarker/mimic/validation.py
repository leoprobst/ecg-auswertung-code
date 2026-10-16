from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from ..core.biomarkers import BiomarkerRegistry
from .adjusted_models import (
    anchor_year_group_start,
    fit_endpoint_adjusted_models,
    fit_repeated_ecg_clustered_marker,
)
from .baseline_registry import BaselineRegistry
from .config import MimicProjectConfig
from .criteria import (
    build_endpoint_criterion_catalog,
    finalize_criterion_catalog_evidence,
    selected_single_marker_ids,
)
from .fiducial_limitation import load_limitation_policy
from .cross_fitting import (
    CROSS_FITTING_VERSION,
    FOLD_A,
    FOLD_B,
    assert_folds_are_subject_disjoint,
    combine_folds,
    cross_fit_multiplicity_family,
    cross_fit_plan,
    cross_fit_summary,
)
from .delineation_sensitivity import (
    INTERVAL_CRITERION_SENSITIVITIES,
    apply_device_interval_comparators,
    evaluate_interval_criterion_sensitivity,
)
from .event_equivalence import (
    verify_clinical_event_evidence_provenance,
)
from .feature_roles import (
    FEATURE_ROLE_POLICY_VERSION,
    endpoint_feature_role,
    feature_profile,
    is_combination_candidate,
)
from .feature_extraction import (
    CAPPED_SAMPLING_VERSION,
    current_development_feature_input_hash,
    feature_implementation_hash,
    verify_feature_artifacts,
)
from .label_registry import ENDPOINT_ALIASES, default_label_definitions
from .labels import (
    development_label_attestation_path,
    verify_development_label_attestation,
)
from .methodology_registry import MethodologyRegistry
from .models import FrozenScoreSpec, ScoreReleaseStage, ValidationSpec
from .optimization import grouped_score_search
from .panels import (
    CLASSICAL_FEATURE_IDS,
    _feature_family,
    default_feature_panels,
)
from .phenotypes import prepare_phenotype_resources
from .ptb_transfer import (
    DEFAULT_PTB_TRANSFER_RUN_ID,
    load_ptb_frozen_transfer_scores,
)
from .result_explorer import build_result_explorer_tables
from .run_log import note_reuse, profile_query, profiled_phase
from .partitioned_records import PartitionedRecords, records_frame, flush_record_groups
from .run_integrity import assert_development_run_mutable
from .selection_bias import (
    MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION,
    verification_bias_bounds,
)
from .snapshots import (
    create_artifact_snapshot,
    verify_artifact_snapshot,
)
from .statistics import (
    MarkerModel,
    apply_fdr_by_family,
    apply_frozen_score,
    classification_metrics,
    enforce_inference_policy,
    evaluate_marker,
    fit_frozen_score,
    fit_marker_model,
    inference_status,
    learn_threshold_for_target,
    marker_bootstrap_intervals,
    paired_discrimination_difference_bootstrap,
    paired_operating_point_bootstrap,
)

def _unify_column_types(tables: list[pa.Table]) -> list[pa.Table]:
    """Bring the per-split cache tables to one type per column.

    A marker with no value in one split is stored there as ``null`` (older
    partitions) or ``double`` (the writer's cast), while the other split holds
    ``bool`` or ``double``; ``concat_tables`` refuses to merge those. Every
    column is cast to the widest type that occurs: bool only if every split
    has bool or null, double otherwise. A093's validate-pilot-1000 failed on
    ``classical_qtc_bazett_positive`` for exactly this reason.
    """

    if len(tables) < 2:
        return tables
    names = tables[0].schema.names
    targets: dict[str, pa.DataType] = {}
    for name in names:
        types = [table.schema.field(name).type for table in tables]
        concrete = [item for item in types if not pa.types.is_null(item)]
        if not concrete:
            targets[name] = pa.float64()
        elif all(item == concrete[0] for item in concrete):
            targets[name] = concrete[0]
        elif all(pa.types.is_boolean(item) or pa.types.is_integer(item) or pa.types.is_floating(item) for item in concrete):
            targets[name] = pa.float64()
        else:
            return tables  # let concat_tables report the genuine conflict
    unified = []
    for table in tables:
        columns = []
        for name in names:
            column = table.column(name)
            target = targets[name]
            columns.append(column if column.type == target else column.cast(target))
        unified.append(pa.table(columns, names=names))
    return unified


class PartialValidationStopped(RuntimeError):
    """Raised when the user ended an open-ended validation before it finished.

    Deliberately an exception rather than a "partial" manifest. A manifest is
    the thing later stages consult to decide whether work can be reused, and a
    partial one invites exactly the confusion this project spends most of its
    guards preventing: endpoints that were never examined looking like
    endpoints where nothing was found. The endpoint checkpoints already on
    disk are untouched and complete, so continuing later resumes from them.
    """

    def __init__(
        self,
        message: str,
        *,
        completed_endpoint_ids: tuple[str, ...],
        requested_endpoint_ids: tuple[str, ...],
        output_paths: tuple = (),
    ) -> None:
        super().__init__(message)
        self.completed_endpoint_ids = completed_endpoint_ids
        self.requested_endpoint_ids = requested_endpoint_ids
        self.output_paths = output_paths


from .refinement import (
    ACTIVITY_VALIDATION_REFINEMENT,
    RefinementActivity,
    publish_activity,
    requested_stop_mode,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    feature_parquet_clause,
    file_content_fingerprint,
    file_fingerprint,
    file_sha256,
    operational_config_hash,
    optional_column_select,
    relation_columns,
    scientific_config_hash,
)
from .transfer import build_ptb_mimic_transfer_matrix
from .validation_provenance import validation_implementation_hash


VALIDATION_STAGE_VERSION = "3.35.0"
LOCKED_TEST_MARKER_SHORTLIST_VERSION = "1.1.0"
VALIDATION_INPUT_PAYLOAD_VERSION = "1.7.0"
VALIDATION_FEATURE_CACHE_VERSION = "1.1.0"
VALIDATION_CHECKPOINT_VERSION = "1.11.0"
VALIDATION_MARKER_CHECKPOINT_CHUNK_SIZE = 128
MAXIMUM_VERIFICATION_BOUND_WIDTH = 0.20
# A single marker cannot enter the replicated criterion catalog below 0.60
# validation AUC. A conservative analytic interval wholly inside 0.40--0.60
# therefore proves that a costly cluster bootstrap cannot change that marker's
# eligibility. Strong effects in either direction remain fully bootstrapped.
VALIDATION_BOOTSTRAP_NEGLIGIBLE_AUC_MARGIN = 0.10
# These are not marker candidates, but every marker record depends on them.
# Keep the list explicit so a bounded marker projection cannot silently turn
# quality diagnostics into all-NaN values while the marker itself still fits.
VALIDATION_MARKER_COMMON_FEATURE_COLUMNS = ("signal_quality_score",)
# Attached to every marker chunk when the feature partitions carry them.
# Older extractions do not; their absence only blanks the stratified columns.
VALIDATION_MARKER_OPTIONAL_COMMON_FEATURE_COLUMNS = (
    "rhythm_context",
    # Audit S7: reported beside every 80-150 Hz marker result.
    "context_hf_qrs_averaged_snr_db",
)
# The A117 10k pilot showed why a stratified AUC belongs next to the pooled
# one: the Direct-8-lead QRS-T angle screened at 0,811 for LVEF < 40 % against
# 0,797 for the Fréchet coupling, but it exists in 56 % of recordings -- 80 %
# of them in regular rhythm -- while the Fréchet distance exists in 88 %. On
# the same rows the order reverses (0,805 against 0,828). The pooled AUC of a
# marker that is only measurable in the easy cases is not comparable with the
# pooled AUC of one measurable everywhere; the AUC inside the regular-rhythm
# stratum is. Descriptive only: no family, no p-value.
RHYTHM_STRATUM_REGULAR = "regular"
RHYTHM_STRATUM_MINIMUM_CLASS = 10
VALIDATION_DOWNSTREAM_COMMON_FEATURE_COLUMNS = (
    "signal_quality_score",
    "signal_quality_tier",
)


class _SharedValidationFeatureMatrix:
    """Decode immutable development features once in Arrow column blocks.

    Endpoint frames only copy the rows they need.  Missing feature rows are
    represented by NaNs and never remove a row from the reference-blind
    endpoint universe.
    """

    def __init__(
        self,
        feature_glob: str,
        feature_ids: tuple[str, ...],
        *,
        column_block_size: int = VALIDATION_MARKER_CHECKPOINT_CHUNK_SIZE,
    ) -> None:
        self._paths = tuple(
            Path(path).resolve()
            for path in sorted(glob.glob(feature_glob))
        )
        if not self._paths:
            raise FileNotFoundError(
                f"No validation feature-cache files match {feature_glob!r}"
            )
        self._feature_ids = tuple(feature_ids)
        self._block_size = max(64, min(256, int(column_block_size)))
        self._blocks: dict[int, pa.Table] = {}
        split_parts: list[np.ndarray] = []
        subject_parts: list[np.ndarray] = []
        study_parts: list[np.ndarray] = []
        self._row_counts: list[int] = []
        self._schemas = []
        self._projected_read_count = 0
        self._maximum_projected_columns = 0
        for path in self._paths:
            split = path.parent.name.removeprefix("split=")
            if split not in {"train", "validation"}:
                raise RuntimeError(
                    f"Unexpected shared feature split path: {path}"
                )
            schema = pq.read_schema(path)
            self._schemas.append(schema)
            missing_keys = {
                "subject_id",
                "study_id",
            }.difference(schema.names)
            if missing_keys:
                raise RuntimeError(
                    f"Feature cache {path} lacks keys {sorted(missing_keys)}"
                )
            keys = pq.read_table(
                path,
                columns=["subject_id", "study_id"],
                memory_map=True,
                use_threads=True,
            )
            row_count = len(keys)
            self._row_counts.append(row_count)
            split_parts.append(np.full(row_count, split, dtype=object))
            subject_parts.append(
                keys["subject_id"].combine_chunks().to_numpy(
                    zero_copy_only=False
                )
            )
            study_parts.append(
                keys["study_id"].combine_chunks().to_numpy(
                    zero_copy_only=False
                )
            )
        available = set(self._schemas[0].names)
        for schema in self._schemas[1:]:
            available.intersection_update(schema.names)
        self._available_columns = frozenset(available)
        self._row_index = pd.MultiIndex.from_arrays(
            (
                np.concatenate(split_parts),
                np.concatenate(subject_parts),
                np.concatenate(study_parts),
            ),
            names=("split", "subject_id", "study_id"),
        )
        if self._row_index.has_duplicates:
            duplicates = int(self._row_index.duplicated(keep=False).sum())
            raise RuntimeError(
                "Shared validation feature cache contains duplicate keys: "
                f"{duplicates:,} rows"
            )

    @property
    def decoded_block_count(self) -> int:
        return len(self._blocks)

    @property
    def available_columns(self) -> frozenset[str]:
        return self._available_columns

    @property
    def projected_read_count(self) -> int:
        return self._projected_read_count

    @property
    def maximum_projected_columns(self) -> int:
        return self._maximum_projected_columns

    def selection_mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Attempted records, including extraction failures; not marker validity."""
        keys = pd.MultiIndex.from_frame(frame[["split", "subject_id", "study_id"]])
        return self._row_index.get_indexer(keys) >= 0

    def attach_columns(
        self,
        frame: pd.DataFrame,
        columns: Iterable[str],
    ) -> pd.DataFrame:
        """Attach one bounded feature projection without retaining its table.

        This is the production path for validation.  Marker checkpoints are
        128 columns wide, so retaining all decoded blocks and then copying all
        1,697 columns into every endpoint frame doubled the full matrix in
        memory (more than 18 GB at the current cohort size).  A projection is
        decoded, aligned to the endpoint rows, converted to compact numeric
        arrays and released before the next checkpoint block.
        """

        selected = tuple(dict.fromkeys(str(item) for item in columns))
        if not selected:
            return frame.drop(
                columns=["__shared_feature_placeholder"],
                errors="ignore",
            ).reset_index(drop=True)
        missing = sorted(set(selected) - self._available_columns)
        if missing:
            raise RuntimeError(
                "Shared validation feature cache lacks requested columns: "
                f"{missing[:10]}"
            )
        query_index = pd.MultiIndex.from_frame(
            frame[["split", "subject_id", "study_id"]]
        )
        positions = self._row_index.get_indexer(query_index)
        present = positions >= 0
        tables = [
            pq.read_table(
                path,
                columns=list(selected),
                memory_map=True,
                use_threads=True,
            )
            for path in self._paths
        ]
        table = pa.concat_tables(_unify_column_types(tables))
        taken = (
            table.take(pa.array(positions[present], type=pa.int64()))
            if np.any(present)
            else None
        )
        values_by_column: dict[str, np.ndarray] = {}
        for column in selected:
            column_type = table.schema.field(column).type
            numeric = (
                pa.types.is_boolean(column_type)
                or pa.types.is_integer(column_type)
                or pa.types.is_floating(column_type)
                or pa.types.is_decimal(column_type)
            )
            values: np.ndarray = np.full(
                len(frame),
                np.nan if numeric else None,
                dtype=np.float64 if numeric else object,
            )
            if taken is not None:
                arrow_values = taken[column].combine_chunks()
                if numeric:
                    decoded = arrow_values.to_numpy(zero_copy_only=False)
                    values[present] = pd.to_numeric(decoded, errors="coerce")
                else:
                    values[present] = arrow_values.to_pandas().to_numpy(
                        copy=False
                    )
            values_by_column[column] = values
        self._projected_read_count += 1
        self._maximum_projected_columns = max(
            self._maximum_projected_columns,
            len(selected),
        )
        base = frame.drop(
            columns=["__shared_feature_placeholder", *selected],
            errors="ignore",
        ).reset_index(drop=True)
        return pd.concat(
            (base, pd.DataFrame(values_by_column)),
            axis=1,
            copy=False,
        )

    def feature_values(
        self,
        frame: pd.DataFrame,
        columns: Iterable[str],
    ) -> dict[str, np.ndarray]:
        """Read a bounded projection as aligned arrays for score screening."""

        selected = tuple(dict.fromkeys(str(item) for item in columns))
        attached = self.attach_columns(frame, selected)
        return {
            column: _numeric_feature(attached, column)
            for column in selected
        }

    def _load_block(self, block_index: int) -> pa.Table:
        cached = self._blocks.get(block_index)
        if cached is not None:
            return cached
        start = block_index * self._block_size
        stop = min(len(self._feature_ids), start + self._block_size)
        columns = list(self._feature_ids[start:stop])
        tables = []
        for path, schema in zip(self._paths, self._schemas):
            missing = set(columns).difference(schema.names)
            if missing:
                raise RuntimeError(
                    f"Feature cache {path} lacks requested columns: "
                    f"{sorted(missing)}"
                )
            tables.append(
                pq.read_table(
                    path,
                    columns=columns,
                    memory_map=True,
                    use_threads=True,
                )
            )
        table = pa.concat_tables(_unify_column_types(tables))
        self._blocks[block_index] = table
        return table

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.drop(
                columns=["__shared_feature_placeholder"],
                errors="ignore",
            )
        query_index = pd.MultiIndex.from_frame(
            frame[["split", "subject_id", "study_id"]]
        )
        positions = self._row_index.get_indexer(query_index)
        present = positions >= 0
        output_parts = [
            frame.drop(
                columns=["__shared_feature_placeholder"],
                errors="ignore",
            ).reset_index(drop=True)
        ]
        for block_index, start in enumerate(
            range(0, len(self._feature_ids), self._block_size)
        ):
            table = self._load_block(block_index)
            stop = min(len(self._feature_ids), start + self._block_size)
            block_values: dict[str, np.ndarray] = {}
            if np.any(present):
                taken = table.take(
                    pa.array(positions[present], type=pa.int64())
                )
            else:
                taken = None
            for feature_id in self._feature_ids[start:stop]:
                values: np.ndarray = np.full(
                    len(frame),
                    np.nan,
                    dtype=np.float64,
                )
                if taken is not None:
                    selected = taken[feature_id].combine_chunks().to_numpy(
                        zero_copy_only=False
                    )
                    values[present] = pd.to_numeric(
                        selected,
                        errors="coerce",
                    )
                block_values[feature_id] = values
            output_parts.append(pd.DataFrame(block_values))
        return pd.concat(output_parts, axis=1, copy=False)

NON_CLINICAL_COMBINATION_FAMILIES = frozenset(
    {
        "Manuelle Review",
        "Qualitaet/Elektroden",
        "Signalbasis",
    }
)
NON_CLINICAL_COMBINATION_FEATURES = frozenset(
    {
        "lead_to_vcg_consistency_score",
    }
)


def _finite_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes"}
    if isinstance(value, (int, np.integer)):
        return int(value) == 1
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    return False


def _build_locked_test_marker_shortlist(
    candidates: pd.DataFrame,
    marker_models: Iterable[dict],
    *,
    development_run_id: str,
    snapshot_hash: str,
) -> dict:
    """Materialize the exact marker x endpoint hypotheses allowed in Test.

    This artifact is generated solely from Train/Validation evidence.  Freeze
    and locked evaluation independently revalidate every entry and its model
    hash, so a broad model registry can never silently become the Test scope.
    """
    model_lookup: dict[tuple[str, str, str], dict] = {}
    for raw_model in marker_models:
        model = dict(raw_model)
        key = (
            str(model.get("endpoint_id", "")),
            str(model.get("panel_id", "")),
            str(model.get("feature_id", "")),
        )
        if not all(key):
            raise ValueError("Marker model is missing endpoint, panel, or feature ID")
        if key in model_lookup:
            raise ValueError(f"Duplicate marker model for locked shortlist: {key}")
        model_lookup[key] = model

    entries: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    if not candidates.empty:
        for candidate in candidates.to_dict(orient="records"):
            eligible = (
                str(candidate.get("stage", "")) == "validation_replicated"
                and _strict_bool(candidate.get("locked_test_eligible", False))
                and _strict_bool(
                    candidate.get("research_diagnosis_allowed", False)
                )
                and _strict_bool(
                    candidate.get("core_replication_evidence_passed", False)
                )
                and _strict_bool(
                    candidate.get("project_wide_multiplicity_confirmed", False)
                )
                and _strict_bool(candidate.get("cross_fit_confirmed", False))
                and _strict_bool(
                    candidate.get("cross_fit_evaluated_in_both_folds", False)
                )
                and _strict_bool(
                    candidate.get("cross_fit_direction_consistent", False)
                )
                and _strict_bool(
                    candidate.get("verification_bias_gate_passed", False)
                )
                and _strict_bool(candidate.get("adjustment_confirmed", False))
                and _strict_bool(
                    candidate.get("train_assessability_gate_passed", False)
                )
                and _strict_bool(
                    candidate.get("validation_assessability_gate_passed", False)
                )
                and _strict_bool(
                    candidate.get(
                        "train_missingness_sensitivity_gate_passed", False
                    )
                )
                and _strict_bool(
                    candidate.get(
                        "validation_missingness_sensitivity_gate_passed", False
                    )
                )
                and _strict_bool(candidate.get("freeze_eligible", True))
                and str(candidate.get("analysis_type", "")) == "single_marker"
                and not _strict_bool(candidate.get("exploratory_score", False))
            )
            if not eligible:
                continue
            key = (
                str(candidate.get("endpoint_id", "")),
                str(candidate.get("panel_id", "")),
                str(candidate.get("feature_id", "")),
            )
            if key in seen:
                raise ValueError(f"Duplicate locked-test shortlist entry: {key}")
            candidate_model = model_lookup.get(key)
            if candidate_model is None:
                raise ValueError(
                    "Locked-test eligible criterion has no exact train-fitted "
                    f"marker model: {key}"
                )
            seen.add(key)
            evidence_payload = {
                "criterion_id": str(candidate.get("criterion_id", "")),
                "train_q_value": _finite_or_none(candidate.get("train_q_value")),
                "validation_q_value": _finite_or_none(candidate.get("q_value")),
                "train_global_q_value": _finite_or_none(
                    candidate.get("train_global_q_value")
                ),
                "validation_global_q_value": _finite_or_none(
                    candidate.get("validation_global_q_value")
                ),
                "train_hierarchical_q_value": _finite_or_none(
                    candidate.get("train_hierarchical_q_value")
                ),
                "validation_hierarchical_q_value": _finite_or_none(
                    candidate.get("validation_hierarchical_q_value")
                ),
                "adjustment_model_id": candidate.get("adjustment_model_id"),
                "adjustment_incremental_auc_ci_low": _finite_or_none(
                    candidate.get("adjustment_incremental_auc_ci_low")
                ),
                "train_test_yield": _finite_or_none(
                    candidate.get("train_test_yield")
                ),
                "validation_test_yield": _finite_or_none(
                    candidate.get("validation_test_yield")
                ),
                "train_missingness_auc_lower_bound": _finite_or_none(
                    candidate.get("train_missingness_auc_lower_bound")
                ),
                "validation_missingness_auc_lower_bound": _finite_or_none(
                    candidate.get("validation_missingness_auc_lower_bound")
                ),
                "cross_fit_p_value": _finite_or_none(
                    candidate.get("cross_fit_p_value")
                ),
                "cross_fit_q_value": _finite_or_none(
                    candidate.get("cross_fit_q_value")
                ),
                "cross_fit_global_q_value": _finite_or_none(
                    candidate.get("cross_fit_global_q_value")
                ),
                "cross_fit_hierarchical_q_value": _finite_or_none(
                    candidate.get("cross_fit_hierarchical_q_value")
                ),
                "cross_fit_pooled_auc": _finite_or_none(
                    candidate.get("cross_fit_pooled_auc")
                ),
                "cross_fit_direction_consistent": _strict_bool(
                    candidate.get("cross_fit_direction_consistent", False)
                ),
                "verification_rate": _finite_or_none(
                    candidate.get("verification_rate")
                ),
                "verification_sensitivity_lower_bound": _finite_or_none(
                    candidate.get("verification_sensitivity_lower_bound")
                ),
                "verification_sensitivity_upper_bound": _finite_or_none(
                    candidate.get("verification_sensitivity_upper_bound")
                ),
                "verification_specificity_lower_bound": _finite_or_none(
                    candidate.get("verification_specificity_lower_bound")
                ),
                "verification_specificity_upper_bound": _finite_or_none(
                    candidate.get("verification_specificity_upper_bound")
                ),
            }
            entries.append(
                {
                    "candidate_kind": "single_marker",
                    "endpoint_id": key[0],
                    "panel_id": key[1],
                    "feature_id": key[2],
                    "criterion_id": evidence_payload["criterion_id"],
                    "marker_model_hash": content_hash(candidate_model),
                    "criterion_evidence_hash": content_hash(evidence_payload),
                    "selection_basis": (
                        "bidirectional_out_of_fold_replication_with_"
                        "adjustment_assessability_and_project_multiplicity"
                    ),
                    **evidence_payload,
                }
            )
    entries.sort(
        key=lambda row: (
            row["endpoint_id"],
            row["panel_id"],
            row["feature_id"],
        )
    )
    hash_payload = {
        "shortlist_version": LOCKED_TEST_MARKER_SHORTLIST_VERSION,
        "entries": entries,
    }
    return {
        "shortlist_version": LOCKED_TEST_MARKER_SHORTLIST_VERSION,
        "development_run_id": development_run_id,
        "feature_snapshot_hash": snapshot_hash,
        "selection_source": "bidirectional_cross_fit_train_validation_only",
        "selection_scope": "explicit_single_marker_by_endpoint",
        "locked_test_read": False,
        "entry_count": len(entries),
        "endpoint_count": len({entry["endpoint_id"] for entry in entries}),
        "shortlist_hash": content_hash(hash_payload),
        "entries": entries,
    }


@dataclass(frozen=True)
class DevelopmentValidationOptions:
    run_id: str = "mimic-development"
    endpoint_ids: Optional[tuple[str, ...]] = None
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel")
    feature_glob: Optional[str] = None
    bootstrap_iterations: Optional[int] = None
    include_predefined_scores: bool = True
    include_ptb_frozen_scores: bool = True
    ptb_transfer_run_id: str = DEFAULT_PTB_TRANSFER_RUN_ID
    include_exploratory_scores: bool = True
    include_all_ecg_secondary: bool = True
    grouped_score_folds: int = 5
    grouped_score_repeats: int = 1
    grouped_score_representatives_per_family: int = 2
    grouped_score_selected_pairs: int = 1
    grouped_score_selected_triples: int = 1
    grouped_score_triple_seed_pairs: int = 25
    maximum_component_abs_correlation: float = 0.90
    minimum_score_incremental_auc: float = 0.005
    maximum_criterion_single_candidates: int = 6
    maximum_criterion_combination_candidates: int = 3
    preview: bool = False
    force: bool = False


def _load_ptb_transfer_inputs(
    config: MimicProjectConfig,
    options: DevelopmentValidationOptions,
) -> tuple[list[dict], dict, bool]:
    """Load frozen PTB transfer reference scores without aborting the
    primary MIMIC validation run when the freeze is stale or missing.

    The PTB transfer matrix is a supplementary cross-dataset comparison
    (PTB and MIMIC AUCs are never pooled); a stale or missing PTB freeze
    must not abort the MIMIC train/validation run, which does not depend
    on it.
    """
    if not options.include_ptb_frozen_scores:
        return (
            [],
            {"status": "disabled", "run_id": options.ptb_transfer_run_id},
            False,
        )
    try:
        ptb_scores, ptb_transfer_provenance = load_ptb_frozen_transfer_scores(
            config,
            run_id=options.ptb_transfer_run_id,
        )
        return ptb_scores, ptb_transfer_provenance, True
    except (RuntimeError, FileNotFoundError) as exc:
        return (
            [],
            {
                "status": "stale_freeze",
                "run_id": options.ptb_transfer_run_id,
                "reason": str(exc),
            },
            False,
        )


@dataclass(frozen=True)
class _SingleMarkerEvaluation:
    feature_id: str
    panel_id: str
    model: Optional[MarkerModel] = None
    train_record: Optional[dict] = None
    validation_record: Optional[dict] = None
    reverse_model: Optional[MarkerModel] = None
    reverse_confirmation_record: Optional[dict] = None
    cross_fit_failure: Optional[dict] = None
    stability_records: tuple[dict, ...] = ()
    classical_records: tuple[dict, ...] = ()
    verification_bias_records: tuple[dict, ...] = ()
    failure: Optional[dict] = None


def run_development_validation(
    config: MimicProjectConfig,
    options: DevelopmentValidationOptions = DevelopmentValidationOptions(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    _validate_run_id(options.run_id)
    if (
        options.maximum_criterion_single_candidates < 0
        or options.maximum_criterion_combination_candidates < 0
    ):
        raise ValueError("Criterion candidate limits cannot be negative")
    definitions = {item.label_id: item for item in default_label_definitions()}
    endpoint_ids = (
        tuple(definitions)
        if options.endpoint_ids is None
        else tuple(options.endpoint_ids)
    )
    unknown_endpoints = sorted(set(endpoint_ids) - set(definitions))
    if unknown_endpoints:
        raise ValueError(f"Unknown clinical endpoints: {unknown_endpoints}")

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    unknown_panels = sorted(set(options.panel_ids) - set(panels))
    if unknown_panels:
        raise ValueError(f"Unknown feature panels: {unknown_panels}")
    panel_map = _feature_panel_map(panels, options.panel_ids)
    feature_ids = tuple(panel_map)
    if not feature_ids:
        raise ValueError("No features were selected")
    ptb_scores, ptb_transfer_provenance, ptb_transfer_available = (
        _load_ptb_transfer_inputs(config, options)
    )
    ptb_scores_by_endpoint: dict[str, list[dict]] = {}
    for payload in ptb_scores:
        for endpoint_id in payload.get("mimic_endpoint_ids", ()):
            ptb_scores_by_endpoint.setdefault(str(endpoint_id), []).append(
                payload
            )

    if options.feature_glob is not None:
        feature_glob = options.feature_glob
        feature_scope = "explicit"
    else:
        panel_root = config.features_root / "_".join(options.panel_ids)
        development_glob = (
            panel_root / "development" / "split=*" / "part-*.parquet"
        )
        feature_glob = str(development_glob)
        feature_scope = "development_only"
    feature_paths = sorted(Path(path) for path in glob.glob(feature_glob))
    if not feature_paths:
        raise FileNotFoundError(
            "Development features are missing. Expected files matching "
            f"{feature_glob!r}; run extract-features first."
        )
    feature_integrity = verify_feature_artifacts(feature_paths)
    feature_scope = _resolved_feature_scope(
        feature_scope,
        feature_integrity,
    )
    sampling_design = _resolved_sampling_design(
        feature_scope,
        feature_integrity,
    )
    probability_models_allowed = _probability_models_allowed(
        preview=options.preview,
        sampling_design=sampling_design,
    )
    prevalence_metrics_interpretable = _prevalence_metrics_allowed(
        preview=options.preview,
        sampling_design=sampling_design,
    )
    if feature_scope == "development_preview" and not options.preview:
        raise RuntimeError(
            "Smoke/partial feature artifacts may only be analyzed with the "
            "explicit preview option. Preview results are not freezable."
        )
    if feature_scope == "development_only":
        _verify_current_development_feature_inputs(
            config,
            feature_integrity=feature_integrity,
            panel_ids=options.panel_ids,
        )
    feature_snapshot = create_artifact_snapshot(
        config,
        artifact_kind="development_features",
        source_glob=feature_glob,
        snapshot_id=(
            f"{options.run_id}-features-"
            f"{str(feature_integrity['artifact_hash'])[:16]}"
        ),
        notes=(
            f"validation_run_id={options.run_id}",
            f"feature_scope={feature_scope}",
            f"sampling_design={sampling_design}",
            "locked_test_included=False",
        ),
        require_complete=not options.preview,
    )
    if not options.preview:
        feature_snapshot.assert_freezable()
    cohort_path = config.labels_root / "endpoint_index_cohorts.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    reports_path = config.evidence_root / "machine_report_labels.parquet"
    context_path = config.evidence_root / "clinical_context_features.parquet"
    window_cohort_path = (
        config.labels_root / "endpoint_window_sensitivity_cohorts.parquet"
    )
    all_label_path = config.labels_root / "clinical_label_evidence.parquet"
    required_paths = (cohort_path, master_path, context_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required validation inputs are missing: {missing}")
    label_attestation = (
        verify_development_label_attestation(config)
        if feature_scope == "development_only"
        else None
    )
    event_evidence_provenance = (
        verify_clinical_event_evidence_provenance(config)
        if feature_scope == "development_only"
        else None
    )

    panel_versions = {
        panel_id: panels[panel_id].version for panel_id in options.panel_ids
    }
    split_hash = _read_split_hash(master_path)
    specification = ValidationSpec(
        run_id=options.run_id,
        split_hash=split_hash,
        endpoint_ids=endpoint_ids,
        panel_versions=panel_versions,
        development_bootstrap_iterations=(
            int(options.bootstrap_iterations)
            if options.bootstrap_iterations is not None
            else (200 if options.preview else 1000)
        ),
        random_seed=config.random_seed,
        grouped_score_folds=options.grouped_score_folds,
        grouped_score_repeats=options.grouped_score_repeats,
        maximum_component_abs_correlation=(
            options.maximum_component_abs_correlation
        ),
        minimum_score_incremental_auc=(
            options.minimum_score_incremental_auc
        ),
        grouped_score_representatives_per_family=(
            options.grouped_score_representatives_per_family
        ),
        grouped_score_selected_pairs=options.grouped_score_selected_pairs,
        grouped_score_selected_triples=options.grouped_score_selected_triples,
        grouped_score_triple_seed_pairs=options.grouped_score_triple_seed_pairs,
    )
    iterations = specification.development_bootstrap_iterations
    if options.preview and iterations > 200:
        raise ValueError(
            "Preview validation is limited to 200 subject-cluster "
            "bootstraps. Use a complete development snapshot for 1,000."
        )
    marker_workers = max(1, min(int(config.workers), len(feature_ids)))
    run_root = config.runs_root / options.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "validate.manifest.json"
    code_hash = validation_implementation_hash()
    config_hash = scientific_config_hash(config)
    baseline_registry = BaselineRegistry()
    methodology_registry = MethodologyRegistry()
    validation_input_payload = _build_validation_input_payload(
        config,
        feature_paths=feature_paths,
        feature_integrity=feature_integrity,
        feature_snapshot_hash=feature_snapshot.snapshot_hash,
        feature_scope=feature_scope,
        sampling_design=sampling_design,
        cohort_path=cohort_path,
        all_label_path=all_label_path,
        master_path=master_path,
        context_path=context_path,
        window_cohort_path=window_cohort_path,
        reports_path=reports_path,
        specification=specification,
        panel_map=panel_map,
        options=options,
        ptb_transfer_provenance=ptb_transfer_provenance,
        baseline_registry=baseline_registry,
        methodology_registry=methodology_registry,
        code_hash=code_hash,
        config_hash=config_hash,
        label_attestation=label_attestation,
        event_evidence_provenance=event_evidence_provenance,
    )
    from .voltage_mass_endpoint import ECHO_GEOMETRY_COLUMNS
    geometry_path = config.evidence_root / "echo_measurements_wide.parquet"
    geometry_hash = file_sha256(geometry_path) if geometry_path.exists() else None
    validation_input_payload["voltage_mass_geometry_sha256"] = geometry_hash
    input_hash = content_hash(validation_input_payload)
    if not options.force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            VALIDATION_STAGE_VERSION,
        ):
            return previous
    note_reuse(
        "validate", reused=False,
        reason="validation_execution_required",
        detail="Checkpoint reuse is recorded separately; this is not a cold-run assertion.",
    )
    assert_development_run_mutable(
        run_root,
        operation="Development validation",
    )
    (
        analysis_feature_glob,
        analysis_feature_cache,
        analysis_feature_cache_manifest_path,
    ) = _prepare_validation_feature_cache(
        catalog_path=config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        run_root=run_root,
        feature_integrity=feature_integrity,
        feature_snapshot_hash=feature_snapshot.snapshot_hash,
        progress=progress,
    )
    shared_feature_matrix: Optional[_SharedValidationFeatureMatrix] = None
    feature_matrix_metrics = {
        "strategy": "bounded_checkpoint_projections",
        "column_block_size": VALIDATION_MARKER_CHECKPOINT_CHUNK_SIZE,
        "projected_read_count": 0,
        "maximum_projected_columns": 0,
        "unbounded_arrow_cache": False,
        "reference_blind_left_join": True,
    }

    tabular_spill_root = run_root / "validation_tabular_parts" / input_hash
    marker_records = PartitionedRecords(tabular_spill_root / "markers")
    classical_records: list[dict] = []
    report_records: list[dict] = []
    paired_comparison_records = PartitionedRecords(tabular_spill_root / "paired_comparison_records")
    comparator_sensitivity_records = PartitionedRecords(tabular_spill_root / "comparator_sensitivity_records")
    combination_records: list[dict] = []
    stability_records = PartitionedRecords(tabular_spill_root / "stability_records")
    marker_models: list[dict] = []
    score_specs: list[dict] = []
    adjusted_records: list[dict] = []
    adjusted_model_specs: list[dict] = []
    decision_curve_records = PartitionedRecords(tabular_spill_root / "decision_curve_records")
    calibration_curve_records = PartitionedRecords(tabular_spill_root / "calibration_curve_records")
    performance_curve_records = PartitionedRecords(tabular_spill_root / "performance_curve_records")
    score_distribution_records = PartitionedRecords(tabular_spill_root / "score_distribution_records")
    all_ecg_records: list[dict] = []
    all_ecg_model_specs: list[dict] = []
    criterion_candidate_records: list[dict] = []
    criterion_threshold_records: list[dict] = []
    failures: list[dict] = []
    endpoint_counts: list[dict] = []
    cross_fit_confirmation_records = PartitionedRecords(tabular_spill_root / "confirmations")
    cross_fit_model_records: list[dict] = []
    cross_fit_plan_records: list[dict] = []
    verification_bias_records = PartitionedRecords(tabular_spill_root / "verification")
    lead_comparison_records: list[dict] = []
    lead_model_records: list[dict] = []
    voltage_mass_records: list[dict] = []
    voltage_mass_model_records: list[dict] = []
    voltage_mass_geometry = None
    from .mechanism_hypotheses import mechanism_hypothesis_rows
    from .mechanism_endpoint import INPUT_COLUMNS as MECHANISM_INPUT_COLUMNS
    from .mechanism_endpoint import analyze_mechanism_endpoint, adjust_mechanism_family
    mechanism_hypotheses = mechanism_hypothesis_rows(
        [{"marker_id": name} for name in BiomarkerRegistry.available()])
    mechanism_records: list[dict] = []
    clock_stress_records: list[dict] = []
    result_groups: dict[str, list[dict]] = {
        "clock_stress_records": clock_stress_records,
        "mechanism_records": mechanism_records,
        "lead_comparison_records": lead_comparison_records,
        "lead_model_records": lead_model_records,
        "voltage_mass_records": voltage_mass_records,
        "voltage_mass_model_records": voltage_mass_model_records,
        "marker_records": marker_records,
        "classical_records": classical_records,
        "report_records": report_records,
        "paired_comparison_records": paired_comparison_records,
        "comparator_sensitivity_records": comparator_sensitivity_records,
        "combination_records": combination_records,
        "stability_records": stability_records,
        "marker_models": marker_models,
        "score_specs": score_specs,
        "adjusted_records": adjusted_records,
        "adjusted_model_specs": adjusted_model_specs,
        "decision_curve_records": decision_curve_records,
        "calibration_curve_records": calibration_curve_records,
        "performance_curve_records": performance_curve_records,
        "score_distribution_records": score_distribution_records,
        "all_ecg_records": all_ecg_records,
        "all_ecg_model_specs": all_ecg_model_specs,
        "criterion_candidate_records": criterion_candidate_records,
        "criterion_threshold_records": criterion_threshold_records,
        "failures": failures,
        "endpoint_counts": endpoint_counts,
        "cross_fit_confirmation_records": cross_fit_confirmation_records,
        "cross_fit_model_records": cross_fit_model_records,
        "cross_fit_plan_records": cross_fit_plan_records,
        "verification_bias_records": verification_bias_records,
    }
    checkpoint_root = (
        run_root
        / "validation_checkpoints"
        / f"input-{input_hash[:16]}"
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    completed_endpoint_ids: list[str] = []
    stopped_early = False
    stop_reason = ""
    power_gate = (
        {}
        if options.preview
        else _development_power_gate(catalog, cohort_path, endpoint_ids)
    )
    for endpoint_index, endpoint_id in enumerate(endpoint_ids, start=1):
        endpoint_checkpoint_path = (
            checkpoint_root / f"{endpoint_id}.json"
        )
        checkpoint_payload = _load_endpoint_validation_checkpoint(
            endpoint_checkpoint_path,
            input_hash=input_hash,
            endpoint_id=endpoint_id,
            materialize_partitions=False,
        )
        if checkpoint_payload is not None:
            note_reuse(
                "validate", reused=True, reason="endpoint_checkpoint_restored",
                detail=str(endpoint_id),
            )
            _extend_validation_result_groups(
                result_groups,
                checkpoint_payload["result_groups"],
                partition_paths=checkpoint_payload.get("result_group_paths"),
            )
            completed_endpoint_ids.append(str(endpoint_id))
            flush_record_groups(result_groups)
            del checkpoint_payload
            if progress:
                progress(
                    f"Validierung {endpoint_index:,}/{len(endpoint_ids):,}: "
                    f"{endpoint_id} aus atomarem Checkpoint geladen"
                )
            stop_mode = requested_stop_mode(run_root)
            if stop_mode is not None:
                stopped_early = True
                stop_reason = f"Stopped after restored endpoint {endpoint_id} ({stop_mode})"
                break
            continue
        group_offsets = {
            name: len(records)
            for name, records in result_groups.items()
        }
        if endpoint_id in ENDPOINT_ALIASES:
            # Audit D7: no test of its own; the canonical endpoint carries it.
            canonical, relation = ENDPOINT_ALIASES[endpoint_id]
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "endpoint_alias",
                    "reason": (
                        f"{relation} of {canonical}; "
                        + (
                            "AUC = 1 - AUC of the canonical endpoint; "
                            if relation == "complement"
                            else "same studies and labels; "
                        )
                        + "not screened, not counted in multiplicity"
                    ),
                }
            )
            _write_endpoint_validation_checkpoint(
                endpoint_checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                result_groups=result_groups,
                group_offsets=group_offsets,
            )
            continue
        if endpoint_id in power_gate:
            # Audit D6: an endpoint whose smaller class has fewer than 30
            # development subjects cannot carry an estimate. Screening it
            # cost a full marker pass and added to every multiplicity family.
            positives, negatives = power_gate[endpoint_id]
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "power_gate",
                    "reason": (
                        "no_inference: "
                        f"{positives} positive / {negatives} negative "
                        "development subjects (< "
                        f"{VALIDATION_MINIMUM_CLASS_SUBJECTS} in a class); "
                        "not screened, not counted in multiplicity"
                    ),
                }
            )
            _write_endpoint_validation_checkpoint(
                endpoint_checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                result_groups=result_groups,
                group_offsets=group_offsets,
            )
            if progress:
                progress(
                    f"Validierung {endpoint_index:,}/{len(endpoint_ids):,}: "
                    f"{endpoint_id} unter der Power-Schwelle, übersprungen"
                )
            continue
        if progress:
            progress(
                f"Validierung {endpoint_index:,}/{len(endpoint_ids):,}: "
                f"{endpoint_id}"
            )
        if shared_feature_matrix is None:
            shared_feature_matrix = _SharedValidationFeatureMatrix(
                analysis_feature_glob,
                feature_ids,
            )
        report_label_id = _report_comparator_for_endpoint(endpoint_id)
        endpoint_universe = _load_endpoint_frame(
            catalog,
            cohort_path=cohort_path,
            master_path=master_path,
            reports_path=reports_path if reports_path.exists() else None,
            context_path=context_path,
            feature_glob=analysis_feature_glob,
            endpoint_id=endpoint_id,
            report_label_id=report_label_id,
            feature_columns=(),
            shared_feature_matrix=shared_feature_matrix,
            selected_records_only=options.preview,
        )
        if endpoint_universe.empty:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "endpoint",
                    "reason": "no endpoint-universe rows with extracted features",
                }
            )
            _write_endpoint_validation_checkpoint(
                endpoint_checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                result_groups=result_groups,
                group_offsets=group_offsets,
            )
            continue
        from .reference_clock_stress import clock_stress_counts
        clock_stress_records.extend(clock_stress_counts(endpoint_universe, definitions[endpoint_id]))
        from .lead_endpoint_analysis import LEAD_INPUT_COLUMNS, analyze_endpoint_leads
        lead_universe = shared_feature_matrix.attach_columns(endpoint_universe,
            [name for name in LEAD_INPUT_COLUMNS if name in shared_feature_matrix.available_columns and name not in endpoint_universe])
        lead_rows, lead_models = analyze_endpoint_leads(lead_universe, endpoint_id=endpoint_id,
            iterations=iterations, seed=config.random_seed, preview=options.preview)
        lead_comparison_records.extend(lead_rows)
        lead_model_records.extend(lead_models)
        endpoint_hypotheses = [row for row in mechanism_hypotheses
                               if row["endpoint_id"] == endpoint_id]
        if endpoint_hypotheses:
            requested = (*MECHANISM_INPUT_COLUMNS,
                         *(row["marker_id"] for row in endpoint_hypotheses))
            mechanism_universe = shared_feature_matrix.attach_columns(endpoint_universe,
                [name for name in dict.fromkeys(requested)
                 if name in shared_feature_matrix.available_columns and name not in endpoint_universe])
            mechanism_records.extend(analyze_mechanism_endpoint(
                mechanism_universe, hypotheses=endpoint_hypotheses, preview=options.preview))
            del mechanism_universe
        from .voltage_mass_endpoint import VOLTAGE_MASS_ENDPOINTS, analyze_voltage_mass_endpoint
        from .voltage_mass import VOLTAGE_MARKERS
        if endpoint_id in VOLTAGE_MASS_ENDPOINTS:
            if voltage_mass_geometry is None:
                voltage_mass_geometry = (pd.read_parquet(geometry_path, columns=list(ECHO_GEOMETRY_COLUMNS))
                    if geometry_hash is not None else pd.DataFrame(columns=ECHO_GEOMETRY_COLUMNS))
            anatomy_universe = shared_feature_matrix.attach_columns(lead_universe,
                [name for name in VOLTAGE_MARKERS if name in shared_feature_matrix.available_columns
                 and name not in lead_universe])
            anatomy_rows, anatomy_models = analyze_voltage_mass_endpoint(anatomy_universe,
                voltage_mass_geometry, endpoint_id=endpoint_id, iterations=iterations,
                seed=config.random_seed, preview=options.preview)
            voltage_mass_records.extend(anatomy_rows)
            voltage_mass_model_records.extend(anatomy_models)
            del anatomy_universe
        del lead_universe
        verified_mask = (
            endpoint_universe["eligibility_state"].eq("eligible")
            & endpoint_universe["label_state"].isin(("positive", "negative"))
        )
        frame = endpoint_universe.loc[verified_mask].reset_index(drop=True)
        if frame.empty:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "endpoint",
                    "reason": (
                        "reference-blind endpoint universe contains no verified "
                        "positive/negative rows"
                    ),
                }
            )
            _write_endpoint_validation_checkpoint(
                endpoint_checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                result_groups=result_groups,
                group_offsets=group_offsets,
            )
            continue
        frame["label"] = (frame["label_state"] == "positive").astype(np.int8)
        endpoint_counts.extend(_endpoint_split_counts(endpoint_id, frame))
        train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
        validation = frame.loc[
            frame["split"] == "validation"
        ].reset_index(drop=True)
        if train.empty or validation.empty:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "endpoint",
                    "reason": "train or validation cohort is empty",
                }
            )
            _write_endpoint_validation_checkpoint(
                endpoint_checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                result_groups=result_groups,
                group_offsets=group_offsets,
            )
            continue

        # Both development folds are used in turn for fitting and for an
        # independent out-of-fold confirmation.  Subject leakage is a hard
        # error, and the sealed test split is never passed into this plan.
        assert_folds_are_subject_disjoint(frame)
        endpoint_plan = cross_fit_plan(frame)
        endpoint_plan["endpoint_id"] = endpoint_id
        cross_fit_plan_records.extend(endpoint_plan.to_dict(orient="records"))

        def attach_endpoint_features(
            requested: Iterable[str],
        ) -> None:
            """Add only downstream-selected columns to the verified frame."""

            nonlocal frame, train, validation
            assert shared_feature_matrix is not None
            missing = tuple(
                feature_id
                for feature_id in dict.fromkeys(
                    str(item) for item in requested
                )
                if feature_id not in frame
                and feature_id
                in shared_feature_matrix.available_columns
            )
            if not missing:
                return
            frame = shared_feature_matrix.attach_columns(frame, missing)
            train = frame.loc[frame["split"].eq("train")].reset_index(
                drop=True
            )
            validation = frame.loc[
                frame["split"].eq("validation")
            ].reset_index(drop=True)

        def projected_endpoint_splits(
            requested: tuple[str, ...],
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Return temporary score-component frames without retaining them."""

            assert shared_feature_matrix is not None
            missing = tuple(
                feature_id
                for feature_id in dict.fromkeys(requested)
                if feature_id not in frame
                and feature_id
                in shared_feature_matrix.available_columns
            )
            projected = (
                shared_feature_matrix.attach_columns(frame, missing)
                if missing
                else frame
            )
            return (
                projected.loc[projected["split"].eq("train")]
                .reset_index(drop=True),
                projected.loc[projected["split"].eq("validation")]
                .reset_index(drop=True),
            )

        def score_feature_values(
            requested: tuple[str, ...],
        ) -> dict[str, np.ndarray]:
            assert shared_feature_matrix is not None
            # Read against the current verified frame instead of closing over
            # ``train``.  ``attach_endpoint_features`` may replace all three
            # endpoint frames, and deriving the split here keeps the callback
            # aligned with that latest frame without a fragile sibling-closure
            # dependency.
            current_train = frame.loc[frame["split"].eq("train")].reset_index(
                drop=True
            )
            return shared_feature_matrix.feature_values(
                current_train,
                requested,
            )

        endpoint_models: dict[str, MarkerModel] = {}
        endpoint_train_records: dict[str, dict] = {}
        endpoint_validation_records: dict[str, dict] = {}
        marker_evaluations = _load_or_evaluate_marker_chunks(
            endpoint_id=endpoint_id,
            feature_ids=feature_ids,
            panel_map=panel_map,
            train=train,
            validation=validation,
            endpoint_universe=endpoint_universe,
            shared_feature_matrix=shared_feature_matrix,
            specification=specification,
            bootstrap_iterations=iterations,
            worker_count=marker_workers,
            checkpoint_root=checkpoint_root,
            input_hash=input_hash,
            endpoint_index=endpoint_index,
            endpoint_total=len(endpoint_ids),
            progress=progress,
        )
        for evaluation in marker_evaluations:
            if evaluation.failure is not None:
                failures.append(evaluation.failure)
                continue
            if (
                evaluation.model is None
                or evaluation.train_record is None
                or evaluation.validation_record is None
            ):
                raise RuntimeError(
                    "Marker evaluation returned an incomplete successful result"
                )
            marker_models.append(
                {
                    "endpoint_id": endpoint_id,
                    "panel_id": evaluation.panel_id,
                    **evaluation.model.to_dict(),
                }
            )
            endpoint_models[evaluation.feature_id] = evaluation.model
            endpoint_train_records[evaluation.feature_id] = (
                evaluation.train_record
            )
            endpoint_validation_records[evaluation.feature_id] = (
                evaluation.validation_record
            )
            marker_records.extend(
                (evaluation.train_record, evaluation.validation_record)
            )
            fold_a_record = dict(evaluation.validation_record)
            fold_a_record.update(
                {
                    "cross_fit_fold": FOLD_A,
                    "fitting_split": "train",
                    "confirming_split": "validation",
                    "cross_fitting_version": CROSS_FITTING_VERSION,
                    "locked_test_read": False,
                }
            )
            cross_fit_confirmation_records.append(fold_a_record)
            cross_fit_model_records.append(
                {
                    "endpoint_id": endpoint_id,
                    "panel_id": evaluation.panel_id,
                    "cross_fit_fold": FOLD_A,
                    "fitting_split": "train",
                    **evaluation.model.to_dict(),
                }
            )
            if (
                evaluation.reverse_model is not None
                and evaluation.reverse_confirmation_record is not None
            ):
                cross_fit_confirmation_records.append(
                    evaluation.reverse_confirmation_record
                )
                cross_fit_model_records.append(
                    {
                        "endpoint_id": endpoint_id,
                        "panel_id": evaluation.panel_id,
                        "cross_fit_fold": FOLD_B,
                        "fitting_split": "validation",
                        **evaluation.reverse_model.to_dict(),
                    }
                )
            if evaluation.cross_fit_failure is not None:
                failures.append(evaluation.cross_fit_failure)
            verification_bias_records.extend(
                evaluation.verification_bias_records
            )
            stability_records.extend(evaluation.stability_records)
            classical_records.extend(evaluation.classical_records)

        visualization_selections = _visualization_marker_selections(
            endpoint_id,
            endpoint_models,
            endpoint_train_records,
            panel_map,
        )
        downstream_features: list[str] = [
            *VALIDATION_DOWNSTREAM_COMMON_FEATURE_COLUMNS,
            "heart_rate_bpm",
            "qrs_duration_ms",
            *(definition.own_value_column
              for definition in INTERVAL_CRITERION_SENSITIVITIES),
            *(feature_id for feature_id, _ in visualization_selections),
        ]
        primary_feature_id = PRIMARY_ADJUSTED_MARKERS.get(endpoint_id)
        if primary_feature_id is not None:
            downstream_features.append(primary_feature_id)
        downstream_features.extend(
            PREDEFINED_SCORE_COMPONENTS.get(endpoint_id, ())
        )
        for transfer_payload in ptb_scores_by_endpoint.get(endpoint_id, ()):
            downstream_features.extend(
                transfer_payload.get("score_spec", {}).get(
                    "components", ()
                )
            )
        for baseline_definition in BaselineRegistry().for_endpoint(
            endpoint_id
        ):
            if not baseline_definition.criterion_id:
                continue
            baseline_feature = (
                f"classical_{baseline_definition.criterion_id}"
            )
            downstream_features.extend(
                (baseline_feature, f"{baseline_feature}_positive")
            )
        primary_baseline = PRIMARY_CLASSICAL_BASELINES.get(endpoint_id)
        if primary_baseline is not None:
            downstream_features.extend(
                (primary_baseline, f"{primary_baseline}_positive")
            )
        for pair in PAIRED_AVAILABILITY_COMPARISONS.get(endpoint_id, ()):
            downstream_features.extend(pair)
        # Own interval values feed the device-scaled Cornell product and the
        # own-delineation sensitivity arm.
        downstream_features.append("classical_qrs_widening_ge_120")
        attach_endpoint_features(downstream_features)
        # Audit S2: the comparator criteria run on the recorder's intervals;
        # the own versions stay as ``own_*`` for the reversed sensitivity
        # analysis. Only the comparator paths below read these frames -- the
        # screened classical markers and the criterion search keep the own
        # measurements their models were fitted on.
        comparator_frame = apply_device_interval_comparators(frame)
        comparator_train = comparator_frame.loc[
            comparator_frame["split"].eq("train")
        ].reset_index(drop=True)
        comparator_validation = comparator_frame.loc[
            comparator_frame["split"].eq("validation")
        ].reset_index(drop=True)
        del comparator_frame

        if report_label_id and "report_reference_state" in frame:
            report_records.extend(
                _machine_report_records(
                    endpoint_id,
                    report_label_id,
                    train,
                    validation,
                    specification,
                )
            )

        primary_vcg_id = PRIMARY_ADJUSTED_MARKERS.get(endpoint_id)
        if primary_vcg_id is not None and primary_vcg_id in endpoint_models:
            # Compared against every classical baseline registered for this
            # endpoint (not just one hardcoded rule), so the "which
            # criterion is good" verdict below covers the same ground the
            # Leitlinienvergleich view already displays. Marker selection
            # itself stays exactly as prespecified in PRIMARY_ADJUSTED_
            # MARKERS above — only the baseline side is generalized, to
            # avoid reintroducing a best-of-many-markers search here.
            for baseline_definition in BaselineRegistry().for_endpoint(
                endpoint_id
            ):
                if not baseline_definition.criterion_id:
                    continue
                baseline_feature_id = (
                    f"classical_{baseline_definition.criterion_id}"
                )
                try:
                    comparison = _paired_primary_baseline_comparison(
                        endpoint_id,
                        primary_vcg_id,
                        baseline_feature_id,
                        endpoint_models[primary_vcg_id],
                        comparator_train,
                        comparator_validation,
                        specification,
                        iterations,
                    )
                    if comparison:
                        verdict, verdict_de = _comparison_verdict(comparison)
                        comparison["comparison_verdict"] = verdict
                        comparison["comparison_verdict_de"] = verdict_de
                        paired_comparison_records.append(comparison)
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append(
                        {
                            "endpoint_id": endpoint_id,
                            "feature_id": primary_vcg_id,
                            "stage": "paired_guideline_comparison",
                            "reason": f"{baseline_feature_id}: {exc}",
                        }
                    )

        for candidate_id, comparator_id in PAIRED_AVAILABILITY_COMPARISONS.get(
            endpoint_id, ()
        ):
            if (
                candidate_id not in endpoint_models
                or comparator_id not in endpoint_models
            ):
                continue
            try:
                comparison = _paired_continuous_baseline_comparison(
                    endpoint_id,
                    candidate_id,
                    comparator_id,
                    endpoint_models[candidate_id],
                    train,
                    validation,
                    specification,
                    iterations,
                )
                if comparison:
                    comparison["comparison_type"] = (
                        PAIRED_AVAILABILITY_COMPARISON_TYPE
                    )
                    verdict, verdict_de = _comparison_verdict(comparison)
                    comparison["comparison_verdict"] = verdict
                    comparison["comparison_verdict_de"] = verdict_de
                    paired_comparison_records.append(comparison)
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "feature_id": candidate_id,
                        "stage": "paired_availability_comparison",
                        "reason": f"{comparator_id}: {exc}",
                    }
                )

        # The comparator criteria above apply absolute clinical cut-offs to
        # this project's own interval measurements, which carry a documented
        # systematic boundary bias (LUDB: T-offset about -20 ms, P-onset
        # about +15 ms). Re-running the same rules on the recorder's own
        # fiducials for the identical waveform shows whether the incremental
        # value survives the swap. Neither estimator is treated as truth.
        try:
            comparator_sensitivity_records.extend(
                evaluate_interval_criterion_sensitivity(
                    comparator_validation,
                    endpoint_id=endpoint_id,
                    evaluation_split="validation",
                ).to_dict(orient="records")
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": "",
                    "stage": "delineation_comparator_sensitivity",
                    "reason": str(exc),
                }
            )

        if options.include_predefined_scores:
            combo_components = PREDEFINED_SCORE_COMPONENTS.get(endpoint_id)
            if combo_components:
                try:
                    records, payload, combo_stability = _fit_predefined_score(
                        endpoint_id,
                        combo_components,
                        train,
                        validation,
                        specification,
                        iterations,
                    )
                    combination_records.extend(records)
                    score_specs.append(payload)
                    stability_records.append(combo_stability)
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append(
                        {
                            "endpoint_id": endpoint_id,
                            "feature_id": " + ".join(combo_components),
                            "stage": "predefined_score_fit",
                            "reason": str(exc),
                        }
                    )

        for transfer_index, transfer_payload in enumerate(
            ptb_scores_by_endpoint.get(endpoint_id, ())
        ):
            try:
                records, payload, transfer_stability = (
                    _evaluate_ptb_frozen_score(
                        endpoint_id,
                        transfer_payload,
                        train,
                        validation,
                        specification,
                        iterations,
                        transfer_index=transfer_index,
                    )
                )
                combination_records.extend(records)
                score_specs.append(payload)
                stability_records.append(transfer_stability)
            except (ValueError, KeyError, TypeError) as exc:
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "feature_id": transfer_payload.get(
                            "ptb_endpoint_id",
                            "ptb_frozen_score",
                        ),
                        "stage": "ptb_frozen_transfer",
                        "reason": str(exc),
                    }
                )

        if options.include_exploratory_scores:
            (
                exploratory_records,
                exploratory_payloads,
                exploratory_stability,
                exploratory_failures,
            ) = _fit_exploratory_scores(
                endpoint_id,
                train,
                validation,
                specification,
                iterations,
                panel_map,
                endpoint_models,
                endpoint_train_records,
                endpoint_validation_records,
                feature_value_loader=score_feature_values,
                feature_frame_loader=projected_endpoint_splits,
            )
            combination_records.extend(exploratory_records)
            score_specs.extend(exploratory_payloads)
            stability_records.extend(exploratory_stability)
            failures.extend(exploratory_failures)

        for feature_id, selection_role in visualization_selections:
            curves, distributions = _visualization_records_for_marker(
                endpoint_id,
                feature_id,
                panel_map[feature_id],
                selection_role,
                endpoint_models[feature_id],
                train,
                validation,
                endpoint_train_records[feature_id],
                endpoint_validation_records[feature_id],
            )
            performance_curve_records.extend(curves)
            score_distribution_records.extend(distributions)

        adjusted_feature_id = PRIMARY_ADJUSTED_MARKERS.get(endpoint_id)
        adjusted_model = (
            endpoint_models.get(adjusted_feature_id)
            if adjusted_feature_id is not None
            else None
        )
        if adjusted_feature_id is not None and adjusted_model is not None:
            if window_cohort_path.exists():
                try:
                    stability_records.extend(
                        _window_sensitivity_records(
                            catalog,
                            cohort_path=window_cohort_path,
                            feature_glob=analysis_feature_glob,
                            endpoint_id=endpoint_id,
                            feature_id=adjusted_feature_id,
                            model=adjusted_model,
                            specification=specification,
                            bootstrap_iterations=iterations,
                        )
                    )
                except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                    failures.append(
                        {
                            "endpoint_id": endpoint_id,
                            "feature_id": adjusted_feature_id,
                            "stage": "label_window_sensitivity",
                            "reason": str(exc),
                        }
                    )
            stability_records.extend(
                _subgroup_stability_records(
                    endpoint_id,
                    adjusted_feature_id,
                    adjusted_model,
                    validation,
                    specification,
                    bootstrap_iterations=iterations,
                )
            )
            stability_records.extend(
                _temporal_validation_records(
                    endpoint_id,
                    adjusted_feature_id,
                    train,
                    validation,
                    specification,
                    bootstrap_iterations=iterations,
                )
            )
            stability_records.extend(
                _covariate_adjusted_records(
                    endpoint_id,
                    adjusted_feature_id,
                    adjusted_model.direction,
                    train,
                    validation,
                )
            )
            if probability_models_allowed:
                (
                    endpoint_adjusted,
                    endpoint_specs,
                    endpoint_decision_curves,
                    endpoint_calibration_curve,
                    endpoint_failures,
                ) = fit_endpoint_adjusted_models(
                    comparator_train,
                    comparator_validation,
                    endpoint_id=endpoint_id,
                    vcg_feature_id=adjusted_feature_id,
                    vcg_direction=adjusted_model.direction,
                    classical_feature_id=PRIMARY_CLASSICAL_BASELINES.get(
                        endpoint_id
                    ),
                    bootstrap_iterations=iterations,
                    random_seed=specification.random_seed,
                )
            else:
                endpoint_adjusted = pd.DataFrame()
                endpoint_specs = []
                endpoint_decision_curves = pd.DataFrame()
                endpoint_calibration_curve = pd.DataFrame()
                endpoint_failures = []
            adjusted_records.extend(
                endpoint_adjusted.to_dict(orient="records")
            )
            adjusted_model_specs.extend(endpoint_specs)
            decision_curve_records.extend(
                endpoint_decision_curves.to_dict(orient="records")
            )
            calibration_curve_records.extend(
                endpoint_calibration_curve.to_dict(orient="records")
            )
            failures.extend(endpoint_failures)
            if (
                options.include_all_ecg_secondary
                and definitions[endpoint_id].target_kind.value != "predictive"
                and all_label_path.exists()
            ):
                try:
                    repeated_frame = _load_all_ecg_endpoint_frame(
                        catalog,
                        label_path=all_label_path,
                        feature_glob=analysis_feature_glob,
                        endpoint_id=endpoint_id,
                        feature_id=adjusted_feature_id,
                    )
                    repeated_train = repeated_frame.loc[
                        repeated_frame["split"] == "train"
                    ].reset_index(drop=True)
                    repeated_validation = repeated_frame.loc[
                        repeated_frame["split"] == "validation"
                    ].reset_index(drop=True)
                    repeated_rows, repeated_payload = (
                        fit_repeated_ecg_clustered_marker(
                            repeated_train,
                            repeated_validation,
                            endpoint_id=endpoint_id,
                            feature_id=adjusted_feature_id,
                            direction=adjusted_model.direction,
                            bootstrap_iterations=iterations,
                            random_seed=specification.random_seed,
                        )
                    )
                    repeated_rows["analysis_family"] = (
                        "secondary_all_ecg_clustered"
                    )
                    all_ecg_records.extend(
                        repeated_rows.to_dict(orient="records")
                    )
                    all_ecg_model_specs.append(repeated_payload)
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    np.linalg.LinAlgError,
                ) as exc:
                    failures.append(
                        {
                            "endpoint_id": endpoint_id,
                            "feature_id": adjusted_feature_id,
                            "stage": "all_ecg_clustered_analysis",
                            "reason": str(exc),
                        }
                    )

        endpoint_screening = apply_fdr_by_family(
            pd.DataFrame(marker_records[group_offsets["marker_records"] :]),
            family_columns=(
                "endpoint_id",
                "panel_id",
                "evaluation_split",
            ),
            alpha_by_panel={
                "primary_core": specification.primary_fdr_alpha,
                "classical": specification.primary_fdr_alpha,
                "sentinel": specification.primary_fdr_alpha,
                "discovery_reserve": specification.discovery_fdr_alpha,
            },
        )
        endpoint_stability = pd.DataFrame(
            stability_records[group_offsets["stability_records"] :]
        )
        endpoint_score_specs = tuple(
            score_specs[group_offsets["score_specs"] :]
        )
        criterion_value_columns = list(
            selected_single_marker_ids(
                endpoint_id=endpoint_id,
                screening=endpoint_screening,
                stability=endpoint_stability,
                marker_models=endpoint_models,
                maximum=options.maximum_criterion_single_candidates,
            )
        )
        for score_payload in endpoint_score_specs:
            criterion_value_columns.extend(
                score_payload.get("score_spec", {}).get("components", ())
            )
        attach_endpoint_features(criterion_value_columns)
        try:
            criterion_tables = build_endpoint_criterion_catalog(
                endpoint_id=endpoint_id,
                train=train,
                screening=endpoint_screening,
                combinations=pd.DataFrame(
                    combination_records[
                        group_offsets["combination_records"] :
                    ]
                ),
                stability=endpoint_stability,
                marker_models=endpoint_models,
                score_payloads=endpoint_score_specs,
                adjusted_results=pd.DataFrame(
                    adjusted_records[group_offsets["adjusted_records"] :]
                ),
                development_run_id=options.run_id,
                snapshot_hash=feature_snapshot.snapshot_hash,
                baseline_id=PRIMARY_CLASSICAL_BASELINES.get(endpoint_id, ""),
                bootstrap_iterations=iterations,
                random_seed=specification.random_seed,
                maximum_single_candidates=(
                    options.maximum_criterion_single_candidates
                ),
                maximum_combination_candidates=(
                    options.maximum_criterion_combination_candidates
                ),
            )
            criterion_candidate_records.extend(
                criterion_tables.candidates.to_dict(orient="records")
            )
            criterion_threshold_records.extend(
                criterion_tables.thresholds.to_dict(orient="records")
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": None,
                    "stage": "criterion_catalog",
                    "reason": str(exc),
                }
            )

        _write_endpoint_validation_checkpoint(
            endpoint_checkpoint_path,
            input_hash=input_hash,
            endpoint_id=endpoint_id,
            result_groups=result_groups,
            group_offsets=group_offsets,
        )
        flush_record_groups(result_groups)
        if progress:
            progress(
                f"Validierung {endpoint_index:,}/{len(endpoint_ids):,}: "
                f"{endpoint_id} atomar abgeschlossen"
            )

        # The endpoint checkpoint has just been written, so everything up to
        # here survives a stop. Publish where we are and, if the user asked to
        # end, honour it at this boundary -- the one place where the result set
        # is complete for every endpoint processed so far.
        publish_activity(
            run_root,
            RefinementActivity(
                stage=ACTIVITY_VALIDATION_REFINEMENT,
                unit_label="Endpunkt",
                position=endpoint_index,
                total=len(endpoint_ids),
                detail=str(endpoint_id),
                extra={"completed_endpoint_ids": list(completed_endpoint_ids)},
            ),
        )
        completed_endpoint_ids.append(str(endpoint_id))
        stop_mode = requested_stop_mode(run_root)
        if stop_mode is not None:
            stopped_early = True
            stop_reason = (
                f"Auf Wunsch beendet nach Endpunkt {endpoint_index:,} von "
                f"{len(endpoint_ids):,} ({stop_mode})."
            )
            if progress:
                progress(f"Validierung: {stop_reason}")
            del endpoint_universe, frame, train, validation
            break

        del endpoint_universe, frame, train, validation

    frame = None  # Endpoint closures are no longer invoked during finalization.
    flush_record_groups(result_groups)
    # Only one checkpoint projection is live at a time; capture the measured
    # bound before releasing the row index ahead of result materialization.
    if shared_feature_matrix is not None:
        feature_matrix_metrics.update(
            {
                "projected_read_count": (
                    shared_feature_matrix.projected_read_count
                ),
                "maximum_projected_columns": (
                    shared_feature_matrix.maximum_projected_columns
                ),
            }
        )
        del shared_feature_matrix

    screening = apply_fdr_by_family(
        records_frame(marker_records),
        family_columns=("endpoint_id", "panel_id", "evaluation_split"),
        alpha_by_panel={
            "primary_core": specification.primary_fdr_alpha,
            "classical": specification.primary_fdr_alpha,
            "sentinel": specification.primary_fdr_alpha,
            "discovery_reserve": specification.discovery_fdr_alpha,
        },
    )
    from .projection_hypotheses import annotate_projection_hypothesis_frame
    screening = annotate_projection_hypothesis_frame(screening)
    from .mechanism_hypotheses import annotate_mechanism_hypotheses
    screening = annotate_mechanism_hypotheses(screening)
    cross_fit_confirmations = records_frame(cross_fit_confirmation_records)
    cross_fit_results = combine_folds(
        {
            FOLD_A: cross_fit_confirmations.loc[
                cross_fit_confirmations.get(
                    "cross_fit_fold",
                    pd.Series("", index=cross_fit_confirmations.index),
                ).eq(FOLD_A)
            ],
            FOLD_B: cross_fit_confirmations.loc[
                cross_fit_confirmations.get(
                    "cross_fit_fold",
                    pd.Series("", index=cross_fit_confirmations.index),
                ).eq(FOLD_B)
            ],
        },
        key_columns=("endpoint_id", "panel_id", "feature_id"),
        effect_column="roc_auc",
    )
    cross_fit_results = apply_fdr_by_family(
        cross_fit_results,
        family_columns=("endpoint_id", "panel_id"),
        alpha_by_panel={
            "primary_core": specification.primary_fdr_alpha,
            "classical": specification.primary_fdr_alpha,
            "sentinel": specification.primary_fdr_alpha,
            "discovery_reserve": specification.discovery_fdr_alpha,
        },
        # There is one IUT p-value per hypothesis; there is no second
        # evaluation-split stratum to select from.
        sensitivity_strata=(),
    )
    if not cross_fit_results.empty:
        cross_fit_results["cross_fit_confirmed"] = (
            cross_fit_results["cross_fit_confirmed_raw"].astype("boolean").fillna(False).astype(bool)
            & cross_fit_results["confirmatory_fdr_significant"].astype("boolean").fillna(False).astype(bool)
            & cross_fit_results[
                "confirmatory_global_fdr_significant"
            ].astype("boolean").fillna(False).astype(bool)
            & cross_fit_results[
                "confirmatory_hierarchical_fdr_significant"
            ].astype("boolean").fillna(False).astype(bool)
        )
    classical = pd.DataFrame(classical_records)
    report_baselines = pd.DataFrame(report_records)
    combinations = pd.DataFrame(combination_records)
    stability = records_frame(stability_records)
    failures_frame = pd.DataFrame(failures)
    label_counts = pd.DataFrame(endpoint_counts)
    comparisons = records_frame(paired_comparison_records)
    comparator_sensitivity = records_frame(comparator_sensitivity_records)
    performance_curves = records_frame(performance_curve_records)
    fixed_curve_points = _fixed_reference_curve_points(
        classical,
        report_baselines,
    )
    if not fixed_curve_points.empty:
        performance_curves = pd.concat(
            (performance_curves, fixed_curve_points),
            ignore_index=True,
            sort=False,
        )
    operating_points = _build_operating_point_results(
        (screening, combinations),
        classical,
        report_baselines,
    )
    descriptive_ranking = _build_comparison_table(
        screening,
        classical,
        report_baselines,
    )
    all_ecg = apply_fdr_by_family(
        pd.DataFrame(all_ecg_records),
        family_columns=("analysis_family", "evaluation_split"),
    )
    verification_bias = records_frame(verification_bias_records)
    # The delineation gate may declare a landmark limited -- measured, and
    # known to miss the external accuracy standard at every operating point.
    # Markers whose reliability collapses under that landmark's error are
    # still screened and reported, but must not be *confirmed*.  Loading this
    # raises if a limitation is declared and no current sensitivity map
    # exists, because continuing would silently grant the claim the
    # limitation exists to withhold.
    fiducial_limitation = load_limitation_policy(
        resources_root=config.resources_root,
        benchmark_suite_path=(
            config.resources_root / "delineation" / "benchmark_suite.json"
        ),
        feature_implementation_hash=feature_implementation_hash(),
    )
    criterion_candidates = finalize_criterion_catalog_evidence(
        pd.DataFrame(criterion_candidate_records),
        screening,
        cross_fit_results=cross_fit_results,
        verification_bias_results=verification_bias,
        fiducial_limitation=fiducial_limitation,
    )
    transfer_matrix = build_ptb_mimic_transfer_matrix(
        config,
        screening=screening,
        combinations=combinations,
        stability=stability,
        ptb_run_id=options.ptb_transfer_run_id,
        enabled=options.include_ptb_frozen_scores and ptb_transfer_available,
    )
    explorer_tables = build_result_explorer_tables(
        screening,
        combinations=combinations,
        comparisons=comparisons,
        stability=stability,
        candidate_evidence=_initial_explorer_evidence(criterion_candidates, preview=options.preview),
    )

    snapshot_verification = verify_artifact_snapshot(feature_snapshot)
    if not snapshot_verification["valid"]:
        raise RuntimeError(
            "Development feature artifacts changed while validation was "
            f"running: {snapshot_verification}"
        )

    if (file_sha256(geometry_path) if geometry_path.exists() else None) != geometry_hash:
        raise RuntimeError("Echo geometry changed during validation")
    from .lead_confirmation import adjust_lead_confirmation
    from .voltage_mass_confirmation import adjust_voltage_mass_confirmation
    output_frames = {
        "reference_clock_stress": lambda: pd.DataFrame(clock_stress_records),
        "mechanism_sensitivity_results": lambda: adjust_mechanism_family(
            pd.DataFrame(mechanism_records), planned_hypotheses=len(mechanism_hypotheses)),
        "voltage_mass_model_results": lambda: adjust_voltage_mass_confirmation(pd.DataFrame(voltage_mass_records)),
        "lead_comparison_results": lambda: adjust_lead_confirmation(pd.DataFrame(lead_comparison_records), planned_endpoints=tuple(endpoint_ids)),
        "screening_results": screening,
        "classical_baseline_results": classical,
        "machine_report_baseline_results": report_baselines,
        "vcg_vs_classical_comparison": comparisons,
        "delineation_comparator_sensitivity": comparator_sensitivity,
        "descriptive_validation_ranking": descriptive_ranking,
        "baseline_registry": baseline_registry.frame(),
        "methodology_registry": methodology_registry.frame(),
        "combination_results": combinations,
        "overfitting_stability_report": stability,
        "validation_failure_report": failures_frame,
        "validation_label_counts": label_counts,
        "adjusted_model_results": lambda: pd.DataFrame(adjusted_records),
        "decision_curve_results": lambda: records_frame(decision_curve_records),
        "calibration_curve_points": lambda: records_frame(calibration_curve_records),
        "all_ecg_clustered_results": all_ecg,
        "performance_curve_points": performance_curves,
        "score_distribution_bins": lambda: records_frame(
            score_distribution_records
        ),
        "operating_point_results": operating_points,
        "ptb_transfer_matrix": transfer_matrix,
        "marker_pathology_explorer": explorer_tables.marker_pathology,
        "pathology_marker_explorer": explorer_tables.pathology_marker,
        "endpoint_summary_explorer": explorer_tables.endpoint_summary,
        "criterion_candidates": criterion_candidates,
        "criterion_thresholds": lambda: pd.DataFrame(
            criterion_threshold_records
        ),
        "cross_fit_confirmations": cross_fit_confirmations,
        "cross_fit_results": cross_fit_results,
        "cross_fit_plan": lambda: pd.DataFrame(cross_fit_plan_records),
        "verification_bias_bounds": verification_bias,
    }
    # Drop redundant local ownership; the consuming exporter releases completed
    # tables. Keep only combinations/cross-fit results needed by later registries.
    del screening, classical, report_baselines, comparisons, comparator_sensitivity
    del descriptive_ranking, stability, failures_frame, label_counts, all_ecg
    del performance_curves, fixed_curve_points, operating_points, transfer_matrix
    del explorer_tables, criterion_candidates, cross_fit_confirmations, verification_bias
    analysis_completeness = (
        "partial_preview" if options.preview else "complete_snapshot"
    )
    def prepare_output(name, frame):
        if not prevalence_metrics_interpretable:
            frame = _suppress_prevalence_dependent_metrics({name: frame})[name]
        frame["analysis_completeness"] = analysis_completeness
        frame["freeze_eligible"] = not options.preview
        frame["sampling_design"] = sampling_design
        frame["probability_metrics_interpretable"] = (
            probability_models_allowed
        )
        frame["prevalence_metrics_interpretable"] = (
            prevalence_metrics_interpretable
        )
        if name == "voltage_mass_model_results":
            frame["freeze_eligible"] = False
        return frame
    from .result_finalization import finalize_result_tables
    output_frames, output_row_counts, output_paths = finalize_result_tables(
        output_frames, run_root, prepare_output,
        retain=("criterion_candidates", "criterion_thresholds"),
    )
    model_path = run_root / "trained_marker_models.json"
    lead_model_path = run_root / "lead_comparison_models.json"
    anatomy_model_path = run_root / "voltage_mass_models.json"
    atomic_write_json(anatomy_model_path, {"models": voltage_mass_model_records,
        "input_hash": input_hash, "clinical_claim_allowed": False, "locked_test_read": False})
    output_paths.append(str(anatomy_model_path.resolve()))
    atomic_write_json(lead_model_path, {"models": lead_model_records, "input_hash": input_hash,
        "clinical_claim_allowed": False, "locked_test_read": False})
    output_paths.append(str(lead_model_path.resolve()))
    cross_fit_models_path = run_root / "cross_fit_marker_models.json"
    score_path = run_root / "frozen_score_candidates.json"
    release_path = run_root / "score_releases.json"
    spec_path = run_root / "validation_spec.json"
    provenance_path = run_root / "development_provenance.json"
    adjusted_specs_path = run_root / "adjusted_model_specs.json"
    all_ecg_specs_path = run_root / "all_ecg_clustered_models.json"
    criterion_registry_path = run_root / "criterion_registry.json"
    locked_shortlist_path = run_root / "locked_test_marker_shortlist.json"
    atomic_write_json(
        model_path,
        {
            "model_type": "univariate_train_fitted_marker",
            "threshold_source": "train",
            "models": marker_models,
        },
    )
    atomic_write_json(
        cross_fit_models_path,
        {
            "model_type": "two_fold_out_of_fold_marker_confirmation",
            "cross_fitting_version": CROSS_FITTING_VERSION,
            "locked_test_read": False,
            "assignments": [
                {
                    "cross_fit_fold": FOLD_A,
                    "fitting_split": "train",
                    "confirming_split": "validation",
                },
                {
                    "cross_fit_fold": FOLD_B,
                    "fitting_split": "validation",
                    "confirming_split": "train",
                },
            ],
            "multiplicity_family": cross_fit_multiplicity_family(
                cross_fit_results
            ),
            "summary": cross_fit_summary(cross_fit_results),
            "models": cross_fit_model_records,
        },
    )
    atomic_write_json(
        score_path,
        {
            "score_type": "exploratory_score",
            "validation": "train_fit_validation_evaluation",
            "prevalence_metrics_interpretable": (
                prevalence_metrics_interpretable
            ),
            "scores": _suppress_prevalence_metrics_in_payload(score_specs)
            if not prevalence_metrics_interpretable
            else score_specs,
        },
    )
    release_created_at = datetime.now(timezone.utc).isoformat()
    releases = []
    validation_combination_lookup = {
        (str(row.endpoint_id), str(row.feature_id)): row
        for row in combinations.loc[
            combinations.get(
                "evaluation_split",
                pd.Series("", index=combinations.index),
            )
            .astype(str)
            .eq("validation")
        ].itertuples(index=False)
    }
    for payload in score_specs:
        score_spec = payload.get("score_spec", {})
        if not score_spec:
            continue
        origin = str(score_spec.get("origin", "mimic_development"))
        endpoint_id = str(score_spec.get("endpoint_id", ""))
        score_id = str(score_spec.get("score_id", ""))
        validation_row = validation_combination_lookup.get(
            (endpoint_id, score_id)
        )
        validation_metrics = (
            {
                "roc_auc": _finite_or_none(validation_row.roc_auc),
                "roc_auc_ci_low": _finite_or_none(
                    validation_row.roc_auc_ci_low
                ),
                "roc_auc_ci_high": _finite_or_none(
                    validation_row.roc_auc_ci_high
                ),
                "average_precision": (
                    _finite_or_none(validation_row.average_precision)
                    if prevalence_metrics_interpretable
                    else None
                ),
                "sensitivity_at_90_specificity": _finite_or_none(
                    validation_row.specificity_90_sensitivity
                ),
                "sensitivity_at_95_specificity": _finite_or_none(
                    validation_row.specificity_95_sensitivity
                ),
                "specificity_at_90_sensitivity": _finite_or_none(
                    validation_row.sensitivity_90_specificity
                ),
                "n_positive": int(validation_row.n_positive),
                "n_negative": int(validation_row.n_negative),
                "inference_status": str(validation_row.inference_status),
                "validation_stable": _strict_bool(
                    getattr(validation_row, "validation_stable", False)
                ),
                "selected_for_freeze": _strict_bool(
                    getattr(validation_row, "selected_for_freeze", False)
                ),
            }
            if validation_row is not None
            else {}
        )
        releases.append(
            {
                "release_id": (
                    f"{score_spec.get('score_id', 'score')}@"
                    f"{score_spec.get('version', 'unknown')}"
                ),
                "endpoint_id": endpoint_id,
                "score_spec": score_spec,
                "marker_model": payload.get("marker_model", {}),
                "validation_metrics": validation_metrics,
                "stage": ScoreReleaseStage.DEVELOPMENT.value,
                "reference_type": (
                    "ptb_ecg_report_label_transfer"
                    if origin.startswith("ptb")
                    else "mimic_clinical_reference_development"
                ),
                "development_run_id": options.run_id,
                "freeze_hash": "",
                "locked_test_artifact_hash": "",
                "primary_case_display_allowed": False,
                "created_at": release_created_at,
            }
        )
    atomic_write_json(
        release_path,
        {
            "release_registry_version": "1.0.0",
            "run_id": options.run_id,
            "prevalence_metrics_interpretable": (
                prevalence_metrics_interpretable
            ),
            "releases": releases,
        },
    )
    atomic_write_json(
        criterion_registry_path,
        {
            "criterion_registry_version": "2.0.0",
            "run_id": options.run_id,
            "feature_snapshot_hash": feature_snapshot.snapshot_hash,
            "threshold_source": "train",
            "evaluation_split": "validation",
            "locked_test_read": False,
            "clinical_claim_allowed": False,
            "prevalence_metrics_interpretable": (
                prevalence_metrics_interpretable
            ),
            "criteria": _validation_json_value(
                output_frames["criterion_candidates"].to_dict(
                    orient="records"
                )
            ),
            "thresholds": _validation_json_value(
                output_frames["criterion_thresholds"].to_dict(
                    orient="records"
                )
            ),
        },
    )
    locked_shortlist = _build_locked_test_marker_shortlist(
        output_frames["criterion_candidates"],
        marker_models,
        development_run_id=options.run_id,
        snapshot_hash=feature_snapshot.snapshot_hash,
    )
    atomic_write_json(locked_shortlist_path, locked_shortlist)
    atomic_write_json(spec_path, specification.to_dict())
    atomic_write_json(
        adjusted_specs_path,
        {
            "model_role": "secondary_adjusted",
            "minimum_events_per_parameter": 20,
            "automatic_feature_selection": False,
            "models": adjusted_model_specs,
        },
    )
    atomic_write_json(
        all_ecg_specs_path,
        {
            "model_role": "secondary_all_ecg_patient_clustered",
            "automatic_feature_selection": False,
            "models": all_ecg_model_specs,
        },
    )
    atomic_write_json(
        provenance_path,
        {
            "run_id": options.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_hash": code_hash,
            "project_config_hash": config_hash,
            "operational_config_hash": operational_config_hash(config),
            "input_hash": input_hash,
            "validation_input_payload": validation_input_payload,
            "feature_files": [
                str(path.resolve()) for path in feature_paths
            ],
            "feature_scope": feature_scope,
            "sampling_design": sampling_design,
            "probability_models_allowed": probability_models_allowed,
            "prevalence_metrics_interpretable": (
                prevalence_metrics_interpretable
            ),
            "preview": options.preview,
            "analysis_completeness": analysis_completeness,
            "feature_integrity": feature_integrity,
            "feature_snapshot": feature_snapshot.to_dict(),
            "analysis_feature_cache": analysis_feature_cache,
            "ptb_transfer": ptb_transfer_provenance,
            "baseline_registry_hash": baseline_registry.registry_hash,
            "methodology_registry_hash": methodology_registry.registry_hash,
            "development_label_attestation_hash": (
                label_attestation.get("attestation_hash")
                if label_attestation is not None
                else None
            ),
            "clinical_event_evidence_provenance": (
                _event_evidence_provenance_summary(
                    event_evidence_provenance
                )
            ),
            "marker_workers": marker_workers,
            "feature_matrix_loading": feature_matrix_metrics,
            "validation_checkpointing": {
                "marker_chunk_size": (
                    VALIDATION_MARKER_CHECKPOINT_CHUNK_SIZE
                ),
                "chunk_size_hardware_independent": True,
                "completed_endpoint_format": (
                    "manifest_with_single_result_group_copy"
                ),
            },
            "cross_fitting": {
                "version": CROSS_FITTING_VERSION,
                "status": "two_fold_out_of_fold_complete",
                "locked_test_read": False,
                "summary": cross_fit_summary(cross_fit_results),
                "multiplicity_family": cross_fit_multiplicity_family(
                    cross_fit_results
                ),
            },
            "locked_test_labels_read": False,
        },
    )
    output_paths.extend(
        str(path.resolve())
        for path in (
            model_path,
            cross_fit_models_path,
            score_path,
            release_path,
            adjusted_specs_path,
            all_ecg_specs_path,
            criterion_registry_path,
            locked_shortlist_path,
            spec_path,
            provenance_path,
            analysis_feature_cache_manifest_path,
        )
    )

    # A validation the user ended early covers fewer endpoints than it was
    # asked for. It must never be indistinguishable from a finished one: a
    # later stage that reuses this manifest would silently treat the missing
    # endpoints as "screened, nothing found" rather than "never looked".
    if stopped_early:
        raise PartialValidationStopped(
            stop_reason,
            completed_endpoint_ids=tuple(completed_endpoint_ids),
            requested_endpoint_ids=tuple(str(item) for item in endpoint_ids),
            output_paths=tuple(output_paths),
        )

    manifest = StageManifest.complete(
        stage=f"validate:{options.run_id}",
        stage_version=VALIDATION_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts=output_row_counts,
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "No locked_test labels were read.",
            (
                "Frozen deployment directions and thresholds remain train-fitted; "
                "internal evidence is additionally confirmed in both swapped, "
                "patient-disjoint out-of-fold directions."
            ),
            (
                "Cross-fit selection uses an intersection-union p-value and "
                "project-wide multiplicity across the union of hypotheses."
            ),
            (
                "Criterion replication requires Train and Validation evidence, "
                "prespecified adjusted-model increment, assessability, and "
                "project-wide multiplicity sensitivity."
            ),
            (
                "Marker missingness is reported as class-specific test yield "
                "with worst-case AUC bounds."
            ),
            (
                "locked_test_marker_shortlist_entries="
                f"{locked_shortlist['entry_count']}"
            ),
            f"analysis_completeness={analysis_completeness}",
            f"freeze_eligible={not options.preview}",
            f"sampling_design={sampling_design}",
            (
                "probability_model_evaluation=enabled"
                if probability_models_allowed
                else "probability_model_evaluation=disabled_for_preview"
            ),
            (
                "prevalence_dependent_metrics=enabled"
                if prevalence_metrics_interpretable
                else "prevalence_dependent_metrics=disabled_for_preview"
            ),
            (
                "decision_curve_scope=internal_development_descriptive; "
                "not_AHA_clinical_utility_evidence"
            ),
        ),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    return manifest


def _resolved_feature_scope(
    requested_scope: str,
    feature_integrity: dict,
) -> str:
    root = Path(str(feature_integrity.get("root", "")))
    manifest_path = Path(
        str(feature_integrity.get("manifest_path", ""))
    )
    if not feature_integrity.get("complete_index"):
        raise RuntimeError(
            "Development validation requires the complete indexed feature "
            "artifact, not a partition subset."
        )
    if not manifest_path.is_file():
        raise RuntimeError(
            "Development feature provenance is missing its stage manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(
            "Development validation requires a complete feature-stage "
            "manifest."
        )
    if manifest.get("input_hash") != feature_integrity.get(
        "stage_input_hash"
    ):
        raise RuntimeError(
            "Feature stage manifest and partition index refer to different "
            "inputs."
        )
    notes = {str(note) for note in manifest.get("notes", ())}
    expected_artifact_note = (
        f"feature_artifact_hash={feature_integrity.get('artifact_hash')}"
    )
    if expected_artifact_note not in notes:
        raise RuntimeError(
            "Feature stage manifest and partition index have inconsistent "
            "artifact hashes."
        )
    includes_locked = (
        "locked_test_features_included=True" in notes
        or "locked_test_only=True" in notes
        or root.name == "all"
        or root.name.startswith("locked-test-")
    )
    if includes_locked:
        raise RuntimeError(
            "Development validation refuses feature artifacts that include or "
            "originate from locked_test. Use the development extraction scope."
        )
    if "locked_test_features_included=False" not in notes:
        raise RuntimeError(
            "Development feature provenance does not explicitly confirm that "
            "locked_test was excluded."
        )
    if root.name.startswith("smoke-development-"):
        return "development_preview"
    if root.name != "development":
        raise RuntimeError(
            "Development validation requires an explicitly development-scoped "
            f"feature artifact; got {root}."
        )
    return "development_only"


def _resolved_sampling_design(
    feature_scope: str,
    feature_integrity: dict,
) -> str:
    if feature_scope != "development_preview":
        return "complete_clinical_phenome_union"
    manifest_path = Path(str(feature_integrity.get("manifest_path", "")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = {str(note) for note in manifest.get("notes", ())}
    expected = f"capped_sampling={CAPPED_SAMPLING_VERSION}"
    if expected in notes:
        return "endpoint_enriched_pilot"
    return "legacy_nonrepresentative_pilot"


def _probability_models_allowed(
    *,
    preview: bool,
    sampling_design: str,
) -> bool:
    return bool(
        not preview
        and sampling_design == "complete_clinical_phenome_union"
    )


def _prevalence_metrics_allowed(
    *,
    preview: bool,
    sampling_design: str,
) -> bool:
    return bool(
        not preview
        and sampling_design == "complete_clinical_phenome_union"
    )


def _suppress_prevalence_dependent_metrics(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    sanitized: dict[str, pd.DataFrame] = {}
    for name, source in frames.items():
        frame = source.copy()
        if (
            name == "performance_curve_points"
            and "curve_type" in frame.columns
        ):
            frame = frame.loc[
                ~frame["curve_type"].astype(str).eq("precision_recall")
            ].copy()
        if "prevalence" in frame.columns:
            frame["pilot_sample_prevalence"] = pd.to_numeric(
                frame["prevalence"],
                errors="coerce",
            )
        for column in frame.columns:
            if _is_prevalence_dependent_metric(column):
                frame[column] = np.nan
        sanitized[name] = frame.reset_index(drop=True)
    return sanitized


def _is_prevalence_dependent_metric(column: str) -> bool:
    normalized = str(column).casefold()
    return bool(
        normalized == "prevalence"
        or "average_precision" in normalized
        or "pr_auc" in normalized
        or "ppv" in normalized
        or "npv" in normalized
        or "brier" in normalized
        or "log_loss" in normalized
        or "calibration_" in normalized
    )


def _suppress_prevalence_metrics_in_payload(value):
    if isinstance(value, dict):
        return {
            key: (
                None
                if _is_prevalence_dependent_metric(key)
                else _suppress_prevalence_metrics_in_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_suppress_prevalence_metrics_in_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _suppress_prevalence_metrics_in_payload(item) for item in value
        )
    return value


def _build_validation_input_payload(
    config: MimicProjectConfig,
    *,
    feature_paths: list[Path],
    feature_integrity: dict,
    feature_snapshot_hash: str,
    feature_scope: str,
    sampling_design: str,
    cohort_path: Path,
    all_label_path: Path,
    master_path: Path,
    context_path: Path,
    window_cohort_path: Path,
    reports_path: Path,
    specification: ValidationSpec,
    panel_map: dict[str, str],
    options: DevelopmentValidationOptions,
    ptb_transfer_provenance: dict,
    baseline_registry: BaselineRegistry,
    methodology_registry: MethodologyRegistry,
    code_hash: str,
    config_hash: str,
    label_attestation: Optional[dict],
    event_evidence_provenance: Optional[dict],
) -> dict:
    scientific_paths: dict[str, Path] = {
        "endpoint_index_cohorts": cohort_path,
        "ecg_master_index": master_path,
        "clinical_context_features": context_path,
        "master_index_manifest": (
            config.master_root / "master_index.manifest.json"
        ),
        "evidence_manifest": (
            config.evidence_root / "evidence.manifest.json"
        ),
        "machine_report_manifest": (
            config.evidence_root / "machine_reports.manifest.json"
        ),
        "label_manifest": config.labels_root / "labels.manifest.json",
        "base_ingest_manifest": (
            config.standardized_root / "ingest.manifest.json"
        ),
        "feature_panel_registry": (
            config.features_root / "feature_panels.json"
        ),
        "baseline_source_lock": baseline_registry.source_lock_path,
        "methodology_source_lock": methodology_registry.source_lock_path,
    }
    if options.include_all_ecg_secondary and all_label_path.is_file():
        scientific_paths["clinical_label_evidence"] = all_label_path
    event_manifest = (
        config.standardized_root
        / "clinical_events"
        / "clinical_events.manifest.json"
    )
    if event_manifest.is_file():
        scientific_paths["clinical_event_manifest"] = event_manifest
    if window_cohort_path.is_file():
        scientific_paths[
            "endpoint_window_sensitivity_cohorts"
        ] = window_cohort_path
    if reports_path.is_file():
        scientific_paths["machine_report_labels"] = reports_path
    if label_attestation is not None:
        scientific_paths["development_label_attestation"] = (
            development_label_attestation_path(config)
        )
    if event_evidence_provenance is not None:
        if (
            event_evidence_provenance.get("status")
            == "direct_event_backed"
        ):
            scientific_paths["clinical_event_evidence_provenance"] = (
                config.evidence_root / "evidence_provenance.json"
            )
        else:
            scientific_paths[
                "clinical_event_evidence_equivalence_attestation"
            ] = (
                config.work_root
                / "provenance"
                / "clinical_event_evidence_equivalence.json"
            )
    missing = [
        str(path)
        for path in scientific_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Validation provenance inputs are missing: "
            + ", ".join(missing)
        )
    return {
        "payload_version": VALIDATION_INPUT_PAYLOAD_VERSION,
        "analysis_scope": feature_scope,
        "sampling_design": sampling_design,
        "locked_test_included": False,
        "scientific_files": {
            name: file_content_fingerprint(path)
            for name, path in sorted(scientific_paths.items())
        },
        "feature_files": [
            file_fingerprint(path) for path in feature_paths
        ],
        "feature_integrity": feature_integrity,
        "feature_snapshot_hash": feature_snapshot_hash,
        "validation_spec": specification.to_dict(),
        "panel_map": panel_map,
        "analysis_options": {
            "include_predefined_scores": options.include_predefined_scores,
            "include_ptb_frozen_scores": options.include_ptb_frozen_scores,
            "ptb_transfer_run_id": options.ptb_transfer_run_id,
            "include_exploratory_scores": (
                options.include_exploratory_scores
            ),
            "include_all_ecg_secondary": (
                options.include_all_ecg_secondary
            ),
            "maximum_criterion_single_candidates": (
                options.maximum_criterion_single_candidates
            ),
            "maximum_criterion_combination_candidates": (
                options.maximum_criterion_combination_candidates
            ),
            "preview": options.preview,
            "feature_cache_version": VALIDATION_FEATURE_CACHE_VERSION,
            "checkpoint_version": VALIDATION_CHECKPOINT_VERSION,
        },
        "ptb_transfer": ptb_transfer_provenance,
        "baseline_registry_hash": baseline_registry.registry_hash,
        "methodology_registry_hash": methodology_registry.registry_hash,
        "development_label_attestation_hash": (
            label_attestation.get("attestation_hash")
            if label_attestation is not None
            else None
        ),
        "clinical_event_evidence_provenance": (
            _event_evidence_provenance_summary(
                event_evidence_provenance
            )
        ),
        "project_config_hash": config_hash,
        "code_hash": code_hash,
    }


def _event_evidence_provenance_summary(
    provenance: Optional[dict],
) -> Optional[dict]:
    if provenance is None:
        return None
    return {
        key: provenance.get(key)
        for key in (
            "status",
            "clinical_event_registry_input_hash",
            "attestation_hash",
        )
        if provenance.get(key) is not None
    }


def verify_validation_input_payload_files(payload: dict) -> dict:
    if payload.get("payload_version") != VALIDATION_INPUT_PAYLOAD_VERSION:
        raise RuntimeError(
            "Validation input payload version is missing or stale"
        )
    if payload.get("locked_test_included") is not False:
        raise RuntimeError(
            "Development validation input payload includes locked_test"
        )
    if (
        payload.get("analysis_scope") == "development_only"
        and not isinstance(
            payload.get("clinical_event_evidence_provenance"),
            dict,
        )
    ):
        raise RuntimeError(
            "Formal validation input has no standardized clinical-event "
            "evidence provenance."
        )
    scientific_files = payload.get("scientific_files")
    feature_files = payload.get("feature_files")
    if not isinstance(scientific_files, dict) or not isinstance(
        feature_files,
        list,
    ):
        raise RuntimeError(
            "Validation input payload has no immutable file inventory"
        )
    verified_scientific = 0
    for name, expected in scientific_files.items():
        if not isinstance(expected, dict) or not expected.get("path"):
            raise RuntimeError(
                f"Validation input fingerprint is malformed: {name}"
            )
        path = Path(str(expected["path"]))
        lowered_parts = {part.lower() for part in path.parts}
        if "sealed" in lowered_parts or "locked_test" in path.name.lower():
            raise RuntimeError(
                "Development validation provenance references a sealed "
                f"artifact: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Validation input disappeared after analysis: {path}"
            )
        current = file_content_fingerprint(path)
        if current != expected:
            raise RuntimeError(
                "Validation input changed after analysis: "
                f"{name} ({path})"
            )
        verified_scientific += 1

    verified_features = 0
    for expected in feature_files:
        if not isinstance(expected, dict) or not expected.get("path"):
            raise RuntimeError(
                "Validation feature fingerprint is malformed"
            )
        path = Path(str(expected["path"]))
        lowered = str(path).lower()
        if "locked-test-" in lowered or "split=locked_test" in lowered:
            raise RuntimeError(
                "Development validation provenance references locked-test "
                f"features: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Validation feature partition disappeared: {path}"
            )
        current = file_fingerprint(path)
        if current != expected:
            raise RuntimeError(
                "Validation feature partition changed after analysis: "
                f"{path}"
            )
        verified_features += 1
    return {
        "status": "verified",
        "scientific_file_count": verified_scientific,
        "feature_file_count": verified_features,
    }


def _verify_current_development_feature_inputs(
    config: MimicProjectConfig,
    *,
    feature_integrity: dict,
    panel_ids: tuple[str, ...],
) -> None:
    root = Path(str(feature_integrity["root"]))
    manifest_path = Path(str(feature_integrity["manifest_path"]))
    index_path = Path(str(feature_integrity["index_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != scientific_config_hash(config):
        raise RuntimeError(
            "Development feature inputs no longer match the current project "
            "configuration. Re-extract or register a newly verified cohort."
        )
    if not any(
        int(item.get("row_count", 0)) > 0
        for item in index.get("partitions", ())
    ):
        raise RuntimeError(
            "Development feature index contains no non-empty partitions"
        )
    stage_input_payload = index.get("stage_input_payload")
    if not isinstance(stage_input_payload, dict):
        raise RuntimeError(
            "Development feature index predates the shared input contract. "
            "Re-extract features before validation."
        )
    recorded_stage_hash = str(index.get("stage_input_hash", ""))
    if content_hash(stage_input_payload) != recorded_stage_hash:
        raise RuntimeError(
            "Development feature input contract is internally inconsistent"
        )
    panel_registry_path = config.features_root / "feature_panels.json"
    required = (panel_registry_path,)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Current development feature inputs are missing: {missing}"
        )
    panel_registry = json.loads(
        panel_registry_path.read_text(encoding="utf-8")
    )
    panel_hash = str(panel_registry.get("panel_hash", ""))
    if panel_hash != str(feature_integrity.get("panel_hash", "")):
        raise RuntimeError(
            "Development feature panel registry changed after extraction"
        )
    current_input_hash = current_development_feature_input_hash(
        config,
        panel_ids=tuple(panel_ids),
        partition_size=int(stage_input_payload.get("partition_size", 0)),
        panel_hash=panel_hash,
    )
    if current_input_hash != feature_integrity.get("stage_input_hash"):
        raise RuntimeError(
            "Development feature cohort provenance is stale: labels, master "
            "index, phenotype mapping, panel selection, partitioning or "
            "feature implementation changed after extraction."
        )


def _validation_cache_schema(connection, paths: list[Path]) -> pa.Schema:
    """Union footers one at a time, then normalize with DuckDB on zero rows.

    Binding union_by_name over thousands of wide files itself consumes RAM
    outside the DuckDB memory limit. Never keep that per-file binder state.
    Unsupported Arrow promotions fail closed, rather than losing columns.
    """
    schema = None
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            incoming = pa.schema([field for field in parquet.schema_arrow
                                  if field.name != "split"])
        schema = (incoming if schema is None else
                  pa.unify_schemas([schema, incoming], promote_options="permissive"))
    if schema is None:
        raise FileNotFoundError("No feature partitions for cache schema")
    relation = "_validation_cache_empty_schema"
    connection.register(relation, pa.Table.from_batches([], schema=schema))
    try:
        return connection.execute(f"SELECT * FROM {relation} LIMIT 0").to_arrow_table().schema
    finally:
        connection.unregister(relation)


def _stream_validation_cache_split(
    connection,
    *,
    source_paths: list[Path],
    schema: pa.Schema,
    temporary: Path,
    split: str,
    expected_rows: int,
    progress: Optional[Callable[[str], None]],
    block_files: int = 32,
) -> int:
    """Bounded, restartable materialization; never buffer a cohort row group.

    DuckDB still supplies the union schema. All consumers sort by clinical
    keys before sampling; the old parallel COPY explicitly disabled order
    preservation. This writer uses sorted partition order deterministically.
    Completed blocks survive a cooperative pause or a killed worker.
    """
    from .storage import flush_to_disk

    schema = schema.remove_metadata()
    schema_names = frozenset(schema.names)
    block_root = temporary.parent / "_stream_blocks"
    block_root.mkdir(parents=True, exist_ok=True)
    state_path = block_root / "checkpoint.json"
    signature = content_hash({
        "version": 1,
        "schema": schema.serialize().to_pybytes().hex(),
        "block_files": block_files,
        "sources": [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                    for p in source_paths],
    })
    state = _read_checkpoint_payload(state_path) or {}
    if state.get("signature") != signature:
        state = {"signature": signature, "blocks": {}}
    batch_rows = min(4096, max(64, (16 * 1024 * 1024) // (8 * max(1, len(schema)))))
    started = time.monotonic()
    fresh_rows = 0

    def emit(phase: str, done: int, *, reused: bool = False) -> None:
        elapsed = time.monotonic() - started
        rate = fresh_rows / elapsed if elapsed > 0 and fresh_rows else None
        eta = (expected_rows - done) / rate if rate else None
        payload = {"split": split, "phase": phase, "completed_rows": done,
                   "total_rows": expected_rows, "fresh_rows": fresh_rows,
                   "elapsed_seconds": elapsed, "rows_per_second": rate,
                   "eta_seconds": eta, "reused": reused,
                   "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_write_json(temporary.parent / "stream.progress.json", payload)
        if progress:
            estimate = f"{max(0, int(eta))}s" if eta is not None else "wird gemessen"
            progress(f"Analyse-Cache {split}: {done:,}/{expected_rows:,} Zeilen; "
                     f"Phase={phase}; ETA={estimate}"
                     + ("; Zwischenstand wiederverwendet" if reused else ""))

    def batches(paths: list[Path]):
        for source in paths:
            with pq.ParquetFile(source) as parquet:
                columns = [n for n in parquet.schema_arrow.names if n in schema_names]
                for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns,
                                                  use_threads=False):
                    arrays = []
                    for field in schema:
                        index = batch.schema.get_field_index(field.name)
                        array = (batch.column(index) if index >= 0
                                 else pa.nulls(batch.num_rows, type=field.type))
                        if array.type != field.type:
                            # Safe casting refuses value loss instead of silently
                            # changing an analysis. NULL and widening casts work.
                            array = array.cast(field.type, safe=True)
                        arrays.append(array)
                    yield pa.RecordBatch.from_arrays(arrays, schema=schema)

    def write(paths: list[Path], target: Path) -> int:
        rows = 0
        pending: list[pa.RecordBatch] = []
        pending_bytes = pending_rows = 0
        with pq.ParquetWriter(target, schema, compression="zstd") as writer:
            for batch in batches(paths):
                pending.append(batch)
                pending_bytes += batch.nbytes
                pending_rows += batch.num_rows
                rows += batch.num_rows
                if pending_bytes >= 32 * 1024 * 1024 or pending_rows >= 4096:
                    writer.write_table(pa.Table.from_batches(pending, schema=schema),
                                       row_group_size=4096)
                    pending.clear()
                    pending_bytes = pending_rows = 0
            if pending:
                writer.write_table(pa.Table.from_batches(pending, schema=schema),
                                   row_group_size=4096)
        flush_to_disk(target)
        return rows

    block_paths = []
    completed = 0
    emit("lesen", 0)
    for offset in range(0, len(source_paths), block_files):
        key = str(offset // block_files)
        target = block_root / f"block-{int(key):06d}.parquet"
        record = state["blocks"].get(key)
        reused = bool(record and _validation_cache_file_matches(block_root, record))
        if not reused:
            staging = block_root / f".{target.name}.{os.getpid()}.tmp"
            rows = write(source_paths[offset:offset + block_files], staging)
            record = {"relative_path": target.name, "row_count": rows,
                      "size": staging.stat().st_size, "sha256": file_sha256(staging)}
            atomic_replace(staging, target)
            state["blocks"][key] = record
            atomic_write_json(state_path, state)
            fresh_rows += rows
        completed += int(record["row_count"])
        block_paths.append(target)
        emit("lesen", completed, reused=reused)
    if completed != expected_rows:
        raise RuntimeError(f"Validation cache source row mismatch: {completed} != {expected_rows}")

    # Only the final concatenation is repeated after an interruption here.
    # Each input block is already durable and verified above.
    started = time.monotonic()
    fresh_rows = 0
    emit("zusammenfuehren", 0)
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for path in block_paths:
            # Blocks already have the exact canonical schema. Preserve a
            # useful 4096-row group size instead of fragmenting every group
            # into the smaller source decode batches (costly on later scans).
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
                    writer.write_batch(batch, row_group_size=4096)
                    fresh_rows += batch.num_rows
            emit("zusammenfuehren", fresh_rows)
    flush_to_disk(temporary)
    return fresh_rows


def _prepare_validation_feature_cache(
    *,
    catalog_path: Path,
    memory_limit_gb: int,
    threads: int,
    run_root: Path,
    feature_integrity: dict,
    feature_snapshot_hash: str,
    progress: Optional[Callable[[str], None]],
) -> tuple[str, dict, Path]:
    source_root = Path(str(feature_integrity["root"])).resolve()
    cache_key = content_hash(
        {
            "cache_version": VALIDATION_FEATURE_CACHE_VERSION,
            "feature_snapshot_hash": feature_snapshot_hash,
            "feature_artifact_hash": feature_integrity.get(
                "artifact_hash"
            ),
        }
    )
    cache_root = (
        run_root
        / "analysis_feature_cache"
        / f"cache-{cache_key[:16]}"
    )
    manifest_path = cache_root / "cache.manifest.json"
    if manifest_path.is_file():
        payload = _verify_validation_feature_cache(
            manifest_path,
            feature_snapshot_hash=feature_snapshot_hash,
            feature_artifact_hash=str(
                feature_integrity.get("artifact_hash", "")
            ),
        )
        return (
            str(cache_root / "split=*" / "features.parquet"),
            payload,
            manifest_path,
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_root / "cache.checkpoint.json"
    checkpoint = _load_validation_cache_checkpoint(
        checkpoint_path,
        cache_key=cache_key,
        feature_snapshot_hash=feature_snapshot_hash,
    )
    index_payload = json.loads(
        Path(str(feature_integrity["index_path"])).read_text(
            encoding="utf-8"
        )
    )
    expected_by_split: dict[str, int] = {}
    for item in index_payload.get("partitions", ()):
        relative_path = str(item.get("relative_path", ""))
        for split in ("train", "validation"):
            if relative_path.startswith(f"split={split}/"):
                expected_by_split[split] = (
                    expected_by_split.get(split, 0)
                    + int(item.get("row_count", 0))
                )
                break
    if set(expected_by_split) != {"train", "validation"}:
        raise RuntimeError(
            "Development feature index does not contain both train and "
            "validation partitions"
        )

    catalog = DuckDBCatalog(
        catalog_path,
        memory_limit_gb=memory_limit_gb,
        threads=threads,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        temp_directory = cache_root / "_duckdb_tmp"
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"SET temp_directory='{_sql_path(temp_directory)}'"
        )
        connection.execute("SET preserve_insertion_order=FALSE")
        for split_index, split in enumerate(
            ("train", "validation"),
            start=1,
        ):
            record = checkpoint["splits"].get(split)
            if record is not None and _validation_cache_file_matches(
                cache_root,
                record,
            ):
                continue
            target = (
                cache_root / f"split={split}" / "features.parquet"
            )
            _remove_validation_cache_path(
                target,
                cache_root=cache_root,
                allow_missing=True,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            for stale in target.parent.glob(
                f".{target.name}.*.tmp"
            ):
                _remove_validation_cache_path(
                    stale,
                    cache_root=cache_root,
                )
            temporary = (
                target.parent
                / f".{target.name}.{os.getpid()}.tmp"
            )
            source_glob = (
                source_root / f"split={split}" / "part-*.parquet"
            )
            source_paths = sorted(source_glob.parent.glob(source_glob.name))
            if not source_paths:
                raise FileNotFoundError(
                    f"No source feature partitions for split {split}"
                )
            if progress:
                progress(
                    "Analyse-Cache "
                    f"{split_index:,}/2: {split} wird kompakt erzeugt"
                )
            with profile_query(
                f"validation:feature-cache:{split}",
                detail=f"source_partitions={feature_integrity.get('partition_count', 0)}",
            ) as profile:
                schema = _validation_cache_schema(connection, source_paths)
                row_count = _stream_validation_cache_split(
                    connection, source_paths=source_paths, schema=schema,
                    temporary=temporary, split=split,
                    expected_rows=int(expected_by_split[split]), progress=progress,
                )
                profile["rows"] = row_count
            expected_rows = int(expected_by_split[split])
            if row_count != expected_rows:
                raise RuntimeError(
                    f"Validation feature cache row mismatch for {split}: "
                    f"{row_count:,} != {expected_rows:,}"
                )
            temporary_size = int(temporary.stat().st_size)
            temporary_sha = file_sha256(temporary)
            atomic_replace(temporary, target)
            record = {
                "split": split,
                "relative_path": target.relative_to(cache_root).as_posix(),
                "row_count": row_count,
                "size": temporary_size,
                "sha256": temporary_sha,
            }
            checkpoint["splits"][split] = record
            checkpoint["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            atomic_write_json(checkpoint_path, checkpoint)
            if progress:
                progress(
                    "Analyse-Cache "
                    f"{split_index:,}/2: {split} atomar abgeschlossen"
                )

    records = [
        checkpoint["splits"][split]
        for split in ("train", "validation")
    ]
    payload = {
        "cache_version": VALIDATION_FEATURE_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "feature_snapshot_hash": feature_snapshot_hash,
        "feature_artifact_hash": feature_integrity.get("artifact_hash"),
        "source_root": str(source_root),
        "source_partition_count": int(
            feature_integrity.get("partition_count", 0)
        ),
        "source_row_count": int(feature_integrity.get("row_count", 0)),
        "immutable_derivation": True,
        "locked_test_included": False,
        "splits": records,
        "row_count": sum(int(item["row_count"]) for item in records),
    }
    payload["cache_hash"] = content_hash(payload)
    atomic_write_json(manifest_path, payload)
    verified = _verify_validation_feature_cache(
        manifest_path,
        feature_snapshot_hash=feature_snapshot_hash,
        feature_artifact_hash=str(
            feature_integrity.get("artifact_hash", "")
        ),
    )
    return (
        str(cache_root / "split=*" / "features.parquet"),
        verified,
        manifest_path,
    )


def _read_checkpoint_payload(path: Path) -> Optional[dict]:
    """Checkpoint contents, or None if the file cannot be trusted.

    Truncation is the normal way a checkpoint dies: the process was killed
    mid-write, or the machine lost power. That must not be fatal either --
    an unreadable checkpoint carries exactly as much information as a
    missing one.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_validation_cache_checkpoint(
    path: Path,
    *,
    cache_key: str,
    feature_snapshot_hash: str,
) -> dict:
    if path.is_file():
        payload = _read_checkpoint_payload(path)
        if payload is not None and (
            payload.get("checkpoint_version")
            == VALIDATION_CHECKPOINT_VERSION
            and payload.get("cache_key") == cache_key
            and payload.get("feature_snapshot_hash")
            == feature_snapshot_hash
        ):
            payload.setdefault("splits", {})
            return payload
    # A checkpoint that describes different work is not an error, it is
    # simply unusable: discard it and start the stage fresh. Refusing to
    # continue would cost a multi-day run over a file whose only purpose is
    # to save time, and whose contents this stage can always produce again.
    payload = {
        "checkpoint_version": VALIDATION_CHECKPOINT_VERSION,
        "cache_key": cache_key,
        "feature_snapshot_hash": feature_snapshot_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    atomic_write_json(path, payload)
    return payload


def _verify_validation_feature_cache(
    manifest_path: Path,
    *,
    feature_snapshot_hash: str,
    feature_artifact_hash: str,
) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("cache_version")
        != VALIDATION_FEATURE_CACHE_VERSION
        or payload.get("feature_snapshot_hash")
        != feature_snapshot_hash
        or str(payload.get("feature_artifact_hash", ""))
        != feature_artifact_hash
        or payload.get("immutable_derivation") is not True
        or payload.get("locked_test_included") is not False
    ):
        raise RuntimeError(
            "Validation feature-cache manifest is stale or unsafe"
        )
    expected_hash = payload.get("cache_hash")
    hash_payload = dict(payload)
    hash_payload.pop("cache_hash", None)
    if expected_hash != content_hash(hash_payload):
        raise RuntimeError(
            "Validation feature-cache manifest hash mismatch"
        )
    root = manifest_path.parent
    splits = payload.get("splits", ())
    if {str(item.get("split")) for item in splits} != {
        "train",
        "validation",
    }:
        raise RuntimeError(
            "Validation feature cache does not contain exact development "
            "splits"
        )
    for record in splits:
        if not _validation_cache_file_matches(root, record):
            raise RuntimeError(
                "Validation feature-cache file failed checksum "
                f"verification: {record.get('split')}"
            )
    return payload


def _validation_cache_file_matches(
    cache_root: Path,
    record: dict,
) -> bool:
    relative_path = Path(str(record.get("relative_path", "")))
    path = (cache_root / relative_path).resolve()
    if (
        not path.is_relative_to(cache_root.resolve())
        or not path.is_file()
    ):
        return False
    return (
        int(path.stat().st_size) == int(record.get("size", -1))
        and file_sha256(path) == record.get("sha256")
    )


def _remove_validation_cache_path(
    path: Path,
    *,
    cache_root: Path,
    allow_missing: bool = False,
) -> None:
    if not path.exists():
        if allow_missing:
            return
        raise FileNotFoundError(path)
    resolved = path.resolve()
    root = cache_root.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(
            f"Refusing to remove unsafe validation cache path: {resolved}"
        )
    if resolved.is_dir():
        raise RuntimeError(
            "Validation cache cleanup only removes derived files"
        )
    resolved.unlink()


@profiled_phase("endpoint_load_and_reference_masks")
def _load_endpoint_frame(
    catalog: DuckDBCatalog,
    *,
    cohort_path: Path,
    master_path: Path,
    reports_path: Optional[Path],
    context_path: Optional[Path],
    feature_glob: str,
    endpoint_id: str,
    report_label_id: Optional[str],
    splits: tuple[str, ...] = ("train", "validation"),
    feature_columns: Optional[tuple[str, ...]] = None,
    context_columns: Optional[tuple[str, ...]] = None,
    selected_records_only: bool = False,
    shared_feature_matrix: Optional[
        _SharedValidationFeatureMatrix
    ] = None,
) -> pd.DataFrame:
    allowed_splits = {"train", "validation", "locked_test"}
    if not splits or not set(splits) <= allowed_splits:
        raise ValueError(f"Invalid validation split selection: {splits}")
    split_sql = ", ".join(f"'{value}'" for value in splits)
    cohort_sql = _sql_path(cohort_path)
    master_sql = _sql_path(master_path)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    report_join = ""
    report_select = "NULL::VARCHAR AS report_reference_state"
    context_join = ""
    selected_context_columns = list(
        dict.fromkeys(
            (
                "bmi_before_ecg",
                "creatinine_nearest",
                "potassium_nearest",
            )
            + tuple(context_columns or ())
        )
    )
    context_select = ",\n                ".join(
        f"NULL AS {_quoted_identifier(column)}"
        for column in selected_context_columns
    )
    if shared_feature_matrix is not None:
        feature_select = (
            "NULL::DOUBLE AS __shared_feature_placeholder"
        )
        feature_join = ""
    elif feature_columns is None:
        read_features = feature_parquet_clause(
            sql_path=feature_sql,
            options=("hive_partitioning=true",),
        )
        feature_select = "f.* EXCLUDE (subject_id, study_id, split, ecg_time)"
        feature_join = (
            f"LEFT JOIN {read_features} f USING (subject_id, study_id)"
        )
    else:
        read_features = feature_parquet_clause(
            sql_path=feature_sql,
            options=("hive_partitioning=true",),
        )
        selected = list(dict.fromkeys(feature_columns))
        feature_select = ",\n                ".join(
            f"f.{_quoted_identifier(column)} AS {_quoted_identifier(column)}"
            for column in selected
        )
        if not feature_select:
            feature_select = "NULL::DOUBLE AS no_feature_selected"
        feature_join = (
            f"LEFT JOIN {read_features} f USING (subject_id, study_id)"
        )
    if selected_records_only and shared_feature_matrix is None:
        # Restrict to attempted records, retaining failed extraction rows.
        # The full-run left join remains necessary to measure true missingness.
        feature_join = feature_join.replace("LEFT JOIN ", "INNER JOIN ", 1)
    if reports_path is not None and report_label_id is not None:
        reports_sql = _sql_path(reports_path)
        state_column = _quoted_identifier(
            f"report_{report_label_id}_state"
        )
        report_join = (
            f"LEFT JOIN read_parquet('{reports_sql}') r "
            "USING (subject_id, study_id)"
        )
        report_select = f"r.{state_column} AS report_reference_state"
    if context_path is not None:
        context_sql = _sql_path(context_path)
        context_join = (
            f"LEFT JOIN read_parquet('{context_sql}') x "
            "USING (subject_id, study_id)"
        )
        context_select = ",\n                ".join(
            f"x.{_quoted_identifier(column)} AS {_quoted_identifier(column)}"
            for column in selected_context_columns
        )
    with catalog.connect(isolated=True) as connection, profile_query(
        f"validation:endpoint-frame:{endpoint_id}",
        detail=f"feature_columns={len(feature_columns or ())}",
    ) as profile:
        # Covariates the current contract requires. A master index written by
        # an older stage version yields typed NULLs rather than a binder
        # error, and the covariate is then reported as unavailable.
        master_columns = relation_columns(
            connection,
            f"read_parquet('{master_sql}')",
        )
        optional_master_columns = ",\n                ".join(
            (
                optional_column_select(
                    master_columns,
                    "anchor_year_group",
                    "VARCHAR",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_heart_rate_bpm",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_qt_interval_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_pr_interval_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_p_onset_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_p_end_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_qrs_onset_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_qrs_end_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
                optional_column_select(
                    master_columns,
                    "machine_p_duration_ms",
                    "DOUBLE",
                    prefix="m.",
                ),
            )
        )
        frame = connection.execute(
            f"""
            SELECT
                c.subject_id,
                c.study_id,
                c.ecg_time,
                c.split,
                c.label_id,
                c.label_state,
                c.eligibility_state,
                c.confidence,
                c.time_delta_hours,
                c.time_precision,
                c.evidence_ids_json,
                c.evidence_summary,
                c.source_tables,
                m.age,
                m.sex,
                m.race,
                {optional_master_columns},
                m.waveform_path,
                m.machine_report,
                m.machine_cart_id,
                m.machine_bandwidth,
                m.machine_filtering,
                m.machine_metadata_temporal_status,
                m.machine_metadata_usable_for_primary,
                m.hadm_id,
                m.ed_stay_id,
                m.nearest_echo_measurement_id,
                m.nearest_echo_delta_hours,
                m.machine_qrs_duration_ms,
                m.signal_quality_score AS master_signal_quality_score,
                {context_select},
                {report_select},
                {feature_select}
            FROM read_parquet('{cohort_sql}') c
            {feature_join}
            LEFT JOIN read_parquet('{master_sql}') m
                USING (subject_id, study_id)
            {report_join}
            {context_join}
            WHERE c.label_id = ?
              AND c.split IN ({split_sql})
            """,
            [endpoint_id],
        ).fetchdf()
        profile["rows"] = len(frame)
    if shared_feature_matrix is not None:
        # A capped pilot does not measure unselected recordings. Keep the
        # reference-blind index choice, then restrict its *sampling* scope.
        # This also avoids repeatedly allocating mostly-empty full-cohort
        # marker blocks for a 1k pilot.
        if selected_records_only or "smoke-development-" in feature_glob:
            frame = frame.loc[shared_feature_matrix.selection_mask(frame)].copy()
        frame = (
            shared_feature_matrix.attach(frame)
            if feature_columns is None
            else shared_feature_matrix.attach_columns(
                frame,
                feature_columns,
            )
        )
    from .reference_tiers import annotate_reference_rows

    frame = annotate_reference_rows(frame)
    return frame.sort_values(
        ["split", "subject_id", "study_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _window_sensitivity_records(
    catalog: DuckDBCatalog,
    *,
    cohort_path: Path,
    feature_glob: str,
    endpoint_id: str,
    feature_id: str,
    model: MarkerModel,
    specification: ValidationSpec,
    bootstrap_iterations: int,
) -> list[dict]:
    cohort_sql = _sql_path(cohort_path)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    # Per endpoint, this query runs once for every marker. Each call used
    # to open all 1,527 partitions with ``union_by_name``, whose per-file
    # schema state sits outside DuckDB's memory limit -- measured at
    # 19.36 GB on one such read. ``hive_partitioning`` is passed through
    # unchanged. See ``storage.feature_parquet_clause``.
    read_features = feature_parquet_clause(
        sql_path=feature_sql, options=("hive_partitioning=true",)
    )
    feature_column = _quoted_identifier(feature_id)
    with catalog.connect(isolated=True) as connection, profile_query(
        f"validation:window-sensitivity:{endpoint_id}:{feature_id}"
    ) as profile:
        frame = connection.execute(
            f"""
            SELECT
                c.subject_id,
                c.study_id,
                c.ecg_time,
                c.split,
                c.label_id,
                c.label_state,
                c.confidence,
                c.time_delta_hours,
                c.window_profile,
                c.window_before_hours,
                c.window_after_hours,
                f.{feature_column} AS {feature_column}
            FROM read_parquet('{cohort_sql}') c
            JOIN {read_features} f USING (subject_id, study_id)
            WHERE c.label_id = ?
              AND c.split IN ('train', 'validation')
              AND c.eligibility_state = 'eligible'
              AND c.label_state IN ('positive', 'negative')
            """,
            [endpoint_id],
        ).fetchdf()
        profile["rows"] = len(frame)
    frame = frame.sort_values(
        [
            "window_before_hours",
            "split",
            "subject_id",
            "study_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    if frame.empty:
        return []
    frame["label"] = frame["label_state"].eq("positive").astype(np.int8)
    records = []
    profile_columns = (
        "window_profile",
        "window_before_hours",
        "window_after_hours",
    )
    for profile_values, profile_frame in frame.groupby(
        list(profile_columns),
        sort=True,
        dropna=False,
    ):
        profile, before, after = profile_values
        for split_index, split in enumerate(("train", "validation")):
            split_frame = profile_frame.loc[
                profile_frame["split"] == split
            ].reset_index(drop=True)
            if split_frame.empty:
                continue
            values = _numeric_feature(split_frame, feature_id)
            record = _marker_record(
                endpoint_id,
                "window_sensitivity",
                feature_id,
                split,
                split_frame,
                values,
                model,
                specification,
                bootstrap_iterations,
                seed_offset=(
                    300_000
                    + int(content_hash(f"{endpoint_id}:{profile}")[:6], 16)
                    + split_index
                ),
            )
            record.update(
                {
                    "analysis_type": "label_window_sensitivity",
                    "sensitivity_dimension": "clinical_time_window",
                    "window_profile": profile,
                    "window_before_hours": float(before),
                    "window_after_hours": float(after),
                    "reference_model_source": "standard_endpoint_train",
                    "threshold_source": "standard_endpoint_train",
                    "exploratory_score": False,
                }
            )
            records.append(record)
    return records


def _load_all_ecg_endpoint_frame(
    catalog: DuckDBCatalog,
    *,
    label_path: Path,
    feature_glob: str,
    endpoint_id: str,
    feature_id: str,
) -> pd.DataFrame:
    label_sql = _sql_path(label_path)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    # Per endpoint, this query runs once for every marker. Each call used
    # to open all 1,527 partitions with ``union_by_name``, whose per-file
    # schema state sits outside DuckDB's memory limit -- measured at
    # 19.36 GB on one such read. ``hive_partitioning`` is passed through
    # unchanged. See ``storage.feature_parquet_clause``.
    read_features = feature_parquet_clause(
        sql_path=feature_sql, options=("hive_partitioning=true",)
    )
    feature_column = _quoted_identifier(feature_id)
    with catalog.connect(isolated=True) as connection, profile_query(
        f"validation:all-ecg-frame:{endpoint_id}:{feature_id}"
    ) as profile:
        frame = connection.execute(
            f"""
            SELECT
                l.subject_id,
                l.study_id,
                l.ecg_time,
                l.split,
                l.label_state,
                f.{feature_column} AS {feature_column}
            FROM read_parquet('{label_sql}') l
            JOIN {read_features} f USING (subject_id, study_id)
            WHERE l.label_id = ?
              AND l.split IN ('train', 'validation')
              AND l.eligibility_state = 'eligible'
              AND l.label_state IN ('positive', 'negative')
            """,
            [endpoint_id],
        ).fetchdf()
        profile["rows"] = len(frame)
    frame = frame.sort_values(
        ["split", "subject_id", "ecg_time", "study_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["label"] = frame["label_state"].eq("positive").astype(np.int8)
    return frame


def _visualization_marker_selections(
    endpoint_id: str,
    endpoint_models: dict[str, MarkerModel],
    train_records: dict[str, dict],
    panel_map: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    selected: dict[str, str] = {}
    prespecified = PRIMARY_ADJUSTED_MARKERS.get(endpoint_id)
    if (
        prespecified in endpoint_models
        and panel_map.get(str(prespecified)) != "classical"
    ):
        selected[str(prespecified)] = "prespecified_primary_marker"
    for panel_id in ("primary_core", "sentinel"):
        candidates = [
            feature_id
            for feature_id, record in train_records.items()
            if panel_map.get(feature_id) == panel_id
            and _optional_float(record.get("roc_auc")) is not None
        ]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda feature_id: (
                -float(train_records[feature_id]["roc_auc"]),
                feature_id,
            ),
        )[0]
        selected.setdefault(best, f"train_{panel_id}_representative")
    return tuple(selected.items())


def _visualization_records_for_marker(
    endpoint_id: str,
    feature_id: str,
    panel_id: str,
    selection_role: str,
    model: MarkerModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_record: dict,
    validation_record: dict,
) -> tuple[list[dict], list[dict]]:
    curves: list[dict] = []
    distributions: list[dict] = []
    bin_edges = np.linspace(-5.0, 5.0, 41)
    for split, frame, metric_record in (
        ("train", train, train_record),
        ("validation", validation, validation_record),
    ):
        values = _numeric_feature(frame, feature_id)
        labels = frame["label"].to_numpy(dtype=np.int8)
        valid = np.isfinite(values) & np.isin(labels, (0, 1))
        clean_labels = labels[valid]
        aligned_z = (
            model.direction * (values[valid] - model.mean)
            / model.standard_deviation
        )
        if np.unique(clean_labels).size < 2:
            continue
        fpr, tpr, roc_thresholds = roc_curve(
            clean_labels,
            aligned_z,
            drop_intermediate=True,
        )
        precision, recall, pr_thresholds = precision_recall_curve(
            clean_labels,
            aligned_z,
        )
        pr_thresholds = np.r_[pr_thresholds, np.nan]
        for curve_type, x_values, y_values, thresholds in (
            ("roc", fpr, tpr, roc_thresholds),
            ("precision_recall", recall, precision, pr_thresholds),
        ):
            indices = _downsample_curve_indices(len(x_values), limit=300)
            for order, index in enumerate(indices):
                curves.append(
                    {
                        "endpoint_id": endpoint_id,
                        "feature_id": feature_id,
                        "panel_id": panel_id,
                        "evaluation_split": split,
                        "selection_role": selection_role,
                        "selection_source": "prespecified_or_train_only",
                        "curve_type": curve_type,
                        "point_kind": "continuous_marker_curve",
                        "point_order": order,
                        "x": float(x_values[index]),
                        "y": float(y_values[index]),
                        "threshold_aligned_train_z": (
                            float(thresholds[index])
                            if np.isfinite(thresholds[index])
                            else np.nan
                        ),
                        "roc_auc": metric_record.get("roc_auc"),
                        "roc_auc_ci_low": metric_record.get(
                            "roc_auc_ci_low"
                        ),
                        "roc_auc_ci_high": metric_record.get(
                            "roc_auc_ci_high"
                        ),
                        "average_precision": metric_record.get(
                            "average_precision"
                        ),
                        "average_precision_ci_low": metric_record.get(
                            "average_precision_ci_low"
                        ),
                        "average_precision_ci_high": metric_record.get(
                            "average_precision_ci_high"
                        ),
                        "inference_status": metric_record.get(
                            "inference_status"
                        ),
                        "n_positive": int(np.sum(clean_labels == 1)),
                        "n_negative": int(np.sum(clean_labels == 0)),
                        "exploratory_score": False,
                    }
                )
        clipped = np.clip(
            aligned_z,
            bin_edges[0] + np.finfo(float).eps,
            bin_edges[-1] - np.finfo(float).eps,
        )
        for label in (0, 1):
            class_values = clipped[clean_labels == label]
            counts, _ = np.histogram(class_values, bins=bin_edges)
            denominator = max(1, len(class_values))
            for bin_index, count in enumerate(counts):
                distributions.append(
                    {
                        "endpoint_id": endpoint_id,
                        "feature_id": feature_id,
                        "panel_id": panel_id,
                        "evaluation_split": split,
                        "selection_role": selection_role,
                        "selection_source": "prespecified_or_train_only",
                        "label": int(label),
                        "label_state": (
                            "positive" if label == 1 else "negative"
                        ),
                        "bin_index": bin_index,
                        "bin_left": float(bin_edges[bin_index]),
                        "bin_right": float(bin_edges[bin_index + 1]),
                        "bin_center": float(
                            (bin_edges[bin_index] + bin_edges[bin_index + 1])
                            / 2.0
                        ),
                        "count": int(count),
                        "fraction_within_class": float(count / denominator),
                        "class_total": int(len(class_values)),
                        "display_transform": (
                            "direction_aligned_train_zscore_clipped_to_5sd"
                        ),
                        "inference_status": metric_record.get(
                            "inference_status"
                        ),
                    }
                )
    return curves, distributions


def _downsample_curve_indices(length: int, *, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.unique(
        np.rint(np.linspace(0, length - 1, limit)).astype(int)
    )


def _fixed_reference_curve_points(
    classical: pd.DataFrame,
    machine_reports: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    for source, frame, identifier_column, panel_id in (
        ("classical_rule", classical, "baseline_id", "classical"),
        (
            "machine_report",
            machine_reports,
            "baseline_id",
            "machine_report",
        ),
    ):
        if frame.empty or identifier_column not in frame:
            continue
        for row in frame.to_dict(orient="records"):
            sensitivity = _optional_float(row.get("sensitivity"))
            specificity = _optional_float(row.get("specificity"))
            ppv = _optional_float(row.get("ppv"))
            if sensitivity is None or specificity is None:
                continue
            shared = {
                "endpoint_id": row.get("endpoint_id"),
                "feature_id": row.get(identifier_column),
                "panel_id": panel_id,
                "evaluation_split": row.get("evaluation_split"),
                "selection_role": "predefined_reference",
                "selection_source": row.get("threshold_source", source),
                "point_kind": "fixed_reference_point",
                "point_order": 0,
                "threshold_aligned_train_z": np.nan,
                "roc_auc": row.get("roc_auc"),
                "average_precision": row.get("average_precision"),
                "inference_status": row.get("inference_status"),
                "n_positive": row.get("n_positive"),
                "n_negative": row.get("n_negative"),
                "exploratory_score": False,
            }
            records.append(
                {
                    **shared,
                    "curve_type": "roc",
                    "x": 1.0 - specificity,
                    "y": sensitivity,
                    "x_ci_low": _one_minus_optional(
                        row.get("specificity_ci_high")
                    ),
                    "x_ci_high": _one_minus_optional(
                        row.get("specificity_ci_low")
                    ),
                    "y_ci_low": row.get("sensitivity_ci_low"),
                    "y_ci_high": row.get("sensitivity_ci_high"),
                }
            )
            if ppv is not None:
                records.append(
                    {
                        **shared,
                        "curve_type": "precision_recall",
                        "x": sensitivity,
                        "y": ppv,
                        "x_ci_low": row.get("sensitivity_ci_low"),
                        "x_ci_high": row.get("sensitivity_ci_high"),
                        "y_ci_low": row.get("ppv_ci_low"),
                        "y_ci_high": row.get("ppv_ci_high"),
                    }
                )
    return pd.DataFrame(records)


def _one_minus_optional(value) -> float:
    numeric = _optional_float(value)
    return 1.0 - numeric if numeric is not None else np.nan


def _build_operating_point_results(
    candidates: pd.DataFrame | tuple[pd.DataFrame, ...],
    classical: pd.DataFrame,
    machine_reports: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []
    target_names = (
        ("specificity_90", "sensitivity_at_90_specificity"),
        ("specificity_95", "sensitivity_at_95_specificity"),
        ("sensitivity_90", "specificity_at_90_sensitivity"),
    )
    candidate_frames = (
        candidates if isinstance(candidates, tuple) else (candidates,)
    )
    for candidate_frame in candidate_frames:
        if candidate_frame.empty:
            continue
        for row in candidate_frame.to_dict(orient="records"):
            for prefix, target in target_names:
                sensitivity = _optional_float(
                    row.get(f"{prefix}_sensitivity")
                )
                specificity = _optional_float(
                    row.get(f"{prefix}_specificity")
                )
                if sensitivity is None or specificity is None:
                    continue
                records.append(
                    {
                        "endpoint_id": row.get("endpoint_id"),
                        "feature_id": row.get("feature_id"),
                        "panel_id": row.get("panel_id"),
                        "analysis_type": row.get("analysis_type"),
                        "evaluation_split": row.get("evaluation_split"),
                        "operating_point": target,
                        "threshold_source": row.get("threshold_source"),
                        "threshold_original": row.get(
                            f"{prefix}_threshold_original"
                        ),
                        "target_attainable": row.get(
                            f"{prefix}_target_attainable"
                        ),
                        "sensitivity": sensitivity,
                        "sensitivity_ci_low": row.get(
                            f"{prefix}_sensitivity_ci_low"
                        ),
                        "sensitivity_ci_high": row.get(
                            f"{prefix}_sensitivity_ci_high"
                        ),
                        "specificity": specificity,
                        "specificity_ci_low": row.get(
                            f"{prefix}_specificity_ci_low"
                        ),
                        "specificity_ci_high": row.get(
                            f"{prefix}_specificity_ci_high"
                        ),
                        "ppv": row.get(f"{prefix}_observed_ppv"),
                        "npv": row.get(f"{prefix}_observed_npv"),
                        "balanced_accuracy": row.get(
                            f"{prefix}_balanced_accuracy"
                        ),
                        "inference_status": row.get("inference_status"),
                        "exploratory_score": row.get(
                            "exploratory_score",
                            False,
                        ),
                    }
                )
    for source, frame, identifier_column, panel_id in (
        ("predefined_clinical_rule", classical, "baseline_id", "classical"),
        (
            "machine_report_positive",
            machine_reports,
            "baseline_id",
            "machine_report",
        ),
    ):
        if frame.empty or identifier_column not in frame:
            continue
        for row in frame.to_dict(orient="records"):
            if (
                _optional_float(row.get("sensitivity")) is None
                or _optional_float(row.get("specificity")) is None
            ):
                continue
            records.append(
                {
                    "endpoint_id": row.get("endpoint_id"),
                    "feature_id": row.get(identifier_column),
                    "panel_id": panel_id,
                    "analysis_type": source,
                    "evaluation_split": row.get("evaluation_split"),
                    "operating_point": "fixed_reference_point",
                    "threshold_source": row.get(
                        "threshold_source",
                        source,
                    ),
                    "threshold_original": np.nan,
                    "target_attainable": None,
                    "sensitivity": row.get("sensitivity"),
                    "sensitivity_ci_low": row.get("sensitivity_ci_low"),
                    "sensitivity_ci_high": row.get("sensitivity_ci_high"),
                    "specificity": row.get("specificity"),
                    "specificity_ci_low": row.get("specificity_ci_low"),
                    "specificity_ci_high": row.get("specificity_ci_high"),
                    "ppv": row.get("observed_ppv", row.get("ppv")),
                    "npv": row.get("observed_npv", row.get("npv")),
                    "balanced_accuracy": row.get("balanced_accuracy"),
                    "inference_status": row.get("inference_status"),
                    "exploratory_score": False,
                }
            )
    return pd.DataFrame(records)


def _load_or_evaluate_marker_chunks(
    *,
    endpoint_id: str,
    feature_ids: tuple[str, ...],
    panel_map: dict[str, str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    endpoint_universe: Optional[pd.DataFrame] = None,
    shared_feature_matrix: Optional[
        _SharedValidationFeatureMatrix
    ] = None,
    specification: ValidationSpec,
    bootstrap_iterations: int,
    worker_count: int,
    checkpoint_root: Path,
    input_hash: str,
    endpoint_index: int,
    endpoint_total: int,
    progress: Optional[Callable[[str], None]],
) -> list[_SingleMarkerEvaluation]:
    endpoint_root = checkpoint_root / endpoint_id / "marker_chunks"
    endpoint_root.mkdir(parents=True, exist_ok=True)
    # Checkpoint boundaries are scientific work units, not a side-effect of
    # available hardware.  This keeps resume artifacts stable across hosts.
    chunk_size = VALIDATION_MARKER_CHECKPOINT_CHUNK_SIZE
    evaluations: list[_SingleMarkerEvaluation] = []
    from .resource_grants import worker_ceiling
    from .marker_process_pool import process_mode_requested
    worker_count = worker_ceiling(worker_count)
    use_processes = (
        process_mode_requested()
        and int(worker_count) > 1
        and len(feature_ids) > 1
    )
    process_state: dict[str, Any] = {}
    executor = (
        ThreadPoolExecutor(
            max_workers=max(1, min(int(worker_count), len(feature_ids))),
            thread_name_prefix="vcg-marker",
        )
        if int(worker_count) > 1 and len(feature_ids) > 1
        else None
    )
    try:
        marker_base_universe = endpoint_universe
        marker_common_ready = shared_feature_matrix is None
        marker_train_mask: Optional[np.ndarray] = None
        marker_validation_mask: Optional[np.ndarray] = None
        marker_train_quality: Optional[np.ndarray] = None
        marker_validation_quality: Optional[np.ndarray] = None
        for start in range(0, len(feature_ids), chunk_size):
            stop = min(len(feature_ids), start + chunk_size)
            chunk_feature_ids = tuple(feature_ids[start:stop])
            checkpoint_path = (
                endpoint_root / f"markers-{start:04d}-{stop:04d}.json"
            )
            chunk = _load_marker_chunk_checkpoint(
                checkpoint_path,
                input_hash=input_hash,
                endpoint_id=endpoint_id,
                feature_ids=chunk_feature_ids,
            )
            if chunk is not None:
                note_reuse(
                    "validate", reused=True, reason="marker_checkpoint_restored",
                    detail=f"{endpoint_id}:{start}:{stop}",
                )
            if chunk is None:
                chunk_train = train
                chunk_validation = validation
                chunk_universe = endpoint_universe
                if shared_feature_matrix is not None:
                    if endpoint_universe is None:
                        raise RuntimeError(
                            "Blockwise marker validation requires the "
                            "reference-blind endpoint universe"
                        )
                    marker_set = set(chunk_feature_ids)
                    if not marker_common_ready:
                        missing_common = sorted(
                            set(VALIDATION_MARKER_COMMON_FEATURE_COLUMNS)
                            - shared_feature_matrix.available_columns
                        )
                        if missing_common:
                            raise RuntimeError(
                                "Blockwise marker validation lacks required "
                                f"common feature columns: {missing_common}"
                            )
                        common_projection = tuple(
                            column
                            for column in (
                                *VALIDATION_MARKER_COMMON_FEATURE_COLUMNS,
                                *(
                                    optional
                                    for optional in VALIDATION_MARKER_OPTIONAL_COMMON_FEATURE_COLUMNS
                                    if optional in shared_feature_matrix.available_columns
                                ),
                            )
                            if column not in endpoint_universe
                        )
                        marker_base_universe = (
                            shared_feature_matrix.attach_columns(
                                endpoint_universe,
                                common_projection,
                            )
                            if common_projection
                            else endpoint_universe.copy()
                        )
                        marker_base_universe["label"] = marker_base_universe[
                            "label_state"
                        ].eq("positive").astype(np.int8)
                        marker_verified = (
                            marker_base_universe["eligibility_state"].eq(
                                "eligible"
                            )
                            & marker_base_universe["label_state"].isin(
                                ("positive", "negative")
                            )
                        ).to_numpy()
                        marker_splits = marker_base_universe[
                            "split"
                        ].astype(str).to_numpy()
                        marker_train_mask = marker_verified & (
                            marker_splits == "train"
                        )
                        marker_validation_mask = marker_verified & (
                            marker_splits == "validation"
                        )
                        marker_quality = _numeric_feature(
                            marker_base_universe,
                            "signal_quality_score",
                        )
                        marker_train_quality = marker_quality[
                            marker_train_mask
                        ]
                        marker_validation_quality = marker_quality[
                            marker_validation_mask
                        ]
                        marker_common_ready = True
                    assert marker_base_universe is not None
                    assert marker_train_mask is not None
                    assert marker_validation_mask is not None
                    assert marker_train_quality is not None
                    assert marker_validation_quality is not None
                    from .projection_hypotheses import (
                        P_PROJECTION_CONTEXT_COLUMNS,
                    )
                    from vcg_biomarker.core.projection_metadata import (
                        projection_metadata,
                    )

                    has_p_projection = any(
                        (metadata := projection_metadata(feature_id)) is not None
                        and metadata["segment"] == "p"
                        for feature_id in chunk_feature_ids
                    )
                    restriction_columns = (
                        tuple(
                            column
                            for column in P_PROJECTION_CONTEXT_COLUMNS
                            if column in shared_feature_matrix.available_columns
                            and column not in marker_base_universe
                        )
                        if has_p_projection
                        else ()
                    )
                    sensitivity_columns = tuple(
                        column
                        for column in sorted(
                            shared_feature_matrix.available_columns
                        )
                        if column.startswith("sensitivity_")
                        and "__" in column
                        and column.split("__", 1)[1] in marker_set
                    )
                    classical_companions = tuple(
                        f"{feature_id}_positive"
                        for feature_id in chunk_feature_ids
                        if feature_id.startswith("classical_")
                        and not feature_id.endswith("_positive")
                        and f"{feature_id}_positive"
                        in shared_feature_matrix.available_columns
                    )
                    chunk_universe = (
                        shared_feature_matrix.attach_columns(
                            marker_base_universe,
                            (
                                *chunk_feature_ids,
                                *restriction_columns,
                                *sensitivity_columns,
                                *classical_companions,
                            ),
                        )
                    )
                    chunk_train = chunk_universe.loc[
                        marker_train_mask
                    ].reset_index(drop=True)
                    chunk_validation = chunk_universe.loc[
                        marker_validation_mask
                    ].reset_index(drop=True)
                chunk = (
                    _evaluate_marker_chunk_in_processes(
                        process_state,
                        endpoint_id=endpoint_id,
                        feature_ids=chunk_feature_ids,
                        panel_map=panel_map,
                        train=chunk_train,
                        validation=chunk_validation,
                        endpoint_universe=chunk_universe,
                        specification=specification,
                        bootstrap_iterations=bootstrap_iterations,
                        worker_count=worker_count,
                        train_quality=marker_train_quality,
                        validation_quality=marker_validation_quality,
                        feature_index_offset=start,
                        progress=progress,
                    )
                    if use_processes
                    else None
                )
                if chunk is None:
                    chunk = _evaluate_endpoint_markers(
                        endpoint_id=endpoint_id,
                        feature_ids=chunk_feature_ids,
                        panel_map=panel_map,
                        train=chunk_train,
                        validation=chunk_validation,
                        endpoint_universe=chunk_universe,
                        specification=specification,
                        bootstrap_iterations=bootstrap_iterations,
                        worker_count=worker_count,
                        train_quality=marker_train_quality,
                        validation_quality=marker_validation_quality,
                        feature_index_offset=start,
                        executor=executor,
                    )
                _write_marker_chunk_checkpoint(
                    checkpoint_path,
                    input_hash=input_hash,
                    endpoint_id=endpoint_id,
                    feature_ids=chunk_feature_ids,
                    evaluations=chunk,
                )
            evaluations.extend(chunk)
            if progress:
                progress(
                    f"Validierung {endpoint_index:,}/{endpoint_total:,}: "
                    f"{endpoint_id}; Marker {stop:,}/{len(feature_ids):,} "
                    "atomar"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if process_state.get("pool") is not None:
            process_state.pop("pool").close()
    if tuple(item.feature_id for item in evaluations) != feature_ids:
        raise RuntimeError(
            f"Marker checkpoint order mismatch for {endpoint_id}"
        )
    return evaluations


@profiled_phase("marker_checkpoint_write")
def _write_marker_chunk_checkpoint(
    path: Path,
    *,
    input_hash: str,
    endpoint_id: str,
    feature_ids: tuple[str, ...],
    evaluations: list[_SingleMarkerEvaluation],
) -> None:
    serialized = [
        {
            "feature_id": item.feature_id,
            "panel_id": item.panel_id,
            "model": (
                item.model.to_dict()
                if item.model is not None
                else None
            ),
            "train_record": item.train_record,
            "validation_record": item.validation_record,
            "reverse_model": (
                item.reverse_model.to_dict()
                if item.reverse_model is not None
                else None
            ),
            "reverse_confirmation_record": item.reverse_confirmation_record,
            "cross_fit_failure": item.cross_fit_failure,
            "stability_records": list(item.stability_records),
            "classical_records": list(item.classical_records),
            "verification_bias_records": list(
                item.verification_bias_records
            ),
            "failure": item.failure,
        }
        for item in evaluations
    ]
    payload = {
        "checkpoint_version": VALIDATION_CHECKPOINT_VERSION,
        "checkpoint_type": "marker_chunk",
        "input_hash": input_hash,
        "endpoint_id": endpoint_id,
        "feature_ids": list(feature_ids),
        "evaluations": _validation_json_value(serialized),
    }
    payload["checkpoint_hash"] = content_hash(payload)
    atomic_write_json(path, payload)


def _load_marker_chunk_checkpoint(
    path: Path,
    *,
    input_hash: str,
    endpoint_id: str,
    feature_ids: tuple[str, ...],
) -> Optional[list[_SingleMarkerEvaluation]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = payload.get("checkpoint_hash")
    hash_payload = dict(payload)
    hash_payload.pop("checkpoint_hash", None)
    if (
        payload.get("checkpoint_version")
        != VALIDATION_CHECKPOINT_VERSION
        or payload.get("checkpoint_type") != "marker_chunk"
        or payload.get("input_hash") != input_hash
        or payload.get("endpoint_id") != endpoint_id
        or tuple(payload.get("feature_ids", ())) != feature_ids
        or expected_hash != content_hash(hash_payload)
    ):
        raise RuntimeError(
            f"Marker checkpoint is stale or corrupt: {path}"
        )
    evaluations = []
    for item in payload.get("evaluations", ()):
        model_payload = item.get("model")
        reverse_model_payload = item.get("reverse_model")
        evaluations.append(
            _SingleMarkerEvaluation(
                feature_id=str(item["feature_id"]),
                panel_id=str(item["panel_id"]),
                model=(
                    MarkerModel.from_dict(model_payload)
                    if model_payload is not None
                    else None
                ),
                train_record=item.get("train_record"),
                validation_record=item.get("validation_record"),
                reverse_model=(
                    MarkerModel.from_dict(reverse_model_payload)
                    if reverse_model_payload is not None
                    else None
                ),
                reverse_confirmation_record=item.get(
                    "reverse_confirmation_record"
                ),
                cross_fit_failure=item.get("cross_fit_failure"),
                stability_records=tuple(
                    item.get("stability_records", ())
                ),
                classical_records=tuple(
                    item.get("classical_records", ())
                ),
                verification_bias_records=tuple(
                    item.get("verification_bias_records", ())
                ),
                failure=item.get("failure"),
            )
        )
    if tuple(item.feature_id for item in evaluations) != feature_ids:
        raise RuntimeError(
            f"Marker checkpoint content is incomplete: {path}"
        )
    return evaluations


def _write_endpoint_validation_checkpoint(
    path: Path,
    *,
    input_hash: str,
    endpoint_id: str,
    result_groups: dict[str, list[dict]],
    group_offsets: dict[str, int],
) -> None:
    groups_root = path.with_suffix(".groups")
    groups_root.mkdir(parents=True, exist_ok=True)
    group_manifest: list[dict] = []
    for group_name, group_records in result_groups.items():
        if isinstance(group_records, PartitionedRecords) and group_records.pending:
            if group_offsets[group_name] != group_records.spilled_rows:
                raise RuntimeError("Endpoint partition offset disagrees with committed rows")
            row_count = len(group_records.pending)
            artifact_path = groups_root / f"{group_name}.parquet"
            group_records.flush(artifact_path)
            group_manifest.append({"group_name": group_name, "row_count": row_count,
                "relative_path": artifact_path.relative_to(path.parent).as_posix(),
                "encoding": "parquet-table-v1", "sha256": group_records.checksums[artifact_path]})
            continue
        # Only one endpoint group is converted at a time, not a second full
        # endpoint-shaped dictionary graph on top of all retained results.
        records = _validation_json_value(group_records[group_offsets[group_name]:])
        if not records:
            group_manifest.append(
                {
                    "group_name": group_name,
                    "row_count": 0,
                    "inline_empty": True,
                }
            )
            continue
        artifact_payload = {
            "checkpoint_version": VALIDATION_CHECKPOINT_VERSION,
            "checkpoint_type": "endpoint_result_group",
            "input_hash": input_hash,
            "endpoint_id": endpoint_id,
            "group_name": group_name,
            "records": records,
        }
        artifact_payload["content_hash"] = content_hash(artifact_payload)
        from .checkpoint_codec import write_group
        artifact_path = groups_root / f"{group_name}.json.gz"
        encoding_metadata = write_group(artifact_path, artifact_payload)
        group_manifest.append(
            {
                "group_name": group_name,
                "row_count": len(records),
                "relative_path": artifact_path.relative_to(
                    path.parent
                ).as_posix(),
                **encoding_metadata,
                "content_hash": artifact_payload["content_hash"],
            }
        )
    payload = {
        "checkpoint_version": VALIDATION_CHECKPOINT_VERSION,
        "checkpoint_type": "completed_endpoint_manifest",
        "input_hash": input_hash,
        "endpoint_id": endpoint_id,
        "result_group_manifest": group_manifest,
    }
    payload["checkpoint_hash"] = content_hash(payload)
    atomic_write_json(path, payload)

    # Marker chunks are restart artifacts only.  Once the endpoint manifest
    # atomically references its sole result copy, retaining them would double
    # the largest marker/model payloads on disk.
    marker_root = (
        path.parent / endpoint_id / "marker_chunks"
    ).resolve()
    checkpoint_root = path.parent.resolve()
    if (
        marker_root != checkpoint_root
        and marker_root.is_relative_to(checkpoint_root)
        and marker_root.is_dir()
    ):
        for chunk_path in marker_root.glob("markers-*.json"):
            try:
                chunk_path.unlink()
            except OSError:
                pass
        try:
            marker_root.rmdir()
        except OSError:
            pass


def _load_endpoint_validation_checkpoint(
    path: Path,
    *,
    input_hash: str,
    endpoint_id: str,
    materialize_partitions: bool = True,
) -> Optional[dict]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = payload.get("checkpoint_hash")
    hash_payload = dict(payload)
    hash_payload.pop("checkpoint_hash", None)
    if (
        payload.get("checkpoint_version")
        != VALIDATION_CHECKPOINT_VERSION
        or payload.get("checkpoint_type")
        != "completed_endpoint_manifest"
        or payload.get("input_hash") != input_hash
        or payload.get("endpoint_id") != endpoint_id
        or expected_hash != content_hash(hash_payload)
    ):
        raise RuntimeError(
            f"Endpoint checkpoint is stale or corrupt: {path}"
        )
    group_manifest = payload.get("result_group_manifest")
    if not isinstance(group_manifest, list):
        raise RuntimeError(
            f"Endpoint checkpoint has no result-group manifest: {path}"
        )
    result_groups: dict[str, list[dict]] = {}
    result_group_paths = {}
    for record in group_manifest:
        group_name = str(record.get("group_name", ""))
        if not group_name or group_name in result_groups:
            raise RuntimeError(
                f"Endpoint checkpoint has duplicate/empty group: {path}"
            )
        if record.get("inline_empty") is True:
            if int(record.get("row_count", -1)) != 0:
                raise RuntimeError(
                    f"Endpoint checkpoint empty-group count mismatch: {path}"
                )
            result_groups[group_name] = []
            continue
        relative_path = Path(str(record.get("relative_path", "")))
        artifact_path = (path.parent / relative_path).resolve()
        if (
            not artifact_path.is_relative_to(path.parent.resolve())
            or not artifact_path.is_file()
        ):
            raise RuntimeError(
                f"Endpoint checkpoint group is missing/corrupt: {group_name}"
            )
        if record.get("encoding") == "parquet-table-v1":
            import pyarrow.parquet as pq
            if file_sha256(artifact_path) != record.get("sha256"):
                raise RuntimeError(f"Endpoint checkpoint hash mismatch: {group_name}")
            row_count = pq.ParquetFile(artifact_path).metadata.num_rows
            if row_count != int(record.get("row_count", -1)):
                raise RuntimeError(f"Endpoint checkpoint row count mismatch: {group_name}")
            result_group_paths[group_name] = (artifact_path, row_count, record["sha256"])
            result_groups[group_name] = (
                _validation_json_value(pd.read_parquet(artifact_path).to_dict(orient="records"))
                if materialize_partitions else [])
            continue
        if record.get("encoding") != "gzip-json-v1":
            raise RuntimeError(f"Unsupported endpoint checkpoint encoding: {group_name}")
        from .checkpoint_codec import read_group
        artifact = read_group(artifact_path, sha256=record.get("sha256"),
                              uncompressed_bytes=record.get("uncompressed_bytes"))
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Invalid endpoint checkpoint group object: {group_name}")
        artifact_hash_payload = dict(artifact)
        artifact_hash = artifact_hash_payload.pop("content_hash", None)
        records = artifact.get("records")
        if (
            artifact.get("checkpoint_version")
            != VALIDATION_CHECKPOINT_VERSION
            or artifact.get("checkpoint_type")
            != "endpoint_result_group"
            or artifact.get("input_hash") != input_hash
            or artifact.get("endpoint_id") != endpoint_id
            or artifact.get("group_name") != group_name
            or artifact_hash != record.get("content_hash")
            or artifact_hash != content_hash(artifact_hash_payload)
            or not isinstance(records, list)
            or len(records) != int(record.get("row_count", -1))
        ):
            raise RuntimeError(
                f"Endpoint checkpoint group content mismatch: {group_name}"
            )
        result_groups[group_name] = records
    return {**payload, "result_groups": result_groups, "result_group_paths": result_group_paths}


def _extend_validation_result_groups(
    destinations: dict[str, list[dict]],
    source: dict,
    *,
    partition_paths: Optional[dict] = None,
) -> None:
    if set(source) != set(destinations):
        raise RuntimeError(
            "Endpoint checkpoint result-group schema mismatch"
        )
    for name, records in destinations.items():
        if isinstance(records, PartitionedRecords) and name in (partition_paths or {}):
            records.adopt_verified_partition(*partition_paths[name])
            continue
        values = source[name]
        if not isinstance(values, list):
            raise RuntimeError(
                f"Endpoint checkpoint group {name} is not a list"
            )
        records.extend(values)


def _validation_json_value(value):
    if isinstance(value, dict):
        return {
            str(key): _validation_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_validation_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_validation_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _validation_json_value(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


VALIDATION_MINIMUM_CLASS_SUBJECTS = 30


def _development_power_gate(
    catalog: DuckDBCatalog,
    cohort_path: Path,
    endpoint_ids: Iterable[str],
) -> dict[str, tuple[int, int]]:
    """Endpoints below the no-inference floor, from development splits only.

    The threshold is the label stage's ``power_status = 'no_inference'``
    floor; it is recomputed here on train and validation so the gate never
    reads locked-test label counts.
    """
    requested = tuple(str(item) for item in endpoint_ids)
    if not requested or not Path(cohort_path).is_file():
        return {}
    with catalog.connect(isolated=True) as connection:
        rows = connection.execute(
            f"""
            SELECT
                label_id,
                COUNT(DISTINCT subject_id) FILTER (
                    WHERE label_state = 'positive'
                ) AS positive_subjects,
                COUNT(DISTINCT subject_id) FILTER (
                    WHERE label_state = 'negative'
                ) AS negative_subjects
            FROM read_parquet('{_sql_path(cohort_path)}')
            WHERE split IN ('train', 'validation')
            GROUP BY label_id
            """
        ).fetchall()
    counts = {str(label): (int(pos), int(neg)) for label, pos, neg in rows}
    gate = {}
    for endpoint_id in requested:
        positives, negatives = counts.get(endpoint_id, (0, 0))
        if min(positives, negatives) < VALIDATION_MINIMUM_CLASS_SUBJECTS:
            gate[endpoint_id] = (positives, negatives)
    return gate


def _evaluate_marker_chunk_in_processes(
    state: dict,
    *,
    endpoint_id: str,
    feature_ids: tuple[str, ...],
    panel_map: dict[str, str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    endpoint_universe: Optional[pd.DataFrame],
    specification: ValidationSpec,
    bootstrap_iterations: int,
    worker_count: int,
    train_quality: Optional[np.ndarray],
    validation_quality: Optional[np.ndarray],
    feature_index_offset: int,
    progress: Optional[Callable[[str], None]],
) -> Optional[list[_SingleMarkerEvaluation]]:
    """One marker chunk in worker processes (E11), or None for the thread path.

    None is returned before any work when the shared copy of the chunk would
    not fit above the memory reserve, or after the pool broke (a worker was
    killed); the thread path then evaluates the chunk. Results do not depend
    on the path: every marker's seeds derive from its index alone.
    """
    from concurrent.futures.process import BrokenProcessPool

    from .marker_process_pool import (
        MarkerProcessPool,
        SharedChunk,
        live_admission,
        shared_bytes_needed,
    )
    from .resources import available_memory_gb

    if state.get("disabled"):
        return None
    if train_quality is None:
        train_quality = _numeric_feature(train, "signal_quality_score")
    if validation_quality is None:
        validation_quality = _numeric_feature(validation, "signal_quality_score")
    if len(train_quality) != len(train) or len(validation_quality) != len(
        validation
    ):
        raise ValueError("Signal-quality arrays must align with endpoint rows")
    frames = {
        "train": train,
        "validation": validation,
        "endpoint_universe": endpoint_universe,
    }
    arrays = {
        "train_quality": train_quality,
        "validation_quality": validation_quality,
    }
    needed_gb = shared_bytes_needed(frames, arrays) / 1024**3
    available = available_memory_gb()
    reserve = float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "2"))
    if available is None or available - needed_gb < reserve + 1.0:
        return None
    if state.get("pool") is None:
        state["pool"] = MarkerProcessPool(worker_count)
    tasks = [
        (feature_index, feature_id, panel_map[feature_id])
        for feature_index, feature_id in enumerate(
            feature_ids,
            start=feature_index_offset,
        )
    ]
    chunk = SharedChunk(
        frames=frames,
        arrays=arrays,
        constants={
            "endpoint_id": endpoint_id,
            "specification": specification,
            "bootstrap_iterations": bootstrap_iterations,
        },
    )
    try:
        return state["pool"].evaluate(chunk, tasks, live_admission(worker_count))
    except BrokenProcessPool as exc:
        state.pop("pool").close()
        state["disabled"] = True
        if progress:
            progress(
                f"Markerprozesse abgebrochen ({type(exc).__name__}); "
                "Validierung läuft mit Threads weiter"
            )
        return None
    finally:
        chunk.close()


def _evaluate_endpoint_markers(
    *,
    endpoint_id: str,
    feature_ids: tuple[str, ...],
    panel_map: dict[str, str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    endpoint_universe: Optional[pd.DataFrame] = None,
    specification: ValidationSpec,
    bootstrap_iterations: int,
    worker_count: int,
    train_quality: Optional[np.ndarray] = None,
    validation_quality: Optional[np.ndarray] = None,
    feature_index_offset: int = 0,
    executor: Optional[ThreadPoolExecutor] = None,
) -> list[_SingleMarkerEvaluation]:
    if train_quality is None:
        train_quality = _numeric_feature(train, "signal_quality_score")
    if validation_quality is None:
        validation_quality = _numeric_feature(
            validation,
            "signal_quality_score",
        )
    if len(train_quality) != len(train) or len(validation_quality) != len(
        validation
    ):
        raise ValueError("Signal-quality arrays must align with endpoint rows")
    arguments = [
        {
            "endpoint_id": endpoint_id,
            "feature_index": feature_index,
            "feature_id": feature_id,
            "panel_id": panel_map[feature_id],
            "train": train,
            "validation": validation,
            "train_quality": train_quality,
            "validation_quality": validation_quality,
            "endpoint_universe": endpoint_universe,
            "specification": specification,
            "bootstrap_iterations": bootstrap_iterations,
        }
        for feature_index, feature_id in enumerate(
            feature_ids,
            start=feature_index_offset,
        )
    ]
    from .resource_grants import worker_ceiling, granted_workers, GRANT_ENV
    worker_count = worker_ceiling(worker_count)
    bounded_workers = max(1, min(int(worker_count), len(arguments)))
    if bounded_workers > 1:
        from .resources import recommend_dynamic_workers
        # At a marker-chunk boundary there are no active fits. RAM admission
        # therefore budgets fresh fit tasks, not resident executor threads.
        bounded_workers = min(bounded_workers, recommend_dynamic_workers(
            bounded_workers, sample_interval=0.05, current_workers=0,
            reserve_gb=float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "6")),
        ))
    if bounded_workers == 1 and not os.environ.get(GRANT_ENV):
        return [
            _evaluate_single_marker(**argument)
            for argument in arguments
        ]
    def _evaluate_with(active_executor: ThreadPoolExecutor):
        from threading import Condition
        admission = Condition()
        active_count = 0
        last_capacity_check = time.monotonic()
        live_capacity = bounded_workers
        # A reused executor may have a larger physical pool. Limit submitted
        # active fits too, otherwise bounded_workers would have no effect.
        # No batch barrier: a completed fit immediately releases the next one.
        def admitted_fit(argument):
            nonlocal active_count, last_capacity_check, live_capacity
            with admission:
                while True:
                    if os.environ.get(GRANT_ENV) and time.monotonic() - last_capacity_check >= 2.0:
                        from .resources import recommend_dynamic_workers
                        live_capacity = recommend_dynamic_workers(
                            worker_count, sample_interval=0.05, current_workers=active_count,
                            reserve_gb=float(os.environ.get("VCG_JOB_OS_RESERVE_GB", "2")),
                        )
                        last_capacity_check = time.monotonic()
                    if active_count < min(live_capacity, granted_workers(worker_count)):
                        break
                    admission.wait(timeout=2.0)
                active_count += 1
            try:
                return _evaluate_single_marker(**argument)
            finally:
                with admission:
                    active_count -= 1
                    admission.notify_all()
        futures = [active_executor.submit(admitted_fit, argument) for argument in arguments]
        return [future.result() for future in futures]
    if executor is not None:
        return _evaluate_with(executor)
    with ThreadPoolExecutor(
        max_workers=worker_count if os.environ.get(GRANT_ENV) else bounded_workers,
        thread_name_prefix="vcg-marker",
    ) as owned_executor:
        return _evaluate_with(owned_executor)


def _initial_explorer_evidence(criteria: pd.DataFrame, *, preview: bool) -> pd.DataFrame:
    """Internal replication only; later guideline gates cannot be anticipated."""
    if criteria.empty:
        return pd.DataFrame()
    from .result_contract import true

    evidence = criteria.copy()
    evidence["analysis_completeness"] = "partial_preview" if preview else "complete_snapshot"
    gates = ("core_replication_evidence_passed", "project_wide_multiplicity_confirmed",
             "cross_fit_confirmed", "verification_bias_gate_passed")
    confirmed = pd.Series(True, index=evidence.index)
    for gate in gates:
        confirmed &= evidence.get(gate, pd.Series(False, index=evidence.index)).map(true)
    # Audit X6: a marker formulated after looking at pilot results on the
    # validation half cannot be replicated in that same half.
    post_outcome = evidence.get(
        "post_outcome_hypothesis", pd.Series(False, index=evidence.index)
    ).map(true)
    evidence["replication_independence"] = np.where(
        post_outcome,
        "not_independent_post_outcome_hypothesis",
        "prespecified_marker",
    )
    evidence["replication_passed"] = (
        confirmed & evidence["stage"].eq("validation_replicated") & ~post_outcome
    )
    low = pd.to_numeric(evidence.get("roc_auc_ci_low", pd.Series(np.nan, index=evidence.index)), errors="coerce")
    high = pd.to_numeric(evidence.get("roc_auc_ci_high", pd.Series(np.nan, index=evidence.index)), errors="coerce")
    effect = pd.to_numeric(evidence.get("roc_auc", pd.Series(np.nan, index=evidence.index)), errors="coerce")
    evidence["effect_ci_passed"] = (low.ge(0) & high.le(1) & low.le(effect) & high.ge(effect))
    return evidence


def _evaluate_single_marker(
    *,
    endpoint_id: str,
    feature_index: int,
    feature_id: str,
    panel_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_quality: np.ndarray,
    validation_quality: np.ndarray,
    endpoint_universe: Optional[pd.DataFrame] = None,
    specification: ValidationSpec,
    bootstrap_iterations: int,
) -> _SingleMarkerEvaluation:
    from .projection_hypotheses import projection_analysis_eligibility

    train_raw_values = _numeric_feature(train, feature_id)
    validation_raw_values = _numeric_feature(validation, feature_id)
    train_eligible, train_restriction = projection_analysis_eligibility(
        train,
        feature_id,
    )
    validation_eligible, validation_restriction = projection_analysis_eligibility(
        validation,
        feature_id,
    )
    train_values = np.where(train_eligible, train_raw_values, np.nan)
    validation_values = np.where(
        validation_eligible,
        validation_raw_values,
        np.nan,
    )
    try:
        model = fit_marker_model(
            train_values,
            train["label"].to_numpy(),
            feature_id,
        )
    except (ValueError, TypeError) as exc:
        return _SingleMarkerEvaluation(
            feature_id=feature_id,
            panel_id=panel_id,
            failure={
                "endpoint_id": endpoint_id,
                "feature_id": feature_id,
                "stage": "single_marker_fit",
                "reason": str(exc),
            },
        )

    train_record = _marker_record(
        endpoint_id,
        panel_id,
        feature_id,
        "train",
        train,
        train_values,
        model,
        specification,
        bootstrap_iterations,
        seed_offset=feature_index * 2,
        quality_values=train_quality,
        raw_values=train_raw_values,
        analysis_eligibility=train_eligible,
        analysis_restriction=train_restriction,
    )
    validation_record = _marker_record(
        endpoint_id,
        panel_id,
        feature_id,
        "validation",
        validation,
        validation_values,
        model,
        specification,
        bootstrap_iterations,
        seed_offset=feature_index * 2 + 1,
        fitting_split="train",
        confirmation_fold=FOLD_A,
        quality_values=validation_quality,
        raw_values=validation_raw_values,
        analysis_eligibility=validation_eligible,
        analysis_restriction=validation_restriction,
    )
    reverse_model: Optional[MarkerModel] = None
    reverse_confirmation_record: Optional[dict] = None
    cross_fit_failure: Optional[dict] = None
    try:
        reverse_model = fit_marker_model(
            validation_values,
            validation["label"].to_numpy(),
            feature_id,
        )
        reverse_confirmation_record = _marker_record(
            endpoint_id,
            panel_id,
            feature_id,
            "train",
            train,
            train_values,
            reverse_model,
            specification,
            bootstrap_iterations,
            seed_offset=feature_index * 2 + 10_000_003,
            fitting_split="validation",
            confirmation_fold=FOLD_B,
            quality_values=train_quality,
            raw_values=train_raw_values,
            analysis_eligibility=train_eligible,
            analysis_restriction=train_restriction,
        )
    except (ValueError, TypeError) as exc:
        cross_fit_failure = {
            "endpoint_id": endpoint_id,
            "feature_id": feature_id,
            "stage": "cross_fit_reverse_marker_fit",
            "reason": str(exc),
        }
    stability = [
        _stability_record(
            endpoint_id,
            panel_id,
            feature_id,
            model,
            train_record,
            validation_record,
            validation_values,
            validation["label"].to_numpy(),
        )
    ]
    if panel_id != "classical":
        stability.extend(
            _transform_sensitivity_records(
                endpoint_id,
                panel_id,
                feature_id,
                train,
                validation,
                train_record,
                validation_record,
            )
        )
    fixed = (
        _fixed_classical_records(
            endpoint_id,
            feature_id,
            train,
            validation,
            specification,
        )
        if panel_id == "classical"
        else []
    )
    verification = (
        _marker_verification_bias_records(
            endpoint_id=endpoint_id,
            panel_id=panel_id,
            feature_id=feature_id,
            endpoint_universe=endpoint_universe,
            model=model,
        )
        if endpoint_universe is not None
        else []
    )
    return _SingleMarkerEvaluation(
        feature_id=feature_id,
        panel_id=panel_id,
        model=model,
        train_record=train_record,
        validation_record=validation_record,
        reverse_model=reverse_model,
        reverse_confirmation_record=reverse_confirmation_record,
        cross_fit_failure=cross_fit_failure,
        stability_records=tuple(stability),
        classical_records=tuple(fixed),
        verification_bias_records=tuple(verification),
    )


def _hf_qrs_snr_fields(
    frame: pd.DataFrame,
    feature_id: str,
    valid: np.ndarray,
) -> dict:
    """Band SNR next to an 80-150 Hz marker result (audit S7)."""
    from .hf_qrs_quality import (
        HF_QRS_BAND_MARKERS,
        MINIMUM_REPORTABLE_AVERAGED_SNR_DB,
    )

    if feature_id not in HF_QRS_BAND_MARKERS:
        return {}
    snr = _numeric_feature(frame, "context_hf_qrs_averaged_snr_db")[
        np.asarray(valid, dtype=bool)
    ]
    measured = snr[np.isfinite(snr)]
    median = float(np.median(measured)) if measured.size else float("nan")
    return {
        "hf_qrs_band_snr_median_db": median,
        "hf_qrs_band_snr_reportable_fraction": (
            float(np.mean(measured >= MINIMUM_REPORTABLE_AVERAGED_SNR_DB))
            if measured.size
            else float("nan")
        ),
        "hf_qrs_band_snr_measured": int(measured.size),
        "hf_qrs_band_reportable": bool(
            np.isfinite(median) and median >= MINIMUM_REPORTABLE_AVERAGED_SNR_DB
        ),
    }


def _regular_rhythm_metrics(
    frame: pd.DataFrame,
    values: np.ndarray,
    labels: np.ndarray,
    direction: int,
) -> dict[str, float | int]:
    """The marker's AUC inside the regular-rhythm stratum, next to the pooled one.

    ``rhythm_context`` is the technical delineation pattern, not a diagnosis;
    ``regular`` is the stratum in which every marker family is measurable, so
    it is the one stratum on which two markers with different availability
    can be read side by side. Fewer than ten of either class leaves the AUC
    blank rather than reporting a number nobody should read.
    """

    output: dict[str, float | int] = {
        "roc_auc_regular_rhythm": np.nan,
        "n_regular_positive": 0,
        "n_regular_negative": 0,
        "regular_rhythm_fraction_of_evaluable": np.nan,
    }
    if "rhythm_context" not in frame:
        return output
    regular = frame["rhythm_context"].astype(str).eq(RHYTHM_STRATUM_REGULAR).to_numpy()
    valid = np.isfinite(values)
    mask = regular & valid
    n_positive = int(np.sum(labels[mask] == 1))
    n_negative = int(np.sum(labels[mask] == 0))
    output["n_regular_positive"] = n_positive
    output["n_regular_negative"] = n_negative
    if valid.any():
        output["regular_rhythm_fraction_of_evaluable"] = float(
            mask.sum() / valid.sum()
        )
    if (
        n_positive >= RHYTHM_STRATUM_MINIMUM_CLASS
        and n_negative >= RHYTHM_STRATUM_MINIMUM_CLASS
    ):
        output["roc_auc_regular_rhythm"] = float(
            roc_auc_score(labels[mask], int(direction) * values[mask])
        )
    return output


def _observed_reference_kind(frame: pd.DataFrame, valid: np.ndarray) -> str:
    """Summarize only observed categories, avoiding row-wise string conversion.

    The mask must remain marker-specific: unavailable values must not make a
    single-reference result appear mixed. String conversion follows unique
    reduction, preserving the existing textual reference contract.
    """
    if "reference_kind" not in frame:
        return "unverified_or_unknown" if np.any(valid) else "mixed_or_unknown"
    observed = {str(value) for value in frame.loc[valid, "reference_kind"].dropna().unique()}
    return next(iter(observed)) if len(observed) == 1 else "mixed_or_unknown"


def _marker_record(
    endpoint_id: str,
    panel_id: str,
    feature_id: str,
    split: str,
    frame: pd.DataFrame,
    values: np.ndarray,
    model: MarkerModel,
    specification: ValidationSpec,
    bootstrap_iterations: int,
    *,
    seed_offset: int,
    fitting_split: str = "train",
    confirmation_fold: Optional[str] = None,
    quality_values: Optional[np.ndarray] = None,
    raw_values: Optional[np.ndarray] = None,
    analysis_eligibility: Optional[np.ndarray] = None,
    analysis_restriction: Optional[dict] = None,
) -> dict:
    labels = frame["label"].to_numpy(dtype=np.int8)
    valid = np.isfinite(values)
    raw = values if raw_values is None else np.asarray(raw_values, dtype=float)
    eligible = (
        np.ones(len(frame), dtype=bool)
        if analysis_eligibility is None
        else np.asarray(analysis_eligibility, dtype=bool)
    )
    if len(raw) != len(frame) or len(eligible) != len(frame):
        raise ValueError("Marker restriction arrays must align with endpoint rows")
    restriction = analysis_restriction or {
        "status": "not_applicable",
        "version": None,
        "required_columns_present": True,
    }
    metrics = evaluate_marker(
        values,
        labels,
        direction=model.direction,
        thresholds=model.thresholds,
    )
    n_positive = int(metrics["n_positive"])
    n_negative = int(metrics["n_negative"])
    status = inference_status(
        n_positive,
        n_negative,
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    metrics = enforce_inference_policy(metrics, status)
    profile = feature_profile(feature_id)
    quality = (
        _numeric_feature(frame, "signal_quality_score")
        if quality_values is None
        else np.asarray(quality_values, dtype=float)
    )
    if len(quality) != len(frame):
        raise ValueError("Signal-quality values must align with marker rows")
    quality_evaluable = valid & np.isfinite(quality)
    reference_kind = _observed_reference_kind(frame, valid)
    record = {
        "endpoint_id": endpoint_id,
        "panel_id": panel_id,
        "feature_id": feature_id,
        "canonical_feature_id": profile.canonical_feature_id,
        "feature_role": profile.role,
        "endpoint_feature_role": endpoint_feature_role(
            feature_id,
            endpoint_id,
        ),
        "feature_modality": profile.modality,
        "conceptual_family": profile.conceptual_family,
        "scientific_tier": profile.scientific_tier,
        "post_outcome_hypothesis": profile.post_outcome_hypothesis,
        "feature_role_policy_version": FEATURE_ROLE_POLICY_VERSION,
        "analysis_type": "single_marker",
        "evaluation_split": split,
        "metric_scope": (
            "cross_fit_out_of_fold_confirmation"
            if confirmation_fold is not None
            else {
                "train": "train_in_sample",
                "validation": "validation_holdout",
                "locked_test": "locked_test_final",
            }.get(split, split)
        ),
        "threshold_source": fitting_split,
        "direction_source": fitting_split,
        "cross_fit_fold": confirmation_fold,
        "fitting_split": fitting_split,
        "confirming_split": split if confirmation_fold is not None else None,
        "cross_fitting_version": (
            CROSS_FITTING_VERSION if confirmation_fold is not None else None
        ),
        "direction": model.direction,
        "missingness": metrics.get("overall_missingness", np.nan),
        "mean_signal_quality": (
            float(np.mean(quality[quality_evaluable]))
            if quality_evaluable.any()
            else np.nan
        ),
        "low_signal_quality_fraction": (
            float(np.mean(quality[quality_evaluable] < 0.60))
            if quality_evaluable.any()
            else np.nan
        ),
        "inference_status": status,
        "exploratory_score": False,
        "reference_kind": reference_kind,
        "n_independent_modality_observed": int(frame.get(
            "independent_reference_modality_available", pd.Series(False, index=frame.index)
        ).loc[valid].eq(True).sum()),
        "n_reference_alignment_confirmed": int(frame.get(
            "reference_alignment_confirmed", pd.Series(False, index=frame.index)
        ).loc[valid].eq(True).sum()),
        "independent_reference_passed": bool(reference_kind == "timed_quantitative"
            and frame.get("reference_alignment_confirmed", pd.Series(False, index=frame.index)).loc[valid].eq(True).all()),
        "n_reference_eligible": len(frame),
        "n_marker_raw_observed": int(np.isfinite(raw).sum()),
        "n_marker_observed": int(valid.sum()),
        "n_marker_unavailable": int((~np.isfinite(raw)).sum()),
        "n_analysis_restriction_excluded": int(
            np.sum(np.isfinite(raw) & ~eligible)
        ),
        "analysis_restriction_status": restriction.get("status"),
        "analysis_restriction_version": restriction.get("version"),
        "analysis_restriction_required_columns_present": bool(
            restriction.get("required_columns_present", False)
        ),
        "analysis_restriction_missing_columns": restriction.get(
            "missing_columns", []
        ),
        "availability_denominator": "selected_reference_eligible_index_records",
        "bootstrap_scope": (
            f"{split}_subject_cluster"
            if split == "validation" or confirmation_fold is not None
            else "not_run_for_train_in_sample"
        ),
        **metrics,
        **_regular_rhythm_metrics(frame, values, labels, model.direction),
        **_hf_qrs_snr_fields(frame, feature_id, valid),
    }
    record.update(
        marker_bootstrap_intervals(
            frame["subject_id"].to_numpy(),
            values,
            labels,
            direction=model.direction,
            thresholds=model.thresholds,
            iterations=(
                bootstrap_iterations
                if status != "descriptive_only"
                and (split == "validation" or confirmation_fold is not None)
                else 0
            ),
            seed=specification.random_seed + seed_offset,
            screen_uninformative=(
                split == "validation" or confirmation_fold is not None
            ),
            negligible_auc_margin=(
                VALIDATION_BOOTSTRAP_NEGLIGIBLE_AUC_MARGIN
            ),
        )
    )
    return record


@profiled_phase("verification_bias")
def _marker_verification_bias_records(
    *,
    endpoint_id: str,
    panel_id: str,
    feature_id: str,
    endpoint_universe: pd.DataFrame,
    model: MarkerModel,
) -> list[dict]:
    """Sharp fixed-threshold bounds over the reference-unverified cohort."""

    if feature_id not in endpoint_universe:
        return []
    from .projection_hypotheses import (
        P_PROJECTION_CONTEXT_COLUMNS,
        projection_analysis_eligibility,
    )

    # Reduce columns before filtering rows. Copying a whole marker chunk for
    # each marker and split scales quadratically in the chunk width.
    context_columns = [
        column
        for column in P_PROJECTION_CONTEXT_COLUMNS
        if column in endpoint_universe
    ]
    required = endpoint_universe.loc[
        :,
        ["split", "label_state", feature_id, *context_columns],
    ]
    records: list[dict] = []
    for split in ("train", "validation"):
        scoped = required.loc[
            required["split"].eq(split),
            ["label_state", feature_id, *context_columns],
        ]
        if scoped.empty:
            continue
        raw_values = _numeric_feature(scoped, feature_id)
        analysis_eligible, restriction = projection_analysis_eligibility(
            scoped,
            feature_id,
        )
        values = np.where(analysis_eligible, raw_values, np.nan)
        marker_observed = np.isfinite(values)
        # Verification and marker availability are two distinct missing-data
        # processes.  Known references count as verified even when the marker
        # is absent; unknown references count as unverified even when their
        # marker is absent.
        verified = scoped["label_state"].isin(
            ("positive", "negative")
        ).to_numpy()
        positive = scoped["label_state"].eq("positive").to_numpy()
        negative = scoped["label_state"].eq("negative").to_numpy()
        for threshold_name in (
            "specificity_95",
            "specificity_90",
            "sensitivity_90",
        ):
            if threshold_name not in model.thresholds:
                continue
            threshold = float(model.thresholds[threshold_name])
            marker_positive = model.direction * values >= threshold
            verified_observed = verified & marker_observed
            unverified_observed = ~verified & marker_observed
            verified_positive = int(np.sum(verified & positive))
            verified_negative = int(np.sum(verified & negative))
            unverified = int(np.sum(~verified))
            bounds = verification_bias_bounds(
                verified_positive,
                verified_negative,
                unverified,
                marker_positive_verified=int(
                    np.sum(verified_observed & positive & marker_positive)
                ),
                marker_positive_unverified=int(
                    np.sum(unverified_observed & marker_positive)
                ),
                marker_positive_verified_negative=int(
                    np.sum(verified_observed & negative & marker_positive)
                ),
                marker_missing_verified_positive=int(
                    np.sum(verified & positive & ~marker_observed)
                ),
                marker_missing_verified_negative=int(
                    np.sum(verified & negative & ~marker_observed)
                ),
                marker_missing_unverified=int(
                    np.sum(~verified & ~marker_observed)
                ),
            )
            sensitivity_width = _finite_or_none(
                bounds.get("sensitivity_bound_width")
            )
            specificity_width = _finite_or_none(
                bounds.get("specificity_bound_width")
            )
            gate_passed = bool(
                bounds.get("status") == "evaluable"
                and (
                    _finite_or_none(bounds.get("verification_rate")) or 0.0
                ) >= MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION
                and sensitivity_width is not None
                and sensitivity_width <= MAXIMUM_VERIFICATION_BOUND_WIDTH
                and specificity_width is not None
                and specificity_width <= MAXIMUM_VERIFICATION_BOUND_WIDTH
            )
            records.append(
                {
                    "endpoint_id": endpoint_id,
                    "panel_id": panel_id,
                    "feature_id": feature_id,
                    "evaluation_split": split,
                    "operating_point": threshold_name,
                    "threshold_aligned": threshold,
                    "threshold_original": float(model.direction * threshold),
                    "direction": int(model.direction),
                    "n_reference_blind_universe": int(len(scoped)),
                    "n_marker_observed": int(np.sum(marker_observed)),
                    "n_marker_missing": int(np.sum(~np.isfinite(raw_values))),
                    "n_analysis_restriction_excluded": int(
                        np.sum(np.isfinite(raw_values) & ~analysis_eligible)
                    ),
                    "analysis_restriction_status": restriction.get("status"),
                    "analysis_restriction_version": restriction.get("version"),
                    "n_verified_marker_observed": int(
                        np.sum(verified_observed)
                    ),
                    "n_unverified_marker_observed": int(
                        np.sum(unverified_observed)
                    ),
                    "verification_bias_gate_passed": gate_passed,
                    "minimum_verification_rate": (
                        MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION
                    ),
                    "maximum_bound_width": MAXIMUM_VERIFICATION_BOUND_WIDTH,
                    "index_selection_strategy": (
                        "first_nonexcluded_ecg_reference_blind"
                    ),
                    **bounds,
                }
            )
    return records


def _fixed_classical_records(
    endpoint_id: str,
    feature_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
) -> list[dict]:
    positive_column = f"{feature_id}_positive"
    if positive_column not in train or positive_column not in validation:
        return []
    records = []
    for split, frame in (("train", train), ("validation", validation)):
        raw = frame[positive_column]
        valid = raw.notna().to_numpy()
        if not valid.any():
            continue
        labels = frame.loc[valid, "label"].to_numpy(dtype=np.int8)
        predictions = raw.loc[valid].astype(bool).to_numpy()
        counts = classification_metrics(labels, predictions)
        marker_metrics = evaluate_marker(
            predictions.astype(float),
            labels,
        )
        status = inference_status(
            int(np.sum(labels == 1)),
            int(np.sum(labels == 0)),
            minimum_inference=specification.min_class_inference,
            minimum_primary=specification.min_class_primary,
        )
        uncertainty = _binary_baseline_intervals(
            frame.loc[valid, "subject_id"].to_numpy(),
            predictions,
            labels,
            specification,
            status=status,
            seed_key=f"{endpoint_id}:{feature_id}:{split}",
            evaluation_split=split,
        )
        records.append(
            {
                "endpoint_id": endpoint_id,
                "baseline_id": feature_id,
                "baseline_type": "classical_rule",
                "evaluation_split": split,
                "threshold_source": "predefined_clinical_rule",
                "n_total": len(labels),
                "n_positive": int(np.sum(labels == 1)),
                "n_negative": int(np.sum(labels == 0)),
                "missingness": float(1.0 - np.mean(valid)),
                "roc_auc": marker_metrics["roc_auc"],
                "average_precision": marker_metrics["average_precision"],
                "pr_auc": marker_metrics["pr_auc"],
                "inference_status": status,
                **counts,
                **uncertainty,
            }
        )
    return records


def _machine_report_records(
    endpoint_id: str,
    report_label_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
) -> list[dict]:
    records = []
    for split, frame in (("train", train), ("validation", validation)):
        states = frame["report_reference_state"].astype("string")
        valid = states.isin(
            ("positive", "negative", "not_mentioned")
        ).to_numpy()
        if not valid.any():
            continue
        labels = frame.loc[valid, "label"].to_numpy(dtype=np.int8)
        state_values = states.loc[valid].astype(str).to_numpy()
        all_state_values = states.dropna().astype(str).to_numpy()
        predictions = state_values == "positive"
        counts = classification_metrics(labels, predictions)
        marker_metrics = evaluate_marker(
            predictions.astype(float),
            labels,
        )
        status = inference_status(
            int(np.sum(labels == 1)),
            int(np.sum(labels == 0)),
            minimum_inference=specification.min_class_inference,
            minimum_primary=specification.min_class_primary,
        )
        uncertainty = _binary_baseline_intervals(
            frame.loc[valid, "subject_id"].to_numpy(),
            predictions,
            labels,
            specification,
            status=status,
            seed_key=f"{endpoint_id}:report:{report_label_id}:{split}",
            evaluation_split=split,
        )
        records.append(
            {
                "endpoint_id": endpoint_id,
                "baseline_id": f"machine_report_{report_label_id}",
                "baseline_type": "machine_report_positive_only",
                "evaluation_split": split,
                "threshold_source": "report_parser_v1",
                "n_total": len(labels),
                "n_source_rows": len(frame),
                "n_positive": int(np.sum(labels == 1)),
                "n_negative": int(np.sum(labels == 0)),
                "missingness": float(1.0 - np.mean(valid)),
                "roc_auc": marker_metrics["roc_auc"],
                "average_precision": marker_metrics["average_precision"],
                "pr_auc": marker_metrics["pr_auc"],
                "positive_count": int(
                    np.sum(all_state_values == "positive")
                ),
                "negative_count": int(
                    np.sum(all_state_values == "negative")
                ),
                "uncertain_count": int(
                    np.sum(all_state_values == "uncertain")
                ),
                "not_mentioned_count": int(
                    np.sum(all_state_values == "not_mentioned")
                ),
                "unavailable_count": int(states.isna().sum()),
                "uncertain_excluded": True,
                "inference_status": status,
                **counts,
                **uncertainty,
            }
        )
    return records


def _binary_baseline_intervals(
    subject_ids: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    specification: ValidationSpec,
    *,
    status: str,
    seed_key: str,
    evaluation_split: str = "validation",
    final: bool = False,
) -> dict:
    iterations = (
        specification.final_bootstrap_iterations
        if final
        else specification.development_bootstrap_iterations
    )
    if status == "descriptive_only":
        iterations = 0
    if not final and evaluation_split == "train":
        iterations = 0
    seed_offset = int(content_hash(seed_key)[:8], 16)
    intervals = marker_bootstrap_intervals(
        subject_ids,
        predictions.astype(float),
        labels,
        direction=1,
        thresholds={"predefined_rule": 0.5},
        iterations=iterations,
        seed=specification.random_seed + seed_offset,
    )
    for metric in (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
    ):
        intervals[f"{metric}_ci_low"] = intervals[
            f"predefined_rule_{metric}_ci_low"
        ]
        intervals[f"{metric}_ci_high"] = intervals[
            f"predefined_rule_{metric}_ci_high"
        ]
    return intervals


def _fit_predefined_score(
    endpoint_id: str,
    components: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    bootstrap_iterations: int,
) -> tuple[list[dict], dict, dict]:
    score_id = f"{endpoint_id}__predefined_vcg_score"
    score_spec = fit_frozen_score(
        train,
        train["label"].to_numpy(),
        score_id=score_id,
        endpoint_id=endpoint_id,
        components=components,
        origin="mimic_development",
        version="mimic-score-v1.0.0",
        subject_ids=train["subject_id"].to_numpy(),
    )
    train_score = apply_frozen_score(train, score_spec)
    validation_score = apply_frozen_score(validation, score_spec)
    score_model = fit_marker_model(
        train_score,
        train["label"].to_numpy(),
        score_id,
    )
    records = []
    for split_index, (split, frame, values) in enumerate(
        (
            ("train", train, train_score),
            ("validation", validation, validation_score),
        )
    ):
        record = _marker_record(
            endpoint_id,
            "predefined_score",
            score_id,
            split,
            frame,
            values,
            score_model,
            specification,
            bootstrap_iterations,
            seed_offset=50_000 + split_index,
        )
        record.update(
            {
                "analysis_type": "predefined_combination",
                "components": "|".join(components),
                "component_count": len(components),
                "component_directions": "|".join(
                    str(value) for value in score_spec.directions
                ),
                "normalization": "train_zscore_complete_case",
                "aggregation": "unweighted_mean",
                "exploratory_score": True,
                "origin": score_spec.origin,
            }
        )
        records.append(record)
    stability = _stability_record(
        endpoint_id,
        "predefined_score",
        score_id,
        score_model,
        records[0],
        records[1],
        validation_score,
        validation["label"].to_numpy(),
    )
    stability["analysis_type"] = "predefined_combination"
    payload = {
        "score_spec": score_spec.to_dict(),
        "marker_model": score_model.to_dict(),
    }
    return records, payload, stability


def _evaluate_ptb_frozen_score(
    endpoint_id: str,
    source_payload: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    bootstrap_iterations: int,
    *,
    transfer_index: int,
) -> tuple[list[dict], dict, dict]:
    source_spec = _frozen_score_from_dict(source_payload["score_spec"])
    source_model = MarkerModel.from_dict(source_payload["marker_model"])
    score_id = (
        f"{endpoint_id}__{source_payload['ptb_endpoint_id']}"
        "__ptb_frozen_transfer"
    )
    score_spec = replace(
        source_spec,
        score_id=score_id,
        endpoint_id=endpoint_id,
        origin=f"ptbxl_frozen:{source_payload['ptb_endpoint_id']}",
    )
    score_model = MarkerModel(
        feature_id=score_id,
        direction=source_model.direction,
        mean=source_model.mean,
        standard_deviation=source_model.standard_deviation,
        thresholds=source_model.thresholds,
        n_train=source_model.n_train,
        n_positive_train=source_model.n_positive_train,
        n_negative_train=source_model.n_negative_train,
    )
    train_score = apply_frozen_score(train, score_spec)
    validation_score = apply_frozen_score(validation, score_spec)
    records = []
    for split_index, (split, frame, values) in enumerate(
        (
            ("train", train, train_score),
            ("validation", validation, validation_score),
        )
    ):
        record = _marker_record(
            endpoint_id,
            "ptb_frozen_score",
            score_id,
            split,
            frame,
            values,
            score_model,
            specification,
            bootstrap_iterations,
            seed_offset=(
                80_000 + transfer_index * 10 + split_index
            ),
        )
        record.update(
            {
                "analysis_type": "ptb_frozen_transfer",
                "components": "|".join(score_spec.components),
                "component_count": len(score_spec.components),
                "component_directions": "|".join(
                    str(value) for value in score_spec.directions
                ),
                "normalization": "ptb_folds_1_8_frozen_zscore",
                "aggregation": score_spec.aggregation,
                "exploratory_score": True,
                "origin": score_spec.origin,
                "threshold_source": "ptb_folds_1_8",
                "ptb_endpoint_id": source_payload["ptb_endpoint_id"],
                "ptb_validation_status": source_payload.get(
                    "ptb_validation_status"
                ),
                "ptb_validation_roc_auc": source_payload.get(
                    "ptb_validation_roc_auc"
                ),
                "mimic_refit_performed": False,
            }
        )
        records.append(record)
    stability = _stability_record(
        endpoint_id,
        "ptb_frozen_score",
        score_id,
        score_model,
        records[0],
        records[1],
        validation_score,
        validation["label"].to_numpy(),
    )
    stability.update(
        {
            "analysis_type": "ptb_frozen_transfer",
            "ptb_endpoint_id": source_payload["ptb_endpoint_id"],
            "mimic_refit_performed": False,
        }
    )
    payload = {
        "score_spec": score_spec.to_dict(),
        "marker_model": score_model.to_dict(),
        "analysis_type": "ptb_frozen_transfer",
        "normalization": "ptb_folds_1_8_frozen_zscore",
        "threshold_source": "ptb_folds_1_8",
        "ptb_endpoint_id": source_payload["ptb_endpoint_id"],
        "mimic_refit_performed": False,
    }
    return records, payload, stability


def _frozen_score_from_dict(payload: dict) -> FrozenScoreSpec:
    values = dict(payload)
    for key in (
        "components",
        "directions",
        "means",
        "standard_deviations",
    ):
        values[key] = tuple(values[key])
    return FrozenScoreSpec(**values)


def _fit_exploratory_scores(
    endpoint_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    bootstrap_iterations: int,
    panel_map: dict[str, str],
    endpoint_models: dict[str, MarkerModel],
    train_records: dict[str, dict],
    validation_records: dict[str, dict],
    feature_value_loader: Optional[
        Callable[[tuple[str, ...]], dict[str, np.ndarray]]
    ] = None,
    feature_frame_loader: Optional[
        Callable[
            [tuple[str, ...]],
            tuple[pd.DataFrame, pd.DataFrame],
        ]
    ] = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    candidate_features = []
    for feature_id in endpoint_models:
        panel_id = panel_map.get(feature_id)
        train_record = train_records.get(feature_id, {})
        if (
            panel_id == "classical"
            or not _is_clinical_combination_candidate(feature_id)
        ):
            continue
        train_auc = _optional_float(train_record.get("roc_auc"))
        if train_auc is None:
            continue
        if float(train_record.get("missingness", 1.0)) > 0.20:
            continue
        candidate_features.append(feature_id)
    grouped_candidates = grouped_score_search(
        train,
        labels=train["label"].to_numpy(dtype=np.int8),
        subject_ids=train["subject_id"].to_numpy(),
        feature_ids=tuple(candidate_features),
        family_for_feature=_feature_family,
        folds=specification.grouped_score_folds,
        max_components=3,
        maximum_pairwise_abs_correlation=(
            specification.maximum_component_abs_correlation
        ),
        minimum_incremental_auc=(
            specification.minimum_score_incremental_auc
        ),
        cv_repeats=specification.grouped_score_repeats,
        random_seed=specification.random_seed,
        required_component_predicate=lambda feature_id: (
            feature_profile(feature_id).modality == "vcg"
        ),
        representatives_per_family=(
            specification.grouped_score_representatives_per_family
        ),
        selected_pairs=specification.grouped_score_selected_pairs,
        selected_triples=specification.grouped_score_selected_triples,
        triple_seed_pairs=specification.grouped_score_triple_seed_pairs,
        feature_value_loader=feature_value_loader,
    )

    if grouped_candidates and feature_frame_loader is not None:
        component_ids = tuple(
            dict.fromkeys(
                component
                for candidate in grouped_candidates
                for component in candidate.components
            )
        )
        train, validation = feature_frame_loader(component_ids)

    records: list[dict] = []
    payloads: list[dict] = []
    stability_rows: list[dict] = []
    failures: list[dict] = []
    for candidate in grouped_candidates:
        components = candidate.components
        component_count = len(components)
        component_modalities = tuple(
            feature_profile(component).modality for component in components
        )
        if set(component_modalities) == {"vcg"}:
            score_modality = "vcg_score"
        else:
            score_modality = "mixed_ecg_vcg_score"
        score_id = (
            f"{endpoint_id}__exploratory_cross_family_{component_count}__"
            f"{content_hash(components)[:10]}"
        )
        try:
            score_spec = fit_frozen_score(
                train,
                train["label"].to_numpy(),
                score_id=score_id,
                endpoint_id=endpoint_id,
                components=components,
                origin="mimic_exploratory_development",
                version="mimic-exploratory-score-v1.0.0",
                subject_ids=train["subject_id"].to_numpy(),
            )
            train_score = apply_frozen_score(train, score_spec)
            validation_score = apply_frozen_score(validation, score_spec)
            score_model = fit_marker_model(
                train_score,
                train["label"].to_numpy(),
                score_id,
            )
            score_records = []
            for split_index, (split, frame, values) in enumerate(
                (
                    ("train", train, train_score),
                    ("validation", validation, validation_score),
                )
            ):
                record = _marker_record(
                    endpoint_id,
                    "exploratory_score",
                    score_id,
                    split,
                    frame,
                    values,
                    score_model,
                    specification,
                    bootstrap_iterations,
                    seed_offset=80_000 + component_count * 10 + split_index,
                )
                record.update(
                    {
                        "analysis_type": "exploratory_mimic_combination",
                        "components": "|".join(components),
                        "component_families": "|".join(
                            _feature_family(component)
                            for component in components
                        ),
                        "component_modalities": "|".join(
                            component_modalities
                        ),
                        "feature_modality": score_modality,
                        "feature_role_policy_version": (
                            FEATURE_ROLE_POLICY_VERSION
                        ),
                        "component_count": component_count,
                        "component_directions": "|".join(
                            str(value) for value in score_spec.directions
                        ),
                        "normalization": "train_zscore_complete_case",
                        "aggregation": "unweighted_mean",
                        "exploratory_score": True,
                        "origin": score_spec.origin,
                    }
                )
                score_records.append(record)
            stability = _stability_record(
                endpoint_id,
                "exploratory_score",
                score_id,
                score_model,
                score_records[0],
                score_records[1],
                validation_score,
                validation["label"].to_numpy(),
            )
            validation_auc = _optional_float(score_records[1].get("roc_auc"))
            gap = _optional_float(stability.get("train_validation_auc_gap"))
            best_component_id = candidate.best_component
            best_component_model = endpoint_models.get(best_component_id)
            paired_increment = {}
            if (
                best_component_id
                and best_component_model is not None
                and best_component_id in validation
            ):
                best_component_values = (
                    _numeric_feature(validation, best_component_id)
                    * best_component_model.direction
                )
                paired_increment = paired_discrimination_difference_bootstrap(
                    validation["subject_id"].to_numpy(),
                    validation["label"].to_numpy(dtype=np.int8),
                    validation_score,
                    best_component_values,
                    iterations=(
                        bootstrap_iterations
                        if score_records[1].get("inference_status")
                        != "descriptive_only"
                        else 0
                    ),
                    seed=(
                        specification.random_seed
                        + 91_000
                        + component_count
                    ),
                )
            validation_increment = _optional_float(
                paired_increment.get("delta_auc")
            )
            stable = bool(
                stability.get("direction_agreement") is True
                and validation_auc is not None
                and validation_auc >= 0.55
                and gap is not None
                and abs(gap) <= 0.10
                and candidate.incremental_auc_vs_best_component
                >= specification.minimum_score_incremental_auc
                and validation_increment is not None
                and validation_increment >= 0.0
                and score_records[1].get("inference_status")
                != "descriptive_only"
            )
            for record in score_records:
                record["validation_stable"] = stable
                record["selected_for_freeze"] = stable
                record["selection_rule"] = (
                    f"{specification.grouped_score_folds} subject-grouped "
                    f"train folds repeated "
                    f"{specification.grouped_score_repeats} time(s); canonical "
                    "cross-family components with |Spearman rho|<"
                    f"{specification.maximum_component_abs_correlation:.2f}; "
                    "train OOF gain >="
                    f"{specification.minimum_score_incremental_auc:.3f} over "
                    "the best component; validation direction "
                    "agreement, AUC>=0.55, non-negative paired validation gain "
                    "and absolute train-validation gap<=0.10"
                )
                record["grouped_cv_mean_auc"] = candidate.mean_auc
                record["grouped_cv_mean_average_precision"] = (
                    candidate.mean_average_precision
                )
                record["grouped_cv_evaluable_folds"] = (
                    candidate.evaluable_folds
                )
                record["grouped_cv_auc_standard_deviation"] = (
                    candidate.fold_auc_standard_deviation
                )
                record["grouped_cv_stability_adjusted_auc"] = (
                    candidate.stability_adjusted_auc
                )
                record["grouped_cv_incremental_auc_vs_best_component"] = (
                    candidate.incremental_auc_vs_best_component
                )
                record["maximum_pairwise_abs_correlation"] = (
                    candidate.maximum_pairwise_abs_correlation
                )
                record["best_component_id"] = best_component_id
                record["validation_delta_auc_vs_best_component"] = (
                    validation_increment
                )
                record["validation_delta_auc_ci_low_vs_best_component"] = (
                    paired_increment.get("delta_auc_ci_low")
                )
                record["validation_delta_auc_ci_high_vs_best_component"] = (
                    paired_increment.get("delta_auc_ci_high")
                )
            stability.update(
                {
                    "analysis_type": "exploratory_mimic_combination",
                    "components": "|".join(components),
                    "component_families": "|".join(
                        _feature_family(component)
                        for component in components
                    ),
                    "component_modalities": "|".join(
                        component_modalities
                    ),
                    "feature_modality": score_modality,
                    "feature_role_policy_version": (
                        FEATURE_ROLE_POLICY_VERSION
                    ),
                    "validation_stable": stable,
                    "selected_for_freeze": stable,
                    "grouped_cv_mean_auc": candidate.mean_auc,
                    "grouped_cv_mean_average_precision": (
                        candidate.mean_average_precision
                    ),
                    "grouped_cv_evaluable_folds": candidate.evaluable_folds,
                    "grouped_cv_positive_direction_folds": (
                        candidate.positive_direction_folds
                    ),
                    "grouped_cv_auc_standard_deviation": (
                        candidate.fold_auc_standard_deviation
                    ),
                    "grouped_cv_stability_adjusted_auc": (
                        candidate.stability_adjusted_auc
                    ),
                    "grouped_cv_incremental_auc_vs_best_component": (
                        candidate.incremental_auc_vs_best_component
                    ),
                    "maximum_pairwise_abs_correlation": (
                        candidate.maximum_pairwise_abs_correlation
                    ),
                    "best_component_id": best_component_id,
                    "validation_delta_auc_vs_best_component": (
                        validation_increment
                    ),
                    "validation_delta_auc_ci_low_vs_best_component": (
                        paired_increment.get("delta_auc_ci_low")
                    ),
                    "validation_delta_auc_ci_high_vs_best_component": (
                        paired_increment.get("delta_auc_ci_high")
                    ),
                }
            )
            records.extend(score_records)
            stability_rows.append(stability)
            if stable:
                payloads.append(
                    {
                        "score_spec": score_spec.to_dict(),
                        "marker_model": score_model.to_dict(),
                        "selection_source": (
                            "five_fold_grouped_train_search_then_validation_gate"
                        ),
                        "grouped_cv": candidate.to_dict(),
                    }
                )
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "endpoint_id": endpoint_id,
                    "feature_id": "|".join(components),
                    "stage": "exploratory_score_fit",
                    "reason": str(exc),
                }
            )
    return records, payloads, stability_rows, failures


def _is_clinical_combination_candidate(feature_id: str) -> bool:
    return is_combination_candidate(feature_id)


def _stability_record(
    endpoint_id: str,
    panel_id: str,
    feature_id: str,
    model: MarkerModel,
    train_record: dict,
    validation_record: dict,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
) -> dict:
    valid = np.isfinite(validation_values)
    validation_direction = None
    if (
        np.sum(valid & (validation_labels == 1)) > 0
        and np.sum(valid & (validation_labels == 0)) > 0
    ):
        raw_auc = roc_auc_score(
            validation_labels[valid],
            validation_values[valid],
        )
        validation_direction = 1 if raw_auc >= 0.5 else -1
    train_auc = _optional_float(train_record.get("roc_auc"))
    validation_auc = _optional_float(validation_record.get("roc_auc"))
    return {
        "endpoint_id": endpoint_id,
        "panel_id": panel_id,
        "feature_id": feature_id,
        "analysis_type": "single_marker",
        "train_direction": model.direction,
        "validation_direction": validation_direction,
        "direction_agreement": (
            validation_direction == model.direction
            if validation_direction is not None
            else None
        ),
        "train_auc": train_auc,
        "validation_auc": validation_auc,
        "train_validation_auc_gap": (
            train_auc - validation_auc
            if train_auc is not None and validation_auc is not None
            else None
        ),
        "train_missingness": train_record.get("missingness"),
        "validation_missingness": validation_record.get("missingness"),
        "threshold_source": "train",
    }


def _transform_sensitivity_records(
    endpoint_id: str,
    panel_id: str,
    feature_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    primary_train_record: dict,
    primary_validation_record: dict,
) -> list[dict]:
    suffix = f"__{feature_id}"
    columns = sorted(
        column
        for column in train.columns
        if column.startswith("sensitivity_")
        and column.endswith(suffix)
        and column in validation
    )
    rows = []
    for column in columns:
        method = column.removeprefix("sensitivity_").removesuffix(suffix)
        train_values = _numeric_feature(train, column)
        validation_values = _numeric_feature(validation, column)
        try:
            model = fit_marker_model(
                train_values,
                train["label"].to_numpy(),
                column,
            )
        except (ValueError, TypeError):
            continue
        train_metrics = evaluate_marker(
            train_values,
            train["label"].to_numpy(),
            direction=model.direction,
            thresholds=model.thresholds,
        )
        validation_metrics = evaluate_marker(
            validation_values,
            validation["label"].to_numpy(),
            direction=model.direction,
            thresholds=model.thresholds,
        )
        primary_train_auc = _optional_float(primary_train_record.get("roc_auc"))
        primary_validation_auc = _optional_float(
            primary_validation_record.get("roc_auc")
        )
        transform_train_auc = _optional_float(train_metrics.get("roc_auc"))
        transform_validation_auc = _optional_float(
            validation_metrics.get("roc_auc")
        )
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "panel_id": panel_id,
                "feature_id": feature_id,
                "analysis_type": "vcg_transform_sensitivity",
                "transform_method": method,
                "train_direction": model.direction,
                "validation_direction": (
                    1
                    if (
                        transform_validation_auc is not None
                        and roc_auc_score(
                            validation.loc[
                                np.isfinite(validation_values),
                                "label",
                            ],
                            validation_values[np.isfinite(validation_values)],
                        )
                        >= 0.5
                    )
                    else -1
                    if transform_validation_auc is not None
                    else None
                ),
                "train_auc": transform_train_auc,
                "validation_auc": transform_validation_auc,
                "primary_kors_train_auc": primary_train_auc,
                "primary_kors_validation_auc": primary_validation_auc,
                "validation_auc_delta_vs_kors": (
                    transform_validation_auc - primary_validation_auc
                    if (
                        transform_validation_auc is not None
                        and primary_validation_auc is not None
                    )
                    else None
                ),
                "train_missingness": float(
                    1.0 - np.mean(np.isfinite(train_values))
                ),
                "validation_missingness": float(
                    1.0 - np.mean(np.isfinite(validation_values))
                ),
                "threshold_source": "transform_specific_train",
            }
        )
    return rows


# Audit 5.2.8: the subgroups a fairness statement is about get a subject-
# cluster bootstrap interval; the technical subgroups (device, QC, context)
# stay point estimates.
FAIRNESS_SUBGROUP_VARIABLES = frozenset(
    {"sex", "age", "race_performance_audit", "acquisition_era"}
)
# Audit 5.2.7: MIMIC-IV's real calendar era lives only in anchor_year_group.
# Early = 2008-2010 and 2011-2013, late = 2014-2016 and 2017-2019.
TEMPORAL_VALIDATION_SPLIT_YEAR = 2014


def _covariate_adjusted_records(
    endpoint_id: str,
    feature_id: str,
    direction: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> list[dict]:
    """Audit 5.3.6: covariate-adjusted AUC (Janes & Pepe 2009) of the primary
    marker and of the endpoint's classical comparator, adjusted for age, sex
    and, where available, heart rate. The comparator is aligned by its own
    training direction."""

    from .covariate_adjusted_roc import (
        covariate_adjusted_auc,
        covariate_matrix,
        heart_rate_column,
    )

    rate = heart_rate_column(train)
    sets = [("age_sex", ("age", "female"))]
    if rate is not None and rate in validation.columns:
        sets.append(("age_sex_heart_rate", ("age", "female", rate)))
    markers = [(feature_id, int(direction), "primary_marker")]
    classical = PRIMARY_CLASSICAL_BASELINES.get(endpoint_id)
    continuous = str(classical).removesuffix("_positive") if classical else None
    if continuous and continuous in train.columns and continuous in validation.columns:
        try:
            comparator = fit_marker_model(
                _numeric_feature(train, continuous),
                train["label"].to_numpy(dtype=np.int8),
                continuous,
            )
            markers.append((continuous, int(comparator.direction), "classical_comparator"))
        except ValueError:
            pass
    rows = []
    for marker, marker_direction, role in markers:
        for set_name, columns in sets:
            if not all(column == "female" or column in train.columns for column in columns):
                continue
            result = covariate_adjusted_auc(
                _numeric_feature(train, marker),
                train["label"].to_numpy(dtype=np.int8),
                covariate_matrix(train, columns),
                _numeric_feature(validation, marker),
                validation["label"].to_numpy(dtype=np.int8),
                covariate_matrix(validation, columns),
                direction=marker_direction,
            )
            rows.append(
                {
                    "endpoint_id": endpoint_id,
                    "panel_id": "primary_marker_subgroup",
                    "feature_id": marker,
                    "analysis_type": "covariate_adjusted_auc",
                    "evaluation_split": "validation",
                    "metric_scope": "validation_holdout",
                    "marker_role": role,
                    "covariate_set": set_name,
                    "covariates": "|".join(columns),
                    "direction": marker_direction,
                    "reference_distribution": "train_controls_linear_location_scale",
                    **result,
                    "method": "janes_pepe_2009_semiparametric_point_estimate",
                }
            )
    return rows


def _temporal_validation_records(
    endpoint_id: str,
    feature_id: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    *,
    bootstrap_iterations: int = 0,
) -> list[dict]:
    """Fit on the early-era train subjects, test in early and late validation.

    The only test of drift in devices, coding and care that MIMIC allows.
    Direction and thresholds are learned from train subjects whose anchor
    group starts before 2014; both validation eras are then scored with that
    model. The late-minus-early difference is the drift; the late AUC is the
    temporally external performance. Subjects never cross splits.
    """
    missing = pd.Series(pd.NA, dtype="object")
    train_era = anchor_year_group_start(train.get("anchor_year_group", missing.reindex(train.index)))
    validation_era = anchor_year_group_start(
        validation.get("anchor_year_group", missing.reindex(validation.index))
    )
    early_train = train.loc[
        (train_era < TEMPORAL_VALIDATION_SPLIT_YEAR).fillna(False).to_numpy()
    ]
    if early_train.empty:
        return []
    try:
        model = fit_marker_model(
            _numeric_feature(early_train, feature_id),
            early_train["label"].to_numpy(dtype=np.int8),
            feature_id,
        )
    except ValueError:
        return []
    rows = []
    for level, mask in (
        ("validation_early_eras", (validation_era < TEMPORAL_VALIDATION_SPLIT_YEAR)),
        ("validation_late_eras", (validation_era >= TEMPORAL_VALIDATION_SPLIT_YEAR)),
    ):
        mask = mask.fillna(False).to_numpy(dtype=bool)
        if int(np.sum(mask)) < 20:
            continue
        values = _numeric_feature(validation, feature_id)[mask]
        labels = validation["label"].to_numpy(dtype=np.int8)[mask]
        metrics = evaluate_marker(
            values, labels, direction=model.direction, thresholds=model.thresholds
        )
        n_positive, n_negative = int(metrics["n_positive"]), int(metrics["n_negative"])
        if n_positive == 0 or n_negative == 0:
            continue
        status = inference_status(
            n_positive,
            n_negative,
            minimum_inference=specification.min_class_inference,
            minimum_primary=specification.min_class_primary,
        )
        metrics = enforce_inference_policy(metrics, status)
        intervals = (
            marker_bootstrap_intervals(
                validation["subject_id"].to_numpy()[mask],
                values,
                labels,
                direction=model.direction,
                thresholds=model.thresholds,
                iterations=int(bootstrap_iterations),
                seed=int(specification.random_seed),
            )
            if status != "descriptive_only" and int(bootstrap_iterations) > 0
            else {}
        )
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "panel_id": "primary_marker_subgroup",
                "feature_id": feature_id,
                "analysis_type": "temporal_validation",
                "evaluation_split": "validation",
                "metric_scope": "validation_holdout",
                "subgroup_variable": "acquisition_era_trained_early",
                "subgroup_level": level,
                "subgroup_time_scope": "anchor_year_group",
                "direction": model.direction,
                "threshold_source": "train_early_eras_only",
                "n_train_early": int(model.n_train),
                "inference_status": status,
                **metrics,
                "roc_auc_ci_low": intervals.get("roc_auc_ci_low", np.nan),
                "roc_auc_ci_high": intervals.get("roc_auc_ci_high", np.nan),
            }
        )
    return rows


def _subgroup_stability_records(
    endpoint_id: str,
    feature_id: str,
    model: MarkerModel,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    *,
    bootstrap_iterations: int = 0,
) -> list[dict]:
    groups: list[tuple[str, str, np.ndarray, str]] = []

    def add_binary_groups(
        variable: str,
        values: pd.Series,
        left_label: str,
        right_label: str,
        split_value: float,
        *,
        time_scope: str = "available_at_ecg",
    ) -> None:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        groups.extend(
            (
                (
                    variable,
                    left_label,
                    np.isfinite(numeric) & (numeric < split_value),
                    time_scope,
                ),
                (
                    variable,
                    right_label,
                    np.isfinite(numeric) & (numeric >= split_value),
                    time_scope,
                ),
            )
        )

    add_binary_groups("age", validation["age"], "<65", ">=65", 65.0)
    add_binary_groups(
        "qrs_duration_ms",
        validation["machine_qrs_duration_ms"],
        "<120",
        ">=120",
        120.0,
    )
    add_binary_groups(
        "bmi",
        validation["bmi_before_ecg"],
        "<30",
        ">=30",
        30.0,
        time_scope="pre_ecg_omr",
    )
    add_binary_groups(
        "creatinine",
        validation["creatinine_nearest"],
        "<1.5",
        ">=1.5",
        1.5,
        time_scope="retrospective_near_ecg",
    )
    sex = validation["sex"].astype("string").str.upper()
    groups.extend(
        (
            ("sex", "female", sex.eq("F").fillna(False).to_numpy(), "baseline"),
            ("sex", "male", sex.eq("M").fillna(False).to_numpy(), "baseline"),
        )
    )
    quality = validation.get(
        "signal_quality_tier",
        pd.Series(pd.NA, index=validation.index, dtype="string"),
    ).astype("string")
    groups.extend(
        (
            (
                "signal_quality",
                "high",
                quality.eq("high").fillna(False).to_numpy(),
                "ecg_signal",
            ),
            (
                "signal_quality",
                "non_high_usable",
                quality.isin(("medium", "low")).fillna(False).to_numpy(),
                "ecg_signal",
            ),
            (
                "clinical_context",
                "ed",
                validation["ed_stay_id"].notna().to_numpy(),
                "index_encounter",
            ),
            (
                "clinical_context",
                "hospital_non_ed",
                (
                    validation["hadm_id"].notna()
                    & validation["ed_stay_id"].isna()
                ).to_numpy(),
                "index_encounter",
            ),
        )
    )
    # Real acquisition era. The year inside ecg_time is de-identification
    # shifted per patient and therefore cannot separate eras at all; splitting
    # on it produced a subgroup whose two arms were the same population.
    era_start = anchor_year_group_start(
        validation.get(
            "anchor_year_group",
            pd.Series(pd.NA, index=validation.index, dtype="object"),
        )
    )
    if era_start.notna().any():
        median_era = float(era_start.median())
        add_binary_groups(
            "acquisition_era",
            era_start,
            f"anchor_group_before_{int(median_era)}",
            f"anchor_group_from_{int(median_era)}",
            median_era,
            time_scope="anchor_year_group",
        )
    for cart_id, count in validation.get(
        "machine_cart_id",
        pd.Series(dtype="object"),
    ).value_counts(dropna=True).items():
        if int(count) >= 100:
            groups.append(
                (
                    "machine_cart_id",
                    str(cart_id),
                    validation["machine_cart_id"].eq(cart_id).to_numpy(),
                    "device_generalizability_audit",
                )
            )
    for temporal_status, count in validation.get(
        "machine_metadata_temporal_status",
        pd.Series(dtype="object"),
    ).value_counts(dropna=True).items():
        if int(count) >= 100:
            groups.append(
                (
                    "machine_metadata_temporal_status",
                    str(temporal_status),
                    validation["machine_metadata_temporal_status"]
                    .eq(temporal_status)
                    .to_numpy(),
                    "machine_metadata_qc",
                )
            )
    for race_value, count in validation["race"].value_counts(dropna=True).items():
        if int(count) >= 100:
            groups.append(
                (
                    "race_performance_audit",
                    str(race_value),
                    validation["race"].eq(race_value).to_numpy(),
                    "administrative_audit_only",
                )
            )

    values = _numeric_feature(validation, feature_id)
    labels = validation["label"].to_numpy(dtype=np.int8)
    rows = []
    for variable, level, group_mask, time_scope in groups:
        group_mask = np.asarray(group_mask, dtype=bool)
        if int(np.sum(group_mask)) < 20:
            continue
        group_values = values[group_mask]
        group_labels = labels[group_mask]
        metrics = evaluate_marker(
            group_values,
            group_labels,
            direction=model.direction,
            thresholds=model.thresholds,
        )
        n_positive = int(metrics["n_positive"])
        n_negative = int(metrics["n_negative"])
        if n_positive == 0 or n_negative == 0:
            continue
        status = inference_status(
            n_positive,
            n_negative,
            minimum_inference=specification.min_class_inference,
            minimum_primary=specification.min_class_primary,
        )
        metrics = enforce_inference_policy(metrics, status)
        intervals = (
            marker_bootstrap_intervals(
                validation["subject_id"].to_numpy()[group_mask],
                group_values,
                group_labels,
                direction=model.direction,
                thresholds=model.thresholds,
                iterations=int(bootstrap_iterations),
                seed=int(specification.random_seed),
            )
            if variable in FAIRNESS_SUBGROUP_VARIABLES
            and status != "descriptive_only"
            and int(bootstrap_iterations) > 0
            else {}
        )
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "panel_id": "primary_marker_subgroup",
                "feature_id": feature_id,
                "roc_auc_ci_low": intervals.get("roc_auc_ci_low", np.nan),
                "roc_auc_ci_high": intervals.get("roc_auc_ci_high", np.nan),
                "interval_scope": (
                    "fairness_subgroup_subject_cluster_bootstrap"
                    if intervals
                    else "point_estimate_only"
                ),
                "analysis_type": "subgroup_stability",
                "evaluation_split": "validation",
                "metric_scope": "validation_holdout",
                "subgroup_variable": variable,
                "subgroup_level": level,
                "subgroup_time_scope": time_scope,
                "direction": model.direction,
                "threshold_source": "train",
                "inference_status": status,
                **metrics,
            }
        )
    return rows


PAIRED_AVAILABILITY_COMPARISON_TYPE = "paired_availability_marker_discrimination"
_COMPARISON_VERDICT_LABELS_DE = {
    "new_marker_significantly_better": "neuer Marker signifikant überlegen",
    "classical_rule_significantly_better": (
        "klassische Regel signifikant überlegen"
    ),
    "comparator_marker_significantly_better": (
        "Vergleichsmarker signifikant überlegen"
    ),
    "equivalent": "kein signifikanter Unterschied",
    "insufficient_evidence": "nicht genügend Evidenz",
}


def _comparison_verdict(record: dict) -> tuple[str, str]:
    """Derive a clear verdict from a paired comparison record produced by
    either _paired_primary_baseline_comparison (fixed classical rule) or
    _paired_continuous_baseline_comparison (continuous classical marker).

    Uses the bootstrap CI of the primary paired delta: for a fixed rule,
    the sensitivity delta at the baseline's own train specificity (i.e.
    the operating point actually comparable to a binary guideline rule);
    for a continuous baseline, the AUC delta. Below formal-inference power
    (per DIAGNOSTIC_CRITERIA_ROADMAP.md: <100 per class), the verdict is
    "insufficient_evidence" rather than a possibly noisy win/loss claim.
    """
    if str(record.get("inference_status", "")) != "formal_inference":
        return "insufficient_evidence", _COMPARISON_VERDICT_LABELS_DE[
            "insufficient_evidence"
        ]
    comparison_type = str(record.get("comparison_type", ""))
    if comparison_type in (
        "paired_continuous_marker_discrimination",
        PAIRED_AVAILABILITY_COMPARISON_TYPE,
    ):
        low = _finite_or_none(record.get("delta_auc_ci_low"))
        high = _finite_or_none(record.get("delta_auc_ci_high"))
    else:
        low = _finite_or_none(record.get("sensitivity_delta_ci_low"))
        high = _finite_or_none(record.get("sensitivity_delta_ci_high"))
    if low is None or high is None:
        return "insufficient_evidence", _COMPARISON_VERDICT_LABELS_DE[
            "insufficient_evidence"
        ]
    if low > 0.0:
        key = "new_marker_significantly_better"
    elif high < 0.0:
        key = (
            "comparator_marker_significantly_better"
            if comparison_type == PAIRED_AVAILABILITY_COMPARISON_TYPE
            else "classical_rule_significantly_better"
        )
    else:
        key = "equivalent"
    return key, _COMPARISON_VERDICT_LABELS_DE[key]


def _paired_primary_baseline_comparison(
    endpoint_id: str,
    vcg_feature_id: str,
    baseline_feature_id: str,
    vcg_model: MarkerModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    bootstrap_iterations: int,
) -> dict:
    positive_column = f"{baseline_feature_id}_positive"
    fixed_rule_available = (
        positive_column in train
        and positive_column in validation
        and train[positive_column].notna().any()
        and validation[positive_column].notna().any()
    )
    if not fixed_rule_available:
        return _paired_continuous_baseline_comparison(
            endpoint_id,
            vcg_feature_id,
            baseline_feature_id,
            vcg_model,
            train,
            validation,
            specification,
            bootstrap_iterations,
        )

    def paired_arrays(frame: pd.DataFrame):
        values = _numeric_feature(frame, vcg_feature_id)
        baseline = frame[positive_column]
        valid = np.isfinite(values) & baseline.notna().to_numpy()
        return (
            frame.loc[valid, "subject_id"].to_numpy(),
            frame.loc[valid, "label"].to_numpy(dtype=np.int8),
            values[valid],
            baseline.loc[valid].astype(bool).to_numpy(),
        )

    (
        train_subjects,
        train_labels,
        train_values,
        train_baseline,
    ) = paired_arrays(train)
    (
        validation_subjects,
        validation_labels,
        validation_values,
        validation_baseline,
    ) = paired_arrays(validation)
    del train_subjects
    if (
        np.unique(train_labels).size < 2
        or np.unique(validation_labels).size < 2
    ):
        return {}

    baseline_train_metrics = classification_metrics(
        train_labels,
        train_baseline,
    )
    baseline_validation_metrics = classification_metrics(
        validation_labels,
        validation_baseline,
    )
    aligned_train = vcg_model.direction * train_values
    aligned_validation = vcg_model.direction * validation_values
    specificity_target = float(baseline_train_metrics["specificity"])
    sensitivity_target = float(baseline_train_metrics["sensitivity"])
    specificity_threshold = learn_threshold_for_target(
        aligned_train,
        train_labels,
        target_specificity=specificity_target,
    )
    sensitivity_threshold = learn_threshold_for_target(
        aligned_train,
        train_labels,
        target_sensitivity=sensitivity_target,
    )
    candidate_at_specificity = aligned_validation >= specificity_threshold
    candidate_at_sensitivity = aligned_validation >= sensitivity_threshold
    candidate_specificity_metrics = classification_metrics(
        validation_labels,
        candidate_at_specificity,
    )
    candidate_sensitivity_metrics = classification_metrics(
        validation_labels,
        candidate_at_sensitivity,
    )
    status = inference_status(
        int(np.sum(validation_labels == 1)),
        int(np.sum(validation_labels == 0)),
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    iterations = (
        int(bootstrap_iterations)
        if status != "descriptive_only"
        else 0
    )
    seed_base = specification.random_seed + int(
        content_hash(
            f"paired:{endpoint_id}:{vcg_feature_id}:{baseline_feature_id}"
        )[:8],
        16,
    )
    sensitivity_difference = paired_operating_point_bootstrap(
        validation_subjects,
        validation_labels,
        candidate_at_specificity,
        validation_baseline,
        metric="sensitivity",
        iterations=iterations,
        seed=seed_base,
    )
    specificity_difference = paired_operating_point_bootstrap(
        validation_subjects,
        validation_labels,
        candidate_at_sensitivity,
        validation_baseline,
        metric="specificity",
        iterations=iterations,
        seed=seed_base + 1,
    )
    criterion_id = baseline_feature_id.removeprefix("classical_")
    registry_match = next(
        (
            item
            for item in BaselineRegistry().for_endpoint(endpoint_id)
            if item.criterion_id == criterion_id
        ),
        None,
    )
    return {
        "endpoint_id": endpoint_id,
        "evaluation_split": "validation",
        "comparison_type": "paired_fixed_rule_operating_point",
        "vcg_feature_id": vcg_feature_id,
        "vcg_direction": vcg_model.direction,
        "baseline_feature_id": baseline_feature_id,
        "baseline_registry_id": (
            registry_match.baseline_id if registry_match is not None else None
        ),
        "baseline_source_version": (
            registry_match.source_version if registry_match is not None else None
        ),
        "threshold_source": "train",
        "inference_status": status,
        "paired_n": int(len(validation_labels)),
        "paired_n_positive": int(np.sum(validation_labels == 1)),
        "paired_n_negative": int(np.sum(validation_labels == 0)),
        "baseline_train_sensitivity": sensitivity_target,
        "baseline_train_specificity": specificity_target,
        "baseline_validation_sensitivity": baseline_validation_metrics[
            "sensitivity"
        ],
        "baseline_validation_specificity": baseline_validation_metrics[
            "specificity"
        ],
        "vcg_specificity_matched_threshold_aligned": specificity_threshold,
        "vcg_specificity_at_specificity_match": (
            candidate_specificity_metrics["specificity"]
        ),
        "vcg_sensitivity_at_specificity_match": (
            candidate_specificity_metrics["sensitivity"]
        ),
        "sensitivity_delta_vs_fixed_rule": sensitivity_difference[
            "delta_sensitivity"
        ],
        "sensitivity_delta_ci_low": sensitivity_difference[
            "delta_sensitivity_ci_low"
        ],
        "sensitivity_delta_ci_high": sensitivity_difference[
            "delta_sensitivity_ci_high"
        ],
        "vcg_sensitivity_matched_threshold_aligned": sensitivity_threshold,
        "vcg_sensitivity_at_sensitivity_match": (
            candidate_sensitivity_metrics["sensitivity"]
        ),
        "vcg_specificity_at_sensitivity_match": (
            candidate_sensitivity_metrics["specificity"]
        ),
        "specificity_delta_vs_fixed_rule": specificity_difference[
            "delta_specificity"
        ],
        "specificity_delta_ci_low": specificity_difference[
            "delta_specificity_ci_low"
        ],
        "specificity_delta_ci_high": specificity_difference[
            "delta_specificity_ci_high"
        ],
        "bootstrap_iterations": iterations,
        "auc_delta_applicable": False,
        "auc_delta_reason": (
            "The guideline comparator is a fixed binary ROC point; compare "
            "paired sensitivity/specificity at train-defined operating points."
        ),
        "exploratory_score": False,
    }


def _paired_continuous_baseline_comparison(
    endpoint_id: str,
    vcg_feature_id: str,
    baseline_feature_id: str,
    vcg_model: MarkerModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specification: ValidationSpec,
    bootstrap_iterations: int,
) -> dict:
    if (
        baseline_feature_id not in train
        or baseline_feature_id not in validation
    ):
        return {}

    train_vcg = _numeric_feature(train, vcg_feature_id)
    train_baseline = _numeric_feature(train, baseline_feature_id)
    train_valid = np.isfinite(train_vcg) & np.isfinite(train_baseline)
    train_labels = train.loc[train_valid, "label"].to_numpy(dtype=np.int8)
    if np.unique(train_labels).size < 2:
        return {}
    baseline_model = fit_marker_model(
        train_baseline[train_valid],
        train_labels,
        baseline_feature_id,
    )

    validation_vcg = _numeric_feature(validation, vcg_feature_id)
    validation_baseline = _numeric_feature(
        validation,
        baseline_feature_id,
    )
    validation_valid = (
        np.isfinite(validation_vcg)
        & np.isfinite(validation_baseline)
    )
    labels = validation.loc[
        validation_valid,
        "label",
    ].to_numpy(dtype=np.int8)
    if np.unique(labels).size < 2:
        return {}
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    status = inference_status(
        n_positive,
        n_negative,
        minimum_inference=specification.min_class_inference,
        minimum_primary=specification.min_class_primary,
    )
    iterations = (
        int(bootstrap_iterations)
        if status != "descriptive_only"
        else 0
    )
    seed = specification.random_seed + int(
        content_hash(
            f"paired-continuous:{endpoint_id}:{vcg_feature_id}:"
            f"{baseline_feature_id}"
        )[:8],
        16,
    )
    paired = paired_discrimination_difference_bootstrap(
        validation.loc[validation_valid, "subject_id"].to_numpy(),
        labels,
        vcg_model.direction * validation_vcg[validation_valid],
        baseline_model.direction
        * validation_baseline[validation_valid],
        iterations=iterations,
        seed=seed,
    )
    criterion_id = baseline_feature_id.removeprefix("classical_")
    registry_match = next(
        (
            item
            for item in BaselineRegistry().for_endpoint(endpoint_id)
            if item.criterion_id == criterion_id
        ),
        None,
    )
    return {
        "endpoint_id": endpoint_id,
        "evaluation_split": "validation",
        "comparison_type": "paired_continuous_marker_discrimination",
        "vcg_feature_id": vcg_feature_id,
        "vcg_direction": vcg_model.direction,
        "baseline_feature_id": baseline_feature_id,
        "baseline_direction": baseline_model.direction,
        "baseline_registry_id": (
            registry_match.baseline_id if registry_match is not None else None
        ),
        "baseline_source_version": (
            registry_match.source_version if registry_match is not None else None
        ),
        "direction_source": "train",
        "threshold_source": "not_applicable_continuous_comparison",
        "inference_status": status,
        "reference_independence": (
            "ecg_derived_non_independent"
            if endpoint_id.startswith("qtc_")
            else "clinical_reference"
        ),
        **paired,
        "exploratory_score": False,
    }


def _build_comparison_table(
    screening: pd.DataFrame,
    classical: pd.DataFrame,
    reports: pd.DataFrame,
) -> pd.DataFrame:
    if screening.empty:
        return pd.DataFrame()
    rows = []
    for endpoint_id, endpoint_frame in screening.groupby("endpoint_id"):
        train_group = endpoint_frame.loc[
            endpoint_frame["evaluation_split"] == "train"
        ]
        validation_group = endpoint_frame.loc[
            endpoint_frame["evaluation_split"] == "validation"
        ]
        train_vcg = train_group.loc[
            train_group["panel_id"] != "classical"
        ].dropna(
            subset=["roc_auc"]
        )
        selected_feature = (
            train_vcg.sort_values("roc_auc", ascending=False).iloc[0][
                "feature_id"
            ]
            if not train_vcg.empty
            else None
        )
        selected_validation = validation_group.loc[
            validation_group["feature_id"] == selected_feature
        ]
        best_vcg = (
            selected_validation.iloc[0]
            if not selected_validation.empty
            else None
        )
        fixed_classical = classical.loc[
            (classical.get("endpoint_id") == endpoint_id)
            & (classical.get("evaluation_split") == "validation")
        ] if not classical.empty else pd.DataFrame()
        best_fixed = (
            fixed_classical.sort_values(
                ["balanced_accuracy", "specificity"],
                ascending=False,
            ).iloc[0]
            if not fixed_classical.empty
            else None
        )
        report = reports.loc[
            (reports.get("endpoint_id") == endpoint_id)
            & (reports.get("evaluation_split") == "validation")
        ] if not reports.empty else pd.DataFrame()
        report_row = report.iloc[0] if not report.empty else None
        vcg_auc = _series_float(best_vcg, "roc_auc")
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "evaluation_split": "validation",
                "train_selected_vcg_feature": selected_feature,
                "validation_vcg_auc": vcg_auc,
                "descriptive_classical_rule": (
                    best_fixed["baseline_id"] if best_fixed is not None else None
                ),
                "classical_sensitivity": _series_float(
                    best_fixed,
                    "sensitivity",
                ),
                "classical_specificity": _series_float(
                    best_fixed,
                    "specificity",
                ),
                "machine_report_baseline": (
                    report_row["baseline_id"] if report_row is not None else None
                ),
                "machine_report_sensitivity": _series_float(
                    report_row,
                    "sensitivity",
                ),
                "machine_report_specificity": _series_float(
                    report_row,
                    "specificity",
                ),
                "selection_source": "train_auc_only",
                "comparison_status": (
                    "descriptive_ranking; fixed-rule deltas are reported only "
                    "in the paired operating-point table"
                ),
            }
        )
    return pd.DataFrame(rows)


def _endpoint_split_counts(endpoint_id: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for split, group in frame.groupby("split"):
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "split": split,
                "n_total": len(group),
                "n_positive": int(group["label_state"].eq("positive").sum()),
                "n_negative": int(group["label_state"].eq("negative").sum()),
                "prevalence": float(group["label_state"].eq("positive").mean()),
                "subject_count": int(group["subject_id"].nunique()),
            }
        )
    return rows


def _feature_panel_map(panels: dict, selected_panel_ids: Iterable[str]) -> dict:
    output = {}
    for panel_id in selected_panel_ids:
        for feature_id in panels[panel_id].feature_ids:
            output[feature_id] = (
                "classical"
                if feature_id.startswith("classical_")
                else panel_id
            )
    return output


def _numeric_feature(frame: pd.DataFrame, feature_id: str) -> np.ndarray:
    if feature_id not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[feature_id], errors="coerce").to_numpy(dtype=float)


def _read_split_hash(master_path: Path) -> str:
    from .storage import DuckDBCatalog

    with DuckDBCatalog(master_path.parent / ".split-audit.duckdb", memory_limit_gb=2,
                       threads=1).connect(isolated=True) as connection:
        values = connection.execute(
            "SELECT DISTINCT split_spec_hash FROM read_parquet(?)",
            [str(master_path)],
        ).fetchall()
    hashes = sorted(str(row[0]) for row in values if row[0])
    if len(hashes) != 1:
        raise RuntimeError(f"Expected one split specification hash, got {hashes}")
    return hashes[0]


def _report_comparator_for_endpoint(endpoint_id: str) -> Optional[str]:
    if endpoint_id.startswith("new_onset_af"):
        return "af"
    if endpoint_id in {
        "acute_mi_high_specificity",
        "acute_mi_probable",
        "omi_revascularization_proxy",
        "troponin_positive_any",
        "acute_troponin_change_proxy",
    }:
        return "ischemia"
    if endpoint_id in {
        "old_mi_icd",
        "mi_or_revascularization_history",
        "regional_wall_motion_abnormality",
    }:
        return "infarct"
    if endpoint_id in {
        "echo_lvh_strict",
        "echo_lvh_sensitive",
        "lv_wall_thickening",
        "lv_dilatation_proxy",
        "concentric_remodeling_proxy",
    }:
        return "lvh"
    if endpoint_id in {
        "echo_rv_dilation",
        "echo_rv_systolic_dysfunction",
        "elevated_rvsp_or_pasp",
        "echo_pulmonary_hypertension",
        "right_heart_strain_proxy",
        "pulmonary_embolism_icd",
        "pulmonary_hypertension_icd",
    }:
        return "rvh"
    if endpoint_id == "echo_lae":
        return "lae"
    if endpoint_id == "echo_rae":
        return "rae"
    if endpoint_id.startswith("qtc_"):
        return "qt_prolongation"
    return None


PREDEFINED_SCORE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "troponin_positive_any": (
        "qrs_t_hausdorff_distance",
        "frontal_xy_qrs_t_angle_deg",
        "t_downstroke_velocity_energy",
        "st_loop_convex_hull_volume_3d",
    ),
    "acute_mi_high_specificity": (
        "qrs_t_hausdorff_distance",
        "frontal_xy_qrs_t_angle_deg",
        "t_downstroke_velocity_energy",
        "st_loop_convex_hull_volume_3d",
    ),
    "omi_revascularization_proxy": (
        "qrs_t_hausdorff_distance",
        "spatial_qrs_t_angle_deg",
        "svg_magnitude",
    ),
    "echo_lvh_strict": (
        "qrs_t_hausdorff_distance",
        "t_downstroke_velocity_energy",
        "classical_sokolow_lyon",
        "svg_qrs_area_ratio",
    ),
    "echo_lvh_sensitive": (
        "qrs_t_hausdorff_distance",
        "t_downstroke_velocity_energy",
        "classical_peguero_lo_presti",
        "svg_qrs_area_ratio",
    ),
    "old_mi_icd": (
        "qrst_loop_frontal_xy_winding_number",
        "qrs_loop_area_best_fit_plane",
        "qrs_t_vector_correlation_distance",
        "st_loop_area_best_fit_plane",
    ),
    "regional_wall_motion_abnormality": (
        "qrst_loop_frontal_xy_winding_number",
        "qrs_loop_area_best_fit_plane",
        "qrs_t_vector_correlation_distance",
        "st_loop_area_best_fit_plane",
    ),
    "echo_rv_dilation": (
        "qrs_terminal_20ms_vector",
        "horizontal_xz_qrs_t_angle_deg",
        "t_loop_planarity",
    ),
    "echo_rv_systolic_dysfunction": (
        "qrs_terminal_20ms_vector",
        "t_loop_tortuosity",
        "spatial_qrs_t_angle_deg",
    ),
    "echo_lae": (
        "p_loop_width_major",
        "p_loop_terminal_force_vector",
        "p_loop_template_instability",
    ),
    "new_onset_af_90d": (
        "p_loop_width_major",
        "p_loop_terminal_force_vector",
        "p_loop_interbeat_axis_variability",
    ),
    "qtc_gt_500": (
        "qtc_fridericia_ms",
        "tpeak_tend_ms",
        "t_loop_tortuosity",
    ),
    "hyperkalemia_near_ecg": (
        "t_loop_width_major",
        "qt_interval_ms",
        "spatial_magnitude_max",
    ),
    "hypokalemia_near_ecg": (
        "qtc_fridericia_ms",
        "tpeak_tend_ms",
        "t_loop_planarity",
    ),
    "reduced_lvef": (
        "qrs_t_hausdorff_distance",
        "svg_magnitude",
        "spatial_qrs_t_angle_deg",
        "t_loop_planarity",
    ),
}


# Two candidate markers read on the same rows. Pooled AUCs of markers with
# different availability are not comparable: in the A117 10k pilot the
# Direct-8-lead QRS-T angle screened at 0,811 for LVEF < 40 % (56 % of
# recordings, 80 % of them in regular rhythm) and the Fréchet coupling at
# 0,797 (88 % of recordings); on the 1.295 rows both exist the order was
# 0,805 against 0,828. The pair is fixed here before any run reads it, the
# Fréchet distance is the candidate and the Direct-8 angle the comparator,
# and the bootstrap of the paired AUC difference is the only inference.
PAIRED_AVAILABILITY_COMPARISONS: dict[str, tuple[tuple[str, str], ...]] = {
    endpoint_id: (
        (
            "qrs_t_normalized_discrete_frechet",
            "qrs_t_integral_angle_direct_8lead_beat_median_deg",
        ),
    )
    for endpoint_id in (
        "reduced_lvef_strict",
        "reduced_lvef",
        "preserved_lvef",
        "heart_failure_icd",
        "cardiomyopathy_icd",
        "lv_dilatation_proxy",
    )
}
PRIMARY_ADJUSTED_MARKERS: dict[str, str] = {
    "troponin_positive_any": "qrs_t_hausdorff_distance",
    "acute_troponin_change_proxy": "qrs_t_hausdorff_distance",
    "acute_mi_high_specificity": "qrs_t_hausdorff_distance",
    "acute_mi_probable": "qrs_t_hausdorff_distance",
    "omi_revascularization_proxy": "qrs_t_hausdorff_distance",
    "myocardial_injury_demand_context": "spatial_qrs_t_angle_deg",
    "echo_lvh_strict": "svg_qrs_area_ratio",
    "echo_lvh_sensitive": "svg_qrs_area_ratio",
    "lv_wall_thickening": "svg_qrs_area_ratio",
    "lv_dilatation_proxy": "qrs_t_hausdorff_distance",
    "concentric_remodeling_proxy": "svg_qrs_area_ratio",
    "reduced_lvef_strict": "qrs_t_hausdorff_distance",
    "reduced_lvef": "qrs_t_hausdorff_distance",
    "preserved_lvef": "qrs_t_hausdorff_distance",
    "echo_lae": "p_loop_width_major",
    "echo_rae": "p_loop_width_major",
    "echo_rv_dilation": "qrs_terminal_20ms_vector",
    "echo_rv_systolic_dysfunction": "qrs_terminal_20ms_vector",
    "elevated_rvsp_or_pasp": "horizontal_xz_qrs_t_angle_deg",
    "echo_pulmonary_hypertension": "qrs_terminal_20ms_vector",
    "regional_wall_motion_abnormality": (
        "qrst_loop_frontal_xy_winding_number"
    ),
    "aortic_stenosis": "svg_magnitude",
    "aortic_regurgitation": "svg_magnitude",
    "mitral_stenosis": "p_loop_width_major",
    "mitral_regurgitation": "p_loop_width_major",
    "tricuspid_regurgitation": "qrs_terminal_20ms_vector",
    "significant_valvular_disease": "spatial_qrs_t_angle_deg",
    "aortic_stenosis_severe": "svg_magnitude",
    "aortic_regurgitation_severe": "svg_magnitude",
    "mitral_stenosis_severe": "p_loop_width_major",
    "mitral_regurgitation_severe": "p_loop_width_major",
    "tricuspid_regurgitation_severe": "qrs_terminal_20ms_vector",
    "significant_valvular_disease_severe": "spatial_qrs_t_angle_deg",
    # Quantitative echo endpoints, each paired with the marker whose
    # physiology matches the reference standard rather than whatever screened
    # best. The atrial ones finally have an atrial-size target: a P-loop
    # marker screened against LVEF was never testing what it measures.
    "diastolic_relaxation_abnormality": "p_loop_width_major",
    "restrictive_filling_pattern": "p_loop_width_major",
    "elevated_filling_pressure_proxy": "p_loop_width_major",
    "la_dilatation_linear": "p_loop_width_major",
    "ra_dilatation_linear": "qrs_terminal_20ms_vector",
    "severe_aortic_stenosis_velocity": "svg_magnitude",
    "moderate_aortic_stenosis_velocity": "svg_magnitude",
    "concentric_remodeling_rwt": "svg_qrs_area_ratio",
    "aortic_root_dilatation": "svg_magnitude",
    "qtc_gt_480": "tpeak_tend_ms",
    "qtc_gt_500": "tpeak_tend_ms",
    "hypokalemia_near_ecg": "tpeak_tend_ms",
    "severe_hypokalemia_near_ecg": "tpeak_tend_ms",
    "hyperkalemia_near_ecg": "t_loop_width_major",
    "severe_hyperkalemia_near_ecg": "t_loop_width_major",
    "hypomagnesemia_near_ecg": "tpeak_tend_ms",
    "hypocalcemia_near_ecg": "tpeak_tend_ms",
    "new_onset_af_30d": "p_loop_template_instability",
    "new_onset_af_90d": "p_loop_template_instability",
    "new_onset_af_1y": "p_loop_template_instability",
    "old_mi_icd": "qrst_loop_frontal_xy_winding_number",
    "mi_or_revascularization_history": (
        "qrst_loop_frontal_xy_winding_number"
    ),
    "heart_failure_icd": "spatial_qrs_t_angle_deg",
    "elevated_ntprobnp": "spatial_qrs_t_angle_deg",
    "acute_decompensated_heart_failure_proxy": "spatial_qrs_t_angle_deg",
    "pulmonary_embolism_icd": "qrs_terminal_20ms_vector",
    "pulmonary_hypertension_icd": "qrs_terminal_20ms_vector",
    "right_heart_strain_proxy": "qrs_terminal_20ms_vector",
    "cardiomyopathy_icd": "qrs_t_hausdorff_distance",
    "pericarditis_icd": "st_loop_area_best_fit_plane",
    "myocarditis_icd": "st_loop_area_best_fit_plane",
    "endocarditis_high_confidence_proxy": "spatial_qrs_t_angle_deg",
    "vt_vf_icd_or_event": "spatial_qrs_t_angle_deg",
    "cardiac_arrest": "spatial_qrs_t_angle_deg",
    "in_hospital_mortality": "spatial_qrs_t_angle_deg",
    "mortality_30d": "spatial_qrs_t_angle_deg",
    "qt_prolonging_drug_exposure": "tpeak_tend_ms",
    "antiarrhythmic_exposure_near_ecg": "spatial_qrs_t_angle_deg",
    "digoxin_exposure_near_ecg": "st_loop_area_best_fit_plane",
}


PRIMARY_CLASSICAL_BASELINES: dict[str, str] = {
    **{
        endpoint: "classical_st_elevation_contiguous"
        for endpoint in (
            "troponin_positive_any",
            "acute_troponin_change_proxy",
            "acute_mi_high_specificity",
            "acute_mi_probable",
            "omi_revascularization_proxy",
        )
    },
    **{
        endpoint: "classical_cornell_voltage"
        for endpoint in (
            "echo_lvh_strict",
            "echo_lvh_sensitive",
            "lv_wall_thickening",
            "lv_dilatation_proxy",
            "concentric_remodeling_proxy",
        )
    },
    **{
        endpoint: "classical_pathological_q_waves"
        for endpoint in (
            "old_mi_icd",
            "mi_or_revascularization_history",
            "regional_wall_motion_abnormality",
        )
    },
    **{
        endpoint: "classical_dominant_r_v1"
        for endpoint in (
            "echo_rv_dilation",
            "echo_rv_systolic_dysfunction",
            "elevated_rvsp_or_pasp",
            "echo_pulmonary_hypertension",
            "pulmonary_embolism_icd",
            "pulmonary_hypertension_icd",
            "right_heart_strain_proxy",
        )
    },
    **{
        endpoint: "classical_p_terminal_force_v1"
        for endpoint in (
            "echo_lae",
            "new_onset_af_30d",
            "new_onset_af_90d",
            "new_onset_af_1y",
        )
    },
    **{
        endpoint: (
            "classical_hyperkalemia_ecg_heuristic"
            if "hyperkalemia" in endpoint
            else "classical_qtc_fridericia"
        )
        for endpoint in (
            "qtc_gt_480",
            "qtc_gt_500",
            "hypokalemia_near_ecg",
            "severe_hypokalemia_near_ecg",
            "hyperkalemia_near_ecg",
            "severe_hyperkalemia_near_ecg",
            "hypomagnesemia_near_ecg",
            "hypocalcemia_near_ecg",
            "qt_prolonging_drug_exposure",
        )
    },
}


def _series_float(series: Optional[pd.Series], key: str) -> Optional[float]:
    if series is None:
        return None
    return _optional_float(series.get(key))


def _optional_float(value) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _validate_run_id(run_id: str) -> None:
    if (
        not run_id
        or not run_id.replace("-", "").replace("_", "").isalnum()
        or run_id in {".", ".."}
    ):
        raise ValueError(f"Unsafe run ID: {run_id!r}")


def _quoted_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")
