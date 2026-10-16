from __future__ import annotations

import glob
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ecg_derived_targets import annotate_targets
from .signal_quality_controls import (
    QUALITY_CONTROL_PANEL_ID,
    annotate_hits_with_quality_verdict,
    compare_quality_against_candidates,
    quality_control_features,
)
from .association_domains import (
    context_association_registry_hash,
    run_context_association_discovery,
)
from .config import MimicProjectConfig
from .discovery_statistics import (
    DISCOVERY_MAX_EFFECT_GAP,
    DISCOVERY_MIN_ALIGNED_AUC,
    DISCOVERY_MIN_EFFECT_RETENTION,
    DISCOVERY_STABILITY_GATE_VERSION,
    add_auc_replication_status,
    discovery_marker_bootstrap_intervals,
)
from .event_equivalence import (
    verify_clinical_event_evidence_provenance,
)
from .feature_extraction import verify_feature_artifacts
from .future_phenome import (
    DEFAULT_FUTURE_HORIZONS_DAYS,
    FUTURE_PHENOME_VERSION,
    run_future_phenome_discovery,
)
from .feature_roles import (
    FEATURE_ROLE_POLICY_VERSION,
    is_candidate_biomarker,
)
from .frame_checkpoints import (
    checkpointed_frame_chunk_path,
    materialize_frame_chunks,
)
from .known_ecg_associations import classify_novelty
from .models import PromotionCandidate
from .panels import PANEL_VERSION, default_feature_panels
from .phenotypes import (
    PHENOTYPE_MAPPING_VERSION,
    prepare_phenotype_resources,
)
from .phenome_adjustment import (
    PHENOME_ADJUSTMENT_COVARIATES,
    PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION,
    PHENOME_ADJUSTMENT_VERSION,
    PhenomeAdjustmentPolicy,
    build_phenome_discovery_status,
    confirm_replicated_phenome_hits,
    finalize_phenome_adjustment_output,
)
from .snapshots import (
    create_artifact_snapshot,
    verify_artifact_snapshot,
)
from .result_explorer import (
    build_new_association_explorer,
    rank_broad_discovery,
)
from .run_log import profile_query
from .run_integrity import assert_development_run_mutable
from .statistics import (
    SortedControlDistribution,
    apply_fdr_by_family,
    evaluate_marker,
    excluded_mann_whitney_auc,
    fast_mann_whitney_auc,
    fit_marker_model,
    inference_status,
    prepare_sorted_control_distribution,
    screen_marker_assessability_from_counts,
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
    operational_config_hash,
    optional_column_select,
    parquet_columns,
    relation_columns,
    scientific_config_hash,
)
from .validation import (
    _event_evidence_provenance_summary,
    _prepare_validation_feature_cache,
    _resolved_feature_scope,
)


PHENOME_STAGE_VERSION = "4.8.0"
DEFAULT_MAX_FUTURE_PHENOTYPES = 100
PHENOME_TRAIN_CHUNK_SIZE = 25
PHENOME_REPLICATION_CHUNK_SIZE = 32
PHENOME_FEATURE_BLOCK_SIZE = 32


def _scientific_mimic_hash() -> str:
    from .code_scope import scientific_source_hash

    return scientific_source_hash(Path(__file__).resolve().parent)


@dataclass(frozen=True)
class _PreparedScreenBlock:
    """Read-only sufficient data shared by every phenotype chunk in a block."""

    study_to_row: Mapping[tuple[int, int], int]
    all_subjects: np.ndarray
    control_feature_rows: np.ndarray
    control_subjects: np.ndarray
    numeric: Mapping[str, np.ndarray]
    global_control_distributions: Mapping[str, SortedControlDistribution]
    all_sex: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    control_sex: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))


def _normalized_sex(features: pd.DataFrame) -> np.ndarray:
    raw = features.get("sex", pd.Series("", index=features.index))
    upper = raw.fillna("").astype(str).str.strip().str.upper()
    return np.where(
        upper.str.startswith("M"), "M", np.where(upper.str.startswith("F"), "F", "")
    )


def _phenotype_sex_restriction(definition) -> Optional[str]:
    """'M' or 'F' for phecodeX sex-specific phenotypes (audit X3), else None."""
    def flag(name: str) -> bool:
        value = definition.get(name, False) if hasattr(definition, "get") else False
        return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}

    if flag("male_only"):
        return "M"
    if flag("female_only"):
        return "F"
    return None


def _phecode_ancestors(phenotype_id: str) -> tuple[str, ...]:
    """Ancestors in phecodeX decimal notation: CV_401.12 -> CV_401.1, CV_401."""
    stem, dot, digits = str(phenotype_id).partition(".")
    if not dot:
        return ()
    return (stem, *(f"{stem}.{digits[:length]}" for length in range(1, len(digits))))


def _related_phenotype_ids(
    phenotype_id: str,
    all_phenotype_ids: Iterable[str],
) -> frozenset[str]:
    """Ancestors and descendants of one phenotype, not the phenotype itself.

    Audit X4: phecodeX has no exclusion ranges any more, and a patient coded
    with a parent or child of the phenotype (hypertension without hypertensive
    heart disease, for example) is not a clean control. Such patients are
    removed from the controls; they are not cases either.
    """
    ancestors = set(_phecode_ancestors(phenotype_id))
    related = {
        other
        for other in all_phenotype_ids
        if other != phenotype_id
        and (other in ancestors or phenotype_id in _phecode_ancestors(other))
    }
    return frozenset(related)


def _current_rows_for_chunk(
    current: pd.DataFrame,
    phenotype_chunk: Iterable[str],
    known_phenotypes: tuple[str, ...],
) -> pd.DataFrame:
    """Diagnosis rows a chunk's screen reads: its phenotypes and relatives."""
    wanted = set(str(item) for item in phenotype_chunk)
    for phenotype_id in list(wanted):
        wanted |= _related_phenotype_ids(phenotype_id, known_phenotypes)
    return current.loc[current["phenotype_id"].astype(str).isin(wanted)].copy()


def _prepared_block_for_rows(
    prepared: "_PreparedScreenBlock",
    rows: pd.DataFrame,
) -> "_PreparedScreenBlock":
    """The shared block with only the study index entries a chunk looks up."""
    from dataclasses import replace

    keys = {
        (int(row.subject_id), int(row.study_id))
        for row in rows[["subject_id", "study_id"]].itertuples(index=False)
    }
    return replace(
        prepared,
        study_to_row={
            key: prepared.study_to_row[key]
            for key in keys
            if key in prepared.study_to_row
        },
    )


def _screen_chunk_task(
    checkpoint_root: Path,
    chunk_index: int,
    input_hash: str,
    phenotype_chunk: tuple[str, ...],
    selected_block: tuple[str, ...],
    current_subset: pd.DataFrame,
    panel_map: Mapping[str, str],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    prepared_block: "_PreparedScreenBlock",
) -> tuple[Path, bool]:
    """One phenotype chunk of the train screen, in a worker process (E11b)."""
    return checkpointed_frame_chunk_path(
        checkpoint_root,
        namespace="current_train_stream_v1",
        chunk_index=chunk_index,
        input_hash=input_hash,
        item_keys=tuple(
            (phenotype_id, feature_id)
            for phenotype_id in phenotype_chunk
            for feature_id in selected_block
        ),
        compute=lambda: pd.DataFrame(
            _screen_split(
                pd.DataFrame(),
                current_subset,
                pd.DataFrame(),
                phenotype_ids=phenotype_chunk,
                feature_ids=selected_block,
                panel_map=panel_map,
                definition_lookup=definition_lookup,
                min_inference=min_inference,
                min_formal=min_formal,
                progress=None,
                prepared_block=prepared_block,
            )
        ),
    )


def _prepare_screen_block(
    features: pd.DataFrame,
    feature_ids: tuple[str, ...],
) -> _PreparedScreenBlock:
    study_to_row = {
        (int(row.subject_id), int(row.study_id)): index
        for index, row in enumerate(
            features[["subject_id", "study_id"]].itertuples(index=False)
        )
    }
    all_subjects = features["subject_id"].astype(int).to_numpy()
    control_feature_rows = np.flatnonzero(
        features["is_control_index"].fillna(False).to_numpy(dtype=bool)
    )
    control_subjects = all_subjects[control_feature_rows]
    all_sex = _normalized_sex(features)
    control_sex = all_sex[control_feature_rows]
    numeric = {
        feature_id: pd.to_numeric(features[feature_id], errors="coerce").to_numpy(
            dtype=float
        )
        for feature_id in feature_ids
    }
    global_control_distributions = {
        feature_id: prepare_sorted_control_distribution(
            numeric[feature_id][control_feature_rows]
        )
        for feature_id in feature_ids
    }
    for array in (
        all_subjects,
        control_feature_rows,
        control_subjects,
        all_sex,
        control_sex,
        *numeric.values(),
        *(item.sorted_values for item in global_control_distributions.values()),
    ):
        array.setflags(write=False)
    return _PreparedScreenBlock(
        study_to_row=study_to_row,
        all_subjects=all_subjects,
        control_feature_rows=control_feature_rows,
        control_subjects=control_subjects,
        numeric=numeric,
        global_control_distributions=global_control_distributions,
        all_sex=all_sex,
        control_sex=control_sex,
    )


@dataclass(frozen=True)
class PhenomeDiscoveryOptions:
    run_id: str = "phenome-discovery"
    feature_glob: Optional[str] = None
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel")
    train_fdr_alpha: float = 0.01
    validation_fdr_alpha: float = 0.05
    min_class_inference: int = 30
    min_class_formal: int = 100
    bootstrap_iterations: int = 1000
    max_phenotypes: Optional[int] = None
    max_future_phenotypes: Optional[int] = DEFAULT_MAX_FUTURE_PHENOTYPES
    future_horizons_days: tuple[int, ...] = DEFAULT_FUTURE_HORIZONS_DAYS
    force: bool = False
    include_promoted_candidates: bool = False

    def __post_init__(self) -> None:
        horizons = tuple(
            sorted({int(value) for value in self.future_horizons_days})
        )
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError(
                "future_horizons_days must contain positive day values"
            )
        object.__setattr__(self, "future_horizons_days", horizons)
        if (
            self.max_future_phenotypes is not None
            and int(self.max_future_phenotypes) <= 0
        ):
            object.__setattr__(self, "max_future_phenotypes", None)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhenomeDiscoveryOptions":
        panel_value = values.get("panel_ids", ("primary_core", "sentinel"))
        if isinstance(panel_value, str):
            panel_ids = tuple(
                item.strip() for item in panel_value.split(",") if item.strip()
            )
        else:
            panel_ids = tuple(str(item) for item in panel_value)
        max_phenotypes = values.get("max_phenotypes")
        max_future_phenotypes = values.get(
            "max_future_phenotypes",
            DEFAULT_MAX_FUTURE_PHENOTYPES,
        )
        horizon_value = values.get(
            "future_horizons_days",
            DEFAULT_FUTURE_HORIZONS_DAYS,
        )
        if isinstance(horizon_value, str):
            horizons = tuple(
                int(item.strip())
                for item in horizon_value.split(",")
                if item.strip()
            )
        else:
            horizons = tuple(int(item) for item in horizon_value)
        return cls(
            run_id=str(values.get("run_id", "phenome-discovery")),
            feature_glob=(
                str(values["feature_glob"])
                if values.get("feature_glob")
                else None
            ),
            panel_ids=panel_ids,
            train_fdr_alpha=float(values.get("train_fdr_alpha", 0.01)),
            validation_fdr_alpha=float(
                values.get("validation_fdr_alpha", 0.05)
            ),
            min_class_inference=int(values.get("min_class_inference", 30)),
            min_class_formal=int(values.get("min_class_formal", 100)),
            bootstrap_iterations=int(values.get("bootstrap_iterations", 1000)),
            max_phenotypes=(
                int(max_phenotypes)
                if max_phenotypes not in (None, "")
                else None
            ),
            max_future_phenotypes=(
                int(max_future_phenotypes)
                if max_future_phenotypes not in (None, "")
                else None
            ),
            future_horizons_days=horizons,
            force=bool(values.get("force", False)),
            include_promoted_candidates=bool(
                values.get("include_promoted_candidates", False)
            ),
        )


@dataclass(frozen=True)
class _PhenomeReplicationCache:
    train_numeric: Mapping[str, np.ndarray]
    validation_numeric: Mapping[str, np.ndarray]
    train_subjects: np.ndarray
    validation_subjects: np.ndarray
    train_row_index: Mapping[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray],
    ]
    validation_row_index: Mapping[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray],
    ]


def run_phenome_discovery(
    config: MimicProjectConfig,
    options: PhenomeDiscoveryOptions = PhenomeDiscoveryOptions(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    if not options.panel_ids:
        raise ValueError("At least one feature panel is required")
    if not 0 < options.train_fdr_alpha < 1:
        raise ValueError("train_fdr_alpha must be between zero and one")
    if not 0 < options.validation_fdr_alpha < 1:
        raise ValueError("validation_fdr_alpha must be between zero and one")

    feature_glob = options.feature_glob or str(
        config.features_root
        / "_".join(options.panel_ids)
        / "development"
        / "split=*"
        / "part-*.parquet"
    )
    feature_paths = tuple(
        sorted(Path(path).resolve() for path in glob.glob(feature_glob))
    )
    if not feature_paths:
        raise FileNotFoundError(
            f"No feature partitions match phenome input: {feature_glob}"
        )
    feature_integrity = verify_feature_artifacts(
        feature_paths,
        require_complete_index=True,
    )
    feature_scope = _resolved_feature_scope("explicit", feature_integrity)
    if feature_scope != "development_only":
        raise RuntimeError(
            "Formal phenome discovery requires the full development feature "
            "artifact; smoke previews cannot produce promotion candidates."
        )
    feature_snapshot = create_artifact_snapshot(
        config,
        artifact_kind="development_features",
        source_glob=feature_glob,
        snapshot_id=(
            f"{options.run_id}-phenome-features-"
            f"{str(feature_integrity['artifact_hash'])[:16]}"
        ),
        require_complete=True,
        notes=(
            f"phenome_run_id={options.run_id}",
            "feature_scope=development_only",
            "locked_test_included=False",
        ),
    )
    feature_snapshot.assert_freezable()

    diagnosis_path = config.standardized_root / "diagnoses_icd.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    context_path = (
        config.evidence_root / "clinical_context_features.parquet"
    )
    for path in (diagnosis_path, master_path, context_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    event_evidence_provenance = (
        verify_clinical_event_evidence_provenance(config)
    )
    event_evidence_provenance_path = (
        config.evidence_root / "evidence_provenance.json"
        if event_evidence_provenance.get("status")
        == "direct_event_backed"
        else (
            config.work_root
            / "provenance"
            / "clinical_event_evidence_equivalence.json"
        )
    )
    resources = prepare_phenotype_resources(config)
    run_root = config.runs_root / options.run_id
    phenome_root = run_root / "phenome"
    phenome_root.mkdir(parents=True, exist_ok=True)
    manifest_path = phenome_root / "phenome.manifest.json"
    input_hash = content_hash(
        {
            "diagnoses": file_fingerprint(diagnosis_path),
            "master": file_fingerprint(master_path),
            "clinical_context": file_fingerprint(context_path),
            "clinical_event_evidence_provenance": (
                file_content_fingerprint(
                    event_evidence_provenance_path
                )
            ),
            "feature_integrity": feature_integrity,
            "feature_snapshot_hash": feature_snapshot.snapshot_hash,
            "phenotype_resource_hash": resources.resource_hash,
            "context_association_registry_hash": (
                context_association_registry_hash()
            ),
            "future_phenome_version": FUTURE_PHENOME_VERSION,
            "options": {
                key: value
                for key, value in asdict(options).items()
                if key != "force"
            },
            # Run machinery (code_scope) cannot change a phenome estimate.
            "code_hash": _scientific_mimic_hash(),
        }
    )
    config_hash = scientific_config_hash(config)
    if not options.force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, PHENOME_STAGE_VERSION):
            return previous
    assert_development_run_mutable(
        run_root,
        operation="Phenome discovery",
    )
    checkpoint_root = (
        phenome_root
        / ".checkpoints"
        / f"phenome-{input_hash[:20]}"
    )
    (
        analysis_feature_glob,
        analysis_feature_cache,
        analysis_feature_cache_manifest,
    ) = _prepare_validation_feature_cache(
        catalog_path=config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        run_root=run_root,
        feature_integrity=feature_integrity,
        feature_snapshot_hash=feature_snapshot.snapshot_hash,
        progress=progress,
    )

    if progress:
        progress("PheWAS 1/9: deterministischen Index und ICD-Phänotypen erstellen")
    cohort_paths = _build_phenome_cohorts(
        config,
        resources=resources,
        diagnosis_path=diagnosis_path,
        master_path=master_path,
        output_root=phenome_root,
    )
    definitions = pd.read_parquet(resources.phecodex_definitions_path)
    summary = pd.read_parquet(cohort_paths["phenotype_summary"])
    # The hypothesis family is fixed from Train before any marker is evaluated
    # in Validation. Validation outcome counts may describe assessability after
    # the freeze, but must never select, rank, or cap the targets.
    eligible = summary.loc[
        summary["train_case_count"] >= options.min_class_inference
    ].copy()
    # Opt-in "seed from prior promotions" cycle (Package 4c): a phenotype
    # promoted in an earlier discover-phenome + refresh-phenome-promotions
    # cycle for this SAME run_id is guaranteed a screening slot below --
    # it still must clear the identical min_class_inference eligibility
    # gate above, this only protects it from the max_phenotypes cap.
    # Re-screening a self-discovered candidate is NOT independent
    # replication even with a fresh train/validation split, since the
    # phenotype selection itself is no longer blind -- _promotion_
    # candidates() tags these rows `seeded_from_promotion=True` so this
    # never looks identical to a first-pass exhaustive PhecodeX scan.
    seeded_phenotype_ids: frozenset[str] = frozenset()
    if options.include_promoted_candidates:
        prior_promotions_path = (
            phenome_root / "phenome_promotion_candidates.parquet"
        )
        if prior_promotions_path.is_file():
            prior_promotions = pd.read_parquet(
                prior_promotions_path, columns=["phenotype_id"]
            )
            seeded_phenotype_ids = frozenset(
                prior_promotions["phenotype_id"].astype(str)
            )
    if options.max_phenotypes is not None:
        eligible = eligible.sort_values(
            ["train_case_count", "phenotype_id"],
            ascending=[False, True],
        )
        if seeded_phenotype_ids:
            is_seeded = eligible["phenotype_id"].astype(str).isin(
                seeded_phenotype_ids
            )
            eligible = pd.concat(
                [eligible.loc[is_seeded], eligible.loc[~is_seeded]],
                ignore_index=True,
            )
        eligible = eligible.head(options.max_phenotypes)
    phenotype_ids = tuple(eligible["phenotype_id"].astype(str))
    if not phenotype_ids:
        raise RuntimeError("No phenotype meets the minimum Train case count")
    target_family_path = phenome_root / "phenome_target_family.parquet"
    target_family = pd.DataFrame(
        {
            "selection_order": np.arange(1, len(phenotype_ids) + 1, dtype=int),
            "phenotype_id": phenotype_ids,
            "selection_split": "train",
            "selection_rule": "train_case_count_then_phenotype_id",
            "validation_outcomes_used_for_selection": False,
            "family_frozen_before_validation_marker_evaluation": True,
        }
    )
    target_family["target_family_hash"] = content_hash(
        {
            "stage_version": PHENOME_STAGE_VERSION,
            "phenotype_ids": phenotype_ids,
            "panel_ids": options.panel_ids,
            "min_class_inference": options.min_class_inference,
            "max_phenotypes": options.max_phenotypes,
        }
    )
    atomic_write_parquet(target_family, target_family_path)
    frozen_ids = tuple(
        pd.read_parquet(target_family_path)
        .sort_values("selection_order")["phenotype_id"]
        .astype(str)
    )
    if frozen_ids != phenotype_ids:
        raise RuntimeError("Frozen PheWAS target family failed round-trip verification")

    panels = {item.panel_id: item for item in default_feature_panels()}
    unknown_panels = set(options.panel_ids) - set(panels)
    if unknown_panels:
        raise ValueError(f"Unknown panels: {sorted(unknown_panels)}")
    feature_ids: list[str] = []
    panel_map: dict[str, str] = {}
    for panel_id in options.panel_ids:
        for feature_id in panels[panel_id].feature_ids:
            if feature_id not in feature_ids:
                feature_ids.append(feature_id)
            panel_map.setdefault(
                feature_id,
                "classical" if feature_id.startswith("classical_") else panel_id,
            )

    if progress:
        progress("PheWAS 2/9: ECG-Features mit dem Index verknüpfen")
    feature_columns = _phenome_feature_columns(
        config,
        feature_glob=analysis_feature_glob,
    )
    candidate_features = tuple(
        feature_id for feature_id in feature_ids if feature_id in feature_columns
    )
    if not candidate_features:
        raise RuntimeError("Selected phenome panels have no extracted columns")
    # The quality markers are screened deliberately, in a family of their own.
    # They were removed from the analysis panels, which stops them being
    # reported as findings and also removes the only instrument able to show
    # that a phenotype's best "marker" is really its recording conditions --
    # the failure the last run produced for major depressive disorder. Their
    # own panel id keeps them out of the candidates' FDR families.
    quality_features = tuple(
        feature_id
        for feature_id in quality_control_features(feature_columns)
        if feature_id not in candidate_features
    )
    for feature_id in quality_features:
        panel_map[feature_id] = QUALITY_CONTROL_PANEL_ID
    screened_features = candidate_features + quality_features
    feature_index = _load_phenome_feature_index(
        config,
        feature_glob=analysis_feature_glob,
        index_path=cohort_paths["index_ecg"],
    )
    if feature_index["split"].eq("locked_test").any():
        raise RuntimeError("Locked-test rows are forbidden in phenome discovery")

    current = pd.read_parquet(
        cohort_paths["current_phenotypes"],
        filters=[("phenotype_id", "in", list(phenotype_ids))],
    )
    history_body = pd.read_parquet(cohort_paths["history_body_systems"])
    definition_lookup = definitions.set_index("phenotype_id")

    if progress:
        progress(
            f"PheWAS 3/9: Train-Screening {len(phenotype_ids):,} Phänotypen × "
            f"bis zu {len(screened_features):,} Marker "
            f"({len(quality_features)} davon Qualitaetskontrollen) in "
            f"{PHENOME_FEATURE_BLOCK_SIZE}-Bloecken"
        )
    train_results_unadjusted, available_features = (
        _checkpointed_train_screen_streaming(
            checkpoint_root,
            config=config,
            feature_glob=analysis_feature_glob,
            index_path=cohort_paths["index_ecg"],
            input_hash=input_hash,
            feature_index=feature_index,
            current=current.loc[current["split"] == "train"],
            history_body=history_body,
            phenotype_ids=phenotype_ids,
            feature_ids=screened_features,
            panel_map=panel_map,
            definition_lookup=definition_lookup,
            min_inference=options.min_class_inference,
            min_formal=options.min_class_formal,
            progress=progress,
        )
    )
    if not available_features:
        raise RuntimeError("No selected phenotype-scan feature is available")
    train_results = apply_fdr_by_family(
        train_results_unadjusted,
        family_columns=("panel_id",),
        alpha_by_panel={
            panel_id: options.train_fdr_alpha for panel_id in set(panel_map.values())
        },
    )
    # Audit 5.3.3: empirical calibration against the negative-control
    # phenotypes, reported beside the screen. The hit selection below reads
    # only the uncalibrated columns and is unchanged.
    from .empirical_calibration import calibrate_screen, negative_control_names

    train_results, empirical_nulls, empirical_summary = calibrate_screen(
        train_results, negative_control_names()
    )
    train_hits = train_results.loc[
        train_results["q_value"].le(options.train_fdr_alpha)
        & train_results["global_fdr_significant"].fillna(False)
        & train_results["hierarchical_fdr_significant"].fillna(False)
        & train_results["confirmatory_assessability_gate_passed"].fillna(False)
        & train_results["inference_status"].eq("formal_inference")
    ].copy().reset_index(drop=True)
    train_hits["replication_seed_index"] = np.arange(
        1,
        len(train_hits) + 1,
        dtype=int,
    )

    if progress:
        progress(
            f"PheWAS 4/9: {len(train_hits):,} Train-Treffer in Validation replizieren"
        )
    validation_unadjusted = _checkpointed_validation_replication_streaming(
        checkpoint_root,
        config=config,
        feature_glob=analysis_feature_glob,
        index_path=cohort_paths["index_ecg"],
        input_hash=input_hash,
        train_hits=train_hits,
        feature_index=feature_index,
        current=current,
        history_body=history_body,
        definition_lookup=definition_lookup,
        min_inference=options.min_class_inference,
        min_formal=options.min_class_formal,
        bootstrap_iterations=0,
        random_seed=config.random_seed,
        worker_count=config.workers,
        progress=progress,
        checkpoint_namespace="current_validation_screen",
        progress_label="PheWAS Validation-Screen",
    )
    if progress:
        progress('PheWAS Validation: gespeicherte Testergebnisse auf Mehrfachtests korrigieren')
    validation_results = apply_fdr_by_family(
        validation_unadjusted,
        family_columns=("panel_id",),
        alpha_by_panel={
            panel_id: options.validation_fdr_alpha
            for panel_id in set(panel_map.values())
        },
    )
    if not validation_results.empty:
        validation_results = add_auc_replication_status(
            validation_results,
            validation_fdr_alpha=options.validation_fdr_alpha,
        )
        validation_results["replicated"] = (
            validation_results["replicated"].fillna(False)
            & validation_results["global_fdr_significant"].fillna(False)
            & validation_results["hierarchical_fdr_significant"].fillna(False)
            & validation_results[
                "confirmatory_assessability_gate_passed"
            ].fillna(False)
        )
        validation_results["replication_multiplicity_scope"] = (
            "arm_wide_global_and_hierarchical_fdr"
        )
        bootstrap_hits = _replicated_bootstrap_hits(
            train_hits,
            validation_results,
        )
        if progress:
            progress(
                "PheWAS Validation-CI: "
                f"{len(bootstrap_hits):,} replizierte Treffer mit "
                f"{options.bootstrap_iterations:,} Subject-Bootstraps"
            )
        bootstrapped = _checkpointed_validation_replication_streaming(
            checkpoint_root,
            config=config,
            feature_glob=analysis_feature_glob,
            index_path=cohort_paths["index_ecg"],
            input_hash=input_hash,
            train_hits=bootstrap_hits,
            feature_index=feature_index,
            current=current,
            history_body=history_body,
            definition_lookup=definition_lookup,
            min_inference=options.min_class_inference,
            min_formal=options.min_class_formal,
            bootstrap_iterations=options.bootstrap_iterations,
            random_seed=config.random_seed,
            worker_count=config.workers,
            progress=progress,
            checkpoint_namespace="current_validation_intervals",
            progress_label="PheWAS Validation-CI",
        )
        if progress:
            progress('PheWAS Validation: gespeicherte Bootstrap-Intervalle zusammenführen')
        validation_results = _overlay_replicated_bootstraps(
            validation_results,
            bootstrapped,
        )
    replicated = (
        validation_results.loc[validation_results["replicated"]].copy()
        if not validation_results.empty
        else pd.DataFrame()
    )
    if progress:
        progress(
            "PheWAS 5/9: replizierte Pathologie-Treffer auf "
            "Alter, Geschlecht und Signalqualität adjustieren"
        )
    # The adjusted validation analysis is evaluated for the complete set of
    # hypotheses selected in Train.  Restricting it to unadjusted Validation
    # hits would select and test on the same observations and would make its
    # q-values anti-conservative.
    adjustment_candidates = train_hits.copy()
    if not adjustment_candidates.empty:
        adjustment_candidates["train_direction"] = (
            adjustment_candidates["direction"].astype(int)
        )
    adjusted_confirmation = _checkpointed_adjusted_confirmation_streaming(
        config,
        feature_glob=analysis_feature_glob,
        index_path=cohort_paths["index_ecg"],
        candidates=adjustment_candidates,
        feature_index=feature_index,
        current=current,
        history_body=history_body,
        policy=PhenomeAdjustmentPolicy(
            minimum_events_per_parameter=20,
            fdr_alpha=options.validation_fdr_alpha,
            minimum_effect_retention=(
                PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION
            ),
        ),
        checkpoint_root=checkpoint_root / "pathology_adjustment",
        checkpoint_input_hash=input_hash,
        progress=progress,
    )
    replicated_enriched = _merge_adjusted_confirmation(
        replicated,
        adjusted_confirmation,
    )
    promotable = replicated_enriched.loc[
        replicated_enriched.get(
            "adjustment_robust",
            pd.Series(False, index=replicated_enriched.index),
        ).fillna(False).astype(bool)
    ].copy()
    discovery_status = build_phenome_discovery_status(
        train_hits,
        validation_results,
        adjusted_confirmation,
    )
    promotions = _promotion_candidates(
        promotable,
        snapshot_hash=feature_snapshot.snapshot_hash,
        seeded_phenotype_ids=seeded_phenotype_ids,
    )
    diagnostics = _phenome_diagnostics(train_results, validation_results)

    if progress:
        progress(
            "PheWAS 6/9: getrennte klinische Assoziationsdomänen prüfen"
        )
    context_paths = run_context_association_discovery(
        config,
        feature_glob=analysis_feature_glob,
        feature_ids=available_features,
        panel_map=panel_map,
        output_root=phenome_root,
        train_fdr_alpha=options.train_fdr_alpha,
        validation_fdr_alpha=options.validation_fdr_alpha,
        min_inference=options.min_class_inference,
        min_formal=options.min_class_formal,
        bootstrap_iterations=options.bootstrap_iterations,
        random_seed=config.random_seed,
        checkpoint_root=checkpoint_root / "context_domains",
        checkpoint_input_hash=input_hash,
        progress=progress,
    )
    context_replicated = pd.read_parquet(context_paths["replicated"])
    if progress:
        progress(
            "PheWAS 7/9: zensierten Zukunfts-Phänotyp-Arm getrennt prüfen"
        )
    future_paths = run_future_phenome_discovery(
        config,
        feature_glob=analysis_feature_glob,
        feature_ids=available_features,
        panel_map=panel_map,
        resources=resources,
        output_root=phenome_root,
        horizons_days=options.future_horizons_days,
        train_fdr_alpha=options.train_fdr_alpha,
        validation_fdr_alpha=options.validation_fdr_alpha,
        min_inference=options.min_class_inference,
        min_formal=options.min_class_formal,
        bootstrap_iterations=options.bootstrap_iterations,
        random_seed=config.random_seed,
        worker_count=config.workers,
        max_phenotypes=_future_phenotype_limit(options),
        checkpoint_root=checkpoint_root / "future_phenome",
        checkpoint_input_hash=input_hash,
        progress=progress,
    )
    future_replicated = pd.read_parquet(future_paths["replicated"])
    broad_replicated = _broad_discovery_table(
        replicated_enriched,
        context_replicated,
        future_replicated,
    )
    replicated = _annotate_ecg_circularity(
        replicated,
        id_column="phenotype_id",
        name_column="phenotype_name",
    )
    # A negative control is only worth having if it is allowed to win.  On the
    # last development run this comparison marked 21 of 261 assessable
    # phenotypes as beaten by a recording-quality marker -- depression, anxiety,
    # cognitive impairment, wheelchair dependence: the phenotypes whose patients
    # move.  Those hits stay in the table; they stop being read as physiology.
    quality_verdicts = compare_quality_against_candidates(
        replicated,
        target_column="phenotype_id",
        quality_features=quality_features,
    )
    replicated = annotate_hits_with_quality_verdict(
        replicated,
        quality_verdicts,
        target_column="phenotype_id",
    )
    broad_replicated = _annotate_ecg_circularity(
        broad_replicated,
        id_column="target_id",
        name_column="target_name",
    )
    broad_ranked = rank_broad_discovery(broad_replicated)
    new_associations = build_new_association_explorer(
        discovery_status,
        broad_ranked,
    )
    snapshot_verification = verify_artifact_snapshot(feature_snapshot)
    if not snapshot_verification["valid"]:
        raise RuntimeError(
            "Development feature artifacts changed while phenome discovery "
            f"was running: {snapshot_verification}"
        )

    if progress:
        progress(
            "PheWAS 8/9: aktuelle Pathologien, Zukunftshypothesen und "
            "Nicht-Pathologie-Treffer "
            "getrennt zusammenfassen"
        )
    outputs = {
        "phenome_train_screening": train_results,
        "phenome_validation_replication": validation_results,
        "phenome_replicated_hits": replicated,
        "phenome_adjusted_confirmation": adjusted_confirmation,
        "phenome_discovery_status": discovery_status,
        "phenome_diagnostics": diagnostics,
        "phenome_promotion_candidates": pd.DataFrame(
            [item.to_dict() for item in promotions]
        ),
        "phenome_label_summary": summary,
        "phenome_quality_control_verdicts": quality_verdicts,
        "ccsr_audit_summary": pd.read_parquet(cohort_paths["ccsr_summary"]),
        "broad_discovery_replicated_hits": broad_replicated,
        "broad_discovery_ranked": broad_ranked,
        "new_associations_explorer": new_associations,
        "phenome_empirical_calibration_nulls": empirical_nulls,
    }
    output_paths: list[str] = []
    for name, frame in outputs.items():
        path = phenome_root / f"{name}.parquet"
        atomic_write_parquet(frame, path)
        output_paths.append(str(path.resolve()))
    calibration_summary_path = phenome_root / "phenome_empirical_calibration.json"
    atomic_write_json(calibration_summary_path, empirical_summary)
    output_paths.append(str(calibration_summary_path.resolve()))
    promotion_path = phenome_root / "promotion_candidates.json"
    atomic_write_json(
        promotion_path,
        {
            "status": "adjusted_replicated_draft_hypotheses_only",
            "locked_test_scanned": False,
            "adjustment_version": PHENOME_ADJUSTMENT_VERSION,
            "automatic_promotion": False,
            "candidates": [item.to_dict() for item in promotions],
        },
    )
    output_paths.append(str(promotion_path.resolve()))
    output_paths.append(str(target_family_path.resolve()))
    output_paths.extend(str(path.resolve()) for path in cohort_paths.values())
    output_paths.extend(
        str(path.resolve()) for path in context_paths.values()
    )
    output_paths.extend(str(path.resolve()) for path in future_paths.values())
    output_paths.append(str(analysis_feature_cache_manifest.resolve()))
    output_paths.append(
        str(event_evidence_provenance_path.resolve())
    )
    if progress:
        progress("PheWAS 9/9: Manifeste und Promotion-Audit versiegeln")
    manifest = StageManifest.complete(
        stage="discover-phenome",
        stage_version=PHENOME_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(dict.fromkeys(output_paths)),
        row_counts={
            "eligible_phenotypes": len(phenotype_ids),
            "frozen_target_family_size": len(target_family),
            "features": len(available_features),
            "train_associations": len(train_results),
            "train_hits": len(train_hits),
            "validation_associations": len(validation_results),
            "replicated_hits": len(replicated),
            "adjusted_confirmation_hits": len(adjusted_confirmation),
            "adjusted_robust_hits": int(
                adjusted_confirmation.get(
                    "adjustment_robust",
                    pd.Series(dtype=bool),
                ).fillna(False).sum()
            ),
            "likely_confounded_hits": int(
                adjusted_confirmation.get(
                    "likely_confounded",
                    pd.Series(dtype=bool),
                ).fillna(False).sum()
            ),
            "discovery_status_rows": len(discovery_status),
            "context_train_associations": len(
                pd.read_parquet(context_paths["train"])
            ),
            "context_validation_associations": len(
                pd.read_parquet(context_paths["validation"])
            ),
            "context_replicated_hits": len(context_replicated),
            "future_train_associations": len(
                pd.read_parquet(future_paths["train"])
            ),
            "future_validation_associations": len(
                pd.read_parquet(future_paths["validation"])
            ),
            "future_replicated_hits": len(future_replicated),
            "broad_replicated_hits": len(broad_replicated),
            "new_association_rows": len(new_associations),
        },
        notes=(
            "target_kind=retrospective_index_admission_association",
            "locked_test_scanned=false",
            f"mapping={PHENOTYPE_MAPPING_VERSION}",
            "index=first_coded_admission_ecg_per_subject_outcome_blind",
            "cases=phenotype_documented_on_or_before_index_discharge",
            "controls=no_phenotype_documentation_through_index_discharge",
            "future_diagnoses_never_used_to_define_cases_or_controls",
            "target_family_selection=train_only_frozen_before_validation",
            (
                "replication_stability_gate="
                f"{DISCOVERY_STABILITY_GATE_VERSION};"
                f"min_validation_auc={DISCOVERY_MIN_ALIGNED_AUC};"
                f"max_auc_gap={DISCOVERY_MAX_EFFECT_GAP};"
                f"min_effect_retention={DISCOVERY_MIN_EFFECT_RETENTION}"
            ),
            "validation_ci_policy=subject_cluster_bootstrap_replicated_hits_only",
            (
                "pathology_adjustment="
                f"{PHENOME_ADJUSTMENT_VERSION};"
                f"covariates={','.join(PHENOME_ADJUSTMENT_COVARIATES)};"
                "selection_source=train;direction_source=train;"
                "fit_split=validation;fdr_family=all_train_selected_hits;"
                f"min_effect_retention={PHENOME_ADJUSTMENT_MIN_EFFECT_RETENTION}"
            ),
            "promotion_requires_adjusted_confirmation=true",
            "automatic_promotion=false",
            "non_pathology_domains=laboratory,medication,procedure,outcome",
            "non_pathology_domains_are_never_promoted_as_diagnoses",
            (
                "future_phenome=censored_separate_arm;"
                f"horizons_days={','.join(map(str, options.future_horizons_days))}"
            ),
            (
                "future_phenotype_limit="
                f"{_future_phenotype_limit(options) or 'all'}"
            ),
            "future_phenome_promoted_automatically=false",
            f"feature_artifact_hash={feature_integrity['artifact_hash']}",
            f"feature_snapshot_hash={feature_snapshot.snapshot_hash}",
            (
                "clinical_event_evidence_provenance="
                f"{content_hash(_event_evidence_provenance_summary(event_evidence_provenance))}"
            ),
            (
                "analysis_feature_cache_hash="
                f"{analysis_feature_cache['cache_hash']}"
            ),
            f"operational_config_hash={operational_config_hash(config)}",
        ),
    )
    manifest.save(manifest_path)
    return manifest


def _feature_blocks(
    feature_ids: tuple[str, ...],
    *,
    size: int = PHENOME_FEATURE_BLOCK_SIZE,
) -> tuple[tuple[str, ...], ...]:
    if size < 1:
        raise ValueError("Phenome feature block size must be positive")
    return tuple(
        feature_ids[start : start + size]
        for start in range(0, len(feature_ids), size)
    )


def _combine_phenome_feature_block(
    feature_index: pd.DataFrame,
    feature_block: pd.DataFrame,
    feature_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Attach one numeric block to the stable, outcome-blind row index."""
    keys = ("subject_id", "study_id", "split")
    if len(feature_index) != len(feature_block):
        raise RuntimeError("Phenome feature block row count changed")
    left_keys = feature_index[list(keys)].reset_index(drop=True)
    right_keys = feature_block[list(keys)].reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise RuntimeError("Phenome feature block row order changed")
    output = feature_index.reset_index(drop=True).copy(deep=False)
    for feature_id in feature_ids:
        if feature_id in feature_block:
            output[feature_id] = feature_block[feature_id].to_numpy(copy=False)
    return output


def _verified_completed_block_paths(
    checkpoint_root: Path,
    *,
    namespace: str,
    input_hash: str,
    chunks: Sequence[tuple[int, tuple]],
) -> Optional[list[Path]]:
    """Reuse a complete block before loading its raw features or worker arrays.

    A matching key hash proves the exact feature selection used by the old
    path. Missing chunks or a different selection use that unchanged path.
    Every admitted file still passes the existing full integrity verifier.
    """
    if not chunks:
        return None
    for index, keys in chunks:
        manifest = checkpoint_root / namespace / f"part-{index:06d}.json"
        if not manifest.is_file():
            return None
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (payload.get("input_hash") != input_hash
                or payload.get("item_hash") != content_hash(list(keys))):
            return None

    def unavailable():
        raise RuntimeError("Checkpoint disappeared during block recovery")

    return [checkpointed_frame_chunk_path(
        checkpoint_root, namespace=namespace, chunk_index=index,
        input_hash=input_hash, item_keys=keys, compute=unavailable,
    )[0] for index, keys in chunks]


def _checkpointed_train_screen_streaming(
    checkpoint_root: Path,
    *,
    config: MimicProjectConfig,
    feature_glob: str,
    index_path: Path,
    input_hash: str,
    feature_index: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    phenotype_ids: tuple[str, ...],
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Screen feature blocks without ever materializing the wide matrix."""
    phenotype_chunks = tuple(
        phenotype_ids[index : index + PHENOME_TRAIN_CHUNK_SIZE]
        for index in range(0, len(phenotype_ids), PHENOME_TRAIN_CHUNK_SIZE)
    )
    blocks = _feature_blocks(feature_ids)
    train_index = feature_index.loc[
        feature_index["split"].astype(str).eq("train")
    ].reset_index(drop=True)
    paths: list[Path] = []
    available: list[str] = []
    completed = 0
    total = max(1, len(blocks) * len(phenotype_chunks))
    for block_index, requested_block in enumerate(blocks):
        saved = _verified_completed_block_paths(
            checkpoint_root, namespace="current_train_stream_v1", input_hash=input_hash,
            chunks=[(block_index * len(phenotype_chunks) + chunk_index,
                     tuple((phenotype, feature) for phenotype in chunk for feature in requested_block))
                    for chunk_index, chunk in enumerate(phenotype_chunks)],
        )
        if saved is not None:
            paths.extend(saved)
            available.extend(requested_block)
            completed += len(phenotype_chunks)
            if progress:
                progress(f"PheWAS Train-Block {completed:,}/{total:,}: Checkpoint fortgesetzt")
            continue
        raw_block = _load_phenome_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=requested_block,
        )
        selected_block = tuple(
            feature_id
            for feature_id in requested_block
            if feature_id in raw_block
            and pd.to_numeric(
                raw_block[feature_id], errors="coerce"
            ).notna().any()
        )
        if not selected_block:
            completed += len(phenotype_chunks)
            continue
        available.extend(selected_block)
        train_block = raw_block.loc[
            raw_block["split"].astype(str).eq("train")
        ].reset_index(drop=True)
        features = _combine_phenome_feature_block(
            train_index,
            train_block,
            selected_block,
        )
        prepared_block = _prepare_screen_block(features, selected_block)
        # The phenotype chunks of one block are independent: they read the same
        # feature frame and write to distinct checkpoint files derived from
        # their chunk index.  Threads rather than processes, because the frame
        # is shared read-only and pickling a hundred megabytes per chunk would
        # cost more than the work.
        #
        # Measured on synthetic data at the real shape (40,000 recordings, 16
        # markers, four chunks): 18.24 s sequential, 10.74 s on two threads
        # (1.70x), 8.47 s on four (2.15x), with identical statistics.  numpy
        # releases the GIL for the sorting and searching this loop is made of,
        # which is why threads help at all -- the scaling is partial because
        # the remaining Python-level bookkeeping does not.
        #
        # Numeric columns and sorted controls are prepared once per feature
        # block and shared read-only.  Rebuilding them in every 25-phenotype
        # chunk previously multiplied the dominant sort by roughly 116.
        def _screen_chunk(item):
            phenotype_chunk_index, phenotype_chunk = item
            chunk_index = (
                block_index * len(phenotype_chunks) + phenotype_chunk_index
            )
            return checkpointed_frame_chunk_path(
                checkpoint_root,
                namespace="current_train_stream_v1",
                chunk_index=chunk_index,
                input_hash=input_hash,
                item_keys=tuple(
                    (phenotype_id, feature_id)
                    for phenotype_id in phenotype_chunk
                    for feature_id in selected_block
                ),
                compute=lambda phenotypes=phenotype_chunk, markers=selected_block: (
                    pd.DataFrame(
                        _screen_split(
                            features,
                            current,
                            history_body,
                            phenotype_ids=phenotypes,
                            feature_ids=markers,
                            panel_map=panel_map,
                            definition_lookup=definition_lookup,
                            min_inference=min_inference,
                            min_formal=min_formal,
                            progress=None,
                            prepared_block=prepared_block,
                        )
                    )
                ),
            )

        worker_count = max(
            1, min(int(config.workers), len(phenotype_chunks))
        )
        items = list(enumerate(phenotype_chunks))
        from .grant_parallel import grant_parallel_map, grant_workers

        if grant_workers(len(items)) > 1:
            # Audit E11b: the screen is GIL-bound (two threads 1.70x, four
            # 2.15x, see above). Supervised runs use processes; the prepared
            # block's arrays reach the workers as shared memory maps, and each
            # chunk carries only the diagnosis rows of its phenotypes and
            # their phecode relatives -- exactly what ``_screen_split`` reads.
            # Results come back in chunk order.
            known = tuple(str(item) for item in definition_lookup.index)
            outcomes = grant_parallel_map(
                _screen_chunk_task,
                [
                    (
                        checkpoint_root,
                        block_index * len(phenotype_chunks) + chunk_number,
                        input_hash,
                        tuple(phenotype_chunk),
                        tuple(selected_block),
                        _current_rows_for_chunk(current, phenotype_chunk, known),
                        dict(panel_map),
                        definition_lookup,
                        int(min_inference),
                        int(min_formal),
                        _prepared_block_for_rows(
                            prepared_block,
                            _current_rows_for_chunk(current, phenotype_chunk, known),
                        ),
                    )
                    for chunk_number, phenotype_chunk in items
                ],
            )
        elif worker_count == 1:
            outcomes = [_screen_chunk(item) for item in items]
        else:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="vcg-phewas-train",
            ) as executor:
                # Results are collected in submission order, so the checkpoint
                # path list -- and therefore the concatenated screen -- does not
                # depend on which thread finished first.
                outcomes = list(executor.map(_screen_chunk, items))
        for path, resumed in outcomes:
            paths.append(path)
            completed += 1
            if progress:
                progress(
                    f"PheWAS Train-Block {completed:,}/{total:,}: "
                    f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
                )
        del prepared_block, features, train_block, raw_block
    if progress:
        progress('PheWAS Train: gespeicherte Ergebnis-Chunks zusammenführen und ursprüngliche Reihenfolge wiederherstellen')
    output = materialize_frame_chunks(paths)
    available_features = tuple(available)
    if not output.empty:
        phenotype_order = _legacy_train_phenotype_order(
            phenotype_ids,
            current,
            definition_lookup,
        )
        feature_order = {
            feature_id: index
            for index, feature_id in enumerate(available_features)
        }
        output = (
            output.assign(
                _phenotype_order=output["phenotype_id"].map(phenotype_order),
                _feature_order=output["feature_id"].map(feature_order),
            )
            .sort_values(
                ["_phenotype_order", "_feature_order"],
                kind="stable",
            )
            .drop(columns=["_phenotype_order", "_feature_order"])
            .reset_index(drop=True)
        )
    return output, available_features


def _legacy_train_phenotype_order(
    phenotype_ids: tuple[str, ...],
    current: pd.DataFrame,
    definition_lookup: pd.DataFrame,
) -> dict[str, int]:
    present = set(current["phenotype_id"].astype(str))
    ordered: list[str] = []
    for start in range(0, len(phenotype_ids), PHENOME_TRAIN_CHUNK_SIZE):
        chunk = phenotype_ids[start : start + PHENOME_TRAIN_CHUNK_SIZE]
        evaluable = [
            phenotype_id
            for phenotype_id in chunk
            if phenotype_id in present and phenotype_id in definition_lookup.index
        ]
        ordered.extend(
            sorted(
                evaluable,
                key=lambda phenotype_id: (
                    str(definition_lookup.loc[phenotype_id, "body_system"]),
                    str(phenotype_id),
                ),
            )
        )
    return {phenotype_id: index for index, phenotype_id in enumerate(ordered)}


def _checkpointed_validation_replication_streaming(
    checkpoint_root: Path,
    *,
    config: MimicProjectConfig,
    feature_glob: str,
    index_path: Path,
    input_hash: str,
    train_hits: pd.DataFrame,
    feature_index: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int,
    progress: Optional[Callable[[str], None]],
    checkpoint_namespace: str,
    progress_label: str,
) -> pd.DataFrame:
    if train_hits.empty:
        return pd.DataFrame()
    ordered_features = tuple(
        dict.fromkeys(train_hits["feature_id"].astype(str).tolist())
    )
    paths: list[Path] = []
    completed = 0
    for block_index, feature_block in enumerate(
        _feature_blocks(ordered_features)
    ):
        selected_hits = train_hits.loc[
            train_hits["feature_id"].astype(str).isin(feature_block)
        ].copy()
        saved = _verified_completed_block_paths(
            checkpoint_root,
            namespace=f"{checkpoint_namespace}_stream_v1/block-{block_index:04d}",
            input_hash=input_hash,
            chunks=[(chunk_index, tuple((str(row.phenotype_id), str(row.feature_id), str(row.panel_id))
                                      for row in selected_hits.iloc[start:start + PHENOME_REPLICATION_CHUNK_SIZE]
                                      .itertuples(index=False)))
                    for chunk_index, start in enumerate(range(0, len(selected_hits), PHENOME_REPLICATION_CHUNK_SIZE))],
        )
        if saved is not None:
            paths.extend(saved)
            completed += len(selected_hits)
            if progress:
                progress(f"{progress_label} {completed:,}/{len(train_hits):,}: Checkpoint fortgesetzt")
            continue
        raw_block = _load_phenome_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=feature_block,
        )
        features = _combine_phenome_feature_block(
            feature_index,
            raw_block,
            feature_block,
        )
        cache = _prepare_phenome_replication_cache(
            selected_hits,
            features,
            current,
            history_body,
            definition_lookup,
        )
        for chunk_index, start in enumerate(
            range(0, len(selected_hits), PHENOME_REPLICATION_CHUNK_SIZE)
        ):
            chunk = selected_hits.iloc[
                start : start + PHENOME_REPLICATION_CHUNK_SIZE
            ].copy()
            keys = tuple(
                (
                    str(row.phenotype_id),
                    str(row.feature_id),
                    str(row.panel_id),
                )
                for row in chunk.itertuples(index=False)
            )
            path, resumed = checkpointed_frame_chunk_path(
                checkpoint_root,
                namespace=(
                    f"{checkpoint_namespace}_stream_v1/"
                    f"block-{block_index:04d}"
                ),
                chunk_index=chunk_index,
                input_hash=input_hash,
                item_keys=keys,
                compute=lambda selected=chunk: pd.DataFrame(
                    _replicate_hits(
                        selected,
                        features,
                        current,
                        history_body,
                        definition_lookup=definition_lookup,
                        min_inference=min_inference,
                        min_formal=min_formal,
                        bootstrap_iterations=bootstrap_iterations,
                        random_seed=random_seed,
                        worker_count=worker_count,
                        progress=None,
                        replication_cache=cache,
                    )
                ),
            )
            paths.append(path)
            completed += len(chunk)
            if progress:
                progress(
                    f"{progress_label} {completed:,}/{len(train_hits):,}: "
                    f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
                )
        del cache, features, raw_block
    if progress:
        progress(f'{progress_label}: gespeicherte Ergebnis-Chunks zusammenführen und ursprüngliche Reihenfolge wiederherstellen')
    output = materialize_frame_chunks(paths)
    if output.empty:
        return output
    order = {
        (
            str(row.phenotype_id),
            str(row.feature_id),
            str(row.panel_id),
        ): index
        for index, row in enumerate(train_hits.itertuples(index=False))
    }
    output["_replication_order"] = [
        order[(str(row.phenotype_id), str(row.feature_id), str(row.panel_id))]
        for row in output.itertuples(index=False)
    ]
    return output.sort_values("_replication_order", kind="stable").drop(
        columns="_replication_order"
    ).reset_index(drop=True)


def _checkpointed_adjusted_confirmation_streaming(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    candidates: pd.DataFrame,
    feature_index: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    policy: PhenomeAdjustmentPolicy,
    checkpoint_root: Path,
    checkpoint_input_hash: str,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if candidates.empty:
        return confirm_replicated_phenome_hits(
            candidates,
            feature_index,
            current,
            history_body,
            policy=policy,
        )
    validation_index = feature_index.loc[
        feature_index["split"].astype(str).eq("validation")
    ].reset_index(drop=True)
    ordered_features = tuple(
        dict.fromkeys(candidates["feature_id"].astype(str).tolist())
    )
    paths: list[Path] = []
    for block_index, feature_block in enumerate(
        _feature_blocks(ordered_features)
    ):
        selected = candidates.loc[
            candidates["feature_id"].astype(str).isin(feature_block)
        ].copy()
        keys = tuple((str(row.phenotype_id), str(row.feature_id), str(row.panel_id))
                     for row in selected.itertuples(index=False))
        saved = _verified_completed_block_paths(checkpoint_root, namespace='stream_blocks_v1',
            input_hash=checkpoint_input_hash, chunks=[(block_index, keys)])
        if saved is not None:
            paths.extend(saved)
            if progress:
                progress(f'PheWAS Adjustierungsblock {block_index + 1:,}/{len(_feature_blocks(ordered_features)):,}: Checkpoint fortgesetzt (ohne Rohdaten-JOIN)')
            continue
        raw_block = _load_phenome_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=feature_block,
            splits=("validation",),
        )
        features = _combine_phenome_feature_block(
            validation_index,
            raw_block,
            feature_block,
        )
        keys = tuple(
            (str(row.phenotype_id), str(row.feature_id), str(row.panel_id))
            for row in selected.itertuples(index=False)
        )
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace="stream_blocks_v1",
            chunk_index=block_index,
            input_hash=checkpoint_input_hash,
            item_keys=keys,
            compute=lambda block_hits=selected, block_features=features, index=block_index: (
                confirm_replicated_phenome_hits(
                    block_hits,
                    block_features,
                    current,
                    history_body,
                    policy=policy,
                    checkpoint_root=(
                        checkpoint_root / f"block-{index:04d}"
                    ),
                    checkpoint_input_hash=checkpoint_input_hash,
                    progress=None,
                    finalize=False,
                )
            ),
        )
        paths.append(path)
        if progress:
            progress(
                f"PheWAS Adjustierungsblock {block_index + 1:,}/"
                f"{len(_feature_blocks(ordered_features)):,}: "
                f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
            )
        del features, raw_block
    raw_output = materialize_frame_chunks(paths)
    order = {
        (
            str(row.phenotype_id),
            str(row.feature_id),
            str(row.panel_id),
        ): index
        for index, row in enumerate(candidates.itertuples(index=False))
    }
    if not raw_output.empty:
        raw_output["_candidate_order"] = [
            order[(str(row.phenotype_id), str(row.feature_id), str(row.panel_id))]
            for row in raw_output.itertuples(index=False)
        ]
        raw_output = raw_output.sort_values(
            "_candidate_order", kind="stable"
        ).drop(columns="_candidate_order").reset_index(drop=True)
    return finalize_phenome_adjustment_output(raw_output, policy)


def _checkpointed_train_screen(
    checkpoint_root: Path,
    *,
    input_hash: str,
    features: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    phenotype_ids: tuple[str, ...],
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    paths: list[Path] = []
    prepared_block = _prepare_screen_block(features, feature_ids)
    chunks = tuple(
        phenotype_ids[index:index + PHENOME_TRAIN_CHUNK_SIZE]
        for index in range(0, len(phenotype_ids), PHENOME_TRAIN_CHUNK_SIZE)
    )
    completed = 0
    for chunk_index, phenotype_chunk in enumerate(chunks):
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace="current_train",
            chunk_index=chunk_index,
            input_hash=input_hash,
            item_keys=phenotype_chunk,
            compute=lambda chunk=phenotype_chunk: pd.DataFrame(
                _screen_split(
                    features,
                    current,
                    history_body,
                    phenotype_ids=chunk,
                    feature_ids=feature_ids,
                    panel_map=panel_map,
                    definition_lookup=definition_lookup,
                    min_inference=min_inference,
                    min_formal=min_formal,
                    progress=None,
                    prepared_block=prepared_block,
                )
            ),
        )
        paths.append(path)
        completed += len(phenotype_chunk)
        if progress:
            progress(
                f"PheWAS Train {completed:,}/{len(phenotype_ids):,}: "
                f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
            )
    return materialize_frame_chunks(paths)


def _checkpointed_validation_replication(
    checkpoint_root: Path,
    *,
    input_hash: str,
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int,
    progress: Optional[Callable[[str], None]],
    replication_cache: Optional[_PhenomeReplicationCache] = None,
    checkpoint_namespace: str = "current_validation",
    progress_label: str = "PheWAS Validation",
) -> pd.DataFrame:
    if train_hits.empty:
        return pd.DataFrame()
    cache = replication_cache or _prepare_phenome_replication_cache(
        train_hits,
        features,
        current,
        history_body,
        definition_lookup,
    )
    paths: list[Path] = []
    completed = 0
    for chunk_index, start in enumerate(
        range(0, len(train_hits), PHENOME_REPLICATION_CHUNK_SIZE)
    ):
        chunk = train_hits.iloc[
            start:start + PHENOME_REPLICATION_CHUNK_SIZE
        ].copy()
        keys = tuple(
            (
                str(row.phenotype_id),
                str(row.feature_id),
                str(row.panel_id),
            )
            for row in chunk.itertuples(index=False)
        )
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace=checkpoint_namespace,
            chunk_index=chunk_index,
            input_hash=input_hash,
            item_keys=keys,
            compute=lambda selected=chunk, offset=start: pd.DataFrame(
                _replicate_hits(
                    selected,
                    features,
                    current,
                    history_body,
                    definition_lookup=definition_lookup,
                    min_inference=min_inference,
                    min_formal=min_formal,
                    bootstrap_iterations=bootstrap_iterations,
                    random_seed=random_seed,
                    worker_count=worker_count,
                    progress=None,
                    seed_offset=offset,
                    replication_cache=cache,
                )
            ),
        )
        paths.append(path)
        completed += len(chunk)
        if progress:
            progress(
                f"{progress_label} {completed:,}/{len(train_hits):,}: "
                f"{'Checkpoint fortgesetzt' if resumed else 'Checkpoint geschrieben'}"
            )
    return materialize_frame_chunks(paths)


def _replicated_bootstrap_hits(
    train_hits: pd.DataFrame,
    validation_results: pd.DataFrame,
) -> pd.DataFrame:
    if train_hits.empty or validation_results.empty:
        return train_hits.iloc[0:0].copy()
    keys = ("phenotype_id", "feature_id", "panel_id")
    replicated_keys = {
        tuple(str(getattr(row, key)) for key in keys)
        for row in validation_results.loc[
            validation_results["replicated"].fillna(False)
        ].itertuples(index=False)
    }
    selected = [
        tuple(str(getattr(row, key)) for key in keys) in replicated_keys
        for row in train_hits.itertuples(index=False)
    ]
    return train_hits.loc[selected].copy()


def _overlay_replicated_bootstraps(
    validation_results: pd.DataFrame,
    bootstrapped: pd.DataFrame,
) -> pd.DataFrame:
    output = validation_results.copy()
    output["bootstrap_scope"] = np.where(
        output["replicated"].fillna(False),
        "subject_cluster_replicated_hit",
        "not_requested_not_replicated",
    )
    if bootstrapped.empty:
        return output
    keys = ["phenotype_id", "feature_id", "panel_id"]
    bootstrap_columns = [
        column
        for column in bootstrapped.columns
        if column.endswith("_ci_low")
        or column.endswith("_ci_high")
        or column in {
            "bootstrap_iterations",
            "bootstrap_successful_iterations",
            "bootstrap_unit",
        }
    ]
    updates = bootstrapped[
        keys + bootstrap_columns
    ].drop_duplicates(keys)
    if updates.duplicated(keys).any():
        raise RuntimeError("Duplicate PheWAS bootstrap result keys")
    merged = output.merge(
        updates.assign(_bootstrap_matched=True),
        on=keys,
        how="left",
        suffixes=("", "__bootstrap"),
        validate="one_to_one",
    )
    matched = merged["_bootstrap_matched"].eq(True)
    drop_columns = ['_bootstrap_matched']
    for column in bootstrap_columns:
        update_column = f"{column}__bootstrap"
        if update_column in merged:
            merged.loc[matched, column] = merged.loc[
                matched,
                update_column,
            ].to_numpy()
            drop_columns.append(update_column)
    merged = merged.drop(columns=drop_columns)
    return merged


def _future_phenotype_limit(
    options: PhenomeDiscoveryOptions,
) -> Optional[int]:
    limits = [
        int(value)
        for value in (
            options.max_phenotypes,
            options.max_future_phenotypes,
        )
        if value is not None
    ]
    return min(limits) if limits else None


def _merge_adjusted_confirmation(
    replicated: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    if replicated.empty or adjusted.empty:
        return replicated.copy()
    keys = ["phenotype_id", "feature_id", "panel_id"]
    if adjusted.duplicated(keys).any():
        raise RuntimeError(
            "Adjusted PheWAS confirmation contains duplicate hit keys"
        )
    new_columns = [
        column
        for column in adjusted.columns
        if column in keys or column not in replicated.columns
    ]
    return replicated.merge(
        adjusted[new_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def _broad_discovery_table(
    pathology_hits: pd.DataFrame,
    context_hits: pd.DataFrame,
    future_hits: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    columns = (
        "discovery_class",
        "discovery_domain",
        "target_id",
        "target_name",
        "target_kind",
        "feature_id",
        "panel_id",
        "train_effect",
        "train_q_value",
        "validation_effect",
        "validation_q_value",
        "direction_consistent",
        "replicated",
        "inference_status",
        "locked_test_scanned",
        "n_total",
        "n_positive",
        "n_negative",
        "missingness",
        "validation_auc",
        "validation_average_precision",
        "validation_ci_low",
        "validation_ci_high",
        "train_validation_gap",
        "effect_retention_ratio",
        "validation_stable",
        "adjustment_status",
        "adjusted_q_value",
        "adjusted_or_per_sd",
        "adjustment_robust",
        "likely_confounded",
        "confounding_control",
        "requires_endpoint_specific_adjustment",
        "feature_role_policy_version",
        "panel_version",
        "phenome_stage_version",
        "novel_vs_literature",
    )
    frames = []
    if not pathology_hits.empty:
        frames.append(
            pd.DataFrame(
                {
                    "discovery_class": "pathology_hypothesis",
                    "discovery_domain": pathology_hits["body_system"],
                    "target_id": pathology_hits["phenotype_id"],
                    "target_name": pathology_hits["phenotype_name"],
                    "target_kind": pathology_hits["target_kind"],
                    "feature_id": pathology_hits["feature_id"],
                    "panel_id": pathology_hits["panel_id"],
                    "train_effect": (
                        2.0 * pathology_hits["train_aligned_auc"] - 1.0
                    )
                    * pathology_hits["train_direction"],
                    "train_q_value": pathology_hits["train_q_value"],
                    "validation_effect": (
                        2.0 * pathology_hits["validation_raw_auc"] - 1.0
                    ),
                    "validation_q_value": pathology_hits["q_value"],
                    "direction_consistent": pathology_hits[
                        "direction_consistent"
                    ],
                    "replicated": pathology_hits["replicated"],
                    "inference_status": pathology_hits["inference_status"],
                    "locked_test_scanned": False,
                    "n_total": _column(pathology_hits, "n_total"),
                    "n_positive": _column(pathology_hits, "n_positive"),
                    "n_negative": _column(pathology_hits, "n_negative"),
                    "missingness": _column(
                        pathology_hits,
                        "case_missingness",
                    ),
                    "validation_auc": _column(
                        pathology_hits,
                        "validation_aligned_auc",
                    ),
                    "validation_average_precision": _column(
                        pathology_hits,
                        "average_precision",
                    ),
                    "validation_ci_low": _column(
                        pathology_hits,
                        "roc_auc_ci_low",
                    ),
                    "validation_ci_high": _column(
                        pathology_hits,
                        "roc_auc_ci_high",
                    ),
                    "train_validation_gap": _column(
                        pathology_hits,
                        "train_validation_auc_gap",
                    ),
                    "effect_retention_ratio": _column(
                        pathology_hits,
                        "effect_retention_ratio",
                    ),
                    "validation_stable": _column(
                        pathology_hits,
                        "validation_stable",
                        False,
                    ),
                    "adjustment_status": _column(
                        pathology_hits,
                        "adjustment_status",
                        "not_evaluable",
                    ),
                    "adjusted_q_value": _column(
                        pathology_hits,
                        "adjusted_q_value",
                    ),
                    "adjusted_or_per_sd": _column(
                        pathology_hits,
                        "adjusted_or_per_sd",
                    ),
                    "adjustment_robust": _column(
                        pathology_hits,
                        "adjustment_robust",
                        False,
                    ),
                    "likely_confounded": _column(
                        pathology_hits,
                        "likely_confounded",
                        False,
                    ),
                    "confounding_control": _column(
                        pathology_hits,
                        "confounding_control",
                        "none_broad_screen",
                    ),
                    "requires_endpoint_specific_adjustment": True,
                }
            )
        )
    if not context_hits.empty:
        frames.append(
            pd.DataFrame(
                {
                    "discovery_class": "non_pathology_association",
                    "discovery_domain": context_hits["domain"],
                    "target_id": context_hits["association_id"],
                    "target_name": context_hits["association_name"],
                    "target_kind": context_hits["target_kind"],
                    "feature_id": context_hits["feature_id"],
                    "panel_id": context_hits["panel_id"],
                    "train_effect": context_hits["train_effect"],
                    "train_q_value": context_hits["train_q_value"],
                    "validation_effect": context_hits[
                        "validation_effect"
                    ],
                    "validation_q_value": context_hits[
                        "validation_q_value"
                    ],
                    "direction_consistent": context_hits[
                        "direction_consistent"
                    ],
                    "replicated": context_hits["replicated"],
                    "inference_status": context_hits["inference_status"],
                    "locked_test_scanned": False,
                    "n_total": _column(context_hits, "n_total"),
                    "n_positive": _column(context_hits, "n_positive"),
                    "n_negative": _column(context_hits, "n_negative"),
                    "missingness": _column(context_hits, "missingness"),
                    "validation_auc": _column(context_hits, "raw_auc"),
                    "validation_average_precision": np.nan,
                    "validation_ci_low": _column(
                        context_hits,
                        "effect_ci_low",
                    ),
                    "validation_ci_high": _column(
                        context_hits,
                        "effect_ci_high",
                    ),
                    "train_validation_gap": _column(
                        context_hits,
                        "train_validation_effect_gap",
                    ),
                    "effect_retention_ratio": _column(
                        context_hits,
                        "effect_retention_ratio",
                    ),
                    "validation_stable": _column(
                        context_hits,
                        "validation_stable",
                        False,
                    ),
                    "adjustment_status": "not_run_separate_domain",
                    "adjusted_q_value": np.nan,
                    "adjusted_or_per_sd": np.nan,
                    "adjustment_robust": False,
                    "likely_confounded": False,
                    "confounding_control": "none_broad_screen",
                    "requires_endpoint_specific_adjustment": True,
                }
            )
        )
    if future_hits is not None and not future_hits.empty:
        frames.append(
            pd.DataFrame(
                {
                    "discovery_class": "future_outcome_hypothesis",
                    "discovery_domain": future_hits["body_system"],
                    "target_id": (
                        future_hits["phenotype_id"].astype(str)
                        + "@"
                        + future_hits["horizon_days"].astype(str)
                        + "d"
                    ),
                    "target_name": (
                        future_hits["phenotype_name"].astype(str)
                        + " within "
                        + future_hits["horizon_days"].astype(str)
                        + " days"
                    ),
                    "target_kind": _column(
                        future_hits,
                        "target_kind",
                        "post_discharge_prognostic_association",
                    ),
                    "feature_id": future_hits["feature_id"],
                    "panel_id": future_hits["panel_id"],
                    "train_effect": _column(
                        future_hits,
                        "cox_log_hazard_ratio_per_sd",
                    ),
                    "train_q_value": future_hits["train_q_value"],
                    "validation_effect": _column(
                        future_hits,
                        "validation_cox_log_hazard_ratio_per_sd",
                    ),
                    "validation_q_value": future_hits[
                        "validation_q_value"
                    ],
                    "direction_consistent": future_hits[
                        "direction_consistent"
                    ],
                    "replicated": future_hits["replicated"],
                    "inference_status": future_hits["inference_status"],
                    "locked_test_scanned": False,
                    "n_total": _column(future_hits, "n_at_risk"),
                    "n_positive": _column(future_hits, "n_positive"),
                    "n_negative": _column(future_hits, "n_negative"),
                    "missingness": _column(
                        future_hits,
                        "marker_missingness_at_risk",
                    ),
                    "validation_auc": _column(
                        future_hits,
                        "validation_aligned_auc",
                    ),
                    "validation_average_precision": _column(
                        future_hits,
                        "average_precision",
                    ),
                    "validation_ci_low": np.log(
                        pd.to_numeric(
                            _column(
                                future_hits,
                                "validation_cox_hr_ci_low",
                            ),
                            errors="coerce",
                        )
                    ),
                    "validation_ci_high": np.log(
                        pd.to_numeric(
                            _column(
                                future_hits,
                                "validation_cox_hr_ci_high",
                            ),
                            errors="coerce",
                        )
                    ),
                    "landmark_auc_ci_low": _column(
                        future_hits,
                        "roc_auc_ci_low",
                    ),
                    "landmark_auc_ci_high": _column(
                        future_hits,
                        "roc_auc_ci_high",
                    ),
                    "train_validation_gap": _column(
                        future_hits,
                        "train_validation_effect_gap",
                    ),
                    "effect_retention_ratio": _column(
                        future_hits,
                        "effect_retention_ratio",
                    ),
                    "validation_stable": _column(
                        future_hits,
                        "validation_stable",
                        False,
                    ),
                    "adjustment_status": "not_run_censored_future_arm",
                    "adjusted_q_value": np.nan,
                    "adjusted_or_per_sd": np.nan,
                    "adjustment_robust": False,
                    "likely_confounded": False,
                    "confounding_control": "none_broad_screen",
                    "requires_endpoint_specific_adjustment": True,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=columns)
    output = pd.concat(frames, ignore_index=True)
    output["feature_role_policy_version"] = FEATURE_ROLE_POLICY_VERSION
    output["panel_version"] = PANEL_VERSION
    output["phenome_stage_version"] = PHENOME_STAGE_VERSION
    output["novel_vs_literature"] = [
        classify_novelty(
            feature_id=str(feature_id),
            target_id=str(target_id),
            target_name=str(target_name),
            discovery_domain=str(discovery_domain),
        )
        for feature_id, target_id, target_name, discovery_domain in zip(
            output["feature_id"],
            output["target_id"],
            output["target_name"],
            output["discovery_domain"],
        )
    ]
    return output[list(columns)]


def _column(
    frame: pd.DataFrame,
    column: str,
    default: object = np.nan,
) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _build_phenome_cohorts(
    config: MimicProjectConfig,
    *,
    resources,
    diagnosis_path: Path,
    master_path: Path,
    output_root: Path,
) -> dict[str, Path]:
    paths = {
        "index_ecg": output_root / "phenome_index_ecg.parquet",
        "current_phenotypes": output_root / "current_phenotypes.parquet",
        "history_body_systems": output_root / "history_body_systems.parquet",
        "phenotype_summary": output_root / "phenotype_summary.parquet",
        "ccsr_current": output_root / "ccsr_current.parquet",
        "ccsr_summary": output_root / "ccsr_summary.parquet",
    }
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    dx = _sql_path(diagnosis_path)
    admissions = _sql_path(config.standardized_root / "admissions.parquet")
    master = _sql_path(master_path)
    mapping = _sql_path(resources.phecodex_mapping_path)
    definitions = _sql_path(resources.phecodex_definitions_path)
    ccsr = _sql_path(resources.ccsr_mapping_path)
    with catalog.connect() as connection:
        # Covariate columns required by the current analysis contract. An
        # ecg_master_index written by an older stage version may not carry
        # them; they then arrive as typed NULLs and the adjusted models
        # report the covariate as unavailable instead of crashing.
        master_columns = relation_columns(
            connection,
            f"read_parquet('{master}')",
        )
        optional_index_columns = ",\n                ".join(
            (
                optional_column_select(
                    master_columns, "anchor_year_group", "VARCHAR"
                ),
                optional_column_select(
                    master_columns, "machine_heart_rate_bpm", "DOUBLE"
                ),
                # Illness-acuity proxies. In an ED/ICU-heavy cohort acuity
                # correlates with nearly every ECG abnormality and nearly
                # every phenotype at once, so an age/sex-only adjustment
                # leaves the dominant confounder of the broad scan in place.
                (
                    "icustay_id IS NOT NULL AS index_icu_stay"
                    if "icustay_id" in master_columns
                    else "NULL::BOOLEAN AS index_icu_stay"
                ),
                optional_column_select(
                    master_columns, "admission_type", "VARCHAR"
                ),
                (
                    "DATE_DIFF('second', admittime, dischtime) / 86400.0"
                    " AS index_length_of_stay_days"
                    if {"admittime", "dischtime"} <= master_columns
                    else "NULL::DOUBLE AS index_length_of_stay_days"
                ),
                (
                    "ed_stay_id IS NOT NULL AS index_ed_encounter"
                    if "ed_stay_id" in master_columns
                    else "NULL::BOOLEAN AS index_ed_encounter"
                ),
            )
        )
        # One index ECG is fixed before phenotype mapping. This prevents a
        # later diagnosis from choosing a different ECG for cases than for
        # controls.
        _copy_query_atomic(
            connection,
            f"""
            WITH coded_admissions AS (
                SELECT DISTINCT subject_id, hadm_id
                FROM read_parquet('{dx}')
                WHERE subject_id IS NOT NULL AND hadm_id IS NOT NULL
            ),
            candidates AS (
                SELECT
                    m.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.subject_id
                        ORDER BY m.ecg_time, m.study_id
                    ) AS index_rank
                FROM read_parquet('{master}') m
                JOIN coded_admissions d USING (subject_id, hadm_id)
                WHERE m.split IN ('train', 'validation')
            )
            SELECT
                subject_id,
                study_id,
                hadm_id,
                ecg_time,
                dischtime,
                COALESCE(
                    GREATEST(ecg_time, dischtime),
                    ecg_time,
                    dischtime
                ) AS index_end_time,
                split,
                age,
                sex,
                race,
                {optional_index_columns},
                waveform_path,
                TRUE AS is_control_index,
                TRUE AS is_case_index,
                'first_coded_admission_ecg_outcome_blind'
                    AS index_selection_strategy
            FROM candidates
            WHERE index_rank = 1
            ORDER BY split, subject_id
            """,
            paths["index_ecg"],
        )
        index_path = _sql_path(paths["index_ecg"])
        _copy_query_atomic(
            connection,
            f"""
            WITH mapped_diagnoses AS (
                SELECT
                    dx.subject_id,
                    dx.hadm_id,
                    COALESCE(a.dischtime, a.admittime) AS documentation_time,
                    p.phenotype_id,
                    ANY_VALUE(d.body_system) AS body_system,
                    MIN(TRY_CAST(dx.seq_num AS INTEGER)) AS seq_num,
                    BOOL_OR(TRY_CAST(dx.seq_num AS INTEGER) = 1)
                        AS principal_diagnosis
                FROM read_parquet('{dx}') dx
                JOIN read_parquet('{mapping}') p
                  ON dx.icd_version = p.icd_version
                 AND dx.icd_code = p.icd_code
                JOIN read_parquet('{definitions}') d USING (phenotype_id)
                JOIN read_parquet('{admissions}') a
                  USING (subject_id, hadm_id)
                WHERE dx.subject_id IS NOT NULL
                  AND dx.hadm_id IS NOT NULL
                GROUP BY
                    dx.subject_id,
                    dx.hadm_id,
                    documentation_time,
                    p.phenotype_id
            )
            SELECT DISTINCT
                i.subject_id,
                i.study_id,
                i.hadm_id,
                i.split,
                d.phenotype_id,
                ANY_VALUE(d.body_system) AS body_system,
                MIN(d.seq_num) AS seq_num,
                BOOL_OR(d.principal_diagnosis) AS principal_diagnosis,
                'pre_or_index_discharge_history' AS phenotype_time_basis
            FROM read_parquet('{index_path}') i
            JOIN mapped_diagnoses d USING (subject_id)
            WHERE d.documentation_time IS NOT NULL
              AND d.documentation_time <= i.index_end_time
            GROUP BY
                i.subject_id,
                i.study_id,
                i.hadm_id,
                i.split,
                d.phenotype_id
            ORDER BY split, subject_id, phenotype_id
            """,
            paths["current_phenotypes"],
        )
        current_path = _sql_path(paths["current_phenotypes"])
        _copy_query_atomic(
            connection,
            f"""
            SELECT DISTINCT
                subject_id,
                body_system,
                'pre_or_index_discharge_only' AS history_time_basis
            FROM read_parquet('{current_path}')
            ORDER BY subject_id, body_system
            """,
            paths["history_body_systems"],
        )
        _copy_query_atomic(
            connection,
            f"""
            SELECT
                phenotype_id,
                ANY_VALUE(body_system) AS body_system,
                COUNT(DISTINCT CASE WHEN split='train' THEN subject_id END)
                    AS train_case_count,
                COUNT(DISTINCT CASE WHEN split='validation' THEN subject_id END)
                    AS validation_case_count,
                COUNT(DISTINCT subject_id) AS total_case_count,
                COUNT(DISTINCT CASE WHEN principal_diagnosis THEN subject_id END)
                    AS principal_case_count
            FROM read_parquet('{current_path}')
            GROUP BY phenotype_id
            ORDER BY phenotype_id
            """,
            paths["phenotype_summary"],
        )
        _copy_query_atomic(
            connection,
            f"""
            WITH mapped_diagnoses AS (
                SELECT DISTINCT
                    dx.subject_id,
                    dx.hadm_id,
                    COALESCE(a.dischtime, a.admittime) AS documentation_time,
                    c.ccsr_category,
                    c.ccsr_description,
                    c.body_system_code
                FROM read_parquet('{dx}') dx
                JOIN read_parquet('{ccsr}') c
                  ON dx.icd_version = 10
                 AND dx.icd_code = c.icd_code
                JOIN read_parquet('{admissions}') a
                  USING (subject_id, hadm_id)
                WHERE dx.subject_id IS NOT NULL
                  AND dx.hadm_id IS NOT NULL
            )
            SELECT DISTINCT
                i.subject_id,
                i.study_id,
                i.split,
                d.ccsr_category,
                d.ccsr_description,
                d.body_system_code,
                'pre_or_index_discharge_history' AS phenotype_time_basis
            FROM read_parquet('{index_path}') i
            JOIN mapped_diagnoses d USING (subject_id)
            WHERE d.documentation_time IS NOT NULL
              AND d.documentation_time <= i.index_end_time
            ORDER BY split, subject_id, ccsr_category
            """,
            paths["ccsr_current"],
        )
        ccsr_current = _sql_path(paths["ccsr_current"])
        _copy_query_atomic(
            connection,
            f"""
            SELECT
                ccsr_category,
                ANY_VALUE(ccsr_description) AS ccsr_description,
                ANY_VALUE(body_system_code) AS body_system_code,
                COUNT(DISTINCT CASE WHEN split='train' THEN subject_id END)
                    AS train_subject_count,
                COUNT(DISTINCT CASE WHEN split='validation' THEN subject_id END)
                    AS validation_subject_count
            FROM read_parquet('{ccsr_current}')
            GROUP BY ccsr_category
            ORDER BY ccsr_category
            """,
            paths["ccsr_summary"],
        )
    index = _read_parquet_columns(
        paths["index_ecg"],
        ("subject_id", "study_id", "split", "is_control_index"),
    )
    if index["split"].eq("locked_test").any():
        raise RuntimeError("Phenome index unexpectedly contains locked-test subjects")
    if index[["subject_id", "study_id"]].duplicated().any():
        raise RuntimeError("Phenome feature index contains duplicate ECG rows")
    control_index = index.loc[index["is_control_index"].fillna(False)]
    if control_index["subject_id"].duplicated().any():
        raise RuntimeError("Phenome controls must have one index ECG per subject")
    cases = _read_parquet_columns(
        paths["current_phenotypes"],
        ("subject_id", "phenotype_id"),
    )
    if cases.duplicated().any():
        raise RuntimeError(
            "Phenome cases must have one index ECG per subject and phenotype"
        )
    return paths


def _read_parquet_columns(
    path: Path,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    # PyArrow's dataset scanner requires a mutable list for projected columns.
    return pd.read_parquet(path, columns=list(columns))


def _load_phenome_features(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    feature_ids: tuple[str, ...],
) -> pd.DataFrame:
    available = _phenome_feature_columns(config, feature_glob=feature_glob)
    selected = tuple(item for item in feature_ids if item in available)
    if not selected:
        raise RuntimeError("Selected phenome panels have no extracted columns")
    feature_index = _load_phenome_feature_index(
        config,
        feature_glob=feature_glob,
        index_path=index_path,
    )
    feature_block = _load_phenome_feature_block(
        config,
        feature_glob=feature_glob,
        feature_ids=selected,
        index_path=index_path,
    )
    return _combine_phenome_feature_block(
        feature_index,
        feature_block,
        selected,
    )


def _phenome_feature_columns(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
) -> frozenset[str]:
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    with catalog.connect(isolated=True) as connection:
        return frozenset(parquet_columns(connection, feature_sql))


def _load_phenome_feature_index(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
) -> pd.DataFrame:
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    # Same measured reason as in validation.py: the option is kept only
    # when the partition footers actually differ.
    read_features = feature_parquet_clause(sql_path=feature_sql)
    index_sql = _sql_path(index_path)
    with catalog.connect(isolated=True) as connection:
        return connection.execute(
            f"""
            SELECT
                i.subject_id,
                i.study_id,
                i.split,
                i.is_control_index,
                i.is_case_index,
                i.age,
                i.sex,
                i.race,
                i.anchor_year_group,
                i.machine_heart_rate_bpm,
                i.index_icu_stay,
                i.admission_type,
                i.index_length_of_stay_days,
                i.index_ed_encounter,
                f.signal_quality_score,
                f.extraction_status
            FROM read_parquet('{index_sql}') i
            JOIN {read_features} f USING (subject_id, study_id, split)
            WHERE i.split IN ('train', 'validation')
            ORDER BY i.split, i.subject_id, i.study_id
            """
        ).fetchdf()


def _load_phenome_feature_block(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    feature_ids: tuple[str, ...],
    index_path: Path,
    splits: tuple[str, ...] = ("train", "validation"),
) -> pd.DataFrame:
    if not feature_ids:
        raise ValueError("Phenome feature block cannot be empty")
    if not splits or any(split not in {"train", "validation"} for split in splits):
        raise ValueError(f"Unsupported phenome splits: {splits!r}")
    selected = feature_ids
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    # Same measured reason as in validation.py: the option is kept only
    # when the partition footers actually differ.
    read_features = feature_parquet_clause(sql_path=feature_sql)
    index_sql = _sql_path(index_path)
    columns = ",\n                ".join(
        f"f.{_quote_identifier(item)}" for item in selected
    )
    split_sql = ", ".join(f"'{split}'" for split in splits)
    with catalog.connect(isolated=True) as connection:
        return connection.execute(
            f"""
            SELECT
                i.subject_id,
                i.study_id,
                i.split,
                {columns}
            FROM read_parquet('{index_sql}') i
            JOIN {read_features} f USING (subject_id, study_id, split)
            WHERE i.split IN ({split_sql})
            ORDER BY i.split, i.subject_id, i.study_id
            """
        ).fetchdf()


def _screen_split(
    features: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    *,
    phenotype_ids: tuple[str, ...],
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
    prepared_block: Optional[_PreparedScreenBlock] = None,
) -> list[dict]:
    prepared = prepared_block or _prepare_screen_block(features, feature_ids)
    study_to_row = prepared.study_to_row
    # Kept as an input artifact for provenance, but never use diagnoses from
    # other (especially future) admissions to manufacture super-controls.
    del history_body
    all_current = current
    current = current.loc[current["phenotype_id"].isin(phenotype_ids)]
    known_phenotypes = tuple(str(item) for item in definition_lookup.index)
    related_by_phenotype = {
        str(phenotype_id): _related_phenotype_ids(str(phenotype_id), known_phenotypes)
        for phenotype_id in phenotype_ids
    }
    related_ids = set().union(*related_by_phenotype.values()) if related_by_phenotype else set()
    related_subjects_by_phenotype = (
        {
            str(key): np.unique(frame["subject_id"].astype(int).to_numpy())
            for key, frame in all_current.loc[
                all_current["phenotype_id"].isin(related_ids),
                ["phenotype_id", "subject_id"],
            ].groupby("phenotype_id")
        }
        if related_ids
        else {}
    )
    records: list[dict] = []
    grouped = tuple(
        sorted(
            current.groupby("phenotype_id", sort=True),
            key=lambda item: (
                (
                    str(
                        definition_lookup.loc[
                            item[0],
                            "body_system",
                        ]
                    )
                    if item[0] in definition_lookup.index
                    else ""
                ),
                str(item[0]),
            ),
        )
    )
    all_subjects = prepared.all_subjects
    control_feature_rows = prepared.control_feature_rows
    control_subjects = prepared.control_subjects
    numeric = prepared.numeric
    # Rank each marker's controls once for the whole chunk instead of once per
    # phenotype.  A phenotype's controls are the global control set minus the
    # rows belonging to its own cases -- typically a fraction of a percent of
    # them -- so re-sorting the whole column for each phenotype repeated almost
    # the same sort 25 times per marker per block.  The exclusion is applied
    # arithmetically afterwards and gives the identical statistic, not an
    # approximation; ``excluded_mann_whitney_auc`` documents the algebra and
    # ``test_control_exclusion_identity`` pins it against the direct form.
    #
    # Measured at the real shape -- 200,000 controls, 400 cases, one 32-marker
    # block against one 25-phenotype chunk: 4.71 s before, 0.56 s after, an 8.4x
    # reduction with a largest AUC difference of exactly zero.  Extrapolated
    # over 55 blocks and 115 chunks that is about 8.3 hours per split before and
    # 1.0 hour after.
    #
    # Memory: one sorted float64 column per marker in the block, so at
    # PHENOME_FEATURE_BLOCK_SIZE = 32 and a few hundred thousand controls this
    # is on the order of a hundred megabytes, not the six gigabytes a full-width
    # cache would need.
    global_control_distributions = prepared.global_control_distributions
    for phenotype_index, (phenotype_id, case_frame) in enumerate(
        grouped,
        start=1,
    ):
        if phenotype_id not in definition_lookup.index:
            continue
        definition = definition_lookup.loc[phenotype_id]
        body_system = str(definition["body_system"])
        case_rows = np.asarray(
            [
                study_to_row[(int(row.subject_id), int(row.study_id))]
                for row in case_frame[
                    ["subject_id", "study_id"]
                ].drop_duplicates().itertuples(index=False)
                if (int(row.subject_id), int(row.study_id)) in study_to_row
            ],
            dtype=int,
        )
        # Audit X3: a sex-specific phenotype compares one sex with itself.
        sex_restriction = _phenotype_sex_restriction(definition)
        if sex_restriction is not None and len(prepared.all_sex):
            case_rows = case_rows[prepared.all_sex[case_rows] == sex_restriction]
        # A subject may contribute several indexable ECGs.  Compute the mask
        # once and use complementary views for the phenotype's controls and
        # the rows that must be subtracted from the global ranked controls.
        case_subjects = np.unique(all_subjects[case_rows])
        # Audit X4: relatives in the phecode hierarchy are no controls.
        related_subjects = np.unique(
            np.concatenate(
                [
                    related_subjects_by_phenotype[item]
                    for item in related_by_phenotype.get(str(phenotype_id), ())
                    if item in related_subjects_by_phenotype
                ]
                or [np.empty(0, dtype=int)]
            )
        )
        excluded_control_mask = np.isin(control_subjects, case_subjects)
        related_control_mask = (
            np.isin(control_subjects, related_subjects) & ~excluded_control_mask
            if related_subjects.size
            else np.zeros(len(control_subjects), dtype=bool)
        )
        excluded_control_mask = excluded_control_mask | related_control_mask
        if sex_restriction is not None and len(prepared.control_sex):
            excluded_control_mask = excluded_control_mask | (
                prepared.control_sex != sex_restriction
            )
        excluded_control_rows = control_feature_rows[excluded_control_mask]
        # Only the count of eligible input controls is needed below. Creating
        # ``control_feature_rows[~mask]`` copied roughly 200k row indices for
        # every phenotype (and several such copies concurrently); subtracting
        # the excluded count is exactly equivalent and allocation-free.
        n_control_input = len(control_feature_rows) - int(
            np.count_nonzero(excluded_control_mask)
        )
        if progress and phenotype_index % 100 == 0:
            progress(
                f"PheWAS Train {phenotype_index:,}/{len(grouped):,}: "
                f"{phenotype_id}"
            )
        for feature_id in feature_ids:
            values = numeric[feature_id]
            case_values = values[case_rows]
            case_values = case_values[np.isfinite(case_values)]
            global_distribution = global_control_distributions[feature_id]
            excluded_values = values[excluded_control_rows]
            excluded_values = excluded_values[np.isfinite(excluded_values)]
            n_case = len(case_values)
            n_control = len(global_distribution.sorted_values) - len(
                excluded_values
            )
            raw_auc, p_value = excluded_mann_whitney_auc(
                case_values,
                global_distribution,
                excluded_values,
            )
            direction = 1 if raw_auc >= 0.5 else -1
            assessability = screen_marker_assessability_from_counts(
                n_positive_input=len(case_rows),
                n_negative_input=n_control_input,
                n_positive_evaluable=n_case,
                n_negative_evaluable=n_control,
                raw_auc=raw_auc,
                direction=direction,
            )
            status = inference_status(
                n_case,
                n_control,
                minimum_inference=min_inference,
                minimum_primary=min_formal,
            )
            records.append(
                {
                    "phenotype_id": phenotype_id,
                    "phenotype_name": definition["display_name"],
                    "body_system": body_system,
                    "panel_id": panel_map[feature_id],
                    "feature_id": feature_id,
                    "evaluation_split": "train",
                    "analysis_type": (
                        "retrospective_index_admission_phewas"
                    ),
                    "n_case": n_case,
                    "n_control": n_control,
                    "case_missingness": (
                        1.0 - n_case / len(case_rows) if len(case_rows) else 1.0
                    ),
                    "control_missingness": (
                        1.0 - n_control / n_control_input
                        if n_control_input
                        else 1.0
                    ),
                    "test_yield": assessability["test_yield"],
                    "positive_test_yield": assessability[
                        "positive_test_yield"
                    ],
                    "negative_test_yield": assessability[
                        "negative_test_yield"
                    ],
                    "test_yield_gap": assessability["test_yield_gap"],
                    "overall_missingness": assessability[
                        "overall_missingness"
                    ],
                    "assessability_status": assessability[
                        "assessability_status"
                    ],
                    "assessability_gate_passed": assessability[
                        "assessability_gate_passed"
                    ],
                    "missingness_auc_lower_bound": assessability[
                        "missingness_auc_lower_bound"
                    ],
                    "missingness_auc_upper_bound": assessability[
                        "missingness_auc_upper_bound"
                    ],
                    "missingness_sensitivity_status": assessability[
                        "missingness_sensitivity_status"
                    ],
                    "missingness_sensitivity_gate_passed": assessability[
                        "missingness_sensitivity_gate_passed"
                    ],
                    "confirmatory_assessability_gate_passed": assessability[
                        "confirmatory_assessability_gate_passed"
                    ],
                    "missing_data_claim_scope": (
                        "complete_case_association_with_adversarial_auc_bounds"
                    ),
                    "raw_auc": raw_auc,
                    "direction": direction,
                    "aligned_auc": max(raw_auc, 1.0 - raw_auc),
                    "rank_biserial": 2.0 * raw_auc - 1.0,
                    "p_value": p_value,
                    "p_value_method": (
                        "mann_whitney_tie_corrected_asymptotic_screen"
                    ),
                    "average_precision": np.nan,
                    "inference_status": status,
                    "label_state_policy": (
                        "shared outcome-blind index; case=phenotype coded "
                        "on or before index discharge; control=no code up to "
                        "index discharge; future diagnoses ignored"
                    ),
                    "target_kind": "retrospective_proxy",
                    "locked_test_scanned": False,
                    "sex_restriction": sex_restriction or "none",
                    "n_related_code_controls_excluded": int(
                        np.count_nonzero(related_control_mask)
                    ),
                    "control_definition": (
                        "no code up to index discharge; phecode ancestors and "
                        "descendants removed; sex-specific phenotypes within sex"
                    ),
                }
            )
    return records


def _replicate_hits(
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    *,
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int,
    progress: Optional[Callable[[str], None]],
    seed_offset: int = 0,
    replication_cache: Optional[_PhenomeReplicationCache] = None,
) -> list[dict]:
    if train_hits.empty:
        return []
    hits = list(enumerate(train_hits.itertuples(index=False), start=1))
    cache = replication_cache or _prepare_phenome_replication_cache(
        train_hits,
        features,
        current,
        history_body,
        definition_lookup,
    )

    def evaluate_hit(hit_index: int, hit) -> Optional[dict]:
        return _evaluate_hit_arrays(
            _hit_arrays(cache, hit),
            _hit_fields(hit, hit_index, seed_offset, definition_lookup),
            min_inference=min_inference,
            min_formal=min_formal,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )

    bounded_workers = max(1, min(int(worker_count), len(hits)))
    from .grant_parallel import grant_parallel_map, grant_workers

    if grant_workers(len(hits)) > 1:
        # Audit E11b: the bootstrap per hit is GIL-bound numpy. Supervised
        # runs evaluate the hits in processes; each task carries only its
        # own arrays, and results come back in train-hit order.
        evaluated = grant_parallel_map(
            _evaluate_hit_arrays_task,
            [
                (
                    _hit_arrays(cache, hit),
                    _hit_fields(hit, hit_index, seed_offset, definition_lookup),
                    int(min_inference),
                    int(min_formal),
                    int(bootstrap_iterations),
                    int(random_seed),
                )
                for hit_index, hit in hits
            ],
        )
    elif bounded_workers == 1:
        evaluated = [
            evaluate_hit(hit_index, hit)
            for hit_index, hit in hits
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=bounded_workers,
            thread_name_prefix="vcg-phewas",
        ) as executor:
            futures = [
                executor.submit(evaluate_hit, hit_index, hit)
                for hit_index, hit in hits
            ]
            # Canonical train-hit order is independent of completion order.
            evaluated = [future.result() for future in futures]

    records: list[dict] = []
    for hit_index, record in enumerate(evaluated, start=1):
        if record is not None:
            records.append(record)
        if progress and hit_index % 100 == 0:
            progress(
                f"PheWAS Validation {hit_index:,}/{len(train_hits):,}"
            )
    return records


def _hit_arrays(cache: "_PhenomeReplicationCache", hit) -> dict:
    """The arrays one replication hit reads from the shared cache."""
    phenotype_id = str(hit.phenotype_id)
    feature_id = str(hit.feature_id)
    body_system = str(hit.body_system)
    train_values, train_labels, _ = _values_from_phenome_replication_rows(
        cache.train_numeric[feature_id],
        cache.train_subjects,
        cache.train_row_index[(phenotype_id, body_system)],
    )
    validation_values, validation_labels, validation_subjects = (
        _values_from_phenome_replication_rows(
            cache.validation_numeric[feature_id],
            cache.validation_subjects,
            cache.validation_row_index[(phenotype_id, body_system)],
        )
    )
    validation_case_rows, validation_control_rows = (
        cache.validation_row_index[(phenotype_id, body_system)]
    )
    validation_selected_rows = np.concatenate(
        (validation_case_rows, validation_control_rows)
    )
    return {
        "train_values": train_values,
        "train_labels": train_labels,
        "validation_values": validation_values,
        "validation_labels": validation_labels,
        "validation_subjects": validation_subjects,
        "validation_all_values": cache.validation_numeric[feature_id][
            validation_selected_rows
        ],
        "validation_all_labels": np.concatenate(
            (
                np.ones(len(validation_case_rows), dtype=np.int8),
                np.zeros(len(validation_control_rows), dtype=np.int8),
            )
        ),
    }


def _hit_fields(hit, hit_index: int, seed_offset: int, definition_lookup) -> dict:
    phenotype_id = str(hit.phenotype_id)
    return {
        "phenotype_id": phenotype_id,
        "phenotype_name": (
            definition_lookup.loc[phenotype_id, "display_name"]
            if phenotype_id in definition_lookup.index
            else phenotype_id
        ),
        "feature_id": str(hit.feature_id),
        "body_system": str(hit.body_system),
        "panel_id": hit.panel_id,
        "q_value": float(hit.q_value),
        "aligned_auc": float(hit.aligned_auc),
        "direction": int(hit.direction),
        "seed_index": int(
            getattr(hit, "replication_seed_index", seed_offset + hit_index)
        ),
    }


def _evaluate_hit_arrays_task(
    arrays: dict,
    fields: dict,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
) -> Optional[dict]:
    return _evaluate_hit_arrays(
        arrays,
        fields,
        min_inference=min_inference,
        min_formal=min_formal,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )


def _evaluate_hit_arrays(
    arrays: dict,
    fields: dict,
    *,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
) -> Optional[dict]:
    """Validation replication of one train hit from its own arrays."""
    feature_id = fields["feature_id"]
    train_values = arrays["train_values"]
    train_labels = arrays["train_labels"]
    validation_values = arrays["validation_values"]
    validation_labels = arrays["validation_labels"]
    if (
        np.unique(train_labels).size < 2
        or np.unique(validation_labels).size < 2
    ):
        return None
    try:
        model = fit_marker_model(train_values, train_labels, feature_id)
    except ValueError:
        return None
    metrics = evaluate_marker(
        arrays["validation_all_values"],
        arrays["validation_all_labels"],
        direction=model.direction,
        thresholds=model.thresholds,
    )
    status = inference_status(
        int(metrics["n_positive"]),
        int(metrics["n_negative"]),
        minimum_inference=min_inference,
        minimum_primary=min_formal,
    )
    intervals = discovery_marker_bootstrap_intervals(
        arrays["validation_subjects"],
        validation_values,
        validation_labels,
        direction=model.direction,
        thresholds=model.thresholds,
        iterations=(
            bootstrap_iterations if status != "descriptive_only" else 0
        ),
        seed=random_seed + fields["seed_index"],
    )
    validation_raw_auc = float(
        metrics["roc_auc"]
        if model.direction > 0
        else 1.0 - float(metrics["roc_auc"])
    )
    validation_direction = 1 if validation_raw_auc >= 0.5 else -1
    return {
        "phenotype_id": fields["phenotype_id"],
        "phenotype_name": fields["phenotype_name"],
        "body_system": fields["body_system"],
        "panel_id": fields["panel_id"],
        "feature_id": feature_id,
        "evaluation_split": "validation",
        "analysis_type": (
            "retrospective_index_admission_phewas_replication"
        ),
        "train_q_value": fields["q_value"],
        "train_aligned_auc": fields["aligned_auc"],
        "train_direction": fields["direction"],
        "direction": model.direction,
        "validation_direction": validation_direction,
        "direction_consistent": fields["direction"] == validation_direction,
        "validation_raw_auc": validation_raw_auc,
        "inference_status": status,
        "target_kind": "retrospective_proxy",
        "locked_test_scanned": False,
        **metrics,
        **intervals,
    }


def _prepare_phenome_replication_cache(
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    current: pd.DataFrame,
    history_body: pd.DataFrame,
    definition_lookup: Optional[pd.DataFrame] = None,
) -> _PhenomeReplicationCache:
    train_features = features.loc[
        features["split"] == "train"
    ].reset_index(drop=True)
    validation_features = features.loc[
        features["split"] == "validation"
    ].reset_index(drop=True)
    train_current = current.loc[current["split"] == "train"]
    validation_current = current.loc[current["split"] == "validation"]
    del history_body
    target_bodies = tuple(
        sorted(
            {
                (str(hit.phenotype_id), str(hit.body_system))
                for hit in train_hits.itertuples(index=False)
            }
        )
    )
    hit_feature_ids = tuple(
        sorted(
            {
                str(hit.feature_id)
                for hit in train_hits.itertuples(index=False)
            }
        )
    )
    return _PhenomeReplicationCache(
        train_numeric={
            feature_id: pd.to_numeric(
                train_features[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            for feature_id in hit_feature_ids
        },
        validation_numeric={
            feature_id: pd.to_numeric(
                validation_features[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            for feature_id in hit_feature_ids
        },
        train_subjects=(
            train_features["subject_id"].astype(int).to_numpy()
        ),
        validation_subjects=(
            validation_features["subject_id"].astype(int).to_numpy()
        ),
        train_row_index=_phenome_replication_row_index(
            train_features,
            train_current,
            target_bodies=target_bodies,
            history_by_body={},
            definition_lookup=definition_lookup,
        ),
        validation_row_index=_phenome_replication_row_index(
            validation_features,
            validation_current,
            target_bodies=target_bodies,
            history_by_body={},
            definition_lookup=definition_lookup,
        ),
    )


def _phenome_replication_row_index(
    features: pd.DataFrame,
    current: pd.DataFrame,
    *,
    target_bodies: tuple[tuple[str, str], ...],
    history_by_body: Mapping[str, set[int]],
    definition_lookup: Optional[pd.DataFrame] = None,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    study_to_row = {
        (int(row.subject_id), int(row.study_id)): index
        for index, row in enumerate(
            features[["subject_id", "study_id"]].itertuples(index=False)
        )
    }
    subjects = features["subject_id"].astype(int).to_numpy()
    control_base = np.flatnonzero(
        features["is_control_index"].fillna(False).to_numpy(dtype=bool)
    )
    # `history_by_body` remains in the signature for checkpoint/API
    # compatibility. Future diagnoses must not influence control eligibility.
    del history_by_body

    case_rows_by_phenotype: dict[str, np.ndarray] = {}
    for phenotype_id in sorted(
        {phenotype for phenotype, _ in target_bodies}
    ):
        case_frame = current.loc[
            current["phenotype_id"].eq(phenotype_id),
            ["subject_id", "study_id"],
        ].drop_duplicates()
        case_rows_by_phenotype[phenotype_id] = np.asarray(
            [
                study_to_row[key]
                for key in (
                    (int(row.subject_id), int(row.study_id))
                    for row in case_frame.itertuples(index=False)
                )
                if key in study_to_row
            ],
            dtype=int,
        )
    # Audits X3 and X4, as in the train screen.
    sex = _normalized_sex(features)
    known = (
        tuple(str(item) for item in definition_lookup.index)
        if definition_lookup is not None
        else tuple(str(item) for item in current["phenotype_id"].unique())
    )
    subjects_by_phenotype = {
        str(key): frame["subject_id"].astype(int).to_numpy()
        for key, frame in current[["phenotype_id", "subject_id"]].groupby("phenotype_id")
    }
    output = {}
    for phenotype_id, body_system in target_bodies:
        case_rows = case_rows_by_phenotype[phenotype_id]
        restriction = (
            _phenotype_sex_restriction(definition_lookup.loc[phenotype_id])
            if definition_lookup is not None and phenotype_id in definition_lookup.index
            else None
        )
        if restriction is not None:
            case_rows = case_rows[sex[case_rows] == restriction]
        excluded_subjects = [subjects[case_rows]]
        for related in _related_phenotype_ids(phenotype_id, known):
            if related in subjects_by_phenotype:
                excluded_subjects.append(subjects_by_phenotype[related])
        control_rows = control_base[
            ~np.isin(subjects[control_base], np.concatenate(excluded_subjects))
        ]
        if restriction is not None:
            control_rows = control_rows[sex[control_rows] == restriction]
        output[(phenotype_id, body_system)] = (case_rows, control_rows)
    return output


def _values_from_phenome_replication_rows(
    feature_values: np.ndarray,
    subjects: np.ndarray,
    rows: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_rows, control_rows = rows
    selected_rows = np.concatenate((case_rows, control_rows))
    labels = np.concatenate(
        (
            np.ones(len(case_rows), dtype=np.int8),
            np.zeros(len(control_rows), dtype=np.int8),
        )
    )
    values = feature_values[selected_rows]
    valid = np.isfinite(values)
    return (
        values[valid],
        labels[valid],
        subjects[selected_rows][valid],
    )


def _case_control_values(
    feature_frame: pd.DataFrame,
    current: pd.DataFrame,
    phenotype_id: str,
    feature_id: str,
    history_subjects: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_frame = current.loc[
        current["phenotype_id"] == phenotype_id,
        ["subject_id", "study_id"],
    ].drop_duplicates()
    case_keys = {
        (int(row.subject_id), int(row.study_id))
        for row in case_frame.itertuples(index=False)
    }
    subjects = feature_frame["subject_id"].astype(int).to_numpy()
    studies = feature_frame["study_id"].astype(int).to_numpy()
    case_mask = np.fromiter(
        (
            (int(subject_id), int(study_id)) in case_keys
            for subject_id, study_id in zip(subjects, studies)
        ),
        dtype=bool,
        count=len(feature_frame),
    )
    values = pd.to_numeric(
        feature_frame[feature_id],
        errors="coerce",
    ).to_numpy(dtype=float)
    control_index = (
        feature_frame["is_control_index"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    # The case and control ECG must come from the same outcome-blind index
    # design.  The legacy `history_subjects` argument is deliberately ignored:
    # excluding on future diagnoses causes immortal-time/healthy-control bias.
    del history_subjects
    case_subjects = subjects[case_mask]
    control_mask = control_index & ~np.isin(subjects, case_subjects)
    eligible = (case_mask | control_mask) & np.isfinite(values)
    labels = case_mask[eligible].astype(np.int8)
    return values[eligible], labels, subjects[eligible]


def _auc_and_p_from_sorted_controls(
    case_values: np.ndarray,
    sorted_control: np.ndarray | SortedControlDistribution,
) -> tuple[float, float]:
    return fast_mann_whitney_auc(case_values, sorted_control)


def _optional_row_float(row: pd.Series, column: str) -> Optional[float]:
    value = row.get(column)
    if value is None:
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if np.isfinite(numeric) else None


def _promotion_candidates(
    replicated: pd.DataFrame,
    *,
    snapshot_hash: str,
    seeded_phenotype_ids: frozenset[str] = frozenset(),
) -> tuple[PromotionCandidate, ...]:
    if replicated.empty:
        return ()
    replicated = replicated.loc[
        replicated["feature_id"].map(is_candidate_biomarker)
    ].copy()
    if replicated.empty:
        return ()
    candidates = []
    for phenotype_id, group in replicated.groupby("phenotype_id", sort=True):
        best = group.sort_values(
            ["q_value", "roc_auc"],
            ascending=[True, False],
        ).iloc[0]
        seeded = str(phenotype_id) in seeded_phenotype_ids
        candidates.append(
            PromotionCandidate(
                candidate_id=(
                    f"phewas-{phenotype_id}-{str(best['feature_id'])}-"
                    f"{snapshot_hash[:8]}"
                ),
                phenotype_id=str(phenotype_id),
                feature_ids=(str(best["feature_id"]),),
                evidence_snapshot_hash=snapshot_hash,
                proposed_label_definition={
                    "label_id": f"candidate_{phenotype_id}",
                    "target_kind": "retrospective_proxy",
                    "index_policy": "protocol_required_before_promotion",
                    "source": (
                        "PhecodeX same-admission association replicated and "
                        "adjusted in validation"
                    ),
                    "status": "draft_not_confirmed",
                    "automatic_promotion": False,
                    "seeded_from_promotion": seeded,
                    **(
                        {
                            "seeded_from_promotion_caveat": (
                                "This phenotype was itself promoted from a "
                                "prior exploratory pass on overlapping "
                                "development data. A fresh train/validation "
                                "split was used again, but the phenotype "
                                "was no longer selected blind -- treat this "
                                "as a materially weaker evidence tier than "
                                "the exhaustive first-pass PhecodeX scan."
                            )
                        }
                        if seeded
                        else {}
                    ),
                },
                effect_size=_optional_row_float(best, "roc_auc"),
                effect_ci_low=_optional_row_float(best, "roc_auc_ci_low"),
                effect_ci_high=_optional_row_float(best, "roc_auc_ci_high"),
                train_q_value=_optional_row_float(best, "train_q_value"),
                validation_q_value=_optional_row_float(best, "q_value"),
            )
        )
    return tuple(candidates)


def _annotate_ecg_circularity(
    frame: pd.DataFrame,
    *,
    id_column: str,
    name_column: str,
) -> pd.DataFrame:
    """Mark hits whose target is the electrocardiogram reading itself.

    Without this the report ranks "right bundle branch block, AUC 0.924" above
    every genuine finding in the study, and a reader has no way to tell that it
    is agreement with the cardiologist who read the same trace rather than
    accuracy against an independent standard.
    """

    if frame.empty:
        for column in (
            "ecg_derived_class",
            "ecg_derived_rule",
            "ecg_circularity_claim_ceiling",
        ):
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame
    ids = (
        frame[id_column] if id_column in frame.columns else [None] * len(frame)
    )
    names = (
        frame[name_column]
        if name_column in frame.columns
        else [None] * len(frame)
    )
    for column, values in annotate_targets(ids, names).items():
        frame[column] = values
    return frame


def _phenome_diagnostics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Per-panel inflation diagnostics, reported only where they mean something.

    The genomic-control lambda is the median test statistic divided by its null
    median.  It is a calibration diagnostic **only over an unselected set of
    tests**.  The validation frame is not one: it holds exactly the hits that
    train already found significant, so its lambda measures the selection, not
    the calibration.  The last development run made that visible -- train
    lambda 7.8 to 12.9, validation lambda 24.9 to 29.1, on a validation set
    that is smaller.  A calibration statistic cannot rise because the sample
    shrank; what rose was the share of true effects among the tests counted.
    Reporting it anyway invited exactly the wrong reading.

    Validation rows therefore carry ``lambda_gc = NaN`` and a stated reason,
    rather than a number that looks like a diagnostic and is not one.
    """

    rows = []
    for split, frame in (("train", train), ("validation", validation)):
        if frame.empty or "p_value" not in frame:
            continue
        selected = split != "train"
        for panel_id, group in frame.groupby("panel_id"):
            p_values = pd.to_numeric(group["p_value"], errors="coerce")
            p_values = p_values[np.isfinite(p_values) & p_values.gt(0)]
            if p_values.empty:
                continue
            chi_square = np.square(norm.isf(p_values.to_numpy() / 2.0))
            rows.append(
                {
                    "evaluation_split": split,
                    "panel_id": panel_id,
                    "test_count": len(p_values),
                    "lambda_gc": (
                        float("nan")
                        if selected
                        else float(np.median(chi_square) / 0.4549364)
                    ),
                    "lambda_gc_status": (
                        "not_applicable_tests_preselected_on_train"
                        if selected
                        else "unselected_test_set"
                    ),
                    "minimum_p_value": float(p_values.min()),
                    "fdr_significant_count": int(
                        group.get("fdr_significant", False).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _copy_query_atomic(connection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    target_sql = _sql_path(temporary)
    with profile_query(f"phenome:{destination.name}") as profile:
        copy_result = connection.execute(
            f"COPY ({query}) TO '{target_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        ).fetchone()
        if copy_result is None:
            raise RuntimeError("DuckDB COPY returned no row count")
        atomic_replace(temporary, destination)
        profile["rows"] = int(copy_result[0])


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
