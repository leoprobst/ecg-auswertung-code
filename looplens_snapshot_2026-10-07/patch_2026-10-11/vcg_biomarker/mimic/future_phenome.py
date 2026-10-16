from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.duration.hazard_regression import PHReg

from .config import MimicProjectConfig
from .discovery_statistics import (
    add_auc_replication_status,
    discovery_marker_bootstrap_intervals,
)
from .absolute_risk import (
    ABSOLUTE_RISK_VERSION,
    EVENT_CENSORED,
    EVENT_COMPETING,
    EVENT_OF_INTEREST,
    aalen_johansen,
    kaplan_meier_complement,
)
from .adjusted_models import anchor_year_group_start
from .frame_checkpoints import (
    checkpointed_frame_chunk_path,
    materialize_frame_chunks,
)
from .run_log import profile_query
from .statistics import (
    apply_fdr_by_family,
    evaluate_marker,
    fast_mann_whitney_auc,
    fit_marker_model,
    inference_status,
    marker_assessability,
)
from .storage import (
    DuckDBCatalog,
    atomic_replace,
    atomic_write_parquet,
    content_hash,
    feature_parquet_clause,
    optional_column_select,
    parquet_columns,
    relation_columns,
)


FUTURE_PHENOME_VERSION = "2.6.0"
DEFAULT_FUTURE_HORIZONS_DAYS = (30, 90, 365)
FUTURE_TRAIN_CHUNK_SIZE = 10
FUTURE_REPLICATION_CHUNK_SIZE = 32
FUTURE_FEATURE_BLOCK_SIZE = 32
MAX_ECG_TO_DISCHARGE_LANDMARK_DAYS = 7.0
MINIMUM_SURVIVAL_MARKER_TEST_YIELD = 0.85
MAXIMUM_SURVIVAL_MARKER_YIELD_GAP = 0.10
PH_DIAGNOSTIC_ALPHA = 0.01


@dataclass(frozen=True)
class _FutureReplicationCache:
    train_numeric: Mapping[str, np.ndarray]
    validation_numeric: Mapping[str, np.ndarray]
    train_subjects: np.ndarray
    validation_subjects: np.ndarray
    state_masks: Mapping[
        tuple[str, int, str],
        Mapping[str, np.ndarray],
    ]


def run_future_phenome_discovery(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    resources,
    output_root: Path,
    horizons_days: tuple[int, ...] = DEFAULT_FUTURE_HORIZONS_DAYS,
    train_fdr_alpha: float,
    validation_fdr_alpha: float,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int = 1,
    max_phenotypes: Optional[int] = None,
    checkpoint_root: Optional[Path] = None,
    checkpoint_input_hash: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Path]:
    horizons = _validate_horizons(horizons_days)
    paths = {
        "index": output_root / "future_phenome_index_ecg.parquet",
        "events": output_root / "future_phenome_first_events.parquet",
        "summary": output_root / "future_phenome_label_summary.parquet",
        "target_family": output_root / "future_phenome_target_family.parquet",
        "train": output_root / "future_phenome_train_screening.parquet",
        "validation": (
            output_root / "future_phenome_validation_replication.parquet"
        ),
        "replicated": output_root / "future_phenome_replicated_hits.parquet",
    }
    if progress:
        progress(
            "Future-PheWAS 1/5: outcome-unabhÃƒÂ¤ngiges Index-ECG und "
            "Beobachtungsende bestimmen"
        )
    _build_future_index_and_events(
        config,
        feature_glob=feature_glob,
        resources=resources,
        index_path=paths["index"],
        event_path=paths["events"],
    )
    index = pd.read_parquet(paths["index"])
    events = pd.read_parquet(paths["events"])
    definitions = pd.read_parquet(resources.phecodex_definitions_path)
    if index.empty:
        raise RuntimeError("No development ECG is available for future PheWAS")
    if index["split"].eq("locked_test").any():
        raise RuntimeError("Locked-test rows are forbidden in future PheWAS")
    if index["subject_id"].duplicated().any():
        raise RuntimeError("Future PheWAS requires exactly one ECG per subject")

    if progress:
        progress(
            "Future-PheWAS 2/5: positive, negative, zensierte und "
            "vorerkrankte ZustÃƒÂ¤nde zÃƒÂ¤hlen"
        )
    summary, selected_targets = _future_label_summary(
        index,
        events,
        definitions,
        horizons_days=horizons,
        min_inference=min_inference,
        max_phenotypes=max_phenotypes,
    )
    atomic_write_parquet(summary, paths["summary"])
    target_family = pd.DataFrame(
        {
            "selection_order": np.arange(
                1, len(selected_targets) + 1, dtype=int
            ),
            "phenotype_id": [item[0] for item in selected_targets],
            "horizon_days": [item[1] for item in selected_targets],
            "selection_split": "train",
            "validation_outcomes_used_for_selection": False,
            "family_frozen_before_validation_marker_evaluation": True,
        }
    )
    family_hash = content_hash(
        {
            "future_phenome_version": FUTURE_PHENOME_VERSION,
            "targets": selected_targets,
            "feature_ids": feature_ids,
            "horizons_days": horizons,
            "min_inference": min_inference,
            "max_phenotypes": max_phenotypes,
        }
    )
    target_family["target_family_hash"] = family_hash
    atomic_write_parquet(target_family, paths["target_family"])
    frozen_targets = tuple(
        (str(row.phenotype_id), int(row.horizon_days))
        for row in pd.read_parquet(paths["target_family"])
        .sort_values("selection_order")
        .itertuples(index=False)
    )
    if frozen_targets != selected_targets:
        raise RuntimeError(
            "Frozen future-PheWAS target family failed round-trip verification"
        )
    if not selected_targets:
        empty = _empty_future_results()
        atomic_write_parquet(empty, paths["train"])
        atomic_write_parquet(empty, paths["validation"])
        atomic_write_parquet(empty, paths["replicated"])
        return paths

    feature_columns = _future_feature_columns(
        config,
        feature_glob=feature_glob,
    )
    available_features = tuple(
        feature_id for feature_id in feature_ids if feature_id in feature_columns
    )
    feature_index = _load_future_feature_index(
        config,
        feature_glob=feature_glob,
        index_path=paths["index"],
    )
    if not available_features:
        raise RuntimeError("No selected marker is available for future PheWAS")
    definition_lookup = definitions.set_index("phenotype_id")
    event_lookup = {
        str(phenotype_id): group.set_index("subject_id")["first_event_time"]
        for phenotype_id, group in events.groupby("phenotype_id", sort=False)
    }

    if progress:
        progress(
            f"Future-PheWAS 3/5: {len(selected_targets):,} zensierte Ziele "
            f"Ãƒâ€” {len(available_features):,} Marker im Train-Split screenen"
        )
    train = _checkpointed_future_train_screen_streaming(
        checkpoint_root,
        config=config,
        feature_glob=feature_glob,
        index_path=paths["index"],
        checkpoint_input_hash=checkpoint_input_hash,
        feature_index=feature_index,
        selected_targets=selected_targets,
        event_lookup=event_lookup,
        definition_lookup=definition_lookup,
        feature_ids=available_features,
        panel_map=panel_map,
        min_inference=min_inference,
        min_formal=min_formal,
        progress=progress,
    )
    if not train.empty:
        train = apply_fdr_by_family(
            train,
            family_columns=("panel_id", "horizon_days"),
        )
        train_hits = train.loc[
            train["q_value"].le(train_fdr_alpha)
            & train["global_fdr_significant"].fillna(False)
            & train["hierarchical_fdr_significant"].fillna(False)
            & train["confirmatory_assessability_gate_passed"].fillna(False)
            & train["adjusted_model_gate_passed"].fillna(False)
            & train["ph_assumption_gate_passed"].fillna(False)
            & train["same_day_tie_sensitivity_gate_passed"].fillna(False)
            & train["inference_status"].eq("formal_inference")
        ].copy().reset_index(drop=True)
    else:
        train_hits = train.copy()

    if progress:
        progress(
            f"Future-PheWAS 4/5: {len(train_hits):,} Train-Treffer einmalig "
            "in Validation replizieren"
        )
    validation = _checkpointed_future_replication_streaming(
        checkpoint_root,
        config=config,
        feature_glob=feature_glob,
        index_path=paths["index"],
        checkpoint_input_hash=checkpoint_input_hash,
        train_hits=train_hits,
        feature_index=feature_index,
        event_lookup=event_lookup,
        definition_lookup=definition_lookup,
        min_inference=min_inference,
        min_formal=min_formal,
        bootstrap_iterations=0,
        random_seed=random_seed,
        worker_count=worker_count,
        progress=progress,
        checkpoint_namespace="future_validation_screen",
        progress_label="Future-PheWAS Validation-Screen",
    )
    if not validation.empty:
        validation = apply_fdr_by_family(
            validation,
            family_columns=("panel_id", "horizon_days"),
        )
        validation["validation_q_value"] = validation["q_value"]
        validation = _add_survival_replication_status(
            validation,
            validation_fdr_alpha=validation_fdr_alpha,
        )
    else:
        validation["replicated"] = pd.Series(dtype=bool)
    replicated = validation.loc[
        validation.get(
            "replicated",
            pd.Series(False, index=validation.index),
        )
    ].copy()

    if progress:
        progress(
            "Future-PheWAS 5/5: getrennte Prognosehypothesen und "
            "Zensierungsmetadaten schreiben"
        )
    atomic_write_parquet(train, paths["train"])
    atomic_write_parquet(validation, paths["validation"])
    atomic_write_parquet(replicated, paths["replicated"])
    return paths


def _future_feature_blocks(
    feature_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        feature_ids[start : start + FUTURE_FEATURE_BLOCK_SIZE]
        for start in range(0, len(feature_ids), FUTURE_FEATURE_BLOCK_SIZE)
    )


def _combine_future_feature_block(
    feature_index: pd.DataFrame,
    feature_block: pd.DataFrame,
    feature_ids: tuple[str, ...],
) -> pd.DataFrame:
    keys = ["subject_id", "study_id", "split"]
    left = feature_index[keys].reset_index(drop=True)
    right = feature_block[keys].reset_index(drop=True)
    if len(left) != len(right) or not left.equals(right):
        raise RuntimeError("Future-PheWAS feature block row identity changed")
    output = feature_index.reset_index(drop=True).copy(deep=False)
    for feature_id in feature_ids:
        output[feature_id] = feature_block[feature_id].to_numpy(copy=False)
    return output


def _checkpointed_future_train_screen_streaming(
    checkpoint_root: Optional[Path],
    *,
    config: MimicProjectConfig,
    feature_glob: str,
    index_path: Path,
    checkpoint_input_hash: Optional[str],
    feature_index: pd.DataFrame,
    selected_targets: tuple[tuple[str, int], ...],
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    train_index = feature_index.loc[
        feature_index["split"].astype(str).eq("train")
    ].reset_index(drop=True)
    target_chunks = tuple(
        selected_targets[start : start + FUTURE_TRAIN_CHUNK_SIZE]
        for start in range(0, len(selected_targets), FUTURE_TRAIN_CHUNK_SIZE)
    )
    paths: list[Path] = []
    records: list[dict] = []
    completed = 0
    blocks = _future_feature_blocks(feature_ids)
    total = max(1, len(blocks) * len(target_chunks))
    for block_index, feature_block in enumerate(blocks):
        # Verify completed blocks before loading raw features. Helpers may
        # finish distant blocks while main is processing earlier ones.
        namespace = f"future_train_stream_v1/block-{block_index:04d}"
        if checkpoint_root is not None and checkpoint_input_hash is not None and all(
            (checkpoint_root / namespace / f"part-{i:06d}.json").is_file()
            for i in range(len(target_chunks))
        ):
            def forbidden():
                raise RuntimeError('Completed Future chunk disappeared')
            for i, targets in enumerate(target_chunks):
                path, resumed = checkpointed_frame_chunk_path(checkpoint_root,
                    namespace=namespace, chunk_index=i, input_hash=checkpoint_input_hash,
                    item_keys=tuple((phenotype, horizon, marker)
                        for phenotype, horizon in targets for marker in feature_block),
                    compute=forbidden)
                paths.append(path)
                completed += 1
                if progress:
                    progress(f"Future-PheWAS Train-Block {completed:,}/{total:,}: fortgesetzt")
            continue
        raw_block = _load_future_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=feature_block,
            splits=("train",),
        )
        features = _combine_future_feature_block(
            train_index,
            raw_block,
            feature_block,
        )
        for target_chunk_index, targets in enumerate(target_chunks):
            compute = lambda selected=targets, markers=feature_block: (
                _screen_future_targets(
                    features,
                    selected_targets=selected,
                    event_lookup=event_lookup,
                    definition_lookup=definition_lookup,
                    feature_ids=markers,
                    panel_map=panel_map,
                    min_inference=min_inference,
                    min_formal=min_formal,
                    progress=None,
                )
            )
            if checkpoint_root is not None and checkpoint_input_hash is not None:
                path, resumed = checkpointed_frame_chunk_path(
                    checkpoint_root,
                    namespace=(
                        "future_train_stream_v1/"
                        f"block-{block_index:04d}"
                    ),
                    chunk_index=target_chunk_index,
                    input_hash=checkpoint_input_hash,
                    item_keys=tuple(
                        (phenotype_id, horizon, feature_id)
                        for phenotype_id, horizon in targets
                        for feature_id in feature_block
                    ),
                    compute=compute,
                )
                paths.append(path)
            else:
                frame = compute()
                records.extend(frame.to_dict(orient="records"))
                resumed = False
            completed += 1
            if progress:
                progress(
                    f"Future-PheWAS Train-Block {completed:,}/{total:,}: "
                    f"{'fortgesetzt' if resumed else 'gesichert'}"
                )
        del features, raw_block
    output = (
        materialize_frame_chunks(paths)
        if paths
        else pd.DataFrame(records)
    )
    if output.empty:
        return _empty_future_results()
    target_order = {target: index for index, target in enumerate(selected_targets)}
    feature_order = {
        feature_id: index for index, feature_id in enumerate(feature_ids)
    }
    output["_target_order"] = [
        target_order[(str(row.phenotype_id), int(row.horizon_days))]
        for row in output.itertuples(index=False)
    ]
    output["_feature_order"] = output["feature_id"].map(feature_order)
    return output.sort_values(
        ["_target_order", "_feature_order"], kind="stable"
    ).drop(columns=["_target_order", "_feature_order"]).reset_index(drop=True)


def _checkpointed_future_replication_streaming(
    checkpoint_root: Optional[Path],
    *,
    config: MimicProjectConfig,
    feature_glob: str,
    index_path: Path,
    checkpoint_input_hash: Optional[str],
    train_hits: pd.DataFrame,
    feature_index: pd.DataFrame,
    event_lookup: Mapping[str, pd.Series],
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
        return _empty_future_results()
    ordered_features = tuple(
        dict.fromkeys(train_hits["feature_id"].astype(str).tolist())
    )
    paths: list[Path] = []
    records: list[dict] = []
    completed = 0
    shared_state_masks = _prepare_future_replication_state_masks(
        train_hits,
        feature_index,
        event_lookup=event_lookup,
        definition_lookup=definition_lookup,
    )
    for block_index, feature_block in enumerate(
        _future_feature_blocks(ordered_features)
    ):
        selected_hits = train_hits.loc[
            train_hits["feature_id"].astype(str).isin(feature_block)
        ].copy()
        raw_block = _load_future_feature_block(
            config,
            feature_glob=feature_glob,
            index_path=index_path,
            feature_ids=feature_block,
        )
        features = _combine_future_feature_block(
            feature_index,
            raw_block,
            feature_block,
        )
        cache = _prepare_future_replication_cache(
            selected_hits,
            features,
            event_lookup=event_lookup,
            definition_lookup=definition_lookup,
            state_masks=shared_state_masks,
        )
        for chunk_index, start in enumerate(
            range(0, len(selected_hits), FUTURE_REPLICATION_CHUNK_SIZE)
        ):
            chunk = selected_hits.iloc[
                start : start + FUTURE_REPLICATION_CHUNK_SIZE
            ].copy()
            keys = tuple(
                (
                    str(row.phenotype_id),
                    int(row.horizon_days),
                    str(row.feature_id),
                )
                for row in chunk.itertuples(index=False)
            )
            compute = lambda selected=chunk: _replicate_future_hits(
                selected,
                features,
                event_lookup=event_lookup,
                definition_lookup=definition_lookup,
                min_inference=min_inference,
                min_formal=min_formal,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed,
                worker_count=worker_count,
                progress=None,
                replication_cache=cache,
            )
            if checkpoint_root is not None and checkpoint_input_hash is not None:
                path, resumed = checkpointed_frame_chunk_path(
                    checkpoint_root,
                    namespace=(
                        f"{checkpoint_namespace}_stream_v1/"
                        f"block-{block_index:04d}"
                    ),
                    chunk_index=chunk_index,
                    input_hash=checkpoint_input_hash,
                    item_keys=keys,
                    compute=compute,
                )
                paths.append(path)
            else:
                frame = compute()
                records.extend(frame.to_dict(orient="records"))
                resumed = False
            completed += len(chunk)
            if progress:
                progress(
                    f"{progress_label} {completed:,}/{len(train_hits):,}: "
                    f"{'fortgesetzt' if resumed else 'gesichert'}"
                )
        del cache, features, raw_block
    output = (
        materialize_frame_chunks(paths)
        if paths
        else pd.DataFrame(records)
    )
    if output.empty:
        return _empty_future_results()
    order = {
        (
            str(row.phenotype_id),
            int(row.horizon_days),
            str(row.feature_id),
        ): index
        for index, row in enumerate(train_hits.itertuples(index=False))
    }
    output["_replication_order"] = [
        order[(str(row.phenotype_id), int(row.horizon_days), str(row.feature_id))]
        for row in output.itertuples(index=False)
    ]
    return output.sort_values("_replication_order", kind="stable").drop(
        columns="_replication_order"
    ).reset_index(drop=True)


def _checkpointed_future_train_screen(
    checkpoint_root: Optional[Path],
    *,
    checkpoint_input_hash: Optional[str],
    features: pd.DataFrame,
    selected_targets: tuple[tuple[str, int], ...],
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    if checkpoint_root is None or checkpoint_input_hash is None:
        return _screen_future_targets(
            features,
            selected_targets=selected_targets,
            event_lookup=event_lookup,
            definition_lookup=definition_lookup,
            feature_ids=feature_ids,
            panel_map=panel_map,
            min_inference=min_inference,
            min_formal=min_formal,
            progress=progress,
        )
    paths: list[Path] = []
    completed = 0
    for chunk_index, start in enumerate(
        range(0, len(selected_targets), FUTURE_TRAIN_CHUNK_SIZE)
    ):
        chunk = selected_targets[start:start + FUTURE_TRAIN_CHUNK_SIZE]
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace="train_screen",
            chunk_index=chunk_index,
            input_hash=checkpoint_input_hash,
            item_keys=chunk,
            compute=lambda targets=chunk: _screen_future_targets(
                features,
                selected_targets=targets,
                event_lookup=event_lookup,
                definition_lookup=definition_lookup,
                feature_ids=feature_ids,
                panel_map=panel_map,
                min_inference=min_inference,
                min_formal=min_formal,
                progress=None,
            ),
        )
        paths.append(path)
        completed += len(chunk)
        if progress:
            progress(
                f"Future-PheWAS Train {completed:,}/"
                f"{len(selected_targets):,}: "
                f"{'fortgesetzt' if resumed else 'gesichert'}"
            )
    output = materialize_frame_chunks(paths)
    return output if not output.empty else _empty_future_results()


def _checkpointed_future_replication(
    checkpoint_root: Optional[Path],
    *,
    checkpoint_input_hash: Optional[str],
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int,
    progress: Optional[Callable[[str], None]],
    replication_cache: Optional[_FutureReplicationCache] = None,
    checkpoint_namespace: str = "validation_replication",
    progress_label: str = "Future-PheWAS Validation",
) -> pd.DataFrame:
    if train_hits.empty:
        return _empty_future_results()
    cache = replication_cache or _prepare_future_replication_cache(
        train_hits,
        features,
        event_lookup=event_lookup,
        definition_lookup=definition_lookup,
    )
    if checkpoint_root is None or checkpoint_input_hash is None:
        return _replicate_future_hits(
            train_hits,
            features,
            event_lookup=event_lookup,
            definition_lookup=definition_lookup,
            min_inference=min_inference,
            min_formal=min_formal,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
            worker_count=worker_count,
            progress=progress,
            replication_cache=cache,
        )
    paths: list[Path] = []
    completed = 0
    for chunk_index, start in enumerate(
        range(0, len(train_hits), FUTURE_REPLICATION_CHUNK_SIZE)
    ):
        chunk = train_hits.iloc[
            start:start + FUTURE_REPLICATION_CHUNK_SIZE
        ].copy()
        keys = tuple(
            (
                str(row.phenotype_id),
                int(row.horizon_days),
                str(row.feature_id),
            )
            for row in chunk.itertuples(index=False)
        )
        path, resumed = checkpointed_frame_chunk_path(
            checkpoint_root,
            namespace=checkpoint_namespace,
            chunk_index=chunk_index,
            input_hash=checkpoint_input_hash,
            item_keys=keys,
            compute=lambda selected=chunk: _replicate_future_hits(
                selected,
                features,
                event_lookup=event_lookup,
                definition_lookup=definition_lookup,
                min_inference=min_inference,
                min_formal=min_formal,
                bootstrap_iterations=bootstrap_iterations,
                random_seed=random_seed,
                worker_count=worker_count,
                progress=None,
                replication_cache=cache,
            ),
        )
        paths.append(path)
        completed += len(chunk)
        if progress:
            progress(
                f"{progress_label} {completed:,}/"
                f"{len(train_hits):,}: "
                f"{'fortgesetzt' if resumed else 'gesichert'}"
            )
    output = materialize_frame_chunks(paths)
    return output if not output.empty else _empty_future_results()


def _future_replicated_bootstrap_hits(
    train_hits: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    if train_hits.empty or validation.empty:
        return train_hits.iloc[0:0].copy()
    keys = ("phenotype_id", "horizon_days", "feature_id", "panel_id")
    replicated_keys = {
        tuple(str(getattr(row, key)) for key in keys)
        for row in validation.loc[
            validation["replicated"].fillna(False)
        ].itertuples(index=False)
    }
    selected = [
        tuple(str(getattr(row, key)) for key in keys) in replicated_keys
        for row in train_hits.itertuples(index=False)
    ]
    return train_hits.loc[selected].copy()


def _overlay_future_bootstraps(
    validation: pd.DataFrame,
    bootstrapped: pd.DataFrame,
) -> pd.DataFrame:
    output = validation.copy()
    output["bootstrap_scope"] = np.where(
        output["replicated"].fillna(False),
        "subject_cluster_replicated_hit",
        "not_requested_not_replicated",
    )
    if bootstrapped.empty:
        return output
    keys = ["phenotype_id", "horizon_days", "feature_id", "panel_id"]
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
    merged = output.merge(
        updates.assign(_bootstrap_matched=True),
        on=keys,
        how="left",
        suffixes=("", "__bootstrap"),
        validate="one_to_one",
    )
    matched = merged["_bootstrap_matched"].eq(True)
    for column in bootstrap_columns:
        update_column = f"{column}__bootstrap"
        if update_column in merged:
            merged.loc[matched, column] = merged.loc[
                matched,
                update_column,
            ].to_numpy()
            merged = merged.drop(columns=update_column)
    return merged.drop(columns="_bootstrap_matched")


def _build_future_index_and_events(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    resources,
    index_path: Path,
    event_path: Path,
) -> None:
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    admissions = _sql_path(config.standardized_root / "admissions.parquet")
    patients = _sql_path(config.standardized_root / "patients.parquet")
    diagnoses = _sql_path(config.standardized_root / "diagnoses_icd.parquet")
    mapping = _sql_path(resources.phecodex_mapping_path)
    definitions = _sql_path(resources.phecodex_definitions_path)
    features = feature_glob.replace("\\", "/").replace("'", "''")
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=features,
        )
        # Covariates required by the current contract; a master index from an
        # older stage version yields typed NULLs instead of a binder error.
        master_columns = relation_columns(
            connection,
            f"read_parquet('{master}')",
        )
        optional_master_columns = ",\n                    ".join(
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
            )
        )
        _copy_query_atomic(
            connection,
            f"""
            WITH feature_identity AS (
                SELECT DISTINCT subject_id, study_id, split
                FROM {read_features}
                WHERE split IN ('train', 'validation')
            ),
            candidates AS (
                SELECT
                    m.subject_id,
                    m.study_id,
                    m.split,
                    m.ecg_time,
                    m.hadm_id,
                    a.admittime,
                    a.dischtime,
                    m.age,
                    m.sex,
                    {optional_master_columns},
                    m.ed_stay_id,
                    DATE_DIFF('second', m.ecg_time, a.dischtime)
                        / 86400.0 AS ecg_to_landmark_days,
                    DATE_DIFF('second', a.admittime, a.dischtime)
                        / 86400.0 AS index_length_of_stay_days,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.subject_id
                        ORDER BY m.ecg_time, m.study_id
                    ) AS index_rank
                FROM read_parquet('{master}') m
                JOIN feature_identity f
                  USING (subject_id, study_id, split)
                JOIN read_parquet('{admissions}') a
                  USING (subject_id, hadm_id)
                WHERE m.split IN ('train', 'validation')
                  AND m.ecg_time IS NOT NULL
                  AND a.admittime IS NOT NULL
                  AND a.dischtime IS NOT NULL
                  AND a.dischtime > a.admittime
                  AND m.ecg_time >= a.admittime
                  AND m.ecg_time <= a.dischtime
                  AND DATE_DIFF('second', m.ecg_time, a.dischtime)
                        / 86400.0 <= {MAX_ECG_TO_DISCHARGE_LANDMARK_DAYS}
            ),
            ecg_observation AS (
                SELECT subject_id, MAX(ecg_time) AS last_ecg_time
                FROM read_parquet('{master}')
                GROUP BY subject_id
            ),
            admission_observation AS (
                SELECT
                    subject_id,
                    MAX(COALESCE(dischtime, admittime)) AS last_admission_time
                FROM read_parquet('{admissions}')
                GROUP BY subject_id
            )
            SELECT
                c.subject_id,
                c.study_id,
                c.split,
                c.ecg_time,
                c.hadm_id,
                c.admittime,
                c.dischtime,
                c.age,
                c.sex,
                c.anchor_year_group,
                c.machine_heart_rate_bpm,
                c.ed_stay_id,
                c.ecg_to_landmark_days,
                c.index_length_of_stay_days,
                c.dischtime AS index_end_time,
                'index_admission_discharge_landmark' AS landmark_time_basis,
                TRUE AS alive_and_event_free_landmark_required,
                -- Audit T2: diagnoses are observed only at contacts. The date
                -- of death used to extend this window, so a patient last seen
                -- on day 10 who died on day 400 counted as diagnosis-free
                -- through day 365. Death stays a competing event below.
                GREATEST(
                    eo.last_ecg_time,
                    ao.last_admission_time
                ) AS last_observed_time,
                p.date_of_death
            FROM candidates c
            LEFT JOIN ecg_observation eo USING (subject_id)
            LEFT JOIN admission_observation ao USING (subject_id)
            LEFT JOIN read_parquet('{patients}') p USING (subject_id)
            WHERE c.index_rank = 1
              AND (
                  p.date_of_death IS NULL
                  OR TRY_CAST(p.date_of_death AS DATE)
                        > CAST(c.dischtime AS DATE)
              )
            ORDER BY c.split, c.subject_id
            """,
            index_path,
        )
        index_sql = _sql_path(index_path)
        _copy_query_atomic(
            connection,
            f"""
            WITH mapped AS (
                SELECT
                    dx.subject_id,
                    p.phenotype_id,
                    COALESCE(a.dischtime, a.admittime) AS event_time,
                    CASE WHEN a.dischtime IS NOT NULL
                        THEN 'discharge_documentation_time'
                        ELSE 'admission_time_fallback'
                    END AS event_time_basis
                FROM read_parquet('{diagnoses}') dx
                JOIN read_parquet('{mapping}') p
                  ON dx.icd_version = p.icd_version
                 AND dx.icd_code = p.icd_code
                JOIN read_parquet('{admissions}') a
                  USING (subject_id, hadm_id)
                WHERE a.admittime IS NOT NULL
            )
            SELECT
                i.subject_id,
                i.split,
                m.phenotype_id,
                ANY_VALUE(d.display_name) AS phenotype_name,
                ANY_VALUE(d.body_system) AS body_system,
                MIN(m.event_time) AS first_event_time,
                ARG_MIN(m.event_time_basis, m.event_time)
                    AS first_event_time_basis
            FROM mapped m
            JOIN read_parquet('{index_sql}') i USING (subject_id)
            JOIN read_parquet('{definitions}') d USING (phenotype_id)
            WHERE i.split IN ('train', 'validation')
            GROUP BY i.subject_id, i.split, m.phenotype_id
            ORDER BY m.phenotype_id, i.split, i.subject_id
            """,
            event_path,
        )


def _future_label_summary(
    index: pd.DataFrame,
    events: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    horizons_days: tuple[int, ...],
    min_inference: int,
    max_phenotypes: Optional[int],
) -> tuple[pd.DataFrame, tuple[tuple[str, int], ...]]:
    definition_lookup = definitions.set_index("phenotype_id")
    records = []
    for phenotype_id, event_frame in events.groupby("phenotype_id", sort=True):
        phenotype_id = str(phenotype_id)
        if phenotype_id not in definition_lookup.index:
            continue
        definition = definition_lookup.loc[phenotype_id]
        event_series = event_frame.set_index("subject_id")["first_event_time"]
        for horizon in horizons_days:
            masks = _future_state_masks(
                index,
                event_series,
                definition,
                horizon_days=horizon,
            )
            record = {
                "phenotype_id": phenotype_id,
                "phenotype_name": str(definition["display_name"]),
                "body_system": str(definition["body_system"]),
                "horizon_days": int(horizon),
                "target_kind": "post_discharge_prognostic_association",
                "index_policy": (
                    "first feature-planned ECG within seven days before a "
                    "completed index-admission discharge landmark, selected "
                    "without outcome or extraction-success information"
                ),
                "positive_rule": (
                    "first phenotype-coded later admission is documented at "
                    f"discharge after the discharge landmark and within {horizon} days"
                ),
                "negative_rule": (
                    "not a binary control definition; all event-free person-"
                    "time contributes until censoring, competing death, or "
                    f"the {horizon}-day horizon"
                ),
                "censoring_rule": (
                    "right censor at last observation or horizon; death before "
                    "phenotype is an explicit competing event and cause-"
                    "specific censoring time"
                ),
                "same_encounter_diagnosis_excluded": True,
                "time_origin": "index_admission_discharge_landmark",
                "ecg_to_landmark_max_days": (
                    MAX_ECG_TO_DISCHARGE_LANDMARK_DAYS
                ),
                "same_day_death_event_primary_policy": "exclude_ambiguous_tie",
                "locked_test_scanned": False,
            }
            for split in ("train", "validation"):
                split_mask = index["split"].eq(split).to_numpy()
                record[f"{split}_positive_count"] = int(
                    np.sum(masks["positive"] & split_mask)
                )
                record[f"{split}_negative_count"] = int(
                    np.sum(masks["negative"] & split_mask)
                )
                record[f"{split}_unavailable_count"] = int(
                    np.sum(masks["unavailable"] & split_mask)
                )
                record[f"{split}_excluded_history_count"] = int(
                    np.sum(masks["excluded_history"] & split_mask)
                )
                record[f"{split}_excluded_sex_count"] = int(
                    np.sum(masks["excluded_sex"] & split_mask)
                )
                record[f"{split}_excluded_same_day_tie_count"] = int(
                    np.sum(masks["excluded_time_tie"] & split_mask)
                )
                record[f"{split}_at_risk_count"] = int(
                    np.sum(masks["at_risk"] & split_mask)
                )
                record[f"{split}_event_count"] = int(
                    np.sum(masks["event_observed"] & split_mask)
                )
                record[f"{split}_censored_count"] = int(
                    np.sum(masks["censored"] & split_mask)
                )
                record[f"{split}_competing_death_count"] = int(
                    np.sum(masks["competing_death"] & split_mask)
                )
            record["eligible_for_screen"] = (
                record["train_event_count"] >= min_inference
                and (
                    record["train_at_risk_count"]
                    - record["train_event_count"]
                ) >= min_inference
            )
            record["selection_split"] = "train"
            record["validation_outcomes_used_for_selection"] = False
            record["validation_assessable_after_freeze"] = (
                record["validation_event_count"] >= min_inference
                and (
                    record["validation_at_risk_count"]
                    - record["validation_event_count"]
                ) >= min_inference
            )
            records.append(record)
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary, ()
    eligible = summary.loc[summary["eligible_for_screen"]].copy()
    if max_phenotypes is not None and not eligible.empty:
        ranked_ids = (
            eligible.assign(
                _train_cases=eligible["train_event_count"]
            )
            .sort_values(
                ["_train_cases", "phenotype_id"],
                ascending=[False, True],
            )["phenotype_id"]
            .drop_duplicates()
            .head(int(max_phenotypes))
        )
        eligible = eligible.loc[
            eligible["phenotype_id"].isin(set(ranked_ids))
        ]
    selected = tuple(
        (str(row.phenotype_id), int(row.horizon_days))
        for row in eligible.sort_values(
            ["phenotype_id", "horizon_days"]
        ).itertuples(index=False)
    )
    return summary, selected


def _future_state_masks(
    frame: pd.DataFrame,
    event_series: pd.Series,
    definition: pd.Series,
    *,
    horizon_days: int,
    same_day_tie_policy: str = "exclude",
) -> dict[str, np.ndarray]:
    if same_day_tie_policy not in {"exclude", "event_first", "death_first"}:
        raise ValueError(f"Invalid same-day tie policy: {same_day_tie_policy}")
    subjects = frame["subject_id"].astype(int)
    index_end = pd.to_datetime(frame["index_end_time"], errors="coerce")
    last_observed = pd.to_datetime(
        frame["last_observed_time"],
        errors="coerce",
    )
    event_time = pd.to_datetime(
        subjects.map(event_series),
        errors="coerce",
    )
    death_source = (
        frame["date_of_death"]
        if "date_of_death" in frame
        else pd.Series(pd.NaT, index=frame.index)
    )
    death_time = pd.to_datetime(death_source, errors="coerce")
    event_delta = (
        event_time.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    followup_delta = (
        last_observed.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    death_delta = (
        death_time.reset_index(drop=True)
        - index_end.reset_index(drop=True)
    ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
    sex = frame.get("sex", pd.Series("", index=frame.index)).astype(str).str.upper()
    male_only = _source_bool(definition.get("male_only", False))
    female_only = _source_bool(definition.get("female_only", False))
    excluded_sex = np.zeros(len(frame), dtype=bool)
    if male_only:
        excluded_sex = ~sex.isin(("M", "MALE")).to_numpy()
    elif female_only:
        excluded_sex = ~sex.isin(("F", "FEMALE")).to_numpy()
    has_event = np.isfinite(event_delta)
    excluded_history = has_event & (event_delta <= 0)
    event_in_horizon = (
        ~excluded_sex
        & ~excluded_history
        & has_event
        & (event_delta > 0)
        & (event_delta <= float(horizon_days))
    )
    death_in_horizon = (
        ~excluded_sex
        & ~excluded_history
        & np.isfinite(death_delta)
        & (death_delta > 0)
        & (death_delta <= float(horizon_days))
    )
    event_calendar_day = event_time.reset_index(drop=True).dt.normalize()
    death_calendar_day = death_time.reset_index(drop=True).dt.normalize()
    ambiguous_same_day_death_event = (
        event_in_horizon
        & death_in_horizon
        & event_calendar_day.eq(death_calendar_day).to_numpy(dtype=bool)
    )
    excluded_time_tie = (
        ambiguous_same_day_death_event
        if same_day_tie_policy == "exclude"
        else np.zeros(len(frame), dtype=bool)
    )
    # For cause-specific hazards, a death before the phenotype event is a
    # competing event and censors observation at death. It is counted and
    # reported explicitly; it is not silently treated as an event-free
    # survivor.
    competing_death = death_in_horizon & ~ambiguous_same_day_death_event & (
        ~event_in_horizon | (death_delta < event_delta)
    )
    if same_day_tie_policy == "death_first":
        competing_death |= ambiguous_same_day_death_event
    positive = (
        event_in_horizon
        & ~competing_death
        & ~excluded_time_tie
    )
    survived_horizon = (
        ~np.isfinite(death_delta)
        | (death_delta >= float(horizon_days))
    )
    complete_followup = (
        np.isfinite(followup_delta)
        & (followup_delta >= float(horizon_days))
        & survived_horizon
    )
    no_event_in_horizon = ~has_event | (event_delta > float(horizon_days))
    negative = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & ~positive
        & no_event_in_horizon
        & complete_followup
    )
    unavailable = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & ~positive
        & ~negative
    )
    administrative_followup = np.minimum(
        np.maximum(followup_delta, 0.0),
        float(horizon_days),
    )
    # Censored rows run to the administrative horizon or to the last observed
    # contact, whichever comes first. A recorded event or death instead sets
    # the exact exit time: both are bounded by the horizon upstream, and the
    # event itself proves observation until it happened even when a
    # source-specific observation-end field is incomplete.
    duration_days = np.where(
        positive,
        event_delta,
        np.where(competing_death, death_delta, administrative_followup),
    )
    at_risk = (
        ~excluded_sex
        & ~excluded_history
        & ~excluded_time_tie
        & np.isfinite(duration_days)
        & (duration_days > 0)
    )
    censored = at_risk & ~positive & ~competing_death
    return {
        "positive": positive,
        "negative": negative,
        "unavailable": unavailable,
        "excluded_history": excluded_history & ~excluded_sex,
        "excluded_sex": excluded_sex,
        "excluded_time_tie": excluded_time_tie,
        "ambiguous_same_day_death_event": ambiguous_same_day_death_event,
        "same_day_tie_policy": same_day_tie_policy,
        "at_risk": at_risk,
        "duration_days": duration_days,
        "event_observed": positive,
        "competing_death": competing_death,
        "censored": censored,
    }


def _load_future_features(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    feature_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    available = _future_feature_columns(config, feature_glob=feature_glob)
    selected = tuple(item for item in feature_ids if item in available)
    if not selected:
        return pd.DataFrame(), ()
    feature_index = _load_future_feature_index(
        config,
        feature_glob=feature_glob,
        index_path=index_path,
    )
    feature_block = _load_future_feature_block(
        config,
        feature_glob=feature_glob,
        index_path=index_path,
        feature_ids=selected,
    )
    return (
        _combine_future_feature_block(feature_index, feature_block, selected),
        selected,
    )


def _future_feature_columns(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
) -> frozenset[str]:
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
        return frozenset(parquet_columns(connection, feature_sql))


def _load_future_feature_index(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
) -> pd.DataFrame:
    available = _future_feature_columns(config, feature_glob=feature_glob)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    index_sql = _sql_path(index_path)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
        # heart_rate_bpm is an ecg_context_comparator, never a candidate,
        # so it is not in feature_ids -- but the Cox adjustment needs it.
        # Pulled in separately, kept out of the screened-marker tuple.
        heart_rate_select = (
            ",\n                f.heart_rate_bpm"
            if "heart_rate_bpm" in available
            else ""
        )
        frame = connection.execute(
            f"""
            SELECT
                i.subject_id,
                i.study_id,
                i.split,
                i.ecg_time,
                i.index_end_time,
                i.last_observed_time,
                i.date_of_death,
                i.age,
                i.sex,
                i.anchor_year_group,
                i.machine_heart_rate_bpm,
                i.ed_stay_id,
                i.ecg_to_landmark_days,
                i.index_length_of_stay_days,
                f.extraction_status,
                f.signal_quality_score
                {heart_rate_select}
            FROM read_parquet('{index_sql}') i
            JOIN {read_features} f USING (subject_id, study_id, split)
            WHERE i.split IN ('train', 'validation')
            ORDER BY i.split, i.subject_id, i.study_id
            """
        ).fetchdf()
    if frame["subject_id"].duplicated().any():
        raise RuntimeError("Future feature frame contains repeated subjects")
    return frame


def _load_future_feature_block(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    index_path: Path,
    feature_ids: tuple[str, ...],
    splits: tuple[str, ...] = ("train", "validation"),
) -> pd.DataFrame:
    if not feature_ids:
        raise ValueError("Future-PheWAS feature block cannot be empty")
    if not splits or any(split not in {"train", "validation"} for split in splits):
        raise ValueError(f"Unsupported future-PheWAS splits: {splits!r}")
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    index_sql = _sql_path(index_path)
    columns = ",\n                ".join(
        f"f.{_quote_identifier(item)}" for item in feature_ids
    )
    split_sql = ", ".join(f"'{split}'" for split in splits)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
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


def _screen_future_targets(
    features: pd.DataFrame,
    *,
    selected_targets: tuple[tuple[str, int], ...],
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    feature_ids: tuple[str, ...],
    panel_map: Mapping[str, str],
    min_inference: int,
    min_formal: int,
    progress: Optional[Callable[[str], None]],
) -> pd.DataFrame:
    records = []
    numeric = {
        feature_id: pd.to_numeric(
            features[feature_id],
            errors="coerce",
        ).to_numpy(dtype=float)
        for feature_id in feature_ids
    }
    for target_index, (phenotype_id, horizon) in enumerate(
        selected_targets,
        start=1,
    ):
        definition = definition_lookup.loc[phenotype_id]
        masks = _future_state_masks(
            features,
            event_lookup.get(phenotype_id, pd.Series(dtype="datetime64[ns]")),
            definition,
            horizon_days=horizon,
        )
        case_rows = np.flatnonzero(masks["positive"])
        control_rows = np.flatnonzero(masks["negative"])
        at_risk_rows = np.flatnonzero(masks["at_risk"])
        adjustments_by_rows = {}
        all_adjustment = _future_adjustment_covariates(features, at_risk_rows)
        for feature_id in feature_ids:
            values = numeric[feature_id]
            marker_assessment = _survival_marker_assessability(values, masks)
            case_values = values[case_rows]
            case_values = case_values[np.isfinite(case_values)]
            control_values = values[control_rows]
            control_values = control_values[np.isfinite(control_values)]
            raw_auc, _ = _auc_and_p(
                case_values,
                control_values,
            )
            valid_at_risk = at_risk_rows[
                np.isfinite(values[at_risk_rows])
            ]
            rows_key = valid_at_risk.tobytes()
            if rows_key not in adjustments_by_rows:
                adjustments_by_rows[rows_key] = _future_adjustment_covariates(features, valid_at_risk)
            adjustment = adjustments_by_rows[rows_key]
            survival = _fit_cause_specific_cox(
                values[valid_at_risk],
                np.asarray(masks["duration_days"])[valid_at_risk],
                np.asarray(masks["event_observed"])[valid_at_risk],
                adjustment,
            )
            direction = int(survival["direction"])
            missingness_sensitivity = _adverse_missing_marker_sensitivity(
                values[at_risk_rows],
                np.asarray(masks["duration_days"])[at_risk_rows],
                np.asarray(masks["event_observed"])[at_risk_rows],
                all_adjustment,
                direction=direction,
            )
            tie_sensitivity = _same_day_tie_sensitivity(
                features,
                values,
                event_lookup.get(
                    phenotype_id, pd.Series(dtype="datetime64[ns]")
                ),
                definition,
                horizon_days=horizon,
                primary_direction=direction,
                primary_masks=masks,
            )
            n_event = int(survival["n_event"])
            n_at_risk = int(survival["n_at_risk"])
            # Absolute risk beside the hazard ratio: a clinician needs the
            # risk difference, and the decision-curve stage of the evidence
            # ladder needs a level, not a ratio.
            absolute = _absolute_risk_block(
                duration_days=np.asarray(masks["duration_days"])[valid_at_risk],
                event_observed=np.asarray(masks["event_observed"])[
                    valid_at_risk
                ],
                competing_death=np.asarray(masks["competing_death"])[
                    valid_at_risk
                ],
                marker_values=values[valid_at_risk],
                direction=direction,
                horizon_days=float(horizon),
            )
            records.append(
                {
                    "phenotype_id": phenotype_id,
                    "phenotype_name": str(definition["display_name"]),
                    "body_system": str(definition["body_system"]),
                    "horizon_days": int(horizon),
                    "panel_id": panel_map[feature_id],
                    "feature_id": feature_id,
                    "evaluation_split": "train",
                    "analysis_type": "future_cause_specific_cox_phewas",
                    "target_kind": "post_discharge_prognostic_association",
                    "n_case": len(case_values),
                    "n_control": len(control_values),
                    "n_at_risk": n_at_risk,
                    "n_event": n_event,
                    "n_censored": int(
                        np.sum(np.asarray(masks["censored"])[valid_at_risk])
                    ),
                    "n_competing_death": int(
                        np.sum(
                            np.asarray(masks["competing_death"])[
                                valid_at_risk
                            ]
                        )
                    ),
                    "marker_missingness_at_risk": (
                        marker_assessment["overall_missingness"]
                    ),
                    "case_missingness": (
                        1.0 - len(case_values) / len(case_rows)
                        if len(case_rows)
                        else 1.0
                    ),
                    "control_missingness": (
                        1.0 - len(control_values) / len(control_rows)
                        if len(control_rows)
                        else 1.0
                    ),
                    "raw_auc": raw_auc,
                    "landmark_complete_followup_auc": raw_auc,
                    "direction": direction,
                    "aligned_auc": (
                        max(raw_auc, 1.0 - raw_auc)
                        if np.isfinite(raw_auc)
                        else np.nan
                    ),
                    "rank_biserial": (
                        2.0 * raw_auc - 1.0
                        if np.isfinite(raw_auc)
                        else np.nan
                    ),
                    **absolute,
                    "cox_log_hazard_ratio_per_sd": survival[
                        "log_hazard_ratio_per_sd"
                    ],
                    "cox_hazard_ratio_per_sd": survival[
                        "hazard_ratio_per_sd"
                    ],
                    "cox_hr_ci_low": survival["hr_ci_low"],
                    "cox_hr_ci_high": survival["hr_ci_high"],
                    "p_value": survival["p_value"],
                    "p_value_method": "adjusted_cause_specific_cox_wald",
                    "survival_model_status": survival["status"],
                    "cox_adjustment_covariates": (
                        "age|sex|heart_rate|ed_setting|acquisition_hour_cyclic|"
                        "anchor_era_start_year|"
                        "signal_quality|ecg_to_landmark_days|"
                        "index_length_of_stay_days"
                    ),
                    "cox_complete_case_test_yield": (
                        n_at_risk / len(at_risk_rows)
                        if len(at_risk_rows)
                        else 0.0
                    ),
                    "ph_diagnostic_p_value": survival[
                        "ph_diagnostic_p_value"
                    ],
                    "ph_assumption_gate_passed": survival[
                        "ph_assumption_gate_passed"
                    ],
                    "adjusted_model_gate_passed": bool(
                        survival["status"] == "evaluable"
                        and int(survival["adjustment_covariate_count"]) >= 2
                    ),
                    **marker_assessment,
                    **missingness_sensitivity,
                    **tie_sensitivity,
                    "confirmatory_assessability_gate_passed": bool(
                        marker_assessment[
                            "survival_marker_assessability_gate_passed"
                        ]
                        and missingness_sensitivity[
                            "missingness_sensitivity_gate_passed"
                        ]
                        and (
                            n_at_risk / len(at_risk_rows)
                            if len(at_risk_rows)
                            else 0.0
                        ) >= MINIMUM_SURVIVAL_MARKER_TEST_YIELD
                    ),
                    "average_precision": np.nan,
                    "inference_status": inference_status(
                        n_event,
                        max(0, n_at_risk - n_event),
                        minimum_inference=min_inference,
                        minimum_primary=min_formal,
                    ),
                    "label_state_policy": (
                        "post-discharge landmark; prior/same-discharge history "
                        "excluded; right censoring modeled; death censored "
                        "cause-specifically; same-day death/event ties excluded"
                    ),
                    "claim_scope": (
                        "adjusted_complete_case_cause_specific_association_only"
                    ),
                    "absolute_risk_claim_allowed": False,
                    "locked_test_scanned": False,
                }
            )
        if progress and target_index % 100 == 0:
            progress(
                f"Future-PheWAS Train {target_index:,}/"
                f"{len(selected_targets):,}"
            )
    return pd.DataFrame(records) if records else _empty_future_results()


def _replicate_future_hits(
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    *,
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    min_inference: int,
    min_formal: int,
    bootstrap_iterations: int,
    random_seed: int,
    worker_count: int,
    progress: Optional[Callable[[str], None]],
    replication_cache: Optional[_FutureReplicationCache] = None,
) -> pd.DataFrame:
    if train_hits.empty:
        return _empty_future_results()
    cache = replication_cache or _prepare_future_replication_cache(
        train_hits,
        features,
        event_lookup=event_lookup,
        definition_lookup=definition_lookup,
    )
    hits = list(enumerate(train_hits.itertuples(index=False), start=1))
    validation_features = features.loc[
        features["split"] == "validation"
    ].reset_index(drop=True)
    del bootstrap_iterations, random_seed

    def evaluate_hit(hit_index: int, hit) -> Optional[dict]:
        phenotype_id = str(hit.phenotype_id)
        horizon = int(hit.horizon_days)
        feature_id = str(hit.feature_id)
        definition = definition_lookup.loc[phenotype_id]
        train_masks = cache.state_masks[(phenotype_id, horizon, "train")]
        validation_masks = cache.state_masks[
            (phenotype_id, horizon, "validation")
        ]
        validation_values, validation_labels, validation_subjects = (
            _future_values_labels_from_numeric(
                cache.validation_numeric[feature_id],
                cache.validation_subjects,
                validation_masks,
            )
        )
        validation_numeric = cache.validation_numeric[feature_id]
        marker_assessment = _survival_marker_assessability(
            validation_numeric,
            validation_masks,
        )
        at_risk_rows = np.flatnonzero(validation_masks["at_risk"])
        valid_rows = at_risk_rows[
            np.isfinite(validation_numeric[at_risk_rows])
        ]
        survival = _fit_cause_specific_cox(
            validation_numeric[valid_rows],
            np.asarray(validation_masks["duration_days"])[valid_rows],
            np.asarray(validation_masks["event_observed"])[valid_rows],
            _future_adjustment_covariates(validation_features, valid_rows),
        )
        if survival["status"] != "evaluable":
            return None
        n_event = int(survival["n_event"])
        n_at_risk = int(survival["n_at_risk"])
        status = inference_status(
            n_event,
            max(0, n_at_risk - n_event),
            minimum_inference=min_inference,
            minimum_primary=min_formal,
        )
        validation_raw_auc, _ = _auc_and_p(
            validation_values[validation_labels == 1],
            validation_values[validation_labels == 0],
        )
        validation_direction = int(survival["direction"])
        train_effect = float(
            getattr(hit, "cox_log_hazard_ratio_per_sd", np.nan)
        )
        validation_effect = float(survival["log_hazard_ratio_per_sd"])
        missingness_sensitivity = _adverse_missing_marker_sensitivity(
            validation_numeric[at_risk_rows],
            np.asarray(validation_masks["duration_days"])[at_risk_rows],
            np.asarray(validation_masks["event_observed"])[at_risk_rows],
            _future_adjustment_covariates(
                validation_features,
                at_risk_rows,
            ),
            direction=int(hit.direction),
        )
        tie_sensitivity = _same_day_tie_sensitivity(
            validation_features,
            validation_numeric,
            event_lookup.get(
                phenotype_id, pd.Series(dtype="datetime64[ns]")
            ),
            definition,
            horizon_days=horizon,
            primary_direction=validation_direction,
        )
        effect_retention = (
            abs(validation_effect) / abs(train_effect)
            if np.isfinite(train_effect) and train_effect != 0
            else np.nan
        )
        return {
            "phenotype_id": phenotype_id,
            "phenotype_name": str(definition["display_name"]),
            "body_system": str(definition["body_system"]),
            "horizon_days": horizon,
            "panel_id": str(hit.panel_id),
            "feature_id": feature_id,
            "evaluation_split": "validation",
            "analysis_type": (
                "future_cause_specific_cox_phewas_replication"
            ),
            "target_kind": "post_discharge_prognostic_association",
            "train_q_value": float(hit.q_value),
            "train_aligned_auc": float(hit.aligned_auc),
            "train_direction": int(hit.direction),
            "train_cox_log_hazard_ratio_per_sd": train_effect,
            "direction": int(hit.direction),
            "validation_direction": validation_direction,
            "direction_consistent": int(hit.direction) == validation_direction,
            "validation_raw_auc": validation_raw_auc,
            "validation_aligned_auc": (
                max(validation_raw_auc, 1.0 - validation_raw_auc)
                if np.isfinite(validation_raw_auc)
                else np.nan
            ),
            "landmark_complete_followup_auc": validation_raw_auc,
            "validation_cox_log_hazard_ratio_per_sd": validation_effect,
            "validation_cox_hazard_ratio_per_sd": survival[
                "hazard_ratio_per_sd"
            ],
            "validation_cox_hr_ci_low": survival["hr_ci_low"],
            "validation_cox_hr_ci_high": survival["hr_ci_high"],
            "effect_retention_ratio": effect_retention,
            "train_validation_effect_gap": (
                abs(train_effect - validation_effect)
                if np.isfinite(train_effect)
                else np.nan
            ),
            "p_value": survival["p_value"],
            "p_value_method": "adjusted_cause_specific_cox_wald",
            "n_at_risk": n_at_risk,
            "n_event": n_event,
            "n_positive": int(np.sum(validation_labels == 1)),
            "n_negative": int(np.sum(validation_labels == 0)),
            "n_censored": int(
                np.sum(np.asarray(validation_masks["censored"])[valid_rows])
            ),
            "n_competing_death": int(
                np.sum(
                    np.asarray(validation_masks["competing_death"])[valid_rows]
                )
            ),
            "marker_missingness_at_risk": (
                marker_assessment["overall_missingness"]
            ),
            "cox_complete_case_test_yield": (
                n_at_risk / len(at_risk_rows) if len(at_risk_rows) else 0.0
            ),
            "ph_diagnostic_p_value": survival["ph_diagnostic_p_value"],
            "ph_assumption_gate_passed": survival[
                "ph_assumption_gate_passed"
            ],
            "adjusted_model_gate_passed": bool(
                survival["status"] == "evaluable"
                and int(survival["adjustment_covariate_count"]) >= 2
            ),
            **marker_assessment,
            **missingness_sensitivity,
            **tie_sensitivity,
            "confirmatory_assessability_gate_passed": bool(
                marker_assessment[
                    "survival_marker_assessability_gate_passed"
                ]
                and missingness_sensitivity[
                    "missingness_sensitivity_gate_passed"
                ]
                and (
                    n_at_risk / len(at_risk_rows)
                    if len(at_risk_rows)
                    else 0.0
                ) >= MINIMUM_SURVIVAL_MARKER_TEST_YIELD
            ),
            "claim_scope": (
                "adjusted_complete_case_cause_specific_association_only"
            ),
            "inference_status": status,
            "absolute_risk_claim_allowed": False,
            "locked_test_scanned": False,
        }

    bounded_workers = max(1, min(int(worker_count), len(hits)))
    if bounded_workers == 1:
        evaluated = [
            evaluate_hit(hit_index, hit)
            for hit_index, hit in hits
        ]
    else:
        with ThreadPoolExecutor(
            max_workers=bounded_workers,
            thread_name_prefix="vcg-future-phewas",
        ) as executor:
            futures = [
                executor.submit(evaluate_hit, hit_index, hit)
                for hit_index, hit in hits
            ]
            evaluated = [future.result() for future in futures]

    records = []
    for hit_index, record in enumerate(evaluated, start=1):
        if record is not None:
            records.append(record)
        if progress and hit_index % 100 == 0:
            progress(
                f"Future-PheWAS Validation {hit_index:,}/"
                f"{len(train_hits):,}"
            )
    return pd.DataFrame(records) if records else _empty_future_results()


def _prepare_future_replication_cache(
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    *,
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
    state_masks: Optional[
        Mapping[tuple[str, int, str], Mapping[str, np.ndarray]]
    ] = None,
) -> _FutureReplicationCache:
    train = features.loc[
        features["split"] == "train"
    ].reset_index(drop=True)
    validation = features.loc[
        features["split"] == "validation"
    ].reset_index(drop=True)
    mask_cache = dict(state_masks) if state_masks is not None else (
        _prepare_future_replication_state_masks(
            train_hits,
            features,
            event_lookup=event_lookup,
            definition_lookup=definition_lookup,
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
    return _FutureReplicationCache(
        train_numeric={
            feature_id: pd.to_numeric(
                train[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            for feature_id in hit_feature_ids
        },
        validation_numeric={
            feature_id: pd.to_numeric(
                validation[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            for feature_id in hit_feature_ids
        },
        train_subjects=train["subject_id"].astype(int).to_numpy(),
        validation_subjects=(
            validation["subject_id"].astype(int).to_numpy()
        ),
        state_masks=mask_cache,
    )


def _prepare_future_replication_state_masks(
    train_hits: pd.DataFrame,
    features: pd.DataFrame,
    *,
    event_lookup: Mapping[str, pd.Series],
    definition_lookup: pd.DataFrame,
) -> dict[tuple[str, int, str], Mapping[str, np.ndarray]]:
    train = features.loc[
        features["split"] == "train"
    ].reset_index(drop=True)
    validation = features.loc[
        features["split"] == "validation"
    ].reset_index(drop=True)
    targets = sorted(
        {
            (str(hit.phenotype_id), int(hit.horizon_days))
            for hit in train_hits.itertuples(index=False)
        }
    )
    mask_cache: dict[
        tuple[str, int, str],
        Mapping[str, np.ndarray],
    ] = {}
    for phenotype_id, horizon in targets:
        definition = definition_lookup.loc[phenotype_id]
        for split, frame in (("train", train), ("validation", validation)):
            mask_cache[(phenotype_id, horizon, split)] = (
                _future_state_masks(
                    frame,
                    event_lookup.get(
                        phenotype_id,
                        pd.Series(dtype="datetime64[ns]"),
                    ),
                    definition,
                    horizon_days=horizon,
                )
            )
    return mask_cache


def _future_values_labels(
    frame: pd.DataFrame,
    feature_id: str,
    masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = pd.to_numeric(frame[feature_id], errors="coerce").to_numpy(
        dtype=float
    )
    subjects = frame["subject_id"].astype(int).to_numpy()
    return _future_values_labels_from_numeric(values, subjects, masks)


def _future_values_labels_from_numeric(
    values: np.ndarray,
    subjects: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eligible = (
        np.asarray(masks["positive"], dtype=bool)
        | np.asarray(masks["negative"], dtype=bool)
    ) & np.isfinite(values)
    labels = np.asarray(masks["positive"], dtype=bool)[eligible].astype(
        np.int8
    )
    return values[eligible], labels, subjects[eligible]


def _auc_and_p(
    case_values: np.ndarray,
    controls: np.ndarray,
) -> tuple[float, float]:
    return fast_mann_whitney_auc(case_values, controls)


def _fit_cause_specific_cox(
    values: np.ndarray,
    duration_days: np.ndarray,
    event_observed: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Fit a marker-first cause-specific Cox model per marker SD.

    Competing deaths must already be represented as censored observations at
    their death time. This estimates a cause-specific association and does not
    estimate cumulative incidence or absolute risk.
    """

    values = np.asarray(values, dtype=float)
    duration = np.asarray(duration_days, dtype=float)
    events = np.asarray(event_observed, dtype=bool)
    if covariates is None:
        adjustment = np.empty((len(values), 0), dtype=float)
    else:
        adjustment = np.asarray(covariates, dtype=float)
        if adjustment.ndim == 1:
            adjustment = adjustment[:, None]
        if adjustment.shape[0] != len(values):
            raise ValueError("Cox covariates must match marker rows")
    valid = np.isfinite(values) & np.isfinite(duration) & (duration > 0)
    if adjustment.shape[1]:
        valid &= np.isfinite(adjustment).all(axis=1)
    values = values[valid]
    duration = duration[valid]
    events = events[valid]
    adjustment = adjustment[valid]
    base: dict[str, object] = {
        "status": "not_evaluable",
        "failure_reason": "",
        "n_at_risk": int(len(values)),
        "n_event": int(np.sum(events)),
        "direction": 0,
        "log_hazard_ratio_per_sd": np.nan,
        "hazard_ratio_per_sd": np.nan,
        "hr_ci_low": np.nan,
        "hr_ci_high": np.nan,
        "p_value": np.nan,
        "ph_diagnostic_p_value": np.nan,
        "ph_assumption_gate_passed": False,
        "adjustment_covariate_count": int(adjustment.shape[1]),
    }
    if len(values) < 3 or np.sum(events) < 2:
        base["failure_reason"] = "insufficient_events_or_person_time"
        return base
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        base["failure_reason"] = "zero_or_invalid_marker_variance"
        return base
    standardized = (values - float(np.mean(values))) / scale
    design_columns = [standardized]
    for column_index in range(adjustment.shape[1]):
        column = adjustment[:, column_index]
        unique = np.unique(column)
        if len(unique) <= 1:
            continue
        if len(unique) > 2:
            column_scale = float(np.std(column, ddof=0))
            if not np.isfinite(column_scale) or column_scale <= 0:
                base["failure_reason"] = (
                    f"invalid_adjustment_covariate_{column_index}"
                )
                return base
            column = (column - float(np.mean(column))) / column_scale
        design_columns.append(column)
    design = np.column_stack(design_columns)
    base["adjustment_covariate_count"] = int(design.shape[1] - 1)
    try:
        result = PHReg(
            duration,
            design,
            status=events.astype(np.int8),
            ties="efron",
        ).fit(disp=0)
        coefficient = float(result.params[0])
        p_value = float(result.pvalues[0])
        ci_low, ci_high = map(float, result.conf_int()[0])
        residuals = np.asarray(result.schoenfeld_residuals, dtype=float)[:, 0]
        diagnostic = events & np.isfinite(residuals)
        diagnostic_times = np.log(duration[diagnostic])
        diagnostic_residuals = residuals[diagnostic]
        if (
            int(np.sum(diagnostic)) >= 10
            and np.unique(diagnostic_times).size > 1
            and np.unique(diagnostic_residuals).size > 1
        ):
            _, ph_p_value = spearmanr(
                diagnostic_times,
                diagnostic_residuals,
            )
            ph_p_value = float(ph_p_value)
        else:
            ph_p_value = np.nan
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        base["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return base
    if not all(np.isfinite(item) for item in (coefficient, p_value, ci_low, ci_high)):
        base["failure_reason"] = "non_finite_cox_estimate"
        return base
    base.update(
        {
            "status": "evaluable",
            "direction": 1 if coefficient > 0 else -1,
            "log_hazard_ratio_per_sd": coefficient,
            "hazard_ratio_per_sd": float(np.exp(np.clip(coefficient, -700, 700))),
            "hr_ci_low": float(np.exp(np.clip(ci_low, -700, 700))),
            "hr_ci_high": float(np.exp(np.clip(ci_high, -700, 700))),
            "p_value": p_value,
            "ph_diagnostic_p_value": ph_p_value,
            "ph_assumption_gate_passed": bool(
                np.isfinite(ph_p_value) and ph_p_value >= PH_DIAGNOSTIC_ALPHA
            ),
        }
    )
    return base



def _absolute_risk_block(
    *,
    duration_days: np.ndarray,
    event_observed: np.ndarray,
    competing_death: np.ndarray,
    marker_values: np.ndarray,
    direction: int,
    horizon_days: float,
) -> dict[str, object]:
    """Absolute risk at the horizon, overall and by marker tertile.

    A hazard ratio says how much faster; it does not say how many. The
    project's evidence ladder needs absolute risk for its decision-curve
    stage, and a clinician reading a result needs the risk difference rather
    than the ratio. Competing deaths enter as competing events, so this is
    cumulative incidence and not the biased ``1 - Kaplan-Meier``.
    """

    empty = {
        "absolute_risk_at_horizon": float("nan"),
        "absolute_risk_ci_low": float("nan"),
        "absolute_risk_ci_high": float("nan"),
        "competing_death_risk_at_horizon": float("nan"),
        "naive_one_minus_km_risk": float("nan"),
        "competing_risk_bias": float("nan"),
        "absolute_risk_high_marker_tertile": float("nan"),
        "absolute_risk_low_marker_tertile": float("nan"),
        "absolute_risk_difference_tertile": float("nan"),
        "absolute_risk_estimator": "aalen_johansen",
        "absolute_risk_version": ABSOLUTE_RISK_VERSION,
    }
    durations = np.asarray(duration_days, dtype=float)
    if len(durations) == 0:
        return empty
    events = np.where(
        np.asarray(event_observed, dtype=bool),
        EVENT_OF_INTEREST,
        np.where(
            np.asarray(competing_death, dtype=bool),
            EVENT_COMPETING,
            EVENT_CENSORED,
        ),
    )
    curve = aalen_johansen(durations, events)
    low, high = curve.confidence_interval_at(horizon_days)
    naive = kaplan_meier_complement(durations, events).risk_at(horizon_days)
    risk = curve.risk_at(horizon_days)
    block = dict(empty)
    block.update(
        {
            "absolute_risk_at_horizon": risk,
            "absolute_risk_ci_low": low,
            "absolute_risk_ci_high": high,
            "competing_death_risk_at_horizon": curve.competing_risk_at(
                horizon_days
            ),
            "naive_one_minus_km_risk": naive,
            "competing_risk_bias": (
                float(naive - risk)
                if np.isfinite(naive) and np.isfinite(risk)
                else float("nan")
            ),
        }
    )
    aligned = int(direction) * np.asarray(marker_values, dtype=float)
    usable = np.isfinite(aligned)
    if usable.sum() >= 30:
        cuts = np.quantile(aligned[usable], (1 / 3, 2 / 3))
        if cuts[0] < cuts[1]:
            high_mask = usable & (aligned >= cuts[1])
            low_mask = usable & (aligned <= cuts[0])
            if high_mask.sum() >= 10 and low_mask.sum() >= 10:
                high_risk = aalen_johansen(
                    durations[high_mask],
                    events[high_mask],
                ).risk_at(horizon_days)
                low_risk = aalen_johansen(
                    durations[low_mask],
                    events[low_mask],
                ).risk_at(horizon_days)
                block.update(
                    {
                        "absolute_risk_high_marker_tertile": high_risk,
                        "absolute_risk_low_marker_tertile": low_risk,
                        "absolute_risk_difference_tertile": (
                            float(high_risk - low_risk)
                            if np.isfinite(high_risk) and np.isfinite(low_risk)
                            else float("nan")
                        ),
                    }
                )
    return block


def _future_adjustment_covariates(
    frame: pd.DataFrame,
    rows: np.ndarray,
) -> np.ndarray:
    scoped = frame.iloc[np.asarray(rows, dtype=int)]
    columns: list[np.ndarray] = []
    if "age" in scoped:
        columns.append(
            pd.to_numeric(scoped["age"], errors="coerce").to_numpy(dtype=float)
        )
    if "sex" in scoped:
        sex = scoped["sex"].astype("string").str.upper()
        columns.append(
            np.where(
                sex.isin(("M", "MALE")),
                1.0,
                np.where(sex.isin(("F", "FEMALE")), 0.0, np.nan),
            )
        )
    # Heart rate: strong non-specific outcome predictor and mechanically
    # correlated with most rate-dependent markers.
    rate = pd.to_numeric(
        scoped.get("heart_rate_bpm", pd.Series(np.nan, index=scoped.index)),
        errors="coerce",
    ).combine_first(
        pd.to_numeric(
            scoped.get(
                "machine_heart_rate_bpm",
                pd.Series(np.nan, index=scoped.index),
            ),
            errors="coerce",
        )
    )
    if rate.notna().any():
        columns.append(rate.to_numpy(dtype=float))
    if "ed_stay_id" in scoped:
        columns.append(scoped["ed_stay_id"].notna().to_numpy(dtype=float))
    if "ecg_time" in scoped:
        # Acquisition hour as a circular pair: a night recording exists for a
        # different reason than a morning one, and the hour is preserved by
        # MIMIC's date shifting.
        stamps = pd.to_datetime(scoped["ecg_time"], errors="coerce")
        hour = (
            stamps.dt.hour.astype(float)
            + stamps.dt.minute.astype(float) / 60.0
        ).to_numpy(dtype=float)
        columns.append(np.sin(2 * np.pi * hour / 24.0))
        columns.append(np.cos(2 * np.pi * hour / 24.0))
    if "anchor_year_group" in scoped:
        # Real era. ecg_time's year is de-identification shifted per patient
        # and carries no cross-patient calendar information.
        era = anchor_year_group_start(scoped["anchor_year_group"])
        if era.notna().any():
            columns.append(era.to_numpy(dtype=float))
    for name in (
        "signal_quality_score",
        "ecg_to_landmark_days",
        "index_length_of_stay_days",
    ):
        if name in scoped:
            columns.append(
                pd.to_numeric(scoped[name], errors="coerce").to_numpy(
                    dtype=float
                )
            )
    return (
        np.column_stack(columns)
        if columns
        else np.empty((len(scoped), 0), dtype=float)
    )


def _survival_marker_assessability(
    values: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, object]:
    at_risk = np.asarray(masks["at_risk"], dtype=bool)
    scoped_values = np.asarray(values, dtype=float)[at_risk]
    event = np.asarray(masks["event_observed"], dtype=bool)[at_risk]
    assessment = marker_assessability(
        scoped_values,
        event.astype(np.int8),
        minimum_test_yield=MINIMUM_SURVIVAL_MARKER_TEST_YIELD,
        maximum_class_yield_gap=MAXIMUM_SURVIVAL_MARKER_YIELD_GAP,
    )
    finite = np.isfinite(scoped_values)
    competing = np.asarray(masks["competing_death"], dtype=bool)[at_risk]
    censored = np.asarray(masks["censored"], dtype=bool)[at_risk]

    def yield_for(state: np.ndarray) -> float:
        count = int(np.sum(state))
        return float(np.sum(finite & state) / count) if count else np.nan

    state_yields = np.asarray(
        (yield_for(event), yield_for(censored), yield_for(competing)),
        dtype=float,
    )
    observed_state_yields = state_yields[np.isfinite(state_yields)]
    state_gap = (
        float(np.max(observed_state_yields) - np.min(observed_state_yields))
        if len(observed_state_yields) >= 2
        else 0.0
    )
    gate = bool(
        assessment["assessability_gate_passed"]
        and (
            not len(observed_state_yields)
            or float(np.min(observed_state_yields))
            >= MINIMUM_SURVIVAL_MARKER_TEST_YIELD
        )
        and state_gap <= MAXIMUM_SURVIVAL_MARKER_YIELD_GAP
    )
    return {
        **assessment,
        "event_marker_test_yield": state_yields[0],
        "censored_marker_test_yield": state_yields[1],
        "competing_death_marker_test_yield": state_yields[2],
        "survival_state_test_yield_gap": state_gap,
        "survival_marker_assessability_gate_passed": gate,
    }


def _adverse_missing_marker_sensitivity(
    values: np.ndarray,
    duration_days: np.ndarray,
    event_observed: np.ndarray,
    covariates: np.ndarray,
    *,
    direction: int,
) -> dict[str, object]:
    marker = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(marker)
    if not missing.any():
        return {
            "missingness_sensitivity_status": "complete_marker_observation",
            "missingness_sensitivity_gate_passed": True,
            "missingness_adverse_log_hr": np.nan,
        }
    observed = marker[~missing]
    if not len(observed) or direction not in (-1, 1):
        return {
            "missingness_sensitivity_status": "not_evaluable",
            "missingness_sensitivity_gate_passed": False,
            "missingness_adverse_log_hr": np.nan,
        }
    width = max(float(np.ptp(observed)), 1.0)
    low = float(np.min(observed) - width)
    high = float(np.max(observed) + width)
    events = np.asarray(event_observed, dtype=bool)
    if direction > 0:
        marker[missing & events] = low
        marker[missing & ~events] = high
    else:
        marker[missing & events] = high
        marker[missing & ~events] = low
    fit = _fit_cause_specific_cox(
        marker,
        duration_days,
        events,
        covariates,
    )
    effect = float(fit["log_hazard_ratio_per_sd"])
    gate = bool(
        fit["status"] == "evaluable"
        and int(fit["direction"]) == int(direction)
        and (
            (direction > 0 and float(fit["hr_ci_low"]) > 1.0)
            or (direction < 0 and float(fit["hr_ci_high"]) < 1.0)
        )
    )
    return {
        "missingness_sensitivity_status": "adverse_event_state_imputation",
        "missingness_sensitivity_gate_passed": gate,
        "missingness_adverse_log_hr": effect,
    }


def _same_day_tie_sensitivity(
    frame: pd.DataFrame,
    values: np.ndarray,
    event_series: pd.Series,
    definition: pd.Series,
    *,
    horizon_days: int,
    primary_direction: int,
    primary_masks: Optional[Mapping] = None,
) -> dict[str, object]:
    primary = primary_masks if primary_masks is not None else _future_state_masks(
        frame,
        event_series,
        definition,
        horizon_days=horizon_days,
        same_day_tie_policy="exclude",
    )
    ambiguous_count = int(
        np.sum(primary["ambiguous_same_day_death_event"])
    )
    if ambiguous_count == 0:
        return {
            "same_day_tie_count": 0,
            "same_day_tie_sensitivity_status": "no_ambiguous_ties",
            "same_day_tie_sensitivity_gate_passed": True,
            "event_first_log_hr": np.nan,
            "death_first_log_hr": np.nan,
        }
    effects: dict[str, float] = {}
    directions: dict[str, int] = {}
    for policy in ("event_first", "death_first"):
        masks = _future_state_masks(
            frame,
            event_series,
            definition,
            horizon_days=horizon_days,
            same_day_tie_policy=policy,
        )
        rows = np.flatnonzero(masks["at_risk"])
        fit = _fit_cause_specific_cox(
            np.asarray(values, dtype=float)[rows],
            np.asarray(masks["duration_days"])[rows],
            np.asarray(masks["event_observed"])[rows],
            _future_adjustment_covariates(frame, rows),
        )
        effects[policy] = float(fit["log_hazard_ratio_per_sd"])
        directions[policy] = int(fit["direction"])
    gate = bool(
        primary_direction in (-1, 1)
        and all(
            directions.get(policy) == primary_direction
            and np.isfinite(effects.get(policy, np.nan))
            for policy in ("event_first", "death_first")
        )
    )
    return {
        "same_day_tie_count": ambiguous_count,
        "same_day_tie_sensitivity_status": (
            "direction_robust" if gate else "direction_not_robust"
        ),
        "same_day_tie_sensitivity_gate_passed": gate,
        "event_first_log_hr": effects.get("event_first", np.nan),
        "death_first_log_hr": effects.get("death_first", np.nan),
    }


def _add_survival_replication_status(
    validation: pd.DataFrame,
    *,
    validation_fdr_alpha: float,
    minimum_effect_retention: float = 0.50,
) -> pd.DataFrame:
    output = validation.copy()
    train_direction = pd.to_numeric(
        output.get("train_direction"), errors="coerce"
    )
    validation_direction = pd.to_numeric(
        output.get("validation_direction"), errors="coerce"
    )
    q_values = pd.to_numeric(
        output.get("validation_q_value"), errors="coerce"
    )
    retention = pd.to_numeric(
        output.get("effect_retention_ratio"), errors="coerce"
    )
    ci_low = pd.to_numeric(
        output.get("validation_cox_hr_ci_low"), errors="coerce"
    )
    ci_high = pd.to_numeric(
        output.get("validation_cox_hr_ci_high"), errors="coerce"
    )
    directional_ci = ((train_direction > 0) & (ci_low > 1.0)) | (
        (train_direction < 0) & (ci_high < 1.0)
    )
    output["direction_consistent"] = (
        train_direction.eq(validation_direction)
        & train_direction.isin((-1, 1))
    )
    output["validation_stable"] = (
        output["direction_consistent"]
        & retention.ge(minimum_effect_retention)
        & directional_ci
    )
    output["replicated"] = (
        q_values.le(float(validation_fdr_alpha))
        & output["global_fdr_significant"].fillna(False)
        & output["hierarchical_fdr_significant"].fillna(False)
        & output["confirmatory_assessability_gate_passed"].fillna(False)
        & output["adjusted_model_gate_passed"].fillna(False)
        & output["ph_assumption_gate_passed"].fillna(False)
        & output["same_day_tie_sensitivity_gate_passed"].fillna(False)
        & output["inference_status"].eq("formal_inference")
        & output["validation_stable"]
    )
    output["replication_policy"] = (
        "train_selected_adjusted_cause_specific_cox; validation_family_global_"
        "and_hierarchical_fdr; assessability; PH; same_day_tie_sensitivity; "
        "same_direction; directional_95ci; minimum_50pct_log_hr_retention"
    )
    output["competing_risk_interpretation"] = (
        "cause_specific_hazard_only; cumulative_incidence_and_absolute_risk_"
        "require_external_competing_risk_model"
    )
    output["absolute_risk_claim_allowed"] = False
    return output


def _empty_future_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "phenotype_id",
            "phenotype_name",
            "body_system",
            "horizon_days",
            "panel_id",
            "feature_id",
            "evaluation_split",
            "analysis_type",
            "target_kind",
            "train_q_value",
            "train_aligned_auc",
            "train_direction",
            "validation_raw_auc",
            "validation_q_value",
            "cox_log_hazard_ratio_per_sd",
            "cox_hazard_ratio_per_sd",
            "validation_cox_log_hazard_ratio_per_sd",
            "validation_cox_hazard_ratio_per_sd",
            "validation_cox_hr_ci_low",
            "validation_cox_hr_ci_high",
            "n_at_risk",
            "n_event",
            "n_censored",
            "n_competing_death",
            "p_value",
            "q_value",
            "inference_status",
            "direction_consistent",
            "replicated",
            "absolute_risk_claim_allowed",
            "locked_test_scanned",
        )
    )


def _source_bool(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def _validate_horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(sorted({int(value) for value in values}))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Future phenotype horizons must be positive days")
    return horizons


def _stable_seed(base: int, *parts: str) -> int:
    payload = "|".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _copy_query_atomic(connection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    with profile_query(f"future-phenome:{destination.name}") as profile:
        copy_result = connection.execute(
            f"COPY ({query}) TO '{_sql_path(temporary)}' "
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
