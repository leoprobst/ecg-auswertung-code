from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .feature_roles import (
    canonical_feature_id,
    is_combination_candidate,
)


@dataclass(frozen=True)
class GroupedScoreCandidate:
    components: tuple[str, ...]
    families: tuple[str, ...]
    mean_auc: float
    mean_average_precision: float
    evaluable_folds: int
    positive_direction_folds: int
    complete_case_fraction: float
    stable: bool
    fold_auc_standard_deviation: float = float("nan")
    minimum_fold_auc: float = float("nan")
    best_component_mean_auc: float = float("nan")
    incremental_auc_vs_best_component: float = float("nan")
    maximum_pairwise_abs_correlation: float = float("nan")
    stability_adjusted_auc: float = float("nan")
    cv_repeats: int = 1
    best_component: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def grouped_score_search(
    train: pd.DataFrame,
    *,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    feature_ids: tuple[str, ...],
    family_for_feature: Callable[[str], str],
    folds: int = 5,
    max_components: int = 3,
    representatives_per_family: int = 2,
    selected_pairs: int = 1,
    selected_triples: int = 1,
    triple_seed_pairs: int = 25,
    minimum_complete_fraction: float = 0.80,
    maximum_pairwise_abs_correlation: Optional[float] = None,
    minimum_incremental_auc: float = 0.0,
    cv_repeats: int = 1,
    random_seed: int = 20260725,
    required_component_predicate: Optional[Callable[[str], bool]] = None,
    feature_value_loader: Optional[
        Callable[[tuple[str, ...]], Mapping[str, np.ndarray]]
    ] = None,
    feature_block_size: int = 128,
    eligibility_predicate: Optional[Callable[[str], bool]] = None,
) -> tuple[GroupedScoreCandidate, ...]:
    labels = np.asarray(labels, dtype=np.int8)
    subjects = np.asarray(subject_ids)
    if len(train) != len(labels) or len(labels) != len(subjects):
        raise ValueError("Train frame, labels and subject IDs must align")
    split_count = int(folds)
    if split_count < 2:
        raise ValueError("folds must be at least 2")
    repeat_count = int(cv_repeats)
    if repeat_count < 1:
        raise ValueError("cv_repeats must be at least 1")
    if maximum_pairwise_abs_correlation is not None and not (
        0.0 < maximum_pairwise_abs_correlation <= 1.0
    ):
        raise ValueError(
            "maximum_pairwise_abs_correlation must be in (0, 1]"
        )
    if minimum_incremental_auc < 0:
        raise ValueError("minimum_incremental_auc cannot be negative")
    if representatives_per_family < 1:
        raise ValueError("representatives_per_family must be at least one")
    if selected_pairs < 0 or selected_triples < 0:
        raise ValueError("selected score counts cannot be negative")
    if triple_seed_pairs < 0:
        raise ValueError("triple_seed_pairs cannot be negative")
    if int(feature_block_size) < 1:
        raise ValueError("feature_block_size must be positive")
    if int(max_components) < 2:
        return ()
    if int(max_components) > 3:
        raise ValueError(
            "Exploratory development scores are limited to three components"
        )
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < split_count:
        return ()
    class_subject_counts = tuple(
        len(np.unique(subjects[labels == label]))
        for label in np.unique(labels)
    )
    if len(class_subject_counts) != 2 or min(class_subject_counts) < split_count:
        return ()
    repeated_splits: list[tuple[int, np.ndarray, np.ndarray]] = []
    try:
        for repeat in range(repeat_count):
            splitter = StratifiedGroupKFold(
                n_splits=split_count,
                shuffle=True,
                random_state=int(random_seed) + 104_729 * repeat,
            )
            repeated_splits.extend(
                (
                    repeat,
                    np.asarray(fit_index),
                    np.asarray(holdout_index),
                )
                for fit_index, holdout_index in splitter.split(
                    np.zeros(len(labels)),
                    labels,
                    groups=subjects,
                )
            )
    except ValueError:
        return ()
    if len(repeated_splits) != split_count * repeat_count:
        return ()
    if any(
        np.unique(labels[fit_index]).size < 2
        or np.unique(labels[holdout_index]).size < 2
        for _, fit_index, holdout_index in repeated_splits
    ):
        return ()

    feature_diagnostics: dict[
        str,
        tuple[float, float, int, int, float, float],
    ] = {}
    canonical_ids: set[str] = set()
    eligible_feature_ids: list[str] = []
    for feature_id in feature_ids:
        if not (eligibility_predicate or is_combination_candidate)(feature_id):
            continue
        canonical = canonical_feature_id(feature_id)
        if canonical != feature_id or canonical in canonical_ids:
            continue
        canonical_ids.add(canonical)
        eligible_feature_ids.append(feature_id)

    def evaluate_feature(
        values: np.ndarray,
    ) -> Optional[
        tuple[np.ndarray, tuple[float, float, int, int, float, float]]
    ]:
        values = np.asarray(values, dtype=float)
        if len(values) != len(train):
            raise ValueError(
                "Loaded score feature does not align with the train frame"
            )
        aligned: np.ndarray = np.full(
            (repeat_count, len(values)),
            np.nan,
            dtype=float,
        )
        fold_auc: list[float] = []
        fold_ap: list[float] = []
        directions: list[int] = []
        for repeat, fit_index, holdout_index in repeated_splits:
            fit_valid = np.isfinite(values[fit_index])
            holdout_valid = np.isfinite(values[holdout_index])
            if (
                np.unique(labels[fit_index][fit_valid]).size < 2
                or np.unique(labels[holdout_index][holdout_valid]).size < 2
            ):
                continue
            fit_values = values[fit_index][fit_valid]
            fit_labels = labels[fit_index][fit_valid]
            raw_auc = float(roc_auc_score(fit_labels, fit_values))
            direction = 1 if raw_auc >= 0.5 else -1
            mean = float(np.mean(fit_values))
            standard_deviation = float(np.std(fit_values, ddof=1))
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                continue
            holdout_values = values[holdout_index][holdout_valid]
            holdout_labels = labels[holdout_index][holdout_valid]
            holdout_score = direction * (
                (holdout_values - mean) / standard_deviation
            )
            aligned[repeat, holdout_index[holdout_valid]] = holdout_score
            fold_auc.append(float(roc_auc_score(holdout_labels, holdout_score)))
            fold_ap.append(
                float(average_precision_score(holdout_labels, holdout_score))
            )
            directions.append(direction)
        required_evaluable = repeat_count * max(2, split_count - 1)
        if len(fold_auc) < required_evaluable:
            return None
        dominant_direction_count = max(
            directions.count(1),
            directions.count(-1),
        )
        complete_fraction = float(
            np.mean(np.all(np.isfinite(aligned), axis=0))
        )
        diagnostic = (
            float(np.mean(fold_auc)),
            float(np.mean(fold_ap)),
            len(fold_auc),
            dominant_direction_count,
            complete_fraction,
            float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else 0.0,
        )
        return aligned, diagnostic

    def load_feature_block(
        selected: tuple[str, ...],
    ) -> Mapping[str, np.ndarray]:
        if feature_value_loader is not None:
            loaded = feature_value_loader(selected)
            return {
                feature_id: np.asarray(loaded[feature_id], dtype=float)
                for feature_id in selected
                if feature_id in loaded
            }
        return {
            feature_id: pd.to_numeric(
                train[feature_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            for feature_id in selected
            if feature_id in train
        }

    # First pass retains only six scalar diagnostics per marker.  The former
    # implementation kept the raw vector plus ``cv_repeats`` OOF vectors for
    # all ~1,700 markers, on top of the already-wide endpoint frame.  At full
    # cohort size that alone can exceed 25 GB.  Re-reading only the few family
    # representatives below is both faster than paging and exactly preserves
    # the CV selection rule.
    block_size = int(feature_block_size)
    for start in range(0, len(eligible_feature_ids), block_size):
        block_feature_ids = tuple(
            eligible_feature_ids[start : start + block_size]
        )
        loaded = load_feature_block(block_feature_ids)
        for feature_id in block_feature_ids:
            values = loaded.get(feature_id)
            if values is None:
                continue
            evaluated = evaluate_feature(values)
            if evaluated is not None:
                _, feature_diagnostics[feature_id] = evaluated

    family_ranked: dict[str, list[str]] = {}
    for feature_id, diagnostic in feature_diagnostics.items():
        (
            mean_auc,
            mean_ap,
            evaluable,
            direction_count,
            complete,
            _,
        ) = diagnostic
        if complete < minimum_complete_fraction:
            continue
        if direction_count < int(np.ceil(0.80 * evaluable)):
            continue
        family = family_for_feature(feature_id)
        family_ranked.setdefault(family, []).append(feature_id)
    representatives: list[str] = []
    for family in sorted(family_ranked):
        ranked = sorted(
            family_ranked[family],
            key=lambda feature_id: (
                -(
                    feature_diagnostics[feature_id][0]
                    - 0.5 * feature_diagnostics[feature_id][5]
                ),
                -feature_diagnostics[feature_id][0],
                -feature_diagnostics[feature_id][1],
                feature_id,
            ),
        )
        representatives.extend(ranked[: max(1, representatives_per_family)])

    raw_values: dict[str, np.ndarray] = {}
    oof: dict[str, np.ndarray] = {}
    for start in range(0, len(representatives), block_size):
        selected = tuple(representatives[start : start + block_size])
        loaded = load_feature_block(selected)
        for feature_id in selected:
            values = loaded.get(feature_id)
            if values is None:
                continue
            evaluated = evaluate_feature(values)
            if evaluated is None:
                continue
            aligned, _ = evaluated
            raw_values[feature_id] = np.asarray(values, dtype=float)
            oof[feature_id] = aligned
    representatives = [
        feature_id
        for feature_id in representatives
        if feature_id in raw_values and feature_id in oof
    ]

    # Every pair inside a triple was already correlated during the pair phase.
    # Ranking dominates the cost of a Spearman coefficient, so recomputing it
    # is the single most wasteful thing this search did.
    correlation_cache: dict[tuple[str, str], float] = {}

    def evaluate_components(
        component_tuple: tuple[str, ...],
    ) -> Optional[GroupedScoreCandidate]:
        component_count = len(component_tuple)
        families = tuple(family_for_feature(item) for item in component_tuple)
        if len(set(families)) != component_count:
            return None
        if (
            required_component_predicate is not None
            and not any(
                required_component_predicate(item)
                for item in component_tuple
            )
        ):
            return None
        pairwise_correlations = _pairwise_abs_spearman(
            component_tuple,
            raw_values,
            cache=correlation_cache,
        )
        maximum_correlation = (
            max(pairwise_correlations) if pairwise_correlations else 0.0
        )
        if (
            maximum_pairwise_abs_correlation is not None
            and maximum_correlation >= maximum_pairwise_abs_correlation
        ):
            return None
        matrix = np.stack([oof[item] for item in component_tuple], axis=2)
        complete = np.all(np.isfinite(matrix), axis=2)
        complete_fraction = float(np.mean(np.all(complete, axis=0)))
        if complete_fraction < minimum_complete_fraction:
            return None
        score = np.mean(matrix, axis=2)
        fold_aucs: list[float] = []
        fold_aps: list[float] = []
        positive_folds = 0
        for repeat, _, holdout_index in repeated_splits:
            valid = complete[repeat, holdout_index]
            fold_labels = labels[holdout_index][valid]
            fold_score = score[repeat, holdout_index][valid]
            if np.unique(fold_labels).size < 2:
                continue
            auc = float(roc_auc_score(fold_labels, fold_score))
            fold_aucs.append(auc)
            fold_aps.append(
                float(average_precision_score(fold_labels, fold_score))
            )
            positive_folds += int(auc > 0.5)
        if not fold_aucs:
            return None
        evaluable = len(fold_aucs)
        fold_auc_std = (
            float(np.std(fold_aucs, ddof=1)) if evaluable > 1 else 0.0
        )
        mean_auc = float(np.mean(fold_aucs))
        best_component = max(
            component_tuple,
            key=lambda item: (
                feature_diagnostics[item][0],
                feature_diagnostics[item][1],
                item,
            ),
        )
        best_component_auc = feature_diagnostics[best_component][0]
        incremental_auc = mean_auc - best_component_auc
        stable = (
            evaluable >= repeat_count * max(2, split_count - 1)
            and positive_folds >= int(np.ceil(0.80 * evaluable))
            and incremental_auc >= minimum_incremental_auc
        )
        return GroupedScoreCandidate(
            components=tuple(component_tuple),
            families=families,
            mean_auc=mean_auc,
            mean_average_precision=float(np.mean(fold_aps)),
            evaluable_folds=evaluable,
            positive_direction_folds=positive_folds,
            complete_case_fraction=complete_fraction,
            stable=stable,
            fold_auc_standard_deviation=fold_auc_std,
            minimum_fold_auc=float(np.min(fold_aucs)),
            best_component_mean_auc=float(best_component_auc),
            incremental_auc_vs_best_component=float(incremental_auc),
            maximum_pairwise_abs_correlation=float(maximum_correlation),
            stability_adjusted_auc=float(mean_auc - 0.5 * fold_auc_std),
            cv_repeats=repeat_count,
            best_component=best_component,
        )

    pair_candidates = [
        candidate
        for component_tuple in combinations(representatives, 2)
        if (candidate := evaluate_components(component_tuple)) is not None
    ]
    stable_pairs = sorted(
        (candidate for candidate in pair_candidates if candidate.stable),
        key=_score_candidate_sort_key,
    )

    triple_candidates: list[GroupedScoreCandidate] = []
    if int(max_components) >= 3 and stable_pairs:
        seeds = (
            stable_pairs
            if triple_seed_pairs == 0
            else stable_pairs[:triple_seed_pairs]
        )
        triple_component_sets: set[tuple[str, ...]] = set()
        representative_order = {
            feature_id: index
            for index, feature_id in enumerate(representatives)
        }
        for pair in seeds:
            pair_families = set(pair.families)
            for third in representatives:
                if third in pair.components:
                    continue
                if family_for_feature(third) in pair_families:
                    continue
                triple_component_sets.add(
                    tuple(
                        sorted(
                            (*pair.components, third),
                            key=representative_order.__getitem__,
                        )
                    )
                )
        for component_tuple in sorted(triple_component_sets):
            candidate = evaluate_components(component_tuple)
            if candidate is not None:
                triple_candidates.append(candidate)

    selected_candidates: list[GroupedScoreCandidate] = []
    selected_candidates.extend(stable_pairs[:selected_pairs])
    selected_candidates.extend(
        sorted(
            (candidate for candidate in triple_candidates if candidate.stable),
            key=_score_candidate_sort_key,
        )[:selected_triples]
    )
    return tuple(selected_candidates)


def _score_candidate_sort_key(item: GroupedScoreCandidate) -> tuple:
    return (
        -item.stability_adjusted_auc,
        -item.incremental_auc_vs_best_component,
        -item.mean_auc,
        -item.mean_average_precision,
        -item.complete_case_fraction,
        item.components,
    )


def _pairwise_abs_spearman(
    components: tuple[str, ...],
    raw_values: Mapping[str, np.ndarray],
    *,
    cache: Optional[dict[tuple[str, str], float]] = None,
) -> list[float]:
    values: list[float] = []
    for first, second in combinations(components, 2):
        key = (first, second) if first <= second else (second, first)
        if cache is not None and key in cache:
            cached = cache[key]
            if np.isfinite(cached):
                values.append(cached)
            continue
        a = raw_values[first]
        b = raw_values[second]
        valid = np.isfinite(a) & np.isfinite(b)
        if np.sum(valid) < 3:
            if cache is not None:
                cache[key] = float("nan")
            continue
        correlation = pd.Series(a[valid]).corr(
            pd.Series(b[valid]),
            method="spearman",
        )
        resolved = (
            float(abs(correlation))
            if correlation is not None and np.isfinite(correlation)
            else float("nan")
        )
        if cache is not None:
            cache[key] = resolved
        if np.isfinite(resolved):
            values.append(resolved)
    return values
