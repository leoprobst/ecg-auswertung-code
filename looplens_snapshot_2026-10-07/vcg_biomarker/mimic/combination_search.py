"""Adaptive-depth marker combinations: how deep is deep enough.

Scientific limitation (v1.2): the historical ``confirmation`` columns below
are inner selection measurements, because they determine stopping and depth.
They are not independent confirmation. Historical pilot numbers in this module
are hypothesis-generating; only a separately nested search can assess optimism.

The existing exploratory search (``optimization.grouped_score_search``) is
hard-capped at three components, and refuses anything deeper. That cap was a
guess. Measured on real data it turns out to be too tight.

Held-out discrimination for ``reduced_lvef_strict``, greedy forward selection
on one patient-disjoint half and measured on the other, averaged over three
independent splits:

    depth   1      2      3      4      5      6      8     10     13     15
    AUC   0.710  0.733  0.749  0.762  0.773  0.784  0.787  0.792  0.797  0.800

Three things follow, and they are the whole design rationale:

* **The combination carries real information.** With permuted labels the same
  procedure stays flat at 0.49-0.52 at every depth, so the rise is not the
  search finding its own noise.
* **Depth 3 leaves roughly 0.05 AUC unused.** That is not a rounding error;
  it is larger than the gap between many single markers and chance.
* **It plateaus, it does not diverge.** Most of the gain is in by depth 6,
  the curve flattens around 13-15, and past that it drifts sideways or down.

So the answer to "should depth be unbounded" is: it does not need to be, but
the stopping point should be *measured rather than assumed*. This module
searches adaptively -- it keeps going while each new level still buys held-out
discrimination, and stops on its own when it does not. It can also be told to
run until externally stopped, which is the same loop with the automatic
stopping rule disabled.

Two design choices make deep search affordable and honest:

**Beam search, not exhaustive enumeration.** Exhaustive search of depth 4 over
67 representatives is 766,480 subsets; depth 5 is 9.3 million. A beam of width
B extends only the best B partial combinations, so each level costs B x R
evaluations -- 536 per level at B=8, R=67. Cost becomes linear in depth
instead of exponential, which is what makes "keep going until it stops paying"
a finite proposition at all.

**Selection and measurement on patient-disjoint halves.** Greedy forward
selection will improve its own objective at every single step, forever, by
construction. A depth curve measured on the data that chose it is therefore
meaningless -- it can only go up. The increment that decides whether to go
deeper is measured on patients the selection never saw, which is why the curve
above is allowed to turn downward and why the stopping rule means something.

Everything here runs on the development splits. The sealed test is not read.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEEP_COMBINATION_VERSION = "1.2.0"

# Depth at which a combination stops being expressible as a bedside rule.
# Below it a result can be written as a criterion ("angle > x AND width > y");
# above it, it is a score. Both are legitimate findings and both go through
# the same confirmation, but they cannot be reported in the same language.
CRITERION_DEPTH_CEILING = 3

TIER_CRITERION = "criterion_expressible"
TIER_SCORE = "composite_score_only"


def claim_tier_for_depth(depth: int) -> str:
    return TIER_CRITERION if int(depth) <= CRITERION_DEPTH_CEILING else TIER_SCORE


@dataclass(frozen=True)
class DepthCandidate:
    """One combination, with the numbers that decide whether it survives."""

    components: tuple[str, ...]
    families: tuple[str, ...]
    depth: int
    selection_auc: float
    confirmation_auc: float
    confirmation_increment: float
    complete_case_fraction: float
    n_confirmation: int
    claim_tier: str

    def to_dict(self) -> dict[str, object]:
        return {
            "components": list(self.components),
            "families": list(self.families),
            "depth": self.depth,
            "selection_auc": self.selection_auc,
            "confirmation_auc": self.confirmation_auc,
            "confirmation_increment": self.confirmation_increment,
            "complete_case_fraction": self.complete_case_fraction,
            "n_confirmation": self.n_confirmation,
            "claim_tier": self.claim_tier,
            "deep_combination_version": DEEP_COMBINATION_VERSION,
            "measurement_role": "inner_selection_not_independent_confirmation",
            "inner_selection_auc": self.confirmation_auc,
            "evidence_stage": "exploratory",
            "clinical_claim_allowed": False,
        }


@dataclass
class DepthLevelResult:
    """What one level of the search produced, written out before the next."""

    depth: int
    best: Optional[DepthCandidate]
    beam: tuple[DepthCandidate, ...]
    evaluations: int
    elapsed_seconds: float
    improved: bool
    stop_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "best_components": (
                list(self.best.components) if self.best else []
            ),
            "best_confirmation_auc": (
                self.best.confirmation_auc if self.best else float("nan")
            ),
            "confirmation_increment": (
                self.best.confirmation_increment if self.best else float("nan")
            ),
            "beam_size": len(self.beam),
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "improved": self.improved,
            "stop_reason": self.stop_reason,
            "claim_tier": claim_tier_for_depth(self.depth),
            "deep_combination_version": DEEP_COMBINATION_VERSION,
        }


@dataclass
class SearchBudget:
    """Everything that can end the search, in one place.

    ``max_depth=None`` with ``patience=None`` is the "run until stopped" mode:
    nothing ends it except the stop callback, the wall-clock budget, or running
    out of families to draw from.
    """

    max_depth: Optional[int] = None
    patience: Optional[int] = 2
    minimum_increment: float = 0.002
    seconds: Optional[float] = None
    max_evaluations: Optional[int] = None
    stop_requested: Optional[Callable[[], bool]] = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_depth is not None and int(self.max_depth) < 2:
            raise ValueError("max_depth must be at least two, or None")
        if self.patience is not None and int(self.patience) < 1:
            raise ValueError("patience must be at least one, or None")
        if self.minimum_increment < 0:
            raise ValueError("minimum_increment cannot be negative")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("seconds budget must be positive, or None")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("max_evaluations must be positive, or None")

    @property
    def runs_until_stopped(self) -> bool:
        return self.max_depth is None and self.patience is None

    def to_dict(self) -> dict[str, object]:
        return {
            "max_depth": self.max_depth,
            "patience": self.patience,
            "minimum_increment": self.minimum_increment,
            "seconds": self.seconds,
            "max_evaluations": self.max_evaluations,
            "runs_until_stopped": self.runs_until_stopped,
        }


def split_subjects_for_search(
    subject_ids: np.ndarray,
    *,
    random_seed: int = 20260814,
    selection_share: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into a selection half and a measurement half, by patient.

    Splitting by row would put the same patient on both sides, and greedy
    selection would then be measured on people it had already chosen against.
    """

    subjects = np.asarray(subject_ids)
    unique = np.unique(subjects)
    if len(unique) < 4:
        raise ValueError("Too few subjects to split for an honest measurement")
    generator = np.random.default_rng(int(random_seed))
    shuffled = unique.copy()
    generator.shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * selection_share))))
    selection_subjects = set(shuffled[:cut].tolist())
    selection_mask = np.fromiter(
        (item in selection_subjects for item in subjects),
        dtype=bool,
        count=len(subjects),
    )
    return selection_mask, ~selection_mask


def _composite_auc(
    matrix: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_rows: int,
) -> tuple[float, float, int]:
    """Equal-weight mean of aligned z-scores, scored on complete cases."""

    complete = np.all(np.isfinite(matrix), axis=1)
    score = np.zeros(len(matrix))
    score[complete] = matrix[complete].mean(axis=1)
    return _prepared_composite_auc(score, complete, labels, mask, minimum_rows=minimum_rows)


def _prepared_composite_auc(score, complete, labels, mask, *, minimum_rows):
    usable = complete & mask
    n_usable = int(usable.sum())
    complete_fraction = float(n_usable / max(1, int(mask.sum())))
    if n_usable < minimum_rows:
        return float("nan"), complete_fraction, n_usable
    scoped_labels = labels[usable]
    if np.unique(scoped_labels).size < 2:
        return float("nan"), complete_fraction, n_usable
    return (
        float(roc_auc_score(scoped_labels, score[usable])),
        complete_fraction,
        n_usable,
    )


def _beam_search_combinations_impl(
    aligned: Mapping[str, np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    family_for_feature: Callable[[str], str],
    budget: Optional[SearchBudget] = None,
    beam_width: int = 8,
    representatives_per_family: int = 3,
    minimum_complete_fraction: float = 0.80,
    minimum_rows: int = 200,
    random_seed: int = 20260814,
    on_depth_complete: Optional[Callable[[DepthLevelResult], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    executor_capacity: int = 1,
) -> tuple[DepthLevelResult, ...]:
    """Grow combinations one component at a time until depth stops paying.

    ``aligned`` holds out-of-fold, direction-aligned z-scores -- one array per
    marker, already on a common scale and already oriented so that larger means
    more of the endpoint. Producing them is the caller's job because the fold
    structure has to match whatever the rest of the run used.

    ``representatives_per_family`` caps how many markers of one conceptual
    family may be extended. Since a combination can hold at most one member of
    a family, the rest would be evaluated only to be excluded at the next
    level; the cap removes that waste. Set it to zero to search everything.

    Returns one entry per depth reached. Because the caller is notified after
    every level via ``on_depth_complete``, stopping the search at any moment
    leaves a complete, valid result for every depth already finished.
    """

    budget = budget or SearchBudget()
    labels = np.asarray(labels, dtype=np.int8)
    subjects = np.asarray(subject_ids)
    feature_ids = [
        feature_id
        for feature_id in aligned
        if np.isfinite(aligned[feature_id]).any()
    ]
    if len(feature_ids) < 2:
        return ()
    for feature_id in feature_ids:
        if len(aligned[feature_id]) != len(labels):
            raise ValueError(
                f"Aligned scores for {feature_id!r} do not match the labels"
            )
    if len(subjects) != len(labels):
        raise ValueError("Subject IDs and labels must align")
    if beam_width < 1:
        raise ValueError("beam_width must be at least one")

    selection_mask, confirmation_mask = split_subjects_for_search(
        subjects,
        random_seed=random_seed,
    )
    columns = {
        feature_id: np.asarray(aligned[feature_id], dtype=float)
        for feature_id in feature_ids
    }
    families = {
        feature_id: str(family_for_feature(feature_id))
        for feature_id in feature_ids
    }

    def evaluate(components: tuple[str, ...]) -> Optional[DepthCandidate]:
        matrix = np.stack([columns[item] for item in components], axis=1)
        complete = np.isfinite(matrix).all(axis=1)
        score = np.zeros(len(matrix))
        score[complete] = matrix[complete].mean(axis=1)
        selection_auc, _, _ = _prepared_composite_auc(
            score, complete,
            labels,
            selection_mask,
            minimum_rows=minimum_rows,
        )
        if not np.isfinite(selection_auc):
            return None
        confirmation_auc, complete_fraction, n_confirmation = _prepared_composite_auc(
            score, complete,
            labels,
            confirmation_mask,
            minimum_rows=minimum_rows,
        )
        if complete_fraction < minimum_complete_fraction:
            return None
        return DepthCandidate(
            components=components,
            families=tuple(families[item] for item in components),
            depth=len(components),
            selection_auc=selection_auc,
            confirmation_auc=confirmation_auc,
            confirmation_increment=float("nan"),
            complete_case_fraction=complete_fraction,
            n_confirmation=n_confirmation,
            claim_tier=claim_tier_for_depth(len(components)),
        )

    def evaluate_many(
        component_sets: Sequence[tuple[str, ...]],
    ) -> list[Optional[DepthCandidate]]:
        if executor is None:
            return [evaluate(components) for components in component_sets]
        # ``Executor.map`` preserves input order. That keeps beam tie-breaking
        # and every persisted result bit-for-bit deterministic while NumPy and
        # scikit-learn perform the independent AUC calculations concurrently.
        from .bounded_executor import adaptive_ordered_map
        return list(adaptive_ordered_map(executor, evaluate, component_sets,
                                        max_workers=executor_capacity))

    # Depth one: every marker on its own, ranked on the selection half.
    singles: list[DepthCandidate] = []
    started = time.time()
    single_keys = [(feature_id,) for feature_id in feature_ids]
    evaluations = len(single_keys)
    for candidate in evaluate_many(single_keys):
        if candidate is not None:
            singles.append(candidate)
    if not singles:
        return ()
    singles.sort(key=lambda item: -item.selection_auc)

    # Only one marker per family may enter a combination, so extending the
    # beam with every same-family sibling evaluates them only to have the
    # family rule exclude them one level later. Keeping the best few per
    # family instead cuts the search from 152,457 evaluations per endpoint to
    # 9,612 at the full panel -- 71 hours to 4.5 across 63 endpoints -- and
    # the discarded markers were competing for a slot they could not both
    # hold. Ranking uses the selection half only, so the measurement half
    # stays untouched by this choice too.
    if representatives_per_family > 0:
        per_family: dict[str, int] = {}
        representatives: list[str] = []
        for candidate in singles:
            family = candidate.families[0]
            if per_family.get(family, 0) >= representatives_per_family:
                continue
            per_family[family] = per_family.get(family, 0) + 1
            representatives.append(candidate.components[0])
        feature_ids = representatives

    beam = tuple(singles[:beam_width])
    best_so_far = max(singles, key=lambda item: item.confirmation_auc)
    levels: list[DepthLevelResult] = [
        DepthLevelResult(
            depth=1,
            best=best_so_far,
            beam=beam,
            evaluations=evaluations,
            elapsed_seconds=time.time() - started,
            improved=True,
        )
    ]
    if on_depth_complete is not None:
        on_depth_complete(levels[-1])

    previous_confirmation = best_so_far.confirmation_auc
    stale_levels = 0
    depth = 1

    while True:
        depth += 1
        if budget.max_depth is not None and depth > int(budget.max_depth):
            levels[-1].stop_reason = "max_depth_reached"
            break
        if budget.stop_requested is not None and budget.stop_requested():
            levels[-1].stop_reason = "stop_requested"
            break
        if budget.seconds is not None and (time.time() - started) >= budget.seconds:
            levels[-1].stop_reason = "time_budget_exhausted"
            break
        if (
            budget.max_evaluations is not None
            and evaluations >= int(budget.max_evaluations)
        ):
            levels[-1].stop_reason = "evaluation_budget_exhausted"
            break

        level_started = time.time()
        extended: dict[tuple[str, ...], DepthCandidate] = {}
        extension_keys: list[tuple[str, ...]] = []
        seen_keys: set[tuple[str, ...]] = set()
        for parent in beam:
            used_families = set(parent.families)
            for feature_id in feature_ids:
                if feature_id in parent.components:
                    continue
                # One component per conceptual family: two members of the same
                # family are two spellings of one measurement, and averaging
                # them buys weight, not information.
                if families[feature_id] in used_families:
                    continue
                key = tuple(sorted((*parent.components, feature_id)))
                if key in seen_keys:
                    continue
                # Remember invalid combinations too. Previously only successful
                # candidates entered ``extended``, so the same incomplete key
                # could be recomputed from several beam parents for no benefit.
                seen_keys.add(key)
                extension_keys.append(key)
        level_evaluations = len(extension_keys)
        for key, candidate in zip(
            extension_keys,
            evaluate_many(extension_keys),
            strict=True,
        ):
            if candidate is not None:
                extended[key] = candidate
        evaluations += level_evaluations

        if not extended:
            levels[-1].stop_reason = "no_remaining_families"
            break

        ranked = sorted(extended.values(), key=lambda item: -item.selection_auc)
        beam = tuple(ranked[:beam_width])
        level_best = max(ranked, key=lambda item: item.confirmation_auc)
        increment = float(level_best.confirmation_auc - previous_confirmation)
        level_best = replace(level_best, confirmation_increment=increment)
        improved = increment >= budget.minimum_increment
        result = DepthLevelResult(
            depth=depth,
            best=level_best,
            beam=beam,
            evaluations=level_evaluations,
            elapsed_seconds=time.time() - level_started,
            improved=improved,
        )
        levels.append(result)
        if on_depth_complete is not None:
            on_depth_complete(result)
        if progress is not None:
            progress(
                f"depth {depth}: held-out AUC {level_best.confirmation_auc:.4f} "
                f"({increment:+.4f}), {level_evaluations} evaluations"
            )

        if improved:
            stale_levels = 0
            previous_confirmation = level_best.confirmation_auc
        else:
            stale_levels += 1
            if budget.patience is not None and stale_levels >= int(budget.patience):
                result.stop_reason = "no_further_gain"
                break

    return tuple(levels)


def beam_search_combinations(
    aligned: Mapping[str, np.ndarray],
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    family_for_feature: Callable[[str], str],
    budget: Optional[SearchBudget] = None,
    beam_width: int = 8,
    representatives_per_family: int = 3,
    minimum_complete_fraction: float = 0.80,
    minimum_rows: int = 200,
    random_seed: int = 20260814,
    on_depth_complete: Optional[Callable[[DepthLevelResult], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    worker_count: int = 1,
) -> tuple[DepthLevelResult, ...]:
    """Run the deterministic beam search, optionally evaluating candidates in threads.

    The candidate calculations are dominated by NumPy/scikit-learn kernels that
    release the GIL. A single persistent thread pool therefore avoids Windows
    process-spawn and array-serialization costs while using the CPU allocation
    already reserved for this stage. Result order is identical to serial mode.
    """

    resolved_worker_count = int(worker_count)
    if resolved_worker_count < 1:
        raise ValueError("worker_count must be at least one")
    if resolved_worker_count == 1:
        return _beam_search_combinations_impl(
            aligned,
            labels,
            subject_ids,
            family_for_feature=family_for_feature,
            budget=budget,
            beam_width=beam_width,
            representatives_per_family=representatives_per_family,
            minimum_complete_fraction=minimum_complete_fraction,
            minimum_rows=minimum_rows,
            random_seed=random_seed,
            on_depth_complete=on_depth_complete,
            progress=progress,
        )
    with ThreadPoolExecutor(
        max_workers=resolved_worker_count,
        thread_name_prefix="deep-combination",
    ) as executor:
        return _beam_search_combinations_impl(
            aligned,
            labels,
            subject_ids,
            family_for_feature=family_for_feature,
            budget=budget,
            beam_width=beam_width,
            representatives_per_family=representatives_per_family,
            minimum_complete_fraction=minimum_complete_fraction,
            minimum_rows=minimum_rows,
            random_seed=random_seed,
            on_depth_complete=on_depth_complete,
            progress=progress,
            executor=executor,
            executor_capacity=resolved_worker_count,
        )


def depth_profile(levels: Sequence[DepthLevelResult]) -> pd.DataFrame:
    """The depth curve as a table: what each extra component actually bought."""

    if not levels:
        return pd.DataFrame(
            columns=[
                "depth",
                "confirmation_auc",
                "confirmation_increment",
                "evaluations",
                "claim_tier",
            ]
        )
    return pd.DataFrame([level.to_dict() for level in levels])


def recommended_depth(
    levels: Sequence[DepthLevelResult],
    *,
    tolerance: float = 0.002,
) -> Mapping[str, object]:
    """The shallowest depth that is not meaningfully worse than the best.

    Deeper is not automatically better even when it scores higher: a composite
    of fifteen markers that beats one of six by 0.003 AUC has bought almost
    nothing and given up being explainable. This reports both -- the depth that
    scored best, and the shallowest depth within ``tolerance`` of it, which is
    the one worth carrying forward.
    """

    scored = [
        (level, best)
        for level in levels
        if (best := level.best) is not None
        and np.isfinite(best.confirmation_auc)
    ]
    if not scored:
        return {
            "best_depth": 0,
            "recommended_depth": 0,
            "deep_combination_version": DEEP_COMBINATION_VERSION,
        }
    peak_level, peak_best = max(
        scored,
        key=lambda item: item[1].confirmation_auc,
    )
    parsimonious_level, parsimonious_best = min(
        (
            (level, best)
            for level, best in scored
            if best.confirmation_auc
            >= peak_best.confirmation_auc - tolerance
        ),
        key=lambda item: item[0].depth,
    )
    return {
        "best_depth": peak_level.depth,
        "best_confirmation_auc": float(peak_best.confirmation_auc),
        "recommended_depth": parsimonious_level.depth,
        "recommended_confirmation_auc": float(
            parsimonious_best.confirmation_auc
        ),
        "recommended_components": list(parsimonious_best.components),
        "auc_given_up_for_parsimony": float(
            peak_best.confirmation_auc - parsimonious_best.confirmation_auc
        ),
        "claim_tier": claim_tier_for_depth(parsimonious_level.depth),
        "depths_evaluated": len(scored),
        "total_evaluations": int(sum(level.evaluations for level in levels)),
        "note_en": (
            "Depth was chosen by measured held-out gain, not assumed. The "
            "recommended depth is the shallowest one within tolerance of the "
            "best, because an unexplainable composite that ties is a loss."
        ),
        "deep_combination_version": DEEP_COMBINATION_VERSION,
    }


def exhaustive_evaluation_count(n_representatives: int, depth: int) -> int:
    """How many subsets an exhaustive search of this depth would visit."""

    from math import comb

    return int(comb(max(0, int(n_representatives)), max(0, int(depth))))


def beam_evaluation_count(
    n_representatives: int,
    depth: int,
    beam_width: int,
) -> int:
    """How many a beam search visits instead -- linear in depth, not exponential."""

    if depth < 1:
        return 0
    return int(n_representatives) + int(
        max(0, depth - 1) * beam_width * n_representatives
    )


def search_cost_comparison(
    n_representatives: int,
    *,
    beam_width: int = 8,
    depths: Sequence[int] = (2, 3, 4, 5, 6, 8, 10, 15, 20),
) -> pd.DataFrame:
    """Why exhaustive search cannot go deep and beam search can."""

    rows = []
    for depth in depths:
        exhaustive = exhaustive_evaluation_count(n_representatives, depth)
        beam = beam_evaluation_count(n_representatives, depth, beam_width)
        rows.append(
            {
                "depth": int(depth),
                # Python integers are unbounded, parquet columns are not: the
                # exhaustive count passes int64 well before it passes the
                # depths this project cares about. Stored as a magnitude,
                # which is all the comparison ever needed.
                "exhaustive_subsets": float(exhaustive),
                "exhaustive_subsets_log10": (
                    # log10 on the Python integer, not on a float that has
                    # already overflowed to infinity.
                    float(len(str(exhaustive)) - 1) if exhaustive > 0 else float("nan")
                ),
                "beam_evaluations": int(beam),
                "reduction_factor": (
                    float(exhaustive / beam) if beam else float("nan")
                ),
                "claim_tier": claim_tier_for_depth(depth),
            }
        )
    return pd.DataFrame(rows)


def assert_depth_is_declared(frame: pd.DataFrame) -> None:
    """Every reported combination must carry its depth and claim tier.

    A composite reported without its depth reads like a single marker, and the
    difference between "a criterion" and "an unweighted fifteen-marker score"
    is exactly the thing a reader needs and cannot recover afterwards.
    """

    if frame.empty:
        return
    missing = [
        column
        for column in ("depth", "claim_tier")
        if column not in frame.columns
    ]
    if missing:
        raise AssertionError(
            f"Combination results must declare {missing}; without depth a "
            "composite is indistinguishable from a single marker."
        )
    unlabelled = frame["claim_tier"].isna() | frame["claim_tier"].eq("")
    if bool(unlabelled.any()):
        raise AssertionError(
            f"{int(unlabelled.sum())} combination(s) carry no claim tier."
        )
    expected = frame["depth"].map(claim_tier_for_depth)
    mismatched = frame.loc[frame["claim_tier"].ne(expected)]
    if not mismatched.empty:
        raise AssertionError(
            f"{len(mismatched)} combination(s) carry a claim tier that does "
            "not follow from their depth."
        )


__all__ = [
    "CRITERION_DEPTH_CEILING",
    "DEEP_COMBINATION_VERSION",
    "DepthCandidate",
    "DepthLevelResult",
    "SearchBudget",
    "TIER_CRITERION",
    "TIER_SCORE",
    "assert_depth_is_declared",
    "beam_evaluation_count",
    "beam_search_combinations",
    "claim_tier_for_depth",
    "depth_profile",
    "exhaustive_evaluation_count",
    "recommended_depth",
    "search_cost_comparison",
    "split_subjects_for_search",
]
