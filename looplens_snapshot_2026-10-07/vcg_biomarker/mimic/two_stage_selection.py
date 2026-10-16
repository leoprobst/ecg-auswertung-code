"""A path from what the screen found to what the study can claim.

The discovery half and the confirmatory half of this project were never
connected. Measured on the completed run:

===========================================  ======
Marker im Katalog                             1.753
replizierte Treffer aus dem breiten Screen   21.099
Promotionskandidaten                            493
davon je befoerdert                               0
Marker, die je bestaetigend geprueft werden      12
===========================================  ======

``PRIMARY_ADJUSTED_MARKERS`` pairs each endpoint with exactly one marker,
hand-curated, chosen "whose physiology matches the reference standard rather
than whatever screened best". That is the right instinct -- prespecification
is what keeps the confirmatory claim honest -- but it means 0.7 % of the
catalog can ever reach a claim, and the broad screen can produce nothing but
hypotheses no matter what it finds.

**This adds a second arm; it replaces nothing.** The curated pairs keep running
exactly as before, under ``model_role="secondary_adjusted"``. What is new is a
selection made from the training split alone, confirmed on validation, and
labelled ``two_stage_confirmatory`` so the two can never be read as one.

**Why this is still inferentially clean.** The rule is fixed before the screen
runs: how many markers per endpoint, ranked by what, deduplicated how. The
validation split sees neither the selection nor the model fit -- only the
evaluation. A marker reaches confirmation because the *training* data ranked it
highly, which is a statement the held-out data never had a hand in.

**What it deliberately does not do.** It does not lower any bar. A selected
marker still has to survive the same covariate adjustment as a curated one, and
its result carries the count of markers that were screened to find it, so a
reader can weigh the multiplicity that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .feature_roles import is_candidate_biomarker


TWO_STAGE_VERSION = "1.0.0"

# Fixed before any screen runs.
#
# Three per endpoint: enough that a curated pairing can be wrong without the
# endpoint going unexamined, few enough that the confirmatory family stays
# small. With 73 endpoints that is at most 219 confirmations against the 72
# the curated arm performs -- a study, not a fishing expedition.
TWO_STAGE_TOP_K = 3

# Screening evidence a marker needs before it is worth confirming. The alpha
# is the train screening alpha; the effect floor is the one the broad screen
# already declares, so the two cannot drift apart.
TWO_STAGE_TRAIN_ALPHA = 0.01
TWO_STAGE_MINIMUM_ABS_RHO = 0.10
TWO_STAGE_MINIMUM_AUC_MARGIN = 0.05


@dataclass(frozen=True)
class TwoStagePolicy:
    """Everything that decides the selection, in one auditable object."""

    top_k: int = TWO_STAGE_TOP_K
    train_alpha: float = TWO_STAGE_TRAIN_ALPHA
    minimum_abs_rho: float = TWO_STAGE_MINIMUM_ABS_RHO
    minimum_auc_margin: float = TWO_STAGE_MINIMUM_AUC_MARGIN
    deduplicate_by_cluster: bool = True

    def to_dict(self) -> dict:
        return {
            "top_k": int(self.top_k),
            "train_alpha": float(self.train_alpha),
            "minimum_abs_rho": float(self.minimum_abs_rho),
            "minimum_auc_margin": float(self.minimum_auc_margin),
            "deduplicate_by_cluster": bool(self.deduplicate_by_cluster),
            "version": TWO_STAGE_VERSION,
        }

    def minimum_effect(self, target_kind: object) -> float:
        return (
            self.minimum_auc_margin
            if str(target_kind) == "binary"
            else self.minimum_abs_rho
        )


def neutral_effect(target_kind: object) -> float:
    return 0.5 if str(target_kind) == "binary" else 0.0


class ValidationLeak(RuntimeError):
    """Raised when the selection is handed anything but training rows."""


def assert_train_only(
    results: pd.DataFrame, *, split_column: str = "evaluation_split"
) -> None:
    """Refuse to select from anything the held-out split has seen.

    A loud failure rather than a filter: silently dropping validation rows
    would let a caller believe it had asked for something it did not get, and
    the whole inferential argument rests on this one property.
    """

    if results is None or results.empty or split_column not in results.columns:
        return
    splits = {str(value) for value in results[split_column].dropna().unique()}
    intruders = splits - {"train"}
    if intruders:
        raise ValidationLeak(
            "Die Auswahl darf nur Trainingsergebnisse sehen; erhalten: "
            + ", ".join(sorted(intruders))
        )


def select_from_train(
    train_results: pd.DataFrame,
    *,
    redundancy: Optional[pd.DataFrame] = None,
    policy: TwoStagePolicy = TwoStagePolicy(),
    prespecified: Optional[Mapping[str, str]] = None,
    endpoint_column: str = "endpoint_id",
    feature_column: str = "feature_id",
    effect_column: str = "roc_auc",
    p_value_column: str = "p_value",
    kind_column: Optional[str] = None,
    default_target_kind: str = "binary",
    split_column: str = "evaluation_split",
) -> pd.DataFrame:
    """The markers each endpoint carries into confirmation, and why."""

    assert_train_only(train_results, split_column=split_column)
    if train_results is None or train_results.empty:
        return pd.DataFrame(columns=_SELECTION_COLUMNS)

    frame = train_results.copy()
    kind = (
        frame[kind_column].astype(str)
        if kind_column and kind_column in frame.columns
        else pd.Series(default_target_kind, index=frame.index)
    )
    effect = pd.to_numeric(frame[effect_column], errors="coerce")
    neutral = kind.map(neutral_effect)
    frame["_delta"] = (effect - neutral).abs()
    frame["_direction"] = np.sign(effect - neutral)
    frame["_floor"] = kind.map(policy.minimum_effect)
    frame["_p"] = pd.to_numeric(frame[p_value_column], errors="coerce")
    frame["_kind"] = kind

    # Candidates only. ``heart_rate`` and ``heart_rate_bpm`` carry the role
    # ``ecg_context_comparator`` -- they are what a finding is adjusted *for*,
    # not what it is about. Left in, they were ranked first for acute
    # myocardial infarction, which is the confounding the role system exists
    # to keep out of the candidate set.
    frame["_candidate"] = frame[feature_column].astype(str).map(
        is_candidate_biomarker
    )
    eligible = frame.loc[
        frame["_candidate"]
        & frame["_delta"].ge(frame["_floor"] - 1e-12)
        & frame["_p"].le(float(policy.train_alpha))
    ]

    cluster_of: dict[str, str] = {}
    if redundancy is not None and not redundancy.empty:
        cluster_of = dict(
            zip(
                redundancy["feature_id"].astype(str),
                redundancy["cluster_id"].astype(str),
            )
        )

    curated = dict(prespecified or {})
    records: list[dict] = []
    for endpoint, group in eligible.groupby(endpoint_column, sort=True):
        ordered = group.sort_values(
            ["_delta", "_p", feature_column], ascending=[False, True, True]
        )
        seen_clusters: set[str] = set()
        seen_statistics: set[tuple] = set()
        rank = 0
        for _, row in ordered.iterrows():
            feature = str(row[feature_column])
            cluster = cluster_of.get(feature, f"__{feature}")
            # Two markers whose screening statistics agree bit for bit did not
            # both get measured. Observed on the completed run:
            # ``conduction_detour_index`` and ``qrs_loop_tortuosity`` shared an
            # AUC, a p-value to seven digits, an n and a missingness across all
            # 63 endpoints -- while their values correlate at rho = 0.34 and
            # their formulas differ. Spending two confirmation slots on that
            # would confirm one marker twice under two names.
            fingerprint = (
                round(float(row["_delta"]), 12),
                round(float(row["_p"]), 15) if np.isfinite(row["_p"]) else None,
                int(row.get("n_total", -1)) if "n_total" in row else -1,
            )
            if fingerprint in seen_statistics:
                continue
            seen_statistics.add(fingerprint)
            if policy.deduplicate_by_cluster and cluster in seen_clusters:
                # A second name for a quantity already selected would spend a
                # confirmation slot on the same finding.
                continue
            seen_clusters.add(cluster)
            rank += 1
            records.append(
                {
                    "endpoint_id": str(endpoint),
                    "feature_id": feature,
                    "selection_rank": rank,
                    "train_effect": float(row[effect_column]),
                    "train_p_value": float(row["_p"]),
                    "train_direction": int(row["_direction"]),
                    "target_kind": str(row["_kind"]),
                    "cluster_id": cluster,
                    "n_screened_for_endpoint": int(len(group)),
                    "also_prespecified": curated.get(str(endpoint)) == feature,
                    "selection_rule": (
                        f"top-{policy.top_k} nach Effektstaerke auf Train, "
                        f"alpha={policy.train_alpha}, ein Vertreter je "
                        "Redundanzgruppe"
                    ),
                }
            )
            if rank >= int(policy.top_k):
                break
    if not records:
        return pd.DataFrame(columns=_SELECTION_COLUMNS)
    return pd.DataFrame(records).sort_values(
        ["endpoint_id", "selection_rank"], ignore_index=True
    )


_SELECTION_COLUMNS = (
    "endpoint_id",
    "feature_id",
    "selection_rank",
    "train_effect",
    "train_p_value",
    "train_direction",
    "target_kind",
    "cluster_id",
    "n_screened_for_endpoint",
    "also_prespecified",
    "selection_rule",
)


def selection_summary(
    selection: pd.DataFrame, prespecified: Optional[Mapping[str, str]] = None
) -> dict:
    """What the second arm adds over the curated one."""

    curated = dict(prespecified or {})
    if selection is None or selection.empty:
        return {
            "version": TWO_STAGE_VERSION,
            "n_selected": 0,
            "n_endpoints": 0,
            "n_markers": 0,
            "n_agreeing_with_curation": 0,
            "n_endpoints_without_curation": 0,
        }
    endpoints = set(selection["endpoint_id"].astype(str))
    return {
        "version": TWO_STAGE_VERSION,
        "n_selected": int(len(selection)),
        "n_endpoints": int(len(endpoints)),
        "n_markers": int(selection["feature_id"].nunique()),
        # Where the screen picks the marker a person curated, the curation is
        # corroborated -- worth counting, and free.
        "n_agreeing_with_curation": int(
            selection["also_prespecified"].astype(bool).sum()
        ),
        "n_endpoints_without_curation": int(
            len(endpoints - set(curated))
        ),
        "curated_endpoints": int(len(curated)),
    }


def confirmation_plan(
    selection: pd.DataFrame,
    *,
    prespecified: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Both arms in one table, each labelled with the arm it belongs to.

    The curated pairs are appended rather than merged: a reader must be able
    to tell a marker chosen before the data was seen from one the training
    split ranked, because only the first supports a confirmatory claim without
    a multiplicity argument.
    """

    rows: list[dict] = []
    for _, row in (
        selection.iterrows()
        if selection is not None and not selection.empty
        else []
    ):
        rows.append(
            {
                "endpoint_id": row["endpoint_id"],
                "feature_id": row["feature_id"],
                "arm": "two_stage_confirmatory",
                "selection_rank": int(row["selection_rank"]),
                "n_screened_for_endpoint": int(row["n_screened_for_endpoint"]),
            }
        )
    for endpoint, feature in dict(prespecified or {}).items():
        rows.append(
            {
                "endpoint_id": str(endpoint),
                "feature_id": str(feature),
                "arm": "prespecified_curated",
                "selection_rank": 0,
                "n_screened_for_endpoint": 0,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "endpoint_id",
                "feature_id",
                "arm",
                "selection_rank",
                "n_screened_for_endpoint",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["endpoint_id", "feature_id", "arm"]
    ).sort_values(["arm", "endpoint_id", "selection_rank"], ignore_index=True)


def run_two_stage_selection(
    config,
    *,
    run_id: str,
    policy: TwoStagePolicy = TwoStagePolicy(),
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Choose, from the training split alone, what deserves confirmation.

    The selection reads only ``evaluation_split == 'train'`` rows, and
    ``assert_train_only`` refuses the frame outright if anything else is in
    it. That refusal is the stage: everything else here is bookkeeping.

    The redundancy table, when the run has one, keeps two names for one
    quantity from spending two confirmation slots.
    """

    import pandas as pd

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        content_hash,
        scientific_config_hash,
    )

    run_root = config.runs_root / run_id
    output_root = run_root / "two_stage"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "two_stage_selection.manifest.json"

    source = run_root / "screening_results.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Die zweistufige Auswahl braucht die Screening-Ergebnisse: {source}"
        )
    redundancy_path = run_root / "redundancy" / "marker_redundancy.parquet"

    input_hash = content_hash(
        {
            "run_id": run_id,
            "source": str(source.resolve()),
            "source_mtime": source.stat().st_mtime_ns,
            "redundancy": (
                redundancy_path.stat().st_mtime_ns
                if redundancy_path.exists()
                else None
            ),
            "policy": policy.to_dict(),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, TWO_STAGE_VERSION):
            return previous

    if progress:
        progress("Zweistufige Auswahl: nur Trainingsergebnisse lesen")
    results = pd.read_parquet(source)
    if "evaluation_split" in results.columns:
        results = results.loc[results["evaluation_split"].astype(str).eq("train")]

    redundancy = (
        pd.read_parquet(redundancy_path) if redundancy_path.exists() else None
    )
    if redundancy is None and progress:
        progress(
            "Zweistufige Auswahl: keine Redundanztabelle -- ein Marker kann "
            "unter zwei Namen zwei Plaetze belegen"
        )

    from .validation import PRIMARY_ADJUSTED_MARKERS

    curated = dict(PRIMARY_ADJUSTED_MARKERS)
    selection = select_from_train(
        results, redundancy=redundancy, policy=policy, prespecified=curated
    )
    summary = selection_summary(selection, curated)
    summary["run_id"] = str(run_id)
    summary["policy"] = policy.to_dict()
    summary["had_redundancy_table"] = redundancy is not None
    plan = confirmation_plan(selection, prespecified=curated)

    if progress:
        progress(
            f"Zweistufige Auswahl: {summary['n_selected']} Paare ueber "
            f"{summary['n_endpoints']} Endpunkte, "
            f"{summary['n_endpoints_without_curation']} davon bisher unkuratiert"
        )

    selection_path = output_root / "two_stage_selection.parquet"
    plan_path = output_root / "confirmation_plan.parquet"
    summary_path = output_root / "two_stage_summary.json"
    atomic_write_parquet(selection, selection_path)
    atomic_write_parquet(plan, plan_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="two-stage-selection",
        stage_version=TWO_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(selection_path.resolve()),
            str(plan_path.resolve()),
            str(summary_path.resolve()),
        ),
        row_counts={
            "selected": int(len(selection)),
            "confirmation_pairs": int(len(plan)),
        },
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "run_two_stage_selection",
    "TWO_STAGE_TOP_K",
    "TWO_STAGE_VERSION",
    "TwoStagePolicy",
    "ValidationLeak",
    "assert_train_only",
    "confirmation_plan",
    "neutral_effect",
    "select_from_train",
    "selection_summary",
]
