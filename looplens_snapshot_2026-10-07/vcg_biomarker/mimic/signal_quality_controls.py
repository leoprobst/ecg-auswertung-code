"""Screen the quality markers on purpose, and let them police the rest.

The broad screen found, in the last development run, that the strongest
predictor of *major depressive disorder within 30 days* was
``einthoven_relation_error_rms`` -- how badly leads I, II and III satisfy
Einthoven's law -- at rank correlation 0.51, with ``vcg_trust_score`` right
behind it at -0.50.  The same pair led hypomagnesaemia (0.47 / -0.47) and coma
(0.33 / -0.31).  Neither is physiology.  Both measure how cleanly the recording
was taken.

The reading is not that the finding is worthless; agitated, tremulous or
uncooperative patients plausibly do produce noisier recordings, and that is
itself informative.  The reading is that for those phenotypes **acquisition
quality explains more than any physiological marker does**, and a screen that
does not say so invites every physiological marker that happens to correlate
with quality to be reported as a discovery.

The quality markers were subsequently taken out of the analysis panels.  That
stops them being reported as findings -- and also removes the only instrument
that could have detected the problem.  This module puts them back deliberately,
as a **declared negative-control family**:

* they are screened against every target, in their own FDR family, so they
  neither borrow power from nor lend power to the candidate markers;
* they are never promotable, whatever their effect size;
* for every target, the strongest quality association is compared with the
  strongest candidate association, and a target where quality wins is marked
  ``technically_dominated``.

A negative control that is allowed to win is the only kind worth having.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .feature_roles import METHODOLOGICAL_CONTROL_FEATURE_IDS

QUALITY_CONTROL_VERSION = "1.0.0"

#: Panel identifier the control family is screened under.  Keeping it distinct
#: is what keeps it out of the candidate markers' FDR families.
QUALITY_CONTROL_PANEL_ID = "quality_control"

#: A target whose best quality association is at least this share of its best
#: candidate association is reported as technically influenced even when the
#: candidate still wins.  Chosen before the first comparison was run.
TECHNICAL_INFLUENCE_SHARE = 0.8

VERDICT_DOMINATED = "technically_dominated"
VERDICT_INFLUENCED = "technically_influenced"
VERDICT_CLEAN = "no_technical_dominance"
VERDICT_UNKNOWN = "not_assessable"


def quality_control_features(available_columns: Iterable[str]) -> tuple[str, ...]:
    """The control family, restricted to what the extraction actually wrote."""

    columns = set(str(name) for name in available_columns)
    return tuple(
        sorted(name for name in METHODOLOGICAL_CONTROL_FEATURE_IDS if name in columns)
    )


@dataclass(frozen=True)
class TargetVerdict:
    target_id: str
    verdict: str
    best_quality_feature: Optional[str]
    best_quality_effect: float
    best_candidate_feature: Optional[str]
    best_candidate_effect: float
    quality_share: float


def _effect_magnitude(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if column.endswith("auc"):
        # An AUC of 0.5 is no association; distance from it is the effect.
        return (values - 0.5).abs()
    return values.abs()


def compare_quality_against_candidates(
    hits: pd.DataFrame,
    *,
    target_column: str,
    feature_column: str = "feature_id",
    effect_column: str = "validation_aligned_auc",
    quality_features: Sequence[str] = (),
) -> pd.DataFrame:
    """One verdict per target: did a quality marker beat every candidate?"""

    columns = [
        "target_id",
        "verdict",
        "best_quality_feature",
        "best_quality_effect",
        "best_candidate_feature",
        "best_candidate_effect",
        "quality_share",
        "quality_control_version",
    ]
    if hits.empty or effect_column not in hits.columns:
        return pd.DataFrame(columns=columns)

    quality = set(str(name) for name in quality_features)
    working = hits[[target_column, feature_column, effect_column]].copy()
    working["_effect"] = _effect_magnitude(working, effect_column)
    working = working[np.isfinite(working["_effect"])]
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["_is_quality"] = working[feature_column].astype(str).isin(quality)

    rows: list[dict] = []
    for target_id, group in working.groupby(target_column, sort=True):
        quality_rows = group[group["_is_quality"]]
        candidate_rows = group[~group["_is_quality"]]
        best_quality = (
            quality_rows.loc[quality_rows["_effect"].idxmax()]
            if not quality_rows.empty
            else None
        )
        best_candidate = (
            candidate_rows.loc[candidate_rows["_effect"].idxmax()]
            if not candidate_rows.empty
            else None
        )
        quality_effect = (
            float(best_quality["_effect"]) if best_quality is not None else float("nan")
        )
        candidate_effect = (
            float(best_candidate["_effect"])
            if best_candidate is not None
            else float("nan")
        )
        if best_quality is None or best_candidate is None:
            verdict = VERDICT_UNKNOWN
            share = float("nan")
        else:
            share = (
                quality_effect / candidate_effect
                if candidate_effect > 0
                else float("inf")
            )
            if quality_effect >= candidate_effect:
                verdict = VERDICT_DOMINATED
            elif share >= TECHNICAL_INFLUENCE_SHARE:
                verdict = VERDICT_INFLUENCED
            else:
                verdict = VERDICT_CLEAN
        rows.append(
            {
                "target_id": str(target_id),
                "verdict": verdict,
                "best_quality_feature": (
                    str(best_quality[feature_column])
                    if best_quality is not None
                    else None
                ),
                "best_quality_effect": quality_effect,
                "best_candidate_feature": (
                    str(best_candidate[feature_column])
                    if best_candidate is not None
                    else None
                ),
                "best_candidate_effect": candidate_effect,
                "quality_share": share,
                "quality_control_version": QUALITY_CONTROL_VERSION,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def annotate_hits_with_quality_verdict(
    hits: pd.DataFrame,
    verdicts: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    """Carry the verdict onto every hit, so a reader cannot miss it."""

    if hits.empty:
        hits["quality_control_verdict"] = pd.Series(dtype="object")
        hits["quality_control_share"] = pd.Series(dtype="float64")
        return hits
    if verdicts.empty:
        hits["quality_control_verdict"] = VERDICT_UNKNOWN
        hits["quality_control_share"] = float("nan")
        return hits
    lookup = verdicts.set_index("target_id")
    keys = hits[target_column].astype(str)
    hits["quality_control_verdict"] = (
        keys.map(lookup["verdict"]).fillna(VERDICT_UNKNOWN)
    )
    hits["quality_control_share"] = keys.map(lookup["quality_share"])
    return hits
