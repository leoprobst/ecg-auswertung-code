"""Five arms, each testing several markers, none of them correcting.

The targeted designs were added one at a time and each judges its own markers
on their own p-values. Nothing looks across them. The within-subject arm tests
seven markers against potassium and seven more per drug class, the survival arm
four markers in two populations, the occlusion arm six markers, and the
voltage-mass arm four numerators against four endpoints in two populations. A
reader taking the smallest p-value from that collection is choosing from
several dozen tests without saying so.

This gathers what the arms already wrote and applies one Benjamini-Hochberg
correction across them, using the project's own
:func:`statistics.apply_fdr_by_family` rather than a second implementation.

**The comparison that matters is not the q-value.** It is the noise floor the
control panel measures: the largest effect produced by a pairing with *no
mechanism*, in this cohort, at this sample size. An FDR calculation asks
whether a p-value is small for the number of tests; the noise floor asks
whether an effect is large for a population where almost everything tracks
being ill. A finding can pass the first and fail the second, and the second is
the harder question. Both are reported side by side for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd


TARGETED_MULTIPLICITY_VERSION = "1.0.0"

# Where each arm writes, and which column carries its p-value. The effect
# column differs by arm -- a rank correlation, a hazard ratio, an AUC -- so it
# is named per arm rather than assumed.
TARGETED_ARMS: Mapping[str, Mapping[str, str]] = {
    "within_subject": {
        "path": "within_subject/within_subject_change.parquet",
        "p_value": "p_value",
        "effect": "spearman_rho",
        "marker": "feature_id",
        "neutral": "0",
    },
    "time_to_event": {
        "path": "survival/time_to_event.parquet",
        "p_value": "p_value",
        "effect": "hazard_ratio_per_sd",
        "marker": "feature_id",
        "neutral": "1",
    },
}


@dataclass(frozen=True)
class MultiplicityPolicy:
    alpha: float = 0.05

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "version": TARGETED_MULTIPLICITY_VERSION,
        }


def collect_arm_results(run_root: Path) -> pd.DataFrame:
    """Every formal test the targeted arms wrote, in one frame.

    Only the arms that report a p-value take part. The occlusion and
    voltage-mass arms report AUCs without one, so correcting them here would
    mean inventing an uncertainty they never computed -- they are listed as
    uncorrected instead of quietly folded in.
    """

    columns = ["arm", "feature_id", "context", "effect", "p_value"]
    records: list[pd.DataFrame] = []
    for arm, spec in TARGETED_ARMS.items():
        path = Path(run_root) / spec["path"]
        if not path.is_file():
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - a broken artefact must not stop it
            continue
        if frame.empty or spec["p_value"] not in frame.columns:
            continue
        block = pd.DataFrame(
            {
                "arm": arm,
                "feature_id": frame[spec["marker"]].astype(str),
                "context": (
                    frame["design"].astype(str)
                    if "design" in frame.columns
                    else frame["population"].astype(str)
                    if "population" in frame.columns
                    else ""
                ),
                "effect": pd.to_numeric(frame[spec["effect"]], errors="coerce"),
                "p_value": pd.to_numeric(frame[spec["p_value"]], errors="coerce"),
            }
        )
        records.append(block)
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True)


def correct_across_arms(
    results: pd.DataFrame,
    *,
    policy: MultiplicityPolicy = MultiplicityPolicy(),
) -> pd.DataFrame:
    """One Benjamini-Hochberg family over every targeted test.

    A single family on purpose. Correcting inside each arm would leave the
    choice *between* arms uncorrected, which is where the selection actually
    happens: a reader who reports the best of five designs has chosen from
    five designs.
    """

    from .statistics import apply_fdr_by_family

    if results is None or results.empty:
        return pd.DataFrame(
            columns=[
                "arm", "feature_id", "context", "effect", "p_value",
                "q_value", "fdr_significant",
            ]
        )
    frame = results.copy()
    frame["_family"] = "targeted_designs"
    corrected = apply_fdr_by_family(
        frame,
        family_columns=("_family",),
        p_column="p_value",
        multiplicity_sensitivity=False,
    )
    return corrected.drop(columns=["_family"]).sort_values(
        "p_value", ignore_index=True
    )


def multiplicity_summary(
    corrected: pd.DataFrame,
    *,
    noise_floor: Optional[float] = None,
    policy: MultiplicityPolicy = MultiplicityPolicy(),
) -> dict:
    """What survives the correction, and what the noise floor says about it."""

    if corrected is None or corrected.empty:
        return {
            "version": TARGETED_MULTIPLICITY_VERSION,
            "n_tests": 0,
            "n_significant": 0,
        }
    significant = corrected.loc[corrected["fdr_significant"].astype(bool)]
    summary = {
        "version": TARGETED_MULTIPLICITY_VERSION,
        "policy": policy.to_dict(),
        "n_tests": int(len(corrected)),
        "n_arms": int(corrected["arm"].nunique()),
        "n_significant": int(len(significant)),
        "smallest_q": (
            float(corrected["q_value"].min())
            if corrected["q_value"].notna().any()
            else float("nan")
        ),
    }
    if noise_floor is not None and np.isfinite(noise_floor):
        summary["confounding_noise_floor"] = float(noise_floor)
        # An AUC-scale floor cannot be compared with a rank correlation
        # directly, so the comparison is stated rather than computed: the
        # floor is what an effect must clear to be distinguishable from
        # confounding in this cohort, whatever scale it is reported on.
        summary["noise_floor_note"] = (
            "Der Rauschboden stammt aus Paarungen ohne Mechanismus und ist "
            "auf der AUC-Skala gemessen. Ein Effekt unterhalb davon laesst "
            "sich in dieser Kohorte nicht von Konfundierung trennen, "
            "unabhaengig vom q-Wert."
        )
    return summary


__all__ = [
    "MultiplicityPolicy",
    "TARGETED_ARMS",
    "TARGETED_MULTIPLICITY_VERSION",
    "collect_arm_results",
    "correct_across_arms",
    "multiplicity_summary",
]
