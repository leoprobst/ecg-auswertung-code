"""Calibrating a ten-second surrogate against the marker it stands in for.

Periodic repolarisation dynamics and deceleration capacity are among the
better-evidenced electrocardiographic risk markers, and neither can be
computed from a ten-second recording. Both need at least five minutes: PRD is
the low-frequency component of the beat-to-beat repolarisation angle series,
and at 0,1 Hz a ten-second strip holds one cycle. A ten-second "PRD" is not a
weak PRD; it is a different quantity that happens to share a name.

There are two honest ways to handle that. Declare the family unreachable, or
calibrate a surrogate against the real thing in the same patients. This module
is the second, and it is written so that the answer may be no.

**Why the MIMIC-IV Waveform Database and not a bigger source.** Version 0.1.0
holds 200 recordings from 198 patients -- far too few to screen anything, and
that is not what it is for. Its value is that the recordings are linked by
``subject_id`` and ``hadm_id`` to the same MIMIC-IV clinical release this
project already uses, so for a patient with both a continuous recording and a
ten-second diagnostic electrocardiogram, the five-minute gold standard and the
ten-second surrogate can be computed **in the same person, hours apart**. No
other available source offers that pairing.

**What a result looks like.** Either a calibration line with a confidence band
and a stated agreement -- in which case the surrogate may be applied to all
630.044 recordings with that band attached -- or the finding that the two do
not track each other, in which case the marker family is declared unreachable
in this project and the declaration is itself a result. Both are written down.
The failure mode this design exists to prevent is the third outcome: computing
a ten-second number, naming it PRD, and never checking.

**The pairing rule is the part that can quietly ruin it.** A continuous
recording and a diagnostic strip taken three days apart in an intensive-care
patient are not two measurements of one state. The window is declared here,
not chosen after seeing the agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


LONG_RECORDING_CALIBRATION_VERSION = "1.1.0"

# The published minimum for periodic repolarisation dynamics. Below it the
# low-frequency band the marker is defined on is not resolved.
MINIMUM_GOLD_STANDARD_SECONDS = 300.0

# How far apart the continuous recording and the diagnostic strip may be and
# still describe one state. Declared before the agreement is seen.
MAXIMUM_PAIRING_HOURS = 6.0

# Below this many pairs a calibration line is not worth drawing, and the
# stage says so rather than drawing one.
MINIMUM_CALIBRATION_PAIRS = 30

# The agreement a surrogate must reach before it may stand in for the gold
# standard. Deliberately set where a reader would set it in advance: below
# this the surrogate carries less than half the variance of the target.
MINIMUM_SURROGATE_CORRELATION = 0.70


@dataclass(frozen=True)
class CalibrationPolicy:
    minimum_gold_standard_seconds: float = MINIMUM_GOLD_STANDARD_SECONDS
    maximum_pairing_hours: float = MAXIMUM_PAIRING_HOURS
    minimum_pairs: int = MINIMUM_CALIBRATION_PAIRS
    minimum_correlation: float = MINIMUM_SURROGATE_CORRELATION
    bootstrap_iterations: int = 1000
    random_seed: int = 20260831

    def to_dict(self) -> dict:
        return {
            "minimum_gold_standard_seconds": float(
                self.minimum_gold_standard_seconds
            ),
            "maximum_pairing_hours": float(self.maximum_pairing_hours),
            "minimum_pairs": int(self.minimum_pairs),
            "minimum_correlation": float(self.minimum_correlation),
            "bootstrap_iterations": int(self.bootstrap_iterations),
            "random_seed": int(self.random_seed),
            "version": LONG_RECORDING_CALIBRATION_VERSION,
        }


def pair_recordings(
    long_recordings: pd.DataFrame,
    diagnostic_recordings: pd.DataFrame,
    policy: CalibrationPolicy = CalibrationPolicy(),
) -> pd.DataFrame:
    """Match each continuous recording to the nearest diagnostic strip.

    One pair per continuous recording, and one continuous recording per
    subject: with 198 patients a subject contributing several pairs would
    dominate a correlation computed over a few dozen.
    """

    columns = [
        "subject_id",
        "long_record_id",
        "study_id",
        "delta_hours",
        "long_duration_seconds",
    ]
    if long_recordings.empty or diagnostic_recordings.empty:
        return pd.DataFrame(columns=columns)

    long_frame = long_recordings.copy()
    long_frame = long_frame[
        pd.to_numeric(long_frame["duration_seconds"], errors="coerce")
        >= float(policy.minimum_gold_standard_seconds)
    ]
    if long_frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    diagnostic = diagnostic_recordings.copy()
    diagnostic["ecg_time"] = pd.to_datetime(diagnostic["ecg_time"])
    long_frame["record_time"] = pd.to_datetime(long_frame["record_time"])
    for subject, block in long_frame.groupby("subject_id"):
        strips = diagnostic[diagnostic["subject_id"] == subject]
        if strips.empty:
            continue
        best: Optional[dict] = None
        for long_row in block.itertuples(index=False):
            deltas = (
                strips["ecg_time"] - long_row.record_time
            ).dt.total_seconds().abs() / 3600.0
            index = int(deltas.to_numpy().argmin())
            delta = float(deltas.iloc[index])
            if delta > float(policy.maximum_pairing_hours):
                continue
            candidate = {
                "subject_id": int(subject),
                "long_record_id": str(long_row.record_id),
                "study_id": int(strips.iloc[index]["study_id"]),
                "delta_hours": delta,
                "long_duration_seconds": float(long_row.duration_seconds),
            }
            if best is None or candidate["delta_hours"] < best["delta_hours"]:
                best = candidate
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows, columns=columns)


def calibrate_surrogate(
    gold_standard: Sequence[float],
    surrogate: Sequence[float],
    *,
    marker_id: str,
    policy: CalibrationPolicy = CalibrationPolicy(),
) -> dict:
    """Fit the surrogate to the gold standard, and say whether it holds.

    The verdict is the point. ``carries`` is true only when there are enough
    pairs and the correlation clears the floor declared in advance; otherwise
    the marker family is reported as not reachable from ten seconds, which is
    a result and not a failure.
    """

    from scipy import stats

    gold = np.asarray(gold_standard, dtype=float)
    proxy = np.asarray(surrogate, dtype=float)
    both = np.isfinite(gold) & np.isfinite(proxy)
    gold = gold[both]
    proxy = proxy[both]
    result = {
        "marker_id": marker_id,
        "pairs": int(gold.size),
        "minimum_pairs": int(policy.minimum_pairs),
        "minimum_correlation": float(policy.minimum_correlation),
        "calibration_version": LONG_RECORDING_CALIBRATION_VERSION,
    }
    if gold.size < policy.minimum_pairs:
        result.update(
            {
                "carries": False,
                "verdict": "too_few_pairs",
                "pearson_r": float("nan"),
                "pearson_ci_low": float("nan"),
                "pearson_ci_high": float("nan"),
                "spearman_rho": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
                "residual_sd": float("nan"),
            }
        )
        return result
    if np.std(gold) <= 0 or np.std(proxy) <= 0:
        result.update(
            {
                "carries": False,
                "verdict": "no_variation_in_one_arm",
                "pearson_r": float("nan"),
                "pearson_ci_low": float("nan"),
                "pearson_ci_high": float("nan"),
                "spearman_rho": float("nan"),
                "slope": float("nan"),
                "intercept": float("nan"),
                "residual_sd": float("nan"),
            }
        )
        return result

    pearson = float(np.corrcoef(proxy, gold)[0, 1])
    rho = float(stats.spearmanr(proxy, gold).statistic)
    slope, intercept = np.polyfit(proxy, gold, 1)
    residuals = gold - (slope * proxy + intercept)
    rng = np.random.default_rng(policy.random_seed)
    draws = rng.integers(0, gold.size, size=(policy.bootstrap_iterations, gold.size))
    samples = np.array(
        [float(np.corrcoef(proxy[draw], gold[draw])[0, 1]) for draw in draws]
    )
    samples = samples[np.isfinite(samples)]
    carries = bool(pearson >= policy.minimum_correlation)
    result.update(
        {
            "carries": carries,
            "verdict": (
                "surrogate_calibrated"
                if carries
                else "surrogate_does_not_track_the_gold_standard"
            ),
            "pearson_r": pearson,
            "pearson_ci_low": (
                float(np.percentile(samples, 2.5)) if samples.size else float("nan")
            ),
            "pearson_ci_high": (
                float(np.percentile(samples, 97.5)) if samples.size else float("nan")
            ),
            "spearman_rho": rho,
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_sd": float(np.std(residuals, ddof=2))
            if gold.size > 2
            else float("nan"),
        }
    )
    return result


def calibration_declaration(results: Sequence[dict]) -> pd.DataFrame:
    """What may be claimed about each surrogate family after calibration."""

    rows = []
    for result in results:
        carries = bool(result.get("carries", False))
        rows.append(
            {
                "marker_id": result.get("marker_id"),
                "pairs": result.get("pairs"),
                "pearson_r": result.get("pearson_r"),
                "verdict": result.get("verdict"),
                "permitted_use": (
                    "may be applied to the full cohort with the calibration "
                    "band attached"
                    if carries
                    else "may not stand in for the five-minute marker; the "
                    "family is declared not reachable from ten seconds"
                ),
                "calibration_version": LONG_RECORDING_CALIBRATION_VERSION,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "marker_id",
            "pairs",
            "pearson_r",
            "verdict",
            "permitted_use",
            "calibration_version",
        ],
    )


def waveform_database_status(root: Optional[Path | str]) -> dict:
    """Whether the long-recording source is present, and what it is for."""

    note = (
        "MIMIC-IV Waveform Database v0.1.0 holds 200 recordings from 198 "
        "patients. Too small to screen; used only to calibrate the "
        "ten-second surrogate against the five-minute gold standard in the "
        "same patients."
    )
    if root is None:
        return {
            "present": False,
            "root": None,
            "record_count": 0,
            "note": note,
            "reason": "no root configured",
        }
    base = Path(root)
    if not base.is_dir():
        return {
            "present": False,
            "root": str(base),
            "record_count": 0,
            "note": note,
            "reason": f"configured root does not exist: {base}",
        }
    headers = list(base.rglob("*.hea"))
    return {
        "present": bool(headers),
        "root": str(base),
        "record_count": len(headers),
        "note": note,
        "reason": "present" if headers else "root exists but holds no records",
    }


__all__ = [
    "LONG_RECORDING_CALIBRATION_VERSION",
    "MAXIMUM_PAIRING_HOURS",
    "MINIMUM_CALIBRATION_PAIRS",
    "MINIMUM_GOLD_STANDARD_SECONDS",
    "MINIMUM_SURROGATE_CORRELATION",
    "CalibrationPolicy",
    "calibrate_surrogate",
    "calibration_declaration",
    "pair_recordings",
    "waveform_database_status",
]
