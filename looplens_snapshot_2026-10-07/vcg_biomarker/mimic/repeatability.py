"""Measurement repeatability of ECG/VCG markers, from repeated recordings.

Every claim this project can make rests on markers whose measurement error is
unknown. The AHA evidence ladder puts analytical validity below any clinical
statement for exactly that reason, and the criterion catalog already carries
the open limitation "technische Wiederholbarkeit muss separat quantifiziert
werden, bevor ein Freeze erlaubt ist". This module closes that gap using data
already in the cohort: 104,008 of 161,352 subjects have at least two ECGs.

The idea is to treat two recordings taken close together, with no intervening
intervention, as repeat measurements of the same underlying physiological
state. That is an assumption, not a fact -- physiology moves on its own -- so
the quantities below are *upper* bounds on measurement error: anything real
that changed between the two recordings inflates them. They are reported as
such, never as a pure device-level precision.

Quantities produced, all on the marker's own scale:

* within-subject standard deviation ``s_w``;
* intraclass correlation ICC(A,1), two-way random effects, absolute
  agreement -- the share of total variance that is genuine between-subject
  signal;
* repeatability coefficient ``RC = 2.77 * s_w`` (Bland-Altman): the interval
  within which 95 % of differences between two measurements fall;
* smallest detectable change ``SDC = 1.96 * sqrt(2) * s_w``: a change in one
  patient below this is indistinguishable from measurement noise;
* Bland-Altman bias and limits of agreement.

A marker whose SDC exceeds the effect size it is meant to detect cannot
support an individual-patient claim, however good its AUC looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


REPEATABILITY_VERSION = "1.2.0"  # X8: gespeicherte Signal-Dubletten entfernt

# Bland-Altman repeatability coefficient factor: 1.96 * sqrt(2).
REPEATABILITY_COEFFICIENT_FACTOR = 2.7718585822512662
SMALLEST_DETECTABLE_CHANGE_FACTOR = 2.7718585822512662

# A marker needs at least this many replicate pairs before its repeatability
# is reported as anything other than descriptive.
MINIMUM_REPLICATE_PAIRS = 30
# ICC below this means most of the observed spread is measurement noise.
MINIMUM_RELIABLE_ICC = 0.75


@dataclass(frozen=True)
class ReplicatePairPolicy:
    """What counts as a technical replicate rather than a real change."""

    maximum_hours_between: float = 2.0
    minimum_hours_between: float = 0.0
    require_same_admission: bool = True
    require_no_intervening_procedure: bool = True
    require_same_care_setting: bool = True
    minimum_signal_quality: float = 0.6

    def __post_init__(self) -> None:
        if self.maximum_hours_between <= 0:
            raise ValueError("maximum_hours_between must be positive")
        if self.minimum_hours_between < 0:
            raise ValueError("minimum_hours_between must not be negative")
        if self.minimum_hours_between >= self.maximum_hours_between:
            raise ValueError(
                "minimum_hours_between must be below maximum_hours_between"
            )
        if not 0.0 <= self.minimum_signal_quality <= 1.0:
            raise ValueError("minimum_signal_quality must be within [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "maximum_hours_between": self.maximum_hours_between,
            "minimum_hours_between": self.minimum_hours_between,
            "require_same_admission": self.require_same_admission,
            "require_no_intervening_procedure": (
                self.require_no_intervening_procedure
            ),
            "require_same_care_setting": self.require_same_care_setting,
            "minimum_signal_quality": self.minimum_signal_quality,
        }


def within_subject_variance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Within-subject variance from paired replicate measurements.

    For duplicate measurements the classical estimator is the mean squared
    difference halved: each difference has variance ``2 * sigma_w^2``.
    """

    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    difference = difference[np.isfinite(difference)]
    if len(difference) == 0:
        return float("nan")
    return float(np.mean(difference ** 2) / 2.0)


def intraclass_correlation_absolute_agreement(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    """ICC(A,1) for paired measurements, two-way random, absolute agreement.

    Absolute agreement -- not consistency -- because a systematic offset
    between two recordings of the same patient is measurement error here, not
    a nuisance to be partialled out.
    """

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]
    n_subjects = int(len(first_array))
    empty = {
        "icc": float("nan"),
        "n_pairs": n_subjects,
        "between_subject_variance": float("nan"),
        "within_subject_variance": float("nan"),
        "systematic_difference_variance": float("nan"),
    }
    if n_subjects < 3:
        return empty
    matrix = np.column_stack((first_array, second_array))
    n_raters = 2
    subject_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    grand_mean = float(matrix.mean())
    ms_rows = (
        n_raters
        * float(np.sum((subject_means - grand_mean) ** 2))
        / (n_subjects - 1)
    )
    ms_columns = (
        n_subjects
        * float(np.sum((rater_means - grand_mean) ** 2))
        / (n_raters - 1)
    )
    residual = (
        matrix
        - subject_means[:, None]
        - rater_means[None, :]
        + grand_mean
    )
    denominator = (n_subjects - 1) * (n_raters - 1)
    ms_error = float(np.sum(residual ** 2)) / denominator if denominator else np.nan
    if not np.isfinite(ms_error):
        return empty
    numerator = ms_rows - ms_error
    denominator_terms = (
        ms_rows
        + (n_raters - 1) * ms_error
        + n_raters * (ms_columns - ms_error) / n_subjects
    )
    icc = (
        float(numerator / denominator_terms)
        if abs(denominator_terms) > 1e-12
        else float("nan")
    )
    return {
        "icc": float(np.clip(icc, -1.0, 1.0)) if np.isfinite(icc) else float("nan"),
        "n_pairs": n_subjects,
        "between_subject_variance": float(max(numerator / n_raters, 0.0)),
        "within_subject_variance": float(ms_error),
        "systematic_difference_variance": float(
            max((ms_columns - ms_error) / n_subjects, 0.0)
        ),
    }


def bland_altman(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    """Bias and 95 % limits of agreement between paired measurements."""

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    difference = first_array[valid] - second_array[valid]
    mean_value = (first_array[valid] + second_array[valid]) / 2.0
    if len(difference) < 2:
        return {
            "bias": float("nan"),
            "difference_sd": float("nan"),
            "loa_low": float("nan"),
            "loa_high": float("nan"),
            "proportional_bias_slope": float("nan"),
            "mean_measurement": float("nan"),
        }
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    slope = float("nan")
    if np.ptp(mean_value) > 0:
        slope = float(np.polyfit(mean_value, difference, 1)[0])
    return {
        "bias": bias,
        "difference_sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "proportional_bias_slope": slope,
        "mean_measurement": float(np.mean(mean_value)),
    }


def marker_repeatability(
    first: Iterable[float],
    second: Iterable[float],
    *,
    marker_id: str = "",
    minimum_pairs: int = MINIMUM_REPLICATE_PAIRS,
    minimum_icc: float = MINIMUM_RELIABLE_ICC,
) -> dict[str, object]:
    """Full repeatability record for one marker from replicate pairs."""

    first_array = np.asarray(list(first), dtype=float)
    second_array = np.asarray(list(second), dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("Replicate arrays must have matching shapes")
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    paired_first = first_array[valid]
    paired_second = second_array[valid]
    n_pairs = int(len(paired_first))
    record: dict[str, object] = {
        "marker_id": str(marker_id),
        "n_pairs_input": int(len(first_array)),
        "n_pairs_evaluable": n_pairs,
        "pair_test_yield": (
            float(n_pairs / len(first_array)) if len(first_array) else float("nan")
        ),
        "repeatability_version": REPEATABILITY_VERSION,
        "estimand": (
            "upper_bound_on_measurement_error_includes_true_short_term_"
            "physiological_variation"
        ),
    }
    if n_pairs < 3:
        record.update(
            {
                "status": "not_evaluable",
                "within_subject_sd": float("nan"),
                "icc": float("nan"),
                "repeatability_coefficient": float("nan"),
                "smallest_detectable_change": float("nan"),
                "reliable_for_individual_change": False,
                "inference_status": "descriptive_only",
            }
        )
        return record
    variance = within_subject_variance(paired_first, paired_second)
    s_w = float(np.sqrt(variance)) if np.isfinite(variance) else float("nan")
    icc_result = intraclass_correlation_absolute_agreement(
        paired_first,
        paired_second,
    )
    agreement = bland_altman(paired_first, paired_second)
    pooled_mean = float(np.mean(np.concatenate((paired_first, paired_second))))
    pooled_sd = float(
        np.std(np.concatenate((paired_first, paired_second)), ddof=1)
    )
    record.update(
        {
            "status": "evaluable",
            "within_subject_sd": s_w,
            "within_subject_variance": variance,
            "icc": icc_result["icc"],
            "repeatability_coefficient": (
                REPEATABILITY_COEFFICIENT_FACTOR * s_w
                if np.isfinite(s_w)
                else float("nan")
            ),
            "smallest_detectable_change": (
                SMALLEST_DETECTABLE_CHANGE_FACTOR * s_w
                if np.isfinite(s_w)
                else float("nan")
            ),
            "coefficient_of_variation": (
                float(s_w / abs(pooled_mean))
                if np.isfinite(s_w) and abs(pooled_mean) > 1e-12
                else float("nan")
            ),
            "measurement_error_share_of_total_sd": (
                float(s_w / pooled_sd)
                if np.isfinite(s_w) and pooled_sd > 1e-12
                else float("nan")
            ),
            "pooled_mean": pooled_mean,
            "pooled_sd": pooled_sd,
            "inference_status": (
                "formal_inference"
                if n_pairs >= int(minimum_pairs)
                else "underpowered"
            ),
        }
    )
    record.update({f"bland_altman_{key}": value for key, value in agreement.items()})
    icc_value = record["icc"]
    record["reliable_for_individual_change"] = bool(
        n_pairs >= int(minimum_pairs)
        and isinstance(icc_value, float)
        and np.isfinite(icc_value)
        and icc_value >= float(minimum_icc)
    )
    record["reliability_gate_reason"] = (
        ""
        if record["reliable_for_individual_change"]
        else (
            "insufficient_replicate_pairs"
            if n_pairs < int(minimum_pairs)
            else "icc_below_threshold"
        )
    )
    return record


REPEATABILITY_COLUMN_BLOCK = 200


def repeatability_table_from_parquet(
    pairs_path: "Path | str",
    marker_ids: Iterable[str],
    *,
    block_size: int = REPEATABILITY_COLUMN_BLOCK,
    first_suffix: str = "_first",
    second_suffix: str = "_second",
    minimum_pairs: int = MINIMUM_REPLICATE_PAIRS,
    minimum_icc: float = MINIMUM_RELIABLE_ICC,
) -> pd.DataFrame:
    """Repeatability read in column blocks instead of one wide frame.

    The replicate-pair artifact is the widest table this project writes: two
    columns per marker, 2 x 1,753 = 3,506 of them. Reading it whole put every
    pair times every marker into memory at once, on a stage that had never
    run.

    Exact rather than approximate: ``marker_repeatability`` consumes exactly
    one marker's two columns, so which markers share a read cannot change a
    single number. Only the peak moves.
    """

    import pyarrow.parquet as pq

    from .trajectory import replicate_pair_parts

    parts = replicate_pair_parts(pairs_path)
    if not parts:
        return repeatability_table(pd.DataFrame(), ())
    # Each part carries the same pair keys plus its own marker block, so the
    # column a marker lives in is found by asking the parts, not by assuming.
    part_of: dict[str, "Path"] = {}
    available: set[str] = set()
    for part in parts:
        names = set(pq.ParquetFile(part).schema_arrow.names)
        available |= names
        for name in names:
            part_of.setdefault(name, part)
    wanted = [
        str(marker)
        for marker in marker_ids
        if f"{marker}{first_suffix}" in available
        and f"{marker}{second_suffix}" in available
    ]
    if not wanted:
        return repeatability_table(pd.DataFrame(), ())

    records: list[dict[str, object]] = []
    step = max(1, int(block_size))
    grouped: dict["Path", list[str]] = {}
    for marker in wanted:
        grouped.setdefault(part_of[f"{marker}{first_suffix}"], []).append(marker)
    ordered_blocks: list[list[str]] = []
    for markers in grouped.values():
        for start in range(0, len(markers), step):
            ordered_blocks.append(markers[start : start + step])
    from .grant_parallel import grant_parallel_map

    # Audit R15: blocks are independent and the table is sorted afterwards.
    for block_records in grant_parallel_map(
        _repeatability_block,
        [
            (
                str(part_of[f"{block[0]}{first_suffix}"]),
                block,
                first_suffix,
                second_suffix,
                minimum_pairs,
                minimum_icc,
            )
            for block in ordered_blocks
        ],
    ):
        records.extend(block_records)
    return pd.DataFrame(records).sort_values("marker_id", ignore_index=True)


def _repeatability_block(
    source: str,
    block: list[str],
    first_suffix: str,
    second_suffix: str,
    minimum_pairs: int,
    minimum_icc: float,
) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    columns = [
        f"{marker}{suffix}"
        for marker in block
        for suffix in (first_suffix, second_suffix)
    ]
    frame = pq.read_table(source, columns=columns).to_pandas()
    return [
        marker_repeatability(
            pd.to_numeric(frame[f"{marker}{first_suffix}"], errors="coerce"),
            pd.to_numeric(frame[f"{marker}{second_suffix}"], errors="coerce"),
            marker_id=str(marker),
            minimum_pairs=minimum_pairs,
            minimum_icc=minimum_icc,
        )
        for marker in block
    ]


def repeatability_table(
    pairs: pd.DataFrame,
    marker_ids: Iterable[str],
    *,
    first_suffix: str = "_first",
    second_suffix: str = "_second",
    minimum_pairs: int = MINIMUM_REPLICATE_PAIRS,
    minimum_icc: float = MINIMUM_RELIABLE_ICC,
) -> pd.DataFrame:
    """Repeatability for every marker present in a replicate-pair frame."""

    records: list[dict[str, object]] = []
    for marker_id in marker_ids:
        first_column = f"{marker_id}{first_suffix}"
        second_column = f"{marker_id}{second_suffix}"
        if first_column not in pairs or second_column not in pairs:
            continue
        records.append(
            marker_repeatability(
                pd.to_numeric(pairs[first_column], errors="coerce"),
                pd.to_numeric(pairs[second_column], errors="coerce"),
                marker_id=str(marker_id),
                minimum_pairs=minimum_pairs,
                minimum_icc=minimum_icc,
            )
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "status",
                "n_pairs_evaluable",
                "within_subject_sd",
                "icc",
                "repeatability_coefficient",
                "smallest_detectable_change",
                "reliable_for_individual_change",
            ]
        )
    return pd.DataFrame(records).sort_values(
        "marker_id",
        kind="mergesort",
        ignore_index=True,
    )


def change_exceeds_measurement_error(
    change: Iterable[float],
    smallest_detectable_change: float,
) -> np.ndarray:
    """Flag per-patient changes that exceed the measurement noise floor.

    The point of a repeatability layer is that this question becomes
    answerable at all: without an SDC, "the marker rose by 12 units" carries
    no information about whether anything happened.
    """

    values = np.abs(np.asarray(list(change), dtype=float))
    if not np.isfinite(smallest_detectable_change) or smallest_detectable_change <= 0:
        return np.zeros(len(values), dtype=bool)
    return values > float(smallest_detectable_change)


def summarize_repeatability(
    table: pd.DataFrame,
    *,
    minimum_icc: float = MINIMUM_RELIABLE_ICC,
) -> Mapping[str, object]:
    """Portfolio-level verdict for the report layer."""

    if table.empty:
        return {
            "markers_assessed": 0,
            "markers_reliable": 0,
            "markers_unreliable": [],
            "minimum_reliable_icc": minimum_icc,
            "repeatability_version": REPEATABILITY_VERSION,
        }
    evaluable = table.loc[table["status"].eq("evaluable")]
    reliable = evaluable.loc[
        evaluable["reliable_for_individual_change"].astype(bool)
    ]
    unreliable = evaluable.loc[
        ~evaluable["reliable_for_individual_change"].astype(bool)
    ]
    return {
        "markers_assessed": int(len(evaluable)),
        "markers_reliable": int(len(reliable)),
        "markers_unreliable": sorted(
            str(value) for value in unreliable["marker_id"]
        )[:200],
        "median_icc": (
            float(evaluable["icc"].median())
            if len(evaluable)
            else float("nan")
        ),
        "minimum_reliable_icc": minimum_icc,
        "estimand": (
            "upper_bound_on_measurement_error; short-term physiological "
            "change is not separable from device and algorithm error here"
        ),
        "repeatability_version": REPEATABILITY_VERSION,
    }


__all__ = [
    "MINIMUM_RELIABLE_ICC",
    "MINIMUM_REPLICATE_PAIRS",
    "REPEATABILITY_COEFFICIENT_FACTOR",
    "REPEATABILITY_VERSION",
    "ReplicatePairPolicy",
    "SMALLEST_DETECTABLE_CHANGE_FACTOR",
    "bland_altman",
    "change_exceeds_measurement_error",
    "intraclass_correlation_absolute_agreement",
    "marker_repeatability",
    "repeatability_table",
    "summarize_repeatability",
    "summarize_repeatability",
    "within_subject_variance",
]
