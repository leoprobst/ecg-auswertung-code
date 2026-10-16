"""Time of day, day of week and season — preserved by MIMIC's de-identification.

MIMIC-IV shifts dates into the future but explicitly conserves the time of
day, the day of the week and approximate seasonality. That makes three real
temporal axes available that the project has never touched, and they are
useful for two opposite reasons at once.

**As signal.** Cardiac repolarization varies over the day: QT and its
rate-corrected forms are longest in the early morning, autonomic tone shifts
across the night-day transition, and the well-known morning peak of
cardiovascular events sits on top of that. A marker that moves with the clock
carries physiology; a marker whose *association with disease* moves with the
clock is telling you something about when patients present, not about disease.

**As confounder.** Measured on this cohort, ECGs peak at 09:00-10:00
(≈67,000 per hour) and bottom out at 05:00 (≈10,000). A recording taken at
04:00 exists for a different reason than one taken at 09:00 — it is far more
likely to be an acute event than a routine round. Acquisition hour is
therefore a proxy for indication, and leaving it out of an adjustment set
leaves a chunk of confounding by indication in place.

Hour is circular: 23:00 and 00:00 are adjacent, so encoding it as a number
0..23 tells a linear model that midnight is maximally far from 23:00. All
covariates here use the sine/cosine pair instead.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


CIRCADIAN_VERSION = "1.0.0"

# Boundaries chosen for clinical rather than statistical reasons: the night
# block spans the period with the fewest routine recordings, the morning block
# the peak of both routine acquisition and cardiovascular events.
TIME_OF_DAY_BANDS: tuple[tuple[str, int, int], ...] = (
    ("night", 0, 6),
    ("morning", 6, 12),
    ("afternoon", 12, 18),
    ("evening", 18, 24),
)

SEASON_BY_MONTH: Mapping[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def circadian_features(times: Iterable[object]) -> pd.DataFrame:
    """Cyclic and categorical temporal features from acquisition timestamps.

    Returns hour-of-day sine/cosine, weekday sine/cosine, seasonal
    sine/cosine, plus the categorical bands. The cyclic pairs are what belongs
    in a regression; the bands are for stratified reporting, where a reader
    needs "night" rather than a pair of trigonometric coefficients.
    """

    stamps = pd.to_datetime(pd.Series(list(times)), errors="coerce")
    hour = stamps.dt.hour + stamps.dt.minute / 60.0
    weekday = stamps.dt.dayofweek.astype(float)
    month = stamps.dt.month.astype(float)
    frame = pd.DataFrame(index=stamps.index)
    frame["acquisition_hour"] = hour
    frame["acquisition_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["acquisition_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["acquisition_weekday"] = weekday
    frame["acquisition_weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    frame["acquisition_weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    frame["acquisition_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    frame["acquisition_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    frame["time_of_day_band"] = _band(hour)
    frame["is_night_acquisition"] = frame["time_of_day_band"].eq("night")
    frame["is_weekend_acquisition"] = weekday >= 5
    frame["season"] = month.map(
        lambda value: SEASON_BY_MONTH.get(int(value)) if pd.notna(value) else None
    )
    frame["circadian_version"] = CIRCADIAN_VERSION
    return frame


def _band(hour: pd.Series) -> pd.Series:
    band = pd.Series(pd.NA, index=hour.index, dtype="object")
    for name, start, end in TIME_OF_DAY_BANDS:
        band.loc[hour.ge(start) & hour.lt(end)] = name
    return band


CIRCADIAN_COVARIATES: tuple[str, ...] = (
    "acquisition_hour_sin",
    "acquisition_hour_cos",
)


def acquisition_intensity_profile(times: Iterable[object]) -> pd.DataFrame:
    """Recordings per hour, and how far each hour sits from the daily mean.

    The deviation is the useful quantity: an hour with a fraction of the usual
    volume is an hour in which an ECG was ordered for a reason, which is what
    makes acquisition time a proxy for indication.
    """

    stamps = pd.to_datetime(pd.Series(list(times)), errors="coerce").dropna()
    if stamps.empty:
        return pd.DataFrame(
            columns=[
                "hour",
                "n",
                "share",
                "relative_intensity",
                "routine_hour",
            ]
        )
    counts = stamps.dt.hour.value_counts().sort_index()
    full = pd.Series(0, index=range(24), dtype=int)
    full.update(counts)
    total = int(full.sum())
    mean_per_hour = total / 24.0
    frame = pd.DataFrame(
        {
            "hour": full.index.astype(int),
            "n": full.to_numpy(dtype=int),
            "share": full.to_numpy(dtype=float) / max(total, 1),
            "relative_intensity": full.to_numpy(dtype=float)
            / max(mean_per_hour, 1e-9),
        }
    )
    frame["routine_hour"] = frame["relative_intensity"] >= 1.0
    frame["circadian_version"] = CIRCADIAN_VERSION
    return frame


def marker_circadian_variation(
    values: Iterable[float],
    times: Iterable[object],
    *,
    marker_id: str = "",
    minimum_per_band: int = 30,
) -> dict[str, object]:
    """Amplitude and acrophase of a marker's daily rhythm.

    Fits ``value ~ a*sin(2*pi*h/24) + b*cos(2*pi*h/24)`` by least squares --
    the standard single-harmonic cosinor. Amplitude is ``sqrt(a^2+b^2)``, the
    acrophase is the clock hour of the peak.

    The amplitude is reported next to the marker's own within-subject
    measurement error where available: a daily swing smaller than the
    measurement noise is not a rhythm anyone can act on.
    """

    array = np.asarray(list(values), dtype=float)
    stamps = pd.to_datetime(pd.Series(list(times)), errors="coerce")
    hour = (stamps.dt.hour + stamps.dt.minute / 60.0).to_numpy(dtype=float)
    usable = np.isfinite(array) & np.isfinite(hour)
    base: dict[str, object] = {
        "marker_id": str(marker_id),
        "n": int(usable.sum()),
        "status": "not_evaluable",
        "amplitude": float("nan"),
        "acrophase_hour": float("nan"),
        "mesor": float("nan"),
        "relative_amplitude": float("nan"),
        "circadian_version": CIRCADIAN_VERSION,
    }
    if usable.sum() < max(int(minimum_per_band) * 4, 50):
        base["failure_reason"] = "insufficient_observations"
        return base
    scoped_values = array[usable]
    scoped_hours = hour[usable]
    if np.unique(np.round(scoped_hours)).size < 6:
        base["failure_reason"] = "insufficient_hour_coverage"
        return base
    design = np.column_stack(
        (
            np.ones(len(scoped_hours)),
            np.sin(2 * np.pi * scoped_hours / 24.0),
            np.cos(2 * np.pi * scoped_hours / 24.0),
        )
    )
    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design,
        scoped_values,
        rcond=None,
    )
    if rank < 3:
        base["failure_reason"] = "rank_deficient_design"
        return base
    mesor, sine, cosine = (float(value) for value in coefficients)
    amplitude = float(np.hypot(sine, cosine))
    acrophase = float((np.arctan2(sine, cosine) * 24.0 / (2 * np.pi)) % 24.0)
    fitted = design @ coefficients
    total_variance = float(np.var(scoped_values, ddof=1))
    explained = (
        float(1.0 - np.var(scoped_values - fitted, ddof=1) / total_variance)
        if total_variance > 1e-12
        else float("nan")
    )
    base.update(
        {
            "status": "evaluable",
            "failure_reason": "",
            "mesor": mesor,
            "amplitude": amplitude,
            "acrophase_hour": acrophase,
            "peak_to_trough": 2.0 * amplitude,
            "variance_explained": explained,
            "relative_amplitude": (
                float(amplitude / abs(mesor)) if abs(mesor) > 1e-12 else float("nan")
            ),
            "model": "single_harmonic_cosinor_24h",
        }
    )
    return base


def exceeds_measurement_noise(
    variation: Mapping[str, object],
    smallest_detectable_change: float,
) -> bool:
    """Whether a daily swing is larger than the marker's own noise floor.

    Ties the circadian layer to the repeatability layer: without that
    comparison an amplitude is just a number, and a rhythm smaller than the
    smallest detectable change cannot inform any individual reading.
    """

    peak_to_trough = variation.get("peak_to_trough")
    if not isinstance(peak_to_trough, (int, float)):
        return False
    if not np.isfinite(peak_to_trough) or not np.isfinite(
        smallest_detectable_change
    ):
        return False
    return bool(float(peak_to_trough) > float(smallest_detectable_change))


def circadian_variation_table(
    frame: pd.DataFrame,
    marker_ids: Iterable[str],
    *,
    time_column: str = "ecg_time",
    repeatability: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Daily rhythm for every marker, judged against measurement error."""

    if time_column not in frame.columns:
        raise ValueError(f"Missing acquisition time column: {time_column}")
    sdc_lookup: dict[str, float] = {}
    if repeatability is not None and not repeatability.empty:
        for _, row in repeatability.iterrows():
            sdc_lookup[str(row.get("marker_id", ""))] = float(
                row.get("smallest_detectable_change", float("nan"))
            )
    records: list[dict[str, object]] = []
    for marker_id in marker_ids:
        if marker_id not in frame.columns:
            continue
        variation = marker_circadian_variation(
            pd.to_numeric(frame[marker_id], errors="coerce"),
            frame[time_column],
            marker_id=str(marker_id),
        )
        sdc = sdc_lookup.get(str(marker_id), float("nan"))
        variation["smallest_detectable_change"] = sdc
        variation["rhythm_exceeds_measurement_noise"] = (
            exceeds_measurement_noise(variation, sdc)
        )
        records.append(variation)
    if not records:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "status",
                "amplitude",
                "acrophase_hour",
                "rhythm_exceeds_measurement_noise",
            ]
        )
    return pd.DataFrame(records)


def summarize_circadian(table: pd.DataFrame) -> Mapping[str, object]:
    """Which markers carry a daily rhythm larger than their own noise."""

    if table.empty:
        return {
            "markers_assessed": 0,
            "markers_with_rhythm": 0,
            "circadian_version": CIRCADIAN_VERSION,
        }
    evaluable = table.loc[table["status"].eq("evaluable")]
    with_rhythm = evaluable.loc[
        evaluable.get(
            "rhythm_exceeds_measurement_noise",
            pd.Series(False, index=evaluable.index),
        ).astype(bool)
    ]
    return {
        "markers_assessed": int(len(evaluable)),
        "markers_with_rhythm": int(len(with_rhythm)),
        "markers_with_rhythm_ids": sorted(
            str(value) for value in with_rhythm["marker_id"]
        )[:200],
        "median_acrophase_hour": (
            float(with_rhythm["acrophase_hour"].median())
            if len(with_rhythm)
            else float("nan")
        ),
        "interpretation": (
            "a marker whose daily swing exceeds its smallest detectable "
            "change must have acquisition time in its adjustment set; one "
            "whose swing is below that noise floor cannot be read per patient"
        ),
        "circadian_version": CIRCADIAN_VERSION,
    }


__all__ = [
    "CIRCADIAN_COVARIATES",
    "CIRCADIAN_VERSION",
    "SEASON_BY_MONTH",
    "TIME_OF_DAY_BANDS",
    "acquisition_intensity_profile",
    "circadian_features",
    "circadian_variation_table",
    "exceeds_measurement_noise",
    "marker_circadian_variation",
    "summarize_circadian",
]
