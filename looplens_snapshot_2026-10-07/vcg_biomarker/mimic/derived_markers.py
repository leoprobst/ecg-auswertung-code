"""Derived markers: quantities that no single recording can carry.

Every marker in the biomarker catalog is computed from one ECG. The four
families here cannot be, because each needs something the recording does not
contain -- the time of day it was taken, the drug dose that preceded it, the
patient's earlier recordings, or the measurement error of the marker itself.

They are therefore produced at the analysis stage and, critically, they are
**declared as a different kind of thing**. A derived marker is not
interchangeable with a per-record marker: it has different missingness, a
different unit of observation, and in two cases it is a *provoked* rather than
a resting phenotype. ``derived_marker_declarations()`` carries that
distinction into every table and report so it survives to the reader.

The four families:

1. **Circadian residual** -- a marker's deviation from its own expected value
   at that hour. Turns the strongest acquisition confounder into a phenotype:
   "long QTc *for four in the morning*" is a different statement from "long
   QTc".
2. **Provoked repolarization reserve** -- the per-patient slope of an ECG
   marker against cumulative drug dose. A resting ECG measures the system at
   equilibrium; only a challenge reveals its reserve. This is the one family
   here that is a trait rather than a state.
3. **Recovery kinetics** -- how fast a marker returns to its own baseline
   after a landmark. Stunning and electrical remodelling have characteristic
   time constants that a cross-section cannot see.
4. **Reliability-weighted composite** -- components weighted by the inverse of
   their measurement variance, which the repeatability layer now supplies.

None of these can enter the frozen Core or Sentinel panels: all four were
conceived after development results were inspected, and two of them are
defined by the analysis design rather than by physiology alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .circadian import marker_circadian_variation
from .medication_dose import dose_response_trend


DERIVED_MARKER_VERSION = "1.3.0"

KIND_CIRCADIAN_RESIDUAL = "circadian_residual"
KIND_PROVOKED_RESERVE = "provoked_pharmacodynamic_reserve"
KIND_RECOVERY_KINETICS = "recovery_kinetics"
KIND_RELIABILITY_COMPOSITE = "reliability_weighted_composite"

DERIVED_MARKER_KINDS: tuple[str, ...] = (
    KIND_CIRCADIAN_RESIDUAL,
    KIND_PROVOKED_RESERVE,
    KIND_RECOVERY_KINETICS,
    KIND_RELIABILITY_COMPOSITE,
)

# What each family needs beyond the recording, and what that costs.
_KIND_METADATA: Mapping[str, Mapping[str, object]] = {
    KIND_CIRCADIAN_RESIDUAL: {
        "observation_unit": "ecg",
        "requires": "acquisition timestamp and a fitted diurnal reference",
        "state_or_trait": "state",
        "caveat_de": (
            "Residuum gegen die eigene Tagesrhythmik. Setzt voraus, dass die "
            "Kosinor-Referenz auf einer Population geschaetzt wurde, die zu "
            "diesem Patienten passt; sie ist keine individuelle Norm."
        ),
    },
    KIND_PROVOKED_RESERVE: {
        "observation_unit": "patient",
        "requires": "repeated ECGs spanning a documented, dosed exposure",
        "state_or_trait": "trait",
        "caveat_de": (
            "Provozierter Phaenotyp, kein Ruhewert. Beobachtungsdaten statt "
            "kontrolliertem Challenge: Dosis folgt der Indikation, daher ist "
            "Confounding by indication nicht ausgeschlossen. Nur mit dem "
            "Target-Trial-Rahmen kausal interpretierbar."
        ),
    },
    KIND_RECOVERY_KINETICS: {
        "observation_unit": "patient_episode",
        "requires": "a landmark plus at least three post-landmark recordings",
        "state_or_trait": "state_trajectory",
        "caveat_de": (
            "Erholungskinetik haengt davon ab, wann EKGs geschrieben wurden. "
            "Die Beobachtungsdichte ist selbst informativ und muss gewichtet "
            "oder mitberichtet werden."
        ),
    },
    KIND_RELIABILITY_COMPOSITE: {
        "observation_unit": "ecg",
        "requires": "component markers plus their within-subject variance",
        "state_or_trait": "state",
        "caveat_de": (
            "Gewichte stammen aus der Wiederholbarkeitsschicht desselben "
            "Datensatzes. Sie sind geschaetzt, nicht extern kalibriert, und "
            "muessen bei einem Freeze mit eingefroren werden."
        ),
    },
}


@dataclass(frozen=True)
class DerivedMarkerDeclaration:
    """How a derived marker must be presented wherever it appears."""

    marker_id: str
    kind: str
    base_marker_id: str
    unit: str
    observation_unit: str
    state_or_trait: str
    requires: str
    caveat_de: str
    per_record_marker: bool = False
    core_sentinel_eligible: bool = False
    post_outcome_hypothesis: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "marker_id": self.marker_id,
            "derived_marker_kind": self.kind,
            "base_marker_id": self.base_marker_id,
            "unit": self.unit,
            "observation_unit": self.observation_unit,
            "state_or_trait": self.state_or_trait,
            "requires": self.requires,
            "presentation_caveat_de": self.caveat_de,
            "per_record_marker": self.per_record_marker,
            "core_sentinel_eligible": self.core_sentinel_eligible,
            "post_outcome_hypothesis": self.post_outcome_hypothesis,
            "derived_marker_version": DERIVED_MARKER_VERSION,
        }


def declare(marker_id: str, kind: str, base_marker_id: str, unit: str) -> DerivedMarkerDeclaration:
    metadata = _KIND_METADATA[kind]
    return DerivedMarkerDeclaration(
        marker_id=marker_id,
        kind=kind,
        base_marker_id=base_marker_id,
        unit=unit,
        observation_unit=str(metadata["observation_unit"]),
        state_or_trait=str(metadata["state_or_trait"]),
        requires=str(metadata["requires"]),
        caveat_de=str(metadata["caveat_de"]),
    )


# --- 1. circadian residual --------------------------------------------------


def circadian_reference(
    values: Iterable[float],
    times: Iterable[object],
    *,
    marker_id: str,
) -> Mapping[str, object]:
    """Fit the diurnal reference a residual is measured against."""

    return marker_circadian_variation(values, times, marker_id=marker_id)


def circadian_residual(
    values: Iterable[float],
    times: Iterable[object],
    reference: Mapping[str, object],
) -> np.ndarray:
    """Observed value minus the value expected at that hour.

    Returns NaN wherever the reference could not be fitted, rather than
    falling back to the raw value: an unadjusted number silently relabelled as
    a residual would be the worst of both.
    """

    array = np.asarray(list(values), dtype=float)
    if str(reference.get("status")) != "evaluable":
        return np.full(len(array), np.nan, dtype=float)
    stamps = pd.to_datetime(pd.Series(list(times)), errors="coerce")
    hour = (stamps.dt.hour + stamps.dt.minute / 60.0).to_numpy(dtype=float)
    mesor = float(reference["mesor"])
    amplitude = float(reference["amplitude"])
    acrophase = float(reference["acrophase_hour"])
    expected = mesor + amplitude * np.cos(2 * np.pi * (hour - acrophase) / 24.0)
    return array - expected


def circadian_residual_table(
    frame: pd.DataFrame,
    marker_ids: Iterable[str],
    *,
    time_column: str = "ecg_time",
) -> tuple[pd.DataFrame, list[DerivedMarkerDeclaration]]:
    """Residual columns plus their declarations."""

    output = pd.DataFrame(index=frame.index)
    declarations: list[DerivedMarkerDeclaration] = []
    if time_column not in frame.columns:
        return output, declarations
    for marker_id in marker_ids:
        if marker_id not in frame.columns:
            continue
        values = pd.to_numeric(frame[marker_id], errors="coerce")
        reference = circadian_reference(
            values,
            frame[time_column],
            marker_id=str(marker_id),
        )
        residual = circadian_residual(values, frame[time_column], reference)
        if not np.isfinite(residual).any():
            continue
        name = f"circadian_residual__{marker_id}"
        output[name] = residual
        declarations.append(
            declare(name, KIND_CIRCADIAN_RESIDUAL, str(marker_id), "base marker unit")
        )
    return output, declarations


# --- 2. provoked repolarization reserve -------------------------------------


def provoked_reserve(
    frame: pd.DataFrame,
    *,
    marker_column: str,
    dose_column: str,
    subject_column: str = "subject_id",
    minimum_doses_per_subject: int = 2,
) -> pd.DataFrame:
    """Per-patient slope of an ECG marker against cumulative drug dose.

    The quantity Roden's repolarization-reserve concept describes: carriers of
    subclinical ion-channel variants look normal at rest and only lengthen
    disproportionately under blockade. Estimating it needs at least two dosed
    observations per patient, which is why the yield is reported alongside.
    """

    required = {marker_column, dose_column, subject_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Provoked-reserve frame is missing columns: {missing}")
    working = frame.loc[:, [subject_column, marker_column, dose_column]].copy()
    working[marker_column] = pd.to_numeric(working[marker_column], errors="coerce")
    working[dose_column] = pd.to_numeric(working[dose_column], errors="coerce")
    working = working.dropna()
    records: list[dict[str, object]] = []
    for subject_id, group in working.groupby(subject_column, sort=True):
        doses = group[dose_column].to_numpy(dtype=float)
        values = group[marker_column].to_numpy(dtype=float)
        distinct = int(np.unique(doses).size)
        record: dict[str, object] = {
            "subject_id": subject_id,
            "marker_id": marker_column,
            "n_observations": int(len(group)),
            "n_distinct_doses": distinct,
            "dose_range": float(np.ptp(doses)) if len(doses) else float("nan"),
            "derived_marker_kind": KIND_PROVOKED_RESERVE,
            "derived_marker_version": DERIVED_MARKER_VERSION,
        }
        if distinct < int(minimum_doses_per_subject) or np.ptp(doses) <= 0:
            record["status"] = "not_evaluable"
            record["failure_reason"] = "insufficient_dose_variation_within_subject"
            record["reserve_slope_per_unit_dose"] = float("nan")
            records.append(record)
            continue
        slope, intercept = np.polyfit(doses, values, 1)
        record.update(
            {
                "status": "evaluable",
                "failure_reason": "",
                "reserve_slope_per_unit_dose": float(slope),
                "baseline_at_zero_dose": float(intercept),
                # The direction that matters: a steep slope means little
                # reserve, i.e. the same dose moves this patient further.
                "low_reserve_direction": "higher_slope_means_lower_reserve",
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def provoked_reserve_population_trend(
    frame: pd.DataFrame,
    *,
    marker_column: str,
    dose_column: str,
    subject_column: str = "subject_id",
) -> Mapping[str, object]:
    """Population dose-response, for comparison with the per-patient slopes."""

    trend = dose_response_trend(
        frame[dose_column],
        frame[marker_column],
        subject_ids=frame[subject_column],
    )
    return {
        **trend,
        "derived_marker_kind": KIND_PROVOKED_RESERVE,
        "interpretation_de": (
            "Populationssteigung als Referenz fuer die individuellen "
            "Steigungen. Eine monotone Dosis-Wirkungs-Beziehung stuetzt eine "
            "pharmakologische Deutung, beweist sie aber nicht: die Dosis "
            "folgt der Indikation."
        ),
    }


# --- 3. recovery kinetics ---------------------------------------------------


def recovery_kinetics(
    frame: pd.DataFrame,
    *,
    marker_column: str,
    time_column: str = "ecg_time",
    landmark_column: str = "landmark_time",
    subject_column: str = "subject_id",
    episode_column: Optional[str] = None,
    normalization_column: Optional[str] = None,
    additional_normalization_columns: Optional[dict[str, str]] = None,
    normalization_window_hours: float = 24.0,
    baseline_window_hours: float = 24.0,
    minimum_post_landmark: int = 3,
) -> pd.DataFrame:
    """How a marker returns towards its own pre-landmark level.

    This is a retrospective observed threshold crossing, not a physiological
    half-life or an exposure available at the landmark. A within-person
    baseline does not remove time-varying confounding or informative sampling.
    Multiple landmarks require explicit episodes; they must never be merged.

    Optional normalization_column must be supplied by a reference-specific
    adapter as the observed normalization time, not inferred from this marker.
    The first ECG at/after that time within normalization_window_hours is fixed
    before marker missingness is inspected. Its residual uses the same baseline;
    neither normalization nor the residual is a causal treatment-effect claim.
    Additional named normalization columns use the same episode and baseline;
    their output fields are prefixed '<name>__'. No replicated episode rows or
    repeated kinetics estimation are introduced for additional analytes.
    """

    additional = dict(additional_normalization_columns or {})
    if any(not key.isidentifier() or key.startswith("_") for key in additional):
        raise ValueError("Additional normalization names must be identifiers")
    required = {marker_column, time_column, landmark_column, subject_column}
    if (set(additional.values()) & (required | {episode_column, normalization_column})
            or len(set(additional.values())) != len(additional)):
        raise ValueError("Additional normalization columns cannot overwrite context")
    required.update(additional.values())
    if episode_column is not None:
        required.add(episode_column)
    if normalization_column is not None:
        required.add(normalization_column)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Recovery frame is missing columns: {missing}")
    if (not np.isfinite(baseline_window_hours) or baseline_window_hours <= 0
            or isinstance(minimum_post_landmark, (bool, np.bool_))
            or not isinstance(minimum_post_landmark, (int, np.integer))
            or minimum_post_landmark < 3):
        raise ValueError("Recovery requires a positive baseline window and at least three post-landmark ECGs")
    if not np.isfinite(normalization_window_hours) or normalization_window_hours <= 0:
        raise ValueError("Normalization ECG window must be positive and finite")
    from .patient_identity import known_patient_mask
    if not known_patient_mask(frame[subject_column]).all():
        raise ValueError("Recovery requires known patient identities")
    if "split" in frame:
        if not frame["split"].isin(["train", "validation"]).all():
            raise PermissionError("Recovery discovery only permits development splits")
        if frame.groupby(subject_column)["split"].nunique().gt(1).any():
            raise ValueError("Recovery patient occurs across development splits")
    # Do not copy thousands of unrelated feature columns for each marker.
    working = frame.loc[:, sorted(required)].copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")
    working[landmark_column] = pd.to_datetime(
        working[landmark_column],
        errors="coerce",
    )
    working[marker_column] = pd.to_numeric(working[marker_column], errors="coerce")
    normalization_sources = {f"{key}__": value for key, value in additional.items()}
    if normalization_column:
        normalization_sources[""] = normalization_column
    for column in normalization_sources.values():
        working[column] = pd.to_datetime(working[column], errors="coerce")
    records: list[dict[str, object]] = []
    group_columns = [subject_column, episode_column] if episode_column else subject_column
    for group_key, group in working.groupby(group_columns, sort=True, dropna=False):
        subject_id = group_key[0] if episode_column else group_key
        group = group.sort_values(time_column, kind="mergesort")
        identity = {"subject_id": subject_id}
        if episode_column:
            identity["episode_id"] = group_key[1]
        record: dict[str, object] = {
            **identity, "marker_id": marker_column,
            "derived_marker_kind": KIND_RECOVERY_KINETICS,
            "derived_marker_version": DERIVED_MARKER_VERSION,
            "analysis_design": "retrospective_observed_post_peak_threshold_crossing",
            "available_at_landmark": False, "physiological_half_life": False,
            "crossing_interpretation": "observation_bracket_unobserved_recrossings_possible",
            "evidence_stage": "exploratory", "clinical_claim_allowed": False,
            "n_input_rows": len(group), "status": "not_evaluable",
        }
        if (group[landmark_column].nunique() != 1 or group[landmark_column].isna().any()
                or (episode_column and not known_patient_mask(group[episode_column]).all())):
            records.append({**record, "failure_reason": "ambiguous_episode_landmark"})
            continue
        # Choose the normalization ECG before looking at marker availability:
        # never replace an unmeasurable first ECG with a later favorable one.
        normalization_values = {}
        record["normalization_status"] = "not_requested"
        for prefix, column in normalization_sources.items():
            observation: dict[str, object] = {}
            normalized = group[column]
            observation["normalization_status"] = "not_evaluable"
            observation["normalization_reason"] = "missing_or_ambiguous_normalization_time"
            if normalized.notna().all() and normalized.nunique() == 1:
                stamp = normalized.iloc[0]
                landmark_stamp = group[landmark_column].iloc[0]
                observation["normalization_reference_time"] = stamp.isoformat()
                observation["normalization_reason"] = "normalization_precedes_landmark"
                if stamp >= landmark_stamp:
                    lag = (group[time_column] - stamp).dt.total_seconds()/3600
                    candidates = group.loc[lag.between(0, normalization_window_hours)]
                    observation["normalization_reason"] = "no_ecg_in_prespecified_normalization_window"
                    if not candidates.empty:
                        selected_time = candidates[time_column].min()
                        values_at_time = candidates.loc[candidates[time_column].eq(selected_time), marker_column]
                        observation["normalization_ecg_time"] = selected_time.isoformat()
                        observation["normalization_ecg_lag_hours"] = (selected_time-stamp).total_seconds()/3600
                        observation["normalization_reason"] = "selected_ecg_marker_missing_or_conflicting"
                        if (np.isfinite(values_at_time.to_numpy(dtype=float, na_value=np.nan)).all()
                                and values_at_time.nunique() == 1):
                            normalization_values[prefix] = float(values_at_time.iloc[0])
                            observation["normalization_reason"] = "baseline_not_yet_verified"
            record.update({prefix + key: value for key, value in observation.items()})
        finite = np.isfinite(group[marker_column].to_numpy(dtype=float, na_value=np.nan))
        valid = group[time_column].notna().to_numpy() & finite
        record["n_unusable_observations"] = int((~valid).sum())
        group = group.loc[valid]
        # Repeated imports of the same measurement must not increase density
        # or satisfy the minimum number of observations. Conflicting values
        # at one timestamp cannot establish an ordered crossing.
        if group.groupby(time_column)[marker_column].nunique().gt(1).any():
            records.append({**record, "failure_reason": "conflicting_same_time_measurements"})
            continue
        before_dedup = len(group)
        group = group.drop_duplicates(subset=[time_column], keep="first")
        record["n_duplicate_observations"] = before_dedup - len(group)
        if group.empty:
            records.append({**record, "n_pre": 0, "n_post": 0,
                            "failure_reason": "no_finite_timed_observations"})
            continue
        landmark = group[landmark_column].iloc[0]
        hours = (
            (group[time_column] - landmark).dt.total_seconds().to_numpy() / 3600.0
        )
        values = group[marker_column].to_numpy(dtype=float)
        pre = values[(hours < 0) & (hours >= -float(baseline_window_hours))]
        post_mask = hours > 0
        record.update({
            "n_pre": int(len(pre)),
            "n_post": int(post_mask.sum()),
            "n_at_landmark_excluded": int((hours == 0).sum()),
        })
        if len(pre) == 0 or post_mask.sum() < int(minimum_post_landmark):
            record["status"] = "not_evaluable"
            record["failure_reason"] = "no_baseline_or_too_few_post_landmark_ecgs"
            records.append(record)
            continue
        baseline = float(np.median(pre))
        for prefix, normalization_value in normalization_values.items():
            record.update({prefix + key: value for key, value in dict(
                normalization_status="evaluable", normalization_reason="",
                normalization_marker_value=normalization_value,
                normalization_residual=normalization_value-baseline,
                normalization_absolute_residual=abs(normalization_value-baseline),
                normalization_interpretation="residual_after_observed_reference_normalization_not_causal_recovery"
            ).items()})
        post_hours = hours[post_mask]
        post_values = values[post_mask]
        deviation = post_values - baseline
        peak_index = int(np.argmax(np.abs(deviation)))
        peak_deviation = float(deviation[peak_index])
        peak_hour = float(post_hours[peak_index])
        # Time at which the deviation first falls below half its peak.
        half_time = float("nan")
        crossing_lower = float("nan")
        crossing_upper = float("nan")
        crossing_status = "undefined_zero_peak"
        if abs(peak_deviation) > 0:
            after = post_hours >= peak_hour
            fraction = np.abs(deviation[after]) / abs(peak_deviation)
            below = np.flatnonzero(fraction <= 0.5)
            if len(below):
                half_time = float(post_hours[after][below[0]] - peak_hour)
                crossing_lower = float(post_hours[after][max(0, below[0] - 1)] - peak_hour)
                crossing_upper = half_time
                crossing_status = "interval_censored"
            else:
                crossing_lower = float(post_hours[-1] - peak_hour)
                crossing_upper = float("inf")
                crossing_status = "right_censored"
        returned = bool(
            np.isfinite(half_time)
            and abs(deviation[-1]) <= 0.5 * abs(peak_deviation)
        )
        record.update(
            {
                "status": "evaluable",
                "failure_reason": "",
                "baseline_value": baseline,
                "peak_deviation": peak_deviation,
                "peak_deviation_hours": peak_hour,
                "recovery_half_time_hours": half_time,
                # Legacy field above is retained for reading old consumers.
                "observed_half_peak_crossing_hours": half_time,
                "half_peak_crossing_lower_hours": crossing_lower,
                "half_peak_crossing_upper_hours": crossing_upper,
                "half_peak_crossing_status": crossing_status,
                "available_after_ecg_time": group[time_column].max().isoformat(),
                "deviation_area_hours": float(
                    np.trapezoid(np.abs(deviation), post_hours)
                ),
                "returned_to_half_baseline": returned,
                "observation_density_per_day": float(
                    post_mask.sum() / max(float(np.ptp(post_hours)) / 24.0, 1e-6)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


# --- 4. reliability-weighted composite --------------------------------------


def reliability_weights(
    repeatability: pd.DataFrame,
    component_ids: Sequence[str],
    *,
    marker_column: str = "marker_id",
    sd_column: str = "within_subject_sd",
) -> dict[str, float]:
    """Inverse-variance weights from the repeatability layer.

    A component the pipeline cannot measure twice the same way should not
    carry the same weight as one it can. Components without an estimate are
    excluded rather than given a default weight.
    """

    if repeatability.empty:
        return {}
    lookup = repeatability.set_index(marker_column)[sd_column].to_dict()
    weights: dict[str, float] = {}
    for component in component_ids:
        sd = lookup.get(component)
        if sd is None:
            continue
        try:
            value = float(sd)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        weights[str(component)] = 1.0 / (value ** 2)
    total = float(sum(weights.values()))
    if total <= 0:
        return {}
    return {key: value / total for key, value in weights.items()}


def reliability_weighted_composite(
    frame: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    standardize: bool = True,
) -> np.ndarray:
    """Weighted mean of standardized components.

    Standardization is on by default because the components carry different
    units; without it the weighting would be dominated by whichever component
    happens to be measured on the largest scale.
    """

    usable = [key for key in weights if key in frame.columns]
    if not usable:
        return np.full(len(frame), np.nan, dtype=float)
    total = float(sum(weights[key] for key in usable))
    if total <= 0:
        return np.full(len(frame), np.nan, dtype=float)
    accumulator = np.zeros(len(frame), dtype=float)
    coverage = np.zeros(len(frame), dtype=float)
    for key in usable:
        column = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)
        if standardize:
            sd = float(np.nanstd(column, ddof=1))
            mean = float(np.nanmean(column))
            if not np.isfinite(sd) or sd <= 0:
                continue
            column = (column - mean) / sd
        weight = weights[key] / total
        finite = np.isfinite(column)
        accumulator[finite] += weight * column[finite]
        coverage[finite] += weight
    result = np.where(coverage > 0, accumulator / np.maximum(coverage, 1e-12), np.nan)
    # A composite assembled from a minority of its components is not the
    # composite that was specified.
    return np.where(coverage >= 0.5, result, np.nan)


# --- presentation contract --------------------------------------------------


def derived_marker_declarations(
    declarations: Iterable[DerivedMarkerDeclaration],
) -> pd.DataFrame:
    """The table every report must join against before showing a derived marker."""

    rows = [item.to_dict() for item in declarations]
    if not rows:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "derived_marker_kind",
                "base_marker_id",
                "observation_unit",
                "state_or_trait",
                "per_record_marker",
                "presentation_caveat_de",
            ]
        )
    return pd.DataFrame(rows)


def annotate_results_with_derived_kind(
    results: pd.DataFrame,
    declarations: Iterable[DerivedMarkerDeclaration],
    *,
    marker_column: str = "feature_id",
) -> pd.DataFrame:
    """Attach the derived-marker declaration to a results table.

    Without this join a derived marker appears in a ranking beside per-record
    markers as if it were the same kind of measurement. It is not: its unit of
    observation differs, and two of the four families are provoked rather than
    resting quantities.
    """

    output = results.copy()
    lookup = {item.marker_id: item for item in declarations}
    if marker_column not in output.columns:
        raise ValueError(f"Results are missing the marker column: {marker_column}")
    kinds: list[str] = []
    per_record: list[bool] = []
    caveats: list[str] = []
    observation_units: list[str] = []
    for value in output[marker_column].astype(str):
        declaration = lookup.get(value)
        kinds.append(declaration.kind if declaration else "per_record_marker")
        per_record.append(declaration is None)
        caveats.append(declaration.caveat_de if declaration else "")
        observation_units.append(
            declaration.observation_unit if declaration else "ecg"
        )
    output["derived_marker_kind"] = kinds
    output["per_record_marker"] = per_record
    output["observation_unit"] = observation_units
    output["presentation_caveat_de"] = caveats
    output["derived_marker_version"] = DERIVED_MARKER_VERSION
    return output


def assert_derived_markers_are_declared(
    results: pd.DataFrame,
    declarations: Iterable[DerivedMarkerDeclaration],
    *,
    marker_column: str = "feature_id",
) -> None:
    """Fail when a derived marker reaches a report without its declaration.

    The guard exists because the failure mode is silent: a residual or a
    provoked slope rendered as an ordinary marker reads as a property of the
    ECG rather than of the analysis design.
    """

    if results.empty or marker_column not in results.columns:
        return
    declared = {item.marker_id for item in declarations}
    prefixes = tuple(f"{kind}__" for kind in DERIVED_MARKER_KINDS)
    offending = sorted(
        {
            str(value)
            for value in results[marker_column].astype(str)
            if str(value).startswith(prefixes) and str(value) not in declared
        }
    )
    if offending:
        raise AssertionError(
            "Derived markers reached the results table without a declaration: "
            f"{offending}. A derived marker shown as an ordinary marker reads "
            "as a property of the recording rather than of the analysis."
        )


__all__ = [
    "DERIVED_MARKER_KINDS",
    "DERIVED_MARKER_VERSION",
    "DerivedMarkerDeclaration",
    "KIND_CIRCADIAN_RESIDUAL",
    "KIND_PROVOKED_RESERVE",
    "KIND_RECOVERY_KINETICS",
    "KIND_RELIABILITY_COMPOSITE",
    "annotate_results_with_derived_kind",
    "assert_derived_markers_are_declared",
    "circadian_reference",
    "circadian_residual",
    "circadian_residual_table",
    "declare",
    "derived_marker_declarations",
    "provoked_reserve",
    "provoked_reserve_population_trend",
    "recovery_kinetics",
    "reliability_weighted_composite",
    "reliability_weights",
]
