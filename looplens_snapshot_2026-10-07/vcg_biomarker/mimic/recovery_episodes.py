"""Admission-scoped, bounded episode adapter for retrospective ECG trajectories."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .derived_markers import recovery_kinetics
from .patient_identity import known_patient_mask

RECOVERY_EPISODE_VERSION = "1.2.0"


def _episode_time_index(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Overlap and latest prior end in O(n log n), preserving strict bounds.

    Input is start-time sorted (NaT last). Retain malformed intervals in the
    comparisons just as the direct pandas predicates do; their own rows abstain.
    Two prefix maxima allow excluding the current episode without rescanning.
    """
    starts = group.landmark_time.to_numpy(dtype="datetime64[ns]").view("i8")
    ends = group.episode_end.to_numpy(dtype="datetime64[ns]").view("i8")
    missing = np.iinfo(np.int64).min
    count = len(starts)
    overlap = np.zeros(count, dtype=bool)
    prior = np.full(count, missing, dtype=np.int64)
    if not count:
        return overlap, prior.view("datetime64[ns]")
    searchable = starts[starts != missing]
    usable_ends = np.where(starts != missing, ends, missing)
    largest = np.maximum.accumulate(usable_ends)
    previous = np.r_[missing, largest[:-1]]
    second = np.maximum.accumulate(np.minimum(previous, usable_ends))
    ordered_ends = np.sort(ends[ends != missing])
    for i, (start, end) in enumerate(zip(starts, ends)):
        if start == missing or end == missing or start >= end:
            continue
        stop = np.searchsorted(searchable, end, side="left")
        best_other = second[stop - 1] if largest[stop - 1] == end else largest[stop - 1]
        overlap[i] = best_other > start
        before = np.searchsorted(ordered_ends, start, side="right")
        if before:
            prior[i] = ordered_ends[before - 1]
    return overlap, prior.view("datetime64[ns]")


def episode_recovery_table(ecgs: pd.DataFrame, episodes: pd.DataFrame, *,
                           marker_ids: tuple[str, ...], baseline_hours: float = 24.,
                           additional_normalization_columns: dict[str, str] | None = None,
                           followup_hours: float = 48.) -> pd.DataFrame:
    """Episodes are supplied independently of ECG marker values.

    Requires explicit start/end boundaries and a single admission. Overlapping
    episodes abstain rather than sharing post-event ECGs. Optional normalization
    is an observed reference timestamp, not inferred from ECGs here. Searchsorted
    range selection avoids a patient ECG x episode Cartesian join.
    """
    additional = dict(additional_normalization_columns or {})
    er = {"subject_id", "hadm_id", "episode_id", "landmark_time", "episode_end"}
    if not set(additional.values()) <= set(episodes):
        raise ValueError("Missing additional normalization timestamps")
    cr = {"subject_id", "hadm_id", "study_id", "ecg_time", "split", "extraction_status", "signal_quality_score"}
    if (set(additional.values()) & (cr | er | set(marker_ids) | {"normalization_time"})
            or len(set(additional.values())) != len(additional)
            or any(not key.isidentifier() or key.startswith("_") for key in additional)):
        raise ValueError("Additional normalization columns must be unique and cannot overwrite context")
    if not er <= set(episodes) or not cr <= set(ecgs):
        raise ValueError("Missing episode/admission or ECG eligibility contract")
    if not marker_ids or len(set(marker_ids)) != len(marker_ids):
        raise ValueError("A unique prespecified marker list is required")
    if set(marker_ids) & (cr | er | {"normalization_time"}):
        raise ValueError("Markers cannot overwrite episode or ECG context")
    if any(not np.isfinite(x) or x <= 0 for x in (baseline_hours, followup_hours)):
        raise ValueError("Episode windows must be finite and positive")
    for frame, columns in ((episodes, ("subject_id", "hadm_id", "episode_id")),
                            (ecgs, ("subject_id", "hadm_id", "study_id"))):
        if any(not known_patient_mask(frame[c]).all() for c in columns):
            raise ValueError("Known patient, admission and record identities are required")
    if episodes.duplicated(["subject_id", "episode_id"]).any() or ecgs.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("Duplicate episode or ECG identity")
    if not ecgs.split.isin(["train", "validation"]).all():
        raise PermissionError("Recovery discovery requires development ECGs only")
    if ecgs.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("Patient crosses development splits")
    event = episodes.copy()
    for c in ("landmark_time", "episode_end", "normalization_time", *additional.values()):
        if c in event:
            event[c] = pd.to_datetime(event[c], errors="coerce")
    records = ecgs.reindex(columns=[*sorted(cr), *marker_ids]).copy()
    records["ecg_time"] = pd.to_datetime(records.ecg_time, errors="coerce")
    if records.ecg_time.isna().any():
        raise ValueError("ECG timestamps must be resolved before episode assignment")
    # Retain QC-failed ECGs for denominator and reference-blind selection, but
    # do not let their numerical values reach recovery estimation.
    quality = pd.to_numeric(records.signal_quality_score, errors="coerce")
    eligible = records.extraction_status.eq("ok") & quality.between(.5, 1.)
    records.loc[~eligible, list(marker_ids)] = np.nan
    grouped = {key: group.sort_values(["ecg_time", "study_id"], kind="stable")
               for key, group in records.groupby(["subject_id", "hadm_id"], sort=False)}
    output: list[dict] = []
    for key, group in event.groupby(["subject_id", "hadm_id"], sort=False):
        group = group.sort_values("landmark_time", kind="stable")
        overlaps, prior_ends = _episode_time_index(group)
        source = grouped.get(key)
        times = source.ecg_time.to_numpy() if source is not None else np.array([], dtype="datetime64[ns]")
        for position, (_, row) in enumerate(group.iterrows()):
            base = dict(subject_id=key[0], hadm_id=key[1], episode_id=row.episode_id,
                        episode_adapter_version=RECOVERY_EPISODE_VERSION,
                        evidence_stage="exploratory", clinical_claim_allowed=False,
                        scientific_tier="post_outcome_discovery",
                        temporal_design="retrospective_episode_response_not_prediction",
                        available_at_landmark=False, status="not_evaluable")
            base.update({name: value for name, value in row.items() if name.startswith("reference_")})
            start, end = row.landmark_time, row.episode_end
            reason = ""
            if pd.isna(start) or pd.isna(end) or start >= end:
                reason = "invalid_episode_boundaries"
            else:
                if overlaps[position]:
                    reason = "overlapping_episode_windows"
            if reason:
                output.extend({**base, "marker_id": marker, "failure_reason": reason} for marker in marker_ids)
                continue
            lower = start - pd.Timedelta(hours=baseline_hours)
            # A prior active episode cannot become the next episode's baseline.
            if not np.isnat(prior_ends[position]):
                lower = max(lower, pd.Timestamp(prior_ends[position]))
            upper = min(end, start + pd.Timedelta(hours=followup_hours))
            left, right = np.searchsorted(times, [lower.to_datetime64(), upper.to_datetime64()])
            scoped = source.iloc[left:right].copy() if source is not None else records.iloc[:0].copy()
            if scoped.empty:
                output.extend({**base, "marker_id": marker, "n_input_rows": 0,
                               "failure_reason": "no_ecg_in_episode_window"} for marker in marker_ids)
                continue
            scoped["landmark_time"] = start
            scoped["episode_id"] = row.episode_id
            normalized = "normalization_time" in event
            if normalized:
                stamp = row.normalization_time
                scoped["normalization_time"] = stamp if pd.notna(stamp) and start <= stamp < upper else pd.NaT
            for column in additional.values():
                stamp = row[column]
                scoped[column] = stamp if pd.notna(stamp) and start <= stamp < upper else pd.NaT
            for marker in marker_ids:
                result = recovery_kinetics(scoped, marker_column=marker, episode_column="episode_id",
                    baseline_window_hours=baseline_hours,
                    additional_normalization_columns=additional,
                    normalization_column="normalization_time" if normalized else None)
                output.extend({**base, **item, "window_start": lower.isoformat(),
                               "window_end_exclusive": upper.isoformat(),
                               "temporal_design": "retrospective_episode_response_not_prediction",
                               "baseline_interpretation": "pre_event_not_proven_healthy_or_recovered"}
                              for item in result.to_dict("records"))
    return pd.DataFrame(output)
