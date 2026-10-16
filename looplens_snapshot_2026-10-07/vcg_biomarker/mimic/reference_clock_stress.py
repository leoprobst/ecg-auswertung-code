"""Prespecified clock shifts of an unchanged index/reference pairing.

These are stress scenarios, NOT estimates of recorder error. No best shift is
selected, no patient is re-indexed and no label is changed or imputed.
"""
import numpy as np
import pandas as pd

VERSION = "clock-stress-fixed-pair-1.1.0"
SHIFTS_HOURS = (-24., -6., -1., -.25, 0., .25, 1., 6., 24.)


def clock_stress_counts(frame, definition):
    if "split" in frame and not frame["split"].isin(["train", "validation"]).all():
        raise ValueError("Clock stress is development-only")
    before, after = definition.window_before_hours, definition.window_after_hours
    if before is None and after is None:
        return [{"endpoint_id": definition.label_id, "status": "not_applicable_no_finite_window",
                 "version": VERSION, "clinical_claim_allowed": False}]
    # LabelDefinition stores signed reference-minus-ECG interval bounds,
    # e.g. (-6, 12) for troponin and (-720, 720) for echo. These are not
    # positive durations; negating the lower bound reverses the window.
    if any(v is not None and not np.isfinite(v) for v in (before, after)):
        raise ValueError("Invalid prespecified reference window")
    lower = -np.inf if before is None else before
    upper = np.inf if after is None else after
    if lower > upper:
        raise ValueError("Invalid prespecified reference window")
    def col(name, default):
        return frame[name] if name in frame else pd.Series(default, index=frame.index)
    delta = pd.to_numeric(col("time_delta_hours", np.nan), errors="coerce")
    observed = (col("reference_evidence_available", False).eq(True)
                & col("time_precision", "").eq("timestamp") & np.isfinite(delta))
    base = observed & delta.between(lower, upper)
    rows = []
    # Sign convention: a positive shift moves ECG time later; reference-ECG
    # delta therefore decreases. Fixed pair selection remains untouched.
    for shift in SHIFTS_HOURS:
        within = observed & (delta - shift).between(lower, upper)
        for label in ("positive", "negative", "unknown"):
            subset = col("label_state", "unknown").eq(label)
            rows.append({"endpoint_id": definition.label_id, "label_state": label,
                "assumed_ecg_clock_shift_hours": shift, "status": "descriptive_stress_scenario",
                "n_index_rows": int(subset.sum()), "n_observed_timed_reference": int((observed & subset).sum()),
                "n_within_shifted_window": int((within & subset).sum()),
                "n_lost_vs_zero_shift": int((base & ~within & subset).sum()),
                "n_entering_vs_zero_shift": int((~base & within & subset).sum()),
                "window_before_hours": before, "window_after_hours": after,
                "pairing": "unchanged_reference_blind_index_and_observed_reference",
                "estimated_clock_error": False, "optimal_shift_selection": False,
                "clinical_claim_allowed": False, "version": VERSION})
    return rows
