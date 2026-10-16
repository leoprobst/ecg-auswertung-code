"""Research-only native precordial downslope timing on short ECG records.

Conventional-band NDAT inspired by DOI 10.1038/s41598-024-55789-w.
This short-record, beatwise implementation is not the original averaged-beat
protocol and does not estimate invasive epicardial activation or UHF delay.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..data_loader import LEAD_MAP

NDAT_VERSION = "1.1.1"
NDAT_LEADS = tuple(f"V{i}" for i in range(1, 7))


def negative_downslope_times(signal: np.ndarray, fs: float,
                            qrs_bounds: list[tuple[int, int]], *,
                            pacing_present: bool = False) -> dict:
    """Input: full native ECG in mV, sample-index QRS bounds in that record.

    Filter once for all beats/leads, then central finite difference. Quality
    constants are prespecified engineering gates, not diagnostic thresholds.
    At least three assessable beats are needed for a record-level median.
    Caller must not pass a 40-Hz delineation signal as a native input.
    """
    result = dict(version=NDAT_VERSION, status="not_evaluable", reason="",
        evidence_stage="post_outcome_discovery", clinical_claim_allowed=False,
        signal_contract="native_mV_not_40Hz_delineation",
        preprocessing="third_order_0.05_100Hz_zero_phase_Butterworth",
        protocol_equivalence="short_record_adaptation_not_original_averaged_beat_protocol",
        leads={}, spread_ms=None, order=[], beat_times_ms=[])
    x = np.asarray(signal, dtype=float)
    if (not np.isfinite(fs) or fs < 250 or x.ndim != 2
            or x.shape[1] < 12 or len(x) < int(3 * fs)):
        result["reason"] = "sampling_shape_or_duration"
        return result
    if pacing_present:
        result["reason"] = "pacing_requires_separate_validated_protocol"
        return result
    precordial = x[:, [LEAD_MAP[lead] for lead in NDAT_LEADS]]
    if not np.isfinite(precordial).all():
        result["reason"] = "nonfinite_precordial_signal"
        return result
    bounds = np.asarray(qrs_bounds)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) < 3:
        result["reason"] = "insufficient_qrs_bounds"
        return result
    if (not np.isfinite(bounds).all() or not np.equal(bounds, np.floor(bounds)).all()
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] >= len(x))
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or np.any(bounds[1:, 0] <= bounds[:-1, 1])):
        result["reason"] = "invalid_or_overlapping_qrs_bounds"
        return result
    filtered = sosfiltfilt(butter(3, [.05, 100], fs=fs, btype="bandpass", output="sos"),
                          precordial, axis=0)
    derivative = np.gradient(filtered, 1 / fs, axis=0)
    times = np.full((len(bounds), 6), np.nan)
    reasons = [[] for _ in NDAT_LEADS]
    for beat, (start, end) in enumerate(bounds.astype(int)):
        if start < .25 * fs or end > len(x) - 1 - .25 * fs or not .04 * fs <= end-start <= .3 * fs:
            for failures in reasons:
                failures.append("edge_or_qrs_duration")
            continue
        for lead in range(6):
            values = filtered[start:end+1, lead]
            slope = derivative[start:end+1, lead]
            peak = int(np.argmin(slope))
            if np.ptp(values) < .05 or slope[peak] >= -1:
                reasons[lead].append("low_amplitude_or_downslope")
                continue
            contenders = np.flatnonzero(slope <= .98 * slope[peak])
            if (peak in (0, len(slope)-1) or
                    (contenders[-1] - contenders[0]) / fs > .012):
                reasons[lead].append("ambiguous_or_boundary_downslope")
                continue
            times[beat, lead] = peak * 1000 / fs
    for i, lead in enumerate(NDAT_LEADS):
        observed = times[:, i][np.isfinite(times[:, i])]
        median = float(np.median(observed)) if len(observed) else None
        mad = float(np.median(np.abs(observed-median))) if len(observed) else None
        central_spread = float(np.ptp(np.quantile(observed, [.1, .9]))) if len(observed) else None
        valid = (len(observed) >= 3 and len(observed) / len(bounds) >= .75
                 and mad <= 8 and central_spread <= 16)
        result["leads"][lead] = dict(time_ms=median if valid else None,
            n_assessable=int(len(observed)), n_beats=len(bounds), mad_ms=mad,
            p90_p10_ms=central_spread,
            status="evaluable" if valid else "not_evaluable",
            reasons=sorted(set(reasons[i] + ([] if valid else ["insufficient_or_unstable_beats"]))))
    result["beat_times_ms"] = [[float(v) if np.isfinite(v) else None for v in row] for row in times]
    if all(result["leads"][lead]["status"] == "evaluable" for lead in NDAT_LEADS):
        medians = [result["leads"][lead]["time_ms"] for lead in NDAT_LEADS]
        result.update(status="evaluable", spread_ms=float(np.ptp(medians)),
                      order=sorted(NDAT_LEADS, key=lambda lead: (result["leads"][lead]["time_ms"], lead)),
                      order_ties="equal_times_are_ties_alphabetical_display_only")
    else:
        result["reason"] = "not_all_precordial_leads_assessable"
    return result
