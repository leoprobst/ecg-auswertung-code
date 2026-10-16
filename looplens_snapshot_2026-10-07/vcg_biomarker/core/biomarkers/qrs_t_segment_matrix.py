"""Fixed temporal thirds, not anatomical regions or selected hypotheses."""
import numpy as np
from ..data_loader import LEAD_MAP

SEGMENT_MATRIX_VERSION = "1.0.0"


def qrs_t_segment_matrix(ecg, *, qrs_bounds, t_bounds, baseline_bounds,
                         segment_evaluable=None):
    result = dict(version=SEGMENT_MATRIX_VERSION, status="not_evaluable",
        reason="required_segments_unavailable", cells=[], clinical_claim_allowed=False,
        scientific_tier="post_outcome_discovery", planned_cells=9,
        partition="fixed_duration_thirds_QRS_and_J_to_T_end",
        unit="degree", minimum_segment_magnitude_fraction=1e-6)
    if segment_evaluable is not None and not all(segment_evaluable.get(s) is True for s in ("qrs", "t", "pr")):
        result["reason"] = "segment_quality_gate_failed"
        return result
    if any(b is None for b in (qrs_bounds, t_bounds, baseline_bounds)):
        return result
    try:
        signal = np.asarray(ecg, dtype=float)
    except (ValueError, TypeError):
        result["reason"] = "invalid_twelve_lead_signal"
        return result
    if signal.ndim != 2 or signal.shape[1] != 12 or not np.isfinite(signal).all():
        result["reason"] = "invalid_twelve_lead_signal"
        return result
    try:
        bounds = np.asarray([qrs_bounds, t_bounds, baseline_bounds])
    except (ValueError, TypeError):
        result["reason"] = "invalid_segment_indices"
        return result
    if bounds.shape != (3, 2) or not np.issubdtype(bounds.dtype, np.integer):
        result["reason"] = "invalid_segment_indices"
        return result
    (q0, q1), (t0, t1), (b0, b1) = bounds
    if not 0 <= b0 < b1 <= q0 < q1 <= t0 < t1 < len(signal):
        result["reason"] = "invalid_segment_order"
        return result
    # Divide integration intervals, not disjoint sample sets: shared endpoints
    # carry half weight in each adjacent section, preserving the full integral.
    q_edges = np.rint(np.linspace(q0, q1, 4)).astype(int)
    t_edges = np.rint(np.linspace(q1, t1, 4)).astype(int)
    if min(np.diff(q_edges).min(), np.diff(t_edges).min()) < 2:
        result["reason"] = "segments_too_short_for_three_parts"
        return result
    native = signal[:, [LEAD_MAP[x] for x in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")]]
    native = native - native[b0:b1+1].mean(axis=0)

    def section(start, end):
        values = native[start:end+1]
        magnitude = np.linalg.norm(values, axis=1)
        weights = magnitude.copy()
        weights[[0, -1]] *= .5
        unit = np.divide(values, magnitude[:, None], out=np.zeros_like(values), where=magnitude[:, None] > 1e-12)
        return unit, weights

    q = [section(a, b) for a, b in zip(q_edges[:-1], q_edges[1:])]
    t = [section(a, b) for a, b in zip(t_edges[:-1], t_edges[1:])]
    q_total, t_total = sum(w.sum() for _, w in q), sum(w.sum() for _, w in t)
    if min(q_total, t_total) <= 1e-12:
        result["reason"] = "zero_segment_energy"
        return result
    total = 0.
    for i, (qu, qw) in enumerate(q):
        for j, (tu, tw) in enumerate(t):
            weight = float(qw.sum()*tw.sum())
            fraction = float(weight/(q_total*t_total))
            valid = min(qw.sum()/q_total, tw.sum()/t_total) >= 1e-6
            angle = float(np.degrees(qw @ np.arccos(np.clip(qu @ tu.T, -1, 1)) @ tw / weight)) if valid else None
            if angle is not None:
                total += fraction * angle
            result["cells"].append(dict(qrs_third=i+1, repolarization_third=j+1,
                angle_deg=angle, magnitude_pair_fraction=fraction,
                status="evaluable" if valid else "not_evaluable",
                reason="" if valid else "low_segment_energy",
                qrs_samples=[int(q_edges[i]), int(q_edges[i+1])],
                repolarization_samples=[int(t_edges[j]), int(t_edges[j+1])]))
    complete = all(cell["status"] == "evaluable" for cell in result["cells"])
    result.update(status="evaluable" if complete else "partial", reason="" if complete else "low_segment_energy",
                  reconstructed_integral_angle_deg=total if complete else None)
    return result
