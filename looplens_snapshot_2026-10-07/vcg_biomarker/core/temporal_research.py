"""Exploratory sequence analyses, deliberately not single-ECG diagnostic markers.

Versioned physical/data contracts for the four proposals in
VCG_ENTSCHEIDUNGEN_UND_ZEITSTRUKTUR_RESEARCH.md. No disease cutoffs.
"""
from __future__ import annotations

import numpy as np

VERSION = "1.3.0"
GRID = np.linspace(0.0, 1.0, 33)
# The record descriptor is written into every feature row. Its 33-point arc
# clock and 33 x 3 phase trajectory are stored to the precision the
# measurement has -- a ten-thousandth of the arc, a tenth of a microvolt --
# not to seventeen digits. Measured on the A117 pilot this halves the
# descriptor (3,7 KB to 1,9 KB) without changing any downstream number
# beyond that precision.
ARC_DECIMALS = 4
MILLIVOLT_DECIMALS = 4
ANALYSES = {
    "serial_similarity_decomposition": "min_(g>0,R in SO(3)) ||QRS_after-g QRS_before R||_F^2 on 33 equal-phase samples; duration retained separately",
    "serial_arc_clock": "integral_0^1 abs(F_after(u)-F_before(u)) du",
    "direction_time_occupancy": "three equal-time bins x eight fixed XYZ octants; valid-time denominator",
    "rr_history_t_response": "within-record OLS F_T(0.5) ~ previous_RR + mean_previous_5_RR + QT",
    "pvc_sinus_transition": "PVC to immediately following explicitly annotated sinus beat: delta_F, direction and coupling interval",
}


def abstain(reason):
    return {"status": "non_estimable", "reason": reason, "version": VERSION,
            "evidence_tier": "exploratory", "clinical_claim_allowed": False}


def _ok(**values):
    return {"status": "descriptive", "version": VERSION,
            "evidence_tier": "exploratory", "clinical_claim_allowed": False, **values}


def _rhythm_context(pattern=None, confidence=None):
    from .delineation.base import RhythmClass
    pattern = getattr(pattern, "value", pattern)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    valid = (isinstance(pattern, str) and pattern in {item.value for item in RhythmClass}
             and confidence is not None and np.isfinite(confidence) and 0 < confidence <= 1)
    return {"pattern": pattern if valid else "unknown",
            "confidence": confidence if valid else None,
            "source": "technical_delineation_pattern_not_clinical_diagnosis"}


def serial_rhythm_context(before, after):
    contexts = []
    for descriptor in (before, after):
        context = descriptor.get("rhythm_context")
        context = context if isinstance(context, dict) else {}
        contexts.append(_rhythm_context(context.get("pattern"), context.get("confidence")))
    a, b = contexts
    status = ("unknown" if "unknown" in (a["pattern"], b["pattern"])
              else "same_recorded_pattern" if a["pattern"] == b["pattern"]
              else "recorded_pattern_change")
    return {"before": a, "after": b, "comparison": status,
            "clinical_rhythm_stability_verified": False,
            "pacing_transition": "unknown",
            "interpretation": "same_technical_class_does_not_prove_stable_activation_or_absent_pacing"}


def arc_clock(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("invalid_or_nonfinite_trajectory")
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if cumulative[-1] <= 1e-12:
        raise ValueError("zero_arc_length")
    return np.interp(GRID, np.linspace(0, 1, len(points)), cumulative / cumulative[-1])


def direction_time_occupancy(points, *, noise_rms_mv):
    """Absolute noise must be supplied, never inferred from the pathological loop.

    3x noise gate is an exploratory measurement rule, not a validated clinical SNR cutoff.
    Axis signs refer to the supplied transform, not to anatomical tissue regions.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        return abstain("invalid_or_nonfinite_trajectory")
    if noise_rms_mv is None or not np.isfinite(noise_rms_mv) or noise_rms_mv <= 0:
        return abstain("independent_noise_estimate_missing")
    valid = np.linalg.norm(points, axis=1) > 3 * noise_rms_mv
    sectors = (points[:, 0] >= 0).astype(int) * 4 + (points[:, 1] >= 0).astype(int) * 2 + (points[:, 2] >= 0).astype(int)
    bins = np.minimum((np.arange(len(points)) * 3) // len(points), 2)
    counts = np.zeros((3, 8), dtype=int)
    np.add.at(counts, (bins[valid], sectors[valid]), 1)
    denominators = counts.sum(axis=1)
    fractions = [None if n == 0 else (row / n).tolist() for row, n in zip(counts, denominators)]
    return _ok(counts=counts.tolist(), fractions=fractions, valid_counts=denominators.tolist(),
               total_counts=np.bincount(bins, minlength=3).tolist(),
               valid_fraction=float(valid.mean()), sector_order="---,--+,-+-, -++, +--,+-+,++-,+++",
               quality_rule="magnitude_gt_3_times_independent_noise_rms_exploratory",
               interpretation="transform_dependent_global_direction_not_local_activation") if valid.any() else abstain("below_noise_gate")


def serial_arc_clock(before, after):
    """Consume stored descriptors only with identical identity/measurement contracts."""
    for key in ("source", "subject_id", "transform", "version", "signal_path"):
        if not before.get(key) or before.get(key) != after.get(key):
            return abstain(f"incompatible_or_missing_{key}")
    if before["version"] != VERSION:
        return abstain("incompatible_version")
    try:
        a, b = (np.asarray(item["qrs"]["arc_clock"], dtype=float) for item in (before, after))
    except (KeyError, TypeError, ValueError):
        return abstain("qrs_arc_clock_unavailable")
    for curve in (a, b):
        if (curve.shape != GRID.shape or not np.isfinite(curve).all()
                or not np.isclose(curve[0], 0) or not np.isclose(curve[-1], 1)
                or np.any(np.diff(curve) < -1e-12)):
            return abstain("invalid_arc_clock")
    delta = b - a
    for item in (before, after):
        if not isinstance(item["qrs"].get("duration_ms"), (int, float)) or not np.isfinite(item["qrs"]["duration_ms"]) or item["qrs"]["duration_ms"] <= 0:
            return abstain("invalid_qrs_duration")
    area_magnitudes = []
    for item in (before, after):
        vector = np.asarray(item["qrs"].get("area_direction", []), dtype=float)
        area_magnitudes.append(float(np.linalg.norm(vector)) * 1000 if vector.shape == (3,) and np.isfinite(vector).all() else None)
    area_delta = None if any(v is None for v in area_magnitudes) else area_magnitudes[1] - area_magnitudes[0]
    return _ok(delta_curve=delta.tolist(), grid=GRID.tolist(),
               rhythm_context=serial_rhythm_context(before, after),
               similarity_decomposition=serial_similarity_decomposition(
                   before["qrs"].get("phase_xyz_mv"), after["qrs"].get("phase_xyz_mv")),
               absolute_distance=float(np.trapezoid(np.abs(delta), GRID)),
               qrs_duration_delta_ms=after["qrs"]["duration_ms"] - before["qrs"]["duration_ms"],
               qrs_vector_area_delta_mv_ms=area_delta,
               claim_scope="physiological_response_not_treatment_effect")


def serial_similarity_decomposition(before, after):
    """Descriptive proper-rotation fit of baseline-corrected equal-phase XYZ.

    No reflection, translation fit or time warping. Pair provenance is checked
    by serial_arc_clock; this numerical primitive does not establish identity.
    """
    try:
        x, y = (np.asarray(points, dtype=float) for points in (before, after))
    except (TypeError, ValueError):
        return abstain("invalid_phase_trajectory")
    if any(a.shape != (len(GRID), 3) or not np.isfinite(a).all() for a in (x, y)):
        return abstain("missing_or_invalid_phase_trajectory")
    sx, sy = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    if min(sx, sy) <= 1e-12:
        return abstain("zero_phase_trajectory")
    x, y = x / sx, y / sy
    u, s, vt = np.linalg.svd(x.T @ y)
    if s[0] <= 1e-12 or s[1] <= 1e-12 * s[0]:
        return abstain("rotation_non_identifiable")
    sign = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
    if sign < 0 and s[1] - s[2] <= 1e-12 * s[0]:
        return abstain("proper_rotation_non_unique")
    rotation = (u * [1., 1., sign]) @ vt
    gain_scaled = float((s[0] + s[1] + sign*s[2]) / np.sum(x*x))
    gain = gain_scaled * (sy / sx)
    residual = float(np.linalg.norm(y - gain_scaled*x @ rotation) / np.linalg.norm(y))
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(residual):
        return abstain("nonfinite_or_nonpositive_similarity_fit")
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))
    return _ok(gain_after_over_before=gain, rotation_deg=angle,
               rotation_matrix=rotation.tolist(), relative_residual=residual,
               interpretation="global_similarity_not_fluid_volume_or_source_localization",
               alignment="equal_phase_no_time_warping_endpoint_midpoint_baseline",
               quality_scope="numerical_identifiability_only_not_validated_signal_snr")


def rr_history_t_response(beats):
    """No removal/compression of unknown beats from RR history; all timestamps retained.

    >=60 s and >=30 complete explicitly sinus observations are feasibility floors,
    not evidence of reliable restitution estimation. No inferential p-value is emitted.
    """
    if len(beats) < 35:
        return abstain("insufficient_sequence_length")
    times = np.asarray([b.get("r_time_s", np.nan) for b in beats], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return abstain("invalid_beat_time_order")
    if times[-1] - times[0] < 60:
        return abstain("recording_shorter_than_60_seconds")
    rr = np.diff(times)
    rows = []
    for i in range(5, len(beats)):
        if any(b.get("beat_type") != "sinus" for b in beats[i - 5:i + 1]):
            continue
        values = [beats[i].get("t_half_fraction"), rr[i - 1], np.mean(rr[i - 5:i]), beats[i].get("qt_ms")]
        if all(v is not None and np.isfinite(v) for v in values):
            rows.append(values)
    if len(rows) < 30:
        return abstain("insufficient_valid_sinus_observations")
    data = np.asarray(rows)
    x = data[:, 1:]
    scale = x.std(axis=0)
    if np.any(scale < 1e-8):
        return abstain("insufficient_rr_or_qt_variation")
    design = np.column_stack([np.ones(len(x)), (x - x.mean(axis=0)) / scale])
    if np.linalg.matrix_rank(design) < 4 or np.linalg.cond(design) > 100:
        return abstain("collinearity")
    coefficients = np.linalg.lstsq(design, data[:, 0], rcond=None)[0]
    return _ok(n_observations=len(rows), n_beats_total=len(beats),
               standardized_coefficients=coefficients.tolist(),
               comparator="QT and preceding RR history", uncertainty="not_estimated_serial_dependence",
               interpretation="descriptive_within_record_response_not_individual_restitution")


def pvc_sinus_transition(beats):
    """Require explicit beat annotations. Morphology outliers alone are not PVC labels."""
    pairs = []
    for i in range(1, len(beats) - 1):
        pvc, sinus = beats[i:i + 2]
        if pvc.get("beat_type") != "pvc" or sinus.get("beat_type") != "sinus":
            continue
        try:
            coupling = float(pvc["r_time_s"] - beats[i - 1]["r_time_s"])
            pause = float(sinus["r_time_s"] - pvc["r_time_s"])
            a, b = np.asarray(pvc["qrs_clock"]), np.asarray(sinus["qrs_clock"])
            if coupling <= 0 or pause <= 0 or not np.isfinite([coupling, pause]).all():
                continue
            if a.shape != GRID.shape or b.shape != GRID.shape or not np.isfinite([a, b]).all():
                continue
            directions = [np.asarray(beat["qrs_area_direction"], dtype=float) for beat in (pvc, sinus)]
            if any(d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-12 for d in directions):
                continue
            cosine = np.dot(*directions) / np.prod([np.linalg.norm(d) for d in directions])
            pairs.append({"pvc_index": i, "sinus_index": i + 1, "coupling_interval_ms": 1000 * coupling,
                          "post_pvc_interval_ms": 1000 * pause, "delta_curve": (b - a).tolist(),
                          "area_direction_difference_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))})
        except (KeyError, TypeError, ValueError):
            continue
    return _ok(pairs=pairs, n_pairs=len(pairs), interpretation="annotated_morphology_transition_not_purkinje_localization") if pairs else abstain("no_valid_explicitly_annotated_pvc_sinus_pairs")


def record_descriptor(delineation, xyz, fs, *, source, subject_id, transform, noise_rms_mv=None):
    """Extraction adapter: use original record coordinates, never median-template indices."""
    from .biomarkers.thesis_derived_biomarkers import _beat_sample
    from .delineation.base import measurement_confidence_threshold
    def sample(beat, attribute):
        return _beat_sample(beat, attribute, "t" if attribute.startswith("t_") else "qrs")
    xyz = np.asarray(xyz, dtype=float)
    output = {"version": VERSION, "source": str(source) if source is not None else "", "subject_id": str(subject_id) if subject_id is not None else "",
              "transform": transform, "signal_path": "record_morphology", "clinical_claim_allowed": False}
    rhythm = getattr(delineation, "rhythm", None)
    output["rhythm_context"] = _rhythm_context(
        getattr(rhythm, "rhythm_class", None), getattr(rhythm, "confidence", None))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(fs) or fs <= 0:
        return {**output, "qrs": abstain("invalid_signal")}

    segment_cache = {}
    def segment(beat, name):
        key = (id(beat), name)
        if key in segment_cache:
            return segment_cache[key]
        left, right = ("r_onset", "r_offset") if name == "qrs" else ("t_onset", "t_offset")
        start, end = sample(beat, left), sample(beat, right)
        confidence = beat.segment_confidence(name)
        if (not np.isfinite(confidence) or not measurement_confidence_threshold(name) <= confidence <= 1
                or start is None or end is None or not 0 <= start < end < len(xyz)):
            raise ValueError("invalid_segment_boundaries_or_quality")
        points = xyz[start:end + 1]
        clock = arc_clock(points)
        # Baseline correction preserves arc lengths and defines direction explicitly.
        points = points - (points[0] + points[-1]) / 2
        area = np.trapezoid(points, dx=1 / fs, axis=0)
        result = ({"arc_clock": np.round(clock, ARC_DECIMALS).tolist(),
                   "duration_ms": round((end - start) / fs * 1000, 3),
                   "area_direction": np.round(area, 6).tolist()}, points)
        segment_cache[key] = result
        return result

    representative = getattr(delineation, "representative_beat", None)
    try:
        if representative is None:
            raise ValueError("representative_beat_missing")
        output["qrs"], points = segment(representative, "qrs")
        phase = np.linspace(0., 1., len(points))
        output["qrs"]["phase_xyz_mv"] = np.round(np.column_stack([
            np.interp(GRID, phase, points[:, axis]) for axis in range(3)
        ]), MILLIVOLT_DECIMALS).tolist()
        output["direction_time_occupancy"] = direction_time_occupancy(points, noise_rms_mv=noise_rms_mv)
    except ValueError as exc:
        output["qrs"] = abstain(str(exc))
        output["direction_time_occupancy"] = abstain("qrs_unavailable")
    # The usual 10-second recordings cannot support the RR-history model.
    # Do not extract every beat twice merely to discover this afterwards.
    has_annotations = any(getattr(b, "beat_type_source", "") for b in delineation.beats)
    has_pvc = any(getattr(b, "beat_type", "unknown") == "pvc" and getattr(b, "beat_type_source", "") for b in delineation.beats)
    if not has_pvc and (len(delineation.beats) < 35 or not has_annotations):
        output["rr_history_t_response"] = abstain("insufficient_sequence_length" if len(delineation.beats) < 35 else "explicit_sinus_annotations_missing")
        output["pvc_sinus_transition"] = abstain("no_valid_explicitly_annotated_pvc_sinus_pairs")
        return output
    sequence = []
    for beat in delineation.beats:
        r = sample(beat, "r_peak")
        entry = {"r_time_s": None if r is None else r / fs,
                 "beat_type": getattr(beat, "beat_type", "unknown") if getattr(beat, "beat_type_source", "") else "unknown"}
        try:
            qrs, _ = segment(beat, "qrs")
            entry.update(qrs_clock=qrs["arc_clock"], qrs_area_direction=qrs["area_direction"])
            t, _ = segment(beat, "t")
            entry["t_half_fraction"] = t["arc_clock"][16]
            entry["qt_ms"] = (sample(beat, "t_offset") - sample(beat, "r_onset")) / fs * 1000
        except ValueError:
            pass  # Missing segment remains explicit; beat stays in temporal sequence.
        sequence.append(entry)
    output["rr_history_t_response"] = rr_history_t_response(sequence)
    output["pvc_sinus_transition"] = pvc_sinus_transition(sequence)
    return output
