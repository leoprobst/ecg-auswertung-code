"""Outcome-blind 1% signal sensitivity sample, using the published angle kernel.

Variants are hypothetical measurement perturbations, not independent repeats.
Fixed noise amplitudes are stress scenarios, not an estimated instrument noise.
"""

import hashlib
from typing import Any

import numpy as np

from .biomarkers.advanced_vcg_biomarkers import direct_qrs_t_angles_8lead
from .biomarkers.qrs_t_method_comparison import compare_qrs_t_methods

VERSION = "1.0.0"
SEED = 20260914


def selected_for_sensitivity(source, record_id):
    digest = hashlib.sha256(f"{SEED}:{source}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 100 == 0


def angle_sensitivity(signal, delineation, fs, *, source, record_id, force=False):
    result: dict[str, Any] = {
        "version": VERSION,
        "seed": SEED,
        "sampling_probability": 0.01,
        "clinical_claim_allowed": False,
        "interval_kind": "sensitivity_not_confidence_interval",
        "status": "not_sampled",
        "variants": [],
        "forced_for_test": bool(force),
    }
    if not force and not selected_for_sensitivity(source, record_id):
        return result
    result["status"] = "non_estimable"
    candidates = []
    for beat in delineation.beats:
        confidence_method = getattr(beat, "segment_confidence", None)
        if not callable(confidence_method):
            continue
        confidences = np.array([confidence_method(wave) for wave in ("pr", "qrs", "t")])
        if not np.isfinite(confidences).all() or (confidences < 0.7).any():
            continue
        if getattr(beat, "boundary_conflict", False):
            continue
        points = [
            getattr(beat, name, None)
            for name in ("p_offset", "r_onset", "r_offset", "t_offset")
        ]
        if any(point is None for point in points):
            continue
        b, q, j, t = [int(point.sample) for point in points if point is not None]
        if 0 <= b < q - 1 < j < t < len(signal):
            candidates.append((beat, b, q, j, t))
    if not candidates:
        result["reason"] = "no_quality_qualified_pr_qrs_t_beat"
        return result
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != 12 or not np.isfinite(fs) or fs <= 0:
        result["reason"] = "invalid_signal_or_sampling_rate"
        return result
    beat, b, q, j, t = candidates[
        0
    ]  # deterministic earliest eligible beat, not best marker
    kwargs = dict(qrs_bounds=(q, j), t_bounds=(j, t), baseline_bounds=(b, q - 1))
    baseline = direct_qrs_t_angles_8lead(signal, **kwargs)
    result["baseline"] = {
        "area_angle_deg": baseline[0],
        "integral_angle_deg": baseline[1],
    }
    result["beat_r_sample"] = int(beat.r_peak.sample)
    result["transform_comparison"] = compare_qrs_t_methods(signal, **kwargs)

    def add(family, name, values):
        result["variants"].append(
            {
                "family": family,
                "scenario": name,
                "area_angle_deg": values[0],
                "integral_angle_deg": values[1],
                "status": "available"
                if all(v is not None for v in values)
                else "non_estimable",
            }
        )

    for name, point, original in (
        ("qrs_onset", beat.r_onset, q),
        ("qrs_offset", beat.r_offset, j),
        ("t_offset", beat.t_offset, t),
    ):
        uncertainty = float(getattr(point, "uncertainty_ms", 0))
        # Fixed 4 ms floor is a declared stress scenario, not calibrated error.
        shift = max(
            1,
            round(
                max(4.0, uncertainty if np.isfinite(uncertainty) else 4.0) * fs / 1000
            ),
        )
        for direction in (-1, 1):
            qq, jj, tt = q, j, t
            changed = original + direction * shift
            if name == "qrs_onset":
                qq = changed
            elif name == "qrs_offset":
                jj = changed
            else:
                tt = changed
            values = (
                direct_qrs_t_angles_8lead(
                    signal,
                    qrs_bounds=(qq, jj),
                    t_bounds=(jj, tt),
                    baseline_bounds=(b, min(q - 1, qq - 1)),
                )
                if b < qq - 1 < jj < tt < len(signal)
                else (None, None)
            )
            add("fiducial", f"{name}:{direction * shift}_samples", values)
    # Same perturbation realisations for every marker; no repeat filtering/delineation.
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(f"{SEED}:{source}:{record_id}".encode()).digest()[:8],
            "little",
        )
    )
    noise = rng.normal(size=(len(signal), 12))
    noise[:, 2] = noise[:, 1] - noise[:, 0]
    noise[:, 3] = -(noise[:, 0] + noise[:, 1]) / 2
    noise[:, 4] = noise[:, 0] - noise[:, 1] / 2
    noise[:, 5] = noise[:, 1] - noise[:, 0] / 2
    for rms in (0.005, 0.01):
        add(
            "synthetic_noise",
            f"independent_lead_noise_{rms}_mV",
            direct_qrs_t_angles_8lead(signal + rms * noise, **kwargs),
        )
    for name, perturbed in (
        ("common_gain_1.02", signal * 1.02),
        ("baseline_shift_0.01mV", signal + 0.01),
    ):
        add("invariance_control", name, direct_qrs_t_angles_8lead(perturbed, **kwargs))
    for other, bb, qq, jj, tt in candidates[1:4]:
        if (
            other.morphology_cluster_id is not None
            and other.morphology_cluster_id == beat.morphology_cluster_id
        ):
            add(
                "same_cluster_observed_beats",
                str(other.r_peak.sample),
                direct_qrs_t_angles_8lead(
                    signal,
                    qrs_bounds=(qq, jj),
                    t_bounds=(jj, tt),
                    baseline_bounds=(bb, qq - 1),
                ),
            )
    result["status"] = (
        "available" if all(v is not None for v in baseline) else "non_estimable"
    )
    return result
