from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sp_signal
from .running_median import centered_median


def bandpass_filter(ecg: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    highcut = min(float(highcut), nyq * 0.95)
    if not 0 < lowcut < highcut:
        raise ValueError(
            f"Invalid bandpass frequencies for fs={fs}: {lowcut}-{highcut} Hz"
        )
    low = lowcut / nyq
    high = highcut / nyq
    sos = sp_signal.butter(order, [low, high], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, ecg, axis=0)


def highpass_filter(ecg: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 5) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = sp_signal.butter(order, normal_cutoff, btype="high", output="sos")
    return sp_signal.sosfiltfilt(sos, ecg, axis=0)


def notch_filter(ecg: np.ndarray, fs: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    if freq <= 0 or freq >= fs / 2:
        return np.asarray(ecg, dtype=float).copy()
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.sosfiltfilt(sp_signal.tf2sos(b, a), ecg, axis=0)


def remove_baseline_wander(ecg: np.ndarray, fs: int) -> np.ndarray:
    window_size = int(2.0 * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = centered_median(ecg, window_size)
    return ecg - baseline


def clean_ecg(ecg: np.ndarray, fs: int, powerline: float = 50.0) -> np.ndarray:
    filtered = highpass_filter(ecg, fs, cutoff=0.5, order=5)
    filtered = notch_filter(filtered, fs, freq=powerline)
    filtered = remove_baseline_wander(filtered, fs)
    return filtered


def prepare_signal_paths(
    ecg: np.ndarray,
    fs: int,
    powerline: float = 50.0,
    *,
    notch: bool = True,
) -> dict[str, np.ndarray]:
    """Create separate signals for amplitudes, delineation and morphology.

    The diagnostic path retains clinically relevant low-frequency ST content,
    while the delineation path remains intentionally narrower and compatible
    with the existing delineators. ``notch=False`` skips the powerline notch
    for recordings the device already notched at that frequency: a second
    zero-phase notch adds nothing and rings symmetrically around steep QRS
    edges (audit S8).
    """
    raw = np.asarray(ecg, dtype=float)
    diagnostic = bandpass_filter(
        raw,
        fs,
        lowcut=0.05,
        highcut=min(150.0, fs * 0.45),
        order=3,
    )
    if notch:
        diagnostic = notch_filter(diagnostic, fs, freq=powerline)

    delineation = bandpass_filter(raw, fs, lowcut=0.5, highcut=40.0, order=4)
    if notch:
        delineation = notch_filter(delineation, fs, freq=powerline)
    delineation = remove_baseline_wander(delineation, fs)

    # Diagnostic amplitudes and morphology require the same adult diagnostic
    # bandwidth. Reuse the exact array instead of performing the identical
    # zero-phase filter and notch twice.
    morphology = diagnostic

    # Adult diagnostic recorders in the primary cohort expose an analogue
    # upper corner of 150 Hz.  A 500-Hz ADC does not recreate content above
    # that front-end corner, so HF-QRS markers use the common, genuinely
    # supported 80--150-Hz sub-band.  ``wideband`` intentionally aliases the
    # already filtered diagnostic array; acquisition-metadata gating happens
    # at the cohort boundary before this source is handed to a biomarker.
    wideband = diagnostic
    return {
        "raw": raw,
        "diagnostic": diagnostic,
        "delineation": delineation,
        "morphology": morphology,
        "wideband": wideband,
    }


def resample_signal(
    ecg: np.ndarray,
    fs_orig: float,
    fs_target: float,
) -> np.ndarray:
    """Anti-aliased rational resampling without a periodic-edge assumption."""

    original = float(fs_orig)
    target = float(fs_target)
    if not np.isfinite(original) or not np.isfinite(target) or min(original, target) <= 0:
        raise ValueError("Sampling rates must be finite and positive")
    values = np.asarray(ecg, dtype=float)
    if np.isclose(original, target, rtol=0.0, atol=1e-12):
        return values
    ratio = Fraction(target / original).limit_denominator(100_000)
    output = sp_signal.resample_poly(
        values,
        ratio.numerator,
        ratio.denominator,
        axis=0,
        padtype="line",
    )
    expected = int(round(values.shape[0] * target / original))
    if output.shape[0] > expected:
        output = output[:expected]
    elif output.shape[0] < expected:
        pad_width = [(0, expected - output.shape[0])] + [
            (0, 0) for _ in range(output.ndim - 1)
        ]
        output = np.pad(output, pad_width, mode="edge")
    return output


def construct_median_beat(ecg: np.ndarray, r_peaks: np.ndarray, fs: int, pre_r: float = 0.3, post_r: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    if len(r_peaks) < 2:
        return ecg, r_peaks

    pre_samples = int(pre_r * fs)
    post_samples = int(post_r * fs)
    beat_length = pre_samples + post_samples

    beats = []
    valid_peaks = []

    for rp in r_peaks:
        start = int(rp) - pre_samples
        end = int(rp) + post_samples
        if start >= 0 and end <= ecg.shape[0]:
            beats.append(ecg[start:end, :])
            valid_peaks.append(pre_samples)

    if len(beats) < 2:
        return ecg, r_peaks

    beats_arr = np.array(beats)

    if beats_arr.shape[1] != beat_length:
        min_len = min(b.shape[0] for b in beats)
        beats_arr = np.array([b[:min_len] for b in beats])

    median_beat = np.median(beats_arr, axis=0)

    return median_beat, np.array([pre_samples])


_MORPHOLOGY_TEMPLATE_FIDUCIALS = (
    "p_onset",
    "p_peak",
    "p_offset",
    "r_onset",
    "q_peak",
    "r_peak",
    "s_peak",
    "r_offset",
    "j_point",
    "t_onset",
    "t_peak",
    "t_offset",
)


def construct_robust_morphology_record(
    ecg: np.ndarray,
    delineation,
    *,
    minimum_beats: int = 2,
    minimum_marker_confidence: float = 0.20,
    padding_ms: float = 20.0,
    morphology_cluster_id: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace the representative cycle with a fiducial-warped median cycle.

    Record-level rhythm and variability features still use the untouched
    record. Morphology features that address the representative cycle receive
    a pointwise median of accepted, non-outlier beats aligned at corresponding
    P/QRS/T fiducials. This avoids letting one otherwise plausible medoid beat
    dominate loop geometry and derivative features.

    ``morphology_cluster_id`` selects a specific beat population identified
    by ``FusionDelineator._mark_morphology_outliers`` (0 = primary/dominant,
    1/2 = secondary/tertiary) instead of the default "all accepted beats
    minus outliers" set. A non-primary cluster builds its own alignment
    anchor from its most confident member, since the record-level
    ``representative_beat`` reflects the dominant population only.
    """

    signal = np.asarray(ecg, dtype=float)
    diagnostics: dict[str, object] = {
        "status": "not_evaluable",
        "candidate_beat_count": 0,
        "used_beat_count": 0,
        "excluded_morphology_outlier_count": 0,
        "minimum_beats": int(minimum_beats),
        "alignment": "piecewise_linear_fiducial_warp",
        "aggregation": "pointwise_median",
        "baseline": "per_beat_isoelectric_median",
        "morphology_cluster_id": morphology_cluster_id,
    }
    if signal.ndim != 2 or signal.shape[0] < 3 or signal.shape[1] < 1:
        diagnostics["reason"] = "invalid_signal_shape"
        return signal.copy(), diagnostics
    if not np.all(np.isfinite(signal)):
        diagnostics["reason"] = "non_finite_signal"
        return signal.copy(), diagnostics

    fs = int(getattr(delineation, "fs", 0))
    accepted = tuple(getattr(delineation, "accepted_beats", ()))
    if morphology_cluster_id is None:
        representative = getattr(delineation, "representative_beat", None)
        diagnostics["candidate_beat_count"] = len(accepted)
        diagnostics["excluded_morphology_outlier_count"] = sum(
            bool(getattr(beat, "morphology_outlier", False))
            for beat in accepted
        )
        candidates = tuple(
            beat
            for beat in accepted
            if not bool(getattr(beat, "morphology_outlier", False))
        )
    else:
        candidates = tuple(
            beat
            for beat in accepted
            if getattr(beat, "morphology_cluster_id", None)
            == morphology_cluster_id
        )
        diagnostics["candidate_beat_count"] = len(candidates)
        representative = _select_cluster_representative(candidates)
    if fs <= 0 or representative is None:
        diagnostics["reason"] = "representative_beat_unavailable"
        return signal.copy(), diagnostics
    if len(candidates) < int(minimum_beats):
        diagnostics["reason"] = "too_few_non_outlier_beats"
        return signal.copy(), diagnostics

    first_attribute = (
        "p_onset"
        if (
            _wave_is_present(representative, "p")
            and float(delineation.segment_confidence("p")) >= 0.35
            and getattr(representative, "p_onset", None) is not None
        )
        else "r_onset"
    )
    if getattr(representative, first_attribute, None) is None:
        first_attribute = "r_peak"
    last_attribute = (
        "t_offset"
        if (
            _wave_is_present(representative, "t")
            and float(delineation.segment_confidence("t")) >= 0.35
            and getattr(representative, "t_offset", None) is not None
        )
        else "r_offset"
    )
    if getattr(representative, last_attribute, None) is None:
        last_attribute = "r_peak"

    target_first = getattr(representative, first_attribute)
    target_last = getattr(representative, last_attribute)
    padding = max(0, int(round(float(padding_ms) / 1000.0 * fs)))
    target_start = max(0, int(target_first.sample) - padding)
    target_end = min(signal.shape[0] - 1, int(target_last.sample) + padding)
    if target_end - target_start < 4:
        diagnostics["reason"] = "representative_window_too_short"
        return signal.copy(), diagnostics

    target_grid = np.arange(target_start, target_end + 1, dtype=float)
    target_baseline = _beat_isoelectric_baseline(
        signal,
        representative,
        fs,
    )
    warped_beats: list[np.ndarray] = []
    used_r_peaks: list[int] = []
    used_anchor_sets: list[tuple[str, ...]] = []
    for beat in candidates:
        source_first = getattr(beat, first_attribute, None)
        source_last = getattr(beat, last_attribute, None)
        if source_first is None or source_last is None:
            continue
        source_start = int(source_first.sample) - padding
        source_end = int(source_last.sample) + padding
        if source_start < 0 or source_end >= signal.shape[0]:
            continue
        if source_end - source_start < 4:
            continue

        pairs: list[tuple[float, float, str]] = [
            (float(target_start), float(source_start), "window_start")
        ]
        for attribute in _MORPHOLOGY_TEMPLATE_FIDUCIALS:
            target_marker = getattr(representative, attribute, None)
            source_marker = getattr(beat, attribute, None)
            if target_marker is None or source_marker is None:
                continue
            if (
                float(getattr(target_marker, "confidence", 0.0))
                < minimum_marker_confidence
                or float(getattr(source_marker, "confidence", 0.0))
                < minimum_marker_confidence
            ):
                continue
            target_sample = float(target_marker.sample)
            source_sample = float(source_marker.sample)
            if not (
                target_start < target_sample < target_end
                and source_start < source_sample < source_end
            ):
                continue
            pairs.append((target_sample, source_sample, attribute))
        pairs.append((float(target_end), float(source_end), "window_end"))
        pairs.sort(key=lambda item: (item[0], item[1]))

        monotonic: list[tuple[float, float, str]] = []
        for pair in pairs:
            if (
                monotonic
                and (
                    pair[0] <= monotonic[-1][0]
                    or pair[1] <= monotonic[-1][1]
                )
            ):
                continue
            monotonic.append(pair)
        anchor_names = tuple(item[2] for item in monotonic)
        if (
            len(monotonic) < 3
            or "r_peak" not in anchor_names
            or monotonic[-1][2] != "window_end"
        ):
            continue

        target_anchors = np.asarray(
            [item[0] for item in monotonic],
            dtype=float,
        )
        source_anchors = np.asarray(
            [item[1] for item in monotonic],
            dtype=float,
        )
        source_positions = np.interp(
            target_grid,
            target_anchors,
            source_anchors,
        )
        if (
            float(np.min(source_positions)) < 0.0
            or float(np.max(source_positions)) > signal.shape[0] - 1
        ):
            continue
        source_grid = np.arange(signal.shape[0], dtype=float)
        warped = np.column_stack(
            [
                np.interp(source_positions, source_grid, signal[:, lead])
                for lead in range(signal.shape[1])
            ]
        )
        warped -= _beat_isoelectric_baseline(signal, beat, fs)
        if not np.all(np.isfinite(warped)):
            continue
        warped_beats.append(warped)
        used_r_peaks.append(int(beat.r_peak.sample))
        used_anchor_sets.append(anchor_names)

    diagnostics["used_beat_count"] = len(warped_beats)
    diagnostics["used_r_peaks"] = used_r_peaks
    if len(warped_beats) < int(minimum_beats):
        diagnostics["reason"] = "too_few_warpable_beats"
        return signal.copy(), diagnostics

    template = np.median(np.stack(warped_beats, axis=0), axis=0)
    template += target_baseline
    output = signal.copy()
    original_window = output[target_start : target_end + 1]
    weights = np.ones(len(template), dtype=float)
    core_start = int(target_first.sample) - target_start
    core_end = int(target_last.sample) - target_start
    if core_start > 0:
        weights[: core_start + 1] = np.linspace(0.0, 1.0, core_start + 1)
    if core_end < len(weights) - 1:
        weights[core_end:] = np.linspace(
            1.0,
            0.0,
            len(weights) - core_end,
        )
    output[target_start : target_end + 1] = (
        original_window * (1.0 - weights[:, None])
        + template * weights[:, None]
    )
    diagnostics.update(
        {
            "status": "used",
            "reason": "",
            "target_start_sample": target_start,
            "target_end_sample": target_end,
            "target_first_fiducial": first_attribute,
            "target_last_fiducial": last_attribute,
            "padding_ms": float(padding_ms),
            "anchor_names": sorted(
                {
                    anchor
                    for anchors in used_anchor_sets
                    for anchor in anchors
                }
            ),
        }
    )
    return output, diagnostics


def _wave_is_present(beat, wave: str) -> bool:
    try:
        return beat.resolved_wave_status(wave).value == "present"
    except (AttributeError, ValueError):
        return False


def _select_cluster_representative(cluster_beats):
    """Pick an anchor beat for a non-primary morphology cluster's template.

    The record-level ``representative_beat`` is a consensus built for the
    dominant (cluster 0) population, so a secondary/tertiary cluster needs
    its own alignment anchor. The beat with the highest R-peak confidence
    among those with established QRS bounds stands in for that role.
    """
    with_qrs = [
        beat
        for beat in cluster_beats
        if getattr(beat, "r_onset", None) is not None
        and getattr(beat, "r_offset", None) is not None
    ]
    pool = with_qrs or list(cluster_beats)
    if not pool:
        return None
    return max(pool, key=lambda beat: beat.r_peak.confidence)


def _beat_isoelectric_baseline(
    signal: np.ndarray,
    beat,
    fs: int,
) -> np.ndarray:
    qrs_onset = getattr(beat, "r_onset", None)
    p_offset = getattr(beat, "p_offset", None)
    intervals: list[tuple[int, int]] = []
    if qrs_onset is not None and p_offset is not None:
        intervals.append(
            (
                int(p_offset.sample) + int(round(0.004 * fs)),
                int(qrs_onset.sample) - int(round(0.008 * fs)),
            )
        )
    if qrs_onset is not None:
        intervals.append(
            (
                int(qrs_onset.sample) - int(round(0.080 * fs)),
                int(qrs_onset.sample) - int(round(0.020 * fs)),
            )
        )
    r_peak = getattr(beat, "r_peak", None)
    if r_peak is not None:
        intervals.append(
            (
                int(r_peak.sample) - int(round(0.140 * fs)),
                int(r_peak.sample) - int(round(0.080 * fs)),
            )
        )
    for start, end in intervals:
        start = max(0, start)
        end = min(signal.shape[0], end)
        if end - start >= 3:
            return np.median(signal[start:end], axis=0, keepdims=True)
    return np.zeros((1, signal.shape[1]), dtype=float)


def detect_r_peaks_multilead(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Detect QRS reference peaks using independent multilead evidence.

    A single lead can be nearly isoelectric, inverted, noisy, or affected by
    lead-specific pathology. Candidate peaks are therefore collected from a
    spatial-energy detector and several usable leads, clustered in time, and
    refined on polarity-invariant multilead QRS magnitude.
    """

    signal = np.asarray(ecg, dtype=float)
    if signal.ndim != 2 or signal.shape[0] < max(8, int(0.5 * fs)):
        return np.empty(0, dtype=int)
    if not np.all(np.isfinite(signal)):
        return np.empty(0, dtype=int)

    filtered = bandpass_filter(
        signal,
        fs,
        lowcut=5.0,
        highcut=min(30.0, fs * 0.45),
        order=3,
    )
    centered = filtered - np.median(filtered, axis=0, keepdims=True)
    scale = np.percentile(np.abs(centered), 95, axis=0)
    good = (scale >= 0.01) & (scale <= 12.0)
    if not np.any(good):
        return np.empty(0, dtype=int)
    normalized = centered[:, good] / np.maximum(scale[good], 0.02)
    spatial_magnitude = np.sqrt(np.median(normalized**2, axis=1))
    spatial_magnitude = sp_signal.savgol_filter(
        spatial_magnitude,
        window_length=_valid_odd_window(
            len(spatial_magnitude),
            max(5, int(round(0.018 * fs))),
        ),
        polyorder=2,
        mode="interp",
    )
    derivative = np.gradient(normalized, axis=0)
    qrs_energy = np.sqrt(np.median(derivative**2, axis=1))
    integration_width = max(1, int(round(0.060 * fs)))
    qrs_energy = np.convolve(
        qrs_energy,
        np.ones(integration_width) / integration_width,
        mode="same",
    )

    refractory = max(1, int(round(0.18 * fs)))
    energy_baseline = float(np.median(qrs_energy))
    energy_noise = max(
        1e-8,
        1.4826
        * float(np.median(np.abs(qrs_energy - energy_baseline))),
    )
    energy_prominence = max(
        3.0 * energy_noise,
        float(np.percentile(qrs_energy, 80) - energy_baseline),
    )
    energy_peaks, _ = sp_signal.find_peaks(
        qrs_energy,
        distance=refractory,
        prominence=energy_prominence,
    )

    candidates: list[tuple[int, str]] = [
        (int(sample), "spatial_energy") for sample in energy_peaks
    ]
    good_indices = np.flatnonzero(good)
    preferred = [1, 0, 5, 6, 7, 8, 9, 10, 11]
    lead_order = [
        index for index in preferred if index in set(good_indices.tolist())
    ]
    lead_order.extend(
        index
        for index in good_indices
        if int(index) not in lead_order
    )
    try:
        from neurokit2.ecg import ecg_peaks
    except ImportError:
        ecg_peaks = None
    if ecg_peaks is not None:
        for lead_index in lead_order[:4]:
            try:
                _, info = ecg_peaks(
                    filtered[:, lead_index],
                    sampling_rate=fs,
                    method="neurokit",
                )
            except Exception:
                continue
            for sample in np.asarray(
                info.get("ECG_R_Peaks", ()),
                dtype=int,
            ):
                if 0 <= sample < len(signal):
                    candidates.append((int(sample), f"lead_{lead_index}"))

    if not candidates:
        fallback, _ = sp_signal.find_peaks(
            spatial_magnitude,
            distance=refractory,
            prominence=max(
                0.15,
                float(np.percentile(spatial_magnitude, 75)) * 0.35,
            ),
        )
        candidates = [(int(sample), "spatial_fallback") for sample in fallback]
    if not candidates:
        return np.empty(0, dtype=int)

    tolerance = max(1, int(round(0.060 * fs)))
    ordered = sorted(candidates, key=lambda item: item[0])
    clusters: list[list[tuple[int, str]]] = []
    for candidate in ordered:
        if (
            not clusters
            or candidate[0]
            - int(np.median([item[0] for item in clusters[-1]]))
            > tolerance
        ):
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    accepted: list[int] = []
    edge_margin = max(1, int(round(0.25 * fs)))
    for cluster in clusters:
        sources = {source for _, source in cluster}
        if (
            "spatial_energy" not in sources
            and "spatial_fallback" not in sources
            and len(sources) < 2
        ):
            continue
        center = int(round(np.median([sample for sample, _ in cluster])))
        lo = max(0, center - int(round(0.055 * fs)))
        hi = min(len(signal), center + int(round(0.055 * fs)) + 1)
        if hi <= lo:
            continue
        refined = lo + int(np.argmax(spatial_magnitude[lo:hi]))
        if refined < edge_margin or refined >= len(signal) - edge_margin:
            continue
        accepted.append(refined)

    if not accepted:
        return np.empty(0, dtype=int)
    accepted = sorted(set(accepted))
    resolved: list[int] = []
    for sample in accepted:
        if not resolved or sample - resolved[-1] >= refractory:
            resolved.append(sample)
            continue
        if qrs_energy[sample] > qrs_energy[resolved[-1]]:
            resolved[-1] = sample
    return _reject_non_qrs_peaks(np.asarray(resolved, dtype=int), qrs_energy, fs)


def _reject_non_qrs_peaks(
    peaks: np.ndarray,
    qrs_energy: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Drop P/T waves that passed the candidate stage as QRS complexes.

    In slow ventricular rhythms (AV block, bradycardia) the fixed prominence
    lets P and T waves through: on LUDB and INCART 232 of 430 false
    detections were such waves, with slope energy 2-20 % of the record's
    strong beats, while no annotated beat, ectopic beats included (lowest
    23 %), fell below 15 %. A candidate within 360 ms of the previous
    accepted beat with less than half its slope energy is a T wave (Pan &
    Tompkins 1985, PMID 3997178). The previous beat counts at most as much
    as a typical strong beat, so an artifact spike cannot suppress the
    next true QRS. Against the reference annotations of LUDB, INCART,
    QTDB, NSTDB and AFDB, extra detections fell from 5,479 to 4,681 while
    missed beats rose by 11 of 47,000, all short-coupled beats in atrial
    fibrillation, most in AFDB 07162, where two thirds are missed either
    way (audit_tools/qrs_detection_regression.py).
    """

    if len(peaks) < 3:
        return peaks
    half = max(1, int(round(0.050 * fs)))
    energy = np.array(
        [
            float(np.max(qrs_energy[max(0, peak - half):peak + half + 1]))
            for peak in peaks
        ]
    )
    strong = float(np.median(energy[energy >= np.median(energy)]))
    if not strong > 0.0:
        return peaks
    t_window = int(round(0.360 * fs))
    kept: list[int] = []
    kept_energy: list[float] = []
    for peak, value in zip(peaks, energy):
        if value < 0.15 * strong:
            continue
        if (
            kept
            and peak - kept[-1] < t_window
            and value < 0.5 * min(kept_energy[-1], strong)
        ):
            continue
        kept.append(int(peak))
        kept_energy.append(float(value))
    return np.asarray(kept, dtype=int)


def _valid_odd_window(length: int, requested: int) -> int:
    if length < 3:
        return 1
    window = min(length if length % 2 else length - 1, requested)
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def preprocess_record(
    ecg: np.ndarray,
    fs: int,
    r_peaks: np.ndarray | None = None,
    powerline: float = 50.0,
) -> dict:
    paths = prepare_signal_paths(ecg, fs, powerline=powerline)
    cleaned = paths["delineation"]

    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = detect_r_peaks_multilead(cleaned, fs)

    median_beat, median_r_peaks = construct_median_beat(cleaned, r_peaks, fs)

    return {
        "raw": paths["raw"],
        "diagnostic": paths["diagnostic"],
        "morphology": paths["morphology"],
        "delineation": cleaned,
        "cleaned": cleaned,
        "r_peaks": r_peaks,
        "median_beat": median_beat,
        "median_r_peaks": median_r_peaks,
        "fs": fs,
    }
