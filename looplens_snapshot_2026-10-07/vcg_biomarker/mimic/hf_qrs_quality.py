"""Noise and quantisation control for the 80-150 Hz QRS markers (audit S7).

MIMIC-IV-ECG is sampled at 500 Hz behind a 0.005-150 Hz analogue front end
and quantised at 5 uV (200 adu/mV). Classical high-frequency QRS work
measures 150-250 Hz on averaged beats at >= 1 kHz with low-noise
acquisition (Traegardh & Schlegel 2007, PMID 17564667; Ringborn et al.
2010, PMID 20060122). In the 80-150 Hz band at the analogue corner the
signal is a few microvolts, close to the quantisation step. A finding from
these markers is therefore only reportable next to the band's own
signal-to-noise ratio.

Per recording, on the Kors VCG of the wideband path and the same axiswise
80-150 Hz zero-phase band the markers use:

* signal RMS in the QRS windows of the accepted beats,
* noise RMS in their TP windows (T offset + 40 ms to P onset - 20 ms of the
  next beat; no TP window, no noise estimate),
* the SNR of one beat and after averaging n beats (+10 log10 n dB, the gain
  for uncorrelated noise),
* the noise RMS relative to the band-limited quantisation noise the Kors
  weights carry from the 5-uV step.

These are context columns: they are not screened and never enlarge a
multiplicity family.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

HF_QRS_QUALITY_VERSION = "1.0.0"
HF_QRS_BAND_MARKERS = frozenset(
    {
        "qrs_fragmentation_high_frequency_energy",
        "terminal_qrs_hf_energy_fraction",
        "hf_qrs_reduced_amplitude_zone_depth",
    }
)
HF_BAND_HZ = (80.0, 150.0)
QUANTISATION_STEP_UV = 5.0
MINIMUM_REPORTABLE_AVERAGED_SNR_DB = 10.0
_TP_START_AFTER_T_OFFSET_S = 0.04
_TP_END_BEFORE_P_ONSET_S = 0.02
_MINIMUM_TP_WINDOW_S = 0.04

EMPTY_HF_QRS_QUALITY = {
    "context_hf_qrs_signal_rms_uv": np.nan,
    "context_hf_qrs_noise_rms_uv": np.nan,
    "context_hf_qrs_single_beat_snr_db": np.nan,
    "context_hf_qrs_averaged_snr_db": np.nan,
    "context_hf_qrs_noise_to_quantisation_ratio": np.nan,
    "context_hf_qrs_snr_beat_count": 0,
}


def band_limited_quantisation_rms_uv(fs: float) -> float:
    """RMS of 5-uV quantisation noise in the band, through the Kors weights.

    Uniform quantisation noise is white with variance step^2 / 12 per lead;
    the band keeps (high - low) / (fs / 2) of it, and each Kors axis sums
    eight leads with its weights, so the spatial variance scales with the
    squared weights over all three axes.
    """
    from ..core.vcg_transform import KORS_MATRIX

    low, high = HF_BAND_HZ
    band_fraction = (high - low) / (fs / 2.0)
    per_lead_variance = QUANTISATION_STEP_UV ** 2 / 12.0 * band_fraction
    return float(np.sqrt(per_lead_variance * float(np.sum(KORS_MATRIX ** 2))))


def hf_qrs_signal_quality(
    xyz_mv: Optional[np.ndarray],
    beats,
    fs: float,
) -> dict[str, float]:
    """Context columns for one recording; NaN where not measurable."""
    from scipy import signal as sp_signal

    result = dict(EMPTY_HF_QRS_QUALITY)
    if xyz_mv is None or fs <= 2 * HF_BAND_HZ[1]:
        return result
    points = np.asarray(xyz_mv, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return result
    nyquist = fs / 2.0
    sos = sp_signal.butter(
        3, [HF_BAND_HZ[0] / nyquist, HF_BAND_HZ[1] / nyquist],
        btype="bandpass", output="sos",
    )
    if len(points) <= 3 * (2 * len(sos) + 1):
        return result
    energy = np.sum(sp_signal.sosfiltfilt(sos, points, axis=0) ** 2, axis=1)
    energy_uv2 = energy * 1e6  # mV^2 -> uV^2

    def sample(point) -> Optional[int]:
        return None if point is None else int(point.sample)

    ordered = sorted(
        (beat for beat in beats if sample(beat.r_onset) is not None),
        key=lambda beat: beat.r_onset.sample,
    )
    qrs_energy: list[np.ndarray] = []
    noise_energy: list[np.ndarray] = []
    for index, beat in enumerate(ordered):
        onset, offset = sample(beat.r_onset), sample(beat.r_offset)
        if offset is not None and 0 <= onset < offset < len(energy_uv2):
            qrs_energy.append(energy_uv2[onset:offset + 1])
        if index == 0:
            continue
        previous_t = sample(ordered[index - 1].t_offset)
        next_p = sample(beat.p_onset)
        if previous_t is None or next_p is None:
            continue
        start = previous_t + int(round(_TP_START_AFTER_T_OFFSET_S * fs))
        stop = next_p - int(round(_TP_END_BEFORE_P_ONSET_S * fs))
        if stop - start >= int(round(_MINIMUM_TP_WINDOW_S * fs)) and 0 <= start and stop <= len(energy_uv2):
            noise_energy.append(energy_uv2[start:stop])
    if not qrs_energy or not noise_energy:
        result["context_hf_qrs_snr_beat_count"] = len(qrs_energy)
        return result
    signal_rms = float(np.sqrt(np.mean(np.concatenate(qrs_energy))))
    noise_rms = float(np.sqrt(np.mean(np.concatenate(noise_energy))))
    result["context_hf_qrs_signal_rms_uv"] = signal_rms
    result["context_hf_qrs_noise_rms_uv"] = noise_rms
    result["context_hf_qrs_snr_beat_count"] = len(qrs_energy)
    result["context_hf_qrs_noise_to_quantisation_ratio"] = (
        noise_rms / band_limited_quantisation_rms_uv(fs)
    )
    if noise_rms > 0 and signal_rms > 0:
        single = 20.0 * np.log10(signal_rms / noise_rms)
        result["context_hf_qrs_single_beat_snr_db"] = float(single)
        result["context_hf_qrs_averaged_snr_db"] = float(
            single + 10.0 * np.log10(len(qrs_energy))
        )
    return result
