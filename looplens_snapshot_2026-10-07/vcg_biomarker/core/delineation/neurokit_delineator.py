from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
try:
    import neurokit2 as nk
except ImportError:
    nk = None

from .base import BeatDelineation, DelineatorBase, MarkerPoint, WaveStatus


class NeuroKitDelineator(DelineatorBase):
    @property
    def name(self) -> str:
        return "neurokit"

    def delineate(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> list[BeatDelineation]:
        lead_ii = ecg_12lead[:, 1]
        beats = []

        if nk is None:
            return self._fallback_delineate(lead_ii, r_peaks, fs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, wave_info = nk.ecg_delineate(
                    ecg_cleaned=lead_ii,
                    rpeaks=r_peaks,
                    sampling_rate=fs,
                    method="dwt",
                )
            except Exception:
                return self._fallback_delineate(lead_ii, r_peaks, fs)

        n_beats = len(r_peaks)

        for i in range(n_beats):
            rp = r_peaks[i]
            beat = BeatDelineation(
                r_peak=MarkerPoint(sample=int(rp), confidence=0.8, source="neurokit")
            )

            if wave_info.get("ECG_P_Onsets") is not None and i < len(wave_info["ECG_P_Onsets"]):
                val = wave_info["ECG_P_Onsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.p_onset = MarkerPoint(sample=int(val), confidence=0.6, source="neurokit")

            if wave_info.get("ECG_P_Peaks") is not None and i < len(wave_info["ECG_P_Peaks"]):
                val = wave_info["ECG_P_Peaks"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.p_peak = MarkerPoint(sample=int(val), confidence=0.6, source="neurokit")

            if wave_info.get("ECG_P_Offsets") is not None and i < len(wave_info["ECG_P_Offsets"]):
                val = wave_info["ECG_P_Offsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.p_offset = MarkerPoint(sample=int(val), confidence=0.5, source="neurokit")

            if wave_info.get("ECG_Q_Peaks") is not None and i < len(wave_info["ECG_Q_Peaks"]):
                val = wave_info["ECG_Q_Peaks"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.q_peak = MarkerPoint(sample=int(val), confidence=0.7, source="neurokit")

            if wave_info.get("ECG_R_Onsets") is not None and i < len(wave_info["ECG_R_Onsets"]):
                val = wave_info["ECG_R_Onsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.r_onset = MarkerPoint(sample=int(val), confidence=0.7, source="neurokit")

            if wave_info.get("ECG_R_Offsets") is not None and i < len(wave_info["ECG_R_Offsets"]):
                val = wave_info["ECG_R_Offsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.r_offset = MarkerPoint(sample=int(val), confidence=0.7, source="neurokit")
                    beat.j_point = MarkerPoint(
                        sample=int(val),
                        confidence=0.65,
                        source="neurokit_qrs_offset",
                    )

            if wave_info.get("ECG_S_Peaks") is not None and i < len(wave_info["ECG_S_Peaks"]):
                val = wave_info["ECG_S_Peaks"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.s_peak = MarkerPoint(sample=int(val), confidence=0.7, source="neurokit")

            if wave_info.get("ECG_T_Onsets") is not None and i < len(wave_info["ECG_T_Onsets"]):
                val = wave_info["ECG_T_Onsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.t_onset = MarkerPoint(sample=int(val), confidence=0.6, source="neurokit")

            if wave_info.get("ECG_T_Peaks") is not None and i < len(wave_info["ECG_T_Peaks"]):
                val = wave_info["ECG_T_Peaks"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.t_peak = MarkerPoint(sample=int(val), confidence=0.7, source="neurokit")

            if wave_info.get("ECG_T_Offsets") is not None and i < len(wave_info["ECG_T_Offsets"]):
                val = wave_info["ECG_T_Offsets"][i]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    beat.t_offset = MarkerPoint(sample=int(val), confidence=0.6, source="neurokit")

            beat = self._validate_beat(beat, fs)
            beats.append(beat)

        return beats

    def _validate_beat(self, beat: BeatDelineation, fs: int) -> BeatDelineation:
        rp = beat.r_peak.sample

        if beat.p_onset is not None and beat.p_onset.sample >= rp:
            beat.p_onset = None
            beat.p_peak = None
            beat.p_offset = None
            beat.p_status = WaveStatus.NOT_ASSESSABLE

        if beat.p_offset is not None and beat.p_offset.sample >= rp:
            beat.p_onset = None
            beat.p_peak = None
            beat.p_offset = None
            beat.p_status = WaveStatus.NOT_ASSESSABLE

        if beat.t_onset is not None and beat.t_onset.sample <= rp:
            beat.t_onset = None
            beat.t_peak = None
            beat.t_offset = None
            beat.t_status = WaveStatus.NOT_ASSESSABLE

        if beat.t_offset is not None and beat.t_offset.sample <= rp:
            beat.t_onset = None
            beat.t_peak = None
            beat.t_offset = None
            beat.t_status = WaveStatus.NOT_ASSESSABLE

        if (
            beat.p_onset is not None
            and beat.p_peak is not None
            and beat.p_offset is not None
        ):
            beat.p_status = WaveStatus.PRESENT
        if beat.r_onset is not None and beat.r_offset is not None:
            beat.qrs_status = WaveStatus.PRESENT
        if (
            beat.t_onset is not None
            and beat.t_peak is not None
            and beat.t_offset is not None
        ):
            beat.t_status = WaveStatus.PRESENT

        qrs_dur = beat.qrs_duration_samples
        if (
            qrs_dur is not None
            and beat.qrs_definitely_exceeds_ms(fs, 120.0)
        ):
            beat.wide_qrs = True
            if beat.qrs_definitely_exceeds_ms(fs, 140.0):
                beat.bundle_branch_block = True

        return beat

    def _fallback_delineate(self, ecg: np.ndarray, r_peaks: np.ndarray, fs: int) -> list[BeatDelineation]:
        beats = []
        for rp in r_peaks:
            beat = BeatDelineation(
                r_peak=MarkerPoint(sample=int(rp), confidence=0.5, source="neurokit_fallback"),
                p_status=WaveStatus.NOT_ASSESSABLE,
                qrs_status=WaveStatus.NOT_ASSESSABLE,
                t_status=WaveStatus.NOT_ASSESSABLE,
            )
            beats.append(beat)
        return beats
