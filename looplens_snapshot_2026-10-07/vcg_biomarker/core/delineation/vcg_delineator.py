from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.signal import find_peaks

from .base import BeatDelineation, DelineatorBase, MarkerPoint, WaveStatus


class VCGDelineator(DelineatorBase):
    @property
    def name(self) -> str:
        return "vcg"

    def delineate(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> list[BeatDelineation]:
        if vcg_xyz is None:
            return [BeatDelineation(
                r_peak=MarkerPoint(sample=int(rp), confidence=0.3, source="vcg_unavailable"),
                p_status=WaveStatus.NOT_ASSESSABLE,
                qrs_status=WaveStatus.NOT_ASSESSABLE,
                t_status=WaveStatus.NOT_ASSESSABLE,
            ) for rp in r_peaks]

        vx, vy, vz = vcg_xyz[:, 0], vcg_xyz[:, 1], vcg_xyz[:, 2]

        sm = np.sqrt(vx**2 + vy**2 + vz**2)
        sv = np.zeros(len(sm) - 1)
        sv = np.sqrt(np.diff(vx)**2 + np.diff(vy)**2 + np.diff(vz)**2)

        origin = self._find_loop_origin(vcg_xyz, r_peaks, fs)

        beats = []
        for i, rp in enumerate(r_peaks):
            rp_int = int(rp)
            beat = BeatDelineation(
                r_peak=MarkerPoint(sample=rp_int, confidence=0.8, source="vcg")
            )

            qrs_start, qrs_end = self._segment_qrs_via_sv(sv, rp_int, fs)
            if qrs_start is not None:
                beat.r_onset = MarkerPoint(sample=qrs_start, confidence=0.75, source="vcg")
            if qrs_end is not None:
                beat.r_offset = MarkerPoint(sample=qrs_end, confidence=0.75, source="vcg")
            if qrs_start is not None and qrs_end is not None:
                beat.qrs_status = WaveStatus.PRESENT

            p_start, p_end = self._segment_p_via_sv_sm(sv, sm, rp_int, fs)
            if p_start is not None:
                beat.p_onset = MarkerPoint(sample=p_start, confidence=0.65, source="vcg")
            if p_end is not None:
                beat.p_offset = MarkerPoint(sample=p_end, confidence=0.6, source="vcg")
            if p_start is not None:
                p_peak = self._find_peak_in_range(sm, p_start, p_end if p_end else rp_int)
                if p_peak is not None:
                    beat.p_peak = MarkerPoint(sample=p_peak, confidence=0.6, source="vcg")

            t_start, t_end = self._segment_t_via_sv_sm(sv, sm, rp_int, fs)
            if t_start is not None:
                beat.t_onset = MarkerPoint(sample=t_start, confidence=0.65, source="vcg")
            if t_end is not None:
                beat.t_offset = MarkerPoint(sample=t_end, confidence=0.6, source="vcg")
            if t_start is not None:
                t_peak = self._find_peak_in_range(sm, t_start, t_end if t_end else min(len(sm), rp_int + int(0.5*fs)))
                if t_peak is not None:
                    beat.t_peak = MarkerPoint(sample=t_peak, confidence=0.7, source="vcg")

            beat.q_peak = self._find_q_in_vcg(vx, vy, vz, rp_int, qrs_start, fs)
            beat.s_peak = self._find_s_in_vcg(vx, vy, vz, rp_int, qrs_end, fs)

            if qrs_end is not None:
                beat.j_point = MarkerPoint(sample=qrs_end, confidence=0.6, source="vcg")

            beat.p_status = (
                WaveStatus.PRESENT
                if beat.p_onset is not None and beat.p_offset is not None
                else WaveStatus.INDETERMINATE
            )
            beat.t_status = (
                WaveStatus.PRESENT
                if beat.t_onset is not None and beat.t_offset is not None
                else WaveStatus.INDETERMINATE
            )

            qrs_dur = beat.qrs_duration_samples
            if (
                qrs_dur is not None
                and beat.qrs_definitely_exceeds_ms(fs, 120.0)
            ):
                beat.wide_qrs = True

            beats.append(beat)

        return beats

    def _find_loop_origin(self, vcg_xyz: np.ndarray, r_peaks: np.ndarray, fs: int) -> np.ndarray:
        if len(r_peaks) < 2:
            return np.zeros(3)

        search_start = int(r_peaks[0])
        search_end = min(len(vcg_xyz), int(r_peaks[-1]))

        if search_end <= search_start:
            return np.zeros(3)

        segment = vcg_xyz[search_start:search_end]
        min_dist = np.inf
        origin = np.zeros(3)

        step = max(1, len(segment) // 1000)
        for i in range(0, len(segment) - 1, step):
            p1 = segment[i]
            p2 = segment[i + step] if i + step < len(segment) else segment[i + 1]
            dist = np.linalg.norm(p2 - p1)
            if dist < min_dist and i > int(0.15 * fs) and i < len(segment) - int(0.15 * fs):
                min_dist = dist
                origin = p1

        return origin

    def _segment_qrs_via_sv(self, sv: np.ndarray, r_peak: int, fs: int) -> tuple[Optional[int], Optional[int]]:
        search_start = max(0, r_peak - int(round(0.22 * fs)))
        search_end = min(len(sv), r_peak + int(round(0.28 * fs)))
        if search_end - search_start < max(8, int(round(0.04 * fs))):
            return None, None

        activity = gaussian_filter1d(
            np.asarray(sv, dtype=float),
            sigma=max(0.75, 0.004 * fs),
            mode="nearest",
        )
        center_start = max(search_start, r_peak - int(round(0.06 * fs)))
        center_end = min(search_end, r_peak + int(round(0.08 * fs)))
        if center_end <= center_start:
            return None, None
        center = center_start + int(np.argmax(activity[center_start:center_end]))
        peak = float(activity[center])

        baseline_parts = []
        pre_end = min(search_end, r_peak - int(round(0.14 * fs)))
        post_start = max(search_start, r_peak + int(round(0.20 * fs)))
        if pre_end > search_start:
            baseline_parts.append(activity[search_start:pre_end])
        if search_end > post_start:
            baseline_parts.append(activity[post_start:search_end])
        baseline_values = (
            np.concatenate(baseline_parts)
            if baseline_parts
            else activity[search_start:search_end]
        )
        baseline = float(np.median(baseline_values))
        noise = max(
            1e-8,
            1.4826
            * float(np.median(np.abs(baseline_values - baseline))),
        )
        if peak <= baseline + 2.0 * noise:
            return None, None
        threshold = max(
            baseline + 2.5 * noise,
            baseline + 0.07 * (peak - baseline),
        )
        low_threshold = max(
            baseline + 1.5 * noise,
            baseline + 0.03 * (peak - baseline),
        )
        local = activity[search_start:search_end]
        active = binary_closing(
            local > threshold,
            structure=np.ones(max(1, int(round(0.018 * fs))), dtype=bool),
        )
        groups = self._group_contiguous(np.flatnonzero(active))
        center_local = center - search_start
        connected = [
            group
            for group in groups
            if group[0] <= center_local <= group[-1]
        ]
        if not connected:
            return None, None
        group = connected[0]
        onset = search_start + group[0]
        offset = search_start + group[-1]
        quiet = max(1, int(round(0.008 * fs)))
        while onset > search_start:
            lo = max(search_start, onset - quiet + 1)
            if np.all(activity[lo : onset + 1] <= low_threshold):
                break
            onset -= 1
        while offset < search_end - 1:
            hi = min(search_end, offset + quiet)
            if np.all(activity[offset:hi] <= low_threshold):
                break
            offset += 1
        duration_ms = (offset - onset) / fs * 1000.0
        if not onset < r_peak < offset or not 25.0 <= duration_ms <= 360.0:
            return None, None
        return int(onset), int(offset)

    def _segment_p_via_sv_sm(self, sv: np.ndarray, sm: np.ndarray, r_peak: int, fs: int) -> tuple[Optional[int], Optional[int]]:
        search_start = max(0, r_peak - int(0.4 * fs))
        search_end = max(0, r_peak - int(0.05 * fs))

        if search_end <= search_start:
            return None, None

        pre_sv = sv[search_start:min(search_end, len(sv))]
        pre_sm = sm[search_start:search_end]

        if len(pre_sv) == 0 or len(pre_sm) == 0:
            return None, None

        baseline_sv = np.median(pre_sv[:max(1, len(pre_sv)//4)])
        baseline_sm = np.median(pre_sm[:max(1, len(pre_sm)//4)])

        elevated = (pre_sv > baseline_sv * 1.5) | (pre_sm > baseline_sm * 1.3)
        elevated_indices = np.where(elevated)[0]

        if len(elevated_indices) < 3:
            return None, None

        groups = self._group_contiguous(elevated_indices)
        if not groups:
            return None, None

        best_group = max(groups, key=len)

        p_start = search_start + best_group[0]
        p_end = search_start + best_group[-1]

        return p_start, p_end

    def _segment_t_via_sv_sm(self, sv: np.ndarray, sm: np.ndarray, r_peak: int, fs: int) -> tuple[Optional[int], Optional[int]]:
        search_start = r_peak + int(0.08 * fs)
        search_end = min(len(sm), r_peak + int(0.5 * fs))

        if search_end <= search_start:
            return None, None

        search_end = min(search_end, len(sm))

        post_sm = sm[search_start:search_end]

        if len(post_sm) == 0:
            return None, None

        baseline = np.median(post_sm[-max(1, len(post_sm)//4):])
        elevated = post_sm > baseline * 1.2

        elevated_indices = np.where(elevated)[0]
        if len(elevated_indices) < 3:
            return None, None

        groups = self._group_contiguous(elevated_indices)
        if not groups:
            return None, None

        largest_group = max(groups, key=len)
        t_start = search_start + largest_group[0]
        t_end = search_start + largest_group[-1]

        return t_start, t_end

    def _find_peak_in_range(self, sm: np.ndarray, start: int, end: int) -> Optional[int]:
        if start >= end or end > len(sm):
            return None
        segment = sm[start:end]
        if len(segment) == 0:
            return None
        return start + np.argmax(segment)

    def _find_q_in_vcg(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, r_peak: int, qrs_start: Optional[int], fs: int) -> Optional[MarkerPoint]:
        search_start = qrs_start if qrs_start else max(0, r_peak - int(0.06 * fs))
        search_end = r_peak

        if search_end <= search_start:
            return None

        sm = np.sqrt(vx**2 + vy**2 + vz**2)
        segment = sm[search_start:search_end]
        if len(segment) == 0:
            return None

        min_idx = np.argmin(segment)
        if segment[min_idx] < 0.5 * sm[r_peak]:
            return MarkerPoint(sample=search_start + min_idx, confidence=0.6, source="vcg")
        return None

    def _find_s_in_vcg(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, r_peak: int, qrs_end: Optional[int], fs: int) -> Optional[MarkerPoint]:
        search_start = r_peak
        search_end = qrs_end if qrs_end else min(len(vx), r_peak + int(0.08 * fs))

        if search_end <= search_start:
            return None

        sm = np.sqrt(vx**2 + vy**2 + vz**2)
        segment = sm[search_start:search_end]
        if len(segment) == 0:
            return None

        min_idx = np.argmin(segment)
        if segment[min_idx] < 0.5 * sm[r_peak]:
            return MarkerPoint(sample=search_start + min_idx, confidence=0.6, source="vcg")
        return None

    def _group_contiguous(self, indices: np.ndarray) -> list[list[int]]:
        if len(indices) == 0:
            return []

        groups = []
        current_group = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= 3:
                current_group.append(indices[i])
            else:
                groups.append(current_group)
                current_group = [indices[i]]

        groups.append(current_group)
        return groups
