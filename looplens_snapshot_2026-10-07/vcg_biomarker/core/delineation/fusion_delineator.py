from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from .base import (
    BeatDelineation,
    DelineatorBase,
    MarkerPoint,
    RecordDelineation,
    RhythmAssessment,
    RhythmClass,
    WaveStatus,
)
from .multilead_delineator import MultileadDelineator
from .neurokit_delineator import NeuroKitDelineator
from .vcg_delineator import VCGDelineator


_MARKER_ATTRIBUTES = (
    "p_onset",
    "p_peak",
    "p_offset",
    "q_peak",
    "r_onset",
    "r_offset",
    "s_peak",
    "j_point",
    "t_onset",
    "t_peak",
    "t_offset",
    "u_onset",
    "u_peak",
    "u_offset",
)

_MARKER_TOLERANCE_MS = {
    "r_peak": 10.0,
    "p_onset": 30.0,
    "p_peak": 25.0,
    "p_offset": 30.0,
    "q_peak": 14.0,
    "r_onset": 16.0,
    "r_offset": 18.0,
    "s_peak": 14.0,
    "j_point": 18.0,
    "t_onset": 35.0,
    "t_peak": 35.0,
    "t_offset": 45.0,
    "u_onset": 60.0,
    "u_peak": 50.0,
    "u_offset": 80.0,
}

_BOUNDARY_SOURCE_FACTORS = {
    "multilead": 1.15,
    "neurokit": 1.00,
    "vcg": 1.05,
}


def _mad(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return 0.0
    median = float(np.median(data))
    return float(np.median(np.abs(data - median)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        return float(np.median(values))
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    return float(values[order[min(index, len(order) - 1)]])


class FusionDelineator(DelineatorBase):
    """Robust detector consensus with physiological abstention.

    Marker locations are never averaged across incompatible clusters.
    Agreement, source support, local detector confidence and marker-specific
    tolerances determine uncertainty. Rhythm context can invalidate a
    conventional discrete P wave without changing ventricular delineation.
    """

    # Sinus-like P coupling from candidate P waves (confidence >= 0.15):
    # coverage, timing MAD and 10-90 % spread of the P-peak-to-R offset.
    # Lowering the coverage to 0.45 or 0.35 was measured and not adopted
    # (audit_tools/p_coupling_variants.py): P on recorder-sinus ECGs rose
    # only from 70 % to 74 %, the newly admitted P agreed worse with the
    # recorder (PR error 14 vs 8.5 ms) and P reported on rhythms without P
    # rose from 7.0 % to 8.5 %; AFDB atrial fibrillation was unchanged.
    stable_p_candidate_coverage: float = 0.60
    stable_p_candidate_mad_ms: float = 20.0
    stable_p_candidate_p10_p90_ms: float = 35.0

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        conflict_threshold: float = 20.0,
        capture_candidates: bool = False,
    ):
        self._nk = NeuroKitDelineator()
        self.capture_candidates = capture_candidates
        self.candidate_snapshots = {}
        self._ml = MultileadDelineator()
        self._vcg = VCGDelineator()
        self._weights = weights or {
            "neurokit": 0.4,
            "multilead": 0.35,
            "vcg": 0.25,
        }
        # Kept for configuration compatibility; marker-specific thresholds
        # are never made tighter than the legacy global setting.
        self._conflict_threshold = max(1.0, float(conflict_threshold))

    @property
    def name(self) -> str:
        return "fusion"

    def delineate(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> list[BeatDelineation]:
        self.candidate_snapshots = {}
        source_beats = {
            "neurokit": self._nk.delineate(
                ecg_12lead,
                r_peaks,
                fs,
                vcg_xyz,
            ),
            "multilead": self._ml.delineate(
                ecg_12lead,
                r_peaks,
                fs,
                vcg_xyz,
            ),
            "vcg": self._vcg.delineate(
                ecg_12lead,
                r_peaks,
                fs,
                vcg_xyz,
            ),
        }
        fused = [
            self._fuse_beat(sources, fs)
            for sources in self._align_beats(source_beats, r_peaks, fs)
            if sources
        ]
        fused.sort(key=lambda beat: beat.r_peak.sample)
        self._assign_beat_coupling_context(fused, fs)
        self._mark_interbeat_overlap(fused, fs)
        return fused

    @staticmethod
    def _assign_beat_coupling_context(
        beats: list[BeatDelineation],
        fs: int,
        *,
        local_window_radius: int = 5,
    ) -> None:
        """Populate per-beat RR-to-previous and coupling-ratio context.

        ``coupling_ratio`` compares a beat's preceding RR interval against
        the median RR of surrounding beats (excluding the beat itself), so a
        short-coupled premature beat reads well below 1.0 regardless of the
        record's overall heart rate. A local rather than record-wide baseline
        keeps this meaningful under sinus arrhythmia, where the "normal" RR
        drifts across the recording.
        """
        if fs <= 0 or len(beats) < 2:
            return
        rr_ms = [
            (beats[index].r_peak.sample - beats[index - 1].r_peak.sample)
            / fs
            * 1000.0
            for index in range(1, len(beats))
        ]
        for index in range(1, len(beats)):
            rr_index = index - 1
            rr_previous = rr_ms[rr_index]
            beats[index].rr_interval_to_previous_ms = float(rr_previous)
            lo = max(0, rr_index - local_window_radius)
            hi = min(len(rr_ms), rr_index + local_window_radius + 1)
            neighbors = [rr_ms[j] for j in range(lo, hi) if j != rr_index]
            if len(neighbors) < 2:
                continue
            baseline = float(np.median(neighbors))
            if baseline > 0:
                beats[index].coupling_ratio = rr_previous / baseline

    def _align_beats(
        self,
        source_beats: dict[str, list[BeatDelineation]],
        reference_r_peaks: np.ndarray,
        fs: int,
    ) -> list[list[tuple[str, BeatDelineation, float]]]:
        if all(not beats for beats in source_beats.values()):
            return []
        references = np.asarray(reference_r_peaks, dtype=int)
        if references.size == 0:
            longest = max(source_beats.values(), key=len)
            references = np.asarray(
                [beat.r_peak.sample for beat in longest],
                dtype=int,
            )
        tolerance_samples = max(1, int(round(0.06 * fs)))
        used: dict[str, set[int]] = {name: set() for name in source_beats}
        aligned: list[list[tuple[str, BeatDelineation, float]]] = []
        for reference in sorted(np.unique(references).tolist()):
            sources: list[tuple[str, BeatDelineation, float]] = []
            for name, beats in source_beats.items():
                candidates = [
                    (abs(beat.r_peak.sample - reference), index, beat)
                    for index, beat in enumerate(beats)
                    if index not in used[name]
                ]
                if not candidates:
                    continue
                distance, index, beat = min(candidates, key=lambda item: item[0])
                if distance > tolerance_samples:
                    continue
                used[name].add(index)
                sources.append((name, beat, self._weights.get(name, 1.0)))
            if sources:
                aligned.append(sources)
        return aligned

    def delineate_record(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        ecg_id: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> RecordDelineation:
        beats = self.delineate(ecg_12lead, r_peaks, fs, vcg_xyz)
        morphology_diagnostics = self._mark_morphology_outliers(
            ecg_12lead,
            beats,
            fs,
        )
        self._detect_u_waves(ecg_12lead, beats, fs)
        rhythm = self._assess_rhythm(ecg_12lead, beats, fs)
        self._apply_rhythm_guards(beats, rhythm)
        self._mark_interbeat_overlap(beats, fs)

        record = RecordDelineation(
            beats=beats,
            fs=fs,
            n_samples=np.asarray(ecg_12lead).shape[0],
            ecg_id=ecg_id,
            source_method="physiology_constrained_fusion_v5",
            rhythm=rhythm,
            diagnostics=morphology_diagnostics,
        )
        quality, diagnostics = self._compute_quality_score(record)
        review_reasons = self._review_reasons(record, quality, diagnostics)
        record.quality_score = quality
        record.diagnostics = diagnostics
        record.review_reasons = review_reasons
        record.review_required = bool(review_reasons)
        return record

    @staticmethod
    def _detect_u_waves(
        ecg_12lead: np.ndarray,
        beats: list[BeatDelineation],
        fs: int,
    ) -> None:
        """Conservatively identify a distinct post-T U wave.

        U waves are optional and often below the reliable measurement floor.
        Therefore lack of a consensus remains indeterminate; it is never
        converted into physiological absence. A present result requires a
        time-coherent candidate in multiple leads and a separate isoelectric
        transition after T.
        """
        signal = np.asarray(ecg_12lead, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, None]
        if signal.ndim != 2 or fs <= 0 or len(signal) < 2:
            for beat in beats:
                beat.u_status = WaveStatus.NOT_ASSESSABLE
            return
        lead_count = signal.shape[1]
        preferred = (
            (1, 7, 8, 9, 10, 5)
            if lead_count >= 12
            else tuple(range(lead_count))
        )
        lead_indices = tuple(index for index in preferred if index < lead_count)
        lead_labels = (
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        )
        ordered = sorted(beats, key=lambda item: item.r_peak.sample)
        for beat_index, beat in enumerate(ordered):
            beat.u_onset = beat.u_peak = beat.u_offset = None
            beat.u_wave_absent = False
            if (
                beat.t_offset is None
                or beat.resolved_wave_status("t") != WaveStatus.PRESENT
            ):
                beat.u_status = WaveStatus.NOT_ASSESSABLE
                continue
            t_end = beat.t_offset.sample
            start = t_end + max(1, int(round(0.012 * fs)))
            end = min(len(signal) - 1, t_end + int(round(0.320 * fs)))
            if beat_index + 1 < len(ordered):
                following = ordered[beat_index + 1]
                next_qrs = (
                    following.r_onset.sample
                    if following.r_onset is not None
                    else following.r_peak.sample
                )
                end = min(end, next_qrs - int(round(0.060 * fs)))
                if following.p_onset is not None:
                    end = min(
                        end,
                        following.p_onset.sample - int(round(0.012 * fs)),
                    )
            if end - start < max(8, int(round(0.050 * fs))):
                beat.u_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("u_wave_window_too_short",)
                continue

            candidates: list[dict[str, float | int | str]] = []
            noise_estimates: list[float] = []
            for lead_index in lead_indices:
                values = signal[start:end, lead_index]
                if len(values) < 8 or not np.all(np.isfinite(values)):
                    continue
                smooth_window = min(
                    len(values) - (1 - len(values) % 2),
                    max(5, int(round(0.035 * fs)) | 1),
                )
                if smooth_window < 5:
                    continue
                smooth = sp_signal.savgol_filter(
                    values,
                    smooth_window,
                    min(3, smooth_window - 2),
                    mode="interp",
                )
                baseline = np.linspace(smooth[0], smooth[-1], len(smooth))
                centered = smooth - baseline
                residual = values - smooth
                noise = max(
                    0.001,
                    1.4826 * _mad(residual),
                )
                noise_estimates.append(float(noise))
                minimum_prominence = max(0.010, 3.5 * noise)
                peaks, properties = sp_signal.find_peaks(
                    np.abs(centered),
                    prominence=minimum_prominence,
                    width=(
                        max(1, int(round(0.018 * fs))),
                        max(2, int(round(0.180 * fs))),
                    ),
                    distance=max(1, int(round(0.030 * fs))),
                )
                if len(peaks) == 0:
                    continue
                t_start = (
                    beat.t_onset.sample
                    if beat.t_onset is not None
                    else max(0, t_end - int(round(0.300 * fs)))
                )
                t_amplitude = float(
                    np.nanmax(np.abs(signal[t_start : t_end + 1, lead_index]))
                )
                ranked = sorted(
                    range(len(peaks)),
                    key=lambda index: float(properties["prominences"][index]),
                    reverse=True,
                )
                for candidate_index in ranked:
                    peak_local = int(peaks[candidate_index])
                    amplitude = float(abs(centered[peak_local]))
                    if t_amplitude > 0 and amplitude > 0.80 * t_amplitude:
                        continue
                    left = int(np.floor(properties["left_ips"][candidate_index]))
                    right = int(np.ceil(properties["right_ips"][candidate_index]))
                    if not 0 <= left < peak_local < right < len(values):
                        continue
                    candidates.append(
                        {
                            "peak": start + peak_local,
                            "onset": start + left,
                            "offset": start + right,
                            "prominence": float(
                                properties["prominences"][candidate_index]
                            ),
                            "snr": amplitude / noise,
                            "lead": (
                                lead_labels[lead_index]
                                if lead_index < len(lead_labels)
                                else f"lead_{lead_index + 1}"
                            ),
                        }
                    )
                    break

            if not candidates:
                beat.u_status = (
                    WaveStatus.NOT_ASSESSABLE
                    if noise_estimates
                    and float(np.median(noise_estimates)) >= 0.05
                    else WaveStatus.INDETERMINATE
                )
                continue
            tolerance = max(1, int(round(0.035 * fs)))
            clusters: list[list[dict[str, float | int | str]]] = []
            for candidate in sorted(candidates, key=lambda item: int(item["peak"])):
                if (
                    not clusters
                    or int(candidate["peak"])
                    - int(np.median([int(item["peak"]) for item in clusters[-1]]))
                    > tolerance
                ):
                    clusters.append([candidate])
                else:
                    clusters[-1].append(candidate)
            supported = [
                cluster
                for cluster in clusters
                if len({str(item["lead"]) for item in cluster}) >= 2
            ]
            if not supported:
                beat.u_status = WaveStatus.INDETERMINATE
                continue
            cluster = max(
                supported,
                key=lambda items: (
                    len({str(item["lead"]) for item in items}),
                    float(np.median([float(item["snr"]) for item in items])),
                ),
            )
            peaks = np.asarray([int(item["peak"]) for item in cluster], dtype=float)
            onsets = np.asarray([int(item["onset"]) for item in cluster], dtype=float)
            offsets = np.asarray([int(item["offset"]) for item in cluster], dtype=float)
            peak_sample = int(round(float(np.median(peaks))))
            onset_sample = int(round(float(np.median(onsets))))
            offset_sample = int(round(float(np.median(offsets))))
            separation_ms = (onset_sample - t_end) / fs * 1000.0
            if separation_ms < 8.0:
                beat.u_status = WaveStatus.FUSED
                beat.t_status = WaveStatus.FUSED
                beat.review_reasons += ("possible_t_u_fusion",)
                continue
            if not t_end <= onset_sample < peak_sample < offset_sample <= end:
                beat.u_status = WaveStatus.NOT_ASSESSABLE
                continue
            dispersion_ms = 1.4826 * _mad(peaks) / fs * 1000.0
            boundary_spread_ms = max(
                1.4826 * _mad(onsets) / fs * 1000.0,
                1.4826 * _mad(offsets) / fs * 1000.0,
            )
            support = len({str(item["lead"]) for item in cluster})
            median_snr = float(np.median([float(item["snr"]) for item in cluster]))
            confidence = float(
                np.clip(
                    (0.35 + 0.10 * min(support, 4))
                    * min(1.0, median_snr / 6.0)
                    * np.exp(-dispersion_ms / 40.0),
                    0.0,
                    0.90,
                )
            )
            if confidence < 0.35:
                beat.u_status = WaveStatus.INDETERMINATE
                continue
            leads = tuple(dict.fromkeys(str(item["lead"]) for item in cluster))
            uncertainty = max(
                1000.0 / fs,
                dispersion_ms,
                boundary_spread_ms,
            )
            common = {
                "confidence": confidence,
                "source": "multilead_u_consensus",
                "uncertainty_ms": uncertainty,
                "support_count": support,
                "lead_names": leads,
                "evidence_sources": ("post_t_multilead_morphology",),
            }
            beat.u_onset = MarkerPoint(sample=onset_sample, **common)
            beat.u_peak = MarkerPoint(sample=peak_sample, **common)
            beat.u_offset = MarkerPoint(sample=offset_sample, **common)
            beat.u_status = WaveStatus.PRESENT

    def refresh_record_quality(
        self,
        record: RecordDelineation,
    ) -> RecordDelineation:
        """Recompute consensus/QC after a validated manual revision."""

        record.invalidate_beat_cache()
        quality, diagnostics = self._compute_quality_score(record)
        review_reasons = self._review_reasons(record, quality, diagnostics)
        record.quality_score = quality
        record.diagnostics = diagnostics
        record.review_reasons = review_reasons
        record.review_required = bool(review_reasons)
        return record

    def _fuse_beat(
        self,
        sources: list[tuple[str, BeatDelineation, float]],
        fs: int,
    ) -> BeatDelineation:
        if not sources:
            raise ValueError("No beat sources provided")
        r_marker = self._fuse_marker(
            [(name, beat.r_peak, weight) for name, beat, weight in sources],
            fs,
            marker_name="r_peak",
        )
        if r_marker is None:
            best = max(
                sources,
                key=lambda item: item[1].r_peak.confidence * item[2],
            )
            r_marker = best[1].r_peak
        fused = BeatDelineation(r_peak=r_marker)
        if self.capture_candidates and len(self.candidate_snapshots) < 2000:
            self.candidate_snapshots[r_marker.sample] = [
                {'method': method, 'landmark': attribute, 'sample': int(point.sample),
                 'time_seconds': point.sample/fs, 'confidence': float(point.confidence),
                 'source': point.source, 'weight': float(weight),
                 'temporally_admissible': bool(attribute == 'r_peak' or self._marker_is_temporally_possible(attribute, point.sample, r_marker.sample, fs))}
                for method, beat, weight in sources
                for attribute in ('r_peak',) + _MARKER_ATTRIBUTES
                for point in (getattr(beat, attribute),) if point is not None]

        for attribute in _MARKER_ATTRIBUTES:
            observations = [
                (source_name, marker, weight)
                for source_name, beat, weight in sources
                for marker in (getattr(beat, attribute),)
                if marker is not None
                and self._marker_is_temporally_possible(
                    attribute,
                    marker.sample,
                    r_marker.sample,
                    fs,
                )
            ]
            marker = self._fuse_marker(
                observations,
                fs,
                marker_name=attribute,
            )
            if marker is not None:
                setattr(fused, attribute, marker)

        self._restore_coherent_multilead_triplet(
            fused,
            sources,
            wave="p",
            fs=fs,
        )
        self._restore_coherent_multilead_triplet(
            fused,
            sources,
            wave="t",
            fs=fs,
        )

        explicit_p_absence = [
            beat
            for _, beat, _ in sources
            if beat.p_wave_absent
            and not beat.r_peak.source.endswith("_unavailable")
            and "fallback" not in beat.r_peak.source
        ]
        explicit_t_absence = [
            beat
            for _, beat, _ in sources
            if beat.t_wave_absent
            and not beat.r_peak.source.endswith("_unavailable")
            and "fallback" not in beat.r_peak.source
        ]
        if (
            fused.p_onset is None
            and fused.p_peak is None
            and len(explicit_p_absence) >= 2
        ):
            fused.p_wave_absent = True
            fused.p_status = WaveStatus.ABSENT
        elif any(
            marker is not None
            for marker in (fused.p_onset, fused.p_peak, fused.p_offset)
        ):
            fused.p_status = WaveStatus.PRESENT
        else:
            fused.p_status = WaveStatus.INDETERMINATE

        if (
            fused.t_onset is None
            and fused.t_peak is None
            and len(explicit_t_absence) >= 2
        ):
            fused.t_wave_absent = True
            fused.t_status = WaveStatus.ABSENT
        elif any(
            marker is not None
            for marker in (fused.t_onset, fused.t_peak, fused.t_offset)
        ):
            fused.t_status = WaveStatus.PRESENT
        else:
            fused.t_status = WaveStatus.INDETERMINATE

        fused.atrial_activity_count = max(
            (beat.atrial_activity_count for _, beat, _ in sources),
            default=0,
        )
        fused.p_t_fusion_risk = any(
            beat.p_t_fusion_risk for _, beat, _ in sources
        )
        fused.wide_qrs = (
            sum(beat.wide_qrs for _, beat, _ in sources) >= 2
        )
        fused.bundle_branch_block = (
            sum(beat.bundle_branch_block for _, beat, _ in sources) >= 2
        )
        fused.boundary_conflict = any(
            marker is not None
            and marker.conflict_ms
            > _MARKER_TOLERANCE_MS.get(attribute, self._conflict_threshold)
            for attribute in _MARKER_ATTRIBUTES
            for marker in (getattr(fused, attribute),)
        )
        self._enforce_physiological_order(fused, fs)

        # H1: retrograde_p_* is not in _MARKER_ATTRIBUTES (it is a
        # single-source hypothesis field, not a per-wave marker every
        # delineator contributes to), so it is not touched by the marker
        # fusion loop above and must be carried through explicitly, or it
        # would be silently discarded here despite being computed
        # correctly by MultileadDelineator. Preferring the highest-weight
        # source that found one keeps this consistent with how every
        # other field in this method resolves multi-source disagreement.
        retrograde_sources = [
            (weight, beat)
            for _, beat, weight in sources
            if beat.retrograde_p_status == WaveStatus.PRESENT
        ]
        if retrograde_sources:
            _, best_retrograde_beat = max(
                retrograde_sources, key=lambda item: item[0]
            )
            fused.retrograde_p_onset = best_retrograde_beat.retrograde_p_onset
            fused.retrograde_p_peak = best_retrograde_beat.retrograde_p_peak
            fused.retrograde_p_offset = best_retrograde_beat.retrograde_p_offset
            fused.retrograde_p_status = WaveStatus.PRESENT
            fused.review_reasons += ("retrograde_p_candidate_found",)
        # H3: the disambiguation tag itself (unlike the P marker it
        # resolves, which already flows through via p_onset/p_peak/
        # p_offset in _MARKER_ATTRIBUTES above) lives only in the source
        # beat's own review_reasons and needs the same explicit carry-through.
        if any(
            "vcg_direction_disambiguated_from_multiple_atrial_deflections"
            in beat.review_reasons
            for _, beat, _ in sources
        ):
            fused.review_reasons += (
                "vcg_direction_disambiguated_from_multiple_atrial_deflections",
            )

        return fused

    @classmethod
    def _restore_coherent_multilead_triplet(
        cls,
        fused: BeatDelineation,
        sources: list[tuple[str, BeatDelineation, float]],
        *,
        wave: str,
        fs: int,
    ) -> None:
        """Repair only incomplete/incoherent fusion from a supported triplet."""

        attributes = (
            ("p_onset", "p_peak", "p_offset")
            if wave == "p"
            else ("t_onset", "t_peak", "t_offset")
        )
        if cls._wave_triplet_is_coherent(fused, wave):
            return
        candidates: list[BeatDelineation] = []
        for source_name, source_beat, _ in sources:
            if source_name != "multilead":
                continue
            markers = [
                getattr(source_beat, attribute)
                for attribute in attributes
            ]
            if (
                all(marker is not None for marker in markers)
                and all(marker.confidence >= 0.30 for marker in markers)
                and all(marker.support_count >= 2 for marker in markers)
                and cls._wave_triplet_is_coherent(source_beat, wave)
            ):
                candidates.append(source_beat)
        if not candidates:
            return
        source_beat = max(
            candidates,
            key=lambda beat: (
                min(
                    getattr(beat, attribute).confidence
                    for attribute in attributes
                ),
                min(
                    getattr(beat, attribute).support_count
                    for attribute in attributes
                ),
            ),
        )
        existing_conflict = max(
            (
                marker.conflict_ms
                for attribute in attributes
                for marker in (getattr(fused, attribute),)
                if marker is not None
            ),
            default=0.0,
        )
        for attribute in attributes:
            point = getattr(source_beat, attribute)
            setattr(
                fused,
                attribute,
                MarkerPoint(
                    sample=point.sample,
                    confidence=min(
                        0.34,
                        0.80 * point.confidence,
                    ),
                    source=f"coherent_multilead_{wave}_triplet",
                    uncertainty_ms=max(
                        point.uncertainty_ms,
                        2.0 * 1000.0 / fs,
                        0.5
                        * _MARKER_TOLERANCE_MS.get(
                            attribute,
                            25.0,
                        ),
                    ),
                    support_count=point.support_count,
                    lead_names=point.lead_names,
                    evidence_sources=tuple(
                        dict.fromkeys(
                            point.evidence_sources
                            + ("coherent_wave_triplet_fallback",)
                        )
                    ),
                    conflict_ms=max(
                        point.conflict_ms,
                        existing_conflict,
                    ),
                ),
            )
        fused.review_reasons += (
            f"{wave}_triplet_restored_from_multilead_consensus",
        )

    @staticmethod
    def _wave_triplet_is_coherent(
        beat: BeatDelineation,
        wave: str,
    ) -> bool:
        if wave == "p":
            markers = (beat.p_onset, beat.p_peak, beat.p_offset)
            qrs_start = (
                beat.r_onset.sample
                if beat.r_onset is not None
                else beat.r_peak.sample
            )
            return bool(
                all(marker is not None for marker in markers)
                and beat.p_onset.sample
                < beat.p_peak.sample
                < beat.p_offset.sample
                <= qrs_start
            )
        if wave == "t":
            markers = (beat.t_onset, beat.t_peak, beat.t_offset)
            qrs_end = (
                beat.r_offset.sample
                if beat.r_offset is not None
                else beat.r_peak.sample
            )
            return bool(
                all(marker is not None for marker in markers)
                and qrs_end
                <= beat.t_onset.sample
                < beat.t_peak.sample
                < beat.t_offset.sample
            )
        raise ValueError(f"Unsupported wave triplet: {wave}")

    def _fuse_marker(
        self,
        markers: list[tuple[str, MarkerPoint, float]],
        fs: int,
        marker_name: Optional[str] = None,
    ) -> Optional[MarkerPoint]:
        if not markers:
            return None
        marker_name = marker_name or "marker"
        tolerance_ms = max(
            self._conflict_threshold
            if marker_name not in _MARKER_TOLERANCE_MS
            else 0.0,
            _MARKER_TOLERANCE_MS.get(
                marker_name,
                self._conflict_threshold,
            ),
        )
        tolerance_samples = max(1, int(round(tolerance_ms / 1000.0 * fs)))
        observations = sorted(markers, key=lambda item: item[1].sample)
        clusters: list[list[tuple[str, MarkerPoint, float]]] = []
        for observation in observations:
            if (
                not clusters
                or observation[1].sample
                - int(np.median([item[1].sample for item in clusters[-1]]))
                > tolerance_samples
            ):
                clusters.append([observation])
            else:
                clusters[-1].append(observation)

        def effective_weight(
            observation: tuple[str, MarkerPoint, float],
        ) -> float:
            source_name, point, configured_weight = observation
            source_factor = (
                _BOUNDARY_SOURCE_FACTORS.get(source_name, 1.0)
                if marker_name.endswith(("onset", "offset"))
                or marker_name in {"r_onset", "r_offset", "j_point"}
                else 1.0
            )
            return (
                max(0.01, configured_weight)
                * max(0.05, point.confidence)
                * source_factor
            )

        global_qrs_boundary = marker_name in {
            "r_onset",
            "r_offset",
            "j_point",
        }
        global_t_boundary = marker_name in {
            "t_onset",
            "t_offset",
        }
        preferred_multilead = [
            observation
            for observation in observations
            if (
                observation[0] == "multilead"
                and (
                    (
                        global_qrs_boundary
                        and observation[1].confidence >= 0.35
                        and observation[1].support_count >= 2
                    )
                    or (
                        global_t_boundary
                        and observation[1].confidence >= 0.65
                        and observation[1].support_count >= 3
                        and any(
                            evidence
                            in {
                                "earliest_supported_lead_boundary",
                                "latest_supported_lead_boundary",
                            }
                            for evidence
                            in observation[1].evidence_sources
                        )
                    )
                )
            )
        ]
        if preferred_multilead:
            preferred_point = max(
                preferred_multilead,
                key=lambda item: (
                    item[1].support_count,
                    item[1].confidence,
                ),
            )[1]
            winning = next(
                cluster
                for cluster in clusters
                if any(point is preferred_point for _, point, _ in cluster)
            )
        else:
            winning = max(
                clusters,
                key=lambda cluster: sum(
                    effective_weight(item) for item in cluster
                ),
            )
        all_samples = np.asarray(
            [item[1].sample for item in observations],
            dtype=float,
        )
        total_conflict_ms = (
            float(np.ptp(all_samples)) / fs * 1000.0
            if len(all_samples) > 1
            else 0.0
        )
        if len(winning) == 1 and len(observations) > 1:
            source_name, point, _ = winning[0]
            high_trust_boundary = (
                source_name == "multilead"
                and marker_name
                in {
                    "p_onset",
                    "p_offset",
                    "r_onset",
                    "r_offset",
                    "j_point",
                    "t_onset",
                    "t_offset",
                }
                and (
                    (
                        global_qrs_boundary
                        and point.confidence >= 0.35
                        and point.support_count >= 2
                    )
                    or (
                        not global_qrs_boundary
                        and point.confidence >= 0.65
                        and point.support_count >= 3
                    )
                )
            )
            if not high_trust_boundary:
                return None

        samples = np.asarray(
            [item[1].sample for item in winning],
            dtype=float,
        )
        weights = np.asarray(
            [effective_weight(item) for item in winning],
            dtype=float,
        )
        global_boundary_names = {
            "r_onset",
            "r_offset",
            "j_point",
            "t_onset",
            "t_offset",
        }
        preferred_global_boundaries = [
            point
            for source_name, point, _ in winning
            if (
                source_name == "multilead"
                and marker_name in global_boundary_names
                and (
                    (
                        global_qrs_boundary
                        and point.confidence >= 0.35
                        and point.support_count >= 2
                    )
                    or (
                        global_t_boundary
                        and point.confidence >= 0.65
                        and point.support_count >= 3
                        and any(
                            evidence
                            in {
                                "earliest_supported_lead_boundary",
                                "latest_supported_lead_boundary",
                            }
                            for evidence in point.evidence_sources
                        )
                    )
                )
            )
        ]
        fused_sample = (
            int(
                max(
                    preferred_global_boundaries,
                    key=lambda point: (
                        point.support_count,
                        point.confidence,
                    ),
                ).sample
            )
            if preferred_global_boundaries
            else int(round(_weighted_median(samples, weights)))
        )
        dispersion_ms = (
            1.4826 * _mad(samples) / fs * 1000.0
            if len(samples) > 1
            else 0.0
        )
        point_confidences = np.asarray(
            [item[1].confidence for item in winning],
            dtype=float,
        )
        base_confidence = float(
            np.average(point_confidences, weights=weights)
        )
        internal_support = max(
            point.support_count for _, point, _ in winning
        )
        support_factor = max(
            min(1.0, 0.68 + 0.16 * (len(winning) - 1)),
            min(1.0, 0.60 + 0.08 * internal_support),
        )
        agreement = float(
            np.exp(-dispersion_ms / max(tolerance_ms, 1.0))
        )
        winning_weight = float(np.sum(weights))
        total_weight = float(
            sum(effective_weight(item) for item in observations)
        )
        consensus_weight_fraction = winning_weight / max(total_weight, 1e-12)
        conflict_penalty = (
            1.0
            if total_conflict_ms <= tolerance_ms
            else max(
                0.72 if internal_support >= 3 else 0.62,
                consensus_weight_fraction,
            )
        )
        confidence = float(
            np.clip(
                base_confidence
                * support_factor
                * agreement
                * conflict_penalty,
                0.0,
                1.0,
            )
        )
        if preferred_global_boundaries and global_qrs_boundary:
            multilead_confidence = max(
                point.confidence
                for point in preferred_global_boundaries
            )
            confidence = max(
                confidence,
                0.88 * multilead_confidence,
            )
        sources = tuple(dict.fromkeys(item[0] for item in winning))
        evidence_sources = tuple(
            dict.fromkeys(
                source
                for _, point, _ in winning
                for source in (
                    point.evidence_sources
                    if point.evidence_sources
                    else (point.source,)
                )
            )
        )
        leads = tuple(
            dict.fromkeys(
                lead
                for _, point, _ in winning
                for lead in point.lead_names
            )
        )
        source = (
            f"fusion_single:{sources[0]}"
            if len(sources) == 1
            else "fusion_consensus:" + "+".join(sources)
        )
        uncertainty = max(
            1000.0 / fs,
            dispersion_ms,
            float(
                np.average(
                    [point.uncertainty_ms for _, point, _ in winning],
                    weights=weights,
                )
            ),
        )
        return MarkerPoint(
            sample=fused_sample,
            confidence=confidence,
            source=source,
            uncertainty_ms=uncertainty,
            support_count=sum(
                point.support_count for _, point, _ in winning
            ),
            lead_names=leads,
            evidence_sources=evidence_sources,
            conflict_ms=total_conflict_ms,
        )

    @staticmethod
    def _marker_is_temporally_possible(
        attribute: str,
        sample: int,
        r_peak: int,
        fs: int,
    ) -> bool:
        relative_ms = (sample - r_peak) / fs * 1000.0
        ranges = {
            "p_onset": (-550.0, -20.0),
            "p_peak": (-480.0, -10.0),
            "p_offset": (-420.0, -2.0),
            "q_peak": (-220.0, 5.0),
            "r_onset": (-240.0, -2.0),
            "r_offset": (2.0, 360.0),
            "s_peak": (-5.0, 300.0),
            "j_point": (2.0, 380.0),
            "t_onset": (20.0, 720.0),
            "t_peak": (30.0, 800.0),
            "t_offset": (50.0, 1000.0),
            "u_onset": (80.0, 1150.0),
            "u_peak": (100.0, 1200.0),
            "u_offset": (120.0, 1300.0),
        }
        lower, upper = ranges.get(attribute, (-np.inf, np.inf))
        return lower <= relative_ms <= upper

    @staticmethod
    def _enforce_physiological_order(
        beat: BeatDelineation,
        fs: int,
    ) -> None:
        if not (
            beat.r_onset is not None
            and beat.r_offset is not None
            and beat.r_onset.sample < beat.r_peak.sample < beat.r_offset.sample
        ):
            beat.r_onset = None
            beat.r_offset = None
            beat.j_point = None
            beat.qrs_status = WaveStatus.NOT_ASSESSABLE
            beat.review_reasons += ("incompatible_qrs_boundaries",)
        else:
            qrs_ms = beat.qrs_duration_ms(fs)
            if qrs_ms is None or not 25.0 <= qrs_ms <= 360.0:
                beat.r_onset = None
                beat.r_offset = None
                beat.j_point = None
                beat.qrs_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("implausible_qrs_duration",)
            else:
                beat.qrs_status = WaveStatus.PRESENT
                beat.wide_qrs = beat.qrs_definitely_exceeds_ms(
                    fs,
                    120.0,
                )
                beat.bundle_branch_block = beat.qrs_definitely_exceeds_ms(
                    fs,
                    140.0,
                )
                if (
                    beat.j_point is None
                    or abs(beat.j_point.sample - beat.r_offset.sample)
                    > int(round(0.04 * fs))
                ):
                    beat.j_point = MarkerPoint(
                        sample=beat.r_offset.sample,
                        confidence=beat.r_offset.confidence,
                        source="qrs_offset_as_j_point",
                        uncertainty_ms=beat.r_offset.uncertainty_ms,
                        support_count=beat.r_offset.support_count,
                        lead_names=beat.r_offset.lead_names,
                        evidence_sources=beat.r_offset.evidence_sources,
                        conflict_ms=beat.r_offset.conflict_ms,
                    )

        if beat.q_peak is not None:
            if not (
                (
                    beat.r_onset.sample
                    if beat.r_onset is not None
                    else beat.q_peak.sample
                )
                <= beat.q_peak.sample
                <= beat.r_peak.sample
            ):
                beat.q_peak = None
        if beat.s_peak is not None:
            if not (
                beat.r_peak.sample
                <= beat.s_peak.sample
                <= (
                    beat.r_offset.sample
                    if beat.r_offset is not None
                    else beat.s_peak.sample
                )
            ):
                beat.s_peak = None

        p_markers = (beat.p_onset, beat.p_peak, beat.p_offset)
        if any(marker is not None for marker in p_markers):
            if not (
                all(marker is not None for marker in p_markers)
                and beat.p_onset.sample
                < beat.p_peak.sample
                < beat.p_offset.sample
                <= (
                    beat.r_onset.sample
                    if beat.r_onset is not None
                    else beat.r_peak.sample
                )
            ):
                beat.p_onset = beat.p_peak = beat.p_offset = None
                beat.p_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("incompatible_p_boundaries",)
            else:
                beat.p_status = WaveStatus.PRESENT

        t_markers = (beat.t_onset, beat.t_peak, beat.t_offset)
        if any(marker is not None for marker in t_markers):
            qrs_end = (
                beat.r_offset.sample
                if beat.r_offset is not None
                else beat.r_peak.sample
            )
            if not (
                all(marker is not None for marker in t_markers)
                and qrs_end
                <= beat.t_onset.sample
                < beat.t_peak.sample
                < beat.t_offset.sample
            ):
                beat.t_onset = beat.t_peak = beat.t_offset = None
                beat.t_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("incompatible_t_boundaries",)
            else:
                beat.t_status = WaveStatus.PRESENT

        u_markers = (beat.u_onset, beat.u_peak, beat.u_offset)
        if any(marker is not None for marker in u_markers):
            t_end = beat.t_offset.sample if beat.t_offset is not None else None
            if not (
                t_end is not None
                and all(marker is not None for marker in u_markers)
                and t_end
                <= beat.u_onset.sample
                < beat.u_peak.sample
                < beat.u_offset.sample
            ):
                beat.u_onset = beat.u_peak = beat.u_offset = None
                beat.u_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("incompatible_u_boundaries",)
            else:
                beat.u_status = WaveStatus.PRESENT

    def _assess_rhythm(
        self,
        ecg: np.ndarray,
        beats: list[BeatDelineation],
        fs: int,
    ) -> RhythmAssessment:
        if len(beats) < 2:
            return RhythmAssessment(
                flags=("too_few_beats_for_rhythm_context",),
            )
        r_samples = np.asarray(
            [beat.r_peak.sample for beat in beats],
            dtype=float,
        )
        rr = np.diff(r_samples) / fs
        rr = rr[np.isfinite(rr) & (rr > 0)]
        if rr.size == 0:
            return RhythmAssessment(flags=("invalid_rr_intervals",))
        rr_median = float(np.median(rr))
        rr_cv = float(1.4826 * _mad(rr) / max(rr_median, 1e-6))
        successive = (
            float(np.median(np.abs(np.diff(rr))) / max(rr_median, 1e-6))
            if rr.size >= 2
            else 0.0
        )
        rr_lag1_correlation = None
        if (
            rr.size >= 4
            and float(np.std(rr[:-1])) > 1e-9
            and float(np.std(rr[1:])) > 1e-9
        ):
            rr_lag1_correlation = float(
                np.corrcoef(rr[:-1], rr[1:])[0, 1]
            )
        rr_turning_point_fraction = (
            float(
                np.mean(
                    (rr[1:-1] - rr[:-2])
                    * (rr[2:] - rr[1:-1])
                    < 0
                )
            )
            if rr.size >= 4
            else None
        )
        p_beats = [
            beat
            for beat in beats
            if beat.p_onset is not None
            and beat.p_peak is not None
            and beat.p_offset is not None
            and beat.marker_confidence("p_onset", "p_offset") >= 0.35
            and not self._uses_reconstructed_p_triplet(beat)
        ]
        p_coverage = len(p_beats) / len(beats)
        p_offsets = np.asarray(
            [
                (beat.p_peak.sample - beat.r_peak.sample) / fs * 1000.0
                for beat in p_beats
            ],
            dtype=float,
        )
        p_timing_mad = (
            float(1.4826 * _mad(p_offsets))
            if p_offsets.size >= 2
            else None
        )
        p_candidates = [
            beat
            for beat in beats
            if beat.p_onset is not None
            and beat.p_peak is not None
            and beat.p_offset is not None
            and beat.resolved_wave_status("p") == WaveStatus.PRESENT
            and beat.marker_confidence("p_onset", "p_offset") >= 0.15
            and not self._uses_reconstructed_p_triplet(beat)
        ]
        p_candidate_coverage = len(p_candidates) / len(beats)
        p_candidate_offsets = np.asarray(
            [
                (beat.p_peak.sample - beat.r_peak.sample) / fs * 1000.0
                for beat in p_candidates
            ],
            dtype=float,
        )
        p_candidate_timing_mad = (
            float(1.4826 * _mad(p_candidate_offsets))
            if p_candidate_offsets.size >= 3
            else None
        )
        p_candidate_timing_p10_p90 = (
            float(
                np.percentile(p_candidate_offsets, 90)
                - np.percentile(p_candidate_offsets, 10)
            )
            if p_candidate_offsets.size >= 3
            else None
        )
        (
            atrial_frequency,
            atrial_concentration,
            atrial_bandpower_ratio,
            independent_atrial_rate_hz,
            independent_atrial_rate_concentration,
        ) = self._atrial_spectrum(
            np.asarray(ecg, dtype=float),
            beats,
            fs,
        )
        multiple_fraction = float(
            np.mean([beat.atrial_activity_count > 1 for beat in beats])
        )
        irregular = rr_cv >= 0.12 or successive >= 0.10
        af_like_irregularity = rr_cv >= 0.12 and successive >= 0.10
        alternating_rr_pattern = (
            rr_lag1_correlation is not None
            and rr_lag1_correlation <= -0.55
            and rr_turning_point_fraction is not None
            and rr_turning_point_fraction >= 0.80
        )
        inconsistent_p = (
            p_coverage < 0.45
            or (
                p_timing_mad is not None
                and p_timing_mad > 38.0
            )
        )
        stable_sinus_like_p_timing = (
            (
                p_coverage >= 0.25
                and p_timing_mad is not None
                and p_timing_mad <= 20.0
            )
            or (
                p_candidate_coverage >= self.stable_p_candidate_coverage
                and p_candidate_timing_mad is not None
                and p_candidate_timing_mad <= self.stable_p_candidate_mad_ms
                and p_candidate_timing_p10_p90 is not None
                and p_candidate_timing_p10_p90
                <= self.stable_p_candidate_p10_p90_ms
            )
            or (
                p_candidate_coverage >= 0.35
                and p_candidate_timing_mad is not None
                and p_candidate_timing_mad <= 5.0
                and p_candidate_timing_p10_p90 is not None
                and p_candidate_timing_p10_p90 <= 12.0
            )
        )
        majority_wide_qrs = (
            sum(beat.wide_qrs for beat in beats) >= len(beats) / 2
        )
        organized_atrial_spectrum = (
            atrial_frequency is not None
            and 4.0 <= atrial_frequency <= 8.5
            and atrial_concentration is not None
            and atrial_concentration >= 0.32
            and p_coverage < 0.45
            and atrial_bandpower_ratio is not None
        )
        flutter_pattern = bool(
            organized_atrial_spectrum
            and (
                atrial_bandpower_ratio >= 0.45
                or (
                    not stable_sinus_like_p_timing
                    and atrial_bandpower_ratio >= 0.35
                )
                or (
                    not stable_sinus_like_p_timing
                    and atrial_frequency >= 4.75
                    and atrial_concentration >= 0.42
                    and atrial_bandpower_ratio >= 0.20
                )
                or (
                    not stable_sinus_like_p_timing
                    and majority_wide_qrs
                    and p_candidate_coverage <= 0.35
                    and atrial_bandpower_ratio >= 0.24
                )
            )
        )
        af_pattern = (
            af_like_irregularity
            and inconsistent_p
            and not stable_sinus_like_p_timing
            and not alternating_rr_pattern
        )
        flags: list[str] = []
        if irregular:
            flags.append("irregular_rr")
        if alternating_rr_pattern:
            flags.append("alternating_rr_ectopic_pattern")
        if inconsistent_p:
            flags.append("no_consistent_discrete_p")
        if stable_sinus_like_p_timing:
            flags.append("stable_discrete_p_coupling")
        if flutter_pattern:
            flags.append("organized_atrial_frequency_pattern")
            flags.append("elevated_atrial_bandpower_burden")
        multiple_atrial_pattern = (
            multiple_fraction >= 0.30
            and not stable_sinus_like_p_timing
        )
        if multiple_atrial_pattern:
            flags.append("multiple_atrial_deflections")

        if multiple_atrial_pattern:
            rhythm_class = RhythmClass.MULTIPLE_ATRIAL_ACTIVITY
            confidence = min(1.0, 0.60 + multiple_fraction * 0.35)
        elif flutter_pattern:
            rhythm_class = RhythmClass.FLUTTER_COMPATIBLE
            confidence = min(
                1.0,
                0.55
                + 0.25 * float(atrial_concentration)
                + 0.10 * float(atrial_bandpower_ratio)
                + 0.10 * (1.0 - p_coverage),
            )
        elif majority_wide_qrs:
            rhythm_class = RhythmClass.WIDE_QRS
            confidence = 0.80
            flags.append("majority_wide_qrs")
        elif alternating_rr_pattern:
            rhythm_class = RhythmClass.MIXED
            confidence = min(
                0.90,
                0.55
                + 0.20 * float(rr_turning_point_fraction)
                + 0.15 * min(1.0, abs(float(rr_lag1_correlation))),
            )
        elif af_pattern:
            rhythm_class = RhythmClass.AF_COMPATIBLE
            confidence = min(
                1.0,
                0.50
                + 0.25 * min(1.0, max(rr_cv, successive) / 0.25)
                + 0.25 * (1.0 - p_coverage),
            )
        elif irregular:
            rhythm_class = RhythmClass.IRREGULAR
            confidence = min(
                0.90,
                0.50 + max(rr_cv, successive),
            )
        else:
            rhythm_class = RhythmClass.REGULAR
            confidence = min(
                0.95,
                0.70
                + 0.20 * p_coverage
                + 0.05 * (1.0 - min(1.0, rr_cv / 0.12)),
            )
        # Rate-aware pattern flags: RhythmClass itself has no rate
        # dimension (a regular narrow-complex rhythm at 180bpm and one at
        # 70bpm both land on REGULAR), so these are layered on as
        # additional flags rather than new RhythmClass values, following
        # the existing additive-flag convention above.
        heart_rate_bpm = 60.0 / rr_median if rr_median > 0 else None
        if heart_rate_bpm is not None:
            if (
                rhythm_class == RhythmClass.REGULAR
                and not majority_wide_qrs
                and heart_rate_bpm >= 150.0
            ):
                flags.append("narrow_complex_tachycardia_pattern")
            if majority_wide_qrs and heart_rate_bpm >= 120.0:
                flags.append("wide_complex_tachycardia_pattern")
            if majority_wide_qrs and 60.0 <= heart_rate_bpm < 110.0:
                # Disjoint from the tachycardia band above: distinguishes
                # the benign accelerated idioventricular rhythm from VT,
                # both otherwise indistinguishable under RhythmClass.WIDE_QRS.
                flags.append("accelerated_idioventricular_pattern")
            if (
                rhythm_class == RhythmClass.REGULAR
                and not majority_wide_qrs
                and p_coverage < 0.15
            ):
                if heart_rate_bpm > 100.0:
                    flags.append("junctional_tachycardia_pattern")
                elif 40.0 <= heart_rate_bpm <= 100.0:
                    flags.append("junctional_rhythm_pattern")
            # AV dissociation: an independently periodic atrial rate (from
            # the QRS-masked spectral search, see _atrial_spectrum) that
            # neither tracks the ventricular rate nor sits in a simple N:1
            # conducted-block ratio to it. This is deliberately a rate
            # comparison, not an attempt to visually resolve a P wave that
            # coincides with or hides inside a QRS -- that target is not
            # reliably visible by amplitude (P is a fraction of typical QRS
            # amplitude) regardless of lead or VCG fusion. It covers both
            # directions: an atrial rate faster than a slow, regular
            # ventricular rate (classic complete AV block -- see
            # third_degree_av_block_pattern below), and a ventricular rate
            # faster than a persisting, unrelated atrial rate (AV
            # dissociation supporting VT over SVT-with-aberrancy in a wide
            # complex tachycardia).
            # Conservative by design: on a single ~10s diagnostic ECG the
            # independent-rate estimate has few RR/spectral cycles to work
            # with, so these thresholds are deliberately stricter than the
            # flutter-search ones above (which benefit from a much sharper,
            # narrowband target). A real-data run across 495 MIMIC-IV
            # diagnostic ECGs at the original thresholds (0.15 / 15% / 5%)
            # flagged av_dissociation_compatible in 9.9% of records --
            # implausibly high for a condition that is clinically rare in
            # an unselected population, and the likely explanation is
            # noise-driven spurious low-frequency peaks on short segments
            # rather than genuine dissociation. Tightened here; still not
            # validated against real annotated AV-block/VT-dissociation
            # cases, only re-checked against the existing synthetic
            # ground-truth scenarios (complete heart block, 2:1 conducted
            # block, silence, VT-with-dissociation).
            if (
                independent_atrial_rate_hz is not None
                and independent_atrial_rate_concentration is not None
                and independent_atrial_rate_concentration >= 0.20
                and rhythm_class in (RhythmClass.REGULAR, RhythmClass.WIDE_QRS)
                and inconsistent_p
                and len(beats) >= 6
            ):
                atrial_rate_bpm = independent_atrial_rate_hz * 60.0
                ratio = atrial_rate_bpm / heart_rate_bpm
                nearest_integer_ratio = round(ratio)
                # Deliberately NOT widened alongside the other thresholds:
                # a genuine N:1 conducted block's ratio is fixed by the
                # conduction mechanism itself and should track very close
                # to exact, whereas two truly independent rates can land
                # near a small integer by pure coincidence (observed with
                # a 75/40 bpm synthetic pair, ratio 1.875, under an 8%
                # tolerance) -- widening this specific tolerance trades
                # away true-positive dissociation detection without
                # reliably cutting the real-data false-positive rate,
                # which is better addressed by the concentration and
                # rate-mismatch thresholds above/below instead.
                integer_ratio_match = (
                    nearest_integer_ratio >= 1
                    and abs(ratio - nearest_integer_ratio)
                    <= 0.05 * nearest_integer_ratio
                )
                rate_mismatch = abs(atrial_rate_bpm - heart_rate_bpm) >= (
                    0.22 * max(atrial_rate_bpm, heart_rate_bpm)
                )
                if rate_mismatch and not integer_ratio_match:
                    flags.append("av_dissociation_compatible")
                    if (
                        atrial_rate_bpm >= heart_rate_bpm * 1.25
                        and 20.0 <= heart_rate_bpm <= 60.0
                    ):
                        flags.append("third_degree_av_block_pattern")
        return RhythmAssessment(
            rhythm_class=rhythm_class,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            rr_median_ms=rr_median * 1000.0,
            rr_cv=rr_cv,
            rr_successive_irregularity=successive,
            rr_lag1_correlation=rr_lag1_correlation,
            rr_turning_point_fraction=rr_turning_point_fraction,
            p_wave_coverage=p_coverage,
            p_timing_mad_ms=p_timing_mad,
            p_candidate_coverage=p_candidate_coverage,
            p_candidate_timing_mad_ms=p_candidate_timing_mad,
            p_candidate_timing_p10_p90_ms=p_candidate_timing_p10_p90,
            atrial_frequency_hz=atrial_frequency,
            atrial_spectral_concentration=atrial_concentration,
            atrial_bandpower_ratio=atrial_bandpower_ratio,
            independent_atrial_rate_hz=independent_atrial_rate_hz,
            independent_atrial_rate_concentration=(
                independent_atrial_rate_concentration
            ),
            flags=tuple(flags),
        )

    @staticmethod
    def _uses_reconstructed_p_triplet(beat: BeatDelineation) -> bool:
        return any(
            marker is not None
            and marker.source == "coherent_multilead_p_triplet"
            for marker in (beat.p_onset, beat.p_peak, beat.p_offset)
        )

    @staticmethod
    def _atrial_spectrum(
        ecg: np.ndarray,
        beats: list[BeatDelineation],
        fs: int,
    ) -> tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Spectral atrial-activity search on the QRS-masked residual.

        Returns ``(flutter_peak_hz, flutter_concentration,
        atrial_bandpower_ratio, independent_atrial_rate_hz,
        independent_atrial_rate_concentration)``. The last two describe a
        second, independent search in the normal/bradycardic sinus band
        (0.5-2.5 Hz) on the SAME QRS-masked-and-interpolated residual used
        for the flutter search above. Because QRS windows are masked out
        before the spectral search runs, this rate estimate never needs to
        visually resolve a P wave that coincides with (or hides inside) a
        QRS -- it infers periodicity from the gaps between beats. Comparing
        it against the independently-known ventricular (RR) rate is how
        AV dissociation (complete heart block, or VT with a slower/faster
        unrelated atrial rhythm) is detected elsewhere in this module,
        without ever attempting direct amplitude-based P detection inside a
        QRS -- a fundamentally unreliable target amplitude-wise.
        """
        if ecg.ndim != 2 or len(ecg) < 2 * fs:
            return None, None, None, None, None
        lead_indices = [
            index for index in (1, 6) if index < ecg.shape[1]
        ]
        if not lead_indices:
            return None, None, None, None, None
        spectra = []
        frequencies = None
        for lead_index in lead_indices:
            data = np.asarray(ecg[:, lead_index], dtype=float).copy()
            mask = np.ones(len(data), dtype=bool)
            for beat in beats:
                start = (
                    beat.r_onset.sample
                    if beat.r_onset is not None
                    else beat.r_peak.sample - int(round(0.10 * fs))
                )
                end = (
                    beat.r_offset.sample
                    if beat.r_offset is not None
                    else beat.r_peak.sample + int(round(0.12 * fs))
                )
                mask[
                    max(0, start - int(round(0.02 * fs))) :
                    min(len(data), end + int(round(0.04 * fs)) + 1)
                ] = False
            usable = np.flatnonzero(mask)
            if usable.size < fs:
                continue
            missing = np.flatnonzero(~mask)
            if missing.size:
                data[missing] = np.interp(missing, usable, data[usable])
            data = sp_signal.detrend(data)
            frequencies, power = sp_signal.welch(
                data,
                fs=fs,
                nperseg=min(len(data), max(512, 4 * fs)),
            )
            spectra.append(power)
        if not spectra or frequencies is None:
            return None, None, None, None, None
        power = np.median(np.vstack(spectra), axis=0)
        band = (frequencies >= 2.5) & (frequencies <= 12.0)
        if not np.any(band):
            return None, None, None, None, None
        flutter_search = (
            (frequencies >= 4.0)
            & (frequencies <= 8.5)
        )
        if not np.any(flutter_search):
            return None, None, None, None, None
        search_frequency = frequencies[flutter_search]
        search_power = power[flutter_search]
        peak_index = int(np.argmax(search_power))
        peak_frequency = float(search_frequency[peak_index])
        local = np.abs(search_frequency - peak_frequency) <= 0.6
        concentration = float(
            np.sum(search_power[local])
            / max(np.sum(power[band]), 1e-12)
        )
        atrial_band = (frequencies >= 3.5) & (frequencies <= 9.0)
        broad_band = (frequencies >= 0.5) & (frequencies <= 40.0)
        if np.count_nonzero(atrial_band) < 2 or np.count_nonzero(broad_band) < 2:
            bandpower_ratio = None
        else:
            atrial_power = float(
                np.trapezoid(
                    power[atrial_band],
                    frequencies[atrial_band],
                )
            )
            broad_power = float(
                np.trapezoid(
                    power[broad_band],
                    frequencies[broad_band],
                )
            )
            bandpower_ratio = float(
                np.clip(
                    atrial_power / max(broad_power, 1e-12),
                    0.0,
                    1.0,
                )
            )
        # Independent atrial-rate search: normal/bradycardic sinus range.
        # A narrower local window than the flutter search above (+/-0.15 Hz
        # vs +/-0.6 Hz) is used because the search band itself is narrower
        # (2.0 Hz vs 4.5 Hz) -- the wider window would span most of the
        # band and stop discriminating a genuine narrow peak from a flat
        # spectrum. Normalized against a dedicated low-frequency reference
        # band (0.3-5.0 Hz), sized the same way `band` (2.5-12 Hz) sizes the
        # flutter search's denominator -- roughly 2x the width of the
        # search band itself -- rather than the full 0.5-40 Hz broad_band,
        # which dilutes a genuine narrow low-frequency peak against
        # unrelated mid/high-frequency masking-residual energy.
        independent_atrial_rate_hz: Optional[float] = None
        independent_atrial_rate_concentration: Optional[float] = None
        sinus_search = (frequencies >= 0.5) & (frequencies <= 2.5)
        low_frequency_reference_band = (
            (frequencies >= 0.3) & (frequencies <= 5.0)
        )
        if (
            np.any(sinus_search)
            and np.count_nonzero(low_frequency_reference_band) >= 2
        ):
            sinus_frequency = frequencies[sinus_search]
            sinus_power = power[sinus_search]
            sinus_peak_index = int(np.argmax(sinus_power))
            candidate_rate = float(sinus_frequency[sinus_peak_index])
            local_sinus = np.abs(sinus_frequency - candidate_rate) <= 0.15
            independent_atrial_rate_hz = candidate_rate
            independent_atrial_rate_concentration = float(
                np.sum(sinus_power[local_sinus])
                / max(np.sum(power[low_frequency_reference_band]), 1e-12)
            )
        return (
            peak_frequency,
            concentration,
            bandpower_ratio,
            independent_atrial_rate_hz,
            independent_atrial_rate_concentration,
        )

    @staticmethod
    def _apply_rhythm_guards(
        beats: list[BeatDelineation],
        rhythm: RhythmAssessment,
    ) -> None:
        rhythm_pattern_guard = rhythm.discrete_p_not_evaluable
        low_coverage_non_sinus_guard = (
            rhythm.rhythm_class
            in {
                RhythmClass.WIDE_QRS,
                RhythmClass.MIXED,
            }
            and rhythm.low_discrete_p_coverage
            and (rhythm.p_wave_coverage or 0.0) == 0.0
        )
        if (
            not (rhythm_pattern_guard or low_coverage_non_sinus_guard)
            or rhythm.confidence < 0.60
        ):
            return
        status = (
            WaveStatus.REPLACED_BY_ATRIAL_ACTIVITY
            if rhythm.rhythm_class
            in {
                RhythmClass.AF_COMPATIBLE,
                RhythmClass.FLUTTER_COMPATIBLE,
            }
            else WaveStatus.NOT_ASSESSABLE
        )
        for beat in beats:
            beat.p_onset = None
            beat.p_peak = None
            beat.p_offset = None
            beat.p_wave_absent = True
            beat.p_status = status
            beat.review_reasons += (
                "discrete_p_measurement_suppressed_by_rhythm_context",
            )

    @staticmethod
    def _mark_interbeat_overlap(
        beats: list[BeatDelineation],
        fs: int,
    ) -> None:
        minimum_gap = int(round(0.020 * fs))
        for current, following in zip(beats[:-1], beats[1:]):
            if current.t_offset is None or following.p_onset is None:
                continue
            gap = following.p_onset.sample - current.t_offset.sample
            if gap < minimum_gap:
                current.p_t_fusion_risk = True
                following.p_t_fusion_risk = True
                if gap < 0:
                    current.t_status = WaveStatus.FUSED
                    following.p_status = WaveStatus.FUSED

    @staticmethod
    def _mark_morphology_outliers(
        ecg: np.ndarray,
        beats: list[BeatDelineation],
        fs: int,
    ) -> dict[str, object]:
        """Identify a coherent dominant QRS morphology, then search any
        remaining beats for a coherent secondary/tertiary population
        (bigeminy, multifocal ectopy, intermittent aberrancy) instead of
        discarding them as undifferentiated noise. No outcome data is used.
        """

        signal = np.asarray(ecg, dtype=float)
        if signal.ndim != 2 or fs <= 0 or len(beats) < 4:
            return {
                "morphology_consensus_assessable": False,
                "morphology_candidate_beats": 0,
                "morphology_dominant_fraction": None,
                "severe_signal_incoherence": False,
                "morphology_cluster_count": 0,
            }
        before = max(1, int(round(0.120 * fs)))
        after = max(1, int(round(0.180 * fs)))
        # All snippets must share one length to be stacked into a matrix
        # for cosine-similarity clustering, so the window is narrowed
        # record-wide (not per beat) using the record's shortest RR --
        # otherwise, at fast heart rates, a beat's fixed-width snippet can
        # reach into an adjacent beat's QRS and contaminate the morphology
        # similarity computation (worst for wide+fast/VT-like rhythms,
        # where the neighbor's own QRS deflection is largest).
        if len(beats) >= 2:
            rr_samples = np.diff(
                np.asarray([beat.r_peak.sample for beat in beats], dtype=float)
            )
            positive_rr = rr_samples[rr_samples > 0]
            if positive_rr.size:
                half_min_rr = max(1, int(np.min(positive_rr)) // 2)
                before = min(before, half_min_rr)
                after = min(after, half_min_rr)
        vectors: list[np.ndarray] = []
        raw_vectors: list[np.ndarray] = []
        beat_indices: list[int] = []
        for beat_index, beat in enumerate(beats):
            start = beat.r_peak.sample - before
            stop = beat.r_peak.sample + after
            if start < 0 or stop > signal.shape[0]:
                continue
            snippet = np.array(signal[start:stop], dtype=float, copy=True)
            if not np.all(np.isfinite(snippet)):
                continue
            snippet -= np.median(snippet, axis=0, keepdims=True)
            vector = snippet.ravel()
            scale = float(np.linalg.norm(vector))
            if scale <= 1e-9:
                continue
            vectors.append(vector / scale)
            raw_vectors.append(vector)
            beat_indices.append(beat_index)
        if len(vectors) < 4:
            return {
                "morphology_consensus_assessable": False,
                "morphology_candidate_beats": len(vectors),
                "morphology_dominant_fraction": None,
                "severe_signal_incoherence": False,
                "morphology_cluster_count": 0,
            }

        matrix = np.asarray(vectors, dtype=float)
        raw_matrix = np.asarray(raw_vectors, dtype=float)
        similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
        medoid_index = int(np.argmax(np.median(similarities, axis=1)))
        initial_members = similarities[medoid_index] >= 0.72
        if int(np.sum(initial_members)) < 2:
            initial_members[medoid_index] = True
        template = np.median(matrix[initial_members], axis=0)
        template_norm = float(np.linalg.norm(template))
        if template_norm <= 1e-9:
            dominant_members = np.zeros(len(matrix), dtype=bool)
            dominant_similarity = 0.0
        else:
            template /= template_norm
            template_similarities = matrix @ template
            dominant_members = template_similarities >= 0.70
            dominant_similarity = (
                float(np.median(template_similarities[dominant_members]))
                if np.any(dominant_members)
                else 0.0
            )
        dominant_fraction = float(np.mean(dominant_members))
        dominant_count = int(np.sum(dominant_members))
        for local_index, beat_index in enumerate(beat_indices):
            beats[beat_index].morphology_outlier = not bool(
                dominant_members[local_index]
            )
            beats[beat_index].morphology_cluster_id = (
                0 if dominant_members[local_index] else None
            )
        severe = dominant_count < 3 or dominant_fraction < 0.55

        result: dict[str, object] = {
            "morphology_consensus_assessable": True,
            "morphology_candidate_beats": len(vectors),
            "morphology_dominant_beat_count": dominant_count,
            "morphology_dominant_fraction": dominant_fraction,
            "morphology_dominant_median_similarity": dominant_similarity,
            "morphology_similarity_threshold": 0.70,
            "morphology_minimum_dominant_fraction": 0.55,
            "severe_signal_incoherence": severe,
            "morphology_cluster_count": 1 if dominant_count else 0,
            "morphology_multi_morphology_compatible": False,
        }

        # Beyond the dominant/primary population, search the remainder for
        # up to two further coherent clusters (secondary, tertiary) using
        # the same medoid + cosine-similarity procedure. A candidate medoid
        # gets a small bonus when its coupling ratio (BeatDelineation.
        # coupling_ratio, from the RR-to-previous context assigned in
        # FusionDelineator._assign_beat_coupling_context) is clearly not 1.0
        # -- a consistently premature or compensated coupling pattern is
        # independent physiological corroboration of a real second
        # population, not just waveform noise. Shape similarity remains the
        # primary criterion; coupling only breaks near-ties.
        minimum_cluster_beats = 3
        max_extra_clusters = 2
        remaining_local = np.flatnonzero(~dominant_members)
        coupling = np.array(
            [
                beats[beat_indices[local]].coupling_ratio
                if beats[beat_indices[local]].coupling_ratio is not None
                else np.nan
                for local in range(len(matrix))
            ],
            dtype=float,
        )
        extra_clusters: list[dict[str, object]] = []
        while (
            len(extra_clusters) < max_extra_clusters
            and len(remaining_local) >= minimum_cluster_beats
        ):
            sub_matrix = matrix[remaining_local]
            sub_similarities = np.clip(sub_matrix @ sub_matrix.T, -1.0, 1.0)
            sub_coupling = coupling[remaining_local]
            coupling_bonus = np.where(
                np.isfinite(sub_coupling) & (np.abs(sub_coupling - 1.0) >= 0.15),
                0.05,
                0.0,
            )
            sub_medoid_local = int(
                np.argmax(np.median(sub_similarities, axis=1) + coupling_bonus)
            )
            sub_initial = sub_similarities[sub_medoid_local] >= 0.72
            if int(np.sum(sub_initial)) < 2:
                sub_initial[sub_medoid_local] = True
            sub_template = np.median(sub_matrix[sub_initial], axis=0)
            sub_template_norm = float(np.linalg.norm(sub_template))
            if sub_template_norm <= 1e-9:
                break
            sub_template /= sub_template_norm
            sub_template_similarities = sub_matrix @ sub_template
            sub_members = sub_template_similarities >= 0.70
            member_count = int(np.sum(sub_members))
            if member_count < minimum_cluster_beats:
                break
            member_local_indices = remaining_local[sub_members]
            cluster_id = len(extra_clusters) + 1
            for local in member_local_indices:
                beats[beat_indices[int(local)]].morphology_cluster_id = cluster_id
            member_coupling = coupling[member_local_indices]
            finite_member_coupling = member_coupling[np.isfinite(member_coupling)]
            raw_cluster_template = np.median(raw_matrix[member_local_indices], axis=0)
            raw_primary_template = (
                np.median(raw_matrix[dominant_members], axis=0)
                if dominant_count
                else None
            )
            centroid_distance_mv = (
                float(
                    np.sqrt(
                        np.mean(
                            (raw_cluster_template - raw_primary_template) ** 2
                        )
                    )
                )
                if raw_primary_template is not None
                else None
            )
            extra_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "beat_count": member_count,
                    "fraction_of_candidates": member_count / len(vectors),
                    "median_similarity": float(
                        np.median(sub_template_similarities[sub_members])
                    ),
                    "centroid_distance_mv": centroid_distance_mv,
                    "mean_coupling_ratio": (
                        float(np.mean(finite_member_coupling))
                        if finite_member_coupling.size
                        else None
                    ),
                }
            )
            remaining_local = remaining_local[~sub_members]

        result["morphology_cluster_count"] = (
            1 if dominant_count else 0
        ) + len(extra_clusters)
        if extra_clusters:
            total_clustered = dominant_count + sum(
                cluster["beat_count"] for cluster in extra_clusters
            )
            clustered_fraction = total_clustered / len(vectors)
            secondary = extra_clusters[0]
            result.update(
                {
                    "morphology_clustered_fraction": clustered_fraction,
                    "morphology_secondary_beat_count": secondary["beat_count"],
                    "morphology_secondary_fraction": (
                        secondary["fraction_of_candidates"]
                    ),
                    "morphology_secondary_median_similarity": (
                        secondary["median_similarity"]
                    ),
                    "morphology_secondary_centroid_distance_mv": (
                        secondary["centroid_distance_mv"]
                    ),
                    "morphology_secondary_mean_coupling_ratio": (
                        secondary["mean_coupling_ratio"]
                    ),
                    "morphology_extra_clusters": extra_clusters,
                }
            )
            multi_morphology_compatible = severe and clustered_fraction >= 0.90
            result["morphology_multi_morphology_compatible"] = (
                multi_morphology_compatible
            )
            if multi_morphology_compatible:
                result["severe_signal_incoherence"] = False
        return result

    def _compute_quality_score(
        self,
        record: RecordDelineation,
    ) -> tuple[float, dict[str, object]]:
        if not record.beats:
            return 0.0, {
                "accepted_fraction": 0.0,
                "marker_conflict_fraction": 1.0,
            }
        representative = record.representative_beat
        if representative is None:
            return 0.0, {
                "accepted_fraction": 0.0,
                "marker_conflict_fraction": 1.0,
            }
        essential = [
            representative.r_peak.confidence,
            representative.marker_confidence("r_onset", "r_offset"),
            representative.marker_confidence("t_onset", "t_offset"),
        ]
        if not record.rhythm.discrete_p_not_evaluable:
            p_confidence = record.segment_confidence("p")
            if p_confidence > 0:
                essential.append(p_confidence)
        marker_quality = float(np.mean(essential))
        accepted_fraction = (
            record.n_accepted_beats / max(record.n_beats, 1)
        )
        conflict_marker_entries = [
            (attribute, marker)
            for beat in record.beats
            for attribute in _MARKER_ATTRIBUTES
            for marker in (getattr(beat, attribute),)
            if marker is not None
            and marker.conflict_ms
            > _MARKER_TOLERANCE_MS.get(attribute, 25.0)
        ]
        available_markers = sum(
            getattr(beat, attribute) is not None
            for beat in record.beats
            for attribute in _MARKER_ATTRIBUTES
        )
        conflict_fraction = (
            len(conflict_marker_entries) / max(available_markers, 1)
        )
        unresolved_conflict_markers = [
            marker
            for attribute, marker in conflict_marker_entries
            if (
                marker.confidence < 0.50
                or marker.uncertainty_ms
                > _MARKER_TOLERANCE_MS.get(attribute, 25.0)
                or marker.support_count < 2
            )
        ]
        unresolved_conflict_fraction = (
            len(unresolved_conflict_markers)
            / max(available_markers, 1)
        )
        representative_attributes = [
            "r_onset",
            "r_offset",
            "t_onset",
            "t_offset",
        ]
        if record.is_segment_evaluable("p"):
            representative_attributes.extend(("p_onset", "p_offset"))
        representative_unresolved_conflicts = [
            marker
            for attribute in representative_attributes
            for marker in (getattr(representative, attribute),)
            if (
                marker is not None
                and marker.conflict_ms
                > _MARKER_TOLERANCE_MS.get(attribute, 25.0)
                and (
                    marker.confidence < 0.50
                    or marker.uncertainty_ms
                    > _MARKER_TOLERANCE_MS.get(attribute, 25.0)
                )
            )
        ]
        agreement = 1.0 - min(
            1.0,
            unresolved_conflict_fraction * 3.0,
        )
        beat_confidences = [
            beat.r_peak.confidence
            for beat in record.accepted_beats
        ]
        r_quality = (
            float(np.median(beat_confidences))
            if beat_confidences
            else 0.0
        )
        quality = float(
            np.clip(
                0.45 * marker_quality
                + 0.25 * accepted_fraction
                + 0.20 * agreement
                + 0.10 * r_quality,
                0.0,
                1.0,
            )
        )
        if bool(record.diagnostics.get("severe_signal_incoherence", False)):
            quality = min(quality, 0.45)
        diagnostics: dict[str, object] = {
            **record.diagnostics,
            "accepted_fraction": accepted_fraction,
            "marker_conflict_fraction": conflict_fraction,
            "unresolved_marker_conflict_fraction": (
                unresolved_conflict_fraction
            ),
            "representative_unresolved_conflict_count": len(
                representative_unresolved_conflicts
            ),
            "marker_quality": marker_quality,
            "source_agreement": agreement,
            "rhythm_class": record.rhythm.rhythm_class.value,
            "rhythm_confidence": record.rhythm.confidence,
            "p_wave_evaluable": record.is_segment_evaluable("p"),
            "qrs_segment_confidence": record.segment_confidence("qrs"),
            "t_segment_confidence": record.segment_confidence("t"),
            "p_segment_confidence": record.segment_confidence("p"),
            "pr_segment_confidence": record.segment_confidence("pr"),
            "qrst_segment_confidence": record.segment_confidence("qrst"),
            "qt_rr_fast_beat_exclusion_count": (
                record.qt_rr_fast_beat_exclusion_count()
            ),
        }
        return quality, diagnostics

    @staticmethod
    def _review_reasons(
        record: RecordDelineation,
        quality: float,
        diagnostics: dict[str, object],
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if quality < 0.55:
            reasons.append("low_fiducial_quality")
        if record.n_beats < 3:
            reasons.append("too_few_beats_for_stable_consensus")
        if float(diagnostics.get("accepted_fraction", 0.0)) < 0.60:
            reasons.append("unstable_beat_morphology_or_fiducials")
        if bool(diagnostics.get("severe_signal_incoherence", False)):
            reasons.append("no_reliable_dominant_beat_morphology")
        if int(
            diagnostics.get(
                "representative_unresolved_conflict_count",
                0,
            )
        ) > 0:
            reasons.append("detector_boundary_conflicts")
        if float(diagnostics.get("qrs_segment_confidence", 0.0)) < 0.35:
            reasons.append("qrs_not_reliably_measurable")
        if float(diagnostics.get("t_segment_confidence", 0.0)) < 0.35:
            reasons.append("t_wave_not_reliably_measurable")
        if record.rhythm.rhythm_class in {
            RhythmClass.AF_COMPATIBLE,
            RhythmClass.FLUTTER_COMPATIBLE,
            RhythmClass.MULTIPLE_ATRIAL_ACTIVITY,
        }:
            reasons.append("p_pr_measurements_not_evaluable_in_rhythm_context")
        if any(beat.p_t_fusion_risk for beat in record.beats):
            reasons.append("p_t_overlap_or_fusion_risk")
        return tuple(dict.fromkeys(reasons))
