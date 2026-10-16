from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.signal import find_peaks

from ..data_loader import LEAD_NAMES_12
from ..vcg_transform import (
    aggregate_loop_axes,
    loop_dominant_axis,
    orthogonal_residual_magnitude,
)
from .base import (
    measurement_confidence_threshold,
    BeatDelineation,
    DelineatorBase,
    MarkerPoint,
    WaveStatus,
)


# P-window's own gap-after-T margin and pre-QRS margin (used to build
# p_start/p_end). t_end's own margin before the next QRS onset must stay
# >= their sum, or a T-offset landing right at its own window boundary can
# invert the next beat's P-window and P is missed even though a real gap
# exists -- common at fast heart rates where beats crowd together.
P_WINDOW_GAP_AFTER_T_FRACTION = 0.020
P_WINDOW_PRE_QRS_MARGIN_FRACTION = 0.018
T_WINDOW_PRE_NEXT_QRS_MARGIN_FRACTION = (
    P_WINDOW_GAP_AFTER_T_FRACTION + P_WINDOW_PRE_QRS_MARGIN_FRACTION
)

P_WAVE_PREFERRED = {1: 1.0, 7: 0.9, 0: 0.8, 5: 0.75, 4: 0.65}
T_WAVE_PREFERRED = {
    9: 1.0,
    10: 1.0,
    11: 0.95,
    8: 0.9,
    7: 0.8,
    1: 0.75,
    0: 0.65,
}
QRS_PREFERRED = {
    1: 0.9,
    5: 0.9,
    6: 0.85,
    7: 0.85,
    8: 0.85,
    9: 0.85,
    10: 0.85,
    11: 0.85,
    0: 0.75,
}

# H4: sentinel lead_index for the derived QRS-orthogonal spatial residual
# channel (see orthogonal_residual_magnitude in core.vcg_transform), kept
# out of the 0-11 real-lead range and distinct from _lead_wave_candidates'
# own -1 placeholder. Its preference weight is set below the weakest real
# P lead (0.65) since it is a derived, less clinically validated channel --
# it can only ever ADD supporting evidence to a candidate a real lead also
# found (the >=2-distinct-lead-index consensus gate still applies), never
# invent a P wave by itself.
SPATIAL_ORTHOGONAL_LEAD_INDEX = -2
SPATIAL_ORTHOGONAL_P_PREFERENCE = 0.55


@dataclass(frozen=True)
class _WaveCandidate:
    onset: int
    peak: int
    offset: int
    snr: float
    lead_index: int


@dataclass(frozen=True)
class _WaveConsensus:
    onset: MarkerPoint
    peak: MarkerPoint
    offset: MarkerPoint
    candidate_count: int
    # (onset_sample, peak_sample, offset_sample) per credible episode,
    # populated only when candidate_count > 1 -- lets a caller attempt
    # secondary disambiguation (e.g. via VCG loop direction) instead of
    # discarding the ambiguous candidates outright.
    episode_windows: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True)
class _QRSBounds:
    onset: MarkerPoint
    offset: MarkerPoint


@dataclass(frozen=True)
class _LeadQRSBounds:
    onset: int
    offset: int
    snr: float
    lead_index: int


def _mad(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return 0.0
    median = float(np.median(data))
    return float(np.median(np.abs(data - median)))


def _robust_spread(values: np.ndarray, fs: int) -> float:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return 0.0
    return float(
        (np.percentile(data, 90) - np.percentile(data, 10))
        / fs
        * 1000.0
    )


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class MultileadDelineator(DelineatorBase):
    """Physiology-constrained, polarity-invariant 12-lead delineator.

    The detector measures time-coherent global boundaries from multiple
    leads. It uses adaptive local noise estimates and can abstain when a
    discrete wave cannot be established. Missing detector output is therefore
    not interpreted as physiological wave absence.
    """

    def __init__(self, *, spatial_p_enhancement: bool = False):
        # H4: QRS-orthogonal spatial residual channel for the P-wave
        # search (see _spatial_orthogonal_candidates). Implemented and
        # unit-tested (it does what it is designed to do: recover a P
        # wave supported by only one real lead once the QRS-orthogonal
        # axis is well separated -- see the H4 synthetic checks), but a
        # 200-record LUDB A/B run (2026-08-11, spatial_p_enhancement=True
        # vs False) showed a net-negative real-data effect: raw P
        # sensitivity 0.852 vs 0.867 and record-level P-evaluable-rate
        # 0.615 vs 0.630 (18 records lost P detections vs 8 gained).
        # Root cause: the per-beat QRS axis estimate is unreliable on
        # wide/ectopic/polymorphic QRS complexes -- exactly the harder
        # rhythms this project's own taxonomy already flags separately
        # (majority_wide_qrs, accelerated_idioventricular_pattern) -- and
        # a spurious spatial candidate there can win episode selection
        # away from a correct real-lead-only detection rather than merely
        # reinforcing it. QRS/T detection is completely unaffected either
        # way (the mechanism is scoped to the P search only). Left as an
        # explicit, off-by-default opt-in rather than deleted, since the
        # narrow scenario it targets is real; enabling it in production
        # would need the episode-selection interaction hardened first,
        # not just a threshold tweak.
        self.spatial_p_enhancement = spatial_p_enhancement

    # The T window used to end 38 ms before the next QRS, so at fast rates
    # it contained the next P wave, and the T end, measured against the
    # window's terminal level, ran into it. Against the QTDB cardiologists,
    # counting only beats whose T the analysis uses, T ends more than 80 ms
    # late fell from 4.0 % to 3.1 % (half A) and 4.7 % to 3.8 % (half B),
    # while beats with a usable T rose from 90.8 % to 96.1 % and 91.5 % to
    # 93.9 %; at 100-120/min half B went from 25 % late to none. The window
    # now ends 160 ms before the next QRS -- before its P wave -- but never
    # before 240 ms after this QRS or a Fridericia QT at QTc 440 ms, so a
    # long T at a fast rate is not cut (above 120/min the T end is 20 ms
    # early on five records; audit_tools/qtdb_tend_by_rate.py). Cutting the
    # T where it still ran at the window's end was tried and not kept: it
    # removed 1.5 % of usable T ends for no gain in accuracy.
    t_window_next_p_margin_s: Optional[float] = 0.16
    t_window_minimum_after_qrs_s: float = 0.24
    t_window_floor_qtc_s: Optional[float] = 0.44

    @property
    def name(self) -> str:
        return "multilead"

    def delineate(
        self,
        ecg_12lead: np.ndarray,
        r_peaks: np.ndarray,
        fs: int,
        vcg_xyz: Optional[np.ndarray] = None,
    ) -> list[BeatDelineation]:
        ecg = np.asarray(ecg_12lead, dtype=float)
        peaks = np.asarray(r_peaks, dtype=int)
        if ecg.ndim != 2 or ecg.shape[0] < 3 or ecg.shape[1] < 1:
            return []
        peaks = np.unique(peaks[(peaks > 0) & (peaks < len(ecg) - 1)])
        if peaks.size == 0:
            return []

        qrs_bounds: list[Optional[_QRSBounds]] = []
        for index, r_peak in enumerate(peaks):
            left_limit, right_limit = self._beat_limits(peaks, index, len(ecg))
            qrs_bounds.append(
                self._find_qrs_bounds(
                    ecg,
                    int(r_peak),
                    fs,
                    left_limit=left_limit,
                    right_limit=right_limit,
                )
            )

        beats: list[BeatDelineation] = []
        # H3: VCG loop-direction disambiguation for ambiguous P-window
        # candidates. reference_p_axes/reference_t_axes accumulate the
        # dominant spatial axis of confidently-single-candidate P/T
        # segments as the loop is processed; ambiguous_p_episodes records,
        # per beat index, the candidate windows that couldn't be resolved
        # by timing alone. Both are consumed in a second pass after the
        # main loop, once a reliable per-record reference axis exists.
        reference_p_axes: list[np.ndarray] = []
        reference_t_axes: list[np.ndarray] = []
        ambiguous_p_episodes: dict[int, tuple[tuple[int, int, int], ...]] = {}
        # H1: beats whose ordinary pre-QRS P search found nothing at all
        # (not the "multiple candidates" case above, which H3 may still
        # resolve) -- candidates for a retrograde (post-QRS) P search,
        # attempted in a later pass once per-record P/T reference axes
        # are available.
        retrograde_p_needed: list[tuple[int, int]] = []
        for index, r_peak in enumerate(peaks):
            rp = int(r_peak)
            beat = BeatDelineation(
                r_peak=MarkerPoint(
                    sample=rp,
                    confidence=0.90,
                    source="multilead_r_reference",
                    uncertainty_ms=1000.0 / fs,
                    support_count=max(1, min(ecg.shape[1], 12)),
                    lead_names=tuple(LEAD_NAMES_12[: ecg.shape[1]]),
                    evidence_sources=("r_peak_detector",),
                )
            )
            qrs = qrs_bounds[index]
            if qrs is not None:
                beat.r_onset = qrs.onset
                beat.r_offset = qrs.offset
                beat.j_point = MarkerPoint(
                    sample=qrs.offset.sample,
                    confidence=qrs.offset.confidence,
                    source="multilead_global_qrs_offset",
                    uncertainty_ms=qrs.offset.uncertainty_ms,
                    support_count=qrs.offset.support_count,
                    lead_names=qrs.offset.lead_names,
                    evidence_sources=qrs.offset.evidence_sources,
                )
                beat.qrs_status = WaveStatus.PRESENT
                qrs_ms = beat.qrs_duration_ms(fs)
                if (
                    qrs_ms is not None
                    and beat.qrs_definitely_exceeds_ms(fs, 120.0)
                ):
                    beat.wide_qrs = True
                    if beat.qrs_definitely_exceeds_ms(fs, 140.0):
                        beat.bundle_branch_block = True
                beat.q_peak = self._find_q_peak_marker(
                    ecg,
                    rp,
                    qrs.onset.sample,
                    fs,
                )
                beat.s_peak = self._find_s_peak_marker(
                    ecg,
                    rp,
                    qrs.offset.sample,
                    fs,
                )
            else:
                beat.qrs_status = WaveStatus.NOT_ASSESSABLE
                beat.review_reasons += ("qrs_boundaries_not_established",)

            current_qrs_onset = (
                beat.r_onset.sample
                if beat.r_onset is not None
                else max(0, rp - int(round(0.12 * fs)))
            )
            previous_qrs_end = (
                qrs_bounds[index - 1].offset.sample
                if index > 0 and qrs_bounds[index - 1] is not None
                else (
                    # Previous beat exists but its own QRS bounds could not
                    # be established: anchor the fallback to the PREVIOUS
                    # beat's own R-peak, not the current beat's -- using
                    # the current beat's rp here placed this point before
                    # the previous beat's own R-peak at short RR (fast
                    # rates), which is physiologically impossible.
                    min(len(ecg) - 1, int(peaks[index - 1]) + int(round(0.16 * fs)))
                    if index > 0
                    else max(0, rp - int(round(0.48 * fs)))
                )
            )
            previous_repolarization_end = (
                beats[index - 1].t_offset.sample
                if index > 0
                and index - 1 < len(beats)
                and beats[index - 1].t_offset is not None
                and beats[index - 1].segment_confidence("t") >= measurement_confidence_threshold("t")
                else previous_qrs_end
            )
            p_start = max(
                previous_repolarization_end
                + int(round(P_WINDOW_GAP_AFTER_T_FRACTION * fs)),
                current_qrs_onset - int(round(0.45 * fs)),
                0,
            )
            p_end = current_qrs_onset - int(
                round(P_WINDOW_PRE_QRS_MARGIN_FRACTION * fs)
            )
            spatial_p_candidates: tuple[tuple[_WaveCandidate, float], ...] = ()
            if (
                self.spatial_p_enhancement
                and vcg_xyz is not None
                and qrs is not None
            ):
                qrs_axis = loop_dominant_axis(
                    vcg_xyz, qrs.onset.sample, qrs.offset.sample
                )
                if qrs_axis is not None:
                    spatial_p_candidates = self._spatial_orthogonal_candidates(
                        vcg_xyz,
                        qrs_axis,
                        p_start,
                        p_end,
                        fs,
                        wave="p",
                        reference_sample=current_qrs_onset,
                    )
            p_consensus = self._find_wave_consensus(
                ecg,
                p_start,
                p_end,
                fs,
                wave="p",
                reference_sample=current_qrs_onset,
                extra_candidates=spatial_p_candidates,
            )
            if p_consensus is not None:
                beat.atrial_activity_count = p_consensus.candidate_count
                if p_consensus.candidate_count == 1:
                    beat.p_onset = p_consensus.onset
                    beat.p_peak = p_consensus.peak
                    beat.p_offset = p_consensus.offset
                    beat.p_status = WaveStatus.PRESENT
                    if vcg_xyz is not None:
                        axis = self._loop_direction(
                            vcg_xyz,
                            beat.p_onset.sample,
                            beat.p_offset.sample,
                        )
                        if axis is not None:
                            reference_p_axes.append(axis)
                else:
                    beat.p_status = WaveStatus.NOT_ASSESSABLE
                    beat.review_reasons += (
                        "multiple_atrial_deflections_before_qrs",
                    )
                    if vcg_xyz is not None and p_consensus.episode_windows:
                        ambiguous_p_episodes[index] = p_consensus.episode_windows
            else:
                beat.p_status = WaveStatus.INDETERMINATE

            current_qrs_end = (
                beat.r_offset.sample
                if beat.r_offset is not None
                else min(len(ecg) - 1, rp + int(round(0.16 * fs)))
            )
            if beat.p_status != WaveStatus.PRESENT:
                retrograde_p_needed.append((index, current_qrs_end))
            next_qrs_onset = (
                qrs_bounds[index + 1].onset.sample
                if index + 1 < len(qrs_bounds)
                and qrs_bounds[index + 1] is not None
                else min(len(ecg), rp + int(round(0.70 * fs)))
            )
            t_start = current_qrs_end + int(round(0.012 * fs))
            t_end = min(
                next_qrs_onset
                - int(round(T_WINDOW_PRE_NEXT_QRS_MARGIN_FRACTION * fs)),
                rp + int(round(0.68 * fs)),
                len(ecg),
            )
            if (
                self.t_window_next_p_margin_s is not None
                and index + 1 < len(qrs_bounds)
                and qrs_bounds[index + 1] is not None
            ):
                guarded_end = max(
                    next_qrs_onset
                    - int(round(self.t_window_next_p_margin_s * fs)),
                    current_qrs_end
                    + int(round(self.t_window_minimum_after_qrs_s * fs)),
                )
                if self.t_window_floor_qtc_s is not None:
                    current_qrs_onset = (
                        beat.r_onset.sample
                        if beat.r_onset is not None
                        else rp - int(round(0.04 * fs))
                    )
                    rr_s = (int(peaks[index + 1]) - rp) / fs
                    guarded_end = max(
                        guarded_end,
                        current_qrs_onset
                        + int(round(
                            self.t_window_floor_qtc_s * rr_s ** (1.0 / 3.0) * fs
                        )),
                    )
                t_end = min(t_end, guarded_end)
            t_consensus = self._find_wave_consensus(
                ecg,
                t_start,
                t_end,
                fs,
                wave="t",
                reference_sample=current_qrs_end,
            )
            if t_consensus is not None:
                beat.t_onset = t_consensus.onset
                beat.t_peak = t_consensus.peak
                beat.t_offset = t_consensus.offset
                beat.t_status = WaveStatus.PRESENT
                if vcg_xyz is not None:
                    axis = self._loop_direction(
                        vcg_xyz,
                        beat.t_onset.sample,
                        beat.t_offset.sample,
                    )
                    if axis is not None:
                        reference_t_axes.append(axis)
            else:
                beat.t_status = WaveStatus.INDETERMINATE

            beats.append(beat)

        if vcg_xyz is not None and ambiguous_p_episodes:
            self._disambiguate_ambiguous_p_candidates(
                beats,
                vcg_xyz,
                reference_p_axes,
                reference_t_axes,
                ambiguous_p_episodes,
                fs,
            )

        if retrograde_p_needed:
            self._search_retrograde_p_candidates(
                beats,
                ecg,
                vcg_xyz,
                fs,
                reference_p_axes,
                reference_t_axes,
                retrograde_p_needed,
            )

        self._mark_overlap_and_edge_risk(beats, fs)
        return beats

    def _search_retrograde_p_candidates(
        self,
        beats: list[BeatDelineation],
        ecg: np.ndarray,
        vcg_xyz: Optional[np.ndarray],
        fs: int,
        reference_p_axes: list[np.ndarray],
        reference_t_axes: list[np.ndarray],
        retrograde_p_needed: list[tuple[int, int]],
    ) -> None:
        """H1: search for a P-like deflection AFTER this beat's own QRS,
        for beats whose ordinary pre-QRS search found nothing (or an
        unresolved multiple-candidate ambiguity) -- the signature of a
        retrograde (AV-junctional) P wave. Reuses the same per-lead
        candidate finder as the forward P/T search (_lead_wave_candidates
        is polarity-invariant: it operates on abs(residual), so no
        separate handling is needed for a retrograde P's typically
        inverted inferior-lead polarity). This deliberately does NOT
        reuse _find_wave_consensus's own scoring for wave='p', which
        assumes the peak PRECEDES the reference sample by ~160ms -- the
        opposite timing relationship for a post-QRS search. When VCG
        data and a reliable per-record P/T reference axis are available,
        a candidate must also sit closer in spatial loop direction to
        the P reference than to the T reference (the same principle as
        _disambiguate_ambiguous_p_candidates), to reduce confusion with
        an early T-wave onset immediately following the QRS.
        """
        p_reference_axis = (
            self._aggregate_axis(reference_p_axes)
            if vcg_xyz is not None
            else None
        )
        t_reference_axis = (
            self._aggregate_axis(reference_t_axes)
            if vcg_xyz is not None
            else None
        )
        gap_samples = int(round(0.012 * fs))
        window_samples = int(round(0.280 * fs))
        tolerance = max(1, int(round(0.045 * fs)))
        min_window = max(4, int(round(0.035 * fs)))
        # _lead_wave_candidates' own physiologic-latency gate for wave="p"
        # requires (reference_sample - peak_sample) to fall in [35, 430]ms,
        # i.e. it expects the peak to PRECEDE reference_sample -- exactly
        # backwards from a post-QRS search, where the peak comes after the
        # window start. Placing reference_sample just past the window's
        # own far end (with a fixed pad) keeps every possible in-window
        # peak position mapped into that same forward-looking gate,
        # without having to special-case the gate itself for this caller.
        latency_pad_samples = int(round(0.050 * fs))
        for beat_index, qrs_end_sample in retrograde_p_needed:
            beat = beats[beat_index]
            if beat.p_status == WaveStatus.PRESENT:
                continue  # resolved by H3's disambiguation pass above
            start = qrs_end_sample + gap_samples
            end = min(len(ecg), qrs_end_sample + window_samples)
            # This beat's own T-wave is already known by this point (T is
            # computed earlier in the same main-loop pass). Without this
            # bound, the retrograde-P window's nominal 280ms reach can run
            # straight into the real T-onset region, and a smooth T
            # upstroke can present as a "candidate" to the same
            # amplitude-residual peak search the P/T detectors share --
            # capping the window here keeps a retrograde-P finding from
            # ever being a relabeled piece of this beat's own T wave.
            if (
                beat.t_onset is not None
                and beat.segment_confidence("t") >= measurement_confidence_threshold("t")
            ):
                end = min(end, beat.t_onset.sample - gap_samples)
            if end - start < min_window:
                continue
            latency_reference = end + latency_pad_samples
            weighted_candidates: list[tuple[_WaveCandidate, float]] = []
            for lead_index, preference in P_WAVE_PREFERRED.items():
                if lead_index >= ecg.shape[1]:
                    continue
                found = self._lead_wave_candidates(
                    ecg[:, lead_index],
                    start,
                    end,
                    fs,
                    wave="p",
                    reference_sample=latency_reference,
                )
                weighted_candidates.extend(
                    (
                        _WaveCandidate(
                            onset=item.onset,
                            peak=item.peak,
                            offset=item.offset,
                            snr=item.snr,
                            lead_index=lead_index,
                        ),
                        preference,
                    )
                    for item in found
                )
            if (
                self.spatial_p_enhancement
                and vcg_xyz is not None
                and beat.r_onset is not None
                and beat.r_offset is not None
            ):
                qrs_axis = loop_dominant_axis(
                    vcg_xyz, beat.r_onset.sample, beat.r_offset.sample
                )
                if qrs_axis is not None:
                    weighted_candidates.extend(
                        self._spatial_orthogonal_candidates(
                            vcg_xyz,
                            qrs_axis,
                            start,
                            end,
                            fs,
                            wave="p",
                            reference_sample=latency_reference,
                        )
                    )
            if not weighted_candidates:
                continue
            clusters = self._cluster_wave_candidates(
                weighted_candidates, tolerance
            )
            supported = [
                cluster
                for cluster in clusters
                if len({item[0].lead_index for item in cluster}) >= 2
            ]
            if not supported:
                continue
            scored = sorted(
                (
                    (
                        cluster,
                        float(
                            np.median(
                                [item[0].snr * item[1] for item in cluster]
                            )
                        )
                        * np.sqrt(
                            len({item[0].lead_index for item in cluster})
                        ),
                    )
                    for cluster in supported
                ),
                key=lambda entry: entry[1],
                reverse=True,
            )
            best_cluster, _ = scored[0]
            candidates = [item[0] for item in best_cluster]
            onset = int(round(float(np.median([c.onset for c in candidates]))))
            peak = int(round(float(np.median([c.peak for c in candidates]))))
            offset = int(round(float(np.median([c.offset for c in candidates]))))
            if not onset < peak < offset:
                continue
            if p_reference_axis is not None:
                axis = self._loop_direction(vcg_xyz, onset, offset)
                if axis is None:
                    continue
                p_similarity = abs(float(np.dot(axis, p_reference_axis)))
                t_similarity = (
                    abs(float(np.dot(axis, t_reference_axis)))
                    if t_reference_axis is not None
                    else 0.0
                )
                if (
                    p_similarity - t_similarity < 0.15
                    or p_similarity < 0.5
                ):
                    continue
            marker_kwargs = dict(
                confidence=0.30,
                source="multilead_retrograde_p",
                uncertainty_ms=40.0,
                support_count=len(
                    {item[0].lead_index for item in best_cluster}
                ),
                evidence_sources=("post_qrs_polarity_invariant_search",),
            )
            beat.retrograde_p_onset = MarkerPoint(sample=onset, **marker_kwargs)
            beat.retrograde_p_peak = MarkerPoint(sample=peak, **marker_kwargs)
            beat.retrograde_p_offset = MarkerPoint(sample=offset, **marker_kwargs)
            beat.retrograde_p_status = WaveStatus.PRESENT
            beat.review_reasons += ("retrograde_p_candidate_found",)

    @classmethod
    def _disambiguate_ambiguous_p_candidates(
        cls,
        beats: list[BeatDelineation],
        vcg_xyz: np.ndarray,
        reference_p_axes: list[np.ndarray],
        reference_t_axes: list[np.ndarray],
        ambiguous_p_episodes: dict[int, tuple[tuple[int, int, int], ...]],
        fs: int,
    ) -> None:
        """Second pass: resolve a subset of 'multiple_atrial_deflections'
        abstentions using the VCG loop's spatial direction rather than
        just timing. A per-record reference P-loop axis is built from
        beats where the ordinary pre-QRS search already found exactly one
        candidate; each ambiguous episode's own loop axis is then scored
        against that reference and against the (usually far more
        abundant) reference T-loop axis. An episode is only accepted when
        it is clearly closer to the P reference than to the T reference
        AND clearly separated from the next-best ambiguous episode --
        otherwise the original conservative NOT_ASSESSABLE abstention is
        left untouched, since misreading a T remnant as a P is worse than
        abstaining.
        """
        p_reference_axis = cls._aggregate_axis(reference_p_axes)
        if p_reference_axis is None:
            return
        t_reference_axis = cls._aggregate_axis(reference_t_axes)
        for beat_index, episode_windows in ambiguous_p_episodes.items():
            if len(episode_windows) < 2:
                continue
            scored: list[tuple[int, int, int, float, float]] = []
            for onset_sample, peak_sample, offset_sample in episode_windows:
                # Defensive: onset/peak/offset are each an independent
                # per-episode median across leads/candidates and are not
                # otherwise guaranteed self-consistent (unlike the single
                # accepted cluster's markers, which go through
                # _supported_wave_edge/_weighted_quantile). A disordered
                # triple here would produce a nonsensical P duration if
                # it were ever assigned to the beat.
                if not onset_sample < peak_sample < offset_sample:
                    continue
                # Same P-duration envelope _lead_wave_candidates enforces
                # on every individual per-lead candidate (22-260 ms) --
                # the episode's cross-lead median onset/offset is not
                # automatically bound by that check, so it is re-applied
                # here before the pair can ever reach the beat.
                duration_ms = (offset_sample - onset_sample) / fs * 1000.0
                if not 22.0 <= duration_ms <= 260.0:
                    continue
                axis = cls._loop_direction(vcg_xyz, onset_sample, offset_sample)
                if axis is None:
                    continue
                p_similarity = abs(float(np.dot(axis, p_reference_axis)))
                t_similarity = (
                    abs(float(np.dot(axis, t_reference_axis)))
                    if t_reference_axis is not None
                    else 0.0
                )
                scored.append(
                    (onset_sample, peak_sample, offset_sample, p_similarity, t_similarity)
                )
            if len(scored) < 2:
                continue
            scored.sort(key=lambda item: item[3], reverse=True)
            best_onset, best_peak, best_offset, best_p_sim, best_t_sim = scored[0]
            second_p_sim = scored[1][3]
            clearly_p_like = (best_p_sim - best_t_sim) >= 0.15
            clearly_separated = (best_p_sim - second_p_sim) >= 0.10
            confidently_aligned = best_p_sim >= 0.5
            if not (clearly_p_like and clearly_separated and confidently_aligned):
                continue
            beat = beats[beat_index]
            marker_kwargs = dict(
                confidence=0.35,
                source="multilead_p_vcg_direction_disambiguated",
                uncertainty_ms=40.0,
                evidence_sources=("vcg_loop_direction_similarity",),
            )
            beat.p_onset = MarkerPoint(sample=best_onset, **marker_kwargs)
            beat.p_peak = MarkerPoint(sample=best_peak, **marker_kwargs)
            beat.p_offset = MarkerPoint(sample=best_offset, **marker_kwargs)
            beat.p_status = WaveStatus.PRESENT
            beat.review_reasons += (
                "vcg_direction_disambiguated_from_multiple_atrial_deflections",
            )

    @staticmethod
    def _loop_direction(
        vcg_xyz: Optional[np.ndarray],
        start: int,
        end: int,
    ) -> Optional[np.ndarray]:
        """Thin delegate to the shared core.vcg_transform.loop_dominant_axis
        -- kept as a method so existing self._loop_direction(...) /
        cls._loop_direction(...) call sites in this file don't need to
        change. See that function's docstring for the AXIS-not-direction
        sign caveat. Also used by advanced_vcg_biomarkers.py directly via
        the shared function, so both consumers share one implementation.
        """
        return loop_dominant_axis(vcg_xyz, start, end)

    @staticmethod
    def _aggregate_axis(axes: list[np.ndarray]) -> Optional[np.ndarray]:
        """Thin delegate to core.vcg_transform.aggregate_loop_axes."""
        return aggregate_loop_axes(axes)

    @staticmethod
    def _beat_limits(
        peaks: np.ndarray,
        index: int,
        n_samples: int,
    ) -> tuple[int, int]:
        current = int(peaks[index])
        left = (
            int(round((int(peaks[index - 1]) + current) / 2.0))
            if index > 0
            else 0
        )
        right = (
            int(round((current + int(peaks[index + 1])) / 2.0))
            if index + 1 < len(peaks)
            else n_samples - 1
        )
        return left, right

    def _find_qrs_bounds(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
        *,
        left_limit: int = 0,
        right_limit: Optional[int] = None,
    ) -> Optional[_QRSBounds]:
        right_limit = len(ecg) - 1 if right_limit is None else right_limit
        search_start = max(
            int(left_limit),
            r_peak - int(round(0.22 * fs)),
            0,
        )
        search_end = min(
            int(right_limit),
            r_peak + int(round(0.26 * fs)),
            len(ecg) - 1,
        )
        if search_end - search_start < max(4, int(round(0.04 * fs))):
            return None

        good_indices, normalized = self._normalized_leads(ecg)
        if normalized.size == 0:
            return None
        derivative = np.gradient(normalized, axis=0)
        activity = np.sqrt(np.median(derivative**2, axis=1))
        activity = gaussian_filter1d(
            activity,
            sigma=max(0.75, 0.004 * fs),
            mode="nearest",
        )
        local = activity[search_start : search_end + 1]
        center_lo = max(search_start, r_peak - int(round(0.06 * fs)))
        center_hi = min(search_end + 1, r_peak + int(round(0.08 * fs)))
        if center_hi <= center_lo:
            return None
        center = center_lo + int(np.argmax(activity[center_lo:center_hi]))
        peak_activity = float(activity[center])

        baseline_parts = []
        before_end = min(
            search_end + 1,
            r_peak - int(round(0.14 * fs)),
        )
        if before_end > search_start:
            baseline_parts.append(activity[search_start:before_end])
        after_start = max(
            search_start,
            r_peak + int(round(0.20 * fs)),
        )
        if search_end + 1 > after_start:
            baseline_parts.append(activity[after_start : search_end + 1])
        baseline_values = (
            np.concatenate(baseline_parts)
            if baseline_parts
            else local
        )
        baseline = float(np.median(baseline_values))
        noise = max(1e-6, 1.4826 * _mad(baseline_values))
        if peak_activity <= baseline + 2.0 * noise:
            return None
        threshold = max(
            baseline + 2.5 * noise,
            baseline + 0.075 * (peak_activity - baseline),
        )
        low_threshold = max(
            baseline + 1.5 * noise,
            baseline + 0.035 * (peak_activity - baseline),
        )

        active = local > threshold
        closing_samples = max(1, int(round(0.020 * fs)))
        active = binary_closing(
            active,
            structure=np.ones(closing_samples, dtype=bool),
        )
        center_local = center - search_start
        groups = self._boolean_groups(active)
        if not groups:
            return None
        connected = [
            group
            for group in groups
            if group[0] <= center_local <= group[1]
        ]
        if connected:
            group_start, group_end = connected[0]
        else:
            group_start, group_end = min(
                groups,
                key=lambda group: min(
                    abs(center_local - group[0]),
                    abs(center_local - group[1]),
                ),
            )
        onset = search_start + group_start
        offset = search_start + group_end

        quiet_samples = max(1, int(round(0.008 * fs)))
        onset = self._walk_to_quiet(
            activity,
            onset,
            search_start,
            direction=-1,
            threshold=low_threshold,
            quiet_samples=quiet_samples,
        )
        offset = self._walk_to_quiet(
            activity,
            offset,
            search_end,
            direction=1,
            threshold=low_threshold,
            quiet_samples=quiet_samples,
        )
        lead_bounds = self._lead_specific_qrs_bounds(
            normalized,
            good_indices,
            r_peak,
            search_start,
            search_end,
            fs,
        )
        onset_candidate_leads: tuple[str, ...] = ()
        offset_candidate_leads: tuple[str, ...] = ()
        onset_spread_ms = 0.0
        offset_spread_ms = 0.0
        if len(lead_bounds) >= 3:
            onset_edge, onset_support = self._supported_global_edge(
                lead_bounds,
                fs,
                boundary="onset",
            )
            offset_edge, offset_support = self._supported_global_edge(
                lead_bounds,
                fs,
                boundary="offset",
            )
            maximum_extension = int(round(0.045 * fs))
            if (
                onset_edge is not None
                and onset - maximum_extension <= onset_edge < onset
            ):
                onset = onset_edge
                onset_candidate_leads = self._lead_names(
                    item.lead_index for item in onset_support
                )
                onset_spread_ms = _robust_spread(
                    np.asarray([item.onset for item in onset_support]),
                    fs,
                )
            if (
                offset_edge is not None
                and offset < offset_edge <= offset + maximum_extension
            ):
                offset = offset_edge
                offset_candidate_leads = self._lead_names(
                    item.lead_index for item in offset_support
                )
                offset_spread_ms = _robust_spread(
                    np.asarray([item.offset for item in offset_support]),
                    fs,
                )
        if not onset < r_peak < offset:
            return None
        duration_ms = (offset - onset) / fs * 1000.0
        if not 25.0 <= duration_ms <= 360.0:
            return None

        support_on = self._boundary_lead_support(
            derivative,
            good_indices,
            onset,
            fs,
        )
        support_off = self._boundary_lead_support(
            derivative,
            good_indices,
            offset,
            fs,
        )
        onset_leads = tuple(
            dict.fromkeys(support_on[1] + onset_candidate_leads)
        )
        offset_leads = tuple(
            dict.fromkeys(support_off[1] + offset_candidate_leads)
        )
        onset_support_count = max(
            support_on[0],
            len(onset_candidate_leads),
        )
        offset_support_count = max(
            support_off[0],
            len(offset_candidate_leads),
        )
        snr = (peak_activity - baseline) / noise
        snr_score = _clip01(np.log1p(snr) / np.log(21.0))
        onset_conf = _clip01(
            0.35 * snr_score
            + 0.50 * onset_support_count / max(len(good_indices), 1)
            + 0.15
        )
        offset_conf = _clip01(
            0.35 * snr_score
            + 0.50 * offset_support_count / max(len(good_indices), 1)
            + 0.15
        )
        return _QRSBounds(
            onset=MarkerPoint(
                sample=onset,
                confidence=onset_conf,
                source="multilead_global_boundary",
                uncertainty_ms=max(
                    1000.0 / fs,
                    16.0 * (1.0 - onset_conf),
                    onset_spread_ms,
                ),
                support_count=max(1, onset_support_count),
                lead_names=onset_leads,
                evidence_sources=(
                    "spatial_derivative",
                    "lead_boundary_support",
                    "earliest_supported_lead_boundary",
                ),
            ),
            offset=MarkerPoint(
                sample=offset,
                confidence=offset_conf,
                source="multilead_global_boundary",
                uncertainty_ms=max(
                    1000.0 / fs,
                    18.0 * (1.0 - offset_conf),
                    offset_spread_ms,
                ),
                support_count=max(1, offset_support_count),
                lead_names=offset_leads,
                evidence_sources=(
                    "spatial_derivative",
                    "lead_boundary_support",
                    "latest_supported_lead_boundary",
                ),
            ),
        )

    def _lead_specific_qrs_bounds(
        self,
        normalized: np.ndarray,
        good_indices: list[int],
        r_peak: int,
        search_start: int,
        search_end: int,
        fs: int,
    ) -> list[_LeadQRSBounds]:
        """Find polarity-invariant QRS limits independently in each lead."""

        candidates: list[_LeadQRSBounds] = []
        center_lo = max(search_start, r_peak - int(round(0.07 * fs)))
        center_hi = min(search_end + 1, r_peak + int(round(0.09 * fs)))
        if center_hi <= center_lo:
            return candidates
        closing_samples = max(1, int(round(0.018 * fs)))
        quiet_samples = max(1, int(round(0.006 * fs)))
        for column, lead_index in enumerate(good_indices):
            lead_activity = gaussian_filter1d(
                np.abs(np.gradient(normalized[:, column])),
                sigma=max(0.60, 0.0025 * fs),
                mode="nearest",
            )
            baseline_parts = []
            before_end = min(
                search_end + 1,
                r_peak - int(round(0.14 * fs)),
            )
            if before_end > search_start:
                baseline_parts.append(
                    lead_activity[search_start:before_end]
                )
            after_start = max(
                search_start,
                r_peak + int(round(0.20 * fs)),
            )
            if search_end + 1 > after_start:
                baseline_parts.append(
                    lead_activity[after_start : search_end + 1]
                )
            if not baseline_parts:
                continue
            baseline_values = np.concatenate(baseline_parts)
            baseline = float(np.median(baseline_values))
            noise = max(1e-6, 1.4826 * _mad(baseline_values))
            center = center_lo + int(
                np.argmax(lead_activity[center_lo:center_hi])
            )
            peak_activity = float(lead_activity[center])
            if peak_activity <= baseline + 3.0 * noise:
                continue
            high_threshold = max(
                baseline + 2.5 * noise,
                baseline + 0.065 * (peak_activity - baseline),
            )
            low_threshold = max(
                baseline + 1.25 * noise,
                baseline + 0.020 * (peak_activity - baseline),
            )
            local = lead_activity[search_start : search_end + 1]
            active = binary_closing(
                local > high_threshold,
                structure=np.ones(closing_samples, dtype=bool),
            )
            center_local = center - search_start
            groups = [
                group
                for group in self._boolean_groups(active)
                if group[0] <= center_local <= group[1]
            ]
            if not groups:
                continue
            onset = search_start + groups[0][0]
            offset = search_start + groups[0][1]
            onset = self._walk_to_quiet(
                lead_activity,
                onset,
                search_start,
                direction=-1,
                threshold=low_threshold,
                quiet_samples=quiet_samples,
            )
            offset = self._walk_to_quiet(
                lead_activity,
                offset,
                search_end,
                direction=1,
                threshold=low_threshold,
                quiet_samples=quiet_samples,
            )
            duration_ms = (offset - onset) / fs * 1000.0
            if (
                not onset < r_peak < offset
                or not 25.0 <= duration_ms <= 360.0
            ):
                continue
            candidates.append(
                _LeadQRSBounds(
                    onset=onset,
                    offset=offset,
                    snr=(peak_activity - baseline) / noise,
                    lead_index=lead_index,
                )
            )
        return candidates

    @staticmethod
    def _supported_global_edge(
        candidates: list[_LeadQRSBounds],
        fs: int,
        *,
        boundary: str,
    ) -> tuple[Optional[int], tuple[_LeadQRSBounds, ...]]:
        """Select the earliest/latest boundary cluster supported by two leads."""

        tolerance = max(1, int(round(0.014 * fs)))
        ordered = sorted(
            candidates,
            key=lambda item: getattr(item, boundary),
            reverse=boundary == "offset",
        )
        for candidate in ordered:
            sample = getattr(candidate, boundary)
            support = tuple(
                item
                for item in candidates
                if abs(getattr(item, boundary) - sample) <= tolerance
            )
            if len(support) >= 2:
                values = np.asarray(
                    [getattr(item, boundary) for item in support],
                    dtype=float,
                )
                selected = int(
                    np.min(values)
                    if boundary == "onset"
                    else np.max(values)
                )
                return selected, support
        return None, ()

    @staticmethod
    def _lead_names(indices) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                LEAD_NAMES_12[index]
                if index < len(LEAD_NAMES_12)
                else f"lead_{index}"
                for index in indices
            )
        )

    @staticmethod
    def _walk_to_quiet(
        activity: np.ndarray,
        position: int,
        limit: int,
        *,
        direction: int,
        threshold: float,
        quiet_samples: int,
    ) -> int:
        index = int(position)
        if direction < 0:
            while index > limit:
                lo = max(limit, index - quiet_samples + 1)
                if np.all(activity[lo : index + 1] <= threshold):
                    return index
                index -= 1
            return limit
        while index < limit:
            hi = min(limit + 1, index + quiet_samples)
            if np.all(activity[index:hi] <= threshold):
                return index
            index += 1
        return limit

    def _find_wave_consensus(
        self,
        ecg: np.ndarray,
        start: int,
        end: int,
        fs: int,
        *,
        wave: str,
        reference_sample: int,
        extra_candidates: tuple[tuple[_WaveCandidate, float], ...] = (),
    ) -> Optional[_WaveConsensus]:
        if end - start < max(4, int(round(0.035 * fs))):
            return None
        preferred = P_WAVE_PREFERRED if wave == "p" else T_WAVE_PREFERRED
        candidates = []
        for lead_index, preference in preferred.items():
            if lead_index >= ecg.shape[1]:
                continue
            candidate = self._lead_wave_candidates(
                ecg[:, lead_index],
                start,
                end,
                fs,
                wave=wave,
                reference_sample=reference_sample,
            )
            candidates.extend(
                (
                    _WaveCandidate(
                        onset=item.onset,
                        peak=item.peak,
                        offset=item.offset,
                        snr=item.snr,
                        lead_index=lead_index,
                    ),
                    preference,
                )
                for item in candidate
            )
        # H4: the QRS-orthogonal spatial residual channel, if the caller
        # computed one, participates in clustering/scoring exactly like a
        # real lead -- including the >=2-distinct-lead-index support gate
        # below, so it can only ever reinforce a candidate a real lead
        # also found, never invent one alone.
        candidates.extend(extra_candidates)
        if not candidates:
            return None

        tolerance = int(round((0.045 if wave == "p" else 0.060) * fs))
        clusters = self._cluster_wave_candidates(
            candidates,
            max(1, tolerance),
        )
        supported = [
            cluster
            for cluster in clusters
            if len({item[0].lead_index for item in cluster}) >= 2
        ]
        if not supported:
            return None

        ranked = sorted(
            supported,
            key=lambda cluster: self._wave_cluster_score(
                cluster,
                wave=wave,
                reference_sample=reference_sample,
                fs=fs,
            ),
            reverse=True,
        )
        physiologic_clusters = [
            cluster
            for cluster in ranked
            if self._wave_cluster_score(
                cluster,
                wave=wave,
                reference_sample=reference_sample,
                fs=fs,
            ) >= 2.0
        ]
        if not physiologic_clusters:
            return None

        episode_windows: tuple[tuple[int, int, int], ...] = ()
        if wave == "p":
            relevant = [
                cluster
                for cluster in physiologic_clusters
                if abs(
                    float(np.median([item[0].peak for item in cluster]))
                    - reference_sample
                )
                <= 0.42 * fs
            ]
            episodes = self._merge_atrial_wave_episodes(
                relevant or physiologic_clusters,
                fs,
            )
            episode_scores = [
                self._wave_cluster_score(
                    episode,
                    wave=wave,
                    reference_sample=reference_sample,
                    fs=fs,
                )
                for episode in episodes
            ]
            best_episode_score = max(episode_scores)
            credible_episodes = [
                episode
                for episode, score in zip(episodes, episode_scores)
                if (
                    len({item[0].lead_index for item in episode}) >= 3
                    and score >= max(4.0, 0.30 * best_episode_score)
                )
            ]
            candidate_count = max(1, len(credible_episodes))
            episode_windows = (
                tuple(
                    (
                        int(round(float(
                            np.median([item[0].onset for item in episode])
                        ))),
                        int(round(float(
                            np.median([item[0].peak for item in episode])
                        ))),
                        int(round(float(
                            np.median([item[0].offset for item in episode])
                        ))),
                    )
                    for episode in credible_episodes
                )
                if len(credible_episodes) > 1
                else ()
            )
            chosen = min(
                episodes,
                key=lambda episode: abs(
                    float(np.median([item[0].peak for item in episode]))
                    - (reference_sample - 0.16 * fs)
                ),
            )
        else:
            candidate_count = 1
            scored_clusters = [
                (
                    cluster,
                    self._wave_cluster_score(
                        cluster,
                        wave=wave,
                        reference_sample=reference_sample,
                        fs=fs,
                    ),
                )
                for cluster in physiologic_clusters
            ]
            best_score = max(score for _, score in scored_clusters)
            strong_clusters = [
                cluster
                for cluster, score in scored_clusters
                if score >= max(2.0, 0.45 * best_score)
            ]
            chosen = min(
                strong_clusters,
                key=lambda cluster: float(
                    np.median([item[0].peak for item in cluster])
                ),
            )

        wave_candidates = [item[0] for item in chosen]
        weights = np.asarray(
            [
                max(0.1, item[1]) * max(1.0, item[0].snr)
                for item in chosen
            ],
            dtype=float,
        )
        onset = self._weighted_quantile(
            np.asarray([item.onset for item in wave_candidates], dtype=float),
            weights,
            0.20,
        )
        peak = self._weighted_quantile(
            np.asarray([item.peak for item in wave_candidates], dtype=float),
            weights,
            0.50,
        )
        offset = self._weighted_quantile(
            np.asarray([item.offset for item in wave_candidates], dtype=float),
            weights,
            0.80,
        )
        onset_support = tuple(wave_candidates)
        offset_support = tuple(wave_candidates)
        supported_onset, supported_onset_candidates = (
            self._supported_wave_edge(
                wave_candidates,
                fs,
                boundary="onset",
                tolerance_ms=22.0,
            )
        )
        supported_offset, supported_offset_candidates = (
            self._supported_wave_edge(
                wave_candidates,
                fs,
                boundary="offset",
                tolerance_ms=22.0,
            )
        )
        if supported_onset is not None:
            onset = float(supported_onset)
            onset_support = supported_onset_candidates
        if supported_offset is not None:
            offset = float(supported_offset)
            offset_support = supported_offset_candidates
        onset_i, peak_i, offset_i = map(
            lambda value: int(round(value)),
            (onset, peak, offset),
        )
        duration_ms = (offset_i - onset_i) / fs * 1000.0
        limits = (25.0, 260.0) if wave == "p" else (45.0, 600.0)
        if (
            not onset_i < peak_i < offset_i
            or not limits[0] <= duration_ms <= limits[1]
        ):
            return None

        lead_indices = tuple(
            dict.fromkeys(item.lead_index for item in wave_candidates)
        )
        snr = float(np.median([item.snr for item in wave_candidates]))
        support_score = min(1.0, len(lead_indices) / 4.0)
        snr_score = _clip01(np.log1p(snr) / np.log(11.0))
        peak_dispersion_ms = (
            1.4826
            * _mad(np.asarray([item.peak for item in wave_candidates]))
            / fs
            * 1000.0
        )
        tolerance_ms = 30.0 if wave == "p" else 45.0
        stability = float(np.exp(-peak_dispersion_ms / tolerance_ms))
        confidence = _clip01(
            0.45 * support_score
            + 0.35 * snr_score
            + 0.20 * stability
        )
        source = f"multilead_{wave}_consensus"

        def marker(
            sample: int,
            boundary_extra_ms: float,
            support_candidates: tuple[_WaveCandidate, ...],
            boundary: str,
            boundary_evidence: Optional[str] = None,
        ) -> MarkerPoint:
            boundary_indices = tuple(
                dict.fromkeys(
                    item.lead_index for item in support_candidates
                )
            )
            boundary_leads = tuple(
                LEAD_NAMES_12[index]
                if index < len(LEAD_NAMES_12)
                else f"lead_{index}"
                for index in boundary_indices
            )
            boundary_values = np.asarray(
                [
                    getattr(item, boundary)
                    for item in support_candidates
                ],
                dtype=float,
            )
            boundary_spread_ms = _robust_spread(boundary_values, fs)
            evidence = (
                "polarity_invariant_amplitude",
                "multilead_timing_cluster",
                "local_noise_threshold",
            )
            if boundary_evidence is not None:
                evidence += (boundary_evidence,)
            return MarkerPoint(
                sample=sample,
                confidence=confidence,
                source=source,
                uncertainty_ms=max(
                    1000.0 / fs,
                    peak_dispersion_ms + boundary_extra_ms * (1.0 - confidence),
                    boundary_spread_ms,
                ),
                support_count=len(boundary_indices),
                lead_names=boundary_leads,
                evidence_sources=evidence,
            )

        return _WaveConsensus(
            onset=marker(
                onset_i,
                20.0 if wave == "p" else 30.0,
                onset_support,
                "onset",
                (
                    "earliest_supported_lead_boundary"
                    if supported_onset is not None
                    else None
                ),
            ),
            peak=marker(
                peak_i,
                12.0 if wave == "p" else 20.0,
                tuple(wave_candidates),
                "peak",
            ),
            offset=marker(
                offset_i,
                24.0 if wave == "p" else 38.0,
                offset_support,
                "offset",
                (
                    "latest_supported_lead_boundary"
                    if supported_offset is not None
                    else None
                ),
            ),
            candidate_count=max(1, candidate_count),
            episode_windows=episode_windows,
        )

    @staticmethod
    def _merge_atrial_wave_episodes(
        clusters: list[list[tuple[_WaveCandidate, float]]],
        fs: int,
    ) -> list[list[tuple[_WaveCandidate, float]]]:
        """Join polarity lobes belonging to one P wave, not separate cycles."""

        ordered = sorted(
            clusters,
            key=lambda cluster: float(
                np.median([item[0].peak for item in cluster])
            ),
        )
        episodes: list[list[tuple[_WaveCandidate, float]]] = []
        maximum_peak_gap = int(round(0.115 * fs))
        maximum_boundary_gap = int(round(0.025 * fs))
        for cluster in ordered:
            if not episodes:
                episodes.append(list(cluster))
                continue
            previous = episodes[-1]
            previous_peak = float(
                np.median([item[0].peak for item in previous])
            )
            current_peak = float(
                np.median([item[0].peak for item in cluster])
            )
            previous_offset = int(
                np.percentile([item[0].offset for item in previous], 80)
            )
            current_onset = int(
                np.percentile([item[0].onset for item in cluster], 20)
            )
            same_wave = (
                current_peak - previous_peak <= maximum_peak_gap
                or current_onset - previous_offset <= maximum_boundary_gap
            )
            if same_wave:
                previous.extend(cluster)
            else:
                episodes.append(list(cluster))
        return episodes

    def _spatial_orthogonal_candidates(
        self,
        vcg_xyz: np.ndarray,
        qrs_axis: np.ndarray,
        start: int,
        end: int,
        fs: int,
        *,
        wave: str,
        reference_sample: int,
    ) -> tuple[tuple[_WaveCandidate, float], ...]:
        """H4: candidates from the derived QRS-orthogonal spatial residual
        channel (see orthogonal_residual_magnitude in core.vcg_transform),
        reusing the exact same per-lead candidate finder as a real ECG
        lead column so it goes through identical onset/offset/SNR logic.
        Projects the WHOLE record once per call (cheap: one Nx3 matrix-
        vector product) rather than a locally-sliced window, so the
        returned onset/peak/offset samples are already in the same
        absolute sample coordinates as every other candidate -- avoiding
        the class of local-window index-shifting bugs H1 hit earlier in
        this same file.
        """
        residual = orthogonal_residual_magnitude(vcg_xyz, qrs_axis)
        if residual is None:
            return ()
        found = self._lead_wave_candidates(
            residual,
            start,
            end,
            fs,
            wave=wave,
            reference_sample=reference_sample,
        )
        return tuple(
            (
                _WaveCandidate(
                    onset=item.onset,
                    peak=item.peak,
                    offset=item.offset,
                    snr=item.snr,
                    lead_index=SPATIAL_ORTHOGONAL_LEAD_INDEX,
                ),
                SPATIAL_ORTHOGONAL_P_PREFERENCE,
            )
            for item in found
        )

    def _lead_wave_candidates(
        self,
        lead: np.ndarray,
        start: int,
        end: int,
        fs: int,
        *,
        wave: str,
        reference_sample: int,
    ) -> list[_WaveCandidate]:
        segment = np.asarray(lead[start:end], dtype=float)
        if segment.size < max(6, int(round(0.035 * fs))):
            return []
        sigma = max(0.75, (0.008 if wave == "p" else 0.014) * fs)
        smooth = gaussian_filter1d(segment, sigma=sigma, mode="nearest")
        edge = max(2, min(len(smooth) // 5, int(round(0.025 * fs))))
        x = np.arange(len(smooth), dtype=float)
        left_level = float(np.median(smooth[:edge]))
        right_level = float(np.median(smooth[-edge:]))
        baseline = left_level + (right_level - left_level) * (
            x / max(len(x) - 1, 1)
        )
        residual = smooth - baseline
        onset_amplitude = np.abs(smooth - left_level)
        offset_amplitude = np.abs(smooth - right_level)
        edge_noise = np.concatenate((residual[:edge], residual[-edge:]))
        noise = max(
            0.002,
            1.4826 * _mad(edge_noise),
            0.5 * 1.4826 * _mad(np.diff(smooth)),
        )
        absolute = np.abs(residual)
        minimum_prominence = max(
            2.2 * noise,
            0.012 if wave == "p" else 0.025,
        )
        peaks, properties = find_peaks(
            absolute,
            prominence=minimum_prominence,
            distance=max(1, int(round((0.045 if wave == "p" else 0.080) * fs))),
        )
        candidates: list[_WaveCandidate] = []
        for position, prominence in zip(
            peaks,
            properties.get("prominences", np.zeros(len(peaks))),
        ):
            peak_sample = start + int(position)
            latency_ms = (
                (reference_sample - peak_sample) / fs * 1000.0
                if wave == "p"
                else (peak_sample - reference_sample) / fs * 1000.0
            )
            latency_limits = (35.0, 430.0) if wave == "p" else (20.0, 610.0)
            if not latency_limits[0] <= latency_ms <= latency_limits[1]:
                continue
            amplitude = float(absolute[position])
            snr = max(amplitude, float(prominence)) / noise
            if snr < (2.4 if wave == "p" else 2.2):
                continue
            onset_threshold = max(
                1.35 * noise,
                float(onset_amplitude[position])
                * (0.12 if wave == "p" else 0.10),
            )
            offset_threshold = max(
                1.35 * noise,
                float(offset_amplitude[position])
                * (0.12 if wave == "p" else 0.10),
            )
            onset_local = self._wave_boundary(
                onset_amplitude,
                int(position),
                direction=-1,
                threshold=onset_threshold,
                fs=fs,
            )
            offset_local = self._wave_boundary(
                offset_amplitude,
                int(position),
                direction=1,
                threshold=offset_threshold,
                fs=fs,
            )
            if wave == "t":
                tangent_offset = self._tangent_offset(
                    smooth,
                    int(position),
                    right_level,
                    fs,
                )
                if tangent_offset is not None:
                    offset_local = tangent_offset
            duration_ms = (offset_local - onset_local) / fs * 1000.0
            limits = (22.0, 260.0) if wave == "p" else (40.0, 600.0)
            if not limits[0] <= duration_ms <= limits[1]:
                continue
            candidates.append(
                _WaveCandidate(
                    onset=start + onset_local,
                    peak=peak_sample,
                    offset=start + offset_local,
                    snr=snr,
                    lead_index=-1,
                )
            )

        # The caller knows the lead index; keeping this method reusable makes
        # edge-window tests deterministic.
        return candidates

    @staticmethod
    def _tangent_offset(
        smooth: np.ndarray,
        peak: int,
        terminal_baseline: float,
        fs: int,
    ) -> Optional[int]:
        derivative = np.gradient(np.asarray(smooth, dtype=float))
        peak_value = float(smooth[peak] - terminal_baseline)
        if abs(peak_value) < 1e-9:
            return None
        start = peak + max(1, int(round(0.008 * fs)))
        stop = min(
            len(smooth),
            peak + max(2, int(round(0.220 * fs))),
        )
        if stop - start < 2:
            return None
        local = derivative[start:stop]
        slope_index = start + int(
            np.argmin(local) if peak_value > 0 else np.argmax(local)
        )
        slope = float(derivative[slope_index])
        if abs(slope) < 1e-9:
            return None
        intersection = slope_index + (
            terminal_baseline - float(smooth[slope_index])
        ) / slope
        lower = peak + int(round(0.012 * fs))
        upper = min(
            len(smooth) - 1,
            peak + int(round(0.320 * fs)),
        )
        if not lower <= intersection <= upper:
            return None
        return int(round(intersection))

    def _cluster_wave_candidates(
        self,
        weighted_candidates: list[tuple[_WaveCandidate, float]],
        tolerance_samples: int,
    ) -> list[list[tuple[_WaveCandidate, float]]]:
        ordered = sorted(weighted_candidates, key=lambda item: item[0].peak)
        clusters: list[list[tuple[_WaveCandidate, float]]] = []
        for item in ordered:
            if (
                not clusters
                or item[0].peak
                - int(np.median([entry[0].peak for entry in clusters[-1]]))
                > tolerance_samples
            ):
                clusters.append([item])
            else:
                clusters[-1].append(item)
        return clusters

    @staticmethod
    def _supported_wave_edge(
        candidates: list[_WaveCandidate],
        fs: int,
        *,
        boundary: str,
        tolerance_ms: float,
    ) -> tuple[Optional[int], tuple[_WaveCandidate, ...]]:
        """Return a global wave edge corroborated by independent leads."""

        if boundary not in {"onset", "offset"}:
            raise ValueError(f"Unsupported wave boundary: {boundary}")
        tolerance = max(
            1,
            int(round(float(tolerance_ms) / 1000.0 * fs)),
        )
        ordered = sorted(
            candidates,
            key=lambda item: getattr(item, boundary),
            reverse=boundary == "offset",
        )
        for candidate in ordered:
            sample = getattr(candidate, boundary)
            nearby = sorted(
                (
                    item
                    for item in candidates
                    if abs(getattr(item, boundary) - sample) <= tolerance
                ),
                key=lambda item: (
                    item.lead_index,
                    abs(getattr(item, boundary) - sample),
                    -item.snr,
                ),
            )
            by_lead: dict[int, _WaveCandidate] = {}
            for item in nearby:
                by_lead.setdefault(item.lead_index, item)
            support = tuple(by_lead.values())
            if len(support) < 2:
                continue
            values = np.asarray(
                [getattr(item, boundary) for item in support],
                dtype=float,
            )
            selected = int(
                np.min(values)
                if boundary == "onset"
                else np.max(values)
            )
            return selected, support
        return None, ()

    @staticmethod
    def _wave_cluster_score(
        cluster: list[tuple[_WaveCandidate, float]],
        *,
        wave: str,
        reference_sample: int,
        fs: int,
    ) -> float:
        candidates = [item[0] for item in cluster]
        support = len({item.lead_index for item in candidates})
        peak = float(np.median([item.peak for item in candidates]))
        latency = (
            reference_sample - peak
            if wave == "p"
            else peak - reference_sample
        ) / fs
        expected = 0.16 if wave == "p" else 0.24
        temporal = float(np.exp(-abs(latency - expected) / 0.25))
        weighted_snr = float(
            np.median([item[0].snr * item[1] for item in cluster])
        )
        return weighted_snr * np.sqrt(max(support, 1)) * (0.6 + 0.4 * temporal)

    @staticmethod
    def _wave_boundary(
        amplitude: np.ndarray,
        peak: int,
        *,
        direction: int,
        threshold: float,
        fs: int,
    ) -> int:
        quiet = max(1, int(round(0.010 * fs)))
        index = int(peak)
        limit = 0 if direction < 0 else len(amplitude) - 1
        while index != limit:
            if direction < 0:
                lo = max(0, index - quiet + 1)
                window = amplitude[lo : index + 1]
                if (
                    window.size
                    and np.mean(window <= 1.10 * threshold) >= 0.75
                    and np.median(window) <= threshold
                ):
                    return index
            else:
                hi = min(len(amplitude), index + quiet)
                window = amplitude[index:hi]
                if (
                    window.size
                    and np.mean(window <= 1.10 * threshold) >= 0.75
                    and np.median(window) <= threshold
                ):
                    return index
            index += direction
        return limit

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        order = np.argsort(values)
        ordered_values = values[order]
        ordered_weights = weights[order]
        cumulative = np.cumsum(ordered_weights)
        if cumulative[-1] <= 0:
            return float(np.quantile(values, quantile))
        target = quantile * cumulative[-1]
        return float(ordered_values[np.searchsorted(cumulative, target)])

    @staticmethod
    def _normalized_leads(ecg: np.ndarray) -> tuple[list[int], np.ndarray]:
        centered = np.asarray(ecg, dtype=float) - np.median(
            ecg,
            axis=0,
            keepdims=True,
        )
        scale = np.percentile(np.abs(centered), 95, axis=0)
        finite = np.all(np.isfinite(centered), axis=0)
        good = finite & (scale >= 0.015) & (scale <= 12.0)
        indices = np.where(good)[0].tolist()
        if not indices:
            return [], np.empty((len(ecg), 0), dtype=float)
        normalized = centered[:, good] / np.maximum(scale[good], 0.02)
        return indices, normalized

    @staticmethod
    def _boundary_lead_support(
        normalized_derivative: np.ndarray,
        good_indices: list[int],
        sample: int,
        fs: int,
    ) -> tuple[int, tuple[str, ...]]:
        radius = max(1, int(round(0.010 * fs)))
        lo = max(0, sample - radius)
        hi = min(len(normalized_derivative), sample + radius + 1)
        supported: list[str] = []
        for column, lead_index in enumerate(good_indices):
            derivative = np.abs(normalized_derivative[:, column])
            local = float(np.max(derivative[lo:hi])) if hi > lo else 0.0
            noise = max(1e-6, 1.4826 * _mad(derivative))
            if local >= 1.8 * noise:
                supported.append(
                    LEAD_NAMES_12[lead_index]
                    if lead_index < len(LEAD_NAMES_12)
                    else f"lead_{lead_index}"
                )
        return len(supported), tuple(supported)

    def _find_q_peak_marker(
        self,
        ecg: np.ndarray,
        r_peak: int,
        onset: int,
        fs: int,
    ) -> Optional[MarkerPoint]:
        candidates = []
        for lead_index, weight in QRS_PREFERRED.items():
            if lead_index >= ecg.shape[1]:
                continue
            segment = ecg[onset:r_peak, lead_index]
            if segment.size < 2:
                continue
            baseline = float(np.median(ecg[max(0, onset - int(0.03 * fs)) : onset + 1, lead_index]))
            local = segment - baseline
            index = int(np.argmin(local))
            amplitude = float(-local[index])
            if amplitude >= 0.015:
                candidates.append((onset + index, amplitude * weight, lead_index))
        return self._extremum_consensus(candidates, fs, "q_peak")

    def _find_s_peak_marker(
        self,
        ecg: np.ndarray,
        r_peak: int,
        offset: int,
        fs: int,
    ) -> Optional[MarkerPoint]:
        candidates = []
        for lead_index, weight in QRS_PREFERRED.items():
            if lead_index >= ecg.shape[1]:
                continue
            segment = ecg[r_peak : offset + 1, lead_index]
            if segment.size < 2:
                continue
            baseline = float(np.median(ecg[offset : min(len(ecg), offset + int(0.03 * fs) + 1), lead_index]))
            local = segment - baseline
            index = int(np.argmin(local))
            amplitude = float(-local[index])
            if amplitude >= 0.015:
                candidates.append((r_peak + index, amplitude * weight, lead_index))
        return self._extremum_consensus(candidates, fs, "s_peak")

    @staticmethod
    def _extremum_consensus(
        candidates: list[tuple[int, float, int]],
        fs: int,
        source: str,
    ) -> Optional[MarkerPoint]:
        if not candidates:
            return None
        strongest = sorted(candidates, key=lambda item: item[1], reverse=True)[:5]
        samples = np.asarray([item[0] for item in strongest], dtype=float)
        sample = int(round(np.median(samples)))
        dispersion_ms = 1.4826 * _mad(samples) / fs * 1000.0
        confidence = _clip01(
            0.45
            + 0.10 * min(len(strongest), 4)
            - 0.02 * dispersion_ms
        )
        leads = tuple(
            LEAD_NAMES_12[item[2]]
            if item[2] < len(LEAD_NAMES_12)
            else f"lead_{item[2]}"
            for item in strongest
        )
        return MarkerPoint(
            sample=sample,
            confidence=confidence,
            source=f"multilead_{source}",
            uncertainty_ms=max(1000.0 / fs, dispersion_ms),
            support_count=len(strongest),
            lead_names=leads,
            evidence_sources=("polarity_specific_extremum",),
        )

    @staticmethod
    def _boolean_groups(mask: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        splits = np.flatnonzero(np.diff(indices) > 1) + 1
        groups = np.split(indices, splits)
        return [(int(group[0]), int(group[-1])) for group in groups]

    @staticmethod
    def _mark_overlap_and_edge_risk(
        beats: list[BeatDelineation],
        fs: int,
    ) -> None:
        minimum_gap = int(round(0.020 * fs))
        for index, beat in enumerate(beats[:-1]):
            following = beats[index + 1]
            if beat.t_offset is None or following.p_onset is None:
                continue
            gap = following.p_onset.sample - beat.t_offset.sample
            if gap < minimum_gap:
                beat.p_t_fusion_risk = True
                following.p_t_fusion_risk = True
                if gap < 0:
                    beat.t_status = WaveStatus.FUSED
                    following.p_status = WaveStatus.FUSED
                    beat.review_reasons += ("t_overlaps_following_atrial_activity",)
                    following.review_reasons += (
                        "p_overlaps_preceding_repolarization",
                    )

    # Compatibility helpers used by the legacy viewer and focused tests.
    def _find_p_onset_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        qrs = self._find_qrs_bounds(ecg, r_peak, fs)
        end = (
            qrs.onset.sample - int(round(0.018 * fs))
            if qrs is not None
            else r_peak - int(round(0.05 * fs))
        )
        result = self._find_wave_consensus(
            ecg,
            max(0, r_peak - int(round(0.45 * fs))),
            end,
            fs,
            wave="p",
            reference_sample=qrs.onset.sample if qrs is not None else r_peak,
        )
        return result.onset.sample if result is not None else None

    def _find_p_peak_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        result = self._p_compat_consensus(ecg, r_peak, fs)
        return result.peak.sample if result is not None else None

    def _find_p_offset_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        result = self._p_compat_consensus(ecg, r_peak, fs)
        return result.offset.sample if result is not None else None

    def _p_compat_consensus(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[_WaveConsensus]:
        qrs = self._find_qrs_bounds(ecg, r_peak, fs)
        reference = qrs.onset.sample if qrs is not None else r_peak
        return self._find_wave_consensus(
            ecg,
            max(0, r_peak - int(round(0.45 * fs))),
            reference - int(round(0.018 * fs)),
            fs,
            wave="p",
            reference_sample=reference,
        )

    def _find_r_onset(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        bounds = self._find_qrs_bounds(ecg, r_peak, fs)
        return bounds.onset.sample if bounds is not None else None

    def _find_r_offset(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        bounds = self._find_qrs_bounds(ecg, r_peak, fs)
        return bounds.offset.sample if bounds is not None else None

    def _find_q_peak(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        bounds = self._find_qrs_bounds(ecg, r_peak, fs)
        if bounds is None:
            return None
        marker = self._find_q_peak_marker(
            ecg,
            r_peak,
            bounds.onset.sample,
            fs,
        )
        return marker.sample if marker is not None else None

    def _find_s_peak(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        bounds = self._find_qrs_bounds(ecg, r_peak, fs)
        if bounds is None:
            return None
        marker = self._find_s_peak_marker(
            ecg,
            r_peak,
            bounds.offset.sample,
            fs,
        )
        return marker.sample if marker is not None else None

    def _find_j_point(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        return self._find_r_offset(ecg, r_peak, fs)

    def _find_t_onset_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        result = self._t_compat_consensus(ecg, r_peak, fs)
        return result.onset.sample if result is not None else None

    def _find_t_peak_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        result = self._t_compat_consensus(ecg, r_peak, fs)
        return result.peak.sample if result is not None else None

    def _find_t_offset_multilead(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[int]:
        result = self._t_compat_consensus(ecg, r_peak, fs)
        return result.offset.sample if result is not None else None

    def _t_compat_consensus(
        self,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> Optional[_WaveConsensus]:
        qrs = self._find_qrs_bounds(ecg, r_peak, fs)
        reference = (
            qrs.offset.sample
            if qrs is not None
            else r_peak + int(round(0.10 * fs))
        )
        return self._find_wave_consensus(
            ecg,
            reference + int(round(0.012 * fs)),
            min(len(ecg), r_peak + int(round(0.68 * fs))),
            fs,
            wave="t",
            reference_sample=reference,
        )

    @staticmethod
    def _find_gradient_onset(
        gradient: np.ndarray,
        direction: int = 1,
    ) -> Optional[int]:
        del direction
        absolute = np.abs(np.asarray(gradient, dtype=float))
        if absolute.size == 0 or not np.any(np.isfinite(absolute)):
            return None
        maximum = float(np.nanmax(absolute))
        if maximum <= 0:
            return None
        above = np.flatnonzero(absolute > 0.15 * maximum)
        if above.size == 0:
            return None
        first = int(above[0])
        for index in range(first, 0, -1):
            if absolute[index] < 0.075 * maximum:
                return index
        return first

    @staticmethod
    def _temporal_plausibility(
        sample: int,
        r_peak: int,
        fs: int,
        marker_type: str,
    ) -> float:
        expected = {
            "p_onset": 0.20,
            "p_peak": 0.15,
            "p_offset": 0.08,
        }.get(marker_type)
        if expected is None:
            return 1.0
        difference = abs(r_peak - sample - expected * fs)
        return _clip01(1.0 - difference / max(0.18 * fs, 1.0))

    @staticmethod
    def _detect_missing_waves(
        beat: BeatDelineation,
        ecg: np.ndarray,
        r_peak: int,
        fs: int,
    ) -> BeatDelineation:
        del ecg, r_peak, fs
        if beat.p_onset is None and beat.p_peak is None:
            beat.p_status = WaveStatus.INDETERMINATE
        if beat.t_onset is None and beat.t_peak is None:
            beat.t_status = WaveStatus.INDETERMINATE
        return beat
