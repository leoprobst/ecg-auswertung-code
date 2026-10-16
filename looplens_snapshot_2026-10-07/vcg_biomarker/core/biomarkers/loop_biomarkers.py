from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull

from .base import BiomarkerBase, BiomarkerResult
from ..delineation.base import RecordDelineation
from ..delineation.base import measurement_confidence_threshold


class QRSLoopAreaBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qrs_loop_area"

    @property
    def description(self) -> str:
        return "QRS loop area projected onto frontal, horizontal, and sagittal planes"

    @property
    def unit(self) -> str:
        return "uV^2"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_start = median_beat.r_onset.sample if median_beat.r_onset else None
        qrs_end = median_beat.r_offset.sample if median_beat.r_offset else None

        if (
            qrs_start is None
            or qrs_end is None
            or delineation.segment_confidence("qrs") < measurement_confidence_threshold("qrs")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_loop = vcg_xyz[qrs_start:qrs_end+1, :] * 1000.0

        if len(qrs_loop) < 3:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        frontal_area = self._shoelace_area(qrs_loop[:, 0], qrs_loop[:, 1])
        horizontal_area = self._shoelace_area(qrs_loop[:, 0], qrs_loop[:, 2])
        sagittal_area = self._shoelace_area(qrs_loop[:, 1], qrs_loop[:, 2])

        total_area = abs(frontal_area) + abs(horizontal_area) + abs(sagittal_area)

        return BiomarkerResult(
            name=self.name, value=total_area, unit=self.unit, category=self.category,
            confidence=min(0.7, delineation.segment_confidence("qrs")), per_lead={
                "frontal": abs(frontal_area),
                "horizontal": abs(horizontal_area),
                "sagittal": abs(sagittal_area),
            },
        )

    def _shoelace_area(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 3:
            return 0.0
        return 0.5 * abs(np.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + x[-1]*y[0] - x[0]*y[-1])


class QRSLoopMaxMinRatioBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qrs_loop_max_min_ratio"

    @property
    def description(self) -> str:
        return "Ratio of the major to minor PCA axis of the QRS loop main plane"

    @property
    def unit(self) -> str:
        return "ratio"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_start = median_beat.r_onset.sample if median_beat.r_onset else None
        qrs_end = median_beat.r_offset.sample if median_beat.r_offset else None

        if (
            qrs_start is None
            or qrs_end is None
            or delineation.segment_confidence("qrs") < measurement_confidence_threshold("qrs")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_loop = vcg_xyz[qrs_start:qrs_end+1, :] * 1000.0

        if len(qrs_loop) < 3:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        centered = qrs_loop - np.mean(qrs_loop, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            major_variance = max(float(eigenvalues[0]), 0.0)
            minor_variance = max(float(eigenvalues[1]), 1e-10)
            ratio = np.sqrt(major_variance / minor_variance)
        except np.linalg.LinAlgError:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        return BiomarkerResult(
            name=self.name,
            value=float(ratio),
            unit=self.unit,
            category=self.category,
            confidence=min(0.7, delineation.segment_confidence("qrs")),
        )


class QRSLoopWidthBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "qrs_loop_width"

    @property
    def description(self) -> str:
        return "Maximum width of the QRS loop (perpendicular to the major axis)"

    @property
    def unit(self) -> str:
        return "uV"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_start = median_beat.r_onset.sample if median_beat.r_onset else None
        qrs_end = median_beat.r_offset.sample if median_beat.r_offset else None

        if (
            qrs_start is None
            or qrs_end is None
            or delineation.segment_confidence("qrs") < measurement_confidence_threshold("qrs")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_loop = vcg_xyz[qrs_start:qrs_end+1, :] * 1000.0

        if len(qrs_loop) < 3:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        centered = qrs_loop - np.mean(qrs_loop, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            sorted_idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_idx]
        except np.linalg.LinAlgError:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        transverse_axis = eigenvectors[:, 1]
        transverse_projections = centered @ transverse_axis
        max_width = float(np.max(transverse_projections) - np.min(transverse_projections))

        return BiomarkerResult(
            name=self.name,
            value=max_width,
            unit=self.unit,
            category=self.category,
            confidence=min(0.65, delineation.segment_confidence("qrs")),
        )


class TLoopAreaBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "t_loop_area"

    @property
    def description(self) -> str:
        return "T loop area projected onto frontal, horizontal, and sagittal planes"

    @property
    def unit(self) -> str:
        return "uV^2"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_start = median_beat.t_onset.sample if median_beat.t_onset else None
        t_end = median_beat.t_offset.sample if median_beat.t_offset else None

        if (
            t_start is None
            or t_end is None
            or delineation.segment_confidence("t") < measurement_confidence_threshold("t")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_loop = vcg_xyz[t_start:t_end+1, :] * 1000.0
        if len(t_loop) < 3:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        area_calc = QRSLoopAreaBiomarker()
        frontal_area = area_calc._shoelace_area(t_loop[:, 0], t_loop[:, 1])
        horizontal_area = area_calc._shoelace_area(t_loop[:, 0], t_loop[:, 2])
        sagittal_area = area_calc._shoelace_area(t_loop[:, 1], t_loop[:, 2])

        total_area = abs(frontal_area) + abs(horizontal_area) + abs(sagittal_area)

        return BiomarkerResult(
            name=self.name, value=total_area, unit=self.unit, category=self.category,
            confidence=min(0.65, delineation.segment_confidence("t")), per_lead={
                "frontal": abs(frontal_area),
                "horizontal": abs(horizontal_area),
                "sagittal": abs(sagittal_area),
            },
        )


class TLoopDispersionBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "t_loop_dispersion"

    @property
    def description(self) -> str:
        return "T loop dispersion: number of subdivisions traversed by the T loop (Voulgari method)"

    @property
    def unit(self) -> str:
        return "subdivisions"

    @property
    def category(self) -> str:
        return "loop"

    @property
    def references(self) -> list[str]:
        return [
            "Voulgari C, Tentolouris N. Assessment of the Spatial QRS-T Angle by Vectorcardiography. Cardiology. 2009;117:21-30.",
        ]

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_start = median_beat.t_onset.sample if median_beat.t_onset else None
        t_end = median_beat.t_offset.sample if median_beat.t_offset else None

        if (
            t_start is None
            or t_end is None
            or delineation.segment_confidence("t") < measurement_confidence_threshold("t")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_loop = vcg_xyz[t_start:t_end+1, :]

        frontal = t_loop[:, [0, 1]]

        x_min, x_max = np.min(frontal[:, 0]), np.max(frontal[:, 0])
        y_min, y_max = np.min(frontal[:, 1]), np.max(frontal[:, 1])

        n_grid = 10
        subdivisions = set()

        for point in frontal:
            xi = int((point[0] - x_min) / (x_max - x_min + 1e-10) * (n_grid - 1))
            yi = int((point[1] - y_min) / (y_max - y_min + 1e-10) * (n_grid - 1))
            xi = min(xi, n_grid - 1)
            yi = min(yi, n_grid - 1)
            subdivisions.add((xi, yi))

        dispersion = len(subdivisions)

        return BiomarkerResult(
            name=self.name,
            value=float(dispersion),
            unit=self.unit,
            category=self.category,
            confidence=min(0.6, delineation.segment_confidence("t")),
        )


class TLoopMaxMinRatioBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "t_loop_max_min_ratio"

    @property
    def description(self) -> str:
        return "Ratio of the major to minor PCA axis of the T loop main plane"

    @property
    def unit(self) -> str:
        return "ratio"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_start = median_beat.t_onset.sample if median_beat.t_onset else None
        t_end = median_beat.t_offset.sample if median_beat.t_offset else None

        if (
            t_start is None
            or t_end is None
            or delineation.segment_confidence("t") < measurement_confidence_threshold("t")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_loop = vcg_xyz[t_start:t_end+1, :] * 1000.0

        if len(t_loop) < 3:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        centered = t_loop - np.mean(t_loop, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            major_variance = max(float(eigenvalues[0]), 0.0)
            minor_variance = max(float(eigenvalues[1]), 1e-10)
            ratio = np.sqrt(major_variance / minor_variance)
        except np.linalg.LinAlgError:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        return BiomarkerResult(
            name=self.name,
            value=float(ratio),
            unit=self.unit,
            category=self.category,
            confidence=min(0.65, delineation.segment_confidence("t")),
        )


class SpatialVelocityQRSPeakBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "spatial_velocity_qrs_peak"

    @property
    def description(self) -> str:
        return "Peak spatial velocity during the QRS complex"

    @property
    def unit(self) -> str:
        return "uV/ms"

    @property
    def category(self) -> str:
        return "loop"

    @property
    def references(self) -> list[str]:
        return [
            "Ray S, et al. Spatial velocity of the dynamic vectorcardiographic loop provides crucial insight in ventricular dysfunction. J Physiol. 2020;598(9):1811-23.",
        ]

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_start = median_beat.r_onset.sample if median_beat.r_onset else None
        qrs_end = median_beat.r_offset.sample if median_beat.r_offset else None

        if (
            qrs_start is None
            or qrs_end is None
            or delineation.segment_confidence("qrs") < measurement_confidence_threshold("qrs")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        qrs_segment = vcg_xyz[qrs_start:qrs_end+1, :]
        if len(qrs_segment) < 2:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)
        sv = np.sqrt(np.sum(np.diff(qrs_segment, axis=0)**2, axis=1)) * 1000.0
        sv_per_ms = sv / (1000.0 / fs)

        peak_sv = float(np.max(sv_per_ms))

        return BiomarkerResult(
            name=self.name,
            value=peak_sv,
            unit=self.unit,
            category=self.category,
            confidence=min(0.7, delineation.segment_confidence("qrs")),
        )


class SpatialVelocityTPeakBiomarker(BiomarkerBase):
    @property
    def name(self) -> str:
        return "spatial_velocity_t_peak"

    @property
    def description(self) -> str:
        return "Peak spatial velocity during the T wave"

    @property
    def unit(self) -> str:
        return "uV/ms"

    @property
    def category(self) -> str:
        return "loop"

    def compute(self, delineation: RecordDelineation, ecg_12lead=None, vcg_xyz=None, fs=500) -> BiomarkerResult:
        if vcg_xyz is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        median_beat = delineation.representative_beat
        if median_beat is None:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_start = median_beat.t_onset.sample if median_beat.t_onset else None
        t_end = median_beat.t_offset.sample if median_beat.t_offset else None

        if (
            t_start is None
            or t_end is None
            or delineation.segment_confidence("t") < measurement_confidence_threshold("t")
        ):
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)

        t_segment = vcg_xyz[t_start:t_end+1, :]
        if len(t_segment) < 2:
            return BiomarkerResult(name=self.name, value=None, unit=self.unit, category=self.category, confidence=0.0)
        sv = np.sqrt(np.sum(np.diff(t_segment, axis=0)**2, axis=1)) * 1000.0
        sv_per_ms = sv / (1000.0 / fs)

        peak_sv = float(np.max(sv_per_ms))

        return BiomarkerResult(
            name=self.name,
            value=peak_sv,
            unit=self.unit,
            category=self.category,
            confidence=min(0.65, delineation.segment_confidence("t")),
        )


def register_loop_biomarkers():
    from .base import BiomarkerRegistry
    for cls in [
        QRSLoopAreaBiomarker, QRSLoopMaxMinRatioBiomarker, QRSLoopWidthBiomarker,
        TLoopAreaBiomarker, TLoopDispersionBiomarker, TLoopMaxMinRatioBiomarker,
        SpatialVelocityQRSPeakBiomarker, SpatialVelocityTPeakBiomarker,
    ]:
        BiomarkerRegistry.register(cls())
