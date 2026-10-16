from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BiomarkerBase, BiomarkerResult
from ..delineation.base import (
    RecordDelineation,
    measurement_confidence_threshold,
)
from ..vcg_geometry import (
    GEH_FORMULA_VERSION,
    VCG_AXIS_CONVENTION,
    origin_sensitivity_metadata,
    vcg_azimuth_deg,
    vcg_elevation_deg,
    vector_angle_deg,
)


_GEH_REFERENCES = [
    "PhysioNet Global Electrical Heterogeneity v1.0.0: "
    "https://physionet.org/content/geh/1.0.0/",
    "Wilson FN, et al. The determination and significance of the areas "
    "of the ventricular deflections. Am Heart J. 1934;10:46-61.",
]


class _NamedVCGBiomarker(BiomarkerBase):
    biomarker_name = ""
    biomarker_description = ""
    biomarker_unit = ""
    biomarker_references: list[str] = []
    reference_range: tuple[Optional[float], Optional[float]] = (None, None)

    @property
    def name(self) -> str:
        return self.biomarker_name

    @property
    def description(self) -> str:
        return self.biomarker_description

    @property
    def unit(self) -> str:
        return self.biomarker_unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def references(self) -> list[str]:
        return list(self.biomarker_references)

    def normal_range(self, age=None, sex=None):
        return self.reference_range

    def _empty(self, note: str) -> BiomarkerResult:
        return BiomarkerResult(
            name=self.name,
            value=None,
            unit=self.unit,
            category=self.category,
            confidence=0.0,
            computation_notes=note,
        )

    def _value(
        self,
        value: Optional[float],
        *,
        confidence: float,
        note: str,
        metadata: Optional[dict[str, float]] = None,
    ) -> BiomarkerResult:
        if value is None or not np.isfinite(value):
            return self._empty(note)
        return BiomarkerResult(
            name=self.name,
            value=float(value),
            unit=self.unit,
            category=self.category,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            per_lead=metadata,
            computation_notes=note,
        )


def _origin_corrected_context(
    delineation: RecordDelineation,
    vcg_xyz: Optional[np.ndarray],
    fs: int,
    ecg_12lead: Optional[np.ndarray] = None,
):
    if vcg_xyz is None:
        return None, None, None, "VCG unavailable"
    # Reuse the record-scoped FeatureContext so eleven legacy VCG markers do
    # not independently repeat origin estimation and QRST integration.
    from .catalog_biomarkers import _context

    context = _context(delineation, ecg_12lead, vcg_xyz, fs)
    beat = context.beat
    if beat is None:
        return None, None, None, "representative beat unavailable"
    assessment = context.vcg_origin_assessment
    corrected = context.origin_corrected_vcg()
    if corrected is None:
        return beat, assessment, None, assessment.provenance_note()
    return beat, assessment, corrected, assessment.provenance_note()


def _geh_context(
    delineation: RecordDelineation,
    vcg_xyz: Optional[np.ndarray],
    fs: int,
    ecg_12lead: Optional[np.ndarray] = None,
):
    beat, assessment, corrected, note = _origin_corrected_context(
        delineation, vcg_xyz, fs, ecg_12lead
    )
    if corrected is None or beat is None or assessment is None:
        return None, assessment, note
    if delineation.segment_confidence("qrst") < measurement_confidence_threshold("qrst"):
        return None, assessment, "QRST boundaries not evaluable; " + note
    if beat.r_onset is None or beat.r_offset is None or beat.t_offset is None:
        return None, assessment, "QRS onset/offset or T offset unavailable; " + note
    from .catalog_biomarkers import _context

    metrics = _context(
        delineation, ecg_12lead, vcg_xyz, fs
    ).geh_area_metrics()
    if metrics is None:
        return None, assessment, "GEH area metrics not evaluable; " + note
    return metrics, assessment, (
        f"{note}; geh_formula_version={GEH_FORMULA_VERSION}; "
        "QRS=QRS-onset:QRS-offset; T-area=JT=QRS-offset:T-offset; "
        f"axes={VCG_AXIS_CONVENTION}"
    )


def _geh_metadata(metrics, assessment) -> dict[str, float]:
    qrs = metrics.qrs_area_vector_uv_ms
    jt = metrics.jt_area_vector_uv_ms
    svg = metrics.qrst_area_vector_uv_ms
    return {
        "qrs_area_x_uv_ms": float(qrs[0]),
        "qrs_area_y_uv_ms": float(qrs[1]),
        "qrs_area_z_uv_ms": float(qrs[2]),
        "jt_area_x_uv_ms": float(jt[0]),
        "jt_area_y_uv_ms": float(jt[1]),
        "jt_area_z_uv_ms": float(jt[2]),
        "svg_x_uv_ms": float(svg[0]),
        "svg_y_uv_ms": float(svg[1]),
        "svg_z_uv_ms": float(svg[2]),
        **origin_sensitivity_metadata(
            assessment,
            qrst_duration_ms=metrics.qrst_duration_ms,
        ),
    }


def _peak_vector_context(
    delineation: RecordDelineation,
    vcg_xyz: Optional[np.ndarray],
    fs: int,
    wave: str,
    ecg_12lead: Optional[np.ndarray] = None,
):
    beat, assessment, corrected, note = _origin_corrected_context(
        delineation, vcg_xyz, fs, ecg_12lead
    )
    if corrected is None or beat is None or assessment is None:
        return None, assessment, note
    if delineation.segment_confidence(wave) < measurement_confidence_threshold(wave):
        return None, assessment, f"{wave.upper()} boundaries not evaluable; {note}"
    if wave == "qrs":
        left = beat.r_onset
        right = beat.r_offset
    elif wave == "t":
        left = beat.t_onset
        right = beat.t_offset
    else:  # pragma: no cover - internal programming guard
        raise ValueError(f"unsupported wave: {wave}")
    if left is None or right is None:
        return None, assessment, f"{wave.upper()} boundaries unavailable; {note}"
    start, end = int(left.sample), int(right.sample)
    if not 0 <= start <= end < len(corrected):
        return None, assessment, f"{wave.upper()} bounds outside VCG; {note}"
    segment = corrected[start : end + 1]
    if len(segment) == 0:
        return None, assessment, f"empty {wave.upper()} segment; {note}"
    magnitudes = np.linalg.norm(segment, axis=1)
    vector = segment[int(np.argmax(magnitudes))]
    if float(np.linalg.norm(vector)) <= 1e-12:
        return None, assessment, f"{wave.upper()} peak vector is too small; {note}"
    return vector, assessment, f"{note}; axes={VCG_AXIS_CONVENTION}"


class SpatialQRSTAngleMeanBiomarker(_NamedVCGBiomarker):
    biomarker_name = "spatial_qrs_t_angle_mean"
    biomarker_description = (
        "Area-based spatial angle between the QRS and complete JT vectors"
    )
    biomarker_unit = "degrees"
    biomarker_references = _GEH_REFERENCES
    reference_range = (0.0, 105.0)

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        metrics, assessment, note = _geh_context(
            delineation, vcg_xyz, fs, ecg_12lead
        )
        if metrics is None or assessment is None:
            return self._empty(note)
        return self._value(
            metrics.qrs_jt_angle_deg,
            confidence=min(
                0.8,
                delineation.segment_confidence("qrst"),
                assessment.primary.confidence,
            ),
            note=(
                note
                if metrics.qrs_jt_angle_deg is not None
                else "QRS or JT area vector is too small; " + note
            ),
            metadata=_geh_metadata(metrics, assessment),
        )

    def _angle_between_vectors(
        self, v1: np.ndarray, v2: np.ndarray
    ) -> Optional[float]:
        """Compatibility shim for callers of the former local helper."""

        return vector_angle_deg(v1, v2)


class SpatialQRSTAnglePeakBiomarker(_NamedVCGBiomarker):
    biomarker_name = "spatial_qrs_t_angle_peak"
    biomarker_description = (
        "Spatial angle between origin-corrected peak QRS and peak T vectors"
    )
    biomarker_unit = "degrees"
    biomarker_references = [
        "Sur S, Han L, Tereshchenko LG. Spatial QRS-T angle as a biomarker "
        "of cardiac risk. J Electrocardiol. 2014;47:621-627."
    ]
    reference_range = (0.0, 50.0)

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        qrs, qrs_assessment, qrs_note = _peak_vector_context(
            delineation, vcg_xyz, fs, "qrs", ecg_12lead
        )
        t, t_assessment, t_note = _peak_vector_context(
            delineation, vcg_xyz, fs, "t", ecg_12lead
        )
        assessment = qrs_assessment or t_assessment
        if qrs is None or t is None or assessment is None:
            return self._empty(qrs_note if qrs is None else t_note)
        angle = vector_angle_deg(qrs, t)
        return self._value(
            angle,
            confidence=min(
                0.75,
                delineation.segment_confidence("qrst"),
                assessment.primary.confidence,
            ),
            note=(
                qrs_note
                if angle is not None
                else "QRS or T peak vector is too small; " + qrs_note
            ),
            metadata=assessment.numeric_metadata(),
        )


class _PeakVectorMagnitudeBiomarker(_NamedVCGBiomarker):
    wave = ""

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        vector, assessment, note = _peak_vector_context(
            delineation, vcg_xyz, fs, self.wave, ecg_12lead
        )
        if vector is None or assessment is None:
            return self._empty(note)
        return self._value(
            float(np.linalg.norm(vector) * 1000.0),
            confidence=min(
                0.8 if self.wave == "qrs" else 0.75,
                delineation.segment_confidence(self.wave),
                assessment.primary.confidence,
            ),
            note=note,
            metadata=assessment.numeric_metadata(),
        )


class QRSVectorMagnitudeBiomarker(_PeakVectorMagnitudeBiomarker):
    biomarker_name = "qrs_vector_magnitude"
    biomarker_description = "Maximum origin-corrected QRS spatial magnitude"
    biomarker_unit = "uV"
    wave = "qrs"
    reference_range = (800.0, 2500.0)


class TVectorMagnitudeBiomarker(_PeakVectorMagnitudeBiomarker):
    biomarker_name = "t_vector_magnitude"
    biomarker_description = "Maximum origin-corrected T spatial magnitude"
    biomarker_unit = "uV"
    wave = "t"
    reference_range = (200.0, 800.0)


class _SVGBiomarker(_NamedVCGBiomarker):
    component = ""

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        metrics, assessment, note = _geh_context(
            delineation, vcg_xyz, fs, ecg_12lead
        )
        if metrics is None or assessment is None:
            return self._empty(note)
        vector = metrics.qrst_area_vector_uv_ms
        if self.component == "magnitude":
            value = metrics.svg_magnitude_uv_ms
        elif self.component == "azimuth":
            value = vcg_azimuth_deg(vector)
        elif self.component == "elevation":
            value = vcg_elevation_deg(vector)
        else:  # pragma: no cover - internal programming guard
            raise ValueError(f"unknown SVG component: {self.component}")
        return self._value(
            value,
            confidence=min(
                0.7,
                delineation.segment_confidence("qrst"),
                assessment.primary.confidence,
            ),
            note=(
                note
                if value is not None
                else f"SVG vector is too small for {self.component}; {note}"
            ),
            metadata=_geh_metadata(metrics, assessment),
        )


class SVGMagnitudeBiomarker(_SVGBiomarker):
    biomarker_name = "svg_magnitude"
    biomarker_description = (
        "Wilson SVG magnitude from the origin-corrected complete QRST area"
    )
    biomarker_unit = "uV*ms"
    biomarker_references = _GEH_REFERENCES
    component = "magnitude"


class SVGAzimuthBiomarker(_SVGBiomarker):
    biomarker_name = "svg_azimuth"
    biomarker_description = "SVG azimuth in the Frank transverse X-Z plane"
    biomarker_unit = "degrees"
    biomarker_references = _GEH_REFERENCES
    component = "azimuth"


class SVGElevationBiomarker(_SVGBiomarker):
    biomarker_name = "svg_elevation"
    biomarker_description = "SVG elevation measured from the inferior +Y axis"
    biomarker_unit = "degrees"
    biomarker_references = _GEH_REFERENCES
    component = "elevation"


class _PeakVectorDirectionBiomarker(_NamedVCGBiomarker):
    wave = ""
    component = ""

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        vector, assessment, note = _peak_vector_context(
            delineation, vcg_xyz, fs, self.wave, ecg_12lead
        )
        if vector is None or assessment is None:
            return self._empty(note)
        value = (
            vcg_azimuth_deg(vector)
            if self.component == "azimuth"
            else vcg_elevation_deg(vector)
        )
        return self._value(
            value,
            confidence=min(
                0.75 if self.wave == "qrs" else 0.7,
                delineation.segment_confidence(self.wave),
                assessment.primary.confidence,
            ),
            note=(
                note
                if value is not None
                else f"{self.wave.upper()} peak vector is too small; {note}"
            ),
            metadata=assessment.numeric_metadata(),
        )


class QRSMaxVectorAzimuthBiomarker(_PeakVectorDirectionBiomarker):
    biomarker_name = "qrs_max_vector_azimuth"
    biomarker_description = "Azimuth of the origin-corrected maximum QRS vector"
    biomarker_unit = "degrees"
    wave = "qrs"
    component = "azimuth"


class QRSMaxVectorElevationBiomarker(_PeakVectorDirectionBiomarker):
    biomarker_name = "qrs_max_vector_elevation"
    biomarker_description = (
        "Elevation of the origin-corrected maximum QRS vector from +Y"
    )
    biomarker_unit = "degrees"
    wave = "qrs"
    component = "elevation"


class TMaxVectorAzimuthBiomarker(_PeakVectorDirectionBiomarker):
    biomarker_name = "t_max_vector_azimuth"
    biomarker_description = "Azimuth of the origin-corrected maximum T vector"
    biomarker_unit = "degrees"
    wave = "t"
    component = "azimuth"


class TMaxVectorElevationBiomarker(_PeakVectorDirectionBiomarker):
    biomarker_name = "t_max_vector_elevation"
    biomarker_description = (
        "Elevation of the origin-corrected maximum T vector from +Y"
    )
    biomarker_unit = "degrees"
    wave = "t"
    component = "elevation"


def register_vcg_biomarkers():
    from .base import BiomarkerRegistry

    for cls in [
        SpatialQRSTAngleMeanBiomarker,
        SpatialQRSTAnglePeakBiomarker,
        QRSVectorMagnitudeBiomarker,
        TVectorMagnitudeBiomarker,
        SVGMagnitudeBiomarker,
        SVGAzimuthBiomarker,
        SVGElevationBiomarker,
        QRSMaxVectorAzimuthBiomarker,
        QRSMaxVectorElevationBiomarker,
        TMaxVectorAzimuthBiomarker,
        TMaxVectorElevationBiomarker,
    ]:
        BiomarkerRegistry.register(cls())
