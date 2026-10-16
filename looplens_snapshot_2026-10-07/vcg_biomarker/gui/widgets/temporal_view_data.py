"""Strict, GUI-independent validation for the temporal research view."""
import numpy as np

from ...core.delineation.base import measurement_confidence_threshold


def qrs_view_points(vcg, beat, fs):
    if vcg is None or beat is None:
        raise ValueError("VCG oder QRS-Delineation fehlt.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Ungültige Abtastrate.")
    confidence = beat.segment_confidence("qrs")
    if not np.isfinite(confidence) or not measurement_confidence_threshold("qrs") <= confidence <= 1:
        raise ValueError("QRS-Segmentqualität unzureichend.")
    xyz = np.asarray(vcg.xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Ungültiges XYZ-Signal.")
    boundaries = [getattr(getattr(beat, key, None), "sample", None) for key in ("r_onset", "r_offset")]
    if any(value is None or not np.isfinite(value) or int(value) != value for value in boundaries):
        raise ValueError("QRS-Grenzen fehlen oder sind ungültig.")
    start, end = map(int, boundaries)
    if not 0 <= start < end < len(xyz) or end - start < 3:
        raise ValueError("QRS-Grenzen außerhalb des Signals oder Segment zu kurz.")
    points = xyz[start:end + 1]
    if not np.all(np.isfinite(points)):
        raise ValueError("QRS enthält nichtfinite Samples; Zeitlücken werden nicht entfernt.")
    return points.copy()
