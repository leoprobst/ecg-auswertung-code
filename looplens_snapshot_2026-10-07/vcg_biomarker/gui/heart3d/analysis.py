"""Anbindung an die Analysemathematik des Hauptprojekts (Delineation, Marker, klassische Kriterien).

Ohne Laborbefund (Prüf-GUI, WFDB-Datei) wird hier dieselbe Kette wie im ECG-Labor
gerechnet – Signalpfade, Mehrkanal-R-Erkennung, Fusions-Delineation mit Rhythmus-
bewertung, klassische Kriterien und die Thesenmarker – und als laborgleiche Nutzlast
zurückgegeben. Alles ist nur lesend gegen ``core``/``mimic`` gekoppelt; schlägt ein
Schritt fehl, bleibt die leichte Schnellschätzung aus ``timeline.py`` in Kraft.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .timeline import LEADS_12

ANALYSIS_FS = 500
THESIS_FEATURES = (
    "qrs_arc_length_fraction_at_half_time", "qrs_early_cancellation_fraction", "qrs_cancellation_fraction",
    "qrs_stall_direction_leftward", "qrs_stall_direction_inferior", "qrs_stall_direction_posterior",
    "qrs_spatial_velocity_initial_burst_ms", "qrs_spatial_velocity_first_to_last_peak_ms",
    "delta_wave_present_flag", "delta_wave_frontal_axis_deg", "pacing_spike_slope_ratio",
    "primary_morphology_qrs_octant_id", "secondary_morphology_qrs_octant_id",
    "secondary_morphology_maximum_deflection_index",
)


def core_available() -> bool:
    try:
        import vcg_biomarker.core.delineation.fusion_delineator  # noqa: F401
        return True
    except Exception:
        return False


def _reorder_to_standard(signal: np.ndarray, names: Iterable[str]) -> Optional[np.ndarray]:
    names = [str(n) for n in names]
    try:
        cols = [names.index(n) for n in LEADS_12]
    except ValueError:
        return None
    return signal[:, cols]


def core_payload_from_signal(signal: np.ndarray, fs: float, lead_names: Iterable[str] = LEADS_12, *,
                             age: Optional[float] = None, sex: Optional[str] = None,
                             powerline_hz: float = 50.0) -> dict:
    """Laborgleiche Nutzlast (Schlaginventar, Segmente, Rhythmus, Kriterien, Marker) aus einem 12-Kanal-Signal."""
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
    from vcg_biomarker.core.preprocessing import (construct_robust_morphology_record, detect_r_peaks_multilead,
                                                  prepare_signal_paths, resample_signal)
    from vcg_biomarker.core.vcg_transform import VCGTransformRegistry
    from vcg_biomarker.mimic.classical import ClassicalECGCriteria
    from vcg_biomarker.mimic.ecg_lab import _wave_segments_for_beat

    sig = _reorder_to_standard(np.asarray(signal, dtype=float), lead_names)
    if sig is None:
        raise ValueError("Zwölf Standardableitungen erforderlich")
    original_fs = float(fs)
    if not np.isclose(original_fs, ANALYSIS_FS):
        sig = resample_signal(sig, original_fs, ANALYSIS_FS)
    paths = prepare_signal_paths(sig, ANALYSIS_FS, powerline=powerline_hz)
    r_peaks = detect_r_peaks_multilead(paths["delineation"], ANALYSIS_FS)
    if len(r_peaks) < 2:
        raise ValueError("weniger als zwei Schläge erkannt")
    vcg_del = VCGTransformRegistry.transform(paths["delineation"], "kors")
    delineator = FusionDelineator(weights={"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25},
                                  conflict_threshold=20.0, capture_candidates=False)
    delineation = delineator.delineate_record(ecg_12lead=paths["delineation"], r_peaks=r_peaks, fs=ANALYSIS_FS,
                                              ecg_id=0, vcg_xyz=vcg_del.xyz)
    accepted_ids = {id(b) for b in delineation.accepted_beats}
    segments = []
    for k, b in enumerate(delineation.beats):
        for seg in _wave_segments_for_beat(b, ANALYSIS_FS):
            seg["beat_index"] = k
            seg["r_peak_sample"] = int(b.r_peak.sample)
            segments.append(seg)
    payload: dict = {
        "lab_processing": {"analysis_sampling_rate_hz": float(ANALYSIS_FS), "source": "heart3d_core_pipeline"},
        "beat_inventory": [
            {"beat_index": k, "r_peak_sample": int(b.r_peak.sample), "morphology_cluster_id": b.morphology_cluster_id,
             "beat_type": b.beat_type, "beat_type_source": b.beat_type_source,
             "morphology_outlier": bool(b.morphology_outlier),
             "accepted": id(b) in accepted_ids}
            for k, b in enumerate(delineation.beats)
        ],
        "delineation": {
            "method": delineation.source_method,
            "accepted_beat_wave_segments": segments,
            "rhythm_context": {
                "class": delineation.rhythm.rhythm_class.value,
                "confidence": float(delineation.rhythm.confidence),
                "flags": list(delineation.rhythm.flags),
                "atrial_frequency_hz": delineation.rhythm.atrial_frequency_hz,
                "independent_atrial_rate_hz": getattr(delineation.rhythm, "independent_atrial_rate_hz", None),
                "p_wave_coverage": delineation.rhythm.p_wave_coverage,
            },
            "quality_score": float(delineation.quality_score),
            "review_reasons": list(delineation.review_reasons),
        },
        "classical_criteria": {},
        "feature_values": {},
        "notes": [],
    }
    try:
        classical = ClassicalECGCriteria().compute(paths["diagnostic"], delineation, age=age, sex=sex)
        payload["classical_criteria"] = {key: value.to_dict() for key, value in classical.items()}
    except Exception as exc:  # Kriterien sind optional
        payload["notes"].append(f"klassische Kriterien nicht berechnet: {exc}")
    try:
        robust, _ = construct_robust_morphology_record(paths["morphology"], delineation)
        vcg_morph = VCGTransformRegistry.transform(robust, "kors")
        record_vcg = VCGTransformRegistry.transform(paths["morphology"], "kors")
        results = BiomarkerRegistry.compute_all(
            delineation=delineation, ecg_12lead=robust, vcg_xyz=vcg_morph.xyz, fs=ANALYSIS_FS,
            names=list(THESIS_FEATURES), record_ecg_12lead=paths["morphology"], record_raw_ecg_12lead=sig,
            record_acquisition={"original_sampling_rate_hz": original_fs}, record_vcg_xyz=record_vcg.xyz,
        )
        payload["feature_values"] = {r.name: (float(r.value) if r.value is not None and np.isfinite(r.value) else None)
                                     for r in results}
    except Exception as exc:
        payload["notes"].append(f"Thesenmarker nicht berechnet: {exc}")
    return payload
