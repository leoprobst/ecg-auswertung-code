"""External validation adapters; missing mapping is never replaced by ECG labels."""
from __future__ import annotations

import numpy as np

EXTERNAL_SOURCES = {
    "edgar_2026": {
        "url": "https://www.ecg-imaging.org/community/ecgi-data-challenge",
        "access": "EDGAR account required; no account created or terms accepted automatically",
        "species_must_be_separated": True,
        "references": ["pacing_site", "epicardial_potential", "ablation_site"],
        "warning": "Successful ablation site is not automatically focal origin; animal experiments are not clinical validation",
    }
}


def compare_waveforms(predicted_mv, observed_mv, times_ms, lead_names, reference):
    """No fitted gain/time shift/polarity reversal to make error artificially small."""
    a, b, t = np.asarray(predicted_mv, float), np.asarray(observed_mv, float), np.asarray(times_ms, float)
    required = ("cohort_id", "patient_id", "episode_id", "signal_id", "source", "model_version", "geometry_source")
    if any(not isinstance(reference.get(k), str) or not reference[k].strip() for k in required):
        raise ValueError("Complete waveform provenance required")
    if reference.get("data_role") != "validation" or reference.get("time_alignment_verified") is not True:
        raise ValueError("Frozen validation role and verified time alignment required")
    if reference.get("species") not in {"human", "animal", "synthetic"}:
        raise ValueError("Explicit species/data class required")
    if reference.get("reference_type") not in {"surface_ecg", "epicardial_potential", "synthetic_reference"}:
        raise ValueError("Unknown waveform reference type")
    if (a.shape != b.shape or a.ndim != 2 or not 4 <= len(a) <= 20000 or not 1 <= a.shape[1] <= 256
            or len(t) != len(a) or t.ndim != 1 or np.any(np.diff(t) <= 0)
            or len(lead_names) != a.shape[1] or len(set(lead_names)) != len(lead_names)
            or not all(np.isfinite(x).all() for x in (a, b, t))):
        raise ValueError("Aligned finite signal arrays, units mV/ms and unique leads required")
    values = []
    for i, name in enumerate(lead_names):
        ac, bc = a[:, i]-a[:, i].mean(), b[:, i]-b[:, i].mean()
        norm = np.linalg.norm(ac)*np.linalg.norm(bc)
        values.append({"lead": name, "rmse_mv": float(np.sqrt(np.mean((a[:, i]-b[:, i])**2))),
                       "mean_signed_error_mv": float(np.mean(a[:, i]-b[:, i])),
                       "correlation": float(ac@bc/norm) if norm > 1e-15 else None})
    return {"schema": "external-waveform-comparison/1", "clinical_use": False, "reference": reference,
            "by_lead": values, "gain_fitted": False, "time_shift_fitted": False,
            "warning": "One record is not independent population validation; aggregate by patient and source."}


def evaluate_regional_mapping(frame, task, predicted_regions):
    from ..mimic.focus_localization import validate_mapping, TASKS
    groups = validate_mapping(frame, task)
    if not frame["split"].eq("validation").all():
        raise ValueError("Mapping evaluation cannot mix development and validation")
    if len(predicted_regions) != len(frame):
        raise ValueError("One frozen prediction/abstention per episode required")
    predictions = np.asarray(predicted_regions, object)
    if any(p is not None and p not in TASKS[task]["regions"] for p in predictions):
        raise ValueError("Unsupported anatomical prediction")
    active = np.array([p is not None for p in predictions])
    truth = frame.region.to_numpy()
    group_coverage, group_accuracy = [], []
    for patient in np.unique(groups):
        mask = groups == patient
        group_coverage.append(float(active[mask].mean()))
        if np.any(active & mask):
            group_accuracy.append(float((predictions[mask & active] == truth[mask & active]).mean()))
    return {"schema": "external-regional-mapping-evaluation/1", "clinical_use": False,
            "patients": len(group_coverage), "episodes": len(frame),
            "reference_type": str(frame.reference_type.iloc[0]),
            "patient_macro_coverage": float(np.mean(group_coverage)),
            "patient_macro_accuracy_when_issued": float(np.mean(group_accuracy)) if group_accuracy else None,
            "patients_with_issued_predictions": len(group_accuracy),
            "abstentions": int(np.sum(~active)), "localization_error_mm": None,
            "warning": "Regional agreement only; no millimetre accuracy without registered coordinates and independent reference uncertainty."}
