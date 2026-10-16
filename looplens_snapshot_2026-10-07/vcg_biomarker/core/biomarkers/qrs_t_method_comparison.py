"""Same-input QRS/J-to-T method comparison; no new diagnostic thresholds."""
import numpy as np

from .advanced_vcg_biomarkers import direct_qrs_t_angles_8lead, DIRECT_QRS_T_LEADS
from ..data_loader import LEAD_MAP
from ..svd_projection import project_ecg_components
from ..vcg_transform import VCGTransformRegistry

METHOD_COMPARISON_VERSION = "1.1.0"
METHODS = ("direct_8lead", "kors", "inverse_dower", "svd3_pr_centered")


def compare_qrs_t_methods(ecg, *, qrs_bounds, t_bounds, baseline_bounds, segment_evaluable=None):
    rows = [{"method": method, "area_angle_deg": None, "integral_angle_deg": None,
        "status": "non_estimable", "reason": "required_segments_unavailable",
        "version": METHOD_COMPARISON_VERSION, "clinical_claim_allowed": False,
        "area_definition": "angle_of_signed_integrals",
        "integral_definition": "magnitude_weighted_pairwise_angle_average",
        "segment_scope": "QRS_and_J_to_T_offset_same_PR_baseline",
        "svd_basis_scope": "whole_supplied_morphology_signal_after_PR_centering"} for method in METHODS]
    if segment_evaluable is not None:
        failed = [name for name in ("qrs", "t", "pr") if segment_evaluable.get(name) is not True]
        if failed:
            for row in rows:
                row["reason"] = "segment_quality_gate_failed:" + ",".join(failed)
            return rows
    if any(bounds is None for bounds in (qrs_bounds, t_bounds, baseline_bounds)):
        return rows
    signal = np.asarray(ecg, dtype=float)
    if signal.ndim != 2 or signal.shape[1] != 12 or len(signal) < 8 or not np.isfinite(signal).all():
        for row in rows: row["reason"] = "invalid_common_twelve_lead_signal"
        return rows
    try:
        bounds_array = np.asarray((qrs_bounds, t_bounds, baseline_bounds))
    except (TypeError, ValueError):
        for row in rows: row["reason"] = "invalid_common_segment_indices"
        return rows
    if (bounds_array.shape != (3, 2)
            or not np.issubdtype(bounds_array.dtype, np.integer)):
        for row in rows: row["reason"] = "invalid_common_segment_indices"
        return rows
    (q0, q1), (t0, t1), (b0, b1) = bounds_array
    if not (0 <= b0 < b1 <= q0 < q1 <= t0 < t1 < len(signal)):
        for row in rows: row["reason"] = "invalid_common_segment_order_or_baseline"
        return rows
    arguments = dict(qrs_bounds=(q0, q1), t_bounds=(q1, t1), baseline_bounds=(b0, b1))
    centered = signal-signal[b0:b1+1].mean(axis=0)
    indices = [LEAD_MAP[name] for name in DIRECT_QRS_T_LEADS]
    for row in rows:
        method = row["method"]
        if method == "direct_8lead":
            projected = signal
        else:
            if method == "svd3_pr_centered":
                svd = project_ecg_components(centered)
                if svd.metadata["status"] != "available":
                    row["reason"] = str(svd.metadata["reason"])
                    continue
                coordinates = svd.coordinates
                row["retained_energy_fraction"] = svd.metadata["retained_energy_fraction"]
            else:
                coordinates = VCGTransformRegistry.transform(centered, str(method)).xyz
            # Isometrically embed 3D into the existing tested 8D angle kernel.
            # Zero-padding does not change dot products, norms or integrals.
            projected = np.zeros_like(signal)
            projected[:, indices[:3]] = coordinates
        area, integral = direct_qrs_t_angles_8lead(projected, **arguments)
        row.update(area_angle_deg=area, integral_angle_deg=integral,
            status="available" if area is not None and integral is not None else "non_estimable",
            reason="" if area is not None and integral is not None else "zero_or_unmeasurable_angle_vectors")
    return rows
