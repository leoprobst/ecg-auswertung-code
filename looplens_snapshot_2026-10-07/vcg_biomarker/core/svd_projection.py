"""Record-local orthogonal ECG components, not anatomical Frank coordinates."""
from dataclasses import dataclass

import numpy as np

from .data_loader import LEAD_MAP

SVD_PROJECTION_VERSION = "1.0.0"
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


@dataclass(frozen=True)
class SVDProjection:
    coordinates: np.ndarray
    loadings: np.ndarray
    metadata: dict


def project_ecg_components(ecg: np.ndarray) -> SVDProjection:
    """Uncentered SVD of the eight independent, equally weighted mV leads.

    Preserve the supplied preprocessing/baseline; never standardize leads or
    fit separate QRS/T bases. Positive largest-magnitude loadings fix signs
    reproducibly, but not anatomical orientation or tied-subspace rotations.
    """
    values = np.asarray(ecg, dtype=float)
    if values.ndim != 2 or values.shape[1] != 12 or len(values) < 8:
        raise ValueError("SVD projection requires at least eight samples of twelve ordered leads")
    if not np.isfinite(values).all():
        raise ValueError("SVD projection requires finite ECG values")
    data = values[:, [LEAD_MAP[name] for name in INDEPENDENT_LEADS]]
    metadata = {"version": SVD_PROJECTION_VERSION, "anatomical_axes": False,
        "axis_names": ["C1", "C2", "C3"], "leads": list(INDEPENDENT_LEADS),
        "units": "mV", "centering": "none_preserve_supplied_baseline",
        "basis_scope": "whole_record_same_basis_for_all_segments",
        "lead_standardization": False, "clinical_claim_allowed": False,
        "status": "non_estimable", "reason": "", "axis_direction_clinically_interpretable": False}
    maximum = float(np.max(np.abs(data)))
    empty = SVDProjection(np.full((len(data), 3), np.nan), np.full((3, 8), np.nan), metadata)
    if maximum == 0:
        metadata["reason"] = "zero_signal_energy"
        return empty
    try:
        _, singular, loadings = np.linalg.svd(data/maximum, full_matrices=False)
    except np.linalg.LinAlgError:
        metadata["reason"] = "svd_nonconvergence"
        return empty
    rank = int(np.sum(singular > singular[0]*max(data.shape)*np.finfo(float).eps))
    metadata["numerical_rank"] = rank
    if rank < 3:
        metadata["reason"] = "insufficient_rank_for_three_components"
        return empty
    basis = loadings[:3].copy()
    signs = np.sign(basis[np.arange(3), np.argmax(np.abs(basis), axis=1)])
    basis *= signs[:, None]
    energy = singular**2
    metadata.update(status="available", retained_energy_fraction=float(energy[:3].sum()/energy.sum()),
        singular_value_ratios=(singular/singular[0]).tolist(),
        near_degenerate_axes=bool(np.any(np.abs(np.diff(singular[:4]))/singular[0] < 1e-6)))
    return SVDProjection(data @ basis.T, basis, metadata)
