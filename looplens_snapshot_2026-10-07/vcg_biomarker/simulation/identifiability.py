"""Local, uncertainty-scaled parameter distinguishability of forward emulators.

No ECG-to-cell inversion, posterior probability or clinical localization claim.
Related motivation: https://doi.org/10.1016/j.media.2025.103641.
"""
from __future__ import annotations

import json
import numpy as np

from .learning import PARAMETERS, OUTPUTS, predict_emulator


def global_polynomial_sensitivity(emulator):
    """Exact Sobol decomposition on the independent uniform conductance box.

    For x_i~U(-1,1): Var(x)=1/3, Var(x^2)=4/45, Var(x_i*x_j)=1/9.
    This avoids Monte Carlo work; it describes the surrogate, not patient causes.
    """
    predict_emulator(emulator, dict.fromkeys(PARAMETERS, 1.))
    coef = np.asarray(emulator["coefficients"], float)
    n = len(PARAMETERS)
    first = coef[1:n+1]**2/3.
    pairs, k = [], n+1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                first[i] += coef[k]**2*(4./45.)
            else:
                pairs.append((i, j, coef[k]**2/9.))
            k += 1
    total_effect, variance = first.copy(), first.sum(0)
    for i, j, contribution in pairs:
        variance += contribution
        total_effect[i] += contribution
        total_effect[j] += contribution
    summaries = {}
    for column, output in enumerate(OUTPUTS):
        if variance[column] <= 0:
            summaries[output] = {"status": "constant_surrogate_output", "variance": 0., "first_order": None, "total_order": None}
            continue
        summaries[output] = {
            "status": "computed", "variance": float(variance[column]),
            "first_order": dict(zip(PARAMETERS, (first[:, column]/variance[column]).tolist())),
            "total_order": dict(zip(PARAMETERS, (total_effect[:, column]/variance[column]).tolist())),
            "second_order": {f"{PARAMETERS[i]}:{PARAMETERS[j]}": float(v[column]/variance[column]) for i, j, v in pairs}}
    return {"schema": "quadratic-emulator-sobol/1", "clinical_use": False, "protocol": emulator.get("protocol"),
            "input_distribution": "independent uniform conductance factors on [0.5, 1.5], not measured patient distribution",
            "outputs": summaries,
            "limitations": ["Exact for the fitted quadratic polynomial only, not the original ionic solver.",
                            "Interactions enter multiple total-order indices; their sum can exceed one.",
                            "Sensitivity does not establish invertibility or biological causality."]}


def emulator_jacobian(emulator, scales):
    """Exact derivative w.r.t. dimensionless x=(conductance_factor-1)/0.5."""
    predict_emulator(emulator, scales)  # Reuse schema, coefficient and domain validation.
    x = (np.array([scales[p] for p in PARAMETERS])-1.)/.5
    coef = np.asarray(emulator["coefficients"], float)
    n = len(PARAMETERS)
    jac = coef[1:n+1].T.copy()
    k = n+1
    for i in range(n):
        for j in range(i, n):
            jac[:, i] += coef[k]*x[j]
            jac[:, j] += coef[k]*x[i]
            k += 1
    return jac


def local_identifiability(emulators, scales, measurement_tolerances):
    """Stack distinct fixed protocols; retain null directions instead of guessing.

    Errors are explicit per-output positive tolerances in the output's units.
    Conservative divisor = measurement tolerance + emulator calibration radius.
    These are not Gaussian SDs: singular values are sensitivity diagnostics,
    NOT Fisher confidence intervals. Marginal emulator undercoverage is retained.
    """
    if not emulators or len(emulators) != len(measurement_tolerances):
        raise ValueError("One explicit measurement tolerance mapping per emulator required")
    seen, blocks, reports = set(), [], []
    for emulator, errors in zip(emulators, measurement_tolerances):
        protocol = emulator.get("protocol")
        if not isinstance(protocol, dict) or not protocol:
            raise ValueError("Frozen protocol required; cannot count an unidentified model as independent information")
        signature = json.dumps(protocol, sort_keys=True, allow_nan=False)
        if signature in seen:
            raise ValueError("Duplicate protocol is not an independent experiment")
        seen.add(signature)
        if set(errors) != set(OUTPUTS):
            raise ValueError("Explicit tolerances for every cellular output required")
        error = np.array([errors[key] for key in OUTPUTS], float)
        if not np.isfinite(error).all() or np.any(error <= 0):
            raise ValueError("Finite positive measurement tolerances required")
        jac = emulator_jacobian(emulator, scales)
        radius = np.asarray(emulator["calibration_radius_90"], float)
        blocks.append(jac/(error+radius)[:, None])
        reports.append({"protocol": protocol, "job_sha256": emulator.get("job_sha256"),
                        "audit_coverage_90": emulator.get("audit_coverage_90"),
                        "jacobian_output_units_per_normalized_parameter": jac.tolist()})
    matrix = np.vstack(blocks)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = max(matrix.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > threshold))
    full = rank == len(PARAMETERS)
    return {"schema": "local-emulator-identifiability/1", "clinical_use": False,
            "parameters": list(PARAMETERS), "at_conductance_factors": dict(scales),
            "protocols": reports, "normalized_sensitivity": matrix.tolist(),
            "singular_values": singular.tolist(), "numerical_rank": rank,
            "unit_perturbation_below_tolerance_directions": int(np.sum(singular < 1.) + len(PARAMETERS)-len(singular)),
            "weakest_normalized_parameter_direction": vh[-1].tolist(),
            "practical_sensitivity_threshold": 1.,
            "practical_sensitivity_meaning": "Local unit normalized-parameter perturbation versus supplied tolerance; not statistical rank or confidence.",
            "parameter_count": len(PARAMETERS), "nullity": len(PARAMETERS)-rank,
            "null_directions": vh[rank:].tolist(),
            "full_column_rank": full,
            "condition_number": float(singular[0]/singular[-1]) if full else None,
            "unique_patient_parameters_identified": False,
            "limitations": ["Local linear diagnostic; nonlinear/global identifiability is not established.",
                            "A full rank does not prove clinical accuracy or sufficient signal-to-noise.",
                            "Calibration radii are marginal, not a simultaneous confidence region.",
                            "Shared emulator/model errors can correlate across protocols."]}
