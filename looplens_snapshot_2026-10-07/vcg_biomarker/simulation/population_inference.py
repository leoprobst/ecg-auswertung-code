"""Joint protocol error calibration and bounded, set-valued cellular inference.

This is NOT ECG inversion: the observations must be cellular AP measurements
or synthetic validation. Finite candidate envelopes are not patient confidence
intervals, and a numerical full rank is not proof of a unique inverse solution.
"""
from __future__ import annotations

import math
import hashlib
import numpy as np

from .learning import PARAMETERS, OUTPUTS, design, digest, predict_emulator


def _models(models):
    if not 1 <= len(models) <= 5:
        raise ValueError("One to five frozen cellular protocols required")
    protocols = []
    for model in models:
        predict_emulator(model, dict.fromkeys(PARAMETERS, 1.))
        if model.get("status") != "research_surrogate" or not model.get("job_sha256"):
            raise ValueError("Computed, provenance-bound emulator required")
        protocol = model.get("protocol")
        required = {"model", "beats", "cycle_ms", "output_ms", "cell_environment", "precondition_max_beats"}
        if not isinstance(protocol, dict) or not required <= protocol.keys():
            raise ValueError("Complete protocol/environment required")
        if protocol in protocols:
            raise ValueError("Duplicate protocol")
        protocols.append(protocol)
    if len({m["job_sha256"] for m in models}) != 1:
        raise ValueError("Protocols from different campaigns cannot be silently combined")
    return protocols


def predict_batch(models, factors):
    """Small BLAS operations; caller controls chunk size, no process nesting."""
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 2 or factors.shape[1] != len(PARAMETERS) or not np.isfinite(factors).all():
        raise ValueError("Finite conductance matrix required")
    if np.any((factors < .5) | (factors > 1.5)):
        raise ValueError("Outside trained conductance domain")
    features = design(factors)
    return np.stack([features @ np.asarray(m["coefficients"]) for m in models], axis=1)


def calibrate_joint(models, rows, coverage=.9):
    """Max absolute standardized residual, ONE score per candidate over all rates.

    Output scales come only from the fit data; the quantile uses calibration
    candidates, never audit outcomes. Exchangeability is needed for a conformal
    guarantee and is not established for a fixed Latin-hypercube simulator bank.
    """
    protocols = _models(models)
    if not .5 <= coverage < 1:
        raise ValueError("Invalid requested coverage")
    cycles = [p["cycle_ms"] for p in protocols]
    if len(set(cycles)) != len(cycles):
        raise ValueError("This bank contract requires distinct pacing rates")
    bank, excluded = {}, 0
    for row in rows:
        if row.get("job_sha256") != models[0]["job_sha256"]:
            raise ValueError("Case belongs to another campaign")
        candidate = row["candidate_id"]
        partition = row["partition"]
        if partition not in {"fit", "calibration", "audit"}:
            raise ValueError("Unknown data role")
        item = bank.setdefault(candidate, {"partition": partition, "scales": row["scales"], "rows": {}})
        if item["partition"] != partition or item["scales"] != row["scales"]:
            raise ValueError("Candidate crossed roles or changed conductances")
        if set(row["scales"]) != set(PARAMETERS):
            raise ValueError("Unknown parameter contract")
        cycle = row["cycle_ms"]
        if cycle not in cycles or cycle in item["rows"]:
            raise ValueError("Unknown or duplicate candidate protocol")
        item["rows"][cycle] = row
    arrays = {}
    for partition in ("fit", "calibration", "audit"):
        selected = []
        for key, item in sorted(bank.items()):
            if item["partition"] != partition:
                continue
            if (len(item["rows"]) != len(cycles) or any(
                    r["status"] != "computed" or not r.get("periodicity_check_passed") for r in item["rows"].values())):
                excluded += 1
                continue
            selected.append((key, item))
        if len(selected) < (40 if partition == "fit" else 10):
            raise ValueError("Insufficient complete candidate protocols for " + partition)
        factors = np.array([[item["scales"][p] for p in PARAMETERS] for _, item in selected])
        truth = np.array([[[item["rows"][cycle]["biomarkers"][o] for o in OUTPUTS] for cycle in cycles]
                          for _, item in selected])
        if not np.isfinite(truth).all():
            raise ValueError("Non-finite cellular targets")
        prediction = predict_batch(models, factors)
        arrays[partition] = (np.array([key for key, _ in selected]), factors, truth, prediction)
    scale = np.maximum(arrays["fit"][2].std(axis=0), 1e-6)
    cal = arrays["calibration"]
    scores = np.max(np.abs(cal[3]-cal[2])/scale, axis=(1, 2))
    order = math.ceil((len(scores)+1)*coverage)
    if order > len(scores):
        raise ValueError("Calibration set too small for a finite requested quantile")
    quantile = float(np.sort(scores)[order-1])
    radius = quantile*scale
    audit = arrays["audit"]
    accepted = np.all(np.abs(audit[3]-audit[2]) <= radius, axis=(1, 2))
    marginal = np.array([m["calibration_radius_90"] for m in models])
    report = {"schema": "joint-cellular-calibration/1", "clinical_use": False,
              "emulators_sha256": digest(models), "job_sha256": models[0]["job_sha256"],
              "protocols": protocols, "outputs": list(OUTPUTS), "parameters": list(PARAMETERS),
              "nominal_coverage": coverage, "quantile_rank": order, "max_residual_quantile": quantile,
              "counts": {k: len(v[0]) for k, v in arrays.items()}, "excluded_incomplete_candidates": excluded,
              "fit_only_scales": scale.tolist(), "simultaneous_radius": radius.tolist(),
              "audit_joint_covered": int(accepted.sum()), "audit_joint_coverage": float(accepted.mean()),
              "audit_joint_coverage_using_old_marginal_90_intervals": float(np.mean(np.all(
                  np.abs(audit[3]-audit[2]) <= marginal, axis=(1, 2)))),
              "calibration_data_sha256": digest({"ids": cal[0].tolist(), "truth": cal[2].tolist()}),
              "audit_role": "posthoc_descriptive_internal_audit_not_new_external_validation",
              "limitations": ["Coverage concerns the simulator, not patients or model discrepancy.",
                              "Fixed Latin-hypercube design: no asserted exchangeability guarantee.",
                              "All protocols must be available; failed/nonperiodic candidates are excluded.",
                              "Correlated protocol errors enter a joint candidate-level maximum."]}
    return report, arrays


class PopulationMatcher:
    """Reuse a bounded prediction bank for many observations; no quadratic matrix."""
    def __init__(self, models, calibration, factors, chunk_size=4096):
        protocols = _models(models)
        if (calibration.get("schema") != "joint-cellular-calibration/1"
                or calibration.get("clinical_use") is not False
                or calibration.get("emulators_sha256") != digest(models)
                or calibration.get("protocols") != protocols):
            raise ValueError("Joint calibration does not belong to these emulators")
        factors = np.asarray(factors, dtype=float)
        if factors.ndim != 2 or not 1 <= len(factors) <= 131072 or not 1 <= chunk_size <= 8192:
            raise ValueError("Bounded candidate bank and chunk size required")
        self.radius = np.array(calibration["simultaneous_radius"], dtype=float)
        if (self.radius.shape != (len(models), len(OUTPUTS)) or not np.isfinite(self.radius).all()
                or np.any(self.radius < 0)):
            raise ValueError("Invalid joint calibration radius")
        self.protocols, self.chunk_size = protocols, chunk_size
        self.factors = factors.copy()
        self.predictions = np.concatenate([predict_batch(models, factors[i:i+chunk_size])
                                           for i in range(0, len(factors), chunk_size)])
        self.bank_sha256 = hashlib.sha256(np.ascontiguousarray(factors, dtype="<f8").tobytes()).hexdigest()

    def match(self, observation):
        kind = observation.get("reference_kind")
        if kind not in {"synthetic_validation", "independent_cellular_action_potential"}:
            raise ValueError("Surface ECG/QT/diagnosis is not cellular ground truth")
        if observation.get("protocols") != self.protocols or observation.get("outputs") != list(OUTPUTS):
            raise ValueError("Observation definitions and full cellular protocols must match")
        if kind == "independent_cellular_action_potential" and (
                not observation.get("reference_provenance") or observation.get("measurement_method") != "ap_features_v1"):
            raise ValueError("Independent cellular provenance and measurement method required")
        values = np.asarray(observation["values"], dtype=float)
        errors = np.asarray(observation["tolerances"], dtype=float)
        if (values.shape != self.radius.shape or errors.shape != values.shape or not np.isfinite(values).all()
                or not np.isfinite(errors).all() or np.any(errors <= 0)):
            raise ValueError("Finite measurements and positive explicit error bounds required")
        allowed = errors+self.radius
        count = 0
        lower, upper = np.full(len(PARAMETERS), np.inf), np.full(len(PARAMETERS), -np.inf)
        for i in range(0, len(self.factors), self.chunk_size):
            accepted = np.all(np.abs(self.predictions[i:i+self.chunk_size]-values) <= allowed, axis=(1, 2))
            points = self.factors[i:i+self.chunk_size][accepted]
            if len(points):
                count += len(points)
                lower, upper = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0))
        return {"schema": "cellular-compatible-set/1", "clinical_use": False,
                "status": "compatible_candidates" if count else "no_candidate_on_finite_design",
                "tested_candidates": len(self.factors), "accepted_candidates": count,
                "candidate_design_sha256": self.bank_sha256,
                "parameter_envelope": {p: [float(lo), float(hi)] for p, lo, hi in zip(PARAMETERS, lower, upper)} if count else None,
                "unique_parameters_identified": False, "posterior_probabilities": None,
                "warning": "Finite candidate envelope, not a confidence region. Zero matches do not prove impossibility; one does not prove uniqueness."}
