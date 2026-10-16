"""Explicit regional substrate scenarios, not inferred coronary anatomy or genes."""
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized

CHANNEL_MECHANISMS = {
    "reference": {"scales": {}, "interpretation": "published cell reference"},
    "IKs_reduction": {"scales": {"IKs": .5}, "interpretation": "reduced repolarization reserve; not an individual KCNQ1 variant"},
    "IKr_reduction": {"scales": {"IKr": .5}, "interpretation": "reduced repolarization reserve; genetic and acquired causes not separable"},
    "INa_reduction": {"scales": {"INa": .5}, "interpretation": "reduced sodium excitability; not a Brugada or SCN5A diagnosis"},
    "IKr_increase": {"scales": {"IKr": 1.5}, "interpretation": "repolarization mechanism; not a mutation-specific short-QT model"},
}

GENETIC_READINESS = {
    "KCNQ1/KCNE1": "IKs conductance experiments available; variant-specific kinetics/heterozygosity missing",
    "KCNH2/KCNE2": "IKr conductance experiments available; variant-specific gating/trafficking missing",
    "SCN5A_loss_of_function": "INa experiments available; variant-specific kinetics and tissue phenotype unverified",
    "SCN5A_gain_of_function_LQT3": "late INa scaling in O'Hara 2011 single cells available; variant-specific gating/Markov kinetics not implemented",
    "RYR2/CASQ2_CPVT": "not implemented; requires stochastic calcium release and adrenergic response",
    "CACNA1C": "ICaL conductance experiments only; Timothy syndrome kinetics not implemented",
    "HCN4": "SAN model available; individual mutation and autonomic parameter mapping not implemented",
    "desmosomal_cardiomyopathy": "requires measured fibrofatty substrate, not an ion conductance multiplier",
}


def coronary_territories(segments, dominance="right", wraparound_lad=False):
    """Schematic scenario weights, NOT frequencies or patient vessel probabilities.

    AHA maps overlap substantially with measured coronary territories. Supply
    dominance alone cannot uniquely label a segment or an infarct culprit.
    """
    if dominance not in {"right", "left", "codominant"}:
        raise ValueError("Unknown dominance must be explored as separate scenarios")
    seg = np.asarray(segments)
    if seg.dtype.kind not in "iu" or np.any((seg < 1) | (seg > 17)):
        raise ValueError("LV AHA segments 1–17 required; no silent RV/atrial assignment")
    weights = np.zeros((len(seg), 3))  # LAD, LCx, RCA
    for i, s in enumerate(seg):
        if s in (1, 2, 7, 8, 13, 14, 17):
            weights[i] = [1, 0, 0]
        elif s in (5, 6, 11, 12, 16):
            weights[i] = [0, 1, 0]
        else:
            weights[i] = {"right": [0, 0, 1], "left": [0, 1, 0], "codominant": [0, .5, .5]}[dominance]
        if wraparound_lad and s == 15:
            weights[i] = [1, 0, 0]
    return {"arteries": ["LAD", "LCx", "RCA"], "weights": weights,
            "interpretation": "schematic_supply_scenario_not_individual_anatomy", "dominance": dominance,
            "not_modelled": ["nodal_artery_origin", "microcirculation", "collaterals", "measured_coronary_tree"]}


def transported_extracellular_field(lap, mass, initial, blood_value, perfusion_rate_per_ms,
                                  source_mmol_l_per_ms, times_ms):
    """Conservative diffusion + prescribed local source + perfusion washout.

    dK/dt = L K + source + perfusion*(blood-K). This is a scenario transport
    equation; perfusion and release must be supplied, not derived from an ECG.
    No assertion that ischemia spreads like an electrical wave or ATP is solved.
    """
    n = len(mass)
    initial, rate, source = (np.asarray(v, float) for v in (initial, perfusion_rate_per_ms, source_mmol_l_per_ms))
    times = np.asarray(times_ms, float)
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in (initial, rate, source)):
        raise ValueError("Finite node fields required")
    if np.any(initial < 0) or np.any(rate < 0) or np.any(source < 0) or not np.isfinite(blood_value) or blood_value < 0:
        raise ValueError("Nonnegative concentrations/rates required")
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Increasing ms times starting at zero required")
    if len(times)*n*8 > 64*1024**2 or lap.shape != (n, n):
        raise ValueError("Transport budget/shape mismatch")
    offdiag = lap - diags(lap.diagonal())
    if offdiag.nnz and offdiag.data.min() < -1e-10:
        raise ValueError("Transport requires a positivity-preserving Laplacian; refine non-monotone mesh")
    if not np.allclose(lap@np.ones(n), 0, atol=1e-8) or not np.allclose(mass@lap, 0, atol=1e-8):
        raise ValueError("Nonconservative transport operator")
    values = np.empty((len(times), n))
    values[0] = initial
    cache = {}
    for i, dt in enumerate(np.diff(times), 1):
        key = float(dt)
        if key not in cache:
            cache.clear()  # variable timesteps must not accumulate sparse factorizations
            cache[key] = factorized((eye(n) - dt*lap + diags(dt*rate)).tocsc())
        values[i] = cache[key](values[i-1] + dt*(source+rate*blood_value))
        if not np.isfinite(values[i]).all() or values[i].min() < -1e-8:
            raise ArithmeticError("Invalid transported concentration; no clipping")
    return values


def perfusion_exposure(mesh, node_segments, artery, dominance, times_ms, *,
                       chemical_diffusivity_mm2_per_ms, blood_potassium_mmol_l,
                       baseline_washout_per_ms, perfusion_fraction, local_release_mmol_l_per_ms,
                       wraparound_lad=False):
    """Connect an explicit supply scenario to transport and the ionic tissue solver.

    No default flow/release constants masquerade as clinical estimates. Electrical
    diffusivity is NEVER reused for extracellular potassium transport. A stationary
    final exposure is exported for a short EP experiment (separate time scales).
    """
    from .monodomain import assemble
    values = [chemical_diffusivity_mm2_per_ms, blood_potassium_mmol_l, baseline_washout_per_ms,
              perfusion_fraction, local_release_mmol_l_per_ms]
    if (not np.isfinite(values).all() or chemical_diffusivity_mm2_per_ms < 0
            or baseline_washout_per_ms < 0 or local_release_mmol_l_per_ms < 0
            or not 0 <= perfusion_fraction <= 1 or not 1.5 <= blood_potassium_mmol_l <= 9):
        raise ValueError("Explicit, physical exposure parameters required")
    if len(node_segments) != len(mesh.xyz_mm) or artery not in {"LAD", "LCx", "RCA"}:
        raise ValueError("Node-level LV segment assignments and named vessel required")
    territories = coronary_territories(node_segments, dominance, wraparound_lad)
    affected = territories["weights"][:, territories["arteries"].index(artery)]
    unit_lap, mass, _ = assemble(mesh, 1., 1.)
    lap = unit_lap*chemical_diffusivity_mm2_per_ms
    perfusion = baseline_washout_per_ms*(1-affected*(1-perfusion_fraction))
    release = local_release_mmol_l_per_ms*affected
    trajectory = transported_extracellular_field(lap, mass, np.full(len(mass), blood_potassium_mmol_l),
                                                 blood_potassium_mmol_l, perfusion, release, times_ms)
    result = RegionalSubstrate(potassium_mmol_l=trajectory[-1],
                              provenance="prescribed perfusion/chemical transport exposure; schematic territory, no ATP/pH solution")
    result.fields(len(mass), len(mesh.tetra))  # reject beyond verified ionic concentration domain
    return result, trajectory, {"artery": artery, "dominance": dominance,
                                "territory_interpretation": territories["interpretation"],
                                "chemical_diffusivity_mm2_per_ms": chemical_diffusivity_mm2_per_ms,
                                "electrical_diffusivity_used_as_chemical": False,
                                "blood_potassium_mmol_l": blood_potassium_mmol_l,
                                "baseline_washout_per_ms": baseline_washout_per_ms,
                                "perfusion_fraction": perfusion_fraction,
                                "local_release_mmol_l_per_ms": local_release_mmol_l_per_ms,
                                "exposure_time_ms": float(np.asarray(times_ms)[-1]),
                                "clinical_use": False}


@dataclass
class RegionalSubstrate:
    potassium_mmol_l: np.ndarray | None = None
    channel_scales: dict[str, np.ndarray] = field(default_factory=dict)
    element_coupling_scale: np.ndarray | None = None
    provenance: str = "user_defined_mechanistic_scenario"
    katp_open_fraction: np.ndarray | None = None

    def fields(self, n, elements):
        out = {}
        if self.potassium_mmol_l is not None:
            out["potassium_mmol_l"] = np.asarray(self.potassium_mmol_l, float)
        if self.katp_open_fraction is not None:
            out["katp_open_fraction"] = np.asarray(self.katp_open_fraction, float)
        for channel, values in self.channel_scales.items():
            if channel not in {"INa", "ICaL", "IKr", "IKs", "IK1", "Ito"}:
                raise ValueError("Unknown regional channel")
            out["scale:"+channel] = np.asarray(values, float)
        for key, values in out.items():
            lo, hi = ((1.5, 9.) if key == "potassium_mmol_l" else
                      (0., .01) if key == "katp_open_fraction" else (0., 3.))
            if values.shape != (n,) or not np.isfinite(values).all() or np.any((values < lo) | (values > hi)):
                raise ValueError("Invalid regional ionic field")
        if self.element_coupling_scale is not None:
            values = np.asarray(self.element_coupling_scale, float)
            if values.shape != (elements,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Coupling field must be 0–1 per element")
        if not self.provenance.strip():
            raise ValueError("Regional scenario source required")
        return out
