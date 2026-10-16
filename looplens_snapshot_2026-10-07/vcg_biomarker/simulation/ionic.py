"""Pinned human ionic models with an event-exact, adaptive single-cell solver.

This is a quantitative cellular assay, not yet a patient-specific whole-heart
bidomain model. No fitting of unidentifiable conductances from demographics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import math
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from .model_assets import COMMIT, MODELS, model_path

CHANNELS = {
    "ohara-2011": {"INa": "ina.gNa", "INaL": "inal.gNaL", "ICaL": "ical.PCa_base", "IKr": "ikr.gKr", "IKs": "iks.gKs", "IK1": "ik1.gK1", "Ito": "ito.gto"},
    "tentusscher-2006": {"INa": "ina.gNa", "ICaL": "ical.gCaL", "IKr": "ikr.gKr", "IKs": "iks.gKs", "IK1": "ik1.gK1", "Ito": "ito.gto"},
    "courtemanche-1998": {"INa": "ina.gNa", "ICaL": "ical.gCaL", "IKr": "ikr.gKr", "IKs": "iks.gKs", "IK1": "ik1.gK1", "Ito": "ito.gto", "IKur": "ikur.gKur_base"},
    "stewart-2009": {"INa": "ina.g_Na", "ICaL": "ical.g_CaL", "IKr": "ikr.g_Kr", "IKs": "iks.g_Ks", "IK1": "ik1.g_K1"},
    "fabbri-2017": {"INa": "ina.gNa", "INaL": "ina.gNaL", "IKr": "ikr.gKr", "IKs": "iks.gKs", "Ito": "ito.gto", "IKur": "ikur.gKur"},
}


def myokit_module():
    # Optional dependency is isolated from the production venv/lock file.
    folder = Path(__file__).resolve().parents[2] / ".simulation-deps"
    if not (folder / "myokit").is_dir():
        raise RuntimeError("Optional ionic engine missing: install Myokit 1.39.2 into .simulation-deps, not the production environment")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import myokit
    if myokit.__version__ != "1.39.2":
        raise RuntimeError("Unverified Myokit version")
    return myokit


@dataclass(frozen=True)
class CellExperiment:
    model: str = "tentusscher-2006"
    cycle_ms: float = 800.
    beats: int = 3
    stimulus_ms: float = .5
    output_ms: float = 1.
    rtol: float = 1e-6
    atol: float = 1e-9
    cell_type: int = 1  # TP06: 0 endo, 1 epi, 2 mid; other models unchanged
    extracellular_mmol_l: dict[str, float] = field(default_factory=dict)
    channel_scales: dict[str, float] = field(default_factory=dict)
    katp_open_fraction: float = 0.
    genetic_variant: str = "reference"
    rhs_backend: str = "python"
    metabolism: dict | None = None

    def validate(self):
        if self.metabolism is not None:
            from .metabolism import validate_config
            validate_config(self.metabolism)
            if self.model != "ohara-2011":
                raise ValueError("Reduced ATP/pH coupling currently requires ORd")
        if self.rhs_backend != "python":
            raise ValueError("Native RHS not released: no sufficient end-to-end gain and Windows LLVM diagnostics in integration")
        if self.model not in MODELS or type(self.beats) is not int or not 1 <= self.beats <= 20:
            raise ValueError("Unknown model or unsafe single-cell budget")
        vals = [self.cycle_ms, self.stimulus_ms, self.output_ms, self.rtol, self.atol]
        if not np.isfinite(vals).all() or not 250 <= self.cycle_ms <= 2500:
            raise ValueError("Invalid pacing cycle")
        if not .1 <= self.stimulus_ms <= 5 or not .1 <= self.output_ms <= 5:
            raise ValueError("Invalid stimulus/output duration")
        if not 1e-9 <= self.rtol <= 1e-4 or not 1e-12 <= self.atol <= 1e-7 or self.cell_type not in (0, 1, 2):
            raise ValueError("Invalid numerical tolerance or cell phenotype")
        bounds = {"potassium_mmol_l": (1.5, 9), "sodium_mmol_l": (100, 180), "ionized_calcium_mmol_l": (.5, 2.5)}
        for name, value in self.extracellular_mmol_l.items():
            if name not in bounds or not np.isfinite(value) or not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError("Invalid extracellular concentration")
        if self.extracellular_mmol_l and self.model not in {"tentusscher-2006", "courtemanche-1998", "ohara-2011"}:
            raise ValueError("Concentration mapping not verified for this cell model")
        for channel, value in self.channel_scales.items():
            if channel not in CHANNELS[self.model] or not np.isfinite(value) or not 0 <= value <= 3:
                raise ValueError("Unsupported channel or multiplier; not a mutation-specific model")
        if (not np.isfinite(self.katp_open_fraction) or not 0 <= self.katp_open_fraction <= .01
                or (self.katp_open_fraction and self.model != "tentusscher-2006")):
            raise ValueError("KATP extension requires TP06 and an opening fraction between 0 and 0.01")
        if self.genetic_variant != "reference":
            from .variants import SAN_VARIANT
            if (self.genetic_variant != SAN_VARIANT or self.model != "fabbri-2017"
                    or {"INa", "INaL"}.intersection(self.channel_scales)):
                raise ValueError("Variant requires the Fabbri SAN model without conflicting Na multipliers")


def compile_model(spec: CellExperiment, vectorized=False, spatial_parameters=()):
    """Translate trusted, hash-pinned expression trees, never arbitrary user code."""
    spec.validate()
    myokit = myokit_module()
    model = myokit.load_model(str(model_path(spec.model)))
    if spec.model == "fabbri-2017":
        # Analytic removable GHK singularities at V=0. Protect the inactive
        # branch too, because NumPy select eagerly evaluates both expressions.
        voltage_name = myokit.Name(model.get("membrane.V"))
        eps = myokit.Number(1e-7, myokit.units.mV)
        condition = myokit.Less(myokit.Abs(voltage_name), eps)
        safe_voltage = myokit.If(condition, eps, voltage_name)
        limits = {
            "ical.IsiCa": "d * f * fCa * (Ca_sub - Ca_o) * P_CaL",
            "ical.IsiK": "d * f * fCa * (K_i - K_o) * P_CaL * 0.000365",
            "ical.IsiNa": "d * f * fCa * (Na_i - Na_o) * P_CaL * 1.85e-5",
            "icat.ICaT": "P_CaT * (calcium.Ca_sub - extra.Ca_o) * d * f",
        }
        for name, expression in limits.items():
            variable = model.get(name)
            original = variable.rhs().clone(subst={voltage_name: safe_voltage})
            limit = myokit.parse_expression(expression, context=variable)
            variable.set_rhs(myokit.If(condition, limit, original))
    if spec.model == "tentusscher-2006":
        model.get("cell.type").set_rhs(spec.cell_type)
    elif spec.model == "ohara-2011":
        model.get("cell.mode").set_rhs(spec.cell_type)
    if spec.katp_open_fraction or "katp_open_fraction" in spatial_parameters:
        if spec.model != "tentusscher-2006":
            raise ValueError("Regional KATP extension requires TP06")
        from .ischemia import add_katp
        add_katp(model, myokit, spec.katp_open_fraction)
    if spec.genetic_variant != "reference":
        from .variants import add_san_1795insd
        add_san_1795insd(model, myokit)
    for channel, scale in spec.channel_scales.items():
        var = model.get(CHANNELS[spec.model][channel])
        var.set_rhs(myokit.Multiply(var.rhs(), myokit.Number(scale)))
    for key, value in spec.extracellular_mmol_l.items():
        names = {"potassium_mmol_l": "extra.Ko", "sodium_mmol_l": "extra.Nao", "ionized_calcium_mmol_l": "extra.Cao"}
        if spec.model == "ohara-2011":
            names = {"potassium_mmol_l": "extra.K_o", "sodium_mmol_l": "extra.Na_o", "ionized_calcium_mmol_l": "extra.Ca_o"}
        name = names[key]
        var = model.get(name)
        var.set_rhs(myokit.Number(value, var.unit()))
    if spec.metabolism is not None:
        if vectorized or spatial_parameters:
            raise ValueError("Reduced energy coupling is single-cell only")
        from .metabolism import add_metabolism
        add_metabolism(model, myokit_module(), spec.metabolism)
    field_bindings = {}
    if spatial_parameters:
        if spec.model != "tentusscher-2006":
            raise ValueError("Regional ionic fields currently verified for TP06 only")
        component = model.add_component("research_fields")
        for parameter in spatial_parameters:
            if parameter == "potassium_mmol_l":
                var = model.get("extra.Ko")
            elif parameter == "katp_open_fraction":
                var = model.get("research_katp.open_fraction")
            elif parameter.startswith("scale:") and parameter[6:] in CHANNELS[spec.model]:
                channel = parameter[6:]
                var = component.add_variable(channel)
                var.set_unit(myokit.units.dimensionless)
                var.set_rhs(1)
                conductance = model.get(CHANNELS[spec.model][channel])
                conductance.set_rhs(myokit.Multiply(conductance.rhs(), myokit.Name(var)))
            else:
                raise ValueError("Unsupported regional field")
            binding = "research_" + var.name()
            var.set_binding(binding)
            field_bindings[binding] = parameter
    model.validate()
    model.check_units(myokit.UNIT_STRICT)
    bindings = {name: var for name, var in model.bindings()}
    if not set(bindings) <= {"time", "pace", "diffusion_current", *field_bindings}:
        raise ValueError("Unsupported model binding")
    factor = float(myokit.Unit.conversion_factor(myokit.units.ms, model.time().unit()))
    states = list(model.states())
    voltage = model.label("membrane_potential")
    if voltage not in states or voltage.unit() != myokit.units.mV:
        raise ValueError("Missing mV membrane state")
    return model, compile_expression_rhs(model, vectorized, field_bindings), states.index(voltage), factor


def compile_expression_rhs(model, vectorized=False, field_bindings=None):
    """Shared AST compiler for pinned ionic and published metabolic models.

    Caller must validate/hash-bind the model; this is not an arbitrary-code loader.
    Model-native time units and stimulus units remain the caller's responsibility.
    """
    myokit = myokit_module()
    from myokit.formats.python import NumPyExpressionWriter, PythonExpressionWriter
    field_bindings = {} if field_bindings is None else field_bindings
    states = list(model.states())
    variables = list(model.variables(deep=True))
    names = {v: f"v_{i}" for i, v in enumerate(variables)}

    def lhs(expression):
        return ("d_" if isinstance(expression, myokit.Derivative) else "") + names[expression.var()]

    writer = NumPyExpressionWriter() if vectorized else PythonExpressionWriter()
    writer.set_lhs_function(lhs)
    constant_writer = PythonExpressionWriter()
    constant_writer.set_lhs_function(lhs)
    lines = ["def rhs(t, y, pace, fields=None):"]
    for i, var in enumerate(states):
        lines.append(f"    {names[var]} = y[{i}]")
    for equations in model.solvable_order().values():
        for equation in equations:
            binding = equation.lhs.var().binding()
            if binding:
                value = (f"fields[{field_bindings[binding]!r}]" if binding in field_bindings
                         else {"time": "t", "pace": "pace", "diffusion_current": "0.0"}[binding])
                lines.append(f"    {lhs(equation.lhs)} = {value}")
            else:
                # Constant conditional branches must stay lazy. NumPy select
                # eagerly evaluates inactive singular expressions (e.g. SAN ACh=0).
                selected_writer = constant_writer if vectorized and equation.rhs.is_constant() else writer
                lines.append("    " + selected_writer.eq(equation))
    lines.append("    return [" + ", ".join("d_" + names[v] for v in states) + "]")
    scope = {"math": math, "numpy": np, "__builtins__": {"abs": abs, "min": min, "max": max}}
    exec(compile("\n".join(lines), "<pinned-myokit-expression-trees>", "exec"), scope)
    return scope["rhs"]


def cycle_state_pattern(end_states, tolerance=.001):
    """Flag multi-cycle candidates without calling every non-P1 trace a failure.

    Requires four repeats (three differences) of every phase. End-state agreement
    alone is not a clinical alternans/TWA measurement or proof of a limit cycle.
    """
    states = np.asarray(end_states, float)
    if states.ndim != 2 or not np.isfinite(states).all() or not 0 < tolerance < 1:
        raise ValueError("Finite end-cycle states and positive numerical tolerance required")
    checked = {}
    for period in (1, 2, 3, 4):
        if len(states) < 4*period:
            continue
        tail = states[-4*period:].reshape(4, period, -1)
        residual = np.max(np.abs(tail[1:]-tail[:-1]) / (1e-6+np.maximum(np.abs(tail[1:]), np.abs(tail[:-1]))))
        checked[str(period)] = float(residual)
        if residual < tolerance:
            return {"status": "single_cycle_candidate" if period == 1 else "multicycle_candidate",
                    "period_beats": period, "residuals": checked, "clinical_alternans": False}
    return {"status": "unresolved_or_drifting" if len(states) >= 4 else "insufficient_cycles",
            "period_beats": None, "residuals": checked, "clinical_alternans": False}


@lru_cache(maxsize=2)
def _compiled_cell(model_name, cell_type, scales, concentrations, katp, variant, backend, metabolic_json=None):
    """Two per-worker kernels only: reuse across pacing chunks, no unbounded JIT cache."""
    import json
    spec = CellExperiment(model=model_name, cell_type=cell_type, channel_scales=dict(scales),
                          extracellular_mmol_l=dict(concentrations), katp_open_fraction=katp,
                          genetic_variant=variant, rhs_backend=backend,
                          metabolism=json.loads(metabolic_json) if metabolic_json is not None else None)
    model, rhs, vi, factor = compile_model(spec)
    return model, rhs, vi, factor


def simulate_cell(spec: CellExperiment, progress=None, cancelled=None, initial_state=None):
    started = time.monotonic()
    spec.validate()
    import json
    model, rhs, voltage_index, factor = _compiled_cell(
        spec.model, spec.cell_type, tuple(sorted(spec.channel_scales.items())),
        tuple(sorted(spec.extracellular_mmol_l.items())), spec.katp_open_fraction, spec.genetic_variant, spec.rhs_backend,
        json.dumps(spec.metabolism, sort_keys=True, allow_nan=False) if spec.metabolism is not None else None)
    published_state = np.asarray(model.initial_values(as_floats=True), float)
    state = published_state if initial_state is None else np.asarray(initial_state, float).copy()
    if state.shape != published_state.shape or not np.isfinite(state).all():
        raise ValueError("Invalid continuation state")
    initial_state_hash = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
    paced = model.binding("pace") is not None
    duration = spec.cycle_ms * spec.beats
    output = np.unique(np.r_[np.arange(0, duration, spec.output_ms), duration])
    result = np.empty((len(output), len(state)))
    boundaries = {0., duration}
    for i in range(spec.beats):
        boundaries.add(i * spec.cycle_ms)
        boundaries.add((i + 1) * spec.cycle_ms)
        if paced:
            boundaries.add(i * spec.cycle_ms + spec.stimulus_ms)
    boundaries = sorted(boundaries)
    previous = state.copy()
    end_states = [state.copy()]
    periodic_residuals = []
    calls = 0

    def derivative(t, y, pace):
        nonlocal calls
        calls += 1
        if calls % 100 == 0 and cancelled and cancelled():
            raise InterruptedError("Ionic assay cancelled")
        return factor * np.asarray(rhs(t * factor, y, pace))

    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if cancelled and cancelled():
            raise InterruptedError("Ionic assay cancelled")
        phase = ((a + b) / 2) % spec.cycle_ms
        pace = float(paced and phase < spec.stimulus_ms)
        fit = solve_ivp(lambda t, y: derivative(t, y, pace), (a, b), state,
                        method="BDF", rtol=spec.rtol, atol=spec.atol, max_step=5., dense_output=True)
        if not fit.success or not np.isfinite(fit.y).all():
            raise RuntimeError("Ionic integration failed: " + fit.message)
        mask = (output >= a) & (output <= b)
        result[mask] = fit.sol(output[mask]).T
        state = fit.y[:, -1]
        if abs(b / spec.cycle_ms - round(b / spec.cycle_ms)) < 1e-8:
            periodic_residuals.append(float(np.max(np.abs(state - previous) / (1e-6 + np.maximum(np.abs(state), np.abs(previous))))))
            previous = state.copy()
            end_states.append(state.copy())
        if progress:
            progress(b, duration, time.monotonic() - started)
    voltage = result[:, voltage_index]
    if not np.isfinite(result).all() or voltage.min() < -200 or voltage.max() > 150:
        raise RuntimeError("Unphysical or incomplete single-cell trajectory; not clipped")
    metadata = {"schema": "ionic-cell-assay/1", "clinical_use": False, "experiment": asdict(spec),
                "model_commit": COMMIT, "model_sha256": MODELS[spec.model], "myokit_version": "1.39.2",
                "numerical_regularizations": ["Fabbri GHK analytic V=0 limits, protected eager branches"] if spec.model == "fabbri-2017" else [],
                "initial_state_sha256": initial_state_hash,
                "initialization": "published" if initial_state is None else "explicit_continuation_state",
                "solver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "state_names": [v.qname() for v in model.states()], "state_units": [str(v.unit()) for v in model.states()],
                "solver": "scipy.solve_ivp BDF; pacing edges are integration boundaries", "rhs_calls": calls,
                "voltage_unit": "mV", "time_unit": "ms", "native_time_per_ms": factor,
                "paced": paced, "periodic_residuals": periodic_residuals,
                "periodicity_check_passed": paced and len(periodic_residuals) >= 3 and max(periodic_residuals[-3:]) < .001,
                "cycle_pattern": cycle_state_pattern(end_states) if paced else {"status": "not_paced", "clinical_alternans": False},
                "elapsed_seconds": time.monotonic() - started,
                "limitations": ["Single cell, no torso, ECG or patient-specific block location.",
                                "Initialization provenance recorded; no assumption of steady state.",
                                "Plasma electrolytes used as a scenario, not verified myocardial concentration.",
                                "Periodic numerical agreement does not validate clinical physiology."]}
    import scipy
    if spec.metabolism is not None:
        from .metabolism import balance_report
        metadata["metabolism"] = balance_report(result, metadata["state_names"], spec.metabolism)
        metadata["metabolism_source_sha256"] = hashlib.sha256(Path(__file__).with_name("metabolism.py").read_bytes()).hexdigest()
    metadata["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version}
    if spec.katp_open_fraction:
        from .ischemia import mechanism_metadata
        metadata["katp"] = mechanism_metadata(spec.katp_open_fraction)
        metadata["katp_source_sha256"] = hashlib.sha256(Path(__file__).with_name("ischemia.py").read_bytes()).hexdigest()
    if spec.genetic_variant != "reference":
        from .variants import variant_metadata
        metadata["variant"] = variant_metadata()
        metadata["variant_source_sha256"] = hashlib.sha256(Path(__file__).with_name("variants.py").read_bytes()).hexdigest()
    return output, voltage, result, metadata


def simulate_preconditioned(spec, max_beats=300, cancelled=None, checkpoint=None, resume=None):
    """Bounded adaptive pre-pacing; stores only small chunks and optional checkpoints.

    Three consecutive end-cycle state residuals <0.001 are a numerical periodicity
    criterion, not physiological/clinical validation. Nonconvergence is explicit.
    """
    from dataclasses import replace
    if spec.metabolism is not None:
        raise ValueError("Transient metabolic balances include cumulative fluxes; P1 preconditioning is not defined")
    if type(max_beats) is not int or not 10 <= max_beats <= 1000:
        raise ValueError("Pre-pacing requires a bounded 10–1000 beat budget")
    count, state, converged = 0, None, False
    if resume is not None:
        count, state = int(resume["beats"]), np.asarray(resume["state"], float)
        if not 0 <= count <= max_beats:
            raise ValueError("Invalid pre-pacing checkpoint")
        converged = bool(resume.get("converged", False))
    began = time.monotonic()
    last_residuals = []
    while count < max_beats and not converged:
        chunk = min(10, max_beats-count)
        _, _, states, meta = simulate_cell(replace(spec, beats=chunk, output_ms=5.),
                                          cancelled=cancelled, initial_state=state)
        state = states[-1].copy()
        count += chunk
        last_residuals = meta["periodic_residuals"]
        converged = bool(meta["periodicity_check_passed"])
        if checkpoint:
            checkpoint({"beats": count, "state": state, "converged": converged})
    times, voltage, states, meta = simulate_cell(spec, cancelled=cancelled, initial_state=state)
    meta["preconditioning"] = {"beats": count, "maximum_beats": max_beats,
                                "converged": converged and bool(meta["periodicity_check_passed"]),
                                "last_prepace_residuals": last_residuals,
                                "criterion": "all states relative end-cycle residual <0.001 for three consecutive cycles",
                                "wall_seconds_this_attempt": time.monotonic()-began}
    return times, voltage, states, meta
