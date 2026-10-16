"""Deterministic reaction-diffusion experiments, outputs and bounded inverse fitting.

No activation-time warping, no normalization to the target QRS duration, no
conversion of fit residuals into disease probabilities. Graph resolution and
the homogeneous infinite conductor limit the interpretation of every output.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from . import MODEL_VERSION
from .model import Experiment, Tissue, experiment_hash, operator

LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LIMITATIONS = [
    "Forschungsmodell; keine Diagnose, kein Ortsnachweis, keine Behandlungsempfehlung.",
    "Mitchell–Schaeffer: normiertes Potential und Erholungstor, keine expliziten Na/K/Ca-Ionenströme.",
    "Generisches grobes 3D-Graphnetz; keine räumlich konvergierte Monodomain-FEM.",
    "Vorhöfe als Oberflächennetze; Kammern als Schalenvolumen; Leitungssystem als approximative 1D-Kabel.",
    "Phänotypparameter und Fasern sind Hypothesen, keine validierten menschlichen Zelltypen.",
    "Homogener unendlicher Leiter; keine Lungen, Thoraxgrenzen oder kalibrierte mV-Amplitude.",
    "Intrakardiale Signale besonders geometrie-/elektrodenabhängig; keine klinischen EGM-Schwellen.",
    "Keine Mechanik, Hämodynamik, autonome Sinusknotenautomatie oder quantitative Elektrolytdosisantwort.",
]


@dataclass
class Result:
    times_ms: np.ndarray
    voltage: np.ndarray
    activation_ms: np.ndarray
    potentials: np.ndarray
    ecg: np.ndarray
    lead_names: tuple[str, ...]
    metadata: dict


def derive_leads(potential, names):
    p = {k: potential[:, i] for i, k in enumerate(names)}
    ra, la, ll = (p[k] for k in ("RA", "LA", "LL"))
    wct = (ra + la + ll) / 3
    leads = {"I": la - ra, "II": ll - ra, "III": ll - la,
             "aVR": ra - (la + ll) / 2, "aVL": la - (ra + ll) / 2, "aVF": ll - (ra + la) / 2}
    leads.update({k: v - wct for k, v in p.items() if k not in {"RA", "LA", "LL"}})
    if "CathA" in p and "CathB" in p:
        leads["CathA-CathB"] = p["CathA"] - p["CathB"]
    return np.column_stack(list(leads.values())), tuple(leads)


def reaction(v, h, tau_close):
    """MS-2003 equations (ms), reference tau_in=.3, tau_out=6, tau_open=120."""
    return h * v * v * (1. - v) / .3 - v / 6., np.where(v < .13, (1. - h) / 120., -h / tau_close)


def simulate(tissue: Tissue, spec: Experiment, progress=None, cancelled=None) -> Result:
    spec.validate()
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("__init__.py", "model.py", "solver.py")}
    if spec.stimulus_site not in tissue.sites:
        raise ValueError("Unknown stimulation site")
    lap, close, live, degree = operator(tissue, spec)
    nodes = len(tissue.xyz)
    nframes = int(np.floor(spec.duration_ms / spec.output_step_ms)) + 1
    if nodes * nframes * 4 > 256 * 1024 ** 2:
        raise ValueError("Frame storage exceeds 256 MiB; shorten recording or increase output step")
    # Conservative explicit diffusion bound plus reaction-step bound; do not clip unstable voltages.
    dt_limit = min(spec.max_dt_ms, .03, .45 / max(float(degree.max()), 1e-12))
    substeps = int(np.ceil(spec.output_step_ms / dt_limit))
    dt = spec.output_step_ms / substeps
    if nframes * substeps > 2_000_000:
        raise ValueError("Time-step budget exceeded by mesh or parameters")
    times = np.arange(nframes) * spec.output_step_ms
    v = np.zeros(nodes)
    h = np.ones(nodes)
    frames = np.empty((nframes, nodes), np.float32)
    frames[0] = v
    activation = np.full(nodes, np.nan)
    selected = np.asarray(tissue.sites[spec.stimulus_site], int)
    selected = selected[live[selected]]
    if not len(selected):
        raise ValueError("Stimulus lies entirely in non-excitable tissue")
    onsets = np.arange(2., spec.duration_ms, spec.cycle_ms)
    if spec.s2_ms is not None:
        onsets = np.sort(np.r_[onsets, spec.s2_ms])
    starts = np.rint(onsets / dt).astype(int)
    ends = starts + max(1, int(round(spec.stimulus_ms / dt)))
    potentials = np.empty((nframes, len(spec.electrodes)))
    # Conservative discrete sources -L*v with common regularized 1/r kernel.
    # This is a pseudo-ECG in arbitrary units; there is no claimed physical mV scale.
    electrode_xyz = np.asarray([e.xyz_mm for e in spec.electrodes])
    distance = np.linalg.norm(electrode_xyz[:, None, :] - tissue.xyz[None, :, :], axis=2)
    if np.any(distance.min(axis=1) < .5):
        raise ValueError("Electrode closer than 0.5 mm to a source node; move it or refine the model")
    kernel = 1. / (4 * np.pi * distance)
    leadfield = -(lap.T @ kernel.T).T
    potentials[0] = 0
    began = time.monotonic()
    for frame in range(1, nframes):
        if cancelled is not None and cancelled():
            raise InterruptedError("Simulation cancelled; no complete result published")
        for k in range(substeps):
            step = (frame - 1) * substeps + k
            dv, dh = reaction(v, h, close)
            dv += lap @ v
            if np.any((starts <= step) & (step < ends)):
                dv[selected] += spec.stimulus_strength
            vn = v + dt * dv * live
            h += dt * dh
            crossing = (v < .5) & (vn >= .5) & np.isnan(activation)
            activation[crossing] = (step + (.5 - v[crossing]) / (vn[crossing] - v[crossing])) * dt
            v = vn
        if not np.isfinite(v).all() or v.min() < -.01 or v.max() > 2 or h.min() < 0 or h.max() > 1.00001:
            raise ArithmeticError("Unstable state; no clipped substitute returned")
        frames[frame] = v
        potentials[frame] = leadfield @ v
        if progress and (frame % 10 == 0 or frame == nframes - 1):
            progress(frame, nframes - 1, time.monotonic() - began)
    ecg, names = derive_leads(potentials, [e.name for e in spec.electrodes])
    metadata = {"schema": "cardiac-simulation/1", "model_version": MODEL_VERSION,
                "source_hashes": source_hashes, "numpy_version": np.__version__,
                "initial_state": "u=0, h=1; not a paced steady state",
                "scar_definition": "non-excitable AND electrically insulating, not passive conducting fibrosis",
                "experiment": spec.to_dict(), "experiment_sha256": experiment_hash(spec, tissue),
                "geometry_sha256": tissue.fingerprint(), "geometry_kind": tissue.geometry_kind,
                "dt_ms": dt, "n_nodes": nodes, "n_edges": len(tissue.edges),
                "computed_duration_ms": float(times[-1]),
                "signal_units": "a.u.", "voltage_units": "normalized", "synthetic": True,
                "clinical_use": False, "runtime_seconds": time.monotonic() - began,
                "activated_fraction": float(np.isfinite(activation[live]).mean()),
                "activation_definition": "first upward crossing of normalized voltage 0.5",
                "limitations": LIMITATIONS}
    return Result(times, frames, activation, potentials, ecg, names, metadata)


def save_result(path: Path, result: Result, tissue: Tissue):
    """Atomic NPZ + self-contained provenance; caller chooses isolated output directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, times_ms=result.times_ms, voltage=result.voltage,
                            activation_ms=result.activation_ms, potentials=result.potentials,
                            ecg=result.ecg, lead_names=np.asarray(result.lead_names), xyz=tissue.xyz,
                            cells=tissue.cells, regions=tissue.regions, edges=tissue.edges,
                            metadata=np.asarray(json.dumps(result.metadata, allow_nan=False)))
    tmp.replace(path)


def load_result(path):
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 512 * 1024 ** 2:
            raise ValueError("Simulation archive exceeds the 512 MiB reader budget")
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        if (not isinstance(metadata, dict) or metadata.get("schema") != "cardiac-simulation/1"
                or metadata.get("synthetic") is not True or metadata.get("clinical_use") is not False):
            raise ValueError("Incompatible simulation result")
        data = {k: z[k].copy() for k in z.files if k != "metadata"} | {"metadata": metadata}
    frames, nodes = data["voltage"].shape
    if (data["xyz"].shape != (nodes, 3) or data["cells"].shape != (nodes,)
            or data["activation_ms"].shape != (nodes,) or data["times_ms"].shape != (frames,)
            or frames < 2 or np.any(np.diff(data["times_ms"]) <= 0)
            or data["ecg"].shape != (frames, len(data["lead_names"]))
            or np.any((data["cells"] < 0) | (data["cells"] > 5))
            or data["edges"].ndim != 2 or data["edges"].shape[1] != 2
            or np.any((data["edges"] < 0) | (data["edges"] >= nodes))):
        raise ValueError("Inconsistent simulation array shapes or indices")
    for key in ("voltage", "xyz", "ecg", "times_ms"):
        if not np.isfinite(data[key]).all():
            raise ValueError("Non-finite simulation array: " + key)
    return data


def export_lab_signal(result_path, output_path, gain_mv_per_unit):
    """Explicit illustrative scale; never normalize amplitudes secretly to a patient ECG."""
    if not np.isfinite(gain_mv_per_unit) or gain_mv_per_unit <= 0:
        raise ValueError("Explicit positive mV/a.u. scale required")
    data = load_result(result_path)
    names = list(data["lead_names"])
    if not set(LEADS_12) <= set(names):
        raise ValueError("12 standard leads required for the ECG lab")
    signal = data["ecg"][:, [names.index(k) for k in LEADS_12]] * gain_mv_per_unit
    provenance = data["metadata"] | {"illustrative_gain_mv_per_unit": gain_mv_per_unit,
                                     "amplitude_calibrated": False,
                                     "result_sha256": hashlib.sha256(Path(result_path).read_bytes()).hexdigest()}
    output_path = Path(output_path)
    # Export uses the native lab NPZ interchange schema; companion JSON retains scientific detail.
    with output_path.open("xb") as f:
        np.savez_compressed(f, ecg=signal, sampling_rate_hz=1000 / np.diff(data["times_ms"])[0],
                            lead_names=np.asarray(LEADS_12), units=np.asarray(["mV"] * 12), synthetic=True,
                            simulation_provenance=np.asarray(json.dumps(provenance)))
    output_path.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output_path


def fit_candidates(tissue, specs, target, target_fs, target_names, progress=None, cancelled=None):
    """Bounded, aligned experiment comparison. No time warping or patient posterior.

    Target must already be an explicitly selected, time-aligned segment. A single
    positive amplitude scale is fitted on alternating time samples; remaining
    samples provide a numerical check only, NOT independent clinical validation.
    """
    if not 1 <= len(specs) <= 8:
        raise ValueError("Inverse search budget: 1–8 explicit candidates")
    target = np.asarray(target, float)
    if (target.ndim != 2 or len(target) < 10 or target.shape[1] != len(target_names)
            or not np.isfinite(target).all() or not np.isfinite(target_fs) or target_fs <= 0):
        raise ValueError("Invalid aligned target")
    rows = []
    for i, spec in enumerate(specs):
        if cancelled and cancelled():
            raise InterruptedError("Search cancelled")
        result = simulate(tissue, spec, cancelled=cancelled)
        shared = [k for k in ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
                  if k in target_names and k in result.lead_names]
        if len(shared) < 3:
            raise ValueError("At least three independent shared leads required")
        t = result.times_ms / 1000
        if t[-1] > (len(target) - 1) / target_fs + 1e-9:
            raise ValueError("Target does not cover the full experiment; no extrapolation")
        observed = np.column_stack([np.interp(t, np.arange(len(target)) / target_fs,
                                              target[:, list(target_names).index(k)]) for k in shared])
        predicted = result.ecg[:, [result.lead_names.index(k) for k in shared]]
        train = np.arange(len(t)) % 2 == 0
        x = predicted - predicted[train].mean(axis=0)
        y = observed - observed[train].mean(axis=0)
        denom = float(np.sum(x[train] ** 2))
        if denom < 1e-18 or np.sum(y[~train] ** 2) < 1e-18:
            rows.append({"preset": spec.preset, "status": "non_estimable", "reason": "constant signal"})
            if progress:
                progress(i + 1, len(specs), 0.)
            continue
        gain = max(0., float(np.sum(x[train] * y[train]) / denom))
        residual = float(np.sum((gain * x[~train] - y[~train]) ** 2) / np.sum(y[~train] ** 2))
        rows.append({"preset": spec.preset, "status": "research_fit", "relative_squared_error": residual,
                     "gain": gain, "experiment_sha256": result.metadata["experiment_sha256"],
                     "experiment": spec.to_dict(), "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                     "target_fs": target_fs, "shared_leads": shared, "clinical_validation": False})
        if progress:
            progress(i + 1, len(specs), 0.)
    return sorted(rows, key=lambda r: r.get("relative_squared_error", float("inf")))
