"""Bounded 3-D tetrahedral monodomain research kernel with human ionic kinetics.

Linear FEM, lumped mass, symmetric fiber diffusivity in mm^2/ms. Adaptive
coupled BDF integrates voltage AND ionic states; no prescribed activation map.
No torso/bidomain, mechanical contraction, personalized fibers or clinical claims.
"""
from dataclasses import asdict, dataclass
import hashlib
import itertools
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.csgraph import connected_components

from .ionic import CellExperiment, compile_model
from .model_assets import MODELS


@dataclass
class TetraMesh:
    xyz_mm: np.ndarray
    tetra: np.ndarray
    fibers: np.ndarray  # one unit direction per element
    source: str = "synthetic_test_tissue"

    def validate(self):
        x, t, f = self.xyz_mm, self.tetra, self.fibers
        if x.ndim != 2 or x.shape[1] != 3 or not 4 <= len(x) <= 2000:
            raise ValueError("Research mesh requires 4–2000 nodes")
        if t.ndim != 2 or t.shape[1] != 4 or not 1 <= len(t) <= 15000 or t.dtype.kind not in "iu":
            raise ValueError("Integer tetrahedral connectivity required")
        if not np.isfinite(x).all() or np.abs(x).max() > 2000 or t.min() < 0 or t.max() >= len(x):
            raise ValueError("Invalid mm coordinates or node indices")
        if f.shape != (len(t), 3) or not np.isfinite(f).all() or not np.allclose(np.linalg.norm(f, axis=1), 1., atol=1e-5):
            raise ValueError("Unit element fiber vectors required")
        if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
            raise ValueError("Duplicate tetrahedron")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Geometry provenance required")

    def fingerprint(self):
        self.validate()
        h = hashlib.sha256(self.source.encode())
        for a in (self.xyz_mm, self.tetra, self.fibers):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class StimulusEvent:
    onset_ms: float
    duration_ms: float
    nodes: tuple[int, ...]
    multiplier: float = 1.

    def validate(self, n, duration):
        values = [self.onset_ms, self.duration_ms, self.multiplier]
        if (not np.isfinite(values).all() or not 0 <= self.onset_ms < duration
                or not .1 <= self.duration_ms <= 5 or not 0 < self.multiplier <= 3):
            raise ValueError("Invalid programmed stimulus")
        if not self.nodes or any(type(i) is not int or not 0 <= i < n for i in self.nodes):
            raise ValueError("Programmed stimulus needs valid node indices")


def block_mesh(shape=(7, 3, 3), spacing_mm=1.):
    if len(shape) != 3 or any(type(n) is not int or not 2 <= n <= 15 for n in shape) or not .2 <= spacing_mm <= 2:
        raise ValueError("Invalid bounded demonstration mesh")
    indices = np.arange(np.prod(shape)).reshape(shape)
    xyz = np.array(list(np.ndindex(shape)), float) * spacing_mm
    tets = []
    for base in np.ndindex(tuple(n-1 for n in shape)):
        for permutation in itertools.permutations(range(3)):
            point = np.array(base)
            corners = [indices[tuple(point)]]
            for axis in permutation:
                point = point.copy()
                point[axis] += 1
                corners.append(indices[tuple(point)])
            tets.append(corners)
    return TetraMesh(xyz, np.array(tets, int), np.tile([1., 0., 0.], (len(tets), 1)))


def load_mesh(path):
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 16 * 1024**2:
            raise ValueError("Mesh archive exceeds budget")
    with np.load(path, allow_pickle=False) as data:
        if str(data["units"]) != "mm" or str(data["coordinate_system"]) != "x_left_y_posterior_z_cranial":
            raise ValueError("Explicit shared mm/anatomical coordinate convention required")
        mesh = TetraMesh(data["xyz_mm"].copy(), data["tetra"].copy(), data["fibers"].copy(), str(data["source"]))
    mesh.validate()
    return mesh


def assemble(mesh, d_long=.1, d_trans=.025, element_coupling_scale=None):
    mesh.validate()
    if not np.isfinite([d_long, d_trans]).all() or not .001 <= d_trans <= d_long <= 1.:
        raise ValueError("Diffusivity must satisfy .001 <= transverse <= longitudinal <= 1 mm²/ms")
    points = mesh.xyz_mm[mesh.tetra]
    basis = np.concatenate([np.ones((*points.shape[:2], 1)), points], axis=2)
    volume = np.abs(np.linalg.det(basis)) / 6
    if np.any(volume < 1e-8) or np.any(np.linalg.cond(basis) > 1e8):
        raise ValueError("Degenerate/ill-conditioned tetrahedra")
    gradient = np.linalg.inv(basis)[:, 1:, :]
    diffusion = d_trans * np.eye(3)[None, :, :] + (d_long-d_trans) * np.einsum("ei,ej->eij", mesh.fibers, mesh.fibers)
    element = volume[:, None, None] * np.einsum("eia,eij,ejb->eab", gradient, diffusion, gradient)
    if element_coupling_scale is not None:
        scale = np.asarray(element_coupling_scale, float)
        if scale.shape != (len(volume),) or not np.isfinite(scale).all() or np.any((scale < 0) | (scale > 1)):
            raise ValueError("Invalid regional coupling")
        element *= scale[:, None, None]
    rows = np.repeat(mesh.tetra, 4, axis=1).ravel()
    cols = np.tile(mesh.tetra, (1, 4)).ravel()
    n = len(mesh.xyz_mm)
    stiffness = coo_matrix((element.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    mass = np.bincount(mesh.tetra.ravel(), weights=np.repeat(volume/4, 4), minlength=n)
    if np.any(mass <= 0) or (element_coupling_scale is None and connected_components(stiffness, directed=False)[0] != 1):
        raise ValueError("Unused node or disconnected tissue; isolate components explicitly")
    lap = -diags(1/mass) @ stiffness
    return lap.tocsr(), mass, stiffness


def simulate_monodomain(mesh, spec=None, duration_ms=60., d_long=.1, d_trans=.025,
                        stimulus_nodes=None, progress=None, cancelled=None, events=None, substrate=None):
    spec = spec or CellExperiment(beats=1)
    if spec.model not in {"tentusscher-2006", "courtemanche-1998", "stewart-2009"}:
        raise ValueError("This bounded tissue kernel requires a paced model")
    if not np.isfinite(duration_ms) or not 5 <= duration_ms <= 1000:
        raise ValueError("Tissue duration exceeds research budget")
    n = len(mesh.xyz_mm)
    fields = substrate.fields(n, len(mesh.tetra)) if substrate is not None else {}
    model, rhs, vi, factor = compile_model(spec, vectorized=True, spatial_parameters=tuple(fields))
    lap, mass, stiffness = assemble(mesh, d_long, d_trans,
                                    substrate.element_coupling_scale if substrate is not None else None)
    states = len(list(model.states()))
    initial = np.repeat(np.array(model.initial_values(as_floats=True))[:, None], n, axis=1)
    stim = np.flatnonzero(mesh.xyz_mm[:, 0] == mesh.xyz_mm[:, 0].min()) if stimulus_nodes is None else np.asarray(stimulus_nodes)
    if stim.ndim != 1 or stim.dtype.kind not in "iu" or not len(stim) or stim.min() < 0 or stim.max() >= n:
        raise ValueError("Invalid stimulus node set")
    if events is None:
        events = [StimulusEvent(float(a), spec.stimulus_ms, tuple(map(int, stim)))
                  for a in np.arange(spec.beats)*spec.cycle_ms if a < duration_ms]
    if not 1 <= len(events) <= 64:
        raise ValueError("One to 64 programmed stimulus events required")
    for event in events:
        event.validate(n, duration_ms)
    times = np.unique(np.r_[np.arange(0, duration_ms, spec.output_ms), duration_ms])
    if len(times) * n * states * 8 > 128 * 1024**2:
        raise ValueError("Tissue ionic-state/output budget exceeded")
    frames = np.empty((len(times), n), np.float32)
    # Safe structural superset of Jacobian: ionic states couple within each node;
    # only voltage diffuses between neighboring nodes.
    pattern = kron(np.ones((states, states)), eye(n, format="csr"), format="lil")
    pattern[vi*n:(vi+1)*n, vi*n:(vi+1)*n] = (stiffness != 0) + eye(n, format="csr")
    pattern = pattern.tocsr()
    boundaries = {0., float(duration_ms)}
    for event in events:
        boundaries.update([float(event.onset_ms), min(float(event.onset_ms+event.duration_ms), duration_ms)])
    boundaries = sorted(boundaries)
    started, calls = time.monotonic(), 0

    def derivative(t, y, pace):
        nonlocal calls
        calls += 1
        if calls > 300000:
            raise RuntimeError("Coupled solver RHS budget exceeded")
        if calls % 100 == 0:
            if cancelled and cancelled():
                raise InterruptedError("Tissue simulation cancelled")
            if progress:
                progress(t, duration_ms, time.monotonic()-started)
        y = y.reshape(states, n)
        # NumPy select evaluates inactive branches, unlike scalar Python if.
        # Only masked singular branches may be ignored; selected RHS must be finite.
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.array([np.broadcast_to(v, (n,)) for v in rhs(t*factor, y, pace, fields)]) * factor
        out[vi] += lap @ y[vi]
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite ionic/FEM derivative")
        return out.ravel()

    state = initial.ravel()
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if cancelled and cancelled():
            raise InterruptedError("Tissue simulation cancelled")
        pace = np.zeros(n)
        for event in events:
            if event.onset_ms <= (a+b)/2 < event.onset_ms+event.duration_ms:
                pace[list(event.nodes)] += event.multiplier
        if pace.max() > 3:
            raise ValueError("Overlapping stimulus amplitude exceeds budget")
        mask = (times >= a) & (times <= b)
        requested = np.unique(np.r_[times[mask], b])
        fit = solve_ivp(lambda t, y: derivative(t, y, pace), (a, b), state, method="BDF",
                        rtol=spec.rtol, atol=spec.atol, max_step=2., jac_sparsity=pattern,
                        t_eval=requested, dense_output=False)
        if not fit.success:
            raise RuntimeError("Coupled solver failed: " + fit.message)
        selected = np.searchsorted(requested, times[mask])
        frames[mask] = fit.y.reshape(states, n, -1)[vi][:, selected].T
        state = fit.y[:, -1]
    if not np.isfinite(frames).all() or frames.min() < -200 or frames.max() > 150:
        raise RuntimeError("Invalid tissue voltage; not clipped")
    if progress:
        progress(duration_ms, duration_ms, time.monotonic()-started)
    meta = {"schema": "ionic-monodomain/1", "clinical_use": False, "model": spec.model,
            "cell_experiment": asdict(spec), "duration_ms": duration_ms,
            "solver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "ionic_source_sha256": hashlib.sha256(Path(__file__).with_name("ionic.py").read_bytes()).hexdigest(),
            "model_sha256": MODELS[spec.model], "mesh_sha256": mesh.fingerprint(), "geometry_source": mesh.source,
            "units": {"time": "ms", "voltage": "mV", "distance": "mm", "diffusivity": "mm^2/ms"},
            "nodes": n, "tetrahedra": len(mesh.tetra), "volume_mm3": float(mass.sum()),
            "d_long": d_long, "d_trans": d_trans, "stimulus_nodes": stim.tolist(), "rhs_calls": calls,
            "programmed_events": [asdict(e) for e in events],
            "regional_fields": {k: v.tolist() for k, v in fields.items()},
            "regional_coupling": (np.asarray(substrate.element_coupling_scale).tolist()
                                  if substrate is not None and substrate.element_coupling_scale is not None else None),
            "substrate_provenance": substrate.provenance if substrate is not None else None,
            "elapsed_seconds": time.monotonic()-started, "mesh_convergence_proven": False,
            "limitations": ["Bounded single-model tissue experiment with optional regional parameters, not a patient whole heart.",
                            "No torso/bidomain, mechanical coupling or inferred clinical diagnosis.",
                            "No automatic demographic mapping into diffusivity or channels."]}
    import scipy
    meta["versions"] = {"numpy": np.__version__, "scipy": scipy.__version__}
    if spec.katp_open_fraction or "katp_open_fraction" in fields:
        from .ischemia import mechanism_metadata
        meta["katp"] = mechanism_metadata(spec.katp_open_fraction, "katp_open_fraction" in fields)
        meta["katp_source_sha256"] = hashlib.sha256(Path(__file__).with_name("ischemia.py").read_bytes()).hexdigest()
    return times, frames, meta
