"""Reduced excitable-tissue model and explicit experiment contracts.

Mitchell–Schaeffer kinetics, not human ion-channel or mechanical equations.
The anatomical adapter uses a coarse generic graph, not a converged FEM mesh.
Coordinates: mm; +x patient-left, +y posterior, +z cranial. Time: ms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import MODEL_VERSION

CELL_NAMES = ("Endokard", "Arbeitsmyokard", "Epikard", "Vorhof", "AV-Übergang", "His/Purkinje")
# These are phenomenological experiment defaults, NOT fitted human cell classes.
TAU_CLOSE = np.array([150., 150., 130., 95., 150., 150.])
DIFFUSION = np.array([0.8, 0.8, 0.8, 0.8, 0.08, 3.2])
PRESETS = {
    "reference": "Referenz · generisches Gewebe",
    "scar": "Isolierende Narbe · nicht erregbare Region",
    "slow_zone": "Langsame Grenzzone · reduzierte Kopplung",
    "short_ap": "Verkürzte Repolarisation · Mechanismusversuch",
    "long_ap": "Verlängerte Repolarisation · Mechanismusversuch",
    "lbbb": "Linker Leitungsast unterbrochen · Modellhypothese",
    "rbbb": "Rechter Leitungsast unterbrochen · Modellhypothese",
}
UNSUPPORTED = {
    "ischemia": "Benötigt validiertes Ionenmodell: K außen, ATP/KATP, Azidose und Grenzzone.",
    "electrolytes": "Keine Abbildung mmol/l → MS-Parameter; Na/K/Ca/Mg nicht quantitativ berechenbar.",
    "hypertrophy": "Benötigt gemessene Wandgeometrie und Faserstruktur; keine reine Spannungsverstärkung.",
    "heart_failure": "Benötigt Remodeling- und gegebenenfalls elektromechanisches Modell; keine EF aus Animation.",
}


@dataclass(frozen=True)
class Electrode:
    name: str
    xyz_mm: tuple[float, float, float]
    kind: str = "surface"  # surface | intracardiac


def electrode_preset(name="torso_12") -> tuple[Electrode, ...]:
    """Illustrative coordinates, not patient anatomy or verified intercostal sites."""
    if name not in {"torso_12", "extended", "intracardiac"}:
        raise ValueError("Unknown electrode configuration")
    points = {"RA": (-160., 0., 150.), "LA": (160., 0., 150.), "LL": (100., 0., -230.),
              "V1": (-48., -95., 22.), "V2": (-20., -95., 22.), "V3": (5., -94., 5.),
              "V4": (35., -85., -10.), "V5": (85., -60., -10.), "V6": (125., -15., -10.)}
    if name == "extended":
        points.update(V3R=(-81., -94., 5.), V4R=(-111., -85., -10.),
                      V7=(115., 70., -10.), V8=(45., 120., -10.), V9=(-10., 125., -10.))
    out = [Electrode(k, v) for k, v in points.items()]
    if name == "intracardiac":
        out.extend([Electrode("CathA", (0., 0., 0.), "intracardiac"),
                    Electrode("CathB", (0., 0., 5.), "intracardiac")])
    return tuple(out)


@dataclass(frozen=True)
class Experiment:
    duration_ms: float = 600.
    output_step_ms: float = 2.
    max_dt_ms: float = .05
    cycle_ms: float = 800.
    stimulus_ms: float = 2.
    stimulus_strength: float = .8  # normalized voltage / ms
    stimulus_site: str = "his"
    diffusion_scale: float = 1.
    anisotropy_ratio: float = 3.
    recovery_scale: float = 1.
    preset: str = "reference"
    region: int = 7
    severity: float = .8
    s2_ms: float | None = None
    electrodes: tuple[Electrode, ...] = field(default_factory=electrode_preset)
    conduction_variant: str = "legacy"
    pathway_scales: dict[str, float] = field(default_factory=dict)

    def validate(self):
        from .conduction import VARIANTS
        if self.conduction_variant not in VARIANTS:
            raise ValueError("Unknown conduction variant")
        if any(not np.isfinite(v) or not 0 <= v <= 3 for v in self.pathway_scales.values()):
            raise ValueError("Pathway coupling multiplier outside 0–3")
        bounds = {"duration_ms": (10., 3000.), "output_step_ms": (.5, 10.),
                  "max_dt_ms": (.001, .1), "cycle_ms": (150., 3000.),
                  "stimulus_ms": (.1, 10.), "stimulus_strength": (.01, 2.),
                  "diffusion_scale": (.05, 5.), "anisotropy_ratio": (1., 10.),
                  "recovery_scale": (.25, 3.), "severity": (0., 1.)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} outside [{lo}, {hi}]")
        if self.preset not in PRESETS or self.region not in range(1, 18):
            raise ValueError("Unknown scenario or AHA region")
        if self.s2_ms is not None and not 0 < self.s2_ms < self.duration_ms:
            raise ValueError("S2 must lie inside the experiment")
        if not 3 <= len(self.electrodes) <= 64:
            raise ValueError("3–64 electrodes required")
        names = [e.name for e in self.electrodes]
        if len(set(names)) != len(names) or not {"RA", "LA", "LL"} <= set(names):
            raise ValueError("Unique electrodes including RA/LA/LL required")
        for e in self.electrodes:
            if (not e.name or e.kind not in {"surface", "intracardiac"}
                    or np.shape(e.xyz_mm) != (3,) or not np.isfinite(e.xyz_mm).all()
                    or np.max(np.abs(e.xyz_mm)) > 2000):
                raise ValueError("Invalid electrode")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if "electrodes" in data:
            data["electrodes"] = tuple(Electrode(e["name"], tuple(e["xyz_mm"]), e["kind"]) for e in data["electrodes"])
        out = cls(**data)
        out.validate()
        return out


@dataclass
class Tissue:
    xyz: np.ndarray
    edges: np.ndarray
    cells: np.ndarray
    regions: np.ndarray
    fibers: np.ndarray
    sites: dict[str, np.ndarray]
    branches: dict[str, np.ndarray]
    geometry_kind: str = "generic_anatomical_graph"

    def fingerprint(self):
        h = hashlib.sha256(MODEL_VERSION.encode())
        for a in (self.xyz, self.edges, self.cells, self.regions, self.fibers):
            h.update(np.ascontiguousarray(a).tobytes())
        for mapping in (self.sites, self.branches):
            for k, v in sorted(mapping.items()):
                h.update(k.encode())
                h.update(np.asarray(v, dtype=np.int64).tobytes())
        return h.hexdigest()


def anatomical_tissue(conduction_variant="legacy") -> Tissue:
    # Geometry-only dependency: no renderer, AnimationEngine or patient timeline.
    from ..gui.heart3d.model import default_model
    from ..gui.heart3d.volume import build_wall_grid
    model = default_model()
    wall = build_wall_grid(model)
    xyz = list(wall.positions)
    cells = list(np.where(wall.endo, 0, np.where(wall.epi, 2, 1)))
    regions = list(wall.segment)
    # Rule-based rotating fiber hypothesis; not DTI or patient-specific fibers.
    axis = np.array([0., 0., 1.])
    circum = np.cross(wall.normal_out, axis)
    circum /= np.maximum(np.linalg.norm(circum, axis=1)[:, None], 1e-8)
    angle = np.deg2rad(60. - 120. * wall.layer / wall.n_layers)
    fiber = circum * np.cos(angle[:, None]) + axis * np.sin(angle[:, None])
    fibers = list(fiber)
    edges = list(zip(wall.edges_i, wall.edges_j))
    sites = {k: v.copy() for k, v in wall.purkinje_sites.items()}
    atrial = {}
    for part_name in ("ra_wall", "la_wall"):
        part = model.parts[part_name]
        offset = len(xyz)
        xyz.extend(part.vertices)
        cells.extend([3] * len(part.vertices))
        regions.extend([0] * len(part.vertices))
        fibers.extend(np.tile(axis, (len(part.vertices), 1)))
        e = np.vstack([part.faces[:, [0, 1]], part.faces[:, [1, 2]], part.faces[:, [2, 0]]])
        edges.extend(np.unique(np.sort(e, axis=1), axis=0) + offset)
        atrial[part_name] = {k: offset + int(v) for k, v in model.atrial_sources[part_name].items()}
        sites[part_name+"_all"] = np.arange(offset, offset+len(part.vertices))
    edges.append((atrial["ra_wall"]["sa_node"], atrial["la_wall"]["bachmann"]))
    sites["sa_node"] = np.array([atrial["ra_wall"]["sa_node"]])
    sites["la_bachmann"] = np.array([atrial["la_wall"]["bachmann"]])
    sites["ra_av_access"] = np.array([atrial["ra_wall"]["av_node"]])
    branches = {"bachmann": np.array([atrial["ra_wall"]["sa_node"], atrial["la_wall"]["bachmann"]])}
    # Explicit 1-D cable nodes along the anatomical guide curves; no preassigned activation times.
    network_nodes = []
    for name, path in model.conduction_paths.items():
        if len(path) < 2:
            continue
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        if distance[-1] < 1e-6:
            continue
        s = np.linspace(0, distance[-1], max(2, int(distance[-1] / 2) + 1))
        points = np.column_stack([np.interp(s, distance, path[:, k]) for k in range(3)])
        idx = np.arange(len(xyz), len(xyz) + len(points))
        if network_nodes:
            existing = np.asarray(xyz)[network_nodes]
            edges.append((int(idx[0]), network_nodes[int(cKDTree(existing).query(points[0])[1])]))
        xyz.extend(points)
        cells.extend([5] * len(points))
        regions.extend([0] * len(points))
        tangent = np.gradient(points, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-8)
        fibers.extend(tangent)
        edges.extend(zip(idx[:-1], idx[1:]))
        network_nodes.extend(idx.tolist())
        branches[name] = idx
        if name == "his":
            sites["his"] = idx[:1]
            # AV bridge: slow phenotype at junction. No claim of physiological PR.
            av = atrial["ra_wall"]["av_node"]
            cells[av] = 4
            edges.append((av, int(idx[0])))
        else:
            chamber = 1 if name == "rbb" else 0
            targets = np.flatnonzero(wall.endo & (wall.chamber == chamber))
            for node in idx[-min(4, len(idx)):]:
                near = targets[cKDTree(wall.positions[targets]).query(np.asarray(xyz)[node], k=3)[1]]
                edges.extend((int(node), int(n)) for n in np.atleast_1d(near))
    # Shared cable junctions are one node, not zero-length diffusion edges.
    positions, first, inverse = np.unique(np.asarray(xyz, float), axis=0, return_index=True, return_inverse=True)
    edges = np.unique(np.sort(inverse[np.asarray(edges, dtype=np.int64)], axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    sites = {k: np.unique(inverse[v]) for k, v in sites.items()}
    branches = {k: np.unique(inverse[v]) for k, v in branches.items()}
    tissue = Tissue(positions, edges, np.asarray(cells, int)[first], np.asarray(regions, int)[first],
                    np.asarray(fibers, float)[first], sites, branches)
    tissue = coarsen_short_edges(tissue)
    from .conduction import extend_network
    return extend_network(tissue, conduction_variant)


def coarsen_short_edges(tissue, minimum_mm=.8):
    """Contract connected sub-mm edges within the SAME cell phenotype.

    Avoid an arbitrarily tiny explicit time step from sliver surface triangles.
    This is a documented preview mesh reduction, not mesh convergence. Never
    joins nearby but electrically unconnected tissue/cables.
    """
    for _ in range(4):
        a, b = tissue.edges.T
        short = ((np.linalg.norm(tissue.xyz[a] - tissue.xyz[b], axis=1) < minimum_mm)
                 & (tissue.cells[a] == tissue.cells[b]))
        if not short.any():
            break
        graph = coo_matrix((np.ones(short.sum()), (a[short], b[short])),
                           shape=(len(tissue.xyz), len(tissue.xyz))).tocsr()
        n, labels = connected_components(graph, directed=False)
        _, first = np.unique(labels, return_index=True)
        count = np.bincount(labels)
        xyz = np.column_stack([np.bincount(labels, weights=tissue.xyz[:, k]) / count for k in range(3)])
        edges = np.unique(np.sort(labels[tissue.edges], axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        tissue = Tissue(xyz, edges, tissue.cells[first], tissue.regions[first], tissue.fibers[first],
                        {k: np.unique(labels[v]) for k, v in tissue.sites.items()},
                        {k: np.unique(labels[v]) for k, v in tissue.branches.items()},
                        "generic_anatomical_graph_short_edge_contraction_0.8mm")
    return tissue


def operator(tissue: Tissue, spec: Experiment):
    """Symmetric conservative graph Laplacian; insulating boundary (no exterior edges)."""
    spec.validate()
    n = len(tissue.xyz)
    a, b = tissue.edges.T
    delta = tissue.xyz[b] - tissue.xyz[a]
    length = np.linalg.norm(delta, axis=1)
    if np.any(length < 1e-8):
        raise ValueError("Coincident graph nodes joined by an edge")
    direction = delta / length[:, None]
    alignment = .5 * ((direction * tissue.fibers[a]).sum(1) ** 2 + (direction * tissue.fibers[b]).sum(1) ** 2)
    d = DIFFUSION[tissue.cells] * spec.diffusion_scale
    close = TAU_CLOSE[tissue.cells] * spec.recovery_scale
    live = np.ones(n, bool)
    region = tissue.regions == spec.region
    if spec.preset == "scar":
        live[region] = False
    elif spec.preset == "slow_zone":
        d[region] *= 1. - .95 * spec.severity
    elif spec.preset == "short_ap":
        close[region] *= 1. - .7 * spec.severity
    elif spec.preset == "long_ap":
        close[region] *= 1. + spec.severity
    elif spec.preset in {"lbbb", "rbbb"}:
        prefixes = ("lbb",) if spec.preset == "lbbb" else ("rbb",)
        for name, nodes in tissue.branches.items():
            if name.startswith(prefixes):
                live[nodes] = False
    weight = 2 * d[a] * d[b] / (d[a] + d[b]) / length ** 2
    weight *= (1. / spec.anisotropy_ratio + (1 - 1. / spec.anisotropy_ratio) * alignment)
    weight *= live[a] & live[b]
    for name, scale in spec.pathway_scales.items():
        if name not in tissue.branches:
            raise ValueError("Pathway is absent from this anatomy: " + name)
        selected = np.zeros(n, bool)
        selected[tissue.branches[name]] = True
        weight[selected[a] & selected[b]] *= scale
    adj = coo_matrix((np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    lap = adj - coo_matrix((degree, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    return lap, close, live, degree


def experiment_hash(spec: Experiment, tissue: Tissue):
    return hashlib.sha256((json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False)
                           + tissue.fingerprint()).encode()).hexdigest()
