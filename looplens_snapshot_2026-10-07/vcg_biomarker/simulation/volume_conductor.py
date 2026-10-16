"""Bounded finite-volume-conductor recovery from prescribed myocardial Vm.

FEM: (Ki+Ke) phi_e = -Ki Vm, insulated outer boundary, one gauge constraint.
Source: Bishop/Plank 2011, https://pmc.ncbi.nlm.nih.gov/articles/PMC3378475/.
This is one-way pseudo-bidomain recovery, NOT a fully coupled bidomain solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


@dataclass
class ConductorMesh:
    xyz_mm: np.ndarray
    tetra: np.ndarray
    sigma_e_s_per_m: np.ndarray  # SPD tensor per element, bath/torso included
    sigma_i_s_per_m: np.ndarray  # SPD in heart, exactly zero outside
    source: str
    coordinate_system: str = "x_left_y_posterior_z_cranial"

    def validate(self):
        x, t = np.asarray(self.xyz_mm), np.asarray(self.tetra)
        if x.ndim != 2 or x.shape[1] != 3 or not 4 <= len(x) <= 4000 or not np.isfinite(x).all():
            raise ValueError("Finite 4–4000-node mm conductor mesh required")
        if (t.ndim != 2 or t.shape[1] != 4 or t.dtype.kind not in "iu" or not 1 <= len(t) <= 32000
                or t.min() < 0 or t.max() >= len(x) or len(np.unique(np.sort(t, axis=1), axis=0)) != len(t)):
            raise ValueError("Invalid/duplicate tetrahedra")
        if not self.source.strip() or self.coordinate_system != "x_left_y_posterior_z_cranial":
            raise ValueError("Explicit shared anatomical coordinates and provenance required")
        for a, intracellular in ((self.sigma_e_s_per_m, False), (self.sigma_i_s_per_m, True)):
            a = np.asarray(a)
            if a.shape != (len(t), 3, 3) or not np.isfinite(a).all() or not np.allclose(a, a.transpose(0, 2, 1)):
                raise ValueError("Symmetric finite conductivity tensors required")
            eigen = np.linalg.eigvalsh(a)
            zero = np.all(a == 0, axis=(1, 2)) if intracellular else np.zeros(len(t), bool)
            if np.any(eigen[~zero] <= 0) or np.any(eigen > 10):
                raise ValueError("Conductivity must be positive definite (or exactly zero for bath intracellular tensor)")
        if not np.any(self.sigma_i_s_per_m):
            raise ValueError("No myocardium in conductor mesh")

    def fingerprint(self):
        self.validate()
        h = hashlib.sha256((self.source+self.coordinate_system).encode())
        for a in (self.xyz_mm, self.tetra, self.sigma_e_s_per_m, self.sigma_i_s_per_m):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def load_conductor(path):
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 16*1024**2:
            raise ValueError("Conductor archive exceeds budget")
    with np.load(path, allow_pickle=False) as data:
        if str(data["units"]) != "mm":
            raise ValueError("Conductor coordinates must be in mm")
        mesh = ConductorMesh(*(data[k].copy() for k in ("xyz_mm", "tetra", "sigma_e_s_per_m", "sigma_i_s_per_m")),
                             str(data["source"]), str(data["coordinate_system"]))
    mesh.validate()
    return mesh


def _geometry(mesh):
    points = mesh.xyz_mm[mesh.tetra]
    edges = (points[:, 1:]-points[:, :1]).transpose(0, 2, 1)
    volume = np.abs(np.linalg.det(edges))/6
    if np.any(volume < 1e-9) or np.any(np.linalg.cond(edges) > 1e7):
        raise ValueError("Degenerate/ill-conditioned conductor tetrahedron")
    inverse = np.linalg.inv(edges)
    gradients = np.concatenate([-inverse.sum(axis=1, keepdims=True), inverse], axis=1)
    return points, volume, inverse, gradients


def _stiffness(mesh, volume, gradients, sigma):
    # Coordinates in mm and sigma in S/m: gradients scale 1e3, volume 1e-9.
    local = 1e-3*volume[:, None, None]*np.einsum("eai,eij,ebj->eab", gradients, sigma, gradients)
    rows, cols = np.repeat(mesh.tetra, 4, axis=1).ravel(), np.tile(mesh.tetra, (1, 4)).ravel()
    return coo_matrix((local.ravel(), (rows, cols)), shape=(len(mesh.xyz_mm),)*2).tocsr()


class VolumeConductor:
    def __init__(self, mesh, gauge_node=0):
        mesh.validate()
        import psutil
        if psutil.virtual_memory().available/2**30 < 3:
            raise RuntimeError("Conductor factorization deferred: RAM reserve")
        if type(gauge_node) is not int or not 0 <= gauge_node < len(mesh.xyz_mm):
            raise ValueError("Invalid gauge node")
        self.mesh = mesh
        self.points, volume, self.inverse, gradients = _geometry(mesh)
        self.ki = _stiffness(mesh, volume, gradients, mesh.sigma_i_s_per_m)
        self.k = self.ki + _stiffness(mesh, volume, gradients, mesh.sigma_e_s_per_m)
        if connected_components(self.k, directed=False)[0] != 1:
            raise ValueError("Disconnected conductor")
        self.mass = np.bincount(mesh.tetra.ravel(), weights=np.repeat(volume/4, 4), minlength=len(mesh.xyz_mm))
        if np.any(self.mass <= 0):
            raise ValueError("Unused conductor node")
        self.free = np.flatnonzero(np.arange(len(mesh.xyz_mm)) != gauge_node)
        self.factor = splu(self.k[self.free][:, self.free].tocsc())
        self.heart_nodes = np.unique(mesh.tetra[np.any(mesh.sigma_i_s_per_m != 0, axis=(1, 2))])

    def electrode_weights(self, xyz_mm):
        xyz = np.asarray(xyz_mm, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not 1 <= len(xyz) <= 64 or not np.isfinite(xyz).all():
            raise ValueError("1–64 finite electrode positions required")
        weights = np.zeros((len(xyz), len(self.mesh.xyz_mm)))
        for j, point in enumerate(xyz):
            local = np.einsum("eij,ej->ei", self.inverse, point-self.points[:, 0])
            bary = np.column_stack([1-local.sum(1), local])
            candidates = np.flatnonzero(np.all((bary >= -1e-9) & (bary <= 1+1e-9), axis=1))
            if not len(candidates):
                raise ValueError("Electrode outside conductor; no nearest-node snapping")
            cell = candidates[0]
            weights[j, self.mesh.tetra[cell]] = bary[cell]
        return weights

    def recover(self, voltage_mv, heart_node_ids, electrode_xyz_mm, cancelled=None):
        voltage, nodes = np.asarray(voltage_mv, float), np.asarray(heart_node_ids)
        if (voltage.ndim != 2 or voltage.shape[1] != len(nodes) or not 1 <= len(voltage) <= 20000
                or voltage.nbytes > 128*1024**2 or not np.isfinite(voltage).all() or np.abs(voltage).max() > 200):
            raise ValueError("Invalid or excessive membrane-potential frames")
        if (nodes.ndim != 1 or nodes.dtype.kind not in "iu" or len(np.unique(nodes)) != len(nodes)
                or np.any(nodes < 0) or np.any(nodes >= len(self.mesh.xyz_mm))
                or not np.isin(self.heart_nodes, nodes).all()):
            raise ValueError("All myocardial nodes need an explicit one-to-one Vm assignment")
        weights = self.electrode_weights(electrode_xyz_mm)
        result = np.empty((len(voltage), len(weights)))
        max_residual = 0.
        import psutil
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            for start in range(0, len(voltage), 32):
                if cancelled and cancelled():
                    raise InterruptedError("Volume conductor cancelled")
                if psutil.virtual_memory().available/2**30 < 2.5:
                    raise RuntimeError("Volume conductor stopped at safe frame boundary: RAM")
                vm = np.zeros((len(self.mesh.xyz_mm), min(32, len(voltage)-start)))
                vm[nodes] = voltage[start:start+32].T
                # Ki annihilates constants. Remove the common Vm baseline first
                # to avoid false residual failures from cancellation at rest.
                vm -= vm[self.heart_nodes[0]][None, :]
                rhs = -self.ki@vm
                phi = np.zeros_like(vm)
                phi[self.free] = self.factor.solve(rhs[self.free])
                relative = np.linalg.norm(self.k@phi-rhs)/(np.linalg.norm(rhs)+1e-15)
                max_residual = max(max_residual, float(relative))
                if relative > 1e-7:
                    raise ArithmeticError("Volume-conductor residual exceeds tolerance")
                phi -= (self.mass@phi/self.mass.sum())[None, :]
                result[start:start+32] = (weights@phi).T
        return result, {"schema": "finite-conductor-recovery/1", "clinical_use": False,
                        "mesh_sha256": self.mesh.fingerprint(), "geometry_source": self.mesh.source,
                        "voltage_unit": "mV", "conductivity_unit": "S/m", "max_relative_residual": max_residual,
                        "gauge": "volume-weighted zero mean", "boundary": "insulating external boundary",
                        "limitations": ["Prescribed Vm, no bidomain feedback or bath-loading correction.",
                                        "Conforming volume mesh and explicit physical conductivity required.",
                                        "No anatomy is inferred from demographics; no clinical localization claim."]}


def recover_tissue(tissue_path, conductor_path, electrode_xyz_mm, electrode_names, cancelled=None):
    """Exact coordinate matching; no registration or invented tissue interpolation."""
    mesh = load_conductor(conductor_path)
    with zipfile.ZipFile(tissue_path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 160*1024**2:
            raise ValueError("Tissue output exceeds input budget")
    with np.load(tissue_path, allow_pickle=False) as data:
        xyz, voltage, times = data["xyz_mm"].copy(), data["voltage_mv"].copy(), data["times_ms"].copy()
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or len(times) != len(voltage):
        raise ValueError("Invalid tissue time axis")
    mapping = []
    for point in xyz:
        found = np.flatnonzero(np.linalg.norm(mesh.xyz_mm-point, axis=1) <= 1e-8)
        if len(found) != 1:
            raise ValueError("Tissue/torso meshes do not conform; explicit registration needed")
        mapping.append(int(found[0]))
    if (len(electrode_names) != len(electrode_xyz_mm) or len(set(electrode_names)) != len(electrode_names)
            or not {"RA", "LA", "LL"} <= set(electrode_names)):
        raise ValueError("Unique electrodes including RA/LA/LL required")
    potential, metadata = VolumeConductor(mesh).recover(voltage, np.array(mapping), electrode_xyz_mm, cancelled)
    from .solver import derive_leads
    ecg, names = derive_leads(potential, electrode_names)
    metadata.update(tissue_sha256=hashlib.sha256(Path(tissue_path).read_bytes()).hexdigest(),
                    electrode_names=list(electrode_names), electrode_xyz_mm=np.asarray(electrode_xyz_mm).tolist())
    return times, ecg, names, metadata


def export_conductor_to_lab(result_path, output_path):
    """Physical model mV, no fitted amplitude gain; remains explicitly synthetic."""
    from .solver import LEADS_12
    with zipfile.ZipFile(result_path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 32*1024**2:
            raise ValueError("Conductor export exceeds budget")
    with np.load(result_path, allow_pickle=False) as data:
        times, signal = data["times_ms"].copy(), data["signal"].copy()
        names = data["lead_names"].tolist()
        metadata = json.loads(str(data["metadata"]))
    if (metadata.get("schema") != "finite-conductor-recovery/1" or metadata.get("clinical_use") is not False
            or metadata.get("voltage_unit") != "mV" or not set(LEADS_12) <= set(names)
            or len(set(names)) != len(names) or len(times) < 2 or signal.shape != (len(times), len(names))
            or not np.isfinite(signal).all() or not np.isfinite(times).all()
            or np.diff(times).min() <= 0 or not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-7, atol=1e-9)):
        raise ValueError("Verified synthetic 12-lead conductor output and uniform time axis required")
    metadata = metadata | {"synthetic": True, "amplitude_source": "conductivity-constrained model in mV, not patient-calibrated",
                           "result_sha256": hashlib.sha256(Path(result_path).read_bytes()).hexdigest()}
    with Path(output_path).open("xb") as handle:
        np.savez_compressed(handle, ecg=signal[:, [names.index(k) for k in LEADS_12]],
                            sampling_rate_hz=1000/np.diff(times)[0], lead_names=np.array(LEADS_12),
                            units=np.array(["mV"]*12), synthetic=True,
                            simulation_provenance=np.array(json.dumps(metadata, allow_nan=False)))
    return Path(output_path)
