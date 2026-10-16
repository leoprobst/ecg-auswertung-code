"""One low-priority process, bounded output, cancellation; no production imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def acquire_worker_lock(path):
    """OS-owned lease: even an abruptly closed GUI cannot admit a second worker."""
    handle = Path(path).open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another simulation process already owns this workspace") from exc
    return handle


def main():
    # Set before NumPy/SciPy imports even when launched outside the GUI.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    import psutil
    from .model import Experiment, anatomical_tissue
    from .solver import fit_candidates, save_result, simulate

    root = Path(sys.argv[1]).resolve()
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    process = psutil.Process()
    try:
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (psutil.AccessDenied, OSError) as exc:
        print(json.dumps({"phase": "priority_warning", "message": str(exc)}), flush=True)

    def cancelled():
        return (root / "cancel").exists()

    def resources():
        notified = False
        while psutil.virtual_memory().available < 2 * 1024 ** 3:
            if cancelled():
                raise InterruptedError("Cancelled while waiting for memory")
            if not notified:
                print(json.dumps({"phase": "waiting_for_ram", "reserve_gib": 2}), flush=True)
                notified = True
            time.sleep(.5)

    def report(done, total, elapsed):
        resources()
        print(json.dumps({"phase": "computing", "done": done, "total": total,
                          "elapsed_seconds": elapsed,
                          "eta_seconds": elapsed * (total - done) / done if done and elapsed else None}), flush=True)

    lease = None
    try:
        lease = acquire_worker_lock(root.parent / "simulation-worker.lock")
        resources()
        if request.get("kind") == "published_ecme":
            from .ecme import ECMEExperiment, simulate_ecme
            times, states, metadata = simulate_ecme(ECMEExperiment(**request["ecme_experiment"]), report, cancelled)
            if cancelled():
                raise InterruptedError("Cancelled before ECME export")
            with (root/"ecme.partial").open("wb") as handle:
                np.savez_compressed(handle, times_ms=times, states=states,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root/"ecme.partial").replace(root/"ecme.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        if request.get("kind") == "mitochondrial_assay":
            from .mitochondria import MitochondrialExperiment, simulate_mitochondria
            times, states, metadata = simulate_mitochondria(MitochondrialExperiment(**request["mitochondrial_experiment"]), report, cancelled)
            if cancelled():
                raise InterruptedError("Cancelled before mitochondrial export")
            with (root/"mitochondria.partial").open("wb") as handle:
                np.savez_compressed(handle, times_s=times, states=states,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root/"mitochondria.partial").replace(root/"mitochondria.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        if request.get("kind") == "volume_conductor":
            from .volume_conductor import recover_tissue
            import hashlib
            for key in ("tissue_path", "conductor_path"):
                path = Path(request[key])
                if path.stat().st_size > 160*1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != request[key+"_sha256"]:
                    raise ValueError("Changed or excessive conductor input")
            times, ecg, names, metadata = recover_tissue(request["tissue_path"], request["conductor_path"],
                request["electrode_xyz_mm"], request["electrode_names"], cancelled)
            if cancelled():
                raise InterruptedError("Cancelled before conductor export")
            with (root/"conductor.partial").open("wb") as handle:
                np.savez_compressed(handle, times_ms=times, signal=ecg, lead_names=np.array(names),
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root/"conductor.partial").replace(root/"conductor.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        if request.get("kind") == "reentry_ring":
            from .reentry import simulate_ring
            times, xyz, voltage, metadata = simulate_ring(
                float(request.get("duration_ms", 1800)), request.get("ablate_at_ms"), report, cancelled)
            if cancelled():
                raise InterruptedError("Cancelled before saving")
            with (root/"ring.partial").open("wb") as handle:
                np.savez_compressed(handle, times_ms=times, xyz_mm=xyz, voltage=voltage,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root/"ring.partial").replace(root/"ring.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        if request.get("kind") == "ionic_tissue":
            from .ionic import CellExperiment
            from .monodomain import StimulusEvent, block_mesh, load_mesh, simulate_monodomain
            from .substrate import RegionalSubstrate
            import hashlib
            if request.get("mesh_path"):
                path = Path(request["mesh_path"])
                if path.stat().st_size > 16 * 1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != request.get("mesh_file_sha256"):
                    raise ValueError("Mesh changed after request or exceeds size budget")
                mesh = load_mesh(path)
            else:
                mesh = block_mesh()
            times, voltage, metadata = simulate_monodomain(
                mesh, CellExperiment(**request["cell_experiment"]),
                duration_ms=float(request.get("duration_ms", 60)), progress=report, cancelled=cancelled,
                events=[StimulusEvent(**e) for e in request["events"]] if "events" in request else None,
                substrate=RegionalSubstrate(**request["substrate"]) if "substrate" in request else None)
            if cancelled():
                raise InterruptedError("Cancelled before saving")
            metadata["patient_profile_applied"] = False
            with (root / "tissue.partial").open("wb") as handle:
                np.savez_compressed(handle, times_ms=times, voltage_mv=voltage,
                                    xyz_mm=mesh.xyz_mm, tetra=mesh.tetra, fibers=mesh.fibers,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root / "tissue.partial").replace(root / "tissue.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        if request.get("kind") == "ionic_cell":
            from .ionic import CellExperiment, simulate_cell, simulate_preconditioned
            from .patient import PatientProfile
            profile = PatientProfile.from_dict(request.get("patient_profile", {}))
            cell_spec = CellExperiment(**request["cell_experiment"])
            if request.get("use_patient_electrolytes"):
                from dataclasses import replace
                constraints = profile.ionic_constraints()
                if not constraints["constraints"]:
                    raise ValueError("No measured and temporally verified patient electrolytes")
                cell_spec = replace(cell_spec, extracellular_mmol_l={k: v["value"] for k, v in constraints["constraints"].items()})
            prepace = request.get("precondition_max_beats", 0)
            if prepace:
                def prepace_progress(values):
                    resources()
                    print(json.dumps({"phase": "preconditioning", "done": values["beats"], "total": prepace,
                                      "eta_seconds": None, "converged": values["converged"]}), flush=True)
                times, voltage, states, metadata = simulate_preconditioned(
                    cell_spec, prepace, cancelled=cancelled, checkpoint=prepace_progress)
            else:
                times, voltage, states, metadata = simulate_cell(cell_spec, report, cancelled)
            metadata["patient_profile"] = profile.to_dict()
            metadata["patient_electrolytes_applied"] = bool(request.get("use_patient_electrolytes"))
            if cancelled():
                raise InterruptedError("Cancelled before saving")
            with (root / "ionic.partial").open("wb") as handle:
                np.savez_compressed(handle, times_ms=times, voltage_mv=voltage, states=states,
                                    metadata=np.array(json.dumps(metadata, allow_nan=False)))
            (root / "ionic.partial").replace(root / "ionic.npz")
            print(json.dumps({"phase": "complete"}), flush=True)
            return 0
        print(json.dumps({"phase": "building_geometry"}), flush=True)
        if request.get("kind") == "inverse":
            variants = {s.get("conduction_variant", "legacy") for s in request["candidates"]}
            if len(variants) != 1:
                raise ValueError("One explicitly frozen geometry variant per inverse job required")
            variant = next(iter(variants))
        else:
            variant = request["experiment"].get("conduction_variant", "legacy")
        tissue = anatomical_tissue(variant)
        if request.get("kind") == "inverse":
            with np.load(root / "target.npz", allow_pickle=False) as z:
                target, fs, names = z["signal"], float(z["fs"]), list(z["lead_names"])
            rows = fit_candidates(tissue, [Experiment.from_dict(s) for s in request["candidates"]],
                                  target, fs, names, report, cancelled)
            temp = root / "fit.partial"
            temp.write_text(json.dumps({"schema": "simulation-fit/1", "results": rows,
                                       "clinical_validation": False}, indent=2), encoding="utf-8")
            temp.replace(root / "fit.json")
        else:
            result = simulate(tissue, Experiment.from_dict(request["experiment"]), report, cancelled)
            if request.get("patient_profile"):
                from .patient import PatientProfile, twin_readiness
                profile = PatientProfile.from_dict(request["patient_profile"])
                result.metadata["patient_profile"] = profile.to_dict()
                result.metadata["patient_constraints"] = twin_readiness(profile)
                result.metadata["patient_geometry_applied"] = False
                result.metadata["demographic_physics_scaling_applied"] = False
            if cancelled():
                raise InterruptedError("Cancelled before saving")
            print(json.dumps({"phase": "saving"}), flush=True)
            save_result(root / "result.npz", result, tissue)
        print(json.dumps({"phase": "complete"}), flush=True)
    except InterruptedError as exc:
        print(json.dumps({"phase": "cancelled", "message": str(exc)}), flush=True)
        return 2
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps({"phase": "failed", "message": str(exc)}), flush=True)
        return 1
    finally:
        if lease is not None:
            lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
