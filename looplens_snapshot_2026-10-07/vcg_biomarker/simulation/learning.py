"""Resumable populations-of-models and transparent forward emulator training.

Simulation parameters are not patient labels. ECG QT is not cellular APD90.
Only identifiable measured cellular targets can constrain the cellular bank.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

PARAMETERS = ("INa", "ICaL", "IKr", "IKs", "IK1", "Ito")
OUTPUTS = ("apd90_ms", "amplitude_mv", "resting_mv", "max_upstroke_mv_per_ms")


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    # Windows indexers/sync clients can briefly hold the destination without
    # delete-sharing. Keep atomic replacement; never fall back to truncation.
    for attempt in range(7):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.02 * 2**attempt)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ap_features(times, voltage, cycle_ms):
    import numpy as np
    times, voltage = np.asarray(times), np.asarray(voltage)
    mask = times >= times[-1]-cycle_ms
    t, v = times[mask], voltage[mask]
    if len(t) < 20 or not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Invalid AP trajectory")
    resting = float(v[0])
    peak = int(np.argmax(v))
    amplitude = float(v[peak]-resting)
    if amplitude < 40 or peak == 0:
        raise ValueError("No captured action potential")
    dv = np.gradient(v, t)
    activation = int(np.argmax(dv[:peak+1]))
    threshold = resting+.1*amplitude
    candidates = np.flatnonzero((v[peak:-1] >= threshold) & (v[peak+1:] < threshold)) + peak
    if not len(candidates):
        raise ValueError("APD90 unmeasurable within the protocol")
    k = int(candidates[0])
    crossing = t[k] + (threshold-v[k])/(v[k+1]-v[k])*(t[k+1]-t[k])
    return {"apd90_ms": float(crossing-t[activation]), "amplitude_mv": amplitude,
            "resting_mv": resting, "max_upstroke_mv_per_ms": float(dv[activation])}


def prepare(root, samples=128, beats=10, max_workers=4, seed=20260923,
            precondition_max_beats=0, max_active_seconds=1800, cycles_ms=(800.,), native_rhs=False,
            cell_model="tentusscher-2006", reference_campaign=None):
    import shutil
    import numpy as np
    from scipy.stats import qmc
    from importlib.metadata import version
    import psutil
    if cell_model not in {"tentusscher-2006", "ohara-2011"}:
        raise ValueError("Population comparison requires a supported ventricular cell model")
    reference = None
    if reference_campaign is not None:
        reference_root = Path(reference_campaign).resolve()
        reference = json.loads((reference_root/"manifest.json").read_text(encoding="utf-8"))
        unsigned = dict(reference)
        reference_hash = unsigned.pop("job_sha256")
        if digest(unsigned) != reference_hash or json.loads((reference_root/"status.json").read_text())["phase"] != "complete":
            raise ValueError("Completed unchanged reference campaign required")
        if (reference["model"] == cell_model or reference["beats"] != beats
                or reference.get("precondition_max_beats") != precondition_max_beats
                or reference["cycles_ms"] != list(cycles_ms) or reference["output_ms"] != .25):
            raise ValueError("Comparison needs different models but identical pacing/output/preconditioning protocols")
    if native_rhs:
        raise ValueError("Native RHS not released; use the verified Python/SciPy kernel")
    if max_workers == 0:
        max_workers = max(1, min(64, (psutil.cpu_count() or 2)-1))
    if not 32 <= samples <= 4096 or not 3 <= beats <= 20 or not 1 <= max_workers <= 64:
        raise ValueError("Bounded population budget required")
    cycles_ms = tuple(float(c) for c in cycles_ms)
    if (not 1 <= len(cycles_ms) <= 5 or len(set(cycles_ms)) != len(cycles_ms)
            or any(not np.isfinite(c) or not 250 <= c <= 2500 for c in cycles_ms)):
        raise ValueError("One to five distinct, valid predeclared pacing protocols required")
    if precondition_max_beats not in (0, *range(10, 1001)) or not 60 <= max_active_seconds <= 86400:
        raise ValueError("Invalid preconditioning or wall-time budget")
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("Use a new job directory, or run/resume the exact existing job")
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    runtime = root / "runtime"
    package = runtime / "vcg_biomarker/simulation"
    package.mkdir(parents=True)
    shutil.copyfile(repo / "vcg_biomarker/__init__.py", package.parent / "__init__.py")
    for name in ("__init__.py", "learning.py", "ionic.py", "ischemia.py", "metabolism.py", "model_comparison.py", "variants.py", "identifiability.py", "resources.py", "model_assets.py", "worker.py"):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    shutil.copytree(repo / "assets/ionic_models", runtime / "assets/ionic_models")
    shutil.copytree(repo / ".simulation-deps/myokit", runtime / ".simulation-deps/myokit",
                    ignore=shutil.ignore_patterns("__pycache__", "tests"))
    hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob("*") if p.is_file()}
    matrix = .5 + qmc.LatinHypercube(d=len(PARAMETERS), seed=seed).random(samples)
    order = np.random.default_rng(seed).permutation(samples)
    parts = {int(k): "fit" if j < int(samples*.7) else "calibration" if j < int(samples*.85) else "audit"
             for j, k in enumerate(order)}
    cases = [{"id": i*len(cycles_ms)+j, "candidate_id": i, "cycle_ms": cycle,
              "scales": dict(zip(PARAMETERS, map(float, row))), "partition": parts[i]}
             for i, row in enumerate(matrix) for j, cycle in enumerate(cycles_ms)]
    if reference is not None:
        cases = [dict(case) for case in reference["cases"] if case["candidate_id"] < samples]
        if len(cases) != samples*len(cycles_ms) or [c["id"] for c in cases] != list(range(len(cases))):
            raise ValueError("Reference design lacks requested deterministic prefix")
    manifest = {"schema": "ionic-population-job/1", "clinical_use": False, "seed": seed, "cases": cases,
                "model": cell_model, "beats": beats, "cycle_ms": cycles_ms[0], "cycles_ms": list(cycles_ms), "output_ms": .25,
                "candidate_count": samples, "partition_unit": "conductance_candidate_across_all_protocols",
                "rhs_backend": "numba" if native_rhs else "python",
                "max_workers": max_workers, "ram_reserve_gib": 2., "max_active_seconds": float(max_active_seconds),
                "precondition_max_beats": precondition_max_beats,
                "parameter_bounds": [.5, 1.5], "runtime_hashes": hashes,
                "source": "synthetic_physics_experiments", "patient_data_used_as_ionic_truth": False,
                "protocol": "published initialization; fixed preconditioning, periodicity reported per candidate",
                "data_roles": "MIMIC anatomical priors remain separate; no QT-to-APD or diagnosis-to-channel substitution"}
    manifest["cell_environment"] = {"cell_type": "TP06_epi" if cell_model == "tentusscher-2006" else "ORd_epi", "temperature_kelvin": 310.,
                                    "potassium_mmol_l": 5.4, "sodium_mmol_l": 140., "calcium_mmol_l": 2. if cell_model == "tentusscher-2006" else 1.8}
    if reference is not None:
        environment = dict(reference["cell_environment"])
        if environment["temperature_kelvin"] != 310. or environment["cell_type"] not in {"TP06_epi", "ORd_epi"}:
            raise ValueError("Reference cell environment not supported for matched comparison")
        environment["cell_type"] = manifest["cell_environment"]["cell_type"]
        manifest["cell_environment"] = environment
        manifest["extracellular_mmol_l"] = {"potassium_mmol_l": environment["potassium_mmol_l"],
            "sodium_mmol_l": environment["sodium_mmol_l"], "ionized_calcium_mmol_l": environment["calcium_mmol_l"]}
        manifest["comparison_reference"] = {"root": str(reference_root), "job_sha256": reference_hash,
            "selection": "first candidate IDs, independent of outcomes", "nominal_scale_caveat": "Same relative channel factors, not identical absolute conductances",
            "case_sha256": {str(case["id"]): hashlib.sha256((reference_root/"cases"/f"{case['id']:05d}.json").read_bytes()).hexdigest()
                            for case in cases}}
    manifest["dependency_versions"] = {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "psutil")}
    manifest["job_sha256"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    (root / "cases").mkdir()
    atomic_json(root / "status.json", {"phase": "prepared", "done": 0, "total": len(cases), "percent": 0.})
    return manifest


def verify_runtime(root, manifest):
    signed = dict(manifest)
    expected = signed.pop("job_sha256")
    if digest(signed) != expected:
        raise ValueError("Job manifest changed; new freeze required")
    from importlib.metadata import version
    if any(version(name) != expected for name, expected in manifest["dependency_versions"].items()):
        raise ValueError("Numerical dependencies changed since job freeze")
    runtime = Path(root) / "runtime"
    for name, expected in manifest["runtime_hashes"].items():
        path = (runtime/name).resolve()
        if not path.is_relative_to(runtime.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen simulation source changed: " + name)
    if Path(__file__).resolve() != (runtime / "vcg_biomarker/simulation/learning.py").resolve():
        raise ValueError("Start the learning job from its frozen runtime directory")


def candidate_task(root_string, case, manifest):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    from .ionic import CellExperiment, simulate_cell, simulate_preconditioned
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (OSError, psutil.AccessDenied):
        pass
    root = Path(root_string)

    def cancelled():
        return ((root/"cancel").exists() or (root/"budget_exhausted.flag").exists()
                or (root/"yield"/str(case["id"])).exists())

    began = time.monotonic()
    result = {"id": case["id"], "candidate_id": case.get("candidate_id", case["id"]),
              "cycle_ms": case.get("cycle_ms", manifest["cycle_ms"]),
              "partition": case["partition"], "scales": case["scales"], "job_sha256": manifest["job_sha256"]}
    try:
        spec = CellExperiment(model=manifest["model"], beats=manifest["beats"], cycle_ms=result["cycle_ms"],
                              output_ms=manifest["output_ms"], channel_scales=case["scales"],
                              rhs_backend=manifest.get("rhs_backend", "python"),
                              extracellular_mmol_l=manifest.get("extracellular_mmol_l", {}))
        precondition = manifest.get("precondition_max_beats", 0)
        if precondition:
            import numpy as np
            checkpoint_path = root/"cases"/f"{case['id']:05d}.state.npz"
            resume = None
            if checkpoint_path.is_file():
                with np.load(checkpoint_path, allow_pickle=False) as data:
                    if str(data["job_sha256"]) != manifest["job_sha256"]:
                        raise ValueError("Pre-pacing checkpoint belongs to a different job")
                    resume = {"beats": int(data["beats"]), "state": data["state"].copy(), "converged": bool(data["converged"])}

            def checkpoint(values):
                partial = checkpoint_path.with_suffix(".partial")
                with partial.open("wb") as handle:
                    np.savez_compressed(handle, **values, job_sha256=np.array(manifest["job_sha256"]))
                partial.replace(checkpoint_path)

            times, voltage, _, meta = simulate_preconditioned(spec, precondition, cancelled, checkpoint, resume)
            result["preconditioning"] = meta["preconditioning"]
        else:
            times, voltage, _, meta = simulate_cell(spec, cancelled=cancelled)
        result.update(status="computed", biomarkers=ap_features(times, voltage, spec.cycle_ms),
                      periodicity_check_passed=meta["periodicity_check_passed"],
                      cycle_pattern=meta.get("cycle_pattern"),
                      periodic_residuals=meta["periodic_residuals"], solver_source_sha256=meta["solver_source_sha256"])
        if precondition and not meta["preconditioning"]["converged"]:
            result.update(status="non_converged", reason="P1 pre-pacing criterion not achieved; excluded from P1 emulator, not declared nonphysiological")
    except InterruptedError:
        return {"id": case["id"], "status": "interrupted"}
    except (ValueError, RuntimeError, ArithmeticError, OverflowError) as exc:
        result.update(status="non_estimable", reason=str(exc), error_type=type(exc).__name__)
    result["elapsed_seconds"] = time.monotonic()-began
    atomic_json(root/"cases"/f"{case['id']:05d}.json", result)
    return result


def design(x):
    import numpy as np
    x = (np.asarray(x)-1.)/.5
    return np.column_stack([np.ones(len(x)), x] + [x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])])


def train_emulator(rows):
    import numpy as np
    from sklearn.model_selection import KFold
    complete = [r for r in rows if r["status"] == "computed"]
    groups = {k: [r for r in complete if r["partition"] == k] for k in ("fit", "calibration", "audit")}
    counts = {k: len(v) for k, v in groups.items()}
    if counts["fit"] < 40 or counts["calibration"] < 10 or counts["audit"] < 10:
        return {"status": "non_estimable", "reason": "insufficient_valid_protocols", "counts": counts}
    arrays = {k: (design([[r["scales"][p] for p in PARAMETERS] for r in values]),
                   np.array([[r["biomarkers"][p] for p in OUTPUTS] for r in values])) for k, values in groups.items()}
    x, y = arrays["fit"]
    weights = np.maximum(y.std(0), 1e-6)

    def fit(a, b, alpha):
        penalty = np.eye(a.shape[1])*alpha
        penalty[0, 0] = 0
        return np.linalg.solve(a.T@a + penalty, a.T@b)

    candidates = []
    for alpha in (.01, .1, 1., 10.):
        errors = []
        for a, b in KFold(3, shuffle=True, random_state=42).split(x):
            scale = np.maximum(y[a].std(0), 1e-6)
            errors.append(float(np.mean(np.abs(x[b]@fit(x[a], y[a], alpha)-y[b])/scale)))
        candidates.append((float(np.mean(errors)), alpha))
    _, alpha = min(candidates)
    coef = fit(x, y, alpha)
    calx, caly = arrays["calibration"]
    residual = np.sort(np.abs(calx@coef-caly), axis=0)
    rank = min(len(residual), int(np.ceil((len(residual)+1)*.9)))
    radius = residual[rank-1]
    ax, ay = arrays["audit"]
    absolute = np.abs(ax@coef-ay)
    return {"schema": "ionic-forward-emulator/1", "status": "research_surrogate", "clinical_use": False,
            "parameters": list(PARAMETERS), "outputs": list(OUTPUTS), "parameter_bounds": [.5, 1.5],
            "design": "intercept, (scale-1)/0.5, then products i<=j in parameter order",
            "coefficients": coef.tolist(), "alpha": alpha, "counts": counts, "calibration_radius_90": radius.tolist(),
            "audit_mae": absolute.mean(0).tolist(), "audit_baseline_mae": np.abs(ay-np.median(y, axis=0)).mean(0).tolist(),
            "audit_coverage_90": (absolute <= radius).mean(0).tolist(), "fit_output_scale": weights.tolist(),
            "warning": "Emulates this simulator and fixed protocol, not patients; intervals marginal per output. No unique inverse conductances."}


def train_campaign(rows, manifest, root):
    """Fit per protocol, retaining candidate-level separation across every rate."""
    root = Path(root)
    cycles = manifest.get("cycles_ms", [manifest["cycle_ms"]])
    membership = {}
    for row in rows:
        candidate = row.get("candidate_id", row["id"])
        if membership.setdefault(candidate, row["partition"]) != row["partition"]:
            raise ValueError("Candidate crossed data roles between protocols")
    models = []
    for cycle in cycles:
        selected = [r for r in rows if r.get("cycle_ms", manifest["cycle_ms"]) == cycle]
        result = train_emulator(selected)
        result["job_sha256"] = manifest["job_sha256"]
        result["protocol"] = {k: manifest[k] for k in ("model", "beats", "output_ms")}
        result["protocol"].update(cycle_ms=cycle, precondition_max_beats=manifest.get("precondition_max_beats", 0),
                                   cell_environment=manifest.get("cell_environment"))
        result["case_status_counts"] = {s: sum(r["status"] == s for r in selected) for s in sorted({r["status"] for r in selected})}
        if len(cycles) > 1:
            (root/"emulators").mkdir(exist_ok=True)
            atomic_json(root/"emulators"/f"cycle_{cycle:g}.json", result)
        models.append(result)
    if len(models) == 1:
        return models[0]
    from .identifiability import local_identifiability
    errors = dict(zip(OUTPUTS, [5., 1., .2, 2.]))  # Explicit hypothetical design tolerances, NOT patient precision.
    valid = [m for m in models if m["status"] == "research_surrogate"]
    diagnostic = local_identifiability(valid, dict.fromkeys(PARAMETERS, 1.), [errors]*len(valid)) if valid else None
    if diagnostic:
        diagnostic["tolerance_role"] = "hypothetical_cellular_experiment_design_only"
        diagnostic["measurement_tolerances"] = errors
        atomic_json(root/"identifiability.json", diagnostic)
    return {"schema": "ionic-protocol-campaign/1", "clinical_use": False,
            "status": "research_surrogates" if len(valid) == len(cycles) else "partial_or_non_estimable",
            "job_sha256": manifest["job_sha256"], "partition_unit": "conductance_candidate_across_all_protocols",
            "models": [{k: v for k, v in m.items() if k != "coefficients"} for m in models],
            "joint_numerical_rank": diagnostic["numerical_rank"] if diagnostic else None,
            "clinical_accuracy_established": False}


def compatible_population(rows, targets, reference_kind, protocol, manifest):
    """Return ALL compatible candidates; no single best genotype or fake posterior."""
    if reference_kind not in {"independent_cellular_action_potential", "synthetic_validation"}:
        raise ValueError("Surface ECG/diagnosis/QT cannot be substituted for cellular action-potential truth")
    if any(protocol.get(key) != manifest[key] for key in ("model", "beats", "cycle_ms", "output_ms")):
        raise ValueError("Observation and simulator protocols differ")
    if "cell_environment" in manifest and protocol.get("cell_environment") != manifest["cell_environment"]:
        raise ValueError("Cell type, temperature and bath concentrations must match")
    if reference_kind == "independent_cellular_action_potential" and (
            not protocol.get("reference_provenance") or protocol.get("measurement_method") != "ap_features_v1"):
        raise ValueError("Independent cellular reference and compatible measurement definitions required")
    if not targets or any(k not in OUTPUTS for k in targets):
        raise ValueError("Unsupported cellular target")
    import numpy as np
    if any(not np.isfinite(pair).all() or len(pair) != 2 or pair[1] <= 0 for pair in targets.values()):
        raise ValueError("Targets require value and positive explicit tolerance in the feature unit")
    accepted = []
    for row in rows:
        if row["status"] != "computed":
            continue
        if (reference_kind == "independent_cellular_action_potential" and protocol.get("steady_state_required", True)
                and not row["periodicity_check_passed"]):
            continue
        if all(abs(row["biomarkers"][k]-value) <= tolerance for k, (value, tolerance) in targets.items()):
            accepted.append({"id": row["id"], "scales": row["scales"], "periodicity_check_passed": row["periodicity_check_passed"]})
    return {"status": "compatible_population" if accepted else "no_compatible_candidate", "candidates": accepted,
            "reference_kind": reference_kind, "unique_parameters_identified": False, "posterior_probabilities": None,
            "clinical_use": False}


def predict_emulator(model, scales):
    import numpy as np
    if (model.get("schema") != "ionic-forward-emulator/1" or model.get("clinical_use") is not False
            or model.get("parameters") != list(PARAMETERS) or model.get("outputs") != list(OUTPUTS)):
        raise ValueError("Unknown forward-emulator contract")
    if set(scales) != set(PARAMETERS):
        raise ValueError("All trained conductance factors required")
    x = np.array([scales[k] for k in PARAMETERS], float)
    if not np.isfinite(x).all() or np.any((x < .5) | (x > 1.5)):
        raise ValueError("Outside trained parameter domain; use the real solver")
    matrix = np.asarray(model["coefficients"], float)
    radius = np.asarray(model["calibration_radius_90"], float)
    features = design(x[None, :])
    if matrix.shape != (features.shape[1], len(OUTPUTS)) or radius.shape != (len(OUTPUTS),):
        raise ValueError("Invalid emulator coefficients")
    if not np.isfinite(matrix).all() or not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError("Non-finite emulator or uncertainty")
    mean = (features@matrix)[0]
    return {"prediction": dict(zip(OUTPUTS, mean.tolist())),
            "marginal_intervals_90": {name: [float(v-r), float(v+r)] for name, v, r in zip(OUTPUTS, mean, radius)},
            "protocol": model.get("protocol"), "clinical_use": False, "meaning": "approximation to frozen simulator only"}


def admission_target(cpus, system_cpu_percent, own_cpu_cores, available_gib, worker_gib, maximum=4):
    import math
    foreign = max(0., cpus*system_cpu_percent/100-own_cpu_cores)
    cpu = max(0, math.floor(cpus-1-foreign))
    ram = max(0, math.floor((available_gib-2.)/max(.5, worker_gib)))
    return min(maximum, cpu, ram)


def run(root, resume=False):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import psutil
    from .worker import acquire_worker_lock
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    verify_runtime(root, manifest)
    coordinator = acquire_worker_lock(root.parent/"population-learning.lock")
    lease = acquire_worker_lock(root/"learning.lock")
    yield_dir = root/"yield"
    yield_dir.mkdir(exist_ok=True)
    for case in manifest["cases"]:
        flag = yield_dir/str(case["id"])
        if flag.exists():
            flag.unlink()
    if resume:
        for name in ("cancel", "pause", "budget_exhausted.flag"):
            path = root/name
            if path.is_file():
                path.unlink()
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    completed = {}
    for path in (root/"cases").glob("*.json"):
        row = json.loads(path.read_text())
        if row["job_sha256"] != manifest["job_sha256"] or row["id"] not in range(len(manifest["cases"])):
            raise ValueError("Incompatible checkpoint")
        completed[row["id"]] = row
    pending = [c for c in manifest["cases"] if c["id"] not in completed]
    limit, stable, worker_gib = 1, 0, (1. if manifest.get("rhs_backend") == "numba" else .5)
    last_sample, began, paused_seconds = 0., time.monotonic(), 0.
    children = {}
    from .resources import PagingMonitor, capacity_target, actionable_paging_pressure
    paging_monitor = PagingMonitor()
    psutil.cpu_percent(None)
    status = {"done": len(completed), "total": len(manifest["cases"]), "pid": os.getpid(),
              "job_sha256": manifest["job_sha256"], "percent": 90*len(completed)/len(manifest["cases"])}
    try:
        # Spawn is lazy: admit one measured task first, not a full eagerly filled queue.
        maximum = min(manifest["max_workers"], max(1, (psutil.cpu_count() or 2)-1))
        while psutil.virtual_memory().available/2**30 < 2.75:
            atomic_json(root/"status.json", {"phase": "waiting_for_startup_ram", "done": len(completed), "total": len(manifest["cases"])})
            if (root/"cancel").exists():
                return
            time.sleep(2)
        with ProcessPoolExecutor(max_workers=maximum, mp_context=multiprocessing.get_context("spawn")) as pool:
            active = {}
            while pending or active:
                now = time.monotonic()
                elapsed = now-began-paused_seconds
                stop = (root/"cancel").exists() or elapsed >= manifest["max_active_seconds"]
                stop = stop or (root/"budget_exhausted.flag").exists()
                if elapsed >= manifest["max_active_seconds"] and not (root/"budget_exhausted.flag").exists():
                    (root/"budget_exhausted.flag").touch()
                paused = (root/"pause").exists()
                if now-last_sample >= 2:
                    own = 0.
                    working_residency = []
                    for child in process.children(recursive=True):
                        try:
                            if child.pid not in children:
                                children[child.pid] = child
                                child.cpu_percent(None)
                            else:
                                cores = children[child.pid].cpu_percent(None)/100
                                own += cores
                                if cores > .01:
                                    working_residency.append(child.memory_info().rss/2**30)
                            memory = child.memory_info()
                            worker_gib = max(worker_gib, getattr(memory, "peak_wset", memory.rss)/2**30*1.5)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    free = psutil.virtual_memory().available/2**30
                    cpu = psutil.cpu_percent(None)
                    paging = paging_monitor.sample()
                    memory_paging_pressure = actionable_paging_pressure(free, worker_gib, paging["paging_pressure"])
                    # Only observed working children, never idle-process residency credits.
                    active_resident = sum(sorted(working_residency)[:len(active)])
                    target = capacity_target(psutil.cpu_count() or 2, cpu, own, free, worker_gib, maximum,
                                             paging["paging_pressure"], len(active), active_resident)
                    if target < limit:
                        limit, stable = target, 0
                    elif target > limit:
                        stable += 1
                        if stable >= 3:
                            limit, stable = min(target, limit+1), 0
                    else:
                        stable = 0
                    last_sample = now
                    # Shed one task at a time at its safe solver boundary. Its last
                    # complete pre-pacing chunk is resumable; never kill all workers.
                    if len(active) > limit and (free < 2+worker_gib or memory_paging_pressure):
                        flagged = any((yield_dir/str(i)).exists() for i in active.values())
                        if not flagged:
                            (yield_dir/str(list(active.values())[-1])).touch()
                    status = {"phase": "cancelling" if stop else "paused" if paused else "computing",
                              "done": len(completed), "total": len(manifest["cases"]),
                              "percent": 90*len(completed)/len(manifest["cases"]),
                              "active": len(active), "admitted": limit, "target": target,
                              "free_gib": free, "cpu_percent": cpu, "estimated_worker_gib": worker_gib,
                              "active_resident_gib_observed": active_resident,
                              "limit_reason": "paging_with_low_ram" if memory_paging_pressure else "ram" if free < 2+worker_gib else "cpu_or_budget" if target < maximum else "configured_budget",
                              "actionable_paging_pressure": memory_paging_pressure,
                              "elapsed_active_seconds": elapsed, "pid": os.getpid(),
                              "eta_seconds": None, "job_sha256": manifest["job_sha256"]}
                    status.update(paging)
                    durations = [r["elapsed_seconds"] for r in completed.values() if r["status"] == "computed"]
                    if len(durations) >= 6 and limit and not paused and not stop:
                        import statistics
                        status["eta_seconds"] = statistics.median(durations)*(len(pending)+len(active))/limit
                        status["eta_scope"] = "empirical simulation work only; emulator fit separately reported"
                    atomic_json(root/"status.json", status)
                    with (root/"resource_telemetry.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(status, allow_nan=False)+"\n")
                while pending and len(active) < limit and not stop and not paused:
                    case = pending.pop(0)
                    active[pool.submit(candidate_task, str(root), case, manifest)] = case["id"]
                if stop and not active:
                    break
                finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED) if active else (set(), set())
                for future in finished:
                    row = future.result()
                    case_id = active.pop(future)
                    flag = yield_dir/str(case_id)
                    if flag.exists():
                        flag.unlink()
                    if row["status"] != "interrupted":
                        completed[row["id"]] = row
                    elif not stop:
                        pending.append(manifest["cases"][case_id])
                if not active:
                    time.sleep(.2)
                    if paused:
                        paused_seconds += .2
            if len(completed) == len(manifest["cases"]):
                atomic_json(root/"status.json", {**status, "phase": "training_emulator", "percent": 95., "active": 0})
                result = train_campaign([completed[k] for k in sorted(completed)], manifest, root)
                atomic_json(root/"emulator.json", result)
                if manifest.get("comparison_reference"):
                    from .model_comparison import compare_campaigns
                    atomic_json(root/"model_comparison.json", compare_campaigns(root, manifest))
                status.update(phase="complete", percent=100., done=len(completed), active=0, result_status=result["status"],
                              eta_seconds=0., elapsed_active_seconds=time.monotonic()-began-paused_seconds)
            else:
                status.update(phase="cancelled" if (root/"cancel").exists() else "budget_exhausted",
                              active=0, done=len(completed), eta_seconds=None,
                              percent=90*len(completed)/len(manifest["cases"]),
                              elapsed_active_seconds=time.monotonic()-began-paused_seconds)
            atomic_json(root/"status.json", status)
    except Exception as exc:
        atomic_json(root/"status.json", {**status, "phase": "failed", "error": str(exc), "done": len(completed)})
        raise
    finally:
        paging_monitor.close()
        lease.close()
        coordinator.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "resume"))
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--beats", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--precondition-max-beats", type=int, default=0)
    parser.add_argument("--max-active-seconds", type=int, default=1800)
    parser.add_argument("--cycles-ms", type=float, nargs="+", default=[800.])
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--native-rhs", action="store_true", help="Not released; requests are explicitly rejected")
    parser.add_argument("--cell-model", choices=("tentusscher-2006", "ohara-2011"), default="tentusscher-2006")
    parser.add_argument("--reference-campaign")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.job_root, args.samples, args.beats, args.max_workers, seed=args.seed,
                         precondition_max_beats=args.precondition_max_beats, max_active_seconds=args.max_active_seconds,
                         cycles_ms=args.cycles_ms, native_rhs=args.native_rhs,
                         cell_model=args.cell_model, reference_campaign=args.reference_campaign)
        print(json.dumps({"status": "prepared", "cases": len(result["cases"]), "job_sha256": result["job_sha256"]}))
    else:
        run(args.job_root, resume=args.action == "resume")


if __name__ == "__main__":
    main()
