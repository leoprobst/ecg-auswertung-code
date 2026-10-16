"""Computed circulating waves on a reduced ring, not an animation or VT diagnosis."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix, eye, kron
from .solver import reaction


def activation_trains(times, voltage, threshold=.5):
    times, voltage = np.asarray(times), np.asarray(voltage)
    if voltage.ndim != 2 or len(times) != len(voltage) or not np.isfinite(voltage).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite ordered trajectories required")
    crossing = (voltage[:-1] < threshold) & (voltage[1:] >= threshold)
    trains = []
    for node in range(voltage.shape[1]):
        idx = np.flatnonzero(crossing[:, node])
        fraction = (threshold-voltage[idx, node])/(voltage[idx+1, node]-voltage[idx, node])
        trains.append(times[idx] + fraction*np.diff(times)[idx])
    return trains


def count_ring_circulation(trains, max_neighbor_delay_ms=30.):
    """Verify propagation around every neighbor and return to the starting node.

    Repeated activation of one node alone is insufficient. Only known ordered
    ring geometry is accepted, not arbitrary clinical tissue or ECG traces.
    """
    if not trains or not len(trains[0]):
        return {"complete_circuits": 0, "cycle_lengths_ms": []}
    now = float(trains[0][0])
    started = now
    circuits, periods = 0, []
    for step in range(1, sum(map(len, trains))+1):
        node = step % len(trains)
        candidates = trains[node][trains[node] >= now-1.]  # sampled-front timing tolerance
        if not len(candidates) or candidates[0]-now > max_neighbor_delay_ms:
            break
        now = max(now, float(candidates[0]))
        if node == 0:
            if now <= started+10:
                break
            periods.append(now-started)
            started = now
            circuits += 1
    return {"complete_circuits": circuits, "cycle_lengths_ms": periods}


def simulate_ring(duration_ms=1800., ablate_at_ms=None, progress=None, cancelled=None):
    if not np.isfinite(duration_ms) or not 300 <= duration_ms <= 3000:
        raise ValueError("Ring duration must be 300–3000 ms")
    if ablate_at_ms is not None and not 300 < ablate_at_ms < duration_ms:
        raise ValueError("Ablation must follow induction and precede the end")
    n, spacing, diffusion = 240, 1.5, .8
    rows = np.arange(n)
    cols = (rows+1) % n

    def laplacian(blocked):
        weight = np.full(n, diffusion/spacing**2)
        if blocked:
            weight[-1] = 0
        adj = coo_matrix((np.r_[weight, weight], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
        return adj - coo_matrix((np.asarray(adj.sum(1)).ravel(), (rows, rows)), shape=(n, n)).tocsr()

    initial = np.r_[np.zeros(n), np.ones(n)]
    times = np.arange(0, duration_ms+1e-6, 1.)
    if times[-1] != duration_ms:
        times = np.r_[times, duration_ms]
    voltage = np.empty((len(times), n), np.float32)
    boundaries = sorted({0., 2., 300., float(duration_ms)} | ({float(ablate_at_ms)} if ablate_at_ms else set()))
    pattern = kron(np.ones((2, 2)), eye(n), format="lil")
    pattern[:n, :n] = (laplacian(False) != 0) + eye(n)
    pattern = pattern.tocsr()
    began = time.monotonic()
    calls = 0
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        lap = laplacian(a < 300 or (ablate_at_ms is not None and a >= ablate_at_ms))
        pacing = a < 2

        def rhs(t, state):
            nonlocal calls
            calls += 1
            if calls % 200 == 0:
                if cancelled and cancelled():
                    raise InterruptedError("Ring experiment cancelled")
                if progress:
                    progress(t, duration_ms, time.monotonic()-began)
            v, h = state[:n], state[n:]
            dv, dh = reaction(v, h, np.full(n, 150.))
            dv += lap@v
            if pacing:
                dv[:3] += .8
            return np.r_[dv, dh]

        mask = (times >= a) & (times <= b)
        output = np.unique(np.r_[times[mask], b])
        fit = solve_ivp(rhs, (a, b), initial, method="BDF", t_eval=output, jac_sparsity=pattern,
                        rtol=1e-5, atol=1e-8, max_step=2)
        if not fit.success or not np.isfinite(fit.y).all():
            raise RuntimeError("Ring solver failed")
        voltage[mask] = fit.y[:n, np.searchsorted(output, times[mask])].T
        initial = fit.y[:, -1]
    if voltage.min() < -.01 or voltage.max() > 2:
        raise ArithmeticError("Invalid ring voltage")
    trains = activation_trains(times, voltage)
    metrics = count_ring_circulation(trains)
    angle = np.arange(n)*2*np.pi/n
    xyz = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(n)])*(n*spacing/(2*np.pi))
    meta = {"schema": "reentry-ring/1", "clinical_use": False, "voltage_units": "normalized",
            "induction": "S1 0–2ms; one edge temporarily blocked until 300ms, then released",
            "ablate_at_ms": ablate_at_ms, "duration_ms": duration_ms, **metrics,
            "node0_activation_times_ms": trains[0].tolist(), "elapsed_seconds": time.monotonic()-began,
            "interpretation": "mechanistic circulation experiment; imposed induction gate, not clinical AVNRT/VT or patient anatomy"}
    return times, xyz, voltage, meta
