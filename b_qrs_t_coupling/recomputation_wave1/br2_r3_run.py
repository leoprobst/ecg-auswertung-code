"""Recomputation R2/R3: extended scenario grid with population uncertainty, differential and persistent error models (development folds only).

Phases:  test | point | boot <fold> | aggregate
Contract: the pre-registered protocol of the recomputation (its SHA-256 is written to the output folder before the first real computation).
Data:    training/validation only (footer check of the split column before every read, read filter, assertion). No test file.
Output:  private aggregates per draw (npz, no identifiers) in the output folder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
import sys
import time
import json
import threading
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "werkzeuge/aw241"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import br2_r3_kern as K  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

OUT = Path(str(data_path("b.aw241_r2r3")))
PROTOKOLL = ROOT / "nachrechnung/AW241/B_R2R3_PROTOKOLL.md"
A, F = C.A, C.F
SEED = 2412026
NBOOT = 1000
MINVALID = 950
FOLDS = ["train", "validation"]
# R2: scenario grid
RA_R2 = np.round(np.arange(40, 81) / 100, 2)
RF_R2 = np.round(np.arange(60, 101, 5) / 100, 2)
RHO_R2 = np.round(np.arange(-4, 9) / 10, 1)
# R3 (a): shift on a 2D grid (point estimate), control SD
B2D = np.round(np.arange(-6, 7) / 10, 1)
K2D = np.round(np.arange(10, 21) / 10, 1)
PI2D = np.round(np.arange(0, 11) / 20, 2)
ANCH = ["P_all", "P_zg", "EF_mix", "EF_pool"]
REF_ANCH = {"N0": None, "I_gesamt": (0.617, 0.837, 0.268), "I_tech": (0.746, 0.977, 0.268)}   # Addendum 1: assumed comparison anchors (N0 = no correction)
SHIFT_ANCH = ["P_all", "EF_mix", "N0", "I_gesamt", "I_tech"]
PERSIST_ANCH = ["P_all", "EF_mix", "N0", "I_gesamt", "I_tech"]
SHIFT_PATHS = ["A", "F", "AF"]
B_AXIS = K.B_COARSE
PEAK = [0]


def guard():
    C.guard()


def start_watch(limit_gb=1.2):
    import psutil

    def watch():
        while True:
            PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss)
            if PEAK[0] > limit_gb * 1024 ** 3:
                os._exit(75)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


NACHTRAG1 = ROOT / "nachrechnung/AW241/B_R2R3_NACHTRAG1.md"


def check_stamp():
    s = C.load(OUT / "protokoll_hash.json")
    assert s["sha256"] == C.sha(PROTOKOLL), "Protocol changed after binding"
    n = C.load(OUT / "nachtrag1_hash.json")
    assert n["sha256"] == C.sha(NACHTRAG1), "Addendum 1 changed after binding"
    return {"protokoll": s, "nachtrag1": n}


# ------------------------------------------------------------------ Data

def load_inputs():
    f, finfo, scale = C.outcome_frame()
    sdA, sdF = scale["sd"][-2], scale["sd"][-1]
    ref = f.loc[f.eligible_reference.eq(True) & np.isfinite(f.preferred_lvef.to_numpy(float))].drop_duplicates("subject_id")
    efmap = pd.Series(classify_lvef(ref.preferred_lvef.to_numpy(float)), index=ref.subject_id.to_numpy())
    cols = ["subject_id", "split", "first_study_id", "second_study_id", "hours_between", A + "_first", A + "_second", F + "_first", F + "_second"]
    p, pinfo = C.read(C.PAIRS, cols)
    q, qinfo = C.pair_frame()
    kq = pd.MultiIndex.from_frame(q[["subject_id", "first_study_id", "second_study_id"]])
    kp = pd.MultiIndex.from_frame(p[["subject_id", "first_study_id", "second_study_id"]])
    amp_ok = kp.isin(kq)
    P = {"sid": p.subject_id.to_numpy(), "amp_ok": np.asarray(amp_ok), "zg": p.hours_between.eq(0).to_numpy(),
         "ef": p.subject_id.map(efmap).to_numpy(float)}
    P["sb"] = P["amp_ok"] & np.isfinite(P["ef"])
    dA = (p[A + "_second"].to_numpy(float) - p[A + "_first"].to_numpy(float)) / sdA
    dF = (p[F + "_second"].to_numpy(float) - p[F + "_first"].to_numpy(float)) / sdF
    P["delta"] = np.column_stack([dA, dF])
    assert np.isfinite(P["delta"]).all()
    g = f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    raw, names = raw_design(g, F)
    z = fixed_design(raw, scale)
    y = classify_lvef(g.preferred_lvef.to_numpy(float))
    folds = {}
    for fold in FOLDS:
        ix = g.split.eq(fold).to_numpy()
        ids = g.subject_id.to_numpy()[ix]
        union = np.union1d(ids, P["sid"])
        folds[fold] = {"z": z[ix], "y": y[ix], "ids": ids, "union": union, "oi": np.searchsorted(union, ids), "pi": np.searchsorted(union, P["sid"])}
    info = {"population": finfo, "pairs": pinfo, "pairs_amp": qinfo, "pairs_n": int(len(p)), "pairs_amp_ok": int(amp_ok.sum()), "pairs_sb": int(P["sb"].sum()),
            "pairs_zeitgleich": int(P["zg"].sum()), "scaling_sd_tail": scale["sd"][-2:], "test_access": False}
    return P, folds, info


# ------------------------------------------------------------------ Evaluation of one weighting (point or draw)

def error_matrices(P, y, w_o, w_p):
    d = P["delta"]
    E = {"P_all": K.pair_error(d, w_p)}
    zg = P["zg"]
    E["P_zg"] = K.pair_error(d[zg], w_p[zg]) if w_p[zg].sum() > 30 else None
    sb = P["sb"]
    ef = P["ef"]
    low, pres = sb & (ef == 1), sb & (ef == 0)
    if w_p[low].sum() < 30 or w_p[pres].sum() < 30:
        raise ValueError("too few pairs per EF group")
    Ec = K.pair_error(d[low], w_p[low])
    Ee = K.pair_error(d[pres], w_p[pres])
    pc = float(np.sum(w_o * y) / np.sum(w_o))
    E["EF_mix"] = (1 - pc) * Ee + pc * Ec
    E["EF_pool"] = K.pair_error(d[sb], w_p[sb])
    return E, Ec, Ee, pc


def anchor_block(c, E):
    out = {}
    for name in ANCH:
        Ex = E[name]
        if Ex is None or np.linalg.eigvalsh(Ex).min() < -1e-10:
            out[name] = None
            continue
        rel = K.rel_of(c, Ex)
        lf, la, lam, pcorr = K.corrected(c, Ex)
        kp = {t: K.kipp(c, [rel["R_F"]], [rel["rho"]], t) for t in K.TARGETS}
        out[name] = {"rel": [rel["R_A"], rel["R_F"], rel["rho"]], "logor": lf, "lam": lam, "pcorr": pcorr,
                     "kipp": {t: (float(kp[t]["ra"][0, 0]), int(kp[t]["status"][0, 0]), float(kp[t]["bound"][0, 0])) for t in K.TARGETS}}
    return out


def shift_design(z, y, w, sA, sF):
    zz = z.copy()
    zz[:, -2] -= sA * y
    zz[:, -1] -= sF * y
    return zz


def control_sd_and_gap(z, y, w):
    out = []
    for col in (-2, -1):
        x = z[:, col]
        m0 = np.average(x[y == 0], weights=w[y == 0])
        m1 = np.average(x[y == 1], weights=w[y == 1])
        sd0 = np.sqrt(np.average((x[y == 0] - m0) ** 2, weights=w[y == 0]))
        out.append((sd0, (m1 - m0) / sd0))
    return out


class ShiftEval:
    """Computes components for shifts (control SD) with a cache; angle A, marker F, both together AF (fraction phi of the observed gap)."""

    def __init__(self, z, y, w, b0, base_c):
        self.z, self.y, self.w, self.b0 = z, y, w, b0
        (self.sdA, self.dA), (self.sdF, self.dF) = control_sd_and_gap(z, y, w)
        self.cache = {}
        self.base = base_c

    def comps(self, path, b):
        key = (path, round(b, 9))
        if key in self.cache:
            return self.cache[key]
        if b == 0:
            c = self.base
        else:
            sA, sF = {"A": (b * self.sdA, 0.0), "F": (0.0, b * self.sdF), "AF": (b * self.dA * self.sdA, b * self.dF * self.sdF)}[path]
            try:
                c = K.components(shift_design(self.z, self.y, self.w, sA, sF), self.y, self.w, logistic_fit, initial=self.b0)
            except Exception:  # noqa: BLE001
                c = None
        self.cache[key] = c
        return c

    def evaluator(self, path, E):
        def f(b):
            c = self.comps(path, b)
            if c is None:
                return np.nan, False
            lf, _, lam, _ = K.corrected(c, E)
            return lf, bool(np.isfinite(lf))
        return f

    def comps_xy(self, sAb, sFb):
        key = ("xy", round(sAb, 9), round(sFb, 9))
        if key in self.cache:
            return self.cache[key]
        try:
            c = K.components(shift_design(self.z, self.y, self.w, sAb * self.sdA, sFb * self.sdF), self.y, self.w, logistic_fit, initial=self.b0) if (sAb or sFb) else self.base
        except Exception:  # noqa: BLE001
            c = None
        self.cache[key] = c
        return c


TIP_KEYS = []
for _t in K.TARGETS:
    for _a in SHIFT_ANCH:
        for _p in SHIFT_PATHS:
            TIP_KEYS.append(("shift", _a, _p, "", _t))
    for _a in PERSIST_ANCH:
        for _rp in ("rho0", "rhoE"):
            for _p in ("A", "F", "AF"):
                TIP_KEYS.append(("persist", _a, _p, _rp, _t))
    for _p in ("A", "F", "AF"):
        TIP_KEYS.append(("kappa", "EF_mix", _p, "", _t))


def tip_row(c, E, Ec, Ee, pc, SE):
    """All tipping points (status, value, bound) of one weighting in the order of TIP_KEYS."""
    res = {}
    for t in K.TARGETS:
        for a in SHIFT_ANCH:
            Ex = E[a]
            for p in SHIFT_PATHS:
                res[("shift", a, p, "", t)] = K.tip_shift(SE.evaluator(p, Ex), B_AXIS, t)
        for a in PERSIST_ANCH:
            Ex = E[a]
            rho_base = float(Ex[0, 1] / np.sqrt(Ex[0, 0] * Ex[1, 1])) if Ex[0, 0] > 0 and Ex[1, 1] > 0 else 0.0
            for rp, rv in (("rho0", 0.0), ("rhoE", rho_base)):
                paths = K.persist_paths(c, Ex, rv)
                for p in ("A", "F", "AF"):
                    lo, ok = paths[p]
                    res[("persist", a, p, rp, t)] = K.tip_on_axis(K.PI_SCAN, lo, ok, t)
        kp = K.kappa_paths(c, Ee, Ec, pc)
        for p in ("A", "F", "AF"):
            lo, ok = kp[p]
            res[("kappa", "EF_mix", p, "", t)] = K.tip_on_axis(K.KAPPA, lo, ok, t)
    return [res[k] for k in TIP_KEYS]


def evaluate(P, fd, w_o, w_p, b0, with_r3=True):
    z, y = fd["z"], fd["y"]
    c = K.components(z, y, w_o, logistic_fit, initial=b0)
    E, Ec, Ee, pc = error_matrices(P, y, w_o, w_p)
    V = c["V"]
    for name, spec in REF_ANCH.items():
        if spec is None:
            E[name] = np.zeros((2, 2))
        else:
            eA, eF = (1 - spec[0]) * V[0, 0], (1 - spec[1]) * V[1, 1]
            E[name] = np.array([[eA, spec[2] * np.sqrt(eA * eF)], [spec[2] * np.sqrt(eA * eF), eF]])
    out = {"c": c, "E": E, "Ec": Ec, "Ee": Ee, "pc": pc}
    out["anchors"] = anchor_block(c, E)
    g = K.grid(c, RA_R2, RF_R2, RHO_R2)
    out["grid"] = g["logor"].astype(np.float32)
    out["kipp"] = {t: K.kipp(c, RF_R2, RHO_R2, t) for t in K.TARGETS}
    if with_r3:
        SE = ShiftEval(z, y, w_o, b0, c)
        out["gap"] = [SE.dA, SE.dF, SE.sdA, SE.sdF]
        out["tips"] = tip_row(c, E, Ec, Ee, pc, SE)
        out["SE"] = SE
    return out


# ------------------------------------------------------------------ Phases

def stamp():
    p = OUT / "protokoll_hash.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise SystemExit("Hash already bound")
    C.dump(p, {"path": str(PROTOKOLL), "sha256": C.sha(PROTOKOLL), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "before_computation": True})
    print("STAMP", C.sha(PROTOKOLL))


def point():
    s = check_stamp()
    P, folds, info = load_inputs()
    res = {"protokoll": s, "script_sha256": {n: C.sha(ROOT / "werkzeuge/aw241" / n) for n in ("br2_r3_run.py", "br2_r3_kern.py")}, "info": info, "folds": {}}
    pub = {"train": {"naive": 2.480, "joint_all": 8.737}, "validation": {"naive": 2.392, "joint_all": 9.259}}
    for fold in FOLDS:
        fd = folds[fold]
        w_o = np.ones(len(fd["y"]))
        w_p = np.ones(len(P["sid"]))
        b0 = logistic_fit(fd["z"], fd["y"], w_o)
        ev = evaluate(P, fd, w_o, w_p, b0)
        an = ev["anchors"]
        gate = {"naive_or": float(np.exp(b0[-1])), "published_naive": pub[fold]["naive"], "P_all_or": float(np.exp(an["P_all"]["logor"])), "published_P_all": pub[fold]["joint_all"]}
        gate["ok"] = abs(gate["naive_or"] - gate["published_naive"]) <= 0.002 and abs(gate["P_all_or"] - gate["published_P_all"]) <= 0.002
        res["folds"][fold] = {"gate": gate, "anchors": {k: v for k, v in an.items()}, "gap": ev["gap"], "pc": ev["pc"],
                              "Ec": ev["Ec"], "Ee": ev["Ee"], "E": {k: v for k, v in ev["E"].items()}, "S": ev["c"]["S"], "V": ev["c"]["V"], "u": ev["c"]["u"], "b": ev["c"]["b"],
                              "grid_logor": ev["grid"], "kipp": {str(t): {kk: vv for kk, vv in v.items()} for t, v in ev["kipp"].items()},
                              "tips": {"|".join(map(str, k)): list(v) for k, v in zip(TIP_KEYS, ev["tips"])}}
        # 2D grid (point estimate)
        SE = ev["SE"]
        for a in SHIFT_ANCH:
            Ex = ev["E"][a]
            M = np.full((len(B2D), len(B2D)), np.nan)
            for i, bA in enumerate(B2D):
                for j, bF in enumerate(B2D):
                    c = SE.comps_xy(float(bA), float(bF))
                    if c is not None:
                        M[i, j] = K.corrected(c, Ex)[0]
            res["folds"][fold]["shift2d_" + a] = M
        Kmat = np.full((len(K2D), len(K2D)), np.nan)
        for i, ka in enumerate(K2D):
            for j, kf in enumerate(K2D):
                Ee, Ec, pc = ev["Ee"], ev["Ec"], ev["pc"]
                D = np.diag([np.sqrt(ka), np.sqrt(kf)])
                Ek = (1 - pc) * Ee + pc * D @ Ec @ D
                Kmat[i, j] = K.corrected(ev["c"], Ek)[0]
        res["folds"][fold]["kappa2d"] = Kmat
        print("POINT", fold, json.dumps(C.clean(gate)), flush=True)
        if not gate["ok"]:
            C.dump(OUT / "point.json", res)
            raise SystemExit("STOP: reproduction gate not passed " + fold)
    C.dump(OUT / "point.json", res)
    print("POINT done", flush=True)


def boot(fold):
    check_stamp()
    P, folds, info = load_inputs()
    fd = folds[fold]
    fi = FOLDS.index(fold)
    path = OUT / f"boot_{fold}.npz"
    n = len(fd["union"])
    nan = np.nan
    b_base = logistic_fit(fd["z"], fd["y"], np.ones(len(fd["y"])))
    store = None
    done = 0
    fails = {}
    if path.exists():
        zf = np.load(path, allow_pickle=True)
        store = {k: list(zf[k]) for k in zf.files if k not in ("done", "fails")}
        done = int(zf["done"])
        fails = json.loads(str(zf["fails"]))
    else:
        store = {k: [] for k in ("r", "S", "V", "u", "b", "grid", "kipp_ra", "kipp_st", "kipp_bd", "a_rel", "a_logor", "a_lam", "a_pc", "a_kra", "a_kst", "a_kbd", "tips", "gap", "Ec", "Ee", "pc")}

    def save(upto):
        arrs = {k: np.asarray(v) for k, v in store.items()}
        np.savez_compressed(path, done=upto, fails=json.dumps(fails), **arrs)

    t0 = time.time()
    for r in range(done, NBOOT):
        if r % 20 == 0:
            guard()
            print("BOOT", fold, r, len(store["r"]), round(time.time() - t0), flush=True)
            if r > done:
                save(r)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, fi, r]))
        mult = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        w_o, w_p = mult[fd["oi"]], mult[fd["pi"]]
        try:
            ev = evaluate(P, fd, w_o, w_p, b_base)
        except Exception as ex:  # noqa: BLE001
            fails[str(ex)[:80]] = fails.get(str(ex)[:80], 0) + 1
            continue
        c = ev["c"]
        an = ev["anchors"]
        store["r"].append(r)
        store["S"].append([c["S"][0, 0], c["S"][0, 1], c["S"][1, 1]])
        store["V"].append([c["V"][0, 0], c["V"][0, 1], c["V"][1, 1]])
        store["u"].append(c["u"])
        store["b"].append(c["b"][-2:])
        store["grid"].append(ev["grid"])
        store["kipp_ra"].append(np.stack([ev["kipp"][t]["ra"] for t in K.TARGETS]).astype(np.float32))
        store["kipp_st"].append(np.stack([ev["kipp"][t]["status"] for t in K.TARGETS]).astype(np.int8))
        store["kipp_bd"].append(np.stack([ev["kipp"][t]["bound"] for t in K.TARGETS]).astype(np.float32))
        store["a_rel"].append([an[a]["rel"] if an[a] else [nan] * 3 for a in ANCH])
        store["a_logor"].append([an[a]["logor"] if an[a] else nan for a in ANCH])
        store["a_lam"].append([an[a]["lam"] if an[a] else nan for a in ANCH])
        store["a_pc"].append([an[a]["pcorr"] if an[a] else nan for a in ANCH])
        store["a_kra"].append([[an[a]["kipp"][t][0] if an[a] else nan for t in K.TARGETS] for a in ANCH])
        store["a_kst"].append([[an[a]["kipp"][t][1] if an[a] else 0 for t in K.TARGETS] for a in ANCH])
        store["a_kbd"].append([[an[a]["kipp"][t][2] if an[a] else nan for t in K.TARGETS] for a in ANCH])
        store["tips"].append([[s, v, g] for (s, v, g) in ev["tips"]])
        store["gap"].append(ev["gap"])
        store["Ec"].append([ev["Ec"][0, 0], ev["Ec"][0, 1], ev["Ec"][1, 1]])
        store["Ee"].append([ev["Ee"][0, 0], ev["Ee"][0, 1], ev["Ee"][1, 1]])
        store["pc"].append(ev["pc"])
    save(NBOOT)
    C.dump(OUT / f"boot_{fold}_meta.json", {"fold": fold, "attempted": NBOOT, "valid": len(store["r"]), "failures": fails, "union_persons": int(n), "seed": [SEED, fi], "peak_rss": PEAK[0], "seconds": round(time.time() - t0)})
    print("BOOT_DONE", fold, len(store["r"]), fails, flush=True)


# ------------------------------------------------------------------ Test with synthetic data

def synthetic_test():
    """Function test without data access: synthetic data with a known structure, gates on internal consistency."""
    rng = np.random.default_rng(3)
    n = 4000
    other = np.column_stack([np.ones(n), rng.normal(size=(n, 4))])
    lat = rng.multivariate_normal([0, 0], [[1, .5], [.5, 1]], size=n)
    E = np.array([[.35, .03], [.03, .18]])
    W = lat + rng.multivariate_normal([0, 0], E, size=n)
    lin = other @ np.array([-2.2, .3, .1, 0, .1]) + lat @ np.array([.2, .5])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(float)
    z = np.column_stack([other, W])
    w = np.ones(n)
    c = K.components(z, y, w, logistic_fit)
    lf, la, lam, pc = K.corrected(c, E)
    print("naive marker OR", np.exp(c["b"][-1]), "corrected", np.exp(lf), "true", np.exp(.5), "lam", lam)
    # Grid consistency: the grid at the point (RA,RF,rho) from E reproduces corrected(E)
    rel = K.rel_of(c, E)
    g = K.grid(c, [rel["R_A"]], [rel["R_F"]], [rel["rho"]])
    assert abs(g["logor"][0, 0, 0] - lf) < 1e-9, (g["logor"][0, 0, 0], lf)
    # tipping point: monotonicity check (value directly below R_A* is < target)
    kp = K.kipp(c, [rel["R_F"]], [rel["rho"]], 1.25)
    print("tipping status", kp["status"], "ra", kp["ra"], "bound", kp["bound"])
    # Shift evaluator
    SE = ShiftEval(z, y, w, c["b"], c)
    f = SE.evaluator("F", E)
    print("Shift F: logor at 0, 0.5, 1.0:", [round(f(b)[0], 3) for b in (0.0, 0.5, 1.0)], "gap A/F (control SD):", SE.dA, SE.dF)
    print("tip_shift F:", K.tip_shift(f, K.B_COARSE, 1.25))
    # persistent paths: pi=0 equals the anchor
    pp = K.persist_paths(c, E, 0.0)
    assert abs(pp["AF"][0][0] - lf) < 1e-9
    print("tip persist AF:", K.tip_on_axis(K.PI_SCAN, *pp["AF"], 1.25))
    # kappa: kappa=1 equals the anchor when Ee=Ec=E
    kp3 = K.kappa_paths(c, E, E, 0.1)
    assert abs(kp3["AF"][0][0] - lf) < 1e-9
    print("tip kappa AF:", K.tip_on_axis(K.KAPPA, *kp3["AF"], 1.25))
    # Complete evaluation pass of one synthetic draw
    P = {"sid": np.arange(1500), "amp_ok": np.ones(1500, bool), "zg": np.zeros(1500, bool), "ef": rng.integers(0, 2, 1500).astype(float)}
    P["zg"][:60] = True
    P["sb"] = P["amp_ok"] & np.isfinite(P["ef"])
    P["delta"] = rng.multivariate_normal([0, 0], 2 * E, size=1500)
    fd = {"z": z, "y": y}
    t0 = time.time()
    ev = evaluate(P, fd, w, np.ones(1500), c["b"])
    print("evaluate duration (s):", round(time.time() - t0, 2), "anchors:", {a: (round(v["rel"][0], 3), round(np.exp(v["logor"]), 2) if np.isfinite(v["logor"]) else None) for a, v in ev["anchors"].items() if v})
    print("Tips (first 6):", ev["tips"][:6])
    print("TEST OK")


if __name__ == "__main__":
    PHASE = sys.argv[1]
    if PHASE == "test":
        synthetic_test()
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = PHASE + ("_" + sys.argv[2] if len(sys.argv) > 2 else "")
    guard()
    start_watch()
    try:
        C.dump(OUT / f"status_{tag}.json", {"phase": tag, "state": "running", "pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if PHASE == "stamp":
            stamp()
        elif PHASE == "point":
            point()
        elif PHASE == "boot":
            boot(sys.argv[2])
        else:
            raise SystemExit("unknown phase")
        C.dump(OUT / f"status_{tag}.json", {"phase": tag, "state": "complete", "pid": os.getpid(), "peak_rss": PEAK[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    except BaseException as ex:
        C.dump(OUT / f"status_{tag}.json", {"phase": tag, "state": "failed", "pid": os.getpid(), "reason": str(ex)[:300]})
        raise
