"""AW-241 B-R4: MNAR erweitert (delta-adjustierte multiple Imputation, nur Entwicklung).

Phasen: test | gate | analyze
Vertrag: nachrechnung/AW241/B_R4_PROTOKOLL.md (Hash wird vor der ersten echten Rechnung nach <local path> geschrieben).
Imputationen: die 100 Imputationen je Fold aus AW-211 (<local path><fold>.npz; klassenspezifisches multivariates Normalmodell,
EM auf Bootstrapstichprobe je Imputation), unveraendert uebernommen. Nur train/validation (Footer-Pruefung vor jedem Lesen).
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
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import expit  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

OUT = Path(str(data_path("b.aw241_r4")))
IMP = Path(str(data_path("b.aw211")))
PROTOKOLL = ROOT / "nachrechnung/AW241/B_R4_PROTOKOLL.md"
A, F = C.A, C.F
VARS = ["age", "sex_numeric", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", A, F]
MISSABLE = [2, 3, 4, 5, 6]
SDA, SDF = 46.84175, 0.76503231
FOLDS = ["train", "validation"]
M_IMP, MIN_IMP = 100, 95
GRID = [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 6, 8]
TABLE_D = [0, 1, 2, 3, 4, 6, 8]
DMAX = 8.0
PEAK = [0]
# Modelle (benannt). scheme: F = Fréchet-Imputationen, A = Winkel-Imputationen, FA = beide; s = Streuungsfaktor; cov = Kovariate der Verschiebung; cls = Ausfallklasse
MODELS = {
    "V0": dict(scheme="F", s=1.0, cov=None, cls="all", text="konstante Verschiebung der imputierten Fréchet-Werte (wie AW-211, Schema S1)"),
    "V1": dict(scheme="F", s=1.5, cov=None, cls="all", text="Verschiebung plus Streuungsfaktor 1,5 der imputierten Fréchet-Werte"),
    "V2": dict(scheme="F", s=2.0, cov=None, cls="all", text="Verschiebung plus Streuungsfaktor 2"),
    "V3": dict(scheme="F", s=1.0, cov="age", cls="all", text="Verschiebung proportional zum Alter"),
    "V4": dict(scheme="F", s=1.0, cov="qrs", cls="all", text="Verschiebung proportional zur QRS-Dauer"),
    "V5": dict(scheme="A", s=1.0, cov=None, cls="all", text="nur imputierte Winkelwerte verschoben (Fréchet unverändert)"),
    "V6": dict(scheme="FA", s=1.0, cov=None, cls="all", text="Winkel und Fréchet gemeinsam verschoben (wie AW-211, Schema S3)"),
    "V7": dict(scheme="F", s=1.0, cov=None, cls="both", text="Mustermischung: nur Ausfälle mit fehlendem Winkel und Fréchet (Schleifen-/Delineationsversagen) verschoben"),
    "V8": dict(scheme="F", s=1.0, cov=None, cls="fonly", text="Mustermischung: nur Ausfälle, bei denen allein der Fréchet-Wert fehlt, verschoben"),
}
PATHS = {"P1": lambda d: (-d, 0.0), "P2": lambda d: (0.0, d), "P3": lambda d: (-d, d)}
TARGETS = [(1.25, "OR 1,25"), (1.0, "OR 1,0")]


def start_watch(limit_gb=1.2):
    import psutil

    def watch():
        while True:
            PEAK[0] = max(PEAK[0], psutil.Process().memory_info().rss)
            if PEAK[0] > limit_gb * 1024 ** 3:
                os._exit(75)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def check_stamp():
    s = C.load(OUT / "protokoll_hash.json")
    assert s["sha256"] == C.sha(PROTOKOLL), "Protokoll nach Bindung veraendert"
    return s


def rubin(q, u):
    q = np.asarray(q)
    u = np.asarray(u)
    m = len(q)
    qb, ub, b = q.mean(), u.mean(), q.var(ddof=1)
    t = ub + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ub if ub > 0 else np.inf
    df = min((m - 1) * (1 + 1 / r) ** 2 if r > 0 else 1e6, 1e6)
    h = stats.t.ppf(.975, df) * np.sqrt(t)
    return qb, t, df, qb - h, qb + h


# ------------------------------------------------------------------ Modifikation (reine Funktionen, getestet)

def shifted_values(vals, y_cells, d_by_class, s, weights_by_cell):
    """vals: imputierte Werte (nur fehlende Zellen); y_cells: Klasse je Zelle; d_by_class: Verschiebung (rohe Einheiten) fuer (Kontrolle, Fall);
    s: Streuungsfaktor um den Klassenmittelwert der imputierten Werte; weights_by_cell: Verschiebungsgewicht je Zelle (Mittel 1 je Klasse).
    Ergebnis: mu + s (x - mu) + d * w   (je Klasse)."""
    out = vals.copy()
    for cls in (0, 1):
        sel = y_cells == cls
        if not sel.any():
            continue
        mu = vals[sel].mean()
        out[sel] = mu + s * (vals[sel] - mu) + d_by_class[cls] * weights_by_cell[sel]
    return out


def cov_weights(cov, y_cells, active):
    """Gewichte proportional zur Kovariate, je Klasse (unter den aktiven Zellen) auf Mittel 1 normiert; inaktive Zellen Gewicht 0."""
    w = np.zeros(len(cov))
    for cls in (0, 1):
        sel = (y_cells == cls) & active
        if sel.any():
            w[sel] = cov[sel] / cov[sel].mean()
    return w


class Fold4:
    def __init__(self, fold):
        self.fold = fold
        f, info, scale = C.outcome_frame()
        g = f.loc[f.eligible_reference.eq(True) & f.split.eq(fold)].copy()
        ok = np.isfinite(g.preferred_lvef.to_numpy(float)) & np.isfinite(g.age.to_numpy(float)) & np.isfinite(g.sex_numeric.to_numpy(float))
        g = g.loc[ok].reset_index(drop=True)
        self.y = classify_lvef(g.preferred_lvef.to_numpy(float))
        self.g = g
        self.scale = scale
        X = np.column_stack([g.age.to_numpy(float), g.sex_numeric.to_numpy(float)] + [g[v].to_numpy(float) for v in VARS[2:]])
        miss = ~np.isfinite(X)
        self.cells = [np.where(miss[:, j])[0] for j in range(X.shape[1])]
        z = np.load(IMP / f"t2_imp_{fold}.npz")
        self.ok = [mi for mi in range(M_IMP) if np.isfinite(z["v6"][mi]).all() and np.isfinite(z["v5"][mi]).all()]
        assert len(self.ok) >= MIN_IMP
        self.vals = {j: z[f"v{j}"] for j in MISSABLE}
        self.mean = np.array(scale["mean"])
        self.sd = np.array(scale["sd"])
        n = len(g)
        self.n = n
        self.w = np.ones(n)
        self.mF = np.zeros(n, bool)
        self.mF[self.cells[6]] = True
        self.mA = np.zeros(n, bool)
        self.mA[self.cells[5]] = True
        self.Z, self.F0, self.A0, self.Q0 = [], [], [], []
        for mi in self.ok:
            D = g[VARS].copy()
            for j in MISSABLE:
                D.iloc[self.cells[j], j] = self.vals[j][mi]
            raw, names = raw_design(D, F)
            self.Z.append(fixed_design(raw, scale))
            self.F0.append(D[F].to_numpy(float))
            self.A0.append(D[A].to_numpy(float))
            self.Q0.append(D["qrs_duration_ms"].to_numpy(float))
        self.age = g.age.to_numpy(float)
        self.b0 = [logistic_fit(zz, self.y, self.w) for zz in self.Z]
        # Ausfallklassen: 'both' = Winkel und Fréchet fehlen, 'fonly' = nur Fréchet fehlt
        self.cls_both = self.mF & self.mA
        self.cls_fonly = self.mF & ~self.mA
        self.cache = {}

    def active(self, cls, mask):
        return mask & {"all": np.ones(self.n, bool), "both": self.cls_both, "fonly": self.cls_fonly}[cls]

    def design(self, k, spec, dc, dk):
        z = self.Z[k].copy()
        if dc == 0 and dk == 0 and spec["s"] == 1.0:
            return z, 0.0
        floor_share = 0.0
        for col, raw0, mask, sdu, mu_col, sd_col in (
                ("F", self.F0[k], self.mF, SDF, self.mean[-1], self.sd[-1]),
                ("A", self.A0[k], self.mA, SDA, self.mean[-2], self.sd[-2])):
            if col not in spec["scheme"]:
                continue
            act = self.active(spec["cls"], mask)
            idx = np.where(act)[0]
            if len(idx) == 0:
                continue
            yc = self.y[idx].astype(int)
            if spec["cov"] == "age":
                wts = cov_weights(self.age[idx], yc, np.ones(len(idx), bool))
            elif spec["cov"] == "qrs":
                wts = cov_weights(self.Q0[k][idx], yc, np.ones(len(idx), bool))
            else:
                wts = np.ones(len(idx))
            d = (dk * sdu, dc * sdu)       # Index 0 Kontrolle, 1 Fall (roh)
            s_eff = spec["s"] if col == "F" else 1.0
            new = shifted_values(raw0[idx], yc, d, s_eff, wts)
            if col == "F":
                floor_share = float(np.mean(new <= 0.0))
                new = np.maximum(0.0, new)
            else:
                new = np.clip(new, 0.0, 180.0)
            full = raw0.copy()
            full[idx] = new
            z[:, -1 if col == "F" else -2] = (full - mu_col) / sd_col
        return z, floor_share

    def fit(self, k, spec, dc, dk):
        z, fl = self.design(k, spec, dc, dk)
        b = logistic_fit(z, self.y, self.w, initial=self.b0[k])
        p = expit(z @ b)
        H = z.T @ (z * (p * (1 - p))[:, None]) + np.diag(np.r_[0, np.full(z.shape[1] - 1, 1e-6)])
        return b[-1], np.linalg.inv(H)[-1, -1], fl

    def evaluate(self, name, dc, dk):
        key = (name, round(dc, 6), round(dk, 6))
        if key in self.cache:
            return self.cache[key]
        spec = MODELS[name]
        q, u, fl = [], [], []
        for k in range(len(self.Z)):
            a, b, c = self.fit(k, spec, dc, dk)
            q.append(a)
            u.append(b)
            fl.append(c)
        r = {"q": np.array(q), "u": np.array(u), "floor_share": float(np.mean(fl))}
        self.cache[key] = r
        return r


def summarize(r, idx=None):
    q = r["q"] if idx is None else r["q"][idx]
    u = r["u"] if idx is None else r["u"][idx]
    qb, t, df, lo, hi = rubin(q, u)
    return {"or": float(np.exp(qb)), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)), "df": float(df)}


def tip(fa, name, path, target, what, idx=None):
    """Kleinstes delta, bei dem OR (what='or'), untere (lo) bzw. obere (hi) Rubin-Grenze auf target faellt."""
    P = PATHS[path]

    def val(d):
        dc, dk = P(d)
        return np.log(summarize(fa.evaluate(name, dc, dk), idx)[what]) - np.log(target)

    v0 = val(0.0)
    if v0 <= 0:
        return {"status": "bei delta=0 schon unter Ziel", "d": 0.0}
    prev = 0.0
    for d in GRID[1:]:
        if val(d) <= 0:
            return {"status": "Kreuzung", "d": float(brentq(val, prev, d, xtol=1e-4))}
        prev = d
    return {"status": "nicht erreicht bis 8", "d": None, "or_at_8": float(np.exp(val(DMAX) + np.log(target)))}


# ------------------------------------------------------------------ Phasen

def test():
    rng = np.random.default_rng(1)
    vals = rng.normal(1.5, 0.5, 200)
    y = rng.integers(0, 2, 200)
    wts = np.ones(200)
    out = shifted_values(vals, y, (0.0, 0.0), 1.0, wts)
    assert np.allclose(out, vals)
    out = shifted_values(vals, y, (0.0, 0.0), 2.0, wts)
    for cls in (0, 1):
        s = y == cls
        assert abs(out[s].mean() - vals[s].mean()) < 1e-12 and abs(out[s].std() / vals[s].std() - 2) < 1e-12
    out = shifted_values(vals, y, (0.3, -0.2), 1.0, wts)
    assert abs((out - vals)[y == 0].mean() - 0.3) < 1e-12 and abs((out - vals)[y == 1].mean() + 0.2) < 1e-12
    age = rng.uniform(30, 90, 200)
    w = cov_weights(age, y, np.ones(200, bool))
    out = shifted_values(vals, y, (0.3, -0.2), 1.0, w)
    assert abs((out - vals)[y == 0].mean() - 0.3) < 1e-12 and abs((out - vals)[y == 1].mean() + 0.2) < 1e-12
    corr = np.corrcoef((out - vals)[y == 0], age[y == 0])[0, 1]
    assert corr > 0.99, corr
    q = rng.normal(0.9, 0.02, 100)
    u = np.full(100, 0.003)
    qb, t, df, lo, hi = rubin(q, u)
    assert lo < qb < hi
    print("TEST OK (Modifikationsfunktionen: Streuung, Verschiebung, Kovariatenproportionalitaet, Rubin)")


def gate():
    """Reproduktionsgate gegen AW-211 (Schema S1 und S3, Pfad P3, Ziel 1,25) und Complete-Case-OR."""
    start_watch()
    ref = C.load(IMP / "t2_results.json")
    res = {}
    for fold in FOLDS:
        fa = Fold4(fold)
        base = summarize(fa.evaluate("V0", 0.0, 0.0))
        t1 = tip(fa, "V0", "P3", 1.25, "or")
        t3 = tip(fa, "V6", "P3", 1.25, "or")
        old = {t["scheme"] + "|" + t["path"] + "|" + t["target"]: t for t in ref["folds"][fold]["tips"]}
        o1 = old["S1|P3|OR=1,25"]["d"]
        o3 = old["S3|P3|OR=1,25"]["d"]
        res[fold] = {"mi_delta0_or": base["or"], "ref_mi_delta0_or": ref["folds"][fold]["mi_delta0"]["or"], "S1_P3_125": t1["d"], "ref_S1_P3_125": o1, "S3_P3_125": t3["d"], "ref_S3_P3_125": o3,
                     "n_imputations": len(fa.ok)}
        res[fold]["ok"] = (abs(base["or"] - ref["folds"][fold]["mi_delta0"]["or"]) < 0.005 and abs(t1["d"] - o1) < 0.02 and abs(t3["d"] - o3) < 0.02)
        print("GATE", fold, json.dumps(C.clean(res[fold])), flush=True)
    C.dump(OUT / "gate.json", res)
    if not all(v["ok"] for v in res.values()):
        raise SystemExit("STOP: Gate nicht bestanden")


def analyze(fold_arg=None):
    check_stamp()
    start_watch()
    res = {"protokoll": check_stamp(), "script_sha256": C.sha(__file__), "models": {k: v["text"] for k, v in MODELS.items()}, "folds": {}}
    gate_res = C.load(OUT / "gate.json")
    assert all(v["ok"] for v in gate_res.values()), "Gate nicht bestanden"
    tag = fold_arg or "alle"
    for fold in ([fold_arg] if fold_arg else FOLDS):
        t0 = time.time()
        fa = Fold4(fold)
        d = {"n_valid_imputations": len(fa.ok), "n_rows": int(fa.n), "imputed_F_cells": {"cases": int(fa.mF[fa.y == 1].sum()), "controls": int(fa.mF[fa.y == 0].sum())},
             "imputed_A_cells": {"cases": int(fa.mA[fa.y == 1].sum()), "controls": int(fa.mA[fa.y == 0].sum())},
             "class_both": {"cases": int(fa.cls_both[fa.y == 1].sum()), "controls": int(fa.cls_both[fa.y == 0].sum())},
             "class_fonly": {"cases": int(fa.cls_fonly[fa.y == 1].sum()), "controls": int(fa.cls_fonly[fa.y == 0].sum())}}
        desc = C.load(IMP / "t2_describe.json")
        d["d_CC"] = desc["folds"][fold]["d_CC"]
        half = [np.arange(len(fa.ok))[h::2] for h in (0, 1)]
        d["models"] = {}
        for name in MODELS:
            m = {"or_delta0": summarize(fa.evaluate(name, 0.0, 0.0)), "or_table": {}, "tips": []}
            for path in PATHS:
                for dd in TABLE_D:
                    dc, dk = PATHS[path](dd)
                    r = fa.evaluate(name, dc, dk)
                    m["or_table"][f"{path}|{dd}"] = {**summarize(r), "floor_share": r["floor_share"]}
                for target, label in TARGETS:
                    row = {"path": path, "target": label}
                    row["point"] = tip(fa, name, path, target, "or")
                    row["delta_low_at_upper_limit"] = tip(fa, name, path, target, "hi")
                    row["delta_high_at_lower_limit"] = tip(fa, name, path, target, "lo")
                    row["half1"] = tip(fa, name, path, target, "or", idx=half[0])
                    row["half2"] = tip(fa, name, path, target, "or", idx=half[1])
                    m["tips"].append(row)
            d["models"][name] = m
            print("MODEL", fold, name, round(time.time() - t0), flush=True)
            res["folds"][fold] = d
            C.dump(OUT / f"r4_partial_{tag}.json", res)
    C.dump(OUT / f"r4_results_{tag}.json", res)
    print("ANALYZED", flush=True)


if __name__ == "__main__":
    PHASE = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if PHASE == "test":
        test()
        sys.exit(0)
    C.guard()
    try:
        C.dump(OUT / f"status_{PHASE}.json", {"phase": PHASE, "state": "running", "pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        (globals()[PHASE](*sys.argv[2:]))
        C.dump(OUT / f"status_{PHASE}.json", {"phase": PHASE, "state": "complete", "pid": os.getpid(), "peak_rss": PEAK[0], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    except BaseException as ex:
        C.dump(OUT / f"status_{PHASE}.json", {"phase": PHASE, "state": "failed", "pid": os.getpid(), "reason": str(ex)[:300]})
        raise
