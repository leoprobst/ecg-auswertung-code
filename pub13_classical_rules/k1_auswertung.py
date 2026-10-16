"""PUB-13 K1 (AW-236): Auswertungsskript nach SPEZIFIKATION_K1_ENTWURF_v3_2.md (SHA-256 195c965c...d683).

Eine Datei. Stufen (Spezifikation Abschnitt 11): prepare (Kohorte, Regeln, Referenzen, Kovariaten) -> run (Einheiten) ->
assemble (Urteile, Ausgabe). Betriebsarten: dev (train + validation), test (locked_test, nur mit Registerzeile 'Beginn'),
code15 (Reproduktion). Der Skriptstand wird vor jeder echten Rechnung gehasht und im Freeze eingetragen.

Aufruf (Beispiele, immer mit python -I):
  python -I k1_auswertung.py prepare --mode dev --spec <Spezifikation> --out <Ausgabeordner>
  python -I k1_auswertung.py run --mode dev --units R02,R03 --out <Ausgabeordner>
  python -I k1_auswertung.py assemble --mode dev --out <Ausgabeordner>
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repo_config import data_path

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # Rechenfenster: begrenzte Threadzahl je Prozess

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter_cache as AC  # noqa: E402
import k1_geruest as G  # noqa: E402
import regeln as R  # noqa: E402

SCRIPT_VERSION = "v3.2"
SPEC_SHA256 = "195c965c57f808a6b077f6563ad1164ea28a38d35f9256d40da195f00455d683"
BOUND_HASHES = {  # gebundene Hilfsdateien (Spezifikation Abschnitt 3)
    "regeln.py": "1e04fc99a5381512ade60654974b7a252548beee804b83225ef0ff50c9a58e11",
    "k1_geruest.py": "7736a0a7eece05f80f8f3c2e6239b63d4f5a1a244607c706d6828d321d44bd62",
    "adapter_cache.py": "f4f651467e742169b844bdc65456bd365c1a99557dbd430bbae6c6a9e32c075d",
}

# ------------------------------------------------------------------ Konfiguration (Pfade)
CFG = dict(
    root=str(data_path("pub13.mimic_validation")),
    counts_db=str(data_path("pub13.zaehlung_aw107_reproduktion_final_counts_duckdb")),
    cache=r"<local path>{s}/features.parquet",
    machine_csv=str(data_path("pub13.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_machine_measurements_csv")),
    hosp=str(data_path("pub13.mimic_iv_3_1_hosp")),
    codelist=str(data_path("pub13.publikationen_pub_13_kardiologie_codelisten_k1_codelisten_k1_csv")),
    register=str(data_path("pub13.koordination_testzugriffe_md")),
    code15=str(data_path("pub13.code15_exams_csv")),
    sealed_labels=str(data_path("pub13.mimic_validation_labels_sealed_locked_clinical_labels_wide_parquet")),
    ram_gb=1.0,
    threads=2,
)
DEV_SPLITS = ("train", "validation")

# ------------------------------------------------------------------ Konstanten der Spezifikation
B_BOOT = G.B_BOOT  # 4000
B_SENS = 1000  # Sensitivitaets- und Sekundaeranalysen (Abweichung A1 des Freeze-Anhangs)
ALPHA = G.ALPHA_FAMILY
KAPPA_GRID = (0.5, 1.0, 2.0)
HR_MARGIN = 1.20
R19_GRID = np.arange(90, 201, dtype=float)
THETA = {"R02": 0.90, "R03": 0.90, "R04": 0.90, "R06_spec": 0.98, "R06_sens": 0.10, "R07": 0.60, "R14": 0.80,
         "R20_A1": 0.01, "R20_A2": 1.0 / 3.0}
AGE_BANDS = (50.0, 70.0)  # T x Altersband: <50, 50-69, >=70
KNOT_PROBS = (0.05, 0.35, 0.65, 0.95)

STREAM_USES = {"primary": 0, "concord": 1, "sens_7d": 2, "sens_24h": 3, "sens_proj": 4, "secondary": 5, "stratum": 6,
               "r19_lbbb": 7, "aux": 8}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_stream(seed: int, rule_idx: int, use: str, mode: str) -> np.random.Generator:
    """Je Regel und Verwendung ein eigener Strom (Spezifikation 5.4); Testmodus mit eigenem Offset."""
    stream = STREAM_USES[use] + (1000 if mode == "test" else 0)
    return G.rng_for(seed, rule_idx, stream)


# ================================================================== Statistik: Splines, Logit, IPW
def rcs_knots(x: np.ndarray, probs=KNOT_PROBS) -> list[float]:
    x = np.asarray(x, dtype=float)
    return [float(v) for v in np.nanquantile(x, probs)]


def rcs_basis(x: np.ndarray, knots) -> np.ndarray:
    """Natuerlicher kubischer Spline nach Harrell: k Knoten -> k-1 Spalten (linear + k-2 nichtlineare)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    k = len(t)
    if k < 3 or np.any(np.diff(t) <= 0):
        raise ValueError(f"Spline-Knoten nicht streng aufsteigend: {list(t)}")  # v3: keine Division durch null
    cols = [x]
    tk, tk1 = t[-1], t[-2]
    denom = (tk - t[0]) ** 2

    def pos3(v):
        return np.maximum(v, 0.0) ** 3

    for j in range(k - 2):
        c = (pos3(x - t[j]) - pos3(x - tk1) * (tk - t[j]) / (tk - tk1) + pos3(x - tk) * (tk1 - t[j]) / (tk - tk1)) / denom
        cols.append(c)
    return np.column_stack(cols)


def expit(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit_newton(Z, y, w=None, beta0=None, pen=None, maxit=50, tol=1e-6):
    """Gewichtete logistische Regression (Newton/IRLS mit Schrittweitenhalbierung).
    pen: Vektor der Strafen (L2) je Spalte (Achsenabschnitt 0). Rueckgabe (beta, konvergiert)."""
    n, p = Z.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    pen = np.zeros(p) if pen is None else np.asarray(pen, dtype=float)
    beta = np.zeros(p) if beta0 is None else np.array(beta0, dtype=float)

    def nll(b):
        eta = Z @ b
        ll = np.sum(w * (y * eta - np.logaddexp(0.0, eta)))
        return -ll + 0.5 * np.sum(pen * b * b)

    cur = nll(beta)
    ok = False
    for _ in range(maxit):
        mu = expit(Z @ beta)
        W = w * mu * (1.0 - mu)
        g = Z.T @ (w * (y - mu)) - pen * beta
        if np.max(np.abs(g)) <= 1e-9 * (1.0 + float(np.sum(w))):  # Gradient praktisch null: konvergiert
            ok = True
            break
        H = (Z * W[:, None]).T @ Z + np.diag(pen)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(p), g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        for _h in range(8):
            new = nll(beta + t * step)
            if np.isfinite(new) and new <= cur + 1e-8 * (1.0 + abs(cur)):
                break
            t *= 0.5
        beta = beta + t * step
        cur = new if np.isfinite(new) else cur
        if np.max(np.abs(t * step)) < tol:
            ok = True
            break
    return beta, ok


def fit_logit_safe(Z, y, w=None, beta0=None, ridge_lambda=1.0):
    """Unbestraft; bei Separation oder Nichtkonvergenz Ridge mit festem lambda = 1 auf standardisierten
    Kovariaten, Achsenabschnitt unbestraft (Spezifikation 5.1, Punkt 5). Rueckgabe (beta, info)."""
    beta, ok = logit_newton(Z, y, w, beta0)
    mu = expit(Z @ beta)
    sep = (not ok) or (np.max(np.abs(beta)) > 15.0) or bool(np.any((mu < 1e-9) | (mu > 1 - 1e-9)))
    if not sep:
        return beta, {"ridge": False, "failed": False}
    # Ridge: Spalten ausser dem Achsenabschnitt (Spalte 0) auf sd 1 bringen, Strafe lambda
    p = Z.shape[1]
    sd = Z.std(axis=0)
    sd[0] = 1.0
    sd[sd < 1e-12] = 1.0
    Zs = Z / sd
    pen = np.full(p, float(ridge_lambda))
    pen[0] = 0.0
    b2, ok2 = logit_newton(Zs, y, w, None if beta0 is None else np.asarray(beta0) * sd, pen)
    b2 = b2 / sd
    failed = (not ok2) or (not np.all(np.isfinite(b2)))  # v3: nicht konvergierte Anpassungen werden zurueckgewiesen
    return b2, {"ridge": True, "converged": bool(ok2), "failed": bool(failed)}


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(x, kind="mergesort")
    xs, ws = x[o], w[o]
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return float("nan")
    return float(xs[np.searchsorted(cw, q * cw[-1], side="left").clip(0, len(xs) - 1)])


def stabilised_weights(p, V, c, trunc=(0.01, 0.99)):
    """w = P(V=1)/p fuer V=1 (stabilisiert), Kappung am 1. und 99. Perzentil unter den Verifizierten; sonst 0."""
    c = np.asarray(c, dtype=float)
    V = np.asarray(V).astype(bool)
    prev = float(np.sum(c * V) / np.sum(c))
    w = np.zeros(len(p))
    w[V] = prev / np.clip(p[V], 1e-6, 1.0)
    sel = V & (c > 0)
    if sel.sum() > 5:
        lo = weighted_quantile(w[sel], c[sel], trunc[0])
        hi = weighted_quantile(w[sel], c[sel], trunc[1])
        w[V] = np.clip(w[V], lo, hi)
    return w


def ess_fraction(w, c, V):
    sel = np.asarray(V).astype(bool) & (c > 0)
    ww = (w * c)[sel]
    if ww.size == 0:
        return float("nan"), 0.0
    ess = float(ww.sum() ** 2 / np.sum((w[sel] ** 2) * c[sel]))
    return ess, ess / float(c[sel].sum())


def two_by_two_ipw(T, D, V, wt):
    """Gewichtete Kennzahlen. T, D in {0,1}; V in {0,1}; wt = Zeilengewicht (0 fuer nicht Verifizierte).
    Rueckgabe sens, spec, ppv, npv, prev, lrp, lrm."""
    T = np.asarray(T, dtype=float)
    Dd = np.nan_to_num(np.asarray(D, dtype=float), nan=0.0)
    m = (np.asarray(V) > 0) * wt
    pos = np.sum(m * Dd)
    neg = np.sum(m * (1 - Dd))
    tot = pos + neg
    tp = np.sum(m * Dd * T)
    tn = np.sum(m * (1 - Dd) * (1 - T))
    sens = tp / pos if pos > 0 else float("nan")
    spec = tn / neg if neg > 0 else float("nan")
    prev = pos / tot if tot > 0 else float("nan")
    fp = 1 - spec
    ppv = sens * prev / (sens * prev + fp * (1 - prev)) if (sens * prev + fp * (1 - prev)) > 0 else float("nan")
    npv = spec * (1 - prev) / (spec * (1 - prev) + (1 - sens) * prev) if (spec * (1 - prev) + (1 - sens) * prev) > 0 else float("nan")
    if np.isnan(sens) or np.isnan(spec):
        lrp = float("nan")  # v3.1: ohne Referenznegative oder -positive ist LR+ undefiniert, nicht +inf
    elif (1 - spec) > 0:
        lrp = sens / (1 - spec)
    else:
        lrp = float("inf") if (sens > 0) else float("nan")  # +inf zulaessig, 0/0 undefiniert
    lrm = (1 - sens) / spec if spec > 0 else float("nan")
    return dict(sens=float(sens), spec=float(spec), ppv=float(ppv), npv=float(npv), prev=float(prev), lrp=float(lrp), lrm=float(lrm))


def completed_metrics(T, D, V, pu, c, kplus, kminus):
    """Kipppunkt (5.2 S2): Odds der Referenz bei Unverifizierten = kappa(T) * Odds kovariablengleicher Verifizierter.
    Verifizierte tragen die beobachtete Referenz, Unverifizierte den (verschobenen) Modellwert; Gewichte = Zaehlungen c."""
    T = np.asarray(T, dtype=float)
    odds = pu / np.clip(1.0 - pu, 1e-9, None)
    k = np.where(T == 1, kplus, kminus)
    pp = odds * k / (1.0 + odds * k)
    Dd = np.where(np.asarray(V) > 0, np.nan_to_num(np.asarray(D, dtype=float), nan=0.0), pp)
    pos = np.sum(c * Dd)
    neg = np.sum(c * (1 - Dd))
    sens = np.sum(c * Dd * T) / pos if pos > 0 else float("nan")
    spec = np.sum(c * (1 - Dd) * (1 - T)) / neg if neg > 0 else float("nan")
    return float(sens), float(spec)


def manski_bounds(T, D, V, c):
    """Worst-Case-Schranken (S1): Unverifizierte alle positiv bzw. alle negativ, zugunsten bzw. zulasten der Regel."""
    T = np.asarray(T, dtype=float)
    Dd = np.nan_to_num(np.asarray(D, dtype=float), nan=0.0)
    ver = np.asarray(V) > 0
    a = np.sum(c * ver * Dd * T)
    b = np.sum(c * ver * Dd * (1 - T))
    d = np.sum(c * ver * (1 - Dd) * (1 - T))
    f = np.sum(c * ver * (1 - Dd) * T)
    u1 = np.sum(c * (~ver) * T)
    u0 = np.sum(c * (~ver) * (1 - T))
    sens_lo = a / (a + b + u0) if (a + b + u0) > 0 else float("nan")
    sens_hi = (a + u1) / (a + u1 + b) if (a + u1 + b) > 0 else float("nan")
    spec_lo = d / (d + f + u1) if (d + f + u1) > 0 else float("nan")
    spec_hi = (d + u0) / (d + u0 + f) if (d + u0 + f) > 0 else float("nan")
    return dict(sens_lo=float(sens_lo), sens_hi=float(sens_hi), spec_lo=float(spec_lo), spec_hi=float(spec_hi))


def classify_gt(lower, upper, theta):
    """Behauptung 'Wert > theta' strikt (R04 Teil 2): haelt bei lower > theta, haelt nicht bei upper <= theta."""
    if lower > theta:
        return G.HAELT
    if upper <= theta:
        return G.HAELT_NICHT
    return G.UNENTSCHIEDEN


def classify_marker3(lower, upper, null=1.0, small=1.25):
    """R08 v3 (Nachtrag 1): Behauptung RR > 1. haelt: lower > 1; haelt nicht erst bei upper <= 1; sonst unentschieden.
    Hinweis 'klein erhoeht', wenn lower > 1 und upper < 1,25."""
    if lower > null:
        return G.HAELT, ("klein erhoeht (obere Schranke < 1,25)" if upper < small else "")
    if upper <= null:
        return G.HAELT_NICHT, ""
    return G.UNENTSCHIEDEN, ""


def design(Xbase, T, female, band, with_interactions=True):
    """Z = [1, Xbase, T, T*weiblich, T*Band2, T*Band3] (Spezifikation 5.1, Punkt 2)."""
    n = Xbase.shape[0]
    cols = [np.ones((n, 1)), Xbase, np.asarray(T, dtype=float)[:, None]]
    if with_interactions:
        Tt = np.asarray(T, dtype=float)
        cols += [(Tt * female)[:, None], (Tt * (band == 1))[:, None], (Tt * (band == 2))[:, None]]
    return np.hstack(cols)


class IPWUnit:
    """Eine Einheit mit Verifikationsmodell: mehrere Regelvarianten T (Dict), gemeinsame Referenz D, Verifikation V.
    stats(c): Kennzahlen unter IPW (Modell wird mit Zaehlungen c neu angepasst), Kipppunkt-Raster und Worst-Case-Schranken.
    weights_from: Zuordnung Variante -> Quellvariante; die Variante verwendet die Gewichte der Quelle (gleiche Gewichte und
    gleiche Population wie in der Spezifikation 5, Konkordanzregel). Nicht konvergierte Anpassungen liefern None."""

    def __init__(self, Tdict, D, V, Xbase, female, band, do_kipp=True, weighted=True, name="", weights_from=None):
        self.name = name
        self.T = {k: np.asarray(v, dtype=float) for k, v in Tdict.items()}
        self.D = np.asarray(D, dtype=float)
        self.V = np.asarray(V).astype(int)
        self.n = len(self.V)
        self.Xbase, self.female, self.band = Xbase, female, band
        self.weights_from = dict(weights_from or {})
        self.sources = [k for k in self.T if k not in self.weights_from]
        self.Z = {k: design(Xbase, self.T[k], female, band) for k in self.sources}
        self.weighted = weighted
        self.do_kipp = do_kipp and weighted
        self.beta_v = {k: None for k in self.sources}
        self.beta_o = {k: None for k in self.sources}
        self.fit_info = {}
        self.failed_fits = 0

    def stats(self, c, warm=True):
        out = {}
        c = np.asarray(c, dtype=float)
        sel = c > 0
        wts = {}
        for k in self.sources:
            Z = self.Z[k]
            if self.weighted:
                bv, info = fit_logit_safe(Z[sel], self.V[sel], c[sel], self.beta_v[k] if warm else None)
                if info.get("failed"):
                    self.failed_fits += 1
                    return None
                if warm and self.beta_v[k] is None:
                    self.beta_v[k] = bv
                self.fit_info.setdefault(k, info)
                p = expit(Z @ bv)
                w = stabilised_weights(p, self.V, c)
                wts[k] = (w, *ess_fraction(w, c, self.V))
            else:
                wts[k] = (c.astype(float), float(np.sum(c * self.V)), 1.0)
        for k, T in self.T.items():
            src = self.weights_from.get(k, k)
            w, ess, essf = wts[src]
            wt = w * c if self.weighted else w
            m = two_by_two_ipw(T, self.D, self.V, wt)
            m["ess"], m["ess_frac"] = ess, essf
            vv = (self.V > 0)
            den = float(np.sum(wt * vv))
            tpos = float(np.sum(wt * vv * T) / den) if den > 0 else float("nan")  # v3.1: gewichtet wie die Praevalenz (Nachtrag 6.3)
            m["p_t"] = tpos
            m["p_t_beobachtet"] = float(np.sum(c * T) / np.sum(c))
            m["trivial_bound"] = G.trivial_specificity_bound(tpos, 1.0 - m["prev"]) if np.isfinite(m["prev"]) else float("nan")
            m["manski"] = manski_bounds(T, self.D, self.V, c)
            if self.do_kipp and k in self.sources:
                Z = self.Z[k]
                v1 = sel & (self.V == 1)
                bo, info_o = fit_logit_safe(Z[v1], self.D[v1], c[v1], self.beta_o[k] if warm else None)
                if info_o.get("failed"):
                    self.failed_fits += 1
                    return None
                if warm and self.beta_o[k] is None:
                    self.beta_o[k] = bo
                pu = expit(Z @ bo)
                grid = {}
                for kp in KAPPA_GRID:
                    for km in KAPPA_GRID:
                        grid[f"{kp}|{km}"] = completed_metrics(T, self.D, self.V, pu, c, kp, km)
                m["kipp"] = grid
            out[k] = m
        return out


def _auc_rank(score, label):
    from scipy.stats import rankdata
    label = np.asarray(label).astype(bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def weight_diagnostics(unit, key, cols=None):
    """Diagnostik des Verifikationsmodells (5.1 Punkt 5): Gewichtsverteilung, effektive Stichprobengroesse, standardisierte
    Mittelwertdifferenzen nach Gewichtung (Ziel < 0,1), c-Statistik, Ueberlappung der Propensity-Verteilungen."""
    if not unit.weighted:
        return {"gewichtet": False}
    c = np.ones(unit.n)
    Z = unit.Z[key]
    bv, info = fit_logit_safe(Z, unit.V, c)
    p = expit(Z @ bv)
    w = stabilised_weights(p, unit.V, c)
    v = unit.V.astype(bool)
    Xb = unit.Xbase
    sd = Xb.std(axis=0)
    sd[sd < 1e-9] = 1.0
    mw = (Xb[v] * w[v][:, None]).sum(axis=0) / w[v].sum()
    smd = np.abs(mw - Xb.mean(axis=0)) / sd
    smd_raw = np.abs(Xb[v].mean(axis=0) - Xb.mean(axis=0)) / sd
    ess, essf = ess_fraction(w, c, unit.V)
    wv = w[v]
    p_small = float(np.mean(p < 0.01))
    return dict(gewichtet=True, anteil_p_unter_0_01=p_small, identifizierbarkeit_begrenzt=bool(p_small > 0.10), ridge=bool(info.get("ridge")), n_verified=int(v.sum()), ess=float(ess), ess_frac=float(essf),
                gewicht_quantile={q: float(np.quantile(wv, q)) for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
                smd_max_nach_gewichtung=float(smd.max()), smd_mittel_nach_gewichtung=float(smd.mean()),
                smd_max_ohne_gewichtung=float(smd_raw.max()), n_smd_ueber_0_1=int((smd > 0.1).sum()), n_spalten=int(len(smd)),
                c_statistik=_auc_rank(p, v),
                propensity_verifizierte_p01_p99=[float(np.quantile(p[v], 0.01)), float(np.quantile(p[v], 0.99))],
                propensity_unverifizierte_p01_p99=[float(np.quantile(p[~v], 0.01)), float(np.quantile(p[~v], 0.99))] if (~v).any() else None)


def bootstrap_counts(rng, n):
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def run_bootstrap(unit: IPWUnit, rng, B, extra=None, every=500, tag=""):
    """Personengeclusterter Bootstrap (eine Person, ein Index-EKG): Zaehlungen c aus Ziehen mit Zuruecklegen;
    das Gewichts- und das Ergebnismodell werden je Wiederholung neu angepasst. Rueckgabe: Punktschaetzer und
    Wiederholungen je Variante und Kennzahl (Arrays). v3: fehlgeschlagene Wiederholungen (nicht konvergierte Anpassung) werden
    gezaehlt und unter reps['_meta'] berichtet, nicht still verworfen; schlaegt die Punktanpassung fehl, gibt es keine Inferenz."""
    point = unit.stats(np.ones(unit.n))
    if point is None:
        return None, {"_meta": {"B": B, "failed": B, "punktschaetzung_fehlgeschlagen": True}}
    reps = {k: {} for k in point}
    for k, m in point.items():  # v3.1: Schluessel immer anlegen (auch bei vollstaendigem Ausfall der Wiederholungen)
        for mk, mv in m.items():
            if mk == "manski":
                for kk in mv:
                    reps[k]["manski_" + kk] = []
            elif mk == "kipp":
                for gk in mv:
                    reps[k][f"kipp_sens_{gk}"] = []
                    reps[k][f"kipp_spec_{gk}"] = []
            else:
                reps[k][mk] = []
    t0 = time.time()
    failed = 0
    for b in range(B):
        c = bootstrap_counts(rng, unit.n)
        st = unit.stats(c)
        if st is None:
            failed += 1
            continue
        for k, m in st.items():
            for mk, mv in m.items():
                if mk == "manski":
                    for kk, vv in mv.items():
                        reps[k].setdefault("manski_" + kk, []).append(vv)
                elif mk == "kipp":
                    for gk, (s_, sp) in mv.items():
                        reps[k].setdefault(f"kipp_sens_{gk}", []).append(s_)
                        reps[k].setdefault(f"kipp_spec_{gk}", []).append(sp)
                else:
                    reps[k].setdefault(mk, []).append(mv)
        if extra is not None:
            extra(b, c, st, reps)
        if (b + 1) % every == 0:
            log(f"  {tag} Bootstrap {b + 1}/{B} ({time.time() - t0:.0f} s)")
    for k in list(reps):
        for mk in reps[k]:
            reps[k][mk] = np.asarray(reps[k][mk], dtype=float)
    undefined = {}
    for k in reps:
        if k == "_meta":
            continue
        for mk, arr in reps[k].items():
            nn = int(np.isnan(arr).sum()) if arr.size else 0  # v3.2: alle Kennzahlen, auch Kipppunkt-, Worst-Case- und Positivratenwerte
            if nn:
                undefined[f"{k}.{mk}"] = nn
    reps["_meta"] = {"B": B, "failed": failed, "punktschaetzung_fehlgeschlagen": False, "undefiniert_je_kennzahl": undefined}
    return point, reps


MIN_VALID_FRACTION = 0.95  # Nachtrag 1: weniger gueltige Wiederholungen -> keine Inferenz
MIN_TAIL_REPS = 10  # Nachtrag 1: mindestens 10 Wiederholungen im Rand der einseitigen Schranke (Monte-Carlo-Praezision)


def qbounds(a, alpha_c):
    """Untere und obere einseitige Perzentilschranke; +inf ist zulaessig (dann ohne Interpolation)."""
    if np.isinf(a).any():
        return float(np.quantile(a, alpha_c, method="lower")), float(np.quantile(a, 1.0 - alpha_c, method="higher"))
    return float(np.quantile(a, alpha_c)), float(np.quantile(a, 1.0 - alpha_c))


def interval(arr, alpha_c, n_expected=None):
    """Einseitige Schranken (Bonferroni-Niveau alpha_c) und unadjustiertes 95 %-Intervall (zweiseitig).
    v3 (Nachtrag 1): +inf bleibt erhalten, NaN (undefinierte Wiederholung) wird gezaehlt; sind weniger als 95 % der erwarteten
    Wiederholungen gueltig oder weniger als 20 vorhanden, gibt es keine Schranken. Monte-Carlo-Praezision: Zahl der Wiederholungen
    im Rand (alpha_c * n) und Schranken aus beiden Haelften der Wiederholungen."""
    a0 = np.asarray(arr, dtype=float)
    a = a0[~np.isnan(a0)]
    n_exp = len(a0) if n_expected is None else int(n_expected)
    nan = float("nan")
    base = dict(nboot=int(a.size), n_erwartet=n_exp, n_undefiniert=int(n_exp - a.size))
    mc_rand = int(alpha_c * a.size)
    if a.size < 20 or a.size < MIN_VALID_FRACTION * n_exp:
        return dict(lo=nan, hi=nan, lo95=nan, hi95=nan, mc_rand=mc_rand, lo_h1=nan, hi_h1=nan, lo_h2=nan, hi_h2=nan,
                    keine_inferenz=True, **base)
    l95, h95 = qbounds(a, 0.025)  # Anzeige (unadjustiert, kein Urteil)
    if mc_rand < MIN_TAIL_REPS:  # Schranken fuer das Urteil nur bei ausreichender Monte-Carlo-Praezision
        return dict(lo=nan, hi=nan, lo95=l95, hi95=h95, mc_rand=mc_rand, lo_h1=nan, hi_h1=nan, lo_h2=nan, hi_h2=nan,
                    keine_inferenz=True, **base)
    lo, hi = qbounds(a, alpha_c)
    half = a.size // 2
    lo1, hi1 = qbounds(a[:half], alpha_c) if int(alpha_c * half) >= 5 else (nan, nan)
    lo2, hi2 = qbounds(a[half:], alpha_c) if int(alpha_c * (a.size - half)) >= 5 else (nan, nan)
    return dict(lo=lo, hi=hi, lo95=l95, hi95=h95, mc_rand=mc_rand, lo_h1=lo1, hi_h1=hi1, lo_h2=lo2, hi_h2=hi2,
                keine_inferenz=False, **base)


# ================================================================== Datenaufbereitung (prepare)
import duckdb  # noqa: E402

AMP_LEADS = AC.LEADS
CACHE_COLS = AC.cache_columns() + ["signal_quality_usable", "classical_pathological_q_waves_positive",
                                   "classical_rbbb_pattern_positive", "classical_qrs_widening_ge_120",
                                   "pr_interval_ms", "context_machine_qrs_duration_ms",
                                   "context_machine_pr_interval_ms", "context_machine_qt_interval_ms"]

RX_LAFB = re.compile(r"left (anterior|posterior) fascicular block|\bLAFB\b|\bLPFB\b|left (anterior|posterior) hemiblock", re.I)
RX_AVB = re.compile(r"\ba-?v block|atrioventricular block|a-?v dissociation|mobitz|wenckebach|complete heart block", re.I)
RX_IVCD = re.compile(r"intraventricular conduction|\bIVCD\b", re.I)
RX_WPW = re.compile(r"wolff|\bWPW\b|pre-?excitation", re.I)
RX_STT = re.compile(r"\bst\b|\bst[- ]t\b|t wave|ischemi|infarct|repolari", re.I)
RX_NORMAL_ANY = re.compile(r"normal ecg", re.I)
RX_ABN = re.compile(r"abnormal ecg|borderline ecg", re.I)
RX_BORDER = re.compile(r"borderline", re.I)
RX_CHESTPAIN = "chest pain"


def gmode(mode: str) -> str:
    return "test" if mode == "test" else "dev"


def splits_for(mode: str):
    return ("locked_test",) if mode == "test" else DEV_SPLITS


def connect(cfg):
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg['ram_gb']}GB'; SET threads={cfg['threads']}")
    return con


def fetch_by_keys(con, rel_sql: str, keys: pd.DataFrame, key_cols, cols) -> pd.DataFrame:
    """Liest nur die Zeilen, deren Schluessel in keys stehen (kein Zugriff auf andere Partitionen)."""
    con.register("k_keys", keys[list(key_cols)].drop_duplicates())
    on = " AND ".join(f"t.{c} = k.{c}" for c in key_cols)
    sel = ", ".join(f"t.{c}" for c in cols)
    df = con.execute(f"SELECT {sel} FROM {rel_sql} t JOIN k_keys k ON {on}").df()
    con.unregister("k_keys")
    return df


def tri_state(s: pd.Series, pos="positive", neg="not_mentioned") -> np.ndarray:
    """positive -> 1, not_mentioned -> 0, sonst (uncertain, fehlend) -> NaN."""
    out = np.full(len(s), np.nan)
    v = s.astype(object).values
    out[v == pos] = 1.0
    out[v == neg] = 0.0
    return out


def load_base(con, cfg, mode, ga) -> pd.DataFrame:
    """Alle erwachsenen EKG der Partition(en) mit Kontext und Maschinenaussagen (ohne Befundtext)."""
    root = cfg["root"]
    ccf = f"read_parquet('{root}/evidence/clinical_context_features.parquet')"
    mrl = f"read_parquet('{root}/evidence/machine_report_labels.parquet')"
    frames = []
    wl = ""
    if mode in ("dev", "dryrun"):
        con.execute(f"ATTACH '{cfg['counts_db']}' AS cn (READ_ONLY)")
        wl = "JOIN cn.ecg w ON w.subject_id = c.subject_id AND w.study_id = c.study_id AND w.fold = c.split"
    for s in splits_for(mode):
        a = part_sql(ccf, s, gmode(mode), **ga)
        b = part_sql(mrl, s, gmode(mode), **ga)
        sql = f"""
        SELECT c.subject_id, c.study_id, c.ecg_time, c.hadm_id, c.ed_stay_id, c.ed_intime, c.icustay_id, c.icu_intime,
               c.icu_outtime, c.sex, c.age, c.chief_complaint, c.machine_heart_rate_bpm AS hr_bpm,
               c.machine_pr_interval_ms AS pr_ms, c.machine_qrs_duration_ms AS qrs_ms, c.machine_qt_interval_ms AS qt_ms,
               c.split, m.machine_metadata_usable AS usable, m.report_af_state, m.report_lbbb_state, m.report_rbbb_state,
               m.report_infarct_state, m.report_lvh_state, m.report_pacemaker_state, m.report_low_voltage_state
        FROM ({a}) c JOIN ({b}) m ON m.subject_id = c.subject_id AND m.study_id = c.study_id {wl}
        WHERE c.age >= 18
        """
        frames.append(con.execute(sql).df())
    df = pd.concat(frames, ignore_index=True)
    df["usable"] = df["usable"].fillna(False).astype(bool)
    return df


def choose_index(base: pd.DataFrame) -> pd.DataFrame:
    """Index-EKG: erstes geeignetes erwachsenes EKG je Person nach EKG-Zeit und study_id (AE-306 D1), ohne Kenntnis
    von Referenz oder Regelergebnis."""
    d = base[base["usable"]].sort_values(["subject_id", "ecg_time", "study_id"], kind="mergesort")
    return d.drop_duplicates("subject_id", keep="first").reset_index(drop=True)


def choose_index_ed(base: pd.DataFrame, trop: pd.DataFrame) -> pd.DataFrame:
    """R20 (v3): erster ED-Aufenthalt je Person mit Brustschmerz-Leitsymptom (Triage) und Troponinmessung; die Troponinmessung
    wird auf Aufenthaltsebene festgestellt (mindestens ein EKG des Aufenthalts mit Troponinzaehlung > 0). Index ist das erste
    verwertbare EKG dieses Aufenthalts am Kalendertag des Triagezeitpunkts (Geraeteuhr), unabhaengig von dessen eigener
    Troponinzaehlung (Sol-Verdacht 4)."""
    d = base[base["usable"] & base["ed_stay_id"].notna() & base["chief_complaint"].fillna("").str.lower().str.contains(RX_CHESTPAIN, regex=False)].copy()
    d = d.merge(trop, on="study_id", how="left")
    d["trop_count"] = d["trop_count"].fillna(0)
    stay_ok = d.groupby("ed_stay_id")["trop_count"].max() > 0
    d = d[d["ed_stay_id"].map(stay_ok).fillna(False).astype(bool)]
    d = d[d["ecg_time"].dt.normalize() == d["ed_intime"].dt.normalize()]
    d = d.sort_values(["subject_id", "ed_intime", "ecg_time", "study_id"], kind="mergesort")
    return d.drop_duplicates("subject_id", keep="first").reset_index(drop=True)


def load_text_and_rr(con, cfg, idx: pd.DataFrame) -> pd.DataFrame:
    cols = ", ".join(f"nullif(trim(report_{i}), '')" for i in range(18))
    con.register("k_idx", idx[["study_id"]].drop_duplicates())
    q = f"""SELECT m.study_id, m.rr_interval AS rr_ms, concat_ws(' | ', {cols}) AS report_text
            FROM read_csv('{cfg['machine_csv']}', header=true) m JOIN k_idx k ON k.study_id = m.study_id"""
    out = con.execute(q).df()
    con.unregister("k_idx")
    return out.drop_duplicates("study_id")


def load_cache(con, cfg, idx: pd.DataFrame, mode, ga) -> pd.DataFrame:
    """A155-Merkmalscache (nur train/validation; fuer locked_test nicht vorhanden): Spannungsspalten und Flags."""
    if mode == "test":
        return pd.DataFrame({"study_id": idx["study_id"].values})
    frames = []
    for s in splits_for(mode):
        guard_phase(gmode(mode), s, **ga)
        path = cfg["cache"].format(s=s)
        sub = idx[idx["split"] == s]
        if len(sub) == 0:
            continue
        frames.append(fetch_by_keys(con, f"read_parquet('{path}')", sub, ["study_id"], ["study_id"] + CACHE_COLS))
    return pd.concat(frames, ignore_index=True).drop_duplicates("study_id")


def _iter_rows(idx: pd.DataFrame, chunk: int = 10000):
    """Zeilenweise als Dict, blockweise erzeugt (Speicher)."""
    for start in range(0, len(idx), chunk):
        for j, row in enumerate(idx.iloc[start:start + chunk].to_dict("records")):
            yield start + j, row


def evaluate_rules(idx: pd.DataFrame, has_cache: bool) -> pd.DataFrame:
    """Regelzustaende 0/1/NaN je Regel und Variante (regeln.py, adapter_cache.py); Maschinenaussagen aus Text und Zustand.
    NaN = nicht auswertbar (Enthaltung), nie stillschweigend negativ."""
    n = len(idx)
    names = ["R02_gt", "R02_ge", "R03_gt", "R03_ge", "R03_prod", "R04_ge", "R04_gt", "R06_a", "R06_c", "R07_a", "R07_b",
             "R14_a155", "R14_o2", "R08_baz", "R08_fri", "R08_lin", "R09_a155", "R18_m", "R18_own",
             "R06_b", "R07_c", "R03_fixM", "R03_fixF", "R04_fixM", "R04_fixF",
             "lvh_volt", "qrs_machine", "qrs_own", "pr_machine", "qtc_baz", "qtc_cat"]
    out = {k: np.full(n, np.nan) for k in names}
    sex_arr = idx["sex"].astype(object).values
    for i, row in _iter_rows(idx):
        sex = sex_arr[i] if sex_arr[i] in ("M", "F") else None
        paced = row.get("report_pacemaker_state") == "positive"
        lbbb = row.get("report_lbbb_state") != "not_mentioned"
        rbbb_ex = row.get("report_rbbb_state") != "not_mentioned"
        af = row.get("report_af_state") != "not_mentioned"
        sinus = row.get("sinus")
        sinus = None if sinus is None or (isinstance(sinus, float) and math.isnan(sinus)) else bool(sinus)
        rr = row.get("rr_ms")
        qrs_m, pr_m, qt_m = row.get("qrs_ms"), row.get("pr_ms"), row.get("qt_ms")
        # --- Rhythmus- und Intervallregeln (Maschinenwerte), unabhaengig vom Cache
        m_int = AC.m_from_cache({}, sex=sex, qrs_ms=qrs_m, pr_ms=pr_m, qt_ms=qt_m, rr_ms=rr, sinus=sinus, af=af, paced=paced, lbbb=lbbb)
        out["qrs_machine"][i] = m_int["qrs_ms"] if m_int["qrs_ms"] is not None else np.nan
        out["pr_machine"][i] = m_int["pr_ms"] if m_int["pr_ms"] is not None else np.nan
        qb = R.qtc_ms(m_int["qt_ms"], m_int["rr_ms"], "bazett")
        if qb is not None and not af:
            out["qtc_baz"][i] = qb
            qc = R.qtc_category(qb, sex)
            out["qtc_cat"][i] = np.nan if qc is None else float(qc)

        def put(key, val):
            out[key][i] = np.nan if val is None else float(bool(val))

        put("R08_baz", R.qtc_gt(m_int, 500.0, "bazett"))
        put("R08_fri", R.qtc_gt(m_int, 500.0, "fridericia"))
        put("R08_lin", R.qtc_gt(m_int, 500.0, "framingham"))
        put("R18_m", R.pr_gt_200(m_int))
        if has_cache:
            q_own = row.get("classical_qrs_widening_ge_120")
            out["qrs_own"][i] = np.nan if q_own is None or not np.isfinite(float(q_own)) else float(q_own)
            m_own = AC.m_from_cache(row, sex=sex, qrs_ms=qrs_m, pr_ms=row.get("pr_interval_ms"), qt_ms=qt_m, rr_ms=rr,
                                    sinus=sinus, af=af, paced=paced, lbbb=lbbb)
            put("R18_own", R.pr_gt_200(m_own))
            sq = row.get("signal_quality_usable")
            sq_ok = bool(sq) if sq is not None and not (isinstance(sq, float) and math.isnan(sq)) else False
            m = m_own
            m["qrs_ms"] = float(qrs_m) if qrs_m is not None and np.isfinite(qrs_m) else None
            lvh = R.lvh_voltage_flag(m)
            out["lvh_volt"][i] = np.nan if lvh is None else float(lvh)
            if sq_ok:
                volt_ok = (not paced) and (not lbbb)
                if volt_ok:
                    put("R02_gt", R.sokolow_lyon(m, strict=True))
                    put("R02_ge", R.sokolow_lyon(m, strict=False))
                    cv = R.cornell_voltage(m)
                    put("R03_gt", cv)
                    cmv, sx = R.cornell_mv(m), R._sex(m)
                    if cmv is not None and sx is not None:
                        out["R03_ge"][i] = float(cmv >= (2.8 if sx == "M" else 2.0))
                    put("R03_prod", R.cornell_product(m))
                    if cmv is not None:  # K2: feste Schwelle fuer beide Geschlechter (Maenner- bzw. Frauengrenze)
                        out["R03_fixM"][i] = float(cmv > 2.8)
                        out["R03_fixF"][i] = float(cmv > 2.0)
                    if not rbbb_ex:
                        put("R04_ge", R.peguero_lo_presti(m, strict=False))
                        put("R04_gt", R.peguero_lo_presti(m, strict=True))
                        pm = R.peguero_mv(m)
                        if pm is not None:
                            out["R04_fixM"][i] = float(pm >= 2.8)
                            out["R04_fixF"][i] = float(pm >= 2.3)
                put("R06_a", R.low_voltage_limb(m, strict=True))
                put("R06_b", R.low_voltage_precordial(m, strict=True))
                put("R06_c", R.low_voltage_ab(m, strict=True))
                put("R07_a", R.low_voltage_limb(m, strict=False))
                put("R07_b", R.low_voltage_precordial(m, strict=False))
                put("R07_c", R.low_voltage_sokolow(m))
                flag = row.get("classical_pathological_q_waves_positive")
                flag_ok = flag is not None and not (isinstance(flag, float) and math.isnan(flag)) and flag is not pd.NA
                if flag_ok and lvh is False and volt_ok:  # v3: Schrittmacher und (auch unsicherer) Maschine-LSB schliessen aus
                    out["R14_a155"][i] = float(bool(flag))
            rb = row.get("classical_rbbb_pattern_positive")
            if rb is not None and rb is not pd.NA and not (isinstance(rb, float) and math.isnan(rb)):
                out["R09_a155"][i] = float(bool(rb))
        # O2 Infarkt (R14 Konkordanz) und RSB-Maschinenaussage
        inf = row.get("report_infarct_state")
        out["R14_o2"][i] = 1.0 if inf == "positive" else (0.0 if inf == "not_mentioned" else np.nan)
    res = pd.DataFrame(out, index=idx.index)
    res["R14_a155"] = res["R14_a155"]
    return res


def machine_flags(idx: pd.DataFrame) -> pd.DataFrame:
    """Aus Befundtext und Zustandsspalten: Normal-EKG, Sinus, Kategorien, Ausschluss- und Expositionsflags (R09, R18, R20)."""
    t = idx["report_text"].astype(object).values
    n = len(idx)
    sinus = np.array([np.nan if x is None or (isinstance(x, float) and math.isnan(x)) else (np.nan if R.maschine_sinus(x) is None else float(R.maschine_sinus(x))) for x in t])
    normal = np.array([np.nan if (x is None or (isinstance(x, float) and math.isnan(x))) or R.maschine_normal(x) is None else float(R.maschine_normal(x)) for x in t])
    has_txt = np.array([isinstance(x, str) and len(x.strip()) > 0 for x in t])

    def rx(r, x):
        return bool(r.search(x)) if isinstance(x, str) else False

    normal_add = np.array([float(rx(RX_NORMAL_ANY, x) and not rx(RX_ABN, x)) if h else np.nan for x, h in zip(t, has_txt)])
    no_stt = np.array([float(not rx(RX_STT, x)) if h else np.nan for x, h in zip(t, has_txt)])
    lafb = np.array([float(rx(RX_LAFB, x)) for x in t])
    avb = np.array([float(rx(RX_AVB, x)) for x in t])
    ivcd = np.array([float(rx(RX_IVCD, x)) for x in t])
    wpw = np.array([float(rx(RX_WPW, x)) for x in t])
    cat = np.where(normal == 1, 0, np.where(np.array([rx(RX_BORDER, x) for x in t]), 1, 2))
    st = lambda c: idx[c].astype(object).values  # noqa: E731
    paced = (st("report_pacemaker_state") == "positive").astype(float)
    lbbb_any = (st("report_lbbb_state") != "not_mentioned").astype(float)
    af_any = (st("report_af_state") != "not_mentioned").astype(float)
    lvh_m = (st("report_lvh_state") != "not_mentioned").astype(float)
    inf_m = (st("report_infarct_state") != "not_mentioned").astype(float)
    rbbb_state = st("report_rbbb_state")
    out = pd.DataFrame({
        "sinus": sinus, "normal": normal, "normal_add": normal_add, "no_stt": no_stt, "mcat": cat, "paced": paced,
        "lbbb_any": lbbb_any, "af_any": af_any, "lvh_m": lvh_m, "inf_m": inf_m, "lafb": lafb, "avb": avb, "ivcd": ivcd, "wpw": wpw,
        "rbbb_pos": (rbbb_state == "positive").astype(float), "rbbb_nm": (rbbb_state == "not_mentioned").astype(float),
    }, index=idx.index)
    both_ex = (out["paced"] > 0) | (out["lvh_m"] > 0) | (out["lafb"] > 0) | (out["avb"] > 0) | (out["inf_m"] > 0) | (out["af_any"] > 0)
    r09 = np.full(n, np.nan)
    r09[(out["rbbb_pos"] > 0) & ~both_ex] = 1.0
    r09[(out["rbbb_nm"] > 0) & (out["lbbb_any"] == 0) & (out["ivcd"] == 0) & ~both_ex] = 0.0
    out["R09_m"] = r09
    out["r09_both_excl"] = both_ex.astype(float)
    return out


def sql_codes(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg["codelist"], dtype={"icd_code": str})
    df["icd_code"] = df["icd_code"].str.strip()
    return df


def history_and_codes(con, cfg, idx: pd.DataFrame) -> pd.DataFrame:
    """Vorgeschichte aus frueheren Aufenthalten (admittime < Bezugszeit des Index, anderer Aufenthalt) und Codes des
    Index-Aufenthalts. Bezugszeit = Aufnahmezeit des Index-Aufenthalts, sonst EKG-Zeit. Nur Code- und Zeitlogik, keine
    Rueckschluesse auf EKG oder Regel."""
    hosp = cfg["hosp"]
    k = idx[["subject_id", "hadm_id", "ecg_time"]].copy()
    k["pidx"] = np.arange(len(k))
    con.register("k_c", k)
    codes = sql_codes(cfg)
    con.register("k_codes", codes[["list_id", "icd_version", "icd_code"]])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE adm AS
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime FROM read_csv('{hosp}/admissions.csv.gz', header=true,
        columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','admittime':'TIMESTAMP','dischtime':'TIMESTAMP','deathtime':'TIMESTAMP','admission_type':'VARCHAR','admit_provider_id':'VARCHAR','admission_location':'VARCHAR','discharge_location':'VARCHAR','insurance':'VARCHAR','language':'VARCHAR','marital_status':'VARCHAR','race':'VARCHAR','edregtime':'TIMESTAMP','edouttime':'TIMESTAMP','hospital_expire_flag':'INTEGER'}}) a
        JOIN (SELECT DISTINCT subject_id FROM k_c) s ON s.subject_id = a.subject_id""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE dxm AS
        SELECT d.subject_id, d.hadm_id, cl.list_id FROM read_csv('{hosp}/diagnoses_icd.csv.gz', header=true,
        columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','seq_num':'INTEGER','icd_code':'VARCHAR','icd_version':'INTEGER'}}) d
        JOIN (SELECT DISTINCT subject_id FROM k_c) s ON s.subject_id = d.subject_id
        JOIN k_codes cl ON cl.icd_version = d.icd_version AND cl.icd_code = trim(d.icd_code)""")
    con.execute("""CREATE OR REPLACE TEMP TABLE dx_any AS
        SELECT DISTINCT d.subject_id, d.hadm_id FROM read_csv('""" + f"{hosp}/diagnoses_icd.csv.gz" + """', header=true,
        columns={'subject_id':'BIGINT','hadm_id':'BIGINT','seq_num':'INTEGER','icd_code':'VARCHAR','icd_version':'INTEGER'}) d
        JOIN (SELECT DISTINCT subject_id FROM k_c) s ON s.subject_id = d.subject_id""")
    con.execute("""CREATE OR REPLACE TEMP TABLE kref AS
        SELECT k.pidx, k.subject_id, k.hadm_id AS idx_hadm, k.ecg_time, COALESCE(a.admittime, k.ecg_time) AS t_ref, a.dischtime AS idx_disch
        FROM k_c k LEFT JOIN adm a ON a.hadm_id = k.hadm_id AND a.subject_id = k.subject_id""")
    con.execute("""CREATE OR REPLACE TEMP TABLE prior AS
        SELECT r.pidx, a.hadm_id FROM kref r JOIN adm a ON a.subject_id = r.subject_id
        WHERE a.admittime < r.t_ref AND (r.idx_hadm IS NULL OR a.hadm_id <> r.idx_hadm)""")
    pf = con.execute("""SELECT p.pidx, d.list_id, count(*) AS n FROM prior p JOIN dxm d ON d.hadm_id = p.hadm_id GROUP BY 1,2""").df()
    npri = con.execute("""SELECT p.pidx, count(DISTINCT p.hadm_id) AS n_prior_dx FROM prior p JOIN dx_any x ON x.hadm_id = p.hadm_id GROUP BY 1""").df()
    ixf = con.execute("""SELECT r.pidx, d.list_id, count(*) AS n FROM kref r JOIN dxm d ON d.hadm_id = r.idx_hadm AND d.subject_id = r.subject_id GROUP BY 1,2""").df()
    ixdx = con.execute("""SELECT r.pidx, 1 AS idx_has_dx FROM kref r JOIN dx_any x ON x.hadm_id = r.idx_hadm AND x.subject_id = r.subject_id GROUP BY 1""").df()
    admi = con.execute("SELECT pidx, t_ref, idx_disch FROM kref").df()
    out = pd.DataFrame({"pidx": np.arange(len(k))})
    for lid in codes["list_id"].unique():
        out["prior_" + lid] = 0
        out["idx_" + lid] = 0
    pv = pf.pivot_table(index="pidx", columns="list_id", values="n", aggfunc="sum").reindex(out["pidx"]).fillna(0)
    iv = ixf.pivot_table(index="pidx", columns="list_id", values="n", aggfunc="sum").reindex(out["pidx"]).fillna(0)
    for lid in pv.columns:
        out["prior_" + lid] = (pv[lid].values > 0).astype(int)
    for lid in iv.columns:
        out["idx_" + lid] = (iv[lid].values > 0).astype(int)
    out["n_prior_dx"] = npri.set_index("pidx")["n_prior_dx"].reindex(out["pidx"]).fillna(0).astype(int).values
    out["idx_has_dx"] = ixdx.set_index("pidx")["idx_has_dx"].reindex(out["pidx"]).fillna(0).astype(int).values
    out["t_ref"] = admi.set_index("pidx")["t_ref"].reindex(out["pidx"]).values
    out["idx_disch"] = admi.set_index("pidx")["idx_disch"].reindex(out["pidx"]).values
    for t in ("k_c", "k_codes"):
        con.unregister(t)
    # PCI/CABG aus der Pipeline-Tabelle patient_code_history (Datum vor dem Index-Aufenthalt)
    pch = fetch_by_keys(con, f"read_parquet('{cfg['root']}/evidence/patient_code_history.parquet')", idx, ["subject_id"],
                        ["subject_id", "first_pci_date", "first_cabg_date"])
    m = idx[["subject_id"]].merge(pch.drop_duplicates("subject_id"), on="subject_id", how="left")
    tref_day = pd.to_datetime(out["t_ref"]).dt.normalize()
    pci = pd.to_datetime(m["first_pci_date"]).values
    cabg = pd.to_datetime(m["first_cabg_date"]).values
    out["prior_pci_cabg"] = (((pd.Series(pci) < tref_day.values) & pd.Series(pci).notna()) |
                             ((pd.Series(cabg) < tref_day.values) & pd.Series(cabg).notna())).astype(int).values
    return out


def covariates(idx: pd.DataFrame, rules_ctx: pd.DataFrame, hist: pd.DataFrame, meds: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Kovariaten des Selektions- und Adjustierungsmodells (nur Information vor dem Index)."""
    n = len(idx)
    icu = idx["icustay_id"].notna() & (idx["icu_intime"] <= idx["ecg_time"]) & (idx["ecg_time"] <= idx["icu_outtime"])
    ed = idx["ed_stay_id"].notna()
    inp = idx["hadm_id"].notna()
    adm = np.where(icu, "icu", np.where(ed, "ed", np.where(inp, "inpatient", "other")))
    hr = idx["hr_bpm"].astype(float).values
    rr = idx["rr_ms"].astype(float).values
    with np.errstate(divide="ignore", invalid="ignore"):
        hr = np.where(np.isfinite(hr), hr, np.where(np.isfinite(rr) & (rr > 0), 60000.0 / rr, np.nan))
    hr = np.clip(hr, 30.0, 220.0)  # Maschinenwerte ausserhalb der plausiblen Spanne werden geklemmt
    cov = pd.DataFrame({
        "age": idx["age"].astype(float).values,
        "female": (idx["sex"] == "F").astype(float).values,
        "sex_known": idx["sex"].isin(["M", "F"]).astype(float).values,
        "adm": adm,
        "hr": hr,
    }, index=idx.index)
    cov["hr_missing"] = (~np.isfinite(cov["hr"])).astype(float)
    cov["hr"] = cov["hr"].fillna(np.nanmedian(cov["hr"]) if np.isfinite(cov["hr"]).any() else 70.0)
    for k, lid in (("hx_hf", "H_HERZINSUFFIZIENZ"), ("hx_af", "H_VHF"), ("hx_cad", "H_KHK_INFARKT"), ("hx_htn", "K_HYPERTONIE"),
                   ("hx_dm", "K_DIABETES"), ("hx_ckd", "K_NIERE"), ("hx_copd", "H_COPD"), ("hx_cancer", "K_KREBS"),
                   ("hx_stroke", "K_SCHLAGANFALL"), ("hx_valve", "H_KLAPPEN"), ("hx_cm", "H_KARDIOMYOPATHIE"),
                   ("hx_pe", "H_LUNGENEMBOLIE"), ("hx_ph", "H_PULM_HYPERTONIE")):
        cov[k] = hist["prior_" + lid].values.astype(float)
    cov["claim_pop"] = ((cov[["hx_hf", "hx_af", "hx_cad", "hx_cm", "hx_valve", "hx_copd", "hx_pe", "hx_ph"]].sum(axis=1)) == 0).astype(float)
    cov["has_prior"] = (hist["n_prior_dx"].values > 0).astype(float)
    cov["mcat_border"] = (rules_ctx["mcat"].values == 1).astype(float)
    cov["mcat_abn"] = (rules_ctx["mcat"].values == 2).astype(float)
    for c in meds.columns:
        cov[c] = meds[c].values
    for c in labs.columns:
        cov[c] = labs[c].values
    return cov


def load_meds_labs(con, cfg, idx: pd.DataFrame):
    root = cfg["root"]
    keys = ["subject_id", "study_id"]
    mcols = ["med_beta_blocker", "med_calcium_channel_blocker", "med_digoxin", "med_antiarrhythmic_class_i_iii", "med_diuretic",
             "med_qt_prolonging_drug"]
    hcols = [c + "_home_reported" for c in mcols]
    med = fetch_by_keys(con, f"read_parquet('{root}/evidence/ecg_medication_features.parquet')", idx, keys, keys + mcols + hcols)
    m = idx[keys].merge(med, on=keys, how="left")
    out = pd.DataFrame(index=idx.index)
    out["med_assessed"] = m[mcols[0]].notna().astype(float).values
    for c in mcols:
        out[c] = (m[c].astype(object).fillna(False).astype(bool) | m[c + "_home_reported"].astype(object).fillna(False).astype(bool)).astype(float).values
    lcols = ["potassium_nearest", "potassium_delta_hours", "magnesium_nearest", "magnesium_delta_hours", "total_calcium_nearest",
             "total_calcium_delta_hours", "troponin_count_24h"]
    lab = fetch_by_keys(con, f"read_parquet('{root}/evidence/ecg_lab_features.parquet')", idx, keys, keys + lcols)
    lm = idx[keys].merge(lab, on=keys, how="left")
    labs = pd.DataFrame(index=idx.index)
    for a in ("potassium", "magnesium", "total_calcium"):
        v = lm[f"{a}_nearest"].where(lm[f"{a}_delta_hours"].abs() <= 24.0)
        labs["lab_" + a] = v.values
    labs["trop_count"] = lm["troponin_count_24h"].values
    return out, labs


def echo_refs(con, cfg, idx: pd.DataFrame) -> pd.DataFrame:
    """Echo-Referenzen im Fenster ±30 Tage, ±7 Tage, ±24 h; naechstliegendes TTE (Pipelinezuordnung)."""
    root = cfg["root"]
    mt = fetch_by_keys(con, f"read_parquet('{root}/evidence/ecg_echo_matches.parquet')", idx, ["study_id"],
                       ["study_id", "nearest_tte_measurement_id", "nearest_tte_delta_hours"])
    m = idx[["study_id", "sex"]].merge(mt.drop_duplicates("study_id"), on="study_id", how="left")
    ids = m["nearest_tte_measurement_id"].dropna().astype("int64").unique()
    con.register("k_mid", pd.DataFrame({"measurement_id": ids}))
    cols = ["measurement_id", "preferred_lvef", "septal_thickness_cm", "posterior_wall_thickness_cm",
            "lv_internal_diameter_diastole_cm", "echo_body_surface_area_m2", "pericardial_effusion", "test_type"]
    e = con.execute(f"SELECT {', '.join('t.' + c for c in cols)} FROM read_parquet('{root}/evidence/echo_measurements_wide.parquet') t JOIN k_mid k ON k.measurement_id = t.measurement_id").df()
    con.unregister("k_mid")
    m = m.merge(e.drop_duplicates("measurement_id"), left_on="nearest_tte_measurement_id", right_on="measurement_id", how="left")
    assert len(m) == len(idx)
    ivs, pw, lvid, bsa = (m[c].astype(float) for c in ("septal_thickness_cm", "posterior_wall_thickness_cm", "lv_internal_diameter_diastole_cm", "echo_body_surface_area_m2"))
    plaus = ivs.between(0.3, 3.0) & pw.between(0.3, 3.0) & lvid.between(2.0, 9.0) & bsa.between(0.8, 3.5)
    mass = 0.8 * 1.04 * ((ivs + lvid + pw) ** 3 - lvid ** 3) + 0.6
    lvmi = mass / bsa
    thr = np.where(m["sex"] == "M", 115.0, np.where(m["sex"] == "F", 95.0, np.nan))
    lvh = np.where(plaus & np.isfinite(thr), (lvmi > thr).astype(float), np.nan)
    lvef = m["preferred_lvef"].astype(float)
    lvef35 = np.where(lvef.notna(), (lvef <= 35.0).astype(float), np.nan)
    pe = m["pericardial_effusion"].astype(object)
    pos_set = {"Moderate (1.0-2.0cm)", "Moderate-large", "Large (>2.0cm)"}
    neg_set = {"Very small (<0.5cm)", "Small (<1.0cm)", "Trivial", "Anterior fat pad"}
    peri = np.where(pe.isin(pos_set), 1.0, np.where(pe.isin(neg_set), 0.0, np.nan))  # v3: leeres Feld = unbekannt
    peri_l = np.where(pe.isin(pos_set), 1.0, np.where(pe.isna() | pe.isin(neg_set), 0.0, np.nan))  # Sensitivitaet (AE-295)
    ivs12 = np.where(ivs.notna() & ivs.between(0.3, 3.0), (ivs >= 1.2).astype(float), np.nan)
    dh = m["nearest_tte_delta_hours"].abs().values
    has_tte = m["nearest_tte_measurement_id"].notna().values & (m["test_type"].astype(object).values == "tte")
    out = pd.DataFrame(index=idx.index)
    for w, lim in (("30d", 720.0), ("7d", 168.0), ("24h", 24.0)):
        inw = has_tte & np.isfinite(dh) & (dh <= lim)
        out[f"echo_{w}"] = inw.astype(float)
        for nm, arr in (("lvh", lvh), ("lvef35", lvef35), ("peri", peri), ("periL", peri_l), ("ivs12", ivs12)):
            out[f"{nm}_{w}"] = np.where(inw, arr, np.nan)
    out["echo_delta_h"] = dh
    return out


def label_refs(con, cfg, idx: pd.DataFrame, mode, ga) -> pd.DataFrame:
    """Pipeline-Labels (Labeldefinition v2.9.0) am Index-EKG: positive -> 1, negative -> 0, sonst NaN."""
    root = cfg["root"]
    want = {"vtvf": "vt_vf_icd_or_event_state", "arrest": "cardiac_arrest_state", "mort30": "mortality_30d_state",
            "mi_prob": "acute_mi_probable_state", "trop_pos": "troponin_positive_any_state",
            "cond1y": "conduction_progression_1y_state", "rv_dil": "echo_rv_dilation_state",
            "ph_echo": "echo_pulmonary_hypertension_state", "peri_icd": "pericardial_effusion_state"}
    if mode == "test":
        guard_phase("test", "locked_test", **ga)
        rel = f"read_parquet('{cfg['sealed_labels']}')"
        have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
        cols = ["study_id"] + [c for c in want.values() if c in have]
        miss = [c for c in want.values() if c not in have]
        if miss:
            log(f"WARNUNG: Spalten in den Testlabels nicht vorhanden (werden als fehlend gefuehrt): {miss}")
        df = fetch_by_keys(con, rel, idx, ["study_id"], cols)
        for c in miss:
            df[c] = None
    else:
        frames = []
        for s in splits_for(mode):
            rel = "(" + part_sql(f"read_parquet('{root}/labels/clinical_labels_wide.parquet')", s, "dev") + ")"
            sub = idx[idx["split"] == s]
            frames.append(fetch_by_keys(con, rel, sub, ["study_id"], ["study_id"] + list(want.values())))
        df = pd.concat(frames, ignore_index=True)
    m = idx[["study_id"]].merge(df.drop_duplicates("study_id"), on="study_id", how="left")
    out = pd.DataFrame(index=idx.index)
    for k, col in want.items():
        v = m[col].astype(object).values
        out["lab_" + k] = np.where(v == "positive", 1.0, np.where(v == "negative", 0.0, np.nan))
    return out


def af_risk_times(af_day, lastobs, death_day, death_in):
    """Risikozeit und Ereignis fuer neues dokumentiertes VHF (cause-specific). Das Ende ist das Minimum aus letztem Kontakt, Tod im
    Fenster (konkurrierend) und 365 Tagen; VHF zaehlt nur, wenn es vor diesem Ende liegt (Sol-Fehler 4). Keine Mindestzeit."""
    af_day = np.asarray(af_day, dtype=float)
    lastobs = np.asarray(lastobs, dtype=float)
    death_day = np.asarray(death_day, dtype=float)
    contact_end = np.minimum(np.where(np.isfinite(lastobs), np.maximum(lastobs, 0.0), 0.0), 365.0)
    end = np.minimum(contact_end, np.where(np.asarray(death_in, dtype=bool), death_day, 365.0))
    ev = np.isfinite(af_day) & (af_day > 0.0001) & (af_day <= end)
    t = np.where(ev, af_day, end)  # v3.1: keine Untergrenze; Risikozeit 0 wird in der Analyse gezaehlt und ausgeschlossen
    return t, ev


def calendar_day_diff(ecg_time, ref_time):
    """Kalendertage zwischen den Datumsanteilen (0 = am selben Kalendertag, 'erster Aufenthaltstag'; Sol-Fehler 5)."""
    a = pd.to_datetime(ecg_time).dt.normalize() if hasattr(ecg_time, "dt") else pd.to_datetime(pd.Series(ecg_time)).dt.normalize()
    b = pd.to_datetime(ref_time).dt.normalize() if hasattr(ref_time, "dt") else pd.to_datetime(pd.Series(ref_time)).dt.normalize()
    return ((a.values - b.values) / np.timedelta64(1, "D")).astype(float)


def timeline_refs(con, cfg, idx: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    cols = ["subject_id", "first_af_event_time", "last_observed_time", "vital_status_known_until", "date_of_death"]
    tl = fetch_by_keys(con, f"read_parquet('{cfg['root']}/evidence/patient_clinical_timeline.parquet')", idx, ["subject_id"], cols)
    m = idx[["subject_id", "ecg_time"]].merge(tl.drop_duplicates("subject_id"), on="subject_id", how="left")
    t0 = pd.to_datetime(m["ecg_time"])
    day = lambda s: (pd.to_datetime(s) - t0).dt.total_seconds() / 86400.0  # noqa: E731
    death_day = (pd.to_datetime(m["date_of_death"]) - t0.dt.normalize()).dt.total_seconds() / 86400.0
    known = day(m["vital_status_known_until"])
    lastobs = day(m["last_observed_time"])
    af_day = day(m["first_af_event_time"])
    out = pd.DataFrame(index=idx.index)
    # Tod 365 Tage
    ev = death_day.between(0, 365)
    t_death = np.where(ev, death_day, np.minimum(known.fillna(np.nan), 365.0))  # v3.1: keine Untergrenze
    ok_death = (death_day.isna() | (death_day >= 0)) & (ev | (known > 0))
    out["death_t"] = np.where(ok_death, t_death, np.nan)
    out["death_ev"] = np.where(ok_death, ev.astype(float), np.nan)
    # neues dokumentiertes VHF (cause-specific; Tod konkurrierend; Zensur am letzten Kontakt). v3: Das Ende der Risikozeit ist das
    # Minimum aus Kontaktende, Tod im Fenster und 365 Tagen; VHF zaehlt nur als Ereignis, wenn es vor diesem Ende liegt.
    prev_af = (af_day <= 0).fillna(False)
    t_af, af_ev = af_risk_times(af_day.values, lastobs.values, death_day.values, ev.values)
    out["af_t"] = t_af
    out["af_ev"] = af_ev.astype(float)
    out["af_prev"] = prev_af.astype(float)
    disch = pd.to_datetime(hist["idx_disch"]).values
    ref_end = np.where(pd.isna(disch), t0.values, np.maximum(disch, t0.values))
    out["followup_contact"] = (((pd.to_datetime(m["last_observed_time"]).values - ref_end) / np.timedelta64(1, "D")) > 1.0).astype(float)
    return out


def work_dir(out: Path, mode: str) -> Path:
    return Path(out) / mode / "work"


def plausibility_checks(idx: pd.DataFrame) -> dict:
    """Vertragspruefung der Cache-Spalten (v3, Sol-Verdacht 1 und 2): S-Tiefen und R-Hoehen sind nichtnegative Betraege
    (Erzeugercode `_qrs_amplitudes`: max(0, ...)); `classical_qrs_widening_ge_120` ist die QRS-Dauer in ms (Erzeugercode `_criterion`
    mit Wert qrs_ms, Einheit ms), `pr_interval_ms` die PR-Zeit in ms. Bei Verletzung bricht prepare ab."""
    out = {}
    amp_cols = [c for c in idx.columns if c.startswith("lead_measurement_") and (c.endswith("_qrs_r_height_mv") or c.endswith("_qrs_s_depth_mv"))]
    if not amp_cols:
        return out
    mn = float(np.nanmin(idx[amp_cols].astype(float).values))
    out["amplituden_minimum_mv"] = mn
    if mn < -1e-9:
        raise SystemExit(f"Vertragsbruch: negative QRS-Amplituden im Cache (Minimum {mn})")
    q = idx["classical_qrs_widening_ge_120"].astype(float)
    out["qrs_eigen_median_ms"] = float(np.nanmedian(q))
    out["qrs_eigen_max"] = float(np.nanmax(q))
    if not (60.0 <= out["qrs_eigen_median_ms"] <= 200.0):
        raise SystemExit(f"Vertragsbruch: classical_qrs_widening_ge_120 ist keine ms-Dauer (Median {out['qrs_eigen_median_ms']})")
    pr = idx["pr_interval_ms"].astype(float)
    out["pr_eigen_median_ms"] = float(np.nanmedian(pr))
    if not (80.0 <= out["pr_eigen_median_ms"] <= 300.0):
        raise SystemExit(f"Vertragsbruch: pr_interval_ms ist keine ms-Dauer (Median {out['pr_eigen_median_ms']})")
    return out


def process_cohort(name, idx, con, cfg, mode, ga, wdir: Path, light: bool, rules_only: bool):
    """Baut Regeltabelle, Kovariaten und Referenztabelle eines Kohorte (getrennte Dateien: Trennschutz, 11 Stufe 3)."""
    log(f"[{name}] Personen {len(idx)}")
    txt = load_text_and_rr(con, cfg, idx)
    idx = idx.merge(txt, on="study_id", how="left")
    mf = machine_flags(idx)
    idx = pd.concat([idx, mf], axis=1)
    cache = load_cache(con, cfg, idx, mode, ga) if not light else pd.DataFrame({"study_id": idx["study_id"].values})
    idx = idx.merge(cache, on="study_id", how="left")
    has_cache = (mode != "test") and not light
    checks = plausibility_checks(idx) if has_cache else {}
    rules = evaluate_rules(idx, has_cache)
    rules = pd.concat([rules, mf], axis=1)
    log(f"[{name}] Regeln bewertet")
    if rules_only:
        hist = pd.DataFrame({"pidx": np.arange(len(idx))})
        codes = sql_codes(cfg)
        for lid in codes["list_id"].unique():
            hist["prior_" + lid] = 0
            hist["idx_" + lid] = 0
        hist["n_prior_dx"], hist["idx_has_dx"], hist["prior_pci_cabg"] = 0, 0, 0
        hist["t_ref"], hist["idx_disch"] = idx["ecg_time"].values, pd.NaT
    else:
        hist = history_and_codes(con, cfg, idx)
    meds, labs = load_meds_labs(con, cfg, idx)
    cov = covariates(idx, mf, hist, meds, labs)
    keys = idx[["subject_id", "study_id", "split", "ecg_time", "hadm_id"]].copy()
    refs = pd.DataFrame(index=idx.index)
    if not rules_only:
        refs = pd.concat([refs, label_refs(con, cfg, idx, mode, ga)], axis=1)
        for c in hist.columns:
            if c.startswith(("idx_", "prior_")) or c in ("n_prior_dx",):
                refs[c] = hist[c].values
        if not light:
            refs = pd.concat([refs, echo_refs(con, cfg, idx), timeline_refs(con, cfg, idx, hist)], axis=1)
        sd = calendar_day_diff(idx["ecg_time"].reset_index(drop=True), hist["t_ref"].reset_index(drop=True))
        refs["ecg_stay_day"] = np.where(idx["hadm_id"].notna().values, sd, np.nan)
    wdir.mkdir(parents=True, exist_ok=True)
    keys.reset_index(drop=True).to_parquet(wdir / f"{name}_keys.parquet")
    rules.reset_index(drop=True).to_parquet(wdir / f"{name}_rules.parquet")
    cov.reset_index(drop=True).to_parquet(wdir / f"{name}_cov.parquet")
    refs.reset_index(drop=True).to_parquet(wdir / f"{name}_refs.parquet")
    log(f"[{name}] gespeichert: Regeln {rules.shape}, Kovariaten {cov.shape}, Referenzen {refs.shape}")
    return dict(n=int(len(idx)), checks=checks)


def prepare(cfg, mode, out, ga, sample_mod=0, rules_only=False):
    out = Path(out)
    wdir = work_dir(out, mode)
    con = connect(cfg)
    t0 = time.time()
    base = load_base(con, cfg, mode, ga)
    if mode == "test":  # Hash der Testlabels erst nach der Zeile Beginn (Spezifikation 3 und Freeze-Anhang A11)
        wdir.mkdir(parents=True, exist_ok=True)
        hf = wdir / "eingaben_test.json"
        h_now = G.sha256_file(cfg["sealed_labels"])
        if hf.exists():  # Fortsetzung: Hash muss mit dem ersten uebereinstimmen (v3.1)
            old = json.loads(hf.read_text(encoding="utf-8")).get(str(cfg["sealed_labels"]))
            if old != h_now:
                raise SystemExit("Testlabels haben sich seit dem ersten Durchgang geaendert (Hash weicht ab)")
        else:
            hf.write_text(json.dumps({str(cfg["sealed_labels"]): h_now}, indent=1), encoding="utf-8")
    n_sex_unknown = int((~base["sex"].isin(["M", "F"])).sum())
    base = base[base["sex"].isin(["M", "F"])]  # v3: unbekanntes Geschlecht wird ausgeschlossen, nicht als maennlich gewertet
    if sample_mod:
        base = base[base["subject_id"] % sample_mod == 0]
    log(f"Basis: {len(base)} erwachsene EKG, {base['subject_id'].nunique()} Personen ({time.time() - t0:.0f} s)")
    cand = base[base["usable"] & base["ed_stay_id"].notna() & base["chief_complaint"].fillna("").str.lower().str.contains(RX_CHESTPAIN, regex=False)]
    trop = fetch_by_keys(con, f"read_parquet('{cfg['root']}/evidence/ecg_lab_features.parquet')", cand, ["study_id"], ["study_id", "troponin_count_24h"])
    trop = trop.rename(columns={"troponin_count_24h": "trop_count"}).drop_duplicates("study_id")
    idx_main = choose_index(base)
    idx_ed = choose_index_ed(base, trop)
    summary = {"mode": mode, "n_adult_ecg": int(len(base)), "n_adult_persons": int(base["subject_id"].nunique()),
               "n_index_main": int(len(idx_main)), "n_index_ed": int(len(idx_ed)), "rules_only": bool(rules_only),
               "sample_mod": int(sample_mod), "n_ecg_geschlecht_unbekannt_ausgeschlossen": n_sex_unknown}
    summary["pruefungen_main"] = process_cohort("main", idx_main, con, cfg, mode, ga, wdir, light=False, rules_only=rules_only)["checks"]
    process_cohort("ed", idx_ed, con, cfg, mode, ga, wdir, light=True, rules_only=rules_only)
    (wdir / "prepare_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if mode == "dev" and not rules_only:  # feste Knoten und Standardisierung der Entwicklung (fuer Entwicklung und Test)
        dm = Data(wdir, "main")
        (Path(out) / "dev").mkdir(parents=True, exist_ok=True)
        (Path(out) / "dev" / "design_dev.json").write_text(json.dumps(Design(dm.cov).state()), encoding="utf-8")
    log(f"prepare abgeschlossen ({time.time() - t0:.0f} s): {summary}")
    return summary


# ================================================================== Laden fuer die Einheiten
class Data:
    """Aufbereitete Tabellen einer Betriebsart. Regeln und Referenzen werden erst in den Einheiten verbunden."""

    def __init__(self, wdir: Path, name: str):
        self.keys = pd.read_parquet(wdir / f"{name}_keys.parquet")
        self.rules = pd.read_parquet(wdir / f"{name}_rules.parquet")
        self.cov = pd.read_parquet(wdir / f"{name}_cov.parquet")
        self.refs = pd.read_parquet(wdir / f"{name}_refs.parquet")
        self.n = len(self.keys)


def age_band(age: np.ndarray) -> np.ndarray:
    return np.digitize(age, AGE_BANDS)  # 0: <50, 1: 50-69, 2: >=70


class Design:
    """Basismatrix Xbase (ohne Regel T): Alter (Spline, 4 Knoten), Geschlecht, Aufnahmeart, Vorgeschichte, Maschinenkategorie,
    Herzfrequenz. Standardisierung und Knoten werden aus der Entwicklung uebernommen, falls vorhanden (feste Knoten)."""

    HX = ["hx_hf", "hx_af", "hx_cad", "hx_htn", "hx_dm", "hx_ckd", "hx_copd", "hx_cancer", "hx_stroke", "hx_valve"]

    def __init__(self, cov: pd.DataFrame, fixed: dict | None = None):
        self.knots = (fixed or {}).get("age_knots") or rcs_knots(cov["age"].values)
        base = self.raw(cov, self.knots)
        self.mu = np.array((fixed or {}).get("mu") or base.mean(axis=0))
        self.sd = np.array((fixed or {}).get("sd") or base.std(axis=0))
        self.sd[self.sd < 1e-9] = 1.0

    def raw(self, cov: pd.DataFrame, knots) -> np.ndarray:
        age = rcs_basis(cov["age"].values, knots)
        cols = [age, cov[["female"]].values]
        for a in ("icu", "ed", "inpatient"):
            cols.append((cov["adm"].values == a).astype(float)[:, None])
        cols.append(cov[self.HX].values)
        cols.append(cov[["mcat_border", "mcat_abn", "hr", "hr_missing"]].values)
        return np.hstack(cols).astype(float)

    def X(self, cov: pd.DataFrame, sel=None) -> np.ndarray:
        c = cov if sel is None else cov[sel]
        x = (self.raw(c, self.knots) - self.mu) / self.sd
        return x

    def state(self):
        return {"age_knots": list(self.knots), "mu": self.mu.tolist(), "sd": self.sd.tolist()}


def cells(T, D, V=None):
    """Zellbesetzung 2x2 (exakt; erst in der Ausgabe gerundet)."""
    T = np.asarray(T)
    D = np.asarray(D)
    ok = np.isfinite(T) & np.isfinite(D)
    if V is not None:
        ok &= np.asarray(V) > 0
    return {"tp": int(np.sum(ok & (T == 1) & (D == 1))), "fp": int(np.sum(ok & (T == 1) & (D == 0))),
            "fn": int(np.sum(ok & (T == 0) & (D == 1))), "tn": int(np.sum(ok & (T == 0) & (D == 0)))}


# ================================================================== Familie E (Einschluss/Struktur) und Familie A
RULE_INDEX = {"R02": 2, "R03": 3, "R04": 4, "R06": 6, "R07": 7, "R08": 8, "R09": 9, "R14": 14, "R18": 18, "R19": 19, "R20": 20}
PROXY_HINT = {"R06": "Referenz als Proxy (Dokumentationsassoziation, keine klinische Bestaetigung)",
              "R07": "Referenz als Proxy (Dokumentationsassoziation Amyloidose-Code mit Hinweis auf Herzbeteiligung)",
              "R14": "Referenz als Proxy (Dokumentationsassoziation frueherer Infarktcodes)",
              "R20_A1": "Referenz als Proxy (Infarkt-Label, P(Infarkt | normal) eher unterschaetzt)",
              "R20_A2": "Referenz als Proxy (Infarkt-Label, nur Infarkt statt gesamtes NSTE-ACS)"}


def class_from(kind, lo, hi, theta):
    if np.isnan(lo) or np.isnan(hi):  # v3.1: unendliche Schranken (LR+) werden klassifiziert
        return G.UNENTSCHIEDEN
    if kind == "ge":
        return G.classify_ge(lo, hi, theta)
    if kind == "le":
        return G.classify_le(lo, hi, theta)
    if kind == "gt":
        return classify_gt(lo, hi, theta)
    raise ValueError(kind)


def itv_of(reps, arr, alpha_c):
    """Intervall mit der Zahl der erwarteten Wiederholungen aus den Metadaten des Bootstraps."""
    return interval(arr, alpha_c, reps.get("_meta", {}).get("B") if isinstance(reps, dict) else None)


def comp_array(reps, comp, prefix="", grid=None):
    """Wiederholungen einer Komponente: spec/sens einer Variante oder Differenz der Sensitivitaeten (R04 Teil 2)."""
    m = comp["metric"]
    key = (lambda mm: mm) if grid is None else (lambda mm: f"kipp_{mm}_{grid}")
    if comp.get("diff"):
        a, b = comp["diff"]
        return reps[a][key(m)] - reps[b][key(m)]
    return reps[comp["T"]][key(m)]


def comp_point(point, comp, grid=None):
    m = comp["metric"]
    if grid is None:
        get = lambda t: point[t][m]  # noqa: E731
    else:
        sidx = 0 if m == "sens" else 1
        get = lambda t: point[t]["kipp"][grid][sidx]  # noqa: E731
    if comp.get("diff"):
        a, b = comp["diff"]
        return get(a) - get(b)
    return get(comp["T"])


def conc_comps(comps, conc_keys):
    """Komponenten fuer die Konkordanzfassung: die Regelschluessel der Variante tragen das Suffix '__c'."""
    out = []
    for c in comps:
        c2 = dict(c)
        c2["id"] = c["id"] + "_konkordanz"
        if c.get("diff"):
            c2["diff"] = tuple(k + "__c" if k in conc_keys else k for k in c["diff"])
        else:
            c2["T"] = c["T"] + "__c" if c["T"] in conc_keys else c["T"]
        out.append(c2)
    return out


def judge_e(point, reps, comps, family, with_kipp):
    """Urteil je Komponente und Einheit (Abschnitt 6.A), Kipppunktraster (5.2), Hinweisklassen. v3: Intervalle verlangen
    mindestens 95 % gueltige Wiederholungen; instabile Klassen zwischen den Haelften der Wiederholungen werden gekennzeichnet."""
    ac = G.alpha_component(family)
    out_c, classes = [], []
    for comp in comps:
        arr = comp_array(reps, comp)
        itv = itv_of(reps, arr, ac)
        cls = class_from(comp["kind"], itv["lo"], itv["hi"], comp["theta"])
        rec = dict(id=comp["id"], claim=comp["claim"], theta=comp["theta"], kind=comp["kind"], estimate=float(comp_point(point, comp)),
                   alpha_c=ac, klasse=cls, **itv)
        hints = []
        if itv.get("keine_inferenz"):
            hints.append("keine Inferenz (zu wenige gueltige Bootstrap-Wiederholungen)")
        else:
            c1 = class_from(comp["kind"], itv["lo_h1"], itv["hi_h1"], comp["theta"])
            c2 = class_from(comp["kind"], itv["lo_h2"], itv["hi_h2"], comp["theta"])
            if not (np.isnan(itv["lo_h1"]) or np.isnan(itv["lo_h2"])) and not (c1 == c2 == cls):
                hints.append("Monte-Carlo-instabil (Klasse wechselt zwischen den Haelften der Wiederholungen)")
        tkey = comp["T"] if not comp.get("diff") else comp["diff"][0]
        if comp["metric"] == "spec" and not comp.get("diff"):
            tb = point[comp["T"]]["trivial_bound"]
            rec["trivial_bound"] = float(tb)
            if np.isfinite(tb) and tb >= comp["theta"]:
                hints.append("durch Positivrate vorbestimmt")
        if point[tkey].get("ess_frac", 1.0) < 0.5:
            hints.append("Verifikationsanteil klein")
        rec["hinweise"] = hints
        out_c.append(rec)
        classes.append(cls)
    unit_cls = G.combine(classes)
    kipp = None
    if with_kipp:
        kipp = {}
        robust = True
        for gk in [f"{a}|{b}" for a in KAPPA_GRID for b in KAPPA_GRID]:
            cl = []
            vals = {}
            for comp in comps:
                arr = comp_array(reps, comp, grid=gk)
                itv = itv_of(reps, arr, ac)
                c1 = class_from(comp["kind"], itv["lo"], itv["hi"], comp["theta"])
                cl.append(c1)
                vals[comp["id"]] = dict(est=float(comp_point(point, comp, grid=gk)), lo=itv["lo"], hi=itv["hi"], klasse=c1)
            uc = G.combine(cl)
            kipp[gk] = dict(unit_klasse=uc, komponenten=vals)
            robust &= (uc == unit_cls)
        kipp["_robust"] = bool(robust)  # robust nur gegenueber den neun geprueften Szenarien
    return dict(components=out_c, unit_class=unit_cls, kipp=kipp)


def summary_metrics(point, reps, names=None, level=(0.025, 0.975)):
    """Punktschaetzer und unadjustiertes 95 %-Intervall der Kennzahlen je Variante."""
    out = {}
    for t in point:
        out[t] = {}
        for m in ("sens", "spec", "ppv", "npv", "prev", "lrp", "lrm", "p_t", "p_t_beobachtet", "ess", "ess_frac", "trivial_bound"):
            if m in point[t]:
                v = reps[t].get(m)
                itv = itv_of(reps, v, 0.025) if v is not None else {}
                out[t][m] = dict(est=float(point[t][m]), lo95=itv.get("lo95"), hi95=itv.get("hi95"))
        out[t]["manski"] = {k: float(v) for k, v in point[t]["manski"].items()}
        out[t]["manski_intervalle"] = {}
        for k, v in point[t]["manski"].items():
            arr = reps[t].get("manski_" + k)
            itv = itv_of(reps, arr, 0.025) if arr is not None else {}
            out[t]["manski_intervalle"][k] = dict(est=float(v), lo95=itv.get("lo95"), hi95=itv.get("hi95"))
    return out


def e_setup(unit, data: Data, win="30d"):
    """Populations- und Referenzdefinition je Einheit (Nachtrag 1: Referenzzustaende positiv/negativ/unbekannt).
    Rueckgabe: dict(mask, Tprim, Tconc, conc_keys, D, V, weighted, comps, ...)"""
    r, f, rf, cov = data.rules, data.refs, data.refs, data.cov
    n = data.n
    allv = np.ones(n, dtype=bool)
    if unit in ("R02", "R03", "R04"):
        D = rf[f"lvh_{win}"].values
        V = np.isfinite(D)
        if unit == "R02":
            T1, T2, ck = {"s": r["R02_gt"].values}, {"s": r["R02_ge"].values}, ["s"]
            mask = np.isfinite(T1["s"]) & np.isfinite(T2["s"])
            comps = [dict(id="R02_spec", T="s", metric="spec", kind="ge", theta=THETA["R02"], claim="Spezifitaet >= 0,90 (Echo-LVH)")]
        elif unit == "R03":
            T1, T2, ck = {"s": r["R03_gt"].values}, {"s": r["R03_ge"].values}, ["s"]
            mask = np.isfinite(T1["s"]) & np.isfinite(T2["s"])
            comps = [dict(id="R03_spec", T="s", metric="spec", kind="ge", theta=THETA["R03"], claim="Spezifitaet >= 0,90 (Echo-LVH)")]
        else:
            T1 = {"P": r["R04_ge"].values, "C": r["R03_gt"].values}
            T2 = {"P": r["R04_gt"].values, "C": r["R03_gt"].values}
            ck = ["P"]
            mask = np.isfinite(T1["P"]) & np.isfinite(T1["C"]) & np.isfinite(T2["P"])
            comps = [dict(id="R04_T1_spec", T="P", metric="spec", kind="ge", theta=THETA["R04"], claim="Spezifitaet (Peguero) >= 0,90"),
                     dict(id="R04_T2_dsens", diff=("P", "C"), metric="sens", kind="gt", theta=0.0, claim="Sens(Peguero) - Sens(Cornell) > 0 (strikt)")]
        return dict(mask=mask, mask_base=allv, Tprim=T1, Tconc=T2, conc_keys=ck, D=D, V=V, weighted=True, comps=comps, family="E")
    if unit == "R06":
        D = rf[f"peri_{win}"].values  # v3: leeres Echo-Feld = unbekannt (nicht negativ), 'Small-moderate' unbekannt
        V = np.isfinite(D)
        T1, T2 = {"s": r["R06_a"].values}, {"s": r["R06_c"].values}
        mask = np.isfinite(T1["s"]) & np.isfinite(T2["s"])
        comps = [dict(id="R06_spec", T="s", metric="spec", kind="ge", theta=THETA["R06_spec"], claim="Spezifitaet >= 0,98 (dokumentierter Erguss)"),
                 dict(id="R06_sens", T="s", metric="sens", kind="le", theta=THETA["R06_sens"], claim="Sensitivitaet <= 0,10 (dokumentierter Erguss)")]
        return dict(mask=mask, mask_base=allv, Tprim=T1, Tconc=T2, conc_keys=["s"], D=D, V=V, weighted=True, comps=comps, family="E")
    if unit == "R07":
        amy = rf["idx_AMYLOID"].values == 1
        cm = rf["idx_KARDIOMYOPATHIE_HINWEIS"].values == 1
        ivs = rf[f"ivs12_{win}"].values
        D = np.where(cm | (ivs == 1), 1.0, np.where(np.isfinite(ivs) & (ivs == 0), 0.0, np.nan))
        V = np.isfinite(D)
        T1, T2 = {"s": r["R07_a"].values}, {"s": r["R07_b"].values}
        mask = amy & np.isfinite(T1["s"]) & np.isfinite(T2["s"])
        comps = [dict(id="R07_sens", T="s", metric="sens", kind="le", theta=THETA["R07"], claim="Sensitivitaet <= 0,60 (Amyloidose-Code mit Hinweis auf Herzbeteiligung)")]
        return dict(mask=mask, mask_base=amy, Tprim=T1, Tconc=T2, conc_keys=["s"], D=D, V=V, weighted=True, comps=comps, family="E")
    if unit == "R14":
        D = r14_reference(rf)
        V = np.isfinite(D)
        T1, T2 = {"s": r["R14_a155"].values}, {"s": r["R14_o2"].values}
        mask = np.isfinite(T1["s"]) & np.isfinite(T2["s"]) & V
        comps = [dict(id="R14_spec", T="s", metric="spec", kind="ge", theta=THETA["R14"], claim="Spezifitaet >= 0,80 (fruehere Infarktcodes)")]
        return dict(mask=mask, mask_base=V.copy(), Tprim=T1, Tconc=T2, conc_keys=["s"], D=D, V=V, weighted=False, comps=comps, family="E")
    raise KeyError(unit)


def r14_reference(rf, index_codes=False, acute_pos=False):
    """R14-Referenz (Nachtrag 1): positiv = Infarktcode (alter Infarkt oder akuter Infarkt) in einem FRUEHEREN Aufenthalt;
    negativ = mindestens ein frueherer Aufenthalt mit Diagnoseliste und keiner dieser Codes; PCI/CABG allein belegt keinen
    Infarkt und zaehlt nicht (Personen nur mit PCI/CABG sind unbekannt); Personen ohne frueheren Aufenthalt unbekannt;
    akuter Infarkt im Index-Aufenthalt ausgeschlossen (Sensitivitaet acute_pos: als positiv; index_codes: Code 'alter Infarkt'
    des Index-Aufenthalts zaehlt als positiv)."""
    pos = (rf["prior_OLD_MI"].values + rf["prior_ACUTE_MI"].values) > 0
    only_proc = (~pos) & (rf["prior_pci_cabg"].values > 0)
    has_prior = rf["n_prior_dx"].values > 0
    idx_acute = rf["idx_ACUTE_MI"].values > 0
    if index_codes:
        pos = pos | (rf["idx_OLD_MI"].values > 0)
    if acute_pos:
        pos = pos | idx_acute
    known_neg = has_prior & ~pos & ~only_proc
    D = np.where(pos, 1.0, np.where(known_neg, 0.0, np.nan))
    if not acute_pos:
        D = np.where(idx_acute, np.nan, D)
    return D


def build_ipw(setup, data: Data, design: Design, conc=False, weighted=None, na_as=None, do_kipp=True, D=None, V=None, keep=None,
              with_conc=False):
    """IPWUnit aus Setup. with_conc: Konkordanzvarianten (Suffix '__c') in dieselbe Einheit aufnehmen; sie verwenden die Gewichte der
    Primaerfassung auf derselben Population (Spezifikation 5). na_as: Regelabstinenz (NaN) als 0 oder 1 werten (Sensitivitaet);
    dann gilt mask_base als Population. keep: Zusatzmaske (z. B. Geschlecht)."""
    T = dict(setup["Tconc"] if conc else setup["Tprim"])
    mask = setup["mask"].copy()
    if na_as is not None:
        T = {k: np.where(np.isfinite(v), v, float(na_as)) for k, v in T.items()}
        mask = setup["mask_base"].copy()
    wf = {}
    if with_conc:
        for k in setup["conc_keys"]:
            T[k + "__c"] = setup["Tconc"][k]
            wf[k + "__c"] = k
    Dv = setup["D"] if D is None else D
    Vv = setup["V"] if V is None else V
    if keep is not None:
        mask = mask & keep
    idx = np.where(mask)[0]
    cov = data.cov.iloc[idx]
    Xb = design.X(cov)
    female = cov["female"].values
    band = age_band(cov["age"].values)
    w = setup["weighted"] if weighted is None else weighted
    u = IPWUnit({k: v[idx] for k, v in T.items()}, np.asarray(Dv)[idx], np.asarray(Vv)[idx], Xb, female, band,
                do_kipp=do_kipp, weighted=w, weights_from=wf)
    u.index_rows = idx
    return u


def drop_constant_cols(unit: IPWUnit):
    """Konstante Spalten (z. B. Geschlecht im Teilkollektiv) aus dem Design nehmen (Singularitaet)."""
    for k, Z in unit.Z.items():
        keep = (Z.std(axis=0) > 1e-9)
        keep[0] = True
        unit.Z[k] = Z[:, keep]


def unit_final(unit_id, judged, judged_c, diag, boot_meta):
    """Endurteil (Nachtrag 1): Primaer- und Konkordanzklasse muessen uebereinstimmen; ist die Selektionspruefung nicht robust
    gegenueber den neun Szenarien oder die Identifizierbarkeit begrenzt, wird das Haupturteil auf 'unentschieden' eingeschraenkt."""
    final = judged["unit_class"]
    hints = []
    if judged_c is not None and judged_c["unit_class"] != judged["unit_class"]:
        final = G.UNENTSCHIEDEN
        hints.append("operationalisierungsabhaengig")
    if judged["kipp"] is not None and not judged["kipp"]["_robust"]:
        final = G.UNENTSCHIEDEN
        hints.append("durch Selektion nicht gedeckt (nicht robust gegenueber den neun geprueften Szenarien; Haupturteil eingeschraenkt)")
    if diag.get("identifizierbarkeit_begrenzt"):
        final = G.UNENTSCHIEDEN
        hints.append("Identifizierbarkeit begrenzt (Ueberlappung des Verifikationsmodells; Haupturteil eingeschraenkt)")
    if boot_meta.get("failed", 0) > 0:
        hints.append(f"{boot_meta['failed']} von {boot_meta['B']} Bootstrap-Wiederholungen ohne konvergierte Anpassung (nicht verworfen, gezaehlt)")
    for c in judged["components"]:
        hints.extend(c["hinweise"])
    if unit_id == "R14":
        hints.append("operationalisierungsabhaengig (Treueklasse von vornherein)")
    if unit_id in PROXY_HINT:
        hints.append(PROXY_HINT[unit_id])
    return final, hints


def run_e(unit_id, data: Data, design: Design, seed: int, mode: str, B: int, outdir: Path, log_prefix=""):
    """Eine Einheit der Familie E: Primaer und Konkordanz in einem Bootstrap (gleiche Gewichte, gleiche Population), Kipppunkt,
    Sensitivitaeten, Geschlecht."""
    t0 = time.time()
    ridx = RULE_INDEX[unit_id]
    setup = e_setup(unit_id, data)
    if setup["mask"].sum() < 30:
        res = dict(unit=unit_id, mode=mode, status="nicht auswertbar: weniger als 30 Personen in der Analysepopulation",
                   n_pop=int(setup["mask"].sum()))
        save_unit(outdir, mode, unit_id, res)
        return res
    u = build_ipw(setup, data, design, with_conc=True)
    log(f"{unit_id}: Population {u.n}, verifiziert {int(u.V.sum())}, Faelle {int(np.nansum(u.D[u.V > 0]))}")
    rng = seed_stream(seed, ridx, "primary", mode)
    point, reps = run_bootstrap(u, rng, B, tag=unit_id)
    if point is None:
        res = dict(unit=unit_id, mode=mode, family="E", status="nicht schaetzbar: Verifikationsmodell der Punktschaetzung nicht konvergiert",
                   final=dict(klasse=G.UNENTSCHIEDEN, primaer=G.UNENTSCHIEDEN, hinweise=["keine Inferenz"]))
        save_unit(outdir, mode, unit_id, res)
        return res
    judged = judge_e(point, reps, setup["comps"], setup["family"], with_kipp=setup["weighted"])
    judged_c = judge_e(point, reps, conc_comps(setup["comps"], setup["conc_keys"]), setup["family"], with_kipp=False)
    first_t = next(iter(u.sources))
    diag = weight_diagnostics(u, first_t)
    final, hints = unit_final(unit_id, judged, judged_c, diag, reps["_meta"])
    res = dict(unit=unit_id, mode=mode, family=setup["family"], status="ok", B=B, bootstrap=reps["_meta"],
               abdeckung=coverage_e(unit_id, data, setup), diagnostik=diag,
               counts=dict(n_pop=int(u.n), n_verified=int(u.V.sum()), cases=int(np.nansum(u.D[u.V > 0])),
                           **{f"cells_{k}": cells(u.T[k], u.D, u.V) for k in u.T}),
               fit=u.fit_info, metrics=summary_metrics(point, reps), judge=judged)
    # Hinweisurteile (explorativ, keine Urteilssprache; nicht in der Familienzaehlung): LR+ >= 5 und Sensitivitaet <= 0,5, einseitig 0,025
    if unit_id in ("R02", "R03"):
        lrp = itv_of(reps, reps["s"]["lrp"], 0.025)
        sn = itv_of(reps, reps["s"]["sens"], 0.025)

        def lage(kind, lo, hi, th):
            c = class_from(kind, lo, hi, th)
            return {G.HAELT: "Intervall innerhalb der Hinweisgrenze", G.HAELT_NICHT: "Intervall ausserhalb der Hinweisgrenze"}.get(c, "Hinweisgrenze im Intervall")

        res["hinweisurteile"] = {
            "LR+>=5": dict(estimate=float(point["s"]["lrp"]), lo=lrp["lo"], hi=lrp["hi"], klasse=lage("ge", lrp["lo"], lrp["hi"], 5.0),
                           herkunft="von den Autoren gesetzt (Orientierung Jaeschke 1994); explorativ, nicht in der Mehrfachkorrektur"),
            "Sens<=0,50": dict(estimate=float(point["s"]["sens"]), lo=sn["lo"], hi=sn["hi"], klasse=lage("le", sn["lo"], sn["hi"], 0.5),
                               herkunft="von den Autoren gesetzt (Orientierung Casale 1987); explorativ, nicht in der Mehrfachkorrektur")}
    res["concordance"] = dict(variante_klasse=judged_c["unit_class"], komponenten=judged_c["components"],
                              gleiche_gewichte_und_population=True,
                              metrics={k: v for k, v in summary_metrics(point, reps).items() if k.endswith("__c")})
    res["final"] = dict(klasse=final, primaer=judged["unit_class"], hinweise=sorted(set(hints)))
    save_unit(outdir, mode, unit_id, res)  # Zwischenspeicher (Checkpoint)
    # --- Sensitivitaeten (explorativ, ohne Urteil), B_SENS
    sens = {}
    variants = []
    if unit_id in ("R02", "R03", "R04", "R06", "R07"):
        variants += [("echo_7d", dict(win="7d")), ("echo_24h_ungewichtet", dict(win="24h", weighted=False))]
    variants += [("regel_NULL_als_negativ", dict(na_as=0)), ("regel_NULL_als_positiv", dict(na_as=1))]
    if unit_id == "R14":
        variants += [("referenz_mit_Codes_des_Indexaufenthalts", dict(ref14="index_oldmi")),
                     ("referenz_akuter_Infarkt_im_Index_positiv", dict(ref14="acute_pos")),
                     ("ohne_LAFB_und_WPW", dict(excl_lafb_wpw=True))]
    if unit_id == "R03":
        variants += [("sekundaer_Cornell_Produkt_Molloy", dict(Tcols={"s": "R03_prod"})),
                     ("K2_feste_Schwelle_2_8_fuer_beide_Geschlechter", dict(Tcols={"s": "R03_fixM"}, k2=True)),
                     ("K2_feste_Schwelle_2_0_fuer_beide_Geschlechter", dict(Tcols={"s": "R03_fixF"}, k2=True))]
    if unit_id == "R04":
        variants += [("K2_feste_Schwelle_2_8_fuer_beide_Geschlechter", dict(Tcols={"P": "R04_fixM", "C": "R03_fixM"}, k2=True)),
                     ("K2_feste_Schwelle_2_3_bzw_2_0_fuer_beide_Geschlechter", dict(Tcols={"P": "R04_fixF", "C": "R03_fixF"}, k2=True))]
    if unit_id == "R06":
        variants += [("sekundaer_Brustwand_R06_b", dict(Tcols={"s": "R06_b"})),
                     ("referenz_leeres_Feld_als_negativ", dict(peri_leer=True)),
                     ("referenz_ICD_Perikarderguss", dict(ref6="icd", weighted=False))]
    if unit_id == "R07":
        variants += [("sekundaer_Sokolow_R07_c", dict(Tcols={"s": "R07_c"})),
                     ("referenz_systemischer_Code_allein_Sensitivitaet", dict(ref7="code_allein", weighted=False))]
    for nm, kw in variants:
        try:
            sens[nm] = e_sensitivity(unit_id, data, design, seed, mode, B_SENS, kw, ridx, nm)
        except Exception as e:  # Sensitivitaeten duerfen den Hauptlauf nicht stoppen
            sens[nm] = dict(status=f"Fehler: {type(e).__name__}: {e}")
    res["sensitivitaeten"] = sens
    # --- Geschlecht (K2): getrennte Kennzahlen und Differenz Frauen - Maenner (explorativ)
    res["geschlecht"] = e_by_sex(unit_id, data, design, seed, mode, setup, ridx)
    res["dauer_s"] = round(time.time() - t0)
    save_unit(outdir, mode, unit_id, res)
    return res


def coverage_e(unit_id, data, setup):
    """Abdeckung: Regelabstinenz (NULL) am Index-EKG wird berichtet, nie stillschweigend als negativ gewertet. Anteil 'TTE vorhanden,
    Referenz nicht berechenbar' (Spezifikation 5.1 Punkt 1)."""
    r, f = data.rules, data.refs
    out = dict(n_index=int(data.n), n_population=int(setup["mask"].sum()), n_basis=int(setup["mask_base"].sum()),
               n_regel_NULL_in_basis=int((setup["mask_base"] & ~setup["mask"]).sum()))
    if "echo_30d" in f.columns and unit_id in ("R02", "R03", "R04", "R06", "R07"):
        has_tte = (f["echo_30d"].values == 1) & setup["mask"]
        out["n_tte_30d_in_population"] = int(has_tte.sum())
        out["n_tte_ohne_berechenbare_referenz"] = int((has_tte & ~setup["V"]).sum())
    if unit_id == "R14":
        out.update(n_lvh_positiv=int(np.nansum(r["lvh_volt"].values == 1)), n_lvh_unbekannt=int(np.isnan(r["lvh_volt"].values).sum()),
                   n_ohne_frueheren_aufenthalt=int((f["n_prior_dx"].values == 0).sum()), n_akuter_infarkt_im_index=int((f["idx_ACUTE_MI"].values == 1).sum()),
                   n_nur_pci_cabg_ohne_infarktcode=int((((f["prior_OLD_MI"].values + f["prior_ACUTE_MI"].values) == 0) & (f["prior_pci_cabg"].values > 0)).sum()))
    return out


def variant_setup(unit_id, data, kw):
    """Setup einer Sensitivitaet (v3.1: gemeinsam fuer die gepoolte und die geschlechtsgetrennte Rechnung, Sol-Punkt 5)."""
    win = kw.get("win", "30d")
    setup = e_setup(unit_id, data, win=win)
    if kw.get("Tcols"):
        setup["Tprim"] = {k: data.rules[c].values for k, c in kw["Tcols"].items()}
        m2 = setup["mask_base"].copy()
        for v in setup["Tprim"].values():
            m2 &= np.isfinite(v)
        if unit_id == "R14":
            m2 &= setup["V"]
        setup["mask"] = m2
        setup["conc_keys"] = []
    if kw.get("peri_leer"):
        setup["D"] = data.refs[f"periL_{win}"].values
        setup["V"] = np.isfinite(setup["D"])
    if kw.get("ref6") == "icd":
        setup["D"] = data.refs["lab_peri_icd"].values
        setup["V"] = np.isfinite(setup["D"])
    if kw.get("ref7") == "code_allein":
        setup["D"] = np.where(data.refs["idx_AMYLOID"].values == 1, 1.0, np.nan)
        setup["V"] = np.isfinite(setup["D"])
    if kw.get("ref14"):
        setup["D"] = r14_reference(data.refs, index_codes=(kw["ref14"] == "index_oldmi"), acute_pos=(kw["ref14"] == "acute_pos"))
        setup["V"] = np.isfinite(setup["D"])
        setup["mask"] = np.isfinite(setup["Tprim"]["s"]) & np.isfinite(setup["Tconc"]["s"]) & setup["V"]  # Maske neu bilden (Sol-Fehler 2)
        setup["mask_base"] = setup["V"].copy()
    if kw.get("excl_lafb_wpw"):
        setup["mask"] = setup["mask"] & (data.rules["lafb"].values == 0) & (data.rules["wpw"].values == 0)
    if kw.get("weighted") is not None:
        setup["weighted"] = bool(kw["weighted"])
    return setup


def e_sensitivity(unit_id, data, design, seed, mode, B, kw, ridx, name):
    setup = variant_setup(unit_id, data, kw)
    u = build_ipw(setup, data, design, na_as=kw.get("na_as"), do_kipp=False)
    rng = seed_stream(seed, ridx, {"echo_7d": "sens_7d", "echo_24h_ungewichtet": "sens_24h"}.get(name, "secondary"), mode)
    point, reps = run_bootstrap(u, rng, B, every=1000, tag=f"{unit_id}-{name}")
    if point is None:
        return dict(status="nicht schaetzbar: Verifikationsmodell nicht konvergiert")
    res = dict(status="ok", B=B, bootstrap=reps["_meta"], n_pop=int(u.n), n_verified=int(u.V.sum()), cases=int(np.nansum(u.D[u.V > 0])),
               metrics=summary_metrics(point, reps))
    if kw.get("k2"):  # K2: feste Schwelle, geschlechtsgetrennt gerechnet (Sol-Punkt 5)
        res["geschlecht"] = e_by_sex(unit_id, data, design, seed, mode, setup, ridx)
    return res


def e_by_sex(unit_id, data, design, seed, mode, setup, ridx):
    out = {}
    reps_by = {}
    for sx, name in ((1.0, "F"), (0.0, "M")):
        keep = data.cov["female"].values == sx
        try:
            u = build_ipw(setup, data, design, keep=keep, do_kipp=False)
            drop_constant_cols(u)
            rng = seed_stream(seed, ridx, "stratum", mode)
            rng = np.random.default_rng(rng.integers(0, 2**31) + (0 if name == "F" else 7919))
            point, reps = run_bootstrap(u, rng, B_SENS, every=1000, tag=f"{unit_id}-{name}")
            if point is None:
                out[name] = dict(status="nicht schaetzbar: Verifikationsmodell nicht konvergiert")
                continue
            out[name] = dict(n_pop=int(u.n), n_verified=int(u.V.sum()), cases=int(np.nansum(u.D[u.V > 0])), bootstrap=reps["_meta"],
                             metrics=summary_metrics(point, reps))
            reps_by[name] = reps
        except Exception as e:
            out[name] = dict(status=f"Fehler: {type(e).__name__}: {e}")
    if "F" in reps_by and "M" in reps_by:
        diff = {}
        for t in reps_by["F"]:
            if t == "_meta":
                continue
            for m in ("sens", "spec"):
                a, b = reps_by["F"][t].get(m), reps_by["M"][t].get(m)
                if a is not None and b is not None:
                    k = min(len(a), len(b))
                    d = np.asarray(a)[:k] - np.asarray(b)[:k]
                    itv = interval(d, 0.025, B_SENS)
                    diff[f"{t}_{m}"] = dict(diff_est=float(np.nanmedian(d)), lo95=itv["lo95"], hi95=itv["hi95"])
        out["interaktion_F_minus_M"] = diff
    return out


# ------------------------------------------------------------------ R20 (Familie A)
def prop_boot(x, rng, B):
    """Anteil mit Bootstrap; x in {0,1}."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), np.full(B, np.nan)
    est = float(x.mean())
    reps = np.empty(B)
    for b in range(B):
        c = bootstrap_counts(rng, n)
        reps[b] = float((c * x).sum() / c.sum())
    return est, reps


def cp_bounds(k, n, alpha):
    """Einseitige exakte (Clopper-Pearson) Schranken zum Niveau alpha; Randfall-Intervall bei 0 oder n Ereignissen."""
    from scipy.stats import beta as _beta
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1.0 - alpha, k + 1, n - k))
    return lo, hi


def prop_interval(x, reps, alpha_c, B):
    """Intervall eines Anteils: Bootstrap-Perzentile; bei 0 oder n Ereignissen (Perzentil-Bootstrap degeneriert) exakte Schranken."""
    n = len(x)
    k = int(np.nansum(x))
    itv = interval(reps, alpha_c, B)
    if n > 0 and (k == 0 or k == n):
        lo, hi = cp_bounds(k, n, alpha_c)
        l95, h95 = cp_bounds(k, n, 0.025)
        itv.update(lo=lo, hi=hi, lo95=l95, hi95=h95, randfall="Clopper-Pearson (0 oder alle Ereignisse)", keine_inferenz=False)
    return itv


def run_r20(data: Data, seed: int, mode: str, B: int, outdir: Path):
    r, f = data.rules, data.refs
    res = {}
    ac = G.alpha_component("A")
    spec = {
        "R20_A1": dict(prim=r["normal"].values, conc=r["normal_add"].values, claim="P(Infarkt | normal) >= 0,01", theta=THETA["R20_A1"]),
        "R20_A2": dict(prim=r["normal"].values, conc=np.fmax(r["normal_add"].values, r["no_stt"].values),
                       claim="Anteil normaler EKG unter NSTE-Infarkt >= 1/3", theta=THETA["R20_A2"]),
    }
    mi = f["lab_mi_prob"].values
    nstemi = f["idx_NSTEMI"].values == 1

    def pick(unit_id, Tcol, ref):
        if unit_id == "R20_A1":
            m = (Tcol == 1) & np.isfinite(ref)
            return ref[m]
        m = (ref == 1) & nstemi & np.isfinite(Tcol)
        return Tcol[m]

    def one(unit_id, Tcol, ref, stream="primary"):
        x = pick(unit_id, Tcol, ref)
        rng = seed_stream(seed, RULE_INDEX["R20"], stream, mode)
        est, reps = prop_boot(x, rng, B)
        itv = prop_interval(x, reps, ac, B)
        cls = class_from("ge", itv["lo"], itv["hi"], spec[unit_id]["theta"])
        mc = False
        if not itv.get("randfall") and not np.isnan(itv.get("lo_h1", float("nan"))) and not np.isnan(itv.get("lo_h2", float("nan"))):
            c1 = class_from("ge", itv["lo_h1"], itv["hi_h1"], spec[unit_id]["theta"])
            c2 = class_from("ge", itv["lo_h2"], itv["hi_h2"], spec[unit_id]["theta"])
            mc = bool(not (c1 == c2 == cls))
        return dict(n=int(len(x)), events=int(np.nansum(x)), estimate=est, alpha_c=ac, klasse=cls, monte_carlo_instabil=mc, **itv)

    for uid in ("R20_A1", "R20_A2"):
        p = one(uid, spec[uid]["prim"], mi, "primary")
        c = one(uid, spec[uid]["conc"], mi, "concord")
        hints = [PROXY_HINT[uid]]
        final = p["klasse"]
        if p["klasse"] != c["klasse"]:
            final = G.UNENTSCHIEDEN
            hints.append("operationalisierungsabhaengig")
        if p.get("keine_inferenz"):
            hints.append("keine Inferenz")
        if p.get("randfall"):
            hints.append("Randfallintervall " + p["randfall"])
        if p.get("monte_carlo_instabil"):
            hints.append("Monte-Carlo-instabil (Klasse wechselt zwischen den Haelften der Wiederholungen)")
        sec = {}
        if uid == "R20_A1":
            sec["referenz_troponin_positiv"] = one(uid, spec[uid]["prim"], f["lab_trop_pos"].values, "secondary")
        fem = data.cov["female"].values == 1
        gsx, reps_g = {}, {}
        for nmx, fm in (("F", fem), ("M", ~fem)):
            if uid == "R20_A1":
                mm = (spec[uid]["prim"] == 1) & np.isfinite(mi) & fm
                xx = mi[mm]
            else:
                mm = (mi == 1) & nstemi & np.isfinite(spec[uid]["prim"]) & fm
                xx = spec[uid]["prim"][mm]
            rng_g = np.random.default_rng(seed_stream(seed, RULE_INDEX["R20"], "stratum", mode).integers(0, 2**31) + (0 if nmx == "F" else 7919))
            est_s, reps_s = prop_boot(xx, rng_g, B_SENS)
            reps_g[nmx] = reps_s
            gsx[nmx] = dict(n=int(len(xx)), events=int(np.nansum(xx)), estimate=est_s, **prop_interval(xx, reps_s, 0.025, B_SENS))
        dd = reps_g["F"] - reps_g["M"]
        di = interval(dd, 0.025, B_SENS)
        gsx["interaktion_F_minus_M"] = dict(diff_est=float(gsx["F"]["estimate"] - gsx["M"]["estimate"]) if (gsx["F"]["n"] and gsx["M"]["n"]) else float("nan"),
                                            lo95=di["lo95"], hi95=di["hi95"])
        res[uid] = dict(unit=uid, mode=mode, status="ok", claim=spec[uid]["claim"], theta=spec[uid]["theta"], primaer=p, konkordanz=c,
                        geschlecht=gsx,
                        final=dict(klasse=final, primaer=p["klasse"], hinweise=hints), sekundaer=sec, family="A",
                        kohorte=dict(n_ed_brustschmerz_troponin=int(data.n), n_normal=int(np.nansum(r["normal"].values)),
                                     n_normal_mit_zusatz=int(np.nansum(r["normal_add"].values))))
        save_unit(outdir, mode, uid, res[uid])
    return res


def save_unit(outdir: Path, mode: str, unit_id: str, res: dict):
    d = Path(outdir) / mode / "units"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{unit_id}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ================================================================== Familie P (R09, R18) und Familie S (R08, R19)
from scipy import stats as _st  # noqa: E402

COX_MED = ["med_beta_blocker", "med_calcium_channel_blocker", "med_digoxin", "med_antiarrhythmic_class_i_iii", "med_diuretic", "med_assessed"]


def z_of(alpha_c: float) -> float:
    return float(_st.norm.ppf(1.0 - alpha_c))


def wald_hr(beta, se, alpha_c):
    z = z_of(alpha_c)
    z95 = float(_st.norm.ppf(0.975))
    return dict(hr=float(np.exp(beta)), lo=float(np.exp(beta - z * se)), hi=float(np.exp(beta + z * se)),
                lo95=float(np.exp(beta - z95 * se)), hi95=float(np.exp(beta + z95 * se)), alpha_c=alpha_c, se_log=float(se))


def cox_matrix(data: Data, design: Design, idx):
    """Adjustierung Familie P (5.3): Alter (Spline), Geschlecht, Aufnahmeart, Vorgeschichte, AV-Knoten-wirksame Medikation,
    Diuretika, Herzfrequenz. Nicht adjustiert: QRS-, PR- und Blockbefunde (Exposition bzw. Weg)."""
    cov = data.cov.iloc[idx]
    X = design.X(cov)
    names = [f"age{i}" for i in range(3)] + ["female", "adm_icu", "adm_ed", "adm_inp"] + Design.HX + ["mcat_b", "mcat_a", "hr", "hr_missing"]
    keep = [i for i, nm in enumerate(names) if nm not in ("mcat_b", "mcat_a")]
    Xc = X[:, keep]
    nm = [names[i] for i in keep]
    med = cov[COX_MED].values.astype(float)
    return np.hstack([Xc, med]), nm + COX_MED


def _cox_prep(T, E, X):
    o = np.argsort(T, kind="mergesort")
    t, e, x = T[o], E[o], X[o]
    ev_times = np.unique(t[e > 0])
    first = np.searchsorted(t, ev_times, side="left")  # erster Index mit t >= u
    d_u = np.bincount(np.searchsorted(ev_times, t[e > 0]), minlength=len(ev_times)).astype(float)
    pos = np.searchsorted(ev_times, t, side="right")  # Anzahl Ereigniszeiten u <= t_i
    own = np.clip(np.searchsorted(ev_times, t, side="left"), 0, len(ev_times) - 1)
    return t, e, x, first, d_u, pos, own


def _cox_pieces(beta, x, e, first, d_u, pos):
    """Breslow-Teillikelihood: Wert, Gradient, Hessesche Matrix (negativ) und Zwischengroessen."""
    w = np.exp(np.clip(x @ beta, -40, 40))
    rev0 = np.cumsum(w[::-1])[::-1]
    rev1 = np.cumsum((x * w[:, None])[::-1], axis=0)[::-1]
    S0, S1 = rev0[first], rev1[first]
    ll = float(np.sum(e * (x @ beta)) - np.sum(d_u * np.log(S0)))
    mu = S1 / S0[:, None]
    grad = x[e > 0].sum(axis=0) - (d_u[:, None] * mu).sum(axis=0)
    A = d_u / S0
    CA = np.concatenate([[0.0], np.cumsum(A)])[pos]
    H = (x * (w * CA)[:, None]).T @ x - (mu * d_u[:, None]).T @ mu
    return ll, grad, H, w, S0, S1, A


def cox_breslow_robust(T, E, X, ridge=0.0, pen_cols=None):
    """Cox-Modell (Breslow-Naeherung fuer Bindungen), Newton-Raphson mit Schrittweitenhalbierung, optional kleine L2-Strafe
    (Rueckfall bei Separation), und robuster Sandwich-Varianz (Lin und Wei) fuer unabhaengige Personen. Rueckgabe beta,
    robuste Kovarianz, modellbasierte Kovarianz, Info."""
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    X = np.asarray(X, dtype=float)
    t, e, x, first, d_u, pos, own = _cox_prep(T, E, X)
    p = X.shape[1]
    pen = np.zeros(p)
    if ridge > 0:
        pen[:] = ridge
        if pen_cols is not None:
            pen[:] = 0.0
            pen[pen_cols] = ridge
    beta = np.zeros(p)
    ll, grad, H, *_ = _cox_pieces(beta, x, e, first, d_u, pos)
    obj = ll - 0.5 * np.sum(pen * beta * beta)
    conv = False
    for _ in range(50):
        g = grad - pen * beta
        step = np.linalg.solve(H + np.diag(pen) + 1e-10 * np.eye(p), g)
        tt = 1.0
        improved = False
        for _h in range(12):
            nb = beta + tt * step
            try:
                nll, ngrad, nH, *_ = _cox_pieces(nb, x, e, first, d_u, pos)
            except FloatingPointError:
                nll = -np.inf
            nobj = nll - 0.5 * np.sum(pen * nb * nb)
            if np.isfinite(nobj) and nobj >= obj - 1e-9:
                improved = True
                break
            tt *= 0.5
        if not improved:
            break
        beta, ll, grad, H, obj = nb, nll, ngrad, nH, nobj
        if np.max(np.abs(tt * step)) < 1e-8:
            conv = True
            break
    ll, grad, H, w, S0, S1, A = _cox_pieces(beta, x, e, first, d_u, pos)
    V = np.linalg.inv(H + np.diag(pen) + 1e-12 * np.eye(p))
    B = (d_u / S0 ** 2)[:, None] * S1
    CB = np.vstack([np.zeros((1, p)), np.cumsum(B, axis=0)])
    CA = np.concatenate([[0.0], np.cumsum(A)])
    term = w[:, None] * (x * CA[pos][:, None] - CB[pos])
    score = e[:, None] * (x - S1[own] / S0[own][:, None]) - term
    Vrob = V @ (score.T @ score) @ V
    return beta, Vrob, V, dict(konvergiert=bool(conv), ridge=float(ridge), max_abs_beta=float(np.max(np.abs(beta))))


def cox_expo(T, E, expo, X, names):
    """Cox-Modell mit robuster Varianz; liefert beta und se der Exposition (Spalte 0). Bei Separation oder Nichtkonvergenz
    (|beta| > 15) Rueckfall auf eine kleine L2-Strafe (0,01) auf den Kovariaten ausser der Exposition."""
    Z = np.column_stack([np.asarray(expo, dtype=float), X])
    keep = np.r_[True, Z[:, 1:].std(axis=0) > 1e-9]
    Z = Z[:, keep]
    Tt = np.asarray(T, dtype=float) / 365.25
    for ridge in (0.0, 0.01):
        try:
            beta, Vrob, V, info = cox_breslow_robust(Tt, E, Z, ridge=ridge, pen_cols=list(range(1, Z.shape[1])) if ridge else None)
            se = float(np.sqrt(Vrob[0, 0]))
            ok = info["konvergiert"] and np.isfinite(se) and se > 0 and info["max_abs_beta"] < 15
            if ok:
                info.update(n_cov=int(keep.sum() - 1), se_modellbasiert=float(np.sqrt(V[0, 0])), ties="breslow")
                return float(beta[0]), se, info
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
    return float("nan"), float("nan"), dict(error=locals().get("last", "nicht konvergiert oder Varianz nicht positiv"))


def p_data(data, mask, T, outcome, t_lo=0.0, t_hi=365.0):
    """Gemeinsame Auswahl und Zeit fuer Cox-Analysen und absolute Risiken (v3.1): dieselben Ausschluesse. Risikozeit = Minimum aus
    Kontaktende, Tod im Fenster und 365 Tagen; Personen mit Risikozeit 0 werden gezaehlt (n_risikozeit_null) und ausgeschlossen."""
    f, cov = data.refs, data.cov
    if outcome == "death":
        base = mask & np.isfinite(T) & np.isfinite(f["death_t"].values)
        t0_, e0_ = f["death_t"].values, f["death_ev"].values
    else:
        base = mask & np.isfinite(T) & (f["af_prev"].values == 0) & (data.rules["af_any"].values == 0) & (cov["hx_af"].values == 0)
        t0_, e0_ = f["af_t"].values, f["af_ev"].values
    n_zero = int((base & (t0_ <= 0)).sum()) if t_lo == 0.0 else 0
    ok = base & (t0_ > t_lo)
    t = np.where(ok, np.minimum(t0_, t_hi) - t_lo, np.nan)
    e = np.where(ok, np.nan_to_num(e0_, nan=0.0) * (t0_ <= t_hi), 0.0)
    return ok, t, e, n_zero


def p_outcome(data, design, mask, T, outcome, alpha_c, t_lo=0.0, t_hi=365.0, verdict=False):
    """Eine Cox-Analyse: mask = Population, T = Exposition (0/1), outcome in {death, af}. t_lo/t_hi: Landmark-Fenster in Tagen.
    verdict=True nur fuer die bindenden Hypothesen P1 bis P4 (Klasse und Hinweis); Sekundaeranalysen tragen keine Urteilsklasse."""
    ok, t, e, n_zero = p_data(data, mask, T, outcome, t_lo, t_hi)
    idx = np.where(ok)[0]
    if len(idx) < 50 or np.nansum(e[idx]) < 10 or T[idx].sum() < 10:
        return dict(status="nicht schaetzbar", n=int(len(idx)), events=int(np.nansum(e[idx])) if len(idx) else 0,
                    n_exposed=int(np.nansum(T[idx])) if len(idx) else 0, n_risikozeit_null=n_zero)
    X, nm = cox_matrix(data, design, idx)
    beta, se, info = cox_expo(t[idx], e[idx], T[idx], X, nm)
    if not np.isfinite(beta):
        return dict(status="Modellfehler", info=info, n=int(len(idx)), n_risikozeit_null=n_zero)
    out = wald_hr(beta, se, alpha_c)
    out.update(status="ok", n=int(len(idx)), events=int(np.nansum(e[idx])), n_exposed=int(T[idx].sum()),
               events_exposed=int(np.nansum(e[idx][T[idx] == 1])), fit=info, n_risikozeit_null=n_zero)
    if verdict:
        cls, hint = G.classify_hr_nonharm(out["lo"], out["hi"], HR_MARGIN)
        out["klasse"], out["hinweis"] = cls, hint
    return out


def km_risk(t, e, days=365.0):
    """Kaplan-Meier-Risiko 1 - S(days) (absolutes Risiko; beim VHF cause-specific, ueberschaetzt das kumulative Risiko)."""
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)
    o = np.argsort(t, kind="mergesort")
    t, e = t[o], e[o]
    n = len(t)
    if n == 0:
        return float("nan")
    ut = np.unique(t[e > 0])
    surv = 1.0
    for u in ut:
        if u > days:
            break
        lo = np.searchsorted(t, u, side="left")
        at_risk = n - lo
        d = int(np.sum((t == u) & (e > 0)))
        surv *= 1.0 - d / at_risk
    return float(1.0 - surv)


def absolute_risks(data, mask, T, outcome):
    """Absolute 365-Tage-Risiken auf derselben Auswahl und Zeit wie die Cox-Analyse (v3.1)."""
    ok, t, e, n_zero = p_data(data, mask, T, outcome)
    return {"exponiert": km_risk(t[ok & (T == 1)], e[ok & (T == 1)]), "nicht_exponiert": km_risk(t[ok & (T == 0)], e[ok & (T == 0)]),
            "n_risikozeit_null": n_zero}


def p_setup(unit_id, data: Data):
    r, cov = data.rules, data.cov
    if unit_id == "R09":
        mask = np.isfinite(r["R09_m"].values)
        Tp = r["R09_m"].values
        Tc = np.where(mask, r["R09_a155"].values, np.nan)
    else:
        mask = np.isfinite(r["R18_m"].values) & (cov["med_digoxin"].values == 0) & (cov["med_antiarrhythmic_class_i_iii"].values == 0)
        Tp = r["R18_m"].values
        Tc = np.where(mask, r["R18_own"].values, np.nan)
    return dict(mask=mask, Tprim=Tp, Tconc=Tc)


def run_p(unit_id, data: Data, design: Design, seed: int, mode: str, outdir: Path):
    """Familie P. v3 (Nachtrag 1): Primaer- und Konkordanzfassung auf DERSELBEN Population (gemeinsame Auswertbarkeit); ist die
    Konkordanzvariante nicht berechenbar, bleibt das Endurteil 'unentschieden'."""
    t0 = time.time()
    st = p_setup(unit_id, data)
    cov, r, f = data.cov, data.rules, data.refs
    ac = G.alpha_component("P")
    claim = cov["claim_pop"].values == 1  # ohne DOKUMENTIERTE Herz- oder Lungenerkrankung vor dem Index
    conc_avail = bool(np.isfinite(st["Tconc"]).any())
    pop = st["mask"] & claim
    pop_both = pop & np.isfinite(st["Tconc"]) if conc_avail else pop
    res = dict(unit=unit_id, mode=mode, family="P", status="ok", alpha_c=ac, population="Krankenhauspatienten ohne dokumentierte Vorerkrankung")
    out = {}
    for variant, T in (("primaer", st["Tprim"]), ("konkordanz", st["Tconc"])):
        out[variant] = {oc: p_outcome(data, design, pop_both, T, oc, ac, verdict=True) for oc in ("death", "af")}
    comp_cls = [out["primaer"][oc].get("klasse", G.UNENTSCHIEDEN) for oc in ("death", "af")]
    comp_conc = [out["konkordanz"][oc].get("klasse", G.UNENTSCHIEDEN) for oc in ("death", "af")]
    uc, vc = G.combine(comp_cls), G.combine(comp_conc)
    conc_ok = conc_avail and all(out["konkordanz"][oc].get("status") == "ok" for oc in ("death", "af"))
    hints = ["Population ohne dokumentierte Vorerkrankung (ohne frueheren Aufenthalt gilt als ohne Dokumentation)",
             "HR < 1,20 bedeutet weder Risikogleichheit noch klinische Harmlosigkeit",
             "neu dokumentiertes VHF: Erfassungsbias (VHF nur bei erneutem Kontakt dokumentiert, Richtung: Nachkontrolle erhoeht die Erfassung)"]
    final = uc
    if not conc_ok:
        final = G.UNENTSCHIEDEN
        vc = None
        hints.append("Konkordanzvariante nicht berechenbar: Endurteil unentschieden (Spezifikation 5, Konkordanzregel)")
    elif uc != vc:
        final = G.UNENTSCHIEDEN
        hints.append("operationalisierungsabhaengig")
    for oc in ("death", "af"):
        h = out["primaer"][oc].get("hinweis")
        if h:
            hints.append(f"{oc}: {h}")
        if out["primaer"][oc].get("status") != "ok":
            hints.append(f"{oc}: keine Inferenz ({out['primaer'][oc].get('status')})")
    res["primaer"], res["konkordanz"] = out["primaer"], out["konkordanz"]
    res["final"] = dict(klasse=final, primaer=uc, konkordanz=vc, hinweise=hints)
    # Sekundaeranalysen (explorativ, ohne Urteilssprache und ohne Familienzaehlung)
    sec = {}
    sec["primaer_volle_population_ohne_konkordanzbeschraenkung"] = {oc: p_outcome(data, design, pop, st["Tprim"], oc, ac) for oc in ("death", "af")}
    sec["gesamtkohorte"] = {oc: p_outcome(data, design, st["mask"], st["Tprim"], oc, ac) for oc in ("death", "af")}
    sec["nur_fruehere_aufenthalte"] = {oc: p_outcome(data, design, pop_both & (cov["has_prior"].values == 1), st["Tprim"], oc, ac) for oc in ("death", "af")}
    sec["af_nur_mit_folgekontakt"] = {"af": p_outcome(data, design, pop_both & (f["followup_contact"].values == 1), st["Tprim"], "af", ac)}
    sec["landmark_0_bis_90_tage"] = {oc: p_outcome(data, design, pop_both, st["Tprim"], oc, ac, 0.0, 90.0) for oc in ("death", "af")}
    sec["landmark_90_bis_365_tage"] = {oc: p_outcome(data, design, pop_both, st["Tprim"], oc, ac, 90.0, 365.0) for oc in ("death", "af")}
    sec["absolutes_risiko_365_tage"] = {oc: absolute_risks(data, pop_both, st["Tprim"], oc) for oc in ("death", "af")}
    if unit_id == "R18":
        qrs = r["qrs_machine"].values
        pr = r["pr_machine"].values
        sec["ohne_PR_unter_120"] = {oc: p_outcome(data, design, pop_both & ~(pr < 120), st["Tprim"], oc, ac) for oc in ("death", "af")}
        sec["QRS_unter_120_in_beiden_Gruppen"] = {oc: p_outcome(data, design, pop_both & (qrs < 120), st["Tprim"], oc, ac) for oc in ("death", "af")}
    sec["or_sekundaer_endpunkte"] = secondary_or(data, design, st, claim, unit_id)
    fem = cov["female"].values == 1
    gs = {}
    for oc in ("death", "af"):
        pf = p_outcome(data, design, pop_both & fem, st["Tprim"], oc, ac)
        pm = p_outcome(data, design, pop_both & ~fem, st["Tprim"], oc, ac)
        gs[oc] = {"F": pf, "M": pm}
        if pf.get("status") == "ok" and pm.get("status") == "ok":
            d = math.log(pf["hr"]) - math.log(pm["hr"])
            se = math.sqrt(pf["se_log"] ** 2 + pm["se_log"] ** 2)
            gs[oc]["interaktion_HR_F_durch_HR_M"] = dict(ratio=float(math.exp(d)), lo95=float(math.exp(d - 1.96 * se)), hi95=float(math.exp(d + 1.96 * se)))
    sec["geschlecht"] = gs
    res["sekundaer"] = sec
    res["kohorte"] = dict(n_index=int(data.n), n_regel_auswertbar=int(np.isfinite(st["Tprim"]).sum()),
                          n_claim_pop=int(pop.sum()), n_claim_pop_gemeinsam=int(pop_both.sum()),
                          n_exponiert_claim=int(np.nansum(st["Tprim"][pop_both])))
    res["dauer_s"] = round(time.time() - t0)
    save_unit(outdir, mode, unit_id, res)
    return res


def weighted_or(data, design, mask, T, y):
    """Sekundaer (Spezifikation 6.3: mit Gewichtung): Verifikationsmodell P(Referenz bestimmbar | T, X), stabilisierte gekappte
    Gewichte, gewichtete logistische Regression mit Sandwich-Varianz (Gewichtsschaetzung bleibt unberuecksichtigt, explorativ)."""
    ok = mask & np.isfinite(T)
    idx = np.where(ok)[0]
    V = np.isfinite(y[idx]).astype(int)
    if V.sum() < 100 or np.nansum(y[idx]) < 10:  # v3.1: wieder 100 vollstaendige Faelle (wie v1/v2)
        return dict(status="nicht schaetzbar", n=int(len(idx)))
    cov = data.cov.iloc[idx]
    Xb = design.X(cov)
    Z = design_z = np.hstack([np.ones((len(idx), 1)), Xb, T[idx][:, None]])
    bv, info = fit_logit_safe(Z, V)
    if info.get("failed"):
        return dict(status="Verifikationsmodell nicht konvergiert", n=int(len(idx)))
    w = stabilised_weights(expit(Z @ bv), V, np.ones(len(idx)))
    v = V == 1
    Xc, nm = cox_matrix(data, design, idx[v])
    keepc = Xc.std(axis=0) > 1e-9
    Zo = np.column_stack([np.ones(v.sum()), T[idx][v], Xc[:, keepc]])
    yo = y[idx][v]
    wv = w[v]
    bo, ok2 = logit_newton(Zo, yo, wv)
    if not ok2:
        return dict(status="Ergebnismodell nicht konvergiert", n=int(len(idx)))
    mu = expit(Zo @ bo)
    H = np.linalg.inv((Zo * (wv * mu * (1 - mu))[:, None]).T @ Zo)
    sc = Zo * (wv * (yo - mu))[:, None]
    Vr = H @ (sc.T @ sc) @ H
    se = float(np.sqrt(Vr[1, 1]))
    return dict(status="ok", n=int(len(idx)), n_bestimmbar=int(v.sum()), events=int(np.nansum(yo)), OR=float(np.exp(bo[1])),
                lo95=float(np.exp(bo[1] - 1.96 * se)), hi95=float(np.exp(bo[1] + 1.96 * se)), gewichtet=True)


def secondary_or(data, design, st, claim, unit_id):
    """Sekundaere Endpunkte (Pipeline-Labels): Leitungsfortschreitung (R09 und R18), Echo-RV-Dilatation und Pulmonalhypertonie (R09)."""
    f = data.refs
    labs = ("cond1y", "rv_dil", "ph_echo") if unit_id == "R09" else ("cond1y",)
    out = {}
    for pop_name, pm in (("ohne_dokumentierte_vorerkrankung", st["mask"] & claim), ("gesamt", st["mask"])):
        for lab in labs:
            out[f"{lab}_{pop_name}"] = weighted_or(data, design, pm, st["Tprim"], f["lab_" + lab].values)
    return out


# ------------------------------------------------------------------ R08 (Familie S), v3: Arrhythmie und Tod getrennt
def rr_counts(T, D, mask):
    ok = mask & np.isfinite(T) & np.isfinite(D)
    a = int(np.sum(ok & (T == 1) & (D == 1)))
    n1 = int(np.sum(ok & (T == 1)))
    c = int(np.sum(ok & (T == 0) & (D == 1)))
    n0 = int(np.sum(ok & (T == 0)))
    return a, n1, c, n0


def rr_wald(a, n1, c, n0, alpha_c):
    if min(a, c) == 0 or n1 == 0 or n0 == 0:
        return dict(status="nicht schaetzbar", a=a, n1=n1, c=c, n0=n0)
    lr = math.log((a / n1) / (c / n0))
    se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    out = wald_hr(lr, se, alpha_c)
    out.update(rr=out.pop("hr"), a=a, n1=n1, c=c, n0=n0, status="ok")
    return out


def rr_adjusted(data, design, mask, T, D, extra_cols=None, minimal=True):
    """Poisson-Regression mit robuster Varianz (sekundaer, explorativ). minimal: Alter, Geschlecht, Aufnahmeart, Herzfrequenz."""
    import statsmodels.api as sm
    cov = data.cov
    ok = mask & np.isfinite(T) & np.isfinite(D)
    idx = np.where(ok)[0]
    if len(idx) < 100 or np.nansum(D[idx]) < 10:
        return dict(status="nicht schaetzbar", n=int(len(idx)))
    X = design.X(cov.iloc[idx])
    sel = list(range(0, 7)) + [19, 20]
    cols = [X[:, sel]]
    if not minimal:
        cols.append(X[:, [7]])
        add = []
        for c in ("med_qt_prolonging_drug", "lab_potassium", "lab_magnesium", "lab_total_calcium"):
            v = cov[c].values[idx].astype(float)
            miss = (~np.isfinite(v)).astype(float)
            med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
            add += [np.where(np.isfinite(v), v, med), miss]
        cols.append(np.column_stack(add))
    Z = np.column_stack([np.ones(len(idx)), T[idx]] + cols)
    keepc = np.r_[True, True, Z[:, 2:].std(axis=0) > 1e-9]
    Z = Z[:, keepc]
    try:
        fit = sm.GLM(D[idx], Z, family=sm.families.Poisson()).fit(cov_type="HC0")
        b, se = float(fit.params[1]), float(fit.bse[1])
        return dict(status="ok", n=int(len(idx)), rr=float(np.exp(b)), lo95=float(np.exp(b - 1.96 * se)), hi95=float(np.exp(b + 1.96 * se)))
    except Exception as e:
        return dict(status=f"Fehler: {type(e).__name__}: {e}")


def run_r08(data: Data, design: Design, seed: int, mode: str, outdir: Path):
    """R08 v3: zwei getrennte Einheiten. 'R08_VTVF' (VT/VF oder Herzstillstand im Aufenthalt) ist zeitlich nicht geordnet und wird nur als
    gleichzeitige Assoziation beurteilt; 'R08_Tod' (30-Tage-Tod) ist die prognostische Einheit. Behauptung RR > 1: haelt bei L > 1,
    haelt nicht erst bei U <= 1 (Nachtrag 1)."""
    t0 = time.time()
    r, f, cov = data.rules, data.refs, data.cov
    ac = G.alpha_component("S")
    vtvf, arr = f["lab_vtvf"].values, f["lab_arrest"].values
    DA = np.where((vtvf == 1) | (arr == 1), 1.0, np.where((vtvf == 0) & (arr == 0), 0.0, np.nan))
    DB = f["lab_mort30"].values
    outc = {"R08_VTVF": ("VTVF_Herzstillstand", DA), "R08_Tod": ("Tod_30_Tage", DB)}
    allm = np.ones(data.n, dtype=bool)
    # gemeinsame Population: Bazett und Fridericia auswertbar
    both = allm & np.isfinite(r["R08_baz"].values) & np.isfinite(r["R08_fri"].values)
    Tb, Tf = r["R08_baz"].values, r["R08_fri"].values
    qrs = r["qrs_machine"].values
    fem = cov["female"].values == 1
    results = {}
    for uid, (nm, D) in outc.items():
        w_p = rr_wald(*rr_counts(Tb, D, both), ac)
        w_c = rr_wald(*rr_counts(Tf, D, both), ac)
        for w in (w_p, w_c):
            if w["status"] == "ok":
                w["klasse"], w["hinweis"] = classify_marker3(w["lo"], w["hi"])
                w["faktor2_obere_schranke_unter_2"] = bool(w["hi"] < 2.0)
        cp, cc = w_p.get("klasse", G.UNENTSCHIEDEN), w_c.get("klasse", G.UNENTSCHIEDEN)
        hints = ["Messwegabhaengigkeit sekundaer (PUB-12)"]
        final = cp
        if cp != cc:
            final = G.UNENTSCHIEDEN
            hints.append("operationalisierungsabhaengig")
        if uid == "R08_VTVF":
            hints.append("gleichzeitige Assoziation: Ereigniszeit relativ zum EKG nicht attestierbar, kein prognostisches Urteil")
        if w_p.get("hinweis"):
            hints.append(w_p["hinweis"])
        sec = {}
        sec["linear_Framingham"] = rr_wald(*rr_counts(r["R08_lin"].values, D, both), ac)
        sec["QRS_unter_120"] = rr_wald(*rr_counts(Tb, D, both & (qrs < 120)), ac)
        sec["nur_Sinusrhythmus"] = rr_wald(*rr_counts(Tb, D, both & (r["sinus"].values == 1)), ac)
        sec["volle_population_Bazett_ohne_Fridericia_Beschraenkung"] = rr_wald(*rr_counts(Tb, D, allm), ac)
        if uid == "R08_VTVF":
            sec["Index_EKG_am_ersten_Aufenthaltstag"] = rr_wald(*rr_counts(Tb, D, both & (f["ecg_stay_day"].values == 0)), ac)
        sec["minimal_adjustiert"] = rr_adjusted(data, design, both, Tb, D, minimal=True)
        sec["vollstaendig_adjustiert"] = rr_adjusted(data, design, both, Tb, D, minimal=False)
        ok = both & np.isfinite(r["qtc_cat"].values) & np.isfinite(D)
        if ok.sum() > 100 and np.nansum(D[ok]) > 10:
            import statsmodels.api as sm
            Z = np.column_stack([np.ones(ok.sum()), r["qtc_cat"].values[ok]])
            try:
                fit = sm.GLM(D[ok], Z, family=sm.families.Poisson()).fit(cov_type="HC0")
                b, se = float(fit.params[1]), float(fit.bse[1])
                sec["Dosis_Wirkung_Stufe_0_bis_3_RR_je_Stufe"] = dict(status="ok", rr=float(np.exp(b)), lo95=float(np.exp(b - 1.96 * se)), hi95=float(np.exp(b + 1.96 * se)), n=int(ok.sum()))
            except Exception as e:
                sec["Dosis_Wirkung_Stufe_0_bis_3_RR_je_Stufe"] = dict(status=f"Fehler: {type(e).__name__}")
        a = rr_wald(*rr_counts(Tb, D, both & fem), ac)
        b_ = rr_wald(*rr_counts(Tb, D, both & ~fem), ac)
        gsx = {"F": a, "M": b_}
        if a.get("status") == "ok" and b_.get("status") == "ok":
            d = math.log(a["rr"]) - math.log(b_["rr"])
            se = math.sqrt(a["se_log"] ** 2 + b_["se_log"] ** 2)
            gsx["interaktion_RR_F_durch_RR_M"] = dict(ratio=float(math.exp(d)), lo95=float(math.exp(d - 1.96 * se)), hi95=float(math.exp(d + 1.96 * se)))
        sec["geschlecht"] = gsx
        results[uid] = dict(unit=uid, mode=mode, family="S", status="ok", alpha_c=ac, ergebnis=nm,
                            primaer={nm: w_p}, konkordanz={nm: w_c},
                            final=dict(klasse=final, primaer=cp, konkordanz=cc, hinweise=hints),
                            sekundaer={nm: sec},
                            kohorte=dict(n_index=int(data.n), n_regel_auswertbar=int(both.sum()), n_positiv=int(np.nansum(Tb[both]))),
                            dauer_s=round(time.time() - t0))
        save_unit(outdir, mode, uid, results[uid])
    return results


# ------------------------------------------------------------------ R19 (Familie S): Delta-Aequivalenz der QRS-Schwelle
def r19_inputs(data: Data, design: Design, qcol: str = "qrs_machine", qconc: str | None = "qrs_own", win="30d", extra_mask=None):
    """v3: Primaer (Maschinenwert) und Konkordanz (eigene Dauer) auf derselben Population, mit denselben Gewichten und derselben
    Standardisierungsstichprobe. Ist die Konkordanzgroesse nicht vorhanden (Testpartition), entfaellt sie."""
    r, f, cov = data.rules, data.refs, data.cov
    q = r[qcol].values
    qc = r[qconc].values if (qconc and qconc in r.columns) else None
    conc_avail = qc is not None and bool(np.isfinite(qc).any())
    mask = np.isfinite(q)
    if conc_avail:
        mask &= np.isfinite(qc)
    D = f[f"lvef35_{win}"].values
    if extra_mask is not None:
        mask &= extra_mask
    idx = np.where(mask)[0]
    c = cov.iloc[idx]
    X = design.X(c)
    knots = list(G.QRS_KNOTS)

    def spl(qv):
        b = rcs_basis(qv, knots)
        mu, sd = b.mean(axis=0), b.std(axis=0)
        sd[sd < 1e-9] = 1.0
        return (b - mu) / sd, mu, sd

    qs_s, mu, sd = spl(q[idx])
    fem = c["female"].values
    Zv = np.column_stack([np.ones(len(idx)), X, qs_s, qs_s * fem[:, None]])
    Xo = np.column_stack([X[:, 0:3], X[:, 4:7], c[["hx_hf", "hx_cad", "hx_cm"]].values.astype(float)])
    inp = dict(idx=idx, q=q[idx], D=D[idx], V=np.isfinite(D[idx]).astype(int), Zv=Zv, Xo=Xo, nq=qs_s.shape[1], fem=fem,
               mu=mu, sd=sd, knots=knots, n=len(idx), conc_avail=conc_avail)
    inp["Zo"] = np.column_stack([np.ones(len(idx)), qs_s, Xo])
    if conc_avail:
        qc_s, muc, sdc = spl(qc[idx])
        inp.update(qc=qc[idx], muc=muc, sdc=sdc, Zoc=np.column_stack([np.ones(len(idx)), qc_s, Xo]))
    return inp


def r19_deltas(inp, c, std_idx, warm=None):
    """Delta = (QRS, bei dem das Frauenrisiko das Maennerrisiko bei 150 ms erreicht) - 150 ms. v3.1: EIN Gewichtsmodell (Primaerfassung,
    mit Zaehlungen c) und je Variante (Primaer 'p', Konkordanz 'c') zwei Ergebnismodelle (je Geschlecht) mit denselben Gewichten.
    Rueckgabe (ergebnis, warm): ergebnis[var] = (delta, status, risks); scheitert nur das Ergebnismodell EINER Variante, ist deren
    Eintrag None, die andere Variante bleibt erhalten (v3.2); scheitert das Gewichtsmodell, ist die Rueckgabe (None, None)."""
    sel = c > 0
    V, D = inp["V"], inp["D"]
    bv, info = fit_logit_safe(inp["Zv"][sel], V[sel], c[sel], warm.get("bv") if warm else None)
    if info.get("failed"):
        return None, None
    w = stabilised_weights(expit(inp["Zv"] @ bv), V, c)
    wt = w * c
    variants = [("p", inp["Zo"], inp["knots"], inp["mu"], inp["sd"])]
    if inp.get("conc_avail"):
        variants.append(("c", inp["Zoc"], inp["knots"], inp["muc"], inp["sdc"]))
    i150 = int(np.where(R19_GRID == 150)[0][0])
    out, new_warm = {}, {"bv": bv}
    for key, Zo, knots, mu, sd in variants:
        grid_basis = (rcs_basis(R19_GRID, knots) - mu) / sd
        nq = inp["nq"]
        risks = {}
        for sx, nm in ((1.0, "F"), (0.0, "M")):
            m = sel & (V == 1) & (inp["fem"] == sx) & (wt > 0)
            bo, info_o = fit_logit_safe(Zo[m], D[m], wt[m], warm.get(f"bo_{nm}{key}") if warm else None)
            if info_o.get("failed"):
                risks = None  # nur diese Variante faellt aus (v3.2)
                break
            new_warm[f"bo_{nm}{key}"] = bo
            xs = inp["Xo"][std_idx]
            a = bo[0] + xs @ bo[1 + nq:]
            risks[nm] = expit(a[:, None] + (grid_basis @ bo[1:1 + nq])[None, :]).mean(axis=0)
        if risks is None:
            out[key] = None
            continue
        val, status = G.equal_risk_qrs(R19_GRID, risks["F"], risks["M"][i150])
        out[key] = (float("nan") if val is None else float(val - 150.0), status, risks)
    return out, new_warm


R19_MIN_EVENTS_PER_SEX = 25


def r19_run_one(inp, rng_std, rng, B, tag, min_verified=200):
    """Punktschaetzer und Bootstrap fuer Primaer (und Konkordanz, falls vorhanden) aus denselben Ziehungen mit denselben Gewichten.
    Ausfaelle werden je Variante gezaehlt (v3.2): Fitfehler des Ergebnismodells der Konkordanzvariante verwerfen die Primaerschaetzung
    nicht; scheitert das Gewichtsmodell, fallen beide Varianten dieser Wiederholung aus. Fitfehler und fehlende Schnittpunkte getrennt."""
    ev_sex = {sx: int(np.nansum(inp["D"][(inp["V"] > 0) & (inp["fem"] == v)])) for sx, v in (("F", 1.0), ("M", 0.0))}
    if inp["V"].sum() < min_verified or min(ev_sex.values()) < R19_MIN_EVENTS_PER_SEX:  # Mindestzahlen wie v1/v2; 25 je Geschlecht: Nachtrag 5.3
        return dict(status="nicht schaetzbar", n=inp["n"], ereignisse_je_geschlecht=ev_sex)
    std_idx = rng_std.choice(inp["n"], size=min(5000, inp["n"]), replace=False)
    res0, warm = r19_deltas(inp, np.ones(inp["n"]), std_idx)
    if res0 is None or res0.get("p") is None:
        return dict(status="nicht schaetzbar: Anpassung der Punktschaetzung nicht konvergiert", n=inp["n"])
    keys = [k for k in res0 if res0[k] is not None]
    conc_point_failed = bool(inp.get("conc_avail")) and res0.get("c") is None
    reps = {k: [] for k in keys}
    bad_fit, bad_cross = {k: 0 for k in keys}, {k: 0 for k in keys}
    for b in range(B):
        c = bootstrap_counts(rng, inp["n"])
        r_, _w = r19_deltas(inp, c, std_idx, warm)
        for k in keys:
            if r_ is None or r_.get(k) is None:
                bad_fit[k] += 1
                reps[k].append(float("nan"))
            else:
                d, st, _r = r_[k]
                reps[k].append(d if st == "ok" else float("nan"))
                bad_cross[k] += int(st != "ok")
        if (b + 1) % 500 == 0:
            log(f"  R19 {tag} Bootstrap {b + 1}/{B}")
    mk = lambda k: dict(d0=res0[k][0], st0=res0[k][1], reps=np.asarray(reps[k]), cross=bad_cross[k], fit_fehler=bad_fit[k])  # noqa: E731
    return dict(status="ok", p=dict(mk("p"), risks0=res0["p"][2]),
                c=(mk("c") if "c" in keys else None), c_punktschaetzung_fehlgeschlagen=conc_point_failed,
                bad_fit=bad_fit["p"], n=inp["n"], n_verified=int(inp["V"].sum()), cases=int(np.nansum(inp["D"][inp["V"] > 0])),
                ereignisse_je_geschlecht=ev_sex)


def r19_verdict(delta0, st0, reps, B, ac):
    """Klasse nach TOST-Logik; 'nicht pruefbar' bei nicht eindeutigem Punktschaetzer oder mehr als 10 % undefinierten Wiederholungen
    (Fitfehler und fehlende Schnittpunkte zusammen); Monte-Carlo-Instabilitaet zwischen den Haelften wird gemeldet."""
    itv = interval(reps, ac, B)
    undefined_share = float(np.isnan(np.asarray(reps)).sum()) / max(B, 1)
    if st0 != "ok" or undefined_share > 0.10 or itv.get("keine_inferenz"):
        return G.UNENTSCHIEDEN, "nicht pruefbar", itv, False
    cls = G.classify_equiv(itv["lo"], itv["hi"], 10.0)
    c1 = G.classify_equiv(itv["lo_h1"], itv["hi_h1"], 10.0) if not np.isnan(itv["lo_h1"]) else cls
    c2 = G.classify_equiv(itv["lo_h2"], itv["hi_h2"], 10.0) if not np.isnan(itv["lo_h2"]) else cls
    return cls, "ok", itv, bool(not (c1 == c2 == cls))


def _r19_rec(run_part, n, B, ac, extra=None):
    kl, anz, itv, mc = r19_verdict(run_part["d0"], run_part["st0"], run_part["reps"], B, ac)
    rec = dict(status="ok", n_pop=n, delta=run_part["d0"], delta_status=run_part["st0"], klasse=kl, anzeige=anz, alpha_c=ac,
               bootstrap_kein_eindeutiger_schnittpunkt=int(run_part["cross"]), bootstrap_fit_fehlgeschlagen=int(run_part["fit_fehler"]),
               monte_carlo_instabil=mc, **itv)
    if extra:
        rec.update(extra)
    return rec


def run_r19(data: Data, design: Design, seed: int, mode: str, B: int, outdir: Path):
    t0 = time.time()
    ac = G.alpha_component("S")
    rng_std = seed_stream(seed, RULE_INDEX["R19"], "aux", mode)
    res = dict(unit="R19", mode=mode, family="S", status="ok", alpha_c=ac,
               population="Index-EKG mit Maschinen-QRS und, falls verfuegbar, eigener QRS-Dauer; TTE-LVEF +-30 Tage")
    inp = r19_inputs(data, design)
    run = r19_run_one(inp, rng_std, seed_stream(seed, RULE_INDEX["R19"], "primary", mode), B, "gemeinsame Population")
    out = {}
    if run.get("status") != "ok":
        out["primaer_Maschinenwert"] = run
        out["konkordanz_eigene_Dauer"] = dict(status="nicht berechenbar" if not inp["conc_avail"] else run.get("status"))
    else:
        r0 = run["p"]["risks0"]
        out["primaer_Maschinenwert"] = _r19_rec(run["p"], run["n"], B, ac, dict(
            n_verified=run["n_verified"], cases=run["cases"],
            ereignisse_je_geschlecht=run["ereignisse_je_geschlecht"], risk_f_bei_150=float(r0["F"][60]), risk_m_bei_150=float(r0["M"][60])))
        if run["c"] is not None:
            out["konkordanz_eigene_Dauer"] = _r19_rec(run["c"], run["n"], B, ac, dict(gleiche_gewichte=True))
        elif run.get("c_punktschaetzung_fehlgeschlagen"):
            out["konkordanz_eigene_Dauer"] = dict(status="nicht schaetzbar: Anpassung der Konkordanzvariante nicht konvergiert (Primaerschaetzung bleibt gueltig)")
        else:
            out["konkordanz_eigene_Dauer"] = dict(status="nicht berechenbar: eigene QRS-Dauer in dieser Partition nicht vorhanden")
    res["varianten"] = out
    # Sekundaer: Primaer auf der vollen Population ohne Konkordanzbeschraenkung (B_SENS) und LSB-Strata (je mindestens 150 Verifizierte)
    sec = {}
    Bs = min(B, B_SENS)
    full = r19_inputs(data, design, qconc=None)
    rf = r19_run_one(full, seed_stream(seed, RULE_INDEX["R19"], "aux", mode), seed_stream(seed, RULE_INDEX["R19"], "secondary", mode), Bs, "volle Population")
    if rf.get("status") == "ok":
        k_, a_, itv_, _mc = r19_verdict(rf["p"]["d0"], rf["p"]["st0"], rf["p"]["reps"], Bs, ac)
        sec["primaer_volle_population"] = dict(status="ok", n_pop=rf["n"], delta=rf["p"]["d0"], delta_status=rf["p"]["st0"], lo95=itv_["lo95"], hi95=itv_["hi95"])
    else:
        sec["primaer_volle_population"] = rf
    lb = data.rules["lbbb_any"].values == 1
    for nm, em in (("LSB_Maschinenaussage", lb), ("kein_LSB", ~lb)):
        sub = r19_inputs(data, design, qconc=None, extra_mask=em)
        rs = r19_run_one(sub, seed_stream(seed, RULE_INDEX["R19"], "aux", mode), seed_stream(seed, RULE_INDEX["R19"], "r19_lbbb", mode), Bs, f"Stratum {nm}", min_verified=150)
        if rs.get("status") == "ok":
            k_, a_, itv_, _mc = r19_verdict(rs["p"]["d0"], rs["p"]["st0"], rs["p"]["reps"], Bs, ac)
            sec[nm] = dict(status="ok", n_pop=rs["n"], delta=rs["p"]["d0"], delta_status=rs["p"]["st0"], lo95=itv_["lo95"], hi95=itv_["hi95"])
        else:
            sec[nm] = rs
    res["sekundaer"] = sec
    p, c_ = out["primaer_Maschinenwert"], out["konkordanz_eigene_Dauer"]
    hints = ["Aussage auf den LVEF-Vergleich bei 150 ms als Bezug begrenzt (nicht automatisch fuer 120 oder 130 ms)"]
    final = p.get("klasse", G.UNENTSCHIEDEN)
    if p.get("anzeige") == "nicht pruefbar":
        hints.append("nicht pruefbar")
    if p.get("monte_carlo_instabil"):
        hints.append("Monte-Carlo-instabil (Klasse wechselt zwischen den Haelften der Wiederholungen)")
    if c_.get("status") != "ok":
        final = G.UNENTSCHIEDEN
        hints.append("Konkordanzvariante nicht berechenbar: Endurteil unentschieden (Spezifikation 5, Konkordanzregel)")
    elif c_.get("klasse") != p.get("klasse"):
        final = G.UNENTSCHIEDEN
        hints.append("operationalisierungsabhaengig")
    res["final"] = dict(klasse=final, primaer=p.get("klasse"), hinweise=hints)
    res["dauer_s"] = round(time.time() - t0)
    save_unit(outdir, mode, "R19", res)
    return res


def run_code15(cfg, seed: int, outdir: Path):
    t0 = time.time()
    df = pd.read_csv(cfg["code15"], usecols=["exam_id", "age", "is_male", "1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "patient_id", "death", "timey", "normal_ecg"])
    pre = dict(n_exams=int(len(df)), n_patients=int(df["patient_id"].nunique()),
               n_exams_mit_timey=int(df["timey"].notna().sum()), n_exams_mit_death=int(df["death"].notna().sum()),
               n_patienten_mit_timey=int(df.loc[df["timey"].notna(), "patient_id"].nunique()),
               max_exams_mit_timey_je_patient=int(df[df["timey"].notna()].groupby("patient_id").size().max()),
               n_timey_nicht_positiv=int((df["timey"] <= 0).sum()), n_tod=int(df.loc[df["timey"].notna(), "death"].astype(float).sum()))
    d = df[df["timey"].notna() & df["death"].notna() & (df["timey"] > 0)].copy()
    d = d.drop_duplicates("patient_id", keep="first")  # nur eine Zeile je Patient (Einheit ist das Exam mit gefuelltem death/timey)
    for c in ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg"):
        d[c] = d[c].astype(float)
    pre["n_analyse_basis"] = int(len(d))
    ac = ALPHA / 2.0  # v3 (Nachtrag 1): zwei beurteilte Regeln (R09 RSB, R18 1dAVb) -> m = 2, einseitig 0,0125
    res = dict(unit="CODE15", status="ok", vorpruefung=pre, alpha_c=ac)
    knots = rcs_knots(d["age"].values)
    ab = rcs_basis(d["age"].values, knots)
    out = {}
    six = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    for nm, expo_col, others in (("R09_RSB", "RBBB", ["LBBB", "1dAVb", "AF"]), ("R18_1dAVb", "1dAVb", ["RBBB", "LBBB", "AF"])):
        expo = (d[expo_col] == 1) & (d[others + ["SB", "ST"]].sum(axis=1) == 0)
        comp = (d[six].sum(axis=1) == 0)
        for variant, cm in (("primaer", comp), ("sens_normal_ecg", comp & (d["normal_ecg"] == 1))):
            m = (expo | cm).values
            X = np.column_stack([ab[m], d.loc[m, "is_male"].astype(float).values])
            ee = d.loc[m, "death"].astype(float).values
            beta, se, info = cox_expo(d.loc[m, "timey"].values * 365.25, ee, expo.values[m].astype(float), X, ["a0", "a1", "a2", "male"])
            w = wald_hr(beta, se, ac)
            frame = pd.DataFrame({"expo": expo.values[m].astype(float), "E": ee})
            cls, hint = G.classify_hr_nonharm(w["lo"], w["hi"], HR_MARGIN)
            w.update(n=int(m.sum()), n_exponiert=int(expo.sum()), events=int(frame["E"].sum()),
                     events_exponiert=int(frame.loc[frame["expo"] == 1, "E"].sum()), klasse=cls, hinweis=hint)
            out[f"{nm}_{variant}"] = w
    res["ergebnisse"] = out
    res["hinweis"] = ("Reproduktion der in der vollstaendigen CODE-Kohorte veroeffentlichten Befunde an der offenen Teilmenge, "
                      "keine unabhaengige Bestaetigung; Labels sind 1dAVb/RBBB der Datenbank, keine Surawicz- oder PR-Regel.")
    res["dauer_s"] = round(time.time() - t0)
    d_ = Path(outdir) / "code15"
    d_.mkdir(parents=True, exist_ok=True)
    (d_ / "code15.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return res


# ================================================================== Ausgabe, Kleinzellenmaske, Hashes
NON_COUNT_KEYS = {"B", "nboot", "n_erwartet", "n_undefiniert", "mc_rand", "n_cov", "n_spalten", "n_smd_ueber_0_1", "dauer_s", "failed",
                  "bootstrap_nicht_pruefbar", "sample_mod", "punktschaetzung_fehlgeschlagen",
                  # v3.2: technische Ausfallzaehler (Bootstrap-Wiederholungen, keine Personenzahlen)
                  "bootstrap_fit_fehlgeschlagen", "bootstrap_kein_eindeutiger_schnittpunkt", "fit_fehler"}
TECH_DICT_KEYS = {"undefiniert_je_kennzahl"}  # Woerterbuch technischer Zaehler (Kennzahl -> Anzahl undefinierter Wiederholungen)


def _is_count(k, v):
    """v3 (Sol-Fehler 1): jede ganzzahlige Ausgabe (Python-int, kein bool) ist eine Anzahl und wird maskiert, ausser den technischen
    Zaehlern (Bootstrap, Spaltenzahlen, Laufzeit)."""
    return isinstance(v, int) and not isinstance(v, bool) and k not in NON_COUNT_KEYS


def _collect_counts(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in TECH_DICT_KEYS:
                continue
            if _is_count(k, v):
                acc.append(v)
            else:
                _collect_counts(v, acc)
    elif isinstance(o, list):
        for v in o:
            _collect_counts(v, acc)


def mask_tree(o, unit_size):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if k in TECH_DICT_KEYS:
                out[k] = v
            elif _is_count(k, v):
                out[k] = "<10" if 0 < v < G.SMALL_CELL else int(round(v / unit_size) * unit_size)
            else:
                out[k] = mask_tree(v, unit_size)
        return out
    if isinstance(o, list):
        return [mask_tree(v, unit_size) for v in o]
    return o


def masked_copy(res: dict) -> dict:
    """Kleinzellenregel mit Komplementvergroeberung auf Einheitsebene: ist irgendeine Zahl 1 bis 9, werden alle Anzahlen
    der Einheit auf 100 gerundet, sonst auf 10; 1 bis 9 erscheinen als '<10'."""
    acc = []
    _collect_counts(res, acc)
    any_small = any(0 < v < G.SMALL_CELL for v in acc)
    return mask_tree(res, 100 if any_small else 10)


def dev_results_hash(outdir: Path) -> str:
    d = Path(outdir) / "dev" / "units"
    lines = []
    for p in sorted(d.glob("*.json")):
        lines.append(f"{p.name}:{G.sha256_file(p)}")
    dd = Path(outdir) / "dev" / "design_dev.json"  # v3.1: Entwicklungs-Design (Knoten, Standardisierung) ist Teil der gebundenen Ergebnisse
    if dd.exists():
        lines.append(f"design_dev.json:{G.sha256_file(dd)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def result_hash(outdir: Path, mode: str) -> str:
    d = Path(outdir) / mode / "units"
    lines = [f"{p.name}:{G.sha256_file(p)}" for p in sorted(d.glob("*.json"))]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


MANIFEST_FILES = [  # (Name, Pfad-Schluessel/Pfad)
    "{root}/evidence/clinical_context_features.parquet", "{root}/evidence/machine_report_labels.parquet",
    "{root}/evidence/ecg_echo_matches.parquet", "{root}/evidence/echo_measurements_wide.parquet",
    "{root}/evidence/ecg_lab_features.parquet", "{root}/evidence/ecg_medication_features.parquet",
    "{root}/evidence/patient_clinical_timeline.parquet", "{root}/evidence/patient_code_history.parquet",
    "{root}/labels/clinical_labels_wide.parquet", "{root}/labels/label_definitions.json",
    "{machine_csv}", "{hosp}/admissions.csv.gz", "{hosp}/diagnoses_icd.csv.gz", "{hosp}/d_icd_diagnoses.csv.gz",
    "{counts_db}", "{codelist}",
]


CACHE_DIR_HASH = "f657dbca318026c1f7a5da365d7941f4982cd1fc34c72fa8cf2f5c6c1e6af24c"  # Spezifikation Abschnitt 3


def dir_manifest_hash(root) -> str:
    """Manifesthash eines Verzeichnisses: sortierte Zeilen 'relativer Pfad Groesse' mit abschliessendem Zeilenumbruch."""
    root = str(root)
    rows = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            rows.append((os.path.relpath(p, root).replace(chr(92), "/"), os.path.getsize(p)))
    txt = "".join(f"{r} {sz}" + chr(10) for r, sz in sorted(rows))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def check_inputs(cfg, out: Path, mode: str):
    """Eingabemanifest (verify_manifest) und Cache-Verzeichnis vor der Aufbereitung pruefen."""
    mf = Path(out) / "manifest.json"
    if mf.exists():
        problems = G.verify_manifest(mf)
        if problems:
            raise SystemExit("Eingabemanifest verletzt: " + "; ".join(problems[:5]))
    else:
        log("WARNUNG: kein Eingabemanifest vorhanden")
    cd = Path(cfg["cache"].format(s="train")).parent.parent
    if cd.exists() and dir_manifest_hash(cd) != CACHE_DIR_HASH:
        raise SystemExit("A155-Merkmalscache weicht vom gebundenen Verzeichnishash ab")


def build_manifest(cfg, spec_path, out_path: Path):
    man = {}
    for t in MANIFEST_FILES:
        p = t.format(**cfg)
        man[p] = G.sha256_file(p)
    for nm in ("regeln.py", "k1_geruest.py", "adapter_cache.py", "k1_auswertung.py"):
        man[str(HERE / nm)] = G.sha256_file(HERE / nm)
    man[str(spec_path)] = G.sha256_file(spec_path)
    nach = Path(str(spec_path)).parent / "SPEZIFIKATION_K1_NACHTRAG1.md"  # v3.1: Nachtrag 1 gehoert zur Bindung
    if nach.exists():
        man[str(nach)] = G.sha256_file(nach)
    cm = Path(cfg["cache"].format(s="train")).parent.parent / "cache.manifest.json"
    if cm.exists():
        man[str(cm)] = G.sha256_file(cm)
    Path(out_path).write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    return man


# ================================================================== Zusammenfassung und Bericht
UNIT_ORDER = ["R02", "R03", "R04", "R06", "R07", "R14", "R20_A1", "R20_A2", "R09", "R18", "R08_VTVF", "R08_Tod", "R19"]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and np.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{nd}f}"


def render_report(units: dict, mode: str, dev_units: dict | None = None) -> str:
    L = [f"# K1 Ergebnis {mode} (automatisch erzeugt, Anzahlen nach Kleinzellenregel)", ""]
    L.append("| Einheit | Familie | Urteil (final) | Primaer | Hinweisklassen |")
    L.append("|---|---|---|---|---|")
    for uid in UNIT_ORDER:
        u = units.get(uid)
        if not u:
            continue
        fin = u.get("final") or {}
        L.append(f"| {uid} | {u.get('family', '')} | {fin.get('klasse', u.get('status', 'n/a'))} | {fin.get('primaer', '')} | {'; '.join(fin.get('hinweise', []))} |")
    L.append("")
    for uid in UNIT_ORDER:
        u = units.get(uid)
        if not u:
            continue
        L.append(f"## {uid}")
        if u.get("status") != "ok":
            L.append(f"Status: {u.get('status')}")
            L.append("")
            continue
        if u.get("family") == "E":
            for c in u["judge"]["components"]:
                L.append(f"- {c['id']} ({c['claim']}): Schaetzer {fmt(c['estimate'])}, einseitige Schranken (alpha/m={c['alpha_c']:.5f}) [{fmt(c['lo'])}; {fmt(c['hi'])}], 95 % [{fmt(c['lo95'])}; {fmt(c['hi95'])}], Klasse {c['klasse']}; Hinweise: {', '.join(c['hinweise']) or '-'}")
            k = u["judge"].get("kipp")
            if k:
                L.append(f"- Kipppunktraster robust: {k['_robust']}")
            L.append(f"- Konkordanzvariante: Klasse {u['concordance']['variante_klasse']}")
            for hk, hv in (u.get("hinweisurteile") or {}).items():
                L.append(f"- Hinweisurteil {hk}: Schaetzer {fmt(hv['estimate'])} [{fmt(hv['lo'])}; {fmt(hv['hi'])}] -> {hv['klasse']} ({hv['herkunft']})")
        elif uid.startswith("R20"):
            for nm in ("primaer", "konkordanz"):
                p = u[nm]
                L.append(f"- {nm}: Schaetzer {fmt(p['estimate'], 4)} [{fmt(p['lo'], 4)}; {fmt(p['hi'], 4)}], 95 % [{fmt(p['lo95'], 4)}; {fmt(p['hi95'], 4)}], n={p['n']}, Ereignisse={p['events']}, Klasse {p['klasse']}")
        elif u.get("family") == "P":
            for var in ("primaer", "konkordanz"):
                for oc, v in u[var].items():
                    if v.get("status") == "ok":
                        L.append(f"- {var} {oc}: HR {fmt(v['hr'])} [{fmt(v['lo'])}; {fmt(v['hi'])}] (95 % [{fmt(v['lo95'])}; {fmt(v['hi95'])}]), n={v['n']}, Ereignisse={v['events']}, Klasse {v['klasse']}")
                    else:
                        L.append(f"- {var} {oc}: {v.get('status')}")
        elif uid.startswith("R08"):
            for var in ("primaer", "konkordanz"):
                for oc, v in u[var].items():
                    if v.get("status") == "ok":
                        L.append(f"- {var} {oc}: RR {fmt(v['rr'])} [{fmt(v['lo'])}; {fmt(v['hi'])}] (95 % [{fmt(v['lo95'])}; {fmt(v['hi95'])}]), Klasse {v['klasse']}; {v.get('hinweis', '')}")
                    else:
                        L.append(f"- {var} {oc}: {v.get('status')}")
        elif uid == "R19":
            for nm, v in u["varianten"].items():
                if v.get("status") == "ok":
                    L.append(f"- {nm}: Delta {fmt(v['delta'], 1)} ms [{fmt(v['lo'], 1)}; {fmt(v['hi'], 1)}], 95 % [{fmt(v['lo95'], 1)}; {fmt(v['hi95'], 1)}], Klasse {v['klasse']} ({v['anzeige']}), n={v['n_pop']}")
                else:
                    L.append(f"- {nm}: {v.get('status')}")
        rp = u.get("replikation")
        if rp and mode == "test":  # v3.1: Bericht und JSON verwenden dasselbe Urteil (replication_check)
            L.append(f"- Replikation: Entwicklung {rp.get('entwicklung', 'n/a')} / Test {rp.get('test', 'n/a')} -> {rp.get('urteil', rp.get('status'))}")
        L.append("")
    return "\n".join(L)


def assemble(mode: str, outdir: Path, pub_dir: Path | None):
    d = Path(outdir) / mode / "units"
    units = {}
    for p in sorted(d.glob("*.json")):
        units[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    dev_units = None
    if mode == "test":
        dd = Path(outdir) / "dev" / "units"
        dev_units = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(dd.glob("*.json"))}
        for uid, u in units.items():
            u["replikation"] = replication_check(dev_units.get(uid), u)
    (Path(outdir) / mode / "results.json").write_text(json.dumps(units, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    masked = {uid: masked_copy(u) for uid, u in units.items()}
    if pub_dir:
        pub_dir = Path(pub_dir)
        pub_dir.mkdir(parents=True, exist_ok=True)
        (pub_dir / f"results_{mode}_aggregate.json").write_text(json.dumps(masked, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
        (pub_dir / f"report_{mode}_aggregate.md").write_text(render_report(masked, mode, {k: masked_copy(v) for k, v in (dev_units or {}).items()}), encoding="utf-8")
    return units


def _diff(est_t, lo_t, hi_t, est_d, lo_d, hi_d, log=False):
    """Differenz Test - Entwicklung mit 95 %-Intervall aus den beiden unadjustierten Intervallen (unabhaengige Schaetzer,
    Standardfehler aus der Intervallbreite). Verträglich, wenn das Intervall die Null (bei Verhaeltnissen die Eins) enthaelt."""
    try:
        f_ = (lambda x: math.log(x)) if log else (lambda x: x)
        dt, dd = f_(est_t), f_(est_d)
        st = (f_(hi_t) - f_(lo_t)) / 3.92
        sd = (f_(hi_d) - f_(lo_d)) / 3.92
        se = math.sqrt(st ** 2 + sd ** 2)
        d = dt - dd
        lo, hi = d - 1.96 * se, d + 1.96 * se
        out = dict(differenz=(math.exp(d) if log else d), lo95=(math.exp(lo) if log else lo), hi95=(math.exp(hi) if log else hi),
                   skala="Verhaeltnis" if log else "absolut", vertraeglich=bool(lo <= 0.0 <= hi))
        return out
    except (ValueError, TypeError, ZeroDivisionError):
        return dict(status="nicht berechenbar")


def replication_check(dev: dict | None, test: dict):
    """9.1 Punkt 6, v3 (Nachtrag 1): 'repliziert' nur bei informativen (haelt oder haelt nicht), uebereinstimmenden Endurteilen; zweimal
    'unentschieden' ist keine Bestaetigung. Effektdifferenz Test minus Entwicklung mit 95 %-Intervall; vertraeglich, wenn die Null
    (bei Verhaeltnissen die Eins) enthalten ist."""
    if not dev or "final" not in dev or "final" not in test:
        return dict(status="nicht vergleichbar")
    cd, ct = dev["final"]["klasse"], test["final"]["klasse"]
    out = dict(entwicklung=cd, test=ct)
    if cd == G.UNENTSCHIEDEN and ct == G.UNENTSCHIEDEN:
        out["urteil"] = "nicht informativ (beide unentschieden)"
    elif cd == ct:
        out["urteil"] = "repliziert"
    else:
        out["urteil"] = "Replikation abweichend"
    cons = {}
    if test.get("family") == "E" and "judge" in dev and "judge" in test:
        dc = {c["id"]: c for c in dev["judge"]["components"]}
        for c in test["judge"]["components"]:
            if c["id"] in dc:
                d_ = dc[c["id"]]
                cons[c["id"]] = _diff(c["estimate"], c["lo95"], c["hi95"], d_["estimate"], d_["lo95"], d_["hi95"])
    elif test.get("family") == "P":
        for oc, v in test["primaer"].items():
            dv = dev["primaer"].get(oc, {})
            if v.get("status") == "ok" and dv.get("status") == "ok":
                cons[oc] = _diff(v["hr"], v["lo95"], v["hi95"], dv["hr"], dv["lo95"], dv["hi95"], log=True)
    elif test.get("family") == "S" and test.get("ergebnis"):
        for oc, v in test["primaer"].items():
            dv = dev["primaer"].get(oc, {})
            if v.get("status") == "ok" and dv.get("status") == "ok":
                cons[oc] = _diff(v["rr"], v["lo95"], v["hi95"], dv["rr"], dv["lo95"], dv["hi95"], log=True)
    elif test.get("unit") == "R19":
        v, dv = test["varianten"].get("primaer_Maschinenwert", {}), dev["varianten"].get("primaer_Maschinenwert", {})
        if v.get("status") == "ok" and dv.get("status") == "ok":
            cons["delta"] = _diff(v["delta"], v["lo95"], v["hi95"], dv["delta"], dv["lo95"], dv["hi95"])
    elif str(test.get("unit", "")).startswith("R20"):
        cons["anteil"] = _diff(test["primaer"]["estimate"], test["primaer"]["lo95"], test["primaer"]["hi95"],
                               dev["primaer"]["estimate"], dev["primaer"]["lo95"], dev["primaer"]["hi95"])
    out["effekt_differenz"] = cons
    return out


# ------------------------------------------------------------------ Testregister als Phasenfolge (Sol-Fehler 9)
_ALLOW_COMPLETED = False


def test_phase(register_path, spec_hash, script_hash, manifest_hash, dev_results_hash) -> str:
    """'fehlt' (keine Zeile Beginn mit allen vier Hashes), 'offen' (Beginn, ggf. Fortsetzung) oder 'abgeschlossen' (Zeile Ende mit
    denselben vier Hashes). Nach Abschluss ist kein weiterer Durchgang mit denselben Hashes zulaessig."""
    pth = Path(register_path)
    if not pth.exists():
        return "fehlt"
    hs = (spec_hash, script_hash, manifest_hash, dev_results_hash)
    began = ended = False
    for line in pth.read_text(encoding="utf-8").splitlines():
        if not all(h and h in line for h in hs):
            continue
        cells_ = [c.strip() for c in line.split("|")]
        ev = cells_[3] if len(cells_) > 3 else ""
        if ev.startswith("Beginn"):
            began = True
        elif ev.startswith("Ende"):
            ended = True
    if ended:
        return "abgeschlossen"
    return "offen" if began else "fehlt"


def _phase_aware_check_test_register(register_path, spec_hash, script_hash, manifest_hash, dev_results_hash):
    """Ersatz fuer G.check_test_register (gebundene Datei bleibt unveraendert): nur ein OFFENER Durchgang ist freigegeben. Wird beim
    Import von k1_auswertung in das Geruest eingehaengt, damit auch direkte Aufrufe von G.guard_split/G.partition_sql nach 'Ende'
    scheitern (v3.1, Sol-Punkt 7)."""
    return test_phase(register_path, spec_hash, script_hash, manifest_hash, dev_results_hash) == "offen"


G.check_test_register = _phase_aware_check_test_register


def guard_phase(mode, split, **ga):
    """Wie G.guard_split, zusaetzlich: nach einer Zeile 'Ende' mit denselben Hashes ist die Testpartition gesperrt."""
    if mode == "test" and split == G.TEST_SPLIT and _ALLOW_COMPLETED and ga:
        if test_phase(ga["register_path"], ga["spec_hash"], ga["script_hash"], ga["manifest_hash"], ga["dev_results_hash"]) != "fehlt":
            return  # Lesen bereits erzeugter Ausgaben nach Abschluss (assemble)
    G.guard_split(mode, split, **ga)


def part_sql(relation, split, mode, split_column="split", **ga):
    """Einziger Ladepfad fuer Partitionen: Phasenwaechter, dann Filter per WHERE-Klausel."""
    guard_phase(mode, split, **ga)
    return G.partition_sql(relation, split, mode, split_column=split_column, **ga)


# ================================================================== Kommandozeile
def unit_list(arg: str):
    allu = ["R02", "R03", "R04", "R06", "R07", "R14", "R20", "R09", "R18", "R08", "R19"]
    if not arg or arg == "all":
        return allu
    return [u.strip() for u in arg.split(",") if u.strip()]


def guard_args_for(mode, args, out):
    if mode != "test":
        return {}
    spec_h = G.sha256_file(args.spec)
    if spec_h != SPEC_SHA256:
        raise SystemExit("Spezifikation weicht vom gebundenen Hash ab")
    script_h = G.sha256_file(__file__)
    manifest_h = G.sha256_file(Path(out) / "manifest.json")
    dev_h = dev_results_hash(out)
    return dict(register_path=CFG["register"], spec_hash=spec_h, script_hash=script_h, manifest_hash=manifest_h, dev_results_hash=dev_h)


def check_bound_files():
    for nm, h in BOUND_HASHES.items():
        if G.sha256_file(HERE / nm) != h:
            raise SystemExit(f"Hilfsdatei {nm} weicht vom gebundenen Hash ab")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "assemble", "code15", "manifest", "hash-dev", "hash-result", "version", "register"])
    ap.add_argument("--event", default="Beginn")
    ap.add_argument("--scope", default="locked_test, ein Durchgang")
    ap.add_argument("--auftrag", default="AW-236")
    ap.add_argument("--mode", choices=["dev", "test"], default="dev")
    ap.add_argument("--spec", default=str(data_path("pub13.publikationen_pub_13_kardiologie_spezifikation_k1_entwurf_v3_2_md")))
    ap.add_argument("--out", default=str(data_path("pub13.aw236")))
    ap.add_argument("--pub", default=str(data_path("pub13.publikationen_pub_13_kardiologie_ergebnisse_aw236")))
    ap.add_argument("--units", default="all")
    ap.add_argument("--B", type=int, default=B_BOOT)
    ap.add_argument("--b-sens", type=int, default=B_SENS)
    ap.add_argument("--sample-mod", type=int, default=0)
    ap.add_argument("--rules-only", action="store_true")
    ap.add_argument("--no-checks", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    globals()["B_SENS"] = int(args.b_sens)
    globals()["_ALLOW_COMPLETED"] = (args.cmd == "assemble")
    if os.environ.get("K1_CFG_JSON"):  # nur fuer Tests der Kommandozeile mit synthetischen Eingaben
        CFG.update(json.loads(Path(os.environ["K1_CFG_JSON"]).read_text(encoding="utf-8")))
    if args.cmd == "version":
        print(SCRIPT_VERSION, G.sha256_file(__file__))
        return
    if not args.no_checks:
        check_bound_files()
    spec_h = G.sha256_file(args.spec)
    if spec_h != SPEC_SHA256 and not args.no_checks:
        raise SystemExit("Spezifikation weicht vom gebundenen Hash ab")
    seed = G.seed_from_spec(SPEC_SHA256)
    if args.cmd == "manifest":
        out.mkdir(parents=True, exist_ok=True)
        build_manifest(CFG, args.spec, out / "manifest.json")
        print("manifest", G.sha256_file(out / "manifest.json"))
        return
    if args.cmd == "hash-dev":
        print(dev_results_hash(out))
        return
    if args.cmd == "hash-result":
        print(result_hash(out, args.mode))
        return
    if args.cmd == "code15":
        res = run_code15(CFG, seed, out)
        print(json.dumps(res["ergebnisse"], indent=1, default=_json_default)[:3000])
        return
    if args.cmd == "register":
        gaa = guard_args_for("test", args, out)
        rh = result_hash(out, "test") if args.event == "Ende" else None
        line = G.register_line(args.event, args.auftrag, gaa["spec_hash"], gaa["script_hash"], gaa["manifest_hash"],
                               gaa["dev_results_hash"], args.scope, result_hash=rh)
        print(line)
        G.append_register(CFG["register"], line)
        return
    ga = guard_args_for(args.mode, args, out)
    if args.cmd == "prepare":
        if not args.no_checks:
            check_inputs(CFG, out, args.mode)
        if args.mode == "test":
            guard_phase("test", "locked_test", **ga)
            continue_entry(out, ga, "prepare")
            append_progress(out, "prepare")
        prepare(CFG, args.mode, out, ga, sample_mod=args.sample_mod, rules_only=args.rules_only)
        return
    if args.cmd == "run":
        if args.mode == "test":
            guard_phase("test", "locked_test", **ga)
            continue_entry(out, ga, "run " + args.units)
            append_progress(out, "run " + args.units)
        wdir = work_dir(out, args.mode)
        fixed = None
        kf = Path(out) / "dev" / "design_dev.json"
        main_data = Data(wdir, "main")
        if kf.exists():
            fixed = json.loads(kf.read_text(encoding="utf-8"))  # feste Knoten und Standardisierung aus der Entwicklung
        elif args.mode == "test":
            raise SystemExit("Entwicklungs-Design (design_dev.json) fehlt: der Testlauf darf keine eigenen Knoten und Standardisierung bilden")
        design = Design(main_data.cov, fixed)
        if args.mode == "dev":
            (Path(out) / "dev").mkdir(parents=True, exist_ok=True)
            if not kf.exists():
                kf.write_text(json.dumps(design.state()), encoding="utf-8")
        for u in unit_list(args.units):
            log(f"=== Einheit {u} ({args.mode}) ===")
            if u in ("R02", "R03", "R04", "R06", "R07", "R14"):
                run_e(u, main_data, design, seed, args.mode, args.B, out)
            elif u == "R20":
                run_r20(Data(wdir, "ed"), seed, args.mode, args.B, out)
            elif u in ("R09", "R18"):
                run_p(u, main_data, design, seed, args.mode, out)
            elif u == "R08":
                run_r08(main_data, design, seed, args.mode, out)
            elif u == "R19":
                run_r19(main_data, design, seed, args.mode, args.B, out)
            else:
                raise SystemExit(f"unbekannte Einheit {u}")
        return
    if args.cmd == "assemble":
        globals()["_ALLOW_COMPLETED"] = True  # Lesen bereits erzeugter Ausgaben bleibt nach Abschluss erlaubt
        if args.mode == "test":
            guard_phase("test", "locked_test", **ga)
        units = assemble(args.mode, out, Path(args.pub))
        if args.mode == "test":
            rh = result_hash(out, "test")
            append_progress(out, f"assemble: Ausgabehash {rh}")
            if test_phase(ga["register_path"], ga["spec_hash"], ga["script_hash"], ga["manifest_hash"], ga["dev_results_hash"]) != "abgeschlossen":
                line = G.register_line("Ende", "AW-236", ga["spec_hash"], ga["script_hash"], ga["manifest_hash"], ga["dev_results_hash"],
                                       "locked_test, Ausgabe " + str(Path(out) / "test" / "results.json"), result_hash=rh)
                G.append_register(ga["register_path"], line)
            print("Ausgabehash E:", rh)
        print("Einheiten:", sorted(units))
        return


def continue_entry(out: Path, ga: dict, what: str):
    """Technische Fortsetzung eines nicht abgeschlossenen Durchgangs: Zeile 'Fortsetzung' im Register (der Lauf schreibt sie selbst);
    vor dem allerersten Schritt eines Durchgangs keine Zeile."""
    log_p = Path(out) / "test" / "fortschritt.log"
    if log_p.exists() and log_p.read_text(encoding="utf-8").strip():
        line = G.register_line("Fortsetzung", "AW-236", ga["spec_hash"], ga["script_hash"], ga["manifest_hash"], ga["dev_results_hash"],
                               f"locked_test, technische Fortsetzung ({what})")
        G.append_register(ga["register_path"], line)


def append_progress(out: Path, text: str):
    p = Path(out) / "test" / "fortschritt.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {text}\n")


if __name__ == "__main__":
    main()
