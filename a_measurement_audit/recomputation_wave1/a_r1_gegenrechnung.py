"""AW-240 / A-R1: unabhaengige Gegenrechnung der Deming-Korrektur (T2, AW-238).

Eigener, neu geschriebener Code (nur numpy/pyarrow), ohne Import aus werkzeuge/aw238.
Eingabe: Personentabelle des T2-Rahmens (nur Spalten split, wall_mm, v6; die Spalte
subject_id wird nicht gelesen) und die Paardifferenzen aus AW-238 Stufe 1 (dy, dx).
Ausgabe: reiner Aggregattext, keine Kennungen.

Aufruf:
  python a_r1_gegenrechnung.py --test    # Kunstdaten-Selbsttest
  python a_r1_gegenrechnung.py --run     # echte Daten
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, hashlib
import numpy as np

SEED = 240101
N_BOOT = 10000
RAHMEN = str(data_path("a.aw238_t2_rahmen_privat_parquet"))
PAARE = str(data_path("a.aw238_stufe1_paardifferenzen_privat_npz"))


# ---------------------------------------------------------------- Kernfunktionen
def momente(x, y):
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = np.sum((x - mx) ** 2)
    syy = np.sum((y - my) ** 2)
    sxy = np.sum((x - mx) * (y - my))
    return n, sxx, syy, sxy


def ols(x, y):
    n, sxx, syy, sxy = momente(x, y)
    return sxy / sxx


def pearson(x, y):
    n, sxx, syy, sxy = momente(x, y)
    return sxy / np.sqrt(sxx * syy)


def deming(x, y, delta):
    """Deming-Steigung, delta = Var(Fehler Y) / Var(Fehler X); geschlossene Form."""
    n, sxx, syy, sxy = momente(x, y)
    a = syy - delta * sxx
    w = np.sqrt(a * a + 4.0 * delta * sxy * sxy)
    if a >= 0:
        return (a + w) / (2.0 * sxy)
    return 2.0 * delta * sxy / (w - a)  # numerisch stabile Form (gleiche Groesse)


def deming_pca(x, y):
    """Gegenprobe delta = 1: Steigung der ersten Hauptachse der 2x2-Kovarianzmatrix."""
    c = np.cov(x, y, ddof=1)
    w, v = np.linalg.eigh(c)
    vec = v[:, np.argmax(w)]
    return vec[1] / vec[0]


def gewichtete_momente(W, x, y):
    """W: (B, n) Zaehlgewichte. Rueckgabe sxx, syy, sxy je Zeile (gewichtete Summen)."""
    sw = W.sum(axis=1)
    mx = (W @ x) / sw
    my = (W @ y) / sw
    sx2 = (W @ (x * x)) - sw * mx * mx
    sy2 = (W @ (y * y)) - sw * my * my
    sxy = (W @ (x * y)) - sw * mx * my
    return sx2, sy2, sxy


def bootstrap(x, y, rx, n_boot, seed, chunk=400):
    rng = np.random.default_rng(seed)
    n = len(x)
    r_l, b_l, c_l = [], [], []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=k).astype(np.float64)
        sxx, syy, sxy = gewichtete_momente(W, x, y)
        r_l.append(sxy / np.sqrt(sxx * syy))
        b = sxy / sxx
        b_l.append(b)
        c_l.append(b / rx)
        done += k
    return np.concatenate(r_l), np.concatenate(b_l), np.concatenate(c_l)


def pi(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def zuverlaessigkeit(var_gesamt, sigma2):
    return 1.0 - sigma2 / var_gesamt


# ---------------------------------------------------------------- Selbsttest
def selftest():
    rng = np.random.default_rng(1)
    fehler = []

    def chk(name, ok):
        print(("ok    " if ok else "FEHLER"), name)
        if not ok:
            fehler.append(name)

    # 1. bekannte Steigung
    n = 5000
    x = rng.normal(10, 2, n)
    y = 3.0 + 1.5 * x + rng.normal(0, 1, n)
    chk("OLS gleich np.polyfit", abs(ols(x, y) - np.polyfit(x, y, 1)[0]) < 1e-10)
    chk("Pearson gleich np.corrcoef", abs(pearson(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-12)
    # 2. Identitaet: standardisiert, delta = 1 -> Betrag 1, Rueckrechnung SD(Y)/SD(X)
    for sgn in (+1, -1):
        y2 = sgn * 0.01 * x + rng.normal(0, 5, n)  # praktisch keine Korrelation
        zx = (x - x.mean()) / x.std(ddof=1)
        zy = (y2 - y2.mean()) / y2.std(ddof=1)
        bz = deming(zx, zy, 1.0)
        chk(f"standardisiert delta=1: Steigung = Vorzeichen(r) ({sgn})",
            abs(bz - np.sign(np.cov(x, y2)[0, 1])) < 1e-9)
        back = bz * y2.std(ddof=1) / x.std(ddof=1)
        chk(f"Rueckrechnung = Vorzeichen * SD(Y)/SD(X) ({sgn})",
            abs(back - np.sign(np.cov(x, y2)[0, 1]) * y2.std(ddof=1) / x.std(ddof=1)) < 1e-9)
        # Originalskala mit delta = Var(Y)/Var(X) liefert dieselbe Zahl
        d2 = deming(x, y2, y2.var(ddof=1) / x.var(ddof=1))
        chk(f"delta=Var(Y)/Var(X) in Originaleinheiten gleich ({sgn})", abs(d2 - back) < 1e-9)
    # 3. PCA-Gegenprobe
    chk("Deming delta=1 gleich erste Hauptachse", abs(deming(x, y, 1.0) - deming_pca(x, y)) < 1e-9)
    # 4. Deming bei delta->0 / gross nahe OLS bzw. inverser OLS
    chk("delta -> 0 gibt inverse OLS", abs(deming(x, y, 1e-12) - (momente(x, y)[2] / momente(x, y)[3])) < 1e-4 * abs(momente(x, y)[2] / momente(x, y)[3]))
    chk("delta -> gross gibt OLS", abs(deming(x, y, 1e12) - ols(x, y)) < 1e-6)
    # 5. Daempfungskorrektur an Kunstdaten mit bekanntem Fehler
    n = 200000
    t = rng.normal(0, 2, n)
    sig_x = 1.0
    xo = t + rng.normal(0, np.sqrt(sig_x), n)
    yo = 2.0 * t + rng.normal(0, 1, n)
    rx = zuverlaessigkeit(xo.var(ddof=1), sig_x)
    chk("Daempfungskorrektur trifft wahre Steigung 2.0 (+-0.03)", abs(ols(xo, yo) / rx - 2.0) < 0.03)
    # 6. Bootstrap: gewichtete Momente gleich direkter Rechnung
    xs = rng.normal(0, 1, 50)
    ys = 0.5 * xs + rng.normal(0, 1, 50)
    W = rng.multinomial(50, np.full(50, 1 / 50), size=3).astype(float)
    sxx, syy, sxy = gewichtete_momente(W, xs, ys)
    ok = True
    for i in range(3):
        idx = np.repeat(np.arange(50), W[i].astype(int))
        ok &= abs(sxy[i] / sxx[i] - ols(xs[idx], ys[idx])) < 1e-10
        ok &= abs(sxy[i] / np.sqrt(sxx[i] * syy[i]) - pearson(xs[idx], ys[idx])) < 1e-10
    chk("gewichteter Bootstrap gleich explizitem Neuziehen", ok)
    # 7. Bootstrap-Intervall deckt die Stichprobenschaetzung ab, Breite plausibel
    n = 3000
    xb = rng.normal(0, 1, n)
    yb = -0.3 * xb + rng.normal(0, 1, n)
    r_b, b_b, c_b = bootstrap(xb, yb, 0.5, 2000, 7)
    lo, hi = pi(b_b)
    se_th = np.sqrt((1 - pearson(xb, yb) ** 2) / (n - 2)) * yb.std(ddof=1) / xb.std(ddof=1)
    chk("Bootstrap-Breite ~ 2*1.96*SE (10%)", abs((hi - lo) / (2 * 1.96 * se_th) - 1) < 0.10)
    chk("Bootstrap-OLS/R_X = OLS/0.5", abs(np.median(c_b) / np.median(b_b) - 2.0) < 1e-9)
    print("Selbsttest:", "BESTANDEN" if not fehler else f"FEHLER {fehler}")
    return not fehler


# ---------------------------------------------------------------- Lauf
def lauf():
    import pyarrow.parquet as pq
    t = pq.read_table(RAHMEN, columns=["split", "wall_mm", "v6"]).to_pandas()
    assert set(t["split"].unique()) <= {"train", "validation"}, "unerwartete Partition"
    x = t["wall_mm"].to_numpy(float)
    y = t["v6"].to_numpy(float)
    sp = t["split"].to_numpy()
    p = np.load(PAARE)
    dy, dx = p["dy"].astype(float), p["dx"].astype(float)

    out = {}
    n, sxx, syy, sxy = momente(x, y)
    vx, vy = sxx / (n - 1), syy / (n - 1)
    out["n"] = int(n)
    out["Var_X"], out["Var_Y"] = vx, vy
    out["SD_X"], out["SD_Y"] = float(np.sqrt(vx)), float(np.sqrt(vy))
    out["SDY_durch_SDX"] = out["SD_Y"] / out["SD_X"]
    out["Cov"] = sxy / (n - 1)
    out["r"] = pearson(x, y)
    out["OLS_YaufX"] = ols(x, y)
    # Fehlervarianzen aus Paardifferenzen: sigma^2 = mean(d^2)/2
    s2y, s2x = float(np.mean(dy ** 2) / 2), float(np.mean(dx ** 2) / 2)
    s2y_v, s2x_v = float(dy.var(ddof=1) / 2), float(dx.var(ddof=1) / 2)
    out.update(sigma2_Y=s2y, sigma2_X=s2x, n_paare_Y=int(len(dy)), n_paare_X=int(len(dx)),
               sigma2_Y_var=s2y_v, sigma2_X_var=s2x_v)
    rx = zuverlaessigkeit(vx, s2x)
    ry = zuverlaessigkeit(vy, s2y)
    out["R_X"], out["R_Y"] = rx, ry
    out["OLS_durch_RX"] = out["OLS_YaufX"] / rx
    out["OLS_durch_RX_varvariante"] = out["OLS_YaufX"] / zuverlaessigkeit(vx, s2x_v)
    out["latente_Korrelation"] = out["r"] / np.sqrt(rx * ry)
    lam0 = s2y / s2x
    out["lambda0"] = lam0
    out["Deming_lambda0"] = deming(x, y, lam0)
    out["Deming_lambda0_x0.25"] = deming(x, y, lam0 * 0.25)
    out["Deming_lambda0_x4"] = deming(x, y, lam0 * 4)
    out["Deming_lambda_robust4"] = deming(x, y, 4.0)
    out["Deming_lambda_VarY_VarX"] = deming(x, y, vy / vx)
    out["Deming_lambda_1_roh"] = deming(x, y, 1.0)
    out["inverse_OLS"] = syy / sxy
    out["VarY_durch_VarX"] = vy / vx
    # Identitaet der alten Zahl
    zx = (x - x.mean()) / x.std(ddof=1)
    zy = (y - y.mean()) / y.std(ddof=1)
    bz = deming(zx, zy, 1.0)
    out["standardisiert_delta1_Steigung"] = bz
    out["standardisiert_zurueck_ms_pro_mm"] = bz * y.std(ddof=1) / x.std(ddof=1)
    out["minus_SDY_durch_SDX"] = -out["SD_Y"] / out["SD_X"]
    out["Hauptachse_delta1_roh"] = deming_pca(x, y)
    # Folds
    for s in ("train", "validation"):
        m = sp == s
        xs_, ys_ = x[m], y[m]
        o = {}
        o["n"] = int(m.sum())
        o["r"] = pearson(xs_, ys_)
        o["OLS"] = ols(xs_, ys_)
        o["OLS_durch_RX"] = o["OLS"] / rx
        o["Deming_lambda0"] = deming(xs_, ys_, lam0)
        o["standardisiert_zurueck"] = deming((xs_ - xs_.mean()) / xs_.std(ddof=1),
                                             (ys_ - ys_.mean()) / ys_.std(ddof=1), 1.0) * ys_.std(ddof=1) / xs_.std(ddof=1)
        o["minus_SDY_durch_SDX"] = -ys_.std(ddof=1) / xs_.std(ddof=1)
        out["fold_" + s] = o
    # Bootstrap (Personenebene, Zeilen = Personen); R_X und lambda fest
    r_b, b_b, c_b = bootstrap(x, y, rx, N_BOOT, SEED)
    out["boot"] = {"seed": SEED, "n_boot": N_BOOT,
                   "r": pi(r_b), "OLS": pi(b_b), "OLS_durch_RX": pi(c_b),
                   "Anteil_r_kleiner_0": float(np.mean(r_b < 0)),
                   "Anteil_OLS_kleiner_0": float(np.mean(b_b < 0))}
    # Bootstrap des Verhaeltnisses SD(Y)/SD(X) (Intervall der alten Zahl)
    rng = np.random.default_rng(SEED + 1)
    nn = len(x)
    sdr = []
    done = 0
    while done < N_BOOT:
        k = min(400, N_BOOT - done)
        W = rng.multinomial(nn, np.full(nn, 1.0 / nn), size=k).astype(np.float64)
        sxx_, syy_, sxy_ = gewichtete_momente(W, x, y)
        sdr.append(np.sqrt(syy_ / sxx_))
        done += k
    out["boot"]["SDY_durch_SDX"] = pi(np.concatenate(sdr))
    return out


def vergleich(o):
    ref = {  # berichtete Werte AW-238 (WELLE0_FEHLERKORREKTUREN_AW238.md, 2.2 bis 2.4)
        "n": (27810, 0), "SD_X": (1.95024, 5e-6), "SD_Y": (6.52812, 5e-6),
        "SDY_durch_SDX": (3.34734, 5e-6), "standardisiert_zurueck_ms_pro_mm": (-3.3473354, 5e-7),
        "r": (-0.0287, 5e-5), "OLS_YaufX": (-0.096, 5e-4), "OLS_durch_RX": (-0.208, 5e-4),
        "sigma2_Y": (14.361, 5e-4), "sigma2_X": (2.0455, 5e-5), "lambda0": (7.02, 5e-3),
        "R_X": (0.462, 5e-4), "R_Y": (0.663, 5e-4), "Deming_lambda0": (-43.7, 0.05),
        "Deming_lambda0_x4": (-0.160, 5e-4), "Deming_lambda_VarY_VarX": (-3.347, 5e-4),
        "VarY_durch_VarX": (11.2, 0.05), "latente_Korrelation": (-0.052, 5e-4),
    }
    zeilen = []
    for k, (wert, tol) in ref.items():
        v = o[k]
        zeilen.append((k, wert, v, abs(v - wert) <= tol + 1e-12))
    return zeilen


if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if "--run" in sys.argv:
        o = lauf()
        v = vergleich(o)
        print(json.dumps(o, indent=1, default=float))
        print("\nVERGLEICH mit AW-238 (berichtet vs. neu):")
        for k, w, n_, ok in v:
            print(f"  {k:36s} berichtet {w!s:>12}  neu {n_: .7f}  {'UEBEREINSTIMMUNG' if ok else 'ABWEICHUNG'}")
        print("script sha256", hashlib.sha256(open(__file__, 'rb').read()).hexdigest())
