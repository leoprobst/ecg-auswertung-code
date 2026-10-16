"""AW-238 A-R1, Stufe 2: Steigungen auf der Originalskala (OLS, umgekehrte OLS, Pearson r, Deming mit lambda0 und
Sensitivitaeten, reliabilitaetskorrigierte OLS). Festlegung: nachrechnung/AW238/A_R1_VORABFESTLEGUNG.md (vorab gehasht).
Eingabe: <local path> (privat), stufe1_fehlervarianz.json, stufe1_paardifferenzen_privat.npz.
Ausgabe: nur Aggregate. Seed 20261015, 10.000 Ziehungen, Personen als Bootstrap-Einheit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import hashlib, json
import numpy as np
import pandas as pd

OUT = str(data_path("a.aw238"))
SEED, B = 20261015, 10000
fr = pd.read_parquet(OUT + "/t2_rahmen_privat.parquet")
s1 = json.load(open(OUT + "/stufe1_fehlervarianz.json"))
pairs = np.load(OUT + "/stufe1_paardifferenzen_privat.npz")
dy, dx = pairs["dy"], pairs["dx"]
x, y = fr["wall_mm"].to_numpy(float), fr["v6"].to_numpy(float)
split = fr["split"].to_numpy()
n = len(x)


def moments(x, y):
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum() / (len(x) - 1)
    syy = ((y - my) ** 2).sum() / (len(x) - 1)
    sxy = ((x - mx) * (y - my)).sum() / (len(x) - 1)
    return sxx, syy, sxy


def deming(sxx, syy, sxy, lam):
    return (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)


lam0 = s1["lambda0"]
sy2, sx2 = s1["sigma2_y"], s1["sigma2_x"]
sxx, syy, sxy = moments(x, y)
r = sxy / np.sqrt(sxx * syy)
RX0 = 1 - sx2 / sxx
RY0 = 1 - sy2 / syy
lam_std = syy / sxx  # Entsprechung der Original-Standardisierung (gleiche Reliabilitaet)
grid = {
    "lambda0 x 0.25": lam0 * 0.25,
    "lambda0 (7.02)": lam0,
    "lambda0 x 4": lam0 * 4,
    "lambda robust (MAD)": s1["lambda_robust"],
    "lambda = Var(Y)/Var(X) (Original-Standardisierung)": lam_std,
    "lambda = 1 roh (einheitenabhaengig)": 1.0,
}


def estimates(x, y, lam0_, rx):
    sxx, syy, sxy = moments(x, y)
    out = {"r": sxy / np.sqrt(sxx * syy), "OLS Y~X": sxy / sxx, "umgekehrte OLS (Syy/Sxy)": syy / sxy,
           "OLS/R_X": sxy / sxx / rx}
    out["Deming lambda0"] = deming(sxx, syy, sxy, lam0_)
    return out


point = {"r": r, "OLS Y~X": sxy / sxx, "umgekehrte OLS (Syy/Sxy)": syy / sxy, "OLS/R_X": sxy / sxx / RX0}
for k, v in grid.items():
    point["Deming " + k] = deming(sxx, syy, sxy, v)
point["R_X (aus Stufe 1)"] = RX0
point["R_Y (aus Stufe 1)"] = RY0
point["implizierte latente Korrelation r/sqrt(RX*RY)"] = r / np.sqrt(RX0 * RY0)

rng = np.random.default_rng(SEED)
boots = {k: np.empty(B) for k in point if k not in ("R_X (aus Stufe 1)", "R_Y (aus Stufe 1)")}
bootsB = {"Deming lambda0": np.empty(B), "OLS/R_X": np.empty(B), "lambda": np.empty(B), "R_X": np.empty(B)}
for b in range(B):
    ii = rng.integers(0, n, size=n)
    pair_yb = rng.integers(0, len(dy), size=len(dy))
    pair_xb = rng.integers(0, len(dx), size=len(dx))
    xb, yb = x[ii], y[ii]
    sxxb, syyb, sxyb = moments(xb, yb)
    boots["r"][b] = sxyb / np.sqrt(sxxb * syyb)
    boots["OLS Y~X"][b] = sxyb / sxxb
    boots["umgekehrte OLS (Syy/Sxy)"][b] = syyb / sxyb
    boots["OLS/R_X"][b] = sxyb / sxxb / RX0
    for k, v in grid.items():
        boots["Deming " + k][b] = deming(sxxb, syyb, sxyb, v)
    boots["implizierte latente Korrelation r/sqrt(RX*RY)"][b] = (sxyb / np.sqrt(sxxb * syyb)) / np.sqrt(RX0 * RY0)
    # Variante B: Fehlerquotient und R_X mit neu gezogenen Paaren
    sy2b = float((dy[pair_yb] ** 2).sum() / (2 * len(dy)))
    sx2b = float((dx[pair_xb] ** 2).sum() / (2 * len(dx)))
    lamb = sy2b / sx2b
    rxb = 1 - sx2b / sxxb
    bootsB["Deming lambda0"][b] = deming(sxxb, syyb, sxyb, lamb)
    bootsB["OLS/R_X"][b] = sxyb / sxxb / rxb
    bootsB["lambda"][b] = lamb
    bootsB["R_X"][b] = rxb


def ci(a):
    return np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)


print(f"n = {n}; lambda0 = {lam0:.3f}; sigma2_Y = {sy2:.3f}; sigma2_X = {sx2:.4f}; Var(X) = {sxx:.4f}; Var(Y) = {syy:.3f}")
print(f"Cov = {sxy:.5f}; r = {r:.5f}")
print("\n| Schaetzer | Punktschaetzung | 95%-Intervall (A: lambda, R_X fest) | Anteil Ziehungen < 0 |")
print("|---|---:|---|---:|")
for k, v in point.items():
    if k in boots:
        lo, hi = ci(boots[k])
        print(f"| {k} | {v:.4f} | [{lo:.4f}; {hi:.4f}] | {np.mean(boots[k] < 0):.4f} |")
    else:
        print(f"| {k} | {v:.4f} | | |")
print("\nVariante B (Fehlervarianzen mit neu gezogenen Paaren):")
for k in ("Deming lambda0", "OLS/R_X", "lambda", "R_X"):
    lo, hi = ci(bootsB[k])
    print(f"  {k}: Median {np.median(bootsB[k]):.4f}  [{lo:.4f}; {hi:.4f}]")

print("\nSplit-Punktschaetzungen (lambda0, R_X fest):")
for sp in ("train", "validation"):
    m = split == sp
    xs, ys = x[m], y[m]
    a, b_, c = moments(xs, ys)
    print(f"  {sp} n={m.sum()}: r={c/np.sqrt(a*b_):.4f}  OLS={c/a:.4f}  inverse={b_/c:.4f}  Deming(lambda0)={deming(a,b_,c,lam0):.4f}  OLS/R_X={c/a/RX0:.4f}")

# Sensitivitaet R_X aus 1-d- und 30-d-Fenstern
raw = s1["raw"]
for key in ("X|24|Personen des T2-Rahmens", "X|720|Personen des T2-Rahmens"):
    sx2k = raw[key][1]
    rxk = 1 - sx2k / sxx
    print(f"R_X mit {key}: sigma2_X={sx2k:.3f}  R_X={rxk:.3f}  OLS/R_X={sxy/sxx/rxk:.4f}")

# Statusregel (Originalspezifikation)
def status(est, lo, hi, tr, va):
    if hi < 0 or lo > 4.0:
        return "widerlegt"
    if 1.7 <= est <= 4.0 and lo > 0 and tr > 0 and va > 0:
        return "bestaetigt (Mechanismus A)"
    if lo <= 0 <= hi:
        return "bestaetigt (Mechanismus B: keine Beziehung)"
    return "unentschieden (zwischen A und B)"


print("\nStatusregel (unveraendert) angewandt:")
for k in ("Deming lambda0", "OLS Y~X", "OLS/R_X"):
    kk = "Deming lambda0 (7.02)" if k == "Deming lambda0" else k
    kk = "Deming lambda0 (7.02)" if k == "Deming lambda0" else k
    key = "Deming lambda0 (7.02)" if k == "Deming lambda0" else k
    est = point[key]
    lo, hi = ci(boots[key])
    vals = []
    for sp in ("train", "validation"):
        m = split == sp
        a, b_, c = moments(x[m], y[m])
        if key.startswith("Deming"):
            vals.append(deming(a, b_, c, lam0))
        elif key == "OLS Y~X":
            vals.append(c / a)
        else:
            vals.append(c / a / RX0)
    print(f"  {key}: Schaetzer {est:.4f} [{lo:.4f}; {hi:.4f}], Folds {vals[0]:.4f}/{vals[1]:.4f} -> {status(est, lo, hi, vals[0], vals[1])}")
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
