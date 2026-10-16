"""AW-241 B-R5, Abschnitt 14 (Zweitmeinung v12, Muss 1 und 2): Machbarkeitsgrenze des Selektionsfaktors B auf DERSELBEN Indexpopulation wie das Hauptmodell.

Indexpopulation: BEF-0063_private.parquet = alle 129.198 Entwicklungspersonen mit ihrem Index-EKG nach der Hauptregel: Verifizierte (eligible_reference) am ersten
verifizierten EKG, nie Verifizierte am ersten nicht ausgeschlossenen EKG. Kovariatenfunktion: B3-Design mit Marker, feste Entwicklungsskalierung (wie Hauptmodell).
S = 1 genau dann, wenn eligible_reference (gueltige LVEF im Fenster), eingeschraenkt auf Personen mit vollstaendigem B3-Design (dann ist S = 1 die Hauptanalysemenge eligible_B3).

Identifizierbarkeit: Der Outcome Y (LVEF < 40 %) ist nur bei S = 1 beobachtet. Ein Verifikationsmodell je Outcomegruppe und damit lambda(x,z) = s1/s0 sind aus den Daten nicht schaetzbar.
Schaetzbar sind (i) q(x,z) = P(Y=1 | x,z,S=1) aus dem Hauptmodell und (ii) die Randwahrscheinlichkeit P_m(x,z) = P(S=1 | x,z) aus einem Verifikationsmodell mit denselben
z-Termen und dem Marker, beides auf der Indexpopulation. Exakte Identitaet (Bayes, ohne Annahme ueber die Form von lambda):
    s1 = P_m (q + lambda (1 - q)),   s0 = P_m ((1 - q) + q / lambda),   P(Y=1|x,z) = q / (q + lambda (1 - q)).
Positivitaet beider Selektionswahrscheinlichkeiten (0 < s0, s1 <= 1) verlangt je Person  lambda_min = q P_m / (1 - P_m (1 - q)) <= lambda <= lambda_max = (1/P_m - q) / (1 - q).
Gemeinsames Modell ln lambda = a(z) + x ln B: lineare Nebenbedingungen in (ln a, ln B); LP liefert das moegliche B-Intervall. Keine Praevalenzannahme noetig.

Aufruf: python br5_index.py [boot]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import os
import sys
import time
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design  # noqa: E402
from werkzeuge.nachrechnung.core import logistic_fit  # noqa: E402

POP = C.POP
AGG = ROOT / "nachrechnung/AW241"
FOLDS = ["train", "validation"]
OR_PUB = {"train": 2.480, "validation": 2.392}
SEED = 2412026


def guard():
    C.guard()


def fit_s(z, s):
    b = logistic_fit(z, s, np.ones(len(s)))
    p = expit(z @ b)
    score = np.max(np.abs(z.T @ (p - s) + np.r_[0, np.full(z.shape[1] - 1, 1e-6)] * b)) / len(s)
    assert np.isfinite(b).all() and score < 1e-7, "Verifikationsmodell nicht konvergiert"
    return b, p


def bounds(P, q):
    """Positivitaetsgrenzen von lambda je Person (Bayes-Identitaet)."""
    assert (P > 0).all() and (P < 1).all() and (q > 0).all() and (q < 1).all()
    lmin = q * P / (1 - P * (1 - q))
    lmax = (1 / P - q) / (1 - q)
    assert (lmin <= 1 + 1e-12).all() and (lmax >= 1 - 1e-12).all(), "lambda = 1 muss immer machbar sein"
    return np.log(np.maximum(lmin, 1e-300)), np.log(lmax)


def lp_range(x, Zc, lo, hi, a_of_z, nonneg):
    """Kleinstes und groesstes ln B unter  lo_i <= a(z_i) + x_i lnB <= hi_i (nonneg: lo_i >= 0). Rueckgabe (v_min, v_max)."""
    lo2 = np.maximum(lo, 0.0) if nonneg else lo
    cols = [np.ones(len(x))] + ([Zc[:, j] for j in range(Zc.shape[1])] if a_of_z else []) + [x]
    A = np.column_stack(cols)
    k = A.shape[1]
    res = []
    for sign in (+1, -1):
        c = np.zeros(k)
        c[-1] = -sign          # sign=+1 maximiert v
        r = linprog(c, A_ub=np.vstack([A, -A]), b_ub=np.r_[hi, -lo2], bounds=[(None, None)] * k, method="highs")
        assert r.status == 0, f"LP nicht optimal: {r.message}"
        res.append(r.x[-1])
    # Machbarkeit des Nullpunkts (lambda = 1, B = 1)
    assert (lo2 <= 1e-12).all() and (hi >= -1e-12).all()
    return res[1], res[0]   # (v_min, v_max)


def flex_design(zf):
    """Flexiblere Verifikationsmodell-Matrix: B3-Spalten, Quadrate der stetigen Terme, Marker mal Winkel, Marker mal Alter."""
    sq = zf[:, [1, 4, 5, 6, 7, 8]] ** 2
    return np.column_stack([zf, sq, zf[:, 8] * zf[:, 7], zf[:, 8] * zf[:, 1]])


VARIANTS_FULL = [("alle", False, False), ("alle", False, True), ("alle", True, False), ("alle", True, True),
                 ("zentral90", False, False), ("zentral90", False, True), ("zentral90", True, False), ("zentral90", True, True)]
VARIANTS_BOOT = [("alle", False, False), ("alle", True, False), ("zentral90", True, False)]


def intervals(zf, lo, hi, variants):
    x = zf[:, -1]
    xl, xu = np.quantile(x, [0.05, 0.95])
    trim = (x >= xl) & (x <= xu)
    out = {}
    for tag, a_of_z, nonneg in variants:
        sel = np.ones(len(x), bool) if tag == "alle" else trim
        vmin, vmax = lp_range(x[sel], zf[sel][:, 1:-1], lo[sel], hi[sel], a_of_z, nonneg)
        out[f"{tag}|a(z)={'linear' if a_of_z else 'konstant'}|lambda>=1={'ja' if nonneg else 'nein'}"] = (float(np.exp(vmin)), float(np.exp(vmax)))
    return out


def boot_fold(zf, sf, yf, B, seed_key, flex=False):
    rng = np.random.default_rng(np.random.SeedSequence(seed_key))
    n = len(sf)
    rows = []
    fails = 0
    for r in range(B):
        idx = rng.integers(0, n, n)
        z, s, y = zf[idx], sf[idx], yf[idx]
        try:
            bo = logistic_fit(z[s == 1], y[s == 1], np.ones(int(s.sum())))
            q = expit(z @ bo)
            Zs = flex_design(z) if flex else z
            bs, P = fit_s(Zs, s)
            lo, hi = bounds(P, q)
            rows.append(intervals(z, lo, hi, VARIANTS_BOOT) | {"_or": float(np.exp(bo[-1]))})
        except (AssertionError, Exception) as ex:   # noqa: BLE001
            fails += 1
            rows.append({"_fehler": str(ex)[:60]})
    return rows, fails


def main(do_boot=False):
    guard()
    f, finfo, scale = C.outcome_frame()   # prueft Footer, liest nur train/validation (Spalten des Hauptmodells)
    cols = ["subject_id", "study_id", "split", "eligible_reference", "eligible_B3", "index_selection_strategy"]
    g, _ = C.read(POP, cols)
    f = f.merge(g[["study_id", "index_selection_strategy"]], on="study_id", validate="one_to_one")
    assert len(f) == 129198 and set(f.split) == set(FOLDS)
    raw, names = raw_design(f, C.F)
    complete = np.isfinite(raw).all(1)
    s_all = f.eligible_reference.to_numpy(bool)
    # Konsistenz mit der Hauptanalyse: S=1 und vollstaendig == eligible_B3
    assert ((s_all & complete) == f.eligible_B3.to_numpy(bool)).all(), "Hauptfallset stimmt nicht mit S=1 und vollstaendig ueberein"
    assert (f.index_selection_strategy[s_all] == "first_verified_ecg_reference_result_blind").all()
    assert (f.index_selection_strategy[~s_all] == "first_nonexcluded_ecg_never_verified").all()
    z_all = np.zeros((len(f), len(names) + 1))
    z_all[complete] = fixed_design(raw[complete], scale)
    y_all = np.full(len(f), np.nan)
    y_all[s_all] = classify_lvef(f.preferred_lvef.to_numpy(float)[s_all])
    out = {"script_sha256": C.sha(__file__), "folds": {}}
    lines = ["AW-241 B-R5 Abschnitt 14 (Entwicklung, nur Aggregate). Skripthash br5_index.py: " + C.sha(__file__), ""]
    rows = []
    for fi, fold in enumerate(FOLDS):
        m = f.split.eq(fold).to_numpy() & complete
        zf, sf, yf = z_all[m], s_all[m].astype(float), y_all[m]
        n, n1 = len(sf), int(sf.sum())
        persons_fold = int(f.split.eq(fold).sum())
        # Hauptmodell (nur S=1): q(x,z) fuer alle Personen
        bo = logistic_fit(zf[sf == 1], yf[sf == 1], np.ones(n1))
        or_obs = float(np.exp(bo[-1]))
        assert abs(or_obs - OR_PUB[fold]) < 0.002, f"Hauptmodell-OR weicht ab: {or_obs}"
        q = expit(zf @ bo)
        # Verifikationsmodell (alle vollstaendigen Personen), gleiche z-Terme plus Marker
        bs, P = fit_s(zf, sf)
        auc = roc_auc_score(sf, P)
        # Kalibrierung: beobachtet gegen vorhergesagt nach Dezilen von P und nach Markerquintilen
        dec = np.quantile(P, np.linspace(0, 1, 11)[1:-1])
        di = np.searchsorted(dec, P)
        cal = [(float(P[di == k].mean()), float(sf[di == k].mean())) for k in range(10)]
        mk = zf[:, -1]
        qi = np.searchsorted(np.quantile(mk, [.2, .4, .6, .8]), mk)
        calq = [(float(P[qi == k].mean()), float(sf[qi == k].mean())) for k in range(5)]
        maxdev = max(abs(a - b) for a, b in cal + calq)
        # Positivitaet
        lo, hi = bounds(P, q)
        x = mk
        xl, xu = np.quantile(x, [0.05, 0.95])
        trim = (x >= xl) & (x <= xu)
        res = {}
        for tag, sel in (("alle", np.ones(n, bool)), ("zentral90", trim)):
            for a_of_z in (False, True):
                for nonneg in (False, True):
                    vmin, vmax = lp_range(x[sel], zf[sel][:, 1:-1], lo[sel], hi[sel], a_of_z, nonneg)
                    res[f"{tag}|a(z)={'linear' if a_of_z else 'konstant'}|lambda>=1={'ja' if nonneg else 'nein'}"] = (float(np.exp(vmin)), float(np.exp(vmax)))
        # flexibleres Verifikationsmodell (Sensitivitaet der Kalibrierung)
        Zx = flex_design(zf)
        bx, Px = fit_s(Zx, sf)
        dec2 = np.quantile(Px, np.linspace(0, 1, 11)[1:-1])
        di2 = np.searchsorted(dec2, Px)
        calx = [(float(Px[di2 == k].mean()), float(sf[di2 == k].mean())) for k in range(10)]
        qi2 = np.searchsorted(np.quantile(mk, [.2, .4, .6, .8]), mk)
        calqx = [(float(Px[qi2 == k].mean()), float(sf[qi2 == k].mean())) for k in range(5)]
        lox, hix = bounds(Px, q)
        resx = intervals(zf, lox, hix, VARIANTS_FULL)
        d = dict(personen_index=persons_fold, vollstaendiges_design=n, verifiziert_vollstaendig=n1, nicht_verifiziert_vollstaendig=n - n1, anteil_verifiziert=n1 / n,
                 or_obs=or_obs, or_marker_verifikation=float(np.exp(bs[-1])), auc_verifikation=float(auc), P_min=float(P.min()), P_max=float(P.max()), P_q05_q95=[float(np.quantile(P, .05)), float(np.quantile(P, .95))],
                 q_q05_q95=[float(np.quantile(q, .05)), float(np.quantile(q, .95))], markerbereich_q05_q95=[float(xl), float(xu)],
                 kalibrierung_dezile_vorhergesagt_beobachtet=cal, kalibrierung_markerquintile=calq, max_abweichung_kalibrierung=float(maxdev),
                 lambda_max_q05_median_q95=[float(np.exp(np.quantile(hi, p))) for p in (.05, .5, .95)], lambda_min_q05_median_q95=[float(np.exp(np.quantile(lo, p))) for p in (.05, .5, .95)],
                 B_stern_1_25=or_obs / 1.25, B_stern_1_0=or_obs, B_intervall=res,
                 flexibel=dict(auc=float(roc_auc_score(sf, Px)), max_abweichung_kalibrierung=float(max(abs(a - b) for a, b in calx + calqx)), P_max=float(Px.max()), B_intervall=resx))
        if do_boot:
            B = 100
            for flex in (False, True):
                t0 = time.time()
                rows_b, fails = boot_fold(zf, sf, yf, B, [SEED, 14, fi, int(flex)], flex)
                good = [r for r in rows_b if "_fehler" not in r]
                summ = {}
                for k in VARIANTS_BOOT:
                    key = f"{k[0]}|a(z)={'linear' if k[1] else 'konstant'}|lambda>=1={'ja' if k[2] else 'nein'}"
                    arr = np.array([r[key] for r in good])
                    summ[key] = dict(B_min_q025_q975=np.quantile(arr[:, 0], [.025, .975]).tolist(), B_max_q025_q975=np.quantile(arr[:, 1], [.025, .975]).tolist(),
                                     anteil_Bmax_ueber_Bstern_1_25=float(np.mean(arr[:, 1] >= or_obs / 1.25)), anteil_Bmax_ueber_Bstern_1_0=float(np.mean(arr[:, 1] >= or_obs)))
                d["bootstrap_flexibel" if flex else "bootstrap_B3"] = dict(gueltig=len(good), versucht=B, fehler=fails, sekunden=round(time.time() - t0), ergebnis=summ)
                print(fold, "boot", flex, len(good), round(time.time() - t0), flush=True)
        out["folds"][fold] = d
        lines.append(f"{fold}: " + json.dumps(d))
    (AGG / "B_R5_index_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    (AGG / "B_R5_index_ausgabe.json").write_text(json.dumps(out, indent=1), encoding="utf8")
    print("\n".join(lines))
    return out, (f, z_all, y_all, s_all, complete)


if __name__ == "__main__":
    main("boot" in sys.argv[1:])
