"""AW-241 B-R5: Verifikationsselektion (Beschreibung, quantitative Biasanalyse, Ersatz-Outcome-Vergleichswert). Nur Entwicklung.
Aufrufe: test | stamp | run.  Vertrag: nachrechnung/AW241/B_R5_PROTOKOLL.md
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import csv
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
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from werkzeuge.b_test import aw211_common as C  # noqa: E402
from werkzeuge.b_test.daten import classify_lvef  # noqa: E402
from werkzeuge.b_test.statistik import raw_design, fixed_design, fit as st_fit  # noqa: E402

OUT = Path(str(data_path("b.aw241_r5")))
AGG = ROOT / "nachrechnung/AW241"
PROTOKOLL = AGG / "B_R5_PROTOKOLL.md"
A, F = C.A, C.F
SDF = 0.76503231
FOLDS = ["train", "validation"]
R_GRID = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
RD = [1, 2, 3, 4, 5]
FT = str(data_path("b.aw131_first_technical_entwicklung_parquet"))
H4 = str(data_path("b.aw168_bef_0049_private_parquet"))


def sup(n):
    return int(n) if n >= 10 else "<10"


def glm_robust(y, X):
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="HC1", maxiter=100)
    return m


def design_tech(df):
    """Rohdesign B3 (ohne Marker-Standardisierung); Rueckgabe X (standardisiert, Marker in Entwicklungs-SD), Maske der vollstaendigen Zeilen."""
    raw, names = raw_design(df, F)
    ok = np.isfinite(raw).all(1)
    x = raw[ok]
    mean, sd = x.mean(0), x.std(0)
    sd[-1] = SDF
    keep = sd > 0
    X = np.column_stack([np.ones(ok.sum()), (x[:, keep] - mean[keep]) / sd[keep]])
    return X, ok, [n for n, k in zip(names, keep) if k]


def or_ci(m, idx):
    b, se = m.params[idx], m.bse[idx]
    return float(np.exp(b)), float(np.exp(b - 1.959964 * se)), float(np.exp(b + 1.959964 * se))


def stamp():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "protokoll_hash.json"
    if p.exists():
        raise SystemExit("bereits gebunden")
    C.dump(p, {"path": str(PROTOKOLL), "sha256": C.sha(PROTOKOLL), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "before_computation": True})
    print("STAMP", C.sha(PROTOKOLL))


def run():
    s = C.load(OUT / "protokoll_hash.json")
    assert s["sha256"] == C.sha(PROTOKOLL)
    cols = ["split", "subject_id", "age", "sex_numeric", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", A, F, "has_tte_30d"]
    ft, info = C.read(FT, cols)
    h4, _ = C.read(H4, ["split", "subject_id", "label_state"])
    f, finfo, scale = C.outcome_frame()
    g = f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    lines = ["AW-241 B-R5 Ausgabe (post hoc, Entwicklung, nur Aggregate). Skripthash br5_run.py: " + C.sha(__file__), ""]
    rows_flow, rows_desc, rows_qba, rows_dich, rows_proxy = [], [], [], [], []
    for fold in FOLDS:
        t = ft.loc[ft.split.eq(fold)].reset_index(drop=True)
        t["V"] = t.has_tte_30d.eq(True)
        X, ok, names = design_tech(t)
        tc = t.loc[ok].reset_index(drop=True)
        V = tc.V.to_numpy(float)
        gm = g.loc[g.split.eq(fold)]
        yy = classify_lvef(gm.preferred_lvef.to_numpy(float))
        rows_flow.append(dict(fold=fold, zielpopulation=sup(len(t)), basis_und_marker_vollstaendig=sup(ok.sum()), verifiziert_30d=sup(int(tc.V.sum())),
                              nicht_verifiziert=sup(int((~tc.V).sum())), hauptfallset_B3=sup(len(gm)), faelle_lvef_unter_40=sup(int(yy.sum()))))
        # Teil A
        mk = X[:, -1]
        mv, mn = mk[V == 1], mk[V == 0]
        smd = (mv.mean() - mn.mean()) / np.sqrt((mv.var() + mn.var()) / 2)
        auc = roc_auc_score(V, mk)
        q = np.quantile(mk, [.2, .4, .6, .8])
        qi = np.searchsorted(q, mk, side="right")
        rate = [float(V[qi == k].mean()) for k in range(5)]
        m = glm_robust(V, X)
        orv = or_ci(m, X.shape[1] - 1)
        rows_desc.append(dict(fold=fold, marker_mittel_verifiziert=round(mv.mean(), 3), marker_mittel_nicht_verifiziert=round(mn.mean(), 3), marker_sd_verifiziert=round(mv.std(), 3),
                              marker_sd_nicht_verifiziert=round(mn.std(), 3), smd=round(smd, 3), auc_marker_fuer_verifikation=round(auc, 3),
                              verifikationsanteil_quintile=" ".join(f"{x:.3f}" for x in rate), or_marker_verifikation_B3=f"{orv[0]:.3f} ({orv[1]:.3f}; {orv[2]:.3f})", n_modell=sup(len(V))))
        # Teil B
        raw, _ = raw_design(gm, F)
        z = fixed_design(raw, scale)
        res = st_fit(z, yy, gm.subject_id.to_numpy())
        o, lo, hi = res["or"], res["or_ci95_wald"][0], res["or_ci95_wald"][1]
        for r in R_GRID:
            rows_qba.append(dict(fold=fold, r=r, OR_korrigiert=round(o / r, 3), ci_low=round(lo / r, 3), ci_high=round(hi / r, 3), OR_beobachtet=round(o, 3)))
        for tg in (1.25, 1.0):
            rows_qba.append(dict(fold=fold, r=f"r_stern_{tg}", OR_korrigiert=tg, ci_low=round(lo / tg, 3), ci_high=round(hi / tg, 3), OR_beobachtet=round(o, 3), r_stern=round(o / tg, 3)))
        # dichotom
        mk_raw = gm[F].to_numpy(float)
        med = np.median(mk_raw[yy == 0])
        hi_ind = (mk_raw > med).astype(float)
        zd = z.copy()
        zd[:, -1] = hi_ind
        md = glm_robust(yy, zd)
        od = or_ci(md, zd.shape[1] - 1)
        for rh in RD:
            for rl in RD:
                rows_dich.append(dict(fold=fold, r_hoch=rh, r_niedrig=rl, OR_dich_beobachtet=round(od[0], 3), OR_dich_korrigiert=round(od[0] * rl / rh, 3),
                                      ci_low=round(od[1] * rl / rh, 3), ci_high=round(od[2] * rl / rh, 3)))
        lines.append(f"{fold}: OR_obs {o:.3f} ({lo:.3f}; {hi:.3f}); r* fuer 1,25: {o / 1.25:.2f} ({lo / 1.25:.2f}; {hi / 1.25:.2f}); r* fuer 1,0: {o:.2f} ({lo:.2f}; {hi:.2f}); dichotom OR {od[0]:.3f} ({od[1]:.3f}; {od[2]:.3f}) Kontrollmedian-Schnitt")
        # Teil C
        c = tc.merge(h4.loc[h4.split.eq(fold), ["subject_id", "label_state"]], on="subject_id", how="inner")
        c = c.loc[c.label_state.isin(["positive", "negative"])].reset_index(drop=True)
        Xc, okc, _ = design_tech(c)
        cc = c.loc[okc].reset_index(drop=True)
        code = cc.label_state.eq("positive").to_numpy(float)
        Vc = cc.V.to_numpy(float)
        xm = Xc[:, -1]
        Xi = np.column_stack([Xc, code, xm * code])
        mi = glm_robust(Vc, Xi)
        o_code = or_ci(mi, Xi.shape[1] - 2)
        o_int = or_ci(mi, Xi.shape[1] - 1)
        tert = np.quantile(xm, [1 / 3, 2 / 3])
        ti = np.searchsorted(tert, xm, side="right")
        cells = {}
        for k in range(3):
            for cd in (0, 1):
                sel = (ti == k) & (code == cd)
                cells[(k, cd)] = (int(sel.sum()), float(Vc[sel].mean()) if sel.sum() >= 10 else np.nan)
        rows_proxy.append(dict(fold=fold, personen=sup(len(cc)), code_positiv=sup(int(code.sum())), verifikationsanteil_code_neg=round(float(Vc[code == 0].mean()), 3),
                               verifikationsanteil_code_pos=round(float(Vc[code == 1].mean()), 3), OR_code_fuer_V_beim_Markermittel=f"{o_code[0]:.3f} ({o_code[1]:.3f}; {o_code[2]:.3f})",
                               wechselwirkung_exp_theta_je_SD=f"{o_int[0]:.3f} ({o_int[1]:.3f}; {o_int[2]:.3f})",
                               anteile_nach_markerdrittel_neg=" ".join(f"{cells[(k, 0)][1]:.3f}" for k in range(3)), anteile_nach_markerdrittel_pos=" ".join(f"{cells[(k, 1)][1]:.3f}" for k in range(3)),
                               zellen_n_pos=" ".join(str(sup(cells[(k, 1)][0])) for k in range(3))))
        lines.append(f"  Teil C: Wechselwirkung Marker x Code fuer Verifikation exp(theta)={o_int[0]:.3f} ({o_int[1]:.3f}; {o_int[2]:.3f}); Code-OR {o_code[0]:.2f}")
    for name, rows in (("b_r5_zahlen.csv", rows_flow), ("b_r5_beschreibung.csv", rows_desc), ("b_r5_biasanalyse.csv", rows_qba), ("b_r5_dichotom.csv", rows_dich), ("b_r5_ersatzoutcome.csv", rows_proxy)):
        with (AGG / name).open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list({k: 1 for r in rows for k in r}.keys()))
            w.writeheader()
            w.writerows(rows)
    for rows in (rows_flow, rows_desc, rows_proxy):
        for r in rows:
            lines.append(str(r))
    (AGG / "B_R5_ausgabe.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


def test():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + .5 * x)))).astype(float)
    m = glm_robust(y, np.column_stack([np.ones(n), x]))
    o, lo, hi = or_ci(m, 1)
    assert lo < np.exp(.5) < hi + .3 and lo < o < hi
    print("TEST OK", o, lo, hi)


if __name__ == "__main__":
    {"test": test, "stamp": stamp, "run": run}[sys.argv[1]]()
