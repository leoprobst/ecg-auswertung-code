"""Addendum to the recomputation (required item 4 of the second opinion): bias-factor formula, synthetic-data check, feasibility limit for the selection factor,
code comparator on the probability scale, figures on the verification rate. Development partition only (train/validation), aggregates only.
Usage: python br5_nachtrag.py sim | run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(data_path("b.werkzeuge_aw241")))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.special import expit, logit  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
import br5_run as B5  # noqa: E402

AGG = B5.AGG
C = B5.C


def sim():
    """Synthetic data: (1) OR_obs = OR_true * B for a log-linear probability ratio; (2) the logit-scale interaction theta is not equal to ln B when selection is not rare."""
    rng = np.random.default_rng(11)
    n = 4_000_000
    from scipy.stats import truncnorm
    x = truncnorm.rvs(-2.5, 2.5, size=n, random_state=11)   # bounded marker range so that probabilities stay <= 1
    or_true = 2.0
    pi0 = expit(-2.5 + np.log(or_true) * x)
    y = (rng.uniform(size=n) < pi0).astype(float)
    lines = ["Addendum, part 1: synthetic-data check of the bias-factor formula (n = 4 million, true OR 2.0 per SD)"]
    rows = []
    for B in (1.0, 1.5, 2.0, 3.0):
        # Probability ratio rho(x) = rho0 * B^x ; P(S|Y=0,x) = s0 (small): P(S|Y=1,x) = rho(x) * s0
        for s0, rho0 in ((0.03, 1.0), (0.10, 1.0)):
            rho = rho0 * B ** x
            p1 = np.minimum(rho * s0, 1.0)
            if (rho * s0 > 1).any():
                continue
            s = rng.uniform(size=n) < np.where(y == 1, p1, s0)
            m = sm.GLM(y[s], sm.add_constant(x[s]), family=sm.families.Binomial()).fit()
            rows.append((B, s0, rho0, float(np.exp(m.params[1])), or_true * B))
    lines.append("B | P(S|Y=0) | rho0 | observed OR | true OR x B")
    for r in rows:
        lines.append(f"{r[0]:.1f} | {r[1]:.2f} | {r[2]:.1f} | {r[3]:.3f} | {r[4]:.3f}")
    # logit-scale interaction: realised B versus exp(theta)
    lines.append("")
    lines.append("Logit selection model logit P(S|x,y) = a + eta*y + theta*x*y: realised B (ratio of the probability ratios per SD, exact) versus exp(theta)")
    lines.append("theta | a | eta | B exact at x=0 -> 1 | exp(theta)")
    for a in (-4.0, -2.0, -1.0):
        for theta in (np.log(2.0),):
            eta = 0.5
            rho = lambda xx: expit(a + eta + theta * xx) / expit(a)  # noqa: E731
            lines.append(f"{theta:.3f} | {a:.1f} | {eta:.1f} | {rho(1.0) / rho(0.0):.3f} | {np.exp(theta):.3f}")
    (AGG / "B_R5_nachtrag_simulation.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


def run():
    cols = ["split", "subject_id", "age", "sex_numeric", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", C.A, C.F, "has_tte_30d"]
    ft, _ = C.read(B5.FT, cols)
    h4, _ = C.read(B5.H4, ["split", "subject_id", "label_state"])
    f, finfo, scale = C.outcome_frame()
    g = f.loc[f.eligible_B3.eq(True)].reset_index(drop=True)
    from werkzeuge.b_test.daten import classify_lvef
    from werkzeuge.b_test.statistik import fit as st_fit, raw_design, fixed_design
    out = {}
    lines = ["Addendum, part 2 (development partition, aggregates only). Script hash br5_nachtrag.py: " + C.sha(__file__), ""]
    for fold in B5.FOLDS:
        t = ft.loc[ft.split.eq(fold)].reset_index(drop=True)
        t["V"] = t.has_tte_30d.eq(True)
        X, ok, names = B5.design_tech(t)
        V = t.V.to_numpy(float)[ok]
        # Figures on the 18 % verification rate
        nall, nver_all = len(t), int(t.V.sum())
        nc, nver_c = int(ok.sum()), int(V.sum())
        gm = g.loc[g.split.eq(fold)]
        yy = classify_lvef(gm.preferred_lvef.to_numpy(float))
        raw, _ = raw_design(gm, C.F)
        res = st_fit(fixed_design(raw, scale), yy, gm.subject_id.to_numpy())
        or_obs = res["or"]
        d = dict(personen_gesamt=nall, verifiziert_gesamt=nver_all, anteil_gesamt=nver_all / nall, modellvollstaendig=nc, verifiziert_modellvollstaendig=nver_c, anteil_modellvollstaendig=nver_c / nc, or_obs=or_obs)
        # Verification model and marginal curve Pm(x)
        m = sm.GLM(V, X, family=sm.families.Binomial()).fit()
        mk = X[:, -1]
        xl, xu = np.quantile(mk, [0.05, 0.95])
        grid = np.linspace(xl, xu, 41)

        def pm(xv):
            Xs = X.copy()
            Xs[:, -1] = xv
            return float(expit(Xs @ m.params).mean())

        P = np.array([pm(v) for v in grid])
        d["marker_q05_q95"] = [float(xl), float(xu)]
        d["Pm_unteres_und_oberes_ende"] = [float(P[0]), float(P[-1])]
        # Feasibility: largest B for which P(S|Y=1,x) <= 1 holds on the range [xl, xu], with rho(xl) = rho_L and rho(x) = rho_L * B^(x - xl)
        feas = {}
        for rho_l in (1.0, 2.0):
            for p_prev in (0.03, 0.06, 0.106):
                best = None
                for Bv in np.arange(1.0, 8.0, 0.01):
                    beta = np.log(or_obs / Bv)
                    a = brentq(lambda aa: expit(aa + beta * mk).mean() - p_prev, -15, 5)
                    pi = expit(a + beta * grid)
                    rho = rho_l * Bv ** (grid - xl)
                    p0 = P / (1 + pi * (rho - 1))
                    p1 = rho * p0
                    if (p1 <= 1 + 1e-12).all() and (p0 > 0).all():
                        best = Bv
                    else:
                        break
                feas[f"rhoL={rho_l}|Praevalenz={p_prev}"] = best
        d["B_max_machbar"] = feas
        d["B_stern_1_25"] = or_obs / 1.25
        d["B_stern_1_0"] = or_obs
        # Code comparator on the probability scale
        c = t.merge(h4.loc[h4.split.eq(fold), ["subject_id", "label_state"]], on="subject_id", how="inner")
        c = c.loc[c.label_state.isin(["positive", "negative"])].reset_index(drop=True)
        Xc, okc, _ = B5.design_tech(c)
        cc = c.loc[okc].reset_index(drop=True)
        code = cc.label_state.eq("positive").to_numpy(float)
        Vc = cc.V.to_numpy(float)

        def bcode(Xc, code, Vc, idx=None):
            if idx is not None:
                Xc, code, Vc = Xc[idx], code[idx], Vc[idx]
            Xi = np.column_stack([Xc, code, Xc[:, -1] * code])
            mm = sm.GLM(Vc, Xi, family=sm.families.Binomial()).fit(maxiter=100)

            def prob(c1, shift):
                Xs = Xc.copy()
                Xs[:, -1] += shift
                Z = np.column_stack([Xs, np.full(len(Xs), c1), Xs[:, -1] * c1])
                return float(expit(Z @ mm.params).mean())

            rho0 = prob(1.0, 0.0) / prob(0.0, 0.0)
            rho1 = prob(1.0, 1.0) / prob(0.0, 1.0)
            return rho1 / rho0, rho0

        bpt, rho0 = bcode(Xc, code, Vc)
        rng = np.random.default_rng(2412026 + (0 if fold == "train" else 1))
        bs = []
        for _ in range(200):
            idx = rng.integers(0, len(Vc), len(Vc))
            try:
                bs.append(bcode(Xc, code, Vc, idx)[0])
            except Exception:  # noqa: BLE001
                pass
        d["code_B_wahrscheinlichkeitsskala"] = dict(punkt=bpt, rho_beim_markermittel=rho0, ci=np.quantile(bs, [0.025, 0.975]).tolist(), n_boot=len(bs))
        out[fold] = d
        lines.append(f"{fold}: " + json.dumps(d))
    (AGG / "B_R5_nachtrag_ausgabe.txt").write_text(json.dumps(out, indent=1), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"sim": sim, "run": run}[sys.argv[1]]()
