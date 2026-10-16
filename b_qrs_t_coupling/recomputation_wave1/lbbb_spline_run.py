"""AW-241 Abschnitt 16 (Schwachstellenpruefung B v15, Eintrag 3): Linksschenkelblock-Stratum mit Firth-Korrektur und QRS-Dauer als Spline im B3-Modell.

Nachtraeglich (post hoc), nur Entwicklungsdaten (train, validation). Hauptmodell-OR wird als Assert reproduziert (2,480 / 2,392).

Teil 1: Stratum klassischer LSB-Befund (classical_lbbb_pattern_positive = True) innerhalb der Hauptanalysemenge eligible_B3, B3-Design mit Marker,
        Firth-penalisierte Logistik (Jeffreys-Prior), Profile-Likelihood-Intervall fuer den Marker; je Fold und gepoolt (nur Entwicklung).
        Zum Vergleich: Stratum ohne LSB (ML, CR1-Wald), Hauptmodell ohne LSB-Personen, ML-Punktschaetzer im LSB-Stratum.
Teil 2: QRS-Dauer als Spline im B3-Modell. Variante A: natuerlicher kubischer Spline, 4 Knoten an den Trainings-Quantilen 5/35/65/95 % (Knotenwahl wie der
        Winkel-Spline in aw168_analysen.py); Variante B: feste Knoten 100/120/150 ms (3 Knoten wie die feste Altersbasis); Variante C: Variante A plus
        Indikatoren fuer klassischen LSB- und RSB-Befund (Personen mit bekanntem Befund; Vergleich mit dem Basismodell in derselben Teilmenge).
        Die nichtlinearen Spline-Terme werden mit Trainings-Mittel/-SD standardisiert und vor den Marker gestellt (Marker = letzte Spalte, feste Entwicklungsskalierung).

Aufruf: python lbbb_spline_run.py test | run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
import sys
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"
sys.dont_write_bytecode = True
ROOT = Path(str(data_path("b.auswertung")))
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import chi2  # noqa: E402

AGG = ROOT / "nachrechnung/AW241"
FOLDS = ["train", "validation"]
OR_PUB = {"train": 2.480, "validation": 2.392}
FLAGS = Path(str(data_path("b.aw131_bef_0063_entwicklung_parquet")))
CHI95 = float(chi2.ppf(0.95, 1))


# ---------------------------------------------------------------- Firth
def _pl(X, y, b):
    eta = X @ b
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    pr = expit(eta)
    W = pr * (1 - pr)
    sign, ld = np.linalg.slogdet(X.T @ (W[:, None] * X))
    assert sign > 0
    return ll + 0.5 * ld


def firth_fit(X, y, fixed=None, maxit=500, tol=1e-9):
    """Firth-penalisierte Logistik. fixed=(Index, Wert) haelt einen Koeffizienten fest (Profile-Likelihood). Rueckgabe: beta, penalisierte Loglik, konvergiert, Info-Inverse."""
    n, p = X.shape
    free = [j for j in range(p) if fixed is None or j != fixed[0]]
    beta = np.zeros(p)
    if fixed is not None:
        beta[fixed[0]] = fixed[1]
    cur = _pl(X, y, beta)
    conv = False
    for _ in range(maxit):
        pr = expit(X @ beta)
        W = pr * (1 - pr)
        I = X.T @ (W[:, None] * X)
        Iinv = np.linalg.inv(I)
        h = W * np.einsum("ij,jk,ik->i", X, Iinv, X)
        U = X.T @ (y - pr + h * (0.5 - pr))
        step = np.zeros(p)
        step[free] = np.linalg.solve(I[np.ix_(free, free)], U[free])
        mx = np.max(np.abs(step))
        if mx > 5:
            step *= 5 / mx
        t = 1.0
        while True:
            new = beta + t * step
            val = _pl(X, y, new)
            if val >= cur - 1e-12 or t < 1e-10:
                break
            t /= 2
        beta, cur = new, val
        if np.max(np.abs(t * step[free])) < tol and np.max(np.abs(U[free])) < 1e-6:
            conv = True
            break
    pr = expit(X @ beta)
    W = pr * (1 - pr)
    return beta, cur, conv, np.linalg.inv(X.T @ (W[:, None] * X))


def firth_profile_ci(X, y, m):
    """Firth-Schaetzer fuer Koeffizient m, Profile-Likelihood-Intervall 95 %, LR-p-Wert gegen 0."""
    beta, plhat, conv, Iinv = firth_fit(X, y)
    assert conv, "Firth-Fit nicht konvergiert"
    bh = float(beta[m])
    se = float(np.sqrt(Iinv[m, m]))

    def g(b0):
        _, v, c, _ = firth_fit(X, y, fixed=(m, b0))
        assert c, "Profil-Fit nicht konvergiert"
        return 2 * (plhat - v) - CHI95

    out = []
    for sgn in (-1, +1):
        step = max(se, 0.05)
        edge = bh + sgn * step
        k = 0
        while g(edge) < 0:
            edge += sgn * step
            k += 1
            assert k < 60, "Profilgrenze nicht gefunden"
        lo, hi = (edge, bh) if sgn < 0 else (bh, edge)
        out.append(brentq(g, lo, hi, xtol=1e-7))
    _, v0, c0, _ = firth_fit(X, y, fixed=(m, 0.0))
    assert c0
    lr = max(0.0, 2 * (plhat - v0))
    return dict(beta=bh, OR=float(np.exp(bh)), CI=[float(np.exp(out[0])), float(np.exp(out[1]))], p_lr=float(chi2.sf(lr, 1)), se_wald_penalized=se, beta_ci=out)


# ---------------------------------------------------------------- Spline
def rcs(x, t):
    """Natuerlicher kubischer Spline (Harrell, restricted cubic), t = Knoten aufsteigend; Rueckgabe [x, nichtlineare Terme]."""
    x = np.asarray(x, float)
    t = np.asarray(t, float)
    k = len(t)
    cols = [x]
    for j in range(k - 2):
        cols.append((np.maximum(x - t[j], 0) ** 3
                     - (t[-1] - t[j]) / (t[-1] - t[-2]) * np.maximum(x - t[-2], 0) ** 3
                     + (t[-2] - t[j]) / (t[-1] - t[-2]) * np.maximum(x - t[-1], 0) ** 3) / (t[-1] - t[0]) ** 2)
    return np.column_stack(cols)


def selftest():
    rng = np.random.default_rng(7)
    # Firth gegen direkte numerische Maximierung der penalisierten Loglik
    from scipy.optimize import minimize
    n = 60
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = (rng.random(n) < expit(X @ np.array([-1.0, 0.8, 0.0, 0.5]))).astype(float)
    b, pl, conv, _ = firth_fit(X, y)
    ref = minimize(lambda v: -_pl(X, y, v), np.zeros(4), method="BFGS", options=dict(gtol=1e-9))
    assert conv and np.allclose(b, ref.x, atol=1e-4), (b, ref.x)
    ci = firth_profile_ci(X, y, 1)
    assert ci["CI"][0] < ci["OR"] < ci["CI"][1] and ci["p_lr"] < 0.5
    # Profil-Grenzen erfuellen die LR-Gleichung
    for bb in ci["beta_ci"]:
        _, v, c, _ = firth_fit(X, y, fixed=(1, bb))
        assert abs(2 * (pl - v) - CHI95) < 1e-4
    # vollstaendige Separation: ML divergiert, Firth endlich
    xs = np.r_[np.linspace(-2, -0.1, 20), np.linspace(0.1, 2, 20)]
    Xs = np.column_stack([np.ones(40), xs])
    ys = (xs > 0).astype(float)
    bs, _, cs, _ = firth_fit(Xs, ys)
    assert cs and np.isfinite(bs).all() and abs(bs[1]) < 20
    cs2 = firth_profile_ci(Xs, ys, 1)
    assert np.isfinite(cs2["CI"]).all()
    # Spline: Linearterm identisch, nichtlineare Terme verschwinden jenseits der Knoten nicht, aber linear ausserhalb des letzten Knotens
    t = np.array([1.0, 2.0, 3.0, 4.0])
    xx = np.array([5.0, 6.0, 7.0, 8.0])
    B = rcs(xx, t)
    assert B.shape == (4, 3) and np.allclose(B[:, 0], xx)
    for j in (1, 2):
        d2 = np.diff(B[:, j], 2)
        assert np.allclose(d2, 0, atol=1e-9), "Spline muss rechts vom letzten Knoten linear sein"
    # fixe Knoten 3 -> eine nichtlineare Spalte
    assert rcs(xx, np.array([100., 120., 150.])).shape == (4, 2)
    print("Selbsttest bestanden")


def main():
    from werkzeuge.b_test import aw211_common as C
    from werkzeuge.b_test.daten import classify_lvef
    from werkzeuge.b_test.statistik import raw_design, fixed_design, fit
    from werkzeuge.nachrechnung.core import logistic_fit
    C.guard()
    f, info, scale = C.outcome_frame()
    assert len(f) == 129198 and set(f.split) == set(FOLDS)
    fl, _ = C.read(FLAGS, ["study_id", "split", "classical_lbbb_pattern_positive", "classical_rbbb_pattern_positive", "qrs_duration_ms"])
    f = f.merge(fl, on="study_id", suffixes=("", "_flag"), validate="one_to_one")
    assert (f.split == f.split_flag).all() and np.allclose(f.qrs_duration_ms.fillna(-1), f.qrs_duration_ms_flag.fillna(-1))
    lb = f.classical_lbbb_pattern_positive
    rb = f.classical_rbbb_pattern_positive
    lbbb = lb.eq(True).to_numpy()
    lbbb_known = lb.notna().to_numpy()
    rbbb = rb.eq(True).to_numpy()
    rbbb_known = rb.notna().to_numpy()
    raw, names = raw_design(f, C.F)
    complete = np.isfinite(raw).all(1)
    elig = f.eligible_B3.to_numpy(bool)
    assert ((f.eligible_reference.to_numpy(bool) & complete) == elig).all()
    z = np.full((len(f), len(names) + 1), np.nan)
    z[complete] = fixed_design(raw[complete], scale)
    y = np.full(len(f), np.nan)
    er = f.eligible_reference.to_numpy(bool)
    y[er] = classify_lvef(f.preferred_lvef.to_numpy(float)[er])
    subj = f.subject_id.to_numpy()
    sp = f.split.to_numpy()
    qrs = f.qrs_duration_ms.to_numpy(float)
    res = {"script_sha256": C.sha(__file__), "names": names, "main": {}, "lbbb": {}, "spline": {}}
    lines = ["AW-241 Abschnitt 16 (Entwicklung, nur Aggregate). Skripthash lbbb_spline_run.py: " + C.sha(__file__), ""]

    def counts(mask):
        n = int(mask.sum())
        c = int((y[mask] == 1).sum())
        return n, c

    def cell(n):
        return str(n) if n >= 10 else "<10"

    # Hauptmodell und Referenz
    for fold in FOLDS:
        m = elig & (sp == fold)
        r = fit(z[m], y[m], subj[m])
        assert abs(r["or"] - OR_PUB[fold]) < 0.002
        res["main"][fold] = dict(OR=r["or"], CI=r["or_ci95_wald"], N=r["N"])
        n_l, c_l = counts(m & lbbb)
        n_unk, _ = counts(m & ~lbbb_known)
        res["main"][fold].update(lbbb_persons=cell(n_l), lbbb_cases=cell(c_l), lbbb_unknown_persons=cell(n_unk))
        lines.append(f"Hauptmodell {fold}: OR={r['or']:.3f} [{r['or_ci95_wald'][0]:.3f};{r['or_ci95_wald'][1]:.3f}] N={r['N']}; LSB-Personen={cell(n_l)}, davon LVEF<40: {cell(c_l)}; LSB-Befund fehlt: {cell(n_unk)}")
    lines.append("")

    # Teil 1: LSB-Stratum
    mk = z.shape[1] - 1
    for tag, folds in (("train", ["train"]), ("validation", ["validation"]), ("entwicklung_gepoolt", FOLDS)):
        m = elig & lbbb & np.isin(sp, folds)
        n, c = counts(m)
        rec = dict(persons=cell(n), cases=cell(c))
        assert c >= 10 and n - c >= 10, "Zelle < 10 im Stratum: manuell behandeln"
        X, yy = z[m], y[m]
        fr = firth_profile_ci(X, yy, mk)
        rec["firth"] = fr
        rec["events_per_parameter"] = float(min(c, n - c) / X.shape[1])
        # ML-Punktschaetzer (Ridge 1e-6) zum Vergleich, ohne Intervall
        try:
            b = logistic_fit(X, yy, np.ones(len(yy)))
            rec["ml_point_or"] = float(np.exp(b[-1]))
        except Exception as ex:  # noqa: BLE001
            rec["ml_point_or"] = None
            rec["ml_note"] = str(ex)[:60]
        # Kompaktes Modell: Marker, Alter (2 Spalten), Geschlecht
        Xc = np.column_stack([z[m][:, [0, 1, 2, 3]], z[m][:, mk]])
        rec["firth_compact_age_sex"] = firth_profile_ci(Xc, yy, Xc.shape[1] - 1)
        res["lbbb"][tag] = rec
        lines.append(f"LSB-Stratum {tag}: Personen={rec['persons']}, LVEF<40={rec['cases']}; Firth OR={fr['OR']:.3f} [{fr['CI'][0]:.3f};{fr['CI'][1]:.3f}] p_LR={fr['p_lr']:.4f}; "
                     f"ML-Punkt={rec['ml_point_or'] if rec['ml_point_or'] is None else round(rec['ml_point_or'], 3)}; Ereignisse/Parameter={rec['events_per_parameter']:.1f}; "
                     f"kompakt (Alter, Geschlecht): OR={rec['firth_compact_age_sex']['OR']:.3f} [{rec['firth_compact_age_sex']['CI'][0]:.3f};{rec['firth_compact_age_sex']['CI'][1]:.3f}]")
    for fold in FOLDS:
        out = {}
        m = elig & ~lbbb & lbbb_known & (sp == fold)
        r = fit(z[m], y[m], subj[m])
        out["ohne_LSB_befund_bekannt"] = dict(OR=r["or"], CI=r["or_ci95_wald"], N=r["N"])
        m2 = elig & ~lbbb & (sp == fold)
        r2 = fit(z[m2], y[m2], subj[m2])
        out["hauptmodell_ohne_LSB_personen"] = dict(OR=r2["or"], CI=r2["or_ci95_wald"], N=r2["N"])
        res["lbbb"].setdefault("vergleich", {})[fold] = out
        lines.append(f"Vergleich {fold}: Stratum LSB negativ (Befund bekannt) OR={r['or']:.3f} [{r['or_ci95_wald'][0]:.3f};{r['or_ci95_wald'][1]:.3f}] N={r['N']}; "
                     f"Hauptmodell ohne LSB-Personen OR={r2['or']:.3f} [{r2['or_ci95_wald'][0]:.3f};{r2['or_ci95_wald'][1]:.3f}] N={r2['N']}")
    lines.append("")

    # Teil 2: QRS-Spline
    tr = elig & (sp == "train")
    kq = np.quantile(qrs[tr], [.05, .35, .65, .95])
    kf = np.array([100., 120., 150.])
    res["spline"]["knots_quantile_ms"] = kq
    res["spline"]["knots_fixed_ms"] = kf

    def with_extra(zz, q, knots, tmask):
        B = rcs(q, knots)[:, 1:]
        mu = B[tmask].mean(0)
        sd = B[tmask].std(0)
        assert (sd > 0).all()
        return np.column_stack([zz[:, :-1], (B - mu) / sd, zz[:, -1:]])

    ok = complete
    zA = np.full((len(f), z.shape[1] + 2), np.nan)
    zA[ok] = with_extra(z[ok], qrs[ok], kq, tr[ok])
    zB = np.full((len(f), z.shape[1] + 1), np.nan)
    zB[ok] = with_extra(z[ok], qrs[ok], kf, tr[ok])
    for fold in FOLDS:
        m = elig & (sp == fold)
        base = fit(z[m], y[m], subj[m])
        rec = dict(basis=dict(OR=base["or"], CI=base["or_ci95_wald"], N=base["N"]))
        for tag, zz in (("A_quantilknoten_5_35_65_95", zA), ("B_feste_knoten_100_120_150", zB)):
            r = fit(zz[m], y[m], subj[m])
            rec[tag] = dict(OR=r["or"], CI=r["or_ci95_wald"], N=r["N"], delta_auc_marker=r["delta_auc"], p_marker=r["p_raw"])
        # Variante C: Spline A plus LSB/RSB-Indikatoren, Personen mit bekanntem Befund
        mc = m & lbbb_known & rbbb_known
        base_c = fit(z[mc], y[mc], subj[mc])
        extra = np.column_stack([lbbb[mc].astype(float), rbbb[mc].astype(float)])
        zc = np.column_stack([zA[mc][:, :-1], extra, zA[mc][:, -1:]])
        # Indikatoren mit < 10 Personen oder konstanten Spalten pruefen
        ln, lc = counts(mc & lbbb)
        rn, rc = counts(mc & rbbb)
        r = fit(zc, y[mc], subj[mc])
        rec["C_spline_A_plus_LSB_RSB_indikatoren"] = dict(OR=r["or"], CI=r["or_ci95_wald"], N=r["N"], lbbb_persons=cell(ln), rbbb_persons=cell(rn),
                                                         basis_gleiche_teilmenge=dict(OR=base_c["or"], CI=base_c["or_ci95_wald"], N=base_c["N"]))
        res["spline"][fold] = rec
        lines.append(f"Spline {fold}: Basis OR={rec['basis']['OR']:.3f} [{rec['basis']['CI'][0]:.3f};{rec['basis']['CI'][1]:.3f}]")
        for tag in ("A_quantilknoten_5_35_65_95", "B_feste_knoten_100_120_150"):
            r = rec[tag]
            lines.append(f"  {tag}: OR={r['OR']:.3f} [{r['CI'][0]:.3f};{r['CI'][1]:.3f}] N={r['N']}")
        r = rec["C_spline_A_plus_LSB_RSB_indikatoren"]
        lines.append(f"  C (A + LSB/RSB-Indikatoren; N={r['N']}, LSB={r['lbbb_persons']}, RSB={r['rbbb_persons']}): OR={r['OR']:.3f} [{r['CI'][0]:.3f};{r['CI'][1]:.3f}]; Basis in derselben Teilmenge OR={r['basis_gleiche_teilmenge']['OR']:.3f} [{r['basis_gleiche_teilmenge']['CI'][0]:.3f};{r['basis_gleiche_teilmenge']['CI'][1]:.3f}]")
    C.dump(AGG / "LB_ergebnis.json", res)
    (AGG / "LB_ausgabe_roh.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        main()
    else:
        raise SystemExit("Aufruf: lbbb_spline_run.py test|run")
