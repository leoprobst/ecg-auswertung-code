"""AW-244 B-R8 (nachtraeglich, post hoc): T-Ende-Kipppunkt (S17) mit kohaerent neu berechnetem QTc und Pruefung der Ausfaelle.

Ausgang: AW-192 (nur Entwicklung): 1.500 Faelle (LVEF < 40 %) und 1.500 Kontrollen, T-Enden aller Schlaege nur bei Faellen
um -32..+32 ms verschoben; Fréchet-Abstand und Winkel neu berechnet (signal_records), die uebrigen Kovariaten blieben
gebunden, auch das QTc. Hier: QT verschiebt sich bei gleicher Verschiebung aller Schlag-T-Enden um denselben Betrag s
(QT = T-Ende minus Q-Beginn), QTcF = QT / RR^(1/3) mit dem Release-RR; daher
    QTcF_neu = QTcF * (QT + s) / QT        (nur Faelle; Kontrollen unveraendert).
QT (qt_interval_ms) stammt aus dem Entwicklungs-Merkmalscache desselben Releases (Gate: QTcF dort gleich QTcF der Stichprobe).
Alle anderen Kovariaten (Alter, Geschlecht, HF, QRS) sind T-Ende-unabhaengig.
Modelle: B3-Design mit fester Skalierung aus AW-166 (scaling_B3), gewichtet (Stichprobengewichte), gleiche Personen wie AW-192
(gemeinsame endliche Menge). Bootstrap: gepaart, geschichtet nach Fold und Klasse, 500 Ziehungen, Seed SeedSequence([19220261013,2])
wie AW-192 (Gate: die Variante 'QTc gebunden' reproduziert results.json). Zusatz: Verfuegbarkeitsmenge je Gitterpunkt;
Merkmale der 50 Faelle ausserhalb der gemeinsamen Menge.
Kein Testzugriff. Ausgabe nur Aggregate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

sys.path.insert(0, str(h244.AUSW))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRID = [-32, -24, -16, -8, -4, 0, 4, 8, 16, 24, 32]
MARKER = "qrs_t_normalized_discrete_frechet"
NBOOT = 500
OLD = Path(str(data_path("b.aw166")))
A192 = Path(str(data_path("b.aw192")))


def qtc_update(qtc, qt, s):
    return qtc * (qt + s) / qt


def design(f, vals_j, qtc_new, sc, finite_mask=None):
    from werkzeuge.b_test.statistik import raw_design, fixed_design
    xx, names = raw_design(f, MARKER)
    xx = xx.copy()
    xx[:, -1] = vals_j[:, 0]
    xx[:, -2] = vals_j[:, 1]
    if qtc_new is not None:
        i = names.index("qtc_fridericia_ms")
        xx[:, i] = qtc_new
    return xx, names


def crossing(betas, target, direction):
    zero = GRID.index(0)
    indices = [zero] + sorted([j for j, ms in enumerate(GRID) if ms * direction > 0], key=lambda j: abs(GRID[j]))
    t = np.log(target)
    if betas[zero] <= t:
        return 0.0
    for a, b in zip(indices, indices[1:]):
        if betas[a] > t and betas[b] <= t:
            return GRID[a] + (GRID[b] - GRID[a]) * (t - betas[a]) / (betas[b] - betas[a])
    return None


def fit_beta(z, y, w):
    from werkzeuge.nachrechnung.core import logistic_fit
    from scipy.special import expit
    b = logistic_fit(z, y, w)
    sc = np.max(np.abs(z.T @ (w * (expit(z @ b) - y)) + np.r_[0, np.full(z.shape[1] - 1, 1e-6)] * b)) / w.sum()
    if not np.isfinite(b).all() or sc >= 1e-7:
        raise ValueError("convergence")
    return b


def core(f, vals, qt, sc, nboot, y, w, seedseq, log=print):
    """f: Stichprobe (DataFrame), vals: (n, len(GRID), 2), qt: (n,), liefert Zeilen/Kipppunkte fuer beide Varianten."""
    from werkzeuge.b_test.statistik import fixed_design
    common = np.isfinite(vals).all(axis=(1, 2))
    shifts = np.array(GRID, float)
    out = {}
    designs = {"gebunden": [], "neu": []}
    avail = {"gebunden": [], "neu": []}
    for j, ms in enumerate(GRID):
        for var in ("gebunden", "neu"):
            q = None
            if var == "neu":
                q = f["qtc_fridericia_ms"].to_numpy(float).copy()
                case = y == 1
                q[case] = qtc_update(q[case], qt[case], ms)
            xx, _ = design(f, vals[:, j, :], q, sc)
            finite = np.isfinite(xx).all(1)
            avail[var].append((finite, xx))
            designs[var].append(fixed_design(xx[common], sc))
    rows = []
    for var in ("gebunden", "neu"):
        for j, ms in enumerate(GRID):
            z = designs[var][j]
            b = fit_beta(z, y[common], w[common])
            fin, xx = avail[var][j]
            zb = fixed_design(xx[fin], sc)
            ba = fit_beta(zb, y[fin], w[fin])
            rows.append({"variante": var, "shift_ms": ms, "or": float(np.exp(b[-1])), "beta": float(b[-1]),
                         "n_common": int(common.sum()), "cases_common": int(y[common].sum()),
                         "or_available": float(np.exp(ba[-1])), "n_available": int(fin.sum()),
                         "cases_available": int(y[fin].sum())})
    split = f["split"].to_numpy()
    strata = [np.flatnonzero((split[common] == fold) & (y[common] == c)) for fold in ["train", "validation"] for c in [0, 1]]
    rng = np.random.default_rng(seedseq)
    boots = {"gebunden": [], "neu": []}
    failures = 0
    for b_ in range(nboot):
        if b_ % 25 == 0:
            h244.guard("b_r8")
        counts = np.zeros(int(common.sum()))
        for ix in strata:
            counts += np.bincount(rng.choice(ix, len(ix), replace=True), minlength=len(counts))
        ww = w[common] * counts
        try:
            res = {}
            for var in ("gebunden", "neu"):
                res[var] = [float(fit_beta(z, y[common], ww)[-1]) for z in designs[var]]
            for var in res:
                boots[var].append(res[var])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
    tips = []
    for var in ("gebunden", "neu"):
        beta_pt = [r["beta"] for r in rows if r["variante"] == var]
        bb = np.array(boots[var])
        for r_i, r in enumerate([r for r in rows if r["variante"] == var]):
            r["boot_lo"], r["boot_hi"] = (np.quantile(np.exp(bb[:, r_i]), [.025, .975]).tolist() if len(bb) >= int(.95 * nboot) else [np.nan, np.nan])
        for direction in (-1, 1):
            for target in (1.25, 1.0):
                vals_c = [crossing(v, target, direction) for v in boots[var]]
                finite = [v for v in vals_c if v is not None]
                tips.append({"variante": var, "richtung": "Verkuerzung" if direction < 0 else "Verlaengerung",
                             "ziel_or": target, "kipppunkt_ms": crossing(beta_pt, target, direction),
                             "endliche_kreuzungen": len(finite), "gueltige_fits": len(bb)})
    return rows, tips, len(boots["gebunden"]), failures, common


def synthetic(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(1500), np.zeros(1500)]
    f = pd.DataFrame({"subject_id": np.arange(n), "study_id": np.arange(n) + 10 ** 6,
                      "split": np.where(rng.random(n) < 0.76, "train", "validation"),
                      "age": rng.uniform(20, 90, n), "sex_numeric": rng.integers(0, 2, n).astype(float),
                      "heart_rate": rng.normal(80, 20, n), "qrs_duration_ms": rng.normal(110, 20, n),
                      "qtc_fridericia_ms": rng.normal(430, 35, n), "spatial_qrs_t_angle_deg": rng.uniform(5, 170, n)})
    f["y"] = y
    f[MARKER] = rng.normal(1.0 + 0.5 * y, 0.77)
    f["sampling_weight"] = np.where(y == 1, 2.3, 20.0)
    qt = f["qtc_fridericia_ms"].to_numpy() * rng.uniform(0.8, 1.0, n)
    base = np.column_stack([f[MARKER].to_numpy(), f["spatial_qrs_t_angle_deg"].to_numpy()])
    vals = np.empty((n, len(GRID), 2))
    for j, ms in enumerate(GRID):
        vals[:, j, 0] = base[:, 0] - (0.004 * ms * (y == 1)) + rng.normal(0, 0.01, n)
        vals[:, j, 1] = base[:, 1] + rng.normal(0, 0.1, n)
    vals[rng.choice(np.flatnonzero(y == 1), 40, replace=False), 3, 0] = np.nan
    return f, vals, qt


def test():
    h244.start_watch()
    assert abs(qtc_update(440.0, 400.0, 20) - 462.0) < 1e-9 and qtc_update(440.0, 400.0, 0) == 440.0
    f, vals, qt = synthetic()
    from werkzeuge.b_test.statistik import raw_design, scaling
    xx, names = raw_design(f, MARKER)
    sc = scaling(xx, names)
    y = f["y"].to_numpy(float)
    w = f["sampling_weight"].to_numpy(float)
    rows, tips, valid, fail, common = core(f, vals, qt, sc, 12, y, w, np.random.SeedSequence([1, 2]))
    assert common.sum() == 3000 - 40
    a = [r for r in rows if r["variante"] == "gebunden" and r["shift_ms"] == 0][0]
    b = [r for r in rows if r["variante"] == "neu" and r["shift_ms"] == 0][0]
    assert abs(a["or"] - b["or"]) < 1e-9, "bei Verschiebung 0 muessen beide Varianten gleich sein"
    a8 = [r for r in rows if r["variante"] == "gebunden" and r["shift_ms"] == 32][0]
    b8 = [r for r in rows if r["variante"] == "neu" and r["shift_ms"] == 32][0]
    assert a8["or"] != b8["or"]
    assert valid >= 10
    print("TEST_OK", round(a["or"], 3), round(a8["or"], 3), round(b8["or"], 3), len(tips))


def run():
    import pyarrow.dataset as ds
    from werkzeuge.b_test.statistik import raw_design
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("b_r8")
    h244.start_watch()
    f = pd.read_parquet(A192 / "sample_private.parquet")
    f = f.sort_values("study_id").reset_index(drop=True)
    assert len(f) == 3000 and set(f.split) == {"train", "validation"}
    entry = json.loads((OLD / "prepare_manifest.json").read_text())["entry"]
    sc = entry["scaling_B3"]
    vals = np.array([[json.loads((A192 / "signal_records" / f"{int(s)}.json").read_text())["values"][str(ms)] for ms in GRID]
                     for s in f.study_id], float)
    ids = f.study_id.tolist()
    parts = []
    for p in sorted(Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e_analysis_feature_cache"))).glob("*/split=*/features.parquet")):
        if "split=test" in str(p):
            raise SystemExit("Testpfad")
        parts.append(ds.dataset(str(p), format="parquet").to_table(
            columns=["study_id", "qt_interval_ms", "qtc_fridericia_ms"], filter=ds.field("study_id").isin(ids)).to_pandas())
    c = pd.concat(parts)
    assert c.study_id.is_unique and len(c) == 3000, len(c)
    f = f.merge(c.rename(columns={"qtc_fridericia_ms": "qtc_cache"}), on="study_id", how="left", validate="one_to_one")
    gate_qtc = float(np.nanmax(np.abs(f.qtc_cache - f.qtc_fridericia_ms)))
    assert gate_qtc < 1e-6, f"QTc-Gate {gate_qtc}"
    assert np.isfinite(f.qt_interval_ms).all() and (f.qt_interval_ms > 0).all()
    y = f["y"].to_numpy(float)
    w = f["sampling_weight"].to_numpy(float)
    qt = f["qt_interval_ms"].to_numpy(float)
    rows, tips, valid, fail, common = core(f, vals, qt, sc, NBOOT, y, w, np.random.SeedSequence([19220261013, 2]))
    # Gate: Variante 'gebunden' reproduziert AW-192
    old = json.loads((A192 / "results.json").read_text())
    gates = []
    for r_old in old["rows"]:
        r = [x for x in rows if x["variante"] == "gebunden" and x["shift_ms"] == r_old["shift_ms"]][0]
        gates.append(abs(r["or"] - r_old["or"]))
        if r_old.get("bootstrap_ci") is not None:
            gates.append(abs(r["boot_lo"] - r_old["bootstrap_ci"][0]))
            gates.append(abs(r["boot_hi"] - r_old["bootstrap_ci"][1]))
    gmax = float(max(gates))
    print("Gate max Abweichung AW-192:", gmax, "gueltig", valid, flush=True)
    assert gmax < 1e-6, gmax
    # Ausfallanalyse
    drop = ~common
    cols = ["age", "heart_rate", "qrs_duration_ms", "qtc_fridericia_ms", "t_duration_ms", "amplitude_mv", "flatness_mv_ms",
            "t_vector_magnitude", "spatial_qrs_t_angle_deg", MARKER]
    desc = []
    for c_ in cols:
        for name, m in (("ausgefallene_Faelle", drop & (y == 1)), ("verbliebene_Faelle", common & (y == 1)),
                        ("verbliebene_Kontrollen", common & (y == 0))):
            v = f.loc[m, c_]
            desc.append({"merkmal": c_, "gruppe": name, "n": int(m.sum()), "mittel": float(v.mean()), "sd": float(v.std()),
                         "median": float(v.median())})
    fail_shift = {str(ms): int((drop & ~np.isfinite(vals[:, j, :]).all(1) & (y == 1)).sum()) for j, ms in enumerate(GRID)}
    pd.DataFrame(rows).to_csv(h244.PUB / "b_r8_or_gitter.csv", index=False)
    pd.DataFrame(tips).to_csv(h244.PUB / "b_r8_kipppunkte.csv", index=False)
    pd.DataFrame(desc).to_csv(h244.PUB / "b_r8_ausfaelle_merkmale.csv", index=False)
    meta = {"skript_sha256": shash, "bootstrap_gueltig": valid, "bootstrap_ausfaelle": fail, "gate_max_abw_aw192": gmax,
            "gate_qtc_cache": gate_qtc, "gemeinsame_menge": int(common.sum()), "gemeinsame_faelle": int(y[common].sum()),
            "ausgefallene_faelle": int((drop & (y == 1)).sum()), "ausgefallene_kontrollen": int((drop & (y == 0)).sum()),
            "ausfall_je_shift_nur_wo_nicht_endlich": {k: h244.small(v) for k, v in fail_shift.items()},
            "qt_median_ms": float(np.median(qt)), "qt_p5_p95": np.percentile(qt, [5, 95]).tolist()}
    h244.dump(h244.PUB / "b_r8_meta.json", meta)
    txt = ["B-R8 T-Ende-Kipppunkt mit kohaerentem QTc (post hoc, nur Entwicklung)", json.dumps(meta, indent=1),
           pd.DataFrame(rows).to_string(index=False), pd.DataFrame(tips).to_string(index=False)]
    (h244.PUB / "B_R8_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
