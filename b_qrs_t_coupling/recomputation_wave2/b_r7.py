"""AW-244 B-R7 (nachtraeglich, post hoc): Komparatoren auf identischen Personen.

Frage (Zweitmeinung B, Sollte 2): Traegt der normalisierte diskrete Fréchet-Abstand (QRS- gegen T-Schleife) ueber weitere
Orientierungs- und Repolarisationsdeskriptoren hinaus zur Unterscheidung LVEF < 40 % bei? Hier: Entwicklungsdaten (train, validation),
Endpunkt LVEF < 40 % (BEF-0063/AW-166-Rahmen, 33.971 Personen mit vollstaendigem B3-Design), eine ECG je Person.
Komparatorenbloecke, die dem B3-Design (Alter natuerliche Basis, Geschlecht, HF, QRS, QTcF, integraler Winkel) hinzugefuegt werden:
  K_ang  : Spitzenwinkel QRS-T plus Restricted-Cubic-Spline des integralen Winkels (Knoten 5/35/65/95 %-Quantile der Stichprobe)
  K_axes : sin/cos von Azimut und Elevation der QRS- und T-Flaechenvektoren, log T-Flaechenvektorbetrag (wie S13d 'axes')
  K_svg  : log SVG-Betrag, sin/cos Azimut und Elevation des SVG, SVG-Winkel zur QRS-Flaeche (raeumlicher Ventrikelgradient). Der SVG-Winkel zur T-Flaeche entfaellt:
           SVG = QRS- plus T-Flaechenvektor liegt in deren Ebene, daher ist Winkel(QRS,T) = Winkel(SVG,QRS) + Winkel(SVG,T), eine exakte lineare
           Abhaengigkeit zum integralen Winkel (Rangdefizit beim ersten Lauf, der vor jedem Ergebnis abbrach)
  K_tcrt : TCRT (total_cosine_r_to_t: mittlerer Kosinus zwischen QRS-Schleifenvektoren und mittlerem T-Vektor) und TMD (t_morphology_dispersion_deg)
  K_plan : Planaritaet von T-, QRS- und QRS-T-Schleife (log(1,001 - Planaritaet)) und Winkel der Schleifenebenen
  K_all  : Vereinigung aller Bloecke
Alle Bloecke stammen aus demselben Release-Merkmalscache (Kors-XYZ, 500 Hz); die Stichprobe ist der Durchschnitt der Personen mit
endlichen Werten in allen Bloecken (identische Personen fuer alle Modelle). Skalierung aus dieser Stichprobe, Marker mit der festen
Entwicklungs-SD 0,76503231 (OR je Entwicklungs-SD wie im Manuskript).
Kennzahlen je Block: OR des Fréchet-Abstands (Wald CR1 und Personenbootstrap), scheinbare Delta-AUC (Fréchet zum Blockmodell), AUC-Zuwachs
des Blocks gegen B3 allein, gekreuzt angepasste (zweifach: Training -> Validierung und umgekehrt) Out-of-fold-Delta-AUC mit
Personenbootstrap der gepoolten Vorhersagen (ohne Neuanpassung der Modelle). Bootstrap: 500 Personenziehungen, je Ziehung
SeedSequence([244070, k]); Intervalle bei mindestens 475 gueltigen Fits. Nur Aggregate. Kein Testzugriff.
Gates: Marker-SD 0,76503231; OR und scheinbare Delta-AUC des B3-Modells reproduzieren die gebundenen Entwicklungswerte (AW-166, original_full_reference:
OR 2,48 train / 2,392 validation sind foldweise; hier gepoolt, daher nur Plausibilitaetsgate OR 2,2 bis 2,6).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

sys.path.insert(0, str(h244.AUSW))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MARKER = "qrs_t_normalized_discrete_frechet"
SD_MARKER = 0.7650323096380155
B = 500
CK = h244.PRIV / "b_r7_ckpt.pkl"

COMP_COLS = ["svg_magnitude", "svg_azimuth", "svg_elevation", "svg_angle_to_qrs_area_deg", "svg_angle_to_t_area_deg",
             "t_loop_planarity", "qrs_loop_planarity", "qrst_loop_planarity", "qrs_t_loop_plane_angle_deg",
             "total_cosine_r_to_t", "t_morphology_dispersion_deg", "spatial_peak_qrs_t_angle_deg",
             "qrs_area_vector_azimuth_deg", "qrs_area_vector_elevation_deg", "t_area_vector_azimuth_deg",
             "t_area_vector_elevation_deg", "t_area_vector_magnitude_mv_ms"]


def rcs_basis(x, knots):
    t = np.asarray(knots, float)
    k = len(t)
    pos = lambda u: np.maximum(u, 0.0) ** 3  # noqa: E731
    cols = []
    for j in range(k - 2):
        c = pos(x - t[j]) - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2]) + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        cols.append(c / ((t[k - 1] - t[0]) ** 2))
    return np.column_stack(cols)


def sincos(deg):
    r = np.deg2rad(np.asarray(deg, float))
    return [np.sin(r), np.cos(r)]


def blocks(f, knots):
    """Gibt Dict Blockname -> (Matrix, Spaltennamen) zurueck."""
    ang = np.column_stack([f["spatial_peak_qrs_t_angle_deg"].to_numpy(float), rcs_basis(f["spatial_qrs_t_angle_deg"].to_numpy(float), knots)])
    ax = np.column_stack(sincos(f["qrs_area_vector_azimuth_deg"]) + sincos(f["t_area_vector_azimuth_deg"]) + sincos(f["qrs_area_vector_elevation_deg"])
                         + sincos(f["t_area_vector_elevation_deg"]) + [np.log(f["t_area_vector_magnitude_mv_ms"].to_numpy(float))])
    svg = np.column_stack([np.log(f["svg_magnitude"].to_numpy(float))] + sincos(f["svg_azimuth"]) + sincos(f["svg_elevation"])
                          + [f["svg_angle_to_qrs_area_deg"].to_numpy(float)])
    tc = np.column_stack([f["total_cosine_r_to_t"].to_numpy(float), f["t_morphology_dispersion_deg"].to_numpy(float)])
    pl = np.column_stack([np.log(1.001 - f["t_loop_planarity"].to_numpy(float)), np.log(1.001 - f["qrs_loop_planarity"].to_numpy(float)),
                          np.log(1.001 - f["qrst_loop_planarity"].to_numpy(float)), f["qrs_t_loop_plane_angle_deg"].to_numpy(float)])
    out = {"K0_B3": np.empty((len(f), 0)), "K_ang": ang, "K_axes": ax, "K_svg": svg, "K_tcrt": tc, "K_plan": pl}
    out["K_all"] = np.column_stack([ang, ax, svg, tc, pl])
    return out


def build(f, extra, params=None):
    from werkzeuge.b_test.statistik import raw_design, scaling, fixed_design
    xx, names = raw_design(f, MARKER)
    x = np.column_stack([xx[:, :-1], extra, xx[:, -1]])
    names = names[:-1] + [f"extra{i}" for i in range(extra.shape[1])] + [MARKER]
    if params is None:
        params = scaling(x, names)
        params = dict(params)
        sd = np.asarray(params["sd"], float)
        sd[-1] = SD_MARKER
        params["sd"] = sd.tolist()
    return fixed_design(x, params), params


def oof_predictions(z, y, fold):
    from werkzeuge.nachrechnung.core import logistic_fit
    full = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for a, b in (("train", "validation"), ("validation", "train")):
        tr, te = fold == a, fold == b
        w = np.ones(tr.sum())
        bf = logistic_fit(z[tr], y[tr], w)
        bb = logistic_fit(z[tr][:, :-1], y[tr], w)
        full[te] = z[te] @ bf
        base[te] = z[te][:, :-1] @ bb
    return full, base


def fit_beta(z, y, w):
    from werkzeuge.nachrechnung.core import logistic_fit
    from scipy.special import expit
    b = logistic_fit(z, y, w)
    pen = np.r_[0, np.full(z.shape[1] - 1, 1e-6)]
    sc = np.max(np.abs(z.T @ (w * (expit(z @ b) - y)) + pen * b)) / w.sum()
    if not np.isfinite(b).all() or sc >= 1e-7:
        raise ValueError("convergence")
    return b


def one_draw(designs, y, k):
    from werkzeuge.nachrechnung.core import auc
    rng = np.random.default_rng(np.random.SeedSequence([244070, k]))
    n = len(y)
    w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    res = {}
    b3 = fit_beta(designs["K0_B3"][:, :-1], y, w)
    auc_b3 = auc(y, designs["K0_B3"][:, :-1] @ b3, w)
    for name, z in designs.items():
        bf = fit_beta(z, y, w)
        bb = b3 if name == "K0_B3" else fit_beta(z[:, :-1], y, w)
        a_full = auc(y, z @ bf, w)
        a_base = auc(y, z[:, :-1] @ bb, w)
        res[name] = (float(np.exp(bf[-1])), a_full - a_base, a_base - auc_b3)
    return res


def synthetic(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"subject_id": np.arange(n), "age": rng.uniform(20, 90, n), "sex_numeric": rng.integers(0, 2, n).astype(float),
                      "heart_rate": rng.normal(80, 20, n), "qrs_duration_ms": rng.normal(105, 20, n), "qtc_fridericia_ms": rng.normal(430, 35, n),
                      "split": np.where(rng.random(n) < 0.75, "train", "validation")})
    u = rng.normal(0, 1, n)                                      # gemeinsame latente Groesse
    f["spatial_qrs_t_angle_deg"] = np.clip(70 + 30 * u + rng.normal(0, 10, n), 1, 179)
    f[MARKER] = 1.6 + 0.76 * (0.7 * u + 0.7 * rng.normal(0, 1, n))
    lp = -2.4 + 0.5 * u + 0.35 * (f[MARKER] - 1.6) / 0.76
    f["y"] = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(float)
    f["spatial_peak_qrs_t_angle_deg"] = np.clip(70 + 25 * u + rng.normal(0, 15, n), 1, 179)
    for c in ("qrs_area_vector_azimuth_deg", "t_area_vector_azimuth_deg"):
        f[c] = rng.uniform(-179, 179, n)
    for c in ("qrs_area_vector_elevation_deg", "t_area_vector_elevation_deg"):
        f[c] = rng.uniform(5, 175, n)
    f["t_area_vector_magnitude_mv_ms"] = np.exp(rng.normal(3, 0.5, n))
    f["svg_magnitude"] = np.exp(rng.normal(10.5, 1, n))
    f["svg_azimuth"] = rng.uniform(-179, 179, n)
    f["svg_elevation"] = rng.uniform(5, 175, n)
    f["svg_angle_to_qrs_area_deg"] = rng.uniform(0, 179, n)
    f["svg_angle_to_t_area_deg"] = rng.uniform(0, 179, n)
    for c in ("t_loop_planarity", "qrs_loop_planarity", "qrst_loop_planarity"):
        f[c] = 1 - np.exp(rng.normal(-4.5, 0.8, n))
    f["qrs_t_loop_plane_angle_deg"] = rng.uniform(0, 89, n)
    f["total_cosine_r_to_t"] = np.clip(0.9 - 0.012 * f["spatial_qrs_t_angle_deg"] + rng.normal(0, 0.1, n), -1, 1)   # TCRT ~ Funktion des Winkels
    f["t_morphology_dispersion_deg"] = rng.uniform(5, 80, n)
    return f


def test():
    h244.start_watch()
    # RCS: lineare Funktion liegt im Spann (lineare Spalte + nichtlineare Terme -> Residuum der linearen Anpassung verschwindet nicht,
    # aber nichtlineare Terme muessen jenseits der aeusseren Knoten linear sein)
    knots = np.array([10., 30., 50., 70.])
    xs = np.array([80., 90., 100., 110.])
    b = rcs_basis(xs, knots)
    d2 = np.diff(b, 2, axis=0)
    assert np.allclose(d2, 0, atol=1e-9), "RCS jenseits des letzten Knotens nicht linear"
    assert np.allclose(rcs_basis(np.array([-5., -2., 3.]), knots), 0.0), "RCS vor dem ersten Knoten nicht null"
    f = synthetic()
    q = np.quantile(f["spatial_qrs_t_angle_deg"], [.05, .35, .65, .95])
    bl = blocks(f, q)
    assert bl["K_all"].shape[1] == sum(bl[k].shape[1] for k in ("K_ang", "K_axes", "K_svg", "K_tcrt", "K_plan"))
    y = f["y"].to_numpy(float)
    designs = {}
    params = None
    for name, ex in bl.items():
        z, p = build(f, ex)
        designs[name] = z
    assert abs(designs["K0_B3"][:, -1].std() - np.std(f[MARKER]) / SD_MARKER) < 1e-9
    r = one_draw(designs, y, 1)
    assert 1.1 < r["K0_B3"][0] < 2.5
    # Block mit starkem Ersatz fuer den Marker senkt dessen OR
    fz = f.copy()
    fz["total_cosine_r_to_t"] = (fz[MARKER] - 1.6) / 0.76 + np.random.default_rng(1).normal(0, 0.1, len(fz))
    bl2 = blocks(fz, q)
    z0, _ = build(fz, bl2["K0_B3"])
    z1, _ = build(fz, bl2["K_tcrt"])
    b0 = fit_beta(z0, y, np.ones(len(y)))
    b1 = fit_beta(z1, y, np.ones(len(y)))
    assert np.exp(b1[-1]) < np.exp(b0[-1]) - 0.05, (np.exp(b0[-1]), np.exp(b1[-1]))
    fold = f["split"].to_numpy()
    full, base = oof_predictions(designs["K_all"], y, fold)
    assert np.isfinite(full).all() and np.isfinite(base).all()
    from werkzeuge.nachrechnung.core import auc
    assert 0.5 < auc(y, full) < 1 and abs(auc(y, full) - auc(y, base)) < 0.1
    print("TEST_OK", round(np.exp(b0[-1]), 3), round(np.exp(b1[-1]), 3))


def run():
    import pyarrow.dataset as ds
    from werkzeuge.b_test.statistik import fit, raw_design
    from werkzeuge.nachrechnung.core import auc
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("b_r7")
    h244.start_watch()
    popf = Path(str(data_path("b.aw166_population_private_parquet")))
    pop = pd.read_parquet(popf)
    assert set(pop.split) == {"train", "validation"} and len(pop) == 33971 and not pop.subject_id.duplicated().any()
    sd_gate = float(np.std(pop[MARKER].to_numpy(float), ddof=0))
    assert abs(sd_gate - SD_MARKER) < 1e-6, sd_gate
    parts = []
    for p in sorted((h244.RUN / "analysis_feature_cache").glob("*/split=*/features.parquet")):
        if "split=test" in str(p):
            raise SystemExit("Testpfad")
        parts.append(ds.dataset(str(p), format="parquet").to_table(columns=["study_id"] + COMP_COLS,
                                                                   filter=ds.field("study_id").isin(pop.study_id.tolist())).to_pandas())
    c = pd.concat(parts)
    assert c.study_id.is_unique
    # Gate: gemeinsame Spalten des Caches stimmen mit der Population ueberein (Winkel)
    f = pop.merge(c.rename(columns={"spatial_peak_qrs_t_angle_deg": "spatial_peak_qrs_t_angle_deg"}), on="study_id", how="left", validate="one_to_one")
    finite_all = np.isfinite(f[COMP_COLS]).all(axis=1) & (f["t_loop_planarity"] < 1.0001)
    f = f[finite_all].reset_index(drop=True)
    n_cases = int(f.y.sum())
    knots = np.quantile(f["spatial_qrs_t_angle_deg"].to_numpy(float), [.05, .35, .65, .95])
    bl = blocks(f, knots)
    y = f["y"].to_numpy(float)
    fold = f["split"].to_numpy()
    subj = f["subject_id"].to_numpy()
    designs, params = {}, {}
    for name, ex in bl.items():
        designs[name], params[name] = build(f, ex)
    rows = []
    b3_apparent = None
    for name, z in designs.items():
        out = fit(z, y, subj)
        full, base = oof_predictions(z, y, fold)
        row = {"block": name, "n": int(len(y)), "faelle": n_cases, "spalten_block": int(z.shape[1] - 1 - bl["K0_B3"].shape[1] - 1 - 0) if False else int(bl[name].shape[1]),
               "or_wald": out["or"], "or_wald_lo": out["or_ci95_wald"][0], "or_wald_hi": out["or_ci95_wald"][1],
               "dauc_scheinbar": out["delta_auc"], "auc_basis_mit_block": out["auc_G"], "auc_voll": out["auc_G_marker"],
               "auc_oof_basis": float(auc(y, base)), "auc_oof_voll": float(auc(y, full)), "dauc_oof": float(auc(y, full) - auc(y, base))}
        rows.append(row)
        if name == "K0_B3":
            b3_apparent = out["auc_G"]
        row["_oof"] = (full, base)
    # AUC-Zuwachs des Blocks gegen B3 allein (scheinbar und out-of-fold)
    b3row = [r for r in rows if r["block"] == "K0_B3"][0]
    for r in rows:
        r["dauc_block_scheinbar"] = r["auc_basis_mit_block"] - b3row["auc_basis_mit_block"]
        r["dauc_block_oof"] = r["auc_oof_basis"] - b3row["auc_oof_basis"]
    # Out-of-fold-Bootstrap (ohne Neuanpassung)
    rng = np.random.default_rng(np.random.SeedSequence([244071]))
    n = len(y)
    idx = np.array([rng.integers(0, n, n) for _ in range(1000)])
    for r in rows:
        full, base = r.pop("_oof")
        d = []
        d2 = []
        for ix in idx:
            d.append(auc(y[ix], full[ix]) - auc(y[ix], base[ix]))
        r["dauc_oof_lo"], r["dauc_oof_hi"] = np.percentile(d, [2.5, 97.5])
        r["_oof_base"] = base
    b3_base = [r for r in rows if r["block"] == "K0_B3"][0]["_oof_base"]
    for r in rows:
        d = [auc(y[ix], r["_oof_base"][ix]) - auc(y[ix], b3_base[ix]) for ix in idx]
        r["dauc_block_oof_lo"], r["dauc_block_oof_hi"] = np.percentile(d, [2.5, 97.5])
        r.pop("_oof_base")
    # Personenbootstrap mit Neuanpassung (OR, scheinbare Delta-AUC), Wiederaufnahme ueber Checkpoint
    done = {}
    if CK.exists():
        cp = pickle.loads(CK.read_bytes())
        if cp["script"] == shash:
            done = cp["done"]
    failures = 0
    for k in range(B):
        if k in done:
            continue
        try:
            done[k] = one_draw(designs, y, k)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            done[k] = None
        if k % 25 == 0:
            CK.write_bytes(pickle.dumps({"script": shash, "done": done}))
            h244.guard("b_r7")
            print("Ziehung", k, flush=True)
    CK.write_bytes(pickle.dumps({"script": shash, "done": done}))
    valid = [v for v in done.values() if v is not None]
    for r in rows:
        nm = r["block"]
        orv = np.array([v[nm][0] for v in valid])
        dv = np.array([v[nm][1] for v in valid])
        bv = np.array([v[nm][2] for v in valid])
        ok = len(valid) >= int(0.95 * B)
        r["boot_gueltig"] = len(valid)
        r["or_boot_lo"], r["or_boot_hi"] = (np.percentile(orv, [2.5, 97.5]) if ok else (np.nan, np.nan))
        r["dauc_boot_lo"], r["dauc_boot_hi"] = (np.percentile(dv, [2.5, 97.5]) if ok else (np.nan, np.nan))
        r["dauc_block_boot_lo"], r["dauc_block_boot_hi"] = (np.percentile(bv, [2.5, 97.5]) if ok else (np.nan, np.nan))
    df = pd.DataFrame(rows)
    # Gate: B3-Modell plausibel (gepoolter OR zwischen den foldweisen AW-166-Werten, Toleranz)
    g = float(b3row["or_wald"])
    assert 2.2 < g < 2.6, g
    df.to_csv(h244.PUB / "b_r7_ergebnisse.csv", index=False)
    meta = {"skript_sha256": shash, "n": int(len(y)), "faelle": n_cases, "verlust_nicht_endlich": int(len(pop) - len(f)), "knoten_winkel": knots.tolist(),
            "marker_sd_gate": sd_gate, "boot_ausfaelle": int(sum(v is None for v in done.values()))}
    h244.dump(h244.PUB / "b_r7_meta.json", meta)
    (h244.PUB / "B_R7_ausgabe.txt").write_text("B-R7 Komparatoren (post hoc, nur Entwicklung)\n" + str(meta) + "\n" + df.to_string(index=False), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
