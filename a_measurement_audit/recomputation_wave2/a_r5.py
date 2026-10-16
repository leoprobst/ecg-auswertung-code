"""AW-244 A-R5 (nachtraeglich, post hoc): periodische und kategoriale Marker.

Schritt 1 (Code-Pruefung, Befund): Die Wiederholbarkeit (rv_a1/wiederholbarkeit.py, aw149_repeat.py) bildet Differenzen roh als
y - x; die Matrixklassifikation (rv_a1/transformation.py, metrics) ebenfalls roh (delta = y - x, Raenge und Quintile der
Rohwerte). Der Katalog fuehrt fuer 27 Marker die Regel 'Release_Signed_Grad_oder_atan2' nur als beschreibende gewickelte
Verzerrung/LoA (train_/validation_/pool_wrapped_*), nicht in der Klassenregel. Dies ist im Supplement offengelegt (S2f: 'Original
raw differences for signed periodic descriptors were retained').
Schritt 2 (diese Rechnung):
 (a) Periodische Marker (27 Katalogmarker): lokale Entwicklung der Paare, y'' = x' + wrap(y - x) mit wrap auf [-180,180) und
     x' = wrap(x - mu), mu = zirkulaerer Mittelwert der Referenzwerte (Kors bzw. alle Messwerte der Population); die Klassenregel
     (Spearman, relative LoA-Breite, Quintilwechsel) und ICC/RC/Bias werden unveraendert auf (x', y'') angewandt. Varianten: R (roh,
     Gate gegen Katalogklasse), W (nur gewickelte Differenz), C (Rotation plus gewickelte Differenz). Klassenaenderungen, Katalogzahlen.
     Wiederholbarkeit: ICC, RC, Bias roh (Gate gegen all_metrics) und gewickelt (C) fuer alle Paare und Proxy, Anteil der Paare mit
     |y - x| > 180 Grad, zirkulaere Korrelation.
 (b) Binaere Marker (55 Katalogmarker): Wiederholbarkeit mit Kappa, PABAK, Uebereinstimmung, Praevalenz (Personenbootstrap); Matrixvergleich
     Kors gegen Dower fuer nicht identische binaere Marker mit Kappa/PABAK.
 (c) Bereichsscreen aller admitted Gradmarker (Kors-Werte, Min/Max) auf nicht markierte periodische Kandidaten.
Daten: nur Entwicklung (train, validation). Seeds: Wiederholbarkeit 1.000 Personenziehungen Seed 77010 (gemeinsame Gewichte),
Klassenbootstrap 300 Ziehungen je Fold SeedSequence([244050, fold, k]). Nur Aggregate, keine Kennungen.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import a_r4 as r4  # noqa: E402  (nur Funktionen wstats/estimate_full/person_weights, unveraendert)

B = 1000
SEED = 77010
BCLASS = 300
RULE = (.95, .5, .10, .80, 2., .40)
CL = ["stabil", "bedingt", "transformabhaengig", "nicht_schaetzbar", "konstant_identisch", "nicht_pruefbar"]
VAR = Path(str(data_path("a.transform_katalog_fortsetzung_v3")))
CAT = h244.AUSW / "nachrechnung" / "transform_katalog.csv"
KEY = ["subject_id", "split", "first_study_id", "second_study_id"]


def wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def circmean(a):
    a = np.deg2rad(np.asarray(a, float))
    a = a[np.isfinite(a)]
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))) if len(a) else 0.0


def unwrap_pair(x, y, mu=None):
    xr = np.asarray(x, float) if mu is None else wrap(np.asarray(x, float) - mu)
    return xr, xr + wrap(np.asarray(y, float) - np.asarray(x, float))


def circ_corr(a, b):
    a = np.deg2rad(a)
    b = np.deg2rad(b)
    ma = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    mb = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    sa, sb = np.sin(a - ma), np.sin(b - mb)
    d = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / d) if d > 0 else float("nan")


# ---- Kopie von rv_a1/transformation.py (metrics, classify, combine); unveraendert uebernommen, nicht importiert
def classify(r, w, s, n, valid, rule=RULE):
    a, b, c, d, e, f = rule
    return np.where(valid & (n >= 100), np.where((r >= a) & (w <= b) & (s <= c), 0, np.where((r < d) | (w > e) | (s > f), 2, 1)), 3).astype("int8")


def metrics(x, y, weights):
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    weights = weights[:, good].astype(float)
    if len(x) < 2:
        return np.full((len(weights), 4), np.nan), np.zeros(len(weights)), np.zeros(len(weights), bool)
    cen = (x.mean() + y.mean()) / 2
    scale = max(np.std(np.r_[x, y]), 1e-100)
    x = (x - cen) / scale
    y = (y - cen) / scale

    def ranks(v):
        order = np.argsort(v, kind="stable")
        v0 = v[order]
        starts = np.r_[0, np.flatnonzero(v0[1:] != v0[:-1]) + 1]
        counts = np.add.reduceat(weights[:, order], starts, axis=1)
        r = np.cumsum(counts, axis=1) - counts / 2 + .5
        ix = np.empty(len(order), int)
        ix[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(v)]))
        return r[:, ix]

    n = weights.sum(1)
    safe = np.maximum(n, 1)
    rx, ry = ranks(x), ranks(y)
    mx = (weights * rx).sum(1) / safe
    my = (weights * ry).sum(1) / safe
    dx = rx - mx[:, None]
    dy = ry - my[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (weights * dx * dy).sum(1) / np.sqrt((weights * dx * dx).sum(1) * (weights * dy * dy).sum(1))
        qx = np.clip(np.floor(5 * (rx - .5) / safe[:, None]), 0, 4)
        qy = np.clip(np.floor(5 * (ry - .5) / safe[:, None]), 0, 4)
        switch = (weights * (qx != qy)).sum(1) / safe
        vx = (weights * x * x).sum(1) - (weights * x).sum(1) ** 2 / safe
        vy = (weights * y * y).sum(1) - (weights * y).sum(1) ** 2 / safe
        delta = y - x
        vd = (weights * delta * delta).sum(1) - (weights * delta).sum(1) ** 2 / safe
        wd = 3.92 * np.sqrt(np.maximum(vd, 0) / vx)
        ws = 3.92 * np.sqrt(np.maximum(vd, 0) / ((vx + vy) / 2))
    return np.array([rho, wd, ws, switch]).T, n, (vx > 0) & (vy > 0) & np.isfinite(rho)


def combine2(a, b):
    return np.where((a < 3) & (b < 3), np.maximum(a, b), 3)


def fold_class(x, y, variant, mu_cache=None):
    """Punktklasse (Kors-SD-Normierung) und Kennzahlen eines Folds fuer die Variante R/W/C."""
    good = np.isfinite(x) & np.isfinite(y)
    xg, yg = x[good], y[good]
    if len(xg) >= 2 and np.array_equal(xg, yg):
        return (4 if np.std(xg) == 0 else 0), np.array([1.0, 0.0, 0.0, 0.0])
    if variant == "R":
        a, b = xg, yg
    elif variant == "W":
        a, b = unwrap_pair(xg, yg, None)
    else:
        a, b = unwrap_pair(xg, yg, circmean(xg))
    m, n, valid = metrics(a, b, np.ones((1, len(a))))
    cl = classify(m[:, 0], m[:, 1], m[:, 3], np.array([len(a)]), np.isfinite(m[:, 0]))
    return int(cl[0]), m[0]


def boot_class(x, y, variant, fold, k, B_=BCLASS):
    good = np.isfinite(x) & np.isfinite(y)
    xg, yg = x[good], y[good]
    if variant == "R":
        a, b = xg, yg
    elif variant == "W":
        a, b = unwrap_pair(xg, yg, None)
    else:
        a, b = unwrap_pair(xg, yg, circmean(xg))
    n = len(a)
    rng = np.random.default_rng(np.random.SeedSequence([244050, fold, k]))
    w = np.zeros((B_, n), np.int16)
    for i in range(B_):
        w[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    m, nn, valid = metrics(a, b, w)
    return classify(m[:, 0], m[:, 1], m[:, 3], nn, valid)


def kappa_counts(n11, n10, n01, n00):
    n = n11 + n10 + n01 + n00
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (n11 + n00) / n
        p1 = (n11 + n10) / n
        p2 = (n11 + n01) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)
        kappa = np.where(pe < 1, (po - pe) / (1 - pe), np.nan)
        pabak = 2 * po - 1
        prev = ((n11 + n10) + (n11 + n01)) / (2 * n)
    return kappa, pabak, po, prev


def synthetic_periodic(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    mu = 170.0                          # Verteilung nahe der Schnittstelle bei +-180
    x = wrap(rng.normal(mu, 20, n))
    y = wrap(x + rng.normal(0, 5, n))   # kleine Abweichung, aber Spruenge an der Schnittstelle
    return x, y


def test():
    h244.start_watch()
    assert np.allclose(wrap([-190, 190, 180, -180, 0, 360, 725]), [170, -170, -180, -180, 0, 0, 5])
    x, y = synthetic_periodic()
    raw_w = np.abs(y - x)
    assert (raw_w > 180).sum() > 50, "Test braucht Spruenge an der Schnittstelle"
    cR, mR = fold_class(x, y, "R")
    cC, mC = fold_class(x, y, "C")
    cW, mW = fold_class(x, y, "W")
    # Rohvariante: Spruenge zerstoeren die Uebereinstimmung -> abhaengig; gewickelte Variante: gute Uebereinstimmung
    assert cR == 2 and cC in (0, 1), (cR, cC, mR, mC)
    assert mC[0] > 0.9 and mR[0] < mC[0]
    assert mC[1] < 1.0 and mR[1] > mC[1]
    # Wickelfreie Daten: R == W == C (Klasse)
    rng = np.random.default_rng(9)
    x0 = rng.normal(0, 20, 2000)
    y0 = x0 + rng.normal(0, 3, 2000)
    assert fold_class(x0, y0, "R")[0] == fold_class(x0, y0, "W")[0] == fold_class(x0, y0, "C")[0]
    # zirkulaere Korrelation
    assert circ_corr(x, y) > 0.9
    # Bootstrap-Klasse
    bc = boot_class(x, y, "C", 0, 1, B_=50)
    assert (bc <= 1).mean() > 0.9
    # Kappa
    k, p, po, prev = kappa_counts(np.array([40.0]), np.array([10.0]), np.array([10.0]), np.array([40.0]))
    assert abs(k[0] - 0.6) < 1e-12 and abs(p[0] - 0.6) < 1e-12 and abs(po[0] - 0.8) < 1e-12 and abs(prev[0] - 0.5) < 1e-12
    # Wiederholbarkeit gewickelt: RC der gewickelten Differenz = 1,96 * SD, Rohdifferenz grossraeumig verfaelscht
    xr, yr = unwrap_pair(x, y, circmean(x))
    sd_w = np.std(yr - xr, ddof=1)
    assert 3.5 < sd_w < 7.5 and np.std(y - x, ddof=1) > 20
    print("TEST_OK", cR, cC, cW, round(mR[0], 3), round(mC[0], 3))


def run():
    import pyarrow.parquet as pq
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("a_r5")
    h244.start_watch()
    cat = pd.read_csv(CAT)
    adm = cat[cat.admitted]
    periodic = adm[adm.train_wrapped_rule == "Release_Signed_Grad_oder_atan2"].marker_id.tolist()
    assert len(periodic) == 27, len(periodic)
    degs = adm[adm.unit.isin(["degrees", "deg"])].marker_id.tolist()
    binary = adm[adm.pool_scale_type == "binaer"].marker_id.tolist()
    assert len(binary) == 55 - 0 or len(binary) >= 54, len(binary)
    files = sorted((VAR / "bloecke").glob("*.parquet"))
    assert len(files) == 29
    for p in files:
        seal = __import__("json").loads(p.with_suffix(".json").read_text())
        assert h244.sha(p) == seal["sha256"], f"Block {p.name}"
    need = sorted(set(periodic) | set(degs) | set(binary))
    cols = ["subject_id", "study_id", "split"] + [p + n for p in ("kors__", "dower__") for n in need]
    data = pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in files], ignore_index=True)
    assert len(data) == 5648 and data.subject_id.is_unique
    folds = [data.split.eq(f).to_numpy() for f in ("train", "validation")]
    # (c) Bereichsscreen
    scr = []
    for m in degs:
        v = data["kors__" + m].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < 100:
            continue
        scr.append({"marker_id": m, "markiert_periodisch": m in periodic, "n": int(len(v)), "min": float(v.min()), "max": float(v.max()),
                    "anteil_negativ": float((v < 0).mean()), "anteil_ueber_180": float((v > 180).mean()),
                    "spannweite": float(v.max() - v.min()), "klasse": cat.set_index("marker_id").loc[m, "class"]})
    scr = pd.DataFrame(scr)
    scr["kandidat_unmarkiert"] = (~scr.markiert_periodisch) & (scr["min"] < -90) & (scr["max"] > 90)
    # (a) Matrixklassen
    orig = cat.set_index("marker_id")
    rows, boots = [], []
    for k, m in enumerate(periodic):
        x = data["kors__" + m].to_numpy(float)
        y = data["dower__" + m].to_numpy(float)
        res = {}
        for var in ("R", "W", "C"):
            cl, mm = [], []
            for f in (0, 1):
                c, met = fold_class(x[folds[f]], y[folds[f]], var)
                cl.append(c)
                mm.append(met)
            comb = int(combine2(np.array([cl[0]]), np.array([cl[1]]))[0])
            res[var] = (comb, cl, mm)
        # Gate: Variante R reproduziert die Katalogklasse
        cat_cl = orig.loc[m, "class"]
        cat_idx = CL.index(cat_cl)
        is_const_identical = cat_cl in ("konstant_identisch", "nicht_pruefbar")
        if not is_const_identical:
            assert res["R"][0] == cat_idx or (cat_cl == "nicht_schaetzbar" and res["R"][0] == 3), (m, res["R"][0], cat_cl)
        row = {"marker_id": m, "klasse_katalog": cat_cl}
        for var in ("R", "W", "C"):
            row["klasse_" + var] = CL[res[var][0]] if res[var][0] < len(CL) else str(res[var][0])
            for f, fn in enumerate(("train", "validation")):
                met = res[var][2][f]
                row[f"{var}_{fn}_spearman"] = float(met[0])
                row[f"{var}_{fn}_breite"] = float(met[1])
                row[f"{var}_{fn}_quintilwechsel"] = float(met[3])
        # Bootstrapklassen (ohne konstante identische)
        if res["C"][0] < 3 and res["R"][0] < 3:
            for var in ("R", "C"):
                bc = [boot_class(x[folds[f]], y[folds[f]], var, f, k) for f in (0, 1)]
                cb = combine2(bc[0], bc[1])
                row[f"p_abhaengig_{var}"] = float((cb == 2).mean())
                row[f"p_bedingt_{var}"] = float((cb == 1).mean())
                row[f"p_stabil_{var}"] = float((cb == 0).mean())
                row[f"p_nicht_schaetzbar_{var}"] = float((cb == 3).mean())
        # zirkulaere Korrelation Kors/Dower und gewickelte Verzerrung
        g = np.isfinite(x) & np.isfinite(y)
        row["zirk_korrelation"] = circ_corr(x[g], y[g])
        row["anteil_spruenge_ueber_180"] = float((np.abs(y[g] - x[g]) > 180).mean())
        row["n_paare"] = int(g.sum())
        rows.append(row)
    mt = pd.DataFrame(rows)
    # Katalogzahlen
    base = {"abhaengig": int((cat["class"] == "transformabhaengig").sum()), "bedingt": int((cat["class"] == "bedingt").sum()),
            "stabil_nicht_identisch": 0}
    cnt = []
    for var in ("R", "W", "C"):
        cc = mt["klasse_" + var].value_counts().to_dict()
        wrongly = int((mt["klasse_" + var] != mt["klasse_katalog"]).sum())
        d_dep = int((mt["klasse_katalog"].eq("transformabhaengig") & ~mt["klasse_" + var].eq("transformabhaengig")).sum())
        to_cond = int((mt["klasse_katalog"].eq("transformabhaengig") & mt["klasse_" + var].eq("bedingt")).sum())
        to_stab = int((mt["klasse_katalog"].eq("transformabhaengig") & mt["klasse_" + var].eq("stabil")).sum())
        cnt.append({"variante": var, "klassenwechsel_gegen_katalog": wrongly, "von_abhaengig_weg": d_dep, "zu_bedingt": to_cond, "zu_stabil": to_stab,
                    "abhaengig_neu_gesamt": base["abhaengig"] - d_dep, "bedingt_neu_gesamt": base["bedingt"] + to_cond}
                   )
    cnt = pd.DataFrame(cnt)
    # Wiederholbarkeit und Binaermarker ueber die Paarbloecke
    z = pd.read_parquet(h244.MA / "repeatability" / "paargruppen_privat.parquet", columns=KEY + ["stable"])
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    rng = np.random.default_rng(SEED)
    bootw = np.empty((B, len(ids)))
    for b in range(B):
        bootw[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    pops = {"gesamt": np.ones(len(z)), "proxy": z.stable.to_numpy().astype(float)}
    parts = sorted((h244.RUN / "repeatability" / "replicate_pairs").glob("part-*.parquet"))
    rrows, brow = [], []
    for part in parts:
        pcols = pq.ParquetFile(part).schema_arrow.names
        have = [m for m in periodic + binary if m + "_first" in pcols and m + "_second" in pcols]
        if not have:
            continue
        f = pd.read_parquet(part, columns=KEY + [m + s for m in have for s in ("_first", "_second")])
        f = f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]))
        for m in have:
            x = f[m + "_first"].to_numpy(float)
            y = f[m + "_second"].to_numpy(float)
            for pn, w in pops.items():
                sel = (w > 0) & np.isfinite(x) & np.isfinite(y)
                if sel.sum() < 2:
                    continue
                if m in periodic:
                    mu = circmean(np.r_[x[sel], y[sel]])
                    xr, yr = unwrap_pair(x, y, mu)
                    raw, rawb, nsr, nnr = r4.wstats(x[:, None], y[:, None], pidx, len(ids), w, bootw)
                    cor, corb, nsc, nnc = r4.wstats(xr[:, None], yr[:, None], pidx, len(ids), w, bootw)
                    row = {"marker_id": m, "population": pn, "n_paare": int(nnr[0]), "n_personen": int(nsr[0]), "mu_zirkulaer": mu}
                    for q, nm in enumerate(["icc", "bias", "sd", "rc"]):
                        row[nm + "_roh"] = float(raw[0, q])
                        row[nm + "_gew"] = float(cor[0, q])
                        row[nm + "_roh_lo"], row[nm + "_roh_hi"] = h244.ci(rawb[:, 0, q], minn=int(0.95 * B))
                        row[nm + "_gew_lo"], row[nm + "_gew_hi"] = h244.ci(corb[:, 0, q], minn=int(0.95 * B))
                    row["anteil_spruenge_ueber_180"] = float((np.abs(y[sel] - x[sel]) > 180).mean())
                    row["zirk_korrelation"] = circ_corr(x[sel], y[sel])
                    rrows.append(row)
                if m in binary:
                    xs, ys = x[sel], y[sel]
                    if not np.isin(np.r_[xs, ys], [0.0, 1.0]).all():
                        continue
                    n11 = ((xs == 1) & (ys == 1)).astype(float)
                    n10 = ((xs == 1) & (ys == 0)).astype(float)
                    n01 = ((xs == 0) & (ys == 1)).astype(float)
                    n00 = ((xs == 0) & (ys == 0)).astype(float)
                    tot = []
                    per = []
                    for v in (n11, n10, n01, n00):
                        a = np.zeros(len(ids))
                        np.add.at(a, pidx[sel], v * w[sel])
                        per.append(a)
                    point = kappa_counts(*[a.sum() for a in per])
                    bk = kappa_counts(*[bootw @ a for a in per])
                    nper = int(len(np.unique(pidx[sel])))
                    brow.append({"marker_id": m, "population": pn, "n_paare": int(sel.sum()), "n_personen": nper,
                                 "kappa": float(point[0]), "kappa_lo": h244.ci(bk[0], minn=int(0.95 * B))[0], "kappa_hi": h244.ci(bk[0], minn=int(0.95 * B))[1],
                                 "pabak": float(point[1]), "pabak_lo": h244.ci(bk[1], minn=int(0.95 * B))[0], "pabak_hi": h244.ci(bk[1], minn=int(0.95 * B))[1],
                                 "uebereinstimmung": float(point[2]), "praevalenz": float(point[3]), "positive_uebereinstimmung_paare": h244.small(int(n11[w[sel] > 0].sum())) if False else h244.small(int(n11.sum()))})
        del f
        h244.guard("a_r5")
        print("Block", Path(part).name, flush=True)
    rep = pd.DataFrame(rrows)
    bin_rep = pd.DataFrame(brow)
    # Gate gegen all_metrics (AW-149): ICC/RC/Bias roh
    am = pd.read_csv(Path(str(data_path("a.aw149_all_metrics_csv"))))
    gate = []
    for pn in ("gesamt", "proxy"):
        a = am[am.population == pn].set_index("marker_id")
        r_ = rep[rep.population == pn].set_index("marker_id")
        common = [m for m in r_.index if m in a.index]
        gate.append({"population": pn, "n": len(common), "max_abw_icc": float(np.nanmax(np.abs(r_.loc[common, "icc_roh"] - a.loc[common, "icc"]))),
                     "max_abw_rc_rel": float(np.nanmax(np.abs(r_.loc[common, "rc_roh"] - a.loc[common, "rc"]) / np.maximum(np.abs(a.loc[common, "rc"]), 1e-12)))})
    gate = pd.DataFrame(gate)
    assert (gate.max_abw_icc < 1e-9).all() and (gate.max_abw_rc_rel < 1e-9).all(), gate
    # ICC der Binaermarker aus AW-149 daneben
    if len(bin_rep):
        am2 = am[["population", "marker_id", "icc"]].rename(columns={"icc": "icc_aw149"})
        bin_rep = bin_rep.merge(am2, on=["population", "marker_id"], how="left")
    # Matrixvergleich fuer nicht identische Binaermarker
    mrows = []
    for m in binary:
        cl = orig.loc[m, "class"]
        x = data["kors__" + m].to_numpy(float)
        y = data["dower__" + m].to_numpy(float)
        g = np.isfinite(x) & np.isfinite(y)
        row = {"marker_id": m, "klasse_katalog": cl, "n_paare": int(g.sum())}
        if g.sum() >= 2 and np.isin(np.r_[x[g], y[g]], [0.0, 1.0]).all():
            c = [((x[g] == 1) & (y[g] == 1)).sum(), ((x[g] == 1) & (y[g] == 0)).sum(), ((x[g] == 0) & (y[g] == 1)).sum(), ((x[g] == 0) & (y[g] == 0)).sum()]
            k_, p_, po_, pr_ = kappa_counts(*[np.array([float(v)]) for v in c])
            row.update({"identisch": bool((x[g] == y[g]).all()), "diskordante_paare": h244.small(int(c[1] + c[2])), "kappa": float(k_[0]), "pabak": float(p_[0]),
                        "uebereinstimmung": float(po_[0]), "praevalenz": float(pr_[0])})
        mrows.append(row)
    mb = pd.DataFrame(mrows)
    out = h244.PUB
    mt.to_csv(out / "a_r5_matrix_periodisch.csv", index=False)
    cnt.to_csv(out / "a_r5_katalogzahlen.csv", index=False)
    rep.to_csv(out / "a_r5_wiederholbarkeit_periodisch.csv", index=False)
    bin_rep.to_csv(out / "a_r5_binaer_wiederholbarkeit.csv", index=False)
    mb.to_csv(out / "a_r5_binaer_matrix.csv", index=False)
    scr.to_csv(out / "a_r5_bereichsscreen.csv", index=False)
    gate.to_csv(out / "a_r5_gate.csv", index=False)
    pd.set_option("display.width", 250)
    txt = ["A-R5 periodische und kategoriale Marker (post hoc, nur Entwicklung)", f"Skript-SHA-256 {shash}", "Gate", gate.to_string(index=False),
           "Katalogzahlen", cnt.to_string(index=False), mt.to_string(index=False), rep.to_string(index=False),
           f"Bereichsscreen: Kandidaten unmarkiert {int(scr.kandidat_unmarkiert.sum())}", scr[scr.kandidat_unmarkiert].to_string(index=False),
           bin_rep.to_string(index=False), mb[mb.klasse_katalog != 'stabil'].to_string(index=False)]
    (out / "A_R5_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
