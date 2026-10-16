"""AW-244 A-R6 (nachtraeglich, post hoc): Replikation, Entwicklungsunsicherheit und Zusammensetzung der Testkohorte.

Die Entscheidungen der Replikationstore (AW-163, S13) bleiben unveraendert. Ergaenzend, zur Erklaerung der Abweichung
(Test-Median-ICC 0,507 gegen Entwicklungs-Benchmark 0,467, Band +-0,05):
 Teil A (nur vorhandene Aggregate, markerweise): zerlegt den Unterschied des Median-ICC in Zwischenpersonenvarianz und
   Fehlervarianz. Aus RC (= 1,96 * SD der zentrierten Paardifferenzen) folgt die Fehlervarianz je Messung s2_w = (RC/1,96)^2 / 2,
   aus dem berichteten ICC folgt s2_b = ICC / (1 - ICC) * s2_w. Gegenfaktische ICC: (s2_b Test, s2_w Entwicklung) und
   (s2_b Entwicklung, s2_w Test); Median ueber die gemeinsamen Marker (2.003 Marker der Replikation). Beide Unsicherheiten:
   Entwicklung aus den gespeicherten 1.000 Personenziehungen (Median ueber die 2.003 Marker), Test aus dem berichteten
   95-%-Intervall des Median-ICC (Normalnaeherung, SE = Breite / 3,92); die Differenz Test minus Entwicklung unter
   Unabhaengigkeit beider Schaetzer (verschiedene Personen).
 Teil B (Zusammensetzung): Randverteilung der Paare nach Abstandsklasse (0 bis 15 min, 15 bis 60 min, 60 bis 120 min),
   Rhythmusklasse und QRS-Klasse in der Testkohorte (3.897 Paare, nur diese drei Paarspalten; keine Markerwerte, keine
   Kennungen in der Ausgabe) gegen die Entwicklung; Standardisierung der Entwicklungspaare auf die Zellverteilung der
   Testkohorte (Gewicht = Testanteil / Entwicklungsanteil je Zelle, fest, nicht neu geschaetzt), Median-ICC der standardisierten
   Entwicklung mit Personenbootstrap (1.000 Ziehungen, Seed 77010, gemeinsame Gewichte mit der Entwicklungsanalyse).
Testzugriff: nur Aggregatdateien und die Paarliste der Testreplikation AW-163 (<local path>), Eintrag in
koordination/TESTZUGRIFFE.md vor und nach dem Lesen. Altersangaben der Testkohorte liegen in diesen Dateien nicht vor und
wurden nicht gesucht. locked_test selbst wird nicht geoeffnet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

B = 1000
SEED = 77010
AW163 = Path(str(data_path("a.aw163")))
TZ = h244.AUSW / "koordination" / "TESTZUGRIFFE.md"
GAP_BINS = [("gap_0bis15min", 0.0, 0.25), ("gap_15bis60min", 0.25, 1.0), ("gap_60bis120min", 1.0, 2.0)]


def tz(event, vorschrift, umfang):
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"| {utc} | AW-244 | {event.replace('|', '/')} | {vorschrift.replace('|', '/')} | {umfang.replace('|', '/')} |\n"
    with open(TZ, "a", encoding="utf8", newline="") as fh:
        fh.write(line)
    return line.strip()


def variance_parts(icc, rc):
    """Implizite Varianzkomponenten aus ICC und RC. Rueckgabe (s2_b, s2_w); NaN wenn nicht definiert."""
    icc = np.asarray(icc, float)
    rc = np.asarray(rc, float)
    s2w = (rc / 1.96) ** 2 / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s2b = np.where((icc > 0) & (icc < 1) & (s2w > 0), icc / (1 - icc) * s2w, np.nan)
    s2w = np.where(np.isfinite(s2b), s2w, np.nan)
    return s2b, s2w


def decompose(icc_d, rc_d, icc_t, rc_t):
    sbd, swd = variance_parts(icc_d, rc_d)
    sbt, swt = variance_parts(icc_t, rc_t)
    ok = np.isfinite(sbd) & np.isfinite(swd) & np.isfinite(sbt) & np.isfinite(swt)
    icc_cf_b = sbt / (sbt + swd)      # nur Zwischenpersonenvarianz wie im Test
    icc_cf_w = sbd / (sbd + swt)      # nur Fehlervarianz wie im Test
    icc_d_i = sbd / (sbd + swd)
    icc_t_i = sbt / (sbt + swt)
    med = lambda a: float(np.nanmedian(np.where(ok, a, np.nan)))  # noqa: E731
    out = {"n_marker": int(ok.sum()),
           "median_icc_dev": med(icc_d_i), "median_icc_test": med(icc_t_i),
           "median_icc_test_sb_dev_sw": med(icc_cf_b), "median_icc_dev_sb_test_sw": med(icc_cf_w),
           "median_log_ratio_sb": float(np.nanmedian(np.where(ok, np.log(sbt / sbd), np.nan))),
           "median_log_ratio_sw": float(np.nanmedian(np.where(ok, np.log(swt / swd), np.nan))),
           "median_ratio_sd_b": float(np.exp(0.5 * np.nanmedian(np.where(ok, np.log(sbt / sbd), np.nan)))),
           "median_ratio_sd_w": float(np.exp(0.5 * np.nanmedian(np.where(ok, np.log(swt / swd), np.nan))))}
    out["delta_total"] = out["median_icc_test"] - out["median_icc_dev"]
    out["delta_nur_zwischenperson"] = out["median_icc_test_sb_dev_sw"] - out["median_icc_dev"]
    out["delta_nur_fehler"] = out["median_icc_dev_sb_test_sw"] - out["median_icc_dev"]
    return out, ok


# ---- gewichtete Momente (wie aw149_repeat.stats, mit Paargewichten)
def estimate(n, sx, sy, sxx, syy, sxy):
    with np.errstate(divide="ignore", invalid="ignore"):
        col = (sx - sy) ** 2 / (2 * n)
        msr = ((sxx + 2 * sxy + syy) / 2 - (sx + sy) ** 2 / (2 * n)) / (n - 1)
        mse = ((sxx + syy - 2 * sxy) / 2 - col) / (n - 1)
        den = msr + mse + 2 * (col - mse) / n
        return np.where((n >= 2) & (den > 0), np.clip((msr - mse) / den, -1, 1), np.nan)


def wstats(x, y, person_idx, n_persons, wpair, bootw):
    ok = np.isfinite(x) & np.isfinite(y) & (wpair[:, None] > 0)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(np.where(ok, (x + y) / 2, np.nan), axis=0)
        scale = np.sqrt(np.nanmean(np.where(ok, ((x - center) ** 2 + (y - center) ** 2) / 2, np.nan), axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    xx = np.where(ok, (x - center) / scale, 0.0)
    yy = np.where(ok, (y - center) / scale, 0.0)
    w = wpair[:, None]
    agg = []
    for v in [ok.astype(float) * w, xx * w, yy * w, xx * xx * w, yy * yy * w, xx * yy * w]:
        a = np.zeros((n_persons, x.shape[1]))
        np.add.at(a, person_idx, v)
        agg.append(a)
    has = np.zeros((n_persons, x.shape[1]))
    np.add.at(has, person_idx, ok.astype(float))
    has = (has > 0).astype(float)
    ns = has.sum(0)
    pt = estimate(*[a.sum(0) for a in agg])
    pt = np.where(ns >= 30, pt, np.nan)
    bb = None
    if bootw is not None:
        bb = np.empty((bootw.shape[0], x.shape[1]))
        for lo in range(0, bootw.shape[0], 50):
            sl = bootw[lo:lo + 50]
            b = estimate(*[sl @ a for a in agg])
            eff = (sl > 0).astype(float) @ has
            b[eff < 30] = np.nan
            bb[lo:lo + 50] = b
    return pt, bb


def cell_labels(rhythm, qrs_class, hours):
    gap = np.full(len(hours), "", dtype=object)
    for name, lo, hi in GAP_BINS:
        m = (hours >= lo) & (hours <= hi) & (gap == "")
        if name != "gap_0bis15min":
            m = (hours > lo) & (hours <= hi) & (gap == "")
        gap[m] = name
    return pd.Series(gap).astype(str) + "|" + pd.Series(np.asarray(rhythm)).astype(str) + "|" + pd.Series(np.asarray(qrs_class)).astype(str)


def cell_weights(dev_cells, test_cells):
    pd_ = dev_cells.value_counts(normalize=True)
    pt_ = test_cells.value_counts(normalize=True)
    w = dev_cells.map(lambda c: (pt_.get(c, 0.0) / pd_[c]) if c in pd_.index else 0.0).to_numpy(float)
    unmatched = float(sum(v for c, v in pt_.items() if c not in pd_.index))
    return w, unmatched


def synthetic_test():
    rng = np.random.default_rng(1)
    n = 3000
    persons = rng.integers(0, 1200, n)
    sb = 1.0
    true = rng.normal(0, sb, 1200)[persons]
    cellA = rng.random(n) < 0.5                 # Zelle A: Fehler-SD 0.5, Zelle B: Fehler-SD 1.5
    e = np.where(cellA, 0.5, 1.5)
    x = (true + rng.normal(0, 1, n) * e)[:, None]
    y = (true + rng.normal(0, 1, n) * e)[:, None]
    return persons, x, y, cellA


def test():
    h244.start_watch()
    # 1. gewichtete Momente mit Gewicht 1 gleich unabhaengiger ANOVA (zwei Personenklassen unbalanciert egal: n gross)
    persons, x, y, cellA = synthetic_test()
    uniq, pidx = np.unique(persons, return_inverse=True)
    w1 = np.ones(len(x))
    pt, _ = wstats(x, y, pidx, len(uniq), w1, None)
    # unabhaengige Pruefung: ICC(A,1) aus Zwei-Faktoren-ANOVA mit den Paaren als Einheiten (Paar = Subjekt) nicht gleich,
    # daher Vergleich mit direkter Momentenformel auf demselben Datensatz
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, syy, sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()
    ref = estimate(np.array([n]), np.array([sx]), np.array([sy]), np.array([sxx]), np.array([syy]), np.array([sxy]))[0]
    assert abs(pt[0] - ref) < 1e-9, (pt, ref)
    # 2. Standardisierung: nur Zelle A -> ICC = 1/(1+0.25) = 0.8; nur Zelle B -> 1/(1+2.25) = 0.3077
    wa = cellA.astype(float)
    wb = (~cellA).astype(float)
    pa, _ = wstats(x, y, pidx, len(uniq), wa, None)
    pb, _ = wstats(x, y, pidx, len(uniq), wb, None)
    assert abs(pa[0] - 0.8) < 0.07 and abs(pb[0] - 0.3077) < 0.07, (pa, pb)
    # 3. Bootstrap-Gewichte
    rng = np.random.default_rng(5)
    bw = np.array([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(100)], float)
    pt2, bb = wstats(x, y, pidx, len(uniq), w1, bw)
    assert np.isfinite(bb).all() and abs(np.median(bb) - pt2[0]) < 0.05
    # 4. Zerlegung: gleiche Fehlervarianz, doppelte Zwischenvarianz -> Aenderung nur ueber Zwischenperson
    icc_d = np.array([0.5, 0.4, 0.6]); rc_d = np.array([2.0, 3.0, 1.5])
    sbd, swd = variance_parts(icc_d, rc_d)
    sbt = 2 * sbd
    icc_t = sbt / (sbt + swd)
    out, ok = decompose(icc_d, rc_d, icc_t, rc_d)
    assert abs(out["delta_total"] - out["delta_nur_zwischenperson"]) < 1e-12 and abs(out["delta_nur_fehler"]) < 1e-12, out
    # 5. Zellgewichte
    dev_c = pd.Series(["a"] * 60 + ["b"] * 40)
    test_c = pd.Series(["a"] * 20 + ["b"] * 70 + ["c"] * 10)
    w, un = cell_weights(dev_c, test_c)
    assert abs(un - 0.1) < 1e-12 and abs(w[0] - (0.2 / 0.6)) < 1e-12 and abs(w[-1] - (0.7 / 0.4)) < 1e-12
    print("TEST_OK", round(pa[0], 3), round(pb[0], 3))


def run():
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("a_r6")
    h244.start_watch()
    fa = AW163 / "repeat_marker.csv"
    fb = AW163 / "development_W_gesamt.csv"
    fp = AW163 / "W_pairs_after_gates_private.parquet"
    ha, hb, hp = h244.sha(fa), h244.sha(fb), h244.sha(fp)
    print(tz("Beginn A-R6 (post hoc): Lesen von Aggregatdateien und Paarspalten der Testreplikation AW-163",
             f"Skript a_r6.py {shash}; Freigabe: Auftrag AW-244 (K), Eintrag nach Regel",
             f"Teil A: repeat_marker.csv (Test-Markeraggregate, Population gesamt) {ha}, development_W_gesamt.csv {hb}; Teil B: aus W_pairs_after_gates_private.parquet {hp} nur Spalten hours_between, rhythm, qrs_class, pacing, stable, year (keine Markerwerte, keine Kennungen in der Ausgabe); locked_test nicht geoeffnet"), flush=True)
    test_m = pd.read_csv(fa)
    test_m = test_m[test_m.population == "gesamt"].set_index("marker_id")
    dev_m = pd.read_csv(fb)
    dev_m = dev_m[dev_m.population == "gesamt"].set_index("marker_id")
    common = [m for m in dev_m.index if m in test_m.index]
    # Teil A
    icc_d = dev_m.loc[common, "icc"].to_numpy(float)
    rc_d = dev_m.loc[common, "rc"].to_numpy(float)
    icc_t = test_m.loc[common, "icc"].to_numpy(float)
    rc_t = test_m.loc[common, "rc"].to_numpy(float)
    dec, ok = decompose(icc_d, rc_d, icc_t, rc_t)
    dec_all = {"median_icc_dev_alle_schaetzbaren": float(np.nanmedian(icc_d)), "median_icc_test_alle_schaetzbaren": float(np.nanmedian(icc_t)),
               "n_gemeinsame_marker": len(common), "n_beide_endlich": int((np.isfinite(icc_d) & np.isfinite(icc_t)).sum())}
    # Entwicklungs-Bootstrap des Medians aus den gespeicherten Ziehungen (gesamt = Index 0)
    bm = []
    names = []
    boots = []
    for p in sorted((Path(str(data_path("a.rv_a1_repeat")))).glob("part-*.npz")):
        v = np.load(p, allow_pickle=False)
        names.extend(v["names"].tolist())
        boots.append(v["boot"][0])
    bd = np.concatenate(boots, axis=1)   # B x Marker (ICC der Population gesamt)
    idx = [names.index(m) for m in common]
    bd = bd[:, idx]
    use = np.isfinite(icc_d) & np.isfinite(icc_t)
    med_dev_draws = np.nanmedian(bd[:, use], axis=1)
    se_dev = float(np.std(med_dev_draws, ddof=1))
    ci_dev = h244.ci(med_dev_draws)
    t_lo, t_hi = 0.48931274, 0.53740378      # berichtetes Test-Intervall des Median-ICC (W_gesamt, 2003 Marker)
    se_test = (t_hi - t_lo) / 3.92
    med_t = float(np.nanmedian(icc_t))
    med_d = float(np.nanmedian(icc_d))
    diff = med_t - med_d
    se = float(np.hypot(se_dev, se_test))
    teilA = {**dec, **dec_all, "se_dev": se_dev, "ci_dev_median": ci_dev, "se_test_aus_ki": se_test,
             "diff_test_minus_dev_median_icc": diff, "diff_ci95_normal": [diff - 1.96 * se, diff + 1.96 * se],
             "band_dev_benchmark": [med_d - 0.05, med_d + 0.05]}
    # Familien/Einheiten: Median log-Verhaeltnis RC Test/Entwicklung
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(rc_t / rc_d)
    teilA["median_rc_verhaeltnis_test_zu_dev"] = float(np.exp(np.nanmedian(lr[np.isfinite(lr)])))
    teilA["anteil_marker_icc_test_hoeher"] = float(np.mean(icc_t[use] > icc_d[use]))
    h244.dump(h244.PUB / "a_r6_teilA.json", teilA)
    # Teil B
    pairs_test = pd.read_parquet(fp, columns=["hours_between", "rhythm", "qrs_class", "pacing", "stable", "year"])
    z = pd.read_parquet(h244.MA / "repeatability" / "paargruppen_privat.parquet",
                        columns=["subject_id", "split", "first_study_id", "second_study_id", "hours_between", "rhythm", "qrs_class", "pacing", "stable", "year"])
    assert len(z) == 15693 and len(pairs_test) == 3897
    tcell = cell_labels(pairs_test.rhythm, pairs_test.qrs_class, pairs_test.hours_between.to_numpy(float))
    dcell = cell_labels(z.rhythm, z.qrs_class, z.hours_between.to_numpy(float))
    wcell, unmatched = cell_weights(dcell, tcell)
    comp = []
    for var in ("rhythm", "qrs_class", "pacing", "stable", "year"):
        a = z[var].astype(str).value_counts(normalize=True)
        b = pairs_test[var].astype(str).value_counts(normalize=True)
        na = z[var].astype(str).value_counts()
        nb = pairs_test[var].astype(str).value_counts()
        for lev in sorted(set(a.index) | set(b.index)):
            comp.append({"variable": var, "stufe": lev, "anteil_entwicklung": float(a.get(lev, 0.0)), "anteil_test": float(b.get(lev, 0.0)),
                         "n_entwicklung": h244.small(int(na.get(lev, 0))), "n_test": h244.small(int(nb.get(lev, 0)))})
    gd = pd.cut(z.hours_between, [-0.001, 0.25, 1.0, 2.0], labels=["0-15min", "15-60min", "60-120min"]).astype(str).value_counts(normalize=True)
    gt = pd.cut(pairs_test.hours_between, [-0.001, 0.25, 1.0, 2.0], labels=["0-15min", "15-60min", "60-120min"]).astype(str).value_counts(normalize=True)
    for lev in ["0-15min", "15-60min", "60-120min"]:
        comp.append({"variable": "abstand", "stufe": lev, "anteil_entwicklung": float(gd.get(lev, 0)), "anteil_test": float(gt.get(lev, 0)),
                     "n_entwicklung": "", "n_test": ""})
    comp.append({"variable": "abstand", "stufe": "median_h", "anteil_entwicklung": float(z.hours_between.median()),
                 "anteil_test": float(pairs_test.hours_between.median()), "n_entwicklung": "", "n_test": ""})
    # Testkohorten-Daten sind gelesen: Ende-Eintrag nach dem Lesen, vor dem Rechnen auf Entwicklungsdaten
    del pairs_test
    pd.DataFrame(comp).to_csv(h244.PUB / "a_r6_zusammensetzung.csv", index=False)
    # Standardisierte Entwicklung
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    rng = np.random.default_rng(SEED)
    bootw = np.empty((B, len(ids)))
    for b in range(B):
        bootw[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    parts = sorted((h244.RUN / "repeatability" / "replicate_pairs").glob("part-*.parquet"))
    KEY = ["subject_id", "split", "first_study_id", "second_study_id"]
    res = {}
    ones = np.ones(len(z))
    order_markers = []
    for part in parts:
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(part).schema_arrow.names
        marker = [c[:-6] for c in cols if c.endswith("_first") and c[:-6] + "_second" in cols and c != "signal_quality_first"]
        f = pd.read_parquet(part, columns=KEY + [m + s for m in marker for s in ["_first", "_second"]])
        f = f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]))
        x = f[[m + "_first" for m in marker]].to_numpy(float)
        y = f[[m + "_second" for m in marker]].to_numpy(float)
        del f
        for label, w in (("roh", ones), ("standardisiert", wcell)):
            for lo in range(0, len(marker), 40):
                sl = slice(lo, lo + 40)
                pt, bb = wstats(x[:, sl], y[:, sl], pidx, len(ids), w, bootw)
                for k, m in enumerate(marker[sl]):
                    res.setdefault(label, {})[m] = (pt[k], bb[:, k])
        order_markers += marker
        h244.guard("a_r6")
    universe = common
    out = {}
    for label in ("roh", "standardisiert"):
        pt = np.array([res[label][m][0] if m in res[label] else np.nan for m in universe])
        bbm = np.array([res[label][m][1] if m in res[label] else np.full(B, np.nan) for m in universe]).T  # B x Marker
        use_m = np.isfinite(pt)
        out[label] = {"median_icc": float(np.nanmedian(pt)), "n_marker_schaetzbar": int(use_m.sum()),
                      "median_ci": h244.ci(np.nanmedian(bbm[:, use_m], axis=1), minn=int(0.95 * B)),
                      "draws": np.nanmedian(bbm[:, use_m], axis=1)}
    # Gate: Rohlauf reproduziert den Entwicklungs-Benchmark der Replikation (0,46669 ueber die 2.003 Marker)
    bench = float(dev_m.loc[common, "icc"].median())
    assert abs(out["roh"]["median_icc"] - bench) < 1e-6, (out["roh"]["median_icc"], bench)
    std = out["standardisiert"]
    d_std = std["median_icc"] - out["roh"]["median_icc"]
    # Differenz Test minus standardisierte Entwicklung (Test-SE aus dem berichteten Intervall)
    sd_std = float(np.std(std["draws"], ddof=1))
    diff2 = med_t - std["median_icc"]
    se2 = float(np.hypot(sd_std, se_test))
    teilB = {"zellen_test": int(tcell.nunique()), "zellen_entwicklung": int(dcell.nunique()), "testanteil_ohne_entwicklungszelle": unmatched,
             "median_icc_entwicklung_roh": out["roh"]["median_icc"], "median_icc_entwicklung_roh_ci": out["roh"]["median_ci"],
             "median_icc_entwicklung_standardisiert": std["median_icc"], "median_icc_standardisiert_ci": std["median_ci"],
             "verschiebung_durch_standardisierung": d_std,
             "median_icc_test": med_t, "diff_test_minus_standardisiert": diff2,
             "diff_ci95_normal": [diff2 - 1.96 * se2, diff2 + 1.96 * se2],
             "anteil_der_differenz_durch_zusammensetzung": d_std / diff if diff != 0 else float("nan"),
             "effektive_stichprobe_gewichte": float(wcell.sum() ** 2 / (wcell ** 2).sum()),
             "max_gewicht": float(wcell.max())}
    h244.dump(h244.PUB / "a_r6_teilB.json", teilB)
    pd.DataFrame({"draw": np.arange(B), "median_roh": out["roh"]["draws"], "median_standardisiert": std["draws"]}).to_csv(
        h244.PRIV / "a_r6_draws_privat.csv", index=False)
    txt = ["A-R6 Replikation: Zerlegung und Zusammensetzung (post hoc)", f"Skript-SHA-256 {shash}", json.dumps(teilA, indent=1),
           json.dumps(teilB, indent=1), pd.DataFrame(comp).to_string(index=False)]
    outtxt = h244.PUB / "A_R6_ausgabe.txt"
    outtxt.write_text("\n".join(txt), encoding="utf8")
    print(tz("Ende A-R6: Lesen beendet (Aggregate und Paarspalten der Testreplikation AW-163; locked_test nicht geoeffnet)",
             f"Skript a_r6.py {shash} unveraendert",
             f"Ausgaben nachrechnung/AW244/A_R6_ausgabe.txt {h244.sha(outtxt)}, a_r6_teilA.json {h244.sha(h244.PUB / 'a_r6_teilA.json')}, a_r6_teilB.json {h244.sha(h244.PUB / 'a_r6_teilB.json')}, a_r6_zusammensetzung.csv {h244.sha(h244.PUB / 'a_r6_zusammensetzung.csv')}; keine Kennungen, keine Personenwerte in den Ausgaben"), flush=True)
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
