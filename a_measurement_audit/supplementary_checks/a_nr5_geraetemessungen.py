"""Paper A, Schwachstelle Nr. 5 (nachtraeglich, post hoc): Wiederholbarkeit der Geraetemessungen (MIMIC-IV-ECG machine_measurements)
auf denselben 15.693 Entwicklungspaaren wie die LoopLens-Werte.

Regel (vorab in nachrechnung/AW249/NR5_REGEL_VOR_LAUF.md mit Zeitstempel und Skripthash festgehalten, danach nicht veraendert):
  Geraetegroessen je Aufnahme (study_id) aus machine_measurements.csv:
    rr (rr_interval), p_dur = p_end - p_onset, pr = qrs_onset - p_onset, qrs_dur = qrs_end - qrs_onset, qt = t_end - qrs_onset,
    qtcF = qt / (rr/1000)^(1/3), qtcB = qt / sqrt(rr/1000), Achsen p_axis, qrs_axis, t_axis (Grad),
    frontaler QRS-T-Winkel = kuerzester Winkelabstand zwischen qrs_axis und t_axis (0 bis 180).
  Ungueltig (NaN): Zeitfelder (rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end) ausserhalb [1, 2999] ms (Sentinelle 29999, 32767,
  65534, 65535, 61440 und 0); Achsen ausserhalb [-180, 360] Grad; abgeleitete Dauern <= 0; rr ausserhalb [200, 3000] ms.
  Schaetzer wie AW-149/AW-244 (a_r4): ICC(A,1) aus normalisierten Momenten (mindestens 30 Personen), RC = 1,96 SD der zentrierten
  Differenzen, Bias, 1.000 Personenziehungen mit Seed 77010 und denselben Gewichten wie die Hauptanalyse (np.unique der subject_id,
  bincount(rng.integers)). Achsen (zirkulaer, Periode 360): D95 = gewichtetes 95-%-Quantil des kuerzesten Winkelabstands
  (kleinster Wert d mit kumuliertem Gewicht >= 0,95 des Gesamtgewichts), Median ebenso (0,50).
  Populationen: 'alle' (alle 15.693 Paare) und 'proxy' (Paare des Stabilitaetsproxys, Spalte stable).
  Vergleich: je Groesse mit LoopLens-Gegenstueck auf den Paaren, in denen Geraet und LoopLens beide endlich sind (gleiche Paarmenge):
  RC-Verhaeltnis = RC(Geraet) / RC(LoopLens) (Werte unter 1: Geraet streut weniger), ICC-Differenz = ICC(Geraet) - ICC(LoopLens),
  Intervalle 2,5/97,5 % der Ziehungen (mindestens 950 endliche).
  Gegenstuecke: qrs_duration_ms, qt_interval_ms, qtc_fridericia_ms, qtc_bazett_ms, p_duration_ms, pr_interval_ms, frontal_xy_qrs_t_angle_deg.
  Gates (Abbruch, wenn verfehlt): LoopLens-Werte der Population 'alle' fuer qrs_duration_ms und qtc_fridericia_ms reproduzieren die
  Zeilen aus nachrechnung/AW244/a_r4_schluesselmarker.csv (ICC, RC, Paarzahl) auf 1e-6.
  Berichtet, aber kein Gate: Uebereinstimmung der abgeleiteten QRS-Dauer mit der Spalte machine_qrs_duration_ms im Paarrahmen.
Nur Aggregate, keine Kennungen in den Ausgaben. Aufruf: <local path> -I -B <skript> test|run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aw244"))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 77010
B = 1000
OUT = h244.AUSW / "nachrechnung" / "AW249"
KEY = ["subject_id", "split", "first_study_id", "second_study_id"]
MM = Path(str(data_path("a.mimic_iv_ecg_diagnostic_electrocardiogram_matched_subset_1_0_machine_measurements_csv")))
TIME_FIELDS = ["p_onset", "p_end", "qrs_onset", "qrs_end", "t_end"]
AXES = ["p_axis", "qrs_axis", "t_axis"]
# Geraetegroesse -> LoopLens-Gegenstueck
PAIRS = {"qrs_dur": "qrs_duration_ms", "qt": "qt_interval_ms", "qtcF": "qtc_fridericia_ms", "qtcB": "qtc_bazett_ms",
         "p_dur": "p_duration_ms", "pr": "pr_interval_ms", "frontal_qrst": "frontal_xy_qrs_t_angle_deg"}


def estimate_full(n, sx, sy, sxx, syy, sxy):
    with np.errstate(divide="ignore", invalid="ignore"):
        col = (sx - sy) ** 2 / (2 * n)
        msr = ((sxx + 2 * sxy + syy) / 2 - (sx + sy) ** 2 / (2 * n)) / (n - 1)
        mse = ((sxx + syy - 2 * sxy) / 2 - col) / (n - 1)
        den = msr + mse + 2 * (col - mse) / n
        icc = np.where((n >= 2) & (den > 0), np.clip((msr - mse) / den, -1, 1), np.nan)
        sd = np.sqrt(np.maximum(0, (sxx + syy - 2 * sxy - (sy - sx) ** 2 / n) / (n - 1)))
        return np.stack([icc, (sy - sx) / n, sd, 1.96 * sd], axis=-1)


def wstats(x, y, pidx, n_persons, wpair, bootw):
    """Wie a_r4.wstats. Rueckgabe Punkt (K,4) [icc,bias,sd,rc], Bootstrap (B,K,4) oder None, Personen je Marker, Paare je Marker."""
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
        np.add.at(a, pidx, v)
        agg.append(a)
    has = np.zeros((n_persons, x.shape[1]))
    np.add.at(has, pidx, ok.astype(float))
    has = (has > 0).astype(float)
    ns = has.sum(0)
    pt = estimate_full(*[a.sum(0) for a in agg])
    pt[ns < 30] = np.nan
    pt[:, 1:] *= scale[:, None]
    bb = None
    if bootw is not None:
        bb = np.empty((bootw.shape[0], x.shape[1], 4))
        for lo in range(0, bootw.shape[0], 50):
            sl = bootw[lo:lo + 50]
            b = estimate_full(*[sl @ a for a in agg])
            eff = (sl > 0).astype(float) @ has
            b[eff < 30] = np.nan
            bb[lo:lo + 50] = b
        bb[:, :, 1:] *= scale[None, :, None]
    return pt, bb, ns, ok.sum(0)


def ang_dist(a, b):
    """Kuerzester Winkelabstand in Grad (0 bis 180), Periode 360."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def clean_time(s):
    s = s.astype(float)
    return s.where((s >= 1) & (s <= 2999))


def derive(m):
    """Geraetegroessen aus den Rohfeldern (DataFrame mit study_id und den Feldern)."""
    assert m.index.equals(pd.RangeIndex(len(m))), "derive: Index muss fortlaufend sein"
    d = pd.DataFrame({"study_id": m["study_id"].to_numpy()})
    t = {c: clean_time(m[c]) for c in TIME_FIELDS}
    rr = m["rr_interval"].astype(float)
    rr = rr.where((rr >= 200) & (rr <= 3000))
    pos = lambda s: s.where(s > 0)  # noqa: E731
    d["p_dur"] = pos(t["p_end"] - t["p_onset"])
    d["pr"] = pos(t["qrs_onset"] - t["p_onset"])
    d["qrs_dur"] = pos(t["qrs_end"] - t["qrs_onset"])
    d["qt"] = pos(t["t_end"] - t["qrs_onset"])
    d["qtcF"] = d["qt"] / (rr / 1000.0) ** (1.0 / 3.0)
    d["qtcB"] = d["qt"] / np.sqrt(rr / 1000.0)
    ax = {}
    for c in AXES:
        a = m[c].astype(float)
        ax[c] = a.where((a >= -180) & (a <= 360))
        d[c] = ax[c]
    d["frontal_qrst"] = ang_dist(ax["qrs_axis"], ax["t_axis"])
    return d


def wquantile(dist, W, q):
    """Gewichtetes Quantil: kleinster Wert mit kumuliertem Gewicht >= q * Gesamtgewicht. W: (R, n) Gewichte, dist: (n,) endlich."""
    o = np.argsort(dist, kind="stable")
    ds = dist[o]
    cw = np.cumsum(W[:, o], axis=1)
    k = (cw >= q * cw[:, -1:]).argmax(axis=1)
    return ds[k]


def circ_stats(a, b, pidx, bootw, w):
    """D95 und Median des kuerzesten Winkelabstands. Punkt (Gewicht w) und Bootstrap ueber Personenziehungen."""
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    d = ang_dist(a[ok], b[ok])
    pw = w[ok]
    pi = pidx[ok]
    out = {"paare": int(ok.sum()), "personen": int(len(np.unique(pi)))}
    pt = {q: float(wquantile(d, pw[None, :], q)[0]) for q in (0.95, 0.5)}
    bq = {0.95: [], 0.5: []}
    for lo in range(0, bootw.shape[0], 100):
        W = bootw[lo:lo + 100][:, pi] * pw[None, :]
        for q in bq:
            bq[q].append(wquantile(d, W, q))
    for q, nm in ((0.95, "d95"), (0.5, "median")):
        v = np.concatenate(bq[q])
        lo_, hi_ = h244.ci(v, minn=int(0.95 * len(v)))
        out[nm] = pt[q]
        out[nm + "_lo"], out[nm + "_hi"] = lo_, hi_
    return out


def boot_weights(ids):
    rng = np.random.default_rng(SEED)
    bw = np.empty((B, len(ids)))
    for b in range(B):
        bw[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    return bw


def linear_row(name, source, x, y, pidx, n_p, w, bw):
    pt, bb, ns, nn = wstats(x[:, None], y[:, None], pidx, n_p, w, bw)
    row = {"groesse": name, "quelle": source, "paare": int(nn[0]), "personen": int(ns[0])}
    for q, nm in enumerate(["icc", "bias", "sd_diff", "rc"]):
        lo, hi = h244.ci(bb[:, 0, q], minn=int(0.95 * bb.shape[0]))
        row[nm] = float(pt[0, q])
        row[nm + "_lo"], row[nm + "_hi"] = lo, hi
    return row


def compare(name, xm, ym, xl, yl, pidx, n_p, w, bw):
    """Geraet gegen LoopLens auf Paaren, in denen alle vier Werte endlich sind."""
    ok = np.isfinite(xm) & np.isfinite(ym) & np.isfinite(xl) & np.isfinite(yl)
    xx = np.where(ok[:, None], np.c_[xm, xl], np.nan)
    yy = np.where(ok[:, None], np.c_[ym, yl], np.nan)
    pt, bb, ns, nn = wstats(xx, yy, pidx, n_p, w, bw)
    rc_ratio = bb[:, 0, 3] / bb[:, 1, 3]
    icc_diff = bb[:, 0, 0] - bb[:, 1, 0]
    r = {"groesse": name, "gemeinsame_paare": int(nn[0]), "gemeinsame_personen": int(ns[0]),
         "rc_geraet": float(pt[0, 3]), "rc_looplens": float(pt[1, 3]), "icc_geraet": float(pt[0, 0]), "icc_looplens": float(pt[1, 0]),
         "rc_verhaeltnis_geraet_zu_looplens": float(pt[0, 3] / pt[1, 3]), "icc_differenz_geraet_minus_looplens": float(pt[0, 0] - pt[1, 0])}
    r["rc_verhaeltnis_lo"], r["rc_verhaeltnis_hi"] = h244.ci(rc_ratio, minn=int(0.95 * bb.shape[0]))
    r["icc_differenz_lo"], r["icc_differenz_hi"] = h244.ci(icc_diff, minn=int(0.95 * bb.shape[0]))
    for k, j in (("geraet", 0), ("looplens", 1)):
        for q, nm in ((0, "icc"), (3, "rc")):
            r[f"{nm}_{k}_lo"], r[f"{nm}_{k}_hi"] = h244.ci(bb[:, j, q], minn=int(0.95 * bb.shape[0]))
    return r


def test():
    h244.start_watch()
    # Ableitungen
    m = pd.DataFrame({"study_id": [1, 2, 3, 4], "rr_interval": [1000, 500, 29999, 800], "p_onset": [40, 40, 40, 29999],
                      "p_end": [140, 29999, 150, 150], "qrs_onset": [200, 200, 200, 200], "qrs_end": [290, 280, 290, 190],
                      "t_end": [600, 500, 600, 600], "p_axis": [60, 32767, 60, 60], "qrs_axis": [350, 30, 30, 30], "t_axis": [10, 40, 29999, 40]})
    d = derive(m)
    # derive verlangt einen fortlaufenden Index (Ausrichtung beim Zuweisen); run() setzt ihn zurueck
    m2 = m.iloc[[3, 1, 0]]
    try:
        d2 = derive(m2.reset_index(drop=True))
    except Exception as e:  # pragma: no cover
        raise AssertionError(e)
    assert d2.p_dur.isna().tolist() == [True, True, False] and d2.qrs_dur[2] == 90
    assert abs(d.qtcF[0] - 400) < 1e-9 and abs(d.qtcB[0] - 400) < 1e-9
    assert abs(d.qtcB[1] - 300 / np.sqrt(0.5)) < 1e-9 and abs(d.qtcF[1] - 300 / 0.5 ** (1 / 3)) < 1e-9
    assert np.isnan(d.qtcF[2]) and np.isnan(d.qtcB[2]) and abs(d.qt[2] - 400) < 1e-9
    assert np.isnan(d.p_dur[1]) and d.p_dur[0] == 100 and np.isnan(d.p_dur[3]) and np.isnan(d.pr[3])
    assert np.isnan(d.qrs_dur[3]) and d.qrs_dur[0] == 90      # qrs_end < qrs_onset
    assert np.isnan(d.p_axis[1]) and np.isnan(d.frontal_qrst[2])
    assert abs(d.frontal_qrst[0] - 20) < 1e-9 and abs(ang_dist(np.array([-170.0]), np.array([170.0]))[0] - 20) < 1e-9
    assert abs(ang_dist(np.array([0.0]), np.array([180.0]))[0] - 180) < 1e-9
    # gewichtetes Quantil
    dist = np.arange(1.0, 101.0)
    assert wquantile(dist, np.ones((1, 100)), 0.95)[0] == 95 and wquantile(dist, np.ones((1, 100)), 0.5)[0] == 50
    assert wquantile(dist, np.where(np.arange(100) < 50, 0.0, 1.0)[None, :], 0.5)[0] == 75
    # Schaetzer: bekannte Struktur, Geraet mit doppeltem Rauschen -> RC-Verhaeltnis etwa 2, ICC niedriger
    rng = np.random.default_rng(5)
    n_p = 1500
    npairs = rng.integers(1, 3, n_p)
    subj = np.repeat(np.arange(n_p), npairs)
    N = len(subj)
    true = rng.normal(0, 10, n_p)[subj]
    xl, yl = true + rng.normal(0, 3, N), true + rng.normal(0, 3, N)
    xm, ym = true + rng.normal(0, 6, N), true + rng.normal(0, 6, N)
    xm[::9] = np.nan
    pidx = subj
    bw = np.empty((200, n_p))
    r2 = np.random.default_rng(1)
    for b in range(200):
        bw[b] = np.bincount(r2.integers(0, n_p, n_p), minlength=n_p)
    ones = np.ones(N)
    r = compare("test", xm, ym, xl, yl, pidx, n_p, ones, bw)
    assert 1.8 < r["rc_verhaeltnis_geraet_zu_looplens"] < 2.2, r
    assert r["rc_verhaeltnis_lo"] < r["rc_verhaeltnis_geraet_zu_looplens"] < r["rc_verhaeltnis_hi"]
    assert r["icc_differenz_geraet_minus_looplens"] < 0 and r["icc_differenz_hi"] < 0
    assert r["gemeinsame_paare"] == int(np.isfinite(xm).sum())
    assert abs(r["rc_looplens"] - 1.96 * np.sqrt(2) * 3) < 0.6
    row = linear_row("t", "g", xl, yl, pidx, n_p, ones, bw[:200])
    assert abs(row["rc"] - 1.96 * np.sqrt(2) * 3) < 0.6 and 0.85 < row["icc"] < 0.97
    # zirkulaer
    a = rng.uniform(-180, 180, N)
    b = ((a + rng.normal(0, 10, N) + 180) % 360) - 180
    cs = circ_stats(a, b, pidx, bw, ones)
    assert 10 < cs["d95"] < 25 and cs["d95_lo"] <= cs["d95"] <= cs["d95_hi"] and cs["median"] < cs["d95"], cs
    print("TEST_OK", round(r["rc_verhaeltnis_geraet_zu_looplens"], 2), round(cs["d95"], 1))


def run():
    import pyarrow.parquet as pq
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("nr5")
    h244.start_watch()
    OUT.mkdir(exist_ok=True)
    z = pd.read_parquet(h244.MA / "repeatability" / "paargruppen_privat.parquet")
    assert len(z) == 15693 and z.subject_id.nunique() == 10187 and set(z.split) <= {"train", "validation"}
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    bw = boot_weights(ids)
    # Geraetemessungen
    cols = ["study_id", "rr_interval"] + TIME_FIELDS + AXES
    needed = set(z.first_study_id) | set(z.second_study_id)
    mm = pd.read_csv(MM, usecols=cols)
    n_mm = len(mm)
    mm = mm[mm.study_id.isin(needed)].reset_index(drop=True)
    assert mm.study_id.is_unique
    share_p40 = float((mm.p_onset == 40).mean())
    dev = derive(mm).set_index("study_id")
    nm_ = list(PAIRS) + AXES
    first = dev.reindex(z.first_study_id.to_numpy())[nm_]
    second = dev.reindex(z.second_study_id.to_numpy())[nm_]
    # LoopLens-Gegenstuecke
    ll_names = list(PAIRS.values())
    LL = {n: [None, None] for n in ll_names}
    parts = sorted((h244.RUN / "repeatability" / "replicate_pairs").glob("part-*.parquet"))
    for part in parts:
        have = pq.ParquetFile(part).schema_arrow.names
        want = [n for n in ll_names if n + "_first" in have and n + "_second" in have]
        if not want:
            continue
        f = pd.read_parquet(part, columns=KEY + [n + s for n in want for s in ("_first", "_second")])
        f = f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]))
        for n in want:
            LL[n] = [f[n + "_first"].to_numpy(float), f[n + "_second"].to_numpy(float)]
    assert all(v[0] is not None for v in LL.values()), "LoopLens-Gegenstueck fehlt"
    # Gates gegen a_r4
    ref = pd.read_csv(h244.AUSW / "nachrechnung" / "AW244" / "a_r4_schluesselmarker.csv")
    ref = ref[ref.population == "alle"].set_index("marker")
    ones = np.ones(len(z))
    gates = []
    for n in ("qrs_duration_ms", "qtc_fridericia_ms"):
        pt, _, ns, nn = wstats(LL[n][0][:, None], LL[n][1][:, None], pidx, len(ids), ones, None)
        g = {"marker": n, "icc": float(pt[0, 0]), "rc": float(pt[0, 3]), "paare": int(nn[0]),
             "icc_ref": float(ref.loc[n, "icc"]), "rc_ref": float(ref.loc[n, "rc"]), "paare_ref": int(ref.loc[n, "paare"])}
        assert abs(g["icc"] - g["icc_ref"]) < 1e-6 and abs(g["rc"] - g["rc_ref"]) < 1e-6 and g["paare"] == g["paare_ref"], g
        gates.append(g)
    # Abgleich abgeleitete QRS-Dauer gegen Spalte im Paarrahmen (berichtet, kein Gate)
    chk = []
    for s, col in (("first", "machine_qrs_duration_ms_a"), ("second", "machine_qrs_duration_ms_b")):
        a = (first if s == "first" else second)["qrs_dur"].to_numpy()
        b = z[col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        chk.append({"aufnahme": s, "beide_endlich": int(both.sum()), "gleich": int((np.abs(a[both] - b[both]) < 1e-9).sum())})
    pops = {"alle": ones, "proxy": z.stable.to_numpy().astype(float)}
    dev_rows, cmp_rows, circ_rows = [], [], []
    for pname, w in pops.items():
        for g, ln in PAIRS.items():
            xm, ym = first[g].to_numpy(float), second[g].to_numpy(float)
            r = linear_row(g, "Geraet", xm, ym, pidx, len(ids), w, bw)
            r["population"] = pname
            dev_rows.append(r)
            r = linear_row(ln, "LoopLens", LL[ln][0], LL[ln][1], pidx, len(ids), w, bw)
            r["population"] = pname
            dev_rows.append(r)
            c = compare(g, xm, ym, LL[ln][0], LL[ln][1], pidx, len(ids), w, bw)
            c["looplens_marker"] = ln
            c["population"] = pname
            cmp_rows.append(c)
        for ax in AXES:
            c = circ_stats(first[ax].to_numpy(float), second[ax].to_numpy(float), pidx, bw, w)
            c.update({"groesse": ax, "population": pname})
            circ_rows.append(c)
    dev_t, cmp_t, circ_t = pd.DataFrame(dev_rows), pd.DataFrame(cmp_rows), pd.DataFrame(circ_rows)
    dev_t.to_csv(OUT / "nr5_wiederholbarkeit.csv", index=False)
    cmp_t.to_csv(OUT / "nr5_vergleich.csv", index=False)
    circ_t.to_csv(OUT / "nr5_achsen.csv", index=False)
    meta = {"skript_sha256": shash, "eingabe_paargruppen_sha256": h244.sha(h244.MA / "repeatability" / "paargruppen_privat.parquet"),
            "eingabe_machine_measurements_sha256": h244.sha(MM), "zeilen_machine_measurements": n_mm,
            "aufnahmen_gebraucht": len(needed), "aufnahmen_mit_geraetewerten": int(len(mm)),
            "anteil_p_onset_gleich_40_ms": share_p40, "gates": gates, "abgleich_qrs_spalte": chk, "ziehungen": B, "seed": SEED}
    h244.dump(OUT / "nr5_meta.json", meta)
    txt = ["NR5 Geraetemessungen (post hoc, nur Entwicklung)", f"Skript-SHA-256 {shash}", dev_t.to_string(index=False), cmp_t.to_string(index=False),
           circ_t.to_string(index=False), str(meta)]
    (OUT / "NR5_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
