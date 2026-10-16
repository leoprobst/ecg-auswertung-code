"""AW-244 A-R4 (nachtraeglich, post hoc): Selektionskopplung des Stabilitaetsproxys.

Frage (Zweitmeinung A, Sollte 1): Der Proxy verlangt u. a. gleiche QRS-Dauer-Klasse (Maschinenwert >= 120 ms auf beiden
Aufnahmen) und waehlt damit nach einer mit den Markern verwandten Messgroesse; Paare mehrfach aufgenommener Personen
tragen mehr Gewicht. Hier:
  (1) Proxy ohne QRS-Klassenbedingung ('noqrs': positiver Abstand <= 6 h, gleiche Aufnahme, Sinusrhythmus laut Maschinenbericht,
      kein Schrittmacherbericht, Metadaten nutzbar, Aufnahmezuordnung eindeutig) und die Differenzmenge 'qrsdiff'
      (noqrs minus Proxy = Paare mit wechselnder oder fehlender QRS-Klasse);
  (2) gleiche Personengewichtung: Paargewicht 1 / (Paare der Person in der Population), und zusaetzlich 200 Zufallsziehungen
      eines Paares je Person (Seeds SeedSequence([244040, rep]));
  (3) Kontraste Proxy bzw. noqrs minus alle Paare mit allen Paaren derselben Personen (Personen festgehalten);
  (4) Qualitaetsgate (Guete >= 0,6): die Paarmenge der Messwerte wurde bei der Extraktion mit diesem Gate gebildet; Paare darunter
      liegen nicht vor. Es wird nur das Gate der Menge dokumentiert (Anteil der Aufnahmen mit Guete < 0,6 im Entwicklungscache).
Daten: nur Entwicklung (train, validation): paargruppen_privat.parquet (15.693 Paare, 10.187 Personen), replicate_pairs-Bloecke
des A155-Laufs, Entwicklungs-Merkmalscache (nur Guete). Schaetzer wie AW-149/rv_a1: ICC(A,1) aus normalisierten Momenten, mindestens 30 Personen,
RC = 1,96 SD der zentrierten Differenzen, 1.000 Personenziehungen Seed 77010 (gemeinsame Gewichte mit der Hauptanalyse), Intervalle
2,5/97,5 % bei mindestens 950 endlichen Ziehungen. Gates: Median-ICC gesamt 0,46652 und Proxy 0,52421, Kontrast
Proxy minus gesamt 0,05773 und mit festgehaltenen Personen 0,02429 (repeat_contrasts.csv) werden reproduziert.
Nur Aggregate, keine Kennungen in den Ausgaben.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h244  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import os  # noqa: E402

SMOKE = os.environ.get("AW244_SMOKE") == "1"   # Funktionstest des Laufpfads: ein Block, wenige Ziehungen, keine Gates
B = 60 if SMOKE else 1000
SEED = 77010
KEYS = ["spatial_qrs_t_angle_deg", "qrs_duration_ms", "p_duration_ms", "qtc_fridericia_ms", "qtc_bazett_ms"]
KEY = ["subject_id", "split", "first_study_id", "second_study_id"]


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
    """Gewichtete Momente je Marker; Rueckgabe Punkt (K,4) [icc,bias,sd,rc] in nativen Einheiten, Bootstrap (B,K,4) oder None."""
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


def recon_stable(z, with_qrs=True):
    m = (z.hadm_id_a.notna() & (z.hadm_id_a == z.hadm_id_b) & z.hours_between.gt(0) & z.hours_between.le(6)
         & (z.rhythm == "Sinus") & (z.pacing == "kein_Report") & ~z.admission_match_ambiguous_a.fillna(True)
         & ~z.admission_match_ambiguous_b.fillna(True) & z.machine_metadata_usable_for_primary_a.fillna(False)
         & z.machine_metadata_usable_for_primary_b.fillna(False))
    if with_qrs:
        m = m & z.qrs_class.isin(["breit", "schmal"])
    return m.to_numpy()


def person_weights(mask, pidx, n_persons):
    cnt = np.bincount(pidx[mask], minlength=n_persons).astype(float)
    w = np.zeros(len(mask))
    w[mask] = 1.0 / cnt[pidx[mask]]
    return w


def median_contrast(bb_a, bb_b, pt_a, pt_b, icc_idx=0):
    """difference_of_medians wie in rv_a1: Marker, die im Punktschaetzer in beiden Populationen schaetzbar sind."""
    mask = np.isfinite(pt_a[:, icc_idx]) & np.isfinite(pt_b[:, icc_idx])
    point = float(np.median(pt_b[mask, icc_idx]) - np.median(pt_a[mask, icc_idx]))
    da = np.nanmedian(bb_a[:, mask, icc_idx], axis=1)
    db = np.nanmedian(bb_b[:, mask, icc_idx], axis=1)
    d = db - da
    return point, h244.ci(d, minn=int(0.95 * len(d))), int(mask.sum()), int(np.isfinite(d).sum())


def one_pair_per_person(mask, pidx, rng):
    idx = np.flatnonzero(mask)
    order = rng.permutation(len(idx))
    idx = idx[order]
    _, first = np.unique(pidx[idx], return_index=True)
    return idx[first]


def icc_rows(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid="ignore"):
        center = np.nanmean(np.where(ok, (x + y) / 2, np.nan), axis=0)
        scale = np.sqrt(np.nanmean(np.where(ok, ((x - center) ** 2 + (y - center) ** 2) / 2, np.nan), axis=0))
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    xx = np.where(ok, (x - center) / scale, 0.0)
    yy = np.where(ok, (y - center) / scale, 0.0)
    n = ok.sum(0).astype(float)
    icc = estimate_full(n, xx.sum(0), yy.sum(0), (xx * xx).sum(0), (yy * yy).sum(0), (xx * yy).sum(0))[:, 0]
    icc[n < 30] = np.nan
    return icc


def synthetic(seed=2):
    rng = np.random.default_rng(seed)
    n_p = 800
    npairs = rng.integers(1, 4, n_p)
    subj = np.repeat(np.arange(n_p), npairs)
    N = len(subj)
    qrs = rng.choice(["schmal", "breit", "wechsel", "fehlend"], N, p=[0.7, 0.18, 0.08, 0.04])
    z = pd.DataFrame({"subject_id": subj, "hours_between": rng.uniform(0, 2, N), "rhythm": rng.choice(["Sinus", "AF"], N, p=[0.5, 0.5]),
                      "pacing": "kein_Report", "qrs_class": qrs})
    z["hadm_id_a"] = 1.0
    z["hadm_id_b"] = 1.0
    z["admission_match_ambiguous_a"] = False
    z["admission_match_ambiguous_b"] = False
    z["machine_metadata_usable_for_primary_a"] = True
    z["machine_metadata_usable_for_primary_b"] = True
    z.loc[z.index[::7], "hours_between"] = 0.0
    z["stable"] = recon_stable(z, True)
    true = rng.normal(0, 1, n_p)[subj]
    noise = np.where(z.qrs_class.isin(["wechsel", "fehlend"]), 1.5, np.where(z.rhythm == "Sinus", 0.5, 1.0))
    x = (true + rng.normal(0, 1, N) * noise)[:, None]
    y = (true + rng.normal(0, 1, N) * noise)[:, None]
    return z, x, y


def test():
    h244.start_watch()
    z, x, y = synthetic()
    assert (recon_stable(z, True) == z.stable.to_numpy()).all()
    nq = recon_stable(z, False)
    assert nq.sum() >= z.stable.sum() and (nq & ~z.stable.to_numpy()).sum() == (z.qrs_class.isin(["wechsel", "fehlend"]).to_numpy() & nq).sum()
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    rng = np.random.default_rng(1)
    bw = np.array([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(200)], float)
    ones = np.ones(len(z))
    pt, bb, ns, nn = wstats(x, y, pidx, len(ids), ones, bw)
    # Vergleich mit direkter ICC-Formel
    ref = icc_rows(x, y)
    assert abs(pt[0, 0] - ref[0]) < 1e-9
    # Personengewichte summieren je Person zu 1
    w = person_weights(nq, pidx, len(ids))
    s = np.bincount(pidx, weights=w, minlength=len(ids))
    assert np.allclose(s[s > 0], 1.0) and abs(w.sum() - len(np.unique(z.subject_id[nq]))) < 1e-9
    # Erwartung: Proxy hat kleineren Fehler -> hoeheres ICC als alle Paare; noqrs dazwischen oder gleich
    p_all = wstats(x, y, pidx, len(ids), ones, None)[0][0, 0]
    p_px = wstats(x, y, pidx, len(ids), z.stable.to_numpy().astype(float), None)[0][0, 0]
    p_nq = wstats(x, y, pidx, len(ids), nq.astype(float), None)[0][0, 0]
    assert p_px > p_all and p_nq > p_all and p_nq <= p_px + 0.02, (p_all, p_px, p_nq)
    # Kontrast
    pt_a, bb_a, _, _ = wstats(x, y, pidx, len(ids), ones, bw)
    pt_b, bb_b, _, _ = wstats(x, y, pidx, len(ids), z.stable.to_numpy().astype(float), bw)
    c, cic, nm, nd = median_contrast(bb_a, bb_b, pt_a, pt_b)
    assert abs(c - (p_px - p_all)) < 1e-9 and cic[0] < c < cic[1]
    # ein Paar je Person
    sel = one_pair_per_person(np.ones(len(z), bool), pidx, np.random.default_rng(3))
    assert len(sel) == len(ids) and len(np.unique(pidx[sel])) == len(ids)
    print("TEST_OK", round(p_all, 3), round(p_px, 3), round(p_nq, 3))


def run():
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    script = Path(__file__)
    shash = h244.require_stamp(script)
    h244.guard("a_r4")
    h244.start_watch()
    zpath = h244.MA / "repeatability" / "paargruppen_privat.parquet"
    z = pd.read_parquet(zpath)
    assert len(z) == 15693 and z.subject_id.nunique() == 10187 and set(z.split) <= {"train", "validation"}
    stable_recon = recon_stable(z, True)
    assert (stable_recon == z.stable.to_numpy()).all(), "Proxy-Rekonstruktion weicht ab"
    noqrs = recon_stable(z, False)
    qrsdiff = noqrs & ~z.stable.to_numpy()
    ids = np.unique(z.subject_id)
    pidx = np.searchsorted(ids, z.subject_id.to_numpy())
    allm = np.ones(len(z), bool)
    prox = z.stable.to_numpy()
    persons_prox = np.isin(pidx, np.unique(pidx[prox]))
    persons_nq = np.isin(pidx, np.unique(pidx[noqrs]))
    pops = {"alle": allm, "proxy": prox, "noqrs": noqrs, "qrsdiff": qrsdiff,
            "alle_personen_proxy": persons_prox, "alle_personen_noqrs": persons_nq}
    weights = {}
    for name, m in pops.items():
        weights[name] = m.astype(float)
        weights[name + "_pw"] = person_weights(m, pidx, len(ids))
    counts = [{"population": n, "paare": int(m.sum()), "personen": int(len(np.unique(pidx[m])))} for n, m in pops.items()]
    qc = z.loc[noqrs, "qrs_class"].value_counts().to_dict()
    qc_diff = z.loc[qrsdiff, "qrs_class"].value_counts().to_dict()
    rng = np.random.default_rng(SEED)
    bootw = np.empty((B, len(ids)))
    for b in range(B):
        bootw[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    parts = sorted((h244.RUN / "repeatability" / "replicate_pairs").glob("part-*.parquet"))
    if SMOKE:
        parts = parts[:1]
    R = 4 if SMOKE else 200
    draw_masks = {"alle": allm, "proxy": prox, "noqrs": noqrs}
    sels = {n: [one_pair_per_person(m, pidx, np.random.default_rng(np.random.SeedSequence([244040, r]))) for r in range(R)] for n, m in draw_masks.items()}
    names, pt_all, bb_all, draw_icc = [], {k: [] for k in weights}, {k: [] for k in weights}, {n: [] for n in draw_masks}
    nn_all = {k: [] for k in weights}
    import pickle
    ckdir = h244.PRIV / ("a_r4_ckpt_smoke" if SMOKE else "a_r4_ckpt")
    ckdir.mkdir(exist_ok=True)
    for part in parts:
        ck = ckdir / f"{Path(part).stem}_B{B}_R{R}.pkl"
        if ck.exists():
            cp = pickle.loads(ck.read_bytes())
            if cp["script"] == shash:
                marker, cpts, cnn, cbb, cdraw = cp["marker"], cp["pt"], cp["nn"], cp["bb"], cp["draw"]
                names += marker
                for k in weights:
                    pt_all[k].append(cpts[k]); nn_all[k].append(cnn[k]); bb_all[k].append(cbb[k])
                for n in sels:
                    draw_icc[n].append(cdraw[n])
                print("Block", Path(part).name, "aus Checkpoint", flush=True)
                continue
        cols = pq.ParquetFile(part).schema_arrow.names
        marker = [c[:-6] for c in cols if c.endswith("_first") and c[:-6] + "_second" in cols and c != "signal_quality_first"]
        f = pd.read_parquet(part, columns=KEY + [m + s for m in marker for s in ["_first", "_second"]])
        f = f.set_index(KEY).reindex(pd.MultiIndex.from_frame(z[KEY]))
        x = f[[m + "_first" for m in marker]].to_numpy(float)
        y = f[[m + "_second" for m in marker]].to_numpy(float)
        del f
        names += marker
        cpts, cnn, cbb, cdraw = {}, {}, {}, {}
        for k, w in weights.items():
            pts, iccs, nns, keyb = [], [], [], {}
            for lo in range(0, len(marker), 40):
                sl = slice(lo, lo + 40)
                pt, bb, ns, nn = wstats(x[:, sl], y[:, sl], pidx, len(ids), w, bootw)
                pts.append(pt)
                nns.append(np.c_[nn, ns])
                iccs.append(bb[:, :, 0])
                for j, m in enumerate(marker[sl]):
                    if m in KEYS:
                        keyb[m] = bb[:, j, :].copy()
                del bb
            cpts[k] = np.concatenate(pts)
            cnn[k] = np.concatenate(nns)
            cbb[k] = (np.concatenate(iccs, axis=1), keyb)
            pt_all[k].append(cpts[k]); nn_all[k].append(cnn[k]); bb_all[k].append(cbb[k])
        for n, ss in sels.items():
            cdraw[n] = np.array([icc_rows(x[s], y[s]) for s in ss])
            draw_icc[n].append(cdraw[n])
        ck.write_bytes(pickle.dumps({"script": shash, "marker": marker, "pt": cpts, "nn": cnn, "bb": cbb, "draw": cdraw}))
        h244.guard("a_r4")
        print("Block", Path(part).name, flush=True)
    names = np.array(names)
    P = {k: np.concatenate(v) for k, v in pt_all.items()}
    BI = {k: np.concatenate([a for a, _ in v], axis=1) for k, v in bb_all.items()}
    KB = {k: {m: kb[m] for _, kb in v for m in kb} for k, v in bb_all.items()}
    NN = {k: np.concatenate(v) for k, v in nn_all.items()}
    D = {n: np.concatenate(v, axis=1) for n, v in draw_icc.items()}
    if SMOKE:
        h244.PUB = h244.PRIV / 'smoke'
        h244.PUB.mkdir(exist_ok=True)
        global_ok = True
    # Gates
    def med(k):
        return float(np.nanmedian(P[k][:, 0]))
    g1, g2 = med("alle"), med("proxy")
    assert SMOKE or (abs(g1 - 0.4665168480877628) < 1e-9 and abs(g2 - 0.5242050093521649) < 1e-9), (g1, g2)
    c_px, c_ci, c_n, _ = median_contrast(BI["alle"][:, :, None], BI["proxy"][:, :, None], P["alle"], P["proxy"])
    c_fix, cf_ci, _, _ = median_contrast(BI["alle_personen_proxy"][:, :, None], BI["proxy"][:, :, None], P["alle_personen_proxy"], P["proxy"])
    assert SMOKE or (abs(c_px - 0.057733740061568284) < 1e-9 and abs(c_fix - 0.02428794121305533) < 1e-9), (c_px, c_fix)
    # Zusammenfassung je Population
    def bb3(k):
        return BI[k][:, :, None]
    rows = []
    for k in weights:
        pt = P[k]
        est = np.isfinite(pt[:, 0])
        draws = np.nanmedian(BI[k][:, est], axis=1)
        frac = (pt[est, 0] >= 0.75).mean()
        rows.append({"population": k, "paare": int(np.nanmax(NN[k][:, 0])), "markers_schaetzbar": int(est.sum()),
                     "median_icc": float(np.median(pt[est, 0])), "median_ci_lo": h244.ci(draws, minn=int(0.95 * B))[0],
                     "median_ci_hi": h244.ci(draws, minn=int(0.95 * B))[1], "anteil_icc_ge_075": float(frac)})
    summ = pd.DataFrame(rows)
    contr = []
    pairs = [("alle", "proxy"), ("alle", "noqrs"), ("noqrs", "proxy"), ("qrsdiff", "proxy"), ("alle_personen_proxy", "proxy"),
             ("alle_personen_noqrs", "noqrs"), ("alle_personen_proxy", "noqrs"),
             ("alle_pw", "proxy_pw"), ("alle_pw", "noqrs_pw"), ("alle_personen_proxy_pw", "proxy_pw"), ("alle_personen_noqrs_pw", "noqrs_pw"),
             ("alle", "alle_pw"), ("proxy", "proxy_pw"), ("noqrs", "noqrs_pw")]
    for a, b in pairs:
        pt, ci_, nm, nd = median_contrast(bb3(a), bb3(b), P[a], P[b])
        contr.append({"von": a, "nach": b, "kontrast_median_icc": pt, "ci_lo": ci_[0], "ci_hi": ci_[1], "n_gemeinsame_marker": nm, "gueltige_ziehungen": nd})
    contr = pd.DataFrame(contr)
    # Schluesselmarker
    keyrows = []
    for k in weights:
        for m in KEYS:
            if m not in names:
                continue
            j = int(np.flatnonzero(names == m)[0])
            kb = KB[k][m]
            row = {"population": k, "marker": m, "paare": int(NN[k][j, 0]), "personen": int(NN[k][j, 1])}
            for q, nm_ in enumerate(["icc", "bias", "sd_diff", "rc"]):
                lo, hi = h244.ci(kb[:, q], minn=int(0.95 * B))
                row[nm_] = float(P[k][j, q])
                row[nm_ + "_lo"], row[nm_ + "_hi"] = lo, hi
            keyrows.append(row)
    keyt = pd.DataFrame(keyrows)
    # Zufallsziehungen
    drows = []
    for n in draw_masks:
        meds = np.array([np.nanmedian(D[n][r]) for r in range(R)])
        drows.append({"population": n + "_ein_paar_je_person_zufall", "wiederholungen": R, "median_icc_mittel": float(meds.mean()),
                      "median_icc_p025": float(np.percentile(meds, 2.5)), "median_icc_p975": float(np.percentile(meds, 97.5)),
                      "median_icc_min": float(meds.min()), "median_icc_max": float(meds.max()),
                      "personen": int(len(sels[n][0]))})
        for m in KEYS:
            if m not in names:
                continue
            j = int(np.flatnonzero(names == m)[0])
            v = D[n][:, j]
            drows[-1][m + "_icc_mittel"] = float(np.nanmean(v))
    draws_t = pd.DataFrame(drows)
    # Kontrast der Zufallsziehungen (gepaarte Wiederholung r)
    dc = []
    for a, b in (("alle", "proxy"), ("alle", "noqrs"), ("noqrs", "proxy")):
        both = np.isfinite(D[a]) & np.isfinite(D[b])
        diffs = np.array([np.median(D[b][r][both[r]]) - np.median(D[a][r][both[r]]) for r in range(R)])
        dc.append({"von": a, "nach": b, "kontrast_mittel": float(diffs.mean()), "p025": float(np.percentile(diffs, 2.5)), "p975": float(np.percentile(diffs, 97.5))})
    dc = pd.DataFrame(dc)
    # Qualitaetsgate: Anteil Aufnahmen mit Guete < 0,6 im Entwicklungscache
    qrows = []
    for p in sorted(h244.RUN.glob("analysis_feature_cache/*/split=*/features.parquet")):
        if "split=test" in str(p):
            raise SystemExit("Testpfad")
        t = ds.dataset(str(p), format="parquet").to_table(columns=["signal_quality_score", "signal_quality_usable"]).to_pandas()
        qrows.append({"split": p.parent.name, "ecg": int(len(t)), "guete_unter_06": int((t.signal_quality_score < 0.6).sum()),
                      "nicht_nutzbar": int((~t.signal_quality_usable.astype(bool)).sum())})
    qt = pd.DataFrame(qrows)
    summ.to_csv(h244.PUB / "a_r4_populationen.csv", index=False)
    contr.to_csv(h244.PUB / "a_r4_kontraste.csv", index=False)
    keyt.to_csv(h244.PUB / "a_r4_schluesselmarker.csv", index=False)
    draws_t.to_csv(h244.PUB / "a_r4_zufallsziehungen.csv", index=False)
    dc.to_csv(h244.PUB / "a_r4_zufallskontraste.csv", index=False)
    qt.to_csv(h244.PUB / "a_r4_qualitaetsgate.csv", index=False)
    pd.DataFrame(counts).to_csv(h244.PUB / "a_r4_zaehlung.csv", index=False)
    meta = {"skript_sha256": shash, "gate_median_alle": g1, "gate_median_proxy": g2, "gate_kontrast_proxy": c_px, "gate_kontrast_personen_fest": c_fix,
            "qrs_klassen_noqrs": {k: h244.small(int(v)) for k, v in qc.items()}, "qrs_klassen_differenzmenge": {k: h244.small(int(v)) for k, v in qc_diff.items()},
            "eingabe_paargruppen_sha256": h244.sha(zpath)}
    h244.dump(h244.PUB / "a_r4_meta.json", meta)
    txt = ["A-R4 Selektionskopplung (post hoc, nur Entwicklung)", f"Skript-SHA-256 {shash}", "Zaehlung", pd.DataFrame(counts).to_string(index=False),
           summ.to_string(index=False), contr.to_string(index=False), keyt.to_string(index=False), draws_t.to_string(index=False), dc.to_string(index=False), qt.to_string(index=False), str(meta)]
    (h244.PUB / "A_R4_ausgabe.txt").write_text("\n".join(txt), encoding="utf8")
    # Markerebene (Punktwerte, ohne Intervalle)
    mk = pd.DataFrame({"marker_id": names})
    for k in weights:
        mk[k + "_icc"] = P[k][:, 0]
    mk.to_csv(h244.PRIV / "a_r4_marker_icc_privat.csv", index=False)
    print("RUN_OK peak_rss", h244.peak(), flush=True)


if __name__ == "__main__":
    {"test": test, "run": run}[sys.argv[1]]()
