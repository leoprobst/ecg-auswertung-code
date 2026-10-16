# -*- coding: utf-8 -*-
"""aw245_r7.py (version 1.0): complete the gain arm (program comparison study, exploratory, post hoc). PTB-XL, S_G (1,000 persons from P0, factors 0.5 / 1 / 2).

(a) Sampling design as a table (strata, population N, sample n, selection probability, weight, Kish sample size); design-aware uncertainty (stratified
    bootstrap, 2,000 draws within the strata) next to the Wilson interval with Kish sample size (rebuild of the published gain-arm values as a check).
(b) Distribution of the actual changes (weighted): median, P95, P99, maximum of the absolute value; share with more than 2, 5, 10 ms (time quantities); threshold crossings (QRS 120; QTcF 480, 500)
    relative to all persons and to persons with a value within +-10 ms of the threshold at factor 1.
(c) Null control: factor 1 against the run without scaling (V_K1 of the benchmark analysis, read) and, where available, the number of deviations; factor 1.001 not computed (see the report).
(d) Rounding to the ADC resolution and clipping: not computed (setting still pending a decision).
(e) Independent beat matching (nearest R peak within +-40 ms resp. +-100 ms) between factor 1 and factor f from looplens_schlaege.csv.gz; beats that cannot be matched.
Locked tables are only read (hash check of the data lock). Usage: python -I -B aw245_r7.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw245_basis as AB  # noqa: E402
import aw245_stat as T  # noqa: E402

S = AB.S
log = AB.log
SEED = 15102457
B = 2000
FAKT = {0.5: "VV0_G0.5", 1.0: "VV0_G1", 2.0: "VV0_G2"}
LL_Q = {"qrs_dauer_global": "QRS-Dauer", "qt_global": "QT", "qtcf": "QTcF", "pr_intervall": "PR-Intervall", "p_dauer": "P-Dauer"}
SCHW = {"qrs_dauer_global": (120,), "qtcf": (480, 500)}
KONTROLLEN = []
BENCH_K1 = Path(str(data_path("pub12.benchmark_auswertung_aw221_v11_v_k1_faktor1_gegen_ohne_skalierung_csv")))
BENCH_S7 = Path(str(data_path("pub12.benchmark_auswertung_aw221_v11_v_verstaerkung_zellen_csv")))


def kontrolle(name, soll, ist, ok):
    KONTROLLEN.append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok)))
    log("CHECK %s: expected %s actual %s -> %s" % (name, soll, ist, "ok" if ok else "DEVIATION"))


def wilson(p, n, z=1.959964):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def lies_ll(ids, vvs, groessen):
    cols = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status"]
    teile = []
    ids_set = set(int(i) for i in ids)
    for ch in pd.read_csv(S.PFADE["erg"] / "looplens/langformat/looplens_langformat.csv.gz", usecols=cols, dtype=str, chunksize=1_500_000):
        ch = ch[(ch.datensatz == "ptbxl") & ch.vorverarbeitung.isin(vvs) & ch.messgroesse.isin(groessen)]
        a = pd.to_numeric(ch["aufnahme"], errors="coerce")
        ch = ch[a.isin(ids_set)]
        if len(ch):
            teile.append(ch)
    d = pd.concat(teile, ignore_index=True)
    d["aufnahme"] = d["aufnahme"].astype(int)
    d["wert"] = pd.to_numeric(d["wert"], errors="coerce")
    out = {}
    for (vv, q), g in d.groupby(["vorverarbeitung", "messgroesse"]):
        g = g[~g.aufnahme.duplicated(keep="first")].set_index("aufnahme")
        v = np.where((g.status == "ok") & np.isfinite(g.wert), g.wert, np.nan)
        out[(vv, q)] = pd.Series(v, index=g.index).reindex(ids).to_numpy(float)
    return out


def lade_nk(ids):
    """NeuroKit2 QRS_g1 and QT_g1 per factor via nk_parse from the benchmark script 1.1 (hash included in the hash check)."""
    spec = importlib.util.spec_from_file_location("bench_v11", str(S.PFADE["v2skript"]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    nk_all = pd.read_csv(S.PFADE["erg"] / "neurokit2/langformat/neurokit2_schlaege.csv.gz", dtype={"datensatz": str, "vorverarbeitung": str, "ableitung": str})
    for f, vv in FAKT.items():
        nk = nk_all[(nk_all.datensatz == "ptbxl") & (nk_all.vorverarbeitung == vv) & nk_all.aufnahme.isin(ids)]
        rec = mod.nk_parse(nk)
        for g in ("QRS_g1", "QT_g1"):
            v = np.full(len(ids), np.nan)
            for i, aid in enumerate(ids):
                r = rec.get(("ptbxl", vv, int(aid)))
                if r is not None:
                    x = r["werte"].get(g, np.nan)
                    if np.isfinite(x):
                        v[i] = x
            out[(vv, g)] = v
        del nk, rec
    return out


def strat_weights(strat, w, B, seed):
    """Row 0: weights w; rows 1..B: w * number of times the person was drawn in the draw within the stratum."""
    rng = np.random.default_rng(seed)
    n = len(strat)
    W = np.empty((B + 1, n))
    W[0] = w
    idx_by = {s: np.flatnonzero(strat == s) for s in np.unique(strat)}
    for b in range(B):
        c = np.zeros(n)
        for s, ix in idx_by.items():
            c_s = np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix))
            c[ix] = c_s
        W[b + 1] = w * c
    return W


def lauf():
    t0 = time.time()
    AB.OUT_ROH.mkdir(parents=True, exist_ok=True)
    AB.OUT_PUB.mkdir(parents=True, exist_ok=True)
    S.STATE["log"] = str(AB.OUT_ROH / "r7_lauf.log")
    log("aw245_r7 Version 1.0")
    pruef = AB.pruefe_sperre()
    sg = S.lese_ids(S.PFADE["listen"] / "S_G_ecg_id.txt")
    st = pd.read_csv(S.PFADE["listen"] / "S_G_stratum.csv").set_index("ecg_id").stratum.reindex(sg).to_numpy()
    n_sg = pd.Series(st).value_counts().to_dict()
    w = np.array([S.STRATUM_N_P0[s] / n_sg[s] for s in st], float)
    kontrolle("S_G persons", 1000, len(sg), len(sg) == 1000)
    kontrolle("Sum of weights = P0", 18869, round(float(w.sum())), round(float(w.sum())) == 18869)
    # (a) design table
    rows = []
    for s in sorted(S.STRATUM_N_P0):
        N, n = S.STRATUM_N_P0[s], n_sg.get(s, 0)
        rows.append(dict(stratum=s, grundgesamtheit_P0=N, stichprobe_S_G=n, auswahlwahrscheinlichkeit=n / N, gewicht=N / n, anteil_P0_pct=100 * N / 18869, anteil_S_G_pct=100 * n / 1000))
    rows.append(dict(stratum="gesamt", grundgesamtheit_P0=18869, stichprobe_S_G=1000, auswahlwahrscheinlichkeit=1000 / 18869, gewicht=np.nan, anteil_P0_pct=100.0, anteil_S_G_pct=100.0))
    n_eff_all = float(w.sum() ** 2 / (w ** 2).sum())
    rows[-1]["kish_n_eff"] = n_eff_all
    df_des = pd.DataFrame(rows)
    # values
    ll = lies_ll(sg, set(FAKT.values()), set(LL_Q))
    nkv = lade_nk(sg)
    Wb = strat_weights(st, w, B, SEED)
    zeilen, rows_s, rows_f = [], [], []
    quant = [("LoopLens", q, LL_Q[q], {f: ll[(FAKT[f], q)] for f in FAKT}) for q in LL_Q]
    quant += [("NeuroKit2", g, g, {f: nkv[(FAKT[f], g)] for f in FAKT}) for g in ("QRS_g1", "QT_g1")]
    for tool, q, name, vals in quant:
        x1 = vals[1.0]
        for f in (0.5, 2.0):
            xf = vals[f]
            ok = np.isfinite(x1) & np.isfinite(xf)
            n_ok = int(ok.sum())
            ch = (np.abs(xf - x1) > 1e-9) & ok
            dlt = np.where(ok, np.abs(xf - x1), 0.0)
            Wo = Wb[:, ok]
            N_ = Wo.sum(axis=1)
            share = (Wb @ (ch.astype(float))) / N_
            p, lo, hi = T.ci(share)
            w_ok = w[ok]
            neff = float(w_ok.sum() ** 2 / (w_ok ** 2).sum())
            pw = float((w_ok * ch[ok]).sum() / w_ok.sum())
            wl, wh = wilson(pw, neff)
            r = dict(werkzeug=tool, groesse=name, faktor=f, n_nenner=n_ok, n_veraendert=int(ch.sum()), anteil_gewichtet_pct=100 * pw, wilson_kish_unten_pct=100 * wl, wilson_kish_oben_pct=100 * wh,
                     kish_n_eff=neff, geschichtet_bootstrap_unten_pct=100 * lo, geschichtet_bootstrap_oben_pct=100 * hi,
                     breite_wilson=100 * (wh - wl), breite_bootstrap=100 * (hi - lo))
            # size of the changes (time quantities in ms only): weighted quantiles of the absolute value, share above thresholds
            da = dlt[ok]
            if tool == "LoopLens" and q in ("qrs_dauer_global", "qt_global", "qtcf", "pr_intervall", "p_dauer") or tool == "NeuroKit2":
                for qq, nmq in ((0.5, "median"), (0.95, "p95"), (0.99, "p99")):
                    e_, l_, h_ = T.ci(T.wquantil(da, Wo, qq))
                    r["betrag_%s_ms" % nmq], r["betrag_%s_unten" % nmq], r["betrag_%s_oben" % nmq] = e_, l_, h_
                r["betrag_max_ms"] = float(da.max())
                for thr in (2, 5, 10):
                    sh = (Wb @ ((dlt > thr) & ok).astype(float)) / N_
                    e_, l_, h_ = T.ci(sh * 100)
                    r["anteil_ueber_%dms_pct" % thr], r["anteil_ueber_%dms_unten" % thr], r["anteil_ueber_%dms_oben" % thr] = e_, l_, h_
                r["median_betrag_unter_den_veraenderten_ms"] = float(np.median(da[ch[ok]])) if ch.any() else np.nan
                r["anteil_veraendert_unter_5ms_der_veraenderten_pct"] = float(100 * np.mean(da[ch[ok]] <= 5)) if ch.any() else np.nan
            zeilen.append(r)
            # threshold crossings
            key = q if tool == "LoopLens" else None
            if key in SCHW:
                for thr in SCHW[key]:
                    fl = ((x1 >= thr) != (xf >= thr)) & ok
                    near = ok & (np.abs(x1 - thr) <= 10)
                    sh_all = (Wb @ fl.astype(float)) / N_
                    Wn = Wb[:, near]
                    Nn = Wn.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sh_near = (Wb @ (fl & near).astype(float)) / (Wb @ near.astype(float))
                    e1, l1, h1 = T.ci(sh_all * 100)
                    e2, l2, h2 = T.ci(sh_near * 100)
                    rows_s.append(dict(werkzeug=tool, groesse=name, faktor=f, schwelle=thr, n_nenner=n_ok, wechsel_n=int(fl.sum()), anteil_gewichtet_pct=e1, bootstrap_unten=l1, bootstrap_oben=h1,
                                       n_nahe_schwelle_pm10=int(near.sum()), wechsel_unter_nahen_n=int((fl & near).sum()), anteil_unter_nahen_gewichtet_pct=e2, nahe_bootstrap_unten=l2, nahe_bootstrap_oben=h2,
                                       unter_nahen_ungewichtet_pct=100 * float((fl & near).sum() / max(near.sum(), 1))))
    df_z = pd.DataFrame(zeilen)
    df_s = pd.DataFrame(rows_s)
    # Checks against the published gain-arm values of the benchmark analysis
    s7 = pd.read_csv(BENCH_S7)
    for tool, q, name, vals in quant[:3]:
        for f in (0.5, 2.0):
            r = df_z[(df_z.werkzeug == tool) & (df_z.groesse == name) & (df_z.faktor == f)].iloc[0]
            ref = s7[(s7.werkzeug == tool) & (s7.groesse == q) & (s7.faktor == f)]
            if len(ref):
                ref = ref.iloc[0]
                kontrolle("Rebuild of published values %s %s %.1f (weighted share, Wilson-Kish)" % (tool, q, f), "%.4f (%.4f to %.4f)" % (ref.anteil_gewichtet, ref.wilson_lo, ref.wilson_hi),
                          "%.4f (%.4f to %.4f)" % (r.anteil_gewichtet_pct / 100, r.wilson_kish_unten_pct / 100, r.wilson_kish_oben_pct / 100),
                          abs(ref.anteil_gewichtet - r.anteil_gewichtet_pct / 100) < 1e-9 and abs(ref.wilson_lo - r.wilson_kish_unten_pct / 100) < 1e-9 and int(ref.n_nenner) == int(r.n_nenner))
    # (c) K1 null control
    k1 = pd.read_csv(BENCH_K1)
    k1p = k1[k1.datensatz == "ptbxl"]
    kontrolle("K1 (factor 1 against run without scaling) all cells passed", "%d of %d" % (len(k1p), len(k1p)), "%d of %d" % (int(k1p.bestanden.sum()), len(k1p)), bool(k1p.bestanden.all()))
    # (e) independent beat matching
    sch = pd.read_csv(S.PFADE["erg"] / "looplens/langformat/looplens_schlaege.csv.gz", usecols=["datensatz", "aufnahme", "vorverarbeitung", "schlag_nr", "r_gipfel", "qrs_beginn", "akzeptiert"])
    sch = sch[(sch.datensatz == "ptbxl") & sch.vorverarbeitung.isin(FAKT.values()) & sch.aufnahme.isin(sg)]
    rows_b = []
    for f in (0.5, 2.0):
        for win_ms in (40, 100):
            win = win_ms // 2
            n1 = nf = paar = u1 = uf = 0
            n_rec_ch = n_rec_pairs = 0
            shifts = []
            rel_ch = 0
            rec_shift = 0
            for aid in sg:
                a = sch[(sch.aufnahme == aid) & (sch.vorverarbeitung == FAKT[1.0])].sort_values("r_gipfel")
                b = sch[(sch.aufnahme == aid) & (sch.vorverarbeitung == FAKT[f])].sort_values("r_gipfel")
                n1 += len(a)
                nf += len(b)
                if len(a) == 0 or len(b) == 0:
                    u1 += len(a)
                    uf += len(b)
                    continue
                ra, rb = a.r_gipfel.to_numpy(float), b.r_gipfel.to_numpy(float)
                qa, qb = (a.qrs_beginn - a.r_gipfel).to_numpy(float), (b.qrs_beginn - b.r_gipfel).to_numpy(float)
                usedb = np.zeros(len(rb), bool)
                prs = 0
                chg = False
                sh_rec = False
                for i in range(len(ra)):
                    dd = np.abs(rb - ra[i])
                    dd[usedb] = 1e9
                    j = int(np.argmin(dd))
                    if dd[j] <= win:
                        usedb[j] = True
                        prs += 1
                        if rb[j] != ra[i]:
                            sh_rec = True
                            shifts.append((rb[j] - ra[i]) * 2)
                        if np.isfinite(qa[i]) and np.isfinite(qb[j]) and qa[i] != qb[j]:
                            chg = True
                            rel_ch += 1
                paar += prs
                u1 += len(ra) - prs
                uf += int((~usedb).sum())
                n_rec_pairs += 1
                n_rec_ch += chg
                rec_shift += sh_rec
            rows_b.append(dict(faktor=f, fenster_ms=win_ms, schlaege_faktor1=n1, schlaege_faktor_f=nf, zugeordnet=paar, nicht_zugeordnet_faktor1=u1, nicht_zugeordnet_faktor_f=uf,
                               aufnahmen_mit_zuordnung=n_rec_pairs, aufnahmen_mit_verschobenem_R=rec_shift, aufnahmen_mit_geaendertem_QRS_beginn_relativ_zu_R=n_rec_ch,
                               verschiebung_ms_median=float(np.median(shifts)) if shifts else np.nan, verschiebung_ms_max_betrag=float(np.max(np.abs(shifts))) if shifts else np.nan))
    df_b = pd.DataFrame(rows_b)

    def schreibe(df, name):
        df.to_csv(AB.OUT_PUB / name, index=False, encoding="utf-8", float_format="%.6g")
        df.to_csv(AB.OUT_ROH / name, index=False, encoding="utf-8", float_format="%.6g")
    schreibe(df_des, "r7_design.csv")
    schreibe(df_z, "r7_veraenderung.csv")
    schreibe(df_s, "r7_schwellenwechsel.csv")
    schreibe(df_b, "r7_schlagzuordnung_s10b.csv")
    schreibe(k1p, "r7_nullkontrolle_k1_gelesen.csv")
    schreibe(pd.DataFrame(KONTROLLEN), "r7_kontrollen.csv")
    schreibe(pd.DataFrame(pruef), "r7_hashpruefung.csv")
    info = dict(version="1.0", seed=SEED, B=B, laufzeit_s=round(time.time() - t0, 1), kontrollen_bestanden=int(sum(k["bestanden"] for k in KONTROLLEN)), kontrollen_gesamt=len(KONTROLLEN),
                kish_n_eff_gesamt=n_eff_all)
    (AB.OUT_PUB / "r7_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("done in %.1f s; checks passed %d of %d" % (time.time() - t0, info["kontrollen_bestanden"], info["kontrollen_gesamt"]))
    if info["kontrollen_bestanden"] != info["kontrollen_gesamt"]:
        raise SystemExit("NOT ALL CHECKS PASSED")


if __name__ == "__main__":
    lauf()
