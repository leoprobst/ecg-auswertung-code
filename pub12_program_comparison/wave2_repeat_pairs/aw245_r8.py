# -*- coding: utf-8 -*-
"""aw245_r8.py (version 1.0): failures, common comparison set and extreme assumptions (program comparison study, exploratory, post hoc).

Population P0 (18,869 persons, one recording per person). Locked tables are only read (hash check of the data lock in aw239_basis.lade_basis).
(a) Failures per program and reason: LoopLens (status of the QRS duration, separately for QT), BRAVEHEART (tool errors with reason), 12SL/Uni-G/ECGDeli (value missing: recording not in the
    feature table or value empty).
(b) Characteristics of the failures against those available (age, sex, CLBBB, CRBBB, IVCD, PACE, atrial fibrillation/flutter, QRS duration of 12SL as a proxy for wide complexes).
(c) Common complete set: QRS (LoopLens, BRAVEHEART, 12SL, Uni-G; and with ECGDeli) and QTc (four programs with QT, RR and delivered QTc): all 10 QRS pairs and the QTc pairs
    on the pairwise set and on the common set (D120/130/150, class change, bias, SD, ICC; CLBBB subset D150; QTc D450/470/500).
(d) Extreme assumptions: all missing values concordant against all discordant for D150 in CLBBB and QTc D450 (denominator = all persons of the subset), and a reweighting by
    cells (QRS band of 12SL x CLBBB x pacemaker code x sex) as a point estimate.
Intervals: percentile bootstrap over persons, 2,000 draws, fixed seed, common draws.
Usage: python -I -B aw245_r8.py --modus lauf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aw245_basis as AB  # noqa: E402

BA = AB.BA
S = AB.S
log = AB.log
SEED = 15102452
WZ = AB.WERKZEUGE
QRS_S = (120, 130, 150)
QTC_S = (450, 470, 500)
KONTROLLEN = []


def kontrolle(name, soll, ist, ok):
    KONTROLLEN.append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok)))
    log("CHECK %s: expected %s actual %s -> %s" % (name, soll, ist, "ok" if ok else "DEVIATION"))


def status_tabelle(ids, pfad, messgroesse):
    cols = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status", "statusgrund"]
    d = AB._chunks_gefiltert(pfad, set(int(i) for i in ids), cols, [messgroesse])
    d = d[d.messgroesse == messgroesse].copy()
    d["aufnahme"] = d["aufnahme"].astype(int)
    d = d[~d.aufnahme.duplicated(keep="first")].set_index("aufnahme")
    return d.reindex(ids)


def kuerze_grund(s):
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    if s.startswith("not_evaluable"):
        return "Konfidenz der QRS-Marken unter 0,35 (nicht auswertbar)"
    if s.startswith("amplitude_outlier") or "clipping" in s or "flatline" in s:
        return "Signalqualitaetskontrolle (" + s.split("=")[0].split("|")[0] + ")"
    if s.startswith("BRAVEHEART errors-Liste"):
        return "BRAVEHEART " + s.split(":", 1)[1].strip()[:60]
    return s[:70]


def merkmale_zeile(B, miss, ok, name):
    rows = []
    qrs12 = B.arch["12SL"]["QRS_Dur_Global"]
    age = np.where(B.age > 120, 90.0, B.age)  # PTB-XL codes ages above 89 as 300 (anonymisation); counted here as 90
    def stat(sel, label):
        n = int(sel.sum())
        a = age[sel]
        r = dict(gruppe=label, n=n, alter_mittel=float(np.nanmean(a)) if n else np.nan, alter_sd=float(np.nanstd(a, ddof=1)) if n > 1 else np.nan,
                 weiblich_pct=100 * float(np.nanmean(B.sex[sel] == 1)) if n else np.nan)
        for c in ("CLBBB", "CRBBB", "IVCD", "PACE"):
            r["%s_pct" % c] = 100 * float(B.code[c][sel].mean()) if n else np.nan
        r["AF_pct"] = 100 * float((B.code["AFIB"] | B.code["AFLT"])[sel].mean()) if n else np.nan
        q = qrs12[sel]
        q = q[np.isfinite(q)]
        r["qrs12sl_median"] = float(np.median(q)) if len(q) else np.nan
        r["qrs12sl_q25"] = float(np.percentile(q, 25)) if len(q) else np.nan
        r["qrs12sl_q75"] = float(np.percentile(q, 75)) if len(q) else np.nan
        r["qrs12sl_ge120_pct"] = 100 * float(np.mean(q >= 120)) if len(q) else np.nan
        return r
    a, b = stat(miss, name + " fehlt"), stat(ok, name + " vorhanden")
    def smd(k):
        # standardised difference
        x1, x2 = a[k], b[k]
        if k in ("alter_mittel",):
            s = np.sqrt((a["alter_sd"] ** 2 + b["alter_sd"] ** 2) / 2)
            return (x1 - x2) / s if s > 0 else np.nan
        if k.endswith("_pct"):
            p1, p2 = x1 / 100, x2 / 100
            s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
            return (p1 - p2) / s if s > 0 else np.nan
        return np.nan
    a["smd_alter"] = smd("alter_mittel")
    for k in ("weiblich_pct", "CLBBB_pct", "CRBBB_pct", "IVCD_pct", "PACE_pct", "AF_pct", "qrs12sl_ge120_pct"):
        a["smd_" + k.replace("_pct", "")] = smd(k)
    rows += [a, b]
    return rows


def poststrat_d(disc, avail, cells, mask_pop):
    """Reweighting: weight 1 / availability share of the cell for persons with a value; returns the weighted share of discordance in mask_pop."""
    df = pd.DataFrame(dict(av=avail & mask_pop, c=cells, pop=mask_pop))
    p = df[df["pop"]].groupby("c")["av"].mean()
    w = df["c"].map(1.0 / p).where(df.av)
    wd = (w * disc.astype(float)).where(df.av).sum()
    return float(wd / w.where(df.av).sum())


def lauf():
    t0 = time.time()
    AB.OUT_ROH.mkdir(parents=True, exist_ok=True)
    AB.OUT_PUB.mkdir(parents=True, exist_ok=True)
    S.STATE["log"] = str(AB.OUT_ROH / "r8_lauf.log")
    log("aw245_r8 version 1.0")
    B = BA.lade_basis(mit_hash=True)
    pruef = list(B.pruef)
    ids, D = B.ids, B.D
    K = len(ids)
    kontrolle("P0 persons", 18869, K, K == 18869)
    qrs = {t: D[t]["qrs"] for t in WZ}
    # ------------------------------------------------------------------ (a) failures
    P = S.PFADE
    stat = {}
    stat["LoopLens"] = status_tabelle(ids, P["erg"] / "looplens/langformat/looplens_langformat.csv.gz", "qrs_dauer_global")
    stat["LoopLens_qt"] = status_tabelle(ids, P["erg"] / "looplens/langformat/looplens_langformat.csv.gz", "qt_global")
    stat["BRAVEHEART"] = status_tabelle(ids, P["erg"] / "ptbxl/ausgabe_langformat.csv", "qrs_dauer_global")
    stat["BRAVEHEART_qt"] = status_tabelle(ids, P["erg"] / "ptbxl/ausgabe_langformat.csv", "qt_global")
    rows_a = []
    for tool, key in (("LoopLens", "LoopLens"), ("LoopLens", "LoopLens_qt"), ("BRAVEHEART", "BRAVEHEART"), ("BRAVEHEART", "BRAVEHEART_qt")):
        d = stat[key]
        gr = d["status"].fillna("keine Zeile") + " | " + d["statusgrund"].map(kuerze_grund)
        gr = gr.where(d["status"] != "ok", "ok")
        g = d["status"].fillna("keine Zeile").where(d["status"] != "ok", "ok")
        for k, v in gr.value_counts().items():
            rows_a.append(dict(programm=tool, groesse="QRS-Dauer" if not key.endswith("_qt") else "QT", grund=k, n=int(v), anteil_pct=100 * v / K))
    # archive products: value missing
    for tool, f in (("12SL", "12sl_features.csv"), ("Uni-G", "unig_features.csv"), ("ECGDeli", "ecgdeli_features.csv")):
        df = pd.read_csv(P["plus"] / f, usecols=["ecg_id", "QRS_Dur_Global", "QT_Int_Global"])
        df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id")
        in_tab = pd.Series(ids).isin(df.index).to_numpy()
        for gname, col in (("QRS-Dauer", "QRS_Dur_Global"), ("QT", "QT_Int_Global")):
            vals = df[col].reindex(ids).to_numpy(float)
            rows_a.append(dict(programm=tool, groesse=gname, grund="ok", n=int(np.isfinite(vals).sum()), anteil_pct=100 * np.isfinite(vals).mean()))
            rows_a.append(dict(programm=tool, groesse=gname, grund="Aufnahme nicht in der Featuretabelle", n=int((~in_tab).sum()), anteil_pct=100 * (~in_tab).mean()))
            rows_a.append(dict(programm=tool, groesse=gname, grund="Aufnahme in Tabelle, Wert leer", n=int((in_tab & ~np.isfinite(vals)).sum()), anteil_pct=100 * (in_tab & ~np.isfinite(vals)).mean()))
    df_a = pd.DataFrame(rows_a)
    # Checks against the plan: LoopLens 85 missing and 4 QC (QRS), BRAVEHEART 206 tool errors (QRS)
    ll = stat["LoopLens"]
    n_fehl = int((ll["status"] == "fehlend").sum())
    n_qc = int((ll["status"] == "ausgeschlossen_qc").sum())
    n_bh = int((stat["BRAVEHEART"]["status"] == "werkzeugfehler").sum())
    kontrolle("LoopLens missing (plan: 85)", 85, n_fehl, n_fehl == 85)
    kontrolle("LoopLens QC exclusion (plan: 4)", 4, n_qc, n_qc == 4)
    kontrolle("BRAVEHEART tool errors (plan: 206)", 206, n_bh, n_bh == 206)
    for t in WZ:
        n_fin = int(np.isfinite(qrs[t]).sum())
        a_ok = int((stat[t]["status"] == "ok").sum()) if t in ("LoopLens", "BRAVEHEART") else n_fin
        if t in ("LoopLens", "BRAVEHEART"):
            kontrolle("%s: number of finite QRS values equals number of status ok" % t, a_ok, n_fin, a_ok == n_fin)
    # ------------------------------------------------------------------ (b) characteristics
    rows_b = []
    for t in WZ:
        ok = np.isfinite(qrs[t])
        rows_b += [dict(programm=t, **r) for r in merkmale_zeile(B, ~ok, ok, t)]
    df_b = pd.DataFrame(rows_b)
    # ------------------------------------------------------------------ (c) common sets, pairwise against common set
    fin = {t: np.isfinite(qrs[t]) for t in WZ}
    c4 = fin["LoopLens"] & fin["BRAVEHEART"] & fin["12SL"] & fin["Uni-G"]
    c5 = c4 & fin["ECGDeli"]
    kontrolle("common QRS set (four programs)", "count", int(c4.sum()), True)
    kontrolle("common QRS set (five programs)", "count", int(c5.sum()), True)
    deliv = {"12SL": B.arch["12SL"]["QT_IntCorr_Global"], "Uni-G": B.arch["Uni-G"]["QT_IntCorr_Global"], "BRAVEHEART": D["BRAVEHEART"]["qtc_nativ"], "LoopLens": D["LoopLens"]["qtc_nativ"]}
    t4 = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G"]
    qc4 = np.all([np.isfinite(D[t]["qt"]) & np.isfinite(D[t]["rr"]) & np.isfinite(deliv[t]) for t in t4], axis=0)
    kontrolle("common QTc set C4 (preceding analysis: 17,833)", 17833, int(qc4.sum()), int(qc4.sum()) == 17833)
    qe5 = np.all([np.isfinite(D[t]["qtc_einh"]) for t in WZ], axis=0)
    qe4 = np.all([np.isfinite(D[t]["qtc_einh"]) for t in t4], axis=0)
    clb = B.code["CLBBB"]
    alle = np.ones(K, bool)
    zieh = BA.ziehungen(K, SEED)
    bk = BA.Baukasten(K)
    pairs = [BA.orientiere(WZ[i], WZ[j]) for i in range(len(WZ)) for j in range(i + 1, len(WZ))]
    for ta, tb in pairs:
        for sub, sm in (("alle", alle), ("CLBBB", clb)):
            BA.spalten_paar(bk, "q|%s|%s|%s|paarweise" % (ta, tb, sub), qrs[ta], qrs[tb], sm)
            BA.spalten_paar(bk, "q|%s|%s|%s|gem4" % (ta, tb, sub), qrs[ta], qrs[tb], sm & c4)
            if ta != "ECGDeli" and tb != "ECGDeli":
                pass
            BA.spalten_paar(bk, "q|%s|%s|%s|gem5" % (ta, tb, sub), qrs[ta], qrs[tb], sm & c5)
    qpairs = [(a, b) for a, b in pairs if a != "ECGDeli" and b != "ECGDeli"]
    for ta, tb in qpairs:
        BA.spalten_paar(bk, "c|%s|%s|geliefert|paarweise" % (ta, tb), deliv[ta], deliv[tb], alle, schwellen=QTC_S, klassen=False)
        BA.spalten_paar(bk, "c|%s|%s|geliefert|gem4" % (ta, tb), deliv[ta], deliv[tb], qc4, schwellen=QTC_S, klassen=False)
    for ta, tb in pairs:
        BA.spalten_paar(bk, "c|%s|%s|einheitlich|paarweise" % (ta, tb), D[ta]["qtc_einh"], D[tb]["qtc_einh"], alle, schwellen=QTC_S, klassen=False)
        BA.spalten_paar(bk, "c|%s|%s|einheitlich|gem5" % (ta, tb), D[ta]["qtc_einh"], D[tb]["qtc_einh"], qe5, schwellen=QTC_S, klassen=False)
        if ta != "ECGDeli" and tb != "ECGDeli":
            BA.spalten_paar(bk, "c|%s|%s|einheitlich|gem4" % (ta, tb), D[ta]["qtc_einh"], D[tb]["qtc_einh"], qe4, schwellen=QTC_S, klassen=False)
    log("Columns built, bootstrap")
    Sx = BA.boot_summen(bk, zieh)
    rows_c = []
    for pref in sorted({k.rsplit("|", 1)[0] for k in Sx if k.endswith("|V")}):
        parts = pref.split("|")
        art = parts[0]
        schw = QRS_S if art == "q" else QTC_S
        n = Sx[pref + "|V"]
        sm = BA.stats_moment(Sx, pref)
        r = dict(messung="QRS" if art == "q" else "QTc", kandidat=parts[1], komparator=parts[2], teilmenge=parts[3] if art == "q" else "alle", menge=parts[-1] if art == "q" else "%s/%s" % (parts[3], parts[4]), n=int(n[0]))
        if art == "c":
            r["variante"] = parts[3]
            r["menge"] = parts[4]
        for t in schw:
            s2 = BA.stats_2x2(Sx, pref, t)
            p, lo, hi = BA.ci(np.asarray(s2["disc"], float) * 100)
            r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t], r["n_disc_%d" % t] = p, lo, hi, int(s2["n_disc"][0])
        if art == "q":
            with np.errstate(invalid="ignore", divide="ignore"):
                w = Sx[pref + "|wechsel"] / n * 100
            r["wechsel_pct"], r["wechsel_ki_u"], r["wechsel_ki_o"] = BA.ci(w)
        for k, nm in (("bias", "bias_ms"), ("sd", "sd_ms"), ("icc", "icc21")):
            r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = BA.ci(sm[k])
        rows_c.append(r)
    df_c = pd.DataFrame(rows_c)
    kontrolle("Rebuild of the preceding analysis: BRAVEHEART against 12SL D150 CLBBB pairwise (14.15 %)", "14.15", "%.2f" % float(df_c[(df_c.messung == "QRS") & (df_c.kandidat == "BRAVEHEART") & (df_c.komparator == "12SL") & (df_c.teilmenge == "CLBBB") & (df_c.menge == "paarweise")].D150_pct.iloc[0]),
              abs(float(df_c[(df_c.messung == "QRS") & (df_c.kandidat == "BRAVEHEART") & (df_c.komparator == "12SL") & (df_c.teilmenge == "CLBBB") & (df_c.menge == "paarweise")].D150_pct.iloc[0]) - 14.15) < 0.01)
    # ------------------------------------------------------------------ (d) extreme assumptions and reweighting
    qrs12 = B.arch["12SL"]["QRS_Dur_Global"]
    band = np.select([~np.isfinite(qrs12), qrs12 < 100, qrs12 < 120, qrs12 < 140, qrs12 < 160], [0, 1, 2, 3, 4], 5)
    cells = pd.Series(band.astype(int).astype(str) + "|" + clb.astype(int).astype(str) + "|" + B.code["PACE"].astype(int).astype(str) + "|" + np.nan_to_num(B.sex, nan=-1).astype(int).astype(str)).to_numpy()
    rows_d = []
    for ta, tb in [p for p in pairs]:
        a, b = qrs[ta], qrs[tb]
        okp = np.isfinite(a) & np.isfinite(b)
        for sub, sm in (("CLBBB", clb), ("alle", alle)):
            ntot = int(sm.sum())
            nok = int((okp & sm).sum())
            disc = ((a >= 150) != (b >= 150)) & okp
            nd = int((disc & sm).sum())
            lo, hi = nd / ntot, (nd + (ntot - nok)) / ntot
            rows_d.append(dict(messung="QRS D150", kandidat=ta, komparator=tb, teilmenge=sub, personen_teilmenge=ntot, vorhanden=nok, fehlend=ntot - nok, diskordant=nd,
                               D_beobachtet_pct=100 * nd / nok if nok else np.nan, schranke_unten_pct=100 * lo, schranke_oben_pct=100 * hi,
                               D_nachgewichtet_pct=100 * poststrat_d(disc, okp, cells, sm) if nok else np.nan))
    for ta, tb in qpairs:
        a, b = deliv[ta], deliv[tb]
        okp = np.isfinite(a) & np.isfinite(b)
        for thr in QTC_S:
            disc = ((a >= thr) != (b >= thr)) & okp
            ntot = K
            nok = int(okp.sum())
            nd = int(disc.sum())
            rows_d.append(dict(messung="QTc geliefert D%d" % thr, kandidat=ta, komparator=tb, teilmenge="alle", personen_teilmenge=ntot, vorhanden=nok, fehlend=ntot - nok, diskordant=nd,
                               D_beobachtet_pct=100 * nd / nok, schranke_unten_pct=100 * nd / ntot, schranke_oben_pct=100 * (nd + ntot - nok) / ntot,
                               D_nachgewichtet_pct=100 * poststrat_d(disc, okp, cells, alle)))
    for ta, tb in pairs:
        a, b = D[ta]["qtc_einh"], D[tb]["qtc_einh"]
        okp = np.isfinite(a) & np.isfinite(b)
        for thr in QTC_S:
            disc = ((a >= thr) != (b >= thr)) & okp
            nok = int(okp.sum())
            nd = int(disc.sum())
            rows_d.append(dict(messung="QTc einheitlich D%d" % thr, kandidat=ta, komparator=tb, teilmenge="alle", personen_teilmenge=K, vorhanden=nok, fehlend=K - nok, diskordant=nd,
                               D_beobachtet_pct=100 * nd / nok, schranke_unten_pct=100 * nd / K, schranke_oben_pct=100 * (nd + K - nok) / K,
                               D_nachgewichtet_pct=100 * poststrat_d(disc, okp, cells, alle)))
    df_d = pd.DataFrame(rows_d)
    # cells of the reweighting (count)
    vc = pd.Series(cells).value_counts()
    kontrolle("Reweighting: cells", "count", int(len(vc)), True)

    def schreibe(df, name):
        df.to_csv(AB.OUT_PUB / name, index=False, encoding="utf-8", float_format="%.6g")
        df.to_csv(AB.OUT_ROH / name, index=False, encoding="utf-8", float_format="%.6g")
    schreibe(df_a, "r8_ausfaelle_gruende.csv")
    schreibe(df_b, "r8_ausfaelle_merkmale.csv")
    schreibe(df_c, "r8_gemeinsame_menge.csv")
    schreibe(df_d, "r8_extremannahmen.csv")
    schreibe(pd.DataFrame(KONTROLLEN), "r8_kontrollen.csv")
    schreibe(pd.DataFrame(pruef), "r8_hashpruefung.csv")
    info = dict(version="1.0", seed=SEED, B=2000, gemeinsame_qrs_vier=int(c4.sum()), gemeinsame_qrs_fuenf=int(c5.sum()), gemeinsame_qtc_c4=int(qc4.sum()),
                gemeinsame_qtc_einheitlich_vier=int(qe4.sum()), gemeinsame_qtc_einheitlich_fuenf=int(qe5.sum()), laufzeit_s=round(time.time() - t0, 1),
                kontrollen_bestanden=int(sum(k["bestanden"] for k in KONTROLLEN)), kontrollen_gesamt=len(KONTROLLEN))
    (AB.OUT_PUB / "r8_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("done in %.1f s; checks passed %d of %d" % (time.time() - t0, info["kontrollen_bestanden"], info["kontrollen_gesamt"]))
    if info["kontrollen_bestanden"] != info["kontrollen_gesamt"]:
        raise SystemExit("NOT ALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", default="lauf")
    ap.parse_args()
    lauf()
