# -*- coding: utf-8 -*-
"""aw239_r5.py (AW-239, P12-R5, Version 1.0): QTc, Formeleffekt isoliert, QT- und RR-Beitrag getrennt (PUB-12, explorativ, post hoc).

Population P0 (18.869 Personen), native Ausgaben. Eingaben: QT und RR je Programm (12SL, Uni-G aus PTB-XL+ mit RR = 60000 / HR__Global; BRAVEHEART RR = 60000 / hr;
LoopLens rr_intervall_ll). Standardformel je Programm: 12SL Bazett, Uni-G Hodge, BRAVEHEART und LoopLens Fridericia. Alle QTc-Werte dieses Skripts werden aus QT und RR
NEU berechnet (qtc_formel in aw239_basis.py), ausser der Spalte "geliefert" (QT_IntCorr_Global bzw. qtcf nativ). Die Neuberechnung wird gegen die gelieferten Spalten
kontrolliert (Anteil innerhalb 1 ms).

(a) Innerhalb eines Programms: QTc nach Bazett, Fridericia, Hodge, Framingham aus demselben QT und RR; Anteil Personen, die eine Schwelle (neutral 450/470/500,
    geschlechtsspezifisch 450 M / 460 F) mit einer Formel erreichen und mit der anderen nicht. Mittlere Differenz der Formeln.
(b) Zwischen Programmen, gemeinsame Personenmenge C4 (alle vier Programme mit gueltigem QT, RR und geliefertem Standard-QTc): Diskordanz je Paar bei
    - geliefertem Standard-QTc,
    - neu berechneter Standardformel je Programm (3 Faktoren: Formel, QT, RR; jeder Faktor von A oder B),
    - einheitlicher Fridericia-Formel (2 Faktoren: QT, RR);
    kontrafaktische Vertauschung: nur die Formel, nur das QT oder nur das RR wird von A auf B gewechselt (bei den Eingaben von A und von B).
    Kontinuierliche Zerlegung der Differenz QTc_A minus QTc_B in Formel-, QT- und RR-Beitrag nach Shapley (alle Reihenfolgen gemittelt; Summe = Differenz exakt),
    mit Kovarianzanteil an der Varianz der Differenz.
(c) Geschlechtsspezifische Schwellen und Herzfrequenzstrata (12SL HR__Global: unter 60, 60 bis 100, ueber 100) fuer alle Vergleiche aus (b).
(d) LoopLens: gelieferte QTcF-, QT- und RR-Werte gegen Kandidatenrechnungen aus den gespeicherten Einzelschlaegen (looplens_schlaege.csv.gz, VV0 nativ).
Intervalle: Perzentil-Bootstrap ueber Personen, 2.000 Ziehungen, fester Seed. Die Schwellenstufen 450/470/500 (460/480 Frauen) sind die Vorgabe aus AW-234; ihre Quelle
(Rautaharju 2009) wurde hier nicht am Volltext nachgelesen.
Aufruf: <local path> -I -B aw239_r5.py --modus lauf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

VERSION = "1.0"
SEED = 15102395
BASIS_PFAD = Path(str(data_path("pub12.werkzeuge_pub12_aw239_aw239_basis_py")))
BASIS_HASH = "50fcf31d788931c671bd0021e112d2f9272795daecd2c24bbd215af0b0d4c7bd"
OUT = Path(str(data_path("pub12.benchmark_pub12_aw239_r5")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw239")))
SCHLAEGE = Path(str(data_path("pub12.benchmark_ergebnisse_looplens_langformat_looplens_schlaege_csv_gz")))


def _lade_basis():
    h = hashlib.sha256(BASIS_PFAD.read_bytes()).hexdigest()
    if BASIS_HASH != "UNGESETZT" and h != BASIS_HASH:
        raise SystemExit("aw239_basis.py weicht ab: %s" % h)
    spec = importlib.util.spec_from_file_location("aw239_basis", str(BASIS_PFAD))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aw239_basis"] = mod
    spec.loader.exec_module(mod)
    return mod, h


BA, BASIS_IST = _lade_basis()
log = BA.log

FORMELN = ("Bazett", "Fridericia", "Hodge", "Framingham")
DEFAULT = {"12SL": "Bazett", "Uni-G": "Hodge", "BRAVEHEART": "Fridericia", "LoopLens": "Fridericia", "ECGDeli": "Framingham"}
T4 = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G"]
SCHEMEN = {"n450": (450, 450), "n470": (470, 470), "n500": (500, 500), "ss450_460": (450, 460), "ss470_480": (470, 480)}
SCHEMEN_INNEN = ("n450", "n470", "n500", "ss450_460")
HR_STRATA = {"alle": None, "HR_unter60": (-np.inf, 60), "HR_60bis100": (60, 100.0000001), "HR_ueber100": (100.0000001, np.inf)}


def schwelle(sex, scheme):
    m, f = SCHEMEN[scheme]
    return np.where(np.asarray(sex) == 0, float(m), float(f))


def paare4():
    return [BA.orientiere(T4[i], T4[j]) for i in range(len(T4)) for j in range(i + 1, len(T4))]


def disc_ind(x, y, thr, ok):
    return ok & ((x >= thr) ^ (y >= thr))


def vergleiche(Q, R, deliv, ta, tb):
    """Alle Vergleichspaare (Name -> (x, y)) fuer ein Paar (A, B). Q, R: dict Programm -> QT, RR. deliv: geliefertes Standard-QTc."""
    fa, fb = DEFAULT[ta], DEFAULT[tb]
    qt = {"A": Q[ta], "B": Q[tb]}
    rr = {"A": R[ta], "B": R[tb]}
    fm = {"A": fa, "B": fb}

    def v(F, q, r):
        return BA.qtc_formel(fm[F], qt[q], rr[r])

    def g(q, r):
        return BA.qtc_formel("Fridericia", qt[q], rr[r])

    c = {
        "geliefert_gesamt": (deliv[ta], deliv[tb]),
        "standard_gesamt": (v("A", "A", "A"), v("B", "B", "B")),
        "standard_nur_formel_bei_A": (v("A", "A", "A"), v("B", "A", "A")),
        "standard_nur_formel_bei_B": (v("A", "B", "B"), v("B", "B", "B")),
        "standard_nur_QT_bei_A": (v("A", "A", "A"), v("A", "B", "A")),
        "standard_nur_QT_bei_B": (v("B", "A", "B"), v("B", "B", "B")),
        "standard_nur_RR_bei_A": (v("A", "A", "A"), v("A", "A", "B")),
        "standard_nur_RR_bei_B": (v("B", "B", "A"), v("B", "B", "B")),
        "fridericia_gesamt": (g("A", "A"), g("B", "B")),
        "fridericia_nur_QT_bei_RR_A": (g("A", "A"), g("B", "A")),
        "fridericia_nur_QT_bei_RR_B": (g("A", "B"), g("B", "B")),
        "fridericia_nur_RR_bei_QT_A": (g("A", "A"), g("A", "B")),
        "fridericia_nur_RR_bei_QT_B": (g("B", "A"), g("B", "B")),
    }
    # Shapley-Zerlegung Standard (3 Faktoren) und Fridericia (2 Faktoren); v(S) mit S = Menge der Faktoren, die von B genommen werden
    cube = {(f, q, r): v(f, q, r) for f in "AB" for q in "AB" for r in "AB"}
    delta3 = cube[("A", "A", "A")] - cube[("B", "B", "B")]

    def shap3(i):
        tot = 0.0
        for S in range(8):
            bits = [(S >> k) & 1 for k in range(3)]
            if bits[i]:
                continue
            n_s = sum(bits)
            w = {0: 1 / 3, 1: 1 / 6, 2: 1 / 3}[n_s]
            b1 = list(bits)
            b1[i] = 1
            key = lambda bb: tuple("B" if x else "A" for x in bb)  # noqa: E731
            tot = tot + w * (cube[key(bits)] - cube[key(b1)])
        return tot

    comp = {"standard": dict(delta=delta3, formel=shap3(0), QT=shap3(1), RR=shap3(2))}
    d2 = g("A", "A") - g("B", "B")
    cq = 0.5 * ((g("A", "A") - g("B", "A")) + (g("A", "B") - g("B", "B")))
    cr = 0.5 * ((g("B", "A") - g("B", "B")) + (g("A", "A") - g("A", "B")))
    comp["fridericia"] = dict(delta=d2, QT=cq, RR=cr)
    return c, comp


def baue(B, C4, strata_hr, Q, R, deliv):
    K = len(B.ids)
    bk = BA.Baukasten(K)
    sex = B.sex
    thr = {s: schwelle(sex, s) for s in SCHEMEN}
    for sname, ok_s in strata_hr.items():
        bk.indikator("C4|V|%s" % sname, C4 & ok_s)
    for ta, tb in paare4():
        cmp_, comp = vergleiche(Q, R, deliv, ta, tb)
        for sname, ok_s in strata_hr.items():
            ok = C4 & ok_s
            for cn, (x, y) in cmp_.items():
                for sc in SCHEMEN:
                    bk.indikator("zw|%s|%s|%s|%s|%s" % (ta, tb, sname, cn, sc), disc_ind(x, y, thr[sc], ok))
        for mode, dd in comp.items():
            for nm, v in dd.items():
                v = np.where(C4, v, 0.0)
                bk.moment("zl|%s|%s|%s|%s|x" % (ta, tb, mode, nm), v)
                bk.moment("zl|%s|%s|%s|%s|x2" % (ta, tb, mode, nm), v * v)
                bk.moment("zl|%s|%s|%s|%s|xd" % (ta, tb, mode, nm), v * np.where(C4, dd["delta"], 0.0))
                bk.moment("zl|%s|%s|%s|%s|abs" % (ta, tb, mode, nm), np.abs(v))
    # innerhalb eines Programms
    for prog in BA.WERKZEUGE:
        qt, rr = B.D[prog]["qt"], B.D[prog]["rr"]
        ok = np.isfinite(qt) & np.isfinite(rr) & (rr > 0)
        bk.indikator("in|%s|V" % prog, ok)
        F = {f: BA.qtc_formel(f, qt, rr) for f in FORMELN}
        for i in range(len(FORMELN)):
            for j in range(i + 1, len(FORMELN)):
                f1, f2 = FORMELN[i], FORMELN[j]
                okp = ok & np.isfinite(F[f1]) & np.isfinite(F[f2])
                bk.indikator("in|%s|%s|%s|V" % (prog, f1, f2), okp)
                for sc in SCHEMEN_INNEN:
                    bk.indikator("in|%s|%s|%s|%s" % (prog, f1, f2, sc), disc_ind(F[f1], F[f2], thr[sc], okp))
                d = np.where(okp, F[f1] - F[f2], 0.0)
                bk.moment("in|%s|%s|%s|d" % (prog, f1, f2), d)
                bk.moment("in|%s|%s|%s|d2" % (prog, f1, f2), d * d)
    return bk


def tabellen(Sx, B, C4, strata_hr, Q, R, deliv):
    T = {k: [] for k in ("innen", "zwischen", "zerlegung")}
    ci = BA.ci
    for prog in BA.WERKZEUGE:
        for i in range(len(FORMELN)):
            for j in range(i + 1, len(FORMELN)):
                f1, f2 = FORMELN[i], FORMELN[j]
                V = Sx["in|%s|%s|%s|V" % (prog, f1, f2)]
                n = int(V[0])
                md = Sx["in|%s|%s|%s|d" % (prog, f1, f2)] / V
                sd = np.sqrt(np.maximum((Sx["in|%s|%s|%s|d2" % (prog, f1, f2)] / V - md ** 2) * V / (V - 1), 0))
                for sc in SCHEMEN_INNEN:
                    num = Sx["in|%s|%s|%s|%s" % (prog, f1, f2, sc)]
                    r = dict(programm=prog, formel_1=f1, formel_2=f2, schwellenstufe=sc, n=n, n_disc=int(num[0]))
                    r["D_pct"], r["D_ki_u"], r["D_ki_o"] = ci(100 * num / V)
                    r["mittlere_diff_f1_minus_f2_ms"], r["diff_ki_u"], r["diff_ki_o"] = ci(md)
                    r["sd_diff_ms"] = float(sd[0])
                    T["innen"].append(r)
    for ta, tb in paare4():
        cmp_, comp = vergleiche(Q, R, deliv, ta, tb)
        for sname in strata_hr:
            V = Sx["C4|V|%s" % sname]
            for cn in cmp_:
                for sc in SCHEMEN:
                    num = Sx["zw|%s|%s|%s|%s|%s" % (ta, tb, sname, cn, sc)]
                    r = dict(kandidat=ta, komparator=tb, stratum=sname, vergleich=cn, schwellenstufe=sc, n=int(V[0]), n_disc=int(num[0]))
                    r["D_pct"], r["D_ki_u"], r["D_ki_o"] = ci(100 * num / V)
                    T["zwischen"].append(r)
        V = Sx["C4|V|alle"]
        for mode, dd in comp.items():
            dlt = "zl|%s|%s|%s|delta|" % (ta, tb, mode)
            m_d = Sx[dlt + "x"] / V
            var_d = Sx[dlt + "x2"] / V - m_d ** 2
            for nm in dd:
                p = "zl|%s|%s|%s|%s|" % (ta, tb, mode, nm)
                mx = Sx[p + "x"] / V
                cov = Sx[p + "xd"] / V - mx * m_d
                r = dict(kandidat=ta, komparator=tb, modus=mode, komponente=nm, n=int(V[0]))
                r["mittel_ms"], r["mittel_ki_u"], r["mittel_ki_o"] = ci(mx)
                r["mittel_betrag_ms"], r["betrag_ki_u"], r["betrag_ki_o"] = ci(Sx[p + "abs"] / V)
                r["sd_ms"] = float(np.sqrt(max(Sx[p + "x2"][0] / V[0] - mx[0] ** 2, 0)))
                r["kovarianzanteil_an_varianz_diff"], r["kovanteil_ki_u"], r["kovanteil_ki_o"] = ci(cov / var_d)
                T["zerlegung"].append(r)
    return {k: pd.DataFrame(v) for k, v in T.items()}


def rekonstruktion_und_median(B):
    rows, med = [], []
    P = B.arch
    spalten = {"12SL": {"Bazett": "QT_IntBazett_Global", "Fridericia": "QT_IntFridericia_Global", "Framingham": "QT_IntFramingham_Global"},
               "Uni-G": {"Bazett": "QT_IntBazett_Global", "Fridericia": "QT_IntFridericia_Global", "Hodge": "QT_IntHodge_Global", "Framingham": "QT_IntFramingham_Global"},
               "ECGDeli": {"Framingham": "QT_IntFramingham_Global"}}
    for prog in BA.WERKZEUGE:
        qt, rr = B.D[prog]["qt"], B.D[prog]["rr"]
        for f in FORMELN:
            x = BA.qtc_formel(f, qt, rr)
            ok = np.isfinite(x)
            med.append(dict(programm=prog, formel=f, n=int(ok.sum()), median_qtc_ms=float(np.median(x[ok])), p05=float(np.percentile(x[ok], 5)),
                            p95=float(np.percentile(x[ok], 95))))
            col = spalten.get(prog, {}).get(f)
            if col is not None:
                y = P[prog][col]
                m = ok & np.isfinite(y)
                dd = x[m] - y[m]
                rows.append(dict(programm=prog, formel=f, spalte=col, n=int(m.sum()), median_diff_ms=float(np.median(dd)), max_betrag_ms=float(np.abs(dd).max()),
                                 anteil_innerhalb_1ms=float(np.mean(np.abs(dd) <= 1.0)), anteil_innerhalb_1_5ms=float(np.mean(np.abs(dd) <= 1.5)), anteil_innerhalb_2ms=float(np.mean(np.abs(dd) <= 2.0))))
    # BRAVEHEART und LoopLens: gelieferte Fridericia-Spalte (qtcf nativ)
    for prog in ("BRAVEHEART", "LoopLens"):
        x = BA.qtc_formel("Fridericia", B.D[prog]["qt"], B.D[prog]["rr"])
        y = B.D[prog]["qtc_nativ"]
        m = np.isfinite(x) & np.isfinite(y)
        dd = x[m] - y[m]
        rows.append(dict(programm=prog, formel="Fridericia", spalte="qtcf (nativ)", n=int(m.sum()), median_diff_ms=float(np.median(dd)), max_betrag_ms=float(np.abs(dd).max()),
                         anteil_innerhalb_1ms=float(np.mean(np.abs(dd) <= 1.0)), anteil_innerhalb_1_5ms=float(np.mean(np.abs(dd) <= 1.5)), anteil_innerhalb_2ms=float(np.mean(np.abs(dd) <= 2.0))))
    return pd.DataFrame(rows), pd.DataFrame(med)


def ll_kandidaten(B):
    """LoopLens: gelieferte QT, RR und QTcF gegen Kandidaten aus den Einzelschlaegen (VV0, nativ). Rueckgabe Tabelle (Anteile innerhalb 0,5 und 1 ms)."""
    sc = pd.read_csv(SCHLAEGE, usecols=["datensatz", "aufnahme", "arm", "vorverarbeitung", "schlag_nr", "r_gipfel", "qrs_beginn", "t_ende", "akzeptiert", "repraesentativ"])
    sc = sc[(sc.datensatz == "ptbxl") & (sc.vorverarbeitung == "VV0") & (sc.arm == "nativ") & sc.aufnahme.isin(B.ids)].sort_values(["aufnahme", "schlag_nr"])
    ids = B.ids
    pos = {int(a): i for i, a in enumerate(ids)}
    n = len(ids)
    kand = {k: np.full(n, np.nan) for k in ("qt_median_akzeptiert", "qt_median_alle", "qt_mittel_akzeptiert", "qt_repraesentativ", "rr_median_akzeptiert", "rr_mittel_akzeptiert",
                                             "rr_median_alle", "qtcf_median_schlagweise_akzeptiert", "qtcf_median_schlagweise_alle")}
    nb = np.zeros(n)
    for aid, g in sc.groupby("aufnahme", sort=False):
        i = pos[int(aid)]
        r = g.r_gipfel.to_numpy(float)
        qt = (g.t_ende.to_numpy(float) - g.qrs_beginn.to_numpy(float)) * 2.0
        acc = g.akzeptiert.to_numpy() == 1
        rep = g.repraesentativ.to_numpy() == 1
        rr_all = np.diff(r) * 2.0
        nb[i] = len(g)
        ok = np.isfinite(qt)
        if (acc & ok).any():
            kand["qt_median_akzeptiert"][i] = np.median(qt[acc & ok])
            kand["qt_mittel_akzeptiert"][i] = np.mean(qt[acc & ok])
        if ok.any():
            kand["qt_median_alle"][i] = np.median(qt[ok])
        if (rep & ok).any():
            kand["qt_repraesentativ"][i] = qt[rep & ok][0]
        if len(rr_all):
            kand["rr_median_alle"][i] = np.median(rr_all)
            acc_rr = acc[1:] & acc[:-1]
            if acc_rr.any():
                kand["rr_median_akzeptiert"][i] = np.median(rr_all[acc_rr])
                kand["rr_mittel_akzeptiert"][i] = np.mean(rr_all[acc_rr])
            # QTcF je Schlag mit dem vorangehenden RR
            qtb, rrb = qt[1:], rr_all
            f = BA.qtc_formel("Fridericia", qtb, rrb)
            okb = np.isfinite(f)
            if okb.any():
                kand["qtcf_median_schlagweise_alle"][i] = np.median(f[okb])
            okb2 = okb & acc_rr
            if okb2.any():
                kand["qtcf_median_schlagweise_akzeptiert"][i] = np.median(f[okb2])
    LL = B.D["LoopLens"]
    rows = []

    def ziel(name, x, y):
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        rows.append(dict(geliefert=name, kandidat="", n=int(m.sum()), median_diff_ms=float(np.median(d)), anteil_innerhalb_0_5ms=float(np.mean(np.abs(d) <= 0.5)),
                         anteil_innerhalb_1ms=float(np.mean(np.abs(d) <= 1.0)), anteil_innerhalb_5ms=float(np.mean(np.abs(d) <= 5.0))))
        return rows[-1]

    for k in ("qt_median_akzeptiert", "qt_median_alle", "qt_mittel_akzeptiert", "qt_repraesentativ"):
        ziel("qt_global", kand[k], LL["qt"])["kandidat"] = k
    for k in ("rr_median_akzeptiert", "rr_mittel_akzeptiert", "rr_median_alle"):
        ziel("rr_intervall_ll", kand[k], LL["rr"])["kandidat"] = k
    cand_qtcf = {
        "F(qt_global, rr_intervall_ll)": BA.qtc_formel("Fridericia", LL["qt"], LL["rr"]),
        "F(Median QT akzeptiert, Median RR akzeptiert)": BA.qtc_formel("Fridericia", kand["qt_median_akzeptiert"], kand["rr_median_akzeptiert"]),
        "F(Median QT akzeptiert, rr_intervall_ll)": BA.qtc_formel("Fridericia", kand["qt_median_akzeptiert"], LL["rr"]),
        "F(qt_global, Median RR akzeptiert)": BA.qtc_formel("Fridericia", LL["qt"], kand["rr_median_akzeptiert"]),
        "Median schlagweise (akzeptiert)": kand["qtcf_median_schlagweise_akzeptiert"],
        "Median schlagweise (alle)": kand["qtcf_median_schlagweise_alle"],
        "F(QT repraesentativ, rr_intervall_ll)": BA.qtc_formel("Fridericia", kand["qt_repraesentativ"], LL["rr"]),
    }
    for k, x in cand_qtcf.items():
        ziel("qtcf", x, LL["qtc_nativ"])["kandidat"] = k
    return pd.DataFrame(rows), pd.Series(nb)


def kontrollen(B, Sx, C4):
    out = []
    for prog, f, col, soll in (("Uni-G", "Hodge", "QT_IntHodge_Global", 1.0), ("12SL", "Bazett", "QT_IntBazett_Global", 0.96), ("Uni-G", "Fridericia", "QT_IntFridericia_Global", 1.0)):
        x = BA.qtc_formel(f, B.D[prog]["qt"], B.D[prog]["rr"])
        y = B.arch[prog][col]
        m = np.isfinite(x) & np.isfinite(y)
        a1 = float(np.mean(np.abs(x[m] - y[m]) <= 1.5))
        out.append(dict(kontrolle="Neuberechnung %s %s gegen %s (innerhalb 1,5 ms; 12SL rundet QT und HR auf ganze Zahlen)" % (prog, f, col), soll=">= %.2f (AW-228: 0,97 bzw. 1,0)" % soll, ist="%.4f" % a1, bestanden=a1 >= soll - 0.005))
    out.append(dict(kontrolle="Gemeinsame Menge C4 nicht leer (>= 15.000)", soll=">= 15000", ist=str(int(C4.sum())), bestanden=int(C4.sum()) >= 15000))
    # Nachbau AW-234: Diskordanz geliefert LoopLens-BRAVEHEART bei neutral 450 liegt auf der paarweisen Menge bei 8,5 %; auf C4 nur Plausibilitaet 5 bis 12
    ta, tb = BA.orientiere("LoopLens", "BRAVEHEART")
    v = Sx["zw|%s|%s|alle|geliefert_gesamt|n450" % (ta, tb)][0] / Sx["C4|V|alle"][0] * 100
    out.append(dict(kontrolle="Plausibilitaet D450 geliefert LoopLens-BRAVEHEART auf C4 (AW-234 paarweise 8,5 %)", soll="5 bis 12 %", ist="%.2f" % v, bestanden=5 <= v <= 12))
    return pd.DataFrame(out)


def lauf():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    BA.S.STATE["log"] = str(OUT / "aw239_r5.log")
    me = Path(__file__)
    log("Start aw239_r5 Version %s, Skript-SHA-256 %s, Basis %s" % (VERSION, BA.sha256_datei(me), BASIS_IST))
    # Hashpruefung inkl. Einzelschlagtabelle (steht im Manifest der Datensperre)
    gel = BA.S.gelesene_dateien() + [SCHLAEGE]
    ok = BA.S.hashpruefung(gel)
    log("Hashpruefung: %d Eintraege, alle stimmen: %s" % (len(BA.S.STATE["pruef"]), ok))
    if not ok:
        raise SystemExit("ABBRUCH: Hashpruefung nicht bestanden")
    pruef = list(BA.S.STATE["pruef"])
    pd.DataFrame(pruef).to_csv(OUT / "00_hashpruefung.csv", index=False, encoding="utf-8")
    B = BA.lade_basis(mit_hash=False)
    Q = {t: B.D[t]["qt"] for t in T4}
    R = {t: B.D[t]["rr"] for t in T4}
    deliv = {"12SL": B.arch["12SL"]["QT_IntCorr_Global"], "Uni-G": B.arch["Uni-G"]["QT_IntCorr_Global"], "BRAVEHEART": B.D["BRAVEHEART"]["qtc_nativ"],
             "LoopLens": B.D["LoopLens"]["qtc_nativ"]}
    C4 = np.ones(len(B.ids), bool)
    for t in T4:
        C4 &= np.isfinite(Q[t]) & np.isfinite(R[t]) & (R[t] > 0) & np.isfinite(deliv[t])
    hr = B.arch["12SL"]["HR__Global"]
    strata_hr = {}
    for s, rng in HR_STRATA.items():
        strata_hr[s] = np.ones(len(B.ids), bool) if rng is None else (np.isfinite(hr) & (hr >= rng[0]) & (hr < rng[1]))
    log("C4: %d Personen; HR-Strata: %s" % (C4.sum(), {k: int((v & C4).sum()) for k, v in strata_hr.items()}))
    bk = baue(B, C4, strata_hr, Q, R, deliv)
    ni, Xi, nm_, Xm = bk.matrizen()
    log("Spalten: %d Indikator, %d Moment" % (len(ni), len(nm_)))
    Sx = BA.boot_summen(bk, BA.ziehungen(len(B.ids), SEED))
    kon = kontrollen(B, Sx, C4)
    log("Kontrollen:\n" + kon.to_string())
    kon.to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
    if not kon.bestanden.all():
        log("ABBRUCH: Kontrolle nicht bestanden")
        return 3
    T = tabellen(Sx, B, C4, strata_hr, Q, R, deliv)
    rek, med = rekonstruktion_und_median(B)
    llk, nb = ll_kandidaten(B)
    dirs = [OUT, KOPIE]
    BA.schreibe(T["innen"], "r5_innerhalb_programm.csv", dirs)
    BA.schreibe(T["zwischen"], "r5_zwischen_programmen.csv", dirs)
    BA.schreibe(T["zerlegung"], "r5_zerlegung_shapley.csv", dirs)
    BA.schreibe(rek, "r5_neuberechnung_gegen_geliefert.csv", dirs)
    BA.schreibe(med, "r5_median_qtc_je_formel.csv", dirs)
    BA.schreibe(llk, "r5_looplens_kandidaten.csv", dirs)
    BA.schreibe(kon, "r5_kontrollen.csv", dirs)
    BA.schreibe(pd.DataFrame(pruef), "r5_hashpruefung.csv", dirs)
    info = dict(version=VERSION, skript_sha256=BA.sha256_datei(me), basis_sha256=BASIS_IST, aw234_sha256=BA.A234_HASH, seed=SEED, ziehungen=BA.B_ZIEHUNGEN,
                start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0],
                numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(pruef), n_C4=int(C4.sum()),
                n_hr_strata={k: int((v & C4).sum()) for k, v in strata_hr.items()}, ll_schlaege_aufnahmen_mit_schlaegen=int((nb > 0).sum()))
    for d in dirs:
        (Path(d) / "r5_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.parse_args(argv)
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
