# -*- coding: utf-8 -*-
"""aw239_r1.py (AW-239, P12-R1, Version 1.0): Schwellenabstand, Strata, Entsprechungsschwellen, Standardmasse, Verteilungen (PUB-12, explorativ, post hoc).

Population P0 (18.869 Personen, eine ECG je Person), native Ausgaben, Paare der fuenf Programme (10 Paare), gemeinsamer Nenner je Paar.
Differenz = Kandidat minus Komparator (LoopLens ist Kandidat, sonst das in ORDNUNG spaetere Programm), wie in AW-228/AW-234.
Alle Intervalle: Perzentil-Bootstrap ueber Personen (2.000 Ziehungen, fester Seed, gemeinsame Ziehungen fuer alle Kennzahlen).

(a) Mittelwert beider Programmwerte (m) in Klassen (<100, 100-109, ..., 150-159, 160-169, ab 170, ab 160) und Abstand |m - Schwelle| (0-5, 5-10, 10-20, ab 20 ms):
    Diskordanz je Klasse, gesamt / CLBBB / ohne CLBBB; QRS-angepasster Vergleich (direkte Standardisierung der Gruppe ohne CLBBB auf die m-Verteilung
    der CLBBB-Gruppe im Bereich 130 bis 169 ms, 10-ms-Klassen) fuer D150.
(b) Proportionaler Bias: Steigung der Differenz auf den Mittelwert (OLS), Heteroskedastizitaet: Steigung des Betrags der Differenz auf den Mittelwert.
(c) Entsprechungsschwellen: Schwelle c im Programm X, bei der der Anteil X > c dem Anteil 12SL >= t (t = 120, 130, 150) auf derselben Personenmenge entspricht.
(d) Strata: CLBBB, CRBBB, IVCD, PACE, Vorhofflimmern/-flattern (AFIB oder AFLT), Alter (<60, 60-79, ab 80), Geschlecht (0 = maennlich): D120, D130, D150, Klassenwechsel.
(e) Masse fuer alle Paare: ICC(2,1), Lin-CCC, Kappa, PABAK, PPA, NPA bei 120/130/150, parametrische und empirische Grenzen der Uebereinstimmung.
(f) Aggregate fuer Abbildungen: Histogramme (5 ms) der QRS-Dauer je Programm (gesamt, CLBBB), Perzentile 5/25/50/75/95 fuer QRS und QT, Bland-Altman-Klassen.

Aufruf: <local path> -I -B aw239_r1.py --modus lauf
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
SEED = 15102391
BASIS_PFAD = Path(str(data_path("pub12.werkzeuge_pub12_aw239_aw239_basis_py")))
BASIS_HASH = "50fcf31d788931c671bd0021e112d2f9272795daecd2c24bbd215af0b0d4c7bd"
OUT = Path(str(data_path("pub12.benchmark_pub12_aw239_r1")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw239")))


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

MBINS = {"lt100": (-np.inf, 100), "100-109": (100, 110), "110-119": (110, 120), "120-129": (120, 130), "130-139": (130, 140), "140-149": (140, 150),
         "150-159": (150, 160), "160-169": (160, 170), "ge170": (170, np.inf), "ge160": (160, np.inf)}
DBINS = {"0-5": (0, 5), "5-10": (5, 10), "10-20": (10, 20), "ge20": (20, np.inf)}
ADJ_BINS = ["130-139", "140-149", "150-159", "160-169"]
BIN_SUBSETS = ("alle", "CLBBB", "ohne_CLBBB")
SCHW = BA.QRS_SCHWELLEN


def paare():
    return [BA.orientiere(BA.WERKZEUGE[i], BA.WERKZEUGE[j]) for i in range(len(BA.WERKZEUGE)) for j in range(i + 1, len(BA.WERKZEUGE))]


def bin_spalten(bk, pref, a, b, mask):
    """Klassen des Mittelwerts und des Abstands zur Schwelle: Nenner- und Zaehlerspalten (Zaehler = Diskordanz)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.asarray(mask, bool)
    a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    m = (a0 + b0) / 2.0
    disc = {t: ((a0 >= t) ^ (b0 >= t)) for t in SCHW}
    for k, (lo, hi) in MBINS.items():
        inb = ok & (m >= lo) & (m < hi)
        bk.indikator("%s|mbin|%s|den" % (pref, k), inb)
        for t in SCHW:
            bk.indikator("%s|mbin|%s|num%d" % (pref, k, t), inb & disc[t])
    for t in SCHW:
        dist = np.abs(m - t)
        for k, (lo, hi) in DBINS.items():
            inb = ok & (dist >= lo) & (dist < hi)
            bk.indikator("%s|dist%d|%s|den" % (pref, t, k), inb)
            bk.indikator("%s|dist%d|%s|num" % (pref, t, k), inb & disc[t])


def baue(B, strata, qrs):
    K = len(B.ids)
    bk = BA.Baukasten(K)
    for ta, tb in paare():
        for sub, mask in strata.items():
            pref = "p|%s|%s|%s" % (ta, tb, sub)
            BA.spalten_paar(bk, pref, qrs[ta], qrs[tb], mask)
            if sub in BIN_SUBSETS:
                bin_spalten(bk, pref, qrs[ta], qrs[tb], mask)
    return bk


def quot(Sx, num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(Sx[den] > 0, Sx[num] / Sx[den], np.nan)


def zeile_ci(v, faktor=1.0):
    p, lo, hi = BA.ci(np.asarray(v, float) * faktor)
    return p, lo, hi


def tabellen_aus_summen(Sx, strata):
    T = {k: [] for k in ("abstand", "mklassen", "standard", "bias", "strata", "masse")}
    for ta, tb in paare():
        for sub in strata:
            pref = "p|%s|%s|%s" % (ta, tb, sub)
            kopf = dict(kandidat=ta, komparator=tb, teilmenge=sub)
            n = Sx[pref + "|V"][0]
            sm = BA.stats_moment(Sx, pref)
            # (d) Strata
            r = dict(kopf, n=int(n))
            for t in SCHW:
                s2 = BA.stats_2x2(Sx, pref, t)
                p, lo, hi = zeile_ci(s2["disc"], 100)
                r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t], r["n_disc_%d" % t] = p, lo, hi, int(s2["n_disc"][0])
            p, lo, hi = zeile_ci(Sx[pref + "|wechsel"] / Sx[pref + "|V"], 100)
            r["klassenwechsel_pct"], r["klassenwechsel_ki_u"], r["klassenwechsel_ki_o"] = p, lo, hi
            T["strata"].append(r)
            # (e) Masse
            r = dict(kopf, n=int(n))
            for k, nm in (("bias", "bias_ms"), ("sd", "sd_diff_ms"), ("loa_u", "loa_param_u_ms"), ("loa_o", "loa_param_o_ms"), ("icc", "icc21"), ("ccc", "lin_ccc")):
                r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = zeile_ci(sm[k])
            for t in SCHW:
                s2 = BA.stats_2x2(Sx, pref, t)
                for k, nm in (("kappa", "kappa"), ("pabak", "pabak"), ("ppa", "ppa"), ("npa", "npa")):
                    r["%s_%d" % (nm, t)], r["%s_%d_ki_u" % (nm, t)], r["%s_%d_ki_o" % (nm, t)] = zeile_ci(s2[k])
            T["masse"].append(r)
            # (b) Bias
            r = dict(kopf, n=int(n))
            for k, nm in (("bias", "bias_ms"), ("slope_d_on_m", "steigung_diff_auf_mittel"), ("slope_absd_on_m", "steigung_betrag_diff_auf_mittel"),
                          ("mean_absd", "mittlerer_betrag_diff_ms")):
                r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = zeile_ci(sm[k])
            T["bias"].append(r)
            if sub not in BIN_SUBSETS:
                continue
            # (a) Klassen
            V = Sx[pref + "|V"]
            for k in MBINS:
                den = pref + "|mbin|%s|den" % k
                r = dict(kopf, klasse_m=k, n=int(Sx[den][0]))
                p, lo, hi = zeile_ci(Sx[den] / V, 100)
                r["anteil_personen_pct"], r["anteil_ki_u"], r["anteil_ki_o"] = p, lo, hi
                for t in SCHW:
                    num = pref + "|mbin|%s|num%d" % (k, t)
                    p, lo, hi = zeile_ci(quot(Sx, num, den), 100)
                    r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t], r["n_disc_%d" % t] = p, lo, hi, int(Sx[num][0])
                T["mklassen"].append(r)
            for t in SCHW:
                for k in DBINS:
                    den = pref + "|dist%d|%s|den" % (t, k)
                    num = pref + "|dist%d|%s|num" % (t, k)
                    r = dict(kopf, schwelle=t, abstand_ms=k, n=int(Sx[den][0]), n_disc=int(Sx[num][0]))
                    p, lo, hi = zeile_ci(Sx[den] / V, 100)
                    r["anteil_personen_pct"], r["anteil_ki_u"], r["anteil_ki_o"] = p, lo, hi
                    p, lo, hi = zeile_ci(quot(Sx, num, den), 100)
                    r["D_pct"], r["D_ki_u"], r["D_ki_o"] = p, lo, hi
                    T["abstand"].append(r)
        # direkte Standardisierung D150, Bereich 130-169
        pc, pn = "p|%s|%s|CLBBB" % (ta, tb), "p|%s|%s|ohne_CLBBB" % (ta, tb)
        denC = sum(Sx[pc + "|mbin|%s|den" % k] for k in ADJ_BINS)
        numC = sum(Sx[pc + "|mbin|%s|num150" % k] for k in ADJ_BINS)
        denN = sum(Sx[pn + "|mbin|%s|den" % k] for k in ADJ_BINS)
        numN = sum(Sx[pn + "|mbin|%s|num150" % k] for k in ADJ_BINS)
        with np.errstate(invalid="ignore", divide="ignore"):
            adj = sum(Sx[pc + "|mbin|%s|den" % k] / denC * quot(Sx, pn + "|mbin|%s|num150" % k, pn + "|mbin|%s|den" % k) for k in ADJ_BINS)
            crudeC, crudeN = numC / denC, numN / denN
            nah = lambda pf: (Sx[pf + "|dist150|0-5|den"] + Sx[pf + "|dist150|5-10|den"]) / Sx[pf + "|V"]  # noqa: E731
        r = dict(kandidat=ta, komparator=tb, bereich_m="130-169", n_CLBBB=int(denC[0]), n_ohne_CLBBB=int(denN[0]))
        for nm, v, f in (("D150_CLBBB_pct", crudeC, 100), ("D150_ohne_CLBBB_roh_pct", crudeN, 100), ("D150_ohne_CLBBB_standardisiert_pct", adj, 100),
                         ("diff_CLBBB_minus_ohne_roh_pp", crudeC - crudeN, 100), ("diff_CLBBB_minus_ohne_standardisiert_pp", crudeC - adj, 100),
                         ("anteil_innerhalb10ms_von_150_CLBBB_pct", nah(pc), 100), ("anteil_innerhalb10ms_von_150_ohne_CLBBB_pct", nah(pn), 100)):
            r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = zeile_ci(v, f)
        T["standard"].append(r)
    return {k: pd.DataFrame(v) for k, v in T.items()}


def empirische_loa(B, qrs, strata, zieh):
    qs = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
    rows = []
    for ta, tb in paare():
        d = np.asarray(qrs[ta], float) - np.asarray(qrs[tb], float)
        for sub in BIN_SUBSETS:
            valid = strata[sub]
            Q = BA.quantile_boot(d, zieh, qs, valid=valid)
            r = dict(kandidat=ta, komparator=tb, teilmenge=sub, n=int((np.isfinite(d) & valid).sum()))
            for j, q in enumerate(qs):
                r["q%g" % (100 * q)], r["q%g_ki_u" % (100 * q)], r["q%g_ki_o" % (100 * q)] = BA.ci(Q[:, j])
            r["laenge_loa_empirisch_ms"] = r["q97.5"] - r["q2.5"]
            rows.append(r)
    return pd.DataFrame(rows)


def entsprechungsschwellen(B, qrs, strata, zieh):
    rows = []
    ref = np.asarray(qrs["12SL"], float)
    for x in ("LoopLens", "BRAVEHEART", "Uni-G", "ECGDeli"):
        xv = np.asarray(qrs[x], float)
        for sub in ("alle", "CLBBB"):
            ok = np.isfinite(xv) & np.isfinite(ref) & strata[sub]
            idx = np.flatnonzero(ok)
            o = np.argsort(xv[idx], kind="stable")
            idx = idx[o]
            xs, rs = xv[idx], ref[idx]
            res = {t: dict(c=[], achieved=[], p=[]) for t in SCHW}
            blocks = [np.ones((1, len(idx)))] + [zieh.chunk_c(c)[:, idx] for c in range(zieh.n_chunks)]
            for W in blocks:
                cs = np.cumsum(W, axis=1)
                N = cs[:, -1]
                for t in SCHW:
                    p = (W * (rs >= t)).sum(axis=1) / N
                    j = (cs >= ((1 - p) * N)[:, None] - 1e-9).argmax(axis=1)
                    res[t]["c"].append(xs[j])
                    res[t]["achieved"].append(1 - cs[np.arange(len(j)), j] / N)
                    res[t]["p"].append(p)
            for t in SCHW:
                c = np.concatenate(res[t]["c"])
                ach = np.concatenate(res[t]["achieved"])
                p = np.concatenate(res[t]["p"])
                r = dict(programm=x, referenz="12SL", teilmenge=sub, referenzschwelle_ms=t, n=int(len(idx)))
                for nm, v, f in (("anteil_ref_ab_schwelle_pct", p, 100), ("aequivalente_schwelle_ms", c, 1), ("erreichter_anteil_ueber_c_pct", ach, 100)):
                    r[nm], r[nm + "_ki_u"], r[nm + "_ki_o"] = BA.ci(v * f)
                rows.append(r)
    return pd.DataFrame(rows)


def verteilungen(B, qrs, qt, strata):
    hist, perz, ba = [], [], []
    kanten = np.arange(40, 305, 5)
    for t in BA.WERKZEUGE:
        for sub in ("alle", "CLBBB"):
            x = np.asarray(qrs[t], float)[strata[sub]]
            x = x[np.isfinite(x)]
            h, _ = np.histogram(np.clip(x, kanten[0], kanten[-1] - 1e-9), bins=kanten)
            for lo, c in zip(kanten[:-1], h):
                hist.append(dict(programm=t, teilmenge=sub, klasse_von_ms=int(lo), klasse_bis_ms=int(lo + 5), anzahl=int(c), n=len(x)))
            for gr, arr in (("QRS", qrs[t]), ("QT", qt[t])):
                y = np.asarray(arr, float)[strata[sub]]
                y = y[np.isfinite(y)]
                pc = np.percentile(y, [5, 25, 50, 75, 95]) if len(y) else [np.nan] * 5
                perz.append(dict(programm=t, teilmenge=sub, groesse=gr, n=len(y), p05=pc[0], p25=pc[1], p50=pc[2], p75=pc[3], p95=pc[4]))
    for ta, tb in paare():
        a, b = np.asarray(qrs[ta], float), np.asarray(qrs[tb], float)
        for sub in ("alle", "CLBBB"):
            ok = np.isfinite(a) & np.isfinite(b) & strata[sub]
            m, d = ((a + b) / 2)[ok], (a - b)[ok]
            kl = np.floor(m / 5).astype(int) * 5
            for k in np.unique(kl):
                dd = d[kl == k]
                ba.append(dict(kandidat=ta, komparator=tb, teilmenge=sub, mittel_von_ms=int(k), mittel_bis_ms=int(k + 5), n=len(dd), mittlere_diff_ms=dd.mean(),
                               sd_diff_ms=dd.std(ddof=1) if len(dd) > 1 else np.nan, p2_5=np.percentile(dd, 2.5), p97_5=np.percentile(dd, 97.5)))
    return pd.DataFrame(hist), pd.DataFrame(perz), pd.DataFrame(ba)


def kontrollen(B, Sx, qrs, strata):
    """Nachbau bekannter Werte aus AW-234 (gleiche Daten): n, D120, ICC, CLBBB-D150, CLBBB-n."""
    erw = [
        ("BRAVEHEART vs 12SL alle: n, D120, ICC", ("BRAVEHEART", "12SL", "alle"), 18663, 2.1, 0.901),
        ("LoopLens vs BRAVEHEART alle: n, D120, ICC", ("LoopLens", "BRAVEHEART", "alle"), 18580, 20.0, 0.551),
        ("Uni-G vs 12SL alle: n, D120, ICC", ("Uni-G", "12SL", "alle"), 18867, 2.5, 0.893),
    ]
    out = []
    for nm, (ta, tb, sub), n_s, d_s, icc_s in erw:
        pref = "p|%s|%s|%s" % (ta, tb, sub)
        n = int(Sx[pref + "|V"][0])
        d120 = 100 * BA.stats_2x2(Sx, pref, 120)["disc"][0]
        icc = BA.stats_moment(Sx, pref)["icc"][0]
        ok = n == n_s and abs(d120 - d_s) < 0.06 and abs(icc - icc_s) < 0.0006
        out.append(dict(kontrolle=nm, soll="%d, %.1f, %.3f" % (n_s, d_s, icc_s), ist="%d, %.2f, %.4f" % (n, d120, icc), bestanden=bool(ok)))
    for (ta, tb), d_s in ((("BRAVEHEART", "12SL"), 14.1), (("Uni-G", "BRAVEHEART"), 15.1), (("Uni-G", "12SL"), 16.3)):
        pref = "p|%s|%s|CLBBB" % (ta, tb)
        d150 = 100 * BA.stats_2x2(Sx, pref, 150)["disc"][0]
        out.append(dict(kontrolle="%s vs %s CLBBB: D150" % (ta, tb), soll="%.1f" % d_s, ist="%.2f" % d150, bestanden=bool(abs(d150 - d_s) < 0.06)))
    out.append(dict(kontrolle="CLBBB-Menge", soll="418", ist=str(int(strata["CLBBB"].sum())), bestanden=int(strata["CLBBB"].sum()) == 418))
    return pd.DataFrame(out)


def lauf():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    BA.S.STATE["log"] = str(OUT / "aw239_r1.log")
    me = Path(__file__)
    log("Start aw239_r1 Version %s, Skript-SHA-256 %s, Basis %s, aw234 %s" % (VERSION, BA.sha256_datei(me), BASIS_IST, BA.A234_HASH))
    B = BA.lade_basis(mit_hash=True)
    pd.DataFrame(B.pruef).to_csv(OUT / "00_hashpruefung.csv", index=False, encoding="utf-8")
    qrs = {t: B.D[t]["qrs"] for t in BA.WERKZEUGE}
    qt = {t: B.D[t]["qt"] for t in BA.WERKZEUGE}
    strata = BA.strata_masken(B)
    log("Strata: " + ", ".join("%s %d" % (k, v.sum()) for k, v in strata.items()))
    bk = baue(B, strata, qrs)
    ni, Xi, nm_, Xm = bk.matrizen()
    log("Spalten: %d Indikator, %d Moment; K = %d" % (len(ni), len(nm_), len(B.ids)))
    zieh = BA.ziehungen(len(B.ids), SEED)
    Sx = BA.boot_summen(bk, zieh)
    kon = kontrollen(B, Sx, qrs, strata)
    log("Kontrollen:\n" + kon.to_string())
    kon.to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
    if not kon.bestanden.all():
        log("ABBRUCH: Kontrolle nicht bestanden")
        return 3
    T = tabellen_aus_summen(Sx, strata)
    dirs = [OUT, KOPIE]
    BA.schreibe(T["abstand"], "r1_abstand_zur_schwelle.csv", dirs)
    BA.schreibe(T["mklassen"], "r1_mittelwertklassen.csv", dirs)
    BA.schreibe(T["standard"], "r1_clbbb_standardisierung_d150.csv", dirs)
    BA.schreibe(T["bias"], "r1_proportionaler_bias.csv", dirs)
    BA.schreibe(T["strata"], "r1_strata.csv", dirs)
    BA.schreibe(T["masse"], "r1_masse_alle_paare.csv", dirs)
    BA.schreibe(empirische_loa(B, qrs, strata, zieh), "r1_empirische_grenzen.csv", dirs)
    BA.schreibe(entsprechungsschwellen(B, qrs, strata, zieh), "r1_entsprechungsschwellen.csv", dirs)
    h, p, ba = verteilungen(B, qrs, qt, strata)
    BA.schreibe(h, "r1_histogramm_qrs_5ms.csv", dirs)
    BA.schreibe(p, "r1_perzentile.csv", dirs)
    BA.schreibe(ba, "r1_blandaltman_klassen.csv", dirs)
    BA.schreibe(kon, "r1_kontrollen.csv", dirs)
    BA.schreibe(pd.DataFrame(B.pruef), "r1_hashpruefung.csv", dirs)
    info = dict(version=VERSION, skript_sha256=BA.sha256_datei(me), basis_sha256=BASIS_IST, aw234_sha256=BA.A234_HASH, seed=SEED, ziehungen=BA.B_ZIEHUNGEN,
                chunk=BA.CHUNK, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), ende=time.strftime("%Y-%m-%d %H:%M:%S"),
                python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, hashpruefung_eintraege=len(B.pruef),
                strata_n={k: int(v.sum()) for k, v in strata.items()})
    for d in dirs:
        (Path(d) / "r1_LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.parse_args(argv)
    return lauf()


if __name__ == "__main__":
    sys.exit(haupt())
