# -*- coding: utf-8 -*-
"""aw234_ablation_auswertung.py (AW-234, Teil 2, Version 1.0): Auswertung der LoopLens-Diagnose-Ablation (NeuroKit2-Fusionsgewicht 0).

Liest den Rohcheckpoint des Ablationslaufs (ll_lauf_ablation.py, Variante ablation, PTB-XL P0 nativ) und den des Kontrolllaufs (Standardgewichte
explizit, 300 Personen aus P0) und vergleicht nur die QRS-Dauer mit dem gesperrten Hauptlauf (Langformat, Hashpruefung wie in Teil 1) und mit
BRAVEHEART, 12SL, Uni-G (und ECGDeli). Rechenbasis (Einheit, Kennzahlen, Ziehungen) ist aw234_auswertung.py.

Auswertung (explorativ, nur QRS-Dauer):
  * Kontrolle: Kontrolllauf gegen Hauptlauf auf den 300 Personen: QRS-Dauer bitgleich (Zaehlung).
  * Ablation gegen Hauptlauf: Anteil unveraenderter Werte, Verteilung der Differenz, Wechsel an 120/130/150 ms.
  * Je Komparator (BRAVEHEART, 12SL, Uni-G, ECGDeli): Bias, D120, D130, D150, Klassenwechsel fuer LoopLens-Hauptlauf und LoopLens-Ablation auf dem
    gemeinsamen Nenner (Personen mit gueltigem Wert in beiden Laeufen und beim Komparator), Differenz Ablation minus Hauptlauf mit Intervall aus denselben
    Ziehungen (Personencluster, 2.000 Ziehungen, fester Seed).
Aufruf: python -I -B aw234_ablation_auswertung.py --modus lauf [--teilweise]
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
HIER = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("aw234_auswertung", str(HIER / "aw234_auswertung.py"))
AW = importlib.util.module_from_spec(_spec)
sys.modules["aw234_auswertung"] = AW
_spec.loader.exec_module(AW)
S = AW.S

VERSION = "1.0"
SEED = 15102342
ABL = Path(str(data_path("pub12.benchmark_pub12_aw234_ablation")))
ROH_ABL = ABL / "roh" / "AW234_ablation.jsonl"
ROH_KONTR = ABL / "roh" / "AW234_kontrolle.jsonl"
QUELLBAUM = ABL / "ABSCHLUSS_ABLATION.json"
OUT = Path(str(data_path("pub12.benchmark_pub12_aw234_ablation_auswertung")))
KOPIE = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw234")))
KOMPARATOREN = ["BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]


def lese_roh(p):
    """Rohcheckpoint -> dict ecg_id -> QRS-Dauer (NaN bei Status ungleich ok, Signal-QC unbrauchbar oder fehlendem Marker). Spaetere Zeile gewinnt."""
    seen = {}
    for line in Path(p).read_text(encoding="utf8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["key"]] = r
    out, stat = {}, {}
    for k, r in seen.items():
        st = r.get("status")
        stat[st] = stat.get(st, 0) + 1
        v = np.nan
        if st == "ok" and r.get("qc_brauchbar", True):
            m = r.get("marker", {}).get("qrs_duration_ms")
            if m is not None and m[0] is not None and np.isfinite(float(m[0])):
                v = float(m[0])
        out[int(r["aufnahme"])] = v
    return out, stat


def ausrichten(d, ids):
    return np.array([d.get(int(i), np.nan) for i in ids], float)


def verteilung(diff):
    """Differenz Ablation minus Hauptlauf: Zaehlklassen."""
    d = diff[np.isfinite(diff)]
    return {"n": len(d), "gleich": int((d == 0).sum()), "abs_le_2": int((np.abs(d) <= 2).sum()), "abs_le_4": int((np.abs(d) <= 4).sum()),
            "abs_gt_10": int((np.abs(d) > 10).sum()), "abs_gt_20": int((np.abs(d) > 20).sum()), "mittel": float(d.mean()), "median": float(np.median(d)),
            "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)), "minimum": float(d.min()), "maximum": float(d.max()),
            "kuerzer": int((d < 0).sum()), "laenger": int((d > 0).sum())}


def lauf(teilweise):
    OUT.mkdir(parents=True, exist_ok=True)
    S.STATE["log"] = str(OUT / "aw234_ablation_auswertung.log")
    t0 = time.time()
    me = Path(__file__)
    log = S.log
    log("Start aw234_ablation_auswertung Version %s, Skript-SHA-256 %s, teilweise=%s" % (VERSION, S.sha256_datei(me), teilweise))
    ok = S.hashpruefung(S.gelesene_dateien())
    pd.DataFrame(S.STATE["pruef"]).to_csv(OUT / "00_hashpruefung.csv", index=False, encoding="utf-8")
    log("Hashpruefung: %d Eintraege, alle stimmen: %s" % (len(S.STATE["pruef"]), ok))
    if not ok:
        log("ABBRUCH: Hashpruefung nicht bestanden")
        return 2
    ids, D = S.lade_alles()
    haupt = D["LoopLens"]["qrs"]
    abl_d, abl_stat = lese_roh(ROH_ABL)
    kon_d, kon_stat = lese_roh(ROH_KONTR)
    abl = ausrichten(abl_d, ids)
    n_abl_roh = len(abl_d)
    S.kontrolle("A1 Ablationslauf vollstaendig (P0)", "18869 Aufgaben", "%d" % n_abl_roh, n_abl_roh == len(ids) or teilweise)
    # Kontrolle: Standardgewichte explizit gegen gesperrten Hauptlauf
    kid = np.array(sorted(kon_d))
    pos = pd.Series(np.arange(len(ids)), index=ids)
    kh = haupt[pos[kid].to_numpy()]
    kk = np.array([kon_d[i] for i in kid])
    gleich = int(np.sum((np.isnan(kh) & np.isnan(kk)) | (kh == kk)))
    S.kontrolle("A2 Kontrolllauf (Standardgewichte explizit) gleich gesperrter Hauptlauf, QRS-Dauer", "alle gleich", "%d von %d" % (gleich, len(kid)), gleich == len(kid))
    # Fehlend-Muster gleich
    mf = np.isfinite(haupt)
    af = np.isfinite(abl)
    if not teilweise:
        S.kontrolle("A3 Verfuegbarkeit Ablation gegen Hauptlauf (Personen mit Wert)", "Zahl ausweisen", "Hauptlauf %d, Ablation %d, nur Hauptlauf %d, nur Ablation %d" % (
            mf.sum(), af.sum(), (mf & ~af).sum(), (~mf & af).sum()), True)
    quell = json.loads(QUELLBAUM.read_text(encoding="utf8")) if QUELLBAUM.exists() else {}
    S.kontrolle("A4 Quellbaum nach dem Lauf unveraendert", "ja", str(quell.get("quellbaum_unveraendert")), bool(quell.get("quellbaum_unveraendert")) or teilweise)
    if not all(k["bestanden"] for k in S.STATE["kontrollen"]):
        pd.DataFrame(S.STATE["kontrollen"]).to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
        log("ABBRUCH: Kontrolle nicht bestanden")
        return 3
    # gemeinsamer Nenner der beiden LoopLens-Laeufe
    beide = mf & af
    w = np.ones(len(ids))
    sch = {"t%d" % t: t for t in AW.QRS_SCHWELLEN}
    einheiten = []
    for c in KOMPARATOREN:
        for arm, x in (("haupt", haupt), ("ablation", abl)):
            einheiten.append(AW.Einheit("%s|%s" % (arm, c), x, D[c]["qrs"], w, beide, sch, klassen=True))
    einheiten.append(AW.Einheit("direkt|ablation_gegen_haupt", abl, haupt, w, beide, sch, klassen=True,
                                 meta=dict(scope="P0", messung="qrs", kandidat="LoopLens Ablation", komparator="LoopLens Hauptlauf", teilmenge="alle", gewichtet=False)))
    bs = AW.boot_lauf(einheiten, S.Ziehungen(len(ids), SEED))
    pk = {e.name: e.punkt() for e in einheiten}
    zl = {e.name: e.zaehlungen() for e in einheiten}
    rows = []
    for c in KOMPARATOREN:
        for arm in ("haupt", "ablation"):
            nm = "%s|%s" % (arm, c)
            r = dict(komparator=c, lauf=("LoopLens Hauptlauf (gesperrt)" if arm == "haupt" else "LoopLens Ablation (NeuroKit2-Gewicht 0)"), n_gemeinsam=zl[nm]["n"])
            for k, nn in (("bias", "bias_ms"), ("sd_diff", "sd_diff_ms"), ("loa_u", "loa_untere_ms"), ("loa_o", "loa_obere_ms"), ("icc21", "icc21"),
                          ("kappa|t120", "kappa_120"), ("klassenwechsel", "klassenwechsel")):
                lo, hi = AW._ki(bs[nm], k)
                f = 100.0 if k == "klassenwechsel" else 1.0
                r[nn], r[nn + "_ki_u"], r[nn + "_ki_o"] = f * pk[nm][k], f * lo, f * hi
            for t in AW.QRS_SCHWELLEN:
                k = "disc|t%d" % t
                lo, hi = AW._ki(bs[nm], k)
                r["D%d_pct" % t], r["D%d_ki_u" % t], r["D%d_ki_o" % t] = 100 * pk[nm][k], 100 * lo, 100 * hi
            rows.append(r)
        # Differenz Ablation minus Hauptlauf aus denselben Ziehungen
        r = dict(komparator=c, lauf="Differenz Ablation minus Hauptlauf", n_gemeinsam=zl["haupt|" + c]["n"])
        for k, nn, f in (("bias", "bias_ms", 1.0), ("klassenwechsel", "klassenwechsel", 100.0), ("disc|t120", "D120_pct", 100.0), ("disc|t130", "D130_pct", 100.0),
                         ("disc|t150", "D150_pct", 100.0)):
            dl = bs["ablation|" + c][k] - bs["haupt|" + c][k]
            lo, hi = S.ki95(dl)
            r[nn] = f * (pk["ablation|" + c][k] - pk["haupt|" + c][k])
            r[nn + "_ki_u"], r[nn + "_ki_o"] = f * lo, f * hi
        rows.append(r)
    AW.OUT, AW.KOPIE = OUT, (OUT if teilweise else KOPIE)
    tab = pd.DataFrame(rows)
    AW.schreibe(tab, "A1_ablation_gegen_komparatoren.csv")
    # direkter Vergleich Ablation gegen Hauptlauf
    e = [x for x in einheiten if x.name == "direkt|ablation_gegen_haupt"][0]
    v = verteilung(abl - haupt)
    v.update(n_haupt=int(mf.sum()), n_ablation=int(af.sum()), n_beide=int(beide.sum()))
    nm = "direkt|ablation_gegen_haupt"
    for t in AW.QRS_SCHWELLEN:
        k = "disc|t%d" % t
        lo, hi = AW._ki(bs[nm], k)
        v["wechsel_an_%d_pct" % t], v["wechsel_an_%d_ki_u" % t], v["wechsel_an_%d_ki_o" % t] = 100 * pk[nm][k], 100 * lo, 100 * hi
        v["n_wechsel_an_%d" % t] = zl[nm][k]
    lo, hi = AW._ki(bs[nm], "bias")
    v["bias_ms_ki_u"], v["bias_ms_ki_o"] = lo, hi
    AW.schreibe(pd.DataFrame([v]), "A2_ablation_gegen_haupt_direkt.csv")
    kreuz, wechsel = AW.tab_kreuz([e], {nm: pk[nm]}, {nm: bs[nm]}, {nm: zl[nm]})
    AW.schreibe(kreuz, "A3_kreuztabelle_ablation_gegen_haupt.csv")
    pd.DataFrame(S.STATE["kontrollen"]).to_csv(OUT / "01_kontrollen.csv", index=False, encoding="utf-8")
    info = dict(version=VERSION, skript_sha256=S.sha256_datei(me), aw234_auswertung_sha256=S.sha256_datei(HIER / "aw234_auswertung.py"), seed=SEED,
                ziehungen=AW.B_ZIEHUNGEN, teilweise=bool(teilweise), roh_ablation_sha256=S.sha256_datei(ROH_ABL), roh_kontrolle_sha256=S.sha256_datei(ROH_KONTR),
                status_ablation=abl_stat, status_kontrolle=kon_stat, quellbaum=quell, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                ende=time.strftime("%Y-%m-%d %H:%M:%S"), gewichte_ablation={"neurokit": 0.0, "multilead": 0.35, "vcg": 0.25},
                gewichte_hauptlauf={"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25})
    (OUT / "LAUF.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if not teilweise:
        KOPIE.mkdir(parents=True, exist_ok=True)
        (KOPIE / "LAUF_ABLATION.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        pd.DataFrame(S.STATE["kontrollen"]).to_csv(KOPIE / "01_kontrollen_ablation.csv", index=False, encoding="utf-8")
    log("Ende nach %.0f s" % (time.time() - t0))
    return 0


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", choices=["lauf"], required=True)
    ap.add_argument("--teilweise", action="store_true", help="Probelauf auf dem unvollstaendigen Checkpoint; schreibt nur nach E:, nicht in die Kopie")
    a = ap.parse_args(argv)
    return lauf(a.teilweise)


if __name__ == "__main__":
    sys.exit(haupt())
