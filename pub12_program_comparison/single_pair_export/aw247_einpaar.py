# -*- coding: utf-8 -*-
"""aw247_einpaar.py (AW-247, Version 1.0): Wiederholung gegen Programmwechsel auf der gemeinsamen Fuenf-Programm-Menge mit genau einem Paar je Person (PUB-12 v8, explorativ, post hoc).

Auftrag K (16.10.2026, Astra-Zweitmeinung v7, Muss 1).  Population: Paare derselben Person aus der gebundenen Paarliste von AW-223 (Primaerpopulation: Abstand bis 24 h, ohne PT-Paare),
davon die gemeinsame Menge, in der alle fuenf Programme an beiden Aufnahmen einen gueltigen Wert liefern (QRS: 296 Paare; QTc einheitlich Fridericia: 280 Paare).
REGEL (vor dem Rechnen in die Ergebnisdatei einpaar_regel_v8.json geschrieben, mit Hash dieses Skripts): je Person wird aus den Paaren der gemeinsamen Menge genau ein Paar gewaehlt, und zwar
das zeitlich erste: kleinstes Aufnahmedatum der ersten Aufnahme (recording_date der PTB-XL-Metadaten), bei Gleichstand kleinere ecg_id_1, dann kleinere ecg_id_2.  Die Wahl haengt von keinem Messwert ab.
Berechnet (QRS 120, 130, 150 ms; QTc einheitlich 450, 470, 500 ms): Wiederholungsdiskordanz je Programm (Anteil der Paare), Programmdiskordanz je Paar von Programmen (Anteil der Paarpositionen,
das Mittel der beiden Aufnahme-Indikatoren je Paar), und die Differenz Wiederholung minus Programmwechsel, alles auf demselben Nenner (denselben Paaren) mit gemeinsamen
personenweisen Bootstrap-Ziehungen (2.000, fester Seed).  Zusaetzlich Zaehlungen zur Ueberlappung der tatsaechlichen Hauptpaare (296 und 280): Personen, Paare, Paarpositionen, eindeutige Aufnahmen.
Gesperrte Benchmark-Tabellen nur lesen (Hashpruefung der Datensperre vorher).
Aufruf: python -I -B aw247_einpaar.py --modus lauf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(data_path("pub12.werkzeuge_pub12_aw245")))

import aw245_basis as AB  # noqa: E402
import aw245_stat as T  # noqa: E402

VERSION = "1.0"
SEED = 16102471
B = 2000
OUT_PUB = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw247")))
OUT_ROH = Path(str(data_path("pub12.benchmark_pub12_aw247")))
WZ = AB.WERKZEUGE
MASSE = {"qrs": T.QRS_SCHW, "qtc_einh": T.QTC_SCHW}
KONTROLLEN = []
REGEL_TEXT = ("Je Person genau ein Paar aus den Paaren der gemeinsamen Fuenf-Programm-Menge (Primaerpopulation: Abstand bis 24 h, ohne Paare aus der LoopLens-Pilotmenge): "
              "das zeitlich erste Paar, d. h. kleinstes recording_date der ersten Aufnahme, bei Gleichstand kleinere ecg_id_1, dann kleinere ecg_id_2. Keine Abhaengigkeit von Messwerten. "
              "Gerechnet: Wiederholungsdiskordanz je Programm, Programmdiskordanz je Paar, Differenz, QRS 120/130/150 ms und QTc einheitlich 450/470/500 ms, 2.000 Ziehungen personenweise, gemeinsame Ziehungen.")


def kontrolle(name, soll, ist, ok):
    KONTROLLEN.append(dict(kontrolle=name, soll=str(soll), ist=str(ist), bestanden=bool(ok)))
    AB.log("KONTROLLE %s: soll %s ist %s -> %s" % (name, soll, ist, "ok" if ok else "ABWEICHUNG"))


def erstes_paar_je_person(pid, t1, id1, id2):
    """Index (in den Eingabefeldern) des zeitlich ersten Paares je Person: Aufnahmezeit der ersten Aufnahme, dann ecg_id_1, dann ecg_id_2."""
    df = pd.DataFrame(dict(pid=np.asarray(pid), t1=pd.to_datetime(pd.Series(t1)).to_numpy(), id1=np.asarray(id1), id2=np.asarray(id2), i=np.arange(len(pid))))
    df = df.sort_values(["pid", "t1", "id1", "id2"], kind="mergesort")
    return df.groupby("pid", sort=True).head(1).sort_values("i").i.to_numpy()


def selbsttest():
    pid = np.array([1, 1, 1, 2, 2, 3])
    t1 = np.array(["2000-01-02", "2000-01-01", "2000-01-03", "2001-05-05", "2001-05-05", "1999-01-01"], dtype="datetime64[ns]")
    id1 = np.array([10, 11, 12, 20, 19, 30])
    id2 = np.array([11, 12, 13, 21, 22, 31])
    sel = erstes_paar_je_person(pid, t1, id1, id2)
    assert list(sel) == [1, 4, 5], sel          # Person 1: Datum 2000-01-01 (Index 1); Person 2: Gleichstand Datum, kleinere ecg_id_1 (19, Index 4); Person 3 (Index 5)
    # Ueberlappungszaehlung
    pairs = pd.DataFrame(dict(pid=[1, 1, 2], a=[1, 2, 5], b=[2, 3, 6]))
    ids = pd.concat([pairs.a, pairs.b])
    vc = ids.value_counts()
    assert int((vc > 1).sum()) == 1 and ids.nunique() == 5
    # Differenz mit gemeinsamen Ziehungen: Person-Cluster = Paar bei genau einem Paar je Person
    a = np.array([100.0, 130.0, 160.0, 119.0]); b = np.array([121.0, 129.0, 150.0, 118.0])
    W = T.person_gewichte(np.array([1, 2, 3, 4]), 50, 1)
    kz = T.paar_kennzahlen(a, b, W, (120, 130, 150))
    assert abs(float(kz["D120"][0]) - 0.25) < 1e-12 and abs(float(kz["D130"][0]) - 0.25) < 1e-12


def lauf(ohne_paritaet=False):
    t0 = time.time()
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    OUT_ROH.mkdir(parents=True, exist_ok=True)
    selbsttest()
    # 1. REGEL VOR DEM RECHNEN schreiben
    skript = Path(__file__).resolve()
    regel = dict(version=VERSION, regel=REGEL_TEXT, skript=skript.name, skript_sha256=hashlib.sha256(skript.read_bytes()).hexdigest(), seed=SEED, ziehungen=B,
                 geschrieben=time.strftime("%Y-%m-%d %H:%M:%S"), hinweis="Datei vor dem Laden jeglicher Messwerte geschrieben")
    (OUT_PUB / "einpaar_regel_v8.json").write_text(json.dumps(regel, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT_ROH / "einpaar_regel_v8.json").write_text(json.dumps(regel, indent=1, ensure_ascii=False), encoding="utf-8")
    AB.S.STATE["log"] = str(OUT_ROH / "einpaar_lauf.log")
    AB.log("aw247_einpaar Version %s, Regel geschrieben, Skript-Hash %s" % (VERSION, regel["skript_sha256"]))
    pruef = AB.pruefe_sperre()
    pl_hash = AB.pruefe_paarliste()
    pairs = pd.read_csv(AB.PAARLISTE)
    pairs["pt_paar"] = pairs["pt_paar"].astype(str).str.lower().isin(["true", "1"])
    prim = ((pairs.abstand_h <= 24) & ~pairs.pt_paar).to_numpy()
    kontrolle("Primaerpopulation Paare", 320, int(prim.sum()), int(prim.sum()) == 320)
    meta = AB.lade_meta()
    ids = np.array(sorted(set(pairs.ecg_id_1.astype(int)) | set(pairs.ecg_id_2.astype(int))), dtype=np.int64)
    pos = {int(e): i for i, e in enumerate(ids)}
    Dv = AB.lade_werte(ids)
    i1 = np.array([pos[int(e)] for e in pairs.ecg_id_1])
    i2 = np.array([pos[int(e)] for e in pairs.ecg_id_2])
    pid_all = pairs.patient_id.astype(int).to_numpy()
    idx = np.flatnonzero(prim)
    rows_re, rows_zw, rows_df, rows_ue, rows_ft = [], [], [], [], []
    erwartet_gem = {"qrs": 296, "qtc_einh": 280}
    for mi, (m, schw) in enumerate(MASSE.items()):
        x1 = {t: Dv[t][m][i1[idx]] for t in WZ}
        x2 = {t: Dv[t][m][i2[idx]] for t in WZ}
        gemeinsam = np.all([np.isfinite(x1[t]) & np.isfinite(x2[t]) for t in WZ], axis=0)
        kontrolle("%s gemeinsame Hauptpaare" % m, erwartet_gem[m], int(gemeinsam.sum()), int(gemeinsam.sum()) == erwartet_gem[m])
        gi = idx[gemeinsam]                                   # Indizes der Hauptpaare in der Paarliste
        # Ueberlappung der tatsaechlichen Hauptpaare
        for name, sub in (("Hauptpaare gemeinsame Menge", gi), ):
            d = pairs.iloc[sub]
            per = d.groupby("patient_id").size()
            rec = pd.concat([d.ecg_id_1, d.ecg_id_2]).value_counts()
            mehrfach_paar = int(d.apply(lambda r: rec[r.ecg_id_1] > 1 or rec[r.ecg_id_2] > 1, axis=1).sum())
            rows_ue.append(dict(mass=m, menge=name, personen=int(per.size), paare=int(len(d)), paarpositionen=int(2 * len(d)), eindeutige_aufnahmen=int(rec.size), aufnahmen_in_zwei_paaren=int((rec > 1).sum()),
                                paare_mit_geteilter_aufnahme=mehrfach_paar, personen_mit_1_paar=int((per == 1).sum()), personen_mit_2_paaren=int((per == 2).sum()), personen_mit_3_oder_mehr_paaren=int((per >= 3).sum()),
                                max_paare_je_person=int(per.max()), effektive_personen_kish=float(per.sum() ** 2 / (per ** 2).sum())))
        # Regel: zeitlich erstes Paar je Person
        t1 = meta.loc[pairs.ecg_id_1.iloc[gi].astype(int), "recording_date"].to_numpy()
        sel_local = erstes_paar_je_person(pid_all[gi], t1, pairs.ecg_id_1.iloc[gi].to_numpy(), pairs.ecg_id_2.iloc[gi].to_numpy())
        gs = gi[sel_local]                                    # gewaehlte Paare (Indizes in der Paarliste)
        d = pairs.iloc[gs]
        rec = pd.concat([d.ecg_id_1, d.ecg_id_2])
        rows_ue.append(dict(mass=m, menge="ein Paar je Person (zeitlich erstes)", personen=int(d.patient_id.nunique()), paare=int(len(d)), paarpositionen=int(2 * len(d)), eindeutige_aufnahmen=int(rec.nunique()),
                            aufnahmen_in_zwei_paaren=0, paare_mit_geteilter_aufnahme=0, personen_mit_1_paar=int(len(d)), personen_mit_2_paaren=0, personen_mit_3_oder_mehr_paaren=0, max_paare_je_person=1, effektive_personen_kish=float(len(d))))
        kontrolle("%s ein Paar je Person: Paare gleich Personen" % m, "gleich", "%d / %d" % (len(d), d.patient_id.nunique()), len(d) == d.patient_id.nunique() == len(sel_local))
        kontrolle("%s ein Paar je Person: eindeutige Aufnahmen gleich Paarpositionen" % m, 2 * len(d), int(rec.nunique()), int(rec.nunique()) == 2 * len(d))
        # Auswahl in den Messfeldern
        pos_sel = np.searchsorted(gi, gs)
        y1 = {t: x1[t][gemeinsam][sel_local] for t in WZ}
        y2 = {t: x2[t][gemeinsam][sel_local] for t in WZ}
        pid_s = pid_all[gs]
        W = T.person_gewichte(pid_s, B, SEED + mi)
        # Kontrolle gegen die bekannte Hauptanalyse auf allen Hauptpaaren (nicht gewaehlt): Wiederholung 12SL D120
        if m == "qrs":
            Wk = T.person_gewichte(pid_all[gi], 50, SEED)
            kk = T.paar_kennzahlen(x1["12SL"][gemeinsam], x2["12SL"][gemeinsam], Wk, schw)
            kontrolle("Kontrolle: Wiederholung 12SL D120 auf allen 296 Hauptpaaren gleich AW-245 (1,69 %)", "0.0169", "%.4f" % float(kk["D120"][0]), abs(float(kk["D120"][0]) - 0.0169) < 5e-4)
        n_pers = int(pd.Series(pid_s).nunique())
        # Wiederholung je Programm
        rep = {}
        for t in WZ:
            kz = T.paar_kennzahlen(y1[t], y2[t], W, schw)
            rep[t] = kz
            for k in ["bias", "sd", "rc", "icc"] + ["D%d" % s for s in schw] + ["wechsel"]:
                est, lo, hi = T.ci(kz[k])
                rows_re.append(dict(mass=m, programm=t, paare=int(kz["n"][0]), personen=n_pers, kennzahl=k, schaetzer=est, ki_unten=lo, ki_oben=hi))
        # Programmwechsel je Paar von Programmen; Differenz gegen die Wiederholung beider Programme
        W2 = np.hstack([W, W])
        for ia in range(len(WZ)):
            for ib in range(ia + 1, len(WZ)):
                ta, tb = AB.S.orientiere(WZ[ia], WZ[ib])
                a = np.concatenate([y1[tb], y2[tb]])
                b = np.concatenate([y1[ta], y2[ta]])
                kz = T.paar_kennzahlen(a, b, W2, schw, voll=True)
                for k in ["bias", "sd", "rc"] + ["D%d" % s for s in schw] + ["wechsel"]:
                    est, lo, hi = T.ci(kz[k])
                    rows_zw.append(dict(mass=m, kandidat=ta, komparator=tb, paarpositionen=int(kz["n"][0]), paare=len(d), personen=n_pers, kennzahl=k, schaetzer=est, ki_unten=lo, ki_oben=hi))
                for tw in (ta, tb):
                    for s in schw:
                        diff = rep[tw]["D%d" % s] - kz["D%d" % s]
                        est, lo, hi = T.ci(diff)
                        rows_df.append(dict(mass=m, schwelle=s, programm_wiederholung=tw, paar_zwischen="%s gegen %s" % (ta, tb), paare=len(d), personen=n_pers, D_wiederholung=float(rep[tw]["D%d" % s][0]),
                                            D_zwischen=float(kz["D%d" % s][0]), differenz_wiederholung_minus_zwischen=est, ki_unten=lo, ki_oben=hi))
        # Fehler-/Ausfallzaehlung
        rows_ft.append(dict(mass=m, hauptpaare=int(len(gi)), ein_paar_je_person=int(len(gs)), personen=n_pers,
                            abstand_median_h_gewaehlt=float(pairs.abstand_h.iloc[gs].median()), abstand_median_h_hauptpaare=float(pairs.abstand_h.iloc[gi].median())))
    df_re, df_zw, df_df, df_ue, df_ft = (pd.DataFrame(x) for x in (rows_re, rows_zw, rows_df, rows_ue, rows_ft))

    def schreibe(df, name):
        df.to_csv(OUT_PUB / name, index=False, encoding="utf-8", float_format="%.6g")
        df.to_csv(OUT_ROH / name, index=False, encoding="utf-8", float_format="%.6g")
    schreibe(df_re, "einpaar_wiederholung.csv")
    schreibe(df_zw, "einpaar_zwischen_programmen.csv")
    schreibe(df_df, "einpaar_differenz.csv")
    schreibe(df_ue, "einpaar_ueberlappung.csv")
    schreibe(df_ft, "einpaar_auswahl.csv")
    schreibe(pd.DataFrame(KONTROLLEN), "einpaar_kontrollen.csv")
    schreibe(pd.DataFrame(pruef), "einpaar_hashpruefung.csv")
    info = dict(version=VERSION, seed=SEED, B=B, paarliste_sha256=pl_hash, laufzeit_s=round(time.time() - t0, 1), kontrollen_bestanden=int(sum(k["bestanden"] for k in KONTROLLEN)), kontrollen_gesamt=len(KONTROLLEN),
                skript_sha256=regel["skript_sha256"])
    (OUT_PUB / "einpaar_LAUF.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    AB.log("fertig in %.1f s; Kontrollen %d von %d bestanden" % (time.time() - t0, info["kontrollen_bestanden"], info["kontrollen_gesamt"]))
    if info["kontrollen_bestanden"] != info["kontrollen_gesamt"]:
        raise SystemExit("KONTROLLEN NICHT ALLE BESTANDEN")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modus", default="lauf")
    ap.add_argument("--selbsttest", action="store_true")
    a = ap.parse_args()
    if a.selbsttest:
        selbsttest()
        print("Selbsttest bestanden")
    else:
        lauf()
