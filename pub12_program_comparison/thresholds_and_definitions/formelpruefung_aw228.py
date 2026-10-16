# -*- coding: utf-8 -*-
"""formelpruefung_aw228.py (AW-228, Teil A): Welche QTc-Formel steckt in den QTc-Spalten der PTB-XL+-Archivtabellen, und welches Zeitraster
tragen QRS- und QT-Werte? Liest nur die drei Archivtabellen (Hash vorher geprueft, Datensperre), keine Ergebnis- und keine Personenebene ausser ecg_id.
Aufruf: <local path> -I -B formelpruefung_aw228.py
Ausgabe: <local path> (Kopie unter publikationen\\PUB-12_etablierung\\ergebnisse). Nur Aggregate."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sw", str(HIER / "schwellen_aw228.py"))
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)
P = sw.PFADE

dateien = [P["plus"] / "12sl_features.csv", P["plus"] / "unig_features.csv", P["plus"] / "ecgdeli_features.csv"]
ok = sw.hashpruefung(dateien)
fehl = [z for z in sw.STATE["pruef"] if not z["ok"]]
print("Hashpruefung:", ok, "Eintraege", len(sw.STATE["pruef"]))
if not ok:
    print(fehl)
    sys.exit(2)


def formeln(qt, rr):
    q, r = qt / 1000.0, rr / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "Bazett": 1000 * q / np.sqrt(r),
            "Fridericia": 1000 * q / np.cbrt(r),
            "Framingham": 1000 * (q + 0.154 * (1 - r)),
            "Hodge": qt + 1.75 * (60000.0 / rr - 60.0),
        }


zeilen = []
for tool, f in (("12SL", "12sl_features.csv"), ("Uni-G", "unig_features.csv")):
    d = pd.read_csv(P["plus"] / f, usecols=["ecg_id", "QT_Int_Global", "HR__Global", "RR_Mean_Global", "QT_IntCorr_Global", "QT_IntBazett_Global",
                                            "QT_IntFridericia_Global", "QT_IntFramingham_Global"] + (["QT_IntHodge_Global"] if tool == "Uni-G" else []))
    qt = d.QT_Int_Global.to_numpy(float)
    rr_hr = 60000.0 / d.HR__Global.to_numpy(float)
    rr_m = d.RR_Mean_Global.to_numpy(float)
    spalten = {"Bazett": "QT_IntBazett_Global", "Fridericia": "QT_IntFridericia_Global", "Framingham": "QT_IntFramingham_Global"}
    if tool == "Uni-G":
        spalten["Hodge"] = "QT_IntHodge_Global"
    for rr_name, rr in (("RR=60000/HR__Global", rr_hr), ("RR_Mean_Global", rr_m)):
        calc = formeln(qt, rr)
        for fm, col in spalten.items():
            c = d[col].to_numpy(float)
            m = np.isfinite(c) & np.isfinite(calc[fm])
            dd = (c - calc[fm])[m]
            zeilen.append(dict(werkzeug=tool, spalte=col, formel=fm, rr_quelle=rr_name, n=int(m.sum()), median_diff_ms=float(np.median(dd)),
                               max_betrag_ms=float(np.max(np.abs(dd))), anteil_innerhalb_0_5ms=float(np.mean(np.abs(dd) <= 0.5)),
                               anteil_innerhalb_1_5ms=float(np.mean(np.abs(dd) <= 1.5))))
    corr = d.QT_IntCorr_Global.to_numpy(float)
    for col in ("QT_IntBazett_Global", "QT_IntFridericia_Global", "QT_IntFramingham_Global") + (("QT_IntHodge_Global",) if tool == "Uni-G" else ()):
        c = d[col].to_numpy(float)
        m = np.isfinite(c) & np.isfinite(corr)
        zeilen.append(dict(werkzeug=tool, spalte="QT_IntCorr_Global", formel="gleich " + col, rr_quelle="-", n=int(m.sum()),
                           median_diff_ms=float(np.median((corr - c)[m])), max_betrag_ms=float(np.max(np.abs((corr - c)[m]))),
                           anteil_innerhalb_0_5ms=float(np.mean(np.abs(corr - c)[m] <= 0.5)), anteil_innerhalb_1_5ms=float(np.mean(np.abs(corr - c)[m] <= 1.5))))
d = pd.read_csv(P["plus"] / "ecgdeli_features.csv", usecols=["ecg_id", "QT_Int_Global", "RR_Mean_Global", "QT_IntFramingham_Global"])
calc = formeln(d.QT_Int_Global.to_numpy(float), d.RR_Mean_Global.to_numpy(float))
for fm in ("Framingham", "Bazett", "Fridericia"):
    c = d.QT_IntFramingham_Global.to_numpy(float)
    m = np.isfinite(c) & np.isfinite(calc[fm])
    dd = (c - calc[fm])[m]
    zeilen.append(dict(werkzeug="ECGDeli", spalte="QT_IntFramingham_Global", formel=fm, rr_quelle="RR_Mean_Global", n=int(m.sum()),
                       median_diff_ms=float(np.median(dd)), max_betrag_ms=float(np.max(np.abs(dd))),
                       anteil_innerhalb_0_5ms=float(np.mean(np.abs(dd) <= 0.5)), anteil_innerhalb_1_5ms=float(np.mean(np.abs(dd) <= 1.5))))
df = pd.DataFrame(zeilen)
df.to_csv(P["out"].parent / "formelpruefung_aw228.csv", index=False, float_format="%.5g")
df.drop(columns=["n"]).to_csv(P["kopie"] / "formelpruefung_aw228.csv", index=False, float_format="%.5g")  # n in der Kopie weggelassen (Komplement zu 21.799 unter 10)
pd.set_option("display.width", 250)
print(df.to_string())

# Zeitraster (Anteil gerader Werte, Anteil durch 4 teilbar) der Archivwerte
r = []
for tool, f in (("12SL", "12sl_features.csv"), ("Uni-G", "unig_features.csv"), ("ECGDeli", "ecgdeli_features.csv")):
    d = pd.read_csv(P["plus"] / f, usecols=["QRS_Dur_Global", "QT_Int_Global"])
    for c in ("QRS_Dur_Global", "QT_Int_Global"):
        v = d[c].to_numpy(float)
        v = v[np.isfinite(v)]
        r.append(dict(werkzeug=tool, groesse=c, n=int(len(v)), anteil_ganzzahlig=float(np.mean(v == np.round(v))), anteil_gerade=float(np.mean(v % 2 == 0)),
                      anteil_durch_4=float(np.mean(v % 4 == 0))))
rd = pd.DataFrame(r)
rd.to_csv(P["out"].parent / "zeitraster_archiv_aw228.csv", index=False, float_format="%.5g")
rd.drop(columns=["n"]).to_csv(P["kopie"] / "zeitraster_archiv_aw228.csv", index=False, float_format="%.5g")
print(rd.to_string())
