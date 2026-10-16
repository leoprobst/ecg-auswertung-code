"""AW-223: Paarliste PTB-XL mit Stundenabstand (nur Metadaten und Statusspalten, nie wert).

Paare: je Person alle Aufnahmen aufsteigend nach (recording_date, ecg_id), aufeinanderfolgende Paare.
abstand_h = (recording_date_2 - recording_date_1) in Stunden aus dem Zeitstempel (Datum und Uhrzeit).
Statusmerker qrs_ok_ll / qrs_ok_bh: status == ok fuer messgroesse qrs_dauer_global, Vorverarbeitung VV0, datensatz ptbxl.
Aufruf: python -I aw223_paarliste.py <ausgabeordner>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import sys, json, hashlib
import pandas as pd

META = str(data_path("pub12.research_diagnostic_identifiability_20261013_ptb_xl_1_0_3_ptbxl_database_csv"))
LL = str(data_path("pub12.benchmark_ergebnisse_looplens_langformat_looplens_langformat_csv_gz"))
BH = str(data_path("pub12.benchmark_ergebnisse_ptbxl_ausgabe_langformat_csv"))
PT = str(data_path("pub12.aw_219_listen_pt_ecg_id_txt"))
STATUSCOLS = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "status"]


def ok_set(path):
    d = pd.read_csv(path, usecols=STATUSCOLS, dtype=str)
    d = d[(d.datensatz == "ptbxl") & (d.vorverarbeitung == "VV0") & (d.messgroesse == "qrs_dauer_global")]
    return set(d.aufnahme[d.status == "ok"].astype(int))


def main(out):
    m = pd.read_csv(META, usecols=["ecg_id", "patient_id", "recording_date"])
    m["t"] = pd.to_datetime(m.recording_date, format="%Y-%m-%d %H:%M:%S")
    m = m.sort_values(["patient_id", "t", "ecg_id"], kind="mergesort")
    g = m.groupby("patient_id", sort=True)
    a = m.assign(ecg_id_2=g.ecg_id.shift(-1), t2=g.t.shift(-1))
    a = a[a.ecg_id_2.notna()].copy()
    a["ecg_id_2"] = a.ecg_id_2.astype(int)
    a["abstand_h"] = (a.t2 - a.t).dt.total_seconds() / 3600.0
    a = a.rename(columns={"ecg_id": "ecg_id_1"})
    ll, bh = ok_set(LL), ok_set(BH)
    pt = set(int(x) for x in open(PT).read().split())
    a["qrs_ok_ll"] = a.ecg_id_1.isin(ll) & a.ecg_id_2.isin(ll)
    a["qrs_ok_bh"] = a.ecg_id_1.isin(bh) & a.ecg_id_2.isin(bh)
    a["pt_paar"] = a.ecg_id_1.isin(pt) | a.ecg_id_2.isin(pt)
    cols = ["patient_id", "ecg_id_1", "ecg_id_2", "abstand_h", "qrs_ok_ll", "qrs_ok_bh", "pt_paar"]
    a = a[cols].sort_values(["patient_id", "ecg_id_1"]).reset_index(drop=True)
    a.to_csv(out + r"\aw223_paarliste_stunden.csv", index=False, float_format="%.6f")
    z = {}
    for lab, h in [("le_2h", 2), ("le_24h", 24), ("le_168h", 168)]:
        s = a[a.abstand_h <= h]
        z[lab] = {"paare": int(len(s)), "personen": int(s.patient_id.nunique())}
    json.dump({"n_paare": int(len(a)), "zaehlung": z}, open(out + r"\aw223_paarliste_zaehlung.json", "w"), indent=1)
    print(len(a), z)


if __name__ == "__main__":
    main(sys.argv[1])
