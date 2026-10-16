# -*- coding: utf-8 -*-
"""aw245_basis.py (version 1.0): shared building blocks for the three post hoc repeat-pair analyses (exploratory) of the program comparison study.

Builds on aw239_basis.py (hash checked, together with aw234_auswertung.py and schwellen_aw228.py version 1.3). The locked benchmark tables are only read;
beforehand the hashes of the data lock are checked (all entries). New: loading the values (QRS, QT, RR, delivered QTc) for an arbitrary set of PTB-XL recordings
(not only P0), reading the large long-format files in chunks; status reasons per program.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

BASIS239 = Path(str(data_path("pub12.werkzeuge_pub12_aw239_aw239_basis_py")))
BASIS239_HASH = "50fcf31d788931c671bd0021e112d2f9272795daecd2c24bbd215af0b0d4c7bd"
OUT_ROH = Path(str(data_path("pub12.benchmark_pub12_aw245")))
OUT_PUB = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw245")))
PAARLISTE = Path(str(data_path("pub12.aw_223_aw223_paarliste_stunden_csv")))
PAARLISTE_HASH = "b8ed58b509f6d72d35c983ad9b303234d947389b80d1a20e54f08e6d8bf13e0b"  # full value from the specification draft of the pair list; a shortened form in a later document, "b8ed58b5...0e0b", has a different ending (3e0b there)
WERKZEUGE = ["LoopLens", "BRAVEHEART", "12SL", "Uni-G", "ECGDeli"]


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _lade239():
    h = sha256_datei(BASIS239)
    if h != BASIS239_HASH:
        raise SystemExit("aw239_basis.py differs: " + h)
    spec = importlib.util.spec_from_file_location("aw239_basis", str(BASIS239))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aw239_basis"] = mod
    spec.loader.exec_module(mod)
    return mod


BA = _lade239()
S = BA.S
log = S.log


def pruefe_sperre():
    ok = S.hashpruefung(S.gelesene_dateien())
    log("Hash check of the data lock: %d entries, all match: %s" % (len(S.STATE["pruef"]), ok))
    if not ok:
        raise SystemExit("ABORT: hash check failed")
    return list(S.STATE["pruef"])


def pruefe_paarliste():
    h = sha256_datei(PAARLISTE)
    ok = h == PAARLISTE_HASH
    log("Pair list SHA-256 %s, equal to the bound full value: %s" % (h, ok))
    if not ok:
        raise SystemExit("ABORT: pair list differs from the bound version")
    return h


def lade_meta():
    P = S.PFADE
    meta = pd.read_csv(P["meta"], usecols=["ecg_id", "patient_id", "sex", "age", "scp_codes", "recording_date"]).set_index("ecg_id")
    return meta


def _chunks_gefiltert(pfad, ids_set, usecols, extra_filter, chunksize=1_500_000):
    teile = []
    for ch in pd.read_csv(pfad, usecols=usecols, dtype=str, chunksize=chunksize):
        ch = ch[(ch.datensatz == "ptbxl") & (ch.vorverarbeitung == "VV0") & ch.messgroesse.isin(extra_filter)]
        a = pd.to_numeric(ch["aufnahme"], errors="coerce")
        ch = ch[a.isin(ids_set)]
        if len(ch):
            teile.append(ch)
    return pd.concat(teile, ignore_index=True) if teile else pd.DataFrame(columns=usecols)


def lade_werte(ids):
    """Like schwellen_aw228.lade_alles(), but for arbitrary ecg_ids (VV0, native outputs). Returns dict tool -> dict (qrs, qt, rr, qtc_nativ, qtc_einh, hr)."""
    ids = np.asarray(ids, dtype=np.int64)
    ids_set = set(int(i) for i in ids)
    P = S.PFADE
    cols = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status"]
    D = {}
    log("loading LoopLens")
    ll = _chunks_gefiltert(P["erg"] / "looplens/langformat/looplens_langformat.csv.gz", ids_set, cols, ["qrs_dauer_global", "qt_global", "qtcf", "rr_intervall_ll"])
    D["LoopLens"] = dict(qrs=S.lang_ok(ll, "qrs_dauer_global", ids), qt=S.lang_ok(ll, "qt_global", ids), rr=S.lang_ok(ll, "rr_intervall_ll", ids),
                         qtc_nativ=S.lang_ok(ll, "qtcf", ids))
    del ll
    log("loading BRAVEHEART")
    bh = _chunks_gefiltert(P["erg"] / "ptbxl/ausgabe_langformat.csv", ids_set, cols, ["qrs_dauer_global", "qt_global", "qtcf"])
    roh = pd.read_csv(P["erg"] / "ptbxl/processed_ecgs_roh.csv.gz", usecols=["filename", "hr"])
    roh["ecg"] = roh.filename.map(lambda s: int(re.search(r"ptbxl_(\d+)", Path(str(s)).name).group(1)))
    roh = roh[~roh.ecg.duplicated(keep="first")].set_index("ecg")
    hr = roh["hr"].reindex(ids).to_numpy(float)
    rr_bh = np.where(np.isfinite(hr) & (hr > 0), 60000.0 / hr, np.nan)
    D["BRAVEHEART"] = dict(qrs=S.lang_ok(bh, "qrs_dauer_global", ids), qt=S.lang_ok(bh, "qt_global", ids), rr=rr_bh, qtc_nativ=S.lang_ok(bh, "qtcf", ids))
    del bh
    log("loading archive products")
    for tool, f, rr_art in (("12SL", "12sl_features.csv", "hr"), ("Uni-G", "unig_features.csv", "hr"), ("ECGDeli", "ecgdeli_features.csv", "rr")):
        cs = ["ecg_id", "QRS_Dur_Global", "QT_Int_Global"]
        if rr_art == "hr":
            cs += ["HR__Global", "QT_IntFridericia_Global", "QT_IntCorr_Global"]
        else:
            cs += ["RR_Mean_Global", "QT_IntFramingham_Global"]
        df = pd.read_csv(P["plus"] / f, usecols=cs)
        df = df[~df.ecg_id.duplicated(keep="first")].set_index("ecg_id").reindex(ids)
        if rr_art == "hr":
            h = df["HR__Global"].to_numpy(float)
            rr = np.where(np.isfinite(h) & (h > 0), 60000.0 / h, np.nan)
            nat = df["QT_IntFridericia_Global"].to_numpy(float)
            deliv = df["QT_IntCorr_Global"].to_numpy(float)
        else:
            rr = df["RR_Mean_Global"].to_numpy(float)
            nat = np.full(len(ids), np.nan)
            deliv = df["QT_IntFramingham_Global"].to_numpy(float)
        D[tool] = dict(qrs=df["QRS_Dur_Global"].to_numpy(float), qt=df["QT_Int_Global"].to_numpy(float), rr=rr, qtc_nativ=nat, qtc_geliefert=deliv)
    # delivered standard QTc per program: 12SL, Uni-G QT_IntCorr_Global; BRAVEHEART, LoopLens qtcf; ECGDeli Framingham column
    D["LoopLens"]["qtc_geliefert"] = D["LoopLens"]["qtc_nativ"]
    D["BRAVEHEART"]["qtc_geliefert"] = D["BRAVEHEART"]["qtc_nativ"]
    for t, d in D.items():
        d["qtc_einh"] = S.fridericia(d["qt"], d["rr"])
    return D


def status_gruende(ids):
    """Status and status reason per program and recording for the QRS duration (VV0): dict tool -> DataFrame(aufnahme, status, grund). 12SL/Uni-G/ECGDeli: missing if NaN."""
    ids_set = set(int(i) for i in ids)
    P = S.PFADE
    cols = ["datensatz", "aufnahme", "vorverarbeitung", "messgroesse", "wert", "status", "statusgrund"]
    out = {}
    for tool, pfad in (("LoopLens", P["erg"] / "looplens/langformat/looplens_langformat.csv.gz"), ("BRAVEHEART", P["erg"] / "ptbxl/ausgabe_langformat.csv")):
        d = _chunks_gefiltert(pfad, ids_set, cols, ["qrs_dauer_global", "qt_global"])
        d = d[d.messgroesse == "qrs_dauer_global"].copy()
        d["aufnahme"] = d["aufnahme"].astype(int)
        d = d[~d.aufnahme.duplicated(keep="first")].set_index("aufnahme")
        out[tool] = d[["status", "statusgrund", "wert"]]
    return out


def zusammen(v):
    return np.isfinite(v)
