# -*- coding: utf-8 -*-
"""aw239_r6_lauf.py (AW-239, P12-R6, Version 1.0): NeuroKit2 0.2.13 auf LUDB 1.0.1 (200 Aufnahmen, 12 Ableitungen, 500 Hz) mit Varianten der Skalierung und der R-Gipfel-Vorgabe.
Nur Rechnen und Speichern der Schlagmarken je Ableitung; die Auswertung steht in aw239_r6_auswertung.py.

Aufruf je Ableitung (wie AW-220, nk_lauf.py, gebunden): clean = ecg_clean(x * g, sampling_rate=500, method="neurokit"); peaks = ecg_peaks(clean, 500, method="neurokit");
ecg_delineate(clean, peaks, 500, method="dwt", show=False, check=False). Alle uebrigen Parameter sind Standardwerte von 0.2.13.
Varianten (g = Verstaerkung je Ableitung):
  V0        g = 1 (LUDB, je Ableitung auf 1 mV Spannweite normiert; entspricht AW-220 VV0)
  G0.1, G0.5, G2, G10   gleichmaessige Faktoren
  Glognorm  g = exp(N(0, 0.3^2)) je Ableitung und Aufnahme (Seed: SeedSequence([15102396, Aufnahme]))
  Gtypisch  g so, dass die mediane QRS-Spannweite (Expertenfenster) der Ableitung der typischen QRS-Spannweite in PTB-XL entspricht (Median ueber PTB-XL+ 12SL: R_Amp_X minus das kleinere von Q_Amp_X und S_Amp_X)
  Rexpert   g = 1, aber die R-Gipfel der Expertenmarken (LUDB-Annotation 'N' der Ableitung) werden an ecg_delineate uebergeben statt der eigenen Detektion
Ausgabe: <local path><Variante>.pkl (dict Aufnahme -> dict Ableitungsindex -> Array (Schlaege, 11): R-Gipfel, P-Beginn, P-Gipfel, P-Ende, Q-Gipfel, R-Beginn,
R-Ende, S-Gipfel, T-Beginn, T-Gipfel, T-Ende; NaN = fehlt) und r6_lauf_info.json.
Aufruf: <local path> -I -B aw239_r6_lauf.py --workers 3
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

VERSION = "1.0"
LUDB = Path(str(data_path("pub12.lobachevsky_university_electrocardiography_database_1_0_1")))
OUT = Path(str(data_path("pub12.benchmark_pub12_aw239_r6")))
F12 = Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features_12sl_features_csv")))
F12_HASH = "cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LUDB_EXT = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
FEAT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
VARIANTEN = ["V0", "G0.1", "G0.5", "G2", "G10", "Glognorm", "Gtypisch", "Rexpert"]
GAIN_UNI = {"G0.1": 0.1, "G0.5": 0.5, "G2": 2.0, "G10": 10.0}
SEED_LOGN = 15102396
WELLEN = ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")
_W = {}


def sha256_datei(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ludb_sums():
    d = {}
    for z in (LUDB / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        p = z.split()
        if len(p) == 2:
            d[p[1].replace("\\", "/").lstrip("*")] = p[0]
    return d


def lese_expert(aid):
    """Expertenmarken je Ableitung: dict Ableitungsindex -> dict {'N': [(Gipfel, Beginn, Ende)], 'p': [...], 't': [...]} (wie lese_ludb_ableitung in bench_auswertung_v11.py)."""
    import wfdb
    res = {}
    for li, ext in enumerate(LUDB_EXT):
        a = wfdb.rdann(str(LUDB / "data" / str(aid)), ext)
        out = {"p": [], "N": [], "t": []}
        on = pk = pksym = None
        for s, sym in zip(a.sample, a.symbol):
            if sym == "(":
                on, pk, pksym = int(s), None, None
            elif sym in out and on is not None:
                pk, pksym = int(s), sym
            elif sym == ")":
                if on is not None and pk is not None and pksym is not None:
                    out[pksym].append((pk, on, int(s)))
                on = pk = pksym = None
        res[li] = out
    return res


def lese_signal(aid):
    import wfdb
    r = wfdb.rdrecord(str(LUDB / "data" / str(aid)))
    names = [str(x).upper() for x in r.sig_name]
    if sorted(names) != sorted(LEADS):
        raise ValueError("Ableitungen %s" % names)
    if sorted({str(u).lower() for u in r.units}) != ["mv"]:
        raise ValueError("Einheiten")
    if int(round(r.fs)) != 500:
        raise ValueError("fs")
    return np.asarray(r.p_signal, float)[:, [names.index(n) for n in LEADS]], 500


def typische_spannweiten():
    if sha256_datei(F12) != F12_HASH:
        raise SystemExit("12sl_features.csv: Hash weicht ab")
    import pandas as pd
    cols = ["R_Amp_" + l for l in FEAT_LEADS] + ["S_Amp_" + l for l in FEAT_LEADS] + ["Q_Amp_" + l for l in FEAT_LEADS]
    d = pd.read_csv(F12, usecols=cols)
    typ = {}
    for l in FEAT_LEADS:
        w = (d["R_Amp_" + l] - np.fmin(d["Q_Amp_" + l], d["S_Amp_" + l])).to_numpy(float)  # Spannweite R-Gipfel bis tiefster Q-/S-Gipfel
        typ[l] = float(np.nanmedian(w))
    return typ


def init_worker():
    import neurokit2 as nk
    warnings.simplefilter("ignore")
    if nk.__version__ != "0.2.13":
        raise RuntimeError("NeuroKit2 " + nk.__version__)
    _W["nk"] = nk


def messe(x, fs, rp_vorgabe=None):
    nk = _W["nk"]
    c = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
    if rp_vorgabe is None:
        _, info = nk.ecg_peaks(c, sampling_rate=fs, method="neurokit")
        rp = np.asarray(info["ECG_R_Peaks"], dtype=int)
    else:
        rp = np.asarray(rp_vorgabe, dtype=int)
    if rp.size == 0:
        return np.zeros((0, 11))
    _, waves = nk.ecg_delineate(c, rp, sampling_rate=fs, method="dwt", show=False, check=False)
    cols = [rp.astype(float)]
    for k in WELLEN:
        v = waves.get(k)
        a = np.full(rp.size, np.nan) if v is None else np.asarray(v, dtype=float)
        if a.size != rp.size:
            raise ValueError("%s: Laenge %d statt %d" % (k, a.size, rp.size))
        cols.append(a)
    return np.column_stack(cols)


def work(task):
    aid, var, typ, _ = task
    gain_info = {}
    t0 = time.time()
    try:
        sig, fs = lese_signal(aid)
        exp = lese_expert(aid) if var in ("Gtypisch", "Rexpert") else None
        res, fehler = {}, {}
        rng = np.random.default_rng(np.random.SeedSequence([SEED_LOGN, aid]))
        logn = np.exp(rng.normal(0.0, 0.3, 12))
        for j in range(12):
            x = np.ascontiguousarray(sig[:, j])
            g = 1.0
            rp = None
            if var in GAIN_UNI:
                g = GAIN_UNI[var]
            elif var == "Glognorm":
                g = float(logn[j])
            elif var == "Gtypisch":
                w = [np.ptp(x[on:off + 1]) for pk, on, off in exp[j]["N"] if off > on]
                q = float(np.median(w)) if w else np.nan
                g = typ[FEAT_LEADS[j]] / q if np.isfinite(q) and q > 0 else 1.0
                gain_info[(aid, j)] = g
            elif var == "Rexpert":
                rp = np.array(sorted(pk for pk, on, off in exp[j]["N"]), dtype=int)
                if rp.size == 0:
                    rp = None
            try:
                gain_info[(aid, j)] = g
                res[j] = messe(x * g, fs, rp)
            except Exception as exc:  # noqa: BLE001
                fehler[j] = "%s: %s" % (type(exc).__name__, str(exc)[:100])
        return aid, var, res, fehler, time.time() - t0, dict(gain_info)
    except Exception as exc:  # noqa: BLE001
        return aid, var, {}, {"alle": "%s: %s" % (type(exc).__name__, str(exc)[:150])}, time.time() - t0, {}


def haupt():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--varianten", default=",".join(VARIANTEN))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sums = ludb_sums()
    ids = sorted(int(x.split("/")[-1]) for x in (LUDB / "RECORDS").read_text().split())
    if a.n:
        ids = ids[: a.n]
    # Hashpruefung der verwendeten LUDB-Dateien
    falsch = 0
    for aid in ids:
        for ext in ["hea", "dat"] + LUDB_EXT:
            rel = "data/%d.%s" % (aid, ext)
            if rel in sums and sha256_datei(LUDB / rel) != sums[rel]:
                falsch += 1
    print("LUDB-Dateien mit abweichendem Hash:", falsch, flush=True)
    if falsch:
        return 2
    typ = typische_spannweiten()
    print("typische QRS-Spannweite (PTB-XL+ 12SL, mV):", {k: round(v, 3) for k, v in typ.items()}, flush=True)
    varianten = a.varianten.split(",")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    info = dict(version=VERSION, skript_sha256=sha256_datei(__file__), ludb_aufnahmen=len(ids), typische_spannweite_mV=typ, seed_lognorm=SEED_LOGN, varianten=varianten,
                start=time.strftime("%Y-%m-%d %H:%M:%S"), python=sys.version.split()[0], numpy=np.__version__)
    import neurokit2 as nk
    info["neurokit2"] = nk.__version__
    gain_info = {}
    with ctx.Pool(a.workers, initializer=init_worker) as pool:
        for var in varianten:
            tasks = [(aid, var, typ, None) for aid in ids]
            store, nfe = {}, 0
            gains = {}
            for aid, v, res, fehler, sek, gi in pool.imap_unordered(work, tasks, chunksize=2):
                gains.update(gi)
                store[aid] = res
                if fehler:
                    nfe += 1
            gv = np.array(list(gains.values()), float)
            info["gain_" + var] = dict(n=int(len(gv)), median=float(np.median(gv)), p05=float(np.percentile(gv, 5)), p95=float(np.percentile(gv, 95)), min=float(gv.min()), max=float(gv.max()))
            if False:
                pass
            with open(OUT / ("nk_%s.pkl" % var), "wb") as f:
                pickle.dump(store, f)
            print(var, "fertig", len(store), "Aufnahmen, mit Ableitungsfehlern:", nfe, "%.0f s" % (time.time() - t0), flush=True)
            info["fehlerhafte_aufnahmen_" + var] = nfe
    info["ende"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "r6_lauf_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(haupt())
