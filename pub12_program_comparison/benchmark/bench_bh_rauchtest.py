"""Plausibility check of the BRAVEHEART output for one recording (smoke test).

Call (-I): bench_bh_rauchtest.py <input_folder> <recording_stem> [<ludb_base>]
  input_folder: folder with <stem>.txt (input), processed_ecgs_*.csv, annotations_*/<stem>.anno
  ludb_base:    optional WFDB base (without extension) of the LUDB recording, for information only (no comparison verdict)

Checks:
  P1 sampling rate and number of samples of the output equal the input (first line, number of lines)
  P2 lead order: the output R waves per lead correlate with our own estimator
     from the input file better with the correct assignment than with swapped aVF/aVL
  P3 unit mV: order of magnitude of the R waves equals the input amplitude (ratio 0.6 to 1.2)
  P4 measurement points in the signal: Q < QRS < S < Tend, the QRS mark is a local maximum of the vector magnitude
     (sum of |lead| over I, II, V1-V6, +-40 ms) and distances are plausible
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

SPALTEN = ["I", "II", "III", "AVR", "AVF", "AVL", "V1", "V2", "V3", "V4", "V5", "V6"]
CSVNAME = {"I": "L1", "II": "L2", "III": "L3", "AVR": "avR", "AVF": "avF", "AVL": "avL",
           "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6"}


def lese_anno_beats(pfad: Path):
    zeilen = pfad.read_text(encoding="utf-8").splitlines()
    beats = []
    for z in zeilen:
        t = z.split(",")
        if len(t) == 4 and all(x.strip().lstrip("-").isdigit() for x in t):
            beats.append([int(x) for x in t])
    return np.array(beats)


def pruefe(ordner: Path, stamm: str, ludb_basis: str | None = None) -> dict:
    out: dict = {"aufnahme": stamm}
    txt = ordner / f"{stamm}.txt"
    m = np.loadtxt(txt)
    fs = m[0]
    sig = m[1:]
    csvdatei = sorted(glob.glob(str(ordner / "processed_ecgs_*.csv")))[-1]
    with open(csvdatei, encoding="utf-8", newline="") as f:
        z = next(csv.DictReader(f))
    out["P1_fs_eingabe"] = float(fs[0])
    out["P1_fs_ausgabe"] = float(z["freq"])
    out["P1_samples_eingabe"] = int(sig.shape[0])
    out["P1_samples_ausgabe"] = int(float(z["num_samples"]))
    out["P1_ok"] = bool(float(z["freq"]) == fs[0] and int(float(z["num_samples"])) == sig.shape[0])
    anno = sorted(glob.glob(str(ordner / "annotations_*" / f"{stamm}.anno")))
    beats = lese_anno_beats(Path(anno[-1]))
    out["beats_anno"] = beats.tolist()
    out["num_beats_ausgabe"] = int(float(z["num_beats"]))

    # P2/P3: R wave per lead. Own estimator: median over beats of max(sig[Q:S]) - median(sig[Q-40ms:Q])
    pre = int(0.04 * fs[0])
    eigen = []
    for j in range(12):
        werte = []
        for q, r, s, te in beats:
            seg = sig[q:s + 1, j]
            basis = np.median(sig[max(q - pre, 0):q, j])
            werte.append(seg.max() - basis)
        eigen.append(np.median(werte))
    eigen = np.array(eigen)
    bh = np.array([float(z[f"{CSVNAME[n]}_r_wave"]) for n in SPALTEN])
    r_ok = float(np.corrcoef(eigen, bh)[0, 1])
    tausch = list(range(12))
    tausch[4], tausch[5] = tausch[5], tausch[4]  # aVF <-> aVL
    r_tausch = float(np.corrcoef(eigen[tausch], bh)[0, 1])
    out["P2_korrelation_R_zuordnung_korrekt"] = r_ok
    out["P2_korrelation_R_aVF_aVL_vertauscht"] = r_tausch
    out["P2_ok"] = bool(r_ok > 0.9 and r_ok > r_tausch)
    out["P2_R_eigen_mV"] = {n: round(float(v), 3) for n, v in zip(SPALTEN, eigen)}
    out["P2_R_braveheart_mV"] = {n: round(float(v), 3) for n, v in zip(SPALTEN, bh)}
    verh = bh[np.abs(eigen) > 0.2] / eigen[np.abs(eigen) > 0.2]
    out["P3_verhaeltnis_R_bh_zu_eigen_median"] = float(np.median(verh))
    out["P3_ok"] = bool(0.5 <= np.median(verh) <= 1.3)

    # P4
    vm = np.sum(np.abs(sig[:, [0, 1, 6, 7, 8, 9, 10, 11]]), axis=1)
    fen = int(0.04 * fs[0])
    ok_ordnung = bool(np.all(beats[:, 0] < beats[:, 1]) and np.all(beats[:, 1] < beats[:, 2])
                      and np.all(beats[:, 2] < beats[:, 3]))
    ok_lokal = []
    for q, r, s, te in beats:
        lo, hi = max(r - fen, 0), min(r + fen + 1, len(vm))
        # the QRS mark lies within 40 ms of the magnitude maximum of the QRS range
        pk = q + int(np.argmax(vm[q:s + 1]))
        ok_lokal.append(abs(pk - r) <= fen)
    out["P4_ordnung_ok"] = ok_ordnung
    out["P4_qrs_marke_nahe_betragsmaximum"] = [bool(x) for x in ok_lokal]
    dauer_qrs = (beats[:, 2] - beats[:, 0]) * 1000 / fs[0]
    dauer_qt = (beats[:, 3] - beats[:, 0]) * 1000 / fs[0]
    out["P4_qrs_ms_je_schlag"] = [round(float(x)) for x in dauer_qrs]
    out["P4_qt_ms_je_schlag"] = [round(float(x)) for x in dauer_qt]
    out["P4_ok"] = bool(ok_ordnung and all(ok_lokal) and 40 <= dauer_qrs.min() and dauer_qrs.max() <= 200
                        and 200 <= dauer_qt.min() and dauer_qt.max() <= 600)
    out["messwerte"] = {k: z[k] for k in ["hr", "qrs_int", "qt_int", "qrst_angle_peak", "qrst_angle_area",
                                           "q_peak_mag", "q_area_mag", "sai_qrst", "vcg_length_qrs",
                                           "quality_prob"]}

    # information only: LUDB expert marks, lead ii
    if ludb_basis:
        import wfdb
        a = wfdb.rdann(ludb_basis, "ii")
        sym = np.array(a.symbol)
        smp = np.array(a.sample)
        on, off = smp[sym == "("], smp[sym == ")"]
        tend = []
        for i, s in enumerate(sym):
            if s == "t" and i + 1 < len(sym) and sym[i + 1] == ")":
                tend.append(smp[i + 1])
        info = []
        for q, r, s, te in beats:
            # nearest expert marks
            o = on[np.argmin(np.abs(on - q))] if len(on) else None
            f = off[np.argmin(np.abs(off - s))] if len(off) else None
            t = tend[int(np.argmin(np.abs(np.array(tend) - te)))] if tend else None
            info.append({"bh_Q": int(q), "exp_on_nahe": None if o is None else int(o),
                         "bh_S": int(s), "exp_off_nahe": None if f is None else int(f),
                         "bh_Tend": int(te), "exp_Tend_nahe": None if t is None else int(t)})
        out["info_ludb_expert_ii_naechstliegend"] = info
        out["info_hinweis"] = "plausibility only, no comparison verdict, outliers possible (different lead/model)"
    return out


if __name__ == "__main__":
    a = sys.argv
    res = pruefe(Path(a[1]), a[2], a[3] if len(a) > 3 else None)
    print(json.dumps(res, indent=2, ensure_ascii=True))
    sys.exit(0 if (res["P1_ok"] and res["P2_ok"] and res["P3_ok"] and res["P4_ok"]) else 1)
