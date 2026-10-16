"""Filter switch check for BRAVEHEART v1.8.2 (clarification 2 of the benchmark preparation).

Question: do the Annoparams `lowpass` and `highpass` switch the internal filters off completely?
Source code (ecgfilter.m, ECG12.m): lowpass controls wden (wavelet denoising, "high-frequency filter"),
highpass controls wander_remove (wavelet baseline removal); both branches are `if ... == 1`.

This script generates synthetic 12-lead signals (Gaussian wave model, 500 Hz, 10 s, mV) with known
contamination (baseline wander, 120 Hz component, noise, DC offset) and runs the compiled
braveheart_batch.exe in a working copy of the app (not the frozen app) for several switch settings
with save_data=1. The .mat file contains ecg_raw and ecg_filtered; ecg_filtered is compared with ecg_raw.

Call (always with -I):
  bench_bh_filtertest.py erzeuge <folder>
  bench_bh_filtertest.py lauf <app_copy> <folder> <case>      (case: A..E)
  bench_bh_filtertest.py auswerte <folder>                       (JSON per case and file)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
FS = 500
DAUER = 10.0
LEADS = ["I", "II", "III", "AVR", "AVF", "AVL", "V1", "V2", "V3", "V4", "V5", "V6"]  # column order of the 'unformatted' format

DATEIEN = ("syn_sauber.txt", "syn_maessig.txt", "syn_verunreinigt.txt")

# Switch cases: lowpass, highpass, baseline_correct_flag
FAELLE = {
    "A": (1, 1, 1),   # default (as in the benchmark arm)
    "B": (0, 0, 0),   # both off, no baseline correction
    "C": (0, 0, 1),   # both off, baseline correction as in the default
    "D": (1, 0, 0),   # lowpass only
    "E": (0, 1, 0),   # highpass only
}


def gauss(t, mu, sig, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sig) ** 2)


def beat_lead(t, r_amp, s_amp, q_amp, t_amp, p_amp):
    return (gauss(t, -0.16, 0.025, p_amp) + gauss(t, -0.025, 0.010, q_amp) + gauss(t, 0.0, 0.011, r_amp)
            + gauss(t, 0.028, 0.012, s_amp) + gauss(t, 0.25, 0.045, t_amp))


def synthese(stufe: int, seed: int) -> np.ndarray:
    """level 0 clean, 1 moderate (wander 0.10 mV, 120 Hz 0.01 mV, noise 0.005 mV, 0.05 mV DC offset), 2 strong."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(DAUER * FS)) / FS
    r_peaks = []
    x = 0.55
    while x < DAUER - 0.5:
        r_peaks.append(x)
        x += 0.8 + rng.uniform(-0.01, 0.01)
    # independent leads I, II, V1..V6 (r, s, q, t, p)
    par = {
        "I": (0.9, -0.10, -0.08, 0.25, 0.10),
        "II": (1.2, -0.25, -0.10, 0.35, 0.14),
        "V1": (0.25, -0.90, 0.0, -0.10, 0.04),
        "V2": (0.55, -1.00, 0.0, 0.30, 0.05),
        "V3": (0.90, -0.60, -0.04, 0.45, 0.06),
        "V4": (1.40, -0.30, -0.08, 0.50, 0.08),
        "V5": (1.30, -0.12, -0.10, 0.40, 0.09),
        "V6": (1.00, -0.06, -0.10, 0.32, 0.08),
    }
    sig = {k: np.zeros_like(t) for k in par}
    for rp in r_peaks:
        sel = np.abs(t - rp) < 0.5
        for k, p in par.items():
            sig[k][sel] += beat_lead(t[sel] - rp, *p)
    sig["III"] = sig["II"] - sig["I"]
    sig["AVR"] = -(sig["I"] + sig["II"]) / 2
    sig["AVL"] = sig["I"] - sig["II"] / 2
    sig["AVF"] = sig["II"] - sig["I"] / 2
    m = np.column_stack([sig[k] for k in LEADS])
    wand, gleich, hf, rausch = {0: (0, 0, 0, 0), 1: (0.10, 0.05, 0.01, 0.005), 2: (0.30, 0.20, 0.03, 0.01)}[stufe]
    if stufe:
        for j in range(12):
            m[:, j] += (wand * np.sin(2 * np.pi * 0.25 * t + 0.5 * j) + gleich + 0.1 * gleich * t
                        + hf * np.sin(2 * np.pi * 120.0 * t + 0.3 * j) + rausch * rng.standard_normal(t.size))
    return m


def schreibe(pfad: Path, m: np.ndarray) -> None:
    with pfad.open("w", encoding="ascii", newline="\n") as f:
        f.write(" ".join([f"{FS:d}"] * 12) + "\n")
        for z in m:
            f.write(" ".join(f"{v:.15g}" for v in z) + "\n")


def cmd_erzeuge(ordner: Path) -> int:
    ordner.mkdir(parents=True, exist_ok=True)
    schreibe(ordner / "syn_sauber.txt", synthese(0, 1))
    schreibe(ordner / "syn_maessig.txt", synthese(1, 3))
    schreibe(ordner / "syn_verunreinigt.txt", synthese(2, 2))
    print("created")
    return 0


def setze_fall(app: Path, fall: str) -> None:
    lp, hp, bc = FAELLE[fall]
    zeilen = (app / "Annoparams.csv").read_text(encoding="utf-8").splitlines()
    neu = []
    for z in zeilen:
        k = z.split(",")[0]
        if k == "lowpass":
            z = f"lowpass,{lp}"
        elif k == "highpass":
            z = f"highpass,{hp}"
        elif k == "baseline_correct_flag":
            z = f"baseline_correct_flag,{bc}"
        neu.append(z)
    (app / "Annoparams.csv").write_text("\n".join(neu) + "\n", encoding="utf-8")
    bs = (app / "batch_settings.csv").read_text(encoding="utf-8").splitlines()
    bs = ["save_data,1" if z.startswith("save_data,") else z for z in bs]
    (app / "batch_settings.csv").write_text("\n".join(bs) + "\n", encoding="utf-8")


def cmd_lauf(app: Path, ordner: Path, fall: str) -> int:
    setze_fall(app, fall)
    ziel = ordner / f"fall_{fall}"
    ziel.mkdir(exist_ok=True)
    for q in DATEIEN:
        (ziel / q).write_bytes((ordner / q).read_bytes())
    r = subprocess.run([sys.executable, "-I", str(HIER / "braveheart_batch_auto.py"), "--app-dir", str(app),
                        "--ecg-dir", str(ziel), "--timeout", "900", "--log", str(ziel / "auto_log.json")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"case {fall} rc={r.returncode}")
    return r.returncode


def felder_aus_mat(mat: Path, n: int = 5000) -> list[np.ndarray]:
    """All double vectors of length n from the MCOS function workspace of the .mat file (storage order).

    The ECG12 objects are not stored as normal variables. The workspace is a MAT5 data stream;
    double fields carry the tag (miDOUBLE=9, byte count 8n). The position in the stream corresponds to the
    property order of the objects (ecg_raw, ecg_filtered, vcg_raw, ...).
    """
    import struct
    import scipy.io as sio
    fw = sio.loadmat(str(mat))["__function_workspace__"].tobytes()
    pat = struct.pack("<II", 9, 8 * n)
    out, i = [], 0
    while True:
        i = fw.find(pat, i)
        if i < 0:
            break
        out.append(np.frombuffer(fw, dtype="<f8", count=n, offset=i + 8).copy())
        i += 1
    return out


def cmd_auswerte(ordner: Path, fall: str, datei: str) -> dict:
    """Compares ecg_filtered with ecg_raw (input) for one case. Returns values averaged per lead."""
    ziel = ordner / f"fall_{fall}"
    mats = sorted(ziel.glob(f"data_*/{datei}.mat"))
    if not mats:
        return {"fall": fall, "datei": datei, "fehler": "no .mat file"}
    arr = felder_aus_mat(mats[0])
    eingabe = np.loadtxt(ordner / f"{datei}.txt", skiprows=1)  # (5000, 12) in the order of LEADS
    # ecg_raw: the 12 fields that match the input exactly (order of the object properties unknown)
    raw_idx = {}
    for j in range(12):
        for k, a in enumerate(arr):
            if np.array_equal(a, eingabe[:, j]):
                raw_idx[j] = k
                break
    if len(raw_idx) < 12:
        return {"fall": fall, "datei": datei, "fehler": f"only {len(raw_idx)} of 12 leads found as exact copy of the input"}
    erstes = min(raw_idx.values())
    # ecg_filtered: the 12 fields that follow the first 12 raw fields (same property order)
    res = {"fall": fall, "datei": datei, "raw_ab_feld": erstes, "ableitungen": {}}
    for j in range(12):
        f = arr[raw_idx[j] + 12] if raw_idx[j] + 12 < len(arr) else None
        r = eingabe[:, j]
        if f is None:
            continue
        d = f - r
        res["ableitungen"][LEADS[j]] = {
            "korr": float(np.corrcoef(f, r)[0, 1]), "max_abs_diff_mV": float(np.max(np.abs(d))),
            "mittel_diff_mV": float(np.mean(d)), "std_diff_mV": float(np.std(d)), "identisch": bool(np.array_equal(f, r)),
            "rms_roh_mV": float(np.sqrt(np.mean(r ** 2))),
        }
    return res


if __name__ == "__main__":
    a = sys.argv
    if len(a) == 3 and a[1] == "erzeuge":
        sys.exit(cmd_erzeuge(Path(a[2])))
    if len(a) == 3 and a[1] == "auswerte":
        erg = [cmd_auswerte(Path(a[2]), f, d) for f in FAELLE for d in [x[:-4] for x in DATEIEN]]
        print(json.dumps(erg, indent=1))
        sys.exit(0)
    if len(a) == 5 and a[1] == "lauf":
        sys.exit(cmd_lauf(Path(a[2]), Path(a[3]), a[4]))
    print(__doc__)
    sys.exit(1)
