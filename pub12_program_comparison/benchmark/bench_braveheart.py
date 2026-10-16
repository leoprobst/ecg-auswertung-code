"""BRAVEHEART adapter: input generation (finished) and invocation (placeholder).

Bound version: BRAVEHEART v1.8.2, tag commit 9b2e03be0ef403bf1701fae2c20e9f7b867425b3.
Input format 'unformatted' (load_unformatted.m): text file with 12 columns in the
order I, II, III, aVR, aVF, aVL, V1-V6 (aVF before aVL!). Each column starts with
the sampling rate in Hz as its first value, from line 2 on the samples in mV (ECG12.m: unitspermv = 1).
Format string in batch_settings.csv: unformatted, extension .txt.

This module does not run BRAVEHEART.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# Column order in 'unformatted' (source: load_unformatted.m, ECG12.m lines 87-90)
SPALTEN_UNFORMATTED = ["I", "II", "III", "AVR", "AVF", "AVL", "V1", "V2", "V3", "V4", "V5", "V6"]


def lese_wfdb_mv(basis: Path):
    """Reads a WFDB recording as physical values (mV). Only for recordings with 12 leads.

    Returns: (signals[n_samples, 12] in the order SPALTEN_UNFORMATTED, fs, raw lead names)
    """
    import wfdb  # imported locally so that schema/register stay usable without wfdb

    rec = wfdb.rdrecord(str(basis))
    namen = [n.upper() for n in rec.sig_name]
    fehlend = [n for n in SPALTEN_UNFORMATTED if n not in namen]
    if fehlend:
        raise ValueError(f"Leads missing in {basis.name}: {fehlend}; found: {namen}")
    einheiten = {u.lower() for u in rec.units}
    if einheiten != {"mv"}:
        raise ValueError(f"Units are not exclusively mV: {sorted(einheiten)}")
    idx = [namen.index(n) for n in SPALTEN_UNFORMATTED]
    return np.asarray(rec.p_signal, dtype=np.float64)[:, idx], int(rec.fs), rec.sig_name


def schreibe_unformatted(pfad: Path, signale: np.ndarray, fs: int) -> str:
    """Writes the BRAVEHEART file 'unformatted'. Returns the SHA-256 of the file."""
    if signale.ndim != 2 or signale.shape[1] != 12:
        raise ValueError("signale must be (n, 12)")
    if not np.isfinite(signale).all():
        raise ValueError("non-finite values in the signal: do not write, assign status nicht_ladbar")
    pfad.parent.mkdir(parents=True, exist_ok=True)
    kopf = np.full((1, 12), float(fs))
    daten = np.vstack([kopf, signale])
    with pfad.open("w", encoding="ascii", newline="\n") as f:
        for zeile in daten:
            f.write(" ".join(f"{v:.15g}" for v in zeile))
            f.write("\n")
    return hashlib.sha256(pfad.read_bytes()).hexdigest()


def konvertiere(basis: Path, ziel: Path) -> dict:
    """One recording -> one BRAVEHEART input file. Returns a manifest (dict)."""
    signale, fs, roh = lese_wfdb_mv(basis)
    sha = schreibe_unformatted(ziel, signale, fs)
    return {
        "quelle_basis": basis.name,
        "ziel": ziel.name,
        "fs_hz": fs,
        "n_samples": int(signale.shape[0]),
        "ableitungen_quelle": list(roh),
        "spalten_ziel": SPALTEN_UNFORMATTED,
        "sha256_ziel": sha,
        "format": "braveheart unformatted (v1.8.2)",
    }


def pruefe_unformatted_roundtrip(pfad: Path, erwartet: np.ndarray, fs: int) -> dict:
    """Reads the file the way load_unformatted.m does (textscan, first line = fs) and compares."""
    m = np.loadtxt(pfad, dtype=np.float64)
    ok_fs = bool(np.all(m[0] == fs))
    werte = m[1:]
    max_abw = float(np.max(np.abs(werte - erwartet)))
    return {"fs_erste_zeile_ok": ok_fs, "form": list(werte.shape), "max_abw_mV": max_abw}


def bh_aufruf_platzhalter(*_args, **_kwargs):
    """Placeholder for the BRAVEHEART run. Deliberately not implemented.

    Finding from the setup of the tool:
    * braveheart_batch.exe (compiled) reads parameters only from batch_settings.csv next to the EXE,
      and the ECG folder only through a Windows folder dialog (uigetdir), no command line argument.
    * Calling the source code braveheart_batch(folder, fmt, ...) needs MATLAB >= R2022a with a license
      and the toolboxes Wavelet, Signal Processing, Deep Learning; Octave is not supported.
    * MATLAB Runtime R2022a (9.12) is a prerequisite of the EXE.
    Open decision: dialog automation of the EXE or licensed MATLAB source.
    One run per folder yields processed_ecgs_<timestamp>.csv, one row per ECG.
    """
    raise NotImplementedError("BRAVEHEART run not released and not set up")


def eingefrorene_dateien(ordner: Path) -> dict:
    """SHA-256 of the configuration-relevant files next to the EXE (for konfig_hash)."""
    namen = ["Annoparams.csv", "batch_settings.csv", "Qualparams.csv", "ecg_formats.csv",
             "generic_csv_params.csv", "transform_mats.csv", "search_presets.csv",
             "braveheart_batch.exe", "braveheart_gui.exe"]
    return {n: hashlib.sha256((ordner / n).read_bytes()).hexdigest() for n in namen if (ordner / n).exists()}


if __name__ == "__main__":
    print(json.dumps({"spalten_unformatted": SPALTEN_UNFORMATTED}))
