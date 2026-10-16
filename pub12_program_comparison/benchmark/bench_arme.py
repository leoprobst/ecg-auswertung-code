"""Common building blocks of the gain and filter arms (BRAVEHEART, NeuroKit2). Computation only, no evaluation.

Preprocessing identifiers (column vorverarbeitung in the long format, as in the LoopLens gain arm):
  VV0        no scaling, native chain of the tool
  VV0_G0.5   factor 0.5 after reading into physical units (mV), before any tool stage (protocol section on gain)
  VV0_G1     factor 1
  VV0_G2     factor 2
  VV1        internal filters of the tool off (protocol section on filters)
  VV2        VV1 plus external band-pass 0.5 to 40 Hz (protocol section on filters)

External filter (protocol section on filters, verbatim): scipy.signal.butter(4, [0.5, 40], btype="bandpass", fs=500, output="sos"),
sosfiltfilt(sos, x, axis=0, padtype="odd", padlen=2500). axis=0 is the time axis of the matrix (n_samples, 12); the protocol does not
name the axis, sosfiltfilt otherwise filters the last axis (leads). Butterworth, order 4 of the band-pass (eight poles),
forward-backward pass (zero phase, magnitude response squared), SOS form.

Scaling: multiplication of the float64 matrix by 0.5, 1.0 or 2.0 (powers of two, exact in floating-point representation as long as
there is no overflow or underflow; control 4 checks the round trip).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import hashlib
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))

PTBXL = Path(str(data_path("pub12.samples_ptb_x")))
LUDB = Path(str(data_path("pub12.lobachevsky_university_electrocardiography_database_1_0_1")))
LISTEN = Path(str(data_path("pub12.aw_219_listen")))
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

VV_KENNUNGEN = ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2", "VV1", "VV2")
FAKTOR = {"VV0_G0.5": 0.5, "VV0_G1": 1.0, "VV0_G2": 2.0}

BP_ORDNUNG = 4
BP_BAND = (0.5, 40.0)
BP_FS = 500
BP_PADLEN = 2500
BP_BESCHREIBUNG = ("scipy.signal.butter(4, [0.5, 40], btype='bandpass', fs=500, output='sos'); "
                   "scipy.signal.sosfiltfilt(sos, x, axis=0, padtype='odd', padlen=2500)")

_SUMS: dict[str, str] = {}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _lade_summen() -> None:
    if _SUMS:
        return
    for praefix, wurzel in (("ptbxl", PTBXL), ("ludb", LUDB)):
        for line in (wurzel / "SHA256SUMS.txt").read_text().splitlines():
            p = line.split(maxsplit=1)
            if len(p) == 2:
                _SUMS[f"{praefix}:" + p[1].strip().lstrip("*")] = p[0]


def basis_pfad(ds: str, rec: str) -> tuple[Path, str]:
    """(Path without extension, relative path for the checksum list)."""
    if ds == "ptbxl":
        n = int(rec)
        rel = f"records500/{n // 1000 * 1000:05d}/{n:05d}_hr"
        return PTBXL / rel, rel
    if ds == "ludb":
        rel = f"data/{rec}"
        return LUDB / rel, rel
    raise ValueError(ds)


def pruefe_quelle(ds: str, rec: str) -> dict:
    """SHA-256 of .hea and .dat against SHA256SUMS.txt of the dataset; raises on deviation."""
    _lade_summen()
    base, rel = basis_pfad(ds, rec)
    shas = {}
    for suf in (".hea", ".dat"):
        h = sha_bytes((base.parent / (base.name + suf)).read_bytes())
        if _SUMS.get(f"{ds}:{rel}{suf}") != h:
            raise RuntimeError(f"File deviates from the checksum: {ds}:{rel}{suf}")
        shas[suf[1:]] = h[:16]
    return shas


def bandpass_vv2(sig: np.ndarray, fs: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    if int(fs) != BP_FS:
        raise ValueError(f"Sampling rate {fs} Hz, the filter is defined for {BP_FS} Hz")
    sos = butter(BP_ORDNUNG, list(BP_BAND), btype="bandpass", fs=BP_FS, output="sos")
    return sosfiltfilt(sos, sig, axis=0, padtype="odd", padlen=BP_PADLEN)


def vorverarbeite(sig: np.ndarray, fs: int, vv: str) -> np.ndarray:
    """Input: float64 (n_samples, 12) in mV. Output per identifier (see module header). VV0 leaves the signal unchanged."""
    if vv == "VV0" or vv == "VV1":
        return sig
    if vv in FAKTOR:
        return sig * FAKTOR[vv]
    if vv == "VV2":
        return bandpass_vv2(sig, fs)
    raise ValueError(vv)


def sha_datei(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(1 << 22)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def lies_liste(name: str) -> list[str]:
    return [x.strip() for x in (LISTEN / f"{name}_ecg_id.txt").read_text().split() if x.strip()]


def pruefe_listen() -> dict:
    """List hashes against ziehung_manifest.json (protocol, drawing section); raises on deviation."""
    import json
    man = json.loads((LISTEN / "ziehung_manifest.json").read_text(encoding="utf8"))
    ergebnis = {}
    for name in ("S_G", "PT"):
        h = sha_datei(LISTEN / f"{name}_ecg_id.txt")
        soll = man["dateien"][name]["sha256"]
        if h != soll:
            raise RuntimeError(f"List {name} deviates from the manifest")
        ergebnis[name] = h
    return ergebnis
