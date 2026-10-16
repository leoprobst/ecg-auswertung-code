"""Effect check of the external band-pass (evaluation protocol section 8, control K6 part 1). Test sine against the theoretical magnitude response.

Call: python -I -B aw220_filtertest.py <output.json>
Theory: square of the magnitude response of the SOS filter (forward-backward). Measured: ratio of the RMS values in the middle
(2.5 to 7.5 s) of a 10-s sine wave at 500 Hz after bandpass_vv2 against the input wave.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfreqz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_arme as A

fs = 500
t = np.arange(5000) / fs
sos = butter(A.BP_ORDNUNG, list(A.BP_BAND), btype="bandpass", fs=fs, output="sos")
erg = {"filter": A.BP_BESCHREIBUNG, "ordnung": A.BP_ORDNUNG, "band_hz": list(A.BP_BAND), "fs": fs, "padlen": A.BP_PADLEN, "frequenzen": []}
for f0 in (0.25, 0.5, 1.0, 20.0, 40.0, 60.0):
    x = np.sin(2 * np.pi * f0 * t)[:, None] * np.ones((1, 12))
    y = A.bandpass_vv2(x, fs)
    mid = slice(1250, 3750)
    gem = np.sqrt(np.mean(y[mid, 0] ** 2)) / np.sqrt(np.mean(x[mid, 0] ** 2))
    _, h = sosfreqz(sos, worN=[f0], fs=fs)
    soll = float(abs(h[0]) ** 2)
    erg["frequenzen"].append({"hz": f0, "gemessen_db": float(round(20 * np.log10(gem), 4)), "soll_db": float(round(20 * np.log10(soll), 4)),
                              "abweichung_db": float(round(abs(20 * np.log10(gem) - 20 * np.log10(soll)), 4))})
erg["max_abweichung_db"] = max(r["abweichung_db"] for r in erg["frequenzen"])
erg["bestanden_0_1_db"] = bool(erg["max_abweichung_db"] <= 0.1)
Path(sys.argv[1]).write_text(json.dumps(erg, indent=1), encoding="utf-8")
print(json.dumps(erg, indent=1))
