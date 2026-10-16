"""AW-232 v2: Adapter vom A155-Merkmalscache zum Messdatensatz von regeln.py.

Liest nur die Spannungs- und Intervallspalten (keine Referenz, keine Endpunkte). Die Spalten
`lead_measurement_<Ableitung>_qrs_r_height_mv` und `..._s_depth_mv` sind die Werte, aus denen
classical.py seine Spannungsregeln rechnet (Maximum und Minimum des QRS-Fensters des Konsens-Schlags).
Bekannte Grenzen: Q-Dauern, R'-Amplituden, S-Dauern und P-Terminalmasse sind im Cache nicht gespeichert;
R09 (Wellenform), R14 (Q) und R17 (PTF Morris) lassen sich aus dem Cache nicht rechnen.
"""
from __future__ import annotations

import math

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def cache_columns():
    cols = []
    for l in LEADS:
        cols += [f"lead_measurement_{l}_qrs_r_height_mv", f"lead_measurement_{l}_qrs_s_depth_mv"]
    return cols


def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def m_from_cache(row: dict, *, sex, qrs_ms=None, pr_ms=None, qt_ms=None, rr_ms=None,
                 sinus=None, af=None, paced=None, lbbb=None) -> dict:
    amp = {}
    for l in LEADS:
        r = _f(row.get(f"lead_measurement_{l}_qrs_r_height_mv"))
        s = _f(row.get(f"lead_measurement_{l}_qrs_s_depth_mv"))
        if r is not None and s is not None:
            amp[l] = {"r": r, "s": s}
    return {
        "sex": sex, "amp": amp,
        "qrs_ms": _f(qrs_ms), "pr_ms": _f(pr_ms), "qt_ms": _f(qt_ms), "rr_ms": _f(rr_ms),
        "rhythm": {"sinus": sinus, "af": af, "paced": paced, "lbbb": lbbb},
    }
