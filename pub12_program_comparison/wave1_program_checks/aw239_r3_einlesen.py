# -*- coding: utf-8 -*-
"""aw239_r3_einlesen.py (AW-239, P12-R3, Version 1.0): liest die oeffentlichen ECGDeli-Fiducialdateien von PTB-XL+ 1.0.1 (WFDB-Annotationen, 500 Hz) fuer alle Aufnahmen
und legt je Aufnahme die Punkte je Schlag und Ableitung in einem Feld ab (npz). Eigener Leser fuer das WFDB-Annotationsformat (Test gegen wfdb in aw239_test_r3.py).

Dateien je Aufnahme: <id:05d>_points_global.atr und <id:05d>_points_lead_<X>.atr (X = I, II, III, aVR, aVL, aVF, V1 bis V6) unter fiducial_points/ecgdeli/<id//1000*1000:05d>/.
Annotationen: aux_note = Name des Punktes (p-wave onset/peak/offset, QRS onset, Q peak, R peak, S peak, QRS offset, L point (for STEMI), t-wave onset/peak/offset).
Zuordnung der Punkte zu Schlaegen (eigene Regel, wie in AW-176 begruendet): Schlaege = R-Gipfel der globalen Datei; je Ableitung gehoert ein R-Gipfel der Ableitungsdatei zum
globalen Schlag, wenn er hoechstens 75 Abtastwerte entfernt ist (naechster). Q-Gipfel und QRS-Beginn: letzter Punkt dieses Namens vor dem Ableitungs-R (hoechstens 150 Abtastwerte
davor); S-Gipfel, QRS-Ende, T-Ende: erster Punkt dieses Namens nach dem Ableitungs-R (S und QRS-Ende hoechstens 150, T-Ende hoechstens 1000 Abtastwerte danach und vor dem
naechsten R-Gipfel). Punkte mit Zeit unter 5 Abtastwerten gelten als fehlend (ECGDeli setzt fehlende Punkte auf NaN, im Export stehen sie am Dateianfang).
Ausgabe: <local path>  (Felder: ids, offsets, R, on, off, Q, S, Toff mit Form (Schlaege gesamt, 12); -1 = fehlt; Gon, Goff, GToff (Schlaege)).
Aufruf: python -I -B aw239_r3_einlesen.py [--n MAX] [--basis PFAD] [--aus PFAD]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
BASIS = Path(str(data_path("pub12.benchmark_pub12_aw239_fid_fiducial_points_ecgdeli")))
AUS = Path(str(data_path("pub12.benchmark_pub12_aw239_r3_ecgdeli_punkte_npz")))
NAMEN = {"QRS onset": "on", "QRS offset": "off", "Q peak": "Q", "R peak": "R", "S peak": "S", "t-wave offset": "Toff"}


def parse_atr(b):
    """WFDB-Annotationsdatei (Bytes) -> Liste (Zeit, Code, Aux-Text). SKIP/NUM/SUB/CHN werden behandelt, AUX haengt an die vorige Annotation."""
    pos, t, out = 0, 0, []
    n = len(b)
    while pos + 2 <= n:
        w = b[pos] | (b[pos + 1] << 8)
        pos += 2
        if w == 0:
            break
        code, delta = w >> 10, w & 1023
        if code == 59:  # SKIP: 32-Bit-Zeitzuwachs (hohes Wort zuerst)
            hi = b[pos] | (b[pos + 1] << 8)
            lo = b[pos + 2] | (b[pos + 3] << 8)
            pos += 4
            inc = (hi << 16) | lo
            if inc >= 1 << 31:
                inc -= 1 << 32
            t += inc
            continue
        if code in (60, 61, 62):
            continue
        if code == 63:
            aux = bytes(b[pos:pos + delta])
            pos += delta + (delta & 1)
            if out:
                out[-1][2] = aux.decode("latin-1")
            continue
        t += delta
        out.append([t, code, ""])
    return [(t_, c_, a_) for t_, c_, a_ in out]


def lese_datei(p):
    return parse_atr(Path(p).read_bytes())


def punkte(ann):
    """Liste (Zeit, Aux) -> dict Kurzname -> sortiertes int-Array der Zeiten (Zeit >= 5)."""
    d = {k: [] for k in NAMEN.values()}
    for t, c, a in ann:
        k = NAMEN.get(a.strip("\x00").strip())
        if k is not None and t >= 5:
            d[k].append(t)
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in d.items()}


def _vor(arr, ref, maxd):
    """letzter Wert von arr, der <= ref ist und hoechstens maxd davor liegt; sonst -1."""
    if len(arr) == 0:
        return -1
    i = np.searchsorted(arr, ref, side="right") - 1
    if i >= 0 and ref - arr[i] <= maxd:
        return int(arr[i])
    return -1


def _nach(arr, ref, maxd, grenze=None):
    """erster Wert von arr, der >= ref ist und hoechstens maxd danach (und < grenze) liegt; sonst -1."""
    if len(arr) == 0:
        return -1
    i = np.searchsorted(arr, ref, side="left")
    if i < len(arr) and arr[i] - ref <= maxd and (grenze is None or arr[i] < grenze):
        return int(arr[i])
    return -1


def recording(basis, ecg):
    """Rueckgabe dict mit R (nb), Matrizen (nb, 12) on, off, Q, S, Toff, R_lead und globalen Gon, Goff, GToff (nb); None, wenn Dateien fehlen."""
    ordner = Path(basis) / ("%05d" % (ecg // 1000 * 1000))
    pg = ordner / ("%05d_points_global.atr" % ecg)
    if not pg.exists():
        return None
    g = punkte(lese_datei(pg))
    Rg = g["R"]
    nb = len(Rg)
    res = {"R": Rg}
    for k in ("on", "off", "Q", "S", "Toff"):
        res[k] = np.full((nb, 12), -1, dtype=np.int64)
    res["Rl"] = np.full((nb, 12), -1, dtype=np.int64)
    res["Gon"] = np.array([_vor(g["on"], r, 150) for r in Rg], dtype=np.int64)
    res["Goff"] = np.array([_nach(g["off"], r, 150) for r in Rg], dtype=np.int64)
    res["GToff"] = np.array([_nach(g["Toff"], r, 1000) for r in Rg], dtype=np.int64)
    for j, lead in enumerate(LEADS):
        pl = ordner / ("%05d_points_lead_%s.atr" % (ecg, lead))
        if not pl.exists():
            return None
        d = punkte(lese_datei(pl))
        Rl = d["R"]
        if len(Rl) == 0:
            continue
        for b, r in enumerate(Rg):
            i = int(np.argmin(np.abs(Rl - r)))
            if abs(Rl[i] - r) > 75:
                continue
            rl = int(Rl[i])
            nxt = int(Rl[i + 1]) if i + 1 < len(Rl) else None
            res["Rl"][b, j] = rl
            res["on"][b, j] = _vor(d["on"], rl, 150)
            res["Q"][b, j] = _vor(d["Q"], rl, 150)
            res["off"][b, j] = _nach(d["off"], rl, 150)
            res["S"][b, j] = _nach(d["S"], rl, 150)
            res["Toff"][b, j] = _nach(d["Toff"], rl, 1000, nxt)
    return res


def haupt(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--basis", default=str(BASIS))
    ap.add_argument("--aus", default=str(AUS))
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    basis = Path(a.basis)
    if a.ids:
        ids = sorted(int(x) for x in Path(a.ids).read_text().split())
    else:
        ids = sorted(int(p.name.split("_")[0]) for o in basis.iterdir() if o.is_dir() for p in o.glob("*_points_global.atr"))
    if a.n:
        ids = ids[: a.n]
    print("Aufnahmen:", len(ids), flush=True)
    t0 = time.time()
    felder = {k: [] for k in ("R", "Gon", "Goff", "GToff")}
    mats = {k: [] for k in ("on", "off", "Q", "S", "Toff", "Rl")}
    offs, ok_ids, fehlend = [0], [], []
    for i, e in enumerate(ids):
        r = recording(basis, e)
        if r is None:
            fehlend.append(e)
            continue
        ok_ids.append(e)
        for k in felder:
            felder[k].append(r[k])
        for k in mats:
            mats[k].append(r[k])
        offs.append(offs[-1] + len(r["R"]))
        if (i + 1) % 2000 == 0:
            print(i + 1, "%.0f s" % (time.time() - t0), flush=True)
    out = {k: np.concatenate(v).astype(np.int32) for k, v in felder.items()}
    out.update({k: np.concatenate(v, axis=0).astype(np.int32) for k, v in mats.items()})
    out["ids"] = np.array(ok_ids, dtype=np.int64)
    out["offsets"] = np.array(offs, dtype=np.int64)
    Path(a.aus).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.aus, **out)
    print("fertig: %d Aufnahmen, %d Schlaege, fehlend %d, %.0f s" % (len(ok_ids), offs[-1], len(fehlend), time.time() - t0), flush=True)
    if fehlend:
        Path(a.aus).with_suffix(".fehlend.txt").write_text("\n".join(map(str, fehlend)))
    return 0


if __name__ == "__main__":
    sys.exit(haupt())
