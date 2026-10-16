# -*- coding: utf-8 -*-
"""aw245_r4.py (version 1.0): signal test of the aVL/aVF labelling in PTB-XL+ (ECGDeli), all recordings with published fiducial files (program comparison study, exploratory, post hoc).

Inputs (read only, open data, no locked tables): ECGDeli points from the preceding ECGDeli stage (<local path>, 21,794 recordings, hash is recorded),
PTB-XL records500 (local), ecgdeli_features.csv (PTB-XL+ 1.0.1, hash against the expected value).
Part A: duration per lead from the annotation file (median of the beat durations, times 2 ms) against the feature column QRS_Dur_<lead> under its own name and under the exchanged name (aVL, aVF)
        for all recordings (previously 100).
Part B: signal test. M[l, y] = share of beats in which the dominant deflection (largest absolute deviation from the window median, window +-15 samples around the annotated
        R peak of the annotation lead l) of the signal y lies at most 1 sample from the annotated R peak. Decision quantity
        D = (M[aVL,aVL] + M[aVF,aVF]) - (M[aVL,aVF] + M[aVF,aVL]); positive: the annotation files fit the signals of the same name. Null threshold tau = 97.5th percentile of |D| when
        combining the annotations of one recording with the signal of the previous recording (no information). Reference: the same statistic for all other 65 lead pairs.
The metric was chosen on the first 400 recordings after a first metric (local extrema, calibrated on the unsuspected leads: own signal best in only 17%) did not hold up;
the aVL/aVF value of these 400 recordings was seen in the process (development look). After the choice (window +-15, tolerance 1) no parameter was changed.
Usage: python -I -B aw245_r4.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw245_r4_lib as L  # noqa: E402

NPZ = Path(str(data_path("pub12.benchmark_pub12_aw239_r3_ecgdeli_punkte_npz")))
PTBXL = Path(str(data_path("pub12.samples_ptb_x")))
FEAT = Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features_ecgdeli_features_csv")))
FEAT_HASH = "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556"
OUT_ROH = Path(str(data_path("pub12.benchmark_pub12_aw245")))
OUT_PUB = Path(str(data_path("pub12.publikationen_pub_12_etablierung_ergebnisse_aw245")))
H_FENSTER = 15
TOL = 1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    z = time.strftime("%H:%M:%S") + " " + str(m)
    print(z, flush=True)
    with open(OUT_ROH / "r4_lauf.log", "a", encoding="utf-8") as f:
        f.write(z + "\n")


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def pfad(e):
    return PTBXL / "records500" / ("%05d" % (e // 1000 * 1000)) / ("%05d_hr" % e)


def lade_punkte(n=0):
    z = np.load(NPZ)
    d = {k: z[k] for k in ("Rl", "Q", "S", "on", "off", "ids", "offsets")}  # unpack once (npz otherwise unpacks on every access)
    ids, offs = d["ids"], d["offsets"]
    if n:
        ids = ids[:n]
    return d, ids, offs


def matrizen(d, ids, offs, nullpaar=True):
    """M (n, 12, 12) and Mn (annotation of recording i against the signal of recording i-1), count of valid beats CN (n, 12)."""
    n = len(ids)
    M = np.full((n, 12, 12), np.nan, dtype=np.float32)
    Mn = np.full((n, 12, 12), np.nan, dtype=np.float32)
    CN = np.zeros((n, 12), dtype=np.int32)
    prev = None
    t0 = time.time()
    for i, e in enumerate(ids):
        a, b = int(offs[i]), int(offs[i + 1])
        R = d["Rl"][a:b]
        x = L.lies_dat(pfad(int(e)))
        M[i], CN[i] = L.dominante_passung(x, R, H_FENSTER, TOL)
        if nullpaar and prev is not None:
            Mn[i], _ = L.dominante_passung(prev, R, H_FENSTER, TOL)
        prev = x
        if (i + 1) % 2000 == 0:
            log("Signals %d of %d (%.0f s)" % (i + 1, n, time.time() - t0))
    return M, CN, Mn


def teil_a(d, ids, offs):
    """Duration per lead from the annotation (median over beats of the difference off-on, times 2 ms) against the feature columns."""
    feat = pd.read_csv(FEAT, usecols=["ecg_id"] + ["QRS_Dur_" + l for l in L.LEADS]).drop_duplicates("ecg_id", keep="first").set_index("ecg_id").reindex(ids)
    dur = np.full((len(ids), 12), np.nan)
    on, off = d["on"], d["off"]
    for i in range(len(ids)):
        a, b = int(offs[i]), int(offs[i + 1])
        o1, o2 = on[a:b], off[a:b]
        for l in range(12):
            ok = (o1[:, l] >= 0) & (o2[:, l] >= 0)
            if ok.any():
                dur[i, l] = np.median((o2[ok, l] - o1[ok, l]) * 2.0)
    F = np.stack([feat["QRS_Dur_" + l].to_numpy(float) for l in L.LEADS], axis=1)
    rows = []
    for l, name in enumerate(L.LEADS):
        own = np.abs(dur[:, l] - F[:, l])
        ok = np.isfinite(dur[:, l]) & np.isfinite(F[:, l])
        r = dict(ableitung=name, n=int(ok.sum()), eigen_exakt=int((own[ok] <= 0.5).sum()), eigen_innerhalb_2ms=int((own[ok] <= 2.0).sum()))
        if l in (L.I_AVL, L.I_AVF):
            other = L.I_AVF if l == L.I_AVL else L.I_AVL
            sw = np.abs(dur[:, other] - F[:, l])
            ok2 = np.isfinite(dur[:, other]) & np.isfinite(F[:, l])
            r.update(vertauscht_exakt=int((sw[ok2] <= 0.5).sum()), vertauscht_innerhalb_2ms=int((sw[ok2] <= 2.0).sum()), n_vertauscht=int(ok2.sum()),
                     beide_exakt=int(((own <= 0.5) & (sw <= 0.5))[ok & ok2].sum()))
        rows.append(r)
    return pd.DataFrame(rows), dur, F


def lauf():
    OUT_ROH.mkdir(parents=True, exist_ok=True)
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hashes = {"npz": sha(NPZ), "features": sha(FEAT)}
    if hashes["features"] != FEAT_HASH:
        raise SystemExit("ecgdeli_features.csv differs")
    log("Hashes " + json.dumps(hashes))
    d, ids, offs = lade_punkte()
    n = len(ids)
    log("Recordings %d, beats %d" % (n, offs[-1]))
    dfa, dur, F = teil_a(d, ids, offs)
    dfa.to_csv(OUT_PUB / "r4_teil_a_dauer_gegen_features.csv", index=False)
    log("Part A\n" + dfa.to_string())
    M, CN, Mn = matrizen(d, ids, offs)
    np.savez_compressed(OUT_ROH / "r4_matrizen.npz", ids=ids, M=M, CN=CN, Mn=Mn)
    ok_n = (CN[:, L.I_AVL] >= 3) & (CN[:, L.I_AVF] >= 3)
    Mf, Mnf = M.astype(float), Mn[1:].astype(float)
    Dn = np.array([L.d_swap(m) for m in Mnf])
    Dn = Dn[np.isfinite(Dn)]
    tau = float(np.percentile(np.abs(Dn), 97.5))
    D = np.array([L.d_swap(m) for m in Mf])
    cls = np.where(ok_n, L.klassifiziere(D, tau), 3)
    namen = {0: "wie benannt (Annotation aVL passt zu Signal aVL)", 1: "vertauscht (Annotation aVL passt zu Signal aVF)", 2: "unklar (|D| <= Nullschwelle)", 3: "keine ausreichenden Daten"}
    rows = []
    for c in range(4):
        k = int((cls == c).sum())
        lo, hi = wilson(k, n)
        rows.append(dict(klasse=namen[c], n=k, anteil_pct=100 * k / n, ki_unten_pct=100 * lo, ki_oben_pct=100 * hi))
    k_wb, k_vt = int((cls == 0).sum()), int((cls == 1).sum())
    k_ent = k_wb + k_vt
    for nm, k in (("unter den Entschiedenen: wie benannt", k_wb), ("unter den Entschiedenen: vertauscht", k_vt)):
        lo, hi = wilson(k, k_ent)
        rows.append(dict(klasse=nm, n=k, anteil_pct=100 * k / k_ent if k_ent else np.nan, ki_unten_pct=100 * lo, ki_oben_pct=100 * hi))
    dfc = pd.DataFrame(rows)
    dfc.to_csv(OUT_PUB / "r4_teil_b_klassen.csv", index=False, float_format="%.4g")
    q = [0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99]
    dist = pd.DataFrame({"quantil": q, "D_aVL_aVF": np.nanquantile(D, q), "D_null_andere_aufnahme": np.quantile(Dn, q)})
    dist.to_csv(OUT_PUB / "r4_teil_b_verteilung_D.csv", index=False, float_format="%.4g")
    rows = []
    for a_, b_ in itertools.combinations(range(12), 2):
        Dc = np.array([L.d_swap(m, a_, b_) for m in Mf])
        Dcn = np.array([L.d_swap(m, a_, b_) for m in Mnf])
        Dcn = Dcn[np.isfinite(Dcn)]
        tc = float(np.percentile(np.abs(Dcn), 97.5))
        okc = np.isfinite(Dc)
        rows.append(dict(paar="%s/%s" % (L.LEADS[a_], L.LEADS[b_]), n=int(okc.sum()), D_median=float(np.median(Dc[okc])), D_positiv_pct=100 * float(np.mean(Dc[okc] > 0)),
                         tau_null=tc, wie_benannt_pct=100 * float(np.mean(Dc[okc] > tc)), vertauscht_pct=100 * float(np.mean(Dc[okc] < -tc)), unklar_pct=100 * float(np.mean(np.abs(Dc[okc]) <= tc))))
    dfk = pd.DataFrame(rows).sort_values("D_median")
    dfk.to_csv(OUT_PUB / "r4_alle_paare.csv", index=False, float_format="%.4g")
    zel = pd.DataFrame({"M_aVLann_aVLsig": Mf[:, 4, 4], "M_aVFann_aVFsig": Mf[:, 5, 5], "M_aVLann_aVFsig": Mf[:, 4, 5], "M_aVFann_aVLsig": Mf[:, 5, 4]}).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    zel.to_csv(OUT_PUB / "r4_passung_vier_zellen.csv", float_format="%.4g")
    a_sw = (np.abs(dur[:, L.I_AVF] - F[:, L.I_AVL]) <= 0.5) & (np.abs(dur[:, L.I_AVL] - F[:, L.I_AVL]) > 0.5)
    info = dict(version="1.0", H=H_FENSTER, tol=TOL, tau=tau, aufnahmen=n, laufzeit_s=round(time.time() - t0, 1), hashes=hashes, null_n=int(len(Dn)), D_median=float(np.nanmedian(D)),
                feature_aVL_gleich_aVF_annotation_nicht_eigene=int(a_sw.sum()), davon_signal_vertauscht=int((a_sw & (cls == 1)).sum()), davon_wie_benannt=int((a_sw & (cls == 0)).sum()),
                davon_unklar=int((a_sw & (cls == 2)).sum()))
    (OUT_PUB / "r4_LAUF.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    log(json.dumps(info, indent=1, default=str))
    log("Classes\n" + dfc.round(2).to_string())
    log("Pairs (first and last)\n" + pd.concat([dfk.head(6), dfk.tail(4)]).round(2).to_string())
    log("Distribution of D\n" + dist.round(3).to_string())


if __name__ == "__main__":
    lauf()
