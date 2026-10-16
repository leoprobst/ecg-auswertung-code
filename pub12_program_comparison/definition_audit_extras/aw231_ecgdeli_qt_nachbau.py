"""Rebuilds the ECGDeli QT in PTB-XL+ from the fiducial files (fixed sample of 100 ECGs).
Open data (PTB-XL+ 1.0.1). Outputs only to a local output folder. Usage: python -I -B aw231_qt.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
warnings.simplefilter("ignore")

BASE = Path(str(data_path("pub12.ptb_xl_plus_1_0_1_features")))
FID = Path(str(data_path("pub12.aw176_fiducial_points_ecgdeli")))
OUT = Path(str(data_path("pub12.benchmark_pub12_aw231")))
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ORDER = ["p-wave onset", "p-wave peak", "p-wave offset", "QRS onset", "Q peak", "R peak", "S peak", "QRS offset",
         "L point (for STEMI)", "t-wave onset", "t-wave peak", "t-wave offset"]
IDX = {n: i for i, n in enumerate(ORDER)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fpt(ecg, lead):
    """Beats x 12 matrix of the fiducial points (sample at 500 Hz), NaN where missing. New row when the order jumps back."""
    stem = FID / f"{(ecg // 1000) * 1000:05d}" / f"{ecg:05d}_points_{'global' if lead == 'global' else 'lead_' + lead}"
    a = wfdb.rdann(str(stem), "atr")
    names = [n.strip(chr(0)) for n in a.aux_note]
    t = np.asarray(a.sample, dtype=float)
    ok = t >= 5                                   # FPT < 5 counts as missing (MATLAB: FPT_mat(FPT_mat<5)=NaN)
    R = np.sort(t[[i for i, n in enumerate(names) if n == "R peak" and ok[i]]])
    mat = np.full((len(R), 12), np.nan)
    doppelt = 0
    BEFORE = {"p-wave onset", "p-wave peak", "p-wave offset", "QRS onset", "Q peak"}
    for i, n in enumerate(names):
        if n not in IDX or not ok[i]:
            continue
        if n == "R peak":
            k = int(np.searchsorted(R, t[i]))
        elif n in BEFORE:
            k = int(np.searchsorted(R, t[i], side="left"))      # next R peak after
            if k >= len(R):
                continue
        else:
            k = int(np.searchsorted(R, t[i], side="right")) - 1  # last R peak before
            if k < 0:
                continue
        if np.isfinite(mat[k, IDX[n]]):
            doppelt += 1
        mat[k, IDX[n]] = t[i]
    rows = list(mat)
    if doppelt:
        print("Double assignment", stem.name, doppelt, flush=True)
    return np.array(rows), len(a.sample)


def nanrank(a, k, largest):
    """k-th extreme value per row across leads, NaN omitted (MATLAB maxk/mink omitnan); NaN if fewer than k values."""
    out = np.full(a.shape[0], np.nan)
    for i, r in enumerate(a):
        v = np.sort(r[np.isfinite(r)])
        if len(v) >= k:
            out[i] = v[-k] if largest else v[k - 1]
    return out


def main():
    hashes = {p.name: sha(p) for p in (BASE / "ecgdeli_features.csv", BASE / "12sl_features.csv", BASE / "unig_features.csv")}
    soll = {"ecgdeli_features.csv": "84143735682f727201cc7f2825c047f98898f9756c6a21729af993b516221556",
            "12sl_features.csv": "cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29",
            "unig_features.csv": "f95ab01df5c9ed329485fd811578f0c39eabb0667a36ec37c7dda427d9d8f5a5"}
    assert hashes == soll, hashes
    smp = pd.read_csv(str(data_path("pub12.aw176_sample_csv")))
    assert sha(str(data_path("pub12.aw176_sample_csv"))) == "ca7db4b2da1f1c10554e6eefa48fbeb19ad6af896d7d3e1fe1c588eb9152de81"
    ids = smp.ecg_id.astype(int).tolist()
    ed = pd.read_csv(BASE / "ecgdeli_features.csv").set_index("ecg_id")
    s12 = pd.read_csv(BASE / "12sl_features.csv", usecols=["ecg_id", "QRS_On_Global", "T_Off_Global", "QT_Int_Global", "QT_IntBazett_Global", "RR_Mean_Global", "HR__Global"]).set_index("ecg_id")
    ug = pd.read_csv(BASE / "unig_features.csv", usecols=["ecg_id", "QRS_On_Global", "T_Off_Global", "QT_Int_Global", "QT_IntCorr_Global", "RR_Mean_Global", "HR__Global"]).set_index("ecg_id")
    rows, lrows = [], []
    for e in ids:
        mats, nb = {}, {}
        for L in LEADS + ["global"]:
            mats[L], nb[L] = fpt(e, L)
        nbeats = {L: len(mats[L]) for L in LEADS + ["global"]}
        same = len(set(nbeats[L] for L in LEADS)) == 1
        # Beat assignment row by row as in the MATLAB table (FPT_Cell: B x 12 x L), otherwise warning
        B = min(nbeats[L] for L in LEADS)
        A = np.stack([mats[L][:B] for L in LEADS], axis=2)  # B x 12 x 12
        A[A < 5] = np.nan
        on = A[:, 3, :]      # column 4 (QRS onset), per lead
        toff = A[:, 11, :]   # column 12 (T offset)
        qrsoff = A[:, 7, :]
        # QT per lead and beat, median over beats
        qt_lead = 2 * (toff - on)
        qt_lead_med = np.nanmedian(qt_lead, axis=0)
        # global variants
        v_code = 2 * (nanrank(toff, 2, True) - nanrank(on, 2, False))      # code: second-latest T end minus second-earliest onset
        v_minmax = 2 * (np.nanmax(toff, axis=1) - np.nanmin(on, axis=1))
        G = mats["global"]
        Gm = G.copy(); Gm[Gm < 5] = np.nan
        v_cons = 2 * (Gm[:, 11] - Gm[:, 3])
        # Framingham per beat with RR of the synchronised R points (last value repeated)
        rr = 2 * np.diff(Gm[:, 5]); rr = np.append(rr, rr[-1]) if len(rr) else rr
        qtc = (v_code + 154 * (1 - 0.001 * rr)) if len(rr) == len(v_code) else np.full(len(v_code), np.nan)
        feat = ed.loc[e]
        r = dict(ecg_id=e, stratum=smp.set_index("ecg_id").loc[e, "stratum"], beats_global=nbeats["global"], beats_min_lead=B,
                 beats_gleich_alle_ableitungen=same, feature_QT_global=feat["QT_Int_Global"], feature_QTc_fram=feat["QT_IntFramingham_Global"],
                 feature_RR=feat["RR_Mean_Global"],
                 qt_code_median=float(np.nanmedian(v_code)), qt_minmax_median=float(np.nanmedian(v_minmax)), qt_consensus_median=float(np.nanmedian(v_cons)),
                 qt_max_der_leadmediane=float(np.nanmax(qt_lead_med)), qtc_fram_code=float(np.nanmedian(qtc)), rr_median_global=float(np.nanmedian(rr)),
                 on_rank2_med=float(np.nanmedian(nanrank(on, 2, False)) * 2), toff_rank2_med=float(np.nanmedian(nanrank(toff, 2, True)) * 2),
                 on_global_med=float(np.nanmedian(Gm[:, 3]) * 2), toff_global_med=float(np.nanmedian(Gm[:, 11]) * 2),
                 q_minus_on_med=float(np.nanmedian(2 * (A[:, 4, :] - A[:, 3, :]))),
                 tpeak_to_toff_med=float(np.nanmedian(2 * (A[:, 11, :] - A[:, 10, :]))), rpk_to_toff_med=float(np.nanmedian(2 * (A[:, 11, :] - A[:, 5, :]))),
                 qt12=s12.loc[e, "QT_Int_Global"] if e in s12.index else np.nan, qtug=ug.loc[e, "QT_Int_Global"] if e in ug.index else np.nan,
                 on12=s12.loc[e, "QRS_On_Global"] if e in s12.index else np.nan, toff12=s12.loc[e, "T_Off_Global"] if e in s12.index else np.nan,
                 onug=ug.loc[e, "QRS_On_Global"] if e in ug.index else np.nan, toffug=ug.loc[e, "T_Off_Global"] if e in ug.index else np.nan,
                 rr12=s12.loc[e, "RR_Mean_Global"] if e in s12.index else np.nan, rrug=ug.loc[e, "RR_Mean_Global"] if e in ug.index else np.nan)
        rows.append(r)
        for j, L in enumerate(LEADS):
            lrows.append(dict(ecg_id=e, lead=L, rebuilt=float(qt_lead_med[j]), feature=feat.get("QT_Int_" + L, np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ecgdeli_qt_nachbau_aw231.csv", index=False)
    pd.DataFrame(lrows).to_csv(OUT / "ecgdeli_qt_ableitungen_aw231.csv", index=False)
    print("done", len(df))


main()
