"""Visual check of NeuroKit2 against the LUDB expert marks (open data). Computation/plotting only, no main analysis.
Usage: python -I -B aw231_ludb.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import sys, json, gzip, csv, warnings
from pathlib import Path
import numpy as np
warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk

assert nk.__version__ == "0.2.13"
LUDB = Path(str(data_path("pub12.lobachevsky_university_electrocardiography_database_1_0_1_data")))
OUT = Path(str(data_path("pub12.benchmark_pub12_aw231")))
OUT.mkdir(parents=True, exist_ok=True)
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXT = [l.lower() for l in LEADS]
RECS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
PLOT_LEADS = ["II", "V5"]
FS = 500
MS = 1000.0 / FS


def expert(rec, ext):
    a = wfdb.rdann(str(LUDB / str(rec)), ext)
    out = {"p": [], "N": [], "t": []}
    on = pk = pksym = None
    for s, sym in zip(a.sample, a.symbol):
        if sym == "(":
            on, pk, pksym = int(s), None, None
        elif sym in out and on is not None:
            pk, pksym = int(s), sym
        elif sym == ")":
            if on is not None and pk is not None:
                out[pksym].append((pk, on, int(s)))
            on = pk = pksym = None
    return out


def nk_lead(x, method):
    """Call as in the adapter (nk_lauf.messe_ableitung, VV0), only the method is variable."""
    c = nk.ecg_clean(x, sampling_rate=FS, method="neurokit")
    _, info = nk.ecg_peaks(c, sampling_rate=FS, method="neurokit")
    rp = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if rp.size == 0:
        return c, rp, {}
    _, w = nk.ecg_delineate(c, rp, sampling_rate=FS, method=method, show=False, check=False)
    w = {k: np.asarray(v, dtype=float) for k, v in w.items()}
    for k, a in w.items():
        if a.size != rp.size:
            raise ValueError(f"{k}: length {a.size} instead of {rp.size}")
    return c, rp, w


def stored_beats():
    """Stored beat table of the NeuroKit2 run (VV0, LUDB) for the adapter comparison."""
    d = {}
    with gzip.open(str(data_path("pub12.benchmark_ergebnisse_neurokit2_langformat_neurokit2_schlaege_csv_gz")), "rt", encoding="utf8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["datensatz"] != "ludb" or row["vorverarbeitung"] != "VV0":
                continue
            if int(row["aufnahme"]) not in RECS:
                continue
            d.setdefault((int(row["aufnahme"]), row["ableitung"]), []).append(row)
    return d


def fnum(s):
    return np.nan if s == "" else float(s)


summary = {}
rows_match = []   # DWT per lead, beat
cnt = {m: {"marken_roh": 0, "ableitungen": 0} for m in ("dwt", "cwt", "peak")}
desc = []         # descriptive rows cwt/peak
adapter_diff = []
sb = stored_beats()

for rec in RECS:
    r = wfdb.rdrecord(str(LUDB / str(rec)))
    names = [n.upper() for n in r.sig_name]
    sig = np.asarray(r.p_signal, float)
    ex = {L: expert(rec, L.lower()) for L in LEADS}
    res = {}
    for method in ("dwt", "cwt", "peak"):
        for L in LEADS:
            x = np.ascontiguousarray(sig[:, names.index(L)])
            try:
                c, rp, w = nk_lead(x, method)
            except Exception as e:
                desc.append(dict(rec=rec, lead=L, method=method, fehler=f"{type(e).__name__}: {str(e)[:80]}"))
                continue
            res[(method, L)] = (c, rp, w)
            e = ex[L]
            en = e["N"]
            # description/distance to LUDB
            def near(rpk):
                best = None
                for pk, on, off in en:
                    if abs(pk - rpk) <= 40 and (best is None or abs(pk - rpk) < abs(best[0] - rpk)):
                        best = (pk, on, off)
                return best
            if method == "dwt":
                # adapter comparison against the stored table
                st = sb.get((rec, L), [])
                a = [[int(rp[i])] + [(-1 if not np.isfinite(w[k][i]) else int(w[k][i])) for k in
                      ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets", "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")] for i in range(len(rp))]
                b = [[(-1 if row[k] == "" else int(float(row[k]))) for k in
                      ("qrs_gipfel_schlag_sample", "p_beginn_schlag_sample", "p_gipfel_schlag_sample", "p_ende_schlag_sample", "q_gipfel_schlag_sample", "qrs_beginn_schlag_sample", "qrs_ende_schlag_sample", "s_gipfel_schlag_sample", "t_beginn_schlag_sample", "t_gipfel_schlag_sample", "t_ende_schlag_sample")] for row in st]
                adapter_diff.append(dict(rec=rec, lead=L, n_frisch=len(a), n_gespeichert=len(b), gleich=(a == b)))
            for i, rpk in enumerate(rp):
                nb = near(rpk)
                if nb is None:
                    continue
                pk, on, off = nb
                # preceding P (expert) of this lead
                pp = [t for t in e["p"] if t[0] < pk]
                pexp = pp[-1] if pp else None
                if method in ("dwt", "cwt"):
                    ron = w.get("ECG_R_Onsets", np.full(len(rp), np.nan))[i]
                    roff = w.get("ECG_R_Offsets", np.full(len(rp), np.nan))[i]
                    rows_match.append(dict(rec=rec, lead=L, method=method, r_nk=int(rpk), n_exp=pk, on_exp=on, off_exp=off,
                                           on_nk=ron, off_nk=roff,
                                           q_nk=w.get("ECG_Q_Peaks", np.full(len(rp), np.nan))[i],
                                           s_nk=w.get("ECG_S_Peaks", np.full(len(rp), np.nan))[i],
                                           p_exp_on=(pexp[1] if pexp else np.nan), p_exp_pk=(pexp[0] if pexp else np.nan), p_exp_off=(pexp[2] if pexp else np.nan),
                                           p_nk_pk=w.get("ECG_P_Peaks", np.full(len(rp), np.nan))[i]))
                else:
                    rows_match.append(dict(rec=rec, lead=L, method="peak", r_nk=int(rpk), n_exp=pk, on_exp=on, off_exp=off,
                                           on_nk=np.nan, off_nk=np.nan, q_nk=w["ECG_Q_Peaks"][i], s_nk=w["ECG_S_Peaks"][i],
                                           p_exp_on=np.nan, p_exp_pk=np.nan, p_exp_off=np.nan, p_nk_pk=w["ECG_P_Peaks"][i]))
            cnt[method]["ableitungen"] += 1
            cnt[method]["marken_roh"] += int(sum(np.isfinite(w.get("ECG_R_Onsets", [])) )) if "ECG_R_Onsets" in w else 0
    # figure
    fig, axes = plt.subplots(len(PLOT_LEADS), 1, figsize=(15, 7.5), sharex=True)
    for ax, L in zip(axes, PLOT_LEADS):
        c, rp, w = res[("dwt", L)]
        t = np.arange(len(c)) / FS
        ax.plot(t, c, color="0.2", lw=0.8, label="Signal (ecg_clean)")
        for key, col, mk, lab in (("ECG_R_Onsets", "tab:red", ">", "NK2 DWT R_Onsets"), ("ECG_R_Offsets", "tab:orange", "<", "NK2 DWT R_Offsets"),
                                  ("ECG_Q_Peaks", "tab:purple", "v", "NK2 Q_Peaks"), ("ECG_S_Peaks", "tab:brown", "v", "NK2 S_Peaks")):
            v = w.get(key, np.array([]))
            v = v[np.isfinite(v)].astype(int)
            v = v[(v >= 0) & (v < len(c))]
            ax.plot(v / FS, c[v], mk, color=col, ms=7, ls="none", label=lab)
        for pk, on, off in ex[L]["N"]:
            ax.plot(on / FS, c[on], "o", mfc="none", mec="tab:green", ms=11, mew=1.8, label="LUDB QRS onset/end")
            ax.plot(off / FS, c[off], "s", mfc="none", mec="tab:green", ms=11, mew=1.8)
        for pk, on, off in ex[L]["p"]:
            ax.axvspan(on / FS, off / FS, color="tab:blue", alpha=0.12, lw=0)
        ax.set_ylabel(f"{L} (mV)")
        ax.set_xlim(2.0, 6.0)
        h, l = ax.get_legend_handles_labels()
        d = dict(zip(l, h))
        ax.legend(d.values(), d.keys(), loc="upper right", fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time (s); blue: LUDB P wave")
    fig.suptitle(f"LUDB {rec}: NeuroKit2 0.2.13 (ecg_clean neurokit, ecg_peaks neurokit, ecg_delineate dwt) against LUDB expert marks")
    fig.tight_layout()
    fig.savefig(OUT / f"ludb_{rec:03d}_nk2_dwt_vs_expert.png", dpi=110)
    plt.close(fig)
    print("done", rec, flush=True)

import pandas as pd
df = pd.DataFrame(rows_match)
df.to_csv(OUT / "schlagpaare_ludb10_aw231.csv", index=False)
pd.DataFrame(adapter_diff).to_csv(OUT / "adaptervergleich_ludb10.csv", index=False)
pd.DataFrame(desc).to_csv(OUT / "fehler_ludb10.csv", index=False)
print("Adapter equal:", pd.DataFrame(adapter_diff).gleich.value_counts().to_dict())
