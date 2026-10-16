"""NeuroKit2 run for the benchmark (S_G, LUDB, PT). Computation only, no evaluation.

NeuroKit2 0.2.13, time quantities only: per lead R waves (ecg_peaks) and beat marks (ecg_delineate, method="dwt").
Call parameters (bound, with hash in the checkpoint and in the long format):
  VV0 (native)   : clean = ecg_clean(x, sampling_rate=fs, method="neurokit"); peaks = ecg_peaks(clean, fs, method="neurokit");
                   ecg_delineate(clean, peaks, fs, method="dwt", show=False, check=False)
  VV0_G0.5/G1/G2 : as VV0 on the signal multiplied by 0.5 / 1 / 2 (mV, after reading, before any tool stage)
  VV1            : raw signal in the peak and delineation call (no ecg_clean), see the pre-registered protocol
  VV2            : VV1 plus external band-pass 0.5 to 40 Hz (bench_arme.bandpass_vv2), see the pre-registered protocol
All other parameters are the defaults of 0.2.13 (ecg_clean: 0.5 Hz high-pass order 5 and 50 Hz powerline filter).
Independent per lead (ECG_R_Peaks per lead, as in the protocol). An error in one lead is a lead error, the
record stays "ok" as long as at least one lead delivers beat marks; lead errors are counted separately.

Tasks (key datensatz|aufnahme|vv). Checkpoint: one line per finished task in <out>/roh/<run>.jsonl;
resumption skips finished keys.

Usage (always run with -I -B):
  nk_lauf.py --lauf haupt --workers 2 --out <output folder>
  nk_lauf.py --lauf rauch --workers 2 --out ...      (PT, LUDB 1 to 5: all identifiers)
  nk_lauf.py --lauf pt --lauf-id <run id> ...         (PT, all identifiers; repeat run for the control)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import numpy as np  # noqa: E402

import bench_arme as A  # noqa: E402

LEADS = A.LEADS
MAX_ALTER_S = 900.0
WELLEN = ("ECG_P_Onsets", "ECG_P_Peaks", "ECG_P_Offsets", "ECG_Q_Peaks", "ECG_R_Onsets", "ECG_R_Offsets",
          "ECG_S_Peaks", "ECG_T_Onsets", "ECG_T_Peaks", "ECG_T_Offsets")
SPALTEN_SCHLAG = ("ECG_R_Peaks",) + WELLEN

AUFRUF = {
    "paket": "neurokit2 0.2.13",
    "VV0": {"ecg_clean": {"method": "neurokit"}, "ecg_peaks": {"method": "neurokit"},
            "ecg_delineate": {"method": "dwt", "show": False, "check": False}, "eingang": "ecg_clean(Ableitung)"},
    "VV1": {"ecg_peaks": {"method": "neurokit"}, "ecg_delineate": {"method": "dwt", "show": False, "check": False},
            "eingang": "Rohsignal"},
    "VV2": {"ecg_peaks": {"method": "neurokit"}, "ecg_delineate": {"method": "dwt", "show": False, "check": False},
            "eingang": "Rohsignal nach Bandpass", "bandpass": A.BP_BESCHREIBUNG},
    "faktoren": {"VV0_G0.5": 0.5, "VV0_G1": 1.0, "VV0_G2": 2.0},
    "abtastrate": "aus Datei (500 Hz)",
}
AUFRUF_HASH = hashlib.sha256(json.dumps(AUFRUF, sort_keys=True).encode()).hexdigest()


class _Mem(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (k, ctypes.c_ulonglong) for k in ("totalPhys", "availPhys", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExtended")
    ]


def frei_gb():
    m = _Mem()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.availPhys, m.availPage) / 1024 ** 3


# ---------------------------------------------------------------- Worker
_W = {}


def init_worker():
    import neurokit2 as nk
    import psutil
    import wfdb
    warnings.simplefilter("ignore")
    if nk.__version__ != "0.2.13":
        raise RuntimeError("NeuroKit2 version " + nk.__version__)
    _W.update(nk=nk, wfdb=wfdb, proc=psutil.Process())


def lade(ds, rec):
    shas = A.pruefe_quelle(ds, rec)
    base, _ = A.basis_pfad(ds, rec)
    r = _W["wfdb"].rdrecord(str(base))
    names = [str(x).upper() for x in r.sig_name]
    if sorted(names) != sorted(LEADS):
        raise ValueError(f"Leads {names}")
    units = sorted({str(u).lower() for u in r.units})
    if units != ["mv"]:
        raise ValueError(f"Units {units}")
    fs = int(round(r.fs))
    sig = np.asarray(r.p_signal, dtype=float)[:, [names.index(n) for n in LEADS]]
    if not np.isfinite(sig).all():
        raise ValueError("non-finite values in the signal")
    return sig, fs, shas


def messe_ableitung(x, fs, vv):
    nk = _W["nk"]
    if vv in ("VV0", "VV0_G0.5", "VV0_G1", "VV0_G2"):
        x = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(x, sampling_rate=fs, method="neurokit")
    rp = np.asarray(info["ECG_R_Peaks"], dtype=int)
    if rp.size == 0:
        return rp, {k: np.zeros(0) for k in WELLEN}
    _, waves = nk.ecg_delineate(x, rp, sampling_rate=fs, method="dwt", show=False, check=False)
    out = {}
    for k in WELLEN:
        v = waves.get(k)
        a = np.full(rp.size, np.nan) if v is None else np.asarray(v, dtype=float)
        if a.size != rp.size:
            raise ValueError(f"{k}: length {a.size} instead of {rp.size}")
        out[k] = a
    return rp, out


def work(task):
    ds, rec, vv = task
    t0 = time.monotonic()
    key = f"{ds}|{rec}|{vv}"
    row = {"key": key, "datensatz": ds, "aufnahme": rec, "vv": vv}
    try:
        sig, fs, shas = lade(ds, rec)
    except Exception as exc:
        row.update(status="nicht_ladbar", grund=f"{type(exc).__name__}: {exc}"[:200], sekunden=round(time.monotonic() - t0, 2))
        return row
    row["fs"] = fs
    row["hash16"] = shas
    row["spannweite_mv"] = [round(float(x), 6) for x in (sig.max(axis=0) - sig.min(axis=0))]
    try:
        sig = A.vorverarbeite(sig, fs, vv if vv != "VV0" else "VV0")
        leads, fehler = {}, {}
        for j, name in enumerate(LEADS):
            try:
                rp, w = messe_ableitung(np.ascontiguousarray(sig[:, j]), fs, vv)
                leads[name] = [[int(rp[i])] + [(-1 if not np.isfinite(w[k][i]) else int(w[k][i])) for k in WELLEN] for i in range(rp.size)]
            except Exception as exc:
                fehler[name] = f"{type(exc).__name__}: {exc}"[:160]
        row["leads"] = leads
        row["lead_fehler"] = fehler
        row["n_rpeaks"] = {k: len(v) for k, v in leads.items()}
        row["status"] = "ok" if any(len(v) > 0 for v in leads.values()) else "werkzeugfehler"
        if row["status"] != "ok":
            row["grund"] = "no lead with R waves" + (": " + next(iter(fehler.values())) if fehler else "")
    except Exception as exc:
        row.update(status="werkzeugfehler", grund=f"{type(exc).__name__}: {exc}"[:200])
    row["sekunden"] = round(time.monotonic() - t0, 2)
    row["rss_mb"] = round(_W["proc"].memory_info().rss / 2 ** 20)
    return row


# ---------------------------------------------------------------- task list
VV_REIHE = ("VV0", "VV0_G1", "VV0_G0.5", "VV0_G2", "VV1", "VV2")


def aufgaben(lauf):
    ludb = [str(i) for i in range(1, 201)]
    if lauf == "rauch":
        pt = A.lies_liste("PT")[:10]
        t = [("ptbxl", r, "VV0") for r in pt] + [("ludb", str(i), "VV0") for i in range(1, 4)]
        t += [("ptbxl", pt[0], v) for v in VV_REIHE[1:]]
        return t
    if lauf == "pt":
        pt = A.lies_liste("PT")
        return [("ptbxl", r, v) for v in VV_REIHE for r in pt] + [("ludb", str(i), v) for v in VV_REIHE for i in range(1, 6)]
    sg = A.lies_liste("S_G")
    t = []
    for v in VV_REIHE:
        t += [("ludb", r, v) for r in ludb]
        t += [("ptbxl", r, v) for r in sg]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lauf", choices=["haupt", "rauch", "pt"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(data_path("pub12.benchmark_ergebnisse_neurokit2")))
    ap.add_argument("--lauf-id", default=None)
    ap.add_argument("--min-frei-gb", type=float, default=1.5)
    args = ap.parse_args()
    A.pruefe_listen()
    out = Path(args.out)
    roh = out / "roh"
    roh.mkdir(parents=True, exist_ok=True)
    lauf_id = args.lauf_id or {"haupt": "AW220_nk_haupt", "rauch": "AW220_nk_rauch", "pt": "AW220_nk_pt"}[args.lauf]
    cp = roh / f"{lauf_id}.jsonl"
    alle = aufgaben(args.lauf)
    keyf = lambda t: f"{t[0]}|{t[1]}|{t[2]}"
    fertig = set()
    if cp.exists():
        for line in cp.read_text(encoding="utf8").splitlines():
            if line.strip():
                try:
                    fertig.add(json.loads(line)["key"])
                except Exception:
                    pass
    offen = [t for t in alle if keyf(t) not in fertig]
    stat_path = out / f"status_{lauf_id}.json"
    zaehl = {"fertig_vorher": len(fertig), "gesamt": len(alle), "neu_ok": 0, "neu_fehler": 0, "neu_nicht_ladbar": 0, "zeitverlust": 0}
    t_start = time.monotonic()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.workers, initializer=init_worker, maxtasksperchild=250)
    inflight = {}
    it = iter(offen)
    pending_done = False
    last_stat = 0.0
    peak_rss = 0
    f = open(cp, "a", encoding="utf8", buffering=1)
    n_since_sync = 0

    def schreibe_status(final=False):
        d = dict(zaehl)
        d.update(lauf_id=lauf_id, pid_master=os.getpid(), workers=args.workers, laufzeit_s=round(time.monotonic() - t_start),
                 offen=len(offen) - (zaehl["neu_ok"] + zaehl["neu_fehler"] + zaehl["neu_nicht_ladbar"] + zaehl["zeitverlust"]),
                 in_arbeit=len(inflight), aelteste_s=round(max([time.monotonic() - v[2] for v in inflight.values()] or [0])),
                 peak_rss_worker_mb=peak_rss, frei_gb=round(frei_gb(), 2), stand=time.strftime("%Y-%m-%d %H:%M:%S"),
                 aufruf_hash=AUFRUF_HASH, beendet=final)
        stat_path.write_text(json.dumps(d), encoding="utf8")
        return d

    schreibe_status()
    try:
        while True:
            while not pending_done and len(inflight) < args.workers * 2:
                if frei_gb() < args.min_frei_gb:
                    break
                try:
                    t = next(it)
                except StopIteration:
                    pending_done = True
                    break
                inflight[keyf(t)] = (pool.apply_async(work, (t,)), t, time.monotonic())
            if pending_done and not inflight:
                break
            progressed = False
            for k in list(inflight):
                ar, t, ts = inflight[k]
                if ar.ready():
                    try:
                        row = ar.get()
                    except Exception as exc:
                        row = {"key": k, "datensatz": t[0], "aufnahme": t[1], "vv": t[2], "status": "werkzeugfehler",
                               "grund": f"Pool: {type(exc).__name__}: {exc}"[:200]}
                    del inflight[k]
                    row["aufruf_hash"] = AUFRUF_HASH
                    f.write(json.dumps(row) + "\n")
                    n_since_sync += 1
                    st = row.get("status")
                    zaehl["neu_ok" if st == "ok" else "neu_nicht_ladbar" if st == "nicht_ladbar" else "neu_fehler"] += 1
                    peak_rss = max(peak_rss, row.get("rss_mb", 0) or 0)
                    progressed = True
                elif time.monotonic() - ts > MAX_ALTER_S:
                    row = {"key": k, "datensatz": t[0], "aufnahme": t[1], "vv": t[2], "status": "werkzeugfehler",
                           "grund": f"Timeout > {int(MAX_ALTER_S)} s (process loss suspected)", "aufruf_hash": AUFRUF_HASH}
                    del inflight[k]
                    f.write(json.dumps(row) + "\n")
                    zaehl["zeitverlust"] += 1
                    progressed = True
            if n_since_sync >= 25:
                f.flush()
                os.fsync(f.fileno())
                n_since_sync = 0
            if time.monotonic() - last_stat > 20:
                last_stat = time.monotonic()
                schreibe_status()
            if not progressed:
                time.sleep(0.4)
    finally:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()
        if inflight:
            pool.terminate()  # only own child processes of this pool
        else:
            pool.close()
        pool.join()
        d = schreibe_status(final=True)
        print(json.dumps(d), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
