"""AW-234 Teil 2: LoopLens-Diagnose-Ablation (Fusionsgewicht des NeuroKit2-Delineators). Nur Rechnen, keine Auswertung.

Abgeleitet von ll_lauf.py (AW-219, unveraendert, gesperrt); Messkette, Signalpruefung und Einlesen sind identisch. Unterschiede:
  * FusionDelineator wird mit explizit uebergebenen Gewichten gebaut (Konstruktorparameter `weights`; keine Codeaenderung im Quellbaum):
      variante kontrolle: {"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25}  (= Standard, explizit; muss den gesperrten Hauptlauf bitgleich
                          reproduzieren, Stichprobe aus P0)
      variante ablation : {"neurokit": 0.0, "multilead": 0.35, "vcg": 0.25}  (NeuroKit2-Gewicht 0, die anderen unveraendert, nicht normiert:
                          Gewichte gehen nur als Verhaeltnisse ein, siehe Bericht; im Quellcode gilt eine Untergrenze max(0.01, Gewicht))
  * Aufgaben: nur PTB-XL P0 nativ (18.869 Personen), keine LUDB, kein Verstaerkungsarm.
  * Berechnet wird nur der Marker qrs_duration_ms (plus Messpunkte des repraesentativen Schlags und die Schlagtabelle).
  * Ausgabe getrennt unter <local path> (der Hauptlauf unter ergebnisse/looplens wird nicht beruehrt).

Urspruenglicher Kopf von ll_lauf.py:
Eigener Wrapper. Der LoopLens-Quellstand <local path> wird nur gelesen
(PYTHONDONTWRITEBYTECODE=1, sys.dont_write_bytecode, python -B). Aufbau der Messkette nach V2 Nachtrag 0 Abschnitt 2:
Referenz ist die Messfunktion der Kalibrierungsaudit (ptbxl_plus_calibration.measure): Signal-QC -> prepare_signal_paths
(50 Hz) -> detect_r_peaks_multilead -> FusionDelineator() mit Standardgewichten -> construct_robust_morphology_record
-> BiomarkerRegistry.compute_all. Abweichung (begruendet): record_vcg_xyz wird wie in core/pipeline.py und a_native/kern.py
uebergeben (Kors der Morphologiespur), weil die Ursprungskorrektur der Vektorgroessen (TP/PR/TQ) den Record-VCG nutzt;
die Intervallmarker sind davon nicht betroffen.

Aufgaben (Schluessel datensatz|aufnahme|arm|faktor):
  arm nativ        : keine Skalierung (Faktor leer)
  arm verstaerkung : Faktor 0.5, 1, 2 nach dem Einlesen in physikalische Einheiten, vor jeder Werkzeugstufe (V2 Abschnitt 7)
Checkpoint: eine Zeile je fertiger Aufgabe in <out>/roh/<lauf>.jsonl; Wiederaufnahme ueberspringt fertige Schluessel.

Aufruf:  python -B ll_lauf_ablation.py --variante ablation --workers 4
         python -B ll_lauf_ablation.py --variante kontrolle --workers 2
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
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(str(data_path("pub12.release_sources_a155_survival_nullable_20261007")))
PTBXL = Path(str(data_path("pub12.samples_ptb_x")))
LUDB = Path(str(data_path("pub12.lobachevsky_university_electrocardiography_database_1_0_1")))
LISTEN = Path(str(data_path("pub12.aw_219_listen")))
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FAKTOREN = (0.5, 1.0, 2.0)
POWERLINE_HZ = 50.0
MIN_DAUER_S = 6.5
# LoopLens-Marker (Registername); Zuordnung zu Schluesseln erst in ll_langformat.py
MARKER = ("qrs_duration_ms",)
GEWICHTE = {
    "kontrolle": {"neurokit": 0.4, "multilead": 0.35, "vcg": 0.25},
    "ablation": {"neurokit": 0.0, "multilead": 0.35, "vcg": 0.25},
}
SEED_KONTROLLE = 23400001
N_KONTROLLE = 300
POINTS = ("p_onset", "p_offset", "qrs_onset", "qrs_offset", "t_offset")
MAX_ALTER_S = 1800.0  # Aufgabe in Arbeit laenger als dies: als Prozessverlust gezaehlt


class _Mem(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (k, ctypes.c_ulonglong) for k in ("totalPhys", "availPhys", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExtended")
    ]


def frei_gb():
    m = _Mem()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return min(m.availPhys, m.availPage) / 1024 ** 3


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- Worker
_W = {}


def init_worker(variante):
    sys.path.insert(0, str(ROOT))
    import vcg_biomarker.mimic.feature_record as fr  # noqa: F401  (laedt die Produktionskette)
    from vcg_biomarker.core.delineation.fusion_delineator import FusionDelineator
    from vcg_biomarker.core.biomarkers import BiomarkerRegistry
    from vcg_biomarker.core.preprocessing import (
        construct_robust_morphology_record, detect_r_peaks_multilead, prepare_signal_paths)
    from vcg_biomarker.core.vcg_transform import VCGTransformRegistry
    from vcg_biomarker.mimic.signals import assess_signal_quality
    from vcg_biomarker.core.biomarkers.catalog_biomarkers import (
        FeatureContext, reset_feature_context, set_feature_context_record_sources)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("vcg_biomarker") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError("Falscher Importstand: " + module.__file__)
    import wfdb
    import psutil
    sums = {}
    for line in (PTBXL / "SHA256SUMS.txt").read_text().splitlines():
        p = line.split(maxsplit=1)
        if len(p) == 2:
            sums["ptbxl:" + p[1].strip().lstrip("*")] = p[0]
    for line in (LUDB / "SHA256SUMS.txt").read_text().splitlines():
        p = line.split(maxsplit=1)
        if len(p) == 2:
            sums["ludb:" + p[1].strip().lstrip("*")] = p[0]
    _W.update(fr=fr, FusionDelineator=FusionDelineator, BiomarkerRegistry=BiomarkerRegistry,
              robust=construct_robust_morphology_record, peaks=detect_r_peaks_multilead, prep=prepare_signal_paths,
              VCG=VCGTransformRegistry, qc=assess_signal_quality, FC=FeatureContext, reset=reset_feature_context,
              setsrc=set_feature_context_record_sources, wfdb=wfdb, sums=sums, proc=psutil.Process())
    _W["delin"] = FusionDelineator(weights=dict(GEWICHTE[variante]))
    _W["variante"] = variante
    _W["markers_ok"] = [n for n in MARKER if n in set(BiomarkerRegistry.available())]


def lade(ds, rec):
    wfdb = _W["wfdb"]
    if ds == "ptbxl":
        n = int(rec)
        rel = f"records500/{n // 1000 * 1000:05d}/{n:05d}_hr"
        base = PTBXL / rel
        keys = [(f"ptbxl:{rel}.hea", base.with_suffix(".hea")), (f"ptbxl:{rel}.dat", base.with_suffix(".dat"))]
    else:
        rel = f"data/{rec}"
        base = LUDB / rel
        keys = [(f"ludb:{rel}.hea", base.with_suffix(".hea")), (f"ludb:{rel}.dat", base.with_suffix(".dat"))]
    shas = {}
    for k, p in keys:
        h = sha_bytes(p.read_bytes())
        if _W["sums"].get(k) != h:
            raise RuntimeError("Datei weicht von der Pruefsumme ab: " + k)
        shas[p.suffix[1:]] = h[:16]
    r = wfdb.rdrecord(str(base))
    names = [str(x).upper() for x in r.sig_name]
    if sorted(names) != sorted(LEADS):
        raise ValueError(f"Ableitungen {names}")
    units = sorted({str(u).lower() for u in r.units})
    if units != ["mv"]:
        raise ValueError(f"Einheiten {units}")
    fs = int(round(r.fs))
    sig = np.asarray(r.p_signal, dtype=float)[:, [names.index(n) for n in LEADS]]
    return sig, fs, shas


def messe(sig, fs, rec):
    fr = _W["fr"]
    out = {}
    q = _W["qc"](sig, fs, expected_powerline_hz=POWERLINE_HZ, minimum_duration_seconds=MIN_DAUER_S)
    out["qc_brauchbar"] = bool(q.usable)
    out["qc_gruende"] = "|".join(str(x) for x in q.reasons)[:200]
    paths = _W["prep"](sig, fs, powerline=POWERLINE_HZ)
    peaks = _W["peaks"](paths["delineation"], fs)
    out["n_rpeaks"] = int(len(peaks))
    if len(peaks) < 2:
        raise ValueError("weniger als zwei R-Zacken")
    initial = _W["VCG"].transform(paths["delineation"], "kors")
    d = _W["delin"].delineate_record(ecg_12lead=paths["delineation"], r_peaks=peaks, fs=fs, ecg_id=int(rec), vcg_xyz=initial.xyz)
    if not d.beats:
        raise ValueError("keine Schlaege abgegrenzt")
    robust, diag = _W["robust"](paths["morphology"], d)
    record_xyz = _W["VCG"].transform(paths["morphology"], "kors").xyz
    robust_xyz = _W["VCG"].transform(robust, "kors").xyz
    acq = {"original_sampling_rate_hz": float(fs)}
    # Messpunkte des repraesentativen Schlags (wie a_native/kern.py)
    _W["reset"]()
    _W["setsrc"](paths["morphology"], record_xyz, None, sig, acq)
    try:
        ctx = _W["FC"](d, robust, robust_xyz, fs)
        pts = {"p_onset": ctx._sample("p_onset"), "p_offset": ctx._sample("p_offset"),
               "qrs_onset": ctx._qrs_start(), "qrs_offset": ctx._qrs_end(), "t_offset": ctx._sample("t_offset")}
    finally:
        _W["reset"]()
    out["punkte"] = {k: (None if v is None else int(v)) for k, v in pts.items()}
    res = _W["BiomarkerRegistry"].compute_all(
        delineation=d, ecg_12lead=robust, vcg_xyz=robust_xyz, fs=fs, names=_W["markers_ok"],
        record_ecg_12lead=paths["morphology"], record_raw_ecg_12lead=sig, record_acquisition=acq,
        record_vcg_xyz=record_xyz, record_vcg_wideband_xyz=None)
    m = {}
    for r in res:
        v = r.value
        v = None if v is None or not np.isfinite(float(v)) else float(v)
        m[r.name] = [v, str(r.unit), None if v is not None else str(getattr(r, "computation_notes", ""))[:80]]
    out["marker"] = m
    rep = d.representative_beat
    out["rep_r"] = None if rep is None else int(rep.r_peak.sample)
    beats = []
    for b in d.beats:
        def g(a):
            mm = getattr(b, a, None)
            return -1 if mm is None else int(mm.sample)
        beats.append([int(b.r_peak.sample), g("p_onset"), g("p_offset"), g("r_onset"), g("r_offset"), g("t_offset"),
                      1 if b.resolved_wave_status("p").value == "present" else 0])
    out["schlaege"] = beats
    try:
        out["akzeptiert_r"] = [int(b.r_peak.sample) for b in d.accepted_beats]
    except Exception:
        out["akzeptiert_r"] = None
    out["robust_status"] = diag.get("status") if isinstance(diag, dict) else None
    return out


def work(task):
    ds, rec, arm, faktor = task
    t0 = time.monotonic()
    key = f"{ds}|{rec}|{arm}|{'' if faktor is None else faktor}"
    row = {"key": key, "datensatz": ds, "aufnahme": rec, "arm": arm, "faktor": faktor}
    try:
        sig, fs, shas = lade(ds, rec)
    except Exception as exc:
        row.update(status="nicht_ladbar", grund=f"{type(exc).__name__}: {exc}"[:200], sekunden=round(time.monotonic() - t0, 2))
        return row
    row["fs"] = fs
    row["hash16"] = shas
    try:
        if arm == "verstaerkung":
            sig = sig * float(faktor)
        row["spannweite_mv"] = [round(float(x), 6) for x in (sig.max(axis=0) - sig.min(axis=0))]
        row.update(messe(sig, fs, rec))
        row["status"] = "ok"
    except Exception as exc:
        row.update(status="werkzeugfehler", grund=f"{type(exc).__name__}: {exc}"[:200])
    row["sekunden"] = round(time.monotonic() - t0, 2)
    row["rss_mb"] = round(_W["proc"].memory_info().rss / 2 ** 20)
    return row


# ---------------------------------------------------------------- Aufgabenliste
def lies_liste(name):
    return [x.strip() for x in (LISTEN / f"{name}_ecg_id.txt").read_text().split() if x.strip()]


def aufgaben(variante, limit=None):
    """Nur PTB-XL P0 nativ. Kontrolle: Stichprobe von N_KONTROLLE Personen aus P0 (fester Seed, aufsteigend). Ablation: ganz P0."""
    p0 = sorted(int(x) for x in lies_liste("P0"))
    if variante == "kontrolle":
        rng = np.random.default_rng(SEED_KONTROLLE)
        p0 = sorted(int(x) for x in rng.choice(np.array(p0), size=limit or N_KONTROLLE, replace=False))
    elif limit:
        p0 = p0[:limit]
    return [("ptbxl", str(r), "nativ", None) for r in p0]


# ---------------------------------------------------------------- Master
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", choices=["kontrolle", "ablation"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(data_path("pub12.benchmark_pub12_aw234_ablation")))
    ap.add_argument("--lauf-id", default=None)
    ap.add_argument("--min-frei-gb", type=float, default=1.5)
    args = ap.parse_args()
    out = Path(args.out)
    roh = out / "roh"
    roh.mkdir(parents=True, exist_ok=True)
    lauf_id = args.lauf_id or ("AW234_" + args.variante)
    cp = roh / f"{lauf_id}.jsonl"
    alle = aufgaben(args.variante, args.limit)
    keyf = lambda t: f"{t[0]}|{t[1]}|{t[2]}|{'' if t[3] is None else t[3]}"
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
    pool = ctx.Pool(processes=args.workers, initializer=init_worker, initargs=(args.variante,), maxtasksperchild=250)
    inflight = {}  # key -> (AsyncResult, task, t_sub)
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
                 beendet=final)
        stat_path.write_text(json.dumps(d), encoding="utf8")
        return d

    schreibe_status()
    try:
        while True:
            # nachfuellen
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
                    except Exception as exc:  # Rueckgabe/Serialisierung
                        row = {"key": k, "datensatz": t[0], "aufnahme": t[1], "arm": t[2], "faktor": t[3], "status": "werkzeugfehler",
                               "grund": f"Pool: {type(exc).__name__}: {exc}"[:200]}
                    del inflight[k]
                    f.write(json.dumps(row) + "\n")
                    n_since_sync += 1
                    st = row.get("status")
                    zaehl["neu_ok" if st == "ok" else "neu_nicht_ladbar" if st == "nicht_ladbar" else "neu_fehler"] += 1
                    peak_rss = max(peak_rss, row.get("rss_mb", 0) or 0)
                    progressed = True
                elif time.monotonic() - ts > MAX_ALTER_S:
                    row = {"key": k, "datensatz": t[0], "aufnahme": t[1], "arm": t[2], "faktor": t[3], "status": "werkzeugfehler",
                           "grund": f"Zeitueberschreitung > {int(MAX_ALTER_S)} s (Prozessverlust vermutet)"}
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
                time.sleep(0.5)
    finally:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()
        if inflight:
            pool.terminate()  # nur eigene Kindprozesse dieses Pools
        else:
            pool.close()
        pool.join()
        d = schreibe_status(final=True)
        print(json.dumps(d), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
