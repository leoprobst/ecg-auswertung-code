"""BRAVEHEART gain and filter arms (S_G, LUDB, PT smoke test). Computation only, no evaluation.

Builds on bench_bh_ptbxl.py: the same dialog automation (P.rechne_stueck: unique folder name per attempt,
output check, retry), the same input format (bench_braveheart.schreibe_unformatted, 15 significant digits).
The tools are not modified; P.DATENSATZ (file prefix) is set per job.

Arms (name -> dataset, preprocessing, app):
  sg_g05 sg_g1 sg_g2 sg_vv1 sg_vv2         PTB-XL S_G list, chunks of 500
  ludb_g05 ludb_g1 ludb_g2 ludb_vv1 ludb_vv2   LUDB, a single chunk
  pt_g05 pt_g1 pt_g2 pt_vv1 pt_vv2         smoke test/controls on the PT list
App: unchanged frozen app (frozen configuration hash) for VV0_G*; for VV1 and VV2 a copy "app_vv1" with
lowpass=0 and highpass=0 in Annoparams.csv (baseline_correct_flag stays 1), otherwise byte-identical.

Usage:
  bench_bh_arme.py vorbereite                  create the app copy for VV1/VV2 (once), print hashes
  bench_bh_arme.py lauf <arm> [<arm> ...]      arms in sequence (resume: finished chunks are skipped)
  bench_bh_arme.py final <arm> [<arm> ...]     aggregation per arm into the long format in the results folder
  bench_bh_arme.py k4                          control on units/scaling on PT and a few LUDB recordings
  bench_bh_arme.py k1 <arm> <native>           control: arm against native run (ptbxl|ludb), equality count only
  bench_bh_arme.py stand
Clean halt as in the PTB-XL arm: state/STOP.txt per arm or the global STOP file in the working folder.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import csv
import glob
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))

import numpy as np  # noqa: E402

import bench_arme as A  # noqa: E402
import bench_bh_lauf as L  # noqa: E402
import bench_bh_ptbxl as P  # noqa: E402
import bench_braveheart as bb  # noqa: E402
import bench_datasets as bd  # noqa: E402
import bench_schema as bs  # noqa: E402

BASIS = Path(str(data_path("pub12.benchmark_bh")))
APP = BASIS / "braveheart" / "app"
APP_VV1 = BASIS / "braveheart" / "app_vv1"
ERG = Path(str(data_path("pub12.benchmark_ergebnisse")))
GLOBAL_STOP = BASIS / "STOP_AW220.txt"
TREIBERLOG = BASIS / "aw220_treiber.log"
STATUS = BASIS / "aw220_status.json"
CHUNK = 500
AUFTRAG = "AW-220"

# arm -> (dataset, list, vv, app, root, target)
ARME: dict[str, dict] = {}


def _reg(name, ds, liste, vv, app, wurzel, ziel):
    ARME[name] = dict(name=name, ds=ds, liste=liste, vv=vv, app=app, wurzel=wurzel, ziel=ziel)


for _n, _vv in (("g05", "VV0_G0.5"), ("g1", "VV0_G1"), ("g2", "VV0_G2")):
    _reg(f"sg_{_n}", "ptbxl", "S_G", _vv, APP, BASIS / "sg_gain" / _n, ERG / "braveheart_sg" / _n)
    _reg(f"ludb_{_n}", "ludb", "LUDB", _vv, APP, BASIS / "ludb_arme" / _n, ERG / "braveheart_ludb_arme" / _n)
    _reg(f"pt_{_n}", "ptbxl", "PT", _vv, APP, BASIS / "aw220_pt" / _n, ERG / "braveheart_pt_rauch" / _n)
for _n, _vv in (("vv1", "VV1"), ("vv2", "VV2")):
    _reg(f"sg_{_n}", "ptbxl", "S_G", _vv, APP_VV1, BASIS / "sg_filter" / _n, ERG / "braveheart_sg" / _n)
    _reg(f"ludb_{_n}", "ludb", "LUDB", _vv, APP_VV1, BASIS / "ludb_arme" / _n, ERG / "braveheart_ludb_arme" / _n)
    _reg(f"pt_{_n}", "ptbxl", "PT", _vv, APP_VV1, BASIS / "aw220_pt" / _n, ERG / "braveheart_pt_rauch" / _n)


def log(text: str) -> None:
    TREIBERLOG.parent.mkdir(parents=True, exist_ok=True)
    with TREIBERLOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + text + "\n")


def ids_von(liste: str) -> list[str]:
    if liste == "LUDB":
        return bd.aufnahmen("ludb")
    return A.lies_liste(liste)


def stuecke(arm: dict) -> list[list[str]]:
    ids = ids_von(arm["liste"])
    return [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]


# ----------------------------------------------------------------------------- Preparation
def cmd_vorbereite() -> int:
    A.pruefe_listen()
    if not APP_VV1.exists():
        shutil.copytree(APP, APP_VV1)
    zeilen = (APP / "Annoparams.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    neu = []
    for z in zeilen:
        k = z.split(",")[0].strip()
        if k == "lowpass":
            z = z.replace("lowpass,1", "lowpass,0")
        elif k == "highpass":
            z = z.replace("highpass,1", "highpass,0")
        neu.append(z)
    (APP_VV1 / "Annoparams.csv").write_bytes("".join(neu).encode("utf-8"))
    hash_app = hashlib.sha256(json.dumps(bb.eingefrorene_dateien(APP), sort_keys=True).encode()).hexdigest()
    hash_vv1 = hashlib.sha256(json.dumps(bb.eingefrorene_dateien(APP_VV1), sort_keys=True).encode()).hexdigest()
    a_alt = (APP / "Annoparams.csv").read_text(encoding="utf-8").splitlines()
    a_neu = (APP_VV1 / "Annoparams.csv").read_text(encoding="utf-8").splitlines()
    diff = [(x, y) for x, y in zip(a_alt, a_neu) if x != y]
    ausgabe = {"konfig_hash_app": hash_app, "konfig_hash_app_vv1": hash_vv1, "unterschiede_annoparams": diff,
               "zeilenzahl_gleich": len(a_alt) == len(a_neu),
               "batch_settings_gleich": (APP / "batch_settings.csv").read_bytes() == (APP_VV1 / "batch_settings.csv").read_bytes(),
               "exe_gleich": (APP / "braveheart_batch.exe").read_bytes() == (APP_VV1 / "braveheart_batch.exe").read_bytes(),
               "annoparams_sha256_app_vv1": hashlib.sha256((APP_VV1 / "Annoparams.csv").read_bytes()).hexdigest()}
    print(json.dumps(ausgabe, indent=1))
    (BASIS / "aw220_app_vv1_konfig.json").write_text(json.dumps(ausgabe, indent=1), encoding="utf-8")
    return 0


# ----------------------------------------------------------------------------- Conversion
def konvertiere_aufnahme(ds: str, rid: str, vv: str, ziel: Path) -> dict:
    base, _ = A.basis_pfad(ds, rid)
    shas = A.pruefe_quelle(ds, rid)
    sig, fs, roh = bb.lese_wfdb_mv(base)
    spann = [round(float(x), 9) for x in (sig.max(axis=0) - sig.min(axis=0))]
    sig2 = A.vorverarbeite(sig, fs, vv)
    sha = bb.schreibe_unformatted(ziel, sig2, fs)
    return {"status": "konvertiert", "fs_hz": fs, "n_samples": int(sig2.shape[0]), "sha256_ziel": sha,
            "quelle_hash16": shas, "spannweite_quelle_mv": spann}


def konvert_stueck(arm: dict, nr: int, ids: list[str]) -> dict:
    wurzel, ds, vv = arm["wurzel"], arm["ds"], arm["vv"]
    state = wurzel / "state"
    state.mkdir(parents=True, exist_ok=True)
    mf = state / f"chunk_{nr:02d}.manifest.json"
    ordner = wurzel / "in" / f"chunk_{nr:02d}"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        fehlt = [r for r in m["ids"] if m["aufnahmen"][r]["status"] == "konvertiert" and not (ordner / f"{ds}_{r}.txt").exists()]
        if not fehlt:
            return m
        mf.unlink()
    ordner.mkdir(parents=True, exist_ok=True)
    m = {"nr": f"{nr:02d}", "ids": ids, "aufnahmen": {}, "vv": vv}
    t0 = time.time()
    for rid in ids:
        ziel = ordner / f"{ds}_{rid}.txt"
        try:
            m["aufnahmen"][rid] = konvertiere_aufnahme(ds, rid, vv, ziel)
        except Exception as e:  # a failure is recorded, never silently dropped
            m["aufnahmen"][rid] = {"status": "nicht_ladbar", "grund": f"{type(e).__name__}: {e}"[:200]}
    m["konvert_sekunden"] = round(time.time() - t0, 1)
    mf.write_text(json.dumps(m), encoding="utf-8")
    return m


# ----------------------------------------------------------------------------- Run
def schreibe_status(extra: dict) -> None:
    d = {"pid": os.getpid(), "stand": time.strftime("%Y-%m-%d %H:%M:%S")}
    d.update(extra)
    STATUS.write_text(json.dumps(d), encoding="utf-8")


def stop_vorhanden(arm: dict | None = None) -> bool:
    if GLOBAL_STOP.exists():
        return True
    return bool(arm and (arm["wurzel"] / "state" / "STOP.txt").exists())


def cmd_lauf(namen: list[str]) -> int:
    A.pruefe_listen()
    jobs = []
    for n in namen:
        arm = ARME[n]
        for nr, ids in enumerate(stuecke(arm)):
            if not (arm["wurzel"] / "state" / f"chunk_{nr:02d}.done.json").exists():
                jobs.append((arm, nr, ids))
    log(f"start lauf arms={namen} open_chunks={len(jobs)} pid={os.getpid()}")
    vorlauf: dict[int, threading.Thread] = {}

    def starte_vorlauf(k: int) -> None:
        if k in vorlauf or k >= len(jobs):
            return
        a, nr, ids = jobs[k]
        t = threading.Thread(target=lambda: konvert_stueck(a, nr, ids), daemon=True)
        vorlauf[k] = t
        t.start()

    fehl_folge = 0
    for k, (arm, nr, ids) in enumerate(jobs):
        state = arm["wurzel"] / "state"
        state.mkdir(parents=True, exist_ok=True)
        if stop_vorhanden(arm):
            log("STOP present, ending")
            return 3
        if not P.warte_entsperrt(state):
            (state / "STOP.txt").write_text("Screen locked for longer than 45 min\n", encoding="utf-8")
            log("STOP: screen locked")
            return 3
        schreibe_status({"arm": arm["name"], "stueck": nr, "job": k, "jobs": len(jobs), "phase": "konvertiere"})
        t0 = time.time()
        if k in vorlauf:
            vorlauf[k].join()
        m = konvert_stueck(arm, nr, ids)
        starte_vorlauf(k + 1)
        nok = sum(1 for a in m["aufnahmen"].values() if a["status"] == "konvertiert")
        if nok == 0:
            log(f"{arm['name']} chunk {nr:02d}: no recording converted, skipped")
            continue
        P.DATENSATZ = arm["ds"]  # file prefix for rechne_stueck and the output check
        erwartet = {f"{arm['ds']}_{r}.txt" for r, a in m["aufnahmen"].items() if a["status"] == "konvertiert"}
        schreibe_status({"arm": arm["name"], "stueck": nr, "job": k, "jobs": len(jobs), "phase": "braveheart"})
        t1 = time.time()
        rc = P.rechne_stueck(arm["wurzel"], nr, arm["app"], erwartet)
        log(f"{arm['name']} chunk {nr:02d} rc={rc} conv_wait={t1 - t0:.0f}s run={time.time() - t1:.0f}s n={len(ids)} ok_conv={nok}")
        if rc == 3:
            return 3
        if rc != 0:
            fehl_folge += 1
            if fehl_folge >= 2:
                (state / "STOP.txt").write_text(f"two chunks in a row without a result (last {arm['name']} {nr:02d})\n", encoding="utf-8")
                log("STOP: two failures in a row")
                return 3
            continue
        fehl_folge = 0
        ordner = arm["wurzel"] / "in" / f"chunk_{nr:02d}"
        geloescht = 0
        for p in ordner.glob(f"{arm['ds']}_*.txt"):
            p.unlink()
            geloescht += 1
        log(f"{arm['name']} chunk {nr:02d}: {geloescht} input files deleted")
    schreibe_status({"phase": "ende", "jobs": len(jobs)})
    alle = all((a["wurzel"] / "state" / f"chunk_{n:02d}.done.json").exists() for a in (ARME[x] for x in namen) for n in range(len(stuecke(a))))
    log(f"end all_done={alle}")
    return 0 if alle else 1


# ----------------------------------------------------------------------------- Aggregation
def _f(x):
    return L._f(x)


def cmd_final(namen: list[str]) -> int:
    for n in namen:
        arm = ARME[n]
        rc = aggregiere(arm)
        if rc != 0:
            return rc
    return 0


def aggregiere(arm: dict) -> int:
    wurzel, ds, vv, app, ziel = arm["wurzel"], arm["ds"], arm["vv"], arm["app"], arm["ziel"]
    state = wurzel / "state"
    P.DATENSATZ = ds
    man = {"datensatz": ds, "stuecke": {}, "aufnahmen": {}}
    for mf in sorted(state.glob("chunk_*.manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        man["stuecke"][m["nr"]] = [r for r in m["ids"] if m["aufnahmen"][r]["status"] == "konvertiert"]
        man["aufnahmen"].update(m["aufnahmen"])
    fehlend = [n for n in range(len(stuecke(arm))) if not (state / f"chunk_{n:02d}.done.json").exists()]
    if fehlend:
        print(f"{arm['name']}: chunks not finished {fehlend}, no aggregation")
        return 1
    # check as in the PTB-XL arm
    for d in sorted(state.glob("chunk_*.done.json")):
        nr = d.name[6:8]
        m = json.loads((state / f"chunk_{nr}.manifest.json").read_text(encoding="utf-8"))
        erw = {f"{ds}_{r}.txt" for r, a in m["aufnahmen"].items() if a["status"] == "konvertiert"}
        pr = P.pruefe_stueck(wurzel / "in" / f"chunk_{nr}", erw)
        if not pr["ok"]:
            print(f"{arm['name']}: chunk {nr} invalid {pr}")
            return 2
    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "logs").mkdir(exist_ok=True)
    (ziel / "logs" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    konf = bb.eingefrorene_dateien(app)
    konfig_hash = hashlib.sha256(json.dumps(konf, sort_keys=True).encode()).hexdigest()
    roh_zeilen: dict[str, dict] = {}
    fehler_bh: dict[str, str] = {}
    qc_flag: dict[str, str] = {}
    spalten_roh: list[str] = []
    for nr in sorted(man["stuecke"]):
        ordner = wurzel / "in" / f"chunk_{nr}"
        for pfad in glob.glob(str(ordner / "processed_ecgs_*.csv")):
            with open(pfad, encoding="utf-8", newline="") as f:
                rd = csv.DictReader(f)
                if not spalten_roh:
                    spalten_roh = rd.fieldnames or []
                for z in rd:
                    roh_zeilen[Path(z["filename"]).stem] = z
        for pfad in glob.glob(str(ordner / "errors_*.csv")):
            with open(pfad, encoding="utf-8", newline="") as f:
                for z in csv.reader(f):
                    if z and z[0] not in ("filename", "file", "ecg"):
                        fehler_bh[Path(z[0]).stem] = ";".join(z[1:])[:200]
        for pfad in glob.glob(str(ordner / "check_ecg_list_*.csv")):
            with open(pfad, encoding="utf-8", newline="") as f:
                for z in csv.reader(f):
                    if z and z[0] not in ("filename", "file", "ecg"):
                        qc_flag[Path(z[0]).stem] = ";".join(z[1:])[:200]
    f_out, w = bs.oeffne_ausgabe(ziel / "ausgabe_langformat.csv")
    zaehl = {"ok": 0, "nicht_ladbar": 0, "werkzeugfehler": 0, "fehlend": 0, "ausgeschlossen_qc": 0}
    ausfaelle = []
    lauf_id = f"{AUFTRAG}_{arm['name']}"
    basis = {"lauf_id": lauf_id, "datensatz": ds, "cluster_id": "", "ableitung": "global", "bezug": "medianschlag",
             "schlag_nr": "", "werkzeug": "braveheart", "werkzeug_version": "1.8.2", "konfig_hash": konfig_hash,
             "arm": "nativ", "vorverarbeitung": vv}
    for rid, info in man["aufnahmen"].items():
        stamm = f"{ds}_{rid}"
        status_auf, grund_auf = "ok", ""
        z = roh_zeilen.get(stamm)
        if info["status"] != "konvertiert":
            status_auf, grund_auf = "nicht_ladbar", info.get("grund", "Konvertierung fehlgeschlagen")
        elif stamm in fehler_bh:
            status_auf, grund_auf = "werkzeugfehler", "BRAVEHEART errors-Liste: " + fehler_bh[stamm]
        elif z is None:
            status_auf, grund_auf = "fehlend", "keine Zeile in processed_ecgs (Stueck nicht gelaufen oder Zeile fehlt)"
        if status_auf != "ok":
            zaehl[status_auf] += 1
            ausfaelle.append({"aufnahme": rid, "status": status_auf, "grund": grund_auf})
        else:
            zaehl["ok"] += 1
        qc = qc_flag.get(stamm)
        for mg, (nativ, einheit) in L.ABBILDUNG.items():
            zeile = dict(basis, aufnahme=rid, messgroesse=mg, native_bezeichnung=nativ, einheit=einheit, wert="",
                         status=status_auf, statusgrund=grund_auf)
            if status_auf == "ok":
                if mg == "qtcf":
                    qt, hr = _f(z["qt_int"]), _f(z["hr"])
                    v = None if (qt is None or hr is None or hr <= 0) else qt / ((60.0 / hr) ** (1.0 / 3.0))
                else:
                    v = _f(z[nativ])
                if v is None:
                    zeile["status"], zeile["statusgrund"] = "fehlend", "Wert NaN oder leer in processed_ecgs"
                else:
                    zeile["wert"] = f"{v:.10g}"
                    if qc:
                        zeile["statusgrund"] = "BRAVEHEART-Qualitaetsmarke (Wert behalten): " + qc
            w.writerow(zeile)
        for mg, grund in L.NICHT_DEFINIERT.items():
            w.writerow(dict(basis, aufnahme=rid, messgroesse=mg, native_bezeichnung="", einheit="ms", wert="",
                            status="nicht_definiert", statusgrund=grund))
    f_out.close()
    with gzip.open(ziel / "processed_ecgs_roh.csv.gz", "wt", encoding="utf-8", newline="") as g:
        wr = csv.DictWriter(g, fieldnames=spalten_roh)
        wr.writeheader()
        for rid in man["aufnahmen"]:
            z = roh_zeilen.get(f"{ds}_{rid}")
            if z:
                wr.writerow(z)
    with tarfile.open(ziel / f"annotations_{arm['name']}.tar.gz", "w:gz") as tf:
        for nr in sorted(man["stuecke"]):
            for d in (wurzel / "in" / f"chunk_{nr}").glob("annotations_*"):
                tf.add(d, arcname=f"chunk_{nr}/{d.name}")
    # check_ecg_list has a header line, errors (file name, message) does not. Unlike cmd_final of the PTB-XL arm, all rows
    # of all chunks are written for errors (cmd_final discards the first error row of every chunk after the first).
    for suffix, name, kopf in (("check_ecg_list_*.csv", "check_ecg_list_alle.csv", True), ("errors_*.csv", "errors_alle.csv", False)):
        with (ziel / name).open("w", encoding="utf-8", newline="") as out:
            erste = True
            for nr in sorted(man["stuecke"]):
                for f in (wurzel / "in" / f"chunk_{nr}").glob(suffix):
                    zeilen = f.read_text(encoding="utf-8").splitlines()
                    if not zeilen:
                        continue
                    if kopf:
                        if erste:
                            out.write(zeilen[0] + "\n")
                            erste = False
                        zeilen = zeilen[1:]
                    for zl in zeilen:
                        out.write(zl + "\n")
    tabellen = {n: hashlib.sha256((ziel / n).read_bytes()).hexdigest() for n in ("ausgabe_langformat.csv", "processed_ecgs_roh.csv.gz")}
    (ziel / "aufstellung.json").write_text(json.dumps({
        "lauf_id": lauf_id, "vorverarbeitung": vv, "aufnahmen_gesamt": len(man["aufnahmen"]), "zaehlung": zaehl,
        "ausfaelle": ausfaelle, "qc_geflaggt": qc_flag, "konfig_hash": konfig_hash, "konfig_dateien": konf,
        "app": str(app), "tabellen_sha256": tabellen, "externer_filter": A.BP_BESCHREIBUNG if vv == "VV2" else None,
    }, indent=1, ensure_ascii=True), encoding="utf-8")
    print(arm["name"], json.dumps(zaehl))
    return 0


# ----------------------------------------------------------------------------- Controls
def cmd_k4() -> int:
    """Units/scaling control: round trip (factor 2 then 0.5) bit-identical; range check PTB-XL (PT) and LUDB. Counting only."""
    erg = {"hin_rueck_speicher": {}, "hin_rueck_text": {}, "spannweite": {}}
    for ds, ids in (("ptbxl", A.lies_liste("PT")), ("ludb", [str(i) for i in range(1, 6)])):
        n = n_mem = n_txt = n_rng = 0
        n_txt_gleich_lesen = 0
        leads_ok = leads_all = 0
        fehler = []
        for rid in ids:
            base, _ = A.basis_pfad(ds, rid)
            sig, fs, _roh = bb.lese_wfdb_mv(base)
            n += 1
            rueck = (sig * 2.0) * 0.5
            n_mem += int(np.array_equal(rueck, sig))
            t_orig = _text(sig, fs)
            t_rueck = _text(rueck, fs)
            n_txt += int(t_orig == t_rueck)
            # text round trip: the text of factor 2 is read (as BRAVEHEART does) and halved
            gelesen2 = np.array([[float(v) for v in z.split()] for z in _text(sig * 2.0, fs).splitlines()[1:]])
            gelesen1 = np.array([[float(v) for v in z.split()] for z in t_orig.splitlines()[1:]])
            n_txt_gleich_lesen += int(np.array_equal(gelesen2 * 0.5, gelesen1))
            spann = sig.max(axis=0) - sig.min(axis=0)
            if ds == "ptbxl":
                ok_leads = (spann >= 0.05) & (spann <= 10.0)
                n_rng += int(ok_leads.all())
                leads_ok += int(ok_leads.sum())
                leads_all += int(ok_leads.size)
            else:
                ok = np.abs(spann - 1.0) <= 1e-6
                n_rng += int(ok.all())
                leads_ok += int(ok.sum())
                leads_all += int(ok.size)
        key = ds
        erg["hin_rueck_speicher"][key] = {"n": n, "bitgleich": n_mem}
        erg["hin_rueck_text"][key] = {"n": n, "text_bitgleich_nach_rueckmultiplikation_im_speicher": n_txt,
                                      "text_faktor2_gelesen_mal_0_5_gleich_text_faktor1_gelesen": n_txt_gleich_lesen}
        erg["spannweite"][key] = {"n": n, "aufnahmen_alle_12_ableitungen_im_bereich": n_rng, "ableitungen_im_bereich": leads_ok,
                                  "ableitungen_gesamt": leads_all,
                                  "bereich": "0.05 to 10 mV" if ds == "ptbxl" else "1.000 mV +- 1e-6"}
    print(json.dumps(erg, indent=1))
    (BASIS / "aw220_k4.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    return 0


def _text(sig: np.ndarray, fs: int) -> str:
    """Same format as bench_braveheart.schreibe_unformatted, as text."""
    kopf = " ".join([f"{float(fs):.15g}"] * 12)
    return "\n".join([kopf] + [" ".join(f"{v:.15g}" for v in z) for z in sig]) + "\n"


EXCL_SPALTEN = {"filename", "note", "proc_date", "proc_time", "source"}


def _lies_roh(pfad: Path, stamm: set[str]) -> tuple[list[str], dict[str, dict]]:
    zeilen = {}
    with gzip.open(pfad, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        kopf = rd.fieldnames or []
        for z in rd:
            s = Path(z["filename"]).stem
            if s in stamm:
                zeilen[s] = z
    return kopf, zeilen


def cmd_k1(armname: str, nativ: str) -> int:
    """Factor-1 arm against the native run (no scaling). BRAVEHEART: bit-identical. Equality counting only."""
    arm = ARME[armname]
    ds = arm["ds"]
    nat_ziel = ERG / ("ptbxl" if nativ == "ptbxl" else "ludb")
    ids = sum((list(json.loads(p.read_text(encoding="utf-8"))["ids"]) for p in sorted((arm["wurzel"] / "state").glob("chunk_*.manifest.json"))), [])
    stamm = {f"{ds}_{r}" for r in ids}
    k_arm, z_arm = _lies_roh(arm["ziel"] / "processed_ecgs_roh.csv.gz", stamm)
    k_nat, z_nat = _lies_roh(nat_ziel / "processed_ecgs_roh.csv.gz", stamm)
    gemeinsam = sorted(set(z_arm) & set(z_nat))
    spalten = [c for c in k_arm if c in k_nat and c not in EXCL_SPALTEN]
    n_gleich = 0
    abw_ids = []
    abw_spalten = set()
    for s in gemeinsam:
        d = [c for c in spalten if z_arm[s].get(c) != z_nat[s].get(c)]
        if d:
            abw_ids.append(s)
            abw_spalten.update(d)
        else:
            n_gleich += 1

    def namen(p: Path) -> set[str]:
        out = set()
        if p.exists():
            for z in csv.reader(p.read_text(encoding="utf-8").splitlines()):
                if z and z[0] not in ("filename", "file", "ecg"):
                    out.add(Path(z[0]).stem)
        return out

    def fehler_aus_aufstellung(p: Path) -> dict[str, str]:
        # errors_alle.csv of the native run (PTB-XL arm, cmd_final) is incomplete; aufstellung.json is authoritative
        a = json.loads(p.read_text(encoding="utf-8"))
        return {f"{ds}_{x['aufnahme']}": f"{x['status']}|{x['grund']}" for x in a["ausfaelle"] if f"{ds}_{x['aufnahme']}" in stamm}
    fe_arm = fehler_aus_aufstellung(arm["ziel"] / "aufstellung.json")
    fe_nat = fehler_aus_aufstellung(nat_ziel / "aufstellung.json")
    # check_ecg_list: rows equal?
    def chk(p: Path) -> dict[str, str]:
        out = {}
        if p.exists():
            for z in p.read_text(encoding="utf-8").splitlines()[1:]:
                s = Path(z.split(",")[0]).stem
                if s in stamm:
                    out[s] = z.split(",", 1)[1] if "," in z else ""
        return out
    c_arm, c_nat = chk(arm["ziel"] / "check_ecg_list_alle.csv"), chk(nat_ziel / "check_ecg_list_alle.csv")
    # input files: SHA against the manifest of the native run (PTB-XL only)
    eingabe_gleich = eingabe_n = None
    nm = BASIS / ds / "state" / "manifest.json"
    if nm.exists():
        nat_man = json.loads(nm.read_text(encoding="utf-8"))["aufnahmen"]
        arm_man = {}
        for p in sorted((arm["wurzel"] / "state").glob("chunk_*.manifest.json")):
            arm_man.update(json.loads(p.read_text(encoding="utf-8"))["aufnahmen"])
        eingabe_n = 0
        eingabe_gleich = 0
        for r in ids:
            if r in nat_man and nat_man[r].get("sha256_ziel") and arm_man.get(r, {}).get("sha256_ziel"):
                eingabe_n += 1
                eingabe_gleich += int(nat_man[r]["sha256_ziel"] == arm_man[r]["sha256_ziel"])
    erg = {"arm": armname, "nativ": nativ, "n_aufnahmen_arm": len(ids), "n_mit_processed_zeile_beide": len(gemeinsam),
           "n_nur_arm": len(set(z_arm) - set(z_nat)), "n_nur_nativ": len(set(z_nat) - set(z_arm)),
           "n_spalten_verglichen": len(spalten), "ausgeschlossene_spalten": sorted(EXCL_SPALTEN),
           "n_zeilen_bitgleich": n_gleich, "n_zeilen_abweichend": len(abw_ids), "abweichende_ids_erste": abw_ids[:10],
           "n_abweichende_spalten": len(abw_spalten),
           "errors_liste_gleich": fe_arm == fe_nat, "n_errors_arm": len(fe_arm), "n_errors_nativ": len(fe_nat),
           "check_ecg_list_gleich": c_arm == c_nat, "n_check_arm": len(c_arm), "n_check_nativ": len(c_nat),
           "eingabedateien_sha_gleich": eingabe_gleich, "eingabedateien_verglichen": eingabe_n}
    erg["bestanden"] = bool(len(gemeinsam) > 0 and not abw_ids and erg["errors_liste_gleich"] and erg["check_ecg_list_gleich"]
                            and erg["n_nur_arm"] == 0 and erg["n_nur_nativ"] == 0)
    print(json.dumps(erg, indent=1))
    if arm["vv"] == "VV0_G1":
        (arm["ziel"] / "k1.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    else:
        # only a count of the changed recordings (effectiveness control for VV1/VV2 on PT); no values, no equality claim
        erg["hinweis"] = "Not an equality check: effectiveness count (recordings with at least one deviating column)"
        (arm["ziel"] / "wirksamkeit_k6.json").write_text(json.dumps(erg, indent=1), encoding="utf-8")
    return 0 if erg["bestanden"] else 1


def cmd_stand() -> int:
    for n, a in ARME.items():
        s = a["wurzel"] / "state"
        if s.exists():
            fertig = len(list(s.glob("chunk_*.done.json")))
            print(f"{n}: {fertig} of {len(stuecke(a))} chunks finished; STOP={'yes' if (s / 'STOP.txt').exists() else 'no'}")
    return 0


def main(a: list[str]) -> int:
    if len(a) >= 2 and a[1] == "vorbereite":
        return cmd_vorbereite()
    if len(a) >= 3 and a[1] == "lauf":
        return cmd_lauf(a[2:])
    if len(a) >= 3 and a[1] == "final":
        return cmd_final(a[2:])
    if len(a) == 2 and a[1] == "k4":
        return cmd_k4()
    if len(a) == 4 and a[1] == "k1":
        return cmd_k1(a[2], a[3])
    if len(a) == 2 and a[1] == "stand":
        return cmd_stand()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
