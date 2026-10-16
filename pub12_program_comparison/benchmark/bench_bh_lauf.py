"""Batch driver for the BRAVEHEART arms of the benchmark: convert, run in chunks, aggregate.

Call (always with -I):
  bench_bh_lauf.py konvert  <dataset> <root> <chunk_size>
  bench_bh_lauf.py lauf     <root> <from> <to> <workers> [--verzoegerung s]
  bench_bh_lauf.py aggregiere <dataset> <root> <target> <arm> <vv>

Folders under <root> (e.g. <local path>):
  in/chunk_NN/     input files <dataset>_<id>.txt plus (after the run) BRAVEHEART outputs
  state/           manifest.json, chunk_NN.done.json, chunk_NN.log.json, STOP.txt, treiber.log
  braveheart_app is taken from <root>/../braveheart/app (frozen configuration)

Checkpoints: a chunk counts as done as soon as chunk_NN.done.json exists. The driver skips
finished chunks and can therefore be restarted after an interruption. On a security or license dialog
(exit code 3 of the dialog automation) it writes STOP.txt and ends without processing further chunks.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
PY = sys.executable
AUTO = HIER / "braveheart_batch_auto.py"


def _log(state: Path, text: str) -> None:
    with (state / "treiber.log").open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + text + "\n")


def cmd_konvert(datensatz: str, wurzel: Path, groesse: int) -> int:
    import bench_braveheart as bb
    import bench_datasets as bd

    state = wurzel / "state"
    state.mkdir(parents=True, exist_ok=True)
    ids = bd.aufnahmen(datensatz)
    manifest = {"datensatz": datensatz, "stueckgroesse": groesse, "stuecke": {}, "aufnahmen": {}}
    for i in range(0, len(ids), groesse):
        nr = i // groesse
        ordner = wurzel / "in" / f"chunk_{nr:02d}"
        ordner.mkdir(parents=True, exist_ok=True)
        manifest["stuecke"][f"{nr:02d}"] = []
        for rid in ids[i:i + groesse]:
            ziel = ordner / f"{datensatz}_{rid}.txt"
            try:
                m = bb.konvertiere(bd.wfdb_basis(datensatz, rid), ziel)
                manifest["aufnahmen"][rid] = {"status": "konvertiert", "chunk": f"{nr:02d}", "fs_hz": m["fs_hz"],
                                              "n_samples": m["n_samples"], "sha256_ziel": m["sha256_ziel"]}
                manifest["stuecke"][f"{nr:02d}"].append(rid)
            except Exception as e:  # failure is recorded, never silently discarded
                manifest["aufnahmen"][rid] = {"status": "nicht_ladbar", "grund": f"{type(e).__name__}: {e}"[:200]}
    (state / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ok = sum(1 for a in manifest["aufnahmen"].values() if a["status"] == "konvertiert")
    print(f"converted {ok} of {len(ids)} in {len(manifest['stuecke'])} chunks")
    return 0


def lauf_stueck(wurzel: Path, nr: str, app: Path) -> int:
    state = wurzel / "state"
    if (state / "STOP.txt").exists():
        return 3
    done = state / f"chunk_{nr}.done.json"
    if done.exists():
        return 0
    ordner = wurzel / "in" / f"chunk_{nr}"
    for versuch in (1, 2):
        # on a retry: move half-finished outputs of the failed attempt into a subfolder
        alt = [p for p in ordner.iterdir() if not p.name.startswith(("ludb_", "ptbxl_", "qtdb_"))
               or p.suffix != ".txt"]
        if alt:
            ab = ordner / f"_fehlversuch_{int(time.time())}"
            ab.mkdir()
            for p in alt:
                if p.parent == ordner and p != ab:
                    p.rename(ab / p.name)
        logp = state / f"chunk_{nr}.log_v{versuch}.json"
        t0 = time.time()
        r = subprocess.run([PY, "-I", str(AUTO), "--app-dir", str(app), "--ecg-dir", str(ordner),
                            "--timeout", "2400", "--log", str(logp)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        _log(state, f"chunk {nr} attempt {versuch} rc={r.returncode} {dt:.0f}s")
        if r.returncode == 3:
            (state / "STOP.txt").write_text(f"chunk {nr}: security/license dialog, see chunk_{nr}.log.json\n",
                                            encoding="utf-8")
            return 3
        if r.returncode == 0:
            if logp.exists():
                (state / f"chunk_{nr}.log.json").write_text(logp.read_text(encoding="utf-8"), encoding="utf-8")
            done.write_text(json.dumps({"chunk": nr, "sekunden": dt, "ende": time.strftime("%Y-%m-%dT%H:%M:%S")}),
                            encoding="utf-8")
            return 0
    return 1


def cmd_lauf(wurzel: Path, von: int, bis: int, worker: int, verzoegerung: float) -> int:
    time.sleep(verzoegerung)
    state = wurzel / "state"
    man = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    app = wurzel.parent / "braveheart" / "app"
    nrs = [n for n in sorted(man["stuecke"]) if von <= int(n) <= bis]
    _log(state, f"start von={von} bis={bis} worker={worker} stuecke={len(nrs)}")
    with ThreadPoolExecutor(max_workers=worker) as ex:
        rcs = list(ex.map(lambda n: lauf_stueck(wurzel, n, app), nrs))
    _log(state, f"end rcs={rcs}")
    (state / f"treiber_ende_{von}_{bis}.json").write_text(json.dumps({"rcs": rcs, "stuecke": nrs}), encoding="utf-8")
    return 0 if all(r == 0 for r in rcs) else 1


# ---- Aggregation ----------------------------------------------------------------------------------------

def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


# Note: the German texts below (native names and status reasons) are written into the output table and are
# kept unchanged, because downstream analysis steps read them.
# measurement -> (native name, unit, column or None=derived)
ABBILDUNG = {
    "qrs_dauer_global": ("qrs_int", "ms"),
    "qt_global": ("qt_int", "ms"),
    "qtcf": ("abgeleitet: qt_int/(60/hr)^(1/3) (Fridericia, hr aus processed_ecgs)", "ms"),
    "qrst_winkel_peak": ("qrst_angle_peak", "grad"),
    "qrst_winkel_integral": ("qrst_angle_area", "grad"),
    "vektorbetrag_max_qrs": ("q_peak_mag", "mV"),
    "integralbetrag_qrs": ("q_area_mag", "mV*ms"),
    "sai_qrst": ("sai_qrst", "mV*ms"),
    "schleifenlaenge_qrs": ("vcg_length_qrs", "mV"),
}
NICHT_DEFINIERT = {
    "qrs_dauer_ableitung": "BRAVEHEART annotiert Fiducials nur am VCG-Medianschlag, keine ableitungsweisen Grenzen",
    "p_dauer": "BRAVEHEART kennt keine P-Fiducials (Beats.m: Q, QRS, S, Tend)",
    "pr_intervall": "BRAVEHEART kennt keine P-Fiducials",
}


def cmd_aggregiere(datensatz: str, wurzel: Path, ziel: Path, arm: str, vv: str, auftrag: str = "AW-208") -> int:
    import bench_braveheart as bb
    import bench_schema as bs

    state = wurzel / "state"
    man = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    app = wurzel.parent / "braveheart" / "app"
    konf = bb.eingefrorene_dateien(app)
    import hashlib
    konfig_hash = hashlib.sha256(json.dumps(konf, sort_keys=True).encode()).hexdigest()
    ziel.mkdir(parents=True, exist_ok=True)
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
                    stamm = Path(z["filename"]).stem
                    roh_zeilen[stamm] = z
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
    lauf_id = f"{auftrag}_{datensatz}_{arm}"
    basis = {"lauf_id": lauf_id, "datensatz": datensatz, "cluster_id": "", "ableitung": "global",
             "bezug": "medianschlag", "schlag_nr": "", "werkzeug": "braveheart", "werkzeug_version": "1.8.2",
             "konfig_hash": konfig_hash, "arm": arm, "vorverarbeitung": vv}
    for rid, info in man["aufnahmen"].items():
        stamm = f"{datensatz}_{rid}"
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
        for mg, (nativ, einheit) in ABBILDUNG.items():
            zeile = dict(basis, aufnahme=rid, messgroesse=mg, native_bezeichnung=nativ, einheit=einheit,
                         wert="", status=status_auf, statusgrund=grund_auf)
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
        for mg, grund in NICHT_DEFINIERT.items():
            zeile = dict(basis, aufnahme=rid, messgroesse=mg, native_bezeichnung="", einheit="ms", wert="",
                         status="nicht_definiert", statusgrund=grund)
            w.writerow(zeile)
    f_out.close()
    # raw table (all columns), gzip-compressed
    import gzip
    with gzip.open(ziel / "processed_ecgs_roh.csv.gz", "wt", encoding="utf-8", newline="") as g:
        wr = csv.DictWriter(g, fieldnames=spalten_roh)
        wr.writeheader()
        for rid in man["aufnahmen"]:
            z = roh_zeilen.get(f"{datensatz}_{rid}")
            if z:
                wr.writerow(z)
    (ziel / "aufstellung.json").write_text(json.dumps({
        "lauf_id": lauf_id, "aufnahmen_gesamt": len(man["aufnahmen"]), "zaehlung": zaehl,
        "ausfaelle": ausfaelle, "qc_geflaggt": qc_flag, "konfig_hash": konfig_hash, "konfig_dateien": konf,
    }, indent=1, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(zaehl))
    return 0


def main(a: list[str]) -> int:
    if len(a) >= 5 and a[1] == "konvert":
        return cmd_konvert(a[2], Path(a[3]), int(a[4]))
    if len(a) >= 6 and a[1] == "lauf":
        verz = float(a[a.index("--verzoegerung") + 1]) if "--verzoegerung" in a else 0.0
        return cmd_lauf(Path(a[2]), int(a[3]), int(a[4]), int(a[5]), verz)
    if len(a) in (7, 8) and a[1] == "aggregiere":
        return cmd_aggregiere(a[2], Path(a[3]), Path(a[4]), a[5], a[6], *(a[7:8] or []))
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
