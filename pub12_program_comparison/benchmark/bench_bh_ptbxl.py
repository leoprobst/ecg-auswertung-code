"""PTB-XL arm of the BRAVEHEART benchmark: per chunk convert, run, delete the inputs.

Call (always with -I):
  bench_bh_ptbxl.py lauf   <root> <chunk_size> [--von N] [--bis M]
  bench_bh_ptbxl.py final  <root> <target>
  bench_bh_ptbxl.py stand  <root>

One worker. Per chunk (default 500 recordings): conversion to <root>/in/chunk_NN (the next part
is converted in the background while a chunk runs), BRAVEHEART through the dialog automation
(bench_bh_lauf.lauf_stueck, unchanged frozen configuration of the app next to <root>/../braveheart/app),
then deletion of the input files (the BRAVEHEART outputs stay, they are small). Checkpoint: state/chunk_NN.done.json.
Resume: the same command, finished chunks are skipped.

Clean stop (STOP.txt in state/, the driver ends, resume with the same command after deleting STOP.txt):
  * security or license dialog (exit code 3 of the dialog automation),
  * locked screen (LogonUI.exe) for longer than 45 min,
  * two chunks in a row that failed (e.g. no interactive session).
'final' merges the manifests, aggregates into the harness schema and packs annotations to <target> (E:).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))

import psutil  # noqa: E402


def _umbenennen(quelle, ziel, versuche=30, pause=10):
    """Rename with retries: virus scanner or indexing keep a folder locked for a short time."""
    letzter = None
    for _ in range(versuche):
        try:
            quelle.rename(ziel)
            return
        except PermissionError as e:
            letzter = e
            time.sleep(pause)
    raise letzter

import bench_bh_lauf as L  # noqa: E402
import bench_braveheart as bb  # noqa: E402
import bench_datasets as bd  # noqa: E402

DATENSATZ = "ptbxl"
SPERR_MAX_S = 45 * 60


def log(state: Path, text: str) -> None:
    L._log(state, text)


def gesperrt() -> bool:
    for p in psutil.process_iter(["name"]):
        if (p.info["name"] or "").lower() == "logonui.exe":
            return True
    return False


def konvert_stueck(wurzel: Path, nr: int, ids: list[str]) -> dict:
    """Converts one chunk; manifest to state/chunk_NN.manifest.json (only when completely finished)."""
    state = wurzel / "state"
    mf = state / f"chunk_{nr:02d}.manifest.json"
    ordner = wurzel / "in" / f"chunk_{nr:02d}"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        fehlt = [r for r in m["ids"] if m["aufnahmen"][r]["status"] == "konvertiert"
                 and not (ordner / f"{DATENSATZ}_{r}.txt").exists()]
        if not fehlt:
            return m
    ordner.mkdir(parents=True, exist_ok=True)
    m = {"nr": f"{nr:02d}", "ids": ids, "aufnahmen": {}}
    t0 = time.time()
    for rid in ids:
        ziel = ordner / f"{DATENSATZ}_{rid}.txt"
        try:
            r = bb.konvertiere(bd.wfdb_basis(DATENSATZ, rid), ziel)
            m["aufnahmen"][rid] = {"status": "konvertiert", "chunk": f"{nr:02d}", "fs_hz": r["fs_hz"],
                                   "n_samples": r["n_samples"], "sha256_ziel": r["sha256_ziel"]}
        except Exception as e:  # failure is recorded, never silently discarded
            m["aufnahmen"][rid] = {"status": "nicht_ladbar", "grund": f"{type(e).__name__}: {e}"[:200]}
    m["konvert_sekunden"] = round(time.time() - t0, 1)
    mf.write_text(json.dumps(m), encoding="utf-8")
    return m


AUTO = HIER / "braveheart_batch_auto.py"


def _namen_csv(pfad: Path, kopf: bool) -> list[str]:
    import csv
    with pfad.open(encoding="utf-8", newline="") as f:
        zeilen = [z for z in csv.reader(f) if z]
    if kopf and zeilen:
        zeilen = zeilen[1:]
    return [z[0] for z in zeilen]


def pruefe_stueck(ordner: Path, erwartet: set[str]) -> dict:
    """Checks that the output in the folder belongs exactly to the inputs of this chunk.

    erwartet = file names (e.g. ptbxl_1000.txt) of the successfully converted recordings of the chunk.
    Requires: exactly one processed_ecgs file; no duplicates; no foreign names (also not in errors,
    check_ecg_list, annotations); union of processed and errors = expected (nothing is missing).
    """
    pe = sorted(ordner.glob("processed_ecgs_*.csv"))
    r: dict = {"n_erwartet": len(erwartet), "n_processed_dateien": len(pe), "ok": False, "probleme": []}
    if len(pe) != 1:
        r["probleme"].append(f"{len(pe)} processed_ecgs files")
        return r
    proc = _namen_csv(pe[0], True)
    fehl_dat = sorted(ordner.glob("errors_*.csv"))
    err = _namen_csv(fehl_dat[0], False) if fehl_dat else []
    chk_dat = sorted(ordner.glob("check_ecg_list_*.csv"))
    chk = _namen_csv(chk_dat[0], True) if chk_dat else []
    anno = [a.stem + ".txt" for d in ordner.glob("annotations_*") for a in d.glob("*.anno")]
    r.update({"n_processed": len(proc), "n_errors": len(err), "n_check": len(chk), "n_anno": len(anno)})
    dop = sorted({n for n in proc if proc.count(n) > 1})
    if dop:
        r["probleme"].append(f"duplicates in processed: {dop[:5]}")
    for name, liste in (("processed", proc), ("errors", err), ("check", chk), ("anno", anno)):
        fremd = sorted(set(liste) - erwartet)
        if fremd:
            r["probleme"].append(f"foreign names in {name}: {len(fremd)}, e.g. {fremd[:3]}")
    fehlt = sorted(erwartet - set(proc) - set(err))
    if fehlt:
        r["probleme"].append(f"{len(fehlt)} recordings neither in processed nor in errors, e.g. {fehlt[:3]}")
    r["ok"] = not r["probleme"]
    return r


def _ausgaben_im_baum(wurzel_in: Path, app: Path) -> set[str]:
    s: set[str] = set()
    for pat in ("processed_ecgs_*.csv", "errors_*.csv", "check_ecg_list_*.csv"):
        s.update(str(x) for x in wurzel_in.rglob(pat))
        s.update(str(x) for x in app.glob(pat))
    s.update(str(x) for x in app.glob("annotations_*"))
    return s


def _raeume_ausgaben_weg(chunk: Path) -> None:
    rest = [x for x in chunk.iterdir() if not (x.name.startswith(f"{DATENSATZ}_") and x.suffix == ".txt")
            and not x.name.startswith("_")]
    if rest:
        ab = chunk / f"_ungueltig_{int(time.time())}"
        ab.mkdir()
        for x in rest:
            x.rename(ab / x.name)


def rechne_stueck(wurzel: Path, nr: int, app: Path, erwartet: set[str], versuche: int = 3) -> int:
    """One chunk with a unique folder name per attempt, output check and retry.

    The chunk folder is renamed to lauf_<nr>_<token> for the run (the name in the folder dialog is thus
    unique per attempt; a folder chosen by mistake elsewhere produces outputs there that stand out).
    Afterwards it is renamed back, the output is checked against the inputs (pruefe_stueck) and new outputs
    outside the run folder are checked for. Failed or invalid outputs move to
    _ungueltig_<time>/ in the chunk folder; then retry.
    """
    import uuid
    state = wurzel / "state"
    chunk = wurzel / "in" / f"chunk_{nr:02d}"
    stat = state / "dialog_statistik.jsonl"
    for versuch in range(1, versuche + 1):
        if (state / "STOP.txt").exists():
            return 3
        _raeume_ausgaben_weg(chunk)
        vorhanden = {x.name for x in chunk.glob(f"{DATENSATZ}_*.txt")}
        if vorhanden != erwartet:
            log(state, f"chunk {nr:02d}: inputs incomplete ({len(vorhanden)} of {len(erwartet)}), abort")
            return 1
        token = uuid.uuid4().hex[:8]
        arbeit = wurzel / "in" / f"lauf_{nr:02d}_{token}"
        vorher = _ausgaben_im_baum(wurzel / "in", app)
        _umbenennen(chunk, arbeit)
        logp = state / f"chunk_{nr:02d}.log_v{versuch}_{token}.json"
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-I", str(AUTO), "--app-dir", str(app), "--ecg-dir", str(arbeit),
                                "--timeout", "2400", "--dialog-timeout", "240", "--log", str(logp)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rc = r.returncode
        finally:
            _umbenennen(arbeit, chunk)
        dt = time.time() - t0
        info = {}
        if logp.exists():
            try:
                info = json.loads(logp.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        od = info.get("ordnerdialog", {})
        eintrag = {"zeit": time.strftime("%Y-%m-%dT%H:%M:%S"), "chunk": nr, "versuch": versuch, "token": token, "rc": rc,
                   "sekunden": round(dt, 1), "dialog_hing": od.get("hing"),
                   "dialog_schon_geschlossen": od.get("schon_geschlossen"),
                   "dialog_versuche": len(od.get("versuche", [])), "t_dialog_s": info.get("t_dialog_s"),
                   "fremdfenster": info.get("fremdfenster", []), "stopp": info.get("stopp"),
                   "exception": (info.get("exception") or "")[-400:] or None, "logdatei_fehlt": not logp.exists()}
        pruef = None
        if rc == 3:
            with stat.open("a", encoding="utf-8") as f:
                f.write(json.dumps(eintrag) + chr(10))
            (state / "STOP.txt").write_text(f"chunk {nr:02d}: security/license dialog, see {logp.name}" + chr(10),
                                            encoding="utf-8")
            return 3
        if rc == 0:
            nachher = _ausgaben_im_baum(wurzel / "in", app)
            fremd = sorted(x for x in nachher - vorher if not x.startswith(str(chunk)))
            pruef = pruefe_stueck(chunk, erwartet)
            if fremd:
                pruef["ok"] = False
                pruef["probleme"].append(f"new outputs outside the run folder: {fremd[:3]}")
            eintrag["pruefung"] = pruef
        with stat.open("a", encoding="utf-8") as f:
            f.write(json.dumps(eintrag) + chr(10))
        status = "ok" if (pruef and pruef["ok"]) else (pruef["probleme"] if pruef else "-")
        log(state, f"chunk {nr:02d} attempt {versuch} token={token} rc={rc} {dt:.0f}s hung={od.get('hing')} check={status}")
        if rc == 0 and pruef and pruef["ok"]:
            (state / f"chunk_{nr:02d}.log.json").write_text(logp.read_text(encoding="utf-8"), encoding="utf-8")
            (state / f"chunk_{nr:02d}.done.json").write_text(json.dumps({
                "chunk": f"{nr:02d}", "sekunden": dt, "ende": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": token,
                "pruefung": pruef}), encoding="utf-8")
            return 0
        _raeume_ausgaben_weg(chunk)
    return 1


def stuecke(groesse: int) -> list[list[str]]:
    ids = bd.aufnahmen(DATENSATZ)
    return [ids[i:i + groesse] for i in range(0, len(ids), groesse)]


def warte_entsperrt(state: Path) -> bool:
    t0 = time.time()
    gemeldet = False
    while gesperrt():
        if not gemeldet:
            log(state, "screen locked (LogonUI.exe), waiting")
            gemeldet = True
        if time.time() - t0 > SPERR_MAX_S:
            return False
        time.sleep(30)
    if gemeldet:
        log(state, f"unlocked after {time.time() - t0:.0f}s")
    return True


def cmd_lauf(wurzel: Path, groesse: int, von: int, bis: int) -> int:
    state = wurzel / "state"
    state.mkdir(parents=True, exist_ok=True)
    app = wurzel.parent / "braveheart" / "app"
    teile = stuecke(groesse)
    nrs = [n for n in range(len(teile)) if von <= n <= bis]
    log(state, f"start ptbxl size={groesse} chunks={len(nrs)} from={von} to={min(bis, len(teile) - 1)} pid={os.getpid()}")
    offen = [n for n in nrs if not (state / f"chunk_{n:02d}.done.json").exists()]
    vorlauf: dict[int, threading.Thread] = {}

    def starte_vorlauf(n: int) -> None:
        if n in vorlauf or n >= len(teile):
            return
        t = threading.Thread(target=lambda: konvert_stueck(wurzel, n, teile[n]), daemon=True)
        vorlauf[n] = t
        t.start()

    fehlschlaege_folge = 0
    ergebnisse: dict[int, int] = {}
    for k, n in enumerate(offen):
        if (state / "STOP.txt").exists():
            log(state, "STOP.txt present, end")
            return 3
        if not warte_entsperrt(state):
            (state / "STOP.txt").write_text(f"screen locked for more than {SPERR_MAX_S // 60} min, before chunk {n:02d}\n",
                                            encoding="utf-8")
            log(state, "STOP: screen locked")
            return 3
        t0 = time.time()
        if n in vorlauf:
            vorlauf[n].join()
        m = konvert_stueck(wurzel, n, teile[n])
        t_konv = time.time() - t0
        if k + 1 < len(offen):
            starte_vorlauf(offen[k + 1])
        nok = sum(1 for a in m["aufnahmen"].values() if a["status"] == "konvertiert")
        if nok == 0:
            log(state, f"chunk {n:02d}: no recording converted, skipped")
            ergebnisse[n] = 2
            continue
        t1 = time.time()
        rc = rechne_stueck(wurzel, n, app, {f"{DATENSATZ}_{r}.txt" for r, a in m["aufnahmen"].items() if a["status"] == "konvertiert"})
        t_lauf = time.time() - t1
        ergebnisse[n] = rc
        log(state, f"chunk {n:02d} rc={rc} convert_wait={t_konv:.0f}s run={t_lauf:.0f}s recordings={len(teile[n])} ok_conv={nok}")
        if rc == 3:
            return 3
        if rc != 0:
            fehlschlaege_folge += 1
            if fehlschlaege_folge >= 2:
                (state / "STOP.txt").write_text(
                    f"two chunks in a row without result (last {n:02d}); probably no interactive "
                    f"session or dialog error, logs chunk_*.log_v*.json\n", encoding="utf-8")
                log(state, "STOP: two failures in a row")
                return 3
            continue
        fehlschlaege_folge = 0
        # delete the inputs of the chunk (outputs stay)
        ordner = wurzel / "in" / f"chunk_{n:02d}"
        geloescht = 0
        for p in ordner.glob(f"{DATENSATZ}_*.txt"):
            p.unlink()
            geloescht += 1
        log(state, f"chunk {n:02d}: {geloescht} input files deleted")
        fertige = len(list(state.glob("chunk_*.done.json")))
        (state / "fortschritt.json").write_text(json.dumps({
            "stuecke_gesamt": len(teile), "fertig": fertige, "zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "letztes_stueck": n, "letzte_laufzeit_s": round(t_lauf, 1)}), encoding="utf-8")
    alle_fertig = all((state / f"chunk_{n:02d}.done.json").exists() for n in range(len(teile)))
    log(state, f"end all_done={alle_fertig} results={ergebnisse}")
    return 0 if alle_fertig else 1


def cmd_pruefe(wurzel: Path) -> int:
    """Checks all finished chunks afterwards against their manifests (the output belongs to the inputs)."""
    state = wurzel / "state"
    schlecht = []
    for d in sorted(state.glob("chunk_*.done.json")):
        nr = d.name[6:8]
        m = json.loads((state / f"chunk_{nr}.manifest.json").read_text(encoding="utf-8"))
        erw = {f"{DATENSATZ}_{r}.txt" for r, a in m["aufnahmen"].items() if a["status"] == "konvertiert"}
        pr = pruefe_stueck(wurzel / "in" / f"chunk_{nr}", erw)
        print(nr, "ok" if pr["ok"] else "DEVIATION", {k: v for k, v in pr.items() if k != "ok"})
        if not pr["ok"]:
            schlecht.append(nr)
    gesehen: dict[str, str] = {}
    for d in sorted(state.glob("chunk_*.done.json")):
        nr = d.name[6:8]
        for pe in (wurzel / "in" / f"chunk_{nr}").glob("processed_ecgs_*.csv"):
            for n in _namen_csv(pe, True):
                if n in gesehen and gesehen[n] != nr:
                    print("DUPLICATE across chunks:", n, gesehen[n], nr)
                    schlecht.append(nr)
                gesehen[n] = nr
    print("invalid:", sorted(set(schlecht)) or "none")
    return 1 if schlecht else 0


def cmd_stand(wurzel: Path) -> int:
    state = wurzel / "state"
    teile = len(stuecke(500)) if not (state / "fortschritt.json").exists() else json.loads(
        (state / "fortschritt.json").read_text())["stuecke_gesamt"]
    fertig = sorted(p.name[6:8] for p in state.glob("chunk_*.done.json"))
    print(f"done {len(fertig)} of {teile}; STOP={'yes' if (state / 'STOP.txt').exists() else 'no'}")
    return 0


def cmd_final(wurzel: Path, ziel: Path) -> int:
    state = wurzel / "state"
    man = {"datensatz": DATENSATZ, "stuecke": {}, "aufnahmen": {}}
    for mf in sorted(state.glob("chunk_*.manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        nr = m["nr"]
        man["stuecke"][nr] = [r for r in m["ids"] if m["aufnahmen"][r]["status"] == "konvertiert"]
        man["aufnahmen"].update(m["aufnahmen"])
    (state / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "logs").mkdir(exist_ok=True)
    (ziel / "logs" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    rc = L.cmd_aggregiere(DATENSATZ, wurzel, ziel, "nativ", "VV0", "AW-209")
    # pack annotations (beat markers)
    with tarfile.open(ziel / "annotations_ptbxl.tar.gz", "w:gz") as tf:
        for nr in sorted(man["stuecke"]):
            for d in (wurzel / "in" / f"chunk_{nr}").glob("annotations_*"):
                tf.add(d, arcname=f"chunk_{nr}/{d.name}")
    # check_ecg_list has a header line, errors (file name, message) does not
    for suffix, name, kopf in (("check_ecg_list_*.csv", "check_ecg_list_alle.csv", True), ("errors_*.csv", "errors_alle.csv", False)):
        with (ziel / name).open("w", encoding="utf-8", newline="") as out:
            erste = True
            for nr in sorted(man["stuecke"]):
                for f in (wurzel / "in" / f"chunk_{nr}").glob(suffix):
                    zeilen = f.read_text(encoding="utf-8").splitlines()
                    if not zeilen:
                        continue
                    if erste:
                        out.write(zeilen[0] + "\n")
                        erste = False
                    for z in zeilen[1:]:
                        out.write(z + "\n")
    return rc


def main(a: list[str]) -> int:
    if len(a) >= 4 and a[1] == "lauf":
        von = int(a[a.index("--von") + 1]) if "--von" in a else 0
        bis = int(a[a.index("--bis") + 1]) if "--bis" in a else 10**6
        return cmd_lauf(Path(a[2]), int(a[3]), von, bis)
    if len(a) == 4 and a[1] == "final":
        return cmd_final(Path(a[2]), Path(a[3]))
    if len(a) == 3 and a[1] == "pruefe":
        return cmd_pruefe(Path(a[2]))
    if len(a) == 3 and a[1] == "stand":
        return cmd_stand(Path(a[2]))
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
