"""Dialog automation for braveheart_batch.exe (BRAVEHEART v1.8.2, compiled).

Purpose: the compiled batch EXE asks for the ECG folder only through a Windows folder dialog
('Select a folder', uigetdir). This script starts the EXE with the MATLAB Runtime on the PATH
(only for the started process), detects the folder dialog, enters the folder path, confirms,
waits for the end and collects the outputs.

SAFETY: only the window 'Select a folder' is operated. If another visible window appears whose title matches
a security prompt (SmartScreen, user account control, Windows security) or a license dialog, NOTHING is clicked: the process
is left untouched, exit code 3, message in the log (user decision needed). Unknown windows are only logged.
Only the process tree started by this script is ended (own PID).

Call (always with -I):
  python -I braveheart_batch_auto.py --app-dir <folder with EXE+settings> --ecg-dir <folder with ECGs>
        [--timeout 3600] [--runtime "<local path> Files/MATLAB/MATLAB Runtime/v912"] [--probe-sekunden 0]
        [--log <file.json>]

Return value: 0 ok (output CSV present), 1 error/timeout/exception (log with traceback), 3 security/license dialog (STOP,
process stays alive), 4 error window in the own process tree (texts in the log, own process tree ended, nothing clicked away).
The log (JSON) contains run times, peak RSS of the process tree, all windows seen and files produced.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import win32con
import win32gui
import win32process

ORDNERDIALOG_TITEL = ("select a folder",)
# System security dialogs (foreign processes): the title begins with one of these parts (lower case).
# The German entries are window titles of the German Windows UI and are compared with data.
STOP_TITEL_GLOBAL = (
    "smartscreen", "windows protected your pc", "windows-schutz", "windows hat ihren pc",
    "user account control", "benutzerkontensteuerung", "windows security", "windows-sicherheit",
)
# Additionally for windows of the started process tree: the title contains one of these parts.
STOP_TITEL_EIGEN = STOP_TITEL_GLOBAL + (
    "sicherheitswarnung", "security warning", "license", "lizenz", "eula", "agreement",
    "do you want to allow", "möchten sie zulassen", "moechten sie zulassen",
)


def ist_stopp(titel: str, eigen: bool) -> bool:
    t = titel.lower().strip()
    if eigen:
        return any(s in t for s in STOP_TITEL_EIGEN)
    return any(t.startswith(s) for s in STOP_TITEL_GLOBAL)


def alle_fenster(pids: set[int]) -> list[dict]:
    """Visible top-level windows that belong to the given PIDs, plus desktop dialogs of foreign
    processes with a security-relevant title (UAC/SmartScreen run in foreign processes)."""
    treffer: list[dict] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return True
        titel = win32gui.GetWindowText(hwnd)
        klasse = win32gui.GetClassName(hwnd)
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            pid = -1
        eigen = pid in pids
        if eigen or ist_stopp(titel, False):
            treffer.append({"hwnd": hwnd, "pid": pid, "titel": titel, "klasse": klasse, "eigen": eigen})
        return True

    win32gui.EnumWindows(cb, None)
    return treffer


def prozessbaum(pid: int) -> set[int]:
    try:
        p = psutil.Process(pid)
        return {pid} | {c.pid for c in p.children(recursive=True)}
    except psutil.NoSuchProcess:
        return set()


def rss_baum(pids: set[int]) -> int:
    s = 0
    for q in pids:
        try:
            s += psutil.Process(q).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return s


def _dialog_offen(hwnd: int) -> bool:
    return bool(win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd))


def ordnerdialog_bedienen(hwnd: int, ordner: str, versuche: int = 4, schliess_warten: float = 12.0) -> dict:
    """Enters the path into the folder dialog window and confirms with OK.

    Supports (a) the classic 'Browse for folder' window (SHBrowseForFolder, edit field, button OK)
    and (b) the modern IFileDialog folder window (edit 'Folder:', button 'Select folder').

    Robustness: enter the path, read it back and compare; after the click the window must disappear
    within `schliess_warten` seconds, otherwise the path is entered again, read back
    and clicked, up to `versuche` times. Every attempt is in the log (hing = window stayed open after the click).
    A window that a user or another process has closed in the meantime is detected
    (return value `schon_geschlossen`); whether the correct folder was chosen then is shown by the output check.
    """
    from pywinauto import Application  # local import: probe mode without pywinauto requirement

    info: dict = {"versuche": [], "hing": 0, "schon_geschlossen": False}
    for nr in range(1, versuche + 1):
        if not _dialog_offen(hwnd):
            info["schon_geschlossen"] = True
            break
        v: dict = {"nr": nr}
        info["versuche"].append(v)
        try:
            app = Application(backend="win32").connect(handle=hwnd)
            dlg = app.window(handle=hwnd)
            dlg.set_focus()
            kinder = []
            for c in dlg.descendants():
                try:
                    kinder.append([c.friendly_class_name(), c.window_text()[:60]])
                except Exception:
                    pass
            v["kinder"] = kinder
            edits = [c for c in dlg.descendants() if c.friendly_class_name() == "Edit"]
            if not edits:
                raise RuntimeError("Folder dialog without edit field, children: " + json.dumps(kinder)[:500])
            edit = edits[0]
            edit.set_edit_text(ordner)
            time.sleep(0.4)
            ist = edit.window_text()
            v["edit_nach_eintrag"] = ist
            if os.path.normcase(os.path.normpath(ist)) != os.path.normcase(os.path.normpath(ordner)):
                raise RuntimeError(f"Edit text differs: {ist!r} != {ordner!r}")
            bestaetigen = None
            for c in [c for c in dlg.descendants() if c.friendly_class_name() == "Button"]:
                t = c.window_text().replace("&", "").strip().lower()
                if t in ("ok", "ordner auswählen", "ordner auswaehlen", "select folder", "ordner auswahlen"):
                    bestaetigen = c
                    break
            if bestaetigen is None:
                raise RuntimeError("No OK button in the folder dialog, children: " + json.dumps(kinder)[:500])
            v["knopf"] = bestaetigen.window_text()
            bestaetigen.click()
        except Exception as e:  # the window may have disappeared meanwhile: checked again below
            v["fehler"] = f"{type(e).__name__}: {e}"[:300]
        t0 = time.time()
        while time.time() - t0 < schliess_warten and _dialog_offen(hwnd):
            time.sleep(0.25)
        if not _dialog_offen(hwnd):
            v["geschlossen_nach_s"] = round(time.time() - t0, 1)
            if "fehler" not in v:
                break
            # error, but window gone: no proof that the path was right -> the output check decides
            info["schon_geschlossen"] = True
            break
        info["hing"] += 1
        time.sleep(1.0)
    info["geschlossen"] = not _dialog_offen(hwnd)
    if not info["geschlossen"]:
        raise RuntimeError("Folder dialog stayed open after all attempts: " + json.dumps(info)[:800])
    return info


# Words in window titles of the (partly German) Windows UI, compared with data
FEHLER_WORTE = ("error", "fehler", "warn", "exception", "failed", "fehlgeschlagen", "runtime", "matlab", "stopped working",
                "funktioniert nicht mehr", "antwortet nicht", "not responding")


def fenster_texte(hwnd: int) -> list[str]:
    """Labels of the child windows (message text of error windows), best effort."""
    texte: list[str] = []

    def cb(h, _):
        try:
            t = win32gui.GetWindowText(h)
            if t:
                texte.append(t[:300])
        except Exception:
            pass
        return True

    try:
        win32gui.EnumChildWindows(hwnd, cb, None)
    except Exception:
        pass
    return texte[:20]


def eigenen_baum_beenden(pid: int) -> None:
    try:
        for c in psutil.Process(pid).children(recursive=True):
            try:
                c.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.Process(pid).terminate()
    except psutil.NoSuchProcess:
        pass


def lauf(app_dir: Path, ecg_dir: Path, timeout: float, runtime: Path, probe: float, log_pfad: Path | None,
         dialog_timeout: float = 240.0) -> int:
    exe = app_dir / "braveheart_batch.exe"
    if not exe.exists():
        print("EXE missing", exe)
        return 1
    vorher = {p.name for p in ecg_dir.iterdir()}
    env = os.environ.copy()
    rt = str(runtime / "runtime" / "win64")
    env["PATH"] = rt + os.pathsep + env["PATH"]
    log: dict = {
        "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exe": str(exe), "ecg_dir": str(ecg_dir), "runtime_pfad": rt,
        "fenster": [], "fremdfenster": [], "peak_rss_mb": 0.0, "dialog_bedient": False,
    }
    t0 = time.time()
    p = None
    rc = 1
    stopgrund = None
    try:
        p = subprocess.Popen([str(exe)], cwd=str(app_dir), env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log["pid"] = p.pid
        gesehen: set[tuple] = set()
        dialog_zeit = None
        ende_zeit = None
        while True:
            jetzt = time.time() - t0
            baum = prozessbaum(p.pid)
            rss = rss_baum(baum) / 2**20
            log["peak_rss_mb"] = max(log["peak_rss_mb"], rss)
            for w in alle_fenster(baum):
                key = (w["pid"], w["titel"], w["klasse"])
                t = w["titel"].lower()
                ist_ordner = any(sx == t.strip() for sx in ORDNERDIALOG_TITEL)
                if key not in gesehen:
                    gesehen.add(key)
                    eintrag = {"t_s": round(jetzt, 1), "titel": w["titel"], "klasse": w["klasse"],
                               "pid": w["pid"], "eigen": w["eigen"]}
                    log["fenster"].append(eintrag)
                    if w["eigen"] and not ist_ordner and not (w["titel"] == "" and w["klasse"] == "Name"):
                        fe = dict(eintrag, texte=fenster_texte(w["hwnd"]))
                        log["fremdfenster"].append(fe)
                        # error notice or system dialog (#32770) in the own process tree: the chunk counts as failed
                        if w["klasse"] == "#32770" or any(x in t for x in FEHLER_WORTE):
                            stopgrund = "FEHLERFENSTER:" + json.dumps(fe, ensure_ascii=True)[:600]
                if ist_stopp(w["titel"], w["eigen"]):
                    stopgrund = f"Security/license window: {w['titel']!r} (class {w['klasse']}, pid {w['pid']})"
                    break
                if stopgrund and stopgrund.startswith("FEHLERFENSTER"):
                    break
                if w["eigen"] and not log["dialog_bedient"] and probe <= 0 and ist_ordner:
                    log["ordnerdialog"] = ordnerdialog_bedienen(w["hwnd"], str(ecg_dir))
                    log["dialog_bedient"] = True
                    dialog_zeit = time.time()
                    log["t_dialog_s"] = round(dialog_zeit - t0, 1)
            if stopgrund:
                log["stopp"] = stopgrund
                rc = 4 if stopgrund.startswith("FEHLERFENSTER") else 3
                break
            if p.poll() is not None:
                ende_zeit = time.time()
                log["exit_code_prozess"] = p.returncode
                rc = 0
                break
            if probe > 0 and jetzt > probe:
                log["probe_ende"] = True
                rc = 0
                break
            if probe <= 0 and not log["dialog_bedient"] and jetzt > dialog_timeout:
                log["stopp"] = f"no folder dialog after {dialog_timeout:.0f} s"
                rc = 1
                break
            if jetzt > timeout:
                log["stopp"] = "Timeout"
                rc = 1
                break
            time.sleep(0.25)
        if dialog_zeit and ende_zeit:
            log["laufzeit_nach_dialog_s"] = round(ende_zeit - dialog_zeit, 1)
    except Exception:
        import traceback
        log["exception"] = traceback.format_exc()[-3000:]
        rc = 1
    finally:
        # only the own process tree, only if still running; on a security/license window (rc 3) the user decides
        if p is not None and p.poll() is None and not (stopgrund and not stopgrund.startswith("FEHLERFENSTER")):
            eigenen_baum_beenden(p.pid)
    log["laufzeit_gesamt_s"] = round(time.time() - t0, 1)
    neu = sorted(p2.name for p2 in ecg_dir.iterdir() if p2.name not in vorher)
    log["neue_ausgaben"] = neu
    if rc == 0 and probe <= 0 and not any(n.startswith("processed_ecgs_") for n in neu):
        log["fehler"] = "no processed_ecgs_* file produced"
        rc = 1
    log["rc"] = rc
    if log_pfad:
        log_pfad.write_text(json.dumps(log, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=True))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-dir", required=True, type=Path)
    ap.add_argument("--ecg-dir", required=True, type=Path)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--runtime", type=Path, default=Path(str(data_path("pub12.matlab_runtime_dir"))))
    ap.add_argument("--probe-sekunden", type=float, default=0, help="observe only (no click), then end")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--dialog-timeout", type=float, default=240.0, help="seconds until the folder dialog, otherwise abort (rc 1)")
    a = ap.parse_args()
    return lauf(a.app_dir.resolve(), a.ecg_dir.resolve(), a.timeout, a.runtime, a.probe_sekunden, a.log,
                a.dialog_timeout)


if __name__ == "__main__":
    sys.exit(main())
