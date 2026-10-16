"""AW-244: gemeinsame Hilfen (Speicherwaechter, Hashbindung, Ausgabe). Nachtraegliche (post hoc) Rechnungen.

Jedes Rechenskript prueft vor dem Lauf, dass sein SHA-256 gleich dem vorab festgehaltenen Stempel
(<skript>.sha256, vor dem Lauf mit `python h244.py stamp <skript>` erzeugt) ist.
Aufruf der Skripte: <local path> -I <skript> test|run
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import ctypes
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

AUSW = Path(str(data_path("b.auswertung")))
PUB = AUSW / "nachrechnung" / "AW244"      # nur Aggregate, keine Kennungen
PRIV = Path(str(data_path("b.aw244")))     # privat (Zwischenwerte)
RUN = Path(str(data_path("b.mimic_validation_runs_a155_2026_10_04_d6e70e")))
MA = Path(str(data_path("b.messaudit")))
MAXRSS = 1.4 * 1024 ** 3


class _Mem(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (k, ctypes.c_ulonglong) for k in ["total", "free", "page", "commit", "vtotal", "vfree", "extended"]]


def memory():
    m = _Mem()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"ram_free": m.free, "commit_free": m.commit}


def guard(tag=""):
    while min(memory().values()) < 1.5 * 1024 ** 3:
        print("WAIT_MEMORY", tag, flush=True)
        time.sleep(30)


_peak = [0]


def start_watch(limit=MAXRSS):
    import psutil

    def w():
        p = psutil.Process()
        while True:
            _peak[0] = max(_peak[0], p.memory_info().rss)
            if _peak[0] > limit:
                print("RSS_LIMIT", flush=True)
                os._exit(75)
            time.sleep(0.2)

    threading.Thread(target=w, daemon=True).start()


def peak():
    return _peak[0]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require_stamp(script):
    """Bricht ab, wenn das Skript nicht dem vorab festgehaltenen Hash entspricht."""
    script = Path(script)
    st = script.with_suffix(".sha256")
    if not st.exists():
        raise SystemExit(f"Stempel fehlt: {st} (erst `h244.py stamp` ausfuehren)")
    soll = st.read_text(encoding="utf8").split()[0]
    ist = sha(script)
    if soll != ist:
        raise SystemExit(f"Skripthash weicht vom Stempel ab: soll {soll} ist {ist}")
    return ist


def dump(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, default=str), encoding="utf8")


def ci(a, lo=2.5, hi=97.5, minn=0):
    import numpy as np
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) < max(minn, 2):
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def small(n, k=10):
    """Kleinzellenregel: Zaehler 1 bis 9 als '<10'."""
    return "<10" if (n is not None and 0 < n < k) else n


def stamp(script):
    script = Path(script)
    h = sha(script)
    stamp_path = script.with_suffix(".sha256")
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp_path.write_text(f"{h}  {script.name}  {utc}\n", encoding="utf8")
    print(h, script.name, utc)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "stamp":
        stamp(sys.argv[2])
    else:
        raise SystemExit("Aufruf: h244.py stamp <skript>")
