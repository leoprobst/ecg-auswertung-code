"""Ressourcenwaechter AW-243 (RECHENFENSTER): Abbruch, wenn weniger als 1,5 GB frei oder eigener RSS ueber der Grenze."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path
import os, sys, json, time, threading, hashlib
from pathlib import Path
sys.dont_write_bytecode = True
OUT = Path(str(data_path("s5.aw243")))
OUT.mkdir(parents=True, exist_ok=True)
PEAK = 0


def start(name, rss_gb=1.2, min_free_gb=1.5):
    import psutil

    def check():
        global PEAK
        PEAK = max(PEAK, psutil.Process().memory_info().rss)
        avail = psutil.virtual_memory().available
        if avail < min_free_gb * 2**30 or PEAK > rss_gb * 2**30:
            (OUT / f'{name}_stop.json').write_text(json.dumps({'pid': os.getpid(), 'reason': 'Ressourcentor', 'available': avail, 'rss': PEAK}))
            os._exit(75)
    check()

    def loop():
        while True:
            check(); time.sleep(1)
    threading.Thread(target=loop, daemon=True).start()


def peak():
    return PEAK


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
