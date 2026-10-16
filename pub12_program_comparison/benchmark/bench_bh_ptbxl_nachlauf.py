"""Follow-up run of the PTB-XL arm: waits until all chunks are done, checks them and aggregates to the target folder.

Call (always with -I):
  bench_bh_ptbxl_nachlauf.py <root> <target> <number_of_chunks> <driver_pid>

Procedure: look every 60 s. End without aggregation if state/STOP.txt exists or the driver process
has ended without all chunks being done (then resume the driver). If all chunks are done:
cmd_pruefe; on a deviation no aggregation (message in state/nachlauf.log); otherwise RAM check and entry
(aggregation needs about 1.3 GB), cmd_final to <target>, release.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repo_config import data_path

import subprocess
import sys
import time
from pathlib import Path

import psutil

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import bench_bh_ptbxl as P  # noqa: E402

RAMTOOL = str(data_path("pub12.scanner_v10_1_tools_k_coord_ram_reservierung_py"))


def log(state: Path, text: str) -> None:
    with (state / "nachlauf.log").open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + text + chr(10))


def main(a: list[str]) -> int:
    wurzel, ziel, n, pid = Path(a[1]), Path(a[2]), int(a[3]), int(a[4])
    state = wurzel / "state"
    log(state, f"follow-up run started, waiting for {n} chunks, driver PID {pid}")
    while True:
        fertig = len(list(state.glob("chunk_*.done.json")))
        if fertig >= n:
            break
        if (state / "STOP.txt").exists() or not psutil.pid_exists(pid):
            log(state, f"end without aggregation: done {fertig} of {n}, STOP={(state / 'STOP.txt').exists()}, "
                       f"driver alive={psutil.pid_exists(pid)}")
            return 1
        time.sleep(60)
    log(state, "all chunks done, checking")
    if P.cmd_pruefe(wurzel) != 0:
        log(state, "check found a deviation, no aggregation")
        return 2
    while subprocess.run([sys.executable, RAMTOOL, "pruefen", "--gb", "2", "--seite", "analyse"],
                         capture_output=True).returncode != 0:
        time.sleep(120)
    r = subprocess.run([sys.executable, RAMTOOL, "eintragen", "--gb", "2", "--seite", "analyse", "--sitzung",
                        "benchmark PTB-XL aggregation", "--pid", str(__import__('os').getpid()), "--stunden", "1",
                        "--zweck", "PTB-XL aggregation"], capture_output=True, text=True)
    rid = r.stdout.strip().split()[-1] if r.returncode == 0 and r.stdout.strip() else None
    try:
        rc = P.cmd_final(wurzel, ziel)
        log(state, f"aggregation rc={rc}")
    finally:
        if rid:
            subprocess.run([sys.executable, RAMTOOL, "freigeben", "--id", rid], capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
