"""Cross-process spill reservations for managed DuckDB connections.

Reservations are conservative upper bounds, not observed directory sizes.
Dead owners are recovered using PID and creation time, not PID alone.
Admission fails explicitly instead of waiting while holding another lease.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import uuid

import psutil


@contextmanager
def reserve_spill(path: Path, amount_gib: float, *, budget_gib: float = 96.0,
                  disk_reserve_gib: float = 10.0, ledger: Path | None = None):
    values = (amount_gib, budget_gib, disk_reserve_gib)
    if any(not math.isfinite(v) or v < 0 for v in values) or amount_gib <= 0:
        raise ValueError("Invalid spill reservation")
    root = path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # st_dev identifies the storage volume, including Windows drive boundaries.
    volume = hashlib.sha256(str(root.stat().st_dev).encode()).hexdigest()[:20]
    ledger = ledger or Path(tempfile.gettempdir()) / "vcg-spill-ledgers" / f"{volume}.sqlite"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    owner = psutil.Process()
    with closing(sqlite3.connect(ledger, timeout=10)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS leases (token TEXT PRIMARY KEY, pid INTEGER, "
                   "created REAL, amount REAL, budget REAL)")
        db.execute("BEGIN IMMEDIATE")
        for old_token, pid, created in db.execute("SELECT token,pid,created FROM leases").fetchall():
            try:
                alive = psutil.Process(pid).create_time() == created
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied:
                alive = True  # Unknown owner remains charged.
            if not alive:
                db.execute("DELETE FROM leases WHERE token=?", (old_token,))
        used, existing_budget = db.execute("SELECT COALESCE(SUM(amount),0),MIN(budget) FROM leases").fetchone()
        ceiling = min(budget_gib, existing_budget) if existing_budget is not None else budget_gib
        free_gib = shutil.disk_usage(root).free / 1024**3
        if used + amount_gib > ceiling or used + amount_gib + disk_reserve_gib > free_gib:
            raise RuntimeError(
                f"Spill admission denied: reserved={used:g} GiB, requested={amount_gib:g} GiB, "
                f"budget={ceiling:g} GiB, free={free_gib:.1f} GiB; retry after connections close")
        db.execute("INSERT INTO leases VALUES (?,?,?,?,?)",
                   (token, os.getpid(), owner.create_time(), amount_gib, ceiling))
    try:
        yield {"token": token, "reserved_gib": amount_gib, "budget_gib": ceiling,
               "ledger": str(ledger), "scope": "managed_connections_on_volume"}
    finally:
        with closing(sqlite3.connect(ledger, timeout=10)) as db, db:
            db.execute("DELETE FROM leases WHERE token=?", (token,))
