import sqlite3
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from vcg_biomarker.mimic.spill_budget import reserve_spill


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("vcg_biomarker.mimic.spill_budget.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1000 * 1024**3))
    return tmp_path / "leases.sqlite"


def test_spill_connections_share_hard_budget_and_release(tmp_path, ledger):
    with reserve_spill(tmp_path, 48, ledger=ledger):
        with reserve_spill(tmp_path, 48, ledger=ledger):
            with pytest.raises(RuntimeError, match="Spill admission denied"):
                with reserve_spill(tmp_path, 1, ledger=ledger):
                    pytest.fail("overbooked")
        with reserve_spill(tmp_path, 48, ledger=ledger):
            pass
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0


def test_spill_reservation_releases_after_exception(tmp_path, ledger):
    with pytest.raises(ValueError):
        with reserve_spill(tmp_path, 48, ledger=ledger):
            raise ValueError("query failed")
    with reserve_spill(tmp_path, 96, ledger=ledger):
        pass


def test_spill_checks_disk_reserve(tmp_path, ledger, monkeypatch):
    monkeypatch.setattr("vcg_biomarker.mimic.spill_budget.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=50 * 1024**3))
    with pytest.raises(RuntimeError, match="Spill admission denied"):
        with reserve_spill(tmp_path, 48, ledger=ledger):
            pytest.fail("disk reserve violated")


def test_other_process_cannot_spend_existing_reservation(tmp_path, ledger):
    script = """
import sys
from pathlib import Path
from vcg_biomarker.mimic.spill_budget import reserve_spill
try:
    with reserve_spill(Path(sys.argv[1]), 49, ledger=Path(sys.argv[2])):
        sys.exit(3)
except RuntimeError as error:
    assert 'Spill admission denied' in str(error)
"""
    with reserve_spill(tmp_path, 48, ledger=ledger):
        child = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(ledger)],
                               capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr


def test_dead_owner_does_not_leak_budget(tmp_path, ledger):
    with reserve_spill(tmp_path, 16, ledger=ledger):
        pass
    with sqlite3.connect(ledger) as db:
        db.execute("INSERT INTO leases VALUES ('dead', ?, 0, 96, 96)", (os.getpid(),))
    with reserve_spill(tmp_path, 96, ledger=ledger):
        pass


def test_windows_lock_release_retries_transient_reader(tmp_path, monkeypatch):
    from vcg_biomarker.mimic.storage import _catalog_writer_lock
    target = tmp_path / "catalog.lock"
    unlink = Path.unlink
    attempts = []
    def temporarily_busy(path, *args, **kwargs):
        if path == target:
            attempts.append(1)
            if len(attempts) == 1:
                raise PermissionError("Windows reader still holds file")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", temporarily_busy)
    with _catalog_writer_lock(target):
        assert target.exists()
    assert len(attempts) == 2
    assert not target.exists()
