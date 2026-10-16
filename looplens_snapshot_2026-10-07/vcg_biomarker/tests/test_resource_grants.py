import os
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic import resource_grants as grants
from vcg_biomarker.mimic.storage import atomic_write_json


@pytest.fixture(autouse=True)
def _no_job_environment(monkeypatch):
    # ``mimic_job_worker._apply_job_resource_environment`` writes the job's
    # CPU and memory caps into ``os.environ`` on purpose (it runs once per
    # worker process). Inside the test process another test's call leaves
    # them behind, and DuckDBCatalog then caps threads and memory with them.
    for name in ("VCG_JOB_CPU_TOKENS", "VCG_JOB_MEMORY_GB",
                 "VCG_JOB_MEMORY_ALLOCATION_GB", "VCG_JOB_OS_RESERVE_GB"):
        monkeypatch.delenv(name, raising=False)


def record():
    return SimpleNamespace(worker_pid=os.getpid(), spec=SimpleNamespace(
        job_id="test/unsafe-path", resource_profile=SimpleNamespace(
            cpu_tokens=6, memory_gb=12.0, interactive=False)))


def test_grant_revoke_remains_charged_until_worker_acknowledges(tmp_path, monkeypatch):
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    payload = {"owner_pid": job.worker_pid, "workers": 6, "ceiling": 15,
               "generation": 2, "unacknowledged_workers": 10, "sampled_at": grants.time.time()}
    coordinator.grants[job.spec.job_id] = payload
    path = coordinator.path(job.spec.job_id)
    assert path.parent == tmp_path
    atomic_write_json(path, payload)
    monkeypatch.setenv(grants.GRANT_ENV, str(path))
    assert coordinator.charged_workers(job) == 10
    assert grants.worker_ceiling(6) == 15
    assert grants.granted_workers(15, resident_workers=9) == 6
    assert coordinator.charged_workers(job) == 9
    grants.granted_workers(15, resident_workers=0)
    assert coordinator.charged_workers(job) == 6


def test_stale_or_child_grants_cannot_expand(tmp_path, monkeypatch):
    path = tmp_path / "grant.json"
    monkeypatch.setenv(grants.GRANT_ENV, str(path))
    assert grants.worker_ceiling(15) == 1
    atomic_write_json(path, {"owner_pid": os.getpid() + 1, "workers": 15,
                            "ceiling": 15, "generation": 1, "sampled_at": grants.time.time()})
    assert grants.granted_workers(15) == 1
    atomic_write_json(path, {"owner_pid": os.getpid(), "workers": 15,
                            "ceiling": 15, "generation": 1, "sampled_at": grants.time.time() - 30})
    assert grants.granted_workers(15) == 1


def test_restart_preserves_expired_reservation_until_ack(tmp_path, monkeypatch):
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    path = coordinator.path(job.spec.job_id)
    atomic_write_json(path, {"owner_pid": job.worker_pid, "job_id": job.spec.job_id,
                            "workers": 6, "ceiling": 15, "generation": 8,
                            "unacknowledged_workers": 12, "sampled_at": 0})
    assert coordinator.charged_workers(job) == 12
    coordinator.sample([job], queued=True, available_gb=20)
    assert coordinator.charged_workers(job) == 12
    monkeypatch.setenv(grants.GRANT_ENV, str(path))
    grants.granted_workers(15, resident_workers=6)
    assert coordinator.charged_workers(job) == 6


def test_non_object_grant_fails_closed(tmp_path):
    path = tmp_path / "grant.json"
    atomic_write_json(path, [1, 2])
    assert grants.read_grant(path) == {}


def test_ui_displays_current_budget_not_asserted_active_workers(tmp_path):
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    atomic_write_json(coordinator.path(job.spec.job_id), {
        "owner_pid": job.worker_pid, "workers": 6, "ceiling": 15,
        "generation": 2, "sampled_at": grants.time.time(),
        "limiting_reason": "ram_pressure_drain",
    })
    label = grants.grant_status_text(tmp_path, job.spec.job_id, job.worker_pid)
    assert "6/15" in label and "RAM-Druck" in label
    assert "kein aktueller Messnachweis" in grants.grant_status_text(
        tmp_path, job.spec.job_id, job.worker_pid + 1)


def test_three_stable_samples_expand_and_pressure_drains_one(tmp_path, monkeypatch):
    import psutil
    clock = {"now": 100.0}
    monkeypatch.setattr(grants.time, "time", lambda: clock["now"])
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kwargs: 0.0)
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    for index in range(4):
        clock["now"] += 2
        coordinator.sample([job], queued=False, available_gb=20)
        if index < 3:
            assert coordinator.grants[job.spec.job_id]["workers"] == 6
    assert coordinator.grants[job.spec.job_id]["workers"] == 7
    clock["now"] += 2
    coordinator.sample([job], queued=False, available_gb=1.5)
    assert coordinator.grants[job.spec.job_id]["workers"] == 6
    assert coordinator.grants[job.spec.job_id]["limiting_reason"] == "ram_pressure_drain"
    assert coordinator.charged_workers(job) == 7
    assert coordinator.drain_in_progress(job, 1.5)
    assert not coordinator.drain_in_progress(job, 0.5)
    clock["now"] += 2
    coordinator.sample([job], queued=True, available_gb=20)
    assert coordinator.grants[job.spec.job_id]["workers"] == 6
    assert not coordinator.drain_in_progress(job, 20)


def test_drain_grace_cannot_delay_emergency_pause_indefinitely(tmp_path, monkeypatch):
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    coordinator.grants[job.spec.job_id] = {"workers": 5, "generation": 1,
        "unacknowledged_workers": 6, "pressure_started_at": 100.0}
    monkeypatch.setattr(grants.time, "time", lambda: 129.0)
    assert coordinator.drain_in_progress(job, 1.5)
    monkeypatch.setattr(grants.time, "time", lambda: 130.0)
    assert not coordinator.drain_in_progress(job, 1.5)


def test_newly_started_worker_cpu_counts_as_own_not_foreign(tmp_path, monkeypatch):
    """E3: a process born between two samples is the job's load."""
    import psutil

    clock = {"now": 1_000.0}
    monkeypatch.setattr(grants.time, "time", lambda: clock["now"])
    busy = {"percent": 0.0}
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kwargs: busy["percent"])
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 16)

    class _Proc:
        def __init__(self, pid, created, cpu, private_gb=0.26):
            self.pid, self._created, self._cpu, self._private = pid, created, cpu, private_gb

        def create_time(self):
            return self._created

        def cpu_times(self):
            return SimpleNamespace(user=self._cpu, system=0.0)

        def memory_info(self):
            return SimpleNamespace(rss=int(self._private * 1024**3), private=int(self._private * 1024**3))

        def children(self, recursive=True):
            return list(tree["children"])

    parent = _Proc(os.getpid(), 10.0, 5.0)
    tree = {"children": [_Proc(111, 20.0, 1.0)]}
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    coordinator.sample([job], queued=False, available_gb=20)
    # 2 s later: the old child used 2 s, a new child (born now) used 2 s of
    # import; the whole machine was busy with exactly those 4 core-seconds.
    clock["now"] += 2
    tree["children"] = [_Proc(111, 20.0, 3.0), _Proc(222, clock["now"] - 1.5, 2.0)]
    busy["percent"] = 100.0 * (4.0 / 2.0) / 16
    coordinator.sample([job], queued=False, available_gb=20)
    grant = coordinator.grants[job.spec.job_id]
    assert grant["cpu_own_cores"] == 2.0
    assert grant["cpu_foreign_cores"] == 0.0
    # Measured worker memory replaces the old 2 GB floor.
    assert abs(grant["memory_per_worker_gb"] - 0.26 * grants.WORKER_MEMORY_SAFETY) < 1e-9


def test_foreign_load_uses_the_median_of_ten_seconds(tmp_path, monkeypatch):
    import psutil

    clock = {"now": 100.0}
    monkeypatch.setattr(grants.time, "time", lambda: clock["now"])
    loads = iter([0.0, 0.0, 0.0, 100.0, 0.0, 0.0])
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kwargs: next(loads))
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    job = record()
    for _ in range(6):
        clock["now"] += 2
        coordinator.sample([job], queued=False, available_gb=20)
    # One sample with the whole machine busy did not cut the grant.
    assert coordinator.grants[job.spec.job_id]["limiting_reason"] == "cpu_or_ram_budget"
    assert coordinator.grants[job.spec.job_id]["cpu_foreign_cores"] < 1.0


def test_worker_memory_share_falls_back_to_profile_without_workers(tmp_path):
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)
    assert coordinator.memory_per_worker_gb("job", configured=2.0) == 2.0
    coordinator.worker_memory_history["job"] = grants.deque([0.3, 0.27])
    assert abs(coordinator.memory_per_worker_gb("job", configured=2.0) - 0.36) < 1e-9


def test_duckdb_threads_follow_a_live_own_grant_only_when_asked(tmp_path, monkeypatch):
    """R12: analysis connections use the grant, explicit caps stay caps."""
    import duckdb

    from vcg_biomarker.mimic.storage import DuckDBCatalog

    path = tmp_path / "grant.json"
    monkeypatch.setenv(grants.GRANT_ENV, str(path))
    following = DuckDBCatalog(tmp_path / "cat.duckdb", memory_limit_gb=1, threads=7, follow_grant=True)
    capped = DuckDBCatalog(tmp_path / "cat.duckdb", memory_limit_gb=1, threads=2)
    # No grant yet: the configured cap applies.
    assert following.effective_threads() == 7
    atomic_write_json(path, {"owner_pid": os.getpid(), "workers": 12, "ceiling": 15,
                             "generation": 1, "sampled_at": grants.time.time()})
    assert following.effective_threads() == 12
    assert capped.effective_threads() == 2
    with following.connect(isolated=True) as connection:
        assert int(connection.execute("SELECT current_setting('threads')").fetchone()[0]) == 12
    # Someone else's grant, or an expired one, never raises the thread count.
    atomic_write_json(path, {"owner_pid": os.getpid() + 1, "workers": 12, "ceiling": 15,
                             "generation": 1, "sampled_at": grants.time.time()})
    assert following.effective_threads() == 7
    assert duckdb is not None


def test_e8_fair_shares_are_max_min_fair():
    shares = grants.fair_cpu_shares({"study": 15.0, "cohorts": 15.0}, 14)
    assert shares == {"cohorts": 7, "study": 7}
    # A job that measurably uses 2 cores keeps 3; the other gets the rest.
    shares = grants.fair_cpu_shares({"study": 15.0, "serial": 3.0}, 14)
    assert shares == {"serial": 3, "study": 11}
    assert grants.fair_cpu_shares({"a": 15.0, "b": 15.0, "c": 15.0}, 2) == {"a": 1, "b": 1, "c": 1}


def test_e8_two_running_jobs_both_grow_to_a_fair_share(tmp_path, monkeypatch):
    import psutil

    clock = {"now": 100.0}
    monkeypatch.setattr(grants.time, "time", lambda: clock["now"])
    monkeypatch.setattr(psutil, "cpu_count", lambda *a, **k: 16)
    usage = {101: 7.0, 102: 7.0}  # cores each job keeps busy
    monkeypatch.setattr(
        psutil, "cpu_percent",
        lambda **kwargs: 100.0 * sum(usage.values()) / 16)

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 1.0

        def cpu_times(self):
            return SimpleNamespace(user=usage[self.pid] * clock["now"], system=0.0)

        def children(self, recursive=True):
            return []

    monkeypatch.setattr(psutil, "Process", _Proc)
    coordinator = grants.ResourceGrantCoordinator(tmp_path, 15, 2)

    def job(name, pid):
        return SimpleNamespace(worker_pid=pid, spec=SimpleNamespace(
            job_id=name, resource_profile=SimpleNamespace(
                cpu_tokens=6, memory_gb=12.0, interactive=False)))

    study, cohorts = job("study", 101), job("cohorts", 102)
    for _ in range(6):
        clock["now"] += 2
        coordinator.sample([study, cohorts], queued=False, available_gb=20)
    for name in ("study", "cohorts"):
        grant = coordinator.grants[name]
        assert grant["workers"] == 7
        assert grant["limiting_reason"] == "fair_share_between_jobs"
        assert grant["ceiling"] == 15
    # A serial job that uses one core gives its share to the busy one.
    usage[102] = 1.0
    for _ in range(12):
        clock["now"] += 2
        coordinator.sample([study, cohorts], queued=False, available_gb=40)
    assert coordinator.grants["cohorts"]["workers"] <= 6
    assert coordinator.grants["study"]["workers"] >= 9
    # A queued job still makes both hold at their admission tokens.
    clock["now"] += 2
    coordinator.sample([study, cohorts], queued=True, available_gb=20)
    assert coordinator.grants["study"]["workers"] == 6


def test_o3_invalid_grant_duration_is_tracked_and_reported(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import workflow

    path = tmp_path / "grant.json"
    monkeypatch.setenv(grants.GRANT_ENV, str(path))
    clock = {"now": 1000.0}
    monkeypatch.setattr(grants.time, "monotonic", lambda: clock["now"])
    grants._GRANT_INVALID_SINCE[0] = None
    assert grants.granted_workers(8) == 1          # no grant file: invalid
    clock["now"] += 90
    assert grants.invalid_grant_seconds() == 90
    import time as real_time
    monkeypatch.setattr(real_time, "monotonic", lambda: clock["now"])
    workflow._GRANT_WARNING_STATE["last"] = 0.0
    assert workflow._with_grant_warning("Schritt 5").startswith("Warnung")
    assert workflow._with_grant_warning("Schritt 5") == "Schritt 5"  # throttled
    atomic_write_json(path, {"owner_pid": os.getpid(), "workers": 4, "ceiling": 8,
                             "generation": 1, "sampled_at": grants.time.time()})
    assert grants.granted_workers(8) == 4
    assert grants.invalid_grant_seconds() is None


def test_r8_duckdb_limit_follows_available_memory_only_when_managed(tmp_path, monkeypatch):
    from vcg_biomarker.mimic import resources
    from vcg_biomarker.mimic.storage import DuckDBCatalog

    catalog = DuckDBCatalog(tmp_path / "c.duckdb", memory_limit_gb=8, threads=2, follow_grant=True)
    fixed = DuckDBCatalog(tmp_path / "c.duckdb", memory_limit_gb=8, threads=2)
    monkeypatch.delenv(grants.GRANT_ENV, raising=False)
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 4.0)
    assert catalog.effective_memory_limit_gb() == 8.0          # unmanaged: unchanged
    monkeypatch.setenv(grants.GRANT_ENV, str(tmp_path / "g.json"))
    monkeypatch.setenv("VCG_JOB_OS_RESERVE_GB", "2")
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 7.5)
    assert catalog.effective_memory_limit_gb() == 4.5          # 7.5 - 2 - 1
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 3.0)
    assert catalog.effective_memory_limit_gb() == 2.0          # floor
    monkeypatch.setattr(resources, "available_memory_gb", lambda: 40.0)
    assert catalog.effective_memory_limit_gb() == 8.0          # never above the configured
    assert fixed.effective_memory_limit_gb() == 8.0            # explicit caps stay caps
    with catalog.connect(isolated=True) as connection:
        limit = connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    assert limit.startswith("7.") or "GiB" in limit or "GB" in limit
