"""Which backend modules can change a result, and which only run the study.

The run identity and the marker freeze used to hash every file in this
package.  A fix to the supervisor, the run log or a release gate therefore
created a new analysis and threw away days of finished pilots or extraction
(A143, A150, A151) although no number would have changed.

``OPERATIONAL_MODULES`` lists the modules that only schedule, guard, log or
report on the work: job queue and supervisor, resource and capacity planning,
telemetry, the pilot go/no-go bookkeeping, benchmarks.  Every other module --
including any module added later -- counts as scientific, so a forgotten
classification errs towards a new identity, never towards a silent change.

A module belongs here only if no result value depends on it.  Worker counts,
memory budgets and progress reports qualify; anything that selects records,
computes a marker, fits a model or writes a result table does not.
"""

from __future__ import annotations

from pathlib import Path

from .storage import content_hash, file_sha256

CODE_SCOPE_VERSION = "1.0.0"

OPERATIONAL_MODULES: frozenset[str] = frozenset(
    {
        # Job queue, supervisor and process isolation.
        "jobs.py",
        "supervisor.py",
        "runtime_preflight.py",
        "runtime_snapshot.py",
        "workflow_side_jobs.py",
        # Resources: workers, memory, spill and disk capacity.
        "resources.py",
        "resource_grants.py",
        "spill_budget.py",
        "throughput_governor.py",
        "capacity_planning.py",
        # Telemetry, progress and runtime planning.
        "run_log.py",
        "live_progress.py",
        "runtime_calibration.py",
        # Pilot go/no-go bookkeeping (it reads results, it never makes them).
        "pilot_go_no_go.py",
        "pilot_storage_profile.py",
        # Static pre-checks and user-facing guidance.
        "preflight.py",
        "stage_exposure.py",
        "user_guidance.py",
        # Benchmarks that never start a scientific run.
        "bounded_executor_benchmark.py",
        "checkpoint_compression_benchmark.py",
        "performance_audit.py",
        "validation_performance_audit.py",
        # This classification itself.
        "code_scope.py",
    }
)


# Result-equivalence attestations (user decision 04.10.2026: a fix that keeps
# mathematics and logic must not cost a run). Each entry states that a new
# version of an analysis module produces the same results as an earlier one,
# named by both SHA-256 values and the tests that show it. The identity then
# keeps counting the earlier version. Marker-defining code is out of reach:
# ``feature_implementation_hash`` hashes its files directly, so a change there
# still moves the identity.
ATTESTATION_FILE = "result_equivalence.json"


def load_attestations(mimic_root: Path | str) -> list[dict]:
    import json

    path = Path(mimic_root) / ATTESTATION_FILE
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    required = {"module", "from_sha256", "to_sha256", "tests", "reason"}
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not required <= set(item) or not item["tests"]
        for item in entries
    ):
        raise ValueError(f"Malformed {ATTESTATION_FILE}")
    return entries


def canonical_sha256(relative: str, sha256: str, attestations: list[dict]) -> str:
    """The earliest attested-equivalent version of this file content."""

    seen = set()
    current = sha256
    while current not in seen:
        seen.add(current)
        previous = next(
            (item["from_sha256"] for item in attestations
             if item["module"] == relative and item["to_sha256"] == current),
            None,
        )
        if previous is None:
            return current
        current = previous
    raise ValueError(f"Attestation cycle for {relative}")


def _entries(root: Path, *, operational: bool) -> list[dict[str, str]]:
    attestations = [] if operational else load_attestations(root)
    entries = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if (relative in OPERATIONAL_MODULES) is operational:
            sha = file_sha256(path)
            if attestations:
                sha = canonical_sha256(relative, sha, attestations)
            entries.append({"relative_path": relative, "sha256": sha})
    return entries


def scientific_source_hash(mimic_root: Path | str) -> str:
    """Hash of every backend module that can change a result."""

    return content_hash(_entries(Path(mimic_root), operational=False))


def operational_source_hash(mimic_root: Path | str) -> str:
    """Hash of the run machinery; recorded for provenance, never identity."""

    return content_hash(_entries(Path(mimic_root), operational=True))


__all__ = [
    "ATTESTATION_FILE",
    "CODE_SCOPE_VERSION",
    "canonical_sha256",
    "load_attestations",
    "OPERATIONAL_MODULES",
    "operational_source_hash",
    "scientific_source_hash",
]
