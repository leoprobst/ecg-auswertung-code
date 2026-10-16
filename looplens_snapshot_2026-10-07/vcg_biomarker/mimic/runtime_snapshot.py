"""Private source tree for a supervisor session, independent of GUI edits.

Dependencies remain in the launching Python environment: do not upgrade that
environment during production. This snapshot is not an environment container.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def create_runtime_snapshot(source_root: Path, snapshots_root: Path) -> Path:
    source_root = source_root.resolve()
    snapshots_root = snapshots_root.resolve()
    package = source_root / "vcg_biomarker"
    if not package.is_dir():
        raise ValueError("Missing application package")
    if snapshots_root.is_relative_to(package):
        raise ValueError("Runtime destination must not be inside the package")
    paths = sorted(p for p in package.rglob("*") if p.is_file()
                   and not any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                               for part in p.relative_to(package).parts)
                   and p.suffix not in {".pyc", ".pyo"})
    paths += [p for p in (source_root / "pyproject.toml",
                         source_root / "requirements-lock-win-py312.txt") if p.is_file()]
    snapshots_root.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="session-", dir=snapshots_root))
    hashes = {}
    for source in paths:
        if source.is_symlink():
            raise ValueError(f"Symlink not allowed in runtime: {source}")
        relative = source.relative_to(source_root)
        data = source.read_bytes()
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise OSError(f"Runtime copy verification failed: {relative}")
        hashes[relative.as_posix()] = digest
    # Reject a source edit while copying; never launch a mixed generation.
    for relative, digest in hashes.items():
        if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during runtime snapshot; retry when editing is idle")
    (target / "runtime_manifest.json").write_text(json.dumps({
        "version": 1, "source_root": str(source_root), "files": hashes,
        "dependency_isolation": False,
    }, indent=2), encoding="utf-8")
    return target


def runtime_environment() -> dict[str, str]:
    """Do not let an inherited development PYTHONPATH override the snapshot."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env
