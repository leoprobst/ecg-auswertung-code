from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .config import MimicProjectConfig
from .models import ArtifactSnapshot
from .storage import atomic_write_json, content_hash, file_sha256


SNAPSHOT_VERSION = "1.1.0"


def create_artifact_snapshot(
    config: MimicProjectConfig,
    *,
    artifact_kind: str,
    source_glob: str | Path,
    snapshot_id: Optional[str] = None,
    expected_files: Optional[int] = None,
    require_complete: bool = False,
    notes: Iterable[str] = (),
) -> ArtifactSnapshot:
    pattern = str(source_glob)
    paths = tuple(
        sorted(
            Path(item).resolve()
            for item in glob.glob(pattern, recursive=True)
            if Path(item).is_file()
        )
    )
    if not paths:
        raise FileNotFoundError(f"No artifacts match snapshot source: {pattern}")
    source_root = _common_parent(paths)
    file_hashes = {
        path.relative_to(source_root).as_posix(): _artifact_file_hash(path)
        for path in paths
    }
    input_hash = _source_input_hash(source_root)
    complete = _is_complete(
        source_root,
        file_count=len(paths),
        expected_files=expected_files,
    )
    if require_complete and not complete:
        raise RuntimeError(
            "A complete artifact was requested, but the feature stage has no "
            "matching complete manifest or the expected file count differs."
        )
    payload_for_hash = {
        "snapshot_version": SNAPSHOT_VERSION,
        "artifact_kind": artifact_kind,
        "source_root": str(source_root),
        "files": file_hashes,
        "input_hash": input_hash,
        "complete": complete,
    }
    snapshot_hash = content_hash(payload_for_hash)
    identifier = snapshot_id or (
        f"{artifact_kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{snapshot_hash[:10]}"
    )
    snapshot = ArtifactSnapshot(
        snapshot_id=identifier,
        artifact_kind=artifact_kind,
        source_root=str(source_root),
        files=tuple(file_hashes),
        file_hashes=file_hashes,
        snapshot_hash=snapshot_hash,
        complete=complete,
        input_hash=input_hash,
        notes=tuple(notes),
    )
    destination = config.snapshots_root / f"{identifier}.json"
    if destination.exists():
        existing = load_artifact_snapshot(destination)
        if existing.snapshot_hash != snapshot.snapshot_hash:
            raise FileExistsError(
                f"Snapshot ID {identifier!r} already refers to another artifact"
            )
        return existing
    atomic_write_json(destination, snapshot.to_dict())
    return snapshot


def load_artifact_snapshot(path: str | Path) -> ArtifactSnapshot:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload.pop("status", None)
    payload["files"] = tuple(payload.get("files", ()))
    payload["file_hashes"] = dict(payload.get("file_hashes", {}))
    payload["notes"] = tuple(payload.get("notes", ()))
    return ArtifactSnapshot(**payload)


def verify_artifact_snapshot(snapshot: ArtifactSnapshot) -> dict[str, object]:
    root = Path(snapshot.source_root)
    missing: list[str] = []
    changed: list[str] = []
    for relative_path, expected_hash in snapshot.file_hashes.items():
        path = root / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        if _artifact_file_hash(path) != expected_hash:
            changed.append(relative_path)
    current_hash = content_hash(
        {
            "snapshot_version": SNAPSHOT_VERSION,
            "artifact_kind": snapshot.artifact_kind,
            "source_root": snapshot.source_root,
            "files": dict(snapshot.file_hashes),
            "input_hash": snapshot.input_hash,
            "complete": snapshot.complete,
        }
    )
    return {
        "valid": not missing and not changed and current_hash == snapshot.snapshot_hash,
        "missing": missing,
        "changed": changed,
        "snapshot_hash_matches": current_hash == snapshot.snapshot_hash,
    }


def _artifact_file_hash(path: Path) -> str:
    # Snapshot integrity must follow the artifact itself. A stale sidecar must
    # never hide a changed Parquet partition.
    return file_sha256(path)


def _source_input_hash(root: Path) -> str:
    for name in ("features.manifest.json", "manifest.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("input_hash", ""))
        except (OSError, json.JSONDecodeError):
            return ""
    return ""


def _is_complete(
    root: Path,
    *,
    file_count: int,
    expected_files: Optional[int],
) -> bool:
    if root.name.startswith("smoke-"):
        return False
    if expected_files is not None and file_count != expected_files:
        return False
    manifest = root / "features.manifest.json"
    index = root / "partition_index.json"
    if manifest.is_file() and index.is_file():
        try:
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            index_payload = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        indexed = int(index_payload.get("partition_count", -1))
        parquet_count = sum(1 for path in root.rglob("part-*.parquet"))
        return (
            manifest_payload.get("status") == "complete"
            and indexed >= 0
            and indexed == parquet_count
        )
    return expected_files is not None and file_count == expected_files


def _common_parent(paths: tuple[Path, ...]) -> Path:
    common = Path(paths[0])
    if common.is_file():
        common = common.parent
    while not all(path.is_relative_to(common) for path in paths):
        common = common.parent
    if common.name.startswith("split="):
        return common.parent
    return common
