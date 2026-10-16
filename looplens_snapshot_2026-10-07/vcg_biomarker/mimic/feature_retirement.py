"""Retire development feature artifacts that a new panel has superseded.

The evidence layer refuses to rebuild while a complete feature artifact is
bound to it, and rightly so: rebuilding underneath finished features would
leave results that no longer correspond to the data they claim to describe.
The guard offers one escape, an exact equivalence proof, which rebuilds the
entire evidence into a shadow root and compares it table by table.

That proof is the right answer when the evidence really is unchanged. It is
the wrong answer when the *features* are what became obsolete -- when the
marker panel itself has moved on. Then nothing about the old artifacts is
worth protecting, and paying an hour of shadow rebuild to prove that the
evidence still matches artifacts nobody will read again is effort spent in
the wrong place.

This module is the other door. It moves superseded artifacts into the archive
instead of deleting them, records why, and leaves the feature root clean so
the evidence can rebuild normally.

Two things it will not do:

* touch anything containing locked-test features -- the sealed split is not
  archived, moved or otherwise handled by convenience tooling;
* run without a stated reason, because "why was this retired" is exactly the
  question a later reader of the archive will have.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from .config import MimicProjectConfig


FEATURE_RETIREMENT_VERSION = "1.0.0"


@dataclass(frozen=True)
class RetiredFeatureArtifact:
    panel_key: str
    scope: str
    archive_path: Path
    size_bytes: int
    partition_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "panel_key": self.panel_key,
            "scope": self.scope,
            "archive_path": str(self.archive_path),
            "size_bytes": self.size_bytes,
            "size_gib": round(self.size_bytes / 2**30, 3),
            "partition_count": self.partition_count,
        }


def bound_development_artifacts(
    config: MimicProjectConfig,
) -> list[tuple[str, str, Path]]:
    """Every development artifact that currently binds the evidence layer.

    Mirrors what ``evidence._has_bound_development_features`` looks for, so
    what this lists is exactly what has to go for the guard to stand down.
    """

    found: list[tuple[str, str, Path]] = []
    for manifest in sorted(
        config.features_root.glob("*/*/features.manifest.json")
    ):
        scope_dir = manifest.parent
        found.append((scope_dir.parent.name, scope_dir.name, scope_dir))
    for pattern in ("*/*/split=*/part-*.parquet", "*/*/split=*/.part-*.resume.parquet"):
        for part in config.features_root.glob(pattern):
            scope_dir = part.parent.parent
            entry = (scope_dir.parent.name, scope_dir.name, scope_dir)
            if entry not in found:
                found.append(entry)
    return sorted(found)


def _contains_locked_test(scope_dir: Path) -> bool:
    if any(scope_dir.glob("split=locked_test*")):
        return True
    manifest = scope_dir / "features.manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Unreadable provenance is not permission to move the artifact.
        return True
    notes = " ".join(str(note) for note in payload.get("notes", ()))
    return "locked_test_features_included=True" in notes


def _directory_size(path: Path) -> tuple[int, int]:
    size = 0
    partitions = 0
    for item in path.rglob("*"):
        if item.is_file():
            size += item.stat().st_size
            if item.suffix == ".parquet":
                partitions += 1
    return size, partitions


def retire_superseded_development_features(
    config: MimicProjectConfig,
    *,
    reason: str,
    scopes: Optional[Iterable[tuple[str, str]]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Move superseded feature artifacts into the archive.

    ``scopes`` restricts the operation to specific ``(panel_key, scope)``
    pairs; the default retires every bound development artifact. Nothing is
    deleted: the archive keeps the files and a manifest saying what was moved,
    when, and why.
    """

    if not str(reason).strip():
        raise ValueError(
            "Retiring feature artifacts requires a stated reason; it is the "
            "first thing a later reader of the archive will ask."
        )
    requested = {tuple(item) for item in scopes} if scopes is not None else None
    archived_at = datetime.now(timezone.utc)
    archive_root = (
        config.work_root
        / "archives"
        / "superseded-features"
        / archived_at.strftime("%Y%m%dT%H%M%SZ")
    )

    retired: list[RetiredFeatureArtifact] = []
    skipped: list[dict[str, str]] = []
    for panel_key, scope, scope_dir in bound_development_artifacts(config):
        if requested is not None and (panel_key, scope) not in requested:
            continue
        if _contains_locked_test(scope_dir):
            skipped.append(
                {
                    "panel_key": panel_key,
                    "scope": scope,
                    "reason": "contains_or_may_contain_locked_test_features",
                }
            )
            continue
        size, partitions = _directory_size(scope_dir)
        target = archive_root / panel_key / scope
        target.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(
                f"Archiviere {panel_key}/{scope} "
                f"({size / 2**30:.2f} GiB, {partitions} Partitionen)"
            )
        shutil.move(str(scope_dir), str(target))
        retired.append(
            RetiredFeatureArtifact(
                panel_key=panel_key,
                scope=scope,
                archive_path=target,
                size_bytes=size,
                partition_count=partitions,
            )
        )

    summary = {
        "archived_at": archived_at.isoformat(),
        "reason": str(reason),
        "archive_root": str(archive_root),
        "retired": [item.to_dict() for item in retired],
        "skipped": skipped,
        "total_bytes": sum(item.size_bytes for item in retired),
        "note_en": (
            "Moved, not deleted. The evidence layer can rebuild once no "
            "complete feature artifact binds it; these files remain readable "
            "here as a record of what the previous panel produced."
        ),
        "feature_retirement_version": FEATURE_RETIREMENT_VERSION,
    }
    if retired:
        archive_root.mkdir(parents=True, exist_ok=True)
        (archive_root / "retirement.manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


__all__ = [
    "FEATURE_RETIREMENT_VERSION",
    "RetiredFeatureArtifact",
    "bound_development_artifacts",
    "retire_superseded_development_features",
]
