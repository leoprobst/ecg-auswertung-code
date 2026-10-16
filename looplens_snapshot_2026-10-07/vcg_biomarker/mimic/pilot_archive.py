from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import MimicProjectConfig
from .jobs import JOB_PROJECT_CONFIG_HASH_KEY, JobStore
from .models import JobState
from .storage import atomic_write_json, atomic_write_parquet, content_hash


PILOT_ARCHIVE_VERSION = "1.0.0"
_ARCHIVABLE_STATES = {
    JobState.PAUSED,
    JobState.CANCELLED,
    JobState.FAILED,
    JobState.BLOCKED,
}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PilotArchiveResult:
    archive_id: str
    job_id: str
    source_root: str
    artifact_root: str
    manifest_path: str
    inventory_path: str
    completed_rows: int
    completed_partitions: int
    checkpoint_rows: int
    total_bytes: int
    scientific_role: str = "historical_partial_pilot_hypothesis_only"
    implementation_compatible: bool = False
    locked_test_read: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def archive_partial_feature_pilot(
    config: MimicProjectConfig,
    job_id: str,
    *,
    reason: str,
) -> PilotArchiveResult:
    """Close and preserve a partial development extraction without copying it."""

    store = JobStore.for_project(config)
    record = store.get(job_id)
    if record.spec.stage != "extract-features":
        raise ValueError("Only feature-extraction jobs can be archived as pilots")
    if record.state not in _ARCHIVABLE_STATES:
        raise ValueError(
            "Pilot archiving requires a paused or terminal non-running job; "
            f"got {record.state.value}"
        )
    arguments = dict(record.spec.arguments)
    if bool(arguments.get("include_locked_test", False)) or bool(
        arguments.get("locked_test_only", False)
    ):
        raise PermissionError("Locked-test feature artifacts cannot be pilot archives")
    panel_ids = tuple(str(value) for value in arguments.get("panel_ids", ()))
    if not panel_ids or any(not _SAFE_COMPONENT.fullmatch(value) for value in panel_ids):
        raise ValueError("The feature job has an invalid or missing panel list")
    max_records = arguments.get("max_records")
    scope = (
        f"smoke-development-{int(max_records)}"
        if max_records is not None
        else "development"
    )
    panel_key = "_".join(panel_ids)
    source_root = (config.features_root / panel_key / scope).resolve()
    features_root = config.features_root.resolve()
    if source_root.parent.parent != features_root:
        raise RuntimeError("Resolved pilot artifacts are outside the feature root")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Pilot feature root does not exist: {source_root}")

    inventory = _partition_inventory(source_root)
    completed_rows = int(
        pd.to_numeric(
            inventory.loc[inventory["artifact_kind"].eq("partition"), "row_count"],
            errors="coerce",
        ).fillna(0).sum()
    )
    checkpoint_rows = int(
        pd.to_numeric(
            inventory.loc[inventory["artifact_kind"].eq("checkpoint"), "row_count"],
            errors="coerce",
        ).fillna(0).sum()
    )
    completed_partitions = int(
        inventory["artifact_kind"].eq("partition").sum()
    )
    total_bytes = int(inventory["size_bytes"].fillna(0).sum())
    if completed_rows == 0 and checkpoint_rows == 0:
        raise RuntimeError("The selected job has no completed pilot rows to archive")

    archived_at = datetime.now(timezone.utc)
    archive_id = (
        f"{archived_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{_safe_archive_component(job_id)}"
    )
    archive_root = (
        config.work_root / "archives" / "feature-pilots" / archive_id
    ).resolve()
    expected_parent = (
        config.work_root / "archives" / "feature-pilots"
    ).resolve()
    if archive_root.parent != expected_parent or archive_root.exists():
        raise RuntimeError(f"Unsafe or existing pilot archive target: {archive_root}")
    archive_root.mkdir(parents=True, exist_ok=False)
    artifact_root = archive_root / "artifacts"

    state_before = record.state.value
    if record.state == JobState.PAUSED:
        record = store.request_cancel(job_id)
    payload = {
        "schema_version": PILOT_ARCHIVE_VERSION,
        "archive_id": archive_id,
        "archived_at": archived_at.isoformat(),
        "reason": str(reason).strip(),
        "scientific_role": "historical_partial_pilot_hypothesis_only",
        "implementation_compatible": False,
        "formal_analysis_allowed": False,
        "freeze_allowed": False,
        "locked_test_read": False,
        "source_job": {
            "job_id": job_id,
            "state_before_archive": state_before,
            "state_after_archive": record.state.value,
            "progress_current": int(record.progress_current),
            "progress_total": int(record.progress_total),
            "progress_message": record.progress_message,
            "created_at": record.spec.created_at,
            "started_at": record.started_at,
            "project_config_hash": str(
                arguments.get(JOB_PROJECT_CONFIG_HASH_KEY, "")
            ),
            "arguments": _public_arguments(arguments),
        },
        "source_root": str(source_root),
        "artifact_root": str(artifact_root),
        "completed_rows": completed_rows,
        "completed_partitions": completed_partitions,
        "checkpoint_rows": checkpoint_rows,
        "total_bytes": total_bytes,
        "archive_hash": "",
    }
    payload["archive_hash"] = content_hash(payload)

    try:
        os.replace(source_root, artifact_root)
        inventory_path = archive_root / "partition_inventory.parquet"
        manifest_path = archive_root / "pilot_archive.json"
        atomic_write_parquet(inventory, inventory_path)
        atomic_write_json(manifest_path, payload)
    except Exception:
        if artifact_root.exists() and not source_root.exists():
            os.replace(artifact_root, source_root)
        raise

    store.append_event(
        job_id,
        "pilot_archive",
        "Partial development artifacts archived as a historical pilot",
        {
            "archive_id": archive_id,
            "manifest_path": str(manifest_path),
            "completed_rows": completed_rows,
            "locked_test_read": False,
        },
    )
    return PilotArchiveResult(
        archive_id=archive_id,
        job_id=job_id,
        source_root=str(source_root),
        artifact_root=str(artifact_root),
        manifest_path=str(manifest_path),
        inventory_path=str(inventory_path),
        completed_rows=completed_rows,
        completed_partitions=completed_partitions,
        checkpoint_rows=checkpoint_rows,
        total_bytes=total_bytes,
    )


def _partition_inventory(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, pattern in (
        ("partition", "split=*/part-*.json"),
        ("checkpoint", "split=*/.part-*.resume.json"),
    ):
        for manifest_path in sorted(root.glob(pattern)):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                payload = {}
            parquet_path = (
                manifest_path.with_suffix(".parquet")
                if kind == "partition"
                else manifest_path.with_name(
                    manifest_path.name.removesuffix(".json") + ".parquet"
                )
            )
            rows.append(
                {
                    "artifact_kind": kind,
                    "relative_parquet_path": str(
                        parquet_path.relative_to(root)
                    ),
                    "relative_manifest_path": str(
                        manifest_path.relative_to(root)
                    ),
                    "split": manifest_path.parent.name.removeprefix("split="),
                    "row_count": int(payload.get("row_count", 0) or 0),
                    "part_hash": str(payload.get("part_hash", "")),
                    "parquet_sha256": str(payload.get("parquet_sha256", "")),
                    "size_bytes": int(
                        parquet_path.stat().st_size
                        if parquet_path.is_file()
                        else payload.get("parquet_size", 0) or 0
                    ),
                    "manifest_valid": bool(payload),
                    "parquet_present": parquet_path.is_file(),
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "artifact_kind",
            "relative_parquet_path",
            "relative_manifest_path",
            "split",
            "row_count",
            "part_hash",
            "parquet_sha256",
            "size_bytes",
            "manifest_valid",
            "parquet_present",
        ),
    )


def _safe_archive_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    if not normalized:
        raise ValueError("job_id cannot be converted to an archive name")
    return normalized[:100]


def _public_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in arguments.items()
        if not str(key).startswith("_job_project_config")
    }


__all__ = [
    "PILOT_ARCHIVE_VERSION",
    "PilotArchiveResult",
    "archive_partial_feature_pilot",
]
