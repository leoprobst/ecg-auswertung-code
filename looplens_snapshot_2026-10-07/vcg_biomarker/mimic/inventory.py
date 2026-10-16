from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .code15_cohort import audit_cohort_release
from .config import MimicProjectConfig
from .schema import ECHO_SPECS, SOURCE_TABLE_SPECS, SourceTableSpec, validate_columns
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
)


INVENTORY_STAGE_VERSION = "1.3.0"


@dataclass(frozen=True)
class InventoryItem:
    table_id: str
    source: str
    path: str
    required: bool
    exists: bool
    size_bytes: Optional[int]
    row_count: Optional[int]
    columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    status: str
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataInventory:
    created_at: str
    items: tuple[InventoryItem, ...]
    input_hash: str

    @property
    def errors(self) -> tuple[InventoryItem, ...]:
        return tuple(item for item in self.items if item.status == "error")

    @property
    def warnings(self) -> tuple[InventoryItem, ...]:
        return tuple(item for item in self.items if item.status == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "input_hash": self.input_hash,
            "items": [item.to_dict() for item in self.items],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


def build_inventory(
    config: MimicProjectConfig,
    *,
    count_rows: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> DataInventory:
    config.ensure_directories()
    items: list[InventoryItem] = []
    resolved_specs = list(_resolve_source_specs(config))
    fingerprints: list[dict[str, Any]] = []
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )

    for spec, path in resolved_specs:
        if progress:
            progress(f"Inventar: {spec.table_id}")
        if not path.exists():
            status = "error" if spec.required else "warning"
            if (
                spec.source == "mimic_note"
                and config.mimic_note_root is None
            ):
                diagnostic = (
                    "optional MIMIC-IV-Note root is not configured; "
                    "note-derived evidence is unavailable"
                )
            else:
                diagnostic = (
                    "required file missing"
                    if spec.required
                    else "optional file missing"
                )
            items.append(
                InventoryItem(
                    table_id=spec.table_id,
                    source=spec.source,
                    path=str(path),
                    required=spec.required,
                    exists=False,
                    size_bytes=None,
                    row_count=None,
                    columns=(),
                    missing_columns=spec.required_columns,
                    status=status,
                    diagnostic=diagnostic,
                )
            )
            continue

        fingerprint = file_fingerprint(path)
        fingerprints.append({"table_id": spec.table_id, **fingerprint})
        try:
            columns, missing = validate_columns(path, spec.required_columns)
            row_count = _count_csv_rows(catalog, path) if count_rows else None
            status = "error" if missing else "ok"
            diagnostic = (
                f"missing required columns: {', '.join(missing)}" if missing else ""
            )
        except Exception as exc:
            columns = ()
            missing = spec.required_columns
            row_count = None
            status = "error"
            diagnostic = f"{type(exc).__name__}: {exc}"

        items.append(
            InventoryItem(
                table_id=spec.table_id,
                source=spec.source,
                path=str(path.resolve()),
                required=spec.required,
                exists=True,
                size_bytes=path.stat().st_size,
                row_count=row_count,
                columns=columns,
                missing_columns=missing,
                status=status,
                diagnostic=diagnostic,
            )
        )

    items.append(_waveform_sample_item(config))
    for cohort, root in (
        ("code15", config.code15_root),
        ("sami_trop", config.sami_trop_root),
    ):
        item, source_fingerprints = _external_cohort_item(root, cohort)
        items.append(item)
        fingerprints.extend(source_fingerprints)
    inventory = DataInventory(
        created_at=datetime.now(timezone.utc).isoformat(),
        items=tuple(items),
        input_hash=content_hash(fingerprints),
    )
    json_path = config.work_root / "data_inventory.json"
    parquet_path = config.work_root / "data_inventory.parquet"
    manifest_path = config.work_root / "inventory.manifest.json"
    # Reuse an identical inventory instead of rewriting it. The file carries
    # its creation time, so every rewrite changed its hash although no source
    # changed; the marker freeze fingerprints these files, and a full run
    # started after its separate pilot job re-ran this stage first and was
    # then refused at audit-biomarkers (A143, 2026-09-26). Other stages already
    # reuse on identical keys; any difference in keys or items rewrites.
    if manifest_path.is_file() and json_path.is_file():
        try:
            previous = StageManifest.load(manifest_path)
            if previous.matches(
                inventory.input_hash,
                scientific_config_hash(config),
                INVENTORY_STAGE_VERSION,
            ):
                existing = load_inventory(config)
                if (
                    existing.input_hash == inventory.input_hash
                    and existing.items == inventory.items
                ):
                    return existing
        except (OSError, ValueError, KeyError, TypeError):
            pass
    atomic_write_json(json_path, inventory.to_dict())
    atomic_write_parquet(
        pd.DataFrame([item.to_dict() for item in inventory.items]),
        parquet_path,
    )

    manifest = StageManifest.complete(
        stage="inventory",
        stage_version=INVENTORY_STAGE_VERSION,
        input_hash=inventory.input_hash,
        config_hash=scientific_config_hash(config),
        output_paths=(str(json_path.resolve()), str(parquet_path.resolve())),
        row_counts={"inventory_items": len(items)},
        notes=(f"operational_config_hash={operational_config_hash(config)}",),
        output_integrity=capture_output_integrity((json_path, parquet_path)),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    config.save()
    return inventory


def load_inventory(config: MimicProjectConfig) -> DataInventory:
    import json

    path = config.work_root / "data_inventory.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for raw in data["items"]:
        raw["columns"] = tuple(raw.get("columns", ()))
        raw["missing_columns"] = tuple(raw.get("missing_columns", ()))
        items.append(InventoryItem(**raw))
    return DataInventory(
        created_at=data["created_at"],
        items=tuple(items),
        input_hash=data["input_hash"],
    )


def _resolve_source_specs(
    config: MimicProjectConfig,
) -> list[tuple[SourceTableSpec, Path]]:
    roots = {
        "mimic_iv": config.mimic_iv_root,
        "mimic_ed": config.mimic_ed_root,
        "mimic_ecg": config.mimic_ecg_root,
        "mimic_note": (
            config.mimic_note_root
            if config.mimic_note_root is not None
            else Path("__mimic_note_root_not_configured__")
        ),
    }
    resolved = [
        (spec, _resolve_spec_path(roots[spec.source], spec))
        for spec in SOURCE_TABLE_SPECS
        if spec.source in roots
    ]
    echo_paths = (
        config.echo_structured_path,
        config.echo_study_list_path,
        config.echo_record_list_path,
    )
    resolved.extend(zip(ECHO_SPECS, echo_paths))
    return resolved


def _resolve_spec_path(root: Path, spec: SourceTableSpec) -> Path:
    candidates = (
        spec.relative_path,
        *spec.alternative_relative_paths,
    )
    for relative_path in candidates:
        path = root / relative_path
        if path.is_file():
            return path
    return root / spec.relative_path


def _count_csv_rows(catalog: DuckDBCatalog, path: Path) -> int:
    normalized = str(path.resolve()).replace("\\", "/").replace("'", "''")
    with catalog.connect() as connection:
        row = connection.execute(
            f"SELECT count(*) FROM read_csv('{normalized}', header=true, "
            "all_varchar=true, ignore_errors=false)"
        ).fetchone()
    return int(row[0])


def _external_cohort_item(
    root: Optional[Path],
    cohort: str,
) -> tuple[InventoryItem, list[dict[str, Any]]]:
    audit = audit_cohort_release(root, cohort)
    source_paths = tuple(Path(path) for path in audit["source_paths"])
    fingerprints = [
        {"table_id": f"{cohort}_release", **file_fingerprint(path)}
        for path in source_paths
        if path.is_file()
    ]
    configured = root is not None
    status = str(audit["status"])
    if status == "unconfigured":
        status = "warning"
    return (
        InventoryItem(
            table_id=f"{cohort}_release",
            source=cohort,
            path=str(root or ""),
            required=configured,
            exists=bool(root is not None and Path(root).is_dir()),
            size_bytes=(
                sum(path.stat().st_size for path in source_paths if path.is_file())
                if source_paths
                else None
            ),
            row_count=int(audit["metadata_rows"]),
            columns=tuple(audit["columns"]),
            missing_columns=tuple(audit["missing_columns"]),
            status=status,
            diagnostic=str(audit["diagnostic"]),
        ),
        fingerprints,
    )


def _waveform_sample_item(
    config: MimicProjectConfig,
    *,
    sample_size: int = 250,
) -> InventoryItem:
    record_list = config.mimic_ecg_root / "record_list.csv"
    sampled = 0
    missing: list[str] = []
    if not record_list.exists():
        return InventoryItem(
            table_id="ecg_waveform_path_sample",
            source="mimic_ecg",
            path=str(config.mimic_ecg_root),
            required=True,
            exists=False,
            size_bytes=None,
            row_count=0,
            columns=(),
            missing_columns=(),
            status="error",
            diagnostic="record_list.csv is unavailable for waveform path checks",
        )
    try:
        with record_list.open(
            "rt",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                relative = str(row.get("path") or "").strip()
                if not relative:
                    missing.append("<empty path>")
                else:
                    target = config.mimic_ecg_root / relative
                    header = (
                        target
                        if target.suffix.casefold() == ".hea"
                        else target.with_suffix(".hea")
                    )
                    if not header.is_file():
                        missing.append(str(header))
                sampled += 1
                if sampled >= sample_size:
                    break
    except (OSError, csv.Error) as exc:
        return InventoryItem(
            table_id="ecg_waveform_path_sample",
            source="mimic_ecg",
            path=str(record_list.resolve()),
            required=True,
            exists=True,
            size_bytes=record_list.stat().st_size,
            row_count=sampled,
            columns=("path",),
            missing_columns=(),
            status="error",
            diagnostic=f"{type(exc).__name__}: {exc}",
        )
    if sampled == 0:
        status = "error"
        diagnostic = "record_list.csv has no data rows"
    elif missing:
        status = "error"
        diagnostic = (
            f"{len(missing)}/{sampled} sampled WFDB headers are missing; "
            f"first={missing[0]}"
        )
    else:
        status = "ok"
        diagnostic = f"{sampled}/{sampled} sampled WFDB headers found"
    return InventoryItem(
        table_id="ecg_waveform_path_sample",
        source="mimic_ecg",
        path=str(config.mimic_ecg_root.resolve()),
        required=True,
        exists=True,
        size_bytes=None,
        row_count=sampled,
        columns=("path",),
        missing_columns=(),
        status=status,
        diagnostic=diagnostic,
    )
