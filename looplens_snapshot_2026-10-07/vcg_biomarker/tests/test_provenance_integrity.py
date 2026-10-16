from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.event_ingest import (
    ingest_clinical_event_tables,
    load_clinical_event_registry,
)
from vcg_biomarker.mimic.ingest import _notes_coverage, _table_specs
from vcg_biomarker.mimic.inventory import _resolve_source_specs
from vcg_biomarker.mimic.note_labels import build_note_labels
from vcg_biomarker.mimic.storage import (
    StageManifest,
    capture_output_integrity,
    file_fingerprint,
)


def _config(
    tmp_path: Path,
    *,
    mimic_note_root: Path | None = None,
) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic",
        mimic_ed_root=tmp_path / "ed",
        mimic_ecg_root=tmp_path / "ecg",
        mimic_note_root=mimic_note_root,
        echo_structured_path=tmp_path / "structured.csv",
        echo_study_list_path=tmp_path / "study.csv",
        echo_record_list_path=tmp_path / "record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def test_content_fingerprint_mode_and_manifest_output_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable clinical source")

    monkeypatch.delenv("VCG_FILE_FINGERPRINT_MODE", raising=False)
    assert "sha256" not in file_fingerprint(source)
    monkeypatch.setenv("VCG_FILE_FINGERPRINT_MODE", "sha256")
    assert len(file_fingerprint(source)["sha256"]) == 64
    assert "sha256" not in file_fingerprint(source, include_sha256=False)

    output = tmp_path / "output.parquet"
    pd.DataFrame({"subject_id": [1, 2], "value": [3.0, 4.0]}).to_parquet(
        output,
        index=False,
    )
    integrity = capture_output_integrity((output,))
    manifest = StageManifest.complete(
        stage="test",
        stage_version="1",
        input_hash="input",
        config_hash="config",
        output_paths=(str(output.resolve()),),
        row_counts={"output": 2},
        output_integrity=integrity,
    )
    assert manifest.matches("input", "config", "1")

    pd.DataFrame({"subject_id": [1, 2], "value": [30.0, 40.0]}).to_parquet(
        output,
        index=False,
    )
    assert not manifest.matches("input", "config", "1")


def test_legacy_manifest_remains_loadable_with_existence_check(
    tmp_path: Path,
) -> None:
    output = tmp_path / "legacy.txt"
    output.write_text("legacy", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "stage": "legacy",
                "stage_version": "1",
                "input_hash": "input",
                "config_hash": "config",
                "output_paths": [str(output.resolve())],
                "row_counts": {},
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "complete",
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = StageManifest.load(path)
    assert manifest.output_integrity == ()
    assert manifest.matches("input", "config", "1")


def test_filtered_event_rows_are_accounted_and_partitions_are_hashed(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    hosp = config.mimic_iv_root / "hosp"
    hosp.mkdir(parents=True)
    pd.DataFrame(
        {
            "labevent_id": [1],
            "subject_id": [100],
            "hadm_id": [10],
            "specimen_id": [20],
            "itemid": [51003],
            "order_provider_id": ["P1"],
            "charttime": ["2024-01-01 09:00:00"],
            "storetime": ["2024-01-01 09:05:00"],
            "value": ["5.0"],
            "valuenum": [5.0],
            "valueuom": ["ng/L"],
            "ref_range_lower": [0.0],
            "ref_range_upper": [14.0],
            "flag": [""],
            "priority": ["ROUTINE"],
            "comments": ["provenance test"],
        }
    ).to_csv(
        hosp / "labevents.csv.gz",
        index=False,
        compression="gzip",
    )
    icu = config.mimic_iv_root / "icu"
    icu.mkdir(parents=True)
    source = icu / "chartevents.csv.gz"
    pd.DataFrame(
        {
            "subject_id": ["100", "101", "102", ""],
            "hadm_id": [10, 11, 12, 13],
            "stay_id": [20, 21, 22, 23],
            "charttime": [
                "2024-01-01 10:00:00",
                "2024-01-01 10:01:00",
                "2024-01-01 10:02:00",
                "2024-01-01 10:03:00",
            ],
            "itemid": [220045, 220045, 999999, 220045],
            "value": ["80", "81", "not curated", "90"],
            "valuenum": [80.0, 81.0, None, 90.0],
            "valueuom": ["bpm", "bpm", "", "bpm"],
        }
    ).to_csv(source, index=False, compression="gzip")

    manifest = ingest_clinical_event_tables(config)
    registry = load_clinical_event_registry(config)
    assert registry is not None
    record = next(
        item
        for item in registry["tables"]
        if item.get("table_id") == "chartevents"
    )
    assert record["source_row_count"] == 4
    assert record["row_count"] == 2
    assert record["quarantine_row_count"] == 1
    assert record["filtered_source_row_count"] == 1
    assert record["accounted_source_row_count"] == 3
    assert manifest.row_counts["chartevents_source"] == 4
    assert manifest.row_counts["chartevents_filtered"] == 1
    assert len(manifest.output_integrity) == 1
    assert len(manifest.output_integrity[0]["sha256"]) == 64
    assert len(record["partitions"]) == 2
    assert all(len(item["sha256"]) == 64 for item in record["partitions"])
    assert [item["row_count"] for item in record["partitions"]] == [1, 1]
    assert len(record["quarantine_partitions"][0]["sha256"]) == 64

    partition = (
        Path(registry["_registry_path"]).parent
        / record["partitions"][0]["relative_path"]
    )
    original_bytes = partition.read_bytes()
    altered = pd.read_parquet(partition)
    altered.loc[:, "value"] = "tampered"
    altered.to_parquet(partition, index=False)
    with pytest.raises(RuntimeError, match="partition integrity failed"):
        load_clinical_event_registry(config)
    partition.write_bytes(original_bytes)

    second_partition = (
        Path(registry["_registry_path"]).parent
        / record["partitions"][1]["relative_path"]
    )
    second_partition.unlink()
    with pytest.raises(RuntimeError, match="partition set mismatch"):
        load_clinical_event_registry(config)


def test_note_gzip_layout_and_missing_coverage_are_explicit(
    tmp_path: Path,
) -> None:
    unconfigured = _config(tmp_path / "unconfigured")
    unconfigured_tables = _table_specs(unconfigured)
    assert _notes_coverage(unconfigured, unconfigured_tables)["status"] == (
        "not_configured"
    )
    notes_manifest = build_note_labels(unconfigured)
    assert "notes_coverage_status=not_configured" in notes_manifest.notes
    assert notes_manifest.row_counts == {
        "note_encounters": 0,
        "note_mentions": 0,
    }
    assert len(notes_manifest.output_integrity) == 2

    note_root = tmp_path / "configured" / "notes"
    nested = note_root / "note"
    nested.mkdir(parents=True)
    discharge = nested / "discharge.csv.gz"
    pd.DataFrame(
        {
            "note_id": ["n1"],
            "subject_id": [1],
            "hadm_id": [2],
            "note_type": ["DS"],
            "charttime": ["2024-01-01 00:00:00"],
            "text": ["No acute issue."],
        }
    ).to_csv(discharge, index=False, compression="gzip")
    configured = _config(
        tmp_path / "configured",
        mimic_note_root=note_root,
    )
    tables = _table_specs(configured)
    discharge_table = next(table for table in tables if table.name == "discharge")
    assert discharge_table.source_path.resolve() == discharge.resolve()
    coverage = _notes_coverage(configured, tables)
    assert coverage["status"] == "partial"
    assert coverage["present_tables"] == ["discharge"]

    inventory_paths = {
        spec.table_id: path
        for spec, path in _resolve_source_specs(configured)
    }
    assert inventory_paths["discharge"].resolve() == discharge.resolve()
