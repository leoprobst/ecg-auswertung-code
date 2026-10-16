"""Deterministic stand-ins for the per-record extraction, importable by spawn.

Used by test_extraction_scheduler.py: the scheduler is exercised with real
process pools, real resume fragments and real partition files, while the
record itself costs microseconds and has a known answer.
"""
from __future__ import annotations

import math

INITIALIZED = {"count": 0}


def initialize(task_config: dict) -> None:
    INITIALIZED["count"] += 1


def record(row: dict) -> dict:
    study_id = int(row["study_id"])
    failed = study_id % 11 == 0
    return {
        "study_id": study_id,
        "split": row.get("split", "train"),
        "extraction_status": "signal_qc_failed" if failed else "ok",
        "error_type": "QC" if failed else None,
        "error_message": "synthetic" if failed else None,
        "biomarker_calculation_failure_count": 0,
        "biomarker_calculation_failure_names": "",
        "marker_a": float("nan") if failed else study_id * 0.5,
        # Abstains for a whole partition in some tests: all-null column.
        "marker_b": math.nan if study_id < 1000 else float(study_id % 7),
    }


def batch(rows: list[dict]) -> list[dict]:
    return [record(row) for row in rows]
