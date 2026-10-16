from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import MimicProjectConfig
from .models import SplitAssignment
from .storage import atomic_write_parquet, content_hash


def assign_subject_split(subject_id: int, config: MimicProjectConfig) -> SplitAssignment:
    raw = f"{config.split_salt}:{int(subject_id)}".encode("ascii")
    digest = hashlib.sha256(raw).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < config.train_fraction:
        split = "train"
    elif value < config.train_fraction + config.validation_fraction:
        split = "validation"
    else:
        split = "locked_test"
    return SplitAssignment(
        subject_id=int(subject_id),
        split=split,
        split_value=value,
        split_hash=digest.hex(),
    )


def split_spec_hash(config: MimicProjectConfig) -> str:
    return content_hash(
        {
            "algorithm": "sha256-first-64-bit",
            "salt": config.split_salt,
            "train_fraction": config.train_fraction,
            "validation_fraction": config.validation_fraction,
            "locked_test_fraction": config.locked_test_fraction,
        }
    )


def build_subject_splits(
    subject_ids: Iterable[int],
    config: MimicProjectConfig,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    unique_ids = sorted({int(subject_id) for subject_id in subject_ids})
    rows = [assign_subject_split(subject_id, config) for subject_id in unique_ids]
    frame = pd.DataFrame(
        {
            "subject_id": [row.subject_id for row in rows],
            "split": [row.split for row in rows],
            "split_value": [row.split_value for row in rows],
            "subject_split_hash": [row.split_hash for row in rows],
            "split_spec_hash": split_spec_hash(config),
        }
    )
    target = (
        Path(output_path)
        if output_path is not None
        else config.master_root / "subject_splits.parquet"
    )
    atomic_write_parquet(frame, target)
    return frame


def assert_subject_split_integrity(frame: pd.DataFrame) -> None:
    required = {"subject_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    invalid = set(frame["split"].dropna().unique()) - {
        "train",
        "validation",
        "locked_test",
    }
    if invalid:
        raise ValueError(f"Unknown split values: {sorted(invalid)}")
    overlap = frame.groupby("subject_id")["split"].nunique()
    overlap = overlap[overlap > 1]
    if not overlap.empty:
        raise RuntimeError(
            f"Patient split leakage detected for {len(overlap)} subject(s)"
        )
