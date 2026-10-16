from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_content_fingerprint,
)


FRAME_CHUNK_CHECKPOINT_VERSION = "1.1.0"


def checkpointed_frame_chunk(
    root: Path,
    *,
    namespace: str,
    chunk_index: int,
    input_hash: str,
    item_keys: Sequence[object],
    compute: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Return one verified frame chunk, computing it atomically if needed."""
    data_path, resumed = checkpointed_frame_chunk_path(
        root,
        namespace=namespace,
        chunk_index=chunk_index,
        input_hash=input_hash,
        item_keys=item_keys,
        compute=compute,
    )
    return pd.read_parquet(data_path), resumed


def checkpointed_frame_chunk_path(
    root: Path,
    *,
    namespace: str,
    chunk_index: int,
    input_hash: str,
    item_keys: Sequence[object],
    compute: Callable[[], pd.DataFrame],
) -> tuple[Path, bool]:
    # Both main and helpers hold the same OS lock from verification through
    # publication. Process death releases it; an existing chunk is rechecked.
    with frame_chunk_lock(root, namespace=namespace, chunk_index=chunk_index):
        return _checkpointed_frame_chunk_path_unlocked(root, namespace=namespace,
            chunk_index=chunk_index, input_hash=input_hash, item_keys=item_keys, compute=compute)


@contextmanager
def frame_chunk_lock(root, *, namespace, chunk_index):
    directory = _checkpoint_namespace(Path(root), namespace)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / f'part-{int(chunk_index):06d}.lock'
    with lock.open('a+b') as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (13, 11, 36):
                        raise
                    time.sleep(.2)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _checkpointed_frame_chunk_path_unlocked(
    root: Path, *, namespace: str, chunk_index: int, input_hash: str,
    item_keys: Sequence[object], compute: Callable[[], pd.DataFrame],
) -> tuple[Path, bool]:
    """Return one verified immutable chunk path without retaining its frame.

    The data file is published before its manifest.  Consequently an
    interruption can leave an uncommitted parquet file, but it can never make
    an incomplete chunk resumable.  The next call simply replaces that orphan
    atomically and publishes a fresh manifest.
    """
    chunk_root = _checkpoint_namespace(root, namespace)
    chunk_root.mkdir(parents=True, exist_ok=True)
    stem = f"part-{int(chunk_index):06d}"
    data_path = chunk_root / f"{stem}.parquet"
    manifest_path = chunk_root / f"{stem}.json"
    item_hash = content_hash(list(item_keys))
    expected = {
        "checkpoint_version": FRAME_CHUNK_CHECKPOINT_VERSION,
        "checkpoint_type": "dataframe_chunk",
        "input_hash": str(input_hash),
        "chunk_index": int(chunk_index),
        "item_hash": item_hash,
        "item_count": len(item_keys),
        "relative_path": data_path.name,
    }

    if manifest_path.is_file():
        _verify_frame_chunk(data_path, manifest_path, expected=expected)
        return data_path, True

    frame = compute()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"Checkpoint computation returned {type(frame).__name__}, "
            "expected pandas.DataFrame"
        )
    atomic_write_parquet(frame, data_path)
    fingerprint = file_content_fingerprint(data_path)
    payload = {
        **expected,
        "row_count": len(frame),
        "columns": [str(column) for column in frame.columns],
        "size": int(fingerprint["size"]),
        "sha256": str(fingerprint["sha256"]),
    }
    payload["checkpoint_hash"] = content_hash(payload)
    atomic_write_json(manifest_path, payload)
    return data_path, False


def materialize_frame_chunks(paths: Sequence[Path]) -> pd.DataFrame:
    """Materialize verified chunks once, avoiding a list of live DataFrames."""
    resolved = [
        str(Path(path).resolve())
        for path in paths
        if pq.ParquetFile(path).metadata.num_rows > 0
    ]
    if not resolved:
        return pd.DataFrame()
    # Passing all paths to the Arrow dataset scanner constructs one final
    # table directly.  In contrast, ``pd.concat([read(path), ...])`` keeps
    # every source frame plus the concatenated copy alive at peak memory.
    return pd.read_parquet(resolved)


def _load_verified_frame_chunk(
    data_path: Path,
    manifest_path: Path,
    *,
    expected: dict,
) -> pd.DataFrame:
    _verify_frame_chunk(data_path, manifest_path, expected=expected)
    return pd.read_parquet(data_path)


def _verify_frame_chunk(
    data_path: Path,
    manifest_path: Path,
    *,
    expected: dict,
) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Frame checkpoint manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Frame checkpoint manifest is invalid: {manifest_path}"
        )
    stored_hash = payload.get("checkpoint_hash")
    hash_payload = dict(payload)
    hash_payload.pop("checkpoint_hash", None)
    if stored_hash != content_hash(hash_payload):
        raise RuntimeError(
            f"Frame checkpoint manifest hash mismatch: {manifest_path}"
        )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Frame checkpoint is stale for {key}: {manifest_path}"
            )
    if not data_path.is_file():
        raise RuntimeError(
            f"Frame checkpoint data is missing: {data_path}"
        )
    fingerprint = file_content_fingerprint(data_path)
    if (
        int(payload.get("size", -1)) != int(fingerprint["size"])
        or payload.get("sha256") != fingerprint["sha256"]
    ):
        raise RuntimeError(
            f"Frame checkpoint data hash mismatch: {data_path}"
        )
    try:
        parquet_file = pq.ParquetFile(data_path)
        metadata = parquet_file.metadata
        columns = list(parquet_file.schema_arrow.names)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Frame checkpoint parquet metadata is unreadable: {data_path}"
        ) from exc
    if int(metadata.num_rows) != int(payload.get("row_count", -1)):
        raise RuntimeError(
            f"Frame checkpoint row count mismatch: {data_path}"
        )
    if [str(column) for column in columns] != payload.get("columns"):
        raise RuntimeError(
            f"Frame checkpoint schema mismatch: {data_path}"
        )


def _checkpoint_namespace(root: Path, namespace: str) -> Path:
    parts = Path(namespace).parts
    if (
        not parts
        or Path(namespace).is_absolute()
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"Unsafe frame checkpoint namespace: {namespace!r}")
    return root.joinpath(*parts)
