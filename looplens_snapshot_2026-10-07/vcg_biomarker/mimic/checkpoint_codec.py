"""Atomic compact gzip JSON for large, content-bound checkpoint groups."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import tempfile
import zlib
from typing import Any

from .storage import _strict_json_default, _strict_json_value, atomic_replace

MAX_GROUP_BYTES = 512 * 1024**2
MAX_ENCODED_BYTES = MAX_GROUP_BYTES + 1024**2


def write_group(path: Path, payload: Any) -> dict[str, Any]:
    raw = json.dumps(_strict_json_value(payload), ensure_ascii=False, separators=(",", ":"),
                     default=_strict_json_default, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_GROUP_BYTES:
        raise ValueError("Checkpoint group exceeds the explicit decoding memory limit")
    compressed = gzip.compress(raw, compresslevel=1, mtime=0)
    metadata = {"encoding": "gzip-json-v1", "uncompressed_bytes": len(raw),
                "sha256": hashlib.sha256(compressed).hexdigest()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(compressed); handle.flush(); os.fsync(handle.fileno())
        atomic_replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return metadata


def read_group(path: Path, *, sha256: str, uncompressed_bytes: int) -> Any:
    if type(uncompressed_bytes) is not int or not 0 <= uncompressed_bytes <= MAX_GROUP_BYTES:
        raise RuntimeError("Invalid checkpoint decompression size")
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise RuntimeError("Invalid checkpoint compressed digest")
    with Path(path).open("rb") as stream:
        encoded = stream.read(MAX_ENCODED_BYTES + 1)
    if len(encoded) > MAX_ENCODED_BYTES:
        raise RuntimeError("Checkpoint compressed input exceeds byte limit")
    if hashlib.sha256(encoded).hexdigest() != sha256:
        raise RuntimeError("Checkpoint group compressed hash mismatch")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(encoded), mode="rb") as stream:
            raw = stream.read(uncompressed_bytes + 1)
        if len(raw) != uncompressed_bytes:
            raise RuntimeError("Checkpoint group decoded size mismatch")
        return json.loads(raw)
    except (OSError, EOFError, ValueError, zlib.error) as exc:
        raise RuntimeError("Checkpoint group cannot be decoded") from exc
