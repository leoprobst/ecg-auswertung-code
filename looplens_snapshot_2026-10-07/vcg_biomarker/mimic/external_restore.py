"""Restore only absent Challenge training files, with published content hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path, PurePosixPath
import re
from urllib.request import urlopen

from .storage import atomic_write_json, file_sha256

OFFICIAL_ROOT = "https://physionet.org/files/challenge-2021/1.0.3/"


def missing_training_files(root: Path, checksums: Path) -> list[dict[str, str]]:
    base = root.resolve()
    missing = []
    seen = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if not relative.startswith("training/") or PurePosixPath(relative).suffix not in (".hea", ".mat"):
            continue
        if (not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", relative)
                or ".." in PurePosixPath(relative).parts):
            raise ValueError("Invalid checksum manifest entry")
        if relative in seen:
            raise ValueError("Duplicate checksum manifest target")
        seen.add(relative)
        target = (base / relative).resolve()
        if not target.is_relative_to(base):
            raise ValueError("Checksum target escapes dataset root")
        if not target.exists():
            missing.append({"relative_path": relative, "sha256": digest})
    return missing


def restore_one(root: Path, item: dict[str, str]) -> dict:
    relative = item["relative_path"]
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base) or not relative.startswith("training/"):
        raise ValueError("Restore target escapes training dataset")
    source_url = OFFICIAL_ROOT + relative
    row: dict[str, object] = dict(item, status="failed", source_url=source_url)
    try:
        if target.exists():
            row["status"] = "already_present_not_modified"
            return row
        with urlopen(source_url, timeout=30) as response:
            payload = response.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("Unexpectedly large individual restoration file")
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError("Official download differs from expected SHA-256")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:  # Exclusive creation: never overwrite.
            stream.write(payload)
        if file_sha256(target) != item["sha256"]:
            raise OSError("Restored file failed on-disk integrity check")
        row.update(status="restored_verified", bytes=len(payload))
    except Exception as exc:
        row.update(error_type=type(exc).__name__, error_message=str(exc))
    return row


def restore(root: Path, output: Path, *, workers: int = 4) -> dict:
    if not 1 <= workers <= 4:
        raise ValueError("Restoration requires one to four workers")
    checksums = root / "SHA256SUMS.txt"
    pending = missing_training_files(root, checksums)
    result = {"status": "running", "source": OFFICIAL_ROOT,
        "checksums_sha256": file_sha256(checksums), "initially_missing": len(pending),
        "existing_files_overwritten": False, "results": []}
    atomic_write_json(output, result)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(lambda item: restore_one(root, item), pending):
            result["results"].append(row)
            print(f"restore {len(result['results'])}/{len(pending)} {row['status']} {row['relative_path']}", flush=True)
            atomic_write_json(output, result)
    result["status"] = "complete" if all(row["status"] == "restored_verified" for row in result["results"]) else "incomplete"
    result["restored"] = sum(row["status"] == "restored_verified" for row in result["results"])
    atomic_write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.root, args.output)
