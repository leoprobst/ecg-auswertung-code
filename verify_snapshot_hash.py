"""Recompute the source-tree hash of the published measurement package.

The hash is the SHA-256 of the canonical JSON list of {relative_path, sha256} entries for every
Python file below vcg_biomarker/ (sorted by path, without __pycache__), the same procedure the
package uses internally.

Usage:
    python verify_snapshot_hash.py                 # release state (expected value in the folder README)
    python verify_snapshot_hash.py --with-patch    # release state overlaid with patch_2026-10-11/

With --with-patch the files of patch_2026-10-11/ replace or extend the release files, which gives the executed
snapshot (891 Python files) and its recorded hash.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "looplens_snapshot_2026-10-07"


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(data):
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def tree_hash(files):
    entries = [{"relative_path": rel, "sha256": sha} for rel, sha in sorted(files.items())]
    return hashlib.sha256(canonical(entries).encode("utf-8")).hexdigest(), len(entries)


def collect(base):
    out = {}
    for path in sorted(base.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        out[str(path.relative_to(base)).replace("\\", "/")] = file_sha(path)
    return out


def main():
    files = collect(ROOT / "vcg_biomarker")
    if "--with-patch" in sys.argv:
        files.update(collect(ROOT / "patch_2026-10-11" / "vcg_biomarker"))
    digest, count = tree_hash(files)
    print("python files: %d" % count)
    print("source tree hash: %s" % digest)


if __name__ == "__main__":
    main()
