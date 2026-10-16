"""Renaming is atomic; that is not the same as durable.

A rename can reach the disk while the bytes it points at are still in the
page cache. Lose power in that window and the result is a correctly named
file with missing or truncated content -- which, unlike a half-written
temporary, cannot be told apart from a complete one.

``atomic_write_text`` had always flushed before renaming. ``atomic_write_parquet``
had not, and every stage result, report and resume fragment goes through it.
The consequence was ordered the wrong way round in the feature extraction:
its resume manifest is JSON and was therefore durable, while the fragment
parquet the manifest names was not, so a power loss could leave a manifest
pointing at bytes that never arrived. The checksum check would then reject the
whole fragment set and the partition would restart from zero.

These tests pin the flush, the ordering it buys, and that a platform refusing
to confirm a flush never costs the write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic import storage
from vcg_biomarker.mimic.storage import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    flush_to_disk,
)


def test_a_written_parquet_still_reads_back_unchanged(tmp_path):
    frame = pd.DataFrame({"study_id": [1, 2, 3], "value": [0.5, 1.5, 2.5]})
    target = tmp_path / "features.parquet"
    atomic_write_parquet(frame, target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), frame)


def test_no_temporary_file_survives_the_write(tmp_path):
    atomic_write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "x.parquet")
    leftovers = [item.name for item in tmp_path.iterdir() if item.name != "x.parquet"]
    assert leftovers == []


def test_the_parquet_is_flushed_before_it_is_renamed(tmp_path):
    """The ordering is the whole point: a rename that beats its own bytes to
    the disk is what produces a file that lies about being complete."""

    calls: list[tuple[str, str]] = []
    original_flush = storage.flush_to_disk
    original_replace = storage.atomic_replace

    def record_flush(path):
        calls.append(("flush", Path(path).name))
        return original_flush(path)

    def record_replace(source, target, **kwargs):
        calls.append(("replace", Path(target).name))
        return original_replace(source, target, **kwargs)

    storage.flush_to_disk = record_flush
    storage.atomic_replace = record_replace
    try:
        atomic_write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "y.parquet")
    finally:
        storage.flush_to_disk = original_flush
        storage.atomic_replace = original_replace

    kinds = [kind for kind, _ in calls]
    assert kinds.index("flush") < kinds.index("replace")
    assert kinds.count("flush") == 2, "the renamed file is flushed as well"


def test_flushing_a_missing_file_is_silent(tmp_path):
    """A durability helper that raises would turn a weakness into a lost stage."""

    flush_to_disk(tmp_path / "does-not-exist.parquet")


def test_a_platform_that_refuses_to_flush_does_not_lose_the_write(
    tmp_path, monkeypatch
):
    def refuse(_fd):
        raise OSError("fsync unsupported on this filesystem")

    monkeypatch.setattr(os, "fsync", refuse)
    frame = pd.DataFrame({"a": [1, 2]})
    target = tmp_path / "z.parquet"
    atomic_write_parquet(frame, target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), frame)


def test_flushing_a_directory_is_only_attempted_where_it_means_something(tmp_path):
    """Windows refuses to open a directory as a file; POSIX needs the
    directory flushed for the rename itself to be durable."""

    import inspect

    source = inspect.getsource(flush_to_disk)
    assert 'os.name == "posix"' in source


def test_the_text_writer_still_flushes_as_it_always_did(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"a": 1})
    assert target.read_text(encoding="utf-8").strip().startswith("{")
    text_target = tmp_path / "note.txt"
    atomic_write_text(text_target, "hello")
    assert text_target.read_text(encoding="utf-8") == "hello"

    import inspect

    assert "os.fsync" in inspect.getsource(atomic_write_text)


def test_no_stage_writes_a_parquet_straight_to_its_final_path():
    """A direct write is worse than an unflushed one: it corrupts the result.

    ``atomic_write_parquet`` writes a temporary and renames, so an interrupted
    write leaves the previous file intact and a dot-prefixed fragment behind.
    Eight call sites in the external-validation and pilot-workspace stages
    wrote straight to the destination instead, where an interruption leaves a
    truncated file under the name a later reader trusts -- and the manifest's
    tracked checksum is taken *after* the write, so it would certify the
    truncation rather than catch it.

    The two survivors write to a temporary that is flushed and renamed by
    hand, which is the same contract spelled out.
    """

    import ast
    from pathlib import Path as _Path

    allowed = {
        ("storage.py", "atomic_write_parquet"),
        ("trajectory.py", "build_trajectory_frame"),
    }
    root = _Path(storage.__file__).parent
    offenders: list[str] = []
    for module_path in sorted(root.glob("*.py")):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "to_parquet":
                continue
            enclosing = node
            name = "<module>"
            while enclosing in parents:
                enclosing = parents[enclosing]
                if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = enclosing.name
                    break
            if (module_path.name, name) not in allowed:
                offenders.append(f"{module_path.name}:{name}:{node.lineno}")
    assert offenders == [], (
        "these write a parquet without a temporary and a rename: "
        + ", ".join(offenders)
    )


def test_the_resume_fragment_is_durable_before_its_manifest_names_it():
    """The extraction commits a fragment parquet, then the JSON manifest that
    names it. With the parquet unflushed that order was a lie: the manifest
    could survive a power loss while its fragment did not, and the checksum
    check would then discard every fragment of the partition."""

    import inspect

    from vcg_biomarker.mimic import feature_extraction

    source = inspect.getsource(feature_extraction._append_resume_fragment)
    assert source.index("atomic_write_parquet") < source.index("atomic_write_json")
