"""The two widest reads in the project, bounded without moving a number.

``build_replicate_pairs`` is a self-join: DuckDB must hold both build sides,
each 640,104 rows x 1,753 marker columns x 8 bytes = 9 GB. Measured
2026-08-29 with a 12 GB DuckDB limit, resident memory climbed past **15.4 GB**
with 4.8 GB free and the output file still empty after six minutes; the run
was stopped before the machine swapped. ``build_trajectory_frame`` has the
same shape in a single ``fetchdf()``, and ``summarize_trajectories`` copies
the frame once more.

Both compute per marker -- ``marker_repeatability`` takes one marker's two
columns, ``summarize_trajectories`` one column per (subject, marker) record --
so blocking is exact. Neither function had a test before the change; these
hold the equivalence to data rather than to that argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.repeatability import repeatability_table_from_parquet
from vcg_biomarker.mimic.trajectory import (
    TRAJECTORY_MARKER_BLOCK,
    TrajectoryOptions,
    _marker_blocks,
    build_replicate_pairs,
    build_trajectory_frame,
    replicate_pair_parts,
    summarize_trajectories,
)


MARKERS = tuple(f"marker_{index:02d}" for index in range(17))


@pytest.fixture()
def project(tmp_path):
    """A small cohort with repeat recordings, written where a run expects it."""

    rng = np.random.default_rng(31)
    database_root = tmp_path / "physionet.org" / "files" / "mimiciv" / "3.1"
    (database_root / "hosp").mkdir(parents=True)
    (database_root / "icu").mkdir(parents=True)
    config = MimicProjectConfig.discover(database_root.parent, tmp_path / "work")
    config.ensure_directories()

    subjects, rows = 60, []
    for subject in range(subjects):
        for visit in range(3):
            rows.append(
                {
                    "subject_id": subject,
                    "study_id": subject * 10 + visit,
                    "hadm_id": subject,
                    "split": "train" if subject % 3 else "validation",
                    "ecg_time": pd.Timestamp("2026-01-01") + pd.Timedelta(
                        hours=visit
                    ),
                }
            )
    master = pd.DataFrame(rows)
    config.master_root.mkdir(parents=True, exist_ok=True)
    master.to_parquet(config.master_root / "ecg_master_index.parquet", index=False)

    features = master[["subject_id", "study_id", "split", "ecg_time"]].copy()
    features["extraction_status"] = "ok"
    features["signal_quality_score"] = 0.9
    for name in MARKERS:
        features[name] = rng.normal(size=len(features))
    root = config.features_root / "primary_core" / "development"
    for split, block in features.groupby("split"):
        folder = root / f"split={split}"
        folder.mkdir(parents=True, exist_ok=True)
        block.to_parquet(folder / "part-0.parquet", index=False)
    glob = str(root / "split=*" / "part-*.parquet").replace("\\", "/")
    return config, glob


def _pairs(config, glob, output, markers=MARKERS, block=None):
    import vcg_biomarker.mimic.trajectory as module

    if block is not None:
        original = module.TRAJECTORY_MARKER_BLOCK
        module.TRAJECTORY_MARKER_BLOCK = block
        try:
            return build_replicate_pairs(
                config,
                feature_glob=glob,
                marker_ids=tuple(markers),
                output_path=output,
            )
        finally:
            module.TRAJECTORY_MARKER_BLOCK = original
    return build_replicate_pairs(
        config, feature_glob=glob, marker_ids=tuple(markers), output_path=output
    )


def test_the_pair_artifact_is_written_in_parts(project, tmp_path):
    config, glob = project
    out = _pairs(config, glob, tmp_path / "pairs", block=5)
    parts = replicate_pair_parts(out)
    assert len(parts) == len(_marker_blocks(MARKERS, 5))
    assert len(parts) > 1, "sonst prueft der Test die Blockung nicht"


def test_every_part_carries_the_same_pairs(project, tmp_path):
    """Parts are joined by their keys, never by row position.

    The query is deliberately unsorted: every consumer reads one part at a
    time, and sorting the full result width filled DuckDB's buffer pool and
    spilled -- measured 2026-08-29, the second block was still running after
    19 minutes against 11 for the first. What must hold is that each part
    describes the same set of pairs, and that each part is internally aligned:
    a marker's two columns come from one part, so a row is one pair.
    """

    config, glob = project
    out = _pairs(config, glob, tmp_path / "pairs", block=4)
    keys = [
        pd.read_parquet(
            part, columns=["subject_id", "first_study_id", "second_study_id"]
        )
        for part in replicate_pair_parts(out)
    ]
    assert len(keys) > 1
    reference = set(map(tuple, keys[0].itertuples(index=False)))
    assert reference, "keine Paare gefunden"
    for frame in keys[1:]:
        assert set(map(tuple, frame.itertuples(index=False))) == reference
        assert len(frame) == len(keys[0])


def test_a_pair_row_stays_one_pair_within_a_part(project, tmp_path):
    """The alignment that does matter, and the one blocking could break."""

    config, glob = project
    out = _pairs(config, glob, tmp_path / "pairs", block=4)
    raw = pd.read_parquet(
        next(iter(replicate_pair_parts(out)))
    )
    markers = sorted(
        name[: -len("_first")] for name in raw.columns if name.endswith("_first")
    )
    assert markers
    features = pd.read_parquet(glob.replace("split=*", "split=train").replace(
        "part-*.parquet", "part-0.parquet"
    ))
    lookup = features.set_index("study_id")
    for marker in markers[:2]:
        for row in raw.head(20).itertuples(index=False):
            first_id = int(getattr(row, "first_study_id"))
            if first_id not in lookup.index:
                continue
            assert getattr(row, f"{marker}_first") == pytest.approx(
                float(lookup.loc[first_id, marker])
            )


def test_the_repeatability_table_is_the_same_whatever_the_part_size(
    project, tmp_path
):
    config, glob = project
    coarse = _pairs(config, glob, tmp_path / "coarse", block=100)
    fine = _pairs(config, glob, tmp_path / "fine", block=3)
    left = repeatability_table_from_parquet(coarse, MARKERS)
    right = repeatability_table_from_parquet(fine, MARKERS)
    assert not left.empty
    pd.testing.assert_frame_equal(left, right)


def test_a_marker_keeps_its_own_two_columns(project, tmp_path):
    config, glob = project
    out = _pairs(config, glob, tmp_path / "pairs", block=2)
    for part in replicate_pair_parts(out):
        names = set(pd.read_parquet(part).columns)
        markers = {
            n[: -len("_first")] for n in names if n.endswith("_first")
        }
        for marker in markers:
            assert f"{marker}_second" in names, marker


def test_stale_parts_from_an_earlier_run_are_removed(project, tmp_path):
    config, glob = project
    out = tmp_path / "pairs"
    _pairs(config, glob, out, block=2)
    many = len(replicate_pair_parts(out))
    _pairs(config, glob, out, block=100)
    few = len(replicate_pair_parts(out))
    assert few < many
    assert few == len(replicate_pair_parts(out)), "keine Reste vom ersten Lauf"


def test_the_trajectory_frame_is_the_same_whatever_the_block_size(
    project, tmp_path
):
    import vcg_biomarker.mimic.trajectory as module

    config, glob = project
    options = TrajectoryOptions(minimum_signal_quality=0.0)

    def build(block, name):
        original = module.TRAJECTORY_MARKER_BLOCK
        module.TRAJECTORY_MARKER_BLOCK = block
        try:
            path = build_trajectory_frame(
                config,
                feature_glob=glob,
                marker_ids=MARKERS,
                output_path=tmp_path / name,
                options=options,
            )
            return pd.read_parquet(path)
        finally:
            module.TRAJECTORY_MARKER_BLOCK = original

    whole = build(1000, "whole.parquet")
    blocked = build(3, "blocked.parquet")
    assert not whole.empty
    pd.testing.assert_frame_equal(blocked, whole)


def test_the_default_block_bounds_the_widest_real_read():
    """1,753 markers over 640,104 rows must not come back as one frame."""

    markers = tuple(f"m{index}" for index in range(1753))
    blocks = _marker_blocks(markers)
    assert len(blocks) > 1
    assert all(len(block) <= TRAJECTORY_MARKER_BLOCK for block in blocks)
    assert sum(len(block) for block in blocks) == len(markers)
    assert len(set(sum(blocks, ()))) == len(markers)


# --- Fertige Blöcke überleben eine Pause -------------------------------------


def test_a_finished_block_is_not_written_again(project, tmp_path):
    """Measured 2026-08-29: five of nine blocks took 4.1 hours, then the
    supervisor paused the job for memory. The loop deleted all five on the way
    back in, so those hours were spent again -- and would be again after the
    next pause, on a machine where memory pauses are expected."""

    config, glob = project
    out = tmp_path / "pairs"
    _pairs(config, glob, out, block=4)
    parts = replicate_pair_parts(out)
    assert len(parts) > 1
    stamps = {p.name: p.stat().st_mtime_ns for p in parts}

    _pairs(config, glob, out, block=4)
    again = replicate_pair_parts(out)
    assert {p.name for p in again} == set(stamps)
    for part in again:
        assert part.stat().st_mtime_ns == stamps[part.name], (
            f"{part.name} wurde neu geschrieben, obwohl es fertig war"
        )


def test_a_part_from_a_different_block_size_is_rewritten(project, tmp_path):
    """Reuse must not mix a part built from other markers into the result."""

    config, glob = project
    out = tmp_path / "pairs"
    _pairs(config, glob, out, block=4)
    coarse_names = {p.name for p in replicate_pair_parts(out)}

    _pairs(config, glob, out, block=9)
    fine = replicate_pair_parts(out)
    for part in fine:
        columns = set(pd.read_parquet(part).columns)
        markers = {
            c[: -len("_first")]
            for c in columns
            if c.endswith("_first") and c != "signal_quality_first"
        }
        assert len(markers) <= 9, sorted(markers)
    assert len(fine) <= len(coarse_names)


def test_a_truncated_part_is_rewritten(project, tmp_path):
    config, glob = project
    out = tmp_path / "pairs"
    _pairs(config, glob, out, block=4)
    victim = replicate_pair_parts(out)[0]
    victim.write_bytes(b"kein parquet")

    _pairs(config, glob, out, block=4)
    frame = pd.read_parquet(victim)
    assert not frame.empty


def test_the_reuse_check_rejects_a_row_count_mismatch(tmp_path):
    """A part from a different cohort or policy has the wrong pair count."""

    from vcg_biomarker.mimic.trajectory import _part_is_current

    path = tmp_path / "part-000.parquet"
    pd.DataFrame(
        {"m0_first": [1.0, 2.0], "m0_second": [1.1, 2.1]}
    ).to_parquet(path, index=False)
    assert _part_is_current(path, ("m0",), 2) is True
    assert _part_is_current(path, ("m0",), 3) is False
    assert _part_is_current(path, ("m0", "m1"), 2) is False
    assert _part_is_current(tmp_path / "absent.parquet", ("m0",), 2) is False


def test_the_stage_manifest_records_every_pair_part_as_a_file(project, monkeypatch):
    """The stage's manifest lists the pair parts, not their directory.

    Listing the directory passed every unit test and ended the first
    production-path preview on real data with "Stage output file is
    missing" -- after the pairs and trajectories had already been written.
    """
    import vcg_biomarker.mimic.trajectory as module
    from vcg_biomarker.mimic.repeatability_stage import run_repeatability_analysis
    from vcg_biomarker.mimic.storage import verify_output_integrity

    config, glob = project
    monkeypatch.setattr(module, "TRAJECTORY_MARKER_BLOCK", 5)
    monkeypatch.setattr(
        "vcg_biomarker.mimic.repeatability_stage.default_feature_panels",
        lambda: (type("Panel", (), {"panel_id": "primary_core", "feature_ids": MARKERS})(),),
    )
    manifest = run_repeatability_analysis(
        config, run_id="synthetic", feature_glob=glob, panel_ids=("primary_core",)
    )
    pair_root = config.runs_root / "synthetic" / "repeatability" / "replicate_pairs"
    parts = replicate_pair_parts(pair_root)
    assert len(parts) > 1
    listed = {str(path) for path in manifest.output_paths}
    assert {str(part.resolve()) for part in parts} <= listed
    assert str(pair_root.resolve()) not in listed
    assert verify_output_integrity(manifest.output_integrity)
    # The reuse path reads the same manifest back unchanged.
    assert run_repeatability_analysis(
        config, run_id="synthetic", feature_glob=glob, panel_ids=("primary_core",)
    ).output_paths == manifest.output_paths


def test_x8_stored_duplicate_signals_are_not_replicates(project):
    """Audit X8: identical waveforms under two study_ids are one recording."""
    import json

    config, glob = project
    root = config.features_root / "primary_core" / "development"
    for part in root.glob("split=*/part-*.parquet"):
        frame = pd.read_parquet(part)
        # Subject 1 visits 0 and 1 carry the same signal; everything else differs.
        frame["signal_sha256"] = [
            "same" if (row.subject_id == 1 and row.study_id in (10, 11)) else f"h{row.study_id}"
            for row in frame.itertuples()
        ]
        frame.to_parquet(part, index=False)
    output = config.work_root / "pairs_dedup"
    build_replicate_pairs(config, feature_glob=glob, marker_ids=MARKERS[:2], output_path=output)
    record = json.loads((output / "duplicate_signals.json").read_text())
    assert record == {"signal_hash_available": True, "identical_signal_pairs_removed": 1}
    pairs = pd.concat([pd.read_parquet(part) for part in replicate_pair_parts(output)])
    subject_one = pairs.loc[pairs["subject_id"] == 1]
    assert sorted(subject_one["first_study_id"].unique()) == [11]
