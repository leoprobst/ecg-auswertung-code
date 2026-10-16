"""Stage 19 must keep what it has computed (A153, 04.10.2026).

The trajectory layer built one Python dict per subject and marker -- about
23 million per block, ~20 GB -- and the replicate-pair reuse check compared a
written part against the unfiltered candidate count, so it never matched.
Every memory pause therefore restarted the whole stage. These tests pin:

* the columnar summary is the reference ``summarize_trajectories`` to
  floating-point rounding, on series with gaps, single recordings, equal
  times, short spans and non-numeric values;
* a finished trajectory part survives an interruption and is not recomputed;
* a finished replicate-pair part is reused even when the quality floor drops
  some candidate pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.trajectory import (
    TrajectoryOptions,
    _summarize_block,
    build_replicate_pairs,
    build_trajectory_frame,
    replicate_pair_parts,
    summarize_trajectories,
)
from vcg_biomarker.tests.test_trajectory_blocking import MARKERS, project  # noqa: F401


def _hard_series(seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(40):
        n = int(rng.integers(1, 7))
        base = pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(subject))
        offsets = np.sort(rng.integers(0, 600, size=n))  # minutes
        if subject % 7 == 0 and n > 1:
            offsets[:] = offsets[0]  # all at one time: no spread
        if subject % 5 == 0:
            offsets = offsets // 120  # very short spans
        for k, minutes in enumerate(offsets):
            rows.append({
                "subject_id": subject,
                "split": "train" if subject % 2 else "validation",
                "study_id": subject * 100 + k,
                "ecg_time": base + pd.Timedelta(minutes=int(minutes)),
                "a": rng.normal() if rng.random() > 0.3 else np.nan,
                "b": rng.normal() * 50,
                "c": np.nan,
                "d": rng.choice([1.0, np.inf, np.nan, 2.0]),
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("minimum_points,span", [(3, 1.0), (2, 0.1), (4, 6.0)])
def test_columnar_summary_matches_the_reference(minimum_points, span):
    series = _hard_series()
    options = TrajectoryOptions(minimum_points=minimum_points, minimum_span_hours=span)
    markers = ("a", "b", "c", "d", "absent")
    reference = summarize_trajectories(series, marker_ids=markers, options=options)
    columnar = _summarize_block(series, markers, options).to_pandas()
    key = ["subject_id", "marker_id"]
    reference = reference.sort_values(key).reset_index(drop=True)
    columnar = columnar.sort_values(key).reset_index(drop=True)[list(reference.columns)]
    assert len(columnar) == len(reference)
    for column in reference.columns:
        left, right = reference[column], columnar[column]
        if pd.api.types.is_float_dtype(left) or pd.api.types.is_float_dtype(right):
            np.testing.assert_allclose(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=column)
        else:
            assert left.astype(str).tolist() == right.astype(str).tolist(), column


def test_a_finished_trajectory_part_is_not_recomputed(project, tmp_path, monkeypatch):  # noqa: F811
    import vcg_biomarker.mimic.trajectory as module

    config, glob = project
    options = TrajectoryOptions(minimum_signal_quality=0.0)
    monkeypatch.setattr(module, "TRAJECTORY_MARKER_BLOCK", 5)
    calls = []
    real = module._write_trajectory_part

    def interrupted(series, block, opts, part_path, **kwargs):
        if len(calls) == 2:
            raise RuntimeError("memory pause")
        calls.append(part_path.name)
        return real(series, block, opts, part_path, **kwargs)

    monkeypatch.setattr(module, "_write_trajectory_part", interrupted)
    output = tmp_path / "trajectories.parquet"
    with pytest.raises(RuntimeError, match="memory pause"):
        build_trajectory_frame(config, feature_glob=glob, marker_ids=MARKERS,
                               output_path=output, options=options)
    assert calls == ["part-000.parquet", "part-001.parquet"]

    calls.clear()
    monkeypatch.setattr(module, "_write_trajectory_part", lambda *a, **k: calls.append(a[3].name) or real(*a, **k))
    build_trajectory_frame(config, feature_glob=glob, marker_ids=MARKERS,
                           output_path=output, options=options)
    assert "part-000.parquet" not in calls and "part-001.parquet" not in calls
    assert calls  # the unfinished blocks were computed

    monkeypatch.setattr(module, "_write_trajectory_part", real)
    fresh = build_trajectory_frame(config, feature_glob=glob, marker_ids=MARKERS,
                                   output_path=tmp_path / "fresh.parquet", options=options)
    pd.testing.assert_frame_equal(pd.read_parquet(output), pd.read_parquet(fresh))
    assert not (tmp_path / ".trajectories.parquet.parts").exists()


def test_pair_parts_are_reused_when_the_quality_floor_drops_pairs(project, tmp_path):  # noqa: F811
    import vcg_biomarker.mimic.trajectory as module

    config, glob = project
    # One recording below the quality floor removes its pairs from every part.
    for part in __import__("pathlib").Path(glob.split("split=")[0]).glob("split=*/part-*.parquet"):
        frame = pd.read_parquet(part)
        frame.loc[frame.study_id == frame.study_id.min(), "signal_quality_score"] = 0.1
        frame.to_parquet(part, index=False)
    out = tmp_path / "pairs"
    original = module.TRAJECTORY_MARKER_BLOCK
    module.TRAJECTORY_MARKER_BLOCK = 4
    try:
        build_replicate_pairs(config, feature_glob=glob, marker_ids=MARKERS, output_path=out)
        stamps = {p.name: p.stat().st_mtime_ns for p in replicate_pair_parts(out)}
        build_replicate_pairs(config, feature_glob=glob, marker_ids=MARKERS, output_path=out)
    finally:
        module.TRAJECTORY_MARKER_BLOCK = original
    assert len(stamps) > 1
    for part in replicate_pair_parts(out):
        assert part.stat().st_mtime_ns == stamps[part.name], f"{part.name} rewritten"
