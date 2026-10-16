"""E11: marker evaluation in processes equals the thread/serial path."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import marker_process_pool as pool_module
from vcg_biomarker.mimic import validation
from vcg_biomarker.mimic.validation import ValidationSpec


def _mixed_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "subject_id": np.arange(rows, dtype=np.int64),
            "label": (np.arange(rows) % 3 == 0).astype(np.int8),
            "value": rng.normal(size=rows),
            "flag": np.arange(rows) % 2 == 0,
            "count32": np.arange(rows, dtype=np.int32),
            "split": ["train"] * rows,
            "tier": pd.Categorical(["good", "poor"] * (rows // 2)),
            "nullable": pd.array([1, None] * (rows // 2), dtype="Int64"),
            "text": pd.array(["a", None] * (rows // 2), dtype="string"),
            "when": pd.date_range("2020-01-01", periods=rows, freq="D"),
        },
        index=pd.RangeIndex(100, 100 + rows),
    )


def test_shared_chunk_round_trip_preserves_frames_exactly():
    frame = _mixed_frame()
    other = frame.iloc[::2].reset_index(drop=True)
    quality = np.linspace(0, 1, len(frame))
    chunk = pool_module.SharedChunk(
        frames={"train": frame, "validation": other, "endpoint_universe": None},
        arrays={"train_quality": quality, "validation_quality": None},
        constants={"endpoint_id": "e", "bootstrap_iterations": 3},
    )
    try:
        payload = pool_module._attach(chunk.name, chunk.token)
        pd.testing.assert_frame_equal(payload["frames"]["train"], frame, check_exact=True)
        pd.testing.assert_frame_equal(payload["frames"]["validation"], other, check_exact=True)
        assert list(payload["frames"]["train"].columns) == list(frame.columns)
        assert payload["frames"]["endpoint_universe"] is None
        np.testing.assert_array_equal(payload["arrays"]["train_quality"], quality)
        assert payload["arrays"]["validation_quality"] is None
        assert payload["constants"] == {"endpoint_id": "e", "bootstrap_iterations": 3}
        # Numeric columns are views on the shared block, not private copies.
        shared_value = payload["frames"]["train"]["value"].to_numpy()
        assert shared_value.base is not None
        assert pool_module.shared_bytes_needed(
            {"train": frame}, {"q": quality}
        ) >= frame["value"].nbytes + quality.nbytes
    finally:
        pool_module._release_attached()
        chunk.close()


def _endpoint(rows: int, offset: int, split: str, rng) -> pd.DataFrame:
    label = (rng.random(rows) < 0.3).astype(np.int8)
    frame = pd.DataFrame(
        {
            "split": split,
            "subject_id": np.repeat(np.arange(offset, offset + rows // 2), 2)[:rows],
            "study_id": np.arange(offset, offset + rows),
            "label": label,
            "label_state": np.where(label == 1, "positive", "negative"),
            "eligibility_state": "eligible",
            "signal_quality_score": rng.uniform(0.2, 1.0, rows),
        }
    )
    for index in range(4):
        values = rng.normal(size=rows) + label * (0.2 * index)
        values[rng.random(rows) < 0.1] = np.nan
        frame[f"marker_{index}"] = values
    frame["classical_sokolow"] = rng.normal(3.0, 1.0, rows) + label
    frame["classical_sokolow_positive"] = frame["classical_sokolow"] > 3.5
    return frame


def _normalized(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _normalized(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_normalized(item) for item in value.tolist()]
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return "NaN"
    if isinstance(value, np.generic):
        return value.item()
    return value


def test_process_chunk_equals_serial_evaluation(monkeypatch):
    rng = np.random.default_rng(17)
    train = _endpoint(240, 0, "train", rng)
    validation_frame = _endpoint(160, 10_000, "validation", rng)
    universe = pd.concat((train, validation_frame), ignore_index=True)
    features = ("marker_0", "marker_1", "marker_2", "marker_3", "classical_sokolow")
    panel_map = {name: "primary_core" for name in features[:4]}
    panel_map["classical_sokolow"] = "classical"
    specification = ValidationSpec(
        run_id="e11",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "1", "classical": "1"},
        development_bootstrap_iterations=40,
        random_seed=13,
    )
    arguments = dict(
        endpoint_id="endpoint",
        feature_ids=features,
        panel_map=panel_map,
        train=train,
        validation=validation_frame,
        endpoint_universe=universe,
        specification=specification,
        bootstrap_iterations=40,
        worker_count=3,
        train_quality=None,
        validation_quality=None,
        feature_index_offset=128,
    )
    monkeypatch.delenv("VCG_RESOURCE_GRANT_FILE", raising=False)
    serial = validation._evaluate_endpoint_markers(
        **{**arguments, "worker_count": 1}
    )
    import vcg_biomarker.mimic.resources as resources

    monkeypatch.setattr(resources, "available_memory_gb", lambda: 64.0)
    monkeypatch.setattr(resources, "recommend_dynamic_workers", lambda requested, **_: requested)
    state: dict = {}
    try:
        processed = validation._evaluate_marker_chunk_in_processes(
            state, progress=None, **arguments
        )
        # A second chunk reuses the same pool.
        again = validation._evaluate_marker_chunk_in_processes(
            state, progress=None, **arguments
        )
    finally:
        if state.get("pool") is not None:
            state.pop("pool").close()
    assert processed is not None
    assert [item.feature_id for item in processed] == list(features)
    assert _normalized(processed) == _normalized(serial)
    assert _normalized(again) == _normalized(serial)
    assert any(item.classical_records for item in serial)
    assert all(item.failure is None for item in serial)


def test_process_path_declines_without_memory_and_mode_follows_environment(monkeypatch):
    import vcg_biomarker.mimic.resources as resources

    monkeypatch.setattr(resources, "available_memory_gb", lambda: 2.5)
    frame = _mixed_frame()
    frame["signal_quality_score"] = 0.5
    state: dict = {}
    declined = validation._evaluate_marker_chunk_in_processes(
        state,
        endpoint_id="e", feature_ids=("value",), panel_map={"value": "primary_core"},
        train=frame, validation=frame, endpoint_universe=None,
        specification=None, bootstrap_iterations=0, worker_count=4,
        train_quality=None, validation_quality=None, feature_index_offset=0,
        progress=None,
    )
    assert declined is None and "pool" not in state

    monkeypatch.delenv("VCG_RESOURCE_GRANT_FILE", raising=False)
    monkeypatch.delenv(pool_module.PROCESS_ENV, raising=False)
    assert pool_module.process_mode_requested() is False
    monkeypatch.setenv("VCG_RESOURCE_GRANT_FILE", "grant.json")
    assert pool_module.process_mode_requested() is True
    monkeypatch.setenv(pool_module.PROCESS_ENV, "0")
    assert pool_module.process_mode_requested() is False
