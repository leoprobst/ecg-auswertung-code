import hashlib
import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.pilot_viewmodel import PilotViewModel


def model_fixture(tmp_path):
    files = []
    for i in range(2):
        relative = f"original/features/1000/split=train/part-{i}.parquet"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(dict(subject_id=[i], study_id=[i+10], extraction_status=["ok"],
                                  unused=["not needed"]))
        if i == 0:
            frame["marker"] = 42.
        frame.to_parquet(path, index=False)
        files.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path / "workspace.json").write_text(json.dumps(dict(version="1.0.0",
        production_resume_allowed=False, locked_test_read=False, run_id="fixture",
        pilot_sizes=[1000], files=files)))
    return PilotViewModel(tmp_path)


def test_projection_preserves_records_missing_marker_and_bounded_cache(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    read = pd.read_parquet
    calls = []
    def projected(path, **kwargs):
        calls.append(kwargs.get("columns"))
        return read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", projected)
    columns = ("subject_id", "study_id", "extraction_status", "marker")
    frame = model.features(1000, columns)
    assert list(frame.columns) == list(columns)
    assert len(frame) == 2 and frame.marker.iloc[0] == 42 and pd.isna(frame.marker.iloc[1])
    assert all("unused" not in names for names in calls)
    pd.testing.assert_frame_equal(model.features(1000, columns), frame)
    frame.loc[0, "marker"] = -999
    assert model.features(1000, columns).marker.iloc[0] == 42.
    assert len(calls) == 2 and not model._frames
    model.features(1000, (*columns[:3], "unused"))
    assert len(model._feature_frames) == 1


def test_feature_cache_hit_still_checks_integrity(tmp_path):
    model = model_fixture(tmp_path)
    columns = ("subject_id", "study_id", "extraction_status", "marker")
    model.features(1000, columns)
    path = tmp_path / next(iter(model._hashes))
    path.write_bytes(b"changed feature file")
    with pytest.raises(ValueError, match="integrity"):
        model.features(1000, columns)


@pytest.mark.parametrize("columns", [None, ("subject_id", "study_id", "extraction_status", "marker")])
def test_missing_identity_schema_is_not_marker_missingness(tmp_path, columns):
    model = model_fixture(tmp_path)
    name = next(iter(model._hashes))
    path = tmp_path / name
    pd.read_parquet(path).drop(columns="study_id").to_parquet(path, index=False)
    # Define an internally hash-consistent but structurally invalid fixture.
    model._hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="record identity"):
        model.features(1000, columns)


def test_absent_marker_keeps_all_records_and_empty_inventory_is_rejected(tmp_path):
    model = model_fixture(tmp_path)
    frame = model.features(1000, ("subject_id", "study_id", "extraction_status", "absent"))
    assert len(frame) == 2 and frame.absent.isna().all()
    with pytest.raises(ValueError, match="Unknown pilot size"):
        model.features(999)
    model._hashes.clear()
    model._feature_frames.clear()
    with pytest.raises(ValueError, match="partitions missing"):
        model.features(1000)


def test_availability_filters_labels_and_reuses_only_current_endpoint(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    relative = "original/labels/endpoint_index_cohorts.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    labels = pd.DataFrame(dict(subject_id=[0, 1, 0, 1], study_id=[10, 11, 10, 11],
        label_id=["A", "A", "B", "B"], label_state=["positive", "unknown", "negative", "positive"],
        eligibility_state=["eligible"]*4, evidence_summary=["fixture"]*4, time_delta_hours=[0.]*4))
    labels.to_parquet(path, index=False)
    model._hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    read = pd.read_parquet
    calls = []
    def counted(source, **kwargs):
        if source == path:
            calls.append(kwargs.get("filters"))
        return read(source, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", counted)
    counts, joined = model.availability(1000, "A", "marker")
    assert counts.n.tolist() == [2, 2, 1, 1]
    assert joined.label_state.tolist() == ["positive", "unknown"]
    model.availability(1000, "A", "absent")
    assert calls == [[("label_id", "==", "A")]]
    model.availability(1000, "B", "marker")
    assert calls[-1] == [("label_id", "==", "B")]
    assert list(model._endpoint_labels) == ["B"]
    assert not model._frames
    path.write_bytes(b"modified labels")
    with pytest.raises(ValueError, match="integrity"):
        model.availability(1000, "B", "marker")
