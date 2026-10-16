import hashlib

import pandas as pd
import pytest

from vcg_biomarker.tests.test_pilot_feature_projection import model_fixture


def context_fixture(tmp_path, duplicate=False):
    model = model_fixture(tmp_path)
    relative = "original/evidence/clinical_context_features.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    pd.DataFrame(dict(study_id=[10, 10 if duplicate else 11],
                      potassium_delta_hours=[-2., 3.], potassium_nearest=[4., 5.],
                      unused=["large unused context", "other"])).to_parquet(path, index=False)
    model._hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, relative, path


def test_timeline_pushdown_cache_and_case_switch(tmp_path, monkeypatch):
    model, relative, _ = context_fixture(tmp_path)
    read = pd.read_parquet
    calls = []
    def counted(path, **kwargs):
        calls.append(kwargs)
        return read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", counted)
    result = model.timeline(10)
    assert result.hours_from_ecg.tolist() == [-2., 0.]
    assert calls[0]["filters"] == [("study_id", "==", 10)]
    assert "unused" not in calls[0]["columns"]
    pd.testing.assert_frame_equal(model.timeline(10), result)
    assert len(calls) == 1
    assert model.timeline(11).hours_from_ecg.tolist() == [0., 3.]
    assert len(model._case_frames[relative][1]) == 1
    assert not model._frames
    assert model.timeline(999).empty


def test_case_cache_integrity_and_defensive_copy(tmp_path):
    model, relative, path = context_fixture(tmp_path)
    columns = ("study_id", "potassium_nearest")
    row = model._case_frame(relative, columns, 10)
    row.loc[0, "potassium_nearest"] = -999
    assert model._case_frame(relative, columns, 10).potassium_nearest.iloc[0] == 4.
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        model._case_frame(relative, columns, 10)


def test_ambiguous_context_is_not_silently_selected(tmp_path):
    model, _, _ = context_fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        model.timeline(10)
