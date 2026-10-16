import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets.dose_pair_widget import DosePairWidget, dose_display_sample


def settle(widget):
    deadline = time.monotonic()+10
    while widget.loading and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    assert not widget.loading


def pairs():
    return pd.DataFrame(dict(drug_class="beta", dose_itemid=1, dose_unit="mg", subject_id=[1, 1, 2, None],
        first_study_id=[10, 11, 20, 30], second_study_id=[11, 12, 21, 31],
        dose_amount=[5., 10., 20., 30.], dose_rank=[.25, .5, .75, 1.],
        dose_rank_status=["eligible", "eligible", "excluded", "eligible"], delta_marker=[1., 2., 3., np.inf]))


def test_group_counts_sampling_and_nonfinite_values():
    frame = pairs()
    original = frame.copy(deep=True)
    points, info = dose_display_sample(frame, "marker", limit=1)
    assert info == dict(n_total=4, n_valid=2, n_patients=1, n_unknown_patient_rows=0, n_shown=1)
    selected_index = points.index[0]
    assert points.first_study_id.iloc[0] == frame.first_study_id.loc[selected_index]
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(points, dose_display_sample(frame, "marker", limit=1)[0])
    frame.loc[1, "dose_itemid"] = 2
    with pytest.raises(ValueError, match="exactly one"):
        dose_display_sample(frame, "marker")


def test_nullable_rank_and_unknown_patient_counts():
    frame = pairs()
    frame["dose_rank"] = pd.Series([pd.NA, 1.2, .75, .8], dtype="Float64")
    frame.loc[3, "delta_marker"] = 4.
    points, info = dose_display_sample(frame, "marker", rank=True)
    assert info["n_valid"] == 1 and info["n_unknown_patient_rows"] == 1
    assert points.iloc[0, 0] == .8


def test_real_projected_parquet_view_switch_and_reset(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = DosePairWidget()
    path = tmp_path/"pairs.parquet"
    frame = pd.concat([pairs(), pairs().assign(dose_itemid=2, dose_unit="mcg", dose_amount=500.)], ignore_index=True)
    frame.to_parquet(path, index=False)
    widget.set_source(path)
    settle(widget)
    app.processEvents()
    assert widget.group.count() == 2
    assert widget.counts["n_total"] == 4
    assert widget.plot.listDataItems()[0].getData()[0].tolist() == [5., 10.]
    scatter = widget.plot.listDataItems()[0].scatter
    widget._point_clicked(scatter, [scatter.points()[1]])
    assert widget.selected_pair == {"subject_id": 1., "first_study_id": 11, "second_study_id": 12}
    requests = []
    widget.open_ecg_requested.connect(lambda pair, side: requests.append((pair, side)))
    widget.open_pair_buttons[1].click()
    assert requests == [(widget.selected_pair, "second")]
    widget._point_clicked(scatter, scatter.points())
    assert widget.selected_pair is None and not widget.copy_pair.isEnabled()
    widget.scale.setCurrentIndex(1)
    assert widget.selected_pair is None
    assert widget.plot.listDataItems()[0].getData()[0].tolist() == [.25, .5]
    widget.group.setCurrentIndex(1)
    widget.scale.setCurrentIndex(0)
    settle(widget)
    assert widget.plot.listDataItems()[0].getData()[0].tolist() == [500., 500.]
    widget.set_source(None)
    assert widget.counts == {} and widget.plot.listDataItems() == []
    pd.DataFrame().to_parquet(path)
    widget.set_source(path)
    settle(widget)
    assert widget.source is None and "Keine" in widget.status.text()
    pd.DataFrame(dict(bad=[1])).to_parquet(path)
    widget.set_source(path)
    settle(widget)
    assert "Inkompatibler" in widget.status.text()
    assert widget.plot.listDataItems() == []
    widget.close()


def test_project_refresh_connects_dose_artifact_and_clears_on_new_run(tmp_path, monkeypatch):
    from vcg_biomarker.gui.widgets import mimic_project_widget
    from vcg_biomarker.tests.test_provenance_integrity import _config
    config = _config(tmp_path/"isolated")
    monkeypatch.setattr(mimic_project_widget, "load_or_discover_config", lambda: config)
    app = QApplication.instance() or QApplication([])
    widget = mimic_project_widget.MimicProjectWidget()
    widget._poll_timer.stop()
    root = tmp_path/"run"
    (root/"within_subject").mkdir(parents=True)
    pairs().to_parquet(root/"within_subject"/"dose_paired_changes.parquet", index=False)
    try:
        widget._refresh_targeted_designs(root)
        settle(widget.dose_pair_view)
        app.processEvents()
        assert widget.dose_pair_view.counts["n_valid"] == 2
        widget._refresh_targeted_designs(tmp_path/"empty_run")
        assert widget.dose_pair_view.counts == {}
        assert widget.dose_pair_view.plot.listDataItems() == []
    finally:
        widget.close()


def test_bounded_cache_avoids_repeat_queries_and_invalidates_changed_source(tmp_path, monkeypatch):
    from vcg_biomarker.gui.widgets import dose_pair_widget as module
    app = QApplication.instance() or QApplication([])
    source = tmp_path/"pairs.parquet"
    pairs().to_parquet(source)
    real_connect = module.duckdb.connect
    calls = []
    def connect(*args, **kwargs):
        calls.append(kwargs)
        return real_connect(*args, **kwargs)
    monkeypatch.setattr(module.duckdb, "connect", connect)
    widget = DosePairWidget()
    widget.set_source(source)
    settle(widget)
    assert len(calls) == 2
    widget.scale.setCurrentIndex(1)
    widget.set_source(source)
    app.processEvents()
    assert len(calls) == 2
    assert widget.scale.currentIndex() == 1
    assert all(call["config"]["threads"] == 1 for call in calls)
    assert all(len(sample[0]) <= 6000 for sample in widget._samples.values())
    pd.concat([pairs(), pairs()], ignore_index=True).to_parquet(source)
    widget.draw()
    assert widget.counts == {} and widget.plot.listDataItems() == []
    assert widget._samples == {}
    widget.set_source(source)
    settle(widget)
    assert len(calls) == 4 and widget.counts["n_total"] == 8
    widget.set_source(None)
    assert widget._samples == {} and widget._source_signature is None
    widget.close()


def test_async_latest_request_only_keeps_gui_responsive(tmp_path, monkeypatch):
    from threading import Event
    from PyQt6.QtCore import QTimer
    from vcg_biomarker.gui.widgets import dose_pair_widget as module
    app = QApplication.instance() or QApplication([])
    started, release = Event(), Event()
    real_load = module._load_dose_view
    calls = []
    def delayed(kind, path, *args):
        calls.append((kind, path.name))
        if path.name == "first.parquet" and kind == "metadata":
            started.set()
            if not release.wait(10):
                raise TimeoutError("Test release missing")
        return real_load(kind, path, *args)
    monkeypatch.setattr(module, "_load_dose_view", delayed)
    first, ignored, last = [tmp_path/name for name in ("first.parquet", "ignored.parquet", "last.parquet")]
    pairs().to_parquet(first)
    pairs().to_parquet(ignored)
    pd.concat([pairs()]*3, ignore_index=True).to_parquet(last)
    widget = DosePairWidget()
    try:
        widget.set_source(first)
        assert started.wait(2)
        tick = []
        QTimer.singleShot(0, lambda: tick.append(True))
        widget.set_source(ignored)
        widget.set_source(last)
        app.processEvents()
        assert tick and widget.counts == {}
        assert widget._pending[1][1] == last
        release.set()
        settle(widget)
        assert widget.counts["n_total"] == 12
        assert calls == [("metadata", "first.parquet"), ("metadata", "last.parquet"), ("sample", "last.parquet")]
    finally:
        release.set()
        widget.close()


def test_closing_during_read_does_not_apply_late_results(tmp_path, monkeypatch):
    from threading import Event
    from vcg_biomarker.gui.widgets import dose_pair_widget as module
    app = QApplication.instance() or QApplication([])
    release = Event()
    real_load = module._load_dose_view
    def delayed(*args):
        if not release.wait(10):
            raise TimeoutError("Test release missing")
        return real_load(*args)
    monkeypatch.setattr(module, "_load_dose_view", delayed)
    path = tmp_path/"pairs.parquet"
    pairs().to_parquet(path)
    widget = DosePairWidget()
    widget.set_source(path)
    future = widget._active[1]
    widget.close()
    assert not widget._load_timer.isActive()
    release.set()
    future.result(timeout=10)
    widget._poll_load()
    app.processEvents()
    assert widget.source == path and widget.counts == {}
    assert widget.group.count() == 0
