from types import SimpleNamespace

import pandas as pd
import pytest

from vcg_biomarker.gui.lab_pair_source import resolve_lab_pair_source, resolve_review_source


def setup_pair(tmp_path):
    config = SimpleNamespace(master_root=tmp_path/"master", mimic_ecg_root=tmp_path/"signals")
    config.master_root.mkdir()
    config.mimic_ecg_root.mkdir()
    for study in (10, 11):
        (config.mimic_ecg_root/f"{study}.hea").write_text(f"{study} 1 500 10\n{study}.dat 16 1000/mV 16 0 0 0 0 I\n")
        (config.mimic_ecg_root/f"{study}.dat").write_bytes(bytes(20))
    master = pd.DataFrame(dict(subject_id=[1, 1], study_id=[10, 11], split=["train", "train"],
        ecg_time=pd.to_datetime(["2020-01-01", "2020-01-02"]), hadm_id=[100, 100],
        relative_waveform_path=["10", "11"]))
    master.to_parquet(config.master_root/"ecg_master_index.parquet")
    return config, master, dict(subject_id=1, first_study_id=10, second_study_id=11)


def test_resolves_both_development_records_and_refuses_wrong_patient(tmp_path):
    config, _, pair = setup_pair(tmp_path)
    assert resolve_lab_pair_source(config, pair, "first").name == "10.hea"
    assert resolve_lab_pair_source(config, pair, "second").name == "11.hea"
    with pytest.raises(ValueError, match="unique matching"):
        resolve_lab_pair_source(config, {**pair, "subject_id": 2}, "first")


def test_review_link_requires_unique_identity_split_and_matching_waveform(tmp_path):
    config, master, _ = setup_pair(tmp_path)
    row = {"subject_id": 1, "study_id": 10, "split": "train", "waveform_path": "10"}
    assert resolve_review_source(config, row).name == "10.hea"
    with pytest.raises(ValueError, match="development"):
        resolve_review_source(config, {**row, "split": "locked_test"})
    with pytest.raises(ValueError, match="conflicts"):
        resolve_review_source(config, {**row, "waveform_path": "11"})
    master = pd.concat([master, master.iloc[:1]], ignore_index=True)
    master.to_parquet(config.master_root / "ecg_master_index.parquet")
    with pytest.raises(ValueError, match="unique"):
        resolve_review_source(config, row)


@pytest.mark.parametrize("failure", ["locked", "cross_split", "duplicate", "chronology", "admission", "escape", "header_escape"])
def test_refuses_unsafe_pair_or_source(tmp_path, failure):
    config, master, pair = setup_pair(tmp_path)
    if failure == "locked":
        master.loc[1, "split"] = "locked_test"
    elif failure == "cross_split":
        master.loc[1, "split"] = "validation"
    elif failure == "duplicate":
        master = pd.concat([master, master.iloc[:1]], ignore_index=True)
    elif failure == "chronology":
        master.loc[1, "ecg_time"] = master.loc[0, "ecg_time"]
    elif failure == "admission":
        master.loc[1, "hadm_id"] = 101
    elif failure == "escape":
        master.loc[0, "relative_waveform_path"] = "../outside"
    else:
        (config.mimic_ecg_root/"10.hea").write_text("10 1 500 10\n../outside.dat 16 1000/mV 16 0 0 0 0 I\n")
    master.to_parquet(config.master_root/"ecg_master_index.parquet")
    with pytest.raises(ValueError):
        resolve_lab_pair_source(config, pair, "first")


def test_actual_wfdb_pair_opens_local_case_without_analysis(tmp_path, monkeypatch):
    import os
    # Do not silently force later desktop/PDF tests into Windows offscreen Qt.
    # Respect an explicit caller choice; use the application's native platform.
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen")
    from dataclasses import replace
    import numpy as np
    import wfdb
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.widgets import mimic_project_widget
    from vcg_biomarker.tests.test_provenance_integrity import _config
    source_config, master, pair = setup_pair(tmp_path)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    waveform = np.tile(np.sin(np.arange(500)/20)[:, None], (1, 12))*.1
    for name in ("10", "11"):
        wfdb.wrsamp(name, fs=500, units=["mV"]*12, sig_name=leads,
            p_signal=waveform, fmt=["16"]*12, write_dir=str(source_config.mimic_ecg_root))
    config = replace(_config(tmp_path/"isolated"), mimic_ecg_root=source_config.mimic_ecg_root)
    config.master_root.mkdir(parents=True, exist_ok=True)
    master.to_parquet(config.master_root/"ecg_master_index.parquet")
    monkeypatch.setattr(mimic_project_widget, "load_or_discover_config", lambda: config)
    app = QApplication.instance() or QApplication([])
    widget = mimic_project_widget.MimicProjectWidget()
    widget._poll_timer.stop()
    assert widget.lab_scroll.widget() is widget.expert_lab
    assert widget.lab_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    endpoint_tab = widget.expert_lab.endpoint_review_tab
    assert endpoint_tab in widget.expert_lab.extra_research_tabs
    widget.expert_lab.mode.setCurrentIndex(1)
    assert widget.expert_lab.tabs.isTabVisible(endpoint_tab)
    widget.expert_lab.blind.setChecked(True)
    assert not widget.expert_lab.tabs.isTabVisible(endpoint_tab)
    widget.expert_lab.blind.setChecked(False)
    assert widget.expert_lab.tabs.isTabVisible(endpoint_tab)
    widget.expert_lab.mode.setCurrentIndex(0)
    assert not widget.expert_lab.tabs.isTabVisible(endpoint_tab)
    monkeypatch.setattr(widget, "analyze_selected_case", lambda: pytest.fail("No automatic analysis allowed"))
    errors = []
    monkeypatch.setattr(mimic_project_widget.QMessageBox, "warning", lambda *args: errors.append(args))
    try:
        widget._open_dose_pair_ecg(pair, "first")
        app.processEvents()
        assert errors == []
        assert widget.workspace_tabs.currentIndex() == 3
        assert widget.case_combo.currentData()
        assert len(widget.case_library.list_cases()) == 1
        assert widget._current_case_ecg.shape == (500, 12)
        assert len(widget.ecg_plot.listDataItems()) == 12
        assert "keine Wellen-/Markeranalyse" in widget.ecg_plot.plotItem.titleLabel.text
        assert widget._current_case_fiducials == {}
        assert not widget.edit_fiducials_button.isEnabled()
        assert "Keine automatische Neuberechnung" in widget.case_status.text()
        assert not list(config.cases_root.glob("*/analyses/*/analysis.json"))
        case_id = widget.case_combo.currentData()
        # A second import selects its target before rendering; no old-case load.
        original_load = widget.case_library.load_signal
        loaded = []
        def counted_load(selected):
            loaded.append(selected)
            return original_load(selected)
        monkeypatch.setattr(widget.case_library, "load_signal", counted_load)
        widget._open_dose_pair_ecg(pair, "second")
        assert loaded == [widget.case_combo.currentData()]
        assert widget.case_combo.currentData() != case_id
        case_id = widget.case_combo.currentData()
        monkeypatch.setattr(widget.case_library, "load_signal", lambda _: (np.full((10, 12), np.inf), 500.))
        widget._open_dose_pair_ecg(pair, "second")
        assert widget.ecg_plot.listDataItems() == []
        assert widget._current_case_ecg is None
        assert "nicht darstellbar" in widget.case_status.text()
        assert widget.case_combo.currentData() == case_id
        widget.case_combo.clear()
        widget._case_changed()
        assert widget.ecg_plot.plotItem.titleLabel.text == ""
        assert widget.case_status.text() == ""
    finally:
        widget.close()
