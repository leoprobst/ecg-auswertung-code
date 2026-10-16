"""Tests für das VCG-Analyselabor: Maße, Drehrichtung, Analysekopplung, Lesung, Fenster."""
from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.gui.vcglab import measures as M
from vcg_biomarker.gui.vcglab.analysis import LEADS_12, analyze_signal, derive_vcg, quick_record
from vcg_biomarker.gui.vcglab.reading import read_beat, st_territory


def test_loop_tail_uses_only_recorded_samples_and_clips_at_boundaries():
    from vcg_biomarker.gui.vcglab.widgets import loop_tail_points, loop_revealed_points

    xyz = np.column_stack([np.arange(100), np.zeros(100), -np.arange(100)])
    np.testing.assert_array_equal(loop_tail_points(xyz, 50, 1000, 8), xyz[43:51])
    np.testing.assert_array_equal(loop_tail_points(xyz, -5, 1000, 8), xyz[:1])
    np.testing.assert_array_equal(loop_tail_points(xyz, 1000, 1000, 8), xyz[92:100])
    assert loop_tail_points(np.empty((0, 3)), 0, 1000).shape == (0, 3)
    np.testing.assert_array_equal(loop_revealed_points(xyz, 50), xyz[:51])
    np.testing.assert_array_equal(loop_revealed_points(xyz, -5), xyz[:1])
    np.testing.assert_array_equal(loop_revealed_points(xyz, 1000), xyz)
    assert loop_revealed_points(np.empty((0, 3)), 0).shape == (0, 3)


def test_presentation_beat_selection_uses_record_time_not_display_order():
    from types import SimpleNamespace
    from vcg_biomarker.gui.widgets.signal_presentation import beat_at_time

    beats = [SimpleNamespace(index=0, r=100), SimpleNamespace(index=1, r=300)]
    record = SimpleNamespace(beats=beats, fs=100.0, ecg=np.zeros((500, 12)))
    assert beat_at_time(record, 0.0) is beats[0]
    assert beat_at_time(record, 2.0) is beats[0]
    assert beat_at_time(record, 3.0) is beats[1]
    assert beat_at_time(record, 20.0) is beats[1]


def test_presentation_cameras_expand_distance_for_wide_panes():
    from vcg_biomarker.gui.widgets.signal_presentation import presentation_camera_scale

    assert presentation_camera_scale(0, 300) == 1.0
    assert presentation_camera_scale(670, 300) == pytest.approx(1.0, rel=0.01)
    assert presentation_camera_scale(945, 330) > 1.25
    assert presentation_camera_scale(500, 500) == 1.0


def test_beat_constellation_preserves_exclusions_and_record_time(core_record):
    from vcg_biomarker.gui.widgets.signal_presentation import beat_constellation_points

    points = beat_constellation_points(core_record)
    assert len(points) == len(core_record.beats)
    for point, beat in zip(points, core_record.beats):
        assert point["index"] == beat.index
        assert point["time"] == pytest.approx(beat.r / core_record.fs)
        assert point["accepted"] is beat.accepted
        assert point["outlier"] is beat.outlier
        assert point["cluster"] == beat.cluster
        assert point["peak_mv"] is None or point["peak_mv"] >= 0


def test_uncertainty_lens_does_not_turn_missing_reference_into_negative(core_record):
    from vcg_biomarker.gui.widgets.signal_presentation import uncertainty_lens_data

    rows = uncertainty_lens_data(core_record)
    assert len(rows) == 4
    assert f"/{len(core_record.beats)}" in rows[0][1]
    assert "P:" in rows[1][1] and "T:" in rows[1][1]
    assert "nicht geprüft" in rows[2][1]
    assert "unbekannt, nicht negativ" in rows[2][2]
    assert "keine Ortswahrscheinlichkeit" in rows[3][1]


def test_guided_presentation_keeps_source_labels_and_manual_step_control(qapp):
    from vcg_biomarker.gui.heart3d.demo import scenario_sinus
    from vcg_biomarker.gui.heart3d.engine import AnimationEngine
    from vcg_biomarker.gui.heart3d.model import default_model
    from vcg_biomarker.gui.vcglab.analysis import quick_record
    from vcg_biomarker.gui.widgets.signal_presentation import SignalPresentationDialog

    timeline = scenario_sinus().timeline()
    record = quick_record(timeline.signal, timeline.fs, timeline.lead_names)
    model = default_model()
    engine = AnimationEngine(model)
    engine.set_timeline(timeline)
    dialog = SignalPresentationDialog(engine=engine, model=model, record=record, synthetic=True)
    assert dialog.draw_loop.isChecked() and dialog.vcg.trace_reveal
    dialog.draw_loop.setChecked(False)
    assert not dialog.vcg.trace_reveal
    dialog.draw_loop.setChecked(True)
    dialog._toggle_tour()
    assert dialog._tour_step == 0
    assert "EKG: simuliertes Signal" in dialog.note.text()
    dialog._set_tour_step(4)
    assert dialog.ecg_tabs.currentIndex() == 4
    assert "keine Ortswahrscheinlichkeit" in dialog.uncertainty.cards[3].findChildren(type(dialog.note))[1].text()
    dialog._advance_tour()
    assert dialog._tour_step is None
    assert dialog.note.text() == dialog._base_note
    dialog.close()


def test_public_demo_capsule_is_hash_checked_and_not_production_input(tmp_path):
    import shutil
    from vcg_biomarker.gui.demo_capsules import (PUBLIC_PTBXL_CAPSULE, load_public_demo_signal,
                                                  verify_demo_capsule)

    signal, fs, names, manifest = load_public_demo_signal()
    assert signal.shape == (5000, 12) and fs == 500
    assert tuple(names) == tuple(manifest["expected_leads"])
    assert manifest["may_enter_production_analysis"] is False
    assert manifest["may_be_used_as_clinical_ground_truth"] is False
    for name in ("manifest.json", "HR00001.hea", "HR00001.mat"):
        shutil.copy2(PUBLIC_PTBXL_CAPSULE / name, tmp_path / name)
    verify_demo_capsule(tmp_path)
    with (tmp_path / "HR00001.hea").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Hashprüfung"):
        verify_demo_capsule(tmp_path)


def test_ecg_lab_exposes_public_demo_separately_from_loaded_case(qapp):
    from vcg_biomarker.gui.widgets.expert_ecg_lab import ExpertECGLab

    lab = ExpertECGLab()
    labels = [action.text() for action in lab.presentation_button.menu().actions()]
    assert any("Geladenen Fall" in label for label in labels)
    assert any("Synthetische Demo" in label for label in labels)
    assert any("PTB-XL-Demofall" in label for label in labels)
    lab.close()


def test_presentation_lead_grid_has_simultaneous_samples_and_shared_scale(qapp):
    from types import SimpleNamespace
    from vcg_biomarker.gui.widgets.signal_presentation import SynchronizedLeadGrid

    signal = np.zeros((1000, 12))
    signal[200, 0] = 4.0
    signal[200, 1] = -2.0
    timeline = SimpleNamespace(signal=signal, fs=500.0,
                               lead_names=("I", "II", "III", "aVR", "aVL", "aVF",
                                           "V1", "V2", "V3", "V4", "V5", "V6"))
    grid = SynchronizedLeadGrid(timeline)
    grid.set_time(0.4)
    assert len(grid._curves) == 12
    assert all(plot.getViewBox().viewRange()[1][1] >= 4.0 for plot in grid._plots)
    for curve, index in grid._curves:
        if index in (0, 1):
            x, y = curve.getData()
            assert x[np.argmax(np.abs(y))] == pytest.approx(0.4)
    grid.close()


def test_vector_bridge_uses_same_record_sample_for_eight_inputs_and_vcg(core_record):
    from vcg_biomarker.gui.widgets.signal_presentation import INDEPENDENT_LEADS, vector_bridge_values

    record = core_record
    t = 0.9
    sample, inputs, xyz = vector_bridge_values(record, t)
    expected = [record.lead_names.index(name) for name in INDEPENDENT_LEADS]
    np.testing.assert_allclose(inputs, record.ecg[sample, expected])
    np.testing.assert_allclose(xyz, record.xyz("kors", corrected=True)[sample])
    assert len(inputs) == 8 and len(xyz) == 3


def test_hypothesis_duel_rejects_misaligned_curves_and_uses_shared_scale(qapp):
    from types import SimpleNamespace
    from vcg_biomarker.gui.heart3d.hypothesis_duel import (
        HypothesisDuelDialog, common_display_limit, paired_qrs_residuals,
    )

    t = np.arange(100, dtype=float)
    measured = np.zeros((100, 12))
    measured[20, 0] = 3.0
    simulated = np.zeros_like(measured)
    simulated[20, 0] = 2.0
    result = SimpleNamespace(hypothesis=SimpleNamespace(label="Modell A"), score=0.7,
                             r_mean=0.6, time_scale_factor=1.0, time_scale_limited=False,
                             r_leads={"I": 0.8}, simulated_leads=np.zeros((60, 12)))
    assert common_display_limit(measured, simulated) >= 3.0
    residuals = paired_qrs_residuals(t, measured, simulated, measured.copy(), 60)
    assert residuals.shape == (2, 12, 2)
    assert residuals[0, 0, 1] == 0
    assert residuals[0, 0, 0] > 0
    assert paired_qrs_residuals(t, measured, simulated, simulated, None) is None
    with pytest.raises(ValueError, match="gleich lange"):
        HypothesisDuelDialog(time_ms=t[:-1], measured=measured, simulated_a=simulated,
                             simulated_b=simulated, result_a=result, result_b=result)
    dialog = HypothesisDuelDialog(time_ms=t, measured=measured, simulated_a=simulated,
                                  simulated_b=simulated, result_a=result, result_b=result)
    assert len(dialog.plots.ci.items) >= 12
    assert dialog.residual_table.rowCount() == 2
    assert dialog.residual_table.columnCount() == 12
    dialog.close()


def test_saved_lab_boundaries_do_not_turn_display_estimates_into_vcg_measures():
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.vcglab.analysis import record_from_payload
    from vcg_biomarker.gui.vcglab.widgets import scene_for

    timeline = demo.scenario_sinus().timeline()
    fs = timeline.fs
    payload = {'lab_processing': {'analysis_sampling_rate_hz': fs},
               'beat_inventory': [
                   {'beat_index': 0, 'r_peak_sample': 500, 'accepted': True,
                    'morphology_cluster_id': 0, 'beat_type': 'unknown'},
                   {'beat_index': 1, 'r_peak_sample': 1000, 'accepted': True,
                    'morphology_cluster_id': 1, 'beat_type': 'unknown'}],
               'delineation': {'accepted_beat_wave_segments': [
                   {'beat_index': 0, 'segment': 'qrs_complex', 'start_sample': 480,
                    'end_sample': 525, 'status': 'not_evaluable'},
                   {'beat_index': 1, 'segment': 'qrs_complex', 'start_sample': 980,
                    'end_sample': 1025, 'status': 'measurable'}]}}
    record = record_from_payload(timeline.signal, fs, timeline.lead_names, payload)
    assert record.source == 'payload'
    assert [beat.accepted for beat in record.beats] == [False, True]
    assert all(beat.t_end is None for beat in record.beats)
    assert scene_for(record, record.beats[0], 'kors', True).measures is None
    assert record.beats[0].label.startswith('#1')


def _circle(plane_axes, n=100, cw=True):
    """Ebene Kreisschleife, die auf dem Bildschirm der gewählten Ebene im Uhrzeigersinn läuft (y nach unten)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    sx, sy = np.cos(th), np.sin(th)  # in einem y-nach-unten-Rahmen: rechts → unten → links = Uhrzeigersinn
    if not cw:
        sy = -sy
    xyz = np.zeros((n, 3))
    (cx, fx), (cy, fy) = M.PLANES[plane_axes]
    xyz[:, cx] = sx / fx
    xyz[:, cy] = sy / fy
    return xyz


def test_angles_and_conventions():
    assert M.azimuth_deg(np.array([1.0, 0, 0])) == pytest.approx(0.0)
    assert M.azimuth_deg(np.array([0, 0, 1.0])) == pytest.approx(90.0)  # posterior
    assert M.elevation_deg(np.array([0, 1.0, 0])) == pytest.approx(0.0)  # inferior
    assert M.elevation_deg(np.array([1.0, 0, 0])) == pytest.approx(90.0)
    assert M.frontal_axis_deg(np.array([0.0, 1.0, 0.0])) == pytest.approx(90.0)
    assert M.angle_deg(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(90.0)
    assert M.angle_deg(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) == pytest.approx(180.0)
    seg = np.tile([0.5, 0.0, 0.0], (101, 1))
    assert M.area_vector_mv_ms(seg, 500.0)[0] == pytest.approx(0.5 * 200.0)  # 100 Intervalle à 2 ms


@pytest.mark.parametrize("plane", ["frontal", "transversal", "sagittal"])
def test_inscription_matches_screen_frame(plane):
    kind, minor, area = M.inscription(_circle(plane, cw=True), plane)
    assert kind == "cw" and minor < 0.05 and area == pytest.approx(np.pi, rel=0.05)
    kind, _, _ = M.inscription(_circle(plane, cw=False), plane)
    assert kind == "ccw"


def test_figure_eight_detected():
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    xyz = np.zeros((200, 3))
    xyz[:, 0] = np.sin(th)
    xyz[:, 1] = np.sin(2 * th)  # Lemniskate: zwei gegenläufige Lappen
    kind, minor, _ = M.inscription(xyz, "frontal")
    assert kind == "figure8" and minor > 0.4


def test_loop_geometry_planar_circle():
    g = M.loop_geometry(_circle("frontal"))
    assert g.planarity_ratio < 1e-6
    assert g.roundness == pytest.approx(1.0, abs=1e-6)
    assert g.perimeter_mv == pytest.approx(2 * np.pi, rel=0.01)
    assert g.planar_area_mv2 == pytest.approx(np.pi, rel=0.01)
    assert abs(g.normal[2]) == pytest.approx(1.0)


def _synthetic_beat(fs=500, qrs_dir=(1.0, 0.6, 0.4), t_dir=(1.0, 0.6, 0.4), t_amp=0.4):
    n = int(0.7 * fs)
    t = np.arange(n) / fs
    q = np.exp(-0.5 * ((t - 0.20) / 0.012) ** 2)
    tw = np.exp(-0.5 * ((t - 0.42) / 0.035) ** 2)
    d = np.asarray(qrs_dir) / np.linalg.norm(qrs_dir)
    e = np.asarray(t_dir) / np.linalg.norm(t_dir)
    xyz = np.outer(q * 1.5, d) + np.outer(tw * t_amp, e)
    b = M.BeatBounds(qrs_on=int(0.16 * fs), qrs_off=int(0.245 * fs), t_on=int(0.33 * fs), t_end=int(0.52 * fs))
    return xyz, b


def test_beat_measures_concordant_and_discordant():
    xyz, b = _synthetic_beat()
    m = M.beat_measures(xyz, 500.0, b, sex="M")
    assert m.get("qrs_t_angle_peak") == pytest.approx(0.0, abs=1.0)
    assert m.get("qrs_max_magnitude") == pytest.approx(1.5, rel=0.02)
    assert m.get("tcrt") > 0.95
    xyz2, b2 = _synthetic_beat(t_dir=(-1.0, -0.6, -0.4))
    m2 = M.beat_measures(xyz2, 500.0, b2, sex="M")
    assert m2.get("qrs_t_angle_peak") == pytest.approx(180.0, abs=1.0)
    assert m2.get("tcrt") < -0.95
    st = next(x for x in m2.measures if x.key == "qrs_t_angle_mean")
    assert st.status == "hoch" and "Kardys" in st.limit.source
    assert m2.get("qrs_duration") == pytest.approx(85.0, abs=2.5)


def test_reference_fills_missing_limits():
    xyz, b = _synthetic_beat()
    ref = {"qrs_max_magnitude": {"p2_5": 0.8, "p97_5": 1.2, "n": 100}}
    m = M.beat_measures(xyz, 500.0, b, sex="F", reference=ref)
    x = next(v for v in m.measures if v.key == "qrs_max_magnitude")
    assert x.status == "hoch" and "MIMIC" in x.limit.source


def test_serial_difference_thresholds():
    xyz, b = _synthetic_beat()
    a = M.beat_measures(xyz, 500.0, b)
    shifted = xyz.copy()
    shifted[b.qrs_off:] += np.array([0.0, 0.08, 0.0])  # ST-Hebung inferior 0,08 mV
    c = M.beat_measures(shifted, 500.0, b)
    d = M.serial_difference(a, c)
    assert d is not None and d.delta_st_mv > 0.05 and d.delta_vg_mv_ms > 16.2 and d.ischemia_flag
    assert st_territory(d.delta_st_vector) == "inferior"


def test_transform_fidelity_high_for_dipole_signal():
    xyz, _ = _synthetic_beat()
    eight = xyz @ M.DOWER_FORWARD.T  # I, II, V1..V6
    ecg = np.zeros((len(xyz), 12))
    names = list(LEADS_12)
    for j, lead in enumerate(M.DOWER_LEADS):
        ecg[:, names.index(lead)] = eight[:, j]
    ecg[:, 2] = ecg[:, 1] - ecg[:, 0]
    ecg[:, 3] = -(ecg[:, 0] + ecg[:, 1]) / 2
    ecg[:, 4] = ecg[:, 0] - ecg[:, 1] / 2
    ecg[:, 5] = ecg[:, 1] - ecg[:, 0] / 2
    kors = derive_vcg(ecg, LEADS_12, "kors")
    fid = M.transform_fidelity(ecg, kors, LEADS_12)
    assert len(fid) == 8 and min(fid.values()) > 0.9


def test_quick_record_and_scenes():
    from vcg_biomarker.gui.heart3d import demo
    tl = demo.scenario_sinus().timeline()
    rec = quick_record(tl.signal, tl.fs, tl.lead_names, "sinus")
    assert rec.source == "quick" and len(rec.beats) >= 5
    assert set(rec.vcg_raw) == {"kors", "inverse_dower"}
    beat = rec.beats[rec.representative_index()]
    m = rec.measures_for(beat)
    assert m.get("qrs_max_magnitude") > 0.3
    from vcg_biomarker.gui.vcglab.widgets import scene_for
    scene = scene_for(rec, None, "kors", True, overlays=True)
    assert scene is not None and scene.segment("qrs") is not None and len(scene.overlays) >= 3


@pytest.fixture(scope="module")
def core_record():
    from vcg_biomarker.gui.heart3d import demo
    tl = demo.scenario_lbbb().timeline()
    return analyze_signal(tl.signal, tl.fs, tl.lead_names, "lbbb")


def test_core_pipeline_and_lbbb_reading(core_record):
    rec = core_record
    assert rec.source == "core" and rec.delineation is not None
    assert rec.median_beat is not None and rec.median_ecg is not None
    assert all(m in rec.origin for m in ("kors", "inverse_dower"))
    m = rec.median_measures("kors")
    assert m is not None and m.get("qrs_duration") >= 120
    r = read_beat(m, qrs_ms=m.get("qrs_duration"), criteria=rec.criteria, rhythm_class=rec.rhythm_class, rhythm_flags=rec.rhythm_flags,
                  fidelity=rec.fidelity["kors"])
    titles = [x.title for x in r.readings]
    assert any("Linksschenkelblock" in t for t in titles)
    assert all(x.tier in ("K", "E", "H", "i") for x in r.readings)


def test_variability_over_dominant_beats(core_record):
    per = [core_record.measures_for(b) for b in core_record.dominant_beats()]
    v = M.beat_variability(per)
    assert v.n_beats >= 3 and v.qrs_axis_sd_deg is not None and v.qrs_axis_sd_deg < 15


@pytest.fixture
def qapp():
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication.instance() or QApplication([])


def test_window_and_preview(qapp, core_record, tmp_path, monkeypatch):
    from vcg_biomarker.gui.vcglab.widgets import MiniVCGWidget, VCGLabWindow
    w = VCGLabWindow()
    w.set_record(core_record)
    qapp.processEvents()
    assert w.scene_data is not None
    assert w.tree.topLevelItemCount() >= 6
    assert "Linksschenkelblock" in w.reading.toPlainText()
    w.beat_combo.setCurrentIndex(1)  # repräsentativer Schlag
    w.method_combo.setCurrentIndex(1)  # inverse Dower
    w.overlay_check.setChecked(True)
    w.set_cursor(10)
    assert w.scene_data.cursor == 10
    w._advance()
    assert w.scene_data.cursor > 10
    out = tmp_path / "masse.csv"
    monkeypatch.setattr("vcg_biomarker.gui.vcglab.widgets.QFileDialog.getSaveFileName", lambda *a, **k: (str(out), ""))
    w._export_measures()
    text = out.read_text(encoding="utf-8-sig")
    assert "qrs_t_angle_mean" in text and "Drehrichtung" in text and len(text.splitlines()) > 40
    w.close()
    mini = MiniVCGWidget(auto_analysis=False, size=120)
    mini.set_record(core_record)
    assert "lbbb" in mini.caption.text()
    mini.stop()
    mini.close()
    qapp.processEvents()


def test_panel_loads_source_lazily_in_background(qapp):
    """set_source rechnet nicht sofort; erst die sichtbare Ansicht löst die Auswertung aus."""
    import time
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.vcglab.widgets import VCGLabPanel
    tl = demo.scenario_sinus().timeline()
    panel = VCGLabPanel()
    panel.set_source(tl.signal, tl.fs, tl.lead_names, "fall", 61, "M")
    assert panel.record is None and panel._pending is not None
    panel.ensure_record()
    deadline = time.time() + 60
    while panel.record is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert panel.record is not None and panel.record.sex == "M"
    assert panel.sex_combo.currentData() == "M"  # Grenzwerte folgen den Patientendaten
    assert panel.scene_data is not None
    panel.stop()
    panel.deleteLater()
    qapp.processEvents()


def test_expert_lab_card_opens_vcg_window(qapp, tmp_path):
    from types import SimpleNamespace
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.widgets.expert_ecg_lab import ExpertECGLab
    tl = demo.scenario_sinus().timeline()
    case = SimpleNamespace(case_id="case-vcg", source_sha256="abc", pseudonym="Synthetic", metadata={"age": 61, "sex": "M"})
    library = SimpleNamespace(root=tmp_path, load_case=lambda _: case, load_signal=lambda _: (tl.signal, tl.fs), list_cases=lambda: [case])
    lab = ExpertECGLab()
    lab.load_case(library, case.case_id)
    qapp.processEvents()
    assert "VCG-Labor" not in [lab.tabs.tabText(i) for i in range(lab.tabs.count())]  # kein zusätzlicher Reiter
    assert lab.vcg_window is None  # Fenster entsteht erst auf Wunsch
    assert lab.vcg_mini is not None and lab.heart is not None
    # The spatial viewers have their own row and cannot cover ECG leads.
    assert lab.heart_overlay is not lab.vcg_overlay
    assert lab.vcg_overlay.collapsed and not lab.heart_overlay.collapsed
    assert "Synthetic" in lab.vcg_mini.caption.text()
    lab.vcg_overlay.toggle()
    lab.show()
    qapp.processEvents()
    assert not lab.vcg_overlay.collapsed
    canvas = lab.canvas.geometry()
    for card in (lab.vcg_overlay, lab.heart_overlay):
        assert card.parentWidget() is lab.viewer_strip
    assert not lab.vcg_overlay.geometry().intersects(lab.heart_overlay.geometry())
    assert not canvas.intersects(lab.viewer_strip.geometry())
    lab._open_vcg_window()
    qapp.processEvents()
    assert lab.vcg_window is not None and lab.vcg_window.isVisible()
    assert lab.vcg_window.panel._pending is not None  # Fall ist übergeben, Auswertung läuft im Fenster
    lab.set_dark(False)  # beide Karten und das Fenster folgen dem hellen Thema
    qapp.processEvents()
    assert "#ffffff" in lab.vcg_overlay.styleSheet() or "255, 255, 255" in lab.vcg_overlay.styleSheet()
    assert lab.vcg_window.panel._dark is False
    lab.clear_analysis()
    qapp.processEvents()
    lab.close()
    lab.deleteLater()
    qapp.processEvents()
