"""Tests für die elektromechanische Herzillustration (ohne OpenGL)."""
from __future__ import annotations

import numpy as np
import pytest

from vcg_biomarker.gui.heart3d.activation import ActivationSolver, AtrialPattern, Pattern, choose_pattern, ectopic_origin_vertex
from vcg_biomarker.gui.heart3d.engine import AnimationEngine
from vcg_biomarker.gui.heart3d.model import CHAMBER_LV, CHAMBER_RV, HeartModel, default_model
from vcg_biomarker.gui.heart3d.timeline import (LEADS_12, mechanical_events, timeline_from_payload, timeline_from_signal,
                                                 weissler)


@pytest.fixture(scope="module")
def model() -> HeartModel:
    return default_model()


@pytest.fixture(scope="module")
def synthetic_sinus():
    from vcg_biomarker.gui.heart3d import demo
    return demo.scenario_sinus().timeline()


def test_model_geometry(model):
    assert model.wall.n_vertices > 10_000
    assert (model.wall_chamber == CHAMBER_LV).sum() > 3_000
    assert (model.wall_chamber == CHAMBER_RV).sum() > 3_000
    assert model.wall_septal.sum() > 1_000
    assert model.wall_endocardial.sum() > 1_000
    assert set(np.unique(model.wall_segment[model.wall_chamber == CHAMBER_LV])) <= set(range(1, 18))
    for key in ("sa_node", "av_node", "his", "lv_apex", "rv_apex"):
        assert np.all(np.isfinite(model.landmarks[key]))
    # Apex liegt kaudal der Basis, Sinusknoten kranial des AV-Knotens
    assert model.landmarks["lv_apex"][2] < model.landmarks["lv_base"][2]
    assert model.landmarks["sa_node"][2] > model.landmarks["av_node"][2]
    for name in ("lv_mid_septal", "rv_anterior_papillary", "rv_apical", "lv_lateral_base"):
        assert len(model.purkinje_sites[name]) > 0


def test_activation_solver_rejects_nonphysical_parameters_and_disconnected_mesh(model):
    solver = ActivationSolver(model)
    original = dict(solver.params)
    for parameters in ({"v_muscle": 0.0}, {"v_purkinje": float("nan")},
                       {"purkinje_lv_u": 1.1}, {"rv_entry_offset_ms": -1.0},
                       {"unknown_parameter": 1.0}):
        with pytest.raises(ValueError):
            solver.set_params(**parameters)
        assert solver.params == original
    solver._edges_i = np.empty(0, dtype=np.int64)
    solver._edges_j = np.empty(0, dtype=np.int64)
    solver._edges_len = np.empty(0, dtype=float)
    solver._edge_transmural = np.empty(0, dtype=bool)
    with pytest.raises(ValueError, match="nicht vollständig erreichbar"):
        solver._solve({0: 0.0}, np.ones(solver._n))


def test_activation_patterns_order(model):
    solver = ActivationSolver(model)
    seg = model.wall_segment
    lv, rv = model.wall_chamber == CHAMBER_LV, model.wall_chamber == CHAMBER_RV
    normal = solver.ventricular(Pattern.NORMAL, 90.0).times_ms
    assert normal.min() >= 0 and np.percentile(normal, 97) == pytest.approx(90.0, abs=1.0)
    # RV-Ausflusstrakt gehört zu den zuletzt erregten Regionen
    assert normal[seg == 104].mean() > normal[lv].mean()
    lbbb = solver.ventricular(Pattern.LBBB, 150.0).times_ms
    assert lbbb[lv].mean() > lbbb[rv].mean() + 15.0
    rbbb = solver.ventricular(Pattern.RBBB, 140.0).times_ms
    assert rbbb[rv].mean() > rbbb[lv].mean() + 15.0
    paced = solver.ventricular(Pattern.RV_PACING, 160.0)
    assert paced.times_ms[model.purkinje_sites["rv_apical"]].max() < 5.0
    assert paced.focal_vertices.size > 0
    ectopic = solver.ventricular(Pattern.ECTOPIC, 150.0, origin_vertex=int(model.purkinje_sites["lv_lateral_base"][0])).times_ms
    assert ectopic[(seg == 5) | (seg == 6)].mean() < ectopic[rv].mean()
    warped = solver.ventricular(Pattern.NORMAL, 90.0, arc_fraction_half_time=0.65)
    assert warped.tier == "H" and warped.notes
    # Zeitverzerrung ändert die Reihenfolge nicht
    assert np.all(np.argsort(warped.times_ms, kind="stable")[:50] == np.argsort(normal, kind="stable")[:50])


def test_atrial_patterns(model):
    solver = ActivationSolver(model)
    sinus = solver.atrial(AtrialPattern.SINUS, 100.0)
    ra_src = model.atrial_sources["ra_wall"]["sa_node"]
    assert sinus.ra_times_ms[ra_src] == pytest.approx(0.0, abs=1e-3)
    assert sinus.la_times_ms.min() > 5.0  # linker Vorhof später
    retro = solver.atrial(AtrialPattern.RETROGRADE, 100.0)
    assert retro.ra_times_ms[model.atrial_sources["ra_wall"]["av_node"]] == pytest.approx(0.0, abs=1e-3)
    assert retro.ra_times_ms[ra_src] > 30.0


def test_weissler_and_events(synthetic_sinus):
    pep, lvet, qs2 = weissler(60.0)
    assert (round(pep), round(lvet), round(qs2)) == (107, 311, 420)
    beat = synthetic_sinus.beats[2]
    ev = mechanical_events(beat)
    assert beat.qrs_on < ev.av_close < ev.ao_open < ev.ao_close < ev.av_open
    assert abs(ev.ao_close - beat.t_end) < 0.08


def test_timeline_from_signal_detects_beats():
    fs = 500.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.zeros((len(t), 12))
    for k in range(11):
        r = 0.5 + k * 0.85
        for lead in range(12):
            signal[:, lead] += 1.0 * np.exp(-0.5 * ((t - r) / 0.012) ** 2) + 0.25 * np.exp(-0.5 * ((t - r - 0.28) / 0.05) ** 2)
    tl = timeline_from_signal(signal, fs, LEADS_12)
    assert 10 <= len(tl.beats) <= 12
    assert tl.rhythm_class == "regular"
    assert all(b.qrs_on < b.r_time < b.qrs_off for b in tl.beats)


def test_timeline_from_payload_uses_delineation():
    fs = 500.0
    signal = np.zeros((5000, 12))
    payload = {
        "lab_processing": {"analysis_sampling_rate_hz": 500.0},
        "beat_inventory": [
            {"beat_index": 0, "r_peak_sample": 500, "morphology_cluster_id": 0, "beat_type": "", "accepted": True},
            {"beat_index": 1, "r_peak_sample": 1000, "morphology_cluster_id": 1, "beat_type": "", "accepted": True},
            {"beat_index": 2, "r_peak_sample": 1500, "morphology_cluster_id": 0, "beat_type": "", "accepted": True},
        ],
        "delineation": {
            "accepted_beat_wave_segments": [
                {"beat_index": 0, "segment": "p", "start_sample": 410, "end_sample": 460},
                {"beat_index": 0, "segment": "qrs", "start_sample": 480, "end_sample": 525},
                {"beat_index": 0, "segment": "t", "start_sample": 560, "end_sample": 680},
                {"beat_index": 2, "segment": "qrs", "start_sample": 1480, "end_sample": 1525},
            ],
            "rhythm_context": {"class": "regular", "flags": ["stable_discrete_p_coupling"]},
        },
        "classical_criteria": {"lbbb_pattern": {"positive": False}, "rbbb_pattern": {"positive": True}},
        "feature_values": {"delta_wave_present_flag": 0.0, "qrs_arc_length_fraction_at_half_time": 0.55},
    }
    tl = timeline_from_payload(payload, signal, fs, LEADS_12)
    assert tl.source == "payload" and len(tl.beats) == 3
    b0 = tl.beats[0]
    assert b0.p_on == pytest.approx(0.82) and b0.qrs_on == pytest.approx(0.96) and b0.t_end == pytest.approx(1.36)
    assert tl.beats[1].beat_type == "?"  # morphology alone does not establish a ventricular focus
    assert tl.conduction_hints["rbbb"] is True and tl.conduction_hints["lbbb"] is False
    assert tl.conduction_hints["qrs_arc_length_fraction_at_half_time"] == pytest.approx(0.55)
    assert choose_pattern(tl.conduction_hints, "N")[0] == Pattern.RBBB


def test_engine_frames_are_finite_and_fast(model, synthetic_sinus):
    import time
    engine = AnimationEngine(model)
    engine.set_timeline(synthetic_sinus)
    beat = synthetic_sinus.beats[2]
    frame = engine.frame(beat.qrs_on + 0.05)
    assert frame.plan is not None and frame.plan.ventricular.pattern == Pattern.NORMAL
    wall = frame.positions["wall"]
    assert wall.shape == model.wall.vertices.shape and np.isfinite(wall).all()
    colors = frame.colors["wall"]
    assert colors.shape == (model.wall.n_vertices, 4) and colors.min() >= 0.0 and colors.max() <= 1.0
    # Gelbe Wellenfront während des QRS
    assert (colors[:, 0] > 0.9).mean() > 0.02
    # Kontraktion nach dem Ejektionsbeginn: Vertices bewegen sich zur Achse
    ev = frame.plan.events
    systole = engine.frame(ev.ao_open + 0.08)
    displacement = np.linalg.norm(systole.positions["wall"] - model.wall.vertices, axis=1)
    assert displacement.max() > 3.0
    diastole = engine.frame(beat.qrs_on - 0.05)
    assert np.linalg.norm(diastole.positions["wall"] - model.wall.vertices, axis=1).max() < 1.5
    t0 = time.time()
    for k in range(20):
        engine.frame(beat.qrs_on + 0.02 * k)
    assert (time.time() - t0) / 20 < 0.08


def test_scenarios_select_expected_patterns(model):
    from vcg_biomarker.gui.heart3d import demo
    engine = AnimationEngine(model)
    expected = {
        "Linksschenkelblock": Pattern.LBBB, "Rechtsschenkelblock": Pattern.RBBB,
        "Präexzitation": Pattern.WPW, "RV-Stimulation": Pattern.RV_PACING, "Sinusrhythmus 72/min": Pattern.NORMAL,
    }
    for name, pattern in expected.items():
        tl = demo.SCENARIOS[name]().timeline()
        engine.set_timeline(tl)
        plan = engine.plan_for(tl.beats[1])
        assert plan.ventricular.pattern == pattern, name
    tl = demo.scenario_bigeminy().timeline()
    engine.set_timeline(tl)
    ectopic = [b for b in tl.beats if b.beat_type == "V"][0]
    plan = engine.plan_for(ectopic)
    assert plan.ventricular.pattern == Pattern.ECTOPIC and plan.ventricular.focal_vertices.size == 1
    assert model.wall_chamber[int(plan.ventricular.focal_vertices[0])] == CHAMBER_LV
    tl = demo.scenario_av_block().timeline()
    engine.set_timeline(tl)
    frame = engine.frame(tl.atrial_events[2][0] + 0.04)
    assert "AV-Dissoziation" in frame.plan.atrial.label
    tl = demo.scenario_af().timeline()
    engine.set_timeline(tl)
    assert engine.plan_for(tl.beats[1]).atrial.pattern == AtrialPattern.FIBRILLATION


def test_ectopic_origin_from_vector(model):
    # Vektor nach links-inferior-anterior -> Fokus rechts-superior-posterior (RV/Basis)
    idx, label = ectopic_origin_vertex(model, np.array([1.0, 1.0, -0.5]), None)
    assert "Kors" in label
    pos = model.wall.vertices[idx]
    assert pos[0] < 0  # rechts vom Zentrum


# ----------------------------------------------------------------------------- Pathologie, Thorax, Schnitt, VCG
def test_pathology_estimate_localizes_st_vector(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.pathology import estimate_pathology
    inferior = estimate_pathology(model, demo.scenario_stemi_inferior().timeline())
    assert inferior.ischemia_kind == "transmural" and inferior.culprit == "RCA"
    assert "inferior" in inferior.territory
    # Läsion liegt kaudal (inferior) und auf der LV-Wand
    hot = inferior.ischemia > 0.6
    assert hot.sum() > 200 and model.wall.vertices[hot, 2].mean() < model.wall.vertices[:, 2].mean()
    assert (model.wall_chamber[hot] == CHAMBER_LV).mean() > 0.9
    anterior = estimate_pathology(model, demo.scenario_stemi_anterior().timeline())
    assert anterior.culprit == "LAD"
    hot = anterior.ischemia > 0.6
    assert model.wall.vertices[hot, 1].mean() < model.wall.vertices[:, 1].mean()  # anterior = -y
    sub = estimate_pathology(model, demo.scenario_subendocardial().timeline())
    assert sub.ischemia_kind == "subendocardial"
    assert sub.ischemia[model.wall_endocardial & (model.wall_chamber == CHAMBER_LV)].mean() > sub.ischemia[~model.wall_endocardial].mean()
    scar = estimate_pathology(model, demo.scenario_old_anterior_mi().timeline())
    assert scar.scar.max() > 0.9 and scar.ischemia.max() < 0.1
    normal = estimate_pathology(model, demo.scenario_sinus().timeline())
    assert not normal.findings and normal.ischemia.max() == 0.0


def test_pathology_uses_analysis_criteria(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.pathology import estimate_pathology
    tl = demo.scenario_sinus().timeline()
    payload = {"classical_criteria": {"sokolow_lyon": {"positive": True}, "low_voltage_qrs": {"positive": False},
                                      "bifid_p_wave": {"positive": True}}}
    est = estimate_pathology(model, tl, payload)
    assert np.abs(est.wall_offset).max() > 1.0 and est.la_scale > 1.0
    assert any(tier == "K" for tier, _, _ in est.findings)


def test_engine_applies_pathology_to_colors_and_motion(model):
    from vcg_biomarker.gui.heart3d import demo
    engine = AnimationEngine(model)
    tl = demo.scenario_old_anterior_mi().timeline()
    engine.set_timeline(tl)
    beat = tl.beats[2]
    frame = engine.frame(beat.qrs_on - 0.05)
    scar = engine.pathology.scar > 0.8
    assert frame.colors["wall"][scar, :3].mean() > frame.colors["wall"][~scar, :3].mean()  # Narbe blass
    ev = engine.plan_for(beat).events
    systole = engine.frame(ev.ao_open + 0.08)
    disp = np.linalg.norm(systole.positions["wall"] - engine._wall_rest, axis=1)
    assert disp[scar].mean() < 0.5 * disp[~scar & (model.wall_chamber == CHAMBER_LV)].mean()  # Akinesie
    engine.set_pathology_visible(False)
    frame2 = engine.frame(beat.qrs_on - 0.05)
    assert abs(frame2.colors["wall"][scar, :3].mean() - frame2.colors["wall"][~scar, :3].mean()) < 0.15


def test_torso_and_dipole():
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.torso import (ELECTRODE_BY_NAME, DipoleSimulator, VIRTUAL_PRESETS, surface_point,
                                                 torso_mesh)
    v, f = torso_mesh()
    assert len(v) > 2000 and f.max() < len(v)
    v1, v6 = ELECTRODE_BY_NAME["V1"].position, ELECTRODE_BY_NAME["V6"].position
    assert v1[1] < 0 and v6[0] > v1[0]  # V1 anterior, V6 links davon
    v9 = surface_point(*VIRTUAL_PRESETS["V9 (paravertebral links)"])
    assert v9[1] > 0  # posterior
    m = default_model()
    sim = DipoleSimulator(0.5 * (m.lv_frame.center + m.rv_frame.center))
    tl = demo.scenario_sinus().timeline()
    sim.set_timeline(tl)
    assert sim.available and sim.scale > 0 and len(sim.fit_r2) == 6
    y = sim.unipolar(v9)
    assert y.shape == (len(tl.signal),) and np.isfinite(y).all()
    pots = sim.surface_potentials(v, int(tl.beats[2].r_time * tl.fs))
    assert pots.shape == (len(v),) and np.abs(pots).max() > 0


def test_section_geometry_and_vcg_loop(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.section import SectionGeometry
    geo = SectionGeometry(model)
    assert len(geo.slab["wall"]) > 1000 and len(geo.slab["ra_wall"]) > 200 and len(geo.slab["la_wall"]) > 200
    xy = geo.project(model.landmarks["la_center"]) - geo.project(model.landmarks["lv_apex"])
    assert xy[0, 1] > 0  # Basis oben
    assert geo.project(model.landmarks["lv_center"])[0, 0] > geo.project(model.landmarks["rv_center"])[0, 0]  # LV rechts im Bild
    engine = AnimationEngine(model)
    tl = demo.scenario_sinus().timeline()
    engine.set_timeline(tl)
    loop = engine.vcg_loop(tl.beats[2])
    assert loop is not None and set(np.unique(loop["segment"])) == {0, 1, 2}
    radius = np.linalg.norm(loop["points"] - loop["centre"], axis=1)
    assert 20.0 < radius.max() < 60.0


def test_conduction_state_follows_pattern(model):
    from vcg_biomarker.gui.heart3d import demo
    engine = AnimationEngine(model)
    tl = demo.scenario_lbbb().timeline()
    engine.set_timeline(tl)
    b = tl.beats[2]
    f = engine.frame(b.qrs_on + 0.01)
    assert "lbb" in f.blocked and "rbb" in f.conduction and "lbb" not in f.conduction and "lbb_to_block" in f.conduction
    tl = demo.scenario_bigeminy().timeline()
    engine.set_timeline(tl)
    v = [x for x in tl.beats if x.beat_type == "V"][0]
    f = engine.frame(v.qrs_on + 0.03)
    assert "focal" in f.conduction and "retro_rbb" in f.conduction and "his" not in f.conduction
    f = engine.frame(v.qrs_on + 0.12)
    assert "retro_av" in f.conduction  # retrograde Vorhoferregung
    tl = demo.scenario_av_block().timeline()
    engine.set_timeline(tl)
    f = engine.frame(tl.atrial_events[2][0] + 0.05)
    assert "av" in f.blocked and "internodal" in f.conduction
    tl = demo.scenario_flutter().timeline()
    engine.set_timeline(tl)
    assert "flutter" in engine.frame(tl.beats[2].qrs_on - 0.15).conduction


def test_timeline_from_payload_accepts_lab_segment_names():
    fs = 500.0
    signal = np.zeros((3000, 12))
    payload = {"lab_processing": {"analysis_sampling_rate_hz": 500.0},
               "beat_inventory": [{"beat_index": 0, "r_peak_sample": 600, "morphology_cluster_id": 0, "beat_type": "unknown"},
                                  {"beat_index": 1, "r_peak_sample": 1100, "morphology_cluster_id": 0, "beat_type": "unknown", "morphology_outlier": True}],
               "delineation": {"accepted_beat_wave_segments": [
                   {"beat_index": 0, "segment": "qrs_complex", "start_sample": 570, "end_sample": 645},
                   {"beat_index": 0, "segment": "t_wave", "start_sample": 700, "end_sample": 820}],
                   "rhythm_context": {"class": "regular", "flags": ["no_consistent_discrete_p"], "atrial_frequency_hz": 5.0}}}
    tl = timeline_from_payload(payload, signal, fs, LEADS_12)
    assert tl.beats[0].qrs_duration_ms == pytest.approx(150.0) and tl.beats[1].beat_type == "?"
    assert tl.conduction_hints["atrial_frequency_hz"] == 5.0
    from vcg_biomarker.gui.heart3d.activation import choose_atrial_pattern
    assert choose_atrial_pattern("regular", tl.rhythm_flags, False, "N", atrial_frequency_hz=5.0) == AtrialPattern.FLUTTER


def test_wall_grid_and_forward_model(model):
    from vcg_biomarker.gui.heart3d.inverse import ForwardModel, activation_from_hypothesis, conduction_hypotheses
    solver = ActivationSolver(model)
    grid = solver.grid
    assert grid.n_nodes > 10_000 and grid.endo.sum() > 1000 and grid.epi.sum() > 1000
    assert 80 < grid.volume[grid.chamber == CHAMBER_LV].sum() / 1000 < 180  # cm³, in der Größenordnung der LV-Masse
    fm = ForwardModel(model, solver)
    H = {h.key: h for h in conduction_hypotheses()}
    act = activation_from_hypothesis(solver, H["normal"], 90.0)
    assert act.node_times_ms is not None and np.percentile(act.times_ms, 97) == pytest.approx(90.0, abs=1.0)
    t = np.arange(-15.0, 120.0, 2.0)
    d = fm.dipole(act.node_times_ms.astype(float), t)
    assert d.shape == (len(t), 3) and np.isfinite(d).all() and np.linalg.norm(d, axis=1).max() > 0
    # Initialkräfte nach anterior (Septum/RV), Freiwandphase nach inferior (datenkalibrierte Herzlage)
    vcg = np.column_stack([d[:, 0], -d[:, 2], d[:, 1]])
    initial = vcg[(t > 0) & (t < 15)].sum(0)
    assert initial[2] < 0
    mid = vcg[(t > 30) & (t < 50)].sum(0)
    assert mid[1] > 0
    # Block: linker Schenkel blockiert -> LV später als RV
    lbbb = activation_from_hypothesis(solver, H["lbbb"], 150.0)
    assert lbbb.times_ms[model.wall_chamber == CHAMBER_LV].mean() > lbbb.times_ms[model.wall_chamber == CHAMBER_RV].mean() + 15
    assert "lbb" in lbbb.blocked


def test_inverse_sweep_ranks_hypotheses(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.inverse import InverseFitter, hypotheses_for_beat
    from vcg_biomarker.gui.heart3d.torso import DipoleSimulator
    solver = ActivationSolver(model)
    tl = demo.scenario_bigeminy().timeline()
    sim = DipoleSimulator(0.5 * (model.lv_frame.center + model.rv_frame.center))
    sim.set_timeline(tl)
    fitter = InverseFitter(model, solver, sim, tl)
    beat = [b for b in tl.beats if b.beat_type == "V"][0]
    hs = hypotheses_for_beat(model, beat, tl.conduction_hints)
    assert all(h.family in ("focus", "pacing") for h in hs) and len(hs) > 30
    results = fitter.sweep(beat, hs[:12], refine=True)
    assert len(results) >= 12 and all(-1.0 <= r.score <= 1.0 for r in results)
    assert results[0].score >= results[-1].score
    assert results[0].simulated_leads.shape[1] == 12
    engine = AnimationEngine(model)
    engine.set_timeline(tl)
    engine.set_hypothesis(results[0].hypothesis)
    plan = engine.plan_for(beat)
    assert plan.pattern_tier == "H" and "Inverse Simulation" in plan.pattern_reason
    normal_beat = [b for b in tl.beats if b.beat_type == "N"][0]
    assert "Inverse" not in engine.plan_for(normal_beat).pattern_reason  # nur Schläge des Hypothesentyps


def test_ischemia_extent_follows_aldrich_and_st_vector(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.pathology import aldrich_area_at_risk, estimate_pathology, lead_based_territories, zone_direction_error_deg
    lv = model.wall_chamber == CHAMBER_LV
    for scenario in (demo.scenario_stemi_inferior, demo.scenario_stemi_anterior):
        est = estimate_pathology(model, scenario().timeline())
        elevated, _ = lead_based_territories(est.measures)
        target, _ = aldrich_area_at_risk(est.measures, elevated)
        frac = 100.0 * ((est.ischemia >= 0.5) & lv).sum() / lv.sum()
        assert abs(frac - target) < 6.0  # Ausdehnung folgt dem Aldrich-Risikogebiet
        assert zone_direction_error_deg(model, est.ischemia, est.measures.st_vector) < 35.0  # Zone liegt in ST-Vektor-Richtung


def test_patient_priors_and_t_wave(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.inverse import InverseFitter, conduction_hypotheses
    from vcg_biomarker.gui.heart3d.priors import patient_prior
    from vcg_biomarker.gui.heart3d.torso import DipoleSimulator
    p = patient_prior(70, "f", 160, 90)
    assert p.axis_shift_deg < 0 and p.duration_factor < 1.0 and "BMI" in p.note  # 70 J: Alterseffekt (×0,97) hebt BMI-Effekt der Leitung auf
    assert patient_prior().note.startswith("keine")
    solver = ActivationSolver(model)
    tl = demo.scenario_sinus().timeline()
    sim = DipoleSimulator(0.5 * (model.lv_frame.center + model.rv_frame.center))
    sim.set_timeline(tl)
    f = InverseFitter(model, solver, sim, tl)
    assert f.forward.solver is not solver
    original_solver_params = dict(solver.params)
    beat = tl.beats[2]
    res = f.evaluate(conduction_hypotheses()[0], beat)
    assert res.r_t_wave is not None and -1.0 <= res.r_t_wave <= 1.0 and res.simulated_t_leads.shape[1] == 12
    assert 0.8 <= res.time_scale_factor <= 1.35
    assert isinstance(res.time_scale_limited, bool)
    f.set_patient(70, "f", 160, 90)
    assert "Achsen-Vorannahme" in f.prior_note
    assert solver.params == original_solver_params
    first_size = f.forward.solver.params["size_factor"]
    assert first_size == pytest.approx(p.duration_factor)
    f.set_patient(70, "f", 160, 90)
    assert f.forward.solver.params["size_factor"] == pytest.approx(first_size)
    res2 = f.evaluate(conduction_hypotheses()[0], beat)
    assert np.isfinite(res2.score)
    assert patient_prior(None, "unknown").axis_shift_deg == 0
    assert patient_prior(float("nan"), "unknown", 40, 400).duration_factor == 1


def test_atrial_inverse_requires_verified_p_boundaries(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.inverse import InverseFitter, atrial_sweep
    from vcg_biomarker.gui.heart3d.torso import DipoleSimulator

    tl = demo.scenario_sinus().timeline()
    beat = tl.beats[2]
    assert beat.has_p and not beat.p_measured
    solver = ActivationSolver(model)
    sim = DipoleSimulator(0.5 * (model.lv_frame.center + model.rv_frame.center))
    sim.set_timeline(tl)
    assert atrial_sweep(InverseFitter(model, solver, sim, tl), beat) == []


def test_rhythm_scenarios_and_csp(model):
    from vcg_biomarker.gui.heart3d import demo
    from vcg_biomarker.gui.heart3d.engine import RhythmScenario
    from vcg_biomarker.gui.heart3d.inverse import activation_from_hypothesis, conduction_hypotheses, csp_hypotheses, Hypothesis
    engine = AnimationEngine(model)
    expected = {"Junktionaler Ersatzrhythmus": RhythmScenario.JUNCTIONAL_ESCAPE, "Ventrikulärer Ersatzrhythmus": RhythmScenario.VENTRICULAR_ESCAPE,
                "AVNRT 170/min": RhythmScenario.AVNRT, "Kammertachykardie (Reentry)": RhythmScenario.VT_REENTRY, "Sinusrhythmus 72/min": RhythmScenario.AUTO}
    for name, scenario in expected.items():
        tl = demo.SCENARIOS[name]().timeline()
        engine.set_timeline(tl)
        assert engine.plan_for(tl.beats[3]).scenario == scenario, name
    tl = demo.SCENARIOS["AVNRT 170/min"]().timeline()
    engine.set_timeline(tl)
    for scenario, key in ((RhythmScenario.AVRT_ORTHO, "retro_accessory"), (RhythmScenario.AVRT_ANTI, "retro_av"), (RhythmScenario.VT_REENTRY, "vt_loop")):
        engine.set_rhythm_scenario(scenario)
        b = tl.beats[3]
        keys = set()
        for dt in (0.01, 0.06, 0.13, 0.2):
            keys |= set(engine.frame(b.qrs_on + dt).conduction)
        assert key in keys, scenario
    engine.set_rhythm_scenario(RhythmScenario.AUTO)
    # Leitungssystem-Stimulation: His korrigiert einen kompletten LSB, nicht einen verzögerten Schenkel
    solver = ActivationSolver(model)
    H = {h.key: h for h in conduction_hypotheses()}
    def raw(h):
        return float(np.percentile(activation_from_hypothesis(solver, h, 100.0, exact_scale=False, scale_limits=(1, 1)).times_ms, 97))
    assert raw(csp_hypotheses(H["lbbb"])[0]) < raw(H["lbbb"]) - 25
    assert abs(raw(csp_hypotheses(H["lbb_d60"])[0]) - raw(H["lbb_d60"])) < 5
    edit = Hypothesis("edit", "Editor", "conduction", extra_delays=(("lbb_anterior", float("inf")),))
    assert "lbb_anterior" in activation_from_hypothesis(solver, edit, 100.0).blocked
