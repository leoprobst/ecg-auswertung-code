"""Mechanism and population learning tests, without production patient analysis."""
from dataclasses import replace
import json
import numpy as np
import pytest

from vcg_biomarker.simulation.ionic import CellExperiment, compile_model, simulate_cell
from vcg_biomarker.simulation.learning import (
    PARAMETERS, OUTPUTS, admission_target, ap_features, compatible_population, predict_emulator, train_emulator,
)
from vcg_biomarker.simulation.monodomain import StimulusEvent, assemble, block_mesh, simulate_monodomain
from vcg_biomarker.simulation.substrate import RegionalSubstrate, coronary_territories, transported_extracellular_field


def test_channel_and_regional_fields_actually_modify_the_equations():
    base = CellExperiment(beats=1)
    m, f, _, _ = compile_model(base)
    _, modified, _, _ = compile_model(replace(base, channel_scales={"IKr": .5}))
    y = np.array(m.initial_values(as_floats=True))
    y[0] = 0.
    assert not np.array_equal(f(0, y, 0), modified(0, y, 0))
    _, vector, _, _ = compile_model(base, True, ("potassium_mmol_l", "scale:INa"))
    values = np.array(vector(0., np.repeat(y[:, None], 2, axis=1), np.zeros(2),
                            {"potassium_mmol_l": np.array([5.4, 7.2]), "scale:INa": np.array([1., .5])}))
    assert np.isfinite(values).all() and not np.array_equal(values[:, 0], values[:, 1])
    np.testing.assert_allclose(values[:, 0], f(0, y, 0), atol=1e-9)
    with pytest.raises(ValueError):
        replace(base, channel_scales={"SCN5A_mutation": .5}).validate()
    with pytest.raises(ValueError):
        replace(base, channel_scales={"INa": np.nan}).validate()


def test_substrate_programmed_pacing_and_coupling():
    mesh = block_mesh((3, 2, 2))
    substrate = RegionalSubstrate(potassium_mmol_l=np.linspace(5.4, 7.2, len(mesh.xyz_mm)),
                                 channel_scales={"INa": np.linspace(1., .7, len(mesh.xyz_mm))},
                                 element_coupling_scale=np.full(len(mesh.tetra), .5))
    normal, _, _ = assemble(mesh)
    reduced, _, _ = assemble(mesh, element_coupling_scale=substrate.element_coupling_scale)
    np.testing.assert_allclose(reduced.toarray(), .5*normal.toarray())
    events = [StimulusEvent(0., .5, (0, 1, 2, 3)), StimulusEvent(15., .5, (8, 9, 10, 11))]
    times, voltage, meta = simulate_monodomain(mesh, CellExperiment(beats=1), duration_ms=25,
                                              events=events, substrate=substrate)
    assert voltage.shape == (len(times), len(mesh.xyz_mm))
    assert len(meta["programmed_events"]) == 2
    assert meta["regional_fields"]["potassium_mmol_l"][-1] == 7.2
    with pytest.raises(ValueError):
        StimulusEvent(40., .5, (0,)).validate(12, 25)
    with pytest.raises(ValueError):
        RegionalSubstrate(potassium_mmol_l=np.full(12, 15.)).fields(12, len(mesh.tetra))


def test_coronary_variation_not_a_fixed_diagnostic_territory():
    segments = np.arange(1, 18)
    right = coronary_territories(segments, "right")
    left = coronary_territories(segments, "left")
    assert not np.array_equal(right["weights"], left["weights"])
    np.testing.assert_allclose(right["weights"].sum(1), 1)
    assert "not_individual" in right["interpretation"]
    with pytest.raises(ValueError):
        coronary_territories(np.array([0, 1]), "right")
    with pytest.raises(ValueError):
        coronary_territories(segments, "unknown")


def test_transport_conserves_mass_and_perfusion_washes_out():
    from scipy.sparse import diags
    n = 8
    lap = diags([np.ones(n-1), np.r_[-1, np.full(n-2, -2), -1], np.ones(n-1)], [-1, 0, 1]).tocsr()*.001
    initial = np.r_[8., np.full(n-1, 5.4)]
    times = np.arange(0, 1001, 100.)
    values = transported_extracellular_field(lap, np.ones(n), initial, 5.4, np.zeros(n), np.zeros(n), times)
    np.testing.assert_allclose(values.sum(1), initial.sum())
    washed = transported_extracellular_field(lap, np.ones(n), initial, 5.4, np.full(n, .001), np.zeros(n), times)
    assert abs(washed[-1].mean()-5.4) < abs(values[-1].mean()-5.4)
    assert washed.min() >= 5.4-1e-10


def test_conduction_variants_have_real_paths_and_blocks():
    from vcg_biomarker.simulation.model import Experiment, anatomical_tissue, operator
    from vcg_biomarker.simulation.conduction import VARIANTS, extend_network
    base = anatomical_tissue()
    for name in VARIANTS:
        mesh = extend_network(base, name)
        if name == "legacy":
            continue
        assert len(mesh.xyz) > len(base.xyz)
        assert "coronary_sinus_musculature" in mesh.branches
        assert any(k.startswith("lbb_septal_from_") for k in mesh.branches)
        matrix, *_ = operator(mesh, Experiment(conduction_variant=name))
        blocked, *_ = operator(mesh, Experiment(conduction_variant=name, pathway_scales={"bachmann": 0.}))
        assert (matrix != blocked).nnz > 0
        np.testing.assert_allclose(blocked@np.ones(len(mesh.xyz)), 0., atol=1e-12)
    with pytest.raises(ValueError, match="absent"):
        operator(base, Experiment(pathway_scales={"nonexistent": 0}))


def test_real_ring_circulation_stops_after_path_interruption():
    from vcg_biomarker.simulation.reentry import simulate_ring, count_ring_circulation
    _, _, _, control = simulate_ring()
    _, _, _, interrupted = simulate_ring(ablate_at_ms=900)
    assert control["complete_circuits"] >= 3
    assert interrupted["complete_circuits"] < control["complete_circuits"]
    assert max(interrupted["node0_activation_times_ms"]) < 900
    synchronous = [np.array([0., 400., 800.]) for _ in range(20)]
    assert count_ring_circulation(synchronous)["complete_circuits"] == 0


def synthetic_bank():
    rng = np.random.default_rng(9)
    rows = []
    for i in range(128):
        x = rng.uniform(.5, 1.5, len(PARAMETERS))
        y = [300-40*x[2]+20*x[1], 110+15*x[0], -85+2*x[4], 200*x[0]]
        rows.append({"id": i, "scales": dict(zip(PARAMETERS, x)), "biomarkers": dict(zip(OUTPUTS, y)),
                     "status": "computed", "partition": "fit" if i < 89 else "calibration" if i < 108 else "audit",
                     "periodicity_check_passed": False})
    return rows


def test_emulator_protocol_no_leakage_or_ecg_as_cell_truth():
    rows = synthetic_bank()
    result = train_emulator(rows)
    assert result["status"] == "research_surrogate"
    assert np.mean(result["audit_mae"]) < .1
    altered = json.loads(json.dumps(rows))
    for row in altered:
        if row["partition"] == "audit":
            row["biomarkers"]["apd90_ms"] += 10
    other = train_emulator(altered)
    assert other["coefficients"] == result["coefficients"]
    assert other["calibration_radius_90"] == result["calibration_radius_90"]
    estimate = predict_emulator(result, {key: 1. for key in PARAMETERS})
    assert estimate["clinical_use"] is False
    assert 200 < estimate["prediction"]["apd90_ms"] < 400
    with pytest.raises(ValueError, match="Outside"):
        predict_emulator(result, {key: 2. for key in PARAMETERS})
    protocol = {"model": "tentusscher-2006", "beats": 10, "cycle_ms": 800., "output_ms": .25}
    targets = {"apd90_ms": [300, 100]}
    with pytest.raises(ValueError, match="QT"):
        compatible_population(rows, targets, "ECG_QT", protocol, protocol)
    accepted = compatible_population(rows, targets, "synthetic_validation", protocol, protocol)
    assert len(accepted["candidates"]) > 1 and accepted["posterior_probabilities"] is None
    measured = {**protocol, "reference_provenance": "local_patch_clamp", "measurement_method": "ap_features_v1"}
    assert not compatible_population(rows, targets, "independent_cellular_action_potential", measured, protocol)["candidates"]


def test_resource_admission_and_ap_biomarker_definition():
    assert admission_target(16, 50, 4, 8, .5, 4) == 4
    assert admission_target(16, 50, 4, 2.1, .5, 4) == 0
    assert admission_target(16, 99, 0, 8, .5, 4) == 0
    t, v, _, _ = simulate_cell(CellExperiment(beats=3, output_ms=.25))
    features = ap_features(t, v, 800)
    assert 100 < features["apd90_ms"] < 600
    assert features["max_upstroke_mv_per_ms"] > 20
    with pytest.raises(ValueError):
        ap_features(t, np.zeros_like(t), 800)


def test_prepacing_checkpoint_resume_preserves_states_without_claiming_convergence():
    from vcg_biomarker.simulation.ionic import simulate_preconditioned
    checkpoints = []
    spec = CellExperiment(beats=1)
    t, v, _, first = simulate_preconditioned(spec, max_beats=10, checkpoint=checkpoints.append)
    assert checkpoints[-1]["beats"] == 10
    assert not first["preconditioning"]["converged"]
    t2, v2, _, resumed = simulate_preconditioned(spec, max_beats=10, resume=checkpoints[-1])
    np.testing.assert_array_equal(t, t2)
    np.testing.assert_array_equal(v, v2)
    assert not resumed["preconditioning"]["converged"]


def test_coronary_exposure_uses_chemical_not_electrical_diffusivity():
    from vcg_biomarker.simulation.substrate import perfusion_exposure
    mesh = block_mesh((2, 2, 2))
    substrate, trajectory, report = perfusion_exposure(
        mesh, np.ones(8, int), "LAD", "right", [0., 1000., 2000.],
        chemical_diffusivity_mm2_per_ms=2e-6, blood_potassium_mmol_l=5.4,
        baseline_washout_per_ms=1e-4, perfusion_fraction=.2, local_release_mmol_l_per_ms=1e-5)
    assert trajectory[-1].min() > 5.4
    assert report["electrical_diffusivity_used_as_chemical"] is False
    assert not report["clinical_use"]
    assert "potassium_mmol_l" in substrate.fields(8, len(mesh.tetra))
