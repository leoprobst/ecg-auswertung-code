from dataclasses import replace

import numpy as np
import pytest

from vcg_biomarker.simulation.monodomain import block_mesh
from vcg_biomarker.simulation.volume_conductor import ConductorMesh, VolumeConductor
from vcg_biomarker.simulation.ionic import CellExperiment, compile_model, simulate_cell, simulate_preconditioned
from vcg_biomarker.simulation.metabolism import synthetic_demo_config, validate_config
from vcg_biomarker.simulation.reference_validation import compare_waveforms, evaluate_regional_mapping


def conductor():
    mesh = block_mesh((3, 3, 3))
    n = len(mesh.tetra)
    return ConductorMesh(mesh.xyz_mm, mesh.tetra, np.tile(np.eye(3)*.3, (n, 1, 1)),
                         np.tile(np.eye(3)*.2, (n, 1, 1)), "manufactured uniform conductor")


def test_volume_conductor_exact_solution_gauge_and_constant_vm():
    mesh = conductor()
    vm = np.stack([mesh.xyz_mm[:, 0]*20-80, np.full(27, -80.)])
    electrodes = np.array([[.3, .5, 1.1], [1.4, .8, .1], [2., 2., 2.]])
    solver = VolumeConductor(mesh)
    phi, meta = solver.recover(vm, np.arange(27), electrodes)
    expected = -.4*(electrodes[:, 0]*20-80 - solver.mass@vm[0]/solver.mass.sum())
    np.testing.assert_allclose(phi[0], expected, atol=1e-10)
    np.testing.assert_allclose(phi[1], 0., atol=1e-10)
    other, _ = VolumeConductor(mesh, gauge_node=19).recover(vm, np.arange(27), electrodes)
    np.testing.assert_allclose(phi, other, atol=1e-10)
    assert meta["max_relative_residual"] < 1e-10


def test_torso_rigid_transform_and_conductivity_scaling():
    mesh = conductor()
    vm = np.array([mesh.xyz_mm[:, 0]*25-80])
    e = np.array([[.2, .3, .4], [1.2, .3, .4]])
    before, _ = VolumeConductor(mesh).recover(vm, np.arange(27), e)
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    displacement = np.array([100., 20., -50.])
    moved = replace(mesh, xyz_mm=mesh.xyz_mm@rotation.T+displacement,
                    sigma_e_s_per_m=rotation@mesh.sigma_e_s_per_m@rotation.T,
                    sigma_i_s_per_m=rotation@mesh.sigma_i_s_per_m@rotation.T)
    after, _ = VolumeConductor(moved).recover(vm, np.arange(27), e@rotation.T+displacement)
    np.testing.assert_allclose(before, after, atol=1e-10)
    scaled = replace(mesh, sigma_e_s_per_m=mesh.sigma_e_s_per_m*2, sigma_i_s_per_m=mesh.sigma_i_s_per_m*2)
    after, _ = VolumeConductor(scaled).recover(vm, np.arange(27), e)
    np.testing.assert_allclose(before, after, atol=1e-10)
    changed = replace(mesh, sigma_e_s_per_m=mesh.sigma_e_s_per_m*2)
    after, _ = VolumeConductor(changed).recover(vm, np.arange(27), e)
    np.testing.assert_allclose(after, before*.625, atol=1e-10)


def test_torso_rejects_missing_sources_bad_geometry_and_outside_electrodes():
    mesh = conductor()
    with pytest.raises(ValueError, match="outside"):
        VolumeConductor(mesh).electrode_weights(np.array([[4., 0., 0.]]))
    with pytest.raises(ValueError, match="myocardial"):
        VolumeConductor(mesh).recover(np.ones((1, 26)), np.arange(26), np.array([[1., 1., 1.]]))
    with pytest.raises(ValueError, match="positive definite"):
        VolumeConductor(replace(mesh, sigma_e_s_per_m=-mesh.sigma_e_s_per_m))
    with pytest.raises(ValueError, match="Degenerate"):
        VolumeConductor(replace(mesh, xyz_mm=np.zeros_like(mesh.xyz_mm)))


def test_ord_common_bath_mapping_and_baseline_metabolism_derivatives():
    spec = CellExperiment(model="ohara-2011", extracellular_mmol_l={"ionized_calcium_mmol_l": 2.})
    model, base, _, _ = compile_model(spec)
    assert model.get("extra.Ca_o").eval() == 2.
    modified, coupled, _, _ = compile_model(replace(spec, metabolism=synthetic_demo_config()))
    state = np.array(modified.initial_values(as_floats=True))
    rhs = coupled(0., state, 0.)
    np.testing.assert_allclose(rhs[:len(list(model.states()))], base(0., state[:len(list(model.states()))], 0.), atol=1e-10)
    by_name = dict(zip((s.qname() for s in modified.states()), rhs))
    assert by_name["research_energy.ATP"]+by_name["research_energy.ADP"] == pytest.approx(0.)
    assert by_name["research_energy.ATP"]+by_name["research_energy.Pi"] == pytest.approx(0.)
    assert (6*by_name["research_energy.glucose"]+3*by_name["research_energy.lactate"]+
            by_name["research_energy.CO2"]-by_name["research_energy.net_carbon_exchange"]) == pytest.approx(0.)


def test_metabolic_trajectory_conserves_pools_and_rejects_fake_steady_state():
    spec = CellExperiment(model="ohara-2011", beats=1, output_ms=2., metabolism=synthetic_demo_config())
    times, voltage, states, meta = simulate_cell(spec)
    assert np.isfinite(states).all() and len(times) == len(voltage)
    assert max(meta["metabolism"]["balance_errors_mmol_l"].values()) < 1e-7
    assert meta["metabolism"]["clinical_use"] is False
    with pytest.raises(ValueError, match="P1"):
        simulate_preconditioned(spec)
    with pytest.raises(ValueError, match="ORd"):
        replace(spec, model="tentusscher-2006").validate()
    config = synthetic_demo_config()
    config.pop("source")
    with pytest.raises(ValueError):
        validate_config(config)


def test_matched_model_design_preserves_pairing_and_bath(tmp_path):
    from vcg_biomarker.simulation.learning import prepare, atomic_json
    parent = prepare(tmp_path/"parent", samples=32, beats=3, cycles_ms=(500., 800., 1200.), precondition_max_beats=10)
    for case in parent["cases"]:
        atomic_json(tmp_path/"parent"/"cases"/f"{case['id']:05d}.json", case)
    atomic_json(tmp_path/"parent"/"status.json", {"phase": "complete"})
    child = prepare(tmp_path/"child", samples=32, beats=3, cycles_ms=(500., 800., 1200.), precondition_max_beats=10,
                    cell_model="ohara-2011", reference_campaign=tmp_path/"parent")
    assert child["cases"] == parent["cases"]
    assert child["extracellular_mmol_l"]["ionized_calcium_mmol_l"] == 2.
    assert child["comparison_reference"]["job_sha256"] == parent["job_sha256"]
    assert len(child["comparison_reference"]["case_sha256"]) == 96
    assert child["cell_environment"]["cell_type"] == "ORd_epi"
    assert (tmp_path/"child/runtime/vcg_biomarker/simulation/metabolism.py").exists()


def test_waveform_validation_does_not_fit_away_error_or_accept_unverified_clock():
    signal = np.arange(20.).reshape(10, 2)
    ref = dict.fromkeys(("cohort_id", "patient_id", "episode_id", "signal_id", "source", "model_version", "geometry_source"), "test")
    ref.update(data_role="validation", time_alignment_verified=True, species="synthetic", reference_type="synthetic_reference")
    result = compare_waveforms(signal+2, signal, np.arange(10.), ["I", "II"], ref)
    assert result["by_lead"][0]["rmse_mv"] == 2
    assert result["by_lead"][0]["correlation"] == pytest.approx(1.)
    ref["time_alignment_verified"] = False
    with pytest.raises(ValueError):
        compare_waveforms(signal, signal, np.arange(10.), ["I", "II"], ref)


def test_mapping_reference_reuses_existing_validation_and_patient_weights():
    import pandas as pd
    from vcg_biomarker.mimic.focus_localization import REQUIRED
    base = dict.fromkeys(REQUIRED, "test")
    base.update(cohort_id="synthetic", rhythm="pvc", reference_type="focal_origin", region="lv_other",
                reference_verified=True, reference_uncertainty="regional", signal_quality_pass=True,
                split="validation", transform="native_ecg", axis_convention="x_left_y_posterior_z_cranial")
    frame = pd.DataFrame([dict(base, patient_id="a", episode_id=str(i)) for i in range(3)]+[dict(base, patient_id="b")])
    result = evaluate_regional_mapping(frame, "ventricular_focus", ["lv_other", "lv_other", "lv_other", "rv_other"])
    assert result["patient_macro_accuracy_when_issued"] == .5  # not 3/4 per-record pooling
    frame.loc[0, "split"] = "locked_test"
    with pytest.raises(ValueError):
        evaluate_regional_mapping(frame, "ventricular_focus", [None]*4)


def test_conductor_worker_roundtrip_is_separate_and_hash_bound(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    mesh = conductor()
    tissue_path, conductor_path = tmp_path/"tissue.npz", tmp_path/"conductor_mesh.npz"
    np.savez_compressed(tissue_path, xyz_mm=mesh.xyz_mm, times_ms=np.arange(4.),
                        voltage_mv=np.stack([mesh.xyz_mm[:, 0]*k-80 for k in (1, 2, 3, 4)]))
    np.savez_compressed(conductor_path, xyz_mm=mesh.xyz_mm, tetra=mesh.tetra,
        sigma_e_s_per_m=mesh.sigma_e_s_per_m, sigma_i_s_per_m=mesh.sigma_i_s_per_m,
        units=np.array("mm"), coordinate_system=np.array(mesh.coordinate_system), source=np.array(mesh.source))
    job = tmp_path/"job"
    job.mkdir()
    request = {"kind": "volume_conductor", "tissue_path": str(tissue_path), "conductor_path": str(conductor_path),
        "tissue_path_sha256": hashlib.sha256(tissue_path.read_bytes()).hexdigest(),
        "conductor_path_sha256": hashlib.sha256(conductor_path.read_bytes()).hexdigest(),
        "electrode_names": ["RA", "LA", "LL"], "electrode_xyz_mm": [[0, 0, 0], [2, 0, 0], [1, 1, 1]]}
    (job/"request.json").write_text(json.dumps(request), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "vcg_biomarker.simulation.worker", str(job)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr+result.stdout
    with np.load(job/"conductor.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["signal"][:, 0], -.8*np.arange(1, 5), atol=1e-10)
        assert json.loads(str(data["metadata"]))["clinical_use"] is False


def test_lab_export_retains_physical_units_and_rejects_nonuniform_time(tmp_path):
    import json
    from vcg_biomarker.simulation.solver import LEADS_12
    from vcg_biomarker.simulation.volume_conductor import export_conductor_to_lab
    path = tmp_path/"forward.npz"
    metadata = {"schema": "finite-conductor-recovery/1", "clinical_use": False, "voltage_unit": "mV"}
    np.savez_compressed(path, times_ms=np.arange(5.), signal=np.ones((5, 12))*.2,
                        lead_names=np.array(LEADS_12), metadata=np.array(json.dumps(metadata)))
    export_conductor_to_lab(path, tmp_path/"lab.npz")
    with np.load(tmp_path/"lab.npz", allow_pickle=False) as data:
        assert bool(data["synthetic"])
        assert float(data["sampling_rate_hz"]) == 1000.
        np.testing.assert_allclose(data["ecg"], .2)
    np.savez_compressed(path, times_ms=np.array([0., 1., 2., 3., 3.5]), signal=np.ones((5, 12)),
                        lead_names=np.array(LEADS_12), metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="uniform"):
        export_conductor_to_lab(path, tmp_path/"bad.npz")
