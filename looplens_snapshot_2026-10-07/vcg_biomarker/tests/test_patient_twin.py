"""Grouped patient-twin acceptance: real equations, data isolation, UI and workers."""
from dataclasses import asdict, replace
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.simulation.patient import Observation, PatientProfile, twin_readiness
from vcg_biomarker.simulation.calibration import build_mimic_training_frame, partition, predict_priors, train_priors
from vcg_biomarker.simulation.ionic import CellExperiment, compile_model, simulate_cell
from vcg_biomarker.simulation.model_assets import MODELS
from vcg_biomarker.simulation.monodomain import assemble, block_mesh, simulate_monodomain


def profile():
    return PatientProfile({"age": Observation(55, "years", "history"),
                           "height_cm": Observation(175, "cm", "measured"),
                           "weight_kg": Observation(75, "kg", "measured")}, "male")


def test_profile_time_censoring_units_and_no_physics_shortcuts():
    p = profile()
    assert PatientProfile.from_dict(p.to_dict()) == p
    assert not twin_readiness(p)["gates"]["clinical_validation"]
    with pytest.raises(ValueError, match="Invalid"):
        Observation(1.75, "m", "measured").validate("height_cm")
    with pytest.raises(ValueError, match="uncensored"):
        Observation(6, "mmol/l", "lab", censoring=">").validate("potassium_mmol_l")
    obs = {"potassium_mmol_l": Observation(4.5, "mmol/l", "lab", "2026-09-23T09:00:00+02:00")}
    p = replace(p, observations=obs, index_time="2026-09-23T10:00:00+02:00")
    assert not p.ionic_constraints()["constraints"]
    p = replace(p, clock_alignment_verified=True)
    assert p.ionic_constraints()["constraints"]["potassium_mmol_l"]["value"] == 4.5
    assert not replace(p, index_time="2026-09-22T10:00:00+02:00").ionic_constraints()["constraints"]
    assert not replace(p, index_time="2026-09-23T10:00:00").ionic_constraints()["constraints"]


def training_data(n=1200):
    rng = np.random.default_rng(3)
    height = rng.normal(172, 8, n)
    return pd.DataFrame({"patient_id": np.arange(n).astype(str), "cohort": "synthetic",
                         "source_split": "train", "age": rng.integers(20, 85, n),
                         "female": rng.integers(0, 2, n), "height_cm": height,
                         "weight_kg": rng.normal(80, 10, n),
                         "lv_diameter_mm": 45 + .4*(height-172)+rng.normal(0, .5, n)})


def test_calibration_is_disjoint_reproducible_and_abstains():
    data = training_data()
    first = train_priors(data)
    assert first == train_priors(data.sample(frac=1, random_state=4))
    model = first["models"]["lv_diameter_mm"]
    assert model["audit_mae"] < model["audit_baseline_mae"] / 2
    assert model["audit_paired_mae_difference_ci95"][1] < 0
    assert model["counts"]["fit"] + model["counts"]["calibration"] + model["counts"]["internal_audit"] == len(data)
    assert predict_priors(first, profile())["lv_diameter_mm"]["status"] == "population_prior"
    assert predict_priors(first, PatientProfile())["lv_diameter_mm"]["reason"] == "missing_covariates"
    out = replace(profile(), observations={**profile().observations, "height_cm": Observation(230, "cm", "measured")})
    assert predict_priors(first, out)["lv_diameter_mm"]["reason"] == "outside_training_support"
    changed = data.copy()
    changed.loc[0, "source_split"] = "locked_test"
    with pytest.raises(ValueError, match="train"):
        train_priors(changed)
    with pytest.raises(ValueError, match="one reference"):
        train_priors(pd.concat([data, data.iloc[:1]]))
    # A held-out audit value cannot alter the fitted coefficients or conformal width.
    changed = data.copy()
    audit = [partition(c, p) == "internal_audit" for c, p in zip(data.cohort, data.patient_id)]
    changed.loc[audit, "lv_diameter_mm"] += 3
    altered = train_priors(changed)["models"]["lv_diameter_mm"]
    for key in ("coefficients", "intercept", "alpha", "conformal_half_width_90"):
        assert altered[key] == model[key]


def test_mimic_join_train_only_reference_blind_and_preceding_tte(tmp_path):
    (tmp_path / "master").mkdir()
    (tmp_path / "evidence").mkdir()
    master = pd.DataFrame({"subject_id": [1, 1, 2, 3], "study_id": [10, 11, 20, 30],
                           "ecg_time": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-03", "2020-01-03"]),
                           "age": [40]*4, "sex": ["M"]*4, "machine_qrs_duration_ms": [90]*4,
                           "machine_metadata_usable_for_primary": [True]*4,
                           "split": ["train", "train", "train", "locked_test"], "age_deidentified_over_89": [False]*4})
    master.to_parquet(tmp_path / "master/ecg_master_index.parquet")
    pd.DataFrame({"subject_id": [1, 2, 3], "split": ["train", "train", "locked_test"]}).to_parquet(tmp_path / "master/subject_splits.parquet")
    echo = pd.DataFrame({"subject_id": [1, 2, 2, 3], "measurement_id": [1, 2, 3, 4],
                         "echo_time": pd.to_datetime(["2020-01-02"]*2+["2020-01-04", "2020-01-02"]),
                         "test_type": ["tte", "tte", "TTE", "tte"], "echo_height_cm": [170]*4,
                         "echo_weight_kg": [70]*4, "lv_internal_diameter_diastole_cm": [5, 5, 8, 5],
                         "septal_thickness_cm": [1]*4, "posterior_wall_thickness_cm": [1]*4,
                         "lv_end_diastolic_volume_ml": [100]*4})
    echo.to_parquet(tmp_path / "evidence/echo_measurements_wide.parquet")
    data, manifest = build_mimic_training_frame(tmp_path)
    assert set(data.patient_id) == {"1", "2"}
    assert data.set_index("patient_id").loc["1", "study_id"] == 10
    assert pd.isna(data.set_index("patient_id").loc["1", "lv_diameter_mm"])
    assert data.set_index("patient_id").loc["2", "lv_diameter_mm"] == 50
    assert manifest["locked_test_accessed"] is False


@pytest.mark.parametrize("name", list(MODELS))
def test_pinned_ionic_equations_match_original_scalar_and_vector(name):
    model, scalar, _, factor = compile_model(CellExperiment(model=name))
    _, vector, _, _ = compile_model(CellExperiment(model=name), vectorized=True)
    state = np.array(model.initial_values(as_floats=True))
    for pace in (0., 1.):
        reference = model.evaluate_derivatives(state, {"time": 0., "pace": pace, "diffusion_current": 0.})
        np.testing.assert_allclose(scalar(0, state, pace), reference, atol=1e-9, rtol=1e-11)
        values = vector(0., np.repeat(state[:, None], 3, axis=1), np.full(3, pace))
        for expected, observed in zip(reference, values):
            np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-9)
    assert factor == (.001 if name == "fabbri-2017" else 1.)
    _, voltage, _, metadata = simulate_cell(CellExperiment(model=name, beats=1))
    assert voltage.min() < -40 and voltage.max() > 0
    assert metadata["periodicity_check_passed"] is False


def test_ionic_tolerance_electrolytes_cancellation_and_bounds():
    spec = CellExperiment(beats=1)
    t, base, _, _ = simulate_cell(spec)
    _, refined, _, _ = simulate_cell(replace(spec, rtol=1e-7, atol=1e-10))
    np.testing.assert_allclose(base, refined, atol=.03)
    _, potassium, _, _ = simulate_cell(replace(spec, extracellular_mmol_l={"potassium_mmol_l": 6.}))
    assert np.max(np.abs(base-potassium)) > 1
    with pytest.raises(InterruptedError):
        simulate_cell(spec, cancelled=lambda: True)
    with pytest.raises(ValueError):
        simulate_cell(replace(spec, extracellular_mmol_l={"magnesium_mmol_l": 2}))
    assert t[-1] == spec.cycle_ms


def test_fem_conservation_anisotropy_degeneracy_and_coupled_wave():
    mesh = block_mesh((4, 2, 2))
    lap, mass, stiff = assemble(mesh)
    np.testing.assert_allclose(stiff.toarray(), stiff.T.toarray(), atol=1e-12)
    np.testing.assert_allclose(lap @ np.ones(len(mass)), 0, atol=1e-12)
    np.testing.assert_allclose(mass @ lap.toarray(), 0, atol=1e-12)
    assert mass.sum() == pytest.approx(3.)
    trans, _, _ = assemble(mesh, d_long=.025, d_trans=.025)
    x = mesh.xyz_mm[:, 0]
    assert -(x*mass) @ (lap@x) == pytest.approx(4*(-(x*mass) @ (trans@x)))
    broken = block_mesh((2, 2, 2))
    broken.xyz_mm[:, 2] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        assemble(broken)
    times, voltage, meta = simulate_monodomain(mesh, duration_ms=30)
    assert (voltage.max(axis=0) > 0).all()
    first = np.argmax(voltage > -20, axis=0)
    assert np.median(first[mesh.xyz_mm[:, 0] == 3]) > np.median(first[mesh.xyz_mm[:, 0] == 0])
    assert not meta["clinical_use"] and not meta["mesh_convergence_proven"]
    assert times[-1] == 30
    uniform = block_mesh((2, 2, 2))
    times, voltage, _ = simulate_monodomain(uniform, duration_ms=10, stimulus_nodes=np.arange(8))
    single_time, single, _, _ = simulate_cell(CellExperiment(beats=1))
    np.testing.assert_allclose(voltage[:, 0], np.interp(times, single_time, single), atol=.05)


def test_ionic_worker_and_gui_contract(tmp_path):
    root = tmp_path / "ionic"
    root.mkdir()
    request = {"kind": "ionic_cell", "cell_experiment": asdict(CellExperiment(beats=1)), "patient_profile": profile().to_dict()}
    (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.run([sys.executable, "-m", "vcg_biomarker.simulation.worker", str(root)],
                             capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    with np.load(root / "ionic.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["patient_profile"]["sex_recorded"] == "male"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.simulation_workspace import SimulationWorkspace
    app = QApplication.instance() or QApplication([])
    widget = SimulationWorkspace(workspace_root=tmp_path)
    assert any(widget.views.tabText(i) == "Patientenzwilling" for i in range(widget.views.count()))
    widget.ionic_panel.show_result(root / "ionic.npz")
    assert len(widget.ionic_panel.plot.listDataItems()) == 1
    widget.patient_panel.evaluate()
    assert "POPULATION" in widget.patient_panel.report.toPlainText()
    widget.shutdown()
    widget.close()
    app.processEvents()
