"""Grouped extension acceptance: KATP, late sodium, uncertainty and GUI/worker."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vcg_biomarker.simulation.ionic import CellExperiment, compile_model, simulate_cell
from vcg_biomarker.simulation.ischemia import katp_current
from vcg_biomarker.simulation.learning import PARAMETERS, OUTPUTS, train_emulator, predict_emulator, ap_features
from vcg_biomarker.simulation.identifiability import emulator_jacobian, local_identifiability
from vcg_biomarker.simulation.monodomain import block_mesh, simulate_monodomain
from vcg_biomarker.simulation.substrate import RegionalSubstrate


def test_katp_equation_units_current_and_potassium_balance():
    base = CellExperiment(beats=1)
    model, reference, vi, _ = compile_model(base)
    modified, rhs, _, _ = compile_model(replace(base, katp_open_fraction=.0055))
    state = np.asarray(model.initial_values(as_floats=True))
    ki = [v.qname() for v in model.states()].index("potassium.Ki")
    # Independent reference expression at multiple potentials, not self-comparison.
    ek = float(model.get("nernst.EK").eval())
    potassium_factor = float(model.get("cell.Cm").eval())
    potassium_factor /= float(model.get("cell.Vc").eval())*float(model.get("phys.F").eval())
    for v in (-100., -80., 0., 40.):
        state[vi] = v
        current = katp_current(v, ek, 5.4, .0055)
        delta = np.asarray(rhs(0., state, 0))-reference(0., state, 0)
        assert delta[vi] == pytest.approx(-current, abs=1e-10)
        assert delta[ki] == pytest.approx(-current*potassium_factor, abs=1e-12)
        other = np.delete(delta, [vi, ki])
        np.testing.assert_allclose(other, 0, atol=1e-12)
    modified.check_units()
    assert katp_current(ek, ek, 5.4, .0055) == 0
    assert katp_current(20., ek, 5.4, 0.) == 0
    with pytest.raises(ValueError):
        replace(base, katp_open_fraction=.55).validate()  # percent/fraction confusion
    with pytest.raises(ValueError):
        replace(base, model="ohara-2011", katp_open_fraction=.0055).validate()


def test_regional_katp_matches_scalar_and_zero_preserves_reference():
    base = CellExperiment(beats=1, output_ms=.5)
    model, baseline, _, _ = compile_model(base)
    _, scalar, _, _ = compile_model(replace(base, katp_open_fraction=.0055))
    _, vector, _, _ = compile_model(base, True, ("katp_open_fraction",))
    state = np.array(model.initial_values(as_floats=True))
    state[0] = 10.
    value = np.array(vector(0., np.repeat(state[:, None], 2, axis=1), np.zeros(2),
                            {"katp_open_fraction": np.array([0., .0055])}))
    np.testing.assert_allclose(value[:, 0], baseline(0., state, 0), atol=1e-9)
    np.testing.assert_allclose(value[:, 1], scalar(0., state, 0), atol=1e-9)
    mesh = block_mesh((2, 2, 2))
    substrate = RegionalSubstrate(katp_open_fraction=np.full(8, .0055))
    t, tissue, metadata = simulate_monodomain(mesh, base, duration_ms=10., stimulus_nodes=np.arange(8), substrate=substrate)
    st, sv, _, _ = simulate_cell(replace(base, katp_open_fraction=.0055))
    np.testing.assert_allclose(tissue[:, 0], np.interp(t, st, sv), atol=.05)
    assert metadata["katp"]["potassium_balance_included"]
    assert metadata["katp"]["regional_override"]


def test_late_sodium_and_katp_change_simulated_action_potential():
    tp = CellExperiment(beats=3, output_ms=.25)
    t, v, _, _ = simulate_cell(tp)
    a = ap_features(t, v, tp.cycle_ms)
    t, v, _, meta = simulate_cell(replace(tp, katp_open_fraction=.0055))
    b = ap_features(t, v, tp.cycle_ms)
    assert b["apd90_ms"] < a["apd90_ms"]
    assert meta["katp"]["source"].endswith("1003891")
    ord_spec = replace(tp, model="ohara-2011")
    t, v, _, _ = simulate_cell(ord_spec)
    a = ap_features(t, v, ord_spec.cycle_ms)
    t, v, _, _ = simulate_cell(replace(ord_spec, channel_scales={"INaL": 3.}))
    b = ap_features(t, v, ord_spec.cycle_ms)
    assert b["apd90_ms"] > a["apd90_ms"]
    # Original sodium formulation is intentionally NOT released for tissue use.
    with pytest.raises(ValueError, match="paced model"):
        simulate_monodomain(block_mesh((2, 2, 2)), ord_spec)


def emulator():
    rng = np.random.default_rng(23)
    rows = []
    for i, x in enumerate(rng.uniform(.5, 1.5, (100, 6))):
        rows.append({"status": "computed", "scales": dict(zip(PARAMETERS, x)),
                     "partition": "fit" if i < 70 else "calibration" if i < 85 else "audit",
                     "biomarkers": dict(zip(OUTPUTS, [x[0]+x[1]*x[2], x[3]**2, x[4]*x[0], x[5]*x[2]]))})
    result = train_emulator(rows)
    result["protocol"] = {"cycle_ms": 800., "model": "synthetic_test_only"}
    return result


def test_identifiability_jacobian_nullspace_and_duplicate_protocol_guard():
    model = emulator()
    scales = dict.fromkeys(PARAMETERS, 1.1)
    jac = emulator_jacobian(model, scales)
    for i, key in enumerate(PARAMETERS):
        # Jacobian is per normalized parameter, hence factors +/-0.5*epsilon.
        lower, upper = dict(scales), dict(scales)
        lower[key] -= .5e-5
        upper[key] += .5e-5
        a, b = predict_emulator(model, lower), predict_emulator(model, upper)
        numerical = [(b["prediction"][k]-a["prediction"][k])/2e-5 for k in OUTPUTS]
        np.testing.assert_allclose(jac[:, i], numerical, atol=1e-8)
    errors = dict.fromkeys(OUTPUTS, .1)
    report = local_identifiability([model], scales, [errors])
    assert report["nullity"] >= 2
    assert report["condition_number"] is None
    assert not report["unique_patient_parameters_identified"]
    matrix = np.array(report["normalized_sensitivity"])
    np.testing.assert_allclose(matrix@np.array(report["null_directions"]).T, 0, atol=1e-10)
    with pytest.raises(ValueError, match="Duplicate"):
        local_identifiability([model, model], scales, [errors, errors])


def test_gui_new_model_reset_and_worker_preconditioning(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.patient_twin_panel import IonicCellPanel
    app = QApplication.instance() or QApplication([])
    panel = IonicCellPanel(lambda request: None, None)
    panel.channel_scales = {"INaL": 2.}
    panel.katp_open_fraction = .005
    panel.reset_model_mechanisms()  # Also exercise directly: Qt slot exceptions must not be swallowed.
    panel.model.setCurrentIndex(panel.model.findData("ohara-2011"))
    assert panel.channel_scales == {} and panel.katp_open_fraction == 0
    panel.close()
    app.processEvents()
    from dataclasses import asdict
    root = tmp_path/"prepace_worker"
    root.mkdir()
    (root/"request.json").write_text(json.dumps({"kind": "ionic_cell", "cell_experiment": asdict(CellExperiment(beats=3)),
                                               "precondition_max_beats": 10}), encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "vcg_biomarker.simulation.worker", str(root)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert '"phase": "preconditioning"' in result.stdout
    with np.load(root/"ionic.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        assert meta["preconditioning"]["beats"] == 10
        assert not meta["preconditioning"]["converged"]


def test_multicycle_state_candidates_are_not_silently_called_solver_failures():
    from vcg_biomarker.simulation.ionic import cycle_state_pattern
    assert cycle_state_pattern(np.ones((5, 3)))["period_beats"] == 1
    alternating = np.tile([[1., 2.], [2., 1.]], (4, 1))
    report = cycle_state_pattern(alternating)
    assert report["period_beats"] == 2 and not report["clinical_alternans"]
    assert cycle_state_pattern(np.arange(24).reshape(12, 2))["status"] == "unresolved_or_drifting"
    assert cycle_state_pattern(alternating[:6])["period_beats"] is None
