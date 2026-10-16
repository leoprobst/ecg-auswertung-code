"""Grouped acceptance of the reduced simulation; no production or patient runs."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from vcg_biomarker.simulation.model import (
    Electrode, Experiment, Tissue, anatomical_tissue, coarsen_short_edges, electrode_preset, operator,
)
from vcg_biomarker.simulation.solver import (
    derive_leads, export_lab_signal, fit_candidates, load_result, reaction, save_result, simulate,
)


def cable(n=31):
    xyz = np.column_stack([np.arange(n) * .5, np.zeros(n), np.zeros(n)])
    return Tissue(xyz, np.column_stack([np.arange(n-1), np.arange(1, n)]), np.ones(n, int),
                  np.where(np.arange(n) == 15, 7, 0), np.tile([1., 0., 0.], (n, 1)),
                  {"his": np.array([0, 1])}, {}, "synthetic_cable")


@pytest.mark.parametrize("update", [{"duration_ms": float("nan")}, {"max_dt_ms": 0},
                                   {"diffusion_scale": -1}, {"preset": "ischemia"},
                                   {"s2_ms": 700}, {"region": 0},
                                   {"electrodes": (Electrode("RA", (0, 0, 0)),) * 3}])
def test_invalid_experiments_abstain(update):
    with pytest.raises(ValueError):
        replace(Experiment(), **update).validate()


def test_contract_roundtrip():
    spec = Experiment(electrodes=electrode_preset("intracardiac"), s2_ms=400)
    assert Experiment.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_process_lease_is_exclusive_and_released(tmp_path):
    from vcg_biomarker.simulation.worker import acquire_worker_lock
    first = acquire_worker_lock(tmp_path / "worker.lock")
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            acquire_worker_lock(tmp_path / "worker.lock")
    finally:
        first.close()
    acquire_worker_lock(tmp_path / "worker.lock").close()


def test_ms_reference_equation_and_graph_conservation():
    v, h = np.array([0., .2]), np.array([1., .7])
    dv, dh = reaction(v, h, np.array([150., 150.]))
    np.testing.assert_allclose(dv, [0., .7 * .2**2 * .8 / .3 - .2 / 6])
    np.testing.assert_allclose(dh, [0., -.7 / 150])
    lap, _, _, _ = operator(cable(), Experiment())
    np.testing.assert_allclose(lap.toarray(), lap.T.toarray())
    np.testing.assert_allclose(lap @ np.ones(31), 0., atol=1e-15)
    assert np.dot(np.arange(31), lap @ np.arange(31)) <= 0


def test_wave_propagation_scar_cancellation_and_time_refinement():
    tissue = cable()
    spec = Experiment(duration_ms=100, output_step_ms=1)
    base = simulate(tissue, spec)
    fine = simulate(tissue, replace(spec, max_dt_ms=.015))
    assert np.isfinite(base.activation_ms).all()
    assert np.all(np.diff(base.activation_ms[2:]) > 0)
    np.testing.assert_allclose(base.activation_ms, fine.activation_ms, atol=1.)
    scar = simulate(tissue, replace(spec, preset="scar"))
    assert np.isnan(scar.activation_ms[15:]).all()
    assert np.all(scar.voltage[:, 15] == 0)
    with pytest.raises(InterruptedError):
        simulate(tissue, spec, cancelled=lambda: True)
    # Check deterministic replay, not just similar plotted curves.
    np.testing.assert_array_equal(base.ecg, simulate(tissue, spec).ecg)


def test_anisotropy_and_disease_mechanisms():
    tissue = cable()
    parallel, _, _, _ = operator(tissue, Experiment())
    tissue.fibers[:] = [0, 1, 0]
    transverse, _, _, _ = operator(tissue, Experiment())
    np.testing.assert_allclose(parallel.toarray(), transverse.toarray() * 3)
    _, long_close, _, _ = operator(tissue, Experiment(preset="long_ap"))
    _, short_close, _, _ = operator(tissue, Experiment(preset="short_ap"))
    assert long_close[15] > short_close[15]
    original = cable()
    compact = coarsen_short_edges(original, .1)
    np.testing.assert_array_equal(compact.xyz, original.xyz)


def test_forward_electrodes_wilson_einthoven_and_bipolar():
    names = [e.name for e in electrode_preset("intracardiac")]
    signal, leads = derive_leads(np.random.default_rng(1).normal(size=(30, len(names))), names)
    d = {k: signal[:, i] for i, k in enumerate(leads)}
    np.testing.assert_allclose(d["II"], d["I"] + d["III"], atol=1e-14)
    np.testing.assert_allclose(d["aVR"] + d["aVL"] + d["aVF"], 0, atol=1e-14)
    np.testing.assert_allclose(d["CathA-CathB"], d["CathA"] - d["CathB"], atol=1e-14)
    tissue = cable()
    electrodes = list(electrode_preset())
    electrodes[0] = replace(electrodes[0], xyz_mm=tuple(tissue.xyz[0]))
    with pytest.raises(ValueError, match="0.5 mm"):
        simulate(tissue, Experiment(electrodes=tuple(electrodes)))


def test_result_roundtrip_and_explicit_synthetic_lab_import(tmp_path):
    from vcg_biomarker.core.external_loader import load_external_ecg_result, ExternalECGMeta
    tissue = cable()
    result = simulate(tissue, Experiment(duration_ms=50))
    path = tmp_path / "result.npz"
    save_result(path, result, tissue)
    data = load_result(path)
    assert data["metadata"]["synthetic"] and not data["metadata"]["clinical_use"]
    assert len(data["metadata"]["source_hashes"]["solver.py"]) == 64
    np.testing.assert_array_equal(data["voltage"], result.voltage)
    with pytest.raises(ValueError):
        export_lab_signal(path, tmp_path / "bad.npz", 0)
    exported = export_lab_signal(path, tmp_path / "signal.npz", 1000.)
    imported = load_external_ecg_result(exported, ExternalECGMeta())
    np.testing.assert_allclose(imported.signal_millivolts, result.ecg * 1000)
    assert imported.source_metadata["synthetic"] is True
    assert imported.source_metadata["simulation_provenance"]["amplitude_calibrated"] is False
    with pytest.raises(FileExistsError):
        export_lab_signal(path, exported, 1000.)


def test_inverse_budget_same_segment_and_no_clinical_probability():
    tissue = cable()
    spec = Experiment(duration_ms=50)
    base = simulate(tissue, spec)
    rows = fit_candidates(tissue, [replace(spec, diffusion_scale=.5), spec], base.ecg, 500, base.lead_names)
    assert rows[0]["relative_squared_error"] < 1e-20
    assert rows[0]["clinical_validation"] is False
    assert "probability" not in rows[0]
    with pytest.raises(ValueError, match="budget"):
        fit_candidates(tissue, [spec] * 9, base.ecg, 500, base.lead_names)
    with pytest.raises(ValueError, match="cover"):
        fit_candidates(tissue, [spec], base.ecg[:10], 500, base.lead_names)


def test_real_anatomy_separation_and_branch_block():
    tissue = anatomical_tissue()
    assert set(tissue.cells) == set(range(6))
    assert len(tissue.xyz) > 1000
    lap, _, live, _ = operator(tissue, Experiment(preset="lbbb"))
    left = np.concatenate([v for k, v in tissue.branches.items() if k.startswith("lbb")])
    assert not live[left].any()
    assert live[tissue.branches["rbb"]].any()
    np.testing.assert_allclose(np.asarray(lap.sum(axis=1)), 0, atol=1e-12)


def test_low_priority_worker_full_roundtrip(tmp_path):
    request = {"kind": "forward", "experiment": Experiment(duration_ms=10).to_dict()}
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    done = subprocess.run([sys.executable, "-m", "vcg_biomarker.simulation.worker", str(tmp_path)],
                          cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=90)
    assert done.returncode == 0, done.stderr + done.stdout
    assert json.loads(done.stdout.splitlines()[-1])["phase"] == "complete"
    assert load_result(tmp_path / "result.npz")["metadata"]["synthetic"]


def test_workspace_controls_result_and_electrode_roundtrip(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.simulation_workspace import SimulationWorkspace, tube_mesh
    app = QApplication.instance() or QApplication([])
    widget = SimulationWorkspace(workspace_root=tmp_path)
    assert widget.specification().preset == "reference"
    assert widget.views.count() == 8
    widget.electrode_preset.setCurrentIndex(2)
    assert len(widget.specification().electrodes) == 11
    tissue = cable()
    path = tmp_path / "result.npz"
    save_result(path, simulate(tissue, Experiment(duration_ms=50)), tissue)
    widget.display_result(path)
    widget.slider.setValue(5)
    assert widget.time_label.text() == "10 ms"
    assert widget.export_button.isEnabled()
    assert "normiert" in widget.legend.text()
    assert len(widget.cell_trace.listDataItems()) == 1
    assert widget.minimumSizeHint().width() <= 1366
    widget.start_fit()
    assert "ausrichten" in widget.status.text()
    v, f = tube_mesh(np.zeros(3), np.array([0., 0., 10.]))
    assert np.isfinite(v).all() and f.max() < len(v)
    widget.shutdown()
    widget.close()
    app.processEvents()


def test_animated_mesh_colors_preserve_geometry():
    from types import SimpleNamespace
    from pyqtgraph.opengl import MeshData
    from vcg_biomarker.gui.simulation_workspace import update_mesh_colors
    vertices = np.eye(3, dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = MeshData(vertexes=vertices, faces=faces)
    calls = []
    item = SimpleNamespace(opts={"meshdata": mesh}, meshDataChanged=lambda: calls.append(True))
    update_mesh_colors(item, np.ones((3, 4)))
    np.testing.assert_array_equal(mesh.vertexes(), vertices)
    np.testing.assert_array_equal(mesh.faces(), faces)
    assert calls == [True]
