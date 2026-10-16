"""Cohort adapters, geometry/units safeguards and published metabolic reference."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.io import savemat

from vcg_biomarker.simulation.edgar import Acquisition, freeze_subject_roles, identity, inventory, physical_record, surface_geometry
from vcg_biomarker.simulation.edgar_inverse import fit_dipole, reference_distance
from vcg_biomarker.simulation.ecme import ECMEExperiment, compile_reference, load_reference, simulate_ecme


def test_subject_grouping():
    a = identity('Utah/Signals1/Run0035-b1-ts.mat')
    b = identity('Utah/Signals1/Run0035-b2-cs.mat')
    assert a['subject_id'] == b['subject_id']
    assert a['parent_record'] == b['parent_record']
    assert identity('Utah/Signals2/Run0035-b1-ts.mat')['subject_id'] != a['subject_id']
    assert identity('Bratislava/Signals/P001-ts.mat')['reference_type'] == 'ablation_site'
    assert identity('Auckland/Signals/Pig3-Run0001-cs.mat')['species'] == 'porcine'


def fixture(root):
    folder = root/'Bratislava/Signals'
    folder.mkdir(parents=True)
    path = folder/'P001-ts.mat'
    savemat(path, {'ts': {'potvals': np.arange(48).reshape(6, 8), 'pacingLoc': [1., 2., 3.],
                         'leadlabels': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']}})
    return path.relative_to(root)


def test_units_explicit_and_raw_immutable(tmp_path):
    root = tmp_path/'raw'
    relative = fixture(root)
    original = (root/relative).read_bytes()
    acquisition = Acquisition(1000., 'uV', 'cm', 'registered-source', 'README', 'source', True)
    record = physical_record(root, relative, acquisition)
    assert record['signal_mv'].shape == (8, 6)
    assert record['signal_mv'][1, 0] == .001
    np.testing.assert_equal(record['reference_xyz_mm'], [10, 20, 30])
    assert (root/relative).read_bytes() == original
    with pytest.raises(ValueError):
        physical_record(root, relative, replace(acquisition, evidence=''))
    with pytest.raises(ValueError):
        physical_record(root, relative, replace(acquisition, registration_verified=False))


@pytest.mark.parametrize('base', [0, 1])
def test_geometry_index_base(tmp_path, base):
    folder = tmp_path/'Halifax/Geometries'
    folder.mkdir(parents=True)
    nodes = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    savemat(folder/'Heart7620.mat', {'Heart7620': {'node': nodes, 'face': faces+base}})
    xyz, got = surface_geometry(tmp_path, 'Halifax/Geometries/Heart7620.mat')
    np.testing.assert_equal(got, faces)
    assert xyz.shape == (4, 3)


def test_nested_geometry_preserves_valid_parts_and_rejects_bad(tmp_path):
    from vcg_biomarker.simulation.edgar import geometry_qc
    folder = tmp_path/'Bratislava/Geometries'
    folder.mkdir(parents=True)
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    valid = {'pts': nodes.T, 'fac': np.array([[0, 1, 2], [1, 2, 3]]).T}
    invalid = {'pts': nodes.T, 'fac': np.array([[0, 0, 2], [1, 2, 3]]).T}
    path = 'Bratislava/Geometries/P001_otherStructures.mat'
    savemat(tmp_path/path, {'otherStructures': {'Llung': valid, 'LV_AO': invalid}})
    qc = geometry_qc(tmp_path, path)
    assert qc['otherStructures.Llung']['status'] == 'validated_native_surface'
    assert qc['otherStructures.LV_AO']['status'] == 'not_usable_for_solver'
    xyz, _ = surface_geometry(tmp_path, path, 'otherStructures.Llung')
    np.testing.assert_equal(xyz, nodes)
    with pytest.raises(ValueError):
        surface_geometry(tmp_path, path, 'otherStructures.LV_AO')


def test_inventory_and_role_freeze(tmp_path):
    root = tmp_path/'raw'
    fixture(root)
    result = inventory(root, tmp_path/'derived')
    assert result['files'] == 1
    assert result['statuses'] == {'needs_acquisition_metadata': 1}
    assert not result['clinical_validation_complete']
    with pytest.raises(ValueError):
        inventory(root, root/'derived')
    rows = [{'kind': 'signal', 'subject_id': s} for s in ('A', 'A', 'B')]
    assert freeze_subject_roles(rows, ['B']) == {'A': 'development', 'B': 'validation'}
    with pytest.raises(ValueError):
        freeze_subject_roles(rows, ['A', 'B'])


def dipole_fixture():
    rng = np.random.default_rng(21)
    electrodes = rng.normal(size=(32, 3))
    electrodes = electrodes/np.linalg.norm(electrodes, axis=1)[:, None]*10
    sites = np.array([[0, 0, 0], [.5, 1, 0], [3, 1, 0], [-1, 2, 2.]])
    delta = electrodes-sites[1]
    matrix = delta/np.linalg.norm(delta, axis=1)[:, None]**3
    potentials = matrix@rng.normal(size=(3, 40)) + rng.normal(size=(1, 40))
    return potentials, electrodes, sites


def test_dipole_exact_and_invariance():
    signals, electrodes, sites = dipole_fixture()
    result = fit_dipole(signals, electrodes, sites)
    assert result['candidate_index'] == 1
    assert result['heldout_electrode_normalized_sse'] < 1e-20
    assert not result['reference_used_in_search']
    assert not result['anatomical_focal_origin_claim']
    transformed = fit_dipole(signals*1000+7, electrodes*3+999, sites*3+999)
    assert transformed['candidate_index'] == 1
    assert reference_distance(result, sites[1], 'ablation_site')['localization_error_mm'] is None


def test_zero_channels_not_treated_as_normal():
    signals, electrodes, sites = dipole_fixture()
    signals[3] = 0
    result = fit_dipole(signals, electrodes, sites)
    assert result['candidate_index'] == 1
    assert result['excluded_constant_channel_indices'] == [3]


def test_no_search_for_constant_signal_or_cancel():
    signals, electrodes, sites = dipole_fixture()
    with pytest.raises(ValueError):
        fit_dipole(np.ones_like(signals), electrodes, sites)
    with pytest.raises(InterruptedError):
        fit_dipole(signals, electrodes, sites, cancelled=lambda: True)


def test_multiple_ablation_sites_are_not_best_reference():
    result = reference_distance({'xyz_native': [0, 0, 0]}, [[1, 0, 0], [1, 0, 0], [5, 0, 0]], 'ablation_site')
    assert result['reference_points'] == 3 and result['unique_reference_points'] == 2
    assert result['distance_native'] is None and not result['reference_point_selected']
    assert result['distances_to_reference_points_native'] == [1., 5.]


def test_mitochondria_units_rhs_and_actual_ph_response():
    from vcg_biomarker.simulation.mitochondria import MitochondrialExperiment, compiled, simulate_mitochondria
    spec = MitochondrialExperiment(duration_s=.2)
    model, rhs = compiled(spec)
    y = model.initial_values(as_floats=True)
    np.testing.assert_allclose(rhs(0, y, 0), model.evaluate_derivatives(y, inputs={'time': 0}), rtol=1e-10, atol=1e-12)
    t, state, metadata = simulate_mitochondria(spec)
    _, acid, acid_metadata = simulate_mitochondria(replace(spec, bath_pH=6.8))
    assert len(y) == 19 and np.isfinite(state).all() and t[-1] == .2
    assert not np.allclose(state, acid)
    assert metadata['unit_check'].startswith('pass')
    assert 6 < metadata['matrix_pH_final'] < 9
    assert acid_metadata['oxygen_protocol'] == 'clamped bath concentration'
    assert not metadata['clinical_use']
    corrected, _ = compiled(replace(spec, parameter_set='paper_2006_corrected'))
    assert corrected.get('Fixed_parameters.W_x').eval() == pytest.approx(.6435)
    assert corrected.get('Adjustable_parameters.x_AK').eval() == 1e6


def test_published_reference_and_ast_rhs_agree():
    model, rhs, details = compile_reference()
    y = model.initial_values(as_floats=True)
    for pace in (0, -100.):
        reference = model.evaluate_derivatives(y, inputs={'time': 0., 'pace': pace})
        np.testing.assert_allclose(rhs(0, y, pace), reference, rtol=1e-10, atol=1e-12)
    assert len(y) == 51
    assert details['unit_status'] == 'source_units_incomplete_not_clinically_released'
    original, _ = load_reference()
    assert original.get('stimulus_current.Istim').binding() is None


def test_ecme_bounds_and_cancellation():
    with pytest.raises(ValueError):
        ECMEExperiment().validate()
    spec = ECMEExperiment(duration_ms=50, acknowledge_source_unit_limitations=True)
    with pytest.raises(InterruptedError):
        simulate_ecme(spec, cancelled=lambda: True)
    with pytest.raises(ValueError):
        replace(spec, duration_ms=1e9).validate()


def test_ecme_actual_transient_and_metabolic_response():
    spec = ECMEExperiment(duration_ms=50, acknowledge_source_unit_limitations=True)
    t, y, meta = simulate_ecme(spec)
    _, stressed, stress_meta = simulate_ecme(replace(spec, respiratory_capacity_scale=.5))
    assert t[0] == 0 and t[-1] == 50 and np.isfinite(y).all()
    assert meta['metrics']['V']['max'] > 0
    assert meta['metrics']['ATPi']['min'] > 0
    assert not np.allclose(y, stressed)
    assert stress_meta['experiment']['respiratory_capacity_scale'] == .5
    assert not meta['clinical_use'] and not meta['human_ORD_coupled']
    json.dumps(meta, allow_nan=False)


def test_pairing_excludes_overlapping_parent():
    from vcg_biomarker.simulation.edgar import paired_signal_manifest
    rows = []
    for episode in ('Run0001', 'Run0001-b1'):
        for kind in ('epicardial', 'body_surface'):
            rows.append({'kind': 'signal', 'signal_type': kind, 'subject_id': 'X',
                'episode_id': episode, 'parent_record': 'Run0001', 'cohort': 'Utah',
                'samples': 100, 'reference_native': [1, 2, 3], 'sha256': 'fixture', 'path': kind+episode})
    pairs = paired_signal_manifest(rows)
    assert pairs[0]['exclude_overlapping_parent_record']
    assert pairs[1]['status'] == 'native_pair'


@pytest.mark.parametrize('kind', ['published_ecme', 'mitochondrial_assay'])
def test_real_metabolic_worker_roundtrip(tmp_path, kind):
    import subprocess
    import sys
    from dataclasses import asdict
    from vcg_biomarker.simulation.mitochondria import MitochondrialExperiment
    if kind == 'published_ecme':
        request = {'kind': kind, 'ecme_experiment': asdict(ECMEExperiment(duration_ms=5,
                    acknowledge_source_unit_limitations=True))}
        target = 'ecme.npz'
    else:
        request = {'kind': kind, 'mitochondrial_experiment': asdict(MitochondrialExperiment(duration_s=.05))}
        target = 'mitochondria.npz'
    (tmp_path/'request.json').write_text(json.dumps(request), encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'vcg_biomarker.simulation.worker', str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr+result.stdout
    with np.load(tmp_path/target, allow_pickle=False) as data:
        assert np.isfinite(data['states']).all()
        assert json.loads(str(data['metadata']))['clinical_use'] is False


def test_metabolic_gui_request_and_result(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from vcg_biomarker.gui.metabolism_panel import MetabolismDialog, MetabolismResult
    app = QApplication.instance() or QApplication([])
    dialog = MetabolismDialog()
    with pytest.raises(ValueError):
        dialog.request()
    dialog.acknowledge.setChecked(True)
    assert dialog.request()['kind'] == 'published_ecme'
    dialog.model.setCurrentIndex(1)
    assert dialog.request()['kind'] == 'mitochondrial_assay'
    assert not dialog.controls['cycle'].isEnabled()
    np.savez(tmp_path/'mito.npz', times_s=[0., 1.], states=[[1., 2.], [2., 3.]],
             metadata=json.dumps({'schema': 'published-mitochondrial-assay/1', 'state_names': ['ATP', 'ADP'],
                                  'state_units': ['M', 'M']}))
    view = MetabolismResult(tmp_path/'mito.npz')
    view.quantity.setCurrentIndex(1)
    app.processEvents()
    assert view.plot.listDataItems()[0].yData.tolist() == [2., 3.]
    view.close()
    dialog.close()
