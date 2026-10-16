from dataclasses import replace

import numpy as np
import pytest

from vcg_biomarker.simulation.twin_hypotheses import TwinObservation, infer_hypotheses, validate_heldout
from vcg_biomarker.simulation.twin_emulator import train_compare, select_active


def fixture():
    obs = [TwinObservation('qrs', 100., 'ms', 'synthetic', None, 'fit', 'marker', 'ecg1'),
           TwinObservation('echo', 50., 'mm', 'synthetic', None, 'additional', 'clinical', 'echo1'),
           TwinObservation('later', 110., 'ms', 'synthetic', None, 'heldout', 'marker', 'ecg2')]
    candidates = [{'id': str(i), 'group': str(i), 'model_version': 'test', 'provenance': 'known synthetic',
                   'physiology_valid': True, 'numerics_valid': True, 'in_domain': True,
                   'predictions': dict(zip(['qrs', 'echo', 'later'], values)),
                   'units': {'qrs': 'ms', 'echo': 'mm', 'later': 'ms'}}
                  for i, values in enumerate([[100., 50., 110.], [101., 52., 160.], [99., 90., 105.]])]
    errors = {k: {'covariance': (np.eye(3)*.25).tolist(), 'source': 'synthetic known error'}
              for k in ['measurement', 'geometry_electrodes', 'physiological_nuisance', 'model_discrepancy']}
    return obs, candidates, errors


def test_ambiguity_additional_evidence_and_heldout_separation():
    obs, candidates, errors = fixture()
    result = infer_hypotheses(obs, candidates, errors)
    assert result.status == 'not_distinguishable'
    assert result.probabilities is None
    assert result.candidates[-1]['status'] == 'contradicted'
    changed = [*obs[:2], replace(obs[2], value=999)]
    assert infer_hypotheses(changed, candidates, errors) == result
    assert validate_heldout(obs, candidates, result, errors).compatible_set_coverage == 1
    assert validate_heldout(changed, candidates, result, errors).compatible_set_coverage == 0
    # Correct structural information distinguishes models; demography is not substituted.
    obs[1] = replace(obs[1], value=90)
    assert infer_hypotheses(obs, candidates, errors).status == 'one_group_on_finite_design'


def test_double_count_units_covariance_and_false_certainty_rejected():
    obs, candidates, errors = fixture()
    obs[1] = replace(obs[1], dependency_group='ecg1', representation='signal')
    with pytest.raises(ValueError, match='twice'):
        infer_hypotheses(obs, candidates, errors)
    obs, candidates, errors = fixture()
    candidates[0]['units']['qrs'] = 's'
    with pytest.raises(ValueError, match='units'):
        infer_hypotheses(obs, candidates, errors)
    obs, candidates, errors = fixture()
    with pytest.raises(ValueError, match='synthetic'):
        infer_hypotheses(obs, candidates, errors, probability_validation={'calibration_passed': True})
    candidates[0]['in_domain'] = False
    assert infer_hypotheses(obs, candidates, errors).candidates[0]['status'] == 'outside_checked_domain'


def test_gp_simple_comparison_and_group_leakage():
    rng = np.random.default_rng(15)
    x = rng.uniform(-1, 1, size=(180, 3))
    y = np.column_stack([x[:, 0]+x[:, 1]**2, np.sin(x[:, 2])])
    roles = np.array(['fit']*100+['calibration']*40+['audit']*40)
    report, uncertainty = train_compare(x, y, roles, np.arange(180), x[:20])
    assert report['status'] == 'research_simulator_emulator'
    assert len(uncertainty) == 20
    assert len(select_active(uncertainty, list(range(20)), 5)) == 5
    with pytest.raises(ValueError, match='crosses'):
        train_compare(x, y, roles, np.zeros(180))
    changed = y.copy()
    changed[140:] += 100
    other, _ = train_compare(x, changed, roles, np.arange(180))
    assert other['selected_by_fit_cv'] == report['selected_by_fit_cv']


def test_clinical_nullable_controls_do_not_become_normal_values():
    import pandas as pd
    from audit_tools.twin_clinical_pilot import numeric_matrix, compare_models
    matrix = numeric_matrix([pd.Series([1, pd.NA], dtype='Int64'), pd.Series([True, pd.NA], dtype='boolean')])
    assert np.isnan(matrix[1]).all()
    assert compare_models(np.array([1., 2.]), np.array([3., 4.]), matrix, np.array([1, 2]))['status'] == 'non_estimable'


def test_missing_heldout_prediction_does_not_change_fit_set():
    obs, candidates, errors = fixture()
    expected = infer_hypotheses(obs, candidates, errors)
    candidates[0]['predictions']['later'] = np.nan
    result = infer_hypotheses(obs, candidates, errors)
    assert result == expected
    assert validate_heldout(obs, candidates, result, errors).compatible_set_coverage is None


def test_information_comparison_requires_identical_heldout_case():
    from vcg_biomarker.simulation.twin_hypotheses import compare_information_levels
    obs, candidates, errors = fixture()
    levels = [{'name': 'ecg', 'observations': obs, 'errors': errors},
              {'name': 'echo', 'observations': [obs[0], replace(obs[1], value=90), obs[2]], 'errors': errors}]
    result = compare_information_levels(levels, candidates, known_reference_id='0')
    # An overly strong wrong structural constraint contracts the set but loses truth.
    assert result['levels'][0]['reference_retained']
    assert not result['levels'][1]['reference_retained']
    levels[1]['observations'][2] = replace(obs[2], value=999)
    with pytest.raises(ValueError, match='identical'):
        compare_information_levels(levels, candidates)


def test_campaign_case_checkpoint_and_safe_interruption(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vcg_biomarker.simulation import twin_campaign, solver
    (tmp_path/'activation/cases').mkdir(parents=True)
    monkeypatch.setattr(twin_campaign, '_load_tissue', lambda *_: None)
    result = SimpleNamespace(activation_ms=np.arange(8.), ecg=np.ones((4, 8)),
        times_ms=np.array([0., 200., 400., 600.]), lead_names=list(twin_campaign.LEADS),
        metadata={'activated_fraction': 1.})
    monkeypatch.setattr(solver, 'simulate', lambda *a, **kw: result)
    case = {'id': 0, 'parameters': [1., 1., 0., 1.], 'preset': 'reference'}
    row = twin_campaign.task(tmp_path, 'activation', case, 'frozen_hash')
    assert row['status'] == 'computed'
    assert row['outputs'][2] == .5
    assert (tmp_path/'activation/cases/00000.json').exists()
    def stop(*a, **kw):
        raise InterruptedError()
    monkeypatch.setattr(solver, 'simulate', stop)
    assert twin_campaign.task(tmp_path, 'activation', dict(case, id=1), 'frozen_hash')['status'] == 'interrupted'
    assert not (tmp_path/'activation/cases/00001.json').exists()
