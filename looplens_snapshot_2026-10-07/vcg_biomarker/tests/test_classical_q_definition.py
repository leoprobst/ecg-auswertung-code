from types import SimpleNamespace

import numpy as np
import pytest

from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.mimic.classical import ClassicalECGCriteria, _initial_q_metric


@pytest.mark.parametrize('r,s,expected', [
    (0., .2, True), (.05, .5, False), (.001, .5, False),
    (0., 0., False), (0., .09, False), (float('nan'), .2, None),
    (0., float('inf'), None), (-.1, .2, None), (True, .2, None),
])
def test_strict_qs_does_not_confuse_small_r_with_absent_r(r, s, expected):
    from vcg_biomarker.mimic.classical import _strict_qs_component
    assert _strict_qs_component({'r_height': r, 's_depth': s}) is expected


def test_qs_group_and_missingness():
    qrs = {lead: {'r_height': 1., 's_depth': .2} for lead in LEAD_NAMES_12}
    q = {lead: {'duration_ms': 0., 'depth_mv': 0.} for lead in LEAD_NAMES_12}
    def evaluate():
        return ClassicalECGCriteria()._scar_criteria(qrs, q, excluded=False)['qs_complexes']
    qrs['V2']['r_height'] = qrs['V3']['r_height'] = .05
    assert evaluate().positive is False
    qrs['V2']['r_height'] = qrs['V3']['r_height'] = 0.
    assert evaluate().positive is True
    qrs['V3']['r_height'] = float('nan')
    assert evaluate().positive is None
    assert not evaluate().evaluable
    assert evaluate().value is None


def test_nonfinite_qrs_is_not_zero_amplitude():
    from vcg_biomarker.mimic.classical import _qrs_amplitudes
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=0), r_offset=SimpleNamespace(sample=5))
    ecg = np.zeros((6, 12))
    ecg[2, 0] = np.nan
    metrics = _qrs_amplitudes(ecg, beat, np.zeros(12), 500)
    assert np.isnan(metrics['I']['r_height'])
    assert np.isnan(metrics['I']['s_depth'])


def scar(changes):
    qrs = {lead: {'r_height': 1., 's_depth': .2} for lead in LEAD_NAMES_12}
    q = {lead: {'duration_ms': 0., 'depth_mv': 0., 'q_to_r_ratio': 0.} for lead in LEAD_NAMES_12}
    for lead, duration, depth in changes:
        q[lead].update(duration_ms=duration, depth_mv=depth, q_to_r_ratio=depth)
    return ClassicalECGCriteria()._scar_criteria(qrs, q, excluded=False)['pathological_q_waves']


@pytest.mark.parametrize('changes,positive', [
    ([('I',20,.2),('aVL',20,.2)],False),
    ([('I',40,.05),('aVL',40,.05)],False),
    ([('I',30,.1),('aVL',30,.1)],True),
    ([('I',30,.1),('V5',30,.1)],False),
    ([('II',30,.1),('aVF',30,.1)],True),
    ([('III',50,.3),('aVR',50,.3)],False),
    ([('V2',20,.05)],False),
    ([('V2',22,.05)],True),
    ([('V3',22,.05)],True),
    ([('V4',30,.1),('V6',30,.1)],True),
    ([('II',30,.1),('III',50,.3)],False),
    ([('aVL',30,.1),('aVF',30,.1)],False),
    ([('V1',50,.3)],False),
    ([('V2',22,0)],False),
])
def test_q_component_explicit_thresholds_and_lead_groups(changes, positive):
    assert scar(changes).positive is positive


def metric(samples, fs=1000, offset=0):
    beat = SimpleNamespace(r_onset=SimpleNamespace(sample=0), r_offset=SimpleNamespace(sample=len(samples)-1))
    return _initial_q_metric(np.asarray(samples)+offset, beat, offset, fs)


def test_small_initial_r_then_s_is_not_q():
    result = metric([0, .01, -.2, -.3, -.2, .5, 1., 0])
    assert result['duration_ms'] == 0
    assert result['depth_mv'] == 0


def test_q_ratio_uses_full_qrs_r_not_pretrigger_fragment():
    result = metric([0]+[-.2]*30+[0, .5, 2., .5, 0])
    assert result['duration_ms'] == 30
    assert result['depth_mv'] == .2
    assert result['q_to_r_ratio'] == .1
    shifted = metric([0]+[-.2]*30+[0, .5, 2., .5, 0], offset=10)
    assert shifted['duration_ms'] == 30
    assert shifted['q_to_r_ratio'] == pytest.approx(.1)


def test_qs_is_not_mislabeled_as_initial_q_followed_by_r():
    assert metric([0,-.1,-.5,-.2,0])['duration_ms'] == 0


def test_missing_q_samples_abstain_instead_of_normal():
    assert np.isnan(metric([0,-.1,np.nan,1.,0])['duration_ms'])
    result = scar([('I',float('nan'), .1)])
    assert result.positive is None
    assert not result.evaluable


@pytest.mark.parametrize('fs', [0, -1, float('nan'), float('inf')])
def test_invalid_sampling_rate_abstains(fs):
    assert np.isnan(metric([0, -.1, 1., 0], fs=fs)['duration_ms'])


def test_baseline_registry_source_lock_accepts_q_definition():
    from vcg_biomarker.mimic.baseline_registry import BaselineRegistry
    BaselineRegistry()


def test_q_component_gui_does_not_claim_complete_udmi_evaluation():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_rule_details
    text = classical_rule_details('pathological_q_waves', {}, 'historical')
    assert 'nicht die vollständige UDMI-Tabelle' in text
    assert '>20 ms, nicht ≥20 ms' in text
    assert 'schließt weder' in text
