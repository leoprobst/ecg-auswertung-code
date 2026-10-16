import pytest
from vcg_biomarker.gui.lab_axis_review import axis_review


def payload(value):
    return {'classical_criteria': {'left_axis_deviation': {'value': value, 'evaluable': True}}}


@pytest.mark.parametrize('value,category', [(-180,'right'),(180,'right'),(-179,'extreme'),
    (-90,'left'),(-30,'normal_range'),(90,'normal_range'),(90.1,'right')])
def test_boundaries(value, category):
    assert axis_review(payload(value), 50)['category'] == category


@pytest.mark.parametrize('value', [None, True, float('nan'), 181, '30'])
def test_invalid_is_not_normal(value):
    assert axis_review(payload(value),50)['category'] == 'unknown'


def test_unknown_age_and_conflicting_axes():
    assert axis_review(payload(0))['category'] == 'unknown'
    p = payload(0)
    p['classical_criteria']['right_axis_deviation'] = {'value': 120, 'evaluable': True}
    assert 'Widersprüchliche' in axis_review(p,50)['text']
