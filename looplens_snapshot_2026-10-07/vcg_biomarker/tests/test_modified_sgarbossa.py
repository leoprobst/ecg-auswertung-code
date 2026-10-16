import pytest
from vcg_biomarker.core.data_loader import LEAD_NAMES_12
from vcg_biomarker.mimic.ischemia_patterns import modified_sgarbossa


def evaluate(lead='V2', r=.1, s=1., j=0., **kwargs):
    qrs = {name: {'r_height': 1., 's_depth': .1} for name in LEAD_NAMES_12}
    st = {name: {'j': 0.} for name in LEAD_NAMES_12}
    qrs[lead] = {'r_height': r, 's_depth': s}
    st[lead]['j'] = j
    return modified_sgarbossa(qrs, st, context=kwargs.get('context', 'lbbb'), quality_ok=kwargs.get('quality_ok', True))


@pytest.mark.parametrize('lead,r,s,j,positive', [
    ('I',1,.1,.1,True), ('V2',.1,1,-.1,True), ('V4',.1,1,-.1,False),
    ('V2',.1,1,.25,True), ('V2',.1,1,.249,False),
    ('V2',.01,.1,.09,False), ('V2',1,.1,-.1,False),
])
def test_components_and_boundaries(lead,r,s,j,positive):
    assert evaluate(lead,r,s,j)['positive'] is positive


def test_context_quality_and_unknown_are_not_negative():
    assert evaluate(context='atrial_pacing')['positive'] is None
    assert evaluate(context=None)['positive'] is None
    assert evaluate(quality_ok=False)['positive'] is None
    assert evaluate(j=float('nan'))['positive'] is None
    assert evaluate(r=1,s=1)['positive'] is None


def test_pacing_has_separate_reference_without_extended_depression_leads():
    result = evaluate(j=.25, context='ventricular_pacing')
    assert result['positive'] is True
    assert result['source'].endswith('/34172301/')
    assert evaluate('V4', .1, 1, -.2, context='ventricular_pacing')['positive'] is False


def test_registry_and_case_bound_context():
    from vcg_biomarker.mimic.baseline_registry import BaselineRegistry
    from vcg_biomarker.mimic.ischemia_patterns import validate_lab_ischemia_context
    BaselineRegistry()
    item = dict(schema_version='1.0.0', case_id='case', context='lbbb',
                author='reviewer', reason='reviewed morphology', created_at='2026-09-21', revision_id='one')
    assert validate_lab_ischemia_context(item, 'case') == item
    with pytest.raises(ValueError):
        validate_lab_ischemia_context(item, 'other')
    with pytest.raises(ValueError):
        validate_lab_ischemia_context({**item, 'reason': ''}, 'case')
