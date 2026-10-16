import pytest
from vcg_biomarker.mimic.classical import _t_inversion_rs_component, _has_contiguous_pair_nullable


@pytest.mark.parametrize('t,ratio,expected', [
    (-.10, 2, False), (-.101, 2, True), (-.3, 1, False),
    (-.3, .5, False), (-.3, float('nan'), None),
    (float('nan'), 2, None), (-.3, -1, None), (.2, float('nan'), False),
])
def test_quantitative_udmi_component(t, ratio, expected):
    assert _t_inversion_rs_component({'t_peak': t, 'r_to_s_ratio': ratio}) is expected


def test_incomplete_negative_t_pair_remains_unknown():
    flags = {'I': True, 'aVL': _t_inversion_rs_component({'t_peak': -.3})}
    assert _has_contiguous_pair_nullable(flags, groups=(('I', 'aVL'),)) is None


@pytest.mark.parametrize('pair,expected', [
    (('I', 'aVL'), True), (('II', 'aVF'), True), (('V2', 'V3'), True),
    (('V1', 'V2'), False), (('aVR', 'aVL'), False), (('II', 'V5'), False),
])
def test_implemented_t_lead_groups(pair, expected):
    from vcg_biomarker.mimic.classical import T_INVERSION_GROUPS
    from vcg_biomarker.core.data_loader import LEAD_NAMES_12
    flags = {lead: lead in pair for lead in LEAD_NAMES_12}
    assert _has_contiguous_pair_nullable(flags, groups=T_INVERSION_GROUPS) is expected


def test_t_view_distinguishes_pattern_from_acute_ischemia():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_rule_details
    text = classical_rule_details('t_inversion_contiguous', {}, 'old')
    assert 'nicht automatisch akute Ischämie' in text
    assert 'prominenten R-Welle' in text
    assert 'nicht vollständig implementiert' in text
