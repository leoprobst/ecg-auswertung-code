import pytest
from vcg_biomarker.gui.lab_marker_viewmodel import marker_row, finding_validation_metrics, classical_rule_view, classical_rule_details


def test_rule_detail_preserves_recorded_definition_and_missing_provenance():
    rule = dict(value=0., evaluable=True, exploratory=False, positive=False,
                definition='QRS <120 ms <b>literal</b>', unit='ms')
    text = classical_rule_details('rule', rule, 'old-analysis')
    assert 'QRS <120 ms <b>literal</b>' in text
    assert 'Quellen-ID: nicht dokumentiert' in text
    assert 'Analysestand: old-analysis' in text
    assert 'Messwert: 0.0' in text
    assert 'source_id' not in rule


def test_rule_detail_shows_explicit_source_and_exclusion():
    rule = dict(evaluable=False, exploratory=False, source_id='saved-source',
                version='v1', exclusion_reason='P missing', leads=['I', 'II'])
    text = classical_rule_details('rule', rule, 'a')
    assert 'saved-source' in text and 'P missing' in text and 'v1' in text


def test_st_profiles_disclose_erratum_without_relabeling_historical_rule():
    rule = dict(definition='historical J+60 definition', value=-.07, unit='mV',
                evaluable=True, exploratory=False, positive=True)
    text = classical_rule_details('st_depression_contiguous', rule, 'a')
    assert 'historical J+60 definition' in text
    assert 'Erratum 2024' in text
    assert 'mindestens einer Ableitung' in text
    assert 'kein zusätzliches ESC-Ergebnis' in text
    assert rule['positive'] is True and rule['value'] == -.07
    assert 'Erratum 2024' not in classical_rule_details('other_rule', rule, 'a')


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '1'])
def test_classical_evaluable_flag_does_not_override_invalid_measurement(value):
    result = classical_rule_view(dict(value=value, evaluable=True, exploratory=False, positive=True))
    assert not result['eligible']


@pytest.mark.parametrize('exploratory', [None, 0, '', 'False', True])
def test_classical_origin_must_be_explicit(exploratory):
    assert not classical_rule_view(dict(value=1., evaluable=True, exploratory=exploratory))['eligible']


def test_classical_zero_and_continuous_measurements_are_supported():
    result = classical_rule_view(dict(value=0., evaluable=True, exploratory=False, positive=False))
    assert result['eligible']
    assert 'kein Krankheitsausschluss' in result['status']
    result = classical_rule_view(dict(value=120., evaluable=True, exploratory=False, positive=None))
    assert result['eligible'] and result['status'] == 'Messwert ohne binäres Urteil'


def test_classical_truthy_strings_are_not_binary_verdicts():
    assert not classical_rule_view(dict(value=1., evaluable=True, exploratory=False, positive='false'))['eligible']


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), '0', True])
def test_invalid_marker_value_never_shown_as_normal(value):
    assert marker_row('x', value, {})['value_text'] == 'nicht berechenbar'


def test_zero_valid_and_recorded_metadata_only():
    view = marker_row('x', 0., {'unit': 'ms', 'family': 'Repolarisation', 'notes': 'six beats'})
    assert view['value_text'] == '0'
    assert view['unit_status'].startswith('ms · Messwert vorhanden')
    assert 'repolarisation' in view['search_text']
    assert 'six beats' in view['tooltip']
    assert 'nicht dokumentiert' in marker_row('x', 0., None)['unit_status']


@pytest.mark.parametrize('source', [None, 'unknown', 'train', 'validation'])
def test_legacy_unqualified_sensitivity_is_not_validation(source):
    value = finding_validation_metrics({'sensitivity': .99, 'sensitivity_source': source})
    assert value['sensitivity'] is None
    assert 'Trainings-Fallback' in value['sensitivity_note']


@pytest.mark.parametrize('value', [None, True, '0.95', float('nan'), float('inf'), -.01, 1.01])
def test_invalid_or_missing_explicit_validation_never_uses_top_level(value):
    result = finding_validation_metrics({'sensitivity': .99, 'sensitivity_source': 'validation_operating_point',
        'primary_operating_point': {'sensitivity_validation': value}})
    assert result['sensitivity'] is None


def test_legacy_explicit_validation_wins_over_unqualified_training_fallback():
    finding = {'sensitivity': .99, 'primary_operating_point': {'sensitivity_validation': .6}}
    assert finding_validation_metrics(finding)['sensitivity'] == .6
    assert finding['sensitivity'] == .99  # Original artifact remains untouched.


def test_conflicting_declared_validation_is_not_silently_selected():
    value = finding_validation_metrics({'sensitivity': .99, 'sensitivity_source': 'validation_operating_point',
        'primary_operating_point': {'sensitivity_validation': .6}})
    assert value['sensitivity'] is None
    assert 'Widersprüchliche' in value['sensitivity_note']


def test_valid_zero_and_actual_specificity_are_preserved_independently():
    value = finding_validation_metrics({'sensitivity': 0., 'sensitivity_source': 'validation_operating_point',
        'specificity': .90, 'specificity_source': 'validation_operating_point'})
    assert value['sensitivity'] == 0.
    assert value['specificity'] == .90  # Never relabel achieved .90 as target .95.
@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True, '0'])
def test_classical_matrix_invalid_value_is_not_normal(value):
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_measurement_cells
    cells = classical_measurement_cells('rule', {'value': value, 'unit': 'mV',
        'evaluable': True, 'exploratory': False, 'positive': False})
    assert cells[1] == 'nicht verfügbar · mV'
    assert 'nicht beurteilbar' in cells[2]


def test_classical_matrix_reports_exclusion_and_real_negative_without_ruleout():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_measurement_cells
    rule = {'value': 0., 'unit': 'mV', 'evaluable': False, 'exploratory': False,
            'exclusion_reason': 'missing segment', 'positive': False}
    assert 'missing segment' in classical_measurement_cells('rule', rule)[2]
    rule['evaluable'] = True
    cells = classical_measurement_cells('rule', rule)
    assert cells[1] == '0.0 · mV'
    assert 'kein Krankheitsausschluss' in cells[2]


@pytest.mark.parametrize('name', ['pr_prolongation_gt_200', 'first_degree_av_block'])
@pytest.mark.parametrize('value,marked', [(200, False), (300, False), (300.1, True)])
def test_pr_display_never_converts_interval_to_verified_av_block(name, value, marked):
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_measurement_cells
    rule = {'value': value, 'unit': 'ms', 'evaluable': True, 'positive': value > 200, 'exploratory': True}
    text = classical_measurement_cells(name, rule)[2]
    assert '1:1-P–QRS' in text
    assert ('Ausgeprägte' in text) == marked


def test_pr_unmeasurable_not_marked_even_if_historical_value_exists():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_measurement_cells
    text = classical_measurement_cells('pr_prolongation_gt_200',
        {'value': 400, 'unit': 'ms', 'evaluable': False})[2]
    assert 'Ausgeprägte' not in text
