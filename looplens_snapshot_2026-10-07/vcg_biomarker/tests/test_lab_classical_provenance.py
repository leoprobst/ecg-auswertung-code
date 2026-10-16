from vcg_biomarker.mimic.baseline_registry import BaselineRegistry
from vcg_biomarker.mimic.lab_classical_provenance import capture_classical_provenance


def test_source_capture_preserves_values_and_original_and_group_semantics():
    original = {'st_depression_contiguous': {'value': 2, 'positive': True},
                't_inversion_contiguous': {'value': None, 'positive': None},
                'pathological_q_waves': {'value': 0, 'positive': False},
                'lab_only_unregistered': {'value': None}}
    registry = BaselineRegistry()
    result = capture_classical_provenance(original, registry)
    assert 'source_records' not in original['st_depression_contiguous']
    assert result['st_depression_contiguous']['value'] == 2
    assert result['st_depression_contiguous']['source_records']
    assert ['I', 'V5'] in result['st_depression_contiguous']['contiguity_groups']
    assert all('V1' not in group for group in result['t_inversion_contiguous']['contiguity_groups'])
    assert 'V2_or_V3' in result['pathological_q_waves']['contiguity_scope']
    assert result['lab_only_unregistered']['source_records'] == []
    assert result['st_depression_contiguous']['registry_hash'] == registry.registry_hash


def test_display_uses_saved_provenance_without_current_registry():
    from vcg_biomarker.gui.lab_marker_viewmodel import classical_rule_details
    text = classical_rule_details('historical', {'source_records': [{'source_version': 'historic_version'}],
        'contiguity_groups': [['I','aVL']], 'registry_hash': 'saved_hash'}, 'old')
    assert 'historic_version' in text and 'saved_hash' in text
    assert 'aVL' in text
