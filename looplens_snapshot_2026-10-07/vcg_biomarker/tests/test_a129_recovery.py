from copy import deepcopy

import pytest

from audit_tools.recover_a129_runtime_binding import validate_administrative_delta


def snapshot():
    return {'source_tree_hash': 'source', 'feature_implementation_hash': 'features',
            'scientific_config_hash': 'science', 'operational_config_hash': 'op',
            'dependency_lock': {'path': 'old/lock', 'sha256': 'lock', 'size': 1},
            'project_definition': {'path': 'old/project', 'sha256': 'project', 'size': 2},
            'inputs': {'evidence_manifest': {'sha256': 'evidence'},
                       'inventory_manifest': {'sha256': 'inventory'}}}


def test_only_known_administrative_delta_is_allowed():
    old = snapshot()
    new = deepcopy(old)
    new['dependency_lock']['path'] = 'runtime/lock'
    new['operational_config_hash'] = 'newop'
    new['inputs']['inventory_manifest']['sha256'] = 'new_inventory'
    differences, inputs = validate_administrative_delta(old, new)
    assert inputs == ['inventory_manifest']
    assert 'dependency_lock' in differences


@pytest.mark.parametrize('key', ['source_tree_hash', 'feature_implementation_hash', 'scientific_config_hash'])
def test_recovery_rejects_scientific_changes(key):
    old = snapshot()
    new = deepcopy(old)
    new[key] = 'changed'
    with pytest.raises(ValueError, match='Non-administrative'):
        validate_administrative_delta(old, new)


def test_recovery_rejects_changed_dependencies_and_evidence():
    old = snapshot()
    for key in ('dependency_lock', 'evidence_manifest'):
        new = deepcopy(old)
        target = new[key] if key == 'dependency_lock' else new['inputs'][key]
        target['sha256'] = 'changed'
        with pytest.raises(ValueError):
            validate_administrative_delta(old, new)
