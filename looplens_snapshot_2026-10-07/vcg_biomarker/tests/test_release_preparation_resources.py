import pytest

from audit_tools.refresh_release_upstream import preparation_resource_budget


def test_preparation_admits_only_with_explicit_headroom():
    required = 10 * 10**9 + 4 * 1024**3
    with pytest.raises(RuntimeError, match='headroom'):
        preparation_resource_budget(required - 1)
    budget = preparation_resource_budget(required)
    assert budget['duckdb_memory_limit_gb'] == 10
    assert budget['workers'] == 1
    assert budget['headroom_bytes'] == 4 * 1024**3
