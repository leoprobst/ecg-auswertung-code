"""Evidence spill allowance must be charged, not just set in DuckDB."""
import ast
from contextlib import contextmanager
import inspect

from vcg_biomarker.mimic import evidence, spill_budget
from vcg_biomarker.mimic.storage import DuckDBCatalog


def test_evidence_build_passes_allowance_to_managed_catalog():
    tree = ast.parse(inspect.getsource(evidence.build_clinical_evidence))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'DuckDBCatalog']
    assert len(calls) == 1
    values = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(values['spill_limit_gb'], ast.Name)
    assert values['spill_limit_gb'].id == 'EVIDENCE_BUILD_SPILL_GIB'


def test_evidence_sql_cap_matches_reservation(tmp_path, monkeypatch):
    reservations = []

    @contextmanager
    def reserve(path, amount, **kwargs):
        reservations.append(amount)
        yield {}

    monkeypatch.setattr(spill_budget, 'reserve_spill', reserve)
    catalog = DuckDBCatalog(tmp_path / 'test.duckdb', memory_limit_gb=1,
                            threads=1, spill_limit_gb=evidence.EVIDENCE_BUILD_SPILL_GIB)
    with catalog.connect(isolated=True) as connection:
        setting = connection.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0]
        assert setting == '48.0 GiB'
    assert reservations == [48]
