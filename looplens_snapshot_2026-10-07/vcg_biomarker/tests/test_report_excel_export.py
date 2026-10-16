"""The workbook is a bounded convenience export, never the full tables."""
from __future__ import annotations

import pandas as pd

from vcg_biomarker.mimic.reports import EXCEL_SHEET_ROW_LIMIT, _atomic_write_excel


def test_large_tables_are_truncated_in_the_workbook_and_listed(tmp_path):
    tables = {
        "screening_results": pd.DataFrame({"a": range(50), "b": range(50)}),
        "run_manifest": pd.DataFrame({"key": ["run_id"], "value": ["x"]}),
    }
    target = tmp_path / "reports.xlsx"
    _atomic_write_excel(tables, target, row_limit=10)
    workbook = pd.read_excel(target, sheet_name=None)
    assert len(workbook["screening_results"]) == 10
    assert len(workbook["run_manifest"]) == 1
    listed = workbook["truncated_tables"].set_index("table")
    assert listed.loc["screening_results", "rows_total"] == 50
    assert listed.loc["screening_results", "rows_in_workbook"] == 10
    assert "run_manifest" not in listed.index
    assert not list(tmp_path.glob(".*.tmp.xlsx"))


def test_the_default_limit_is_far_below_the_screen_size():
    # 308,000 screening rows at the old million-row limit cost 16.6 GiB.
    assert EXCEL_SHEET_ROW_LIMIT <= 50_000
