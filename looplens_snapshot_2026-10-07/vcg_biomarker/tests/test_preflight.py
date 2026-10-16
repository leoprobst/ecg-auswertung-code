"""Finding the next missing column in minutes instead of days.

``therapy-response`` had never run. It reached the front of a month-long
study, bound a query against ``heart_rate_bpm`` -- a column the extraction
does not write, because its role takes it out of every panel -- and failed.
Three stages behind it had also never run.

A checker is only useful if it is trusted, and it is only trusted if it does
not cry wolf. The first version flagged the very stage whose fix had just
introduced ``AS heart_rate_bpm``: a name the query *produces* read as a name
it *requires*. Both directions are tested here, because a checker that is
ignored is worth less than none.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vcg_biomarker.mimic.preflight import (
    STAGE_MODULES,
    _WIDE_COLUMN_LIST,
    catalog_marker_names,
    check_stage,
    marker_names_referenced,
    render_report,
    run_preflight,
)


CATALOG = ("qrs_duration_ms", "qt_interval_ms", "heart_rate_bpm", "svg_azimuth_deg")


def test_a_required_column_that_is_missing_is_reported(tmp_path):
    (tmp_path / "stage.py").write_text(
        'QUERY = "SELECT f.qrs_duration_ms, f.qt_interval_ms FROM f"\n'
        "x = connection.execute(QUERY).fetchdf()\n",
        encoding="utf-8",
    )
    finding = check_stage(
        "demo",
        "stage",
        feature_columns={"qrs_duration_ms"},
        catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.ok is False
    assert finding.missing_columns == ("qt_interval_ms",)


def test_a_column_the_query_creates_is_not_a_column_it_needs(tmp_path):
    """The false positive that would have got this checker switched off."""

    (tmp_path / "stage.py").write_text(
        'QUERY = "SELECT m.machine_heart_rate_bpm AS heart_rate_bpm FROM m"\n',
        encoding="utf-8",
    )
    finding = check_stage(
        "demo",
        "stage",
        feature_columns={"qrs_duration_ms"},
        catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.missing_columns == ()
    assert finding.ok is True


def test_an_alias_anywhere_in_the_module_counts(tmp_path):
    """The alias and its use are rarely in the same string literal."""

    (tmp_path / "stage.py").write_text(
        'RATE = "m.machine_heart_rate_bpm AS heart_rate_bpm"\n'
        'QUERY = f"SELECT heart_rate_bpm FROM ({RATE})"\n',
        encoding="utf-8",
    )
    finding = check_stage(
        "demo",
        "stage",
        feature_columns=set(),
        catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.missing_columns == ()


def test_only_catalog_names_are_treated_as_markers(tmp_path):
    """Otherwise every SQL keyword becomes a missing column."""

    (tmp_path / "stage.py").write_text(
        'QUERY = "SELECT count(*) FROM read_parquet(x) WHERE split = 1"\n',
        encoding="utf-8",
    )
    finding = check_stage(
        "demo",
        "stage",
        feature_columns=set(),
        catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.missing_columns == ()


def test_a_missing_module_is_said_rather_than_passed(tmp_path):
    finding = check_stage(
        "demo", "absent", feature_columns=set(), catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.notes
    assert "fehlt" in finding.notes[0]


def test_the_wide_column_shape_is_recognised():
    source = (
        'columns = ", ".join(\n'
        '    f"f.{_quote_identifier(item)}" for item in selected\n'
        ")\n"
    )
    assert _WIDE_COLUMN_LIST.search(source)


def test_one_marker_at_a_time_is_not_the_wide_shape():
    """validation.py selects ``f.{feature_column}``; that must not flag."""

    assert not _WIDE_COLUMN_LIST.search('sql = f"SELECT f.{feature_column} FROM f"')


def test_the_size_number_is_labelled_as_a_bound_not_a_measurement(tmp_path):
    """A constant identical for every stage once passed as a finding.

    Measured afterwards: the gap stage was reported at 9.01 GB and actually
    materialises 0.41 GB, because it selects 80 of the 1,889 columns. The
    number may only appear as what it is.
    """

    (tmp_path / "stage.py").write_text(
        'columns = ", ".join(f"f.{item}" for item in selected)\n'
        "frame = connection.execute(query).fetchdf()\n",
        encoding="utf-8",
    )
    finding = check_stage(
        "demo",
        "stage",
        feature_columns={"a", "b"},
        catalog=CATALOG,
        package_root=tmp_path,
        cohort_rows=1_000_000,
    )
    assert finding.worst_case_materialisation_gb is not None
    assert any("Laufzeit" in note for note in finding.notes)

    class _Report:
        feature_column_count = 1889
        catalog_marker_count = 1906
        findings = [finding]

    text = render_report(_Report())
    assert "obere Schranke" in text
    assert "keine Messung" in text


def test_every_listed_module_exists():
    root = Path("vcg_biomarker/mimic")
    for stage, modules in STAGE_MODULES.items():
        for module in modules:
            assert (root / f"{module}.py").is_file(), f"{stage}: {module}"


def test_the_catalog_is_not_empty():
    """An empty catalog would make every check pass for the wrong reason."""

    assert len(catalog_marker_names()) > 500


def test_every_module_a_stage_reads_through_is_checked():
    """The gap that made this checker report "ok" on the worst case.

    ``repeatability`` was listed alone. It issues no queries: the reads live
    in ``trajectory.py``, where a self-join carries 2 x 1,753 marker columns
    against the 1,889-column glob. A checker pointed at the wrong file is
    worse than no checker, because it produces a clean line.
    """

    assert "trajectory" in STAGE_MODULES["repeatability"]
    for stage, modules in STAGE_MODULES.items():
        assert isinstance(modules, tuple), stage
        assert modules, stage


def test_the_widest_build_in_the_project_is_recognised():
    """``for marker in marker_ids`` was not in the pattern."""

    source = (
        'marker_first = ",\n".join(\n'
        '    f\'a."{marker}" AS "{marker}_first"\' for marker in marker_ids\n'
        ")\n"
    )
    assert _WIDE_COLUMN_LIST.search(source)


def test_an_alias_from_a_neighbouring_module_still_counts(tmp_path):
    """Modules are concatenated, so a name declared in one covers the other."""

    (tmp_path / "left.py").write_text(
        'RATE = "m.machine_heart_rate_bpm AS heart_rate_bpm"\n', encoding="utf-8"
    )
    (tmp_path / "right.py").write_text(
        'QUERY = "SELECT heart_rate_bpm FROM x"\n', encoding="utf-8"
    )
    lonely = check_stage(
        "demo", ("right",), feature_columns=set(), catalog=CATALOG,
        package_root=tmp_path,
    )
    assert lonely.missing_columns == ("heart_rate_bpm",)

    together = check_stage(
        "demo", ("left", "right"), feature_columns=set(), catalog=CATALOG,
        package_root=tmp_path,
    )
    assert together.missing_columns == ()


def test_one_missing_module_does_not_hide_the_others(tmp_path):
    (tmp_path / "present.py").write_text(
        'QUERY = "SELECT f.qt_interval_ms FROM f"\n', encoding="utf-8"
    )
    finding = check_stage(
        "demo", ("absent", "present"), feature_columns=set(), catalog=CATALOG,
        package_root=tmp_path,
    )
    assert finding.missing_columns == ("qt_interval_ms",)


def test_no_feature_reading_stage_still_carries_a_bare_union_read():
    """The 19.36 GB pattern, on the stages that have not run yet."""

    import re

    from pathlib import Path as _Path

    pattern = re.compile(r"read_parquet\(\s*'?\{(\w*feature\w*)\}'?\s*,\s*union_by_name=true")
    offenders = []
    for stage, modules in STAGE_MODULES.items():
        for name in modules:
            path = _Path("vcg_biomarker/mimic") / f"{name}.py"
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                # therapy_response keeps one as the fallback default for a
                # caller that passes no clause; the guided path always does.
                if name == "therapy_response":
                    continue
                offenders.append(f"{name}.py:{line} ({stage})")
    assert not offenders, offenders


def test_preflight_uses_registered_panel_contract_before_first_extraction(
    tmp_path,
):
    report = run_preflight(
        SimpleNamespace(features_root=tmp_path / "features"),
        stages=("validate",),
        cohort_rows=1_000,
    )
    assert report.ok
    assert report.feature_column_count > 1_000
    assert (
        report.feature_schema_source
        == "registered_panel_contract_before_extraction"
    )
    assert report.feature_glob.endswith("split=train/part-*.parquet")


def test_preflight_rejects_unknown_missing_panel_contract(tmp_path):
    with pytest.raises(ValueError, match="Unknown feature panel key"):
        run_preflight(
            SimpleNamespace(features_root=tmp_path / "features"),
            panel_key="not_a_real_panel",
            stages=("validate",),
        )
