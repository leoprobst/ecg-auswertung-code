"""Below-limit laboratory results live in ``comments``; the label must see them."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.evidence import _lab_censoring_sql
from vcg_biomarker.mimic.lab_intervals import censored_result_sql, upper_reference_state_sql


def test_censored_comments_become_bounded_values_with_a_marker():
    rows = pd.DataFrame(
        {
            "valuenum": [None, None, None, 0.05, None],
            "comments": [
                "<0.01.  CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.",
                "LESS THAN 0.01.  cTropnT > 0.10 ng/mL suggests Acute MI.",
                "GREATER THAN 25.0",
                "cTropnT > 0.10 ng/mL suggests Acute MI.",
                "___",
            ],
        }
    )
    effective, censoring = _lab_censoring_sql()
    connection = duckdb.connect()
    connection.register("rows", rows)
    frame = connection.execute(
        f"SELECT {effective} AS value_numeric, {censoring} AS censoring FROM rows"
    ).fetchdf()
    assert frame.value_numeric.tolist()[:4] == [0.01, 0.01, 25.0, 0.05]
    assert pd.isna(frame.value_numeric.iloc[4])
    assert frame.censoring.tolist()[:3] == ["below_limit", "below_limit", "above_limit"]
    # A measured value keeps precedence over any comment, and a comment that
    # only quotes the assay note is not a result.
    assert pd.isna(frame.censoring.iloc[3]) and pd.isna(frame.censoring.iloc[4])


@pytest.mark.parametrize(("text", "expected"), [
    (">0.01", "positive"), (">=0.01", "indeterminate"),
    ("<0.10", "indeterminate"), ("<0.01", "negative"),
    ("<=0.01", "negative"), (">=0.02", "positive"),
    ("LESS THAN OR EQUAL TO 1e-2", "negative"),
    (">+1e-2", "positive"), ("<-1e-2", "negative"),
])
def test_whole_interval_classification(text, expected):
    bound, censor = censored_result_sql("value")
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE labs(value VARCHAR, comments VARCHAR, valuenum DOUBLE)")
        # Qualifiers remain authoritative even when valuenum already holds the limit.
        connection.execute("INSERT INTO labs VALUES (?, '', 0.01)", [text])
        state = upper_reference_state_sql("bound", "censor", "0.01")
        result = connection.execute(
            f"SELECT {state} FROM (SELECT {bound} AS bound, {censor} AS censor FROM labs)"
        ).fetchone()[0]
    assert result == expected


@pytest.mark.parametrize("text", [">unknown", "<NaN", ">=inf", "LESS THAN invalid"])
def test_malformed_qualifier_never_falls_back_to_exact_numeric(text):
    bound, censor = censored_result_sql("value")
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE labs(value VARCHAR, comments VARCHAR, valuenum DOUBLE)")
        connection.execute("INSERT INTO labs VALUES (?, '', 0.01)", [text])
        result = connection.execute(f"SELECT {bound}, {censor} FROM labs").fetchone()
    assert result == (None, None)


@pytest.mark.parametrize("text,endpoint,expected", [
    ("<3.0", "severe_hypokalemia_near_ecg", "positive"),
    ("<=3.0", "severe_hypokalemia_near_ecg", "indeterminate"),
    (">=3.0", "severe_hypokalemia_near_ecg", "negative"),
    (">=6.5", "severe_hyperkalemia_near_ecg", "positive"),
    ("<6.5", "severe_hyperkalemia_near_ecg", "negative"),
    ("<=6.5", "severe_hyperkalemia_near_ecg", "indeterminate"),
    (">5.5", "hyperkalemia_near_ecg", "positive"),
    ("<3.5", "hypokalemia_near_ecg", "positive"),
    ("<5.5", "hypokalemia_near_ecg", "indeterminate"),
    (">=3.5", "hyperkalemia_near_ecg", "indeterminate"),
])
def test_electrolyte_label_uses_entire_interval(text, endpoint, expected):
    from vcg_biomarker.mimic.labels import _label_rules
    bound, censor = censored_result_sql("value")
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE labs(value VARCHAR, comments VARCHAR, valuenum DOUBLE)")
        connection.execute("INSERT INTO labs VALUES (?, '', NULL)", [text])
        rule = _label_rules()[endpoint]
        row = connection.execute(f"""SELECT {rule.state} FROM (
            SELECT {bound} AS potassium_nearest_bound, {censor} AS potassium_nearest_censoring,
            1 AS potassium_count_6h FROM labs)""").fetchone()
    assert row[0] == expected


@pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
@pytest.mark.parametrize("censor", [None, "below_limit", "at_or_below_limit", "above_limit", "at_or_above_limit"])
def test_interval_comparisons_against_explicit_allowed_values(operator, censor):
    from vcg_biomarker.mimic.lab_intervals import interval_comparison_sql
    import operator as op
    operations = {"<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge}
    samples = {None: [3.0], "below_limit": [-100, 2.99], "at_or_below_limit": [-100, 3.0],
               "above_limit": [3.01, 100], "at_or_above_limit": [3.0, 100]}
    with duckdb.connect() as connection:
        for threshold in (2.0, 3.0, 4.0):
            expression = interval_comparison_sql("test", operator, str(threshold))
            actual = connection.execute(f"SELECT {expression} FROM "
                "(SELECT 3.0 AS test_nearest_bound, ?::VARCHAR AS test_nearest_censoring)", [censor]).fetchone()[0]
            assert actual == all(operations[operator](value, threshold) for value in samples[censor])


def test_interval_survives_actual_evidence_windows_pivot_and_labels(tmp_path, monkeypatch):
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic import evidence
    from vcg_biomarker.mimic.labels import _label_rules

    config = MimicProjectConfig.discover(database_root=tmp_path / "db", work_root=tmp_path / "work")
    config.ensure_directories()
    time = pd.Timestamp("2020-01-01 12:00")
    raw = pd.DataFrame({
        "labevent_id": range(1, 6), "subject_id": [1, 2, 3, 4, 1],
        "hadm_id": [1, 2, 3, 4, 1], "itemid": evidence.LAB_ITEM_GROUPS["troponin_t"][0],
        "charttime": [time] * 4 + [time + pd.Timedelta(hours=1)],
        "storetime": [time] * 5, "value": [">0.01", "<0.10", "<0.01", "0.02", "0.005"],
        "valuenum": [0.01, None, None, 0.02, 0.005], "comments": [""] * 5,
        "valueuom": ["ng/mL"] * 5, "ref_range_lower": [0.0] * 5,
        "ref_range_upper": [0.01] * 5, "flag": [""] * 5,
    })
    master = pd.DataFrame({"subject_id": [1, 2, 3, 4], "study_id": [11, 12, 13, 14],
                           "hadm_id": [1, 2, 3, 4], "ecg_time": [time] * 4})
    master.to_parquet(config.master_root / "ecg_master_index.parquet", index=False)
    monkeypatch.setattr(evidence, "_event_or_raw_relation", lambda *args: "test_labs")
    with duckdb.connect() as connection:
        connection.execute("SET threads=1")
        connection.register("test_labs", raw)
        evidence._build_lab_evidence(connection, config, None)
        labs = pd.read_parquet(config.evidence_root / "labs.parquet").set_index("labevent_id")
        assert labs.loc[[1, 2, 3], "value_numeric"].isna().all()
        assert labs.loc[1, "value_interval_lower"] == 0.01
        assert bool(labs.loc[1, "value_interval_lower_open"])
        windows = config.evidence_root / "ecg_lab_windows.parquet"
        result = connection.execute(evidence._lab_feature_pivot_query(
            windows.as_posix(), "", partition_index=0, partition_count=1,
        )).fetchdf().sort_values("subject_id")
        assert result.troponin_interval_state_standard.tolist() == [
            "positive", "indeterminate", "negative", "positive"
        ]
        connection.register("lab_pivot", result)
        rule = _label_rules()["troponin_positive_any"]
        states = connection.execute(f"SELECT {rule.state} FROM lab_pivot ORDER BY subject_id").fetchall()
        assert [row[0] for row in states] == ["positive", "indeterminate", "negative", "positive"]
    serial = pd.read_parquet(config.evidence_root / "ecg_troponin_serial.parquet")
    serial = serial.set_index("subject_id")
    # Audit D3: ">0.01" followed by 0.005 (URL 0.01) certainly crosses the
    # URL, so the censored series is decided instead of dropped. The single
    # values of subjects 2-4 give no serial assessment.
    assert bool(serial.loc[1, "serial_assessment_available"])
    assert bool(serial.loc[1, "acute_change_proxy"])
    assert serial.loc[1, "serial_interval_state"] == "positive"
    assert not serial.loc[[2, 3, 4], "serial_assessment_available"].any()
    # Relative change stays an exact-values statistic.
    assert serial.max_relative_delta.isna().all()
