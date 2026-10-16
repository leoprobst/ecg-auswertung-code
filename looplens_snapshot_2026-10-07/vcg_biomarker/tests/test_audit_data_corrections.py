"""Audit 2026-09-23, data and label findings D1-D10."""
from __future__ import annotations

import duckdb


def test_d1_machine_axes_in_minus90_to_270_convention_are_kept():
    from vcg_biomarker.mimic.ingest import _axis_sql, _machine_measurements_query

    rows = duckdb.connect().execute(
        f"SELECT raw, {_axis_sql('raw')} FROM (VALUES "
        "('-90'),('180'),('181'),('269'),('270'),('271'),('999'),('29999'),"
        "(''),('abc'),('-181')) t(raw)"
    ).fetchall()
    assert dict(rows) == {
        "-90": -90.0, "180": 180.0, "181": -179.0, "269": -91.0, "270": -90.0,
        "271": None, "999": None, "29999": None, "": None, "abc": None, "-181": None,
    }
    query = _machine_measurements_query()
    for axis in ("p", "qrs", "t"):
        assert f"TRY_CAST({axis}_axis AS DOUBLE) - 360" in query
        assert f"AS {axis}_axis_deg" in query


def test_d2_hemolyzed_potassium_is_not_a_measurement_end_to_end(tmp_path, monkeypatch):
    import pandas as pd

    from vcg_biomarker.mimic import evidence
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.labels import _label_rules

    config = MimicProjectConfig.discover(database_root=tmp_path / "db", work_root=tmp_path / "work")
    config.ensure_directories()
    time = pd.Timestamp("2020-01-01 12:00")
    comments = [
        "HEMOLYSIS FALSELY ELEVATES K.",   # 1: only a hemolyzed value
        "",                                # 2: clean
        "SPECIMEN NOT HEMOLYZED.",         # 3: explicitly clean
        "SLIGHTLY HEMOLYZED SPECIMEN.",    # 4: hemolyzed, repeat draw 1 h later
        "",                                # 4: the repeat
    ]
    raw = pd.DataFrame({
        "labevent_id": range(1, 6), "subject_id": [1, 2, 3, 4, 4],
        "hadm_id": [1, 2, 3, 4, 4], "itemid": 50971,
        "charttime": [time] * 4 + [time + pd.Timedelta(hours=1)],
        "storetime": [time] * 5, "value": ["6.0", "6.0", "6.0", "6.0", "4.5"],
        "valuenum": [6.0, 6.0, 6.0, 6.0, 4.5], "comments": comments,
        "valueuom": ["mEq/L"] * 5, "ref_range_lower": [3.3] * 5,
        "ref_range_upper": [5.1] * 5, "flag": ["abnormal"] * 4 + [""],
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
        assert labs.loc[[1, 4], "value_status"].tolist() == ["hemolysis_reported"] * 2
        assert labs.loc[[2, 3, 5], "value_status"].tolist() == ["usable"] * 3
        assert labs.loc[1, "hemolysis_state"] == "reported"
        assert labs.loc[3, "hemolysis_state"] == "explicitly_absent"
        assert pd.isna(labs.loc[2, "hemolysis_state"])
        issues = pd.read_parquet(config.quarantine_root / "lab_value_issues.parquet")
        assert sorted(issues.labevent_id) == [1, 4]
        windows = config.evidence_root / "ecg_lab_windows.parquet"
        pivot = connection.execute(evidence._lab_feature_pivot_query(
            windows.as_posix(), "", partition_index=0, partition_count=1,
        )).fetchdf().sort_values("subject_id")
        connection.register("lab_pivot", pivot)
        rule = _label_rules()["hyperkalemia_near_ecg"]
        states = [row[0] for row in connection.execute(
            f"SELECT {rule.state} FROM lab_pivot ORDER BY subject_id").fetchall()]
    # 1: pseudohyperkalemia is no case; 2/3: real; 4: the clean repeat decides.
    assert states[0] != "positive"
    assert states[1:] == ["positive", "positive", "negative"]


def _serial_frame(tmp_path, series, name="labs"):
    """series: {subject: [(value_text, url), ...]} one assay, hourly draws."""
    import pandas as pd

    from vcg_biomarker.mimic.lab_intervals import censored_result_sql

    time = pd.Timestamp("2020-01-01 12:00")
    rows, event = [], 0
    for subject, values in series.items():
        for position, (text, url) in enumerate(values):
            event += 1
            rows.append({"labevent_id": event, "subject_id": subject, "itemid": 51003,
                         "value": text, "url": url, "comments": "",
                         "valuenum": None if text[0] in "<>" else float(text),
                         "charttime": time + pd.Timedelta(hours=position)})
    raw = pd.DataFrame(rows)
    bound, censor = censored_result_sql("value")
    with duckdb.connect() as connection:
        connection.register("raw", raw)
        labs = connection.execute(f"""
            SELECT *, CASE WHEN censoring IS NULL THEN bound END AS value_numeric,
                CASE WHEN censoring IS NULL OR censoring IN ('above_limit','at_or_above_limit')
                     THEN bound END AS value_interval_lower,
                CASE WHEN censoring IS NULL OR censoring IN ('below_limit','at_or_below_limit')
                     THEN bound END AS value_interval_upper,
                COALESCE(censoring = 'above_limit', FALSE) AS value_interval_lower_open,
                COALESCE(censoring = 'below_limit', FALSE) AS value_interval_upper_open,
                url AS reference_upper,
                CASE WHEN censoring IS NULL AND url > 0 THEN bound / url END AS upper_reference_ratio,
                'troponin_t' AS lab_group, 'ng/mL' AS unit, 'usable' AS value_status
            FROM (SELECT *, TRY_CAST({bound} AS DOUBLE) AS bound, {censor} AS censoring FROM raw)
        """).df()
    master = pd.DataFrame({"subject_id": list(series), "study_id": [s * 10 for s in series],
                           "ecg_time": time})
    labs_path, master_path = tmp_path / f"{name}.parquet", tmp_path / f"{name}_master.parquet"
    labs.to_parquet(labs_path, index=False)
    master.to_parquet(master_path, index=False)
    return master_path.as_posix(), labs_path.as_posix()


def _run_serial(query_builder, master, labs):
    with duckdb.connect() as connection:
        return connection.execute(query_builder(master, labs, partition_index=0,
                                                partition_count=1)).df().set_index("subject_id")


def test_d3_censored_troponin_series_are_decided_on_the_whole_interval(tmp_path):
    from vcg_biomarker.mimic.evidence import _lab_troponin_serial_query

    series = {
        1: [("<0.01", 0.01), ("0.30", 0.01)],     # rise from undetectable
        2: [("<0.01", 0.01), ("<0.01", 0.01)],    # undetectable throughout
        3: [("<0.10", 0.01), ("0.05", 0.01)],     # limit above URL: undecidable
        4: [("0.02", 0.01), ("0.021", 0.01)],     # stable elevation, exact
        5: [("0.005", 0.01), ("0.03", 0.01)],     # exact crossing
        6: [("0.50", None), ("0.60", None)],      # no URL: undecidable
        7: [(">0.01", 0.01), ("0.005", 0.01)],    # falls below URL
        8: [("0.30", 0.01)],                      # single value
        9: [("<0.01", 0.01), ("0.008", 0.01)],    # below URL throughout
    }
    result = _run_serial(_lab_troponin_serial_query, *_serial_frame(tmp_path, series))
    states = result["serial_interval_state"].to_dict()
    assert states == {1: "positive", 2: "negative", 3: "indeterminate", 4: "negative",
                      5: "positive", 6: "indeterminate", 7: "positive",
                      8: "single_value", 9: "negative"}
    assessed = result["serial_assessment_available"].to_dict()
    assert assessed == {1: True, 2: True, 3: False, 4: True, 5: True, 6: False,
                        7: True, 8: False, 9: True}
    assert result["acute_change_proxy"].to_dict() == {
        subject: state == "positive" for subject, state in states.items()}
    assert result.loc[1, "qualifying_first_labevent_id"] == 1
    assert result.loc[1, "qualifying_last_labevent_id"] == 2
    # Audit 5.2.3: the rise from "not detectable" as its own phenotype.
    # Series 3 starts below a limit that lies above the URL: not "undetectable".
    assert result["undetectable_rise_state"].to_dict() == {
        1: "rise_from_undetectable", 2: "undetectable_without_rise",
        3: "not_applicable", 4: "not_applicable", 5: "not_applicable",
        6: "not_applicable", 7: "not_applicable", 8: "not_applicable",
        9: "undetectable_without_rise"}


def test_d3_interval_rule_equals_the_old_rule_on_exact_series(tmp_path):
    """Exact series must be classified exactly as before the change."""
    import ast
    import random

    from pathlib import Path

    source = (Path(__file__).parent / "fixtures" / "old_lab_troponin_serial_query.py.txt").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "_lab_troponin_serial_query")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "old", "exec"), namespace)
    old_query = namespace["_lab_troponin_serial_query"]
    from vcg_biomarker.mimic.evidence import _lab_troponin_serial_query

    rng = random.Random(3)
    series = {}
    for subject in range(1, 301):
        count = rng.choice([1, 2, 2, 3, 4])
        url = rng.choice([0.01, 0.04, None]) if subject % 7 else 0.01
        values = [rng.choice([0.0, 0.004, 0.009, 0.01, 0.011, 0.012, 0.02, 0.03, 0.3])
                  for _ in range(count)]
        series[subject] = [(f"{value}", url if url is not None else 0.01) for value in values]
    master, labs = _serial_frame(tmp_path, series)
    old = _run_serial(old_query, master, labs)
    new = _run_serial(_lab_troponin_serial_query, master, labs)
    for column in ("acute_change_proxy", "serial_assessment_available", "max_serial_count",
                   "max_url_ratio", "max_relative_delta", "acute_change_event_time",
                   "qualifying_first_labevent_id", "qualifying_last_labevent_id"):
        import pandas as pd
        pd.testing.assert_series_equal(old[column].sort_index(), new[column].sort_index(),
                                       check_dtype=False, obj=column)


def test_d4_survival_is_ascertained_through_the_registry_window():
    import pandas as pd

    from vcg_biomarker.mimic.labels import _label_rules

    rule = _label_rules()["mortality_30d"]
    frame = pd.DataFrame({
        "death_delta_days": [5, 30, 40, None, None, None, -2],
        # 6 h to the next contact, but discharged: registry covers a year.
        "followup_hours": [None, None, None, 6, 6, 6, None],
        "vital_status_followup_hours": [None, None, None, 24 * 400, 24 * 10, None, None],
    })
    with duckdb.connect() as connection:
        connection.register("f", frame)
        rows = connection.execute(
            f"SELECT {rule.state}, {rule.eligibility} FROM f").fetchall()
    assert [row[0] for row in rows] == [
        "positive", "positive", "negative", "negative", "indeterminate",
        "indeterminate", "indeterminate"]
    assert [row[1] for row in rows] == [
        "eligible", "eligible", "eligible", "eligible", "insufficient_evidence",
        "insufficient_evidence", "insufficient_evidence"]


def test_d4_timeline_carries_the_registry_window():
    import inspect

    from vcg_biomarker.mimic import evidence

    source = inspect.getsource(evidence)
    assert "e.last_encounter_time + INTERVAL 365 DAY" in source
    assert "AS vital_status_followup_hours" in source


def test_d5_lvh_follows_ase_mass_and_geometry():
    import pandas as pd

    from vcg_biomarker.mimic.evidence import _devereux_context_columns_sql
    from vcg_biomarker.mimic.labels import _label_rules

    echo = pd.DataFrame({
        "septal_thickness_cm": [1.0, 1.3, 1.0, 0.9, 1.6, None],
        "posterior_wall_thickness_cm": [1.0, 1.3, 1.0, 0.8, 1.6, 1.0],
        "lv_internal_diameter_diastole_cm": [4.8, 4.4, 6.4, 5.0, 4.0, 4.8],
        "echo_body_surface_area_m2": [1.9, 1.9, 1.9, 1.9, 1.9, 1.9],
    })
    master = pd.DataFrame({"sex": ["M", "F", "M", "F", "M", "M"]})
    with duckdb.connect() as connection:
        connection.register("ew", echo)
        connection.register("m", master)
        context = connection.execute(
            f"SELECT m.sex, {_devereux_context_columns_sql()} FROM ew POSITIONAL JOIN m"
        ).df()
    # Hand-checked: 0.832 * (6.8^3 - 4.8^3) + 0.6 = 170.19 g.
    assert round(context.loc[0, "lv_mass_devereux_g"], 2) == 170.19
    assert context.lv_geometry_ase.tolist()[:5] == [
        "normal_geometry", "concentric_hypertrophy", "eccentric_hypertrophy",
        "normal_geometry", "concentric_hypertrophy"]
    assert context.lv_geometry_ase.isna().tolist()[5]
    context["lv_mass_index"] = None
    context["has_tte_30d"] = True
    context["lv_wall_category"] = [None, None, None, "Normal", None, "Mod symmetric LVH"]
    rules = _label_rules()
    with duckdb.connect() as connection:
        connection.register("c", context)
        states = {
            label: [row[0] for row in connection.execute(
                f"SELECT {rules[label].state} FROM c").fetchall()]
            for label in ("echo_lvh_strict", "lvh_concentric_ase", "lvh_eccentric_ase")
        }
    # The eccentric LVH (row 2, normal walls) is a case, not a control; the
    # wall-category text alone (row 5, no mass) no longer decides.
    assert states["echo_lvh_strict"][:5] == [
        "negative", "positive", "positive", "negative", "positive"]
    assert states["echo_lvh_strict"][5] != "positive"
    assert states["lvh_concentric_ase"][:5] == [
        "negative", "positive", states["lvh_concentric_ase"][2], "negative", "positive"]
    assert states["lvh_concentric_ase"][2] not in {"positive", "negative"}
    assert states["lvh_eccentric_ase"][2] == "positive"
    assert states["lvh_eccentric_ase"][1] not in {"positive", "negative"}


def test_d6_power_gate_uses_development_splits_only(tmp_path):
    import pandas as pd

    from vcg_biomarker.mimic.storage import DuckDBCatalog
    from vcg_biomarker.mimic.validation import _development_power_gate

    rows = []
    for subject in range(200):
        split = "train" if subject % 2 else "validation"
        rows.append(("broad", subject, split, "positive" if subject < 60 else "negative"))
        rows.append(("dead", subject, split, "positive" if subject < 20 else "negative"))
    # 500 locked-test positives must not rescue the dead endpoint.
    rows += [("dead", 10_000 + subject, "locked_test", "positive") for subject in range(500)]
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame(rows, columns=["label_id", "subject_id", "split", "label_state"]).to_parquet(cohort)
    catalog = DuckDBCatalog(tmp_path / "c.duckdb", memory_limit_gb=1, threads=1)
    gate = _development_power_gate(catalog, cohort, ("broad", "dead", "missing"))
    assert gate == {"dead": (20, 180), "missing": (0, 0)}


def test_d7_aliases_point_at_screened_endpoints():
    from vcg_biomarker.mimic.label_registry import ENDPOINT_ALIASES, definitions_by_id

    definitions = definitions_by_id()
    for alias, (canonical, relation) in ENDPOINT_ALIASES.items():
        assert alias in definitions and canonical in definitions
        assert canonical not in ENDPOINT_ALIASES
        assert relation in {"identical", "complement"}


def test_d9_af_after_the_ecg_in_the_same_admission_is_new_onset():
    import pandas as pd

    from vcg_biomarker.mimic.labels import _label_rules

    time = pd.Timestamp("2020-01-01 12:00")
    frame = pd.DataFrame({
        "ecg_time": [time] * 6,
        "current_report_af_state": ["negative", "negative", "negative", "positive",
                                    "negative", "negative"],
        # 0: post-operative AF, same admission, timed by a later AF ECG
        # 1: index-admission code only, no timed event
        # 2: AF-coded admission before this one
        # 3: this ECG already shows AF
        # 4: no AF, long follow-up
        # 5: earlier AF report
        "first_af_report_time": [time + pd.Timedelta(days=2), None, None, time, None,
                                 time - pd.Timedelta(days=5)],
        "af_dx_prior_admission": [False, False, True, False, False, False],
        "af_dx_index_admission": [True, True, False, True, False, False],
        "next_af_event_delta_hours": [48.0, None, None, None, None, None],
        "next_af_event_source": ["machine_report", None, None, None, None, None],
        "next_af_dx_hadm_id": [None] * 6,
        "first_af_report_study_id": pd.array([7, None, None, 3, None, 1], dtype="Int64"),
        "death_delta_days": [None] * 6,
        "followup_hours": [24 * 400] * 6,
    })
    rule = _label_rules()["new_onset_af_30d"]
    with duckdb.connect() as connection:
        connection.register("f", frame)
        rows = connection.execute(
            f"SELECT {rule.state}, {rule.eligibility}, {rule.evidence_ids} FROM f").fetchall()
    assert [row[0] for row in rows] == [
        "positive", "indeterminate", "indeterminate", "indeterminate", "negative",
        "indeterminate"]
    assert [row[1] for row in rows] == [
        "eligible", "insufficient_evidence", "excluded", "excluded", "eligible", "excluded"]
    assert rows[0][2] == '["study:7"]'


def test_d10_equal_distance_prefers_the_earlier_echo():
    import inspect

    from vcg_biomarker.mimic import evidence

    source = inspect.getsource(evidence)
    assert source.count("ORDER BY ABS(delta_hours), delta_hours ASC, measurement_id DESC") == 2
    assert "ORDER BY ABS(delta_hours), measurement_id DESC" not in source
    with duckdb.connect() as connection:
        winner = connection.execute(
            "SELECT measurement_id FROM (VALUES (1, 24.0), (2, -24.0), (3, 30.0)) t(measurement_id, delta_hours) "
            "ORDER BY ABS(delta_hours), delta_hours ASC, measurement_id DESC LIMIT 1").fetchone()[0]
    assert winner == 2


def test_t2_future_phenome_window_ends_at_the_last_contact():
    import inspect

    import pandas as pd

    from vcg_biomarker.mimic import future_phenome

    source = inspect.getsource(future_phenome)
    assert "TRY_CAST(p.date_of_death AS TIMESTAMP)\n                ) AS last_observed_time" not in source
    start = pd.Timestamp("2020-01-01")
    frame = pd.DataFrame({
        "subject_id": [1, 2],
        "index_end_time": [start, start],
        # 1: last contact day 10, death day 400 -> not observed to day 365
        # 2: contact through day 500
        "last_observed_time": [start + pd.Timedelta(days=10), start + pd.Timedelta(days=500)],
        "date_of_death": [start + pd.Timedelta(days=400), pd.NaT],
        "sex": ["M", "F"],
    })
    masks = future_phenome._future_state_masks(
        frame, pd.Series(dtype="datetime64[ns]"), pd.Series(dtype=object), horizon_days=365)
    assert not bool(masks["negative"][0])
    assert bool(masks["negative"][1])


def test_x3_x4_phewas_controls_respect_sex_and_the_phecode_hierarchy():
    import numpy as np
    import pandas as pd

    from vcg_biomarker.mimic import phenome

    assert phenome._phecode_ancestors("CV_401.12") == ("CV_401", "CV_401.1")
    assert phenome._related_phenotype_ids(
        "CV_401.1", ["CV_401", "CV_401.1", "CV_401.12", "CV_40", "CV_402"]
    ) == frozenset({"CV_401", "CV_401.12"})

    subjects = np.arange(1, 13)
    features = pd.DataFrame({
        "subject_id": subjects, "study_id": subjects * 10, "split": "train",
        "sex": ["M"] * 6 + ["F"] * 6,
        "is_control_index": True,
        "marker": np.linspace(0.0, 1.0, 12),
    })
    current = pd.DataFrame({
        "subject_id": [1, 2, 7, 3], "study_id": [10, 20, 70, 30], "split": "train",
        # 1, 2 (men) and 7 (woman, a coding error) carry the prostate code;
        # 3 carries its parent code.
        "phenotype_id": ["GU_627.1", "GU_627.1", "GU_627.1", "GU_627"],
    })
    definitions = pd.DataFrame({
        "phenotype_id": ["GU_627", "GU_627.1"],
        "display_name": ["parent", "child"], "body_system": ["GU", "GU"],
        "male_only": ["TRUE", "TRUE"], "female_only": ["FALSE", "FALSE"],
    }).set_index("phenotype_id")
    records = phenome._screen_split(
        features, current, pd.DataFrame(), phenotype_ids=("GU_627.1",),
        feature_ids=("marker",), panel_map={"marker": "p"},
        definition_lookup=definitions, min_inference=1, min_formal=1, progress=None,
    )
    record = records[0]
    # Cases: the two men. Controls: men 4-6; subject 3 (parent code) and all
    # women are removed.
    assert record["n_case"] == 2 and record["n_control"] == 3
    assert record["sex_restriction"] == "M"
    assert record["n_related_code_controls_excluded"] == 1
    rows = phenome._phenome_replication_row_index(
        features, current, target_bodies=(("GU_627.1", "GU"),), history_by_body={},
        definition_lookup=definitions)
    case_rows, control_rows = rows[("GU_627.1", "GU")]
    assert sorted(features.loc[case_rows, "subject_id"]) == [1, 2]
    assert sorted(features.loc[control_rows, "subject_id"]) == [4, 5, 6]


def test_x5_spline_sensitivity_and_admission_urgency():
    import numpy as np
    import pandas as pd

    from vcg_biomarker.mimic import phenome_adjustment as adjustment

    knots = np.array([20.0, 40.0, 60.0, 80.0])
    x = np.linspace(0, 100, 201)
    basis = adjustment.restricted_cubic_spline_basis(x, knots)
    assert basis.shape == (201, 3)
    np.testing.assert_allclose(basis[:, 0], x)
    # Linear beyond the outer knots: second differences vanish there.
    for column in (1, 2):
        tail = basis[x > 80, column]
        head = basis[x < 20, column]
        assert np.allclose(np.diff(tail, 2), 0, atol=1e-9)
        assert np.allclose(head, 0)
    frame = pd.DataFrame({
        "admission_type": ["EW EMER.", "ELECTIVE", "URGENT", None, None, "EU OBSERVATION"],
        "index_ed_encounter": [0, 0, 0, 1, 0, 0],
    })
    assert adjustment._emergency_admission(frame).tolist()[:4] == [1.0, 0.0, 1.0, 1.0]
    assert np.isnan(adjustment._emergency_admission(frame)[4])
    assert adjustment._emergency_admission(frame)[5] == 1.0

    rng = np.random.default_rng(8)
    n = 4000
    age = rng.uniform(20, 90, n)
    heart = rng.uniform(50, 120, n)
    marker = rng.normal(size=n)
    logit = -1.0 + 0.8 * marker + 0.0004 * (age - 55) ** 2
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    design = pd.DataFrame({
        "subject_id": np.arange(n), "label": label, "aligned_marker": marker,
        "age": age, "heart_rate": heart, "sex_male": rng.integers(0, 2, n).astype(float),
    })
    context = pd.DataFrame({"admission_type": rng.choice(["EW EMER.", "ELECTIVE"], n)})
    result = adjustment._spline_sensitivity(
        design, context, phenotype_id="p", feature_id="m",
        policy=adjustment.PhenomeAdjustmentPolicy(), unadjusted_coefficient=0.8,
    )
    assert result["spline_adjustment_status"] == "evaluable", result
    assert 1.8 < result["spline_adjusted_or_per_sd"] < 3.0


def test_x6_post_outcome_markers_never_count_as_replicated():
    import pandas as pd

    from vcg_biomarker.mimic.validation import _initial_explorer_evidence

    criteria = pd.DataFrame({
        "stage": ["validation_replicated"] * 2,
        "core_replication_evidence_passed": [True, True],
        "project_wide_multiplicity_confirmed": [True, True],
        "cross_fit_confirmed": [True, True],
        "verification_bias_gate_passed": [True, True],
        "post_outcome_hypothesis": [False, True],
        "roc_auc": [0.7, 0.7], "roc_auc_ci_low": [0.65, 0.65], "roc_auc_ci_high": [0.75, 0.75],
    })
    evidence = _initial_explorer_evidence(criteria, preview=False)
    assert evidence["replication_passed"].tolist() == [True, False]
    assert evidence["replication_independence"].tolist() == [
        "prespecified_marker", "not_independent_post_outcome_hypothesis"]


def test_x7_untreated_pairs_are_compared_like_with_like():
    import numpy as np
    import pandas as pd

    from vcg_biomarker.mimic.therapy_response import (
        _untreated_control_pair_query,
        untreated_comparison,
    )

    def pairs(n, normalized_share, gap, rhythm, start):
        k = int(round(n * normalized_share))
        return pd.DataFrame({
            "subject_id": np.arange(start, start + n),
            "tachycardia_pattern_before": True,
            "rhythm_normalized_flag": [True] * k + [False] * (n - k),
            "rhythm_class_before": rhythm,
            "hours_before_to_exposure": gap / 2, "hours_exposure_to_after": gap / 2,
        })

    # Treated: 60 % normalise in the 0-1 h bin; untreated normalise 30 % there
    # but 90 % in the 3-4 h bin, which dominates the raw controls.
    treated = pairs(100, 0.6, 0.8, "af", 0)
    near = pairs(50, 0.3, 0.8, "af", 1000).drop(columns=["hours_before_to_exposure", "hours_exposure_to_after"])
    far = pairs(500, 0.9, 3.5, "af", 2000).drop(columns=["hours_before_to_exposure", "hours_exposure_to_after"])
    near["ecg_gap_hours"], far["ecg_gap_hours"] = 0.8, 3.5
    controls = pd.concat([near, far], ignore_index=True)
    result = untreated_comparison(treated, controls, iterations=200, seed=1)
    # Only the matched 0-1 h stratum counts: 0.6 - 0.3, not 0.6 - (raw 0.85).
    assert result["n_strata_matched"] == 1
    assert abs(result["difference"] - 0.3) < 1e-9
    assert result["ci_low"] < 0.3 < result["ci_high"]
    sql = _untreated_control_pair_query("m.parquet", "med.parquet")
    assert "INTERVAL 240 MINUTE" in sql and "INTERVAL 24 HOUR" in sql
    assert "next_hadm_id = hadm_id" in sql


def test_x7_untreated_pair_sql_runs_on_real_shapes(tmp_path):
    import pandas as pd

    from vcg_biomarker.mimic.therapy_response import _untreated_control_pair_query

    t = pd.Timestamp("2020-01-01 08:00")
    hour = pd.Timedelta(hours=1)
    master = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3, 3],
        "study_id": [11, 12, 13, 21, 22, 31, 32],
        "hadm_id": [100, 100, 100, 200, 200, 300, 301],
        "ecg_time": [t, t + hour, t + 10 * hour, t, t + 2 * hour, t, t + hour],
    })
    medications = pd.DataFrame({
        "subject_id": [2], "medication_class": ["beta_blocker"],
        "event_time": [t - 5 * hour], "source_table": ["emar"],
    })
    master.to_parquet(tmp_path / "m.parquet")
    medications.to_parquet(tmp_path / "med.parquet")
    with duckdb.connect() as connection:
        rows = connection.execute(_untreated_control_pair_query(
            (tmp_path / "m.parquet").as_posix(), (tmp_path / "med.parquet").as_posix())).df()
    # 11->12 kept; 12->13 too far apart; subject 2 treated within the washout;
    # subject 3 changes admission.
    assert rows[["study_id_before", "study_id_after"]].values.tolist() == [[11, 12]]
    assert set(rows["medication_class"]) == {"untreated_control"}


def test_x9_reference_ranges_name_their_comparability_and_exclude_hypertension():
    from vcg_biomarker.mimic.mappings import classify_icd
    from vcg_biomarker.mimic.reference_ranges import (
        EXCLUDING_DIAGNOSIS_COLUMNS,
        reference_comparability,
    )

    assert classify_icd("I10", 10) == "essential_hypertension"
    assert classify_icd("4019", 9) == "essential_hypertension"
    assert classify_icd("E119", 10) == "diabetes_mellitus"
    assert classify_icd("25000", 9) == "diabetes_mellitus"
    assert classify_icd("I110", 10) == "hypertensive_heart_disease"
    for column in ("dx_essential_hypertension", "dx_diabetes_mellitus",
                   "dx_hypertensive_heart_disease"):
        assert column in EXCLUDING_DIAGNOSIS_COLUMNS
    assert "ludb_bias" in reference_comparability("qtc_fridericia_ms")["comparability_note"]
    assert "kors" in reference_comparability("svg_magnitude")["comparability_note"]
    assert reference_comparability("p_duration_ms")["literature_comparable"] is False


def test_e11b_process_screen_chunks_equal_the_full_screen(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    from vcg_biomarker.mimic import phenome
    from vcg_biomarker.mimic.grant_parallel import PARALLEL_ENV, grant_parallel_map

    rng = np.random.default_rng(12)
    n = 400
    subjects = np.arange(n)
    features = pd.DataFrame({
        "subject_id": subjects, "study_id": subjects * 10, "split": "train",
        "sex": np.where(subjects % 2, "M", "F"), "is_control_index": True,
        **{f"m{k}": rng.normal(size=n) for k in range(3)},
    })
    codes = ["CV_401", "CV_401.1", "CV_401.12", "GU_627", "EM_250"]
    rows = [(int(s), int(s) * 10, "train", rng.choice(codes)) for s in rng.choice(subjects, 160)]
    current = pd.DataFrame(rows, columns=["subject_id", "study_id", "split", "phenotype_id"]).drop_duplicates()
    definitions = pd.DataFrame({
        "phenotype_id": codes, "display_name": codes, "body_system": [c[:2] for c in codes],
        "male_only": ["FALSE", "FALSE", "FALSE", "TRUE", "FALSE"], "female_only": "FALSE",
    }).set_index("phenotype_id")
    markers = ("m0", "m1", "m2")
    prepared = phenome._prepare_screen_block(features, markers)
    full = pd.DataFrame(phenome._screen_split(
        features, current, pd.DataFrame(), phenotype_ids=tuple(codes), feature_ids=markers,
        panel_map=dict.fromkeys(markers, "p"), definition_lookup=definitions,
        min_inference=1, min_formal=1, progress=None, prepared_block=prepared))
    chunks = [("CV_401", "CV_401.1"), ("CV_401.12", "GU_627"), ("EM_250",)]
    known = tuple(codes)
    monkeypatch.setenv(PARALLEL_ENV, "1")
    outcomes = grant_parallel_map(phenome._screen_chunk_task, [
        (tmp_path, index, "hash", chunk, markers,
         phenome._current_rows_for_chunk(current, chunk, known),
         dict.fromkeys(markers, "p"), definitions, 1, 1,
         phenome._prepared_block_for_rows(prepared, phenome._current_rows_for_chunk(current, chunk, known)))
        for index, chunk in enumerate(chunks)])
    parts = pd.concat([pd.read_parquet(path) for path, _ in outcomes], ignore_index=True)
    key = ["phenotype_id", "feature_id"]
    left = full.sort_values(key).reset_index(drop=True)
    right = parts.sort_values(key).reset_index(drop=True)[left.columns]
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)


def test_e11b_process_hit_replication_equals_serial(monkeypatch):
    import math

    import numpy as np
    import pandas as pd

    from vcg_biomarker.mimic import phenome
    from vcg_biomarker.mimic.grant_parallel import PARALLEL_ENV

    rng = np.random.default_rng(21)
    n = 600
    subjects = np.arange(n)
    features = pd.DataFrame({
        "subject_id": subjects, "study_id": subjects * 10,
        "split": np.where(subjects % 2, "train", "validation"),
        "sex": np.where(subjects % 3, "M", "F"), "is_control_index": True,
        **{f"m{k}": rng.normal(size=n) for k in range(2)},
    })
    cases = rng.choice(subjects, 200, replace=False)
    current = pd.DataFrame({
        "subject_id": cases, "study_id": cases * 10,
        "split": np.where(cases % 2, "train", "validation"),
        "phenotype_id": np.where(cases % 4 == 0, "CV_401", "EM_250"),
    })
    definitions = pd.DataFrame({"phenotype_id": ["CV_401", "EM_250"],
                                "display_name": ["a", "b"], "body_system": ["CV", "EM"],
                                "male_only": "FALSE", "female_only": "FALSE"}).set_index("phenotype_id")
    hits = pd.DataFrame({
        "phenotype_id": ["CV_401", "EM_250", "CV_401"], "feature_id": ["m0", "m1", "m1"],
        "body_system": ["CV", "EM", "CV"], "panel_id": "p", "q_value": 0.01,
        "aligned_auc": 0.6, "direction": 1,
    })

    def run(setting):
        monkeypatch.setenv(PARALLEL_ENV, setting)
        return phenome._replicate_hits(
            hits, features, current, pd.DataFrame(), definition_lookup=definitions,
            min_inference=5, min_formal=10, bootstrap_iterations=50, random_seed=4,
            worker_count=1, progress=None)

    serial, parallel = run("0"), run("1")
    assert len(serial) == len(parallel) > 0

    def same(left, right):
        if isinstance(left, float) and isinstance(right, float):
            return (math.isnan(left) and math.isnan(right)) or left == right
        return left == right

    for left, right in zip(serial, parallel):
        assert left.keys() == right.keys()
        assert all(same(left[key], right[key]) for key in left), left
