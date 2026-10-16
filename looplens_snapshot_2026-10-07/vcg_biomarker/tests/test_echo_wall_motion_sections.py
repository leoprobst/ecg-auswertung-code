"""Section existence is documentation evidence, not proof of normal motion."""
from __future__ import annotations

import duckdb
import pandas as pd

from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.evidence import _build_echo_evidence

from vcg_biomarker.mimic.echo_segments import RWMA_CORE_SEGMENTS

SEGMENTS = tuple(f"wm_{segment}" for segment in (*RWMA_CORE_SEGMENTS, "apical_cap"))


def _rows(measurement_id, subject_id, results):
    rows = []
    for segment, result in zip(SEGMENTS, results):
        rows.append(
            {
                "subject_id": subject_id,
                "measurement_id": measurement_id,
                "measurement_datetime": pd.Timestamp("2020-01-02 08:00"),
                "test_type": "tte",
                "measurement": segment,
                "measurement_description": segment,
                "result_raw": result,
                "result_numeric": None,
                "unit": None,
                "parse_status": "text" if result is not None else "blank",
            }
        )
    rows.append(
        {
            "subject_id": subject_id,
            "measurement_id": measurement_id,
            "measurement_datetime": pd.Timestamp("2020-01-02 08:00"),
            "test_type": "tte",
            "measurement": "lvef",
            "measurement_description": "lvef",
            "result_raw": "55",
            "result_numeric": 55.0,
            "unit": "%",
            "parse_status": "numeric",
        }
    )
    return rows


def test_blank_wall_motion_rows_are_retained_for_coverage_only(tmp_path):
    config = MimicProjectConfig.discover(
        database_root=tmp_path / "db", work_root=tmp_path / "work"
    )
    config.ensure_directories()
    # Study 1: every segment blank; study 2: one hypokinetic segment, rest
    # blank; study 3: two explicit normals, rest blank.
    all_rows = pd.DataFrame(
        _rows(1, 10, [None] * 17)
        + _rows(2, 10, ["hypokinetic"] + [None] * 16)
        + _rows(3, 11, ["normal", "normal"] + [None] * 15)
    )
    all_rows.to_parquet(
        config.standardized_root / "echo_measurements_all.parquet", index=False
    )
    all_rows[all_rows.result_raw.notna()].to_parquet(
        config.standardized_root / "echo_measurements_long.parquet", index=False
    )
    pd.DataFrame(
        {
            "measurement_id": [1, 2, 3],
            "subject_id": [10, 10, 11],
            "echo_time": [pd.Timestamp("2020-01-02 08:00")] * 3,
            "test_type": ["tte"] * 3,
            "identity_conflict": [False] * 3,
        }
    ).to_parquet(config.standardized_root / "echo_study_index.parquet", index=False)
    pd.DataFrame(
        {
            "subject_id": [10, 11],
            "study_id": [100, 101],
            "ecg_time": [pd.Timestamp("2020-01-02 07:00")] * 2,
        }
    ).to_parquet(config.master_root / "ecg_master_index.parquet", index=False)

    connection = duckdb.connect()
    try:
        _build_echo_evidence(connection, config, None)
    finally:
        connection.close()
    wide = pd.read_parquet(
        config.evidence_root / "echo_measurements_wide.parquet"
    ).set_index("measurement_id")

    assert wide.loc[1, "rwma_reported_segments"] == 17
    assert wide.loc[1, "rwma_abnormal_segments"] == 0
    assert wide.loc[1, "rwma_normal_segments"] == 0
    assert wide.loc[2, "rwma_abnormal_segments"] == 1
    assert wide.loc[3, "rwma_normal_segments"] == 2
    # Only the documentation proxy may treat blank coverage this way.
    negative = (wide.rwma_reported_segments > 0) & (wide.rwma_abnormal_segments == 0)
    assert negative.loc[[1, 3]].all() and not negative.loc[2]


def test_real_label_rules_distinguish_clinical_negative_from_documentation():
    from vcg_biomarker.mimic.labels import _label_rules

    rows = pd.DataFrame({
        "rwma_reported_segments": [17] * 5,
        "rwma_abnormal_segments": [0, 1, 0, 0, 0],
        "rwma_normal_segments": [0, 0, 2, 17, 16],
        "rwma_not_visualized_segments": [0, 0, 0, 0, 1],
        "rwma_complete_normal_coverage": [False, False, False, True, False],
        "rwma_reference_conflict": [False] * 5,
        "has_tte_7d": [True] * 5,
        "has_tte_30d": [True] * 5,
    })
    rules = _label_rules()
    with duckdb.connect() as connection:
        connection.register("rows", rows)
        clinical = connection.execute(
            f"SELECT {rules['regional_wall_motion_abnormality'].state} FROM rows"
        ).fetchall()
        proxy = connection.execute(
            f"SELECT {rules['regional_wall_motion_documentation_proxy'].state} FROM rows"
        ).fetchall()
    assert [item[0] for item in clinical] == [
        "indeterminate", "positive", "indeterminate", "negative", "indeterminate"
    ]
    assert [item[0] for item in proxy] == [
        "negative", "positive", "negative", "negative", "negative"
    ]


def test_incomplete_negative_coverage_is_not_registered_as_diagnostic_truth():
    from vcg_biomarker.mimic.label_registry import default_label_definitions
    from vcg_biomarker.mimic.models import TargetKind

    definitions = {item.label_id: item for item in default_label_definitions()}
    for key in ("regional_wall_motion_documentation_proxy", "regional_wall_motion_abnormality_explicit"):
        assert definitions[key].target_kind == TargetKind.RETROSPECTIVE_PROXY


def test_distinct_anatomical_segments_and_rest_stress_separation(tmp_path):
    from vcg_biomarker.mimic.echo_segments import resting_wall_motion_coverage_sql

    rows = []
    for study in (1, 2, 3):
        for segment in RWMA_CORE_SEGMENTS:
            rows.append({"measurement_id": study, "measurement": f"wm_{segment}", "result_raw": "normal"})
    # Stress abnormality is not a resting abnormality; duplicated rest/base is one segment.
    rows.extend([
        {"measurement_id": 1, "measurement": "wm_stress_ant_base", "result_raw": "akinetic"},
        {"measurement_id": 1, "measurement": "wm_rest_ant_base", "result_raw": "normal"},
        {"measurement_id": 2, "measurement": "wm_rest_ant_base", "result_raw": "akinetic"},
    ])
    # Sixteen copies of one segment cannot certify sixteen anatomical regions.
    rows = [row for row in rows if row["measurement_id"] != 3] + [
        {"measurement_id": 3, "measurement": "wm_ant_base", "result_raw": "normal"}
    ] * 16
    path = tmp_path / "segments.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    with duckdb.connect() as connection:
        result = connection.execute(resting_wall_motion_coverage_sql(path.as_posix())).fetchdf().set_index("measurement_id")
    assert result.loc[1, "rwma_reported_segments"] == 16
    assert result.loc[1, "rwma_complete_normal_coverage"]
    assert result.loc[1, "rwma_abnormal_segments"] == 0
    assert result.loc[2, "rwma_reference_conflict"]
    assert not result.loc[2, "rwma_complete_normal_coverage"]
    assert result.loc[3, "rwma_reported_segments"] == 1
    assert not result.loc[3, "rwma_complete_normal_coverage"]


def test_documentation_proxy_cannot_be_upgraded_by_generic_echo_gates():
    from vcg_biomarker.mimic.result_contract import (
        CANDIDATE_GATES, CLINICAL_GATES, scientific_result_contract,
    )
    row = {f"{key}_passed": True for key in (*CANDIDATE_GATES, *CLINICAL_GATES)}
    row.update(endpoint_id="regional_wall_motion_documentation_proxy",
               reference_kind="timed_quantitative", analysis_scope="complete_snapshot")
    result = scientific_result_contract(row)
    assert not result.clinical_claim_allowed
    assert result.evidence_stage not in {"clinically_validated", "guideline_candidate"}
    assert result.claim_ceiling == "documentation_association_only"
