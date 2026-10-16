"""Data-driven axes: screening what the database measured, not what was listed.

The properties worth pinning are the ones that decide whether a hit means
anything: the two target kinds need different comparison groups, the rank
shortcut must give the project's own statistic rather than a near-miss, each
axis must carry its own multiplicity, and the frequency floor must be fixed
before the screen rather than adjustable after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.broad_axis_stage import (
    MINIMUM_CASES,
    PRESPECIFIED_FLOORS,
    TRAIN_FDR_ALPHA,
    VALIDATION_FDR_ALPHA,
    _average_ranks,
    _adjustment_common_select,
    _adjustment_marker_projection,
    _rank_auc,
    _screen_binary,
    _screen_continuous,
    _spearman,
    _tie_term,
    available_markers,
    axis_sources,
    screening_marker_ids,
)
from vcg_biomarker.mimic.broad_event_screen import (
    AXIS_ICU_MEASUREMENT,
    AXIS_CCSR,
    AXIS_LABORATORY,
    AXIS_MICROBIOLOGY,
    AXIS_PROCEDURE,
    BROAD_AXES,
    DEFAULT_POLICIES,
    assert_policies_prespecified,
    binary_axis_values_sql,
    continuous_axis_values_sql,
    icu_measurement_target_counts_sql,
    policy_for,
)
from vcg_biomarker.mimic.statistics import (
    fast_mann_whitney_auc,
    prepare_sorted_control_distribution,
)


# ------------------------------------------------------------ the statistic


@pytest.mark.parametrize(
    ("n_cases", "n_controls", "rounded"),
    [(80, 4000, False), (150, 20000, False), (60, 3000, True), (35, 900, True)],
)
def test_the_rank_shortcut_is_the_projects_own_statistic(
    n_cases: int, n_controls: int, rounded: bool
) -> None:
    """Not "close enough": the same number, ties and continuity included.

    One sort per marker replaces one sort per (target, marker). At 671 binary
    targets and 1,713 markers that is 3,426 sorts instead of 1.15 million, so
    the shortcut is worth having -- but only if it does not quietly become a
    second, slightly different statistic alongside the phenotype screen's.
    """

    generator = np.random.default_rng(7)
    values = generator.standard_normal(n_cases + n_controls)
    if rounded:
        values = np.round(values, 1)  # force ties
    values[:n_cases] += 0.4
    mask = np.zeros(len(values), dtype=bool)
    mask[:n_cases] = True

    auc, p_value = _rank_auc(
        _average_ranks(values),
        case_mask=mask,
        n_cases=n_cases,
        n_controls=n_controls,
        tie_term=_tie_term(values),
    )
    reference_auc, reference_p = fast_mann_whitney_auc(
        values[mask], prepare_sorted_control_distribution(values[~mask])
    )
    assert auc == pytest.approx(reference_auc, abs=1e-12)
    assert p_value == pytest.approx(reference_p, rel=1e-9)


def test_midranks_handle_ties_the_way_the_statistic_requires() -> None:
    ranks = _average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(ranks) == [1.0, 2.5, 2.5, 4.0]
    assert _tie_term(np.array([10.0, 20.0, 20.0, 30.0])) == pytest.approx(6.0)
    assert _tie_term(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_rank_auc_refuses_an_empty_group() -> None:
    ranks = _average_ranks(np.arange(10.0))
    mask = np.zeros(10, dtype=bool)
    auc, p_value = _rank_auc(
        ranks, case_mask=mask, n_cases=0, n_controls=10, tie_term=0.0
    )
    assert np.isnan(auc) and np.isnan(p_value)


def test_spearman_finds_a_monotone_relation_and_ignores_noise() -> None:
    generator = np.random.default_rng(3)
    target = generator.standard_normal(500)
    rho, p_value = _spearman(2.0 * target + 0.1 * generator.standard_normal(500), target)
    assert rho > 0.9 and p_value < 1e-20
    rho_null, p_null = _spearman(generator.standard_normal(500), target)
    assert abs(rho_null) < 0.2 and p_null > 0.01
    # Too few points to say anything.
    assert np.isnan(_spearman(np.arange(5.0), np.arange(5.0))[1])


def test_adjusted_marker_block_projection_is_valid_and_collision_proof() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    feature = pd.DataFrame(
        {
            "study_id": [11],
            "signal_quality_score": [0.9],
            "rhythm_context": ["sinus"],
            "qrs_duration_ms": [100.0],
            "marker_a": [1.25],
            'odd"marker': [2.5],
        }
    )
    master = pd.DataFrame(
        {
            "study_id": [11],
            "subject_id": [1],
            "ecg_time": pd.to_datetime(["2026-01-01"]),
            "split": ["train"],
            "age": [60.0],
            "sex": ["M"],
            "machine_heart_rate_bpm": [70.0],
            "triage_heart_rate": [71.0],
            "icustay_id": [pd.NA],
            "ed_stay_id": [pd.NA],
            "hadm_id": [5],
            "machine_qrs_duration_ms": [101.0],
        }
    )
    connection.register("f", feature)
    connection.register("m", master)
    projection, aliases = _adjustment_marker_projection(
        ("marker_a", 'odd"marker')
    )
    result = connection.execute(
        "SELECT "
        + _adjustment_common_select()
        + ", "
        + projection
        + " FROM f JOIN m USING (study_id)"
    ).fetchdf()

    assert len(set(aliases.values())) == 2
    assert result[aliases["marker_a"]].iloc[0] == pytest.approx(1.25)
    assert result[aliases['odd"marker']].iloc[0] == pytest.approx(2.5)
    assert result["care_setting"].iloc[0] == "inpatient"


# ------------------------------------------------------ comparison groups


def _cohort(n: int = 800, seed: int = 5) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "study_id": np.arange(n),
            "split": ["train"] * (n // 2) + ["validation"] * (n - n // 2),
            "marker_signal": generator.standard_normal(n),
            "marker_noise": generator.standard_normal(n),
        }
    )


def test_a_binary_target_compares_against_the_rest_of_the_split() -> None:
    """The bug this pins: emitting only the ECGs that carry the entity.

    The value table lists presence, so every value in it is 1.0. Screening
    that directly leaves a case group and no controls -- the whole axis then
    produces zero tests while reporting success.
    """

    cohort = _cohort()
    present = set(range(0, 120))
    cohort.loc[cohort["study_id"].isin(present), "marker_signal"] += 1.2
    presence = pd.DataFrame(
        {"study_id": sorted(present), "target_id": ["ORGANISM_A"] * len(present)}
    )
    records = _screen_binary(
        cohort,
        presence,
        AXIS_MICROBIOLOGY,
        policy_for(AXIS_MICROBIOLOGY),
        {"ORGANISM_A": "Organism A"},
        ["marker_signal", "marker_noise"],
    )
    assert records, "a binary target must produce tests"
    frame = pd.DataFrame(records)
    # Cases and controls both present, and controls are the rest of the split.
    assert (frame["n_positive"] > 0).all()
    assert (frame["n_negative"] > frame["n_positive"]).all()
    signal = frame.loc[frame["feature_id"] == "marker_signal"]
    noise = frame.loc[frame["feature_id"] == "marker_noise"]
    assert (signal["effect"] > 0.7).all()
    assert (noise["p_value"] > 0.01).any()


def test_a_target_carried_by_too_few_ecgs_is_not_tested() -> None:
    cohort = _cohort()
    presence = pd.DataFrame(
        {
            "study_id": list(range(MINIMUM_CASES - 1)),
            "target_id": ["RARE"] * (MINIMUM_CASES - 1),
        }
    )
    assert (
        _screen_binary(
            cohort,
            presence,
            AXIS_MICROBIOLOGY,
            policy_for(AXIS_MICROBIOLOGY),
            {"RARE": "Rare"},
            ["marker_signal"],
        )
        == []
    )


def test_a_continuous_target_uses_only_the_ecgs_that_were_measured() -> None:
    """ECGs without a measurement are not controls; they are uninformative."""

    generator = np.random.default_rng(11)
    n = 400
    value = generator.standard_normal(n)
    frame = pd.DataFrame(
        {
            "target_id": ["220074"] * n,
            "split": ["train"] * (n // 2) + ["validation"] * (n // 2),
            "value": value,
            "marker_signal": 1.5 * value + 0.3 * generator.standard_normal(n),
            "marker_noise": generator.standard_normal(n),
        }
    )
    records = _screen_continuous(
        frame,
        AXIS_ICU_MEASUREMENT,
        policy_for(AXIS_ICU_MEASUREMENT),
        {"220074": "Central Venous Pressure"},
        ["marker_signal", "marker_noise"],
    )
    result = pd.DataFrame(records)
    assert set(result["evaluation_split"]) == {"train", "validation"}
    signal = result.loc[result["feature_id"] == "marker_signal"]
    assert (signal["effect"] > 0.8).all()
    assert (signal["p_value"] < 1e-10).all()
    assert result["target_label"].eq("Central Venous Pressure").all()


def test_binary_screen_counts_patients_not_repeated_ecgs() -> None:
    rng = np.random.default_rng(31)
    n_subjects = 800
    subject_id = np.repeat(np.arange(n_subjects), 2)
    split = np.where(subject_id < n_subjects // 2, "train", "validation")
    cohort = pd.DataFrame(
        {
            "study_id": np.arange(2 * n_subjects),
            "subject_id": subject_id,
            "ecg_time": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(np.tile([0, 1], n_subjects), unit="D"),
            "split": split,
            "marker": rng.normal(size=2 * n_subjects),
        }
    )
    case_subjects = set(range(80)) | set(range(400, 480))
    first_studies = cohort.loc[
        cohort["subject_id"].isin(case_subjects)
        & cohort["study_id"].mod(2).eq(0),
        "study_id",
    ]
    cohort.loc[cohort["subject_id"].isin(case_subjects), "marker"] += 1.0
    presence = pd.DataFrame(
        {"study_id": first_studies, "target_id": "EVENT"}
    )
    result = pd.DataFrame(
        _screen_binary(
            cohort,
            presence,
            AXIS_MICROBIOLOGY,
            policy_for(AXIS_MICROBIOLOGY),
            {"EVENT": "Event"},
            ["marker"],
        )
    )
    assert result["n_positive"].tolist() == [80, 80]
    assert result["n_negative"].tolist() == [320, 320]


def test_continuous_screen_uses_one_index_ecg_per_patient_and_endpoint() -> None:
    rng = np.random.default_rng(37)
    n_subjects = 400
    subject_id = np.repeat(np.arange(n_subjects), 2)
    target = rng.normal(size=2 * n_subjects)
    frame = pd.DataFrame(
        {
            "study_id": np.arange(2 * n_subjects),
            "subject_id": subject_id,
            "ecg_time": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(np.tile([0, 1], n_subjects), unit="D"),
            "target_id": "LAB",
            "split": np.where(subject_id < n_subjects // 2, "train", "validation"),
            "value": target,
            "marker": target + rng.normal(0.0, 0.1, 2 * n_subjects),
        }
    )
    result = pd.DataFrame(
        _screen_continuous(
            frame,
            AXIS_ICU_MEASUREMENT,
            policy_for(AXIS_ICU_MEASUREMENT),
            {"LAB": "Laboratory"},
            ["marker"],
        )
    )
    assert result["n_observations"].tolist() == [200, 200]


# --------------------------------------------------------------- policy


def test_the_frequency_floor_is_prespecified_and_covers_every_axis() -> None:
    assert set(PRESPECIFIED_FLOORS) == set(BROAD_AXES)
    assert_policies_prespecified(DEFAULT_POLICIES, expected=dict(PRESPECIFIED_FLOORS))
    moved = dict(PRESPECIFIED_FLOORS)
    moved[AXIS_LABORATORY] += 50
    with pytest.raises(AssertionError, match="preregistered"):
        assert_policies_prespecified(DEFAULT_POLICIES, expected=moved)


def test_the_icu_axis_is_a_measurement_axis_not_an_administrative_one() -> None:
    """Why it is the scientifically interesting one.

    A pulmonary artery pressure is obtained with a catheter. Comparing an ECG
    marker against it asks whether the marker could replace the catheter,
    which is a different question from whether it correlates with a billing
    code -- and the claim ceiling has to say so.
    """

    policy = policy_for(AXIS_ICU_MEASUREMENT)
    assert policy.target_kind == "continuous"
    assert policy.reference_tier.startswith("A_")
    assert policy.claim_ceiling != "association_only_not_accuracy"
    for axis in (AXIS_PROCEDURE, AXIS_MICROBIOLOGY):
        assert policy_for(axis).reference_tier.startswith("C_")
        assert policy_for(axis).target_kind == "binary"


def test_replication_is_stricter_on_train_than_on_validation() -> None:
    # Screening alpha must be the tight one: the train family is the large,
    # entirely hypothesis-free one.
    assert TRAIN_FDR_ALPHA < VALIDATION_FDR_ALPHA


# ------------------------------------------------------------------- SQL


def test_every_axis_query_compiles_and_returns_the_expected_shape() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE ce AS SELECT * FROM (VALUES "
        "(1, 220074, 12.0, TIMESTAMP '2110-01-01 10:00:00'), "
        "(2, 220074, 14.0, TIMESTAMP '2110-01-01 10:30:00'), "
        "(3, 220074, NULL, TIMESTAMP '2110-01-01 11:00:00')) "
        "AS t(subject_id, itemid, valuenum, charttime)"
    )
    connection.execute(
        "CREATE TABLE d AS SELECT * FROM (VALUES "
        "(220074, 'Central Venous Pressure')) AS t(itemid, label)"
    )
    counts = connection.execute(
        icu_measurement_target_counts_sql("ce", "d")
    ).fetchdf()
    assert set(counts.columns) == {
        "entity_id",
        "entity_label",
        "n_subjects",
        "n_observations",
    }
    # Rows without a numeric value do not count as measurements.
    assert int(counts.iloc[0]["n_observations"]) == 2

    connection.execute(
        "CREATE TABLE master AS SELECT * FROM (VALUES "
        "(1, 100, 10, 'train', TIMESTAMP '2110-01-01 10:05:00'), "
        "(2, 200, 10, 'validation', TIMESTAMP '2110-01-01 10:20:00')) "
        "AS t(subject_id, study_id, hadm_id, split, ecg_time)"
    )
    values = connection.execute(
        continuous_axis_values_sql(
            "ce", "master", axis="icu_measurement", entity_ids=["220074"]
        )
    ).fetchdf()
    assert set(values["target_id"]) == {"220074"}
    assert set(values["split"]) == {"train", "validation"}
    assert set(values["temporal_relation"]) == {"before_ecg", "after_ecg"}
    # Nearest in time, not an average over the window.
    assert float(values.loc[values["study_id"] == 100, "value"].iloc[0]) == 12.0

    connection.execute(
        "CREATE TABLE proc AS SELECT * FROM (VALUES "
        "(1, 10, '0270'), (2, 99, '0270')) AS t(subject_id, hadm_id, icd_code)"
    )
    presence = connection.execute(
        binary_axis_values_sql(
            "proc",
            "master",
            axis="procedure_code",
            entity_column="icd_code",
            entity_ids=["0270"],
        )
    ).fetchdf()
    # Scoped to the ECG's own admission: subject 2 coded under hadm 99 does
    # not count for an ECG recorded during hadm 10.
    assert list(presence["study_id"]) == [100]


def test_a_query_without_targets_is_refused_rather_than_built() -> None:
    with pytest.raises(ValueError, match="At least one target"):
        continuous_axis_values_sql("e", "m", axis="a", entity_ids=[])
    with pytest.raises(ValueError, match="At least one target"):
        binary_axis_values_sql(
            "e", "m", axis="a", entity_column="c", entity_ids=[]
        )


def test_entity_ids_with_quotes_cannot_break_out_of_the_query() -> None:
    sql = binary_axis_values_sql(
        "e",
        "m",
        axis="a",
        entity_column="organism",
        entity_ids=["O'BRIEN SPECIES"],
    )
    assert "'O''BRIEN SPECIES'" in sql


# ----------------------------------------------------------- robustness


def test_a_panel_newer_than_the_feature_files_does_not_kill_the_axis() -> None:
    """Markers added after an extraction exist in the panel, not in the files."""

    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE f AS SELECT * FROM (VALUES (1, 0.5)) AS t(study_id, marker_a)"
    )
    connection.execute("COPY f TO 'test_features.parquet' (FORMAT PARQUET)")
    try:
        resolved = available_markers(
            connection, "test_features.parquet", ["marker_a", "marker_added_later"]
        )
        assert resolved == ["marker_a"]
    finally:
        import os

        os.remove("test_features.parquet")


def test_an_unreadable_feature_set_yields_no_markers_instead_of_raising() -> None:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    assert available_markers(connection, "does_not_exist.parquet", ["a"]) == []


def test_screenable_markers_exclude_aliases_and_classical_baselines() -> None:
    markers = screening_marker_ids(("primary_core", "sentinel"))
    assert markers, "the core panels must contribute screenable markers"
    assert not any(name.startswith("classical_") for name in markers)
    assert len(set(markers)) == len(markers)
    with pytest.raises(KeyError, match="Unknown feature panel"):
        screening_marker_ids(("not_a_panel",))


def test_a_missing_source_is_reported_rather_than_raised(tmp_path) -> None:
    class _Config:
        mimic_iv_root = tmp_path / "absent"
        work_root = tmp_path

    from vcg_biomarker.mimic.broad_event_screen import AXIS_DRG

    sources = axis_sources(_Config())
    assert set(sources) == {
        AXIS_CCSR,
        AXIS_LABORATORY,
        AXIS_ICU_MEASUREMENT,
        AXIS_PROCEDURE,
        AXIS_MICROBIOLOGY,
        AXIS_DRG,
    }
    assert not any(source["available"] for source in sources.values())
    assert sources[AXIS_CCSR]["reason"]


def test_the_ccsr_axis_has_a_source_and_its_queries_run(tmp_path, monkeypatch) -> None:
    """Policy, counting query and source entry belong together.

    The axis had the first two and not the third, so every sweep skipped it
    as unavailable and then refused to complete; the production-path preview
    was the first run to get that far.
    """
    import duckdb
    from types import SimpleNamespace

    class _Config:
        mimic_iv_root = tmp_path / "mimic"
        work_root = tmp_path

    standardized = tmp_path / "standardized"
    standardized.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [10, 10, 20, 30],
            "icd_code": ["I50", "E11", "I50", "I50"],
            "icd_version": [10, 10, 10, 9],
        }
    ).to_parquet(standardized / "diagnoses_icd.parquet", index=False)
    mapping = tmp_path / "ccsr_mapping.parquet"
    pd.DataFrame(
        {
            "icd_code": ["I50", "E11"],
            "ccsr_category": ["CIR019", "END002"],
            "ccsr_description": ["Heart failure", "Diabetes"],
        }
    ).to_parquet(mapping, index=False)
    monkeypatch.setattr(
        "vcg_biomarker.mimic.phenotypes.prepare_phenotype_resources",
        lambda config: SimpleNamespace(ccsr_mapping_path=mapping),
    )

    source = axis_sources(_Config())[AXIS_CCSR]
    assert source["available"] is True
    assert source["kind"] == "binary" and source["entity_column"] == "ccsr_category"

    connection = duckdb.connect()
    counts = connection.execute(source["counts_sql"]()).fetchdf().set_index("entity_id")
    # The ICD-9 row of subject 3 is not mapped; two subjects carry CIR019.
    assert counts.loc["CIR019", "n_subjects"] == 2
    assert counts.loc["END002", "entity_label"] == "Diabetes"
    events = connection.execute(
        f"SELECT * FROM {source['relation']} ORDER BY subject_id, ccsr_category"
    ).fetchdf()
    assert events["ccsr_category"].tolist() == ["CIR019", "END002", "CIR019"]


def test_axis_sources_prefer_standardized_parquet_without_raw_rescan(
    tmp_path,
) -> None:
    class _Config:
        mimic_iv_root = tmp_path / "mimic"
        work_root = tmp_path

    standardized = tmp_path / "standardized"
    standardized.mkdir()
    for name in (
        "labevents.parquet",
        "d_labitems.parquet",
        "chartevents.parquet",
        "d_items.parquet",
        "procedures_icd.parquet",
        "microbiologyevents.parquet",
    ):
        (standardized / name).touch()

    sources = axis_sources(_Config())

    for axis in (
        AXIS_LABORATORY,
        AXIS_ICU_MEASUREMENT,
        AXIS_PROCEDURE,
        AXIS_MICROBIOLOGY,
    ):
        source = sources[axis]
        assert source["available"] is True
        assert "read_parquet" in str(source["relation"])
        assert "read_csv" not in str(source["relation"])
        assert "read_parquet" in str(source["counts_sql"]())
        assert "read_csv" not in str(source["counts_sql"]())


# ------------------------------------------------- readable targets


def test_a_procedure_target_carries_its_title_not_only_its_code() -> None:
    """A hit reading "0270" cannot be adjudicated by anyone.

    Every other axis carries a label -- laboratory items from d_labitems, ICU
    items from d_items, organisms are text already. Procedures were the one
    axis whose label was the code itself.
    """

    duckdb = pytest.importorskip("duckdb")
    from vcg_biomarker.mimic.broad_event_screen import procedure_target_counts_sql

    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE proc AS SELECT * FROM (VALUES "
        "(1, '0270', 9), (2, '0270', 9), (3, '9999', 9)) "
        "AS t(subject_id, icd_code, icd_version)"
    )
    connection.execute(
        "CREATE TABLE d AS SELECT * FROM (VALUES "
        "('0270', 9, 'Craniotomy')) AS t(icd_code, icd_version, long_title)"
    )
    labelled = connection.execute(
        procedure_target_counts_sql("proc", "d")
    ).fetchdf().set_index("entity_id")
    assert labelled.loc["0270", "entity_label"] == "Craniotomy"
    # A code the dictionary does not know still gets a usable label.
    assert labelled.loc["9999", "entity_label"] == "9999"

    # Without a dictionary the query still works, just without titles.
    bare = connection.execute(
        procedure_target_counts_sql("proc")
    ).fetchdf().set_index("entity_id")
    assert bare.loc["0270", "entity_label"] == "0270"


def test_the_drg_axis_groups_by_treatment_effort_not_by_diagnosis() -> None:
    """A third taxonomy, deliberately unlike the other two.

    PheCodeX and CCSR both group by diagnosis. DRG groups an admission by what
    it took to treat, which merges constellations a diagnosis axis separates.
    """

    duckdb = pytest.importorskip("duckdb")
    from vcg_biomarker.mimic.broad_event_screen import (
        AXIS_DRG,
        drg_target_counts_sql,
        policy_for,
    )

    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE drg AS SELECT * FROM (VALUES "
        "(1, 10, 291, 'HEART FAILURE'), (2, 20, 291, 'HEART FAILURE'), "
        "(3, 30, 871, 'SEPTICEMIA')) "
        "AS t(subject_id, hadm_id, drg_code, description)"
    )
    counts = connection.execute(
        drg_target_counts_sql("drg")
    ).fetchdf().set_index("entity_id")
    assert counts.loc["291", "entity_label"] == "HEART FAILURE"
    assert int(counts.loc["291", "n_subjects"]) == 2

    policy = policy_for(AXIS_DRG)
    assert policy.target_kind == "binary"
    # An administrative grouping, so the same claim ceiling as diagnoses.
    assert policy.reference_tier.startswith("C_")
    assert AXIS_DRG in PRESPECIFIED_FLOORS


# --------------------------------------------- temporal interpretability


def test_a_dated_target_is_split_by_whether_it_preceded_the_ecg() -> None:
    """The finding that decides what a hit means.

    Measured on the real data, 38.9 % of ECG/procedure pairs inside one
    admission have the procedure after the ECG and 43.0 % before it. Pooled,
    "marker associates with pacemaker implantation" could mean the ECG
    predicted the implant or that it shows a paced heart -- opposite claims
    from one number.
    """

    duckdb = pytest.importorskip("duckdb")
    from vcg_biomarker.mimic.broad_event_screen import (
        RELATION_AFTER,
        RELATION_BEFORE,
        RELATION_SAME_DAY,
        RELATION_UNDATED,
    )

    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE master AS SELECT * FROM (VALUES "
        "(1, 100, 10, 'train', TIMESTAMP '2110-06-15 12:00:00')) "
        "AS t(subject_id, study_id, hadm_id, split, ecg_time)"
    )
    connection.execute(
        "CREATE TABLE proc AS SELECT * FROM (VALUES "
        "(1, 10, '0270', DATE '2110-06-10'), "
        "(1, 10, '0271', DATE '2110-06-20'), "
        "(1, 10, '0272', DATE '2110-06-15')) "
        "AS t(subject_id, hadm_id, icd_code, chartdate)"
    )
    rows = connection.execute(
        binary_axis_values_sql(
            "proc",
            "master",
            axis=AXIS_PROCEDURE,
            entity_column="icd_code",
            entity_ids=["0270", "0271", "0272"],
            date_column="chartdate",
        )
    ).fetchdf().set_index("target_id")
    assert rows.loc["0270", "temporal_relation"] == RELATION_BEFORE
    assert rows.loc["0271", "temporal_relation"] == RELATION_AFTER
    assert rows.loc["0272", "temporal_relation"] == RELATION_SAME_DAY

    # An axis without a date says so rather than guessing.
    undated = connection.execute(
        binary_axis_values_sql(
            "proc",
            "master",
            axis=AXIS_PROCEDURE,
            entity_column="icd_code",
            entity_ids=["0270"],
        )
    ).fetchdf()
    assert set(undated["temporal_relation"]) == {RELATION_UNDATED}


def test_the_drg_axis_declares_itself_an_end_of_stay_label() -> None:
    """A DRG is assigned when the admission is billed, never before."""

    from vcg_biomarker.mimic.broad_event_screen import AXIS_DRG, policy_for

    assert policy_for(AXIS_DRG).temporal_role == "assigned_at_discharge_end_of_stay"


def test_before_and_after_are_screened_as_separate_targets() -> None:
    """Same entity, opposite timing, separate case groups."""

    from vcg_biomarker.mimic.broad_event_screen import (
        RELATION_AFTER,
        RELATION_BEFORE,
    )

    cohort = _cohort(n=1200)
    before = set(range(0, 100))
    after = set(range(100, 200))
    cohort.loc[cohort["study_id"].isin(before), "marker_signal"] += 1.5
    presence = pd.DataFrame(
        {
            "study_id": sorted(before) + sorted(after),
            "target_id": ["0270"] * (len(before) + len(after)),
            "temporal_relation": [RELATION_BEFORE] * len(before)
            + [RELATION_AFTER] * len(after),
        }
    )
    records = _screen_binary(
        cohort,
        presence,
        AXIS_PROCEDURE,
        policy_for(AXIS_PROCEDURE),
        {"0270": "Craniotomy"},
        ["marker_signal"],
    )
    frame = pd.DataFrame(records)
    assert set(frame["temporal_relation"]) == {RELATION_BEFORE, RELATION_AFTER}
    # Only the group that actually carries the shift shows the effect.
    shifted = frame.loc[frame["temporal_relation"] == RELATION_BEFORE, "effect"]
    unshifted = frame.loc[frame["temporal_relation"] == RELATION_AFTER, "effect"]
    assert (shifted > 0.8).all()
    assert (unshifted.abs() - 0.5).abs().max() < 0.15
    # Every record states how it may be read.
    assert frame["temporal_role"].notna().all()


def test_the_sealed_split_is_proven_absent_not_merely_claimed() -> None:
    from vcg_biomarker.mimic.broad_axis_stage import assert_no_sealed_rows

    development = pd.DataFrame({"split": ["train", "validation"]})
    assert_no_sealed_rows(development, where="test")
    assert_no_sealed_rows(pd.DataFrame(), where="test")
    with pytest.raises(PermissionError, match="locked_test"):
        assert_no_sealed_rows(
            pd.DataFrame({"split": ["train", "locked_test"]}), where="test"
        )
