"""Class sizes pooled over both splits are not a power statement.

The gate asked whether each class had at least 30 and then 100 subjects, over
train and validation together, and never looked at prevalence.  Two things got
through:

* ``significant_valvular_disease`` ran at a prevalence of 0.969 -- 9,400
  positive subjects against 301 negative ones -- and was still reported as
  ``formal_analysis_eligible``, because 301 clears the hundred-subject floor.
  Split in two, the validation half holds roughly sixty negatives.
* the screen replicates on validation, so the binding constraint is the smaller
  half, which the pooled count cannot see.

Replaying the corrected prevalence rule over the last run's 63 endpoints flags
exactly three -- regional wall motion abnormality (0.996), significant valvular
disease (0.969) and troponin positive (0.922) -- and leaves the genuinely rare
ones alone.  That asymmetry is deliberate and is what the last two tests pin:
a rare disease is not a broken endpoint.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.labels import _label_summary_query


def _summarise(rows, tmp_path):
    """Run the production summary query over a small synthetic cohort."""

    frame = pd.DataFrame(
        rows,
        columns=["label_id", "subject_id", "split", "label_state"],
    )
    frame["eligibility_state"] = "eligible"
    frame["target_kind"] = "diagnostic"
    frame["panel"] = "primary"
    frame["source_tables"] = "test"
    frame["index_policy"] = "test"
    frame["window_before_hours"] = -6.0
    frame["window_after_hours"] = 6.0
    frame["rule_version"] = "1.0.0"
    long_path = tmp_path / "long.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    frame.to_parquet(long_path)
    frame.to_parquet(cohort_path)
    query = _label_summary_query(long_path, cohort_path)
    with duckdb.connect() as connection:
        return connection.execute(query).df().set_index("label_id")


def _cohort(label_id, *, positives, negatives, split_positives=None,
            split_negatives=None):
    """Subjects split evenly unless a per-split breakdown is given."""

    rows = []
    subject = 0
    if split_positives is None:
        split_positives = {"train": positives // 2, "validation": positives - positives // 2}
    if split_negatives is None:
        split_negatives = {"train": negatives // 2, "validation": negatives - negatives // 2}
    for split, count in split_positives.items():
        for _ in range(count):
            subject += 1
            rows.append((label_id, subject, split, "positive"))
    for split, count in split_negatives.items():
        for _ in range(count):
            subject += 1
            rows.append((label_id, subject, split, "negative"))
    return rows


def test_a_balanced_well_populated_endpoint_is_eligible(tmp_path):
    summary = _summarise(_cohort("good", positives=400, negatives=400), tmp_path)
    row = summary.loc["good"]
    assert row["power_status"] == "formal_analysis_eligible"
    assert row["prevalence_status"] == "prevalence_supports_inference"


def test_a_class_that_is_only_large_when_pooled_is_not_eligible(tmp_path):
    # 300 negatives in total clears the floor; 40 in validation cannot carry a
    # replication, which is what the screen actually asks of it.
    rows = _cohort(
        "pooled",
        positives=600,
        negatives=300,
        split_negatives={"train": 260, "validation": 40},
    )
    row = _summarise(rows, tmp_path).loc["pooled"]
    assert row["smallest_split_minority_subject_count"] == 40
    assert row["power_status"] == "underpowered_descriptive"


def test_the_valvular_disease_shape_is_no_longer_reported_as_eligible(tmp_path):
    # The real numbers: 9,400 positive subjects, 301 negative, scaled down by
    # ten to keep the fixture small but the shape identical.
    rows = _cohort("valvular", positives=940, negatives=30)
    row = _summarise(rows, tmp_path).loc["valvular"]
    assert row["prevalence_status"] == "high_prevalence_verification_conditioned"
    assert row["power_status"] != "formal_analysis_eligible"


def test_a_rare_disease_is_not_treated_as_a_broken_endpoint(tmp_path):
    # Severe aortic regurgitation: 238 positives against 14,019 negatives, a
    # prevalence of 0.017. Legitimate, and the gate must leave it alone.
    rows = _cohort("rare", positives=238, negatives=1402)
    row = _summarise(rows, tmp_path).loc["rare"]
    assert row["prevalence_status"] == "prevalence_supports_inference"
    assert row["power_status"] == "formal_analysis_eligible"


def test_a_rare_disease_with_too_few_cases_still_fails_on_count(tmp_path):
    rows = _cohort("very_rare", positives=40, negatives=4000)
    row = _summarise(rows, tmp_path).loc["very_rare"]
    assert row["prevalence_status"] == "prevalence_supports_inference"
    assert row["power_status"] == "underpowered_descriptive"


def test_an_endpoint_with_no_usable_class_is_named_not_assessable(tmp_path):
    rows = [("empty", 1, "train", "unavailable")]
    row = _summarise(rows, tmp_path).loc["empty"]
    assert row["prevalence_status"] == "not_assessable"
    assert row["power_status"] == "no_inference"


def test_label_build_session_bounds_spill_below_the_work_root(tmp_path):
    """The canonical label build spills; it must do so where the space is,
    without insertion-order state and with an allowance above the 16 GiB
    per-connection default that ended pilot A123 on 2026-09-19."""
    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.labels import LABEL_BUILD_SPILL_GIB, label_build_session_options

    config = MimicProjectConfig.discover(database_root=tmp_path / "db", work_root=tmp_path / "work")
    statements = label_build_session_options(config)
    assert statements[0] == "SET preserve_insertion_order=false"
    assert "tmp/labels_duckdb" in statements[1].replace("\\", "/")
    assert (tmp_path / "work" / "tmp" / "labels_duckdb").is_dir()
    assert statements[2] == f"SET max_temp_directory_size='{LABEL_BUILD_SPILL_GIB}GiB'"
    assert LABEL_BUILD_SPILL_GIB > 16
