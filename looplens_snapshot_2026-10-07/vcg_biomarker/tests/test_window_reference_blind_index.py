from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from vcg_biomarker.mimic.labels import _reference_blind_window_index_query, _window_sensitivity_summary_query


def candidates():
    rows = []
    for subject in (1, 2):
        for study in (1, 2):
            for profile in ("strict", "standard"):
                available = study == 2 or profile == "standard"
                rows.append({"subject_id": subject, "study_id": subject*100+study,
                    "ecg_time": pd.Timestamp(f"2020-01-0{study}"), "label_id": "endpoint",
                    "window_profile": profile, "window_before_hours": -3 if profile == "strict" else -6,
                    "window_after_hours": 3 if profile == "strict" else 6, "split": "train",
                    "eligibility_state": "eligible" if available else "insufficient_evidence",
                    "label_state": "positive" if study == 2 else "negative" if available else "unavailable",
                    "time_delta_hours": 1. if available else 100., "confidence": "high" if available else "none"})
    return pd.DataFrame(rows)


def select(frame):
    with duckdb.connect() as connection:
        connection.register("fixture", frame)
        return connection.execute(_reference_blind_window_index_query("SELECT * FROM fixture")).fetchdf()


def test_missing_reference_never_promotes_later_record_in_any_window():
    result = select(candidates())
    assert result.groupby("subject_id").study_id.nunique().eq(1).all()
    assert set(result.study_id) == {101, 201}
    assert result.loc[result.window_profile.eq("strict"), "label_state"].eq("unavailable").all()
    assert result.loc[result.window_profile.eq("standard"), "label_state"].eq("negative").all()
    assert len(result) == 4


def test_all_profile_exclusions_and_ties_are_deterministic():
    frame = candidates()
    frame.loc[frame.study_id.eq(101), "eligibility_state"] = "excluded"
    frame.loc[frame.subject_id.eq(2), "ecg_time"] = pd.Timestamp("2020-01-01")
    expected = select(frame)
    shuffled = select(frame.sample(frac=1, random_state=7))
    assert set(expected.study_id) == set(shuffled.study_id) == {102, 201}


def test_summary_retains_unavailable_denominator_without_fake_prevalence(tmp_path):
    path = tmp_path / "windows.parquet"
    select(candidates()).to_parquet(path)
    with duckdb.connect() as connection:
        summary = connection.execute(_window_sensitivity_summary_query(path)).fetchdf().set_index("window_profile")
    assert summary.loc["strict", "subject_count"] == 2
    assert summary.loc["strict", "unavailable_subject_count"] == 2
    assert summary.loc["strict", "reference_available_fraction"] == 0
    assert pd.isna(summary.loc["strict", "prevalence"])
    assert pd.isna(summary.loc["strict", "median_absolute_delta_hours"])


def test_materialized_audit_distinguishes_old_reference_dependent_indices(tmp_path):
    from vcg_biomarker.mimic.window_index_audit import audit
    frame = candidates()
    # Reproduce the old eligibility-first strategy: earliest known reference per profile.
    old = frame.loc[frame.eligibility_state.eq("eligible")].sort_values("ecg_time").drop_duplicates(
        ["subject_id", "label_id", "window_profile"])
    old["window_rule_version"] = "synthetic-old"
    path = tmp_path / "old.parquet"; old.to_parquet(path)
    result = audit(path)
    assert result["pairs_with_index_drift"] == 2
    new = select(frame); new["window_rule_version"] = "synthetic-new"
    path = tmp_path / "new.parquet"; new.to_parquet(path)
    assert audit(path)["pairs_with_index_drift"] == 0


@pytest.mark.parametrize("locked", [False, True])
def test_window_consumers_do_not_use_ineligible_positive_labels(tmp_path, locked):
    from vcg_biomarker.mimic.validation import _window_sensitivity_records
    from vcg_biomarker.mimic.locking import _locked_window_sensitivity_records
    frame = select(candidates()).iloc[:1].copy()
    frame["label_state"] = "positive"; frame["eligibility_state"] = "excluded"
    frame["split"] = "locked_test" if locked else "train"
    cohort = tmp_path / "cohort.parquet"; frame.to_parquet(cohort)
    features = tmp_path / "features.parquet"
    frame[["subject_id", "study_id"]].assign(marker=1.).to_parquet(features)
    catalog = SimpleNamespace(connect=lambda **kwargs: duckdb.connect())
    kwargs = dict(cohort_path=cohort, feature_glob=str(features), endpoint_id="endpoint",
                  feature_id="marker", model=None, specification=None)
    if locked:
        assert _locked_window_sensitivity_records(catalog, **kwargs) == []
    else:
        assert _window_sensitivity_records(catalog, **kwargs, bootstrap_iterations=0) == []
