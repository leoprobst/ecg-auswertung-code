"""Ranking the controls once per chunk must give the same statistic.

The phenotype scan compares every marker against every phenotype's controls.
Those control sets differ only by the recordings belonging to that phenotype's
own cases -- typically well under one percent of them -- yet the scan sorted the
whole control column again for each phenotype: 2,885 phenotypes multiplied by
1,753 markers multiplied by two splits.

The replacement sorts each marker's controls once and subtracts the excluded
ones arithmetically.  That is only worth anything if it is exact, so these tests
compare it against the direct computation rather than against a stored number:
first the statistic in isolation, over tie-heavy random draws, then the whole
screen end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.phenome import _screen_split
from vcg_biomarker.mimic.statistics import (
    excluded_mann_whitney_auc,
    fast_mann_whitney_auc,
    prepare_sorted_control_distribution,
)


@pytest.mark.parametrize("seed", range(25))
def test_control_exclusion_identity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_control = int(rng.integers(20, 900))
    n_case = int(rng.integers(5, 150))
    # Rounding to few decimals forces heavy ties, which is where a tie-corrected
    # statistic is easiest to get wrong.
    controls = np.round(rng.normal(size=n_control), int(rng.integers(0, 3)))
    cases = np.round(rng.normal(loc=0.4, size=n_case), int(rng.integers(0, 3)))
    removed = int(rng.integers(0, max(1, n_control // 2)))
    positions = rng.choice(n_control, size=removed, replace=False)
    reduced = np.delete(controls, positions)
    if len(reduced) == 0:
        pytest.skip("degenerate draw")

    direct = fast_mann_whitney_auc(
        cases, prepare_sorted_control_distribution(reduced)
    )
    subtracted = excluded_mann_whitney_auc(
        cases,
        prepare_sorted_control_distribution(controls),
        controls[positions],
    )
    assert subtracted[0] == direct[0]
    assert subtracted[1] == direct[1]


def test_non_finite_controls_do_not_shift_the_count() -> None:
    controls = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 2.0])
    cases = np.array([2.5, np.nan, 0.5])
    excluded = np.array([2.0, np.nan])
    direct = fast_mann_whitney_auc(
        cases, prepare_sorted_control_distribution([1.0, 2.0, 3.0])
    )
    subtracted = excluded_mann_whitney_auc(
        cases, prepare_sorted_control_distribution(controls), excluded
    )
    assert subtracted == direct


def test_removing_every_control_is_reported_as_not_assessable() -> None:
    controls = np.array([1.0, 2.0, 3.0])
    auc, p_value = excluded_mann_whitney_auc(
        np.array([1.5]),
        prepare_sorted_control_distribution(controls),
        controls,
    )
    assert np.isnan(auc)
    assert p_value == 1.0


def _synthetic_screen(seed: int):
    rng = np.random.default_rng(seed)
    n = 240
    subject_id = np.repeat(np.arange(1, n // 2 + 1), 2)
    study_id = np.arange(1, n + 1)
    features = pd.DataFrame(
        {
            "subject_id": subject_id,
            "study_id": study_id,
            "split": "train",
            "is_control_index": rng.random(n) < 0.85,
            "marker_a": np.round(rng.normal(size=n), 1),
            "marker_b": rng.normal(size=n),
        }
    )
    features.loc[rng.random(n) < 0.1, "marker_b"] = np.nan
    rows = []
    for phenotype_id in ("phe_one", "phe_two"):
        chosen = rng.choice(study_id, size=40, replace=False)
        for value in chosen:
            rows.append(
                {
                    "phenotype_id": phenotype_id,
                    "subject_id": int(subject_id[value - 1]),
                    "study_id": int(value),
                    "split": "train",
                }
            )
    current = pd.DataFrame(rows)
    definitions = pd.DataFrame(
        {
            "phenotype_id": ["phe_one", "phe_two"],
            "display_name": ["Phenotype one", "Phenotype two"],
            "body_system": ["Cardiovascular", "Respiratory"],
        }
    ).set_index("phenotype_id")
    return features, current, definitions


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_the_screen_matches_a_control_set_built_by_hand(seed: int) -> None:
    features, current, definitions = _synthetic_screen(seed)
    records = _screen_split(
        features,
        current,
        pd.DataFrame(),
        phenotype_ids=("phe_one", "phe_two"),
        feature_ids=("marker_a", "marker_b"),
        panel_map={"marker_a": "primary_core", "marker_b": "sentinel"},
        definition_lookup=definitions,
        min_inference=1,
        min_formal=1,
        progress=None,
    )
    assert records

    subjects = features["subject_id"].to_numpy()
    control_rows = np.flatnonzero(features["is_control_index"].to_numpy())
    for record in records:
        phenotype_id = record["phenotype_id"]
        feature_id = record["feature_id"]
        case_studies = set(
            current.loc[current["phenotype_id"] == phenotype_id, "study_id"]
        )
        case_rows = np.flatnonzero(features["study_id"].isin(case_studies))
        case_subjects = subjects[case_rows]
        kept = control_rows[~np.isin(subjects[control_rows], case_subjects)]
        values = features[feature_id].to_numpy(dtype=float)
        cases = values[case_rows]
        cases = cases[np.isfinite(cases)]
        controls = values[kept]
        controls = controls[np.isfinite(controls)]
        expected_auc, expected_p = fast_mann_whitney_auc(
            cases, prepare_sorted_control_distribution(controls)
        )
        assert record["n_case"] == len(cases)
        assert record["n_control"] == len(controls)
        assert record["raw_auc"] == pytest.approx(expected_auc, abs=0.0)
        assert record["p_value"] == pytest.approx(expected_p, abs=0.0)


def test_the_screen_is_safe_to_run_on_several_threads() -> None:
    """The train screen runs its phenotype chunks on a thread pool.

    That is only allowed if ``_screen_split`` shares nothing mutable between
    calls -- it now caches a sorted control column per marker, which would be a
    problem if the cache outlived the call.  Measured scaling on synthetic data
    at the real shape was 2.15x on four threads; this test pins the correctness
    half of that, that concurrency does not change a single number.
    """

    from concurrent.futures import ThreadPoolExecutor

    features, current, definitions = _synthetic_screen(5)
    chunks = [("phe_one",), ("phe_two",), ("phe_one", "phe_two")]

    def run(chunk):
        return _screen_split(
            features,
            current,
            pd.DataFrame(),
            phenotype_ids=chunk,
            feature_ids=("marker_a", "marker_b"),
            panel_map={"marker_a": "primary_core", "marker_b": "sentinel"},
            definition_lookup=definitions,
            min_inference=1,
            min_formal=1,
            progress=None,
        )

    sequential = [run(chunk) for chunk in chunks]
    with ThreadPoolExecutor(max_workers=3) as pool:
        concurrent = list(pool.map(run, chunks))

    assert [len(part) for part in concurrent] == [len(part) for part in sequential]
    for expected, produced in zip(sequential, concurrent):
        for left, right in zip(expected, produced):
            assert left["phenotype_id"] == right["phenotype_id"]
            assert left["feature_id"] == right["feature_id"]
            assert left["raw_auc"] == right["raw_auc"]
            assert left["p_value"] == right["p_value"]
