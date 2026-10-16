"""Markers that cannot be told apart must be grouped before they are screened.

Measured on 5,924 development ECGs: at |rho| >= 0.99 the 1,753-marker catalog
holds 131 groups with more than one member and 162 redundant markers, leaving
1,591 independent quantities. Two consequences, and only the first is cosmetic:

* a "top 20 markers" list is dominated by whichever quantity has the most
  spellings -- the 2026-08-28 results showed one quantity under five names in
  the same ranking;
* the FDR family counts perfectly correlated tests as separate ones, which
  makes "21,099 replicated hits" a number about the catalog, not the data.

The grouping must never trade information for familiarity, and must never let
a comparator displace a candidate. Both are tested below, because the first
draft got both wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.marker_redundancy import (
    REDUNDANCY_RHO_THRESHOLD,
    RedundancyPolicy,
    build_redundancy_table,
    cluster_markers,
    elect_representative,
    panel_rank_map,
    rank_correlation_matrix,
    redundancy_summary,
    screening_family,
)


def _frame(rng, n=900):
    base = rng.normal(size=n)
    other = rng.normal(size=n)
    return pd.DataFrame(
        {
            # One quantity under three names, one of them negated.
            "alpha": base,
            "alpha_scaled": base * 7.5 + 3.0,
            "alpha_negated": -base,
            # Related but distinguishable.
            "beta": 0.6 * base + 0.8 * rng.normal(size=n),
            # Unrelated.
            "gamma": other,
            # Constant: cannot be compared at all.
            "delta": np.ones(n),
        }
    )


@pytest.fixture()
def policy():
    return RedundancyPolicy(minimum_observations=100)


def test_a_monotone_rescaling_is_the_same_marker(policy):
    rng = np.random.default_rng(3)
    table = build_redundancy_table(
        _frame(rng), ["alpha", "alpha_scaled", "gamma"], policy=policy
    )
    groups = list(table.groupby("cluster_id")["feature_id"].apply(set))
    assert {"alpha", "alpha_scaled"} in groups


def test_a_negated_marker_is_the_same_marker(policy):
    """A marker and its negation give one finding with the sign flipped."""

    rng = np.random.default_rng(4)
    table = build_redundancy_table(
        _frame(rng), ["alpha", "alpha_negated"], policy=policy
    )
    assert table["cluster_id"].nunique() == 1


def test_a_merely_correlated_marker_stays_separate(policy):
    rng = np.random.default_rng(5)
    table = build_redundancy_table(_frame(rng), ["alpha", "beta"], policy=policy)
    assert table["cluster_id"].nunique() == 2


def test_a_measurement_beats_a_threshold_cut_from_it():
    """The millisecond value carries strictly more than the yes/no."""

    coverage = {"measure_ms": 1.0, "over_threshold": 1.0}
    panels = {"measure_ms": 2, "over_threshold": 0}
    distinct = {"measure_ms": 4000, "over_threshold": 2}
    assert (
        elect_representative(
            ["measure_ms", "over_threshold"], coverage, panels, distinct
        )
        == "measure_ms"
    )


def test_the_panel_decides_between_equals():
    coverage = {"a_marker": 1.0, "b_marker": 1.0}
    panels = {"a_marker": 2, "b_marker": 0}
    distinct = {"a_marker": 4000, "b_marker": 4000}
    assert (
        elect_representative(["a_marker", "b_marker"], coverage, panels, distinct)
        == "b_marker"
    )


def test_coverage_breaks_a_remaining_tie():
    coverage = {"sparse_one": 0.2, "dense_one": 0.9}
    panels = {"sparse_one": 1, "dense_one": 1}
    distinct = {"sparse_one": 500, "dense_one": 500}
    assert (
        elect_representative(
            ["sparse_one", "dense_one"], coverage, panels, distinct
        )
        == "dense_one"
    )


def test_the_election_does_not_depend_on_input_order():
    coverage = {"x": 1.0, "y": 1.0, "z": 1.0}
    panels = {"x": 1, "y": 1, "z": 1}
    distinct = {"x": 100, "y": 100, "z": 100}
    first = elect_representative(["x", "y", "z"], coverage, panels, distinct)
    second = elect_representative(["z", "y", "x"], coverage, panels, distinct)
    assert first == second


def test_a_classical_baseline_never_represents_a_candidate():
    """Electing the yardstick would delete the candidate from the screen."""

    rng = np.random.default_rng(6)
    base = rng.normal(size=600)
    frame = pd.DataFrame(
        {"qrs_duration_ms": base, "classical_qrs_widening_ge_120": base * 2}
    )
    table = build_redundancy_table(
        frame,
        ["qrs_duration_ms", "classical_qrs_widening_ge_120"],
        policy=RedundancyPolicy(minimum_observations=100),
    )
    chosen = set(screening_family(table))
    assert "classical_qrs_widening_ge_120" not in chosen
    assert "qrs_duration_ms" in chosen


def test_the_classical_twin_is_recorded_not_hidden():
    """A candidate that *is* a classical criterion is a finding, not noise."""

    rng = np.random.default_rng(7)
    base = rng.normal(size=600)
    frame = pd.DataFrame(
        {"qrs_duration_ms": base, "classical_qrs_widening_ge_120": base * 3.0}
    )
    table = build_redundancy_table(
        frame,
        ["qrs_duration_ms", "classical_qrs_widening_ge_120"],
        policy=RedundancyPolicy(minimum_observations=100),
    )
    row = table.loc[table["feature_id"].eq("qrs_duration_ms")].iloc[0]
    assert row["classical_twin"] == "classical_qrs_widening_ge_120"
    assert row["abs_rho_to_classical_twin"] == pytest.approx(1.0, abs=1e-9)


def test_clustering_is_transitive():
    """A and B agree, B and C agree: all three are one quantity."""

    names = ["a", "b", "c"]
    matrix = np.array(
        [[1.0, 0.995, 0.98], [0.995, 1.0, 0.995], [0.98, 0.995, 1.0]]
    )
    assert cluster_markers(matrix, names, threshold=0.99) == [["a", "b", "c"]]


def test_a_marker_that_cannot_be_compared_is_kept_as_its_own_cluster(policy):
    """Dropping it would shrink the family for a reason nobody could see."""

    rng = np.random.default_rng(8)
    table = build_redundancy_table(
        _frame(rng), ["alpha", "gamma", "delta"], policy=policy
    )
    assert "delta" in set(table["feature_id"])
    row = table.loc[table["feature_id"].eq("delta")].iloc[0]
    assert bool(row["is_representative"]) is True
    assert int(row["cluster_size"]) == 1


def test_every_marker_appears_exactly_once(policy):
    rng = np.random.default_rng(9)
    ids = ["alpha", "alpha_scaled", "alpha_negated", "beta", "gamma", "delta"]
    table = build_redundancy_table(_frame(rng), ids, policy=policy)
    assert sorted(table["feature_id"]) == sorted(ids)
    assert len(table) == len(ids)


def test_each_cluster_has_exactly_one_representative(policy):
    rng = np.random.default_rng(10)
    ids = ["alpha", "alpha_scaled", "alpha_negated", "beta", "gamma", "delta"]
    table = build_redundancy_table(_frame(rng), ids, policy=policy)
    counts = table.groupby("cluster_id")["is_representative"].sum()
    assert set(counts.unique()) == {1}


def test_the_screening_family_is_the_representatives(policy):
    rng = np.random.default_rng(11)
    ids = ["alpha", "alpha_scaled", "beta", "gamma"]
    table = build_redundancy_table(_frame(rng), ids, policy=policy)
    family = screening_family(table)
    assert len(family) == table["cluster_id"].nunique()
    assert set(family) <= set(ids)


def test_the_summary_counts_what_it_says(policy):
    rng = np.random.default_rng(12)
    ids = ["alpha", "alpha_scaled", "alpha_negated", "beta", "gamma"]
    table = build_redundancy_table(_frame(rng), ids, policy=policy)
    summary = redundancy_summary(table)
    assert summary["n_markers"] == len(ids)
    assert summary["n_redundant"] == len(ids) - summary["n_clusters"]
    assert summary["largest_cluster"] >= 3


def test_the_threshold_is_prespecified():
    assert REDUNDANCY_RHO_THRESHOLD == 0.99
    assert RedundancyPolicy().rho_threshold == 0.99


def test_the_policy_is_recorded_in_full():
    payload = RedundancyPolicy().to_dict()
    for key in ("rho_threshold", "sample_rows", "sample_seed", "version"):
        assert key in payload


def test_an_empty_catalog_does_not_crash():
    table = build_redundancy_table(pd.DataFrame(), [], policy=RedundancyPolicy())
    assert table.empty
    assert screening_family(table) == ()
    assert redundancy_summary(table)["n_markers"] == 0


def test_the_panel_map_covers_the_real_catalog():
    ranks = panel_rank_map()
    assert len(ranks) > 1_500
    assert ranks["qrs_duration_ms"] >= 0


def test_the_correlation_matrix_skips_unusable_columns():
    rng = np.random.default_rng(13)
    matrix, names = rank_correlation_matrix(
        _frame(rng), ["alpha", "delta"], minimum_observations=100
    )
    assert names == ["alpha"]
    assert matrix.shape == (1, 1)


# --- Die Stichprobe ----------------------------------------------------------


def _write_partitions(root, *, files=8, rows=50, markers=("m0", "m1")):
    import numpy as np

    rng = np.random.default_rng(17)
    for split in ("train", "validation"):
        folder = root / f"split={split}"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(files):
            frame = pd.DataFrame({name: rng.normal(size=rows) for name in markers})
            frame["extraction_status"] = "ok" if index % 4 else "signal_qc_failed"
            frame["split"] = split
            frame.to_parquet(folder / f"part-{index:04d}.parquet", index=False)
    return str(root / "split=*" / "part-*.parquet").replace("\\", "/")


def test_the_sample_never_reads_the_validation_split(tmp_path):
    """Which markers count as one quantity is a preprocessing decision.

    Fitting it on validation rows would let the held-out split shape the
    family that is later tested on it.
    """

    from vcg_biomarker.mimic.marker_redundancy import sample_feature_rows

    glob = _write_partitions(tmp_path / "features")
    frame = sample_feature_rows(
        glob, ["m0", "m1"], policy=RedundancyPolicy(sample_rows=10_000)
    )
    assert not frame.empty
    assert "split" not in frame.columns
    # Every train partition holds 50 rows, three quarters of them usable.
    assert len(frame) <= 8 * 50


def test_the_sample_drops_failed_extractions(tmp_path):
    from vcg_biomarker.mimic.marker_redundancy import sample_feature_rows

    glob = _write_partitions(tmp_path / "features")
    frame = sample_feature_rows(
        glob, ["m0"], policy=RedundancyPolicy(sample_rows=10_000)
    )
    # One partition in four is entirely failed; those rows must not appear.
    assert len(frame) == 6 * 50


def test_the_sample_spans_the_whole_set_not_its_beginning(tmp_path):
    """Systematic, so nothing that varies along extraction order can bias it."""

    from vcg_biomarker.mimic.marker_redundancy import sample_feature_rows

    root = tmp_path / "features"
    folder = root / "split=train"
    folder.mkdir(parents=True)
    for index in range(20):
        pd.DataFrame(
            {"m0": [float(index)] * 10, "extraction_status": ["ok"] * 10}
        ).to_parquet(folder / f"part-{index:04d}.parquet", index=False)
    glob = str(root / "split=*" / "part-*.parquet").replace("\\", "/")

    frame = sample_feature_rows(
        glob, ["m0"], policy=RedundancyPolicy(sample_rows=40)
    )
    seen = sorted(frame["m0"].unique())
    assert len(seen) >= 4
    assert max(seen) >= 15, f"nur der Anfang gelesen: {seen}"


def test_no_matching_partitions_returns_the_empty_shape(tmp_path):
    from vcg_biomarker.mimic.marker_redundancy import sample_feature_rows

    glob = str(tmp_path / "absent" / "split=*" / "part-*.parquet")
    frame = sample_feature_rows(glob, ["m0"], policy=RedundancyPolicy())
    assert frame.empty
    assert list(frame.columns) == ["m0"]
