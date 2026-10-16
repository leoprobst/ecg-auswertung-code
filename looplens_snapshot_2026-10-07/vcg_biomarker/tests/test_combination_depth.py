"""Adaptive combination depth.

The properties worth pinning are the ones that make a deep search honest
rather than merely fast: the depth curve is measured on patients the selection
never saw, a level that buys nothing is recognised as buying nothing, and a
composite always carries the depth it was built at -- because a fifteen-marker
score reported as if it were a single marker is the most misleading thing this
module could produce.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.combination_search import (
    CRITERION_DEPTH_CEILING,
    TIER_CRITERION,
    TIER_SCORE,
    SearchBudget,
    assert_depth_is_declared,
    beam_evaluation_count,
    beam_search_combinations,
    claim_tier_for_depth,
    depth_profile,
    exhaustive_evaluation_count,
    recommended_depth,
    search_cost_comparison,
    split_subjects_for_search,
)
from vcg_biomarker.mimic.combination_stage import (
    STOP_FILE_NAME,
    _load_endpoint_frame,
    aligned_out_of_fold_scores,
    stop_file_requested,
)


def test_combination_loader_projects_frozen_markers_and_is_deterministic(
    tmp_path: Path,
) -> None:
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    pd.DataFrame(
        {
            "subject_id": [2, 1, 3],
            "study_id": [22, 11, 33],
            "split": ["train", "train", "validation"],
            "endpoint_state": ["negative", "positive", "positive"],
        }
    ).to_parquet(labels_root / "clinical_labels_wide.parquet", index=False)
    feature_root = tmp_path / "features" / "split=train"
    feature_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [2, 1, 3],
            "study_id": [22, 11, 33],
            "split": ["train", "train", "validation"],
            "extraction_status": ["ok", "ok", "error"],
            "wanted_marker": [2.0, 1.0, 3.0],
            "unrequested_wide_marker": [20.0, 10.0, 30.0],
        }
    ).to_parquet(feature_root / "part-0000.parquet", index=False)
    config = SimpleNamespace(
        labels_root=labels_root,
        catalog_path=tmp_path / "catalog.duckdb",
        duckdb_memory_limit_gb=1,
        workers=1,
    )
    feature_glob = str(tmp_path / "features" / "split=*" / "part-*.parquet")

    frame = _load_endpoint_frame(
        config,
        feature_glob=feature_glob,
        endpoint_id="endpoint",
        marker_ids=("wanted_marker",),
        max_rows=1,
    )

    assert frame.columns.tolist() == [
        "subject_id",
        "study_id",
        "wanted_marker",
        "label",
    ]
    assert frame.to_dict("records") == [
        {
            "subject_id": 1,
            "study_id": 11,
            "wanted_marker": 1.0,
            "label": 1,
        }
    ]


def _informative_panel(
    n: int = 3000,
    n_informative: int = 6,
    n_noise: int = 10,
    seed: int = 11,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Markers that each carry a slice of the signal, plus pure noise.

    Every informative marker is weak on its own, so the composite has to work
    for the search to find anything -- which is the situation the module exists
    for.
    """

    generator = np.random.default_rng(seed)
    labels = (generator.random(n) < 0.3).astype(np.int8)
    subjects = np.repeat(np.arange(n // 2), 2)[:n]
    aligned: dict[str, np.ndarray] = {}
    for index in range(n_informative):
        aligned[f"family{index}_signal"] = (
            0.35 * labels + generator.standard_normal(n)
        )
    for index in range(n_noise):
        aligned[f"noise{index}_random"] = generator.standard_normal(n)
    return aligned, labels, subjects


def _family(feature_id: str) -> str:
    return str(feature_id).split("_", 1)[0]


# ------------------------------------------------------------- claim tiers


def test_depth_decides_whether_a_result_is_a_criterion_or_a_score() -> None:
    for depth in range(1, CRITERION_DEPTH_CEILING + 1):
        assert claim_tier_for_depth(depth) == TIER_CRITERION
    assert claim_tier_for_depth(CRITERION_DEPTH_CEILING + 1) == TIER_SCORE
    assert claim_tier_for_depth(15) == TIER_SCORE


def test_a_combination_may_not_be_reported_without_its_depth() -> None:
    good = pd.DataFrame(
        {"depth": [2, 7], "claim_tier": [TIER_CRITERION, TIER_SCORE]}
    )
    assert_depth_is_declared(good)
    assert_depth_is_declared(pd.DataFrame())

    with pytest.raises(AssertionError, match="depth"):
        assert_depth_is_declared(pd.DataFrame({"claim_tier": [TIER_SCORE]}))
    with pytest.raises(AssertionError, match="no claim tier"):
        assert_depth_is_declared(pd.DataFrame({"depth": [4], "claim_tier": [""]}))
    # A seven-component score labelled as a bedside criterion is the exact
    # misrepresentation this check exists to stop.
    with pytest.raises(AssertionError, match="does not follow from their depth"):
        assert_depth_is_declared(
            pd.DataFrame({"depth": [7], "claim_tier": [TIER_CRITERION]})
        )


# ------------------------------------------------------------------ budget


def test_budget_rejects_settings_that_cannot_mean_anything() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        SearchBudget(max_depth=1)
    with pytest.raises(ValueError, match="patience"):
        SearchBudget(patience=0)
    with pytest.raises(ValueError, match="minimum_increment"):
        SearchBudget(minimum_increment=-0.1)
    with pytest.raises(ValueError, match="seconds"):
        SearchBudget(seconds=0)
    with pytest.raises(ValueError, match="max_evaluations"):
        SearchBudget(max_evaluations=0)


def test_run_until_stopped_is_exactly_no_depth_cap_and_no_patience() -> None:
    assert SearchBudget(max_depth=None, patience=None).runs_until_stopped
    assert not SearchBudget(max_depth=None, patience=2).runs_until_stopped
    assert not SearchBudget(max_depth=6, patience=None).runs_until_stopped


# ------------------------------------------------------------------ search


def test_the_split_that_makes_the_depth_curve_honest_is_by_patient() -> None:
    subjects = np.repeat(np.arange(100), 3)
    selection, confirmation = split_subjects_for_search(subjects)
    assert selection.sum() + confirmation.sum() == len(subjects)
    assert not (selection & confirmation).any()
    # No patient may appear on both sides; otherwise greedy selection is
    # measured on people it already chose against.
    overlap = set(subjects[selection]) & set(subjects[confirmation])
    assert overlap == set()
    with pytest.raises(ValueError, match="Too few subjects"):
        split_subjects_for_search(np.array([1, 1, 2]))


def test_depth_pays_while_there_is_signal_left_to_add() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=6, patience=None, minimum_increment=0.0),
        beam_width=4,
        minimum_complete_fraction=0.5,
    )
    assert len(levels) >= 4
    aucs = [level.best.confirmation_auc for level in levels]
    # Held-out discrimination at depth four beats a single marker.
    assert aucs[3] > aucs[0] + 0.02
    assert all(level.best.depth == level.depth for level in levels)


def test_parallel_candidate_evaluation_is_identical_to_serial() -> None:
    aligned, labels, subjects = _informative_panel(n=1200, seed=29)
    kwargs = {
        "family_for_feature": _family,
        "budget": SearchBudget(
            max_depth=4,
            patience=None,
            minimum_increment=0.0,
        ),
        "beam_width": 4,
        "minimum_complete_fraction": 0.5,
    }

    serial = beam_search_combinations(
        aligned,
        labels,
        subjects,
        worker_count=1,
        **kwargs,
    )
    parallel = beam_search_combinations(
        aligned,
        labels,
        subjects,
        worker_count=4,
        **kwargs,
    )

    # Wall-clock durations are deliberately excluded; every scientific and
    # stopping result must otherwise be exactly the same.
    def stable_candidate(candidate):
        payload = candidate.to_dict() if candidate is not None else None
        if payload is None:
            return None
        for key, value in payload.items():
            if isinstance(value, float) and np.isnan(value):
                payload[key] = "nan"
        return payload

    def stable_level(level):
        return {
            "depth": level.depth,
            "best": stable_candidate(level.best),
            "beam": [stable_candidate(candidate) for candidate in level.beam],
            "evaluations": level.evaluations,
            "improved": level.improved,
            "stop_reason": level.stop_reason,
        }

    assert [stable_level(level) for level in parallel] == [
        stable_level(level) for level in serial
    ]


def test_parallel_candidate_evaluation_rejects_no_workers() -> None:
    aligned, labels, subjects = _informative_panel(n=200)
    with pytest.raises(ValueError, match="worker_count"):
        beam_search_combinations(
            aligned,
            labels,
            subjects,
            family_for_feature=_family,
            worker_count=0,
        )


def test_the_search_stops_itself_when_depth_stops_paying() -> None:
    # Only two markers carry signal; everything after depth two is noise.
    aligned, labels, subjects = _informative_panel(n_informative=2, n_noise=12)
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=None, patience=2, minimum_increment=0.01),
        beam_width=4,
        minimum_complete_fraction=0.5,
    )
    assert levels[-1].stop_reason == "no_further_gain"
    # It stopped on its own, well short of the 14 available families.
    assert levels[-1].depth < 10


def test_an_explicit_depth_cap_is_honoured_and_named() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=3, patience=None, minimum_increment=0.0),
        beam_width=3,
        minimum_complete_fraction=0.5,
    )
    assert max(level.depth for level in levels) == 3
    assert levels[-1].stop_reason == "max_depth_reached"


def test_an_external_stop_ends_the_search_between_levels() -> None:
    aligned, labels, subjects = _informative_panel()
    calls = {"n": 0}

    def stop_now() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(
            max_depth=None,
            patience=None,
            stop_requested=stop_now,
        ),
        beam_width=3,
        minimum_complete_fraction=0.5,
    )
    assert levels[-1].stop_reason == "stop_requested"
    # Everything reached before the stop is still a complete result.
    assert all(level.best is not None for level in levels)


def test_every_level_is_handed_over_before_the_next_one_starts() -> None:
    aligned, labels, subjects = _informative_panel()
    seen: list[int] = []
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=4, patience=None, minimum_increment=0.0),
        beam_width=3,
        minimum_complete_fraction=0.5,
        on_depth_complete=lambda level: seen.append(level.depth),
    )
    # This is what makes stopping safe at any moment.
    assert seen == [level.depth for level in levels]
    assert seen == sorted(seen)


def test_one_component_per_family_is_enforced() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=lambda _: "one_family_for_everything",
        budget=SearchBudget(max_depth=5, patience=None),
        beam_width=3,
        minimum_complete_fraction=0.5,
    )
    # With a single family nothing can be extended past depth one.
    assert max(level.depth for level in levels) == 1
    assert levels[-1].stop_reason == "no_remaining_families"


def test_families_never_repeat_inside_a_combination() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=5, patience=None, minimum_increment=0.0),
        beam_width=4,
        minimum_complete_fraction=0.5,
    )
    for level in levels:
        for candidate in level.beam:
            assert len(set(candidate.families)) == len(candidate.families)
            assert len(set(candidate.components)) == candidate.depth


def test_permuted_labels_produce_no_depth_gain() -> None:
    aligned, labels, subjects = _informative_panel()
    generator = np.random.default_rng(5)
    shuffled = labels.copy()
    generator.shuffle(shuffled)
    levels = beam_search_combinations(
        aligned,
        shuffled,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=6, patience=None, minimum_increment=0.0),
        beam_width=4,
        minimum_complete_fraction=0.5,
    )
    # Selection can always improve its own half; the measurement half is what
    # keeps a search from believing its own noise.
    best = max(level.best.confirmation_auc for level in levels)
    assert best < 0.60


def test_mismatched_inputs_are_refused() -> None:
    aligned, labels, subjects = _informative_panel(n=200)
    with pytest.raises(ValueError, match="do not match the labels"):
        beam_search_combinations(
            {**aligned, "bad_marker": np.zeros(5)},
            labels,
            subjects,
            family_for_feature=_family,
        )
    with pytest.raises(ValueError, match="Subject IDs and labels"):
        beam_search_combinations(
            aligned,
            labels,
            subjects[:10],
            family_for_feature=_family,
        )
    with pytest.raises(ValueError, match="beam_width"):
        beam_search_combinations(
            aligned,
            labels,
            subjects,
            family_for_feature=_family,
            beam_width=0,
        )
    assert beam_search_combinations(
        {"only_one": np.zeros(len(labels))},
        labels,
        subjects,
        family_for_feature=_family,
    ) == ()


# ------------------------------------------------------------ reporting


def test_the_recommendation_prefers_the_shallowest_depth_that_ties() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=8, patience=None, minimum_increment=0.0),
        beam_width=4,
        minimum_complete_fraction=0.5,
    )
    report = recommended_depth(levels, tolerance=0.005)
    assert report["recommended_depth"] <= report["best_depth"]
    assert report["auc_given_up_for_parsimony"] <= 0.005
    assert len(report["recommended_components"]) == report["recommended_depth"]
    assert report["claim_tier"] == claim_tier_for_depth(
        report["recommended_depth"]
    )
    assert recommended_depth(())["recommended_depth"] == 0


def test_the_depth_profile_is_a_readable_table() -> None:
    aligned, labels, subjects = _informative_panel()
    levels = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=_family,
        budget=SearchBudget(max_depth=4, patience=None, minimum_increment=0.0),
        beam_width=3,
        minimum_complete_fraction=0.5,
    )
    profile = depth_profile(levels)
    assert list(profile["depth"]) == [1, 2, 3, 4]
    assert profile["claim_tier"].iloc[-1] == TIER_SCORE
    assert profile["evaluations"].gt(0).all()
    assert depth_profile(()).empty


def test_beam_search_is_linear_in_depth_where_exhaustive_is_not() -> None:
    assert exhaustive_evaluation_count(67, 4) == 766_480
    assert beam_evaluation_count(67, 4, 8) < 2_000
    table = search_cost_comparison(67, beam_width=8)
    # Beam cost grows by a constant per level; exhaustive cost explodes.
    steps = table["beam_evaluations"].diff().dropna()
    assert (steps > 0).all()
    assert table["reduction_factor"].is_monotonic_increasing
    assert table.loc[table["depth"] <= 3, "claim_tier"].eq(TIER_CRITERION).all()


# --------------------------------------------------------------- stage bits


def test_aligned_scores_are_out_of_fold_and_direction_stable() -> None:
    generator = np.random.default_rng(3)
    n = 2000
    labels = (generator.random(n) < 0.4).astype(np.int8)
    subjects = np.repeat(np.arange(n // 2), 2)
    frame = pd.DataFrame(
        {
            "up": 0.8 * labels + generator.standard_normal(n),
            "down": -0.8 * labels + generator.standard_normal(n),
            "pure_noise": generator.standard_normal(n),
            "mostly_missing": np.where(
                generator.random(n) < 0.5, np.nan, generator.standard_normal(n)
            ),
        }
    )
    aligned = aligned_out_of_fold_scores(
        frame,
        labels,
        subjects,
        ["up", "down", "pure_noise", "mostly_missing", "absent"],
    )
    # A marker that points the other way is flipped, not discarded.
    assert {"up", "down"} <= set(aligned)
    for marker_id in ("up", "down"):
        finite = np.isfinite(aligned[marker_id])
        assert np.mean(aligned[marker_id][finite][labels[finite] == 1]) > 0
    # Too sparse to be scored against everything else, and not present at all.
    assert "mostly_missing" not in aligned
    assert "absent" not in aligned


def test_aligned_scores_survive_degenerate_input() -> None:
    n = 100
    frame = pd.DataFrame({"flat": np.ones(n)})
    labels = np.zeros(n, dtype=np.int8)
    subjects = np.arange(n)
    assert aligned_out_of_fold_scores(frame, labels, subjects, ["flat"]) == {}


def test_the_stop_file_is_how_a_user_ends_an_open_ended_search(tmp_path) -> None:
    assert not stop_file_requested(tmp_path)
    (tmp_path / STOP_FILE_NAME).write_text("stop", encoding="utf-8")
    assert stop_file_requested(tmp_path)


def test_only_a_few_markers_per_family_are_extended() -> None:
    """Why the search stays affordable at the full panel.

    A combination holds at most one member of a family, so extending every
    same-family sibling evaluates them only to have the family rule exclude
    them one level later. At 1,713 markers that was 152,457 evaluations per
    endpoint against 9,612 with the cap -- 71 hours against 4.5 over 63
    endpoints.
    """

    generator = np.random.default_rng(21)
    n = 2000
    labels = (generator.random(n) < 0.3).astype(np.int8)
    subjects = np.repeat(np.arange(n // 2), 2)
    # Three families of ten near-duplicate markers each.
    aligned = {}
    for family in range(3):
        base = 0.4 * labels + generator.standard_normal(n)
        for sibling in range(10):
            aligned[f"fam{family}_marker{sibling}"] = (
                base + 0.05 * generator.standard_normal(n)
            )

    def family_of(feature_id: str) -> str:
        return str(feature_id).split("_", 1)[0]

    capped = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=family_of,
        budget=SearchBudget(max_depth=3, patience=None, minimum_increment=0.0),
        beam_width=4,
        representatives_per_family=2,
        minimum_complete_fraction=0.5,
    )
    uncapped = beam_search_combinations(
        aligned,
        labels,
        subjects,
        family_for_feature=family_of,
        budget=SearchBudget(max_depth=3, patience=None, minimum_increment=0.0),
        beam_width=4,
        representatives_per_family=0,
        minimum_complete_fraction=0.5,
    )
    capped_evaluations = sum(level.evaluations for level in capped)
    uncapped_evaluations = sum(level.evaluations for level in uncapped)
    assert capped_evaluations < uncapped_evaluations / 2

    # And the cheaper search must not have given up the answer: the families
    # it can reach are the same, so held-out discrimination is comparable.
    assert capped[-1].best.confirmation_auc > uncapped[-1].best.confirmation_auc - 0.02
    # Depth one still ranks every marker, so nothing is hidden from the report.
    assert capped[0].evaluations == len(aligned)


# --------------------------------------------------- stopping an open search


def test_the_three_stop_modes_are_distinct_and_the_safe_one_is_default(tmp_path) -> None:
    from vcg_biomarker.mimic.combination_stage import (
        DEFAULT_STOP_MODE,
        STOP_MODE_AFTER_ENDPOINT,
        STOP_MODE_AFTER_SWEEP,
        STOP_MODE_IMMEDIATE,
        STOP_MODES,
        request_stop,
        requested_stop_mode,
    )

    assert set(STOP_MODES) == {
        STOP_MODE_IMMEDIATE,
        STOP_MODE_AFTER_ENDPOINT,
        STOP_MODE_AFTER_SWEEP,
    }
    # The default must be the one that leaves no endpoint without any
    # combination, because that is the failure a user cannot repair later.
    assert DEFAULT_STOP_MODE == STOP_MODE_AFTER_SWEEP

    assert requested_stop_mode(tmp_path) is None
    request_stop(tmp_path, mode=STOP_MODE_IMMEDIATE, requested_by="test")
    assert requested_stop_mode(tmp_path) == STOP_MODE_IMMEDIATE
    request_stop(tmp_path, mode=STOP_MODE_AFTER_SWEEP)
    assert requested_stop_mode(tmp_path) == STOP_MODE_AFTER_SWEEP

    with pytest.raises(ValueError, match="Unknown stop mode"):
        request_stop(tmp_path, mode="whenever")


def test_an_unreadable_stop_file_is_read_as_the_safe_mode(tmp_path) -> None:
    from vcg_biomarker.mimic.combination_stage import (
        DEFAULT_STOP_MODE,
        STOP_FILE_NAME,
        requested_stop_mode,
    )

    # A hand-written or older stop file carries no mode. Reading it as the
    # abrupt one would discard work the user never asked to discard.
    (tmp_path / STOP_FILE_NAME).write_text("stop", encoding="utf-8")
    assert requested_stop_mode(tmp_path) == DEFAULT_STOP_MODE
    (tmp_path / STOP_FILE_NAME).write_text('{"mode": "nonsense"}', encoding="utf-8")
    assert requested_stop_mode(tmp_path) == DEFAULT_STOP_MODE


def test_live_state_reports_depth_and_the_estimate_for_the_next_level(tmp_path) -> None:
    from vcg_biomarker.mimic.combination_stage import (
        LIVE_FILE_NAME,
        describe_live_state,
        read_live_state,
    )
    import json as _json

    assert read_live_state(tmp_path) is None
    assert "kein Live-Status" in describe_live_state(None)

    (tmp_path / LIVE_FILE_NAME).write_text(
        _json.dumps(
            {
                "endpoint_id": "reduced_lvef_strict",
                "endpoint_position": 7,
                "endpoints_total": 63,
                "current_depth": 5,
                "estimated_next_level_seconds": 42.0,
                "stop_mode_pending": None,
            }
        ),
        encoding="utf-8",
    )
    state = read_live_state(tmp_path)
    assert state["current_depth"] == 5
    text = describe_live_state(state)
    # Everything the user asked to see: where in the sweep, how deep, how long.
    assert "7/63" in text
    assert "Tiefe 5" in text
    assert "42" in text

    (tmp_path / LIVE_FILE_NAME).write_text("{not json", encoding="utf-8")
    assert read_live_state(tmp_path) is None


def test_the_gui_exposes_one_button_per_stop_mode() -> None:
    """The GUI must be able to end the only open-ended stage in the run."""

    import inspect

    from vcg_biomarker.gui.widgets import mimic_project_widget as widget

    source = inspect.getsource(widget.MimicProjectWidget)
    for handler in (
        "stop_deep_search_after_sweep",
        "stop_deep_search_after_endpoint",
        "stop_deep_search_immediately",
        "_refresh_deep_search_status",
        "_build_deep_search_box",
    ):
        assert hasattr(widget.MimicProjectWidget, handler), handler
    # The status must be refreshed by the poll loop, not only on demand.
    assert "_refresh_deep_search_status()" in inspect.getsource(
        widget.MimicProjectWidget._poll_background_state
    )
    # Each handler must request its own mode.
    for mode in ("after_sweep", "after_endpoint", "immediate"):
        assert f'"{mode}"' in source


def test_a_leftover_stop_file_does_not_kill_the_next_run(tmp_path, monkeypatch) -> None:
    """Regression: restarting after a stop stopped again, silently.

    A stop is a legitimate outcome, not an error, so a stale stop file would
    have ended the restarted stage without anything looking wrong. Starting
    the stage is the intent to search, so the file is cleared first.
    """

    from vcg_biomarker.mimic import combination_stage as stage
    from vcg_biomarker.mimic.combination_stage import (
        STOP_FILE_NAME,
        STOP_MODE_IMMEDIATE,
        request_stop,
    )

    class _Config:
        runs_root = tmp_path
        labels_root = tmp_path
        features_root = tmp_path

        def to_dict(self):
            return {}

    run_root = tmp_path / "some-run" / "deep_combinations"
    run_root.mkdir(parents=True)
    request_stop(run_root, mode=STOP_MODE_IMMEDIATE, requested_by="previous_run")
    assert (run_root / STOP_FILE_NAME).exists()

    # No endpoints, so the stage returns early -- but only after it has done
    # the one thing this test is about.
    monkeypatch.setattr(stage, "available_endpoints", lambda config: ())
    with pytest.raises(ValueError, match="No endpoints"):
        stage.run_deep_combination_search(_Config(), run_id="some-run")
    assert not (run_root / STOP_FILE_NAME).exists()
