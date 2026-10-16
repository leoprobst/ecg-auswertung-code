"""Live run reporting and the train/validation swap.

Two properties are worth pinning here. The progress board must weight stages
by cost, because a board that counts stages equally reports 60 % done when the
single most expensive stage has not started. And the cross-fit must refuse,
loudly, to touch the sealed split -- that is the one guarantee in the pipeline
that no amount of recomputation can restore.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from vcg_biomarker.mimic.cross_fitting import (
    CROSS_FIT_ASSIGNMENTS,
    SEALED_SPLITS,
    SWAPPABLE_SPLITS,
    assert_folds_are_subject_disjoint,
    assert_locked_test_excluded,
    combine_folds,
    cross_fit_multiplicity_family,
    cross_fit_plan,
    cross_fit_summary,
    efficiency_gain,
)
from vcg_biomarker.mimic.live_progress import (
    STAGE_DESCRIPTIONS,
    STAGE_WEIGHTS,
    LiveProgressReporter,
    RunProgress,
    format_duration,
    parse_fraction,
    read_live_state,
    render_snapshot,
    stage_description,
    stage_weight,
)


# ---------------------------------------------------------------- progress


def _run(stages=("ingest", "extract-features", "validate")) -> RunProgress:
    return RunProgress(run_id="test-run", stages=list(stages))


def test_every_weighted_stage_is_also_explained() -> None:
    # A stage that shows up in the board with a weight but no sentence is a
    # progress bar the reader cannot interpret.
    missing = sorted(set(STAGE_WEIGHTS) - set(STAGE_DESCRIPTIONS))
    assert missing == []
    for text in STAGE_DESCRIPTIONS.values():
        assert text.strip().endswith(".")


def test_unknown_stage_still_renders_with_a_default() -> None:
    assert stage_weight("nothing-like-this") == 1.0
    assert stage_description("nothing-like-this") == "Running a pipeline stage."


def test_progress_is_weighted_by_cost_not_by_stage_count() -> None:
    """A finished cheap stage must not read as a third of the run.

    The expectations are derived from STAGE_WEIGHTS rather than written as
    fixed ranges. The weights are measurements and they move: the August 2026
    pre-run raised validate-pilot-1000 from 0.3 h to a measured 25.3, and a
    test that hard-codes "extract-features is the big one" would then fail for
    being out of date rather than for finding a defect.
    """

    from vcg_biomarker.mimic.live_progress import stage_weight

    stages = ("ingest", "extract-features", "validate")
    total = sum(stage_weight(stage) for stage in stages)
    progress = _run(stages)

    progress.finish_stage("ingest")
    assert progress.stages_completed_text() == "stage 1/3"
    expected_after_cheap = stage_weight("ingest") / total
    assert progress.overall_fraction == pytest.approx(
        expected_after_cheap, abs=1e-6
    )
    # The point of weighting: one of three stages done is nowhere near a third.
    assert progress.overall_fraction < 0.20

    progress.begin_stage("extract-features")
    progress.update_stage(0.5, "12,500/25,000 records")
    expected_at_half = (
        stage_weight("ingest") + 0.5 * stage_weight("extract-features")
    ) / total
    assert progress.overall_fraction == pytest.approx(
        expected_at_half, abs=1e-6
    )

    half_way = progress.overall_fraction
    progress.finish_stage("extract-features")
    assert progress.overall_fraction > half_way


def test_fraction_is_clamped_and_current_stage_is_cleared_on_finish() -> None:
    progress = _run()
    progress.begin_stage("validate")
    progress.update_stage(5.0)
    assert progress.current_fraction == 1.0
    progress.update_stage(-2.0)
    assert progress.current_fraction == 0.0
    progress.finish_stage("validate")
    assert progress.current is None
    assert progress.overall_fraction > 0.0


def test_the_eta_does_not_extrapolate_the_pace_of_the_cheap_stages() -> None:
    """The defect this replaced: an order of magnitude, in the wrong direction.

    The estimate used to project the pace observed so far across the rest of
    the run. The stages that finish first are cheap preparation work and run
    far faster per weighted hour than the two that dominate the total, so two
    hours into the August 2026 run the board reported forty hours remaining
    while the weights -- measurements of exactly those remaining stages --
    said 453.

    The weights are the estimate now, so how fast the preparation happened to
    run must not move it.
    """

    progress = _run()
    progress.begin_stage("ingest")
    progress.update_stage(0.01)
    early = progress.estimated_remaining_seconds()
    # Available immediately: an estimate that does not extrapolate needs no
    # warm-up before it means something.
    assert early is not None and early > 0.0

    # Pretend the first stage crawled. The remaining stages are unaffected.
    slow = _run()
    slow.begin_stage("ingest")
    slow.update_stage(0.01)
    slow.started_at -= 10_000.0
    assert slow.estimated_remaining_seconds() == pytest.approx(early)


def test_the_eta_falls_as_stages_finish() -> None:
    progress = _run()
    before = progress.estimated_remaining_seconds()
    progress.begin_stage("ingest")
    progress.finish_stage("ingest")
    after = progress.estimated_remaining_seconds()
    assert after is not None and before is not None
    assert after < before


def test_an_open_ended_stage_is_left_out_of_the_estimate() -> None:
    """A stage that never ends must not carry a finish time."""

    from vcg_biomarker.mimic.live_progress import (
        OPEN_ENDED_STAGES,
        RunProgress,
        stage_weight,
    )

    open_stage = sorted(OPEN_ENDED_STAGES)[0]
    progress = RunProgress(run_id="r", stages=["ingest", open_stage])
    assert progress.remaining_weight == pytest.approx(stage_weight("ingest"))


def test_incomplete_prior_is_not_a_guaranteed_time_bound() -> None:
    """Unknown work must not turn an uncertain prior into a lower bound."""

    from vcg_biomarker.mimic.live_progress import RunProgress

    progress = RunProgress(run_id="r", stages=["ingest", "validate"])
    assert not progress.estimate_is_lower_bound
    assert progress.estimate_has_unmodelled_work
    progress.completed.add("validate")
    assert not progress.estimate_is_lower_bound
    assert not progress.estimate_has_unmodelled_work


def test_the_observed_ratio_is_reported_but_never_applied() -> None:
    """Re-measuring the weights is a person's decision, not the board's."""

    progress = _run()
    progress.begin_stage("ingest")
    baseline = progress.estimated_remaining_seconds()
    progress.finish_stage("ingest")
    record = progress.stage_history["ingest"]
    record.started_at = 0.0
    record.finished_at = 10.0  # absurdly fast against a 1.0 h weight
    ratio = progress.observed_weight_ratio()
    assert ratio is not None and ratio < 0.01
    # The estimate dropped only by ingest's own weight, not by the ratio.
    remaining = progress.estimated_remaining_seconds()
    assert remaining == pytest.approx(
        baseline - stage_weight("ingest") * 3600.0
    )


def test_board_names_the_stage_its_meaning_and_the_share_of_the_run() -> None:
    stream = io.StringIO()
    progress = _run()
    reporter = LiveProgressReporter(
        progress, stream=stream, enable_ansi=False, min_redraw_seconds=0.0
    )
    reporter.begin_stage("extract-features")
    reporter.update(0.25, "5,000/20,000 records")
    text = stream.getvalue()
    assert "OVERALL" in text and "STAGE" in text
    assert "extract-features" in text
    assert STAGE_DESCRIPTIONS["extract-features"] in text
    assert "5,000/20,000 records" in text
    assert "% of run" in text
    assert "[x]" not in text  # nothing finished yet
    reporter.finish_stage("extract-features")
    assert "[x]" in stream.getvalue()


def test_redraw_is_throttled_but_lifecycle_events_always_draw() -> None:
    stream = io.StringIO()
    reporter = LiveProgressReporter(
        _run(), stream=stream, enable_ansi=False, min_redraw_seconds=3600.0
    )
    reporter.begin_stage("ingest")
    after_begin = len(stream.getvalue())
    reporter.update(0.5, "throttled away")
    assert len(stream.getvalue()) == after_begin
    reporter.finish_stage("ingest")
    assert len(stream.getvalue()) > after_begin


def test_state_file_survives_the_terminal_and_renders_from_disk(tmp_path) -> None:
    state_path = tmp_path / "nested" / "live_progress.json"
    reporter = LiveProgressReporter(
        _run(), stream=io.StringIO(), state_path=state_path, enable_ansi=False
    )
    reporter.begin_stage("extract-features")
    reporter.update(0.4, "8,000/20,000 records")
    reporter.finish_run()

    state = read_live_state(state_path)
    assert state is not None
    assert state["current_stage"] == "extract-features"
    assert state["current_stage_description"].startswith("Computing all")
    assert {entry["stage"] for entry in state["stages"]} == {
        "ingest",
        "extract-features",
        "validate",
    }
    assert pytest.approx(sum(e["weight_share"] for e in state["stages"])) == 1.0

    board = render_snapshot(state)
    assert "extract-features" in board and "8,000/20,000 records" in board
    assert "letzte Aktualisierung" in board


def test_unwritable_state_path_never_stops_the_run(tmp_path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    reporter = LiveProgressReporter(
        _run(),
        stream=io.StringIO(),
        state_path=blocker / "live_progress.json",
        enable_ansi=False,
    )
    reporter.begin_stage("ingest")  # must not raise
    reporter.finish_stage("ingest")


def test_missing_or_corrupt_state_file_reads_as_none(tmp_path) -> None:
    assert read_live_state(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_live_state(broken) is None


def test_counts_already_in_stage_messages_are_reused_as_progress() -> None:
    assert parse_fraction("processed 25/100 shards") == pytest.approx(0.25)
    assert parse_fraction("12,500 / 25,000 records") == pytest.approx(0.5)
    assert parse_fraction("no numbers here") is None
    assert parse_fraction("7/0") is None
    # Never above one, even when a stage overshoots its own estimate.
    assert parse_fraction("120/100") == pytest.approx(1.0)


def test_durations_read_as_durations() -> None:
    assert format_duration(None) == "--"
    assert format_duration(45) == "45s"
    assert format_duration(3661).startswith("1h")
    assert format_duration(90000).startswith("1d")


# ------------------------------------------------------------ cross-fitting


def _split_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4, 5],
            "split": [
                "train",
                "train",
                "train",
                "validation",
                "validation",
                "locked_test",
            ],
        }
    )


def test_the_sealed_split_is_never_swappable() -> None:
    assert SWAPPABLE_SPLITS == {"train", "validation"}
    assert SEALED_SPLITS == {"locked_test"}
    assert_locked_test_excluded(["train", "validation"])
    with pytest.raises(PermissionError, match="never include the sealed split"):
        assert_locked_test_excluded(["train", "locked_test"])


def test_plan_covers_both_directions_and_leaves_the_seal_unread() -> None:
    plan = cross_fit_plan(_split_frame())
    assert len(plan) == len(CROSS_FIT_ASSIGNMENTS) == 2
    assert set(plan["fitting_split"]) == {"train", "validation"}
    assert set(plan["confirming_split"]) == {"train", "validation"}
    # Each split is used once for fitting and once for confirming.
    assert (plan["n_fitting_subjects"] + plan["n_confirming_subjects"]).nunique() == 1
    assert not plan["locked_test_read"].any()


def test_a_subject_in_both_folds_is_refused() -> None:
    leaking = _split_frame()
    leaking.loc[0, "split"] = "validation"  # subject 1 now in both
    with pytest.raises(RuntimeError, match="both swappable splits"):
        assert_folds_are_subject_disjoint(leaking)
    with pytest.raises(RuntimeError):
        cross_fit_plan(leaking)
    # A subject shared with the sealed split is a different problem and is not
    # this check's business.
    assert_folds_are_subject_disjoint(pd.DataFrame())


def _fold(auc_a: float, auc_b: float, n: int = 1000) -> dict[str, pd.DataFrame]:
    return {
        "train_fits_validation_confirms": pd.DataFrame(
            {
                "endpoint_id": ["e1"],
                "feature_id": ["f1"],
                "aligned_auc": [auc_a],
                "n_total": [n],
                "p_value": [0.01],
            }
        ),
        "validation_fits_train_confirms": pd.DataFrame(
            {
                "endpoint_id": ["e1"],
                "feature_id": ["f1"],
                "aligned_auc": [auc_b],
                "n_total": [n],
                "p_value": [0.02],
            }
        ),
    }


def test_agreeing_folds_pool_and_are_marked_consistent() -> None:
    pooled = combine_folds(_fold(0.70, 0.66))
    assert len(pooled) == 1
    row = pooled.iloc[0]
    assert row["pooled_effect"] == pytest.approx(0.68)
    assert row["folds_available"] == 2
    assert bool(row["direction_consistent_across_folds"])
    assert row["fold_agreement"] == "consistent"
    assert row["effect_spread"] == pytest.approx(0.04)


def test_a_marker_that_reverses_between_folds_is_flagged_not_averaged_away() -> None:
    pooled = combine_folds(_fold(0.70, 0.40))
    row = pooled.iloc[0]
    # The mean sits near chance, which on its own would read as "weak marker".
    assert row["pooled_effect"] == pytest.approx(0.55)
    assert row["fold_agreement"] == "direction_reversed"
    summary = cross_fit_summary(pooled)
    assert summary["direction_reversed"] == 1
    assert summary["direction_consistent"] == 0
    assert summary["locked_test_read"] is False


def test_legacy_rows_without_uncertainty_use_labelled_sample_size_fallback() -> None:
    folds = _fold(0.80, 0.60)
    folds["validation_fits_train_confirms"]["n_total"] = [3000]
    pooled = combine_folds(folds)
    # 0.80 on 1000 and 0.60 on 3000 -> closer to the larger fold.
    assert pooled.iloc[0]["pooled_effect"] == pytest.approx(0.65)
    assert pooled.iloc[0]["n_confirming_total"] == 4000
    assert pooled.iloc[0]["pooling_weight_method"] == (
        "sample_size_fallback_no_uncertainty"
    )


def test_pooling_uses_auc_uncertainty_not_sample_size_when_available() -> None:
    folds = _fold(0.80, 0.60)
    folds["train_fits_validation_confirms"].loc[:, "roc_auc_se"] = 0.10
    folds["validation_fits_train_confirms"].loc[:, "roc_auc_se"] = 0.05
    # Inverse variance weights 100 and 400 -> 0.64. A sample-size average
    # would be 0.70 because both rows have n=1000.
    pooled = combine_folds(folds)
    assert pooled.iloc[0]["pooled_effect"] == pytest.approx(0.64)
    assert pooled.iloc[0]["pooled_effect_se"] == pytest.approx(1 / (500 ** 0.5))
    assert pooled.iloc[0]["pooling_weight_method"] == "explicit_auc_se"


def test_cross_fit_confirmation_is_intersection_union_not_best_fold() -> None:
    folds = _fold(0.75, 0.70)
    folds["train_fits_validation_confirms"].loc[:, "p_value"] = 0.001
    folds["validation_fits_train_confirms"].loc[:, "p_value"] = 0.20
    row = combine_folds(folds).iloc[0]
    assert row["p_value"] == pytest.approx(0.20)
    assert not bool(row["cross_fit_confirmed_raw"])


def test_a_single_available_fold_is_labelled_as_such() -> None:
    folds = _fold(0.70, 0.70)
    folds.pop("validation_fits_train_confirms")
    pooled = combine_folds(folds)
    assert pooled.iloc[0]["folds_available"] == 1
    assert pooled.iloc[0]["fold_agreement"] == "single_fold"


def test_unusable_effects_are_kept_as_not_evaluable() -> None:
    folds = _fold(float("nan"), float("nan"))
    pooled = combine_folds(folds)
    assert pooled.iloc[0]["fold_agreement"] == "not_evaluable"
    assert pd.isna(pooled.iloc[0]["pooled_effect"])
    empty = combine_folds({"a": pd.DataFrame(), "b": None})
    assert empty.empty and "pooled_effect" in empty.columns
    assert cross_fit_summary(empty)["hypotheses"] == 0


def test_multiplicity_family_spans_the_union_of_the_folds() -> None:
    family = cross_fit_multiplicity_family(combine_folds(_fold(0.7, 0.68)))
    assert family["hypotheses"] == 1
    assert family["folds"] == 2
    assert family["family_definition"] == "union_over_folds_not_per_fold"


def test_the_swap_recovers_fitting_data_without_spending_the_seal() -> None:
    gain = efficiency_gain(60_000, 20_000, 20_000)
    assert gain["fitting_share_single_direction"] == pytest.approx(0.60)
    assert gain["fitting_share_cross_fitted"] == pytest.approx(0.80)
    assert gain["relative_gain"] == pytest.approx(4 / 3)
    # The point of the whole exercise: the sealed fifth stays sealed.
    assert gain["sealed_share_retained"] == pytest.approx(0.20)


def test_key_columns_must_actually_identify_a_hypothesis() -> None:
    with pytest.raises(ValueError, match="key columns"):
        combine_folds(
            {"a": pd.DataFrame({"x": [1], "aligned_auc": [0.7], "n_total": [10]})},
            key_columns=("endpoint_id",),
        )


def test_state_file_is_valid_json_for_external_watchers(tmp_path) -> None:
    path = tmp_path / "live_progress.json"
    reporter = LiveProgressReporter(
        _run(), stream=io.StringIO(), state_path=path, enable_ansi=False
    )
    reporter.begin_stage("ingest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "test-run"
    assert payload["live_progress_version"]


# ------------------------------------------------- the watchable board


def test_the_board_answers_the_questions_a_watcher_actually_has() -> None:
    """A percentage alone is not a status for a four-day run."""

    from vcg_biomarker.mimic.live_progress import render_live_board, stage_weight

    stages = ["ingest", "extract-features", "screen-broad-axes", "audit"]
    progress = RunProgress(run_id="grosser-lauf", stages=stages)
    progress.started_at -= 7200.0
    progress.begin_stage("ingest")
    progress.finish_stage("ingest")
    progress.begin_stage("extract-features")
    progress.update_stage(0.37, "296.013/800.035 EKGs | 6 Worker")
    state = progress.to_dict()
    state["stages"] = [
        {
            "stage": stage,
            "state": (
                "done"
                if stage in progress.completed
                else "running"
                if stage == progress.current
                else "pending"
            ),
            "weight_share": stage_weight(stage) / progress.total_weight,
            "elapsed_seconds": progress.stage_history[stage].elapsed_seconds,
        }
        for stage in stages
    ]
    board = render_live_board(
        state, run_id="grosser-lauf", state_path="x", width=100
    )

    assert "GESAMT" in board and "AKTUELL" in board and "STUFE" in board
    # Which stage, how far into it, and what it is doing right now.
    assert "extract-features" in board
    assert "296.013/800.035 EKGs" in board
    # What is finished, what is queued, and what each cost.
    assert board.count("[x]") == 1
    assert board.count("[ ]") == 2
    assert "fertig" in board and "wartet" in board
    # A run this long needs a wall-clock answer, not only a duration.
    assert "unverbindlicher Planungstermin" in board
    # Long stage ids must not shove the columns out of line.
    rows = [line for line in board.splitlines() if line.startswith("  [")]
    assert len({len(line.rstrip()) - len(line.rstrip().rstrip("0123456789smhd -")) for line in rows}) <= 3


def test_a_run_without_a_status_file_is_explained_not_left_blank() -> None:
    from vcg_biomarker.mimic.live_progress import render_live_board

    board = render_live_board(
        None, run_id="noch-nicht-da", state_path="/pfad/live_progress.json"
    )
    assert "noch-nicht-da" in board
    assert "/pfad/live_progress.json" in board
    # Tell the reader what to try instead of leaving them guessing.
    assert "jobs list" in board


def test_watch_defaults_to_a_readable_interval_without_an_argument() -> None:
    """``--watch`` alone used to be an argparse error."""

    from vcg_biomarker.mimic_cli import _build_parser

    parser = _build_parser()
    bare = parser.parse_args(["progress", "--run-id", "r", "--watch"])
    assert bare.watch == 10.0
    explicit = parser.parse_args(["progress", "--run-id", "r", "--watch", "30"])
    assert explicit.watch == 30.0
    once = parser.parse_args(["progress", "--run-id", "r"])
    assert once.watch is None
    assert once.no_clear is False
    # Width zero means "measure the terminal": a board wider than the window
    # wraps, and the in-place redraw then leaves the top of the previous board
    # on screen, which reads as duplicated headers.
    assert once.width == 0
    assert parser.parse_args(
        ["progress", "--run-id", "r", "--width", "120"]
    ).width == 120
