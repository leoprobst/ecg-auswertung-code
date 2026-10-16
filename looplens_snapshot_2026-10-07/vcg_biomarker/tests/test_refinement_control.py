"""One control for every stage that has no natural end.

The combination search and the marker bootstrap are both open-ended: either can
always be pushed further, and both are meant to be stopped when the user has
seen enough and continued later. They grew separate stop files, separate live
status and separate buttons, which is two controls for what is one activity
from the user's seat.

What the tests below pin is not the plumbing but the promises: a stop reaches a
stage started from older code, progress is never reported as a percentage of an
unknown total, and a run ended early can never be mistaken for a finished one.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.refinement import (
    DEFAULT_STOP_MODE,
    LIVE_FILE_NAME,
    STOP_FILE_NAME,
    STOP_MODE_AFTER_ENDPOINT,
    STOP_MODE_AFTER_SWEEP,
    STOP_MODE_IMMEDIATE,
    RefinementActivity,
    clear_stop,
    describe_activity,
    publish_activity,
    read_activity,
    refinement_has_saturated,
    request_stop,
    requested_stop_mode,
    stop_file_requested,
)


class TestStopRequest:
    def test_no_request_reads_as_no_stop(self, tmp_path):
        assert requested_stop_mode(tmp_path) is None
        assert not stop_file_requested(tmp_path)

    @pytest.mark.parametrize(
        "mode",
        [STOP_MODE_IMMEDIATE, STOP_MODE_AFTER_ENDPOINT, STOP_MODE_AFTER_SWEEP],
    )
    def test_each_mode_round_trips(self, tmp_path, mode):
        request_stop(tmp_path, mode=mode)
        assert requested_stop_mode(tmp_path) == mode

    def test_an_unknown_mode_is_refused_at_the_source(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown stop mode"):
            request_stop(tmp_path, mode="halt_and_catch_fire")

    def test_a_damaged_stop_file_reads_as_the_safe_mode(self, tmp_path):
        """A stop must never fail open, and never fail abruptly either."""

        (tmp_path / STOP_FILE_NAME).write_text("{ truncated", encoding="utf-8")
        assert requested_stop_mode(tmp_path) == DEFAULT_STOP_MODE

    def test_an_unrecognised_mode_falls_back_rather_than_aborting(self, tmp_path):
        (tmp_path / STOP_FILE_NAME).write_text(
            json.dumps({"mode": "from_a_future_version"}), encoding="utf-8"
        )
        assert requested_stop_mode(tmp_path) == DEFAULT_STOP_MODE

    def test_clearing_lets_the_run_be_continued(self, tmp_path):
        """Continuing is the normal case; the request must not outlive its stop."""

        request_stop(tmp_path, mode=STOP_MODE_AFTER_SWEEP)
        clear_stop(tmp_path)
        assert requested_stop_mode(tmp_path) is None

    def test_clearing_an_absent_request_is_not_an_error(self, tmp_path):
        clear_stop(tmp_path)

    def test_the_file_name_is_unchanged_from_the_deep_search(self):
        """A stop from a new GUI has to reach a search already running."""

        from vcg_biomarker.mimic import combination_stage

        assert combination_stage.STOP_FILE_NAME == STOP_FILE_NAME
        assert STOP_FILE_NAME == "STOP_DEEP_SEARCH"


class TestLiveActivity:
    def test_nothing_published_reads_as_nothing_running(self, tmp_path):
        assert read_activity(tmp_path) is None
        assert "Keine offene Verfeinerung" in describe_activity(None)

    def test_activity_round_trips(self, tmp_path):
        publish_activity(
            tmp_path,
            RefinementActivity(
                stage="validate",
                unit_label="Endpunkt",
                position=7,
                total=72,
                detail="heart_failure",
                gain=0.00031,
                gain_label="Verschiebung",
            ),
        )
        state = read_activity(tmp_path)
        assert state["position"] == 7
        assert state["total"] == 72
        assert state["detail"] == "heart_failure"
        line = describe_activity(state)
        assert "7/72" in line
        assert "heart_failure" in line
        assert "0.00031" in line

    def test_the_older_deep_search_file_is_still_read(self, tmp_path):
        """A GUI must not go blank against a stage from before this module."""

        (tmp_path / "deep_search_live.json").write_text(
            json.dumps(
                {
                    "stage": "deep-combination-search",
                    "endpoint_position": 3,
                    "endpoints_total": 40,
                    "current_depth": 4,
                }
            ),
            encoding="utf-8",
        )
        state = read_activity(tmp_path)
        assert state is not None
        line = describe_activity(state)
        assert "3/40" in line
        assert "Stufe 4" in line

    def test_the_current_file_wins_over_the_older_one(self, tmp_path):
        (tmp_path / "deep_search_live.json").write_text(
            json.dumps({"endpoint_position": 1, "endpoints_total": 9}),
            encoding="utf-8",
        )
        publish_activity(
            tmp_path,
            RefinementActivity(
                stage="validate", unit_label="Endpunkt", position=5, total=9
            ),
        )
        assert read_activity(tmp_path)["position"] == 5

    def test_a_damaged_live_file_reads_as_no_state(self, tmp_path):
        (tmp_path / LIVE_FILE_NAME).write_text("{ truncated", encoding="utf-8")
        assert read_activity(tmp_path) is None

    def test_progress_is_never_a_percentage_of_an_unknown_total(self, tmp_path):
        """There is no denominator: the work is unbounded by design."""

        publish_activity(
            tmp_path,
            RefinementActivity(
                stage="deep-combination-search",
                unit_label="Endpunkt",
                position=2,
                total=0,
                level=6,
            ),
        )
        line = describe_activity(read_activity(tmp_path))
        assert "%" not in line
        assert "Stufe 6" in line


class TestSaturationHint:
    def test_nothing_to_compare_yet_says_so(self):
        assert refinement_has_saturated(None, threshold=0.001) is None

    def test_a_small_gain_reads_as_saturated(self):
        assert refinement_has_saturated(0.00003, threshold=0.001) is True

    def test_a_large_gain_reads_as_still_improving(self):
        assert refinement_has_saturated(0.02, threshold=0.001) is False


def test_a_validation_ended_early_cannot_look_finished():
    """The guard that keeps 'never looked' apart from 'nothing found'."""

    from vcg_biomarker.mimic.validation import PartialValidationStopped

    error = PartialValidationStopped(
        "beendet nach 7 von 72",
        completed_endpoint_ids=("a", "b"),
        requested_endpoint_ids=("a", "b", "c"),
    )
    assert isinstance(error, RuntimeError)
    assert error.completed_endpoint_ids == ("a", "b")
    assert len(error.requested_endpoint_ids) > len(error.completed_endpoint_ids)
