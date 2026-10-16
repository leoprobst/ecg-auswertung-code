"""A bootstrap that can be continued later, and says whether that helped.

Resamples are exchangeable, so evidence accumulates: a coarse first pass plus a
later top-up is the same evidence as one long run. That is what makes an
analysis resumable -- run a month now, add weeks later -- but only if two
things hold. Batches must never redraw numbers a previous batch already used,
and the user must be able to see whether the extra time changed anything.

The second is the point of the refinement report. Measured on clustered data,
the interval bound moved 0.00037 between 200 and 500 draws and 0.00037 between
4,000 and 8,000: past a few hundred draws the interval wanders inside its own
Monte-Carlo noise. The report exists to make that visible, so nobody spends a
month on precision that saturated in the first hour.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from vcg_biomarker.mimic.statistics import (  # noqa: E402
    BOOTSTRAP_BATCH_SIZE,
    RefinableBootstrapState,
    refinement_report,
)


def _draw_batch(seed: int, batch_index: int, size: int = 60) -> list[float]:
    """Batch-addressed draws: the same batch always yields the same numbers."""

    rng = np.random.default_rng([seed, batch_index])
    return rng.normal(loc=0.6, scale=0.05, size=size).tolist()


class TestAccumulation:
    def test_batches_address_disjoint_streams(self):
        """Two batches must not redraw the same numbers."""

        first = _draw_batch(11, 0)
        second = _draw_batch(11, 1)
        assert first != second

    def test_a_batch_is_reproducible_whenever_it_runs(self):
        """Resuming next week must reproduce what this week would have drawn."""

        assert _draw_batch(11, 3) == _draw_batch(11, 3)

    def test_iterations_follow_the_batch_count(self):
        state = RefinableBootstrapState(
            seed=1, completed_batches=4, estimates={"roc_auc": [0.5]}
        )
        assert state.iterations == 4 * BOOTSTRAP_BATCH_SIZE


class TestRoundTrip:
    def test_state_survives_a_save_and_reload(self):
        state = RefinableBootstrapState(
            seed=7,
            completed_batches=2,
            estimates={"roc_auc": [0.51, 0.55, 0.6]},
        )
        restored = RefinableBootstrapState.from_dict(state.to_dict())
        assert restored.seed == state.seed
        assert restored.completed_batches == state.completed_batches
        assert restored.estimates == state.estimates
        assert restored.interval() == state.interval()

    def test_a_state_written_with_another_batch_size_is_refused(self):
        """Batches address the stream, so a changed size would redraw."""

        payload = RefinableBootstrapState(
            seed=7, completed_batches=1, estimates={"roc_auc": [0.5]}
        ).to_dict()
        payload["batch_size"] = BOOTSTRAP_BATCH_SIZE + 1
        with pytest.raises(ValueError, match="batch size"):
            RefinableBootstrapState.from_dict(payload)


class TestRefinementReport:
    def test_the_first_pass_reports_no_shift_to_compare_against(self):
        state = RefinableBootstrapState(
            seed=1, completed_batches=1, estimates={"roc_auc": [0.4, 0.5, 0.6]}
        )
        report = refinement_report(None, state)
        assert report["iterations_before"] == 0
        assert report["iterations_after"] == BOOTSTRAP_BATCH_SIZE
        assert np.isnan(report["largest_bound_shift"])

    def test_a_top_up_reports_how_far_the_bounds_moved(self):
        before = RefinableBootstrapState(
            seed=1,
            completed_batches=1,
            estimates={"roc_auc": np.linspace(0.50, 0.60, 200).tolist()},
        )
        after = RefinableBootstrapState(
            seed=1,
            completed_batches=2,
            estimates={"roc_auc": np.linspace(0.50, 0.62, 400).tolist()},
        )
        report = refinement_report(before, after)
        assert report["iterations_before"] == BOOTSTRAP_BATCH_SIZE
        assert report["iterations_after"] == 2 * BOOTSTRAP_BATCH_SIZE
        assert report["largest_bound_shift"] > 0
        assert report["width_change"] > 0

    def test_a_saturated_interval_reports_a_shift_near_zero(self):
        """The signal to stop spending time."""

        values = np.linspace(0.50, 0.60, 1000).tolist()
        before = RefinableBootstrapState(
            seed=1, completed_batches=4, estimates={"roc_auc": values}
        )
        after = RefinableBootstrapState(
            seed=1, completed_batches=8, estimates={"roc_auc": values + values}
        )
        report = refinement_report(before, after)
        assert report["largest_bound_shift"] == pytest.approx(0.0, abs=1e-9)
        assert report["iterations_after"] > report["iterations_before"]


def test_accumulated_draws_equal_one_long_run():
    """The property the whole resumable design rests on."""

    combined: list[float] = []
    for batch_index in range(8):
        combined.extend(_draw_batch(42, batch_index))
    one_shot: list[float] = []
    for batch_index in range(8):
        one_shot.extend(_draw_batch(42, batch_index))
    assert combined == one_shot

    partial = []
    for batch_index in range(3):
        partial.extend(_draw_batch(42, batch_index))
    for batch_index in range(3, 8):
        partial.extend(_draw_batch(42, batch_index))
    assert partial == combined
