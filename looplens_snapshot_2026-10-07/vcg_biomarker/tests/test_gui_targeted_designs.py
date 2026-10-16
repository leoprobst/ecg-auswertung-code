"""Six stages were added to the backend; the interface has to show them.

The GUI already had one drift of exactly this kind: twelve runnable stages were
missing from its dropdown and four more were offered under an unnamed fallback
label, and nothing failed because the contract check only asked whether every
*offered* stage is runnable. Result views are the same trap one level up -- a
stage can run, write its parquet, and have nowhere to appear.

These tests hold the wiring in place without starting Qt: the widget module is
read as source, which is enough to prove every stage output has a table, every
table has a tooltip that says what its columns mean, and the refresh method
reads the paths the stages actually write.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from vcg_biomarker.gui.widgets import mimic_project_widget as widget_module


TABLES = (
    "control_panel_table",
    "control_panel_shrinkage_table",
    "availability_table",
    "redundancy_table",
    "within_subject_table",
    "voltage_mass_table",
    "voltage_mass_models_table",
    "time_to_event_table",
    "occlusion_yield_table",
    "two_stage_table",
)

# What each stage writes, relative to the run root. Kept here rather than
# imported so that a stage renaming its output breaks this test instead of
# silently emptying a tab.
STAGE_OUTPUTS = {
    "control-panel": "controls/control_panel.parquet",
    "within-subject-change": "within_subject/within_subject_change.parquet",
    "voltage-mass-discordance": "voltage_mass/voltage_mass_discordance.parquet",
    "time-to-event": "survival/time_to_event.parquet",
    "occlusion-yield": "occlusion/criterion_yield.parquet",
    "two-stage-selection": "two_stage/confirmation_plan.parquet",
}


def _widget_source() -> str:
    return Path(widget_module.__file__).read_text(encoding="utf-8")


def _tooltip(table: str) -> str:
    """The tooltip text of one table.

    Reads to the closing parenthesis at the call's own indentation rather than
    the first one found: the tooltips contain parentheses of their own -- the
    dose-arm text says "(Betablocker senkt die Herzfrequenz)" -- and stopping
    there truncated the tooltip mid-sentence and failed a test about text that
    was present.
    """

    source = _widget_source()
    marker = f"self.{table}.setToolTip("
    assert marker in source, f"{table} hat keinen Tooltip"
    start = source.index(marker)
    end = source.index(chr(10) + "        )", start)
    return source[start:end]


# --- Every stage output has somewhere to appear -------------------------------


@pytest.mark.parametrize("table", TABLES)
def test_the_table_exists(table):
    assert f"self.{table} = PagedParquetTable" in _widget_source()


@pytest.mark.parametrize("stage, output", sorted(STAGE_OUTPUTS.items()))
def test_the_stage_output_is_read_by_the_interface(stage, output):
    source = inspect.getsource(widget_module.MimicProjectWidget._refresh_targeted_designs)
    for part in output.split("/"):
        assert part in source, f"{stage}: {part} wird nicht gelesen"


@pytest.mark.parametrize("table", TABLES)
def test_every_table_explains_its_columns(table):
    """A table of unexplained column names is not a result view."""

    tooltip = _tooltip(table)
    assert len(tooltip) > 200, f"{table}: Tooltip zu knapp, um etwas zu erklaeren"


def test_both_new_tabs_are_added():
    source = _widget_source()
    assert 'tabs.addTab(controls, "Kontrolltafel")' in source
    assert 'tabs.addTab(designs, "Zielgerichtete Designs")' in source


def test_the_new_tabs_carry_their_own_explanations():
    source = _widget_source()
    for title in ("Kontrolltafel", "Zielgerichtete Designs"):
        assert f'"{title}": (' in source, f"{title} fehlt im Tooltip-Verzeichnis"


# --- What the views must not let a reader believe -----------------------------


def test_the_occlusion_view_carries_the_surrogate_caveat():
    """81,3 % is not "STEMI criteria miss 81 % of occlusions", and the number
    must never appear without what the endpoint actually is."""

    tooltip = _tooltip("occlusion_yield_table")
    assert "Surrogat" in tooltip
    assert "angiographisch" in tooltip.lower()

    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "Surrogat" in refresh


def test_dose_controls_are_run_specific_not_a_historical_fixed_result():
    """An old pilot p value cannot describe every subsequently selected run."""

    tooltip = _tooltip("within_subject_table")
    assert "Positivkontrolle" in tooltip
    assert "Nullbefund" in tooltip
    assert "ausgewählten Lauf" in tooltip
    assert "erwartete Richtung, aber keine Signifikanz" not in tooltip


def test_the_voltage_mass_view_demands_the_ratio_beat_its_parts():
    tooltip = _tooltip("voltage_mass_table")
    assert "auc_voltage" in tooltip and "auc_mass" in tooltip
    assert "gain_over_best_part" in tooltip


def test_the_two_stage_view_separates_the_two_arms():
    tooltip = _tooltip("two_stage_table")
    assert "prespecified_curated" in tooltip
    assert "two_stage_confirmatory" in tooltip


def test_a_control_pointing_the_wrong_way_is_called_out():
    """Recovering fewer controls is a weak run; a reversed one is a defect."""

    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "n_positive_wrong_direction" in refresh
    assert "falsche Richtung" in refresh


# --- Older runs must not break the view ---------------------------------------


def test_a_run_without_the_new_stages_shows_a_hint_rather_than_failing():
    """Every existing run predates all six stages."""

    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "is_file()" in refresh
    assert "Noch keine" in refresh


def test_the_missing_hint_says_the_stages_can_be_added_afterwards():
    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "nachtragen" in refresh


# --- The pictures, and what each is there to show -----------------------------


PLOTS = (
    "survival_plot",
    "hazard_forest_plot",
    "voltage_mass_plot",
    "within_subject_plot",
    "availability_plot",
)


@pytest.mark.parametrize("plot", PLOTS)
def test_the_plot_exists(plot):
    assert f"self.{plot} = pg.PlotWidget()" in _widget_source()


@pytest.mark.parametrize("plot", PLOTS)
def test_the_plot_is_drawn_by_something(plot):
    """A widget nothing draws into is an empty rectangle forever."""

    source = _widget_source()
    assert f"self.{plot}.clear()" in source


def test_the_availability_view_draws_both_groups_side_by_side():
    """The gap between the bars is the finding; one bar cannot show it."""

    source = inspect.getsource(
        widget_module.MimicProjectWidget._redraw_segment_availability
    )
    assert "available_positive_pct" in source
    assert "available_negative_pct" in source
    assert "BarGraphItem" in source


def test_the_availability_view_offers_the_worst_endpoint_first():
    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "ascending=False" in refresh


def test_the_survival_view_warns_about_non_monotone_markers():
    """A hazard ratio assumes a linear course; the curves can show it is not.

    The assertion is on the warning, not on the figures that prompted it. An
    earlier version pinned the tertile percentages of one particular marker
    from one particular run into the interface text, which made the test pass
    and the label go stale on the next run.
    """

    source = _widget_source()
    assert "monoton" in source
    assert "Hazard Ratio" in source
    assert "linearen" in source


def test_the_forest_plot_marks_the_line_of_no_effect():
    source = inspect.getsource(
        widget_module.MimicProjectWidget._redraw_hazard_forest
    )
    assert "InfiniteLine" in source
    assert "pos=1.0" in source


def test_the_forest_plot_shows_both_populations():
    source = inspect.getsource(
        widget_module.MimicProjectWidget._redraw_hazard_forest
    )
    assert "erstes ECG" in source
    assert "robust" in source


def test_the_scatter_plots_are_thinned_rather_than_drawn_whole():
    """173.374 points draw slowly and a reader cannot see through them."""

    source = inspect.getsource(widget_module.MimicProjectWidget._thinned)
    assert "step" in source
    assert "_SCATTER_POINT_LIMIT" in _widget_source()


def test_the_multiplicity_correction_reaches_the_summary():
    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert "correct_across_arms" in refresh
    assert "Rauschboden" in refresh


def test_a_broken_artefact_does_not_take_the_view_down():
    """Every read of a parquet in the refresh is guarded."""

    refresh = inspect.getsource(
        widget_module.MimicProjectWidget._refresh_targeted_designs
    )
    assert refresh.count("try:") >= refresh.count("pd.read_parquet(")


def test_the_refresh_is_actually_called():
    """A method nothing calls is a tab that stays empty forever."""

    source = _widget_source()
    assert source.count("_refresh_targeted_designs") >= 2
