import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication
from vcg_biomarker.gui.widgets.lead_comparison_widget import LeadComparisonWidget


def test_anchor_comparison_uses_its_own_effect_and_rejects_ambiguous_anchor():
    app = QApplication.instance() or QApplication([])
    widget = LeadComparisonWidget()
    row = {"endpoint_id": "a", "family": "qrs", "lead": "I", "status": "estimable",
        "n_validation_common": 100, "reference_kind": "simulation",
        "delta_auc_clinical_baseline": .1, "anchor_selected_in_train": "II",
        "delta_auc_train_anchor": -.03, "delta_auc_train_anchor_ci_low": -.07,
        "delta_auc_train_anchor_ci_high": .01}
    widget.set_frame(pd.DataFrame([row]))
    widget.comparator.setCurrentIndex(1)
    app.processEvents()
    assert widget.rendered_effects == [{"lead": "I", "value": -.03, "interval_drawn": True}]
    assert "keine Neuauswahl" in widget.summary.text()
    widget.set_frame(pd.DataFrame([row, {**row, "lead": "III", "status": "non_estimable",
                                        "anchor_selected_in_train": None}]))
    assert len(widget.rendered_effects) == 1
    widget.set_frame(pd.DataFrame([row, {**row, "lead": "III", "anchor_selected_in_train": None}]))
    assert not widget.rendered_effects
    widget.set_frame(pd.DataFrame([row, {**row, "lead": "III", "anchor_selected_in_train": "V1"}]))
    assert not widget.rendered_effects
    assert "widersprüchlich" in widget.summary.text()
    widget.set_frame(pd.DataFrame([{key: value for key, value in row.items()
                                  if not key.startswith("delta_auc_train_anchor")}]))
    assert not widget.rendered_effects
    widget.close()


def test_descriptive_ranking_preserves_ties_and_abstentions():
    from vcg_biomarker.gui.widgets.lead_comparison_widget import _rank_lead_results
    frame = pd.DataFrame([
        {"lead": lead, "status": status, "validation_auc": auc}
        for lead, status, auc in [("I", "estimable", .6), ("II", "estimable", .8),
                                  ("III", "estimable", .8), ("V1", "non_estimable", .99),
                                  ("V2", "estimable", None), ("V3", "estimable", 1.5)]])
    ranked = _rank_lead_results(frame).set_index("lead")
    assert ranked.index[:3].tolist() == ["II", "III", "I"]
    assert ranked.loc[["II", "III", "I"], "validation_auc_rank"].tolist() == [1, 1, 3]
    assert ranked.loc[["V1", "V2", "V3"], "validation_auc_rank"].isna().all()


def test_lead_view_handles_missing_partial_and_ambiguous_results(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = LeadComparisonWidget()
    widget.set_source(tmp_path/"missing.parquet")
    assert "Keine berechneten" in widget.note.text()
    frame = pd.DataFrame([{"endpoint_id": "echo_test", "family": "qrs", "lead": "I", "status": "estimable",
        "n_validation_common": 100, "reference_kind": "quantitative_reference",
        "delta_auc_clinical_baseline": .05}])
    widget.set_frame(frame)
    app.processEvents()
    assert widget.table.rowCount() == 1
    assert "Keine Freigabe" in widget.note.text()
    assert "Basis + Ableitung" in widget.note.text()
    assert "keine Trefferquote" in widget.note.text()
    with pytest.raises(ValueError, match="Mehrdeutige"):
        widget.set_frame(pd.concat([frame, frame]))
    with pytest.raises(ValueError, match="Inkompatibler"):
        widget.set_frame(pd.DataFrame({"unknown": [1]}))
    widget.close()


def test_sequential_missingness_counts_are_visible_without_inventing_old_counts():
    app = QApplication.instance() or QApplication([])
    widget = LeadComparisonWidget()
    row = {"endpoint_id": "test", "family": "qrs", "lead": "I", "status": "non_estimable",
           "n_validation_common": 0, "reference_kind": "simulation", "n_index": 10,
           "n_excluded_reference": 3, "n_excluded_extraction_after_reference": 2,
           "n_excluded_covariates_after_extraction": 1, "n_excluded_leads_after_covariates": 4,
           "n_common_complete": 0}
    row["delta_auc_clinical_baseline"] = .02
    widget.set_frame(pd.DataFrame([row]))
    app.processEvents()
    fields = [widget.table.horizontalHeaderItem(i).toolTip() for i in range(widget.table.columnCount())]
    assert widget.table.item(0, fields.index("n_excluded_reference")).text() == "3"
    assert fields.index("delta_auc_clinical_baseline") < fields.index("n_excluded_reference")
    old = {key: value for key, value in row.items() if not key.startswith("n_excluded")}
    widget.set_frame(pd.DataFrame([old]))
    fields = [widget.table.horizontalHeaderItem(i).toolTip() for i in range(widget.table.columnCount())]
    assert "n_excluded_reference" not in fields
    widget.close()


def test_filters_only_offer_available_families_and_clear_stale_axes():
    app = QApplication.instance() or QApplication([])
    widget = LeadComparisonWidget()
    rows = [{"endpoint_id": endpoint, "family": family, "lead": "I", "status": "estimable",
             "n_validation_common": 50, "reference_kind": "quantitative_reference",
             "delta_auc_clinical_baseline": .1} for endpoint, family in [("a", "qrs"), ("b", "p")]]
    widget.set_frame(pd.DataFrame(rows))
    assert widget.family.currentText() == "qrs" and widget.family.count() == 1
    widget.endpoint.setCurrentText("b"); app.processEvents()
    assert widget.family.currentText() == "p" and widget.family.count() == 1
    assert widget.table.rowCount() == 1
    widget.set_frame(pd.DataFrame())
    assert not widget.rendered_effects and not widget.summary.text()
    assert widget.table.columnCount() == 0
    widget.close()


@pytest.mark.parametrize("low,high,expected", [(.01, .2, True), (.01, None, False),
    (float("inf"), .2, False), (.3, .1, False), (-2, 2, False), ("bad", .2, False)])
def test_partial_or_invalid_intervals_never_look_complete(low, high, expected):
    app = QApplication.instance() or QApplication([])
    widget = LeadComparisonWidget()
    widget.set_frame(pd.DataFrame([{"endpoint_id": "a", "family": "qrs", "lead": "I", "status": "estimable",
        "n_validation_common": 123, "reference_kind": "quantitative_reference",
        "delta_auc_clinical_baseline": .1, "delta_auc_clinical_baseline_ci_low": low,
        "delta_auc_clinical_baseline_ci_high": high}]))
    app.processEvents()
    assert widget.rendered_effects == [{"lead": "I", "value": .1, "interval_drawn": expected}]
    widget.close()
