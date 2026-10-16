"""Real project widget and paged Parquet selection, in an isolated project."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QApplication

from vcg_biomarker.gui.widgets import mimic_project_widget
from vcg_biomarker.tests.test_provenance_integrity import _config


def test_project_selection_and_source_reset_update_effect_view(tmp_path, monkeypatch):
    config = _config(tmp_path/"isolated")
    monkeypatch.setattr(mimic_project_widget, "load_or_discover_config", lambda: config)
    app = QApplication.instance() or QApplication([])
    widget = mimic_project_widget.MimicProjectWidget()
    widget._poll_timer.stop()  # isolate selection from unrelated periodic discovery
    try:
        first = dict(design="dialysis_potassium", feature_id="synthetic_first",
            association_status="estimable", spearman_rho=.4, n_pairs=100, n_patients=50,
            adjusted_model="timing_quality_sensitivity_only", adjusted_status="estimable",
            adjusted_rank_slope=.2, adjusted_ci_low=.1, adjusted_ci_high=.3,
            adjusted_n_complete=80, adjusted_n_patients=40, adjusted_global_q_value=.01,
            unadjusted_same_sample_rank_slope=.35, adjusted_abstention_reason="")
        second = {**first, "feature_id": "synthetic_second", "adjusted_status": "non_estimable",
                  "adjusted_abstention_reason": "collinear_or_ill_conditioned_design"}
        source = tmp_path/"effects.parquet"
        pd.DataFrame([first, second]).to_parquet(source, index=False)
        widget.within_subject_table.set_source(source)
        widget.within_subject_table.selectRow(0)
        app.processEvents()
        assert "synthetic_first" in widget.within_effect_view.title.text()
        assert len(widget.within_effect_view.rendered_effects) == 3
        widget.within_subject_table.selectRow(1)
        app.processEvents()
        assert "synthetic_second" in widget.within_effect_view.title.text()
        assert not any(point["row"] == 2 for point in widget.within_effect_view.rendered_effects)
        assert "collinear" in widget.within_effect_view.table.item(2, 4).text()
        widget.within_subject_table.set_source(None)
        app.processEvents()
        assert widget.within_effect_view.rendered_effects == []
        assert widget.within_effect_view.plot.isHidden()
        widget._within_subject_changes = pd.DataFrame(dict(delta_lab=[1., 2., np.inf],
            delta_marker=[2., 3., 4.], subject_id=[1, 1, 2], analyte_id="potassium",
            analyte_unit="mmol/l", lab_pair_status="eligible"))
        widget.within_analyte_combo.clear()
        widget.within_analyte_combo.addItem("Kalium", "potassium")
        widget.within_marker_combo.clear()
        widget.within_marker_combo.addItem("Synthetic marker", "marker")
        widget._SCATTER_POINT_LIMIT = 1
        widget._redraw_within_subject_scatter()
        assert widget._within_scatter_counts["n_total"] == 3
        assert widget._within_scatter_counts["n_valid"] == 2
        assert widget._within_scatter_counts["n_shown"] == 1
        data = widget.within_subject_plot.listDataItems()[0].getData()
        assert len(data[0]) == 1 and np.isfinite(data[0]).all()
        widget._within_subject_changes = None
        widget._redraw_within_subject_scatter()
        assert widget.within_subject_plot.listDataItems() == []
        assert widget._within_scatter_counts == {}
    finally:
        widget.close()
