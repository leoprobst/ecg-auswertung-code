import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace

import pandas as pd
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QTableWidget

from vcg_biomarker.gui.widgets.mimic_project_widget import MimicProjectWidget
from vcg_biomarker.mimic.result_explorer import build_result_explorer_tables


@pytest.mark.parametrize("excluded,raw", [(float('inf'), 100), ('unknown', 100),
    (True, 100), (2.5, 100), (-1, 100), (101, 100), (None, 100)])
def test_invalid_projection_counts_are_explicit_without_crashing(excluded, raw):
    app = QApplication.instance() or QApplication([])
    table = QTableWidget()
    frame = pd.DataFrame([{'feature_name_de': 'Test', 'projection_plane': 'horizontal_xz',
        'projection_hypothesis_restriction_operationalized': 'applied',
        'n_analysis_restriction_excluded': excluded, 'n_marker_raw_observed': raw}])
    MimicProjectWidget._set_primary_result_frame(SimpleNamespace(), table, frame, 'biomarkers')
    labels = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    tooltip = table.item(0, labels.index('2D-Restriktionsstatus')).toolTip()
    assert 'ungültig' in tooltip and 'nicht als null' in tooltip
    table.close()
    app.processEvents()


def test_projection_columns_and_limits_are_visible_with_nonrange_source_index():
    app = QApplication.instance() or QApplication([])
    table = QTableWidget()
    frame = pd.DataFrame([{
        "feature_name_de": "T-Loop-Fläche", "endpoint_name_de": "Hyperkaliämie",
        "projection_plane": "horizontal_xz", "projection_hypothesis_q_value": .012,
        "projection_hypothesis_restriction_operationalized":
            "true_marker_availability_and_endpoint_window",
        "n_analysis_restriction_excluded": 7,
        "n_marker_raw_observed": 107,
        "feature_tooltip_de": "Markerbeschreibung",
    }], index=[99])
    MimicProjectWidget._set_primary_result_frame(SimpleNamespace(), table, frame, "biomarkers")
    fields = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert "2D-Ebene" in fields and "2D-Hypothese FDR q" in fields
    first = table.item(0, 0)
    assert first.text() == "T-Loop-Fläche"
    assert first.toolTip() == "Markerbeschreibung"
    source = first.data(Qt.ItemDataRole.UserRole)
    assert source["projection_plane"] == "horizontal_xz"
    q_item = table.item(0, fields.index("2D-Hypothese FDR q"))
    assert "ersetzt weder globales FDR" in q_item.toolTip()
    restriction_item = table.item(0, fields.index("2D-Restriktionsstatus"))
    assert "7 von 107" in restriction_item.toolTip()
    table.close(); app.processEvents()


def test_historical_frame_does_not_gain_empty_projection_columns():
    app = QApplication.instance() or QApplication([])
    table = QTableWidget()
    frame = pd.DataFrame([{"feature_name_de": "Alt", "endpoint_name_de": "Alt",
                           "projection_plane": None,
                           "projection_hypothesis_q_value": float("nan")}])
    MimicProjectWidget._set_primary_result_frame(SimpleNamespace(), table, frame, "biomarkers")
    labels = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert all(not label.startswith("2D-") for label in labels)
    table.close(); app.processEvents()


def test_result_explorer_preserves_projection_contract_for_gui():
    screening = pd.DataFrame([{
        "endpoint_id": "hyperkalemia_near_ecg", "feature_id": "t_loop_horizontal_xz_area",
        "panel_id": "primary_core", "evaluation_split": "validation", "analysis_type": "single_marker",
        "inference_status": "formal_inference", "n_positive": 50, "n_negative": 200,
        "roc_auc": .7, "q_value": .02, "projection_hypothesis_prespecified": True,
        "projection_plane": "horizontal_xz", "projection_hypothesis_q_value": .03,
        "projection_hypothesis_restriction_operationalized":
            "true_marker_availability_and_endpoint_window",
        "projection_hypothesis_version": "1.0.0",
        "n_marker_raw_observed": 250,
        "n_marker_observed": 240,
        "n_marker_unavailable": 10,
        "n_analysis_restriction_excluded": 0,
        "analysis_restriction_status": "not_applicable",
        "analysis_restriction_version": "1.0.0",
    }])
    row = build_result_explorer_tables(screening).marker_pathology.iloc[0]
    assert row.projection_plane == "horizontal_xz"
    assert row.projection_hypothesis_q_value == .03
    assert bool(row.projection_hypothesis_prespecified)
    assert row.n_marker_raw_observed == 250
    assert row.n_marker_unavailable == 10
