from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from vcg_biomarker.gui.widgets.pilot_workspace_widget import PilotWorkspaceDialog


def test_reset_removes_table_export_and_plot_state():
    target = SimpleNamespace(plot=Mock(), availability_plot=Mock(), result_table=Mock(),
        summary=Mock(), export_frame=pd.DataFrame({"old": [1]}))
    PilotWorkspaceDialog._clear_result_view(target)
    assert target.export_frame.empty
    target.result_table.setRowCount.assert_called_once_with(0)
    target.plot.clear.assert_called_once()
    target.availability_plot.getAxis("bottom").setTicks.assert_called_once_with([])


def test_failed_result_load_clears_partial_data_again():
    events = []
    def clear():
        events.append("clear")
    def draw():
        events.append("draw")
        raise ValueError("invalid marker artifact")
    target = SimpleNamespace(_clear_result_view=clear, _draw_result_view=draw, summary=Mock())
    PilotWorkspaceDialog.show_result(target)
    assert events == ["clear", "draw", "clear"]
    assert "invalid marker artifact" in target.summary.setText.call_args.args[0]


def test_empty_export_does_not_touch_filesystem_or_model():
    target = SimpleNamespace(export_frame=pd.DataFrame(), summary=Mock())
    # Deliberately no model, path or selected case: guard must run first.
    PilotWorkspaceDialog.export(target)
    assert "Kein Export" in target.summary.setText.call_args.args[0]
