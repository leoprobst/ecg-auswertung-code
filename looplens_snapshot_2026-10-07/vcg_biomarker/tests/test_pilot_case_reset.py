from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from vcg_biomarker.gui.widgets.pilot_workspace_widget import PilotWorkspaceDialog


def test_clear_covers_markers_and_context_not_only_curves():
    names = ("ecg_plot", "loop_plot", "timeline_plot", "context_table", "context_title",
             "method_comparison_table", "lead_table", "ndat_view", "segment_angle_view",
             "time_slider", "loop_time_gradient", "loop_time_label", "projection_status",
             "lab_status", "loop3d")
    target = SimpleNamespace(**{name: Mock() for name in names}, gl=True,
                             xyz=np.ones((3, 3)), array_data={"old": 1})
    PilotWorkspaceDialog._clear_case_views(target)
    assert target.xyz.shape == (0, 3) and target.array_data == {}
    target.method_comparison_table.setRowCount.assert_called_once_with(0)
    target.context_table.setRowCount.assert_called_once_with(0)
    target.ndat_view.set_result.assert_called_once_with({})
    target.segment_angle_view.set_result.assert_called_once_with({})
    target.loop3d.clear.assert_called_once()
    target.lab_status.setToolTip.assert_called_once_with("")


def test_case_is_cleared_before_load_and_again_after_failure():
    events = []
    def clear():
        events.append("clear")
    def fail():
        events.append("load")
        raise ValueError("damaged archive")
    target = SimpleNamespace(_clear_case_views=clear, _load_case_views=fail,
        case=SimpleNamespace(currentData=lambda: 12), lab_status=Mock())
    PilotWorkspaceDialog.show_case(target)
    assert events == ["clear", "load", "clear"]
    assert "damaged archive" in target.lab_status.setText.call_args.args[0]
    events.clear()
    target.case.currentData = lambda: None
    PilotWorkspaceDialog.show_case(target)
    assert events == ["clear"]
