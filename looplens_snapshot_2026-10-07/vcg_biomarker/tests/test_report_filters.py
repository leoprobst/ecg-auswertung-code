import pandas as pd

from vcg_biomarker.mimic.models import ReportSpec, ReportType
from vcg_biomarker.mimic.report_builder import _feature_members, _prepare_table


def test_marker_report_filters_direct_and_combination_components() -> None:
    frame = pd.DataFrame(
        {
            "endpoint_id": ["a", "a", "b"],
            "feature_id": ["marker_a", "marker_b", "score_ab"],
            "components": ["[]", "[]", '["marker_a", "marker_c"]'],
            "panel_id": ["primary_core", "sentinel", "primary_core"],
        }
    )
    spec = ReportSpec(
        report_id="marker-a",
        report_type=ReportType.MARKER,
        run_id="development",
        formats=("html",),
        feature_ids=("marker_a",),
    )

    filtered = _prepare_table(frame, name="screening_results", spec=spec)

    assert filtered["feature_id"].tolist() == ["marker_a", "score_ab"]
    assert spec.to_dict()["feature_ids"] == ("marker_a",)


def test_feature_member_parser_prefers_structured_json_and_known_delimiters() -> None:
    assert _feature_members('["a", "b"]') == ("a", "b")
    assert _feature_members("a|b") == ("a", "b")
    assert _feature_members(["a", "b"]) == ("a", "b")
