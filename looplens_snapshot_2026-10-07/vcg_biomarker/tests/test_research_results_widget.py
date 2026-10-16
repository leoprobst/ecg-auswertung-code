import json
import pytest
from PyQt6.QtWidgets import QApplication
from vcg_biomarker.gui.widgets.research_results_widget import (
    ResearchResultsWidget,
    CaseResearchStatusWidget,
    research_rows,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_extension_report_counts_are_not_patients():
    from vcg_biomarker.mimic.pre_run_extensions import research_contract

    report = dict(contract=research_contract(), clinical_claim_allowed=False,
                  locked_test_read=False, status="complete",
                  tasks=[dict(kind="personal_deviation", n_rows=100)])
    row = research_rows(report)[0]
    assert row["value"] == 100
    assert row["n"] is None
    assert row["ci"] is None
    report["status"] = "running"
    with pytest.raises(ValueError):
        research_rows(report)


def test_extension_run_selection_clears_old_results(app, tmp_path):
    from vcg_biomarker.mimic.pre_run_extensions import research_contract

    root = tmp_path / "pre_run_extensions" / "job"
    root.mkdir(parents=True)
    (root / "report.json").write_text(json.dumps(dict(
        contract=research_contract(), clinical_claim_allowed=False,
        locked_test_read=False, status="complete",
        tasks=[dict(kind="long_windows", n_rows=6)])), encoding="utf-8")
    widget = ResearchResultsWidget()
    widget.set_run_root(tmp_path)
    assert widget.table.rowCount() == 1
    widget.set_run_root(None)
    assert widget.table.rowCount() == 0
    widget.close()


def payload():
    return {
        "version": "1.0.0",
        "clinical_claim_allowed": False,
        "task_registry": {},
        "results": [
            {
                "endpoint_id": "Reduzierte LVEF (synthetisch)",
                "models": {
                    "vcg": {
                        "evaluation": {
                            "metrics": {
                                "auc": 0.78,
                                "auc_ci95": [0.7, 0.85],
                                "n_patients": 123,
                                "calibration_bins": [
                                    {
                                        "predicted": 0.4,
                                        "observed": 0.38,
                                        "observed_ci95": [0.3, 0.5],
                                    }
                                ],
                            }
                        }
                    }
                },
            }
        ],
    }


def test_missing_metrics_are_not_neutral_results():
    data = payload()
    data["results"][0]["models"]["vcg"] = {
        "status": "non_estimable",
        "reasons": ["few_events"],
    }
    assert research_rows(data)[0]["value"] is None
    with pytest.raises(ValueError):
        research_rows({**data, "version": "99"})


def test_real_widget_clears_stale_report_on_error_and_run_switch(app, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload()), encoding="utf-8")
    widget = ResearchResultsWidget()
    widget.load_report(path)
    assert widget.table.rowCount() == 1
    assert widget.calibration.listDataItems()
    widget.set_run_root(None)
    assert widget.table.rowCount() == 0
    assert not widget.calibration.listDataItems()
    path.write_text("broken", encoding="utf-8")
    widget.load_report(path)
    assert "nicht darstellbar" in widget.status.text()
    widget.close()


def test_optional_research_contracts_and_labor_abstention(app):
    data = {
        "version": "1.0.0",
        "clinical_claim_allowed": False,
        "status": "research_estimate",
        "n_patients": 300,
        "effects": {
            "benefit": {"ate": -0.1, "ci95": [-0.2, 0], "scale": "risk_difference"}
        },
    }
    assert research_rows(data)[0]["value"] == -0.1
    widget = CaseResearchStatusWidget()
    assert "nicht berechnet" in widget.toPlainText()
    widget.set_case_payload({"localization_result": {"clinical_claim_allowed": True}})
    assert "nicht verifiziert" in widget.toPlainText()
    widget.close()


def test_deep_search_is_never_labelled_independent_confirmation():
    data = {
        "deep_combination_version": "1.2.0",
        "clinical_claim_allowed": False,
        "locked_test_read": False,
        "measurement_role": "inner_selection_not_independent_confirmation",
        "depth_profile_rows": [
            {"endpoint_id": "ef", "depth": 2, "best_confirmation_auc": 0.75}
        ],
    }
    row = research_rows(data)[0]
    assert "Interne" in row["metric"]
    assert "keine unabhängige" in row["status"]
    with pytest.raises(ValueError):
        research_rows({**data, "measurement_role": "independent_confirmation"})
