import pandas as pd
import pytest

from vcg_biomarker.core.analysis.clinical_baseline import (
    ClinicalRuleResult,
    evaluate_clinical_rules,
)
from vcg_biomarker.core.analysis.results import AnalysisRun
from vcg_biomarker.core.parallel_pipeline import ParallelPipeline
from vcg_biomarker.config import AppConfig


def _result_by_id(results, rule_id):
    return next(item for item in results if item.rule_id == rule_id)


def test_clinical_rule_sex_specific_thresholds():
    frame = pd.DataFrame({
        "sex": ["M", "M", "F", "F"],
        "bm_qtc_bazett": [440.0, 451.0, 455.0, 461.0],
        "path_LNGQT": [0, 1, 0, 1],
    })

    result = _result_by_id(evaluate_clinical_rules(frame), "lngqt_qtc_bazett_sex")

    assert result.status == "ok"
    assert result.tp == 2
    assert result.tn == 2
    assert result.sensitivity == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)


def test_clinical_rule_missing_columns_are_reported():
    frame = pd.DataFrame({"path_LVH": [0, 1]})

    result = _result_by_id(evaluate_clinical_rules(frame), "lvh_sokolow_lyon_ge_3500")

    assert result.status == "missing_columns"
    assert result.missing_columns == ["bm_sokolow_lyon_voltage_surrogate"]
    assert result.reference_sensitivity == pytest.approx(0.24)
    assert result.reference_specificity == pytest.approx(0.94)


def test_clinical_rule_uses_alternative_catalog_column():
    frame = pd.DataFrame({
        "sex": ["M", "M", "F", "F"],
        "bm_qtc_bazett_ms": [439.0, 441.0, 461.0, 459.0],
        "path_LNGQT": [0, 1, 1, 0],
    })

    result = _result_by_id(evaluate_clinical_rules(frame), "lngqt_qtc_bazett_sex")

    assert result.status == "ok"
    assert result.biomarker_name == "qtc_bazett_ms"
    assert result.sensitivity == pytest.approx(1.0)
    assert result.specificity == pytest.approx(1.0)


def test_clinical_rule_multi_condition_wpw_screen():
    frame = pd.DataFrame({
        "bm_pr_interval": [100.0, 130.0, 100.0, 100.0],
        "bm_qrs_duration": [130.0, 130.0, 100.0, 130.0],
        "path_WPW": [1, 0, 0, 1],
    })

    result = _result_by_id(evaluate_clinical_rules(frame), "wpw_short_pr_wide_qrs_screen")

    assert result.status == "ok"
    assert result.biomarker_name == "pr_interval + qrs_duration"
    assert (result.tp, result.fp, result.tn, result.fn) == (2, 0, 2, 0)
    assert result.balanced_accuracy == pytest.approx(1.0)


def test_fixed_threshold_metrics_are_computed():
    frame = pd.DataFrame({
        "bm_pr_interval": [180.0, 220.0, 210.0, 190.0],
        "path__AVB": [0, 0, 1, 1],
    })

    result = _result_by_id(evaluate_clinical_rules(frame), "avb_pr_gt_200")

    assert result.status == "ok"
    assert (result.tp, result.fp, result.tn, result.fn) == (1, 1, 1, 1)
    assert result.sensitivity == pytest.approx(0.5)
    assert result.specificity == pytest.approx(0.5)
    assert result.ppv == pytest.approx(0.5)
    assert result.npv == pytest.approx(0.5)
    assert result.accuracy == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)


def test_analysis_run_clinical_baseline_json_roundtrip(tmp_path):
    run = AnalysisRun(
        run_id="baseline_json",
        biomarker_names=[],
        pathology_names=["_AVB"],
        vcg_method="kors",
        clinical_baseline_results=[
            ClinicalRuleResult(
                rule_id="avb_pr_gt_200",
                pathology_name="_AVB",
                biomarker_name="pr_interval",
                display_name="PR interval",
                threshold_label="> 200",
                source_label="test",
                source_url="https://example.test",
                note="context",
                status="ok",
                n_total=4,
                n_evaluable=4,
                n_positive=2,
                n_negative=2,
                tp=1,
                fp=1,
                tn=1,
                fn=1,
                sensitivity=0.5,
                specificity=0.5,
                ppv=0.5,
                npv=0.5,
                accuracy=0.5,
                balanced_accuracy=0.5,
            )
        ],
    )

    path = tmp_path / "baseline.json"
    run.export_json(path)
    loaded = AnalysisRun.import_json(path)

    assert len(loaded.clinical_baseline_results) == 1
    assert loaded.clinical_baseline_results[0].rule_id == "avb_pr_gt_200"
    assert loaded.clinical_baseline_results[0].balanced_accuracy == pytest.approx(0.5)


def test_parallel_pipeline_attaches_clinical_baseline_from_cache(tmp_path):
    pipeline = ParallelPipeline(AppConfig())
    pipeline.cache.cache_dir = tmp_path
    for idx, (pr, label) in enumerate([
        (180.0, False),
        (220.0, False),
        (210.0, True),
        (190.0, True),
    ]):
        pipeline.cache.store(
            ecg_id=idx,
            biomarker_dict={"pr_interval": pr},
            pathology_labels={"_AVB": label},
            quality_score=1.0,
        )

    run = AnalysisRun(
        run_id="cache_baseline",
        biomarker_names=["pr_interval"],
        pathology_names=["_AVB"],
        vcg_method="kors",
    )
    pipeline._attach_clinical_baseline(run)

    result = _result_by_id(run.clinical_baseline_results, "avb_pr_gt_200")
    assert run.comparison_baseline["status"] == "computed"
    assert result.status == "ok"
    assert result.balanced_accuracy == pytest.approx(0.5)
