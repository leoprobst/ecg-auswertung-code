from __future__ import annotations

import pandas as pd
import pytest

from vcg_biomarker.core.delineation.benchmark_suite import (
    _af_guard_confusion_metrics,
    _majority_rhythm_label,
    run_afdb_af_guard_confusion_benchmark,
)


def test_majority_rhythm_label_picks_larger_overlap() -> None:
    segments = [(0, 1000, "af"), (1000, 3000, "other")]
    assert _majority_rhythm_label(segments, 500, 1500) == "af"
    assert _majority_rhythm_label(segments, 900, 3000) == "other"
    assert _majority_rhythm_label(segments, 5000, 6000) is None
    assert _majority_rhythm_label([], 0, 100) is None


def test_af_guard_confusion_metrics_computes_sensitivity_and_specificity() -> None:
    frame = pd.DataFrame(
        [
            {
                "ground_truth_rhythm": "af",
                "predicted_af_compatible": True,
                "predicted_flutter_compatible": False,
            },
            {
                "ground_truth_rhythm": "af",
                "predicted_af_compatible": False,
                "predicted_flutter_compatible": False,
            },
            {
                "ground_truth_rhythm": "other",
                "predicted_af_compatible": False,
                "predicted_flutter_compatible": False,
            },
            {
                "ground_truth_rhythm": "other",
                "predicted_af_compatible": True,
                "predicted_flutter_compatible": False,
            },
            {
                "ground_truth_rhythm": "flutter",
                "predicted_af_compatible": False,
                "predicted_flutter_compatible": True,
            },
        ]
    )

    metrics = _af_guard_confusion_metrics(frame)

    af = metrics["af"]
    assert af["true_positive"] == 1
    assert af["false_negative"] == 1
    assert af["false_positive"] == 1
    assert af["true_negative"] == 2
    assert af["sensitivity"] == 0.5
    assert af["specificity"] == pytest.approx(2 / 3)
    flutter = metrics["flutter"]
    assert flutter["true_positive"] == 1
    assert flutter["sensitivity"] == 1.0
    assert metrics["window_count"] == 5
    assert metrics["ground_truth_label_counts"]["af"] == 2


def test_af_guard_confusion_metrics_empty_frame_is_none() -> None:
    metrics = _af_guard_confusion_metrics(pd.DataFrame())
    assert metrics == {"af": None, "flutter": None}


def test_missing_afdb_dataset_reports_no_windows_completed(tmp_path) -> None:
    (tmp_path / "RECORDS").write_text("00001\n", encoding="utf-8")
    result = run_afdb_af_guard_confusion_benchmark(tmp_path, max_records=1)
    assert result["status"] == "failed_no_windows"
    assert result["windows_completed"] == 0
    assert result["records_quarantined"] == 1
