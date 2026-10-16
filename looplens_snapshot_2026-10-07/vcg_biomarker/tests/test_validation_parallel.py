from __future__ import annotations

import numpy as np
import pandas as pd

from vcg_biomarker.mimic.models import ValidationSpec
from vcg_biomarker.mimic.validation import _evaluate_endpoint_markers


def test_parallel_marker_evaluation_matches_sequential_results() -> None:
    rng = np.random.default_rng(20260726)
    labels = np.repeat((0, 1), 60)
    train = pd.DataFrame(
        {
            "subject_id": np.arange(120),
            "label": labels,
            "marker_up": labels + rng.normal(0.0, 0.35, 120),
            "marker_down": -labels + rng.normal(0.0, 0.35, 120),
        }
    )
    validation = pd.DataFrame(
        {
            "subject_id": np.arange(1_000, 1_120),
            "label": labels,
            "marker_up": labels + rng.normal(0.0, 0.4, 120),
            "marker_down": -labels + rng.normal(0.0, 0.4, 120),
        }
    )
    specification = ValidationSpec(
        run_id="parallel-test",
        split_hash="split",
        endpoint_ids=("endpoint",),
        panel_versions={"primary_core": "test", "sentinel": "test"},
        development_bootstrap_iterations=20,
        min_class_inference=10,
        min_class_primary=20,
    )
    arguments = {
        "endpoint_id": "endpoint",
        "feature_ids": ("marker_up", "marker_down"),
        "panel_map": {
            "marker_up": "primary_core",
            "marker_down": "sentinel",
        },
        "train": train,
        "validation": validation,
        "specification": specification,
        "bootstrap_iterations": 20,
    }

    sequential = _evaluate_endpoint_markers(**arguments, worker_count=1)
    parallel = _evaluate_endpoint_markers(**arguments, worker_count=2)

    assert [item.feature_id for item in parallel] == [
        "marker_up",
        "marker_down",
    ]
    assert [item.model.to_dict() for item in parallel if item.model] == [
        item.model.to_dict() for item in sequential if item.model
    ]
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            [
                {
                    **(item.train_record or {}),
                    **{
                        f"validation_{key}": value
                        for key, value in (item.validation_record or {}).items()
                    },
                }
                for item in parallel
            ]
        ),
        pd.DataFrame(
            [
                {
                    **(item.train_record or {}),
                    **{
                        f"validation_{key}": value
                        for key, value in (item.validation_record or {}).items()
                    },
                }
                for item in sequential
            ]
        ),
        check_exact=True,
    )


def test_p_projection_evaluation_separates_restriction_from_raw_missingness() -> None:
    labels = np.repeat((0, 1), 60)

    def frame(subject_start: int) -> pd.DataFrame:
        rhythm = np.full(120, "regular", dtype=object)
        rhythm[:10] = "af_compatible_pattern"
        rhythm[10:20] = "irregular"
        return pd.DataFrame({
            "subject_id": np.arange(subject_start, subject_start + 120),
            "label": labels,
            "p_loop_frontal_xy_area": labels + np.linspace(-.2, .2, 120),
            "rhythm_context": rhythm,
            "rhythm_context_confidence": np.full(120, .90),
            "p_wave_coverage": np.full(120, .80),
        })

    specification = ValidationSpec(
        run_id="p-restriction-test",
        split_hash="split",
        endpoint_ids=("echo_lae",),
        panel_versions={"sentinel": "test"},
        development_bootstrap_iterations=0,
        min_class_inference=10,
        min_class_primary=20,
    )
    result = _evaluate_endpoint_markers(
        endpoint_id="echo_lae",
        feature_ids=("p_loop_frontal_xy_area",),
        panel_map={"p_loop_frontal_xy_area": "sentinel"},
        train=frame(0),
        validation=frame(1_000),
        specification=specification,
        bootstrap_iterations=0,
        worker_count=1,
    )[0]
    assert result.failure is None
    assert result.validation_record is not None
    assert result.validation_record["n_marker_raw_observed"] == 120
    assert result.validation_record["n_marker_observed"] == 100
    assert result.validation_record["n_marker_unavailable"] == 0
    assert result.validation_record["n_analysis_restriction_excluded"] == 20
    assert result.validation_record["analysis_restriction_required_columns_present"] is True
