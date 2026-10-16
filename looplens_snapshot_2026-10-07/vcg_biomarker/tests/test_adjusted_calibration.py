from __future__ import annotations

import numpy as np
import pandas as pd

from vcg_biomarker.mimic.adjusted_models import (
    FrozenLogisticModel,
    calibration_curve_points,
)


def test_calibration_curve_points_are_observed_predictions_with_wilson_ci() -> None:
    predictor = np.linspace(-3.0, 3.0, 200)
    probability = 1.0 / (1.0 + np.exp(-predictor))
    labels = (np.arange(200) / 200.0 < probability).astype(int)
    frame = pd.DataFrame({"label": labels, "marker": predictor})
    model = FrozenLogisticModel(
        model_id="candidate",
        endpoint_id="endpoint",
        predictor_names=("marker",),
        means=(0.0,),
        standard_deviations=(1.0,),
        standardize=(False,),
        coefficients=(0.0, 1.0),
        baseline_model_id=None,
        training_subject_hash="synthetic",
        events_per_parameter=50.0,
    )

    rows = calibration_curve_points(
        frame,
        model,
        evaluation_split="validation",
    )

    assert 2 <= len(rows) <= 10
    assert sum(row["n"] for row in rows) == len(frame)
    assert all(0.0 <= row["mean_predicted"] <= 1.0 for row in rows)
    assert all(
        row["observed_ci_low"]
        <= row["observed_rate"]
        <= row["observed_ci_high"]
        for row in rows
    )
    assert [row["mean_predicted"] for row in rows] == sorted(
        row["mean_predicted"] for row in rows
    )
