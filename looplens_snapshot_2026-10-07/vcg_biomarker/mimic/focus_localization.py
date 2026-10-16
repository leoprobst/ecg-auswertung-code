"""Regional research localization: mapping references, never VCG-octant labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

VERSION = "1.0.0"
TASKS = {
    "atrial_focus": {
        "rhythms": {"pac", "focal_at"},
        "references": {"focal_origin", "earliest_breakthrough"},
        "regions": {"right_atrial", "left_atrial", "septal_paraseptal"},
    },
    "ventricular_focus": {
        "rhythms": {"pvc", "focal_vt"},
        "references": {"focal_origin", "earliest_breakthrough"},
        "regions": {
            "rv_outflow",
            "rv_other",
            "lv_outflow",
            "lv_other",
            "boundary_uncertain",
        },
    },
    "ventricular_exit": {
        "rhythms": {"scar_vt"},
        "references": {"reentry_exit"},
        "regions": {
            "rv_outflow",
            "rv_other",
            "lv_outflow",
            "lv_other",
            "boundary_uncertain",
        },
    },
    "pacing_site": {
        "rhythms": {"paced_ventricular"},
        "references": {"pacing_site"},
        "regions": {
            "rv_outflow",
            "rv_other",
            "lv_outflow",
            "lv_other",
            "boundary_uncertain",
        },
    },
}
REQUIRED = (
    "cohort_id",
    "patient_id",
    "episode_id",
    "morphology_id",
    "signal_id",
    "rhythm",
    "reference_type",
    "region",
    "evidence_id",
    "reference_verified",
    "reference_uncertainty",
    "signal_quality_pass",
    "split",
    "center_id",
    "transform",
    "axis_convention",
)


@dataclass(frozen=True)
class LocalizationResult:
    task: str
    reference_type: str
    probabilities: dict[str, float]
    status: str
    abstention_reason: str = ""
    version: str = VERSION
    clinical_claim_allowed: bool = False
    coordinates: None = None


def mapping_inventory(root: Path):
    """Only a dedicated mapping import is accepted, never ICD/report surrogates."""
    root = Path(root)
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return {
        "version": VERSION,
        "import_root": str(root),
        "files": [str(p) for p in files],
        "required_columns": list(REQUIRED),
        "status": "requires_record_level_validation"
        if files
        else "missing_mapping_reference",
        "clinical_claim_allowed": False,
        "note": "Presence is not validation. Ablation success alone is not origin ground truth.",
    }


def validate_mapping(frame: pd.DataFrame, task: str):
    if task not in TASKS:
        raise ValueError("unsupported_localization_task")
    missing = sorted(set(REQUIRED) - set(frame))
    if missing:
        raise ValueError("missing_mapping_columns:" + ",".join(missing))
    if frame.empty:
        raise ValueError("empty_mapping_reference")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("missing_mapping_values")
    if not frame["split"].isin(["train", "validation"]).all():
        raise ValueError("locked_or_unknown_split_forbidden")
    spec = TASKS[task]
    if not frame.rhythm.isin(spec["rhythms"]).all():
        raise ValueError("rhythm_has_no_matching_focal_task")
    if not frame.reference_type.isin(spec["references"]).all():
        raise ValueError("incompatible_mapping_reference")
    if frame.reference_type.nunique() != 1:
        raise ValueError("origin_and_breakthrough_must_be_analyzed_separately")
    if not frame.region.isin(spec["regions"]).all():
        raise ValueError("unknown_anatomical_region")
    for name in ("reference_verified", "signal_quality_pass"):
        if (
            not frame[name]
            .map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x))
            .all()
        ):
            raise ValueError("unverified_reference_or_signal_quality")
    for name in (
        "cohort_id",
        "patient_id",
        "episode_id",
        "morphology_id",
        "signal_id",
        "evidence_id",
        "axis_convention",
    ):
        if frame[name].astype(str).str.strip().eq("").any():
            raise ValueError("empty_mapping_identity")
    if frame.axis_convention.nunique() != 1 or frame["transform"].nunique() != 1:
        raise ValueError("compare_transforms_separately_with_explicit_axes")
    if (
        not frame["transform"]
        .isin(["kors", "inverse_dower", "svd", "native_ecg"])
        .all()
    ):
        raise ValueError("unknown_transform")
    if task == "atrial_focus":
        if (
            "p_quality_pass" not in frame
            or not frame.p_quality_pass.map(
                lambda x: isinstance(x, (bool, np.bool_)) and bool(x)
            ).all()
        ):
            raise ValueError("insufficient_p_quality")
    keys = [
        "cohort_id",
        "patient_id",
        "episode_id",
        "morphology_id",
        "signal_id",
        "transform",
    ]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate_mapping_episode")
    # Preserve the source namespace; IDs alone are not globally unique.
    patients = pd.MultiIndex.from_frame(frame[["cohort_id", "patient_id"]])
    groups = pd.factorize(patients)[0]
    if frame.groupby(["cohort_id", "patient_id"]).split.nunique().gt(1).any():
        raise ValueError("patient_crosses_splits")
    return groups


def regional_result(
    task,
    reference_type,
    classes,
    probabilities,
    *,
    quality_pass,
    threshold=0.70,
    minimum_margin=0.15,
):
    if task not in TASKS or reference_type not in TASKS[task]["references"]:
        raise ValueError("unsupported_task_reference")
    p = np.asarray(probabilities, dtype=float)
    if not 0 <= threshold <= 1 or not 0 <= minimum_margin <= 1:
        raise ValueError("invalid_abstention_threshold")
    if len(p) != len(classes) or len(set(classes)) != len(classes) or len(p) < 2:
        raise ValueError("invalid_region_probabilities")
    if (
        not set(classes) <= TASKS[task]["regions"]
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("invalid_region_probabilities")
    ordered = np.sort(p)
    reason = (
        "insufficient_signal_quality"
        if quality_pass is not True
        else (
            "ambiguous_region"
            if ordered[-1] < threshold or ordered[-1] - ordered[-2] < minimum_margin
            else ""
        )
    )
    if classes[int(np.argmax(p))] == "boundary_uncertain":
        reason = "anatomical_region_uncertain"
    return LocalizationResult(
        task,
        reference_type,
        dict(zip(classes, map(float, p))),
        "abstained" if reason else "research_region",
        reason,
    )


def evaluate_regions(
    mapping: pd.DataFrame,
    feature_columns,
    *,
    task,
    folds=5,
    seed=20260914,
    evaluation="grouped_cv",
    return_predictions=False,
    model_family="multinomial",
):
    """Grouped out-of-fold regional evaluation; no anatomy or mapping features as predictors."""
    groups = validate_mapping(mapping, task)
    if evaluation not in {"grouped_cv", "leave_center_out", "frozen_validation"}:
        raise ValueError("unknown_evaluation_design")
    if model_family not in {"multinomial", "hist_gradient_boosting"}:
        raise ValueError("unknown_localization_model_family")
    if evaluation != "frozen_validation" and not mapping.split.eq("train").all():
        raise ValueError("only_train_mapping_for_development")
    eligible = mapping.reference_uncertainty.isin(
        ["low", "regional"]
    ) & mapping.region.ne("boundary_uncertain")
    n_uncertain_excluded = int((~eligible).sum())
    mapping = mapping.loc[eligible].copy()
    groups = groups[eligible.to_numpy()]
    if set(feature_columns) & set(REQUIRED) or not feature_columns:
        raise ValueError("reference_metadata_is_not_a_predictor")
    y = mapping.region.to_numpy()
    classes = np.unique(y)
    if len(classes) < 2 or any(
        len(np.unique(groups[(y == c) & mapping.split.eq("train").to_numpy()]))
        < max(10, folds)
        for c in classes
    ):
        raise ValueError("insufficient_region_patients")
    x = mapping[list(feature_columns)].astype(float).replace([np.inf, -np.inf], np.nan)
    p = np.full((len(y), len(classes)), np.nan)
    if evaluation == "leave_center_out":
        centers = mapping.center_id.to_numpy()
        if len(np.unique(centers)) < 2:
            raise ValueError("multiple_centers_required")
        if mapping.groupby(["cohort_id", "patient_id"]).center_id.nunique().gt(1).any():
            raise ValueError("patient_crosses_centers")
        partitions = [
            (np.flatnonzero(centers != c), np.flatnonzero(centers == c))
            for c in np.unique(centers)
        ]
    elif evaluation == "frozen_validation":
        train = np.flatnonzero(mapping.split.eq("train"))
        test = np.flatnonzero(mapping.split.eq("validation"))
        if not len(train) or not len(test):
            raise ValueError("train_and_validation_required")
        partitions = [(train, test)]
    else:
        partitions = list(
            StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(
                x, y, groups
            )
        )
    with threadpool_limits(limits=1):
        for a, b in partitions:
            if set(y[a]) != set(classes):
                raise ValueError("region_absent_from_training_fold")
            if any(len(np.unique(groups[a][y[a] == c])) < 5 for c in classes):
                raise ValueError("insufficient_region_patients_in_training_fold")
            valid = x.iloc[a].notna().mean().ge(0.8) & x.iloc[a].std().gt(1e-12)
            names = list(x.columns[valid])
            if not names or x.iloc[b][names].isna().all(axis=1).any():
                raise ValueError("insufficient_localization_features")
            estimator = (
                LogisticRegression(C=1, max_iter=500)
                if model_family == "multinomial"
                else HistGradientBoostingClassifier(
                    max_iter=60,
                    max_leaf_nodes=7,
                    min_samples_leaf=10,
                    early_stopping=False,
                    random_state=seed,
                )
            )
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), estimator
            )
            # Give each patient's repeated morphologies equal total weight.
            _, inverse, counts = np.unique(
                groups[a], return_inverse=True, return_counts=True
            )
            weights = 1 / counts[inverse]
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(
                    x.iloc[a][names],
                    y[a],
                    **{model.steps[-1][0] + "__sample_weight": weights},
                )
            p[b] = model.predict_proba(x.iloc[b][names])
    evaluated = np.isfinite(p).all(axis=1)
    evaluated_indices = np.flatnonzero(evaluated)
    p, y, groups = p[evaluated], y[evaluated], groups[evaluated]
    predicted = classes[p.argmax(axis=1)]
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    curve = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        accepted = (p.max(axis=1) >= threshold) & (
            np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2] >= 0.15
        )
        curve.append(
            {
                "threshold": threshold,
                "coverage": float(np.average(accepted, weights=weights)),
                "error": float(
                    np.average(
                        predicted[accepted] != y[accepted], weights=weights[accepted]
                    )
                )
                if accepted.any()
                else None,
            }
        )
    return {
        "version": VERSION,
        "task": task,
        "reference_type": mapping.reference_type.iloc[0],
        "classes": classes.tolist(),
        "n_patients": len(np.unique(groups)),
        "n_uncertain_reference_rows_excluded": n_uncertain_excluded,
        "macro_f1": float(
            f1_score(y, predicted, average="macro", sample_weight=weights)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted, sample_weight=weights)
        ),
        "log_loss": float(log_loss(y, p, labels=classes, sample_weight=weights)),
        "top2_accuracy": float(
            np.average(
                np.any(classes[np.argsort(p, axis=1)[:, -2:]] == y[:, None], axis=1),
                weights=weights,
            )
        ),
        "classification_report": classification_report(
            y, predicted, output_dict=True, zero_division=0, sample_weight=weights
        ),
        "confusion_matrix": confusion_matrix(
            y, predicted, labels=classes, sample_weight=weights
        ).tolist(),
        "coverage_error": curve,
        "clinical_claim_allowed": False,
        "evidence_stage": "exploratory",
        "coordinates": None,
        "evaluation_design": evaluation,
        "model_family": model_family,
        **(
            {
                "predictions": {
                    "row_indices": evaluated_indices.tolist(),
                    "probabilities": p.tolist(),
                    "truth": y.tolist(),
                    "patient_groups": groups.tolist(),
                }
            }
            if return_predictions
            else {}
        ),
    }


def beat_activation_features(
    xyz,
    sampling_rate,
    onset,
    offset,
    *,
    segment,
    baseline_xyz,
    quality_pass,
    morphology_id,
):
    """One already-delineated beat, in calibrated XYZ mV; never an all-rhythm median.

    Absolute axes are retained. A caller must supply a validated transformation
    and a physiological baseline, not rotate the loop to an arbitrary PCA frame.
    These are research task inputs, not additions to the frozen marker panel.
    """
    xyz = np.asarray(xyz, dtype=float)
    baseline = np.asarray(baseline_xyz, dtype=float)
    if segment not in {"p", "qrs"} or quality_pass is not True or not morphology_id:
        raise ValueError("segment_quality_or_morphology_required")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or baseline.shape != (3,):
        raise ValueError("expected_samples_by_xyz")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("invalid_sampling_rate")
    if (
        not isinstance(onset, (int, np.integer))
        or not isinstance(offset, (int, np.integer))
        or not 0 <= onset < offset <= len(xyz)
    ):
        raise ValueError("invalid_segment_bounds")
    loop = xyz[onset:offset] - baseline
    if not np.isfinite(loop).all() or not np.isfinite(baseline).all():
        raise ValueError("nonfinite_segment")
    result = {
        "segment": segment,
        "morphology_id": morphology_id,
        "duration_ms": 1000 * len(loop) / sampling_rate,
        "axis_preserved": True,
        "clinical_claim_allowed": False,
    }
    windows = (
        {f"{ms}ms": ms / 1000 for ms in (10, 20, 30, 40)}
        if segment == "qrs"
        else {
            "first_quarter": len(loop) / (4 * sampling_rate),
            "first_half": len(loop) / (2 * sampling_rate),
        }
    )
    times = np.arange(len(loop)) / sampling_rate
    for label, end in windows.items():
        if len(loop) < 2 or end > times[-1]:
            result[label] = {"status": "segment_too_short"}
            continue
        before = times < end
        # Include/interpolate the exact endpoint: ten samples at 1 kHz span
        # nine, not ten milliseconds. Never mislabel that integral as 10 ms.
        window_times = np.append(times[before], end)
        endpoint = np.array([np.interp(end, times, loop[:, axis]) for axis in range(3)])
        values = np.vstack([loop[before], endpoint])
        integral = np.trapezoid(values, x=window_times, axis=0)
        result[label] = {
            "integral_xyz_mv_s": integral.tolist(),
            "mean_xyz_mv": (integral / end).tolist(),
            "window_seconds": end,
            "status": "measured",
        }
    return result
