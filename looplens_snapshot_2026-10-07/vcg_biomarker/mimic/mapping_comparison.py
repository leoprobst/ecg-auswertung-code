"""Paired regional comparisons; requires independently verified mapping imports."""

from itertools import combinations
import hashlib

import numpy as np
import pandas as pd

from .focus_localization import REQUIRED, validate_mapping, evaluate_regions
from ..core.vcg_transform import VCGTransformRegistry

METHODS = ("native_ecg", "kors", "inverse_dower", "svd")
KEYS = ["cohort_id", "patient_id", "episode_id", "morphology_id", "signal_id"]


def beat_transform_features(
    ecg_mv, baseline_mv, sampling_rate, onset, offset, *, segment="qrs"
):
    """Same delineated beat, standard 12-lead order; no morphology mixing.

    SVD comparator uses only singular-value invariants of the eight independent
    leads. Its arbitrary axes are deliberately not anatomical localization inputs.
    """
    ecg = np.asarray(ecg_mv, float)
    baseline = np.asarray(baseline_mv, float)
    if (
        segment not in {"p", "qrs"}
        or ecg.ndim != 2
        or ecg.shape[1] != 12
        or baseline.shape != (12,)
        or not np.isfinite(ecg).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(sampling_rate)
        or sampling_rate <= 0
        or not isinstance(onset, (int, np.integer))
        or not isinstance(offset, (int, np.integer))
        or not 0 <= onset < offset <= len(ecg)
        or offset - onset < 3
    ):
        raise ValueError("invalid_calibrated_delineated_beat")
    beat = ecg[onset:offset] - baseline
    representations = {"native_ecg": beat}
    representations.update(
        {
            name: VCGTransformRegistry.transform(beat, name).xyz
            for name in ("kors", "inverse_dower")
        }
    )
    out = {}
    for name, values in representations.items():
        integrals = np.trapezoid(values, dx=1 / sampling_rate, axis=0)
        out[name] = {
            **{f"integral_{i}_mv_s": float(v) for i, v in enumerate(integrals)},
            **{
                f"peak_to_peak_{i}_mv": float(v)
                for i, v in enumerate(np.ptp(values, axis=0))
            },
        }
        times = np.arange(len(values)) / sampling_rate
        windows = (
            {f"early_{ms}ms": ms / 1000 for ms in (10, 20, 30, 40)}
            if segment == "qrs"
            else {
                "early_quarter": len(values) / (4 * sampling_rate),
                "early_half": len(values) / (2 * sampling_rate),
            }
        )
        for label, end in windows.items():
            if end > times[-1]:
                integral = np.full(values.shape[1], np.nan)
            else:
                before = times < end
                endpoint = [
                    np.interp(end, times, values[:, i]) for i in range(values.shape[1])
                ]
                integral = np.trapezoid(
                    np.vstack([values[before], endpoint]),
                    x=np.append(times[before], end),
                    axis=0,
                )
            out[name].update(
                {f"{label}_integral_{i}_mv_s": float(v) for i, v in enumerate(integral)}
            )
    # Standard order I, II, III, aVR, aVL, aVF, V1..V6.
    independent = beat[:, [0, 1, 6, 7, 8, 9, 10, 11]]
    singular = np.linalg.svd(independent, compute_uv=False)
    out["svd"] = {
        f"singular_rms_{i}_mv": float(v / np.sqrt(len(beat)))
        for i, v in enumerate(singular[:3])
    }
    return out


def compare_mapping_transforms(
    frames,
    feature_columns,
    *,
    task,
    folds=5,
    evaluation="grouped_cv",
    seed=20260914,
    bootstraps=500,
    identity_audit_verified=False,
    nonlinear_benchmark=True,
):
    """All methods share signal intersection, ordered rows and evaluation folds.

    This compares the supplied feature representations, not diagnostic superiority
    of transformations in isolation. External validation is never used for tuning.
    """
    if identity_audit_verified is not True:
        raise ValueError("cross_source_identity_audit_required")
    if set(frames) != set(METHODS) or set(feature_columns) != set(METHODS):
        raise ValueError("all_four_comparators_required")
    if bootstraps < 20:
        raise ValueError("at_least_20_bootstraps_required")
    prepared, availability = {}, {}
    for method in METHODS:
        frame = frames[method].copy()
        validate_mapping(frame, task)
        if not frame["transform"].eq(method).all():
            raise ValueError("transform_identity_mismatch")
        columns = feature_columns[method]
        if not columns or set(columns) & set(REQUIRED):
            raise ValueError("invalid_feature_columns")
        if (
            method == "svd"
            and not frame.axis_convention.eq("svd_invariant_no_anatomical_axes").all()
        ):
            raise ValueError("svd_requires_invariant_axis_contract")
        eligible = frame.reference_uncertainty.isin(
            ["low", "regional"]
        ) & frame.region.ne("boundary_uncertain")
        measured = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
        availability[method] = {
            "import_rows": len(frame),
            "measured_rows": int(measured.sum()),
            "reference_eligible_rows": int(eligible.sum()),
        }
        prepared[method] = frame.loc[eligible & measured].set_index(KEYS).sort_index()
    common = prepared[METHODS[0]].index
    for frame in prepared.values():
        common = common.intersection(frame.index)
    if common.empty:
        raise ValueError("no_common_measured_mapping_signals")
    base = prepared[METHODS[0]].loc[common]
    metadata = [c for c in REQUIRED if c not in KEYS + ["transform", "axis_convention"]]
    results, benchmarks = {}, {}
    for method, frame in prepared.items():
        paired = frame.loc[common]
        if not paired[metadata].equals(base[metadata]):
            raise ValueError("paired_mapping_reference_mismatch")
        results[method] = evaluate_regions(
            paired.reset_index(),
            feature_columns[method],
            task=task,
            folds=folds,
            seed=seed,
            evaluation=evaluation,
            return_predictions=True,
        )
        if nonlinear_benchmark:
            benchmarks[method] = evaluate_regions(
                paired.reset_index(),
                feature_columns[method],
                task=task,
                folds=folds,
                seed=seed,
                evaluation=evaluation,
                model_family="hist_gradient_boosting",
            )
    differences = {}
    for first, second in combinations(METHODS, 2):
        r1, r2 = results[first], results[second]
        p1, p2 = r1["predictions"], r2["predictions"]
        if p1["row_indices"] != p2["row_indices"] or r1["classes"] != r2["classes"]:
            raise ValueError("evaluation_alignment_failure")
        truth = np.asarray(p1["truth"])
        classes = np.asarray(r1["classes"])
        correct1 = classes[np.argmax(p1["probabilities"], axis=1)] == truth
        correct2 = classes[np.argmax(p2["probabilities"], axis=1)] == truth
        patient_delta = (
            pd.Series(correct1.astype(float) - correct2)
            .groupby(p1["patient_groups"])
            .mean()
            .to_numpy()
        )
        rng = np.random.default_rng(seed)
        draws = [
            float(rng.choice(patient_delta, len(patient_delta), replace=True).mean())
            for _ in range(bootstraps)
        ]
        differences[first + "_minus_" + second] = {
            "patient_equal_accuracy_difference": float(patient_delta.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "uncertainty_scope": "paired_patient_bootstrap_conditional_on_predictions",
        }
    # Agreement is meaningful only for like units and physical XYZ axes.
    agreement = {}
    kors, dower = prepared["kors"].loc[common], prepared["inverse_dower"].loc[common]
    if not kors.axis_convention.equals(dower.axis_convention):
        raise ValueError("physical_axis_conventions_disagree")
    for name in sorted(
        set(feature_columns["kors"]) & set(feature_columns["inverse_dower"])
    ):
        if not name.endswith(("_mv", "_mv_s")):
            continue
        difference = (
            (kors[name] - dower[name]).groupby(level=["cohort_id", "patient_id"]).mean()
        )
        if len(difference) < 3:
            continue
        bias = float(difference.mean())
        sd = float(difference.std(ddof=1))
        agreement[name] = {
            "bias_kors_minus_dower": bias,
            "limits_of_agreement": [bias - 1.96 * sd, bias + 1.96 * sd],
            "unit": "mV*s" if name.endswith("_mv_s") else "mV",
            "n_patients": len(difference),
            "scope": "patient_mean_not_individual_beat_repeatability",
        }
    return {
        "version": "1.0.0",
        "status": "research_comparison",
        "task": task,
        "evaluation": evaluation,
        "availability_before_intersection": availability,
        "paired_signal_count": len(common),
        "results": results,
        "paired_differences": differences,
        "kors_dower_agreement": agreement,
        "nonlinear_benchmarks": benchmarks,
        "paired_identity_hash": hashlib.sha256(
            pd.util.hash_pandas_object(
                common.to_frame(index=False), index=False
            ).values.tobytes()
        ).hexdigest(),
        "svd_scope": "rotation_invariants_not_anatomical_xyz",
        "clinical_claim_allowed": False,
    }
