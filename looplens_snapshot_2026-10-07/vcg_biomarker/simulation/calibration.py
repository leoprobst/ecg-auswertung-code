"""Local, patient-disjoint population priors. Observable targets != ionic ground truth."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("age", "female", "height_cm", "weight_kg")
FEATURE_PROFILES = {"body_measures": FEATURES, "age_sex": ("age", "female")}
TARGETS = {
    "lv_diameter_mm": (20., 100., "mm", "independent_echo"),
    "septum_mm": (3., 40., "mm", "independent_echo"),
    "posterior_wall_mm": (3., 40., "mm", "independent_echo"),
    "lv_edv_ml": (20., 500., "ml", "independent_echo"),
    "qrs_duration_ms": (40., 250., "ms", "machine_measurement_not_independent_truth"),
}
SALT = "patient-twin-prior-v1"


def partition(cohort, patient):
    h = hashlib.sha256(f"{SALT}|{cohort}|{patient}".encode()).digest()
    value = int.from_bytes(h[:8], "big") / 2**64
    return "fit" if value < .7 else "calibration" if value < .85 else "internal_audit"


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()


def train_priors(frame: pd.DataFrame, seed=20260923, feature_profile="body_measures"):
    """All inputs must be original DEVELOPMENT TRAIN patients; never relabel validation/test."""
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError("Unknown predeclared covariate profile")
    features = FEATURE_PROFILES[feature_profile]
    required = {"patient_id", "cohort", "source_split", *features}
    if not required <= set(frame):
        raise ValueError("Missing calibration schema fields")
    if not frame["source_split"].eq("train").all():
        raise ValueError("Only the existing train split may enter population-prior training")
    if frame.duplicated(["cohort", "patient_id"]).any():
        raise ValueError("Exactly one reference-blind index per patient required")
    if frame[["patient_id", "cohort"]].isna().any().any():
        raise ValueError("Patient identity required")
    frame = frame.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["cohort"] = frame["cohort"].astype(str)
    if frame.duplicated(["cohort", "patient_id"]).any() or frame["patient_id"].str.strip().eq("").any():
        raise ValueError("Ambiguous patient identity")
    df = frame.sort_values(["cohort", "patient_id"]).reset_index(drop=True).copy()
    parts = np.array([partition(c, str(p)) for c, p in zip(df.cohort, df.patient_id)])
    x = df[list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(x).all(1)
    valid &= (x[:, 0] >= 18) & (x[:, 0] <= 89) & np.isin(x[:, 1], [0., 1.])
    if feature_profile == "body_measures":
        valid &= (x[:, 2] >= 100) & (x[:, 2] <= 230) & (x[:, 3] >= 25) & (x[:, 3] <= 300)
    report = {"schema": "twin-population-priors/1", "clinical_use": False, "seed": seed,
              "partition_salt": SALT, "features": list(features), "feature_profile": feature_profile, "n_index_patients": len(df),
              "n_complete_covariates": int(valid.sum()), "input_sha256": frame_hash(df),
              "population": "hospital-based, not healthy normative values", "models": {}}
    for target, (lo, hi, unit, reference) in TARGETS.items():
        if target not in df:
            report["models"][target] = {"status": "not_available"}
            continue
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
        mask = valid & np.isfinite(y) & (y >= lo) & (y <= hi)
        indices = {k: np.flatnonzero(mask & (parts == k)) for k in ("fit", "calibration", "internal_audit")}
        counts = {k: len(v) for k, v in indices.items()}
        if counts["fit"] < 150 or counts["calibration"] < 40 or counts["internal_audit"] < 40:
            report["models"][target] = {"status": "non_estimable", "reason": "insufficient_patients", "counts": counts}
            continue
        tr, cal, audit = (indices[k] for k in ("fit", "calibration", "internal_audit"))
        scores = []
        for alpha in (.1, 1., 10., 100.):
            errors, base_errors = [], []
            for local_tr, local_val in KFold(3, shuffle=True, random_state=seed).split(tr):
                a, b = tr[local_tr], tr[local_val]
                fit = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[a], y[a])
                errors.extend(np.abs(fit.predict(x[b]) - y[b]))
                base_errors.extend(np.abs(np.median(y[a]) - y[b]))
            scores.append((float(np.mean(errors)), alpha, float(np.mean(base_errors))))
        score, alpha, baseline_cv = min(scores)
        # Selection only from development fit folds, never from audit performance.
        use_covariates = score < .98 * baseline_cv
        fit = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[tr], y[tr])
        scaler, model = fit.steps[0][1], fit.steps[1][1]
        coef = model.coef_ / scaler.scale_ if use_covariates else np.zeros(len(features))
        intercept = float(model.intercept_ - np.dot(scaler.mean_, coef)) if use_covariates else float(np.median(y[tr]))
        prediction = x @ coef + intercept
        residual = np.sort(np.abs(y[cal] - prediction[cal]))
        # Finite-sample split conformal quantile for nominal 90% marginal coverage.
        rank = min(len(residual), int(np.ceil((len(residual) + 1) * .9)))
        half_width = float(residual[rank - 1])
        model_mae = float(np.mean(np.abs(y[audit] - prediction[audit])))
        base_mae = float(np.mean(np.abs(y[audit] - np.median(y[tr]))))
        paired_errors = np.abs(y[audit] - prediction[audit]) - np.abs(y[audit] - np.median(y[tr]))
        rng = np.random.default_rng(seed)
        bootstrap = [float(paired_errors[rng.integers(0, len(audit), len(audit))].mean()) for _ in range(500)]
        subgroups = {}
        for label, group in (("female", x[audit, 1] == 1), ("male", x[audit, 1] == 0),
                             ("age_lt65", x[audit, 0] < 65), ("age_ge65", x[audit, 0] >= 65)):
            subset = audit[group]
            subgroups[label] = {"n": len(subset), "status": "descriptive_only" if len(subset) >= 30 else "insufficient_patients"}
            if len(subset) >= 30:
                subgroups[label].update(mae=float(np.mean(np.abs(y[subset] - prediction[subset]))),
                                        coverage_90=float(np.mean(np.abs(y[subset] - prediction[subset]) <= half_width)))
        report["models"][target] = {
            "status": "research_prior", "unit": unit, "reference": reference, "counts": counts,
            "selected_by_fit_cv": "ridge" if use_covariates else "intercept_only",
            "alpha": alpha, "coefficients": coef.tolist(), "intercept": intercept,
            "input_bounds": [[float(x[tr, k].min()), float(x[tr, k].max())] for k in range(x.shape[1])],
            "fit_cv_mae": score, "fit_cv_baseline_mae": baseline_cv,
            "conformal_half_width_90": half_width, "audit_mae": model_mae,
            "audit_baseline_mae": base_mae, "audit_coverage_90": float(np.mean(np.abs(y[audit] - prediction[audit]) <= half_width)),
            "audit_paired_mae_difference_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "audit_subgroups": subgroups,
            "audit_n": len(audit), "physics_parameter_identified": False,
            "required_independent_validation": True,
        }
    return report


def predict_priors(bundle, profile):
    features = tuple(bundle.get("features", ()))
    if (bundle.get("schema") != "twin-population-priors/1" or features not in FEATURE_PROFILES.values()
            or bundle.get("clinical_use") is not False):
        raise ValueError("Incompatible population prior")
    values = profile.covariates()
    result = {}
    for target, model in bundle["models"].items():
        if model["status"] != "research_prior":
            result[target] = {"status": "unavailable"}
            continue
        coef = np.asarray(model["coefficients"], float)
        bounds = np.asarray(model["input_bounds"], float)
        if (coef.shape != (len(features),) or bounds.shape != (len(features), 2)
                or not np.isfinite(coef).all() or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
                or not np.isfinite([model["intercept"], model["conformal_half_width_90"]]).all()
                or model["conformal_half_width_90"] < 0):
            raise ValueError("Invalid numerical prior contract")
        # Even an intercept-only model was calibrated on this complete-case population.
        needed = list(features)
        if any(k not in values for k in needed):
            result[target] = {"status": "abstain", "reason": "missing_covariates", "required": needed}
            continue
        if any(not lo <= values[k] <= hi for k, (lo, hi) in zip(features, model["input_bounds"]) if k in needed):
            result[target] = {"status": "abstain", "reason": "outside_training_support"}
            continue
        mean = float(model["intercept"] + sum(c * values.get(k, 0) for c, k in zip(model["coefficients"], features)))
        width = model["conformal_half_width_90"]
        result[target] = {"status": "population_prior", "center": mean, "interval_90": [mean-width, mean+width],
                          "required_covariates": needed, "feature_profile": bundle.get("feature_profile", "body_measures"),
                          "unit": model["unit"], "reference": model["reference"],
                          "note": "Marginal cohort interval, not a measured patient value or ionic parameter."}
    return result


def build_mimic_training_frame(work_root, max_patients=20000):
    """Read only sealed train membership, then reference-blind earliest ECG.

    Existing OMR's nearest_omr_delta_days is NOT per-measurement age; do not use
    it as a guarantee of contemporaneous height/weight. Prefer echo's own body
    measures, and only a preceding TTE within seven days. Outputs stay local.
    """
    import duckdb
    if type(max_patients) is not int or not 1 <= max_patients <= 200000:
        raise ValueError("Bounded positive patient count required")
    root = Path(work_root).resolve()
    paths = {"master": root / "master/ecg_master_index.parquet",
             "splits": root / "master/subject_splits.parquet",
             "echo": root / "evidence/echo_measurements_wide.parquet"}
    before = {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}
    con = duckdb.connect(":memory:", config={"threads": "1", "memory_limit": "512MB", "max_temp_directory_size": "0B"})
    try:
        for name, path in paths.items():
            con.read_parquet(str(path)).create_view(name)
        frame = con.execute("""
            WITH development AS (
              SELECT m.subject_id, m.study_id, m.ecg_time, m.age, m.sex,
                     m.machine_qrs_duration_ms, m.machine_metadata_usable_for_primary,
                     ROW_NUMBER() OVER(PARTITION BY m.subject_id ORDER BY m.ecg_time, m.study_id) AS rn
              FROM master m JOIN splits s USING(subject_id)
              WHERE s.split = 'train' AND m.split = 'train'
                AND m.age BETWEEN 18 AND 89 AND NOT COALESCE(m.age_deidentified_over_89, true)
            ), selected AS (
              SELECT * FROM development WHERE rn=1
              ORDER BY sha256(CAST(subject_id AS VARCHAR) || 'twin-index-v1') LIMIT ?
            ), matched AS (
              SELECT s.*, e.measurement_id, e.echo_time, e.echo_height_cm, e.echo_weight_kg,
                     e.lv_internal_diameter_diastole_cm, e.septal_thickness_cm,
                     e.posterior_wall_thickness_cm, e.lv_end_diastolic_volume_ml,
                     ROW_NUMBER() OVER(PARTITION BY s.subject_id ORDER BY e.echo_time DESC, e.measurement_id) AS en
              FROM selected s LEFT JOIN echo e ON s.subject_id=e.subject_id
                AND LOWER(e.test_type)='tte' AND e.echo_time <= s.ecg_time
                AND e.echo_time >= s.ecg_time - INTERVAL 7 DAY
            )
            SELECT CAST(subject_id AS VARCHAR) AS patient_id, 'mimic' AS cohort, 'train' AS source_split,
                   study_id, age, CASE WHEN sex='F' THEN 1. WHEN sex='M' THEN 0. ELSE NULL END AS female,
                   echo_height_cm AS height_cm, echo_weight_kg AS weight_kg,
                   10*lv_internal_diameter_diastole_cm AS lv_diameter_mm,
                   10*septal_thickness_cm AS septum_mm, 10*posterior_wall_thickness_cm AS posterior_wall_mm,
                   lv_end_diastolic_volume_ml AS lv_edv_ml,
                   CASE WHEN machine_metadata_usable_for_primary THEN machine_qrs_duration_ms END AS qrs_duration_ms
            FROM matched WHERE en=1
        """, [int(max_patients)]).df()
    finally:
        con.close()
    if before != {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}:
        raise RuntimeError("Upstream source changed during read; calibration aborted")
    manifest = {"selection": "existing train only; earliest ECG per patient before echo/missingness inspection",
                "source_stats": before, "echo_window": "preceding 7 days, TTE only",
                "clock_caveat": "ECG clocks not externally verified; structural prior, not acute treatment response",
                "max_patients": max_patients, "locked_test_accessed": False}
    return frame, manifest


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-patients", type=int, default=20000)
    parser.add_argument("--feature-profile", choices=tuple(FEATURE_PROFILES), default="body_measures")
    parser.add_argument("--retain-local-indices", action="store_true", help="Optional restricted patient-level audit file; never publish")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.is_relative_to(Path(args.work_root).resolve()):
        raise ValueError("Calibration output must be outside the production workspace")
    if output.exists():
        raise FileExistsError("New output directory required; do not silently overwrite a prior")
    output.mkdir(parents=True)
    import os
    import psutil
    if psutil.virtual_memory().available < 3*1024**3:
        raise RuntimeError("Insufficient spare RAM; no spill or reserve bypass allowed")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    frame, manifest = build_mimic_training_frame(args.work_root, args.max_patients)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        bundle = train_priors(frame, feature_profile=args.feature_profile)
    import platform
    import sklearn
    bundle["software"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "sklearn": sklearn.__version__,
                          "calibration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    bundle["source_manifest"] = manifest
    if args.retain_local_indices:
        frame.to_parquet(output / "development_indices.parquet", index=False)
    import shutil
    frozen = output / "frozen_source"
    frozen.mkdir()
    for name in ("calibration.py", "patient.py"):
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    (output / "population_priors.json").write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"n": len(frame), "models": {k: v["status"] for k, v in bundle["models"].items()}}))


if __name__ == "__main__":
    main()
