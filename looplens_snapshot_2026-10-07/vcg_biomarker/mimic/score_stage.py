"""Bounded, restartable research follow-up. Does not modify upstream results."""

from __future__ import annotations

from dataclasses import asdict, replace
import glob
import json
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_research import TreatmentProfile, PROFILES
from .feature_roles import feature_profile, is_combination_candidate
from .focus_localization import mapping_inventory
from .research_extensions import research_readiness
from .pre_run_extensions import research_contract, activation_residuals
from .pre_run_extension_stage import QRS_T_COUPLING_MARKERS

# Columns the activation-conditioned residual models read; projected beside
# the selected markers and blanked by the same extraction-QC gate.
ACTIVATION_MODEL_COLUMNS = (
    "rhythm_context", "qrs_duration_ms", "context_heart_rate_bpm", "t_loop_width_major",
    *QRS_T_COUPLING_MARKERS,
)
from .label_registry import definitions_by_id
from .panels import default_feature_panels, conceptual_feature_family
from .reference_tiers import annotate_reference_rows
from .score_research import (
    NotEstimable,
    SearchPolicy,
    VERSION,
    discover_binary_score,
    paired_probability_comparison,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    development_feature_glob,
    file_sha256,
    feature_parquet_clause,
    parquet_columns,
    scientific_config_hash,
)

PRIORITY_ENDPOINTS = (
    "reduced_lvef_strict",
    "hypokalemia_near_ecg",
    "hyperkalemia_near_ecg",
    "hypomagnesemia_near_ecg",
    "hypocalcemia_near_ecg",
    "severe_hypokalemia_near_ecg",
)
NORMALITY_FLAGS = (
    "classical_st_elevation_contiguous",
    "classical_st_depression_contiguous",
    "classical_t_inversion_contiguous",
    "classical_pathological_q_waves",
    "classical_rbbb_pattern",
    "classical_lbbb_pattern",
    "classical_low_voltage_qrs",
)
PROFILE_DESIGNS = {
    "urgent_angiography": "diagnosis",
    "pci": "treatment_benefit_research",
    "pretreatment": "treatment_benefit_research",
    "icd_prognosis": "prognosis",
    "af_monitoring": "prognosis",
    "pulmonary_embolism": "diagnosis",
    "amyloid_hcm_lvh": "diagnosis",
    "af_stroke_bleeding": "prognosis",
}

CLINICAL_COMPONENTS = (
    "context_age",
    "context_sex_male",
    "context_heart_rate",
    "context_quality",
    "context_icu",
    "context_ed",
    "context_inpatient",
)

DIRECT_ECG_GEOMETRY = frozenset(
    {"qrs_t_area_angle_direct_8lead_deg", "qrs_t_integral_angle_direct_8lead_deg"}
)


def score_modality(name):
    # Do not reinterpret the frozen feature registry globally. In this new
    # comparison direct eight-lead geometry is not a transformed XYZ measurement.
    return (
        "direct_ecg_geometry"
        if name in DIRECT_ECG_GEOMETRY or "direct_8lead" in name or name.startswith(("direct8_", "postectopic_direct8_"))
        else feature_profile(name).modality
    )


def classical_negative_mask(frame):
    """Prespecified research subgroup, NOT a comprehensive normal-ECG diagnosis."""
    values = frame.reindex(columns=NORMALITY_FLAGS).apply(
        pd.to_numeric, errors="coerce"
    )
    assessable = np.isfinite(values).all(axis=1)
    return assessable & values.eq(0).all(axis=1), {
        "version": "1.0.0",
        "rule": "all_required_flags_observed_and_zero",
        "flags": list(NORMALITY_FLAGS),
        "n_assessable": int(assessable.sum()),
        "missing_columns": sorted(set(NORMALITY_FLAGS) - set(frame)),
        "scope": "prespecified_classical_criteria_negative_not_proof_of_normal_ecg",
    }


def task_registry():
    return {
        key: {
            "endpoint_id": key,
            "endpoint_kind": value.target_kind.value,
            "index_policy": value.index_policy,
            "positive_rule": value.positive_rule,
            "negative_rule": value.negative_rule,
            "reference_sources": value.source_tables,
            "window_before_hours": value.window_before_hours,
            "window_after_hours": value.window_after_hours,
            "record_level_reference_required": True,
            "clinical_claim_allowed": False,
        }
        for key, value in definitions_by_id().items()
    }


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def score_family(name):
    # The legacy catalog puts *all* classical rules in one family. A two-per-
    # family search cap must not handicap the whole ECG comparison to two rules.
    if feature_profile(name).modality in {"ecg", "classical_ecg"}:
        return "ecg:" + name
    return conceptual_feature_family(name)


def load_score_cohort(
    config, feature_glob, endpoint, marker_ids, *, max_patients=None, seed=20260914
):
    """Index is fixed upstream BEFORE unknown labels/extraction failures are removed."""
    path = (
        (config.labels_root / "endpoint_index_cohorts.parquet")
        .as_posix()
        .replace("'", "''")
    )
    sql_path = feature_glob.replace("\\", "/").replace("'", "''")
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=min(2, config.workers),
    )
    with catalog.connect(isolated=True, read_only=True) as connection:
        # Sample the already-fixed index cohort, before inspecting reference
        # availability or marker values. This is not case/control enrichment.
        n_available_index = connection.execute(
            f"SELECT count(*) FROM read_parquet('{path}') WHERE split='train' AND label_id=?",
            [endpoint],
        ).fetchone()[0]
        limit = (
            f" ORDER BY hash(subject_id, {int(seed)}), subject_id LIMIT {int(max_patients)}"
            if max_patients
            else ""
        )
        cohort = connection.execute(
            f"SELECT * FROM read_parquet('{path}') WHERE split = 'train' AND label_id = ?"
            + limit,
            [endpoint],
        ).fetchdf()
        if cohort.empty:
            return pd.DataFrame(), {"n_index": 0}
        if cohort.subject_id.duplicated().any():
            raise ValueError("duplicate_reference_blind_index_patient")
        original_count = len(cohort)
        cohort = annotate_reference_rows(cohort)
        kind_counts = cohort.reference_kind.value_counts().to_dict()
        # This default discovery pass is for independent quantitative diagnoses.
        # No table-name-only or device-report label can pass this gate.
        cohort = cohort.loc[
            cohort.reference_kind.eq("timed_quantitative")
            & cohort.label_state.isin(["positive", "negative"])
            & cohort.eligibility_state.eq("eligible")
        ].copy()
        available = parquet_columns(connection, sql_path)
        selected = [m for m in marker_ids if m in available]
        counts = {
            "n_index": original_count,
            "n_index_available": int(n_available_index),
            "sampling": "deterministic_patient_hash_before_reference_filter",
            "n_not_selected_by_budget": int(n_available_index) - original_count,
            "n_verified_reference": len(cohort),
            "reference_kind_counts": kind_counts,
            "n_clock_alignment_confirmed": int(
                cohort.reference_alignment_confirmed.sum()
            ),
            "time_claim": "recorded_time_association; device_EHR_clock_alignment_not_assumed",
        }
        if not selected or cohort.empty:
            return pd.DataFrame(), counts
        features = feature_parquet_clause(connection=connection, sql_path=sql_path)
        connection.register("score_index", cohort)
        columns = ",".join("f." + _quote(m) for m in selected)
        master_path = (
            getattr(config, "master_root", Path("__absent_master__"))
            / "ecg_master_index.parquet"
        )
        context_projection, context_join = "", ""
        activation_projection = ""
        for name in ACTIVATION_MODEL_COLUMNS:
            if name in available and name not in selected:
                activation_projection += ", f." + _quote(name)
        required_master = {
            "subject_id",
            "study_id",
            "split",
            "age",
            "sex",
            "machine_heart_rate_bpm",
            "icustay_id",
            "ed_stay_id",
            "hadm_id",
        }
        if master_path.is_file():
            master_sql = master_path.as_posix().replace("'", "''")
            master_columns = set(parquet_columns(connection, master_sql))
            if (
                required_master <= master_columns
                and "signal_quality_score" in available
            ):
                hr = (
                    "COALESCE(f.context_heart_rate_bpm, m.machine_heart_rate_bpm)"
                    if "context_heart_rate_bpm" in available
                    else "m.machine_heart_rate_bpm"
                )
                context_projection = f""", m.age AS context_age,
                    CASE WHEN m.sex='M' THEN 1.0 WHEN m.sex='F' THEN 0.0 END AS context_sex_male,
                    {hr} AS context_heart_rate, f.signal_quality_score AS context_quality,
                    CASE WHEN m.study_id IS NULL THEN NULL ELSE CAST(m.icustay_id IS NOT NULL AS DOUBLE) END AS context_icu,
                    CASE WHEN m.study_id IS NULL THEN NULL ELSE CAST(m.icustay_id IS NULL AND m.ed_stay_id IS NOT NULL AS DOUBLE) END AS context_ed,
                    CASE WHEN m.study_id IS NULL THEN NULL ELSE CAST(m.icustay_id IS NULL AND m.ed_stay_id IS NULL AND m.hadm_id IS NOT NULL AS DOUBLE) END AS context_inpatient
                """
                # master_index only links stays overlapping the index ECG.
                # No length of stay, future treatments or subsequent labs enter.
                context_join = f"LEFT JOIN read_parquet('{master_sql}') m ON c.subject_id=m.subject_id AND c.study_id=m.study_id AND m.split='train'"
        frame = connection.execute(f"""
            SELECT c.*, f.extraction_status, {columns} {context_projection} {activation_projection}
            FROM score_index c LEFT JOIN {features} f
              ON c.subject_id=f.subject_id AND c.study_id=f.study_id AND f.split='train'
            {context_join}
            ORDER BY c.subject_id, c.study_id
        """).fetchdf()
    if frame.subject_id.duplicated().any():
        raise ValueError("duplicate_feature_record")
    counts["n_extraction_ok"] = int(frame.extraction_status.eq("ok").sum())
    counts["marker_availability_before_filter"] = {
        name: float(np.isfinite(pd.to_numeric(frame[name], errors="coerce")).mean())
        for name in selected
    }
    # Keep unmeasurable index rows in the denominator; no substitution ECG.
    frame.loc[~frame.extraction_status.eq("ok"), selected] = np.nan
    for name in ACTIVATION_MODEL_COLUMNS:
        if name in frame:
            frame.loc[~frame.extraction_status.eq("ok"), name] = np.nan
    counts["clinical_context_available"] = all(
        name in frame for name in CLINICAL_COMPONENTS
    )
    return frame, counts


def _source_hash():
    root = Path(__file__).parent
    return content_hash(
        {
            name: file_sha256(root / name)
            for name in (
                "score_stage.py",
                "pre_run_extensions.py",
                "pre_run_extension_stage.py",
                "score_research.py",
                "optimization.py",
                "decision_research.py",
                "focus_localization.py",
                "research_extensions.py",
                "causal_treatment.py",
                "mapping_comparison.py",
                "score_outcomes.py",
                "feature_roles.py",
                "reference_tiers.py",
                "label_registry.py",
                "master_index.py",
            )
        }
    )


def run_score_discovery(
    config,
    *,
    run_id,
    panel_ids=("primary_core", "sentinel", "classical_criteria"),
    feature_glob=None,
    endpoint_ids=None,
    max_endpoints=6,
    max_patients=50000,
    policy=None,
    preview=False,
    force=False,
    progress=None,
):
    """Automatic finite first pass; deep search remains separately invoked."""
    if Path(run_id).name != run_id or run_id in ("", ".", "..") or ":" in run_id:
        raise ValueError("invalid_run_id")
    if max_endpoints < 1:
        raise ValueError("positive_endpoint_budget_required")
    if max_patients is not None and max_patients < 1:
        raise ValueError("positive_patient_budget_required")
    # The fixed cap of two processes had no recorded reason; memory is already
    # bounded per fold wave in score_research. The live grant is the ceiling
    # now (audit R13).
    from .resource_grants import worker_ceiling

    policy = policy or SearchPolicy(
        seed=config.random_seed,
        workers=max(1, worker_ceiling(max(1, int(config.workers)))),
    )
    root = config.runs_root / run_id / "score_discovery"
    root.mkdir(parents=True, exist_ok=True)
    registry = task_registry()
    requested = tuple(dict.fromkeys(endpoint_ids or PRIORITY_ENDPOINTS))
    if set(requested) - set(registry):
        raise ValueError("unregistered_score_endpoint")
    glob_pattern = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    from .validation import (
        verify_feature_artifacts,
        verify_development_label_attestation,
        _verify_current_development_feature_inputs,
        _resolved_feature_scope,
        _resolved_sampling_design,
        _probability_models_allowed,
    )

    paths = sorted(Path(p) for p in glob.glob(glob_pattern))
    # Reject the paths BEFORE opening any parquet footer or reading rows.
    if any(
        any(
            part.casefold() in {"sealed", "locked_test"}
            or (
                part.casefold().startswith("split=")
                and part.casefold() not in {"split=train", "split=validation"}
            )
            for part in p.parts
        )
        for p in paths
    ):
        raise ValueError("locked_feature_paths_forbidden")
    paths = [
        p for p in paths if "split=train" in p.parts or "split=validation" in p.parts
    ]
    reasons = []
    integrity = None
    attestation = None
    sampling_design = "unverified"
    if preview:
        reasons.append("preview_not_eligible_for_automatic_score_discovery")
    elif not paths:
        reasons.append("missing_development_features")
    else:
        integrity = verify_feature_artifacts(paths)
        scope = _resolved_feature_scope("explicit", integrity)
        sampling_design = _resolved_sampling_design(scope, integrity)
        if not _probability_models_allowed(
            preview=False, sampling_design=sampling_design
        ):
            raise ValueError(
                "nonrepresentative_feature_sample_cannot_train_probability_score"
            )
        _verify_current_development_feature_inputs(
            config, feature_integrity=integrity, panel_ids=panel_ids
        )
        attestation = verify_development_label_attestation(config)
    cohort_path = config.labels_root / "endpoint_index_cohorts.parquet"
    master_path = config.master_root / "ecg_master_index.parquet"
    if not cohort_path.is_file():
        reasons.append("missing_reference_blind_index")
    inputs = {
        "version": VERSION,
        "source_hash": _source_hash(),
        "policy": asdict(policy),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "duckdb",
            )
        },
        "feature_integrity": integrity,
        "sampling_design": sampling_design,
        "label_attestation": attestation,
        "cohort_sha256": file_sha256(cohort_path) if cohort_path.is_file() else None,
        "master_sha256": file_sha256(master_path) if master_path.is_file() else None,
        "endpoints": requested,
        "max_endpoints": max_endpoints,
        "panels": panel_ids,
        "max_patients": max_patients,
        "candidate_panels": [p for p in panel_ids if p != "discovery_reserve"],
        "registry_hash": content_hash(registry),
        "config_hash": scientific_config_hash(config),
    }
    input_hash = content_hash(inputs)
    # A changed algorithm/input gets a new directory, never overwrites old pilot output.
    destination = root / input_hash[:20]
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "score_discovery.manifest.json"
    if manifest_path.exists() and not force:
        manifest = StageManifest.load(manifest_path)
        if manifest.matches(input_hash, scientific_config_hash(config), VERSION):
            return manifest
    atomic_write_json(destination / "inputs.json", inputs)
    panels = {p.panel_id: p for p in default_feature_panels()}
    marker_ids = tuple(
        dict.fromkeys(
            m
            for panel in panel_ids
            if panel != "discovery_reserve"
            for m in panels[panel].feature_ids
            if is_combination_candidate(m)
            or feature_profile(m).modality in {"classical_ecg", "ecg"}
        )
    )
    from ..core.biomarkers import BiomarkerRegistry

    registered = set(BiomarkerRegistry.available(include_metadata_only=True))
    units = {
        name: BiomarkerRegistry.definition(name).unit
        if name in registered
        else "unverified"
        for name in marker_ids
    }
    units.update(
        dict(
            zip(
                CLINICAL_COMPONENTS,
                (
                    "years",
                    "binary",
                    "beats/min",
                    "unitless",
                    "binary",
                    "binary",
                    "binary",
                ),
            )
        )
    )
    summaries, outputs = [], [destination / "inputs.json"]
    for position, endpoint in enumerate(requested):
        if progress:
            progress(
                f"Score research endpoint {position + 1}/{len(requested)}: {endpoint}"
            )
        if reasons or position >= max_endpoints:
            summaries.append(
                {
                    "endpoint_id": endpoint,
                    "status": "preflight_abstention" if reasons else "budget_exhausted",
                    "reasons": reasons or ["endpoint_budget"],
                    "clinical_claim_allowed": False,
                }
            )
            continue
        checkpoint = destination / (endpoint + ".json")
        checkpoint_integrity = destination / (endpoint + ".sha256.json")
        if checkpoint.exists() and checkpoint_integrity.exists() and not force:
            digest = json.loads(checkpoint_integrity.read_text(encoding="utf-8"))
            if (
                digest.get("sha256") == file_sha256(checkpoint)
                and digest.get("input_hash") == input_hash
            ):
                summaries.append(json.loads(checkpoint.read_text(encoding="utf-8")))
                outputs.extend([checkpoint, checkpoint_integrity])
                prediction_file = destination / (endpoint + ".oof.parquet")
                if digest.get("prediction_sha256") and not prediction_file.exists():
                    raise ValueError("checkpoint_predictions_missing")
                if prediction_file.exists():
                    if digest.get("prediction_sha256") != file_sha256(prediction_file):
                        raise ValueError("checkpoint_prediction_integrity_mismatch")
                    outputs.append(prediction_file)
                continue
        frame, counts = load_score_cohort(
            config,
            glob_pattern,
            endpoint,
            marker_ids,
            max_patients=max_patients,
            seed=policy.seed,
        )
        result = {
            "endpoint_id": endpoint,
            "availability": counts,
            "models": {},
            "clinical_claim_allowed": False,
            "status": "evaluated",
        }
        prediction_path = None
        if frame.empty:
            result.update(
                status="non_estimable", reason="no_independently_verified_index_records"
            )
        else:
            columns = [name for name in marker_ids if name in frame]
            ecg = [
                name
                for name in columns
                if score_modality(name) in {"ecg", "classical_ecg"}
            ]
            vcg = [name for name in columns if score_modality(name) == "vcg"]
            direct = [
                name
                for name in columns
                if score_modality(name) == "direct_ecg_geometry"
            ]
            predictions = frame[["subject_id", "study_id"]].copy()
            activation_columns = ("qrs_duration_ms", "context_heart_rate_bpm", "rhythm_context", "t_loop_width_major")
            result["activation_conditioned_repolarization"] = {
                "status": "non_estimable", "reason": "required_activation_or_target_columns_missing",
                "clinical_claim_allowed": False,
            }
            if all(name in frame for name in activation_columns):
                activation_frame = frame[["subject_id", "study_id", *activation_columns]].copy()
                activation_frame["source"] = "mimic"
                activation_frame["split"] = "train"
                known_rhythm = activation_frame.rhythm_context.isin([
                    "regular", "irregular", "af_compatible_pattern", "flutter_compatible_pattern",
                    "multiple_atrial_activity", "wide_qrs_pattern", "mixed_or_ectopic_pattern"])
                rhythm_predictors = []
                for rhythm in ("irregular", "af_compatible_pattern", "flutter_compatible_pattern",
                               "multiple_atrial_activity", "wide_qrs_pattern", "mixed_or_ectopic_pattern"):
                    name = "activation_rhythm_" + rhythm
                    activation_frame[name] = activation_frame.rhythm_context.eq(rhythm).astype(float).where(known_rhythm)
                    rhythm_predictors.append(name)
                activation_predictors = ["qrs_duration_ms", "context_heart_rate_bpm", *rhythm_predictors]
                residuals, models = activation_residuals(activation_frame,
                    activation_predictors, ["t_loop_width_major"])
                predictions["exploratory_t_width_activation_residual"] = residuals.residual.to_numpy()
                result["activation_conditioned_repolarization"] = {
                    "status": "evaluated", "models": models, "n_available": int(residuals.residual.notna().sum()),
                    "score_search_input": False, "clinical_claim_allowed": False,
                    "interpretation": "cross_fitted_conditional_morphology_residual_not_causal_decomposition",
                }
                # The same conditioning for the QRS-T coupling markers, one
                # model per marker so that each keeps its own availability
                # instead of the intersection of all of them.
                coupling = {}
                for target in QRS_T_COUPLING_MARKERS:
                    if target not in frame:
                        coupling[target] = {"status": "non_estimable", "reason": "target_column_missing"}
                        continue
                    activation_frame[target] = frame[target].to_numpy()
                    target_residuals, target_models = activation_residuals(
                        activation_frame, activation_predictors, [target])
                    predictions[f"exploratory_{target}_activation_residual"] = (
                        target_residuals.residual.to_numpy())
                    coupling[target] = {
                        "status": "evaluated", "models": target_models,
                        "n_available": int(target_residuals.residual.notna().sum()),
                    }
                result["activation_conditioned_qrs_t_coupling"] = {
                    "targets": coupling, "score_search_input": False, "clinical_claim_allowed": False,
                    "interpretation": "cross_fitted_conditional_morphology_residual_not_causal_decomposition",
                }
            y = frame.label_state.eq("positive").astype(int).to_numpy()
            groups = frame.subject_id.to_numpy()
            normal_mask, normal_rule = classical_negative_mask(frame)
            result["classical_negative_subgroup"] = {
                "rule": normal_rule,
                "n": int(normal_mask.sum()),
                "models": {},
            }
            comparisons: list[tuple[str, list[str], tuple[str, ...]]] = [
                ("ecg", ecg, ()),
                ("vcg", vcg, ()),
                ("ecg_vcg", ecg + direct + vcg, ()),
            ]
            if direct:
                comparisons.append(("ecg_extended", ecg + direct, ()))
            result["modality_contract"] = {
                "direct_eight_lead_geometry_is_transformed_vcg": False,
                "direct_ecg_geometry_features": direct,
            }
            context_ready = all(
                name in frame
                and np.isfinite(pd.to_numeric(frame[name], errors="coerce")).mean()
                >= policy.minimum_available
                for name in CLINICAL_COMPONENTS
            )
            if context_ready:
                comparisons.extend(
                    (name, list(CLINICAL_COMPONENTS) + markers, CLINICAL_COMPONENTS)
                    for name, markers in (
                        ("clinical_context", []),
                        ("clinical_ecg", ecg),
                        ("clinical_vcg", vcg),
                        ("clinical_ecg_vcg", ecg + direct + vcg),
                    )
                )
                if direct:
                    comparisons.append(
                        (
                            "clinical_ecg_extended",
                            list(CLINICAL_COMPONENTS) + ecg + direct,
                            CLINICAL_COMPONENTS,
                        )
                    )
            else:
                result["models"]["clinical_context"] = {
                    "status": "non_estimable",
                    "reason": "insufficient_prediction_time_context",
                }
            for group, names, fixed in comparisons:

                def phase_progress(phase, completed, total):
                    if progress:
                        progress(
                            f"Score endpoints completed {position}/{len(requested)} | {endpoint} | {group} | {phase}: {completed}/{total}"
                        )

                try:
                    if not names:
                        raise NotEstimable("no_features_for_modality")
                    model = discover_binary_score(
                        frame[names],
                        y,
                        groups,
                        endpoint_id=endpoint,
                        model_group=group,
                        family=score_family,
                        policy=replace(policy, workers=1) if len(frame) < 5000 else policy,
                        progress=phase_progress,
                        fixed_components=fixed,
                        provenance={
                            "split": "train",
                            "input_hash": input_hash,
                            "reference_kind": "timed_quantitative",
                            "index_policy": "reference_blind",
                            "target_population": "measured_reference_subset_not_general_screening_population",
                            "feature_units": units,
                            "context_available_at_decision_verified": bool(fixed),
                            "context_time_scope": "master_index_recorded_stay_overlap; not external device_clock_synchronization",
                            "pilot_informed_hypothesis": True,
                        },
                    )
                    predictions[group] = model.pop("oof_predictions")
                    from .score_research import probability_metrics

                    result["classical_negative_subgroup"]["models"][group] = (
                        probability_metrics(
                            y[normal_mask],
                            predictions[group].to_numpy()[normal_mask],
                            groups[normal_mask],
                            iterations=policy.bootstrap_iterations,
                            seed=policy.seed,
                        )
                        if normal_mask.any()
                        else {
                            "status": "non_estimable",
                            "reason": "no_assessable_classical_negative_records",
                        }
                    )
                    result["models"][group] = model
                except NotEstimable as exc:
                    result["models"][group] = {
                        "status": "non_estimable",
                        "reason": str(exc),
                    }
            result["paired_comparisons"] = {
                name + "_vs_ecg": paired_probability_comparison(
                    y,
                    predictions["ecg"],
                    predictions[name],
                    groups,
                    iterations=policy.bootstrap_iterations,
                    seed=policy.seed,
                )
                for name in ("vcg", "ecg_vcg", "ecg_extended")
                if name in predictions and "ecg" in predictions
            }
            result["classical_negative_subgroup"]["paired_comparisons"] = {
                name + "_vs_ecg": paired_probability_comparison(
                    y[normal_mask], predictions["ecg"].to_numpy()[normal_mask],
                    predictions[name].to_numpy()[normal_mask], groups[normal_mask],
                    iterations=policy.bootstrap_iterations, seed=policy.seed)
                for name in ("vcg", "ecg_vcg", "ecg_extended")
                if normal_mask.any() and name in predictions and "ecg" in predictions
            }
            for baseline, augmented in (
                ("clinical_context", "clinical_ecg"),
                ("clinical_context", "clinical_vcg"),
                ("clinical_context", "clinical_ecg_vcg"),
                ("clinical_ecg", "clinical_ecg_vcg"),
                ("ecg_extended", "ecg_vcg"),
                ("clinical_ecg_extended", "clinical_ecg_vcg"),
            ):
                if baseline in predictions and augmented in predictions:
                    result["paired_comparisons"][augmented + "_vs_" + baseline] = (
                        paired_probability_comparison(
                            y,
                            predictions[baseline],
                            predictions[augmented],
                            groups,
                            iterations=policy.bootstrap_iterations,
                            seed=policy.seed,
                        )
                    )
            prediction_path = destination / (endpoint + ".oof.parquet")
            atomic_write_parquet(predictions, prediction_path)
            outputs.append(prediction_path)
        atomic_write_json(checkpoint, result)
        atomic_write_json(
            checkpoint_integrity,
            {
                "input_hash": input_hash,
                "sha256": file_sha256(checkpoint),
                "prediction_sha256": file_sha256(prediction_path)
                if prediction_path
                else None,
            },
        )
        summaries.append(result)
        outputs.extend([checkpoint, checkpoint_integrity])
    report = {
        "version": VERSION,
        "input_hash": input_hash,
        "results": summaries,
        "task_registry": registry,
        "clinical_claim_allowed": False,
        "locked_test_read": False,
        "pre_run_followup_contract": research_contract(),
        "treatment_readiness": {
            name: TreatmentProfile(
                name, PROFILE_DESIGNS.get(name, "physiological_response")
            ).readiness()
            for name in PROFILES
        },
        "localization": mapping_inventory(config.work_root / "mapping_imports"),
        "research_extensions": {
            "readiness": research_readiness(config.work_root),
            "causal_backend": "cross_fitted_aipw_fixed_horizon_time_zero_strategies",
            "mapping_backend": "paired_native_ecg_kors_inverse_dower_svd_invariants",
            "entrypoint": "python -m vcg_biomarker.mimic.research_extensions",
            "execution": "explicit_verified_protocol_manifest_only",
            "clinical_claim_allowed": False,
        },
        "separate_validation": "not_run; apply_train_frozen_specs_only",
        "normal_ecg_subgroup": "prespecified_classical_criteria_negative; see per-endpoint coverage; not comprehensive ECG normality",
        "continuous_and_survival": "requires_typed_outcome_adapter",
        "deep_search": "separate_opt_in_stage",
        "reporting_limit": "nested estimates describe the search procedure, not independent replication of the final formula",
    }
    if inputs["cohort_sha256"] and file_sha256(cohort_path) != inputs["cohort_sha256"]:
        raise RuntimeError("upstream_index_changed_during_score_search")
    if _source_hash() != inputs["source_hash"]:
        raise RuntimeError("analysis_source_changed_during_score_search")
    if inputs["master_sha256"] and file_sha256(master_path) != inputs["master_sha256"]:
        raise RuntimeError("upstream_master_changed_during_score_search")
    report_path = destination / "score_research_report.json"
    atomic_write_json(report_path, report)
    outputs.append(report_path)
    manifest = StageManifest.complete(
        stage="score-discovery",
        stage_version=VERSION,
        input_hash=input_hash,
        config_hash=scientific_config_hash(config),
        output_paths=tuple(str(p.resolve()) for p in outputs),
        row_counts={"endpoints": len(summaries)},
        output_integrity=capture_output_integrity(tuple(outputs)),
        status="complete" if not reasons else "complete_without_eligible_inputs",
        notes=("research_only", "locked_test_read=False", "gui_independent_followup"),
    )
    manifest.save(manifest_path)
    return manifest
