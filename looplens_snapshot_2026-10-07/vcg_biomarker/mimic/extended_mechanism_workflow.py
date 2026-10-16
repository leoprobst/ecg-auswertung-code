"""Development-only production adapter for the extended research models."""
import json

import numpy as np
import pandas as pd

from .extended_mechanism_models import (
    abstain, electromechanical_residuals, pr_geometry_coupling, within_between_echo_model,
)
from .mechanism_echo_context import load_echo_context, attach_echo_context, compare_echo_pair_context

VERSION = "1.1.0"
MARKERS = ("p_duration_ms", "p_loop_terminal_area_fraction", "qrs_local24ms_cancellation_fraction")
ECHO_COLUMNS = ("la_volume_index_direct", "tapse_cm")


def analyze_extended_records(records, measurements):
    """Input is clock-deduplicated before reference/feature missingness selection."""
    if records.subject_id.isna().any() or records.duplicated(["subject_id", "ecg_time"]).any():
        raise ValueError("Known patients and unique selected ECG clocks required")
    if not records.split.isin(["train", "validation"]).all():
        raise ValueError("Locked/unknown records forbidden")
    if records.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("Patients cross development splits")
    records = records.sort_values(["subject_id", "ecg_time", "study_id"], kind="stable").reset_index(drop=True)
    enriched = attach_echo_context(records, measurements, columns=ECHO_COLUMNS)
    qc = (enriched.extraction_status.eq("ok") & enriched.signal_quality_score.between(.5, 1)
          & np.isfinite(enriched.signal_quality_score))
    enriched.loc[~qc, list(MARKERS)] = np.nan
    # Select adjacent ECGs BEFORE looking at any reference or marker values.
    shift_source = records.copy()
    # Preserve integer identities beyond float64's exact range when shift adds NA.
    for name in ("study_id", "nearest_echo_measurement_id"):
        shift_source[name] = shift_source[name].astype(object)
    next_row = shift_source.groupby("subject_id").shift(-1)
    mask = next_row.study_id.notna()
    before = records.loc[mask].copy()
    after = next_row.loc[mask].copy()
    after["subject_id"] = before.subject_id
    before["pair_id"] = np.arange(len(before))
    after["pair_id"] = before.pair_id
    pairs = compare_echo_pair_context(before, after, measurements, columns=ECHO_COLUMNS)
    pairs["first_study_id"] = before.study_id.to_numpy()
    pairs["second_study_id"] = after.study_id.to_numpy()
    # Restrict repeated-measure models to endpoints of qualified distinct pairs.
    models = []
    for split in ("train", "validation"):
        for echo in ECHO_COLUMNS:
            qualified = pairs.loc[pairs.split.eq(split) & pairs[echo + "_available"]]
            keys = pd.concat([qualified[["subject_id", name]].rename(columns={name: "study_id"})
                              for name in ("first_study_id", "second_study_id")]).drop_duplicates()
            repeated = enriched.merge(keys, on=["subject_id", "study_id"], how="inner", validate="one_to_one")
            # The same echo reused for several ECGs is not another observation.
            repeated = repeated.drop_duplicates(["subject_id", "nearest_echo_measurement_id"], keep="first")
            for marker in MARKERS:
                fit = within_between_echo_model(repeated.rename(columns={marker: "marker_value", echo: "echo_value"}))
                models.append({"split": split, "feature_id": marker, "echo_measure": echo,
                               "status": fit["status"], "details_json": json.dumps(fit, allow_nan=False)})
    from statsmodels.stats.multitest import multipletests
    # All prespecified split/marker/echo/term tests retain their family positions.
    planned = len(MARKERS) * len(ECHO_COLUMNS) * 2 * 2
    pvalues, locations = np.ones(planned), []
    details = [json.loads(row["details_json"]) for row in models]
    for row_index, fit in enumerate(details):
        for term_index, term in enumerate(fit.get("terms", [])):
            pvalues[len(locations)] = term["p_value"]
            locations.append((row_index, term_index))
    qvalues = multipletests(pvalues, method="fdr_bh")[1]
    for value, (row_index, term_index) in zip(qvalues, locations):
        details[row_index]["terms"][term_index]["q_value"] = float(value)
    for row, fit in zip(models, details):
        fit["fdr_planned_tests"] = planned
        fit["fdr_family"] = "extended_echo_within_between_all_prespecified_terms"
        row["details_json"] = json.dumps(fit, allow_nan=False)
    # Earliest patient index is selected BEFORE complete-case filtering.
    index = enriched.drop_duplicates("subject_id", keep="first")
    index = index.copy()
    index["sex_male"] = index.sex.astype("string").str.upper().map({"M": 1., "F": 0.})
    index["care_ed"] = np.where(index.ed_stay_id.notna(), 1.,
                                np.where(index.hadm_id.notna(), 0., np.nan))
    residuals, residual_model = electromechanical_residuals(index,
        predictors=(*MARKERS, "age", "sex_male", "context_heart_rate_bpm", "signal_quality_score", "care_ed"),
        mechanical=ECHO_COLUMNS)
    return pairs, pd.DataFrame(models), residuals, residual_model


def run_extended_mechanisms(connection, config, feature_relation, available, *, preview=False):
    """Called by within-subject stage; no separate patient job is launched."""
    wide = config.evidence_root / "echo_measurements_wide.parquet"
    identities = config.evidence_root / "echo_study_index.parquet"
    empty = pd.DataFrame({"status": pd.Series(dtype=str)})
    summary = {"version": VERSION, "clinical_claim_allowed": False,
               "unverified_requirements": ["rhythm_and_pacing", "echo_measurement_method_comparability",
                   "time_varying_confounders", "informative_observation_times", "external_validation"],
               "interpretation": "exploratory_cross_sectional_and_serial_not_clinical_validation"}
    outputs = {"echo_pairs": empty.copy(), "echo_models": empty.copy(),
               "electromechanical_residuals": empty.copy(), "pr_geometry_models": empty.copy()}
    if preview:
        return outputs, {**summary, "status": "non_estimable", "reason": "pilot_no_confirmation"}
    if wide.is_file() and identities.is_file():
        from .within_subject import _paired_ecg_source, _sql
        import pyarrow.parquet as pq
        if not set(ECHO_COLUMNS) <= set(pq.read_schema(wide).names):
            summary["echo"] = abstain("requested_echo_measurement_columns_missing")
        else:
            measurements = load_echo_context(wide, identities, columns=ECHO_COLUMNS)
            source = _paired_ecg_source(_sql(config.master_root / "ecg_master_index.parquet"), "'train','validation'")
            projection = ", ".join(f'f."{name}"' if name in available else f'NULL::DOUBLE AS "{name}"'
                                   for name in (*MARKERS, "context_heart_rate_bpm"))
            outer_projection = ", ".join(f'f."{name}"' for name in (*MARKERS, "context_heart_rate_bpm"))
            records = connection.execute(f"""WITH projected_features AS MATERIALIZED (
                SELECT f.subject_id, f.study_id, f.split, f.extraction_status, f.signal_quality_score,
                    {projection} FROM {feature_relation} f WHERE f.split IN ('train','validation'))
                SELECT m.subject_id, m.study_id, m.split, m.ecg_time,
                m.age, m.sex, m.hadm_id, m.ed_stay_id, m.nearest_echo_measurement_id, m.nearest_echo_delta_hours,
                f.extraction_status, f.signal_quality_score, {outer_projection}
                FROM {source} m LEFT JOIN projected_features f
                ON m.subject_id=f.subject_id AND m.study_id=f.study_id AND m.split=f.split
                WHERE m.subject_id IN (SELECT subject_id FROM projected_features)""").df()
            pairs, models, residuals, residual_model = analyze_extended_records(records, measurements)
            outputs.update(echo_pairs=pairs, echo_models=models, electromechanical_residuals=residuals)
            summary["echo"] = {"status": "computed", "n_selected_pairs": len(pairs),
                               "pair_policy": "adjacent_reference_blind_ECGs_no_fallback"}
            summary["electromechanical_model"] = residual_model
    else:
        summary["echo"] = abstain("echo_measurements_or_identity_index_missing")
    # This contract must be provided by independently qualified beat-level data,
    # never manufactured from a ten-second record median or assumed zero error.
    pr_path = config.evidence_root / "pr_geometry_qualified.parquet"
    if not pr_path.is_file():
        summary["pr_geometry"] = abstain("qualified_beat_pairs_and_error_calibration_missing")
    else:
        from .pr_calibration_contract import load_approved_calibration
        certificate, reason = load_approved_calibration(
            pr_path.with_name("pr_geometry_calibration.json"), pr_path)
        if certificate is None:
            summary["pr_geometry"] = abstain(reason)
            return outputs, summary
        pr = pd.read_parquet(pr_path)
        required = {"subject_id", "study_id", "split", "beat_time_s", "pr_ms", "geometry",
                    "pr_error_variance_ms2", "error_covariance", "qualification_verified", "calibration_reference",
                    "geometry_marker_id", "transform", "measurement_version"}
        if not required <= set(pr):
            raise ValueError("Incomplete qualified PR measurement contract")
        if not pr.split.isin(["train", "validation"]).all() or pr.groupby("subject_id").split.nunique().gt(1).any():
            raise ValueError("PR records violate development isolation")
        rows = []
        for (subject, study, split), group in pr.groupby(["subject_id", "study_id", "split"], dropna=False):
            qualified = (group.qualification_verified.eq(True).all()
                and group.calibration_reference.astype("string").str.strip().ne("").fillna(False).all()
                and group.calibration_reference.nunique() == 1
                and group.pr_error_variance_ms2.nunique() == 1 and group.error_covariance.nunique() == 1
                and group.pr_error_variance_ms2.eq(certificate["pr_error_variance_ms2"]).all()
                and group.error_covariance.eq(certificate["error_covariance"]).all()
                and group["transform"].eq(certificate["transform"]).all()
                and group.measurement_version.eq(certificate["measurement_version"]).all()
                and group.beat_time_s.notna().all() and group.beat_time_s.is_unique
                and np.isfinite(group.beat_time_s).all()
                and group.beat_time_s.ge(0).all()
                and group.geometry_marker_id.eq("qrs_initial_terminal_energy_balance").all()
                and all(group[key].astype("string").str.strip().ne("").fillna(False).all()
                        and group[key].nunique() == 1 for key in ("transform", "measurement_version"))
                and pd.notna(subject) and pd.notna(study))
            fit = (pr_geometry_coupling(group.pr_ms, group.geometry,
                    pr_error_variance_ms2=group.pr_error_variance_ms2.iloc[0],
                    error_covariance=[group.error_covariance.iloc[0]]) if qualified else abstain("unqualified_PR_error_or_timing_contract"))
            rows.append({"subject_id": subject, "study_id": study, "split": split,
                         "geometry_marker_id": "qrs_initial_terminal_energy_balance",
                         "calibration_reference": str(group.calibration_reference.iloc[0]),
                         "transform": str(group["transform"].iloc[0]),
                         "measurement_version": str(group.measurement_version.iloc[0]),
                         "calibration_certificate_sha256": certificate["certificate_sha256"],
                         "status": fit["status"], "details_json": json.dumps(fit, allow_nan=False)})
        outputs["pr_geometry_models"] = pd.DataFrame(rows)
        summary["pr_geometry"] = {"status": "computed", "n_records": len(rows)}
    return outputs, summary
