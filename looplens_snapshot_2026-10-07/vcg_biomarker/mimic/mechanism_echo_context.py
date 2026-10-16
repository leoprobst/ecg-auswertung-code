"""Attach numeric echo context without choosing a better-covered index ECG."""

import numpy as np
import pandas as pd

VERSION = "1.3.0"


def load_echo_context(wide_path, study_index_path, *, columns):
    """Read only requested measurements and independently recorded identity QC."""
    wide = pd.read_parquet(wide_path, columns=["measurement_id", "subject_id", "echo_time", *columns])
    identities = pd.read_parquet(study_index_path,
                                 columns=["measurement_id", "subject_id", "identity_conflict"])
    if (identities.measurement_id.isna().any()
            or identities.measurement_id.duplicated().any()):
        raise ValueError("Echo study index requires unique known identities")
    identities = identities.rename(columns={"subject_id": "identity_subject_id"})
    result = wide.merge(identities, on="measurement_id", how="left", validate="one_to_one")
    valid = (result.identity_conflict.eq(False).fillna(False)
             & result.subject_id.notna() & result.identity_subject_id.notna()
             & result.subject_id.eq(result.identity_subject_id).fillna(False))
    result["identity_conflict"] = ~valid
    return result.drop(columns="identity_subject_id")


def attach_echo_context(index, measurements, *, columns, max_delta_hours=720.0):
    """Cross-sectional nearest-echo context, never ECG-time prediction inputs.

    Requires explicit patient identity on both sides. No fallback to a different
    measurement or record is allowed. Existing row order and index are preserved.
    """
    if not np.isfinite(max_delta_hours) or max_delta_hours < 0:
        raise ValueError("Finite nonnegative echo window required")
    reserved = {"mechanism_echo_context_eligible", "mechanism_echo_context_temporality"}
    if (len(set(columns)) != len(columns) or set(columns) & (set(index.columns) | reserved)
            or reserved & set(index.columns)):
        raise ValueError("Context columns must be unique and cannot overwrite index fields")
    required = {"subject_id", "ecg_time", "nearest_echo_measurement_id", "nearest_echo_delta_hours"}
    if not required <= set(index):
        raise ValueError("Missing index echo identity or alignment fields")
    if not {"subject_id", "measurement_id", "echo_time", "identity_conflict", *columns} <= set(measurements):
        raise ValueError("Missing echo patient identity or requested measurements")
    if measurements.measurement_id.isna().any() or measurements.measurement_id.duplicated().any():
        raise ValueError("Echo measurement identities must be known and unique")
    output = index.copy()
    aligned = measurements.set_index("measurement_id").reindex(
        index.nearest_echo_measurement_id.to_numpy()).reset_index(drop=True)
    delta = pd.to_numeric(index.nearest_echo_delta_hours, errors="coerce").to_numpy(dtype=float)
    ecg_time = pd.to_datetime(index.ecg_time.reset_index(drop=True), errors="coerce", utc=True)
    echo_time = pd.to_datetime(aligned.echo_time, errors="coerce", utc=True)
    measured_delta = (echo_time - ecg_time).dt.total_seconds().to_numpy() / 3600.0
    # Upstream DATE_DIFF counts integer second boundaries. No sign reversal.
    clock_ok = (np.isfinite(measured_delta)
                & np.isclose(delta, measured_delta, rtol=0, atol=1/3600)
                & (np.abs(measured_delta) <= max_delta_hours))
    same_patient = (aligned.subject_id.notna().to_numpy()
                    & index.subject_id.notna().to_numpy()
                    & aligned.subject_id.eq(index.subject_id.reset_index(drop=True)).fillna(False).to_numpy(dtype=bool))
    identity_ok = aligned.identity_conflict.eq(False).fillna(False).to_numpy(dtype=bool)
    eligible = same_patient & identity_ok & clock_ok & np.isfinite(delta) & (np.abs(delta) <= max_delta_hours)
    for name in columns:
        values = pd.to_numeric(aligned[name], errors="coerce").to_numpy(dtype=float)
        output[name] = np.where(eligible & np.isfinite(values), values, np.nan)
    output["mechanism_echo_context_eligible"] = eligible
    output["mechanism_echo_context_temporality"] = "cross_sectional_not_ecg_time_prediction"
    return output


def compare_echo_pair_context(before, after, measurements, *, columns, max_delta_hours=720.0):
    """Measure changes for caller-selected pairs; never search or replace a pair.

    Each input has a unique pair_id and may contain repeated patients. Paired
    observations are descriptive, not independent inferential samples. Echo
    measurements may follow ECGs: this is response/context, never prediction.
    Units and measurement definitions must be identical in the supplied columns.
    """
    required = {"pair_id", "subject_id", "split", "ecg_time",
                "nearest_echo_measurement_id", "nearest_echo_delta_hours"}
    for frame in (before, after):
        if not required <= set(frame):
            raise ValueError("Missing echo pair identity fields")
        if frame.pair_id.isna().any() or frame.pair_id.duplicated().any():
            raise ValueError("Echo pair identities must be unique and known")
        if not frame.split.isin(["train", "validation"]).all():
            raise ValueError("Echo pairs forbid locked or unknown splits")
    if set(before.pair_id) != set(after.pair_id):
        raise ValueError("Before and after must describe the same selected pairs")
    # Patient membership must not cross development roles, including across pairs.
    roles = pd.concat([before[["subject_id", "split"]], after[["subject_id", "split"]]])
    if roles.groupby("subject_id").split.nunique().gt(1).any():
        raise ValueError("Patient crosses development splits")
    left = before.reset_index(drop=True)
    right = after.set_index("pair_id").reindex(left.pair_id).reset_index()
    a = attach_echo_context(left, measurements, columns=columns, max_delta_hours=max_delta_hours)
    b = attach_echo_context(right, measurements, columns=columns, max_delta_hours=max_delta_hours)
    echo_times = measurements.set_index("measurement_id").echo_time
    ta = pd.to_datetime(left.nearest_echo_measurement_id.map(echo_times), errors="coerce", utc=True)
    tb = pd.to_datetime(right.nearest_echo_measurement_id.map(echo_times), errors="coerce", utc=True)
    ecg_a = pd.to_datetime(left.ecg_time, errors="coerce", utc=True)
    ecg_b = pd.to_datetime(right.ecg_time, errors="coerce", utc=True)
    reason = pd.Series("", index=left.index, dtype="string")
    checks = (
        (left.subject_id.notna() & left.subject_id.eq(right.subject_id).fillna(False), "patient_mismatch_or_missing"),
        (left.split.eq(right.split), "split_mismatch"),
        (ecg_b.gt(ecg_a), "nonforward_ecg_pair"),
        (a.mechanism_echo_context_eligible & b.mechanism_echo_context_eligible, "echo_identity_or_alignment_unqualified"),
        (left.nearest_echo_measurement_id.ne(right.nearest_echo_measurement_id), "same_echo_measurement"),
        (tb.gt(ta), "nonforward_echo_pair"),
    )
    for valid, label in checks:
        reason.loc[reason.eq("") & ~valid.fillna(False)] = label
    eligible = reason.eq("")
    output = left[["pair_id", "subject_id", "split"]].copy()
    output["echo_before_id"] = left.nearest_echo_measurement_id
    output["echo_after_id"] = right.nearest_echo_measurement_id
    output["echo_interval_hours"] = (tb - ta).dt.total_seconds() / 3600
    output["status"] = np.where(eligible, "descriptive", "non_estimable")
    output["reason"] = reason
    for name in columns:
        available = eligible & np.isfinite(a[name]) & np.isfinite(b[name])
        output[name + "_before"] = a[name].where(available)
        output[name + "_after"] = b[name].where(available)
        output[name + "_delta"] = (b[name] - a[name]).where(available)
        output[name + "_available"] = available
    output["version"] = VERSION
    output["clinical_claim_allowed"] = False
    output["analysis_temporality"] = "paired_echo_response_not_ecg_time_prediction"
    output["measurement_definition_compatibility_verified"] = False
    output["inference_requires_patient_clustering"] = True
    return output
