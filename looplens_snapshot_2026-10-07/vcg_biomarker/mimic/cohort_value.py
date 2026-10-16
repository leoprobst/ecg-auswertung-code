"""Measure what a cohort contributes without equating more rows with better models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from .patient_identity import known_patient_mask


def cohort_value_table(features: pd.DataFrame, *, cohort_column: str,
                       marker_ids: tuple[str, ...], agreement: pd.DataFrame | None = None,
                       survival: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if features.empty:
        return pd.DataFrame(columns=["cohort_id", "n_selected", "n_unique_known_patients"])
    for cohort, group in features.groupby(cohort_column, sort=True):
        patients = group.get("patient_id", pd.Series(None, index=group.index, dtype=object))
        identified = known_patient_mask(patients)
        ok = group.get("extraction_status", pd.Series("unknown", index=group.index)).eq("ok")
        available = [name for name in marker_ids if name in group]
        measured = np.isfinite(group[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan))
        measured &= ok.to_numpy()[:, None]
        from .external_identity_audit import measurement_representative_mask
        representative = measurement_representative_mask(group).to_numpy(dtype=bool)
        unique_denominator = int(representative.sum()) * len(marker_ids)
        row = {"cohort_id": str(cohort), "n_selected": len(group), "n_extraction_ok": int(ok.sum()),
            "n_unique_known_patients": int(patients.loc[identified].nunique()),
            "n_records_unknown_patient": int((~identified).sum()),
            "n_repeated_known_patient_records": int(identified.sum()-patients.loc[identified].nunique()),
            "n_markers_requested": len(marker_ids), "n_markers_with_any_measurement": int(measured.any(axis=0).sum()),
            "marker_cell_availability": float(measured.sum()/(len(group)*len(marker_ids))) if marker_ids else np.nan,
            "reference_scope": "measurement_and_annotation_agreement", "n_outcome_bound_patients": 0,
            "n_observed_deaths": 0, "independence_across_cohorts": "not_proven_by_different_record_ids",
            "model_improvement": "not_established_by_additional_validation_rows", "clinical_claim_allowed": False}
        row["n_measurement_representatives_including_failures"] = int(representative.sum())
        row["unique_measurement_marker_cell_availability"] = float(measured[representative].sum()/unique_denominator) if unique_denominator else np.nan
        if {"waveform_sha256", "fingerprint_status", "exact_copy_group_size", "exact_copy_representative"} <= set(group):
            measured_identity = group.fingerprint_status.eq("measured")
            row["n_distinct_exact_measurements"] = int(group.loc[measured_identity, "waveform_sha256"].nunique())
            row["n_exact_copy_records"] = int(group.exact_copy_group_size.gt(1).sum())
            row["n_nonrepresentative_exact_copies"] = int((measured_identity & ~group.exact_copy_representative.eq(True)).sum())
            row["n_signal_identity_unavailable"] = int((~measured_identity).sum())
            row["exact_copy_scope"] = ("|".join(sorted(group.exact_copy_evidence_scope.dropna().astype(str).unique()))
                                       if "exact_copy_evidence_scope" in group else "selected_extraction_only_not_patient_identity")
        if {"death", "follow_up_years"} <= set(group):
            duration = pd.to_numeric(group.follow_up_years, errors="coerce")
            # Do not interpret a string such as 'False' as an observed death.
            valid_death = group.death.map(lambda value: isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)) and value in (0, 1))
            outcome_valid = valid_death & np.isfinite(duration) & duration.gt(0)
            row["n_identified_records_without_valid_outcome"] = int((identified & ~outcome_valid).sum())
            followed = group.loc[identified & outcome_valid]
            if followed.patient_id.duplicated().any():
                row["outcome_identity_status"] = "ambiguous_multiple_index_rows"
            else:
                row["n_outcome_bound_patients"] = int(followed.patient_id.nunique())
                row["n_observed_deaths"] = int(followed.death.astype(bool).sum())
                row["reference_scope"] = "annotation_plus_independent_longitudinal_outcome"
        if agreement is not None and not agreement.empty and "source_id" in agreement:
            scoped = agreement.loc[agreement.source_id.eq(cohort)]
            row["evaluable_annotation_hypotheses"] = int(scoped.get("status", pd.Series(dtype=str)).eq("evaluable").sum())
            if {"roc_auc_ci_low", "roc_auc_ci_high"} <= set(scoped):
                width = pd.to_numeric(scoped.roc_auc_ci_high, errors="coerce")-pd.to_numeric(scoped.roc_auc_ci_low, errors="coerce")
                row["median_patient_ci_width"] = float(width.dropna().median()) if width.notna().any() else np.nan
        if survival is not None and not survival.empty:
            scoped = survival.loc[survival.cohort.eq(cohort)]
            row["estimable_mortality_associations"] = int(scoped.status.eq("estimable").sum())
            row["ecg_age_incremental_comparisons"] = int(scoped.get("ecg_age_comparison_status", pd.Series(dtype=str)).eq("estimable").sum())
        rows.append(row)
    return pd.DataFrame(rows)
