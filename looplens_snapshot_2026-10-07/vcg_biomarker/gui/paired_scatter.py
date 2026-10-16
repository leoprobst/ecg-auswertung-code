"""Bounded paired display sample; never used as the model's analysis cohort."""
import numpy as np
import pandas as pd

from ..mimic.patient_identity import known_patient_mask


def prepare_paired_scatter(changes, marker, analyte, *, limit=6000):
    if type(limit) is not int or limit < 1:
        raise ValueError("Positive display limit required")
    x, y = "delta_lab", f"delta_{marker}"
    empty = pd.DataFrame(columns=[x, y])
    info = dict(status="no_data", n_total=0, n_valid=0, n_shown=0,
        n_known_patients=0, n_missing_patient_rows=0, unit="", legacy_reference=False,
        sampling="none")
    if changes is None or changes.empty or not {x, y} <= set(changes):
        return empty, info
    columns = [name for name in (x, y, "subject_id", "analyte_unit", "analyte_id", "lab_pair_status") if name in changes]
    # Select columns together with rows, never copy every marker column first.
    if analyte and "analyte_id" in changes:
        selected = changes.loc[changes.analyte_id.astype(str).eq(str(analyte)), columns]
    else:
        selected = changes.loc[:, columns]
    info["n_total"] = len(selected)
    units = selected.analyte_unit.dropna().astype(str).unique() if "analyte_unit" in selected else []
    if len(units) > 1 or ("analyte_id" in selected and selected.analyte_id.nunique() > 1):
        info["status"] = "ambiguous_analyte_or_unit"
        return empty, info
    info["unit"] = str(units[0]) if len(units) else ""
    values = selected[[x, y]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    if "lab_pair_status" in selected:
        valid &= selected.lab_pair_status.eq("eligible").fillna(False).to_numpy(dtype=bool)
    else:
        info["legacy_reference"] = True
    values = values.loc[valid]
    info["n_valid"] = len(values)
    if "subject_id" in selected:
        patients = selected.loc[valid, "subject_id"]
        known = known_patient_mask(patients)
        info["n_known_patients"] = int(patients[known].nunique())
        info["n_missing_patient_rows"] = int((~known).sum())
    else:
        info["n_missing_patient_rows"] = len(values)
    if len(values) > limit:
        indices = np.sort(np.random.default_rng(0).choice(len(values), size=limit, replace=False))
        values = values.iloc[indices]
        info["sampling"] = "uniform_pairs_fixed_seed_0_display_only"
    info.update(n_shown=len(values), status="ready" if len(values) else "no_eligible_finite_pairs")
    return values, info
