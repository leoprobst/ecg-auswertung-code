"""Same-ECG-pair laboratory coverage, never imputed cross-pair physiology."""
import numpy as np
import pandas as pd

from .patient_identity import known_patient_mask

PAIR_KEYS = ("subject_id", "first_study_id", "second_study_id")
LAB_CONTEXT_ANALYTES = ("potassium", "calcium_total", "calcium_ionized", "magnesium", "ph")
PHYSIOLOGY_CORE = ("potassium", "calcium_ionized", "magnesium", "ph")


def joint_physiology_sensitivity(changes, context, marker_ids):
    """Conditional response model, not isolated/total causal electrolyte effect.

    Adjust baseline clinical/timing/QC plus other core laboratory changes.
    Do not substitute total calcium for ionized calcium or select controls
    based on observed significance/availability. Complete cases only.
    """
    from .paired_adjustment import BASELINE_CLINICAL_COVARIATES, _paired_sensitivity
    outputs = []
    if changes.empty or context.empty:
        return pd.DataFrame()
    for analyte in LAB_CONTEXT_ANALYTES:
        selected = changes.loc[changes.analyte_id.eq(analyte)]
        if selected.empty:
            continue
        controls = tuple(f"context_delta_{name}" for name in PHYSIOLOGY_CORE if name != analyte)
        joined = selected.merge(context[[*PAIR_KEYS, *controls]], on=list(PAIR_KEYS),
            how="left", validate="one_to_one", sort=False)
        # Guard target reference, even when called independently of the stage.
        joined["delta_lab"] = joined.delta_lab.where(joined.lab_pair_status.eq("eligible"))
        result = _paired_sensitivity(joined, marker_ids, exposure_column="delta_lab",
            covariates=(*BASELINE_CLINICAL_COVARIATES, *controls),
            model="joint_physiology_baseline_timing_quality_sensitivity")
        if result.empty:
            continue
        result = result.rename(columns={name: f"joint_{name}" for name in result if name != "feature_id"})
        result.insert(0, "design", f"dialysis_{analyte}")
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def joint_lab_context(changes):
    """Wide context plus coverage, without choosing a better-measured ECG.

    Input reference validity and accepted-unit policy come from paired_change_table.
    Total and ionized calcium remain distinct, not alternative imputations.
    """
    columns = [*PAIR_KEYS, *[f"context_delta_{name}" for name in LAB_CONTEXT_ANALYTES]]
    summary = dict(n_pairs=0, n_pairs_any_lab=0, n_pairs_complete_core=0,
        n_pairs_complete_all_five=0, analytes={}, coverage_patterns={},
        status="no_pairs", clinical_claim_allowed=False,
        interpretation="same_pair_retrospective_response_coverage_not_independent_causal_effects",
        core=list(PHYSIOLOGY_CORE), imputation="none_total_and_ionized_calcium_not_interchangeable")
    if changes is None or changes.empty:
        return pd.DataFrame(columns=columns), summary
    required = {*PAIR_KEYS, "analyte_id", "delta_lab", "lab_pair_status"}
    if not required <= set(changes):
        raise ValueError("Missing same-pair laboratory reference contract")
    selected = changes.loc[changes.analyte_id.isin(LAB_CONTEXT_ANALYTES), sorted(required)].copy()
    if selected.duplicated([*PAIR_KEYS, "analyte_id"]).any():
        raise ValueError("Ambiguous repeated analyte for same patient/ECG pair")
    if selected.empty:
        return pd.DataFrame(columns=columns), summary
    pair_rows = selected[list(PAIR_KEYS)].drop_duplicates()
    values = pd.to_numeric(selected.delta_lab, errors="coerce")
    valid = selected.lab_pair_status.eq("eligible").fillna(False) & np.isfinite(values)
    for name in PAIR_KEYS:
        valid &= known_patient_mask(selected[name])
    selected["delta_lab"] = values.where(valid)
    wide = selected.pivot(index=list(PAIR_KEYS), columns="analyte_id", values="delta_lab")
    wide = wide.reindex(columns=LAB_CONTEXT_ANALYTES).rename(columns=lambda name: f"context_delta_{name}")
    wide.columns.name = None
    wide = pair_rows.merge(wide.reset_index(), on=list(PAIR_KEYS), how="left", validate="one_to_one")
    available = wide[[f"context_delta_{name}" for name in LAB_CONTEXT_ANALYTES]].notna()
    core = available[[f"context_delta_{name}" for name in PHYSIOLOGY_CORE]].all(axis=1)
    summary.update(status="computed", n_pairs=len(wide), n_pairs_any_lab=int(available.any(axis=1).sum()),
        n_pairs_complete_core=int(core.sum()), n_pairs_complete_all_five=int(available.all(axis=1).sum()))
    summary["n_patients_complete_core"] = int(wide.loc[core, "subject_id"].nunique())
    for name in LAB_CONTEXT_ANALYTES:
        mask = available[f"context_delta_{name}"]
        summary["analytes"][name] = dict(n_pairs=int(mask.sum()), n_patients=int(wide.loc[mask, "subject_id"].nunique()))
    # Five analytes have only 32 possible patterns: no Python callback per pair.
    codes = available.to_numpy(dtype=np.int64) @ (1 << np.arange(len(LAB_CONTEXT_ANALYTES)))
    counts = np.bincount(codes, minlength=1 << len(LAB_CONTEXT_ANALYTES))
    summary["coverage_patterns"] = {
        "|".join(name for bit, name in enumerate(LAB_CONTEXT_ANALYTES) if code & (1 << bit)) or "none": int(count)
        for code, count in enumerate(counts) if count
    }
    return wide, summary
