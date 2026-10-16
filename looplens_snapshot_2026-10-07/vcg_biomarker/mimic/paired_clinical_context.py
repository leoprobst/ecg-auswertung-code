"""Reference-blind baseline context; no post-ECG or fallback enrichment."""
import numpy as np
import pandas as pd


def attach_baseline_context(connection, pairs, master_path):
    """Join only the first selected ECG, by patient AND study, on development.

    No complete adjustment is implied. Admission/ED linkage is not a diagnosis
    or proof that an unlinked ECG was outpatient. Missing columns stay missing.
    """
    output = pairs.copy()
    columns = ("age", "sex", "machine_heart_rate_bpm", "hadm_id", "ed_stay_id")
    targets = [f"baseline_{name}" for name in columns]
    if any(name in output for name in (*targets, "baseline_context_status")):
        raise ValueError("Baseline context already present; refusing overwrite")
    if output.empty:
        for name in targets:
            output[name] = pd.Series(index=output.index, dtype=object)
        for name in ("age", "heart_rate", "sex_male"):
            output[f"baseline_adjust_{name}"] = pd.Series(index=output.index, dtype=float)
        output["baseline_context_status"] = pd.Series(index=output.index, dtype=str)
        return output
    keys = output[["subject_id", "first_study_id"]].drop_duplicates()
    path = str(master_path).replace("'", "''")
    relation = f"read_parquet('{path}')"
    available = set(connection.execute(f"DESCRIBE SELECT * FROM {relation}").df()["column_name"])
    if not {"subject_id", "study_id", "split"} <= available:
        raise ValueError("Master lacks patient/study/split identity for baseline context")
    select = ", ".join(f'm."{name}"' if name in available else f'NULL AS "{name}"' for name in columns)
    # Per-connection temporary registration is removed even if validation fails.
    connection.register("paired_baseline_keys", keys)
    try:
        context = connection.execute(f"""
            SELECT m.subject_id, m.study_id AS first_study_id, {select}
            FROM {relation} m SEMI JOIN paired_baseline_keys k
              ON m.subject_id=k.subject_id AND m.study_id=k.first_study_id
            WHERE m.split IN ('train', 'validation')
        """).df()
    finally:
        connection.unregister("paired_baseline_keys")
    if context.duplicated(["subject_id", "first_study_id"]).any():
        raise ValueError("Ambiguous baseline master identity; no arbitrary row selection")
    context = context.rename(columns={name: f"baseline_{name}" for name in columns})
    context["baseline_context_status"] = "matched_development_index"
    index = output.index.copy()
    output = output.merge(context, on=["subject_id", "first_study_id"], how="left", validate="many_to_one", sort=False)
    output.index = index
    output["baseline_context_status"] = output.baseline_context_status.fillna("missing_development_index")
    # Retain source values, but provide explicitly numeric model candidates.
    for source, target in (("age", "age"), ("machine_heart_rate_bpm", "heart_rate")):
        values = pd.to_numeric(output[f"baseline_{source}"], errors="coerce")
        output[f"baseline_adjust_{target}"] = values.where(np.isfinite(values) & values.gt(0))
    output["baseline_adjust_sex_male"] = output.baseline_sex.astype("string").str.upper().map({"M": 1., "F": 0.})
    return output
