"""Production adapter for development-only, serial temporal morphology responses."""
import json

import numpy as np
import pandas as pd

from ..core.temporal_research import VERSION, abstain, serial_arc_clock


def _positive_count(value):
    """Reject missing/nonfinite/fractional counts; never serialize them as NaN."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if np.isfinite(numeric) and numeric > 0 and numeric.is_integer() else None


def temporal_response_table(pairs):
    rows = []
    for row in pairs.to_dict("records"):
        result = abstain("temporal_descriptor_unavailable")
        if row.get("split_before") not in ("train", "validation") or row.get("split_after") != row.get("split_before"):
            result = abstain("nondevelopment_or_cross_split_pair")
        else:
            try:
                before = json.loads(row.get("temporal_before") or "null")
                after = json.loads(row.get("temporal_after") or "null")
                hours = [float(row["hours_before_to_exposure"]), float(row["hours_exposure_to_after"])]
                if not np.isfinite(hours).all() or min(hours) <= 0 or max(hours) > 2:
                    result = abstain("ambiguous_or_outside_prespecified_exposure_window")
                elif isinstance(before, dict) and isinstance(after, dict):
                    if str(before.get("subject_id")) != str(row.get("subject_id")):
                        result = abstain("pair_subject_mismatch")
                    else:
                        result = serial_arc_clock(before, after)
            except (TypeError, ValueError, KeyError):
                result = abstain("invalid_temporal_descriptor_or_timing")
        rows.append({**{key: row.get(key) for key in ("subject_id", "study_id_before", "study_id_after", "medication_class")},
                     "status": result["status"], "reason": result.get("reason", ""),
                     "absolute_arc_clock_distance": result.get("absolute_distance"),
                     "qrs_duration_delta_ms": result.get("qrs_duration_delta_ms"),
                     "heart_rate_delta_bpm": row.get("heart_rate_delta_bpm"),
                     "details_json": json.dumps(result, allow_nan=False),
                     "version": VERSION, "clinical_claim_allowed": False})
    return pd.DataFrame(rows, columns=["subject_id", "study_id_before", "study_id_after", "medication_class",
        "status", "reason", "absolute_arc_clock_distance", "qrs_duration_delta_ms", "heart_rate_delta_bpm",
        "details_json", "version", "clinical_claim_allowed"])


def dialysis_temporal_response(connection, feature_relation, pairs, *, descriptor_available):
    """Reuse already selected dialysis pairs; never select a better-referenced ECG.

    One projected read restricted to selected development keys. Dialysis start
    between ECGs is not proof that the second ECG follows treatment completion.
    """
    columns = ["subject_id", "first_study_id", "second_study_id", "status", "reason",
               "absolute_arc_clock_distance", "details_json", "clinical_claim_allowed"]
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    lookup = {}
    if descriptor_available:
        keys = pd.concat([
            pairs[["subject_id", "split", name]].rename(columns={name: "study_id"})
            for name in ("first_study_id", "second_study_id")
        ], ignore_index=True).drop_duplicates()
        keys = keys.loc[keys["split"].isin(["train", "validation"])]
        connection.register("temporal_selected_keys", keys)
        try:
            features = connection.execute(f"""
                SELECT f.subject_id, f.study_id, f.split, f.context_temporal_research_json
                FROM {feature_relation} f JOIN temporal_selected_keys k
                  ON f.subject_id=k.subject_id AND f.study_id=k.study_id AND f.split=k.split
                WHERE f.extraction_status='ok' AND f.split IN ('train','validation')
            """).df()
        finally:
            connection.unregister("temporal_selected_keys")
        if features.duplicated(["subject_id", "study_id", "split"]).any():
            raise ValueError("duplicate_temporal_feature_identity")
        lookup = {(row.subject_id, row.study_id, row.split): row.context_temporal_research_json
                  for row in features.itertuples(index=False)}
    rows = []
    for row in pairs.to_dict("records"):
        result = abstain("temporal_descriptor_unavailable")
        procedure_count = _positive_count(row.get("n_procedure_rows", row.get("n_sessions")))
        try:
            if row.get("split") not in ("train", "validation"):
                result = abstain("nondevelopment_pair")
            elif (not np.isfinite(row["hours_between"]) or not 1 <= row["hours_between"] <= 48
                  or procedure_count is None):
                result = abstain("invalid_selected_dialysis_window")
            else:
                before, after = [json.loads(lookup.get((row["subject_id"], row[name], row["split"]), "null"))
                                 for name in ("first_study_id", "second_study_id")]
                if isinstance(before, dict) and isinstance(after, dict):
                    if str(before.get("subject_id")) != str(row["subject_id"]):
                        result = abstain("pair_subject_mismatch")
                    else:
                        result = serial_arc_clock(before, after)
        except (ValueError, TypeError, KeyError):
            result = abstain("invalid_temporal_descriptor_or_timing")
        result["interpretation"] = "selected_pair_with_intervening_dialysis_start_not_proven_post_completion_or_causal_effect"
        result["intervention_context"] = {
            "count_semantics": "procedure_rows_not_independent_sessions",
            "n_procedure_rows": procedure_count,
            "n_procedure_types": _positive_count(row.get("n_procedure_types")),
            "procedure_itemids": (row["procedure_itemids"]
                                  if isinstance(row.get("procedure_itemids"), str) else None),
            "session_identity_verified": False,
            "treatment_completion_verified": False,
        }
        rows.append({**{key: row.get(key) for key in columns[:3]}, "status": result["status"],
                     "reason": result.get("reason", ""), "absolute_arc_clock_distance": result.get("absolute_distance"),
                     "details_json": json.dumps(result, allow_nan=False), "clinical_claim_allowed": False})
    return pd.DataFrame(rows, columns=columns)
