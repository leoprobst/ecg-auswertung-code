"""Development-only dialysis trajectories from standardized MIMIC relations."""
from __future__ import annotations

import uuid
import pandas as pd

from .recovery_episodes import episode_recovery_table
from .recovery_reference import attach_observed_normalization

RECOVERY_LOADER_VERSION = "1.2.0"
RECOVERY_LAB_GROUPS = ("potassium", "total_calcium", "ionized_calcium", "magnesium", "blood_gas_ph")


def dialysis_recovery(connection, *, master_relation: str, procedure_relation: str,
                       feature_relation: str, lab_relation: str, marker_ids: tuple[str, ...],
                       dialysis_itemids: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    """Materialize the narrow, selected-feature ECG cohort once per call.

    Relations are trusted SQL constructed by the stage, not user SQL. No raw
    patient-ID pooling, no locked ECG selection, no quality filtering before
    reference-blind trajectory selection. Episodes are 48-hour response windows
    truncated at the next dialysis start, not claims about treatment duration.
    """
    if not marker_ids or not dialysis_itemids:
        raise ValueError("Prespecified markers and dialysis item IDs required")
    def ident(value):
        return '"' + str(value).replace('"', '""') + '"'
    table = "recovery_" + uuid.uuid4().hex
    events_table = table + "_events"
    marker_sql = ", ".join("f." + ident(marker) for marker in marker_ids)
    items = ",".join(str(int(item)) for item in dialysis_itemids)
    groups_sql = ",".join("'" + group + "'" for group in RECOVERY_LAB_GROUPS)
    try:
        connection.execute(f"""CREATE TEMP TABLE {table} AS
            SELECT m.subject_id, m.hadm_id, m.study_id, m.ecg_time, m.split,
                   f.extraction_status, f.signal_quality_score, {marker_sql}
            FROM {master_relation} m JOIN {feature_relation} f
              ON m.subject_id=f.subject_id AND m.study_id=f.study_id
            WHERE m.split IN ('train','validation') AND m.hadm_id IS NOT NULL
        """)
        connection.execute(f"""CREATE TEMP TABLE {events_table} AS
            WITH starts AS (
                SELECT DISTINCT p.subject_id, p.hadm_id, p.starttime AS landmark_time
                FROM {procedure_relation} p
                WHERE p.itemid IN ({items}) AND p.starttime IS NOT NULL
                AND EXISTS (SELECT 1 FROM {table} m
                            WHERE m.subject_id=p.subject_id AND m.hadm_id=p.hadm_id)
            )
            SELECT *, CAST(landmark_time AS VARCHAR) AS episode_id,
                   least(landmark_time + INTERVAL 48 HOUR,
                         lead(landmark_time) OVER (PARTITION BY subject_id, hadm_id ORDER BY landmark_time)) AS episode_end
            FROM starts
        """)
        episodes = connection.execute(f"SELECT * FROM {events_table} ORDER BY subject_id, hadm_id, landmark_time").df()
        if episodes.empty:
            return pd.DataFrame(), {"status": "no_eligible_episodes", "n_episodes": 0,
                                    "version": RECOVERY_LOADER_VERSION}
        # Include admission in the key even if deidentified timestamps happen
        # to repeat in distinct admissions of one patient.
        episodes["episode_id"] = episodes.hadm_id.astype(str) + ":" + episodes.episode_id.astype(str)
        ecgs = connection.execute(f"""SELECT m.* FROM {table} m WHERE EXISTS (
            SELECT 1 FROM {events_table} e WHERE e.subject_id=m.subject_id AND e.hadm_id=m.hadm_id
            AND m.ecg_time >= e.landmark_time-INTERVAL 24 HOUR AND m.ecg_time < e.episode_end)
        """).df()
        labs = connection.execute(f"""SELECT l.subject_id, l.hadm_id, l.labevent_id,
                l.itemid, l.lab_group, l.charttime, l.storetime, l.value_numeric,
                l.unit, l.reference_lower, l.reference_upper, l.value_status
            FROM {lab_relation} l WHERE l.lab_group IN ({groups_sql}) AND EXISTS (
                SELECT 1 FROM {events_table} e WHERE e.subject_id=l.subject_id AND e.hadm_id=l.hadm_id
                AND l.charttime >= e.landmark_time-INTERVAL 6 HOUR AND l.charttime < e.episode_end)
        """).df()
        attached = attach_observed_normalization(episodes, labs, lab_group="potassium")
        additional = {}
        reference_columns = {}
        references = {"potassium": attached}
        for lab_group in RECOVERY_LAB_GROUPS[1:]:
            reference = attach_observed_normalization(episodes, labs, lab_group=lab_group)
            references[lab_group] = reference
            column = f"{lab_group}__normalization_time"
            additional[lab_group] = column
            # attach_observed_normalization preserves the input episode order.
            if not reference.episode_id.equals(attached.episode_id):
                raise ValueError("Normalization reference episode alignment changed")
            reference_columns[column] = reference.normalization_time
            for name in reference:
                if name.startswith("reference_"):
                    reference_columns[f"reference_{lab_group}_{name.removeprefix('reference_')}"] = reference[name]
        attached = pd.concat([attached, pd.DataFrame(reference_columns, index=attached.index)], axis=1)
        results = episode_recovery_table(ecgs, attached, marker_ids=marker_ids,
                                         additional_normalization_columns=additional)
        summary = dict(version=RECOVERY_LOADER_VERSION, status="computed",
            n_episodes=len(episodes), n_selected_ecgs=len(ecgs), n_labs=len(labs),
            n_results=len(results), n_evaluable=int(results.status.eq("evaluable").sum()),
            sampling_scope="development_ecgs_present_in_selected_feature_artifact",
            episode_definition="dialysis_start_to_next_start_or_48h_same_admission",
            interpretation="retrospective_response_not_isolated_causal_dialysis_effect",
            normalization_reference="potassium_same_assay_unit_reported_interval",
            normalization_references={name: {
                "n_labs": int(labs.lab_group.eq(name).sum()),
                "n_observed_normalizations": int(reference.reference_normalization_status.eq("observed").sum()),
                "n_episodes": len(reference),
                "observed_course_episode_counts": {
                    str(key): int(value) for key, value in reference.reference_observed_course.value_counts(dropna=False).items()
                },
                "observed_course_patient_counts": {
                    str(key): int(value) for key, value in reference.groupby("reference_observed_course").subject_id.nunique().items()
                },
                "course_counts_interpretation": "patients_may_occur_in_multiple_courses_no_independent_control_groups",
                "interpretation": "same_assay_unit_reported_interval_not_sustained_normality",
            } for name, reference in references.items()},
            control_comparison_status="not_computed", clinical_claim_allowed=False)
        return results, summary
    finally:
        connection.execute(f"DROP TABLE IF EXISTS {events_table}")
        connection.execute(f"DROP TABLE IF EXISTS {table}")
