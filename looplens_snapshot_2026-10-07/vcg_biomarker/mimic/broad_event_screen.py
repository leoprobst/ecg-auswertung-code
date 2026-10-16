"""Data-driven breadth: screen what the database has, not what someone listed.

Every event table in this project reaches the analysis through a hand-curated
subset, and the curation is far narrower than the data. Measured here:

| Achse | vorhanden | kuratiert | erfasste Zeilen |
|---|---:|---:|---:|
| Diagnosen (PheCodeX) | 28.583 Codes | 20.774 | 84,4 % |
| Labor | 976 Itemids | 56 | **32,9 %** |
| Prozeduren | 14.911 Codes | 6 Gruppen | nicht gemessen |
| Mikrobiologie | 718 Organismen | Flags | nicht gemessen |
| CCSR | 19.436 Codes gemappt | **nie gescreent** | 0 % |

Curation is the right tool when a marker needs a physiological definition. It
is the wrong tool for deciding *which of the database's own measurements to
look at* -- there the only defensible rule is frequency, applied uniformly and
declared in advance.

This module builds that: for each axis it enumerates every entity with enough
observations to support inference, emits it as a screening target, and reports
what remains below the floor. Two consequences are deliberate:

* **The multiplicity grows, and that is correct.** Screening 599 laboratory
  items instead of 56 is a larger family, and the global and hierarchical FDR
  the project already applies is what makes it honest. Hiding tests to keep
  q-values small is not a method.
* **A frequency floor is not a relevance judgement.** An entity below the
  floor is reported as *underpowered here*, never as uninformative.

CCSR is included because it was already built and summarised in every run and
then never screened. It is a second, coarser taxonomy over the same codes:
where PheCodeX splits a concept into several phenotypes each too small to
test, the CCSR category can still reach the inference floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


BROAD_EVENT_SCREEN_VERSION = "1.1.0"

AXIS_CCSR = "ccsr_category"
AXIS_LABORATORY = "laboratory_item"
AXIS_PROCEDURE = "procedure_code"
AXIS_MICROBIOLOGY = "microbiology_organism"
AXIS_ICU_MEASUREMENT = "icu_measurement"
AXIS_DRG = "drg_case_mix"

BROAD_AXES: tuple[str, ...] = (
    AXIS_CCSR,
    AXIS_LABORATORY,
    AXIS_PROCEDURE,
    AXIS_MICROBIOLOGY,
    AXIS_ICU_MEASUREMENT,
    AXIS_DRG,
)


@dataclass(frozen=True)
class BroadAxisPolicy:
    """Prespecified inclusion rule for one axis.

    ``minimum_subjects`` is the only inclusion criterion. It is fixed before
    the screen so that the family size cannot be tuned after seeing results,
    which is the mechanism by which a "broad" screen quietly becomes a
    selected one.
    """

    axis: str
    minimum_subjects: int
    minimum_observations: int = 0
    target_kind: str = "binary"
    temporal_role: str = "retrospective_index_admission_context"
    claim_ceiling: str = "association_only_not_accuracy"
    reference_tier: str = "C_single_administrative_or_text"
    notes_de: str = ""

    def __post_init__(self) -> None:
        if self.axis not in BROAD_AXES:
            raise ValueError(f"Unknown broad axis: {self.axis}")
        if self.minimum_subjects < 30:
            raise ValueError(
                "A frequency floor below 30 subjects per class cannot support "
                "formal inference in this pipeline"
            )
        if self.target_kind not in ("binary", "continuous"):
            raise ValueError("target_kind must be binary or continuous")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "minimum_subjects": self.minimum_subjects,
            "minimum_observations": self.minimum_observations,
            "target_kind": self.target_kind,
            "temporal_role": self.temporal_role,
            "claim_ceiling": self.claim_ceiling,
            "reference_tier": self.reference_tier,
            "notes_de": self.notes_de,
            "broad_event_screen_version": BROAD_EVENT_SCREEN_VERSION,
        }


DEFAULT_POLICIES: tuple[BroadAxisPolicy, ...] = (
    BroadAxisPolicy(
        AXIS_DRG,
        minimum_subjects=100,
        target_kind="binary",
        # A DRG is assigned when the admission is billed, so it is always an
        # end-of-stay label. No association with it can be diagnostic.
        temporal_role="assigned_at_discharge_end_of_stay",
        notes_de=(
            "DRG-Fallgruppen: die abrechnungsseitige Zusammenfassung eines "
            "Aufenthalts in eine Kategorie mit Schweregrad. Eine dritte, "
            "gaenzlich andere Taxonomie neben PheCodeX und CCSR -- sie "
            "gruppiert nach Behandlungsaufwand statt nach Diagnose und faengt "
            "damit Konstellationen, die eine reine Diagnoseachse trennt."
        ),
    ),
    BroadAxisPolicy(
        AXIS_ICU_MEASUREMENT,
        minimum_subjects=200,
        minimum_observations=1000,
        target_kind="continuous",
        temporal_role="concurrent_invasive_or_bedside_measurement",
        # An arterial or pulmonary-artery pressure is a measured number, not
        # an administrative code, so the ceiling is the same as for laboratory
        # values rather than the one for diagnoses.
        claim_ceiling="association_and_analytical_agreement",
        reference_tier="A_instrumented_measurement",
        notes_de=(
            "Invasive und bettseitige ICU-Messwerte. Wissenschaftlich die "
            "wertvollste Achse: hier stehen die invasiv gemessenen Groessen "
            "(Pulmonalarteriendruck, ZVD, Herzzeitvolumen, SvO2), fuer die ein "
            "VCG-Marker ein nichtinvasiver Ersatz sein koennte. Das ist eine "
            "Surrogatfrage, keine blosse Assoziation."
        ),
    ),

    BroadAxisPolicy(
        AXIS_CCSR,
        minimum_subjects=100,
        target_kind="binary",
        notes_de=(
            "Zweite, groebere Diagnosetaxonomie ueber dieselben Codes. Faengt "
            "Konzepte, die PheCodeX in mehrere je zu kleine Phaenotypen "
            "aufteilt."
        ),
    ),
    BroadAxisPolicy(
        AXIS_LABORATORY,
        minimum_subjects=200,
        minimum_observations=1000,
        target_kind="continuous",
        temporal_role="concurrent_clinical_measurement",
        reference_tier="A_quantitative_or_adjudicated",
        claim_ceiling="diagnostic_accuracy_with_reference_uncertainty",
        notes_de=(
            "Alle Laboritems mit ausreichender Messzahl, nicht nur die 56 "
            "kuratierten. Numerische Werte im selben Zeitfenster wie die "
            "kuratierten Gruppen."
        ),
    ),
    BroadAxisPolicy(
        AXIS_PROCEDURE,
        minimum_subjects=100,
        target_kind="binary",
        notes_de=(
            "Prozedurcodes des Indexaufenthalts. Eine Prozedur ist eine "
            "Behandlungsentscheidung, kein Krankheitsnachweis; sie folgt dem "
            "EKG oft nach und ist entsprechend zu lesen."
        ),
    ),
    BroadAxisPolicy(
        AXIS_MICROBIOLOGY,
        minimum_subjects=100,
        target_kind="binary",
        notes_de=(
            "Nachgewiesene Organismen statt eines binaeren Infektionsflags. "
            "Sepsis-Erreger unterscheiden sich in Toxinwirkung und "
            "Kreislaufverhalten."
        ),
    ),
)


def policy_for(axis: str, policies: Iterable[BroadAxisPolicy] = DEFAULT_POLICIES) -> BroadAxisPolicy:
    for policy in policies:
        if policy.axis == axis:
            return policy
    raise KeyError(f"No policy registered for axis {axis!r}")


# --- target enumeration -----------------------------------------------------


def enumerate_targets(
    counts: pd.DataFrame,
    policy: BroadAxisPolicy,
    *,
    entity_column: str,
    subject_column: str = "n_subjects",
    observation_column: str = "n_observations",
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """Turn a per-entity count table into screening targets.

    Entities below the floor are kept in the output with
    ``eligible_for_screen=False`` and a reason, so the report can state how
    much of the axis was left untested instead of showing only what passed.
    """

    if counts.empty:
        return pd.DataFrame(
            columns=[
                "axis",
                "entity_id",
                "entity_label",
                "n_subjects",
                "n_observations",
                "eligible_for_screen",
                "ineligibility_reason",
            ]
        )
    output = pd.DataFrame()
    output["axis"] = [policy.axis] * len(counts)
    output["entity_id"] = counts[entity_column].astype(str).to_numpy()
    output["entity_label"] = (
        counts[label_column].astype(str).to_numpy()
        if label_column and label_column in counts.columns
        else output["entity_id"].to_numpy()
    )
    subjects = pd.to_numeric(counts[subject_column], errors="coerce").fillna(0)
    observations = (
        pd.to_numeric(counts[observation_column], errors="coerce").fillna(0)
        if observation_column in counts.columns
        else subjects
    )
    output["n_subjects"] = subjects.astype(int).to_numpy()
    output["n_observations"] = observations.astype(int).to_numpy()
    enough_subjects = subjects >= policy.minimum_subjects
    enough_observations = observations >= policy.minimum_observations
    output["eligible_for_screen"] = (
        (enough_subjects & enough_observations).to_numpy()
    )
    output["ineligibility_reason"] = np.select(
        [
            (~enough_subjects).to_numpy(),
            (~enough_observations).to_numpy(),
        ],
        [
            "below_subject_floor_underpowered_here_not_uninformative",
            "below_observation_floor_underpowered_here_not_uninformative",
        ],
        default="",
    )
    output["target_kind"] = policy.target_kind
    output["temporal_role"] = policy.temporal_role
    output["reference_tier"] = policy.reference_tier
    output["claim_ceiling"] = policy.claim_ceiling
    output["minimum_subjects"] = policy.minimum_subjects
    output["broad_event_screen_version"] = BROAD_EVENT_SCREEN_VERSION
    return output.sort_values(
        ["eligible_for_screen", "n_subjects"],
        ascending=[False, False],
        kind="mergesort",
        ignore_index=True,
    )


def ccsr_target_counts_sql(ccsr_relation: str, diagnoses_relation: str) -> str:
    """Subjects per CCSR category."""

    return f"""
        SELECT
            c.ccsr_category AS entity_id,
            ANY_VALUE(c.ccsr_description) AS entity_label,
            COUNT(DISTINCT d.subject_id) AS n_subjects,
            COUNT(*) AS n_observations
        FROM {diagnoses_relation} d
        JOIN {ccsr_relation} c ON c.icd_code = d.icd_code
        WHERE c.ccsr_category IS NOT NULL
        GROUP BY c.ccsr_category
    """


def laboratory_target_counts_sql(
    labevents_relation: str,
    dictionary_relation: str,
) -> str:
    """Subjects and numeric measurements per laboratory item."""

    return f"""
        SELECT
            l.itemid::VARCHAR AS entity_id,
            ANY_VALUE(COALESCE(d.label, l.itemid::VARCHAR)) AS entity_label,
            COUNT(DISTINCT l.subject_id) AS n_subjects,
            COUNT(l.valuenum) AS n_observations
        FROM {labevents_relation} l
        LEFT JOIN {dictionary_relation} d USING (itemid)
        GROUP BY l.itemid
    """


def procedure_target_counts_sql(
    procedures_relation: str,
    dictionary_relation: Optional[str] = None,
) -> str:
    """Subjects per procedure code, with the human-readable title.

    Without the dictionary a screening hit reads as "0270", which no clinician
    can adjudicate. Every other axis carries a label; this one used to be the
    exception.
    """

    if dictionary_relation is None:
        return f"""
            SELECT
                icd_code AS entity_id,
                icd_code AS entity_label,
                COUNT(DISTINCT subject_id) AS n_subjects,
                COUNT(*) AS n_observations
            FROM {procedures_relation}
            WHERE icd_code IS NOT NULL
            GROUP BY icd_code
        """
    return f"""
        SELECT
            p.icd_code AS entity_id,
            ANY_VALUE(COALESCE(d.long_title, p.icd_code)) AS entity_label,
            COUNT(DISTINCT p.subject_id) AS n_subjects,
            COUNT(*) AS n_observations
        FROM {procedures_relation} p
        LEFT JOIN {dictionary_relation} d
          ON d.icd_code = p.icd_code
         AND d.icd_version = p.icd_version
        WHERE p.icd_code IS NOT NULL
        GROUP BY p.icd_code
    """


def drg_target_counts_sql(drg_relation: str) -> str:
    """Subjects per DRG case-mix group, with its description."""

    return f"""
        SELECT
            drg_code::VARCHAR AS entity_id,
            ANY_VALUE(COALESCE(description, drg_code::VARCHAR)) AS entity_label,
            COUNT(DISTINCT subject_id) AS n_subjects,
            COUNT(*) AS n_observations
        FROM {drg_relation}
        WHERE drg_code IS NOT NULL
        GROUP BY drg_code
    """


def microbiology_target_counts_sql(microbiology_relation: str) -> str:
    return f"""
        SELECT
            organism AS entity_id,
            organism AS entity_label,
            COUNT(DISTINCT subject_id) AS n_subjects,
            COUNT(*) AS n_observations
        FROM {microbiology_relation}
        WHERE organism IS NOT NULL AND TRIM(organism) <> ''
        GROUP BY organism
    """


def icu_measurement_target_counts_sql(
    chartevents_relation: str,
    dictionary_relation: str,
) -> str:
    """Subjects and numeric measurements per ICU chart item."""

    return f"""
        SELECT
            c.itemid::VARCHAR AS entity_id,
            ANY_VALUE(COALESCE(d.label, c.itemid::VARCHAR)) AS entity_label,
            COUNT(DISTINCT c.subject_id) AS n_subjects,
            COUNT(c.valuenum) AS n_observations
        FROM {chartevents_relation} c
        LEFT JOIN {dictionary_relation} d USING (itemid)
        WHERE c.valuenum IS NOT NULL
        GROUP BY c.itemid
    """


# --- long-format value extraction -------------------------------------------
#
# A wide pivot is the wrong shape here. Six columns for each of ~1,500 targets
# would be a 9,000-column evidence table, almost entirely null, rebuilt every
# time a target clears or falls below the floor. The screen needs one value per
# ECG and target, so the artifact is tall and sparse instead: only measurements
# that exist occupy a row.


def continuous_axis_values_sql(
    event_relation: str,
    master_relation: str,
    *,
    axis: str,
    entity_ids: Sequence[str],
    time_column: str = "charttime",
    window_hours: float = 6.0,
) -> str:
    """Nearest numeric value per ECG and target, inside a symmetric window.

    Nearest rather than mean, because a physiological state near the ECG is
    the quantity of interest; an admission-wide average is a different and
    much noisier construct.
    """

    if not entity_ids:
        raise ValueError("At least one target is required")
    id_list = ", ".join(str(int(entity)) for entity in entity_ids)
    # An integer minute literal: DuckDB rejects a fractional hour literal
    # ("INTERVAL 6.0 HOUR" is a parser error), and rounding to whole hours
    # would silently coarsen a window the caller asked for in fractions.
    window_minutes = int(round(float(window_hours) * 60.0))
    return f"""
        WITH scoped AS (
            SELECT
                m.subject_id,
                m.study_id,
                m.split,
                e.itemid::VARCHAR AS target_id,
                e.valuenum AS value,
                DATE_DIFF('second', m.ecg_time, e.{time_column}) / 3600.0
                    AS delta_hours
            FROM {master_relation} m
            JOIN {event_relation} e
              ON e.subject_id = m.subject_id
             AND e.{time_column}
                 BETWEEN m.ecg_time - INTERVAL {window_minutes} MINUTE
                     AND m.ecg_time + INTERVAL {window_minutes} MINUTE
            WHERE e.valuenum IS NOT NULL
              AND e.itemid IN ({id_list})
        ), nearest AS (
            SELECT
                subject_id,
                study_id,
                split,
                '{axis}' AS axis,
                target_id,
                ARG_MIN(value, ABS(delta_hours)) AS value,
                ARG_MIN(delta_hours, ABS(delta_hours)) AS delta_hours,
                COUNT(*) AS n_measurements
            FROM scoped
            GROUP BY subject_id, study_id, split, target_id
        )
        SELECT
            *,
            CASE
                WHEN delta_hours < 0 THEN '{RELATION_BEFORE}'
                WHEN delta_hours > 0 THEN '{RELATION_AFTER}'
                ELSE '{RELATION_SAME_TIME}'
            END AS temporal_relation
        FROM nearest
    """


# How a dated administrative entity sits relative to the ECG. The three are
# different findings and must never be pooled into one number:
RELATION_BEFORE = "before_ecg"
RELATION_AFTER = "after_ecg"
RELATION_SAME_DAY = "same_day_as_ecg"
RELATION_SAME_TIME = "same_time_as_ecg"
RELATION_UNDATED = "undated_admission_context"


def binary_axis_values_sql(
    event_relation: str,
    master_relation: str,
    *,
    axis: str,
    entity_column: str,
    entity_ids: Sequence[str],
    date_column: Optional[str] = None,
) -> str:
    """Presence of an administrative entity for the ECG's own admission.

    Scoped to the admission rather than to the patient's lifetime: a procedure
    coded three years later says nothing about the ECG in front of you, and
    treating it as if it did is how a screen invents associations.

    Scoping to the admission is not enough on its own, though. Measured on the
    real data, 38.9 % of ECG/procedure pairs inside one admission have the
    procedure *after* the ECG and 43.0 % before it. Those are opposite
    findings -- "this ECG anticipates an intervention that was still to come"
    versus "this ECG shows a heart that had already been operated on" -- and
    pooling them produces a number that means neither. When a date is
    available the relation is emitted alongside the presence so the two are
    screened, corrected and reported separately.
    """

    if not entity_ids:
        raise ValueError("At least one target is required")
    quoted = ", ".join(
        "'" + str(entity).replace("'", "''") + "'" for entity in entity_ids
    )
    if date_column is None:
        relation_sql = f"'{RELATION_UNDATED}'"
    else:
        relation_sql = f"""
            CASE
                WHEN CAST(e.{date_column} AS DATE)
                     > CAST(m.ecg_time AS DATE) THEN '{RELATION_AFTER}'
                WHEN CAST(e.{date_column} AS DATE)
                     < CAST(m.ecg_time AS DATE) THEN '{RELATION_BEFORE}'
                ELSE '{RELATION_SAME_DAY}'
            END"""
    return f"""
        SELECT DISTINCT
            m.subject_id,
            m.study_id,
            m.split,
            '{axis}' AS axis,
            e.{entity_column}::VARCHAR AS target_id,
            {relation_sql} AS temporal_relation,
            1.0 AS value,
            NULL::DOUBLE AS delta_hours,
            1 AS n_measurements
        FROM {master_relation} m
        JOIN {event_relation} e
          ON e.subject_id = m.subject_id
         AND e.hadm_id = m.hadm_id
        WHERE e.{entity_column} IS NOT NULL
          AND e.{entity_column}::VARCHAR IN ({quoted})
    """


# --- coverage accounting ----------------------------------------------------


def axis_coverage(
    targets: pd.DataFrame,
    *,
    curated_entity_ids: Iterable[str] = (),
) -> Mapping[str, object]:
    """How much of an axis the broad screen adds over the curated subset."""

    if targets.empty:
        return {
            "axis": "",
            "entities_total": 0,
            "entities_screened": 0,
            "broad_event_screen_version": BROAD_EVENT_SCREEN_VERSION,
        }
    curated = {str(item) for item in curated_entity_ids}
    eligible = targets.loc[targets["eligible_for_screen"].astype(bool)]
    added = eligible.loc[~eligible["entity_id"].astype(str).isin(curated)]
    total_observations = float(targets["n_observations"].sum())
    screened_observations = float(eligible["n_observations"].sum())
    return {
        "axis": str(targets["axis"].iloc[0]),
        "entities_total": int(len(targets)),
        "entities_screened": int(len(eligible)),
        "entities_below_floor": int(len(targets) - len(eligible)),
        "entities_curated_before": len(curated),
        "entities_added_by_broad_screen": int(len(added)),
        "observation_coverage": (
            screened_observations / max(total_observations, 1.0)
        ),
        "minimum_subjects": int(targets["minimum_subjects"].iloc[0]),
        "interpretation_de": (
            "Entitaeten unterhalb der Schwelle sind hier unterpowert, nicht "
            "uninformativ. Sie zaehlen nicht als geprueft."
        ),
        "broad_event_screen_version": BROAD_EVENT_SCREEN_VERSION,
    }


def screen_family_size(targets: pd.DataFrame, n_markers: int) -> Mapping[str, object]:
    """Family size the multiplicity control has to absorb.

    Reported explicitly because widening a screen without saying by how much
    is how a discovery family quietly stops being controllable.
    """

    eligible = int(targets["eligible_for_screen"].astype(bool).sum())
    return {
        "axis": str(targets["axis"].iloc[0]) if len(targets) else "",
        "eligible_targets": eligible,
        "markers": int(n_markers),
        "tests_in_family": eligible * int(n_markers),
        "multiplicity_note_de": (
            "Die Familie waechst mit der Breite. Das ist beabsichtigt: die "
            "globale und hierarchische FDR-Kontrolle des Projekts traegt sie, "
            "und Tests zu verschweigen, um q-Werte klein zu halten, waere "
            "keine Methode."
        ),
        "broad_event_screen_version": BROAD_EVENT_SCREEN_VERSION,
    }


def combined_coverage_report(
    coverages: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """One row per axis, for the report layer."""

    rows = [dict(item) for item in coverages if item.get("axis")]
    if not rows:
        return pd.DataFrame(
            columns=[
                "axis",
                "entities_total",
                "entities_screened",
                "entities_added_by_broad_screen",
                "observation_coverage",
            ]
        )
    return pd.DataFrame(rows)


def assert_policies_prespecified(
    policies: Iterable[BroadAxisPolicy],
    *,
    expected: Mapping[str, int],
) -> None:
    """Fail when a frequency floor differs from its preregistered value.

    The floor decides the family size, so moving it after seeing results is a
    researcher degree of freedom with a direct effect on every q-value in the
    screen.
    """

    actual = {policy.axis: policy.minimum_subjects for policy in policies}
    mismatches = {
        axis: (value, actual.get(axis))
        for axis, value in expected.items()
        if actual.get(axis) != value
    }
    if mismatches:
        raise AssertionError(
            "Broad-screen frequency floors differ from the preregistered "
            f"values: {mismatches}. The floor determines the family size and "
            "therefore every q-value in the screen."
        )


__all__ = [
    "AXIS_CCSR",
    "AXIS_LABORATORY",
    "AXIS_MICROBIOLOGY",
    "AXIS_PROCEDURE",
    "BROAD_AXES",
    "BROAD_EVENT_SCREEN_VERSION",
    "BroadAxisPolicy",
    "DEFAULT_POLICIES",
    "assert_policies_prespecified",
    "axis_coverage",
    "AXIS_DRG",
    "AXIS_ICU_MEASUREMENT",
    "drg_target_counts_sql",
    "RELATION_AFTER",
    "RELATION_BEFORE",
    "RELATION_SAME_DAY",
    "RELATION_SAME_TIME",
    "RELATION_UNDATED",
    "binary_axis_values_sql",
    "ccsr_target_counts_sql",
    "continuous_axis_values_sql",
    "icu_measurement_target_counts_sql",
    "combined_coverage_report",
    "enumerate_targets",
    "laboratory_target_counts_sql",
    "microbiology_target_counts_sql",
    "policy_for",
    "procedure_target_counts_sql",
    "screen_family_size",
]
