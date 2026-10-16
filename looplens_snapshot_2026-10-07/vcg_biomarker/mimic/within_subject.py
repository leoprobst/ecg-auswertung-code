"""Does the marker move when the patient's state moves -- in the same patient?

The cross-sectional screen asks whether people with a condition have different
markers from people without. In an intensive-care population that question is
answered mostly by confounding: measured on 2026-08-30, a pair with no
mechanism at all (digoxin exposure against P-wave duration) reached AUC 0.616,
larger than every mechanism-backed control in the same run. Anatomy, sex,
chronic disease and illness severity all sit between the marker and the label.

Comparing a patient with themselves controls stable patient attributes, but
does not remove changing illness severity, concurrent therapies, acquisition
changes, informative ECG timing or regression to the mean. A within-person
association is not by itself a causal intervention effect.

**Why the potassium question needs exactly this.** The 2026-08-28 run found
essentially nothing for severe hyperkalaemia. Part of that was the wrong marker
pairing, but part is structural: T-wave morphology varies far more between
people than potassium moves it within one. A cross-sectional comparison drowns
the effect in that between-person spread. Dialysis provides the natural
experiment -- a large, fast, measured potassium shift in a single patient, with
ECGs on both sides of it.

**This is the mirror image of the repeatability pairing.**
``trajectory.build_replicate_pairs`` looks for pairs with *no* intervening
event, because it wants measurement error. Here the intervening event is the
whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .storage import DuckDBCatalog, feature_parquet_clause
from .panels import STUDY_PANEL_IDS
from .clustered_rank_association import ClusteredRankBatch
from .paired_adjustment import paired_sensitivities, add_all_paired_adjustment_fdr
from .paired_clinical_context import attach_baseline_context
from .paired_lab_context import joint_lab_context, joint_physiology_sensitivity


WITHIN_SUBJECT_VERSION = "1.32.0"


@dataclass(frozen=True)
class InterventionWindow:
    """An event that is expected to move the state, and the pairing around it."""

    intervention_id: str
    display_name: str
    procedure_itemids: tuple[int, ...] = ()
    minimum_hours_between: float = 1.0
    maximum_hours_between: float = 48.0
    require_same_admission: bool = True
    minimum_signal_quality: float = 0.5

    def to_dict(self) -> dict:
        return {
            "intervention_id": self.intervention_id,
            "display_name": self.display_name,
            "procedure_itemids": list(self.procedure_itemids),
            "minimum_hours_between": float(self.minimum_hours_between),
            "maximum_hours_between": float(self.maximum_hours_between),
            "require_same_admission": bool(self.require_same_admission),
            "minimum_signal_quality": float(self.minimum_signal_quality),
            "version": WITHIN_SUBJECT_VERSION,
        }


# The dialysis *treatments*, taken from d_items. Catheter insertion (224270)
# and filter changes (225436) are deliberately excluded: they are procedures
# on the equipment, not a session that shifts electrolytes.
DIALYSIS_ITEMIDS: tuple[int, ...] = (
    225441,  # Hemodialysis
    225802,  # Dialysis - CRRT
    225803,  # Dialysis - CVVHD
    225809,  # Dialysis - CVVHDF
    225955,  # Dialysis - SCUF
    225805,  # Peritoneal Dialysis
)

DIALYSIS_WINDOW = InterventionWindow(
    intervention_id="dialysis",
    display_name="Dialyse",
    procedure_itemids=DIALYSIS_ITEMIDS,
    # Prespecified ECG-pair window, not treatment duration. This item set also
    # includes continuous therapies; a start between ECGs does not establish
    # completion or a single independent session.
    minimum_hours_between=1.0,
    maximum_hours_between=48.0,
)

# Serum and whole-blood potassium only. Urine, ascites, pleural, stool, CSF
# and joint fluid all carry the word and none of them is what the myocardium
# sees.
SERUM_POTASSIUM_ITEMIDS: tuple[int, ...] = (50822, 50971, 52452, 52610)

# Plausibility bounds for the paired lab, applied before anything is fitted.
# Same principle as the gap-marker target bounds: a unit error at one end of a
# difference destroys the whole distribution, and the count of exclusions is
# what tells a reader how much was dropped.
POTASSIUM_BOUNDS = (1.5, 10.0)


@dataclass(frozen=True)
class LabAnalyte:
    """One non-interchangeable laboratory quantity in the paired design."""

    analyte_id: str
    display_name: str
    itemids: tuple[int, ...]
    bounds: tuple[float, float]
    unit: str
    accepted_units: tuple[str, ...] = ()


def _analyte_units(analyte: LabAnalyte) -> tuple[str, ...]:
    """Use the evidence pipeline's unit policy; never infer a scale from values."""
    from .evidence import LAB_EXPECTED_UNITS
    group = {"calcium_total": "total_calcium", "calcium_ionized": "ionized_calcium",
             "ph": "blood_gas_ph"}.get(analyte.analyte_id, analyte.analyte_id)
    units = analyte.accepted_units or LAB_EXPECTED_UNITS.get(group, ())
    if not units or any(not isinstance(unit, str) or not unit.strip() for unit in units):
        raise ValueError(f"Explicit accepted units required for {analyte.analyte_id}")
    return tuple(dict.fromkeys(unit.strip().lower() for unit in units))


# Total and ionized calcium measure different fractions. Changing concentration
# units does not make them interchangeable. No albumin-based correction is
# inferred here; item ids are restricted to blood measurements.
DIALYSIS_LAB_ANALYTES: tuple[LabAnalyte, ...] = (
    LabAnalyte(
        "potassium",
        "Potassium",
        SERUM_POTASSIUM_ITEMIDS,
        POTASSIUM_BOUNDS,
        "mEq/L (numerically mmol/L for K+)",
    ),
    LabAnalyte(
        "calcium_total",
        "Total calcium",
        (50893,),
        (4.0, 15.0),
        "mg/dL",
    ),
    LabAnalyte(
        "calcium_ionized",
        "Ionized calcium",
        (50808, 51624),
        (0.4, 2.5),
        "mmol/L",
    ),
    LabAnalyte(
        "magnesium",
        "Magnesium",
        (50960,),
        (0.4, 8.0),
        "mg/dL",
    ),
    LabAnalyte(
        "ph",
        "Blood pH",
        (50820,),
        (6.5, 8.0),
        "pH units",
    ),
)


def _sql(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _paired_ecg_source(master_sql: str, split_list: str) -> str:
    """One reference-blind representative per patient/clock time, before QC.

    Study identifiers break clock ties, not inferred chronology. Picking once
    ensures the preceding and following pair use the same ECG at that time.
    """
    return f"""(SELECT *, count(*) OVER (PARTITION BY subject_id, ecg_time) AS ecg_clock_tie_count
        FROM read_parquet('{master_sql}')
        WHERE split IN ({split_list}) AND ecg_time IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY subject_id, ecg_time ORDER BY study_id) = 1)"""


def _pair_quality_clause(window: InterventionWindow) -> str:
    """Both already-selected recordings need finite normalized quality scores."""
    minimum = float(window.minimum_signal_quality)
    if isinstance(window.minimum_signal_quality, (bool, np.bool_)) or not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError("Minimum signal quality must be finite and between zero and one")
    return " AND ".join(
        f"(isfinite({alias}.signal_quality_score) AND {alias}.signal_quality_score BETWEEN {minimum} AND 1.0)"
        for alias in ("a", "b"))


def intervention_pair_query(
    master_sql: str,
    procedure_sql: str,
    feature_clause: str,
    *,
    window: InterventionWindow,
    marker_columns: str,
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """Two ECGs of one patient with the intervention between them.

    The intervention has to fall *strictly* between the recordings. A session
    that started before the first ECG would put both on the same side of it,
    and the difference would measure time rather than treatment.
    """

    id_list = ", ".join(str(int(item)) for item in window.procedure_itemids)
    split_list = ", ".join(f"'{value}'" for value in splits)
    # The admission columns live on the candidate pairs, not on the feature
    # tables -- ``a`` and ``b`` are ECG measurements and carry no hadm_id.
    admission = (
        "AND t.first_hadm_id IS NOT NULL "
        "AND t.first_hadm_id = t.second_hadm_id"
        if window.require_same_admission
        else ""
    )
    return f"""
        WITH ordered AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time, m.hadm_id,
                m.ecg_clock_tie_count, LEAD(m.ecg_clock_tie_count) OVER w AS next_clock_tie_count,
                LEAD(m.study_id) OVER w AS next_study_id,
                LEAD(m.ecg_time) OVER w AS next_ecg_time,
                LEAD(m.hadm_id)  OVER w AS next_hadm_id
            FROM {_paired_ecg_source(master_sql, split_list)} m
            WINDOW w AS (PARTITION BY m.subject_id ORDER BY m.ecg_time)
        ),
        candidate AS (
            SELECT
                o.subject_id, o.split,
                o.ecg_clock_tie_count AS first_clock_tie_count, o.next_clock_tie_count AS second_clock_tie_count,
                o.study_id      AS first_study_id,
                o.next_study_id AS second_study_id,
                o.ecg_time      AS first_ecg_time,
                o.next_ecg_time AS second_ecg_time,
                o.hadm_id       AS first_hadm_id,
                o.next_hadm_id  AS second_hadm_id,
                DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                    AS hours_between
            FROM ordered o
            WHERE o.next_study_id IS NOT NULL
              AND DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                  BETWEEN {window.minimum_hours_between}
                      AND {window.maximum_hours_between}
        ),
        treated AS (
            SELECT c.*, count(p.itemid) AS n_sessions,
                count(p.itemid) AS n_procedure_rows,
                count(DISTINCT p.itemid) AS n_procedure_types,
                string_agg(DISTINCT CAST(p.itemid AS VARCHAR), ',' ORDER BY CAST(p.itemid AS VARCHAR))
                    AS procedure_itemids,
                'procedure_rows_not_independent_sessions' AS intervention_count_semantics
            FROM candidate c
            JOIN read_parquet('{procedure_sql}') p
              ON p.subject_id = c.subject_id
             AND p.hadm_id = c.first_hadm_id
             AND p.itemid IN ({id_list})
             AND p.starttime > c.first_ecg_time
             AND p.starttime < c.second_ecg_time
            GROUP BY ALL
        )
        SELECT
            t.subject_id, t.split, t.first_study_id, t.second_study_id,
            t.first_ecg_time, t.second_ecg_time, t.hours_between, t.n_sessions,
            t.n_procedure_rows, t.n_procedure_types, t.procedure_itemids,
            t.intervention_count_semantics,
            t.first_clock_tie_count, t.second_clock_tie_count,
            a.signal_quality_score AS signal_quality_first,
            b.signal_quality_score AS signal_quality_second,
            {marker_columns}
        FROM treated t
        JOIN {feature_clause} a
          ON a.subject_id = t.subject_id AND a.study_id = t.first_study_id
        JOIN {feature_clause} b
          ON b.subject_id = t.subject_id AND b.study_id = t.second_study_id
        WHERE a.extraction_status = 'ok' AND b.extraction_status = 'ok'
          AND {_pair_quality_clause(window)}
          {admission}
    """


def nearest_lab_query(
    master_sql: str,
    lab_sql: str,
    *,
    itemids: Sequence[int],
    window_hours: float = 6.0,
) -> str:
    """Compatibility projection of the shared retrospective laboratory query."""
    matches = [item for item in DIALYSIS_LAB_ANALYTES if itemids and set(itemids) <= set(item.itemids)]
    if len(matches) != 1:
        raise ValueError("Requested items require one known analyte and explicit unit policy")
    definition = matches[0]
    analyte = LabAnalyte(definition.analyte_id, definition.display_name, tuple(itemids),
                         definition.bounds, definition.unit, _analyte_units(definition))
    query = nearest_labs_query(master_sql, lab_sql, analytes=(analyte,), window_hours=window_hours)
    return f"SELECT * EXCLUDE(analyte_id) FROM ({query}) selected_laboratory"


def nearest_labs_query(
    master_sql: str,
    lab_sql: str,
    *,
    analytes: Sequence[LabAnalyte] = DIALYSIS_LAB_ANALYTES,
    window_hours: float = 6.0,
    selected_ecgs_relation: str | None = None,
) -> str:
    """Same-admission retrospective response reference, not prediction input.

    Select the nearest time before checking its value; no fallback to a more
    distant valid measurement. Equidistant times prefer the earlier sample.
    Conflicting measurements at that time abstain instead of choosing a row.
    """

    if not np.isfinite(window_hours) or window_hours <= 0:
        raise ValueError("A finite positive laboratory window is required")
    if not analytes:
        raise ValueError("At least one laboratory analyte is required")
    seen_ids: set[int] = set()
    rows: list[str] = []
    for analyte in analytes:
        if not analyte.itemids:
            raise ValueError(f"Analyte {analyte.analyte_id} has no item ids")
        quoted = analyte.analyte_id.replace("'", "''")
        allowed = "[" + ", ".join("'" + unit.replace("'", "''") + "'" for unit in _analyte_units(analyte)) + "]"
        for itemid in analyte.itemids:
            numeric = int(itemid)
            if numeric in seen_ids:
                raise ValueError(f"Lab item {numeric} belongs to multiple analytes")
            seen_ids.add(numeric)
            rows.append(f"({numeric}, '{quoted}', {allowed})")
    mapping = ", ".join(rows)
    selection = ""
    if selected_ecgs_relation is not None:
        identifier = selected_ecgs_relation.replace('"', '""')
        selection = f'SEMI JOIN "{identifier}" selected_ecgs ON selected_ecgs.subject_id=m.subject_id AND selected_ecgs.study_id=m.study_id'
    minutes = int(round(float(window_hours) * 60.0))
    from .lab_quality import hemolysis_invalidates_sql

    hemolyzed = hemolysis_invalidates_sql("l.comments", "l.itemid")
    return f"""
        WITH analyte_map(itemid, analyte_id, allowed_units) AS (
            VALUES {mapping}
        ), ranked AS (
            SELECT
                m.subject_id,
                m.study_id,
                a.analyte_id,
                l.valuenum AS raw_value, l.itemid,
                lower(trim(l.valueuom)) AS unit, l.charttime,
                list_contains(a.allowed_units, lower(trim(l.valueuom))) AS unit_allowed,
                {hemolyzed} AS hemolysis_invalidated,
                DATE_DIFF('microseconds', m.ecg_time, l.charttime) / 3600000000.0
                    AS delta_hours,
                DENSE_RANK() OVER (
                    PARTITION BY m.study_id, a.analyte_id
                    ORDER BY
                        abs(DATE_DIFF('microseconds', m.ecg_time, l.charttime)),
                        l.charttime
                ) AS rank_by_distance
            FROM read_parquet('{master_sql}') m
            {selection}
            JOIN read_parquet('{lab_sql}') l
              ON l.subject_id = m.subject_id
             AND l.hadm_id = m.hadm_id
            JOIN analyte_map a USING (itemid)
            WHERE m.split IN ('train', 'validation')
              AND l.charttime BETWEEN m.ecg_time - INTERVAL {minutes} MINUTE
                                  AND m.ecg_time + INTERVAL {minutes} MINUTE
        ), nearest AS (
            SELECT subject_id, study_id, analyte_id, min(charttime) AS charttime,
                min(delta_hours) AS delta_hours, count(*) AS lab_source_rows,
                min(raw_value) AS candidate_value, min(itemid) AS itemid, min(unit) AS unit,
                CASE WHEN count(DISTINCT itemid) <> 1
                           OR count(DISTINCT raw_value) > 1 OR count(DISTINCT unit) > 1
                     THEN 'conflicting_nearest_measurements'
                     WHEN NOT bool_and(coalesce(isfinite(raw_value), FALSE))
                     THEN 'invalid_or_missing_nearest_value'
                     WHEN count(nullif(unit, '')) <> count(*)
                     THEN 'missing_nearest_unit'
                     WHEN NOT bool_and(coalesce(unit_allowed, FALSE))
                     THEN 'unit_not_supported_for_analyte'
                     -- Audit D2: the nearest sample is hemolyzed; no fallback
                     -- to a more distant one, as for every other defect.
                     WHEN bool_or(hemolysis_invalidated)
                     THEN 'hemolysis_reported'
                     ELSE 'eligible' END AS lab_reference_status
            FROM ranked WHERE rank_by_distance = 1
            GROUP BY subject_id, study_id, analyte_id
        )
        SELECT * EXCLUDE(candidate_value),
            CASE WHEN lab_reference_status = 'eligible' THEN candidate_value ELSE NULL END AS value
        FROM nearest
    """


def paired_change_table(
    pairs: pd.DataFrame,
    labs: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    bounds: tuple[float, float] = POTASSIUM_BOUNDS,
    first_suffix: str = "_first",
    second_suffix: str = "_second",
) -> pd.DataFrame:
    """One row per pair: how far the lab moved, how far each marker moved."""

    if pairs is None or pairs.empty:
        # Keep the output contract even for an empty pilot/subgroup. The stage
        # summary reads delta_lab unconditionally; an untyped empty frame used
        # to crash instead of reporting zero paired measurements.
        empty = pairs.copy() if pairs is not None else pd.DataFrame()
        for name in ("lab_first", "lab_second", "delta_lab", *[f"delta_{marker}" for marker in marker_ids]):
            empty[name] = pd.Series(index=empty.index, dtype=float)
        empty["lab_within_bounds"] = pd.Series(index=empty.index, dtype=bool)
        return empty
    lookup = labs.set_index("study_id")["value"] if labs is not None and not labs.empty else pd.Series(dtype=float)
    frame = pairs.copy()
    frame["lab_first"] = frame["first_study_id"].map(lookup)
    frame["lab_second"] = frame["second_study_id"].map(lookup)

    metadata = labs.set_index("study_id") if labs is not None and not labs.empty else pd.DataFrame()
    for source, target in (("itemid", "lab_itemid"), ("unit", "lab_unit"),
                           ("subject_id", "lab_subject_id"), ("lab_reference_status", "lab_reference_status"),
                           ("charttime", "lab_charttime"), ("delta_hours", "lab_delta_hours")):
        values = metadata[source] if source in metadata else pd.Series(dtype=object)
        for side in ("first", "second"):
            frame[f"{target}_{side}"] = frame[f"{side}_study_id"].map(values)
    reference_ok = (frame.lab_reference_status_first.eq("eligible")
                    & frame.lab_reference_status_second.eq("eligible")
                    & frame.lab_subject_id_first.eq(frame.subject_id)
                    & frame.lab_subject_id_second.eq(frame.subject_id)).fillna(False)
    compatible = (frame.lab_itemid_first.notna()
                  & frame.lab_itemid_first.eq(frame.lab_itemid_second)
                  & frame.lab_unit_first.notna()
                  & frame.lab_unit_first.eq(frame.lab_unit_second)).fillna(False)

    lower, upper = bounds
    plausible = (
        frame["lab_first"].between(lower, upper)
        & frame["lab_second"].between(lower, upper)
    )
    frame["lab_within_bounds"] = plausible
    frame["lab_pair_status"] = np.select(
        [~reference_ok, ~compatible, ~plausible],
        ["reference_ineligible_or_missing", "different_assay_or_unit", "outside_plausibility_bounds"],
        default="eligible")
    frame["delta_lab"] = np.where(
        plausible & reference_ok & compatible, frame["lab_second"] - frame["lab_first"], np.nan
    )
    for marker in marker_ids:
        first, second = f"{marker}{first_suffix}", f"{marker}{second_suffix}"
        if first in frame and second in frame:
            frame[f"delta_{marker}"] = pd.to_numeric(
                frame[second], errors="coerce"
            ) - pd.to_numeric(frame[first], errors="coerce")
    return frame


def within_subject_association(
    changes: pd.DataFrame,
    marker_ids: Sequence[str],
    *,
    minimum_pairs: int = 30,
) -> pd.DataFrame:
    """Correlate the change in each marker with the change in the lab.

    Spearman, because neither difference is expected to be normal and a single
    unit error would dominate a Pearson coefficient.

    Differencing reduces stable between-person differences but does not prove
    exact removal of confounding or interactions. Repeated pairs from the same
    patient require clustered uncertainty. Concurrent changes remain relevant --
    the session itself removes fluid as well as potassium, and that is a
    limitation of the design, not of the arithmetic.
    """

    if changes is None or changes.empty or "delta_lab" not in changes:
        return pd.DataFrame(
            columns=[
                "feature_id",
                "n_pairs",
                "spearman_rho",
                "p_value",
                "median_delta_marker",
                "median_delta_lab",
            ]
        )
    records: list[dict] = []
    lab = pd.to_numeric(changes["delta_lab"], errors="coerce")
    prepared = ClusteredRankBatch(lab, changes["subject_id"] if "subject_id" in changes else None)
    for marker in marker_ids:
        column = f"delta_{marker}"
        if column not in changes:
            continue
        value = pd.to_numeric(changes[column], errors="coerce")
        estimate, usable = prepared.estimate(value, minimum_pairs=int(minimum_pairs))
        if estimate is None:
            continue
        n = int(usable.sum())
        records.append(
            {
                "feature_id": str(marker),
                "n_pairs": n,
                **estimate,
                "median_delta_marker": float(value[usable].median()),
                "median_delta_lab": float(lab[usable].median()),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "feature_id",
                "n_pairs",
                "spearman_rho",
                "p_value",
                "median_delta_marker",
                "median_delta_lab",
            ]
        )
    return pd.DataFrame(records).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False, ignore_index=True
    )


def add_within_subject_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    """Correct each prespecified design family and the full sensitivity family."""

    output = frame.copy()
    output["design_q_value"] = np.nan
    output["global_q_value"] = np.nan
    output["design_fdr_significant"] = False
    output["global_fdr_significant"] = False
    if output.empty or not {"design", "p_value"} <= set(output.columns):
        return output
    from statsmodels.stats.multitest import multipletests

    p_values = pd.to_numeric(output["p_value"], errors="coerce")
    valid = p_values.notna() & np.isfinite(p_values)
    for _, indices in output.loc[valid].groupby("design", sort=True).groups.items():
        adjusted = multipletests(p_values.loc[indices], method="fdr_bh")[1]
        output.loc[indices, "design_q_value"] = adjusted
        output.loc[indices, "design_fdr_significant"] = adjusted <= 0.05
    if valid.any():
        indices = output.index[valid]
        adjusted = multipletests(p_values.loc[indices], method="fdr_bh")[1]
        output.loc[indices, "global_q_value"] = adjusted
        output.loc[indices, "global_fdr_significant"] = adjusted <= 0.05
    return output


def build_dialysis_lab_pairs(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    marker_ids: Sequence[str],
    window: InterventionWindow = DIALYSIS_WINDOW,
    analytes: Sequence[LabAnalyte] = DIALYSIS_LAB_ANALYTES,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read dialysis pairs and every prespecified lab in one database scan."""

    master = config.master_root / "ecg_master_index.parquet"
    if not master.is_file():
        raise FileNotFoundError(master)
    procedures = _events_glob(config, "procedureevents")
    labs = _events_glob(config, "labevents")

    feature_sql = str(feature_glob).replace("\\", "/").replace("'", "''")
    clause = feature_parquet_clause(sql_path=feature_sql)
    columns = ",\n            ".join(
        f'a."{marker}" AS "{marker}_first", b."{marker}" AS "{marker}_second"'
        for marker in marker_ids
    ) or "NULL::DOUBLE AS no_marker_selected"

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        if progress:
            progress(f"{window.display_name}: EKG-Paare um die Intervention suchen")
        pairs = connection.execute(
            intervention_pair_query(
                _sql(master),
                procedures,
                clause,
                window=window,
                marker_columns=columns,
            )
        ).fetchdf()
        if progress:
            progress(
                f"{window.display_name}: {len(pairs):,} Paare, "
                "Laborwerte zuordnen"
            )
        if pairs.empty:
            values = pd.DataFrame(columns=["subject_id", "study_id", "analyte_id", "charttime",
                "delta_hours", "lab_source_rows", "itemid", "unit", "lab_reference_status", "value"])
        else:
            selected = pd.concat([
                pairs[["subject_id", f"{side}_study_id"]].rename(columns={f"{side}_study_id": "study_id"})
                for side in ("first", "second")
            ], ignore_index=True).drop_duplicates()
            connection.register("paired_reference_ecgs", selected)
            try:
                values = connection.execute(
                    nearest_labs_query(_sql(master), labs, analytes=analytes,
                                       selected_ecgs_relation="paired_reference_ecgs")
                ).fetchdf()
            finally:
                connection.unregister("paired_reference_ecgs")
    return pairs, values


def build_dialysis_potassium_pairs(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    marker_ids: Sequence[str],
    window: InterventionWindow = DIALYSIS_WINDOW,
    lab_itemids: Sequence[int] = SERUM_POTASSIUM_ITEMIDS,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper for callers that request potassium only."""

    analyte = LabAnalyte(
        "potassium",
        "Potassium",
        tuple(int(item) for item in lab_itemids),
        POTASSIUM_BOUNDS,
        "mEq/L",
    )
    pairs, values = build_dialysis_lab_pairs(
        config,
        feature_glob=feature_glob,
        marker_ids=marker_ids,
        window=window,
        analytes=(analyte,),
        progress=progress,
    )
    return pairs, values.drop(columns="analyte_id", errors="ignore")


# --------------------------------------------------------- dose response


# Minimum administrations of one drug before its dose can be ranked. Below
# this the "dose" axis is a handful of points and the correlation is noise.
DOSE_MINIMUM_ADMINISTRATIONS = 50

# How long after a dose the second ECG may be taken.
#
# Measured on beta blockers, 2026-08-30, RR-interval variability against
# ranked dose:
#
# ========  =======  ======  =====
# Fenster   Paare    rho     p
# ========  =======  ======  =====
#     4 h       204  +0.151  0.057
#    24 h     1,818  +0.036  0.523
# ========  =======  ======  =====
#
# Four times the effect on a ninth of the data. Metoprolol given intravenously
# acts within minutes and is largely gone within hours; a day-wide window pairs
# ECGs with doses that had stopped acting, and averages the effect towards
# nothing. The width is therefore part of the design, not a convenience.
#
# A caveat that belongs with any result from this: heart rate itself is not in
# the feature set -- its role is ``ecg_context_comparator`` -- so the marker
# with the clearest expected dose-response cannot be tested here at all.
DRUG_WINDOW = InterventionWindow(
    intervention_id="drug_dose",
    display_name="Medikamentengabe",
    procedure_itemids=(),
    minimum_hours_between=0.5,
    maximum_hours_between=6.0,
)


def _dose_identity_ctes(*, classified=False):
    """Keep rate segments, collapse identical copies, flag conflicting orders."""
    category = "drug_class, " if classified else ""
    return f"""dose_unique AS (
        SELECT *, count(*) AS source_row_count FROM dosed GROUP BY ALL
    ), dose_identity AS (
        SELECT *, count(*) OVER (PARTITION BY {category}subject_id, hadm_id,
            orderid, itemid, starttime) AS n_identity_versions
        FROM dose_unique
    ),"""


def _dose_aggregate_columns(*, bounded=False):
    """Do not sum different drugs or units into an arbitrarily labelled dose."""
    valid = "COUNT(DISTINCT x.itemid)=1 AND COUNT(DISTINCT lower(trim(x.amountuom)))=1 AND COUNT(NULLIF(trim(x.amountuom), ''))=COUNT(*)"
    temporal = "BOOL_AND(COALESCE(x.starttime > c.first_ecg_time AND x.endtime >= x.starttime AND x.endtime < c.second_ecg_time, FALSE))" if bounded else "TRUE"
    # MIMIC-IV records completed rate segments as ChangeDose/Rate; older
    # descriptions use Changed. Keep the existing timing/identity/amount gates.
    documented = "BOOL_AND(COALESCE(lower(trim(x.statusdescription)) IN ('changed','changedose/rate','paused','finishedrunning','stopped','flushed'), FALSE))" if bounded else "TRUE"
    identity = "BOOL_AND(x.orderid IS NOT NULL AND x.orderid > 0 AND x.n_identity_versions=1)" if bounded else "TRUE"
    amount_valid = "BOOL_AND(COALESCE(isfinite(x.amount) AND x.amount > 0, FALSE)) AND COALESCE(isfinite(sum(x.amount)), FALSE)"
    valid = f"({valid}) AND ({temporal}) AND ({documented}) AND ({identity}) AND ({amount_valid})"
    source_count = "sum(x.source_row_count)" if bounded else "count(*)"
    return f"""CASE WHEN {valid} THEN sum(x.amount) ELSE NULL END AS dose_amount,
        count(*) AS n_administrations,
        {source_count} AS dose_source_rows,
        CASE WHEN {valid} THEN min(x.itemid) ELSE NULL END AS dose_itemid,
        CASE WHEN {valid} THEN min(lower(trim(x.amountuom))) ELSE NULL END AS dose_unit,
        CASE WHEN {valid} THEN min(x.label) ELSE NULL END AS dose_label,
        CASE WHEN NOT ({identity}) THEN 'unknown_or_conflicting_order_identity'
             WHEN NOT ({documented}) THEN 'unknown_or_unusable_administration_status'
             WHEN NOT ({temporal}) THEN 'unbounded_or_invalid_exposure_interval'
             WHEN NOT ({amount_valid}) THEN 'invalid_or_missing_dose_amount'
             WHEN {valid} THEN 'eligible' ELSE 'mixed_drug_or_unit_or_missing_unit' END AS dose_aggregation_status"""


def dose_pair_query(
    master_sql: str,
    input_sql: str,
    items_sql: str,
    feature_clause: str,
    *,
    window: InterventionWindow,
    marker_columns: str,
    label_pattern: str,
    context_columns: Sequence[str] = (),
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """ECG before and after an administration, with the dose that was given.

    ``context_columns`` names columns to pair from the *master index* rather
    than the feature partitions. There is exactly one reason this exists:
    ``machine_heart_rate_bpm``. Heart rate is a mechanistic comparator, not a
    deterministic pairing validator: indication, rhythm, exposure timing and
    concurrent treatment can obscure or reverse the observed association. It is
    deliberately not a biomarker candidate (its role is
    ``ecg_context_comparator``, the thing findings are adjusted *for*), so it
    is absent from the feature panels and has to come from the index.

    Restricted to documented exposure within the ECG pair. This estimates
    an association between recorded dose and marker change, not a causal
    response. Restricting to treated patients does not remove confounding by
    indication, severity, treatment duration, or concurrent therapies. Drug
    administration alone also does not establish the suspected diagnosis.
    """

    split_list = ", ".join(f"'{value}'" for value in splits)
    admission = (
        "AND t.first_hadm_id IS NOT NULL "
        "AND t.first_hadm_id = t.second_hadm_id"
        if window.require_same_admission
        else ""
    )
    context = tuple(str(name) for name in context_columns)
    context_select = "".join(
        f", mb.{name} - ma.{name} AS delta_{name}" for name in context
    )
    context_join = (
        f"""JOIN read_parquet('{master_sql}') ma
          ON ma.study_id = t.first_study_id
        JOIN read_parquet('{master_sql}') mb
          ON mb.study_id = t.second_study_id"""
        if context
        else ""
    )
    return f"""
        WITH dosed AS (
            SELECT
                i.subject_id, i.hadm_id, i.starttime, i.endtime, i.itemid,
                i.amount, i.amountuom, i.statusdescription, i.orderid, i.linkorderid, d.label
            FROM read_parquet('{input_sql}') i
            JOIN read_parquet('{items_sql}') d ON d.itemid = i.itemid
            WHERE lower(trim(COALESCE(i.statusdescription, ''))) <> 'rewritten'
              AND regexp_matches(lower(d.label), '{label_pattern}')
        ),
        {_dose_identity_ctes()}
        ordered AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time, m.hadm_id,
                m.ecg_clock_tie_count, LEAD(m.ecg_clock_tie_count) OVER w AS next_clock_tie_count,
                LEAD(m.study_id) OVER w AS next_study_id,
                LEAD(m.ecg_time) OVER w AS next_ecg_time,
                LEAD(m.hadm_id)  OVER w AS next_hadm_id
            FROM {_paired_ecg_source(master_sql, split_list)} m
            WINDOW w AS (PARTITION BY m.subject_id ORDER BY m.ecg_time)
        ),
        candidate AS (
            SELECT
                o.subject_id, o.split,
                o.ecg_clock_tie_count AS first_clock_tie_count, o.next_clock_tie_count AS second_clock_tie_count,
                o.study_id      AS first_study_id,
                o.next_study_id AS second_study_id,
                o.ecg_time      AS first_ecg_time,
                o.next_ecg_time AS second_ecg_time,
                o.hadm_id       AS first_hadm_id,
                o.next_hadm_id  AS second_hadm_id,
                DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                    AS hours_between
            FROM ordered o
            WHERE o.next_study_id IS NOT NULL
              AND DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                  BETWEEN {window.minimum_hours_between}
                      AND {window.maximum_hours_between}
        ),
        treated AS (
            SELECT
                c.*,
                {_dose_aggregate_columns(bounded=True)}
            FROM candidate c
            JOIN dose_identity x
              ON x.subject_id = c.subject_id
             AND x.hadm_id = c.first_hadm_id
             AND x.starttime < c.second_ecg_time
             AND (x.starttime >= c.first_ecg_time OR x.endtime > c.first_ecg_time)
            GROUP BY ALL
        )
        SELECT
            t.subject_id, t.split, t.first_study_id, t.second_study_id,
            t.first_ecg_time, t.second_ecg_time, t.hours_between,
            t.first_clock_tie_count, t.second_clock_tie_count,
            t.dose_amount, t.n_administrations, t.dose_itemid, t.dose_unit,
            t.dose_label, t.dose_aggregation_status, t.dose_source_rows,
            a.signal_quality_score AS signal_quality_first, b.signal_quality_score AS signal_quality_second,
            {marker_columns}{context_select}
        FROM treated t
        JOIN {feature_clause} a
          ON a.subject_id = t.subject_id AND a.study_id = t.first_study_id
        JOIN {feature_clause} b
          ON b.subject_id = t.subject_id AND b.study_id = t.second_study_id
        {context_join}
        WHERE a.extraction_status = 'ok' AND b.extraction_status = 'ok'
          AND {_pair_quality_clause(window)}
          {admission}
    """


def dose_pairs_all_classes_query(
    master_sql: str,
    input_sql: str,
    items_sql: str,
    feature_clause: str,
    *,
    window: InterventionWindow,
    marker_ids: Sequence[str],
    class_patterns: Mapping[str, str],
    context_columns: Sequence[str] = (),
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """Build every prespecified drug-class pair in one physical scan plan.

    ``dose_pair_query`` remains the small, single-class compatibility API.
    The workflow, however, must not rescan inputevents, the master index and
    both feature sides once per class. A class-map relation makes class an
    ordinary grouping column, while a narrow materialized feature CTE lets the
    two ECG sides reuse one projection from Parquet.
    """

    classes = tuple(
        (str(name), str(pattern)) for name, pattern in class_patterns.items()
    )
    if not classes:
        raise ValueError("At least one drug class is required")
    markers = tuple(dict.fromkeys(str(name) for name in marker_ids))
    if not markers:
        raise ValueError("At least one marker is required")

    def sql_literal(value: str) -> str:
        return value.replace("'", "''")

    def identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    classified_items = "\n            UNION ALL\n            ".join(
        "SELECT itemid, label, "
        f"'{sql_literal(name)}' AS drug_class "
        "FROM item_dictionary "
        f"WHERE regexp_matches(label_lower, '{sql_literal(pattern)}')"
        for name, pattern in classes
    )
    split_list = ", ".join(f"'{sql_literal(str(value))}'" for value in splits)
    context = tuple(dict.fromkeys(str(name) for name in context_columns))
    marker_projection = ", ".join(identifier(name) for name in markers)
    marker_deltas = ", ".join(
        f"b.{identifier(name)} - a.{identifier(name)} AS "
        f"{identifier('delta_' + name)}"
        for name in markers
    )
    ordered_context = "".join(
        f", m.{identifier(name)}, "
        f"LEAD(m.{identifier(name)}) OVER w AS {identifier('next_' + name)}"
        for name in context
    )
    candidate_context = "".join(
        f", o.{identifier('next_' + name)} - o.{identifier(name)} AS "
        f"{identifier('delta_' + name)}"
        for name in context
    )
    selected_context = "".join(
        f", t.{identifier('delta_' + name)}" for name in context
    )
    admission = (
        "AND t.first_hadm_id IS NOT NULL "
        "AND t.first_hadm_id = t.second_hadm_id"
        if window.require_same_admission
        else ""
    )
    return f"""
        WITH item_dictionary AS MATERIALIZED (
            SELECT itemid, label, lower(label) AS label_lower
            FROM read_parquet('{items_sql}')
        ),
        classified_items AS (
            {classified_items}
        ),
        dosed AS (
            SELECT
                d.drug_class,
                i.subject_id, i.hadm_id, i.starttime, i.endtime, i.itemid,
                i.amount, i.amountuom, i.statusdescription, i.orderid, i.linkorderid, d.label
            FROM read_parquet('{input_sql}') i
            JOIN classified_items d ON d.itemid = i.itemid
            WHERE lower(trim(COALESCE(i.statusdescription, ''))) <> 'rewritten'
        ),
        {_dose_identity_ctes(classified=True)}
        ordered AS (
            SELECT
                m.subject_id, m.study_id, m.split, m.ecg_time, m.hadm_id,
                m.ecg_clock_tie_count, LEAD(m.ecg_clock_tie_count) OVER w AS next_clock_tie_count,
                LEAD(m.study_id) OVER w AS next_study_id,
                LEAD(m.ecg_time) OVER w AS next_ecg_time,
                LEAD(m.hadm_id)  OVER w AS next_hadm_id
                {ordered_context}
            FROM {_paired_ecg_source(master_sql, split_list)} m
            WINDOW w AS (PARTITION BY m.subject_id ORDER BY m.ecg_time)
        ),
        candidate AS (
            SELECT
                o.subject_id, o.split,
                o.ecg_clock_tie_count AS first_clock_tie_count, o.next_clock_tie_count AS second_clock_tie_count,
                o.study_id      AS first_study_id,
                o.next_study_id AS second_study_id,
                o.ecg_time      AS first_ecg_time,
                o.next_ecg_time AS second_ecg_time,
                o.hadm_id       AS first_hadm_id,
                o.next_hadm_id  AS second_hadm_id,
                DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                    AS hours_between
                {candidate_context}
            FROM ordered o
            WHERE o.next_study_id IS NOT NULL
              AND DATE_DIFF('second', o.ecg_time, o.next_ecg_time) / 3600.0
                  BETWEEN {window.minimum_hours_between}
                      AND {window.maximum_hours_between}
        ),
        treated AS (
            SELECT
                c.*,
                x.drug_class,
                {_dose_aggregate_columns(bounded=True)}
            FROM candidate c
            JOIN dose_identity x
              ON x.subject_id = c.subject_id
             AND x.hadm_id = c.first_hadm_id
             AND x.starttime < c.second_ecg_time
             AND (x.starttime >= c.first_ecg_time OR x.endtime > c.first_ecg_time)
            GROUP BY ALL
        ),
        feature_rows AS MATERIALIZED (
            SELECT subject_id, study_id, signal_quality_score, {marker_projection}
            FROM {feature_clause}
            WHERE extraction_status = 'ok'
        )
        SELECT
            t.drug_class,
            t.subject_id, t.split, t.first_study_id, t.second_study_id,
            t.first_ecg_time, t.second_ecg_time, t.hours_between,
            t.first_clock_tie_count, t.second_clock_tie_count,
            t.dose_amount, t.n_administrations, t.dose_itemid, t.dose_unit,
            t.dose_label, t.dose_aggregation_status, t.dose_source_rows,
            a.signal_quality_score AS signal_quality_first, b.signal_quality_score AS signal_quality_second,
            {marker_deltas}{selected_context}
        FROM treated t
        JOIN feature_rows a
          ON a.subject_id = t.subject_id AND a.study_id = t.first_study_id
        JOIN feature_rows b
          ON b.subject_id = t.subject_id AND b.study_id = t.second_study_id
        WHERE 1 = 1
          AND {_pair_quality_clause(window)}
          {admission}
    """


def normalise_dose_within_drug(
    pairs,
    *,
    dose_column: str = "dose_amount",
    item_column: str = "dose_itemid",
    minimum_administrations: int = DOSE_MINIMUM_ADMINISTRATIONS,
):
    """Rank the dose within each drug before pooling drugs together.

    Milligrams of metoprolol and micrograms of esmolol are not comparable
    numbers, and ``amountuom`` proves it: pooling the raw amounts would rank
    patients by which drug they happened to receive. Ranking within the item
    asks whether this was a large recorded dose for this item and unit. It
    does not remove confounding by indication, duration, route or drug choice.
    The legacy minimum_administrations argument counts comparable pair doses,
    not unique patients or individual administrations.
    """

    if pairs is None or len(pairs) == 0 or dose_column not in pairs:
        return pairs
    if type(minimum_administrations) is not int or minimum_administrations < 1:
        raise ValueError("Positive comparable-pair threshold required")
    frame = pairs.copy()
    dose = pd.to_numeric(frame[dose_column], errors="coerce")
    from .patient_identity import known_patient_mask
    unit = frame.get("dose_unit", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip().str.lower()
    item = frame.get(item_column, pd.Series(pd.NA, index=frame.index))
    valid = np.isfinite(dose) & dose.gt(0) & known_patient_mask(item) & unit.notna() & unit.ne("")
    if "dose_aggregation_status" in frame:
        valid &= frame.dose_aggregation_status.eq("eligible")
    counts = dose.where(valid).groupby([item, unit]).transform("count").fillna(0)
    eligible = valid & counts.ge(int(minimum_administrations))
    frame["dose_rank"] = dose.where(eligible).groupby([item, unit]).rank(pct=True)
    frame["dose_drug_n"] = counts
    frame["dose_rank_status"] = np.where(eligible, "eligible",
        np.where(valid, "too_few_comparable_doses", "invalid_or_mixed_dose_identity_unit_amount"))
    return frame


def dose_response_association(
    changes,
    marker_ids: Sequence[str],
    *,
    minimum_pairs: int = 30,
    dose_column: str = "dose_rank",
):
    """Does more drug move the marker further, within the treated?"""

    columns = [
        "feature_id",
        "n_pairs",
        "spearman_rho",
        "p_value",
        "median_delta_marker",
    ]
    if changes is None or len(changes) == 0 or dose_column not in changes:
        return pd.DataFrame(columns=columns)
    dose = pd.to_numeric(changes[dose_column], errors="coerce")
    prepared = ClusteredRankBatch(dose, changes["subject_id"] if "subject_id" in changes else None)
    records: list[dict] = []
    for marker in marker_ids:
        column = f"delta_{marker}"
        if column not in changes:
            continue
        value = pd.to_numeric(changes[column], errors="coerce")
        estimate, usable = prepared.estimate(value, minimum_pairs=int(minimum_pairs))
        if estimate is None:
            continue
        n = int(usable.sum())
        records.append(
            {
                "feature_id": str(marker),
                "n_pairs": n,
                **estimate,
                "median_delta_marker": float(value[usable].median()),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).sort_values(
        "spearman_rho", key=lambda s: s.abs(), ascending=False, ignore_index=True
    )


def drug_class_pattern(drug_class: str) -> str:
    """A SQL-safe regular expression for one medication class."""

    from .mappings import MEDICATION_PATTERNS

    patterns = MEDICATION_PATTERNS.get(str(drug_class))
    if not patterns:
        raise ValueError(f"Unbekannte Medikamentenklasse: {drug_class!r}")
    return "|".join(str(item).lower().replace("'", "''") for item in patterns)


def _events_glob(config: MimicProjectConfig, table: str) -> str:
    """Resolve only the active integrity-checked ingest, never directory order."""
    from .event_ingest import load_clinical_event_registry

    registry = load_clinical_event_registry(config)
    if registry is None:
        raise FileNotFoundError("Kein vollständiges aktives Ereignis-Manifest; ingest-events zuerst")
    records = [item for item in registry.get("tables", ())
               if item.get("table_id") == table and item.get("status") == "ok"]
    if len(records) != 1:
        raise RuntimeError(f"Aktive Ereignis-Registry enthält keine eindeutige Tabelle {table}")
    root = Path(registry["_registry_path"]).parent.resolve()
    source = (root / records[0]["relative_glob"]).resolve()
    if not source.is_relative_to(root):
        raise RuntimeError("Ereignispfad liegt außerhalb des aktiven Registry-Verzeichnisses")
    return _sql(source)


# The markers the paired designs are run on. Repolarisation first, because
# that is what potassium moves and what the dialysis design exists to test;
# the conduction and voltage markers follow as the comparison that says
# whether the effect is specific or the whole ECG simply moved.
#
# Historical, unadjusted pilot observations over 604 dialysis pairs, NOT a
# current regression reference or evidence of mechanistic specificity:
#
# =================================  =======  =========  ====
# Marker                             rho      p          n
# =================================  =======  =========  ====
# tpeak_tend_ms                      -0,210   5e-06      463
# qrs_t_peak_magnitude_ratio         -0,168   4e-04      435
# classical_peaked_t_wave_heuristic  +0,149   0,0018     438
# t_loop_width_major                 +0,142   0,0027     446
# qrs_duration_ms                    +0,132   0,0016     569
# qtc_fridericia_ms                  -0,101   0,045      396
# st_vector_magnitude                -0,034   0,47       465
# =================================  =======  =========  ====
WITHIN_SUBJECT_MARKERS: tuple[str, ...] = (
    "tpeak_tend_ms",
    "t_loop_width_major",
    "qrs_t_peak_magnitude_ratio",
    "classical_peaked_t_wave_heuristic",
    "qtc_fridericia_ms",
    "qrs_duration_ms",
    "st_vector_magnitude",
)

# A contextual comparator with drug-/rhythm-/timing-dependent expectations.
# Absence of a negative observed dose-HR slope does not establish a pairing
# error, and the heterogeneous CCB/class-I/III groups have no universal sign.
# It is read from the master index
# because it is a context comparator rather than a candidate marker, so no
# feature panel contains it.
DOSE_CONTEXT_COLUMNS: tuple[str, ...] = ("machine_heart_rate_bpm",)

# Historical pilot estimates, not a current positive-control gate. In
# particular, a negative sign is not uniformly expected across these classes:
#
# ========================  =======  ======  ====
# Klasse                    rho      p       n
# ========================  =======  ======  ====
# beta_blocker              -0,098   0,127   246
# calcium_channel_blocker   -0,060   0,532   112
# antiarrhythmic_class_i_3  -0,028   0,694   198
# ========================  =======  ======  ====
#
# Repeated pairs are not independent patients. These legacy p-values/pair
# counts cannot calibrate current clustered power or certify a new marker.

# Drug classes whose dose is ranked within the item. Each has an expected
# pharmacological rationale, but a broad class is not a uniform HR control.
DOSE_RESPONSE_CLASSES: tuple[str, ...] = (
    "beta_blocker",
    "calcium_channel_blocker",
    "antiarrhythmic_class_i_iii",
)


def run_within_subject_change(
    config,
    *,
    run_id: str,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    markers: Sequence[str] = WITHIN_SUBJECT_MARKERS,
    drug_classes: Sequence[str] = DOSE_RESPONSE_CLASSES,
    feature_glob: Optional[str] = None,
    force: bool = False,
    preview: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Both paired designs: the dialysis pairs, and the dose gradient.

    They are one stage because they answer the same question in the same way
    -- a change inside one patient, controlling stable patient attributes but
    not time-varying confounding -- and because they read the same two event
    tables. They do not establish isolated causal treatment effects.
    """

    import duckdb

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        capture_output_integrity,
        content_hash,
        development_feature_glob,
        feature_manifest_path,
        feature_parquet_clause,
        input_artifact_fingerprints,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "within_subject"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "within_subject.manifest.json"

    marker_ids = tuple(dict.fromkeys(str(item) for item in markers))
    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    feature_manifest = feature_manifest_path(resolved_glob)
    extended_preview = bool(preview or "smoke-development-" in str(resolved_glob))
    recovery_labs = config.evidence_root / "labs.parquet"
    upstream = input_artifact_fingerprints(
        {
            "feature_manifest": feature_manifest,
            **{name: config.evidence_root / name for name in (
                "echo_measurements_wide.parquet", "echo_study_index.parquet", "pr_geometry_qualified.parquet",
                "pr_geometry_calibration.json", "pr_calibration_review_report.json")
                if (config.evidence_root / name).is_file()},
            **({"recovery_labs": recovery_labs} if recovery_labs.is_file() else {}),
            "feature_partition_index": feature_manifest.parent
            / "partition_index.json",
            "master_index_manifest": config.master_root
            / "master_index.manifest.json",
            "ingest_manifest": config.standardized_root / "ingest.manifest.json",
            "clinical_events_manifest": config.standardized_root
            / "clinical_events"
            / "clinical_events.manifest.json",
        }
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "extended_preview": extended_preview,
            "feature_glob": resolved_glob,
            "markers": list(marker_ids),
            "drug_classes": list(drug_classes),
            "dialysis_itemids": list(DIALYSIS_ITEMIDS),
            "dialysis_labs": [
                {
                    "analyte_id": analyte.analyte_id,
                    "itemids": list(analyte.itemids),
                    "bounds": list(analyte.bounds),
                    "accepted_units": list(_analyte_units(analyte)),
                    "unit": analyte.unit,
                }
                for analyte in DIALYSIS_LAB_ANALYTES
            ],
            "drug_window_hours": float(DRUG_WINDOW.maximum_hours_between),
            "drug_window_policy": DRUG_WINDOW.to_dict(),
            "dialysis_window_policy": DIALYSIS_WINDOW.to_dict(),
            "recovery_labs_available": recovery_labs.is_file(),
            "recovery_implementation": {
                name: content_hash((Path(__file__).parent / name).read_text(encoding="utf-8"))
                for name in ("recovery_loader.py", "recovery_reference.py", "recovery_episodes.py", "derived_markers.py", "clustered_rank_association.py", "paired_adjustment.py", "paired_clinical_context.py", "paired_lab_context.py", "temporal_response.py", "extended_mechanism_models.py", "extended_mechanism_workflow.py", "mechanism_echo_context.py", "pr_calibration_contract.py")
            },
            "upstream": upstream,
            "personal_extension_implementation": {
                name: content_hash((Path(__file__).parent / name).read_text(encoding="utf-8"))
                for name in ("pre_run_extensions.py", "pre_run_extension_stage.py")
            },
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, WITHIN_SUBJECT_VERSION):
            return previous

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    frames: list[pd.DataFrame] = []
    summary: dict = {
        "version": WITHIN_SUBJECT_VERSION,
        "run_id": str(run_id),
        "markers": list(marker_ids),
        "ecg_clock_tie_policy": "lowest_study_id_per_patient_timestamp_before_feature_or_reference_selection_no_fallback",
        "pair_quality_policy": "both_selected_recordings_finite_score_between_configured_minimum_and_one_no_fallback",
        "drug_window_policy": DRUG_WINDOW.to_dict(),
        "dialysis_window_policy": DIALYSIS_WINDOW.to_dict(),
    }

    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        clause = feature_parquet_clause(connection, sql_path(resolved_glob))
        available = set(
            connection.execute(f"SELECT * FROM {clause} LIMIT 0").df().columns
        )
        present = [name for name in marker_ids if name in available]
        if not present:
            raise ValueError("Kein Marker des Paardesigns im Merkmalssatz")

        from .pre_run_extension_stage import stream_personal_deviations
        personal_paths, summary["personal_deviation"] = stream_personal_deviations(
            connection, clause, available, output_root)

        from .extended_mechanism_workflow import run_extended_mechanisms
        if progress:
            progress("Intraindividuell: Echo-Verlauf, elektromechanische Residuen und qualifizierte PR-Kopplung")
        extended_outputs, summary["extended_mechanisms"] = run_extended_mechanisms(
            connection, config, clause, available, preview=extended_preview)

        recovery_results = pd.DataFrame()
        if recovery_labs.is_file():
            from .recovery_loader import dialysis_recovery
            if progress:
                progress("Intraindividuell: retrospektive Dialyse-Erholungsepisoden und Laborreferenzen")
            recovery_results, summary["recovery"] = dialysis_recovery(
                connection,
                master_relation=f"read_parquet('{_sql(config.master_root / 'ecg_master_index.parquet')}')",
                procedure_relation=f"read_parquet('{_events_glob(config, 'procedureevents')}')",
                feature_relation=clause,
                lab_relation=f"read_parquet('{_sql(recovery_labs)}')",
                marker_ids=tuple(present), dialysis_itemids=DIALYSIS_ITEMIDS)
        else:
            summary["recovery"] = {"status": "not_computed", "reason": "standardized_laboratory_evidence_missing",
                                   "clinical_claim_allowed": False}

        if progress:
            progress("Intraindividuell: Dialyse-Paare bilden")
        raw_pairs, labs = build_dialysis_lab_pairs(
            config,
            feature_glob=sql_path(resolved_glob),
            marker_ids=present,
            analytes=DIALYSIS_LAB_ANALYTES,
            progress=progress,
        )
        raw_pairs = attach_baseline_context(connection, raw_pairs, config.master_root / "ecg_master_index.parquet")
        from .temporal_response import dialysis_temporal_response
        temporal_pairs = dialysis_temporal_response(connection, clause, raw_pairs,
            descriptor_available="context_temporal_research_json" in available)
        summary["temporal_response"] = {
            "n_selected_pairs_before_missingness_filter": len(temporal_pairs),
            "status_counts": temporal_pairs["status"].value_counts().to_dict(),
            "clinical_claim_allowed": False,
        }
        change_frames: list[pd.DataFrame] = []
        summary["n_dialysis_pairs"] = int(len(raw_pairs))
        tie_columns = ["first_clock_tie_count", "second_clock_tie_count"]
        summary["n_retained_dialysis_pairs_with_clock_ties"] = (
            int(raw_pairs[tie_columns].gt(1).any(axis=1).sum()) if set(tie_columns) <= set(raw_pairs) else None)
        summary["dialysis_analytes"] = {}
        for analyte in DIALYSIS_LAB_ANALYTES:
            analyte_labs = labs.loc[
                labs["analyte_id"].astype(str).eq(analyte.analyte_id)
            ].drop(columns="analyte_id", errors="ignore")
            analyte_changes = paired_change_table(
                raw_pairs,
                analyte_labs,
                present,
                bounds=analyte.bounds,
            )
            analyte_changes.insert(0, "analyte_id", analyte.analyte_id)
            analyte_changes.insert(1, "analyte_unit", analyte.unit)
            change_frames.append(analyte_changes)
            association = within_subject_association(analyte_changes, present)
            if not association.empty:
                association = association.merge(paired_sensitivities(
                    analyte_changes, present, exposure_column="delta_lab"), on="feature_id", how="left", validate="one_to_one")
                association.insert(0, "design", f"dialysis_{analyte.analyte_id}")
                association.insert(1, "analyte_id", analyte.analyte_id)
                association.insert(2, "analyte_unit", analyte.unit)
                frames.append(association)
            lab_delta = pd.to_numeric(
                analyte_changes["delta_lab"], errors="coerce"
            )
            analyte_summary = {
                "display_name": analyte.display_name,
                "unit": analyte.unit,
                "n_pairs_with_lab": int(lab_delta.notna().sum()),
                "median_delta": (
                    float(lab_delta.median()) if lab_delta.notna().any() else None
                ),
                "plausibility_bounds": list(analyte.bounds),
                "accepted_units": list(_analyte_units(analyte)),
                "unit_conversion": "none_same_recorded_unit_required_across_pair",
                "paired_reference_status_counts": {
                    str(key): int(value) for key, value in analyte_changes["lab_pair_status"].value_counts().items()
                } if "lab_pair_status" in analyte_changes else {},
                "reference_design": "retrospective_same_admission_nearest_time_no_validity_fallback_same_assay_unit",
            }
            summary["dialysis_analytes"][analyte.analyte_id] = analyte_summary
            if analyte.analyte_id == "potassium":
                summary["n_dialysis_pairs_with_lab"] = analyte_summary[
                    "n_pairs_with_lab"
                ]
                summary["median_delta_potassium"] = analyte_summary[
                    "median_delta"
                ]
        changes = (
            pd.concat(change_frames, ignore_index=True)
            if change_frames
            else pd.DataFrame()
        )
        if progress:
            analyte_counts = ", ".join(
                f"{key}={value['n_pairs_with_lab']:,}"
                for key, value in summary["dialysis_analytes"].items()
            )
            progress(
                f"Intraindividuell: {len(raw_pairs):,} Dialyse-Paare; "
                + analyte_counts
            )

        summary["dose_response"] = {}
        class_patterns = {
            str(drug_class): drug_class_pattern(str(drug_class))
            for drug_class in drug_classes
        }
        try:
            all_dosed = connection.execute(
                dose_pairs_all_classes_query(
                    sql_path(config.master_root / "ecg_master_index.parquet"),
                    _events_glob(config, "inputevents"),
                    # The standardized copy, like every other table this
                    # stage reads. The raw csv.gz is not parquet and the
                    # query says read_parquet.
                    sql_path(config.standardized_root / "d_items.parquet"),
                    clause,
                    window=DRUG_WINDOW,
                    marker_ids=present,
                    class_patterns=class_patterns,
                    context_columns=DOSE_CONTEXT_COLUMNS,
                )
            ).df()
        except Exception as error:
            raise RuntimeError(
                "Within-subject combined dose query failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        all_dosed = attach_baseline_context(connection, all_dosed, config.master_root / "ecg_master_index.parquet")
        # Reuse the existing combined table: do not concatenate another wide
        # marker matrix merely to persist three per-class rank columns.
        all_dosed["dose_rank"] = np.nan
        all_dosed["dose_drug_n"] = np.nan
        all_dosed["dose_rank_status"] = pd.Series("not_evaluated", index=all_dosed.index, dtype=object)
        summary["baseline_context_policy"] = "first_selected_ecg_patient_and_study_development_only_no_second_ecg_fallback_partial_clinical_sensitivity"
        for drug_class in drug_classes:
            if progress:
                progress(f"Intraindividuell: Dosis-Wirkung {drug_class}")
            dosed = all_dosed.loc[
                all_dosed["drug_class"].astype(str).eq(str(drug_class))
            ].drop(columns="drug_class", errors="ignore")
            ranked = normalise_dose_within_drug(dosed)
            for column in ("dose_rank", "dose_drug_n", "dose_rank_status"):
                all_dosed.loc[ranked.index, column] = ranked[column]
            result = dose_response_association(
                ranked, list(present) + list(DOSE_CONTEXT_COLUMNS)
            )
            if not result.empty:
                result.insert(0, "design", f"dose_response_{drug_class}")
                result = result.merge(paired_sensitivities(
                    ranked, list(present) + list(DOSE_CONTEXT_COLUMNS), exposure_column="dose_rank"),
                    on="feature_id", how="left", validate="one_to_one")
                frames.append(result)
            summary["dose_response"][drug_class] = {
                "n_pairs": int(len(dosed)),
                "n_retained_pairs_with_clock_ties": (
                    int(dosed[tie_columns].gt(1).any(axis=1).sum()) if set(tie_columns) <= set(dosed) else None),
                "n_rank_eligible_pairs": int(ranked["dose_rank"].notna().sum()) if "dose_rank" in ranked else 0,
                "dose_rank_exclusions": {str(key): int(value) for key, value in
                    ranked.loc[ranked["dose_rank_status"].ne("eligible"), "dose_rank_status"].value_counts().items()}
                    if "dose_rank_status" in ranked else {},
                "rank_basis": "recorded_amount_within_itemid_and_unit_not_equivalent_doses_across_drugs",
                "aggregation_status_counts": {str(key): int(value) for key, value in
                    dosed["dose_aggregation_status"].value_counts(dropna=False).items()}
                    if "dose_aggregation_status" in dosed else {},
                "order_identity_policy": "collapse_identical_selected_exposure_rows_reject_conflicting_order_item_start_versions_preserve_distinct_segments",
                "amount_policy": "all_linked_nonrewritten_amounts_finite_positive_and_finite_sum_otherwise_abstain",
                "temporal_policy": "documented_start_and_end_strictly_inside_ecg_pair_same_admission",
                "temporal_limit": "overlapping_or_unresolved_administrations_abstain_no_prorating_no_complete_exposure_history_claim",
                "n_drugs": (
                    int(dosed["dose_itemid"].nunique()) if len(dosed) else 0
                ),
                "n_tested_markers": int(len(result)),
                "heart_rate_comparator": {
                    "role": "contextual_mechanistic_comparator_not_deterministic_positive_control",
                    "expected_direction": "negative_pharmacological_tendency_not_guaranteed_observed_slope"
                        if drug_class == "beta_blocker" else "no_uniform_class_wide_direction",
                    "null_or_opposite_sign_proves_pairing_error": False,
                    "requires_review": "specific_drug_rhythm_indication_timing_concurrent_therapy",
                },
            }

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lab_context, summary["joint_lab_coverage"] = joint_lab_context(changes)
    joint_models = joint_physiology_sensitivity(changes, lab_context, present)
    if not joint_models.empty and not table.empty:
        table = table.merge(joint_models, on=["design", "feature_id"], how="left", validate="one_to_one")
    table = add_within_subject_fdr(table)
    table = add_all_paired_adjustment_fdr(table)
    summary["joint_physiology_sensitivity"] = {
        "model": "joint_physiology_baseline_timing_quality_sensitivity",
        "n_estimable": int(table.joint_adjusted_status.eq("estimable").sum()) if "joint_adjusted_status" in table else 0,
        "clinical_claim_allowed": False,
        "limit": "conditional_response_association_not_causal_effect_fluid_therapy_indication_uncontrolled",
        "multiplicity": "separate_joint_BH_family_not_best_of_models_control",
        "covariate_policy": "baseline_timing_quality_plus_other_core_deltas_K_ionizedCa_Mg_pH_no_totalCa_substitution",
    }
    summary["baseline_clinical_sensitivity"] = {
        "model": "baseline_clinical_timing_quality_sensitivity",
        "n_estimable": int(table["clinical_adjusted_status"].eq("estimable").sum()) if "clinical_adjusted_status" in table else 0,
        "complete_time_varying_confounder_control": False,
        "covariates": "first_ecg_age_sex_heart_rate_pair_interval_both_quality_scores",
        "uncontrolled": "acuity_indication_concurrent_therapies_volume_electrolyte_context_setting_subtype",
        "multiplicity": "separate_BH_clinical_family_not_best_of_models_or_joint_primary_and_sensitivity_control",
    }
    summary["adjusted_sensitivity"] = {
        "model": "timing_quality_sensitivity_only",
        "n_estimable": int(table["adjusted_status"].eq("estimable").sum()) if "adjusted_status" in table else 0,
        "complete_time_varying_confounder_control": False,
        "multiplicity": "separate_BH_within_design_and_across_timing_quality_models_not_combined_with_primary_tests",
        "interpretation": "empirical_rank_slope_not_spearman_coefficient_not_causal",
    }
    summary["n_results"] = int(len(table))
    summary["inference"] = {
        "method": "patient_clustered_CR1_standardized_rank_OLS_t_Gminus1",
        "minimum_independent_patients": 30,
        "n_estimable": int(table["association_status"].eq("estimable").sum()) if "association_status" in table else 0,
        "n_non_estimable": int(table["association_status"].eq("non_estimable").sum()) if "association_status" in table else 0,
        "limit": "conditional_empirical_rank_regression_not_exact_population_spearman_inference",
        "time_varying_confounders_adjusted": False,
        "clinical_claim_allowed": False,
    }
    summary["multiplicity"] = {
        "design_family": "Benjamini-Hochberg within each intervention/analyte",
        "global_sensitivity_family": "Benjamini-Hochberg across all paired tests",
        "alpha": 0.05,
    }
    summary["dialysis_interpretation_limit"] = (
        "Dialysis changes electrolytes, acid-base state and volume together; "
        "single-analyte correlations are physiological associations, not "
        "isolated causal electrolyte effects."
    )

    # The paired changes themselves. A rank correlation of -0,21 over 604
    # pairs is a number; the cloud shows whether it holds throughout or rests
    # on a handful of large potassium shifts, which is a different finding.
    table_path = output_root / "within_subject_change.parquet"
    extended_paths = list(personal_paths)
    for name, frame in extended_outputs.items():
        path = output_root / (name + ".parquet")
        atomic_write_parquet(frame, path)
        extended_paths.append(path)
    changes_path = output_root / "dialysis_paired_changes.parquet"
    temporal_path = output_root / "dialysis_temporal_response.parquet"
    atomic_write_parquet(temporal_pairs, temporal_path)
    dose_changes_path = output_root / "dose_paired_changes.parquet"
    lab_context_path = output_root / "dialysis_joint_lab_context.parquet"
    atomic_write_parquet(lab_context, lab_context_path)
    summary_path = output_root / "within_subject_summary.json"
    recovery_path = output_root / "recovery_episode_results.parquet"
    atomic_write_parquet(recovery_results, recovery_path)
    atomic_write_parquet(table, table_path)
    atomic_write_parquet(all_dosed, dose_changes_path)
    summary["dose_pair_artifact"] = {
        "file": dose_changes_path.name,
        "n_pairs": int(len(all_dosed)),
        "retention": "all_post_pair_selection_rows_including_rank_abstentions",
        "limit": "not_all_source_administrations_not_pre_pair_QC_exclusion_denominator",
    }
    atomic_write_parquet(
        changes if len(changes) else pd.DataFrame(), changes_path
    )
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="within-subject-change",
        stage_version=WITHIN_SUBJECT_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            *(str(path.resolve()) for path in extended_paths),
            str(table_path.resolve()),
            str(changes_path.resolve()),
            str(dose_changes_path.resolve()),
            str(lab_context_path.resolve()),
            str(summary_path.resolve()),
            str(recovery_path.resolve()),
            str(temporal_path.resolve()),
        ),
        row_counts={
            "results": int(len(table)),
            "dialysis_pairs": int(summary.get("n_dialysis_pairs", 0)),
            "dose_pairs": int(len(all_dosed)),
            "joint_lab_pairs": int(len(lab_context)),
            "recovery_results": int(len(recovery_results)),
        },
        output_integrity=capture_output_integrity(
            (table_path, changes_path, dose_changes_path, lab_context_path, summary_path, recovery_path, temporal_path, *extended_paths)
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "DIALYSIS_ITEMIDS",
    "DIALYSIS_LAB_ANALYTES",
    "DIALYSIS_WINDOW",
    "POTASSIUM_BOUNDS",
    "SERUM_POTASSIUM_ITEMIDS",
    "WITHIN_SUBJECT_VERSION",
    "InterventionWindow",
    "LabAnalyte",
    "add_within_subject_fdr",
    "build_dialysis_lab_pairs",
    "build_dialysis_potassium_pairs",
    "intervention_pair_query",
    "nearest_lab_query",
    "nearest_labs_query",
    "paired_change_table",
    "DOSE_CONTEXT_COLUMNS",
    "DOSE_MINIMUM_ADMINISTRATIONS",
    "DOSE_RESPONSE_CLASSES",
    "WITHIN_SUBJECT_MARKERS",
    "DRUG_WINDOW",
    "dose_pair_query",
    "dose_response_association",
    "drug_class_pattern",
    "normalise_dose_within_drug",
    "run_within_subject_change",
    "within_subject_association",
]
