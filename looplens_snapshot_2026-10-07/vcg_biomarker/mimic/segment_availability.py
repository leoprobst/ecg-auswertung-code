"""Was the marker measurable, and does that depend on the diagnosis?

Every quality view in this project answers "does this ECG have a label". None
answers "was this marker measurable, and is the answer different in the
patients who have the disease". That second question is where a whole class of
defect lives, and until now nothing could show it.

The case that prompted this: the P segment is unusable in 61 % of recordings,
and the loss is not random. Measured over 1.377 ECGs drawn balanced on the
machine report, the P wave was measurable in **48,6 %** of recordings without
atrial fibrillation and **3,7 %** of recordings with it. Every atrial finding
in the project is therefore conditioned on not having the pathology being
looked for -- and the effect on a result is not a loss of power but a bias of
unknown direction.

A marker missing at the same rate in both groups costs power and nothing else.
A marker missing at different rates is measuring the endpoint through its own
absence. The two look identical in every table this project had, which is why
the gap between the two bars is the whole output here.

**How availability is decided.** There is no segment-state column in the
feature partitions; a marker that could not be measured is null. So one
sentinel marker per segment stands for that segment's delineation, chosen
because it needs the segment's own boundaries and nothing else. Counting nulls
of a marker that also needs, say, a U wave would blame the segment for a
different failure.

**What it found, across the whole endpoint set.** 288 endpoint-segment pairs
over 630.044 ECGs; 34 of them differ by 10 points or more, and every one of the
largest is the P segment:

=======================================  =========  =========  =======
Endpunkt                                 positiv    negativ    Abstand
=======================================  =========  =========  =======
digoxin_exposure_near_ecg                12,7 %     38,0 %     -25,3
acute_decompensated_heart_failure_proxy  20,9 %     42,9 %     -22,0
qtc_gt_480                               20,5 %     40,6 %     -20,2
qtc_gt_500                               20,8 %     39,6 %     -18,8
new_onset_af_30d                         29,7 %     48,2 %     -18,5
heart_failure_icd                        23,6 %     41,9 %     -18,3
=======================================  =========  =========  =======

This is the finding, and it is larger than the one that prompted the module.
The differential loss of the P wave is *not* specific to atrial fibrillation:
it is nearly as severe for digoxin exposure, for decompensated heart failure
and for the QTc endpoints. Every P-wave marker in this project is therefore
measured preferentially in the patients who do not have whichever condition is
being studied -- across most of the endpoint set, not one corner of it.

The direction is the damaging one. A marker missing at random costs power. A
marker missing twice as often in the cases is partly measuring the endpoint
through its own absence, and the bias it introduces has no fixed sign.

And it is the P segment specifically, which is what makes it fixable:

========  ============  ===================  ===============
Segment   Endpunkte     davon auffaellig     groesster Abstand
========  ============  ===================  ===============
p         72            **32**               25,3
t         72            1                    16,8
st        72            1                    16,0
qrs       72            **0**                4,7
========  ============  ===================  ===============

QRS availability does not depend on the diagnosis in a single one of the 72
endpoints. That contrast is the argument for the TQ window in one table: a
window anchored on the two landmarks the delineator is most confident about
inherits QRS's indifference rather than P's dependence.

**What this is not.** It is a description, not a correction. Knowing that the P
wave is missing differentially does not repair an atrial analysis -- it says
the analysis needs a window that does not depend on the P wave, which is what
the TQ markers are for. The comparison here is how one finds out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from .panels import STUDY_PANEL_IDS


SEGMENT_AVAILABILITY_VERSION = "1.0.0"

# One marker per segment, each needing that segment's boundaries and nothing
# beyond them. A marker that also depends on a U wave or on a second segment
# would report someone else's failure as this segment's.
SEGMENT_SENTINELS: Mapping[str, str] = {
    "p": "p_loop_width_major",
    "qrs": "qrs_duration_ms",
    "st": "st_vector_magnitude",
    "t": "t_loop_width_major",
    # The window this module exists to argue for. Absent from runs that
    # predate it, and then simply not reported.
    "tq": "tq_atrial_rms",
}

# Below this an availability difference is a handful of ECGs and reading a
# direction into it would be over-reading.
MINIMUM_GROUP_SIZE = 100


@dataclass(frozen=True)
class AvailabilityPolicy:
    """Everything that decides an availability number."""

    minimum_group_size: int = MINIMUM_GROUP_SIZE
    # A gap this wide or wider is called out. 10 points is not a threshold for
    # significance -- at these cohort sizes almost any gap is significant --
    # but the point at which a difference starts to matter for reading a
    # result.
    notable_gap_points: float = 10.0

    def to_dict(self) -> dict:
        return {
            "minimum_group_size": int(self.minimum_group_size),
            "notable_gap_points": float(self.notable_gap_points),
            "version": SEGMENT_AVAILABILITY_VERSION,
        }


def availability_query(
    master_sql: str,
    feature_clause: str,
    *,
    sentinels: Mapping[str, str],
    splits: Sequence[str] = ("train", "validation"),
) -> str:
    """One row per ECG: was each segment's sentinel measurable?

    Read once, for every endpoint at once. The first version issued one query
    per endpoint, each scanning the feature partitions -- 70 full scans of a
    1.889-column partition set for five columns of answer. Reading those five
    columns once and grouping in memory is the same result at a seventieth of
    the cost, and it is the pattern the rest of this project already uses.
    """

    split_list = ", ".join(f"'{value}'" for value in splits)
    flags = ",\n            ".join(
        f"(f.{marker} IS NOT NULL) AS has_{segment}"
        for segment, marker in sentinels.items()
    )
    return f"""
        SELECT
            m.subject_id,
            m.study_id,
            {flags}
        FROM read_parquet('{master_sql}') m
        JOIN {feature_clause} f
          ON f.subject_id = m.subject_id AND f.study_id = m.study_id
        WHERE f.extraction_status = 'ok'
          AND m.split IN ({split_list})
    """


def count_by_state(
    availability: pd.DataFrame,
    states: pd.Series,
    *,
    sentinels: Mapping[str, str],
) -> pd.DataFrame:
    """The per-state counts the table expects, from the read-once frame."""

    frame = availability.copy()
    frame["endpoint_state"] = states.to_numpy()
    frame = frame.loc[frame["endpoint_state"].isin(["positive", "negative"])]
    if frame.empty:
        return pd.DataFrame(columns=["endpoint_state", "n_ecgs"])
    grouped = frame.groupby("endpoint_state")
    counts = grouped.size().rename("n_ecgs").reset_index()
    for segment in sentinels:
        column = f"has_{segment}"
        if column not in frame.columns:
            continue
        totals = grouped[column].sum().rename(f"n_{segment}").reset_index()
        counts = counts.merge(totals, on="endpoint_state", how="left")
    return counts


def availability_table(
    counts: pd.DataFrame,
    endpoint_id: str,
    *,
    sentinels: Mapping[str, str] = SEGMENT_SENTINELS,
    policy: AvailabilityPolicy = AvailabilityPolicy(),
) -> pd.DataFrame:
    """One row per segment: availability in each group and the gap.

    The gap is signed as *positive minus negative*, so a negative number means
    the marker is missing more often in the patients who have the condition --
    the direction that biases a finding rather than merely weakening it.
    """

    columns = [
        "endpoint_id", "segment", "sentinel_marker",
        "n_positive", "n_negative",
        "available_positive_pct", "available_negative_pct",
        "gap_points", "differential",
    ]
    if counts is None or len(counts) == 0:
        return pd.DataFrame(columns=columns)
    indexed = counts.set_index("endpoint_state")
    if not {"positive", "negative"} <= set(indexed.index):
        return pd.DataFrame(columns=columns)
    n_positive = int(indexed.loc["positive", "n_ecgs"])
    n_negative = int(indexed.loc["negative", "n_ecgs"])
    if min(n_positive, n_negative) < int(policy.minimum_group_size):
        return pd.DataFrame(columns=columns)

    records: list[dict] = []
    for segment, marker in sentinels.items():
        column = f"n_{segment}"
        if column not in indexed.columns:
            continue
        positive = 100.0 * float(indexed.loc["positive", column]) / n_positive
        negative = 100.0 * float(indexed.loc["negative", column]) / n_negative
        gap = positive - negative
        records.append(
            {
                "endpoint_id": str(endpoint_id),
                "segment": str(segment),
                "sentinel_marker": str(marker),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "available_positive_pct": positive,
                "available_negative_pct": negative,
                "gap_points": gap,
                "differential": bool(
                    abs(gap) >= float(policy.notable_gap_points)
                ),
            }
        )
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


def worst_differential(table: pd.DataFrame) -> Optional[dict]:
    """The segment whose availability depends most on the endpoint."""

    if table is None or len(table) == 0 or "gap_points" not in table:
        return None
    row = table.loc[table["gap_points"].abs().idxmax()]
    return {
        "endpoint_id": str(row["endpoint_id"]),
        "segment": str(row["segment"]),
        "gap_points": float(row["gap_points"]),
        "available_positive_pct": float(row["available_positive_pct"]),
        "available_negative_pct": float(row["available_negative_pct"]),
    }


def run_segment_availability(
    config,
    *,
    run_id: str,
    endpoint_ids: Optional[Sequence[str]] = None,
    panel_ids: Sequence[str] = STUDY_PANEL_IDS,
    sentinels: Mapping[str, str] = SEGMENT_SENTINELS,
    policy: AvailabilityPolicy = AvailabilityPolicy(),
    feature_glob: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Measure, for every endpoint, whether missingness tracks the diagnosis."""

    import duckdb

    from .label_registry import definitions_by_id
    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        content_hash,
        development_feature_glob,
        feature_parquet_clause,
        scientific_config_hash,
    )

    output_root = config.runs_root / run_id / "availability"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "segment_availability.manifest.json"

    endpoints = tuple(
        str(item) for item in (endpoint_ids or sorted(definitions_by_id()))
    )
    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "endpoints": list(endpoints),
            "sentinels": dict(sentinels),
            "policy": policy.to_dict(),
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash, config_hash, SEGMENT_AVAILABILITY_VERSION
        ):
            return previous

    labels_path = config.labels_root / "clinical_labels_wide.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Die Beurteilbarkeitsstufe braucht die Labels: {labels_path}"
        )

    def sql_path(value) -> str:
        return str(value).replace(chr(92), "/")

    frames: list[pd.DataFrame] = []
    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        clause = feature_parquet_clause(connection, sql_path(resolved_glob))
        available_columns = set(
            connection.execute(f"SELECT * FROM {clause} LIMIT 0").df().columns
        )
        present = {
            segment: marker
            for segment, marker in sentinels.items()
            if marker in available_columns
        }
        if not present:
            raise ValueError(
                "Kein Sentinel-Marker im Merkmalssatz -- ohne einen laesst "
                "sich Beurteilbarkeit nicht messen"
            )
        label_columns = set(
            connection.execute(
                f"SELECT * FROM read_parquet('{sql_path(labels_path)}') LIMIT 0"
            ).df().columns
        )
        usable_endpoints = [
            name for name in endpoints if f"{name}_state" in label_columns
        ]
        if progress:
            progress(
                f"Beurteilbarkeit: {len(usable_endpoints)} Endpunkte, "
                f"{len(present)} Segmente -- ein Durchgang"
            )
        availability = connection.execute(
            availability_query(
                sql_path(config.master_root / "ecg_master_index.parquet"),
                clause,
                sentinels=present,
            )
        ).df()
        state_columns = ", ".join(
            f"{name}_state" for name in usable_endpoints
        )
        states = connection.execute(
            "SELECT subject_id, study_id"
            + (f", {state_columns}" if state_columns else "")
            + f" FROM read_parquet('{sql_path(labels_path)}')"
        ).df()

    if progress:
        progress(
            f"Beurteilbarkeit: {len(availability):,} EKGs gelesen, "
            "Endpunkte auswerten"
        )
    merged = availability.merge(states, on=["subject_id", "study_id"], how="inner")
    for endpoint in usable_endpoints:
        column = f"{endpoint}_state"
        if column not in merged.columns:
            continue
        counts = count_by_state(
            merged[[f"has_{segment}" for segment in present]],
            merged[column],
            sentinels=present,
        )
        block = availability_table(
            counts, endpoint, sentinels=present, policy=policy
        )
        if not block.empty:
            frames.append(block)

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = {
        "version": SEGMENT_AVAILABILITY_VERSION,
        "run_id": str(run_id),
        "policy": policy.to_dict(),
        "sentinels": dict(present),
        "n_endpoints": int(table["endpoint_id"].nunique()) if len(table) else 0,
        "n_differential": int(table["differential"].sum()) if len(table) else 0,
    }
    if len(table):
        worst = worst_differential(table)
        if worst:
            summary["worst_differential"] = worst
            if progress:
                progress(
                    "Beurteilbarkeit: groesster Unterschied "
                    f"{worst['gap_points']:+.1f} Punkte bei "
                    f"{worst['segment']} / {worst['endpoint_id']}"
                )

    table_path = output_root / "segment_availability.parquet"
    summary_path = output_root / "segment_availability_summary.json"
    atomic_write_parquet(table, table_path)
    atomic_write_json(summary_path, summary)

    manifest = StageManifest.complete(
        stage="segment-availability",
        stage_version=SEGMENT_AVAILABILITY_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(str(table_path.resolve()), str(summary_path.resolve())),
        row_counts={
            "rows": int(len(table)),
            "differential": int(summary["n_differential"]),
        },
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "AvailabilityPolicy",
    "MINIMUM_GROUP_SIZE",
    "SEGMENT_AVAILABILITY_VERSION",
    "SEGMENT_SENTINELS",
    "availability_query",
    "count_by_state",
    "availability_table",
    "run_segment_availability",
    "worst_differential",
]
