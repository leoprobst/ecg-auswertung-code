"""Within-patient ECG trajectories: repeated recordings as their own signal.

The cohort holds 800,035 ECGs over 161,352 subjects -- five per person on
average, and 104,008 subjects with at least two. Every analysis so far has
reduced that to one cross-sectional index ECG per person. This module uses the
rest, for two purposes that a cross-section cannot serve:

1. **Measurement error.** Two recordings taken minutes apart with nothing
   happening in between estimate how much of a marker's spread is noise.
   ``repeatability.py`` holds the estimators; this module finds the pairs.

2. **Change as the exposure.** A marker's *trajectory* -- slope, admission-to-
   discharge delta, recovery after an intervention -- is a different and often
   more specific quantity than its level. Each patient acts as their own
   control, which removes every time-invariant confounder at a stroke.

Three traps are handled explicitly rather than assumed away:

* **Informative observation.** Sicker patients get more ECGs, so the number of
  recordings is not ancillary. ``observation_intensity_weights`` models the
  count and returns stabilised weights, and the raw association between count
  and outcome is reported so the informativeness is visible even when the
  weights are not used.
* **Regression to the mean.** A change score correlates with its own baseline
  by construction. Trajectory models therefore carry the baseline as a
  covariate, and the uncorrected version is kept beside it.
* **Repeated measurements.** All inference is GEE with subject clusters and
  robust covariance; a naive per-ECG model would treat five recordings of one
  person as five independent patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from .config import MimicProjectConfig
from .evidence import _icu_event_relations
from .repeatability import ReplicatePairPolicy
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    atomic_replace,
    flush_to_disk,
    feature_parquet_clause,
    optional_column_select,
)
from .storage import atomic_write_json, content_hash, parquet_columns, relation_columns


TRAJECTORY_VERSION = "1.4.0"  # columnar per-block summaries with resumable parts; values unchanged

# A subject needs this many usable recordings before a slope is fitted.
MINIMUM_TRAJECTORY_POINTS = 3
# Minimum span so a slope is not extrapolated from a few minutes.
MINIMUM_TRAJECTORY_SPAN_HOURS = 1.0
# Weight clipping for the observation-intensity model, to stop a handful of
# very frequently recorded patients dominating a weighted analysis.
WEIGHT_PERCENTILE_CLIP = (1.0, 99.0)


@dataclass(frozen=True)
class TrajectoryOptions:
    minimum_points: int = MINIMUM_TRAJECTORY_POINTS
    minimum_span_hours: float = MINIMUM_TRAJECTORY_SPAN_HOURS
    minimum_signal_quality: float = 0.6
    splits: tuple[str, ...] = ("train", "validation")

    def __post_init__(self) -> None:
        if self.minimum_points < 2:
            raise ValueError("minimum_points must be at least two")
        if self.minimum_span_hours <= 0:
            raise ValueError("minimum_span_hours must be positive")
        if set(self.splits) - {"train", "validation"}:
            raise ValueError(
                "Trajectory analysis never reads the locked test split"
            )


# Markers written per part file / held per fetch.
#
# The replicate-pair query is a self-join: DuckDB must hold both build sides,
# each 640,104 rows x 1,753 marker columns x 8 bytes = 9 GB. Measured
# 2026-08-29 with a 12 GB DuckDB limit: resident memory climbed steadily past
# 15.4 GB with 4.8 GB free and the output file still empty after six minutes,
# so the run was stopped before the machine started swapping.
#
# 200 markers per part puts each build side at about 1 GB. The statistics are
# per marker, so which markers share a part cannot change a number.
TRAJECTORY_MARKER_BLOCK = 200


def _marker_blocks(
    marker_ids: Sequence[str], block_size: Optional[int] = None
) -> tuple[tuple[str, ...], ...]:
    """Split the markers into materialisable groups.

    ``block_size`` is resolved at call time rather than captured as a default
    argument: a default is bound once, when the module is defined, so the
    constant could never be adjusted -- not by a test, and not by a caller
    that knows its own memory budget.
    """

    ordered = tuple(marker_ids)
    if not ordered:
        return ()
    step = max(
        1,
        int(TRAJECTORY_MARKER_BLOCK if block_size is None else block_size),
    )
    return tuple(
        ordered[start : start + step] for start in range(0, len(ordered), step)
    )


def replicate_pair_parts(output_path: Path | str) -> tuple[Path, ...]:
    """The part files of a replicate-pair artifact, in written order."""

    root = Path(output_path)
    if root.is_file():
        return (root,)
    return tuple(sorted(root.glob("part-*.parquet")))


# Columns every part carries besides its marker block. Two of them end in the
# marker suffixes, which is why they have to be listed rather than inferred.
_PAIR_KEY_COLUMNS = frozenset(
    {
        "subject_id",
        "split",
        "first_study_id",
        "second_study_id",
        "first_ecg_time",
        "second_ecg_time",
        "hours_between",
        "intervening_intervention_checked",
        "signal_quality_first",
        "signal_quality_second",
    }
)


def _part_is_current(
    part_path: Path,
    block: Sequence[str],
    expected_rows: int,
    *,
    first_suffix: str = "_first",
    second_suffix: str = "_second",
) -> bool:
    """Whether an existing part file is the one this loop would write.

    Two things have to agree, and both are read from the footer alone:

    * the marker columns, so a part left over from a different marker set or
      a different block size is rewritten rather than silently mixed in;
    * the pair count, so a part from a different cohort or policy is rejected.

    Any doubt -- an unreadable footer, a missing library -- answers no. The
    expensive path is the safe one.
    """

    if not part_path.is_file():
        return False
    try:
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(part_path)
        names = set(handle.schema_arrow.names)
        rows = int(handle.metadata.num_rows)
    except Exception:
        return False
    if rows != int(expected_rows):
        return False
    wanted = {f"{marker}{suffix}" for marker in block
              for suffix in (first_suffix, second_suffix)}
    # The pair keys carry the same suffixes -- ``signal_quality_first`` ends
    # in ``_first`` without being a marker -- so they are named and removed
    # rather than pattern-matched away.
    present = {
        name for name in names
        if (name.endswith(first_suffix) or name.endswith(second_suffix))
        and name not in _PAIR_KEY_COLUMNS
    }
    return present == wanted


def build_replicate_pairs(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    marker_ids: tuple[str, ...],
    output_path: Path,
    policy: ReplicatePairPolicy = ReplicatePairPolicy(),
    splits: tuple[str, ...] = ("train", "validation"),
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Find consecutive ECG pairs that plausibly measure the same state.

    Consecutive, within the policy window, inside one admission, both above
    the quality floor, and -- when ICU event data is available -- with no
    procedure and no infusion start recorded between them. The last condition
    is what separates "measured twice" from "measured before and after
    something happened"; without ICU events it cannot be checked and the pair
    set is labelled accordingly.
    """

    if set(splits) - {"train", "validation"}:
        raise ValueError("Replicate pairs never read the locked test split")
    master = config.master_root / "ecg_master_index.parquet"
    if not master.is_file():
        raise FileNotFoundError(master)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    # One clause for both sides of the self-join. This is the widest read
    # in the project -- 2 x 1,753 marker columns against the 1,889-column
    # glob -- and it carried ``union_by_name`` twice. Measured once on a
    # single such read: 19.36 GB resident against an 8 GB DuckDB limit,
    # because the per-file schema state is not what that limit governs.
    # The option is kept only when the partition footers actually differ;
    # see ``storage.feature_parquet_clause``.
    read_features = feature_parquet_clause(sql_path=feature_sql)
    split_sql = ", ".join(f"'{value}'" for value in splits)
    icu_relations = _icu_event_relations(config)
    intervention_checked = bool(
        policy.require_no_intervening_procedure
        and {"procedureevents", "inputevents"} <= set(icu_relations)
    )
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    # One part file per marker block. The self-join makes DuckDB hold
    # both build sides -- each 640,104 rows x 1,753 markers x 8 bytes =
    # 9 GB -- and measured on 2026-08-29 the process passed 15.4 GB with
    # 4.8 GB free and an empty output file. Blocking bounds both sides.
    blocks = _marker_blocks(marker_ids) or ((),)
    if progress:
        progress(
            "Wiederholbarkeit: technische Replikatpaare suchen "
            f"(<= {policy.maximum_hours_between:g} h, gleiche Aufnahme)"
        )
    with catalog.connect(isolated=True) as connection:
        master_columns = relation_columns(
            connection,
            f"read_parquet('{_sql_path(master)}')",
        )
        admission_filter = (
            "AND f.hadm_id IS NOT NULL AND f.hadm_id = s.hadm_id"
            if policy.require_same_admission and "hadm_id" in master_columns
            else ""
        )
        setting_filter = (
            "AND COALESCE(f.ed_stay_id, -1) = COALESCE(s.ed_stay_id, -1)"
            if policy.require_same_care_setting
            and "ed_stay_id" in master_columns
            else ""
        )
        intervention_filter = ""
        if intervention_checked:
            intervention_filter = f"""
              AND NOT EXISTS (
                  SELECT 1 FROM {icu_relations['procedureevents']} p
                  WHERE p.subject_id = f.subject_id
                    AND p.starttime > f.ecg_time
                    AND p.starttime < s.ecg_time
              )
              AND NOT EXISTS (
                  SELECT 1 FROM {icu_relations['inputevents']} v
                  WHERE v.subject_id = f.subject_id
                    AND v.starttime > f.ecg_time
                    AND v.starttime < s.ecg_time
              )
            """
        # The candidate pairs do not depend on the markers: window
        # functions over the master index plus two ICU exclusion
        # sub-selects, identical for every block. Measured on the first
        # blocked attempt, recomputing them per block kept one block
        # busy for four minutes at 8.6 GB with nothing written. Done
        # once here instead.
        candidate_sql = f"""
            WITH usable AS (
                SELECT
                    m.subject_id,
                    m.study_id,
                    m.split,
                    m.ecg_time,
                    {optional_column_select(master_columns, 'hadm_id', 'BIGINT', prefix='m.')},
                    {optional_column_select(master_columns, 'ed_stay_id', 'BIGINT', prefix='m.')}
                FROM read_parquet('{_sql_path(master)}') m
                WHERE m.split IN ({split_sql})
                  AND m.ecg_time IS NOT NULL
            ),
            ordered AS (
                SELECT
                    u.*,
                    LEAD(u.study_id) OVER w AS next_study_id,
                    LEAD(u.ecg_time) OVER w AS next_ecg_time
                FROM usable u
                WINDOW w AS (
                    PARTITION BY u.subject_id
                    ORDER BY u.ecg_time, u.study_id
                )
            ),
            candidate AS (
                SELECT
                    f.subject_id,
                    f.split,
                    f.study_id AS first_study_id,
                    f.next_study_id AS second_study_id,
                    f.ecg_time AS first_ecg_time,
                    f.next_ecg_time AS second_ecg_time,
                    DATE_DIFF('second', f.ecg_time, f.next_ecg_time)
                        / 3600.0 AS hours_between
                FROM ordered f
                JOIN ordered s
                  ON s.subject_id = f.subject_id
                 AND s.study_id = f.next_study_id
                WHERE f.next_study_id IS NOT NULL
                  AND DATE_DIFF('second', f.ecg_time, f.next_ecg_time) / 3600.0
                      BETWEEN {policy.minimum_hours_between}
                          AND {policy.maximum_hours_between}
                  {admission_filter}
                  {setting_filter}
                  {intervention_filter}
            )
            SELECT * FROM candidate
        """
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE vcg_replicate_candidate AS "
            + candidate_sql
        )
        # Audit X8: stored duplicates (identical waveform under two
        # study_ids) would report perfect repeatability. Removed by content
        # hash when the extraction carries one.
        feature_names = parquet_columns(connection, feature_sql)
        duplicate_record = {
            "signal_hash_available": "signal_sha256" in feature_names,
            "identical_signal_pairs_removed": 0,
        }
        if duplicate_record["signal_hash_available"]:
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE vcg_replicate_hash AS
                SELECT subject_id, study_id, signal_sha256
                FROM {read_features}
                WHERE signal_sha256 IS NOT NULL
                """
            )
            before = int(connection.execute(
                "SELECT count(*) FROM vcg_replicate_candidate").fetchone()[0])
            connection.execute(
                """
                CREATE OR REPLACE TEMP TABLE vcg_replicate_candidate AS
                SELECT c.*
                FROM vcg_replicate_candidate c
                LEFT JOIN vcg_replicate_hash a
                  ON a.subject_id = c.subject_id AND a.study_id = c.first_study_id
                LEFT JOIN vcg_replicate_hash b
                  ON b.subject_id = c.subject_id AND b.study_id = c.second_study_id
                WHERE a.signal_sha256 IS NULL
                   OR b.signal_sha256 IS NULL
                   OR a.signal_sha256 <> b.signal_sha256
                """
            )
            after = int(connection.execute(
                "SELECT count(*) FROM vcg_replicate_candidate").fetchone()[0])
            duplicate_record["identical_signal_pairs_removed"] = before - after
        output_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_path / "duplicate_signals.json", duplicate_record)
        if progress and duplicate_record["identical_signal_pairs_removed"]:
            progress(
                "Wiederholbarkeit: "
                f"{duplicate_record['identical_signal_pairs_removed']:,} "
                "gespeicherte Dubletten entfernt"
            )
        # Both sides of every marker-block query need the same ECG rows.  A
        # direct self-join against ``read_parquet`` makes DuckDB bind and scan
        # the complete partition set twice per block.  Keep the small key set
        # once, then project each marker block from Parquet in one pass and
        # self-join that narrow in-memory/spillable table below.
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE vcg_replicate_study_key AS
            SELECT subject_id, first_study_id AS study_id
            FROM vcg_replicate_candidate
            UNION
            SELECT subject_id, second_study_id AS study_id
            FROM vcg_replicate_candidate
            """
        )
        output_path.mkdir(parents=True, exist_ok=True)
        # Parts beyond the current block count belong to a larger marker set
        # and can never be right again.
        for stale in output_path.glob('part-*.parquet'):
            try:
                if int(stale.stem.split('-')[-1]) >= len(blocks):
                    stale.unlink()
            except ValueError:
                stale.unlink()
        # The pairs a part holds: candidates whose two recordings were both
        # extracted and pass the quality floor -- the same joins and filter as
        # the block query below. Counting the raw candidates instead never
        # matched a written part, so after every memory pause all blocks were
        # written again (A153, 04.10.2026: three cycles of 2.7 h each).
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE vcg_replicate_quality AS
            SELECT f.subject_id, f.study_id, f.signal_quality_score
            FROM {read_features} f
            JOIN vcg_replicate_study_key k
              ON k.subject_id = f.subject_id
             AND k.study_id = f.study_id
            WHERE f.extraction_status = 'ok'
            """
        )
        expected_rows = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM vcg_replicate_candidate c
                JOIN vcg_replicate_quality a
                  ON a.subject_id = c.subject_id
                 AND a.study_id = c.first_study_id
                JOIN vcg_replicate_quality b
                  ON b.subject_id = c.subject_id
                 AND b.study_id = c.second_study_id
                WHERE COALESCE(a.signal_quality_score, 0)
                      >= {policy.minimum_signal_quality}
                  AND COALESCE(b.signal_quality_score, 0)
                      >= {policy.minimum_signal_quality}
                """
            ).fetchone()[0]
        )
        connection.execute("DROP TABLE IF EXISTS vcg_replicate_quality")
        for index, block in enumerate(blocks):
            part_path = output_path / f'part-{index:03d}.parquet'
            marker_first = ',\n                '.join(
                f'a."{marker}" AS "{marker}_first"' for marker in block
            ) or 'NULL::DOUBLE AS no_marker_selected_first'
            marker_second = ',\n                '.join(
                f'b."{marker}" AS "{marker}_second"' for marker in block
            ) or 'NULL::DOUBLE AS no_marker_selected_second'
            if _part_is_current(part_path, block, expected_rows):
                # Measured 2026-08-29: this stage wrote five of nine blocks in
                # 4.1 hours and was then paused. Deleting them unconditionally
                # spent those hours again -- and again after the next pause,
                # on a machine where memory pauses are expected. A part that
                # already holds exactly this block's columns for exactly this
                # many pairs is the file this loop would write.
                if progress and len(blocks) > 1:
                    progress(
                        'Wiederholbarkeit: Markerblock '
                        f'{index + 1}/{len(blocks)} liegt bereits vor'
                    )
                continue
            if part_path.exists():
                part_path.unlink()
            if progress and len(blocks) > 1:
                progress(
                    'Wiederholbarkeit: Replikatpaare schreiben '
                    f'(Markerblock {index + 1}/{len(blocks)})'
                )
            feature_markers = ',\n                    '.join(
                f'f."{marker}"' for marker in block
            )
            feature_marker_projection = (
                f",\n                    {feature_markers}"
                if feature_markers
                else ""
            )
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE vcg_replicate_feature_block AS
                SELECT
                    f.subject_id,
                    f.study_id,
                    f.signal_quality_score
                    {feature_marker_projection}
                FROM {read_features} f
                JOIN vcg_replicate_study_key k
                  ON k.subject_id = f.subject_id
                 AND k.study_id = f.study_id
                WHERE f.extraction_status = 'ok'
                """
            )
            query = f"""
                SELECT
                    c.subject_id,
                    c.split,
                    c.first_study_id,
                    c.second_study_id,
                    c.first_ecg_time,
                    c.second_ecg_time,
                    c.hours_between,
                    {'TRUE' if intervention_checked else 'FALSE'}
                        AS intervening_intervention_checked,
                    a.signal_quality_score AS signal_quality_first,
                    b.signal_quality_score AS signal_quality_second,
                    {marker_first},
                    {marker_second}
                FROM vcg_replicate_candidate c
                JOIN vcg_replicate_feature_block a
                  ON a.subject_id = c.subject_id
                 AND a.study_id = c.first_study_id
                JOIN vcg_replicate_feature_block b
                  ON b.subject_id = c.subject_id
                 AND b.study_id = c.second_study_id
                WHERE COALESCE(a.signal_quality_score, 0)
                      >= {policy.minimum_signal_quality}
                  AND COALESCE(b.signal_quality_score, 0)
                      >= {policy.minimum_signal_quality}
                -- Deliberately unordered. Every consumer reads one part at
                -- a time -- ``repeatability_table_from_parquet`` groups the
                -- markers by the part that holds them and never aligns rows
                -- across parts -- so the parts are joined by their pair keys,
                -- which each of them carries, and never by row position.
                --
                -- Sorting cost real time: the sort ran over the full result
                -- width (two columns per marker plus keys), filled DuckDB's
                -- 12 GB buffer pool and spilled. Measured 2026-08-29, the
                -- second block was still running after 19 minutes against 11
                -- for the first, with memory pinned at 14.1 GB.
            """
            _copy_query_atomic(connection, query, part_path)
        connection.execute("DROP TABLE IF EXISTS vcg_replicate_feature_block")
        connection.execute("DROP TABLE IF EXISTS vcg_replicate_study_key")
        connection.execute("DROP TABLE IF EXISTS vcg_replicate_candidate")
    return output_path


def build_trajectory_frame(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    marker_ids: tuple[str, ...],
    output_path: Path,
    options: TrajectoryOptions = TrajectoryOptions(),
    progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """One row per subject and marker: level, slope and span.

    Slopes are ordinary least squares in time, in marker units per day. They
    are descriptive summaries of an irregularly sampled series, not a growth
    model -- the spacing of ECGs is itself informative and is reported
    alongside so that a slope is never read without it.
    """

    master = config.master_root / "ecg_master_index.parquet"
    if not master.is_file():
        raise FileNotFoundError(master)
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    read_features = feature_parquet_clause(sql_path=feature_sql)
    split_sql = ", ".join(f"'{value}'" for value in options.splits)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    # Same reason as the pair query: 640,104 rows x 1,753 markers x 8 bytes is
    # 9 GB before pandas overhead. Each block is summarised column by column
    # (``_summarize_block``: the same per subject and marker definitions as
    # ``summarize_trajectories``, without one Python dict per record), written
    # to its own part file with a content key, and only then merged. A153
    # (04.10.2026) built ~23 million dict records per block -- about 20 GB --
    # and lost every finished block at each memory pause.
    blocks = _marker_blocks(marker_ids) or ((),)
    parts_root = output_path.parent / f".{output_path.name}.parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    for stale in parts_root.glob("part-*"):
        try:
            if int(stale.name.split(".")[0].split("-")[-1]) >= len(blocks):
                stale.unlink()
        except ValueError:
            stale.unlink()
    source_key = _trajectory_source_key(master, feature_glob)
    part_paths: list[Path] = []
    with catalog.connect(isolated=True) as connection:
        for index, block in enumerate(blocks):
            part_path = parts_root / f"part-{index:03d}.parquet"
            part_paths.append(part_path)
            key = content_hash({
                "markers": list(block),
                "options": {
                    "minimum_points": int(options.minimum_points),
                    "minimum_span_hours": float(options.minimum_span_hours),
                    "minimum_signal_quality": float(options.minimum_signal_quality),
                    "splits": list(options.splits),
                },
                "trajectory_version": TRAJECTORY_VERSION,
                "source": source_key,
            })
            suffix = (
                f" (Markerblock {index + 1}/{len(blocks)})"
                if len(blocks) > 1
                else ""
            )
            if _trajectory_part_is_current(part_path, key):
                if progress:
                    progress(f"Verlaufsschicht: Serien je Person liegen bereits vor{suffix}")
                continue
            if progress:
                progress(f"Verlaufsschicht: Serien je Person laden{suffix}")
            marker_columns = ",\n            ".join(
                f'f."{marker}"' for marker in block
            ) or "NULL::DOUBLE AS no_marker_selected"
            query = f"""
                SELECT
                    m.subject_id,
                    m.split,
                    m.study_id,
                    m.ecg_time,
                    f.signal_quality_score,
                    {marker_columns}
                FROM read_parquet('{_sql_path(master)}') m
                JOIN {read_features} f USING (subject_id, study_id)
                WHERE m.split IN ({split_sql})
                  AND m.ecg_time IS NOT NULL
                  AND f.extraction_status = 'ok'
                  AND COALESCE(f.signal_quality_score, 0)
                      >= {options.minimum_signal_quality}
                ORDER BY m.subject_id, m.ecg_time
            """
            series = connection.execute(query).fetchdf()
            rows = _write_trajectory_part(series, block, options, part_path)
            del series
            atomic_write_json(
                part_path.with_suffix(".json"),
                {"key": key, "rows": int(rows), "markers": list(block)},
            )
        if progress and len(blocks) > 1:
            progress("Verlaufsschicht: Teilblöcke nach Person und Marker zusammenführen")
        # The single-frame order -- subject, then marker in the requested
        # order -- restored by DuckDB's external sort, so the merged file does
        # not depend on the block size and never has to fit in memory.
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE vcg_trajectory_marker_rank "
            "(marker_id VARCHAR, marker_rank BIGINT)"
        )
        if marker_ids:
            connection.executemany(
                "INSERT INTO vcg_trajectory_marker_rank VALUES (?, ?)",
                [(str(name), rank) for rank, name in enumerate(marker_ids)],
            )
        sources = ", ".join(f"'{_sql_path(path)}'" for path in part_paths)
        columns = ", ".join(f't."{name}"' for name in _TRAJECTORY_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_query_atomic(
            connection,
            f"""
            SELECT {columns}
            FROM read_parquet([{sources}]) t
            LEFT JOIN vcg_trajectory_marker_rank r ON r.marker_id = t.marker_id
            ORDER BY t.subject_id, r.marker_rank
            """,
            output_path,
        )
        connection.execute("DROP TABLE IF EXISTS vcg_trajectory_marker_rank")
    for path in parts_root.glob("part-*"):
        path.unlink()
    try:
        parts_root.rmdir()
    except OSError:
        pass
    return output_path


_TRAJECTORY_ARROW_TYPES = None


def _trajectory_schema():
    import pyarrow as pa

    global _TRAJECTORY_ARROW_TYPES
    if _TRAJECTORY_ARROW_TYPES is None:
        strings = {"split", "marker_id", "slope_status", "status", "trajectory_version"}
        integers = {"subject_id", "n_ecgs", "n_points"}
        _TRAJECTORY_ARROW_TYPES = pa.schema([
            (name, pa.string() if name in strings else pa.int64() if name in integers else pa.float64())
            for name in _TRAJECTORY_COLUMNS
        ])
    return _TRAJECTORY_ARROW_TYPES


def _trajectory_source_key(master: Path, feature_glob: str) -> dict:
    """What a finished part was computed from: index, features, code version."""

    from .storage import feature_manifest_path, file_sha256

    manifest = feature_manifest_path(feature_glob)
    return {
        "master_index_sha256": file_sha256(master),
        "feature_glob": str(feature_glob),
        "feature_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
        "implementation": "trajectory-block-columnar-1",
    }


def _trajectory_part_is_current(part_path: Path, key: str) -> bool:
    import json

    stamp = part_path.with_suffix(".json")
    if not part_path.is_file() or not stamp.is_file():
        return False
    try:
        import pyarrow.parquet as pq

        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        rows = int(pq.ParquetFile(part_path).metadata.num_rows)
    except Exception:
        return False
    return recorded.get("key") == key and int(recorded.get("rows", -1)) == rows


def _write_trajectory_part(
    series: pd.DataFrame,
    block: Sequence[str],
    options: TrajectoryOptions,
    part_path: Path,
    *,
    markers_per_batch: int = 16,
) -> int:
    import pyarrow.parquet as pq

    schema = _trajectory_schema()
    temporary = part_path.parent / f".{part_path.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    written = 0
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for start in range(0, max(len(block), 1), markers_per_batch):
            chunk = tuple(block[start:start + markers_per_batch])
            if not chunk:
                break
            table = _summarize_block(series, chunk, options)
            if table is None or table.num_rows == 0:
                continue
            writer.write_table(table.cast(schema))
            written += table.num_rows
    flush_to_disk(temporary)
    atomic_replace(temporary, part_path)
    flush_to_disk(part_path)
    return written


def _summarize_block(series: pd.DataFrame, marker_ids: Sequence[str], options: TrajectoryOptions):
    """Columnar ``summarize_trajectories``: same records, same definitions.

    Per subject: n_ecgs, span_hours, split and the day offsets are taken from
    all of the subject's recordings, exactly as the reference does. Per marker
    only the finite values count. Mean, standard deviation and slope are the
    closed forms of what the reference computes with ``np.mean``, ``np.std``
    and ``np.polyfit``; they agree to floating-point rounding.
    """

    import pyarrow as pa

    required = {"subject_id", "ecg_time"}
    missing = sorted(required - set(series.columns))
    if missing:
        raise ValueError(f"Trajectory series is missing columns: {missing}")
    if series.empty:
        return None
    working = series.copy()
    working["ecg_time"] = pd.to_datetime(working["ecg_time"], errors="coerce")
    working = working.dropna(subset=["ecg_time"])
    working = working.sort_values(["subject_id", "ecg_time"], kind="mergesort")
    if working.empty:
        return None
    subjects = working["subject_id"].to_numpy()
    change = np.r_[True, subjects[1:] != subjects[:-1]]
    starts = np.flatnonzero(change)
    counts = np.diff(np.r_[starts, len(subjects)])
    subject_ids = subjects[starts]
    group_index = np.repeat(np.arange(len(starts)), counts)
    times = working["ecg_time"].to_numpy()
    first_time = times[starts]
    last_time = times[starts + counts - 1]
    span_hours = ((last_time - first_time) / np.timedelta64(1, "h")).astype(float)
    first_per_row = pd.Series(first_time[group_index], index=working.index)
    days = (working["ecg_time"] - first_per_row).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    if "split" in working:
        split = working["split"].iloc[starts].astype(str).to_numpy()
    else:
        split = np.full(len(starts), "", dtype=object)
    n_groups = len(starts)
    position = np.arange(len(subjects))
    tables = []
    for marker_id in marker_ids:
        if marker_id not in working:
            continue
        values = pd.to_numeric(working[marker_id], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(values)
        n_points = np.add.reduceat(usable.astype(np.int64), starts)
        has = n_points > 0
        first_idx = np.minimum.reduceat(np.where(usable, position, len(position)), starts)
        last_idx = np.maximum.reduceat(np.where(usable, position, -1), starts)
        baseline = np.full(n_groups, np.nan)
        last = np.full(n_groups, np.nan)
        baseline[has] = values[first_idx[has]]
        last[has] = values[last_idx[has]]
        safe_n = np.where(has, n_points, 1)
        total = np.add.reduceat(np.where(usable, values, 0.0), starts)
        mean = np.where(has, total / safe_n, np.nan)
        maximum = np.where(has, np.maximum.reduceat(np.where(usable, values, -np.inf), starts), np.nan)
        minimum = np.where(has, np.minimum.reduceat(np.where(usable, values, np.inf), starts), np.nan)
        centred = np.where(usable, values - mean[group_index], 0.0)
        two_or_more = n_points >= 2
        sumsq = np.add.reduceat(centred * centred, starts)
        sd = np.where(two_or_more, np.sqrt(sumsq / np.where(two_or_more, n_points - 1, 1)), np.nan)
        # Median spacing of the usable recordings, per subject.
        usable_rows = np.flatnonzero(usable)
        usable_group = group_index[usable_rows]
        usable_days = days[usable_rows]
        same = usable_group[1:] == usable_group[:-1]
        gaps = np.diff(usable_days)[same]
        gap_group = usable_group[1:][same]
        median_interval = np.full(n_groups, np.nan)
        if len(gaps):
            medians = pd.Series(gaps).groupby(gap_group).median()
            median_interval[medians.index.to_numpy()] = medians.to_numpy() * 24.0
        # Least squares slope over the usable recordings.
        day_mean = np.add.reduceat(np.where(usable, days, 0.0), starts) / safe_n
        dx = np.where(usable, days - day_mean[group_index], 0.0)
        dy = np.where(usable, values - mean[group_index], 0.0)
        sxx = np.add.reduceat(dx * dx, starts)
        sxy = np.add.reduceat(dx * dy, starts)
        day_max = np.maximum.reduceat(np.where(usable, days, -np.inf), starts)
        day_min = np.minimum.reduceat(np.where(usable, days, np.inf), starts)
        spread = np.where(has, day_max - day_min, 0.0)
        fitted = (n_points >= int(options.minimum_points)) & (
            span_hours >= float(options.minimum_span_hours)) & (spread > 0)
        short = ~fitted & two_or_more & (span_hours < float(options.minimum_span_hours))
        slope = np.where(fitted, sxy / np.where(sxx > 0, sxx, 1.0), np.nan)
        slope_status = np.where(fitted, "fitted", np.where(short, "span_too_short", "insufficient_points"))
        status = np.where(has, "evaluable", "no_measurable_value")
        tables.append(pa.table({
            "subject_id": pa.array(subject_ids.astype(np.int64)),
            "split": pa.array(split.astype(str)),
            "marker_id": pa.array(np.full(n_groups, str(marker_id))),
            "n_ecgs": pa.array(counts.astype(np.int64)),
            "n_points": pa.array(n_points.astype(np.int64)),
            "span_hours": pa.array(span_hours),
            "baseline_value": pa.array(baseline),
            "last_value": pa.array(last),
            "delta": pa.array(last - baseline),
            "mean_value": pa.array(mean),
            "max_value": pa.array(maximum),
            "min_value": pa.array(minimum),
            "within_subject_sd": pa.array(sd),
            "median_interval_hours": pa.array(median_interval),
            "slope_per_day": pa.array(slope),
            "slope_status": pa.array(slope_status.astype(str)),
            "status": pa.array(status.astype(str)),
            "trajectory_version": pa.array(np.full(n_groups, TRAJECTORY_VERSION)),
        }))
    if not tables:
        return None
    return pa.concat_tables(tables)


def summarize_trajectories(
    series: pd.DataFrame,
    *,
    marker_ids: Iterable[str],
    options: TrajectoryOptions = TrajectoryOptions(),
) -> pd.DataFrame:
    """Per subject and marker: baseline, last value, delta, slope, span."""

    required = {"subject_id", "ecg_time"}
    missing = sorted(required - set(series.columns))
    if missing:
        raise ValueError(f"Trajectory series is missing columns: {missing}")
    if series.empty:
        return pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
    working = series.copy()
    working["ecg_time"] = pd.to_datetime(working["ecg_time"], errors="coerce")
    working = working.dropna(subset=["ecg_time"])
    working = working.sort_values(
        ["subject_id", "ecg_time"],
        kind="mergesort",
    )
    records: list[dict[str, object]] = []
    for subject_id, group in working.groupby("subject_id", sort=True):
        times = group["ecg_time"].to_numpy()
        span_hours = (
            float((times[-1] - times[0]) / np.timedelta64(1, "h"))
            if len(times) >= 2
            else 0.0
        )
        days = (
            (group["ecg_time"] - group["ecg_time"].iloc[0])
            .dt.total_seconds()
            .to_numpy(dtype=float)
            / 86400.0
        )
        split = str(group["split"].iloc[0]) if "split" in group else ""
        for marker_id in marker_ids:
            if marker_id not in group:
                continue
            values = pd.to_numeric(
                group[marker_id],
                errors="coerce",
            ).to_numpy(dtype=float)
            usable = np.isfinite(values)
            n_points = int(np.sum(usable))
            record: dict[str, object] = {
                "subject_id": subject_id,
                "split": split,
                "marker_id": str(marker_id),
                "n_ecgs": int(len(values)),
                "n_points": n_points,
                "span_hours": span_hours,
                "trajectory_version": TRAJECTORY_VERSION,
            }
            if n_points == 0:
                record.update(_EMPTY_TRAJECTORY_VALUES)
                record["status"] = "no_measurable_value"
                records.append(record)
                continue
            usable_values = values[usable]
            usable_days = days[usable]
            baseline = float(usable_values[0])
            last = float(usable_values[-1])
            record.update(
                {
                    "baseline_value": baseline,
                    "last_value": last,
                    "delta": last - baseline,
                    "mean_value": float(np.mean(usable_values)),
                    "max_value": float(np.max(usable_values)),
                    "min_value": float(np.min(usable_values)),
                    "within_subject_sd": (
                        float(np.std(usable_values, ddof=1))
                        if n_points >= 2
                        else float("nan")
                    ),
                    "median_interval_hours": (
                        float(np.median(np.diff(usable_days)) * 24.0)
                        if n_points >= 2
                        else float("nan")
                    ),
                }
            )
            slope = float("nan")
            slope_status = "insufficient_points"
            if (
                n_points >= int(options.minimum_points)
                and span_hours >= float(options.minimum_span_hours)
                and np.ptp(usable_days) > 0
            ):
                slope = float(np.polyfit(usable_days, usable_values, 1)[0])
                slope_status = "fitted"
            elif n_points >= 2 and span_hours < float(options.minimum_span_hours):
                slope_status = "span_too_short"
            record["slope_per_day"] = slope
            record["slope_status"] = slope_status
            record["status"] = "evaluable"
            records.append(record)
    if not records:
        return pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
    return pd.DataFrame(records)


_TRAJECTORY_COLUMNS = (
    "subject_id",
    "split",
    "marker_id",
    "n_ecgs",
    "n_points",
    "span_hours",
    "baseline_value",
    "last_value",
    "delta",
    "mean_value",
    "max_value",
    "min_value",
    "within_subject_sd",
    "median_interval_hours",
    "slope_per_day",
    "slope_status",
    "status",
    "trajectory_version",
)

_EMPTY_TRAJECTORY_VALUES = {
    "baseline_value": float("nan"),
    "last_value": float("nan"),
    "delta": float("nan"),
    "mean_value": float("nan"),
    "max_value": float("nan"),
    "min_value": float("nan"),
    "within_subject_sd": float("nan"),
    "median_interval_hours": float("nan"),
    "slope_per_day": float("nan"),
    "slope_status": "insufficient_points",
}


def observation_intensity_weights(
    counts: Iterable[float],
    covariates: Optional[pd.DataFrame] = None,
    *,
    clip_percentiles: tuple[float, float] = WEIGHT_PERCENTILE_CLIP,
) -> dict[str, object]:
    """Stabilised weights for informative ECG frequency.

    Being recorded often is a marker of being unwell, so subjects contribute
    unequally *and* non-randomly to any analysis that pools recordings. The
    weight is the marginal probability of the observed count divided by the
    probability conditional on the covariates, i.e. a stabilised
    inverse-probability-of-observation weight; without covariates it reduces
    to the inverse relative intensity, which at least removes the mechanical
    over-representation of frequently recorded patients.
    """

    count_array = np.asarray(list(counts), dtype=float)
    valid = np.isfinite(count_array) & (count_array > 0)
    weights = np.full(len(count_array), np.nan, dtype=float)
    if not valid.any():
        return {
            "weights": weights,
            "status": "not_evaluable",
            "method": "none",
            "effective_sample_size": 0.0,
        }
    intensity = count_array[valid]
    marginal = float(np.mean(intensity))
    method = "inverse_relative_intensity"
    # Without covariates the expected count *is* the subject's own count, so
    # the weight becomes mean/count: a patient recorded ten times contributes
    # the same total as one recorded twice. Using the marginal mean on both
    # sides would return all-ones and silently do nothing.
    expected = intensity.astype(float)
    if covariates is not None and len(covariates) == len(count_array):
        design = covariates.loc[valid].apply(
            pd.to_numeric,
            errors="coerce",
        )
        design = design.loc[:, design.notna().any(axis=0)]
        complete = design.notna().all(axis=1).to_numpy()
        if design.shape[1] and complete.sum() > design.shape[1] + 10:
            matrix = np.column_stack(
                (np.ones(int(complete.sum())), design.loc[complete].to_numpy())
            )
            observed = intensity[complete]
            coefficients, *_ = np.linalg.lstsq(matrix, observed, rcond=None)
            fitted = matrix @ coefficients
            expected = np.full(len(intensity), marginal, dtype=float)
            expected[complete] = np.clip(fitted, 0.5, None)
            method = "stabilised_linear_intensity_model"
    raw = marginal / expected
    low, high = np.percentile(raw, clip_percentiles)
    clipped = np.clip(raw, low, high)
    weights[valid] = clipped
    total = float(np.sum(clipped))
    effective = (
        float(total ** 2 / np.sum(clipped ** 2))
        if np.sum(clipped ** 2) > 0
        else 0.0
    )
    return {
        "weights": weights,
        "status": "evaluable",
        "method": method,
        "mean_count": marginal,
        "weight_min": float(np.min(clipped)),
        "weight_max": float(np.max(clipped)),
        "effective_sample_size": effective,
        "clip_percentiles": list(clip_percentiles),
        "interpretation": (
            "removes mechanical over-representation of frequently recorded "
            "patients; does not identify a causal effect of recording"
        ),
    }


def observation_intensity_diagnostics(
    counts: Iterable[float],
    outcome: Iterable[float],
) -> dict[str, object]:
    """How strongly the number of recordings itself predicts the outcome.

    Reported whether or not weights are applied: if the count is strongly
    associated with the outcome, any pooled repeated-measures result carries
    that association whether the analyst wanted it or not.
    """

    count_array = np.asarray(list(counts), dtype=float)
    outcome_array = np.asarray(list(outcome), dtype=float)
    valid = (
        np.isfinite(count_array)
        & np.isfinite(outcome_array)
        & np.isin(outcome_array, (0.0, 1.0))
    )
    if valid.sum() < 10 or np.unique(outcome_array[valid]).size < 2:
        return {
            "status": "not_evaluable",
            "mean_count_positive": float("nan"),
            "mean_count_negative": float("nan"),
            "count_outcome_auc": float("nan"),
        }
    counts_valid = count_array[valid]
    labels = outcome_array[valid].astype(int)
    positive = counts_valid[labels == 1]
    negative = counts_valid[labels == 0]
    order = np.argsort(counts_valid, kind="mergesort")
    ranks = np.empty(len(counts_valid), dtype=float)
    sorted_values = counts_valid[order]
    ranks[order] = np.arange(1, len(counts_valid) + 1, dtype=float)
    # Average ranks for ties so the AUC is exact under ties.
    unique_values, inverse, tie_counts = np.unique(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    cumulative = np.cumsum(tie_counts)
    starts = cumulative - tie_counts
    mean_ranks = (starts + cumulative + 1.0) / 2.0
    ranks[order] = mean_ranks[inverse]
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )
    return {
        "status": "evaluable",
        "mean_count_positive": float(np.mean(positive)),
        "mean_count_negative": float(np.mean(negative)),
        "count_outcome_auc": float(auc),
        "informative_observation_suspected": bool(abs(auc - 0.5) >= 0.05),
        "interpretation": (
            "AUC of the recording count alone against the outcome; far from "
            "0.5 means observation frequency is not ancillary"
        ),
    }


def fit_trajectory_association(
    frame: pd.DataFrame,
    *,
    marker_column: str,
    outcome_column: str,
    subject_column: str = "subject_id",
    covariate_columns: tuple[str, ...] = (),
    baseline_column: Optional[str] = None,
    binary_outcome: bool = True,
    weights: Optional[Iterable[float]] = None,
) -> dict[str, object]:
    """GEE for a repeated-measures or change-score association.

    Subject-clustered with robust covariance, because the unit of independence
    is the patient and not the recording. When ``baseline_column`` is given it
    enters the model: a change score is mechanically correlated with its own
    starting point, and leaving that uncontrolled produces regression to the
    mean dressed up as an effect.
    """

    columns = [subject_column, marker_column, outcome_column]
    columns.extend(covariate_columns)
    if baseline_column:
        columns.append(baseline_column)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Trajectory model is missing columns: {missing}")
    design = frame.loc[:, list(dict.fromkeys(columns))].copy()
    for column in design.columns:
        if column != subject_column:
            design[column] = pd.to_numeric(design[column], errors="coerce")
    weight_values = (
        pd.to_numeric(pd.Series(list(weights)), errors="coerce")
        if weights is not None
        else None
    )
    if weight_values is not None:
        design["__weight"] = weight_values.to_numpy()
    design = design.dropna()
    base: dict[str, object] = {
        "marker_column": marker_column,
        "outcome_column": outcome_column,
        "baseline_adjusted": bool(baseline_column),
        "n_observations": int(len(design)),
        "n_subjects": int(design[subject_column].nunique()) if len(design) else 0,
        "inference_method": "gee_exchangeable_subject_clusters_robust_se",
        "trajectory_version": TRAJECTORY_VERSION,
        "status": "not_evaluable",
        "coefficient": float("nan"),
        "std_error": float("nan"),
        "p_value": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "effect_scale": "log_odds_per_marker_unit" if binary_outcome else "outcome_units_per_marker_unit",
    }
    if len(design) < 20 or base["n_subjects"] < 10:
        base["failure_reason"] = "insufficient_subjects_or_observations"
        return base
    outcome = design[outcome_column].to_numpy(dtype=float)
    if binary_outcome:
        if not np.isin(outcome, (0.0, 1.0)).all():
            base["failure_reason"] = "binary_outcome_is_not_zero_one"
            return base
        if np.unique(outcome).size < 2:
            base["failure_reason"] = "outcome_has_a_single_class"
            return base
    predictors = [marker_column]
    if baseline_column:
        predictors.append(baseline_column)
    predictors.extend(covariate_columns)
    predictors = list(dict.fromkeys(predictors))
    matrix = design.loc[:, predictors].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        base["failure_reason"] = "non_finite_design"
        return base
    constant = [
        name
        for name, column in zip(predictors, matrix.T)
        if np.unique(column).size < 2
    ]
    if marker_column in constant:
        base["failure_reason"] = "marker_is_constant"
        return base
    keep = [name not in constant for name in predictors]
    predictors = [name for name, flag in zip(predictors, keep) if flag]
    matrix = matrix[:, np.asarray(keep, dtype=bool)]
    exog = np.column_stack((np.ones(len(matrix)), matrix))
    groups = design[subject_column].to_numpy()
    family = Binomial() if binary_outcome else Gaussian()
    repeated = pd.Series(groups).duplicated().any()
    structure = Exchangeable() if repeated else Independence()
    try:
        model = GEE(
            outcome,
            exog,
            groups=groups,
            family=family,
            cov_struct=structure,
            weights=(
                design["__weight"].to_numpy(dtype=float)
                if "__weight" in design
                else None
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit()
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        base["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return base
    if any("separation" in str(item.message).lower() for item in caught):
        base["failure_reason"] = "separation_or_unstable_estimate"
        base["separation_suspected"] = True
        return base
    coefficient = float(result.params[1])
    std_error = float(result.bse[1])
    confidence = result.conf_int()
    # Separation and near-separation produce enormous coefficients with
    # meaningless standard errors. Reporting an odds ratio of 1e8 as a finding
    # would be worse than reporting nothing.
    if (
        not np.isfinite(coefficient)
        or not np.isfinite(std_error)
        or abs(coefficient) > 20.0
        or std_error > 50.0
    ):
        base["failure_reason"] = "separation_or_unstable_estimate"
        base["status"] = "not_evaluable"
        base["separation_suspected"] = True
        return base
    base.update(
        {
            "status": "evaluable",
            "failure_reason": "",
            "separation_suspected": False,
            "coefficient": coefficient,
            "std_error": std_error,
            "p_value": float(result.pvalues[1]),
            "ci_low": float(confidence[1][0]),
            "ci_high": float(confidence[1][1]),
            "predictors": "|".join(predictors),
            "correlation_structure": (
                "exchangeable" if repeated else "independence_single_row_per_subject"
            ),
            "weighted": "__weight" in design,
        }
    )
    if binary_outcome:
        base["odds_ratio"] = float(np.exp(np.clip(coefficient, -700, 700)))
        base["odds_ratio_ci_low"] = float(
            np.exp(np.clip(base["ci_low"], -700, 700))
        )
        base["odds_ratio_ci_high"] = float(
            np.exp(np.clip(base["ci_high"], -700, 700))
        )
    return base


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _copy_query_atomic(connection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    with profile_query(f"trajectory:{destination.name}") as profile:
        copy_result = connection.execute(
            f"COPY ({query}) TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        ).fetchone()
        if copy_result is None:
            raise RuntimeError("DuckDB COPY returned no row count")
        flush_to_disk(temporary)
        atomic_replace(temporary, destination)
        flush_to_disk(destination)
        profile["rows"] = int(copy_result[0])


__all__ = [
    "MINIMUM_TRAJECTORY_POINTS",
    "TRAJECTORY_VERSION",
    "TrajectoryOptions",
    "build_replicate_pairs",
    "build_trajectory_frame",
    "fit_trajectory_association",
    "observation_intensity_diagnostics",
    "observation_intensity_weights",
    "summarize_trajectories",
]
