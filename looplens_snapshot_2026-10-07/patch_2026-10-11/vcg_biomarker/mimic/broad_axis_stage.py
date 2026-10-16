"""Pipeline stage: screen every marker against every frequent database entity.

The project already screens diagnoses exhaustively -- 2,885 PheCodeX
phenotypes across 18 body systems. The other side, everything the hospital
*measured* rather than *coded*, was curated: 113 hand-written associations.
Measured against what the database actually holds:

| Achse | Ziele ab 200 Patienten | bisher gescreent |
|---|---:|---:|
| Laborparameter | 460 | 26 |
| Prozedurencodes | 590 | 6 |
| ICU-Messgroessen | 342 | 6 |
| Erreger | 81 | 1 |

This stage removes that asymmetry. Frequency is the only inclusion rule, it is
fixed before the screen, and an entity below the floor is reported as
underpowered rather than dropped.

**Why the ICU axis is the scientifically interesting one.** It carries the
invasively measured quantities, for patients who also have an ECG: central
venous pressure (14,572 patients, 809,472 measurements), pulmonary artery
pressure (~6,200 patients, 938,576 measurements), cardiac output (5,784),
SvO2 (3,427), PCWP (820). Everywhere else in this project an ECG marker is
compared against a *code* or an *echo report*. Here it is compared against a
number obtained with a catheter -- which turns "is this associated" into "is
this a non-invasive surrogate", and that is the question with clinical
consequences.

**Multiplicity.** Each axis is its own FDR family, matching how the existing
context screen is corrected (``family_columns=("domain", "panel_id")``). A new
axis therefore does not raise the bar for the phenotype screen or for the
curated associations; it carries its own. The global sensitivity column spans
everything, as before.

Train screens, validation replicates, and the sealed test is not read.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .broad_event_screen import (
    AXIS_CCSR,
    AXIS_DRG,
    RELATION_UNDATED,
    AXIS_ICU_MEASUREMENT,
    AXIS_LABORATORY,
    AXIS_MICROBIOLOGY,
    AXIS_PROCEDURE,
    BROAD_EVENT_SCREEN_VERSION,
    BroadAxisPolicy,
    DEFAULT_POLICIES,
    assert_policies_prespecified,
    axis_coverage,
    binary_axis_values_sql,
    ccsr_target_counts_sql,
    drg_target_counts_sql,
    combined_coverage_report,
    continuous_axis_values_sql,
    enumerate_targets,
    icu_measurement_target_counts_sql,
    laboratory_target_counts_sql,
    microbiology_target_counts_sql,
    policy_for,
    procedure_target_counts_sql,
    screen_family_size,
)
from .broad_axis_adjustment import (
    BROAD_AXIS_ADJUSTMENT_VERSION,
    PRIMARY_COVARIATES,
    SENSITIVITY_COVARIATES,
    add_adjustment_effect_change,
    add_adjusted_fdr,
    fit_adjusted_association,
)
from .config import MimicProjectConfig
from .feature_roles import feature_profile
from .panels import default_feature_panels
from .statistics import (
    apply_fdr_by_family,
    fast_mann_whitney_auc,
    inference_status,
    prepare_sorted_control_distribution,
)
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    development_feature_glob,
    feature_parquet_clause,
    operational_config_hash,
    scientific_config_hash,
)


BROAD_AXIS_STAGE_VERSION = "1.5.0"  # hemolysis-falsified lab results excluded (D2)

DEVELOPMENT_SPLITS: tuple[str, ...] = ("train", "validation")

# Prespecified floors. Asserted against the policies at run time so that a
# later edit to either side is a failure rather than a silent redefinition of
# what "broad" meant.
PRESPECIFIED_FLOORS: Mapping[str, int] = {
    AXIS_CCSR: 100,
    AXIS_LABORATORY: 200,
    AXIS_PROCEDURE: 100,
    AXIS_MICROBIOLOGY: 100,
    AXIS_ICU_MEASUREMENT: 200,
    AXIS_DRG: 100,
}

# Screening alpha on train, replication alpha on validation. Deliberately
# stricter than the curated screen: this family is larger and entirely
# hypothesis-free.
TRAIN_FDR_ALPHA = 0.01
VALIDATION_FDR_ALPHA = 0.05

# Prespecified minimum effect. Fixed here, before any screen has run, because
# a floor chosen after seeing the results is not a floor.
#
# The laboratory axis alone contributes 13,232,238 usable rows. At that size a
# Spearman rho of 0.001 is significant at any alpha anybody would write down,
# so "significant" would measure the cohort size and nothing else. The floor
# turns the question back into "is this large enough to matter".
#
# |rho| >= 0.10 for a continuous target: below that, under 1 % of the variance
# is shared, and the measurement error of a single ECG marker is of the same
# order. AUC >= 0.55 for a binary target is the same statement -- rank
# agreement 5 points above chance -- and both are far below what a diagnostic
# claim needs, which is the point: this is a floor for reporting a hit, not a
# threshold for believing one.
MINIMUM_ABSOLUTE_RHO = 0.10
MINIMUM_ABSOLUTE_AUC_MARGIN = 0.05

MINIMUM_CASES = 30
MINIMUM_MARKER_COVERAGE = 0.50


def _effect_floor(target_kind: object) -> float:
    return (
        MINIMUM_ABSOLUTE_AUC_MARGIN
        if str(target_kind) == "binary"
        else MINIMUM_ABSOLUTE_RHO
    )


def meets_effect_floor(effect: object, target_kind: object) -> bool:
    """Whether one screening effect clears its prespecified floor."""

    try:
        value = float(effect)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value):
        return False
    # The tolerance is not slack, it is the declared boundary. An AUC of 0.45
    # is exactly the floor away from 0.5, but ``0.5 - 0.45`` evaluates to
    # 0.04999999999999999, so a bare ``>=`` would exclude the very value the
    # threshold names. A prespecified threshold must not depend on which side
    # of it a float lands.
    return (
        abs(value - _neutral(target_kind))
        >= _effect_floor(target_kind) - 1e-12
    )


# Execution order. ``DEFAULT_POLICIES`` is part of the preregistration and is
# left exactly as it stands; this only decides which axis is computed first,
# which changes no result -- the reuse key sorts the axis names before hashing.
#
# The ICU axis leads because it is the only one whose reference is a *number
# obtained with a catheter*: central venous pressure (14,572 patients),
# pulmonary artery pressure (~6,200), cardiac output (5,784), SvO2 (3,427),
# PCWP (820). Everywhere else a marker is compared against a billing code or a
# report someone typed. That turns "is this associated" into "is this a
# non-invasive surrogate", which is the only question here with an immediate
# clinical consequence. It is also the cheaper axis: 360 targets above the
# floor against 727 in the laboratory axis.
#
# Anything not named runs afterwards, in the registry's own order.
BROAD_AXIS_PRIORITY: tuple[str, ...] = (
    AXIS_ICU_MEASUREMENT,
    AXIS_LABORATORY,
)


def axis_priority(axis: object) -> tuple[int, str]:
    name = str(axis)
    if name in BROAD_AXIS_PRIORITY:
        return (BROAD_AXIS_PRIORITY.index(name), name)
    return (len(BROAD_AXIS_PRIORITY), name)


def _sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _parquet(path: Path | str) -> str:
    return f"read_parquet('{_sql_path(path)}')"


def _feature_parquet(glob: str) -> str:
    """Read the feature partitions tolerantly.

    Feature partitions written at different times do not share a column set --
    a marker added later exists only in newer parts. Without union_by_name
    DuckDB binds the first file's schema and then fails on the next one, which
    surfaces as a missing ``study_id`` rather than as a schema mismatch.
    """

    # Measured on the real data: the option is what makes a wide feature read
    # cost 19.36 GB resident against an 8 GB DuckDB limit. It is still the
    # right answer when the partitions really do differ, so the choice is made
    # from their footers rather than assumed either way.
    return feature_parquet_clause(sql_path=_sql_path(glob))


def _csv(path: Path | str) -> str:
    return f"read_csv_auto('{_sql_path(path)}')"


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sensitivity_adjustment_allowed(target_id: object, label: object) -> bool:
    """Do not adjust away a rhythm/conduction target with its own mediators."""

    text = f"{target_id} {label}".lower()
    return not any(
        token in text
        for token in (
            "arrhythm",
            "rhythm",
            "fibrillation",
            "flutter",
            "tachy",
            "brady",
            "bundle",
            "conduction",
            "qrs",
            "atrioventric",
            "heart block",
        )
    )


def _adjustment_select(marker_id: str) -> str:
    marker = marker_id.replace('"', '""')
    return f"""
        SELECT
            {_adjustment_common_select()},
            f."{marker}" AS marker_value
    """


def _adjustment_common_select() -> str:
    return """
            f.study_id,
            m.subject_id,
            m.ecg_time,
            m.split AS evaluation_split,
            m.age,
            m.sex,
            COALESCE(m.machine_heart_rate_bpm, m.triage_heart_rate)
                AS heart_rate_bpm,
            f.signal_quality_score,
            CASE
                WHEN m.icustay_id IS NOT NULL THEN 'ICU'
                WHEN m.ed_stay_id IS NOT NULL THEN 'ED'
                WHEN m.hadm_id IS NOT NULL THEN 'inpatient'
                ELSE 'unlinked'
            END AS care_setting,
            f.rhythm_context,
            COALESCE(f.qrs_duration_ms, m.machine_qrs_duration_ms)
                AS qrs_duration_ms
    """.strip()


ADJUSTMENT_MARKER_BLOCK_SIZE = 64
BROAD_ADJUSTMENT_PACKET_NAMESPACE = 'numpy-design-v1'


def _adjustment_marker_projection(
    marker_ids: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """Project a marker block once and return collision-proof aliases."""

    aliases = {
        marker_id: f"__vcg_adjustment_marker_{index:04d}"
        for index, marker_id in enumerate(marker_ids)
    }
    projection = ",\n            ".join(
        f'f."{marker_id.replace(chr(34), chr(34) * 2)}" AS "{alias}"'
        for marker_id, alias in aliases.items()
    )
    return projection, aliases


def _fit_adjusted_candidate(
    frame: pd.DataFrame,
    *,
    axis: str,
    target_id: str,
    target_label: str,
    temporal_relation: str,
    marker_id: str,
    target_kind: str,
    prepared: bool = False,
    reusable: Optional[Mapping[tuple, dict]] = None,
) -> list[dict]:
    records: list[dict] = []
    for split in DEVELOPMENT_SPLITS:
        scoped = frame.loc[frame["evaluation_split"].astype(str).eq(split)]
        # The unadjusted regression uses the same standardized estimand as the
        # primary model.  Rank-AUC/Spearman from the broad screen remain the
        # robust discovery statistics, but cannot be subtracted from an odds
        # ratio or standardized beta to quantify confounding.
        model_specs = [
            ("unadjusted_regression", ()),
            ("primary", PRIMARY_COVARIATES),
        ]
        if _sensitivity_adjustment_allowed(target_id, target_label):
            model_specs.append(("rhythm_conduction_sensitivity", SENSITIVITY_COVARIATES))
        for model_id, covariates in model_specs:
            old = (reusable or {}).get((marker_id, split, model_id))
            if old is not None:
                records.append(dict(old))
                continue
            result = fit_adjusted_association(
                scoped,
                target_kind=target_kind,
                covariates=covariates,
                model_id=model_id,
                _prepared=prepared,
            )
            result.update(
                {
                    "axis": axis,
                    "target_id": target_id,
                    "target_label": target_label,
                    "temporal_relation": temporal_relation,
                    "feature_id": marker_id,
                    "evaluation_split": split,
                }
            )
            records.append(result)
    return records


def _adjust_replicated_hits(
    config: MimicProjectConfig,
    *,
    replicated: pd.DataFrame,
    feature_glob: str,
    sources: Mapping[str, Mapping[str, object]],
    progress: Optional[Callable[[str], None]] = None,
    checkpoint_root: Optional[Path] = None,
    checkpoint_input_hash: str = "",
    helper_shard: Optional[tuple[int, int]] = None,
) -> pd.DataFrame:
    """Reload only replicated candidates and fit prespecified robust models."""

    if replicated.empty:
        return pd.DataFrame()
    import duckdb

    master = _parquet(config.master_root / "ecg_master_index.parquet")
    records: list[dict] = []
    packet_paths = []
    requested_marker_ids = sorted(
        set(replicated["feature_id"].astype(str))
    )
    groups = list(
        replicated.groupby(
            ["axis", "target_id", "temporal_relation"],
            sort=True,
            dropna=False,
        )
    )
    from .frame_checkpoints import checkpointed_frame_chunk_path, materialize_frame_chunks
    # Persist the immutable candidate plan before helpers may attach. Scientific
    # input identity, ordered candidate rows and model versions bind every packet.
    plan_hash = content_hash({"input": checkpoint_input_hash,
        "version": BROAD_AXIS_ADJUSTMENT_VERSION,
        "candidates": replicated[['axis', 'target_id', 'temporal_relation', 'feature_id']].to_dict(orient="records")})
    packet_input_hash = content_hash({'plan': plan_hash,
        'numeric_encoding': BROAD_ADJUSTMENT_PACKET_NAMESPACE})
    if checkpoint_root is not None:
        checkpoint_root = Path(checkpoint_root) / plan_hash
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        plan_path = checkpoint_root / "plan.pkl"
        from .frame_checkpoints import frame_chunk_lock
        with frame_chunk_lock(checkpoint_root, namespace="plan", chunk_index=0):
            if not plan_path.exists():
                import pickle, os
                temporary = plan_path.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    slim = replicated[['axis', 'target_id', 'target_label', 'temporal_relation', 'feature_id']].copy()
                    serial_sources = {axis: {k: v for k, v in source.items() if not callable(v)}
                                      for axis, source in sources.items()}
                    pickle.dump(dict(replicated=slim, feature_glob=feature_glob,
                        sources=serial_sources, input_hash=checkpoint_input_hash, plan_hash=plan_hash), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, plan_path)
                atomic_write_json(checkpoint_root / 'plan_metadata.json', dict(
                    plan_hash=plan_hash, input_hash=checkpoint_input_hash, groups=len(groups),
                    total_models=sum(len(set(hits['feature_id'].astype(str))) *
                        (6 if _sensitivity_adjustment_allowed(str(key[1]), str(hits.iloc[0].get('target_label', key[1]))) else 4)
                        for key, hits in groups), packet_size=8))
    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        # Binding a wide, partitioned feature relation is not free: DuckDB
        # has to inspect the Parquet schema before it can answer even a
        # ``LIMIT 0`` query. Resolve the replicated shortlist once for this
        # connection instead of repeating the same schema probe for every
        # target-marker pair.
        basis_manifest = checkpoint_root / 'development_basis.json' if checkpoint_root is not None else None
        certified = (__import__('json').loads(basis_manifest.read_text()) if basis_manifest is not None and basis_manifest.exists() else {})
        requested_basis_hash = content_hash({'input': plan_hash, 'markers': requested_marker_ids, 'projection': 1})
        if certified.get('input_hash') == requested_basis_hash:
            # This exact marker list was certified on initial publication. The
            # basis verifier below rechecks the file SHA before any packet read.
            available_marker_set = set(requested_marker_ids)
        else:
            available_marker_set = set(available_markers(connection, feature_glob, requested_marker_ids))
        basis_relation = None
        basis_aliases = _adjustment_marker_projection(sorted(available_marker_set))[1]
        if checkpoint_root is not None:
            # COPY streams a shared development-only projection to disk; Python
            # never materializes the complete wide feature matrix. Packet scans
            # then read only their eight markers and avoid repeated master joins.
            basis_relation = _adjustment_basis(connection, checkpoint_root,
                feature_glob=feature_glob, master=master,
                markers=sorted(available_marker_set), input_hash=plan_hash)
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_INDEX_COHORT_TABLE} AS
            SELECT f.study_id
            FROM (
                SELECT DISTINCT study_id
                FROM {basis_relation if basis_relation is not None else _feature_parquet(feature_glob)}
            ) f
            JOIN {master} m USING (study_id)
            WHERE m.split IN {tuple(DEVELOPMENT_SPLITS)}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY m.subject_id
                ORDER BY m.ecg_time, f.study_id
            ) = 1
            """
        )
        for position, ((axis, target_id, relation), hits) in enumerate(groups, start=1):
            if helper_shard is not None and (position - 1) % helper_shard[1] != helper_shard[0]:
                continue
            source = sources.get(str(axis))
            if source is None or not source.get("available"):
                continue
            policy = policy_for(str(axis))
            target_kind = str(policy.target_kind)
            label = str(hits.iloc[0].get("target_label", target_id))
            if progress:
                progress(
                    f"{axis}: adjustierte Bestaetigung {position}/{len(groups)}"
                )
            if target_kind == "continuous":
                values_sql = continuous_axis_values_sql(
                    str(source["relation"]),
                    master,
                    axis=str(axis),
                    entity_ids=[str(target_id)],
                    time_column=str(source.get("time_column", "charttime")),
                )
                connection.execute(
                    f"""
                    CREATE OR REPLACE TEMP TABLE {_AXIS_VALUES_TABLE} AS
                    SELECT v.*
                    FROM ({values_sql}) v
                    JOIN {master} m USING (study_id)
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY
                            v.subject_id,
                            v.target_id,
                            v.temporal_relation
                        ORDER BY m.ecg_time, v.study_id
                    ) = 1
                    """
                )
            else:
                presence_sql = binary_axis_values_sql(
                    str(source["relation"]),
                    master,
                    axis=str(axis),
                    entity_column=str(source["entity_column"]),
                    entity_ids=[str(target_id)],
                    date_column=source.get("date_column"),
                )
                connection.execute(
                    "CREATE OR REPLACE TEMP TABLE vcg_adjustment_presence AS "
                    + presence_sql
                )
            marker_ids = [
                marker_id
                for marker_id in sorted(set(hits["feature_id"].astype(str)))
                if marker_id in available_marker_set
            ]
            # Small durable packets bound Python materialization and pause loss.
            packet_size = 8 if checkpoint_root is not None else ADJUSTMENT_MARKER_BLOCK_SIZE
            for block_start in range(
                0,
                len(marker_ids),
                packet_size,
            ):
                block = marker_ids[
                    block_start : block_start + packet_size
                ]
                marker_projection, aliases = _adjustment_marker_projection(
                    block
                )
                select = (
                    "SELECT\n            "
                    + (_adjustment_common_select() if basis_relation is None else
                        ', '.join('f.'+name for name in ('study_id', 'subject_id', 'ecg_time',
                        'evaluation_split', 'age', 'sex', 'heart_rate_bpm', 'signal_quality_score',
                        'care_setting', 'rhythm_context', 'qrs_duration_ms')))
                    + ",\n            "
                    + marker_projection
                )
                if target_kind == "continuous":
                    query = f"""
                        {select}, v.value AS target_value
                        FROM {_AXIS_VALUES_TABLE} v
                        JOIN {_feature_parquet(feature_glob)} f USING (study_id)
                        JOIN {master} m USING (study_id)
                        WHERE v.target_id = {_sql_literal(target_id)}
                          AND v.temporal_relation = {_sql_literal(relation)}
                          AND m.split IN {tuple(DEVELOPMENT_SPLITS)}
                    """
                else:
                    query = f"""
                        {select}, CASE WHEN p.study_id IS NULL THEN 0 ELSE 1 END
                            AS target_value
                        FROM {_INDEX_COHORT_TABLE} i
                        JOIN {_feature_parquet(feature_glob)} f USING (study_id)
                        JOIN {master} m USING (study_id)
                        LEFT JOIN (
                            SELECT DISTINCT study_id
                            FROM vcg_adjustment_presence
                            WHERE target_id = {_sql_literal(target_id)}
                              AND temporal_relation = {_sql_literal(relation)}
                        ) p USING (study_id)
                        WHERE m.split IN {tuple(DEVELOPMENT_SPLITS)}
                    """
                def compute_packet():
                    old_frame = None
                    if checkpoint_root is not None:
                        legacy_namespace=f'group-{position:04d}'
                        legacy_index=block_start//packet_size
                        if (checkpoint_root/legacy_namespace/f'part-{legacy_index:06d}.json').exists():
                            old_path, _ = checkpointed_frame_chunk_path(checkpoint_root,
                                namespace=legacy_namespace, chunk_index=legacy_index,
                                input_hash=plan_hash, item_keys=list(block),
                                compute=lambda: (_ for _ in ()).throw(RuntimeError('Legacy result missing; refuse reuse')))
                            old_frame=pd.read_parquet(old_path)
                            if not old_frame['reason'].eq('fit_failed:ValueError').any():
                                return old_frame
                    packet_query = query
                    if basis_relation is not None:
                        for marker in block:
                            packet_query = packet_query.replace('f."'+marker.replace('"', '""')+'"',
                                'f."'+basis_aliases[marker]+'"')
                        packet_query = packet_query.replace(_feature_parquet(feature_glob), basis_relation)
                        packet_query = packet_query.replace(f'JOIN {master} m USING (study_id)', '')
                        packet_query = packet_query.replace('m.split IN', 'f.evaluation_split IN')
                    return _compute_adjustment_packet(connection, query=packet_query, block=block,
                        aliases=aliases, axis=str(axis), target_id=str(target_id), label=label,
                        relation=str(relation), target_kind=target_kind, old_frame=old_frame)
                if checkpoint_root is not None:
                    path, resumed = checkpointed_frame_chunk_path(checkpoint_root / BROAD_ADJUSTMENT_PACKET_NAMESPACE,
                        namespace=f"group-{position:04d}", chunk_index=block_start // packet_size,
                        input_hash=packet_input_hash, item_keys=list(block), compute=compute_packet)
                    if helper_shard is None:
                        packet_paths.append(path)
                    if progress:
                        progress(f"{axis}: Adjustierung Gruppe {position}/{len(groups)}, "
                            f"Marker {min(block_start+len(block), len(marker_ids))}/{len(marker_ids)} "
                            + ("uebernommen" if resumed else "gesichert"))
                    continue
                records.extend(compute_packet().to_dict(orient='records'))
    if helper_shard is not None:
        return pd.DataFrame()
    if packet_paths:
        result = materialize_frame_chunks(packet_paths)
        return add_adjustment_effect_change(add_adjusted_fdr(result))
    if not records:
        return pd.DataFrame()
    if helper_shard is not None:
        return pd.DataFrame(records)
    return add_adjustment_effect_change(add_adjusted_fdr(pd.DataFrame(records)))


def _compute_adjustment_packet(connection, *, query, block, aliases, axis,
                               target_id, label, relation, target_kind, old_frame=None):
    reusable = {}
    if old_frame is not None:
        for row in old_frame.to_dict(orient='records'):
            if row.get('reason') != 'fit_failed:ValueError':
                reusable[(str(row['feature_id']), str(row['evaluation_split']), str(row['model_id']))] = row
    try:
        block_frame = connection.execute(query).fetchdf()
    except Exception as error:
        raise RuntimeError(
            "Broad-axis adjusted data load failed for "
            f"axis={axis}, target={target_id}, relation={relation}, "
            f"markers={block}: {type(error).__name__}: {error}"
        ) from error
    assert_no_sealed_rows(
        block_frame.rename(
            columns={"evaluation_split": "split"}, copy=False
        ),
        where="broad-axis adjusted confirmation",
    )
    # All covariates and outcome rows are identical for the block.
    # Give each fit only those common columns plus its own marker:
    # ``fit_adjusted_association`` defensively copies its input,
    # so leaving the other 63 marker columns attached would copy
    # them 64 times and trade an I/O win for a memory regression.
    # The old fitter sorted and selected the same index ECG on each
    # of six model calls. Prepare once, preserving split-specific
    # ordering, duplicate selection and the original row indices.
    scoped_frames = []
    split_values = block_frame['evaluation_split'].astype(str)
    for split in DEVELOPMENT_SPLITS:
        scoped = block_frame.loc[split_values.eq(split)]
        if 'subject_id' in scoped:
            order = [name for name in ('subject_id', 'ecg_time', 'study_id') if name in scoped]
            scoped = scoped.sort_values(order, kind='mergesort').drop_duplicates('subject_id', keep='first')
        scoped_frames.append(scoped)
    block_frame = pd.concat(scoped_frames)
    records = []
    marker_aliases = set(aliases.values())
    common_columns = [
        name
        for name in block_frame.columns
        if name not in marker_aliases
    ]
    for marker_id in block:
        marker_frame = block_frame.loc[
            :, [*common_columns, aliases[marker_id]]
        ].rename(
            columns={aliases[marker_id]: "marker_value"},
            copy=False,
        )
        records.extend(
            _fit_adjusted_candidate(
                marker_frame,
                axis=str(axis),
                target_id=str(target_id),
                target_label=label,
                temporal_relation=str(relation),
                marker_id=marker_id,
                target_kind=target_kind,
                prepared=True,
                reusable=reusable,
            )
        )
    del block_frame
    return pd.DataFrame(records)


def _adjustment_basis(connection, root, *, feature_glob, master, markers, input_hash):
    """Publish one immutable shared feature/covariate projection under an OS lock."""
    from .frame_checkpoints import frame_chunk_lock
    from .storage import file_sha256
    path = root / 'development_basis.parquet'
    manifest_path = root / 'development_basis.json'
    expected = content_hash({'input': input_hash, 'markers': markers, 'projection': 1})
    with frame_chunk_lock(root, namespace='basis', chunk_index=0):
        if manifest_path.exists():
            manifest = __import__('json').loads(manifest_path.read_text(encoding='utf-8'))
            if (manifest['input_hash'] != expected or not path.is_file()
                    or path.stat().st_size != manifest['size'] or file_sha256(path) != manifest['sha256']):
                raise RuntimeError('Invalid shared adjustment basis; refusing overwrite')
        else:
            temporary = path.with_suffix('.tmp.parquet')
            projection = _adjustment_marker_projection(markers)[0]
            query = (f'SELECT {_adjustment_common_select()}, {projection} '
                     f'FROM {_feature_parquet(feature_glob)} f JOIN {master} m USING (study_id) '
                     f'WHERE m.split IN {tuple(DEVELOPMENT_SPLITS)}')
            connection.execute(f'COPY ({query}) TO {_sql_literal(str(temporary))} '
                               "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 16384)")
            import os
            with temporary.open('r+b') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            atomic_write_json(manifest_path, dict(input_hash=expected, size=path.stat().st_size,
                sha256=file_sha256(path), locked_test_read=False))
    return _parquet(path)


def _annotate_adjusted_confirmation(
    replicated: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    output = replicated.copy()
    output["adjusted_confirmed"] = False
    output["adjusted_status"] = "not_assessed"
    output["adjusted_reason"] = "adjusted_model_not_available"
    for column in (
        "adjusted_effect",
        "adjusted_ci_low",
        "adjusted_ci_high",
        "adjusted_domain_q_value",
        "adjusted_global_q_value",
    ):
        output[column] = np.nan
    if output.empty or adjusted.empty:
        return output
    primary = adjusted.loc[adjusted["model_id"].eq("primary")]
    key_columns = ["axis", "target_id", "temporal_relation", "feature_id"]
    train = primary.loc[primary["evaluation_split"].eq("train")].set_index(key_columns)
    validation = primary.loc[primary["evaluation_split"].eq("validation")].set_index(key_columns)
    for index, row in output.iterrows():
        key = tuple(row[column] for column in key_columns)
        if key not in validation.index:
            continue
        val = validation.loc[key]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[0]
        output.at[index, "adjusted_status"] = str(val.get("status", "non_estimable"))
        output.at[index, "adjusted_reason"] = str(val.get("reason", ""))
        for source_column, target_column in (
            ("effect", "adjusted_effect"),
            ("ci_low", "adjusted_ci_low"),
            ("ci_high", "adjusted_ci_high"),
            ("domain_q_value", "adjusted_domain_q_value"),
            ("global_q_value", "adjusted_global_q_value"),
        ):
            output.at[index, target_column] = val.get(source_column, np.nan)
        if key not in train.index:
            continue
        trn = train.loc[key]
        if isinstance(trn, pd.DataFrame):
            trn = trn.iloc[0]
        confirmed = bool(
            trn.get("status") == "estimable"
            and val.get("status") == "estimable"
            and float(trn.get("domain_q_value", np.nan)) <= TRAIN_FDR_ALPHA
            and float(val.get("domain_q_value", np.nan)) <= VALIDATION_FDR_ALPHA
            and float(trn.get("estimate", 0.0)) * float(val.get("estimate", 0.0)) > 0
        )
        output.at[index, "adjusted_confirmed"] = confirmed
        if not confirmed and not str(output.at[index, "adjusted_reason"]):
            output.at[index, "adjusted_reason"] = (
                "adjusted_fdr_or_direction_confirmation_failed"
            )
    return output


def screening_marker_ids(panel_ids: Sequence[str]) -> tuple[str, ...]:
    """Markers eligible to be screened, in a stable order."""

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    names: list[str] = []
    for panel_id in panel_ids:
        panel = panels.get(panel_id)
        if panel is None:
            raise KeyError(f"Unknown feature panel: {panel_id}")
        names.extend(panel.feature_ids)
    ordered = list(dict.fromkeys(names))
    return tuple(
        name
        for name in ordered
        if not name.startswith("classical_")
        and feature_profile(name).role not in {"alias", "qc"}
    )


def axis_sources(config: MimicProjectConfig) -> dict[str, dict[str, object]]:
    """Where each axis reads its entities and values from.

    Returned rather than hard-coded inside the run so a missing source is a
    reported skip instead of an exception halfway through the sweep.
    """

    hosp = config.mimic_iv_root / "hosp"
    icu = config.mimic_iv_root / "icu"
    standardized = config.work_root / "standardized"
    events = standardized / "clinical_events"
    registry = sorted(events.glob("versions/schema-*"))
    latest = registry[-1] if registry else None

    def registry_glob(table: str) -> Optional[str]:
        if latest is None:
            return None
        directory = latest / table
        if not directory.is_dir():
            return None
        return _parquet(directory / "**" / "*.parquet")

    def standardized_or_raw(
        standardized_name: str,
        raw_path: Path,
    ) -> tuple[str, bool]:
        """Prefer the one-time ingest result over another compressed scan."""

        standardized_path = standardized / standardized_name
        if standardized_path.is_file():
            return _parquet(standardized_path), True
        return _csv(raw_path), raw_path.is_file()

    laboratory_relation = registry_glob("labevents")
    laboratory_available = laboratory_relation is not None
    if laboratory_relation is not None:
        # Audit D2: the event layer carries comments; hemolysis-falsified
        # results are not measurements of the target.
        from .lab_quality import without_hemolyzed_results

        laboratory_relation = without_hemolyzed_results(laboratory_relation)
    if laboratory_relation is None:
        laboratory_relation, laboratory_available = standardized_or_raw(
            "labevents.parquet", hosp / "labevents.csv.gz"
        )
    lab_dictionary, _ = standardized_or_raw(
        "d_labitems.parquet", hosp / "d_labitems.csv.gz"
    )

    icu_relation = registry_glob("chartevents")
    icu_available = icu_relation is not None
    if icu_relation is None:
        icu_relation, icu_available = standardized_or_raw(
            "chartevents.parquet", icu / "chartevents.csv.gz"
        )
    icu_dictionary, _ = standardized_or_raw(
        "d_items.parquet", icu / "d_items.csv.gz"
    )

    procedure_relation, procedure_available = standardized_or_raw(
        "procedures_icd.parquet", hosp / "procedures_icd.csv.gz"
    )
    microbiology_relation, microbiology_available = standardized_or_raw(
        "microbiologyevents.parquet", hosp / "microbiologyevents.csv.gz"
    )
    if not microbiology_relation.startswith("read_parquet"):
        microbiology_relation = (
            f"(SELECT subject_id, hadm_id, org_name AS organism FROM "
            f"{microbiology_relation})"
        )

    # The CCSR axis: ICD-10 admission diagnoses mapped to the coarser CCSR
    # taxonomy. The axis had a policy and a counting query but no source
    # entry, so every run skipped it as "source unavailable" and the stage
    # then refused to complete; the production-path preview was the first
    # run to reach it. The mapping is the pinned package resource the phenome
    # scan uses; a missing resource is a reported skip, as for the others.
    diagnoses_relation, diagnoses_available = standardized_or_raw(
        "diagnoses_icd.parquet", hosp / "diagnoses_icd.csv.gz"
    )
    ccsr_available = False
    ccsr_relation = "(SELECT NULL::BIGINT AS subject_id, NULL::BIGINT AS hadm_id, "         "NULL::VARCHAR AS ccsr_category)"
    ccsr_mapping = "(SELECT NULL::VARCHAR AS icd_code, NULL::VARCHAR AS ccsr_category, "         "NULL::VARCHAR AS ccsr_description)"
    ccsr_reason = ""
    if diagnoses_available:
        try:
            from .phenotypes import prepare_phenotype_resources

            ccsr_mapping = _parquet(prepare_phenotype_resources(config).ccsr_mapping_path)
            ccsr_relation = (
                "(SELECT DISTINCT d.subject_id, d.hadm_id, c.ccsr_category "
                f"FROM {diagnoses_relation} d JOIN {ccsr_mapping} c "
                "ON d.icd_version = 10 AND d.icd_code = c.icd_code "
                "WHERE d.subject_id IS NOT NULL AND d.hadm_id IS NOT NULL)"
            )
            ccsr_available = True
        except Exception as error:  # noqa: BLE001 - a missing resource is a reported skip
            ccsr_reason = f"{type(error).__name__}: {error}"
    else:
        ccsr_reason = "diagnoses_icd source missing"

    return {
        AXIS_CCSR: {
            "counts_sql": lambda: ccsr_target_counts_sql(
                ccsr_mapping,
                f"(SELECT subject_id, hadm_id, icd_code FROM {diagnoses_relation} "
                "WHERE icd_version = 10)",
            ),
            "relation": ccsr_relation,
            "kind": "binary",
            "entity_column": "ccsr_category",
            "available": ccsr_available,
            "reason": ccsr_reason,
        },
        AXIS_LABORATORY: {
            "counts_sql": lambda: laboratory_target_counts_sql(
                laboratory_relation,
                lab_dictionary,
            ),
            "relation": laboratory_relation,
            "kind": "continuous",
            "time_column": "charttime",
            "available": laboratory_available,
        },
        AXIS_ICU_MEASUREMENT: {
            "counts_sql": lambda: icu_measurement_target_counts_sql(
                icu_relation,
                icu_dictionary,
            ),
            "relation": icu_relation,
            "kind": "continuous",
            "time_column": "charttime",
            "available": icu_available,
        },
        AXIS_DRG: {
            "counts_sql": lambda: drg_target_counts_sql(
                _csv(hosp / "drgcodes.csv.gz")
            ),
            "relation": (
                f"(SELECT subject_id, hadm_id, drg_code::VARCHAR AS drg_code "
                f"FROM {_csv(hosp / 'drgcodes.csv.gz')})"
            ),
            "kind": "binary",
            "entity_column": "drg_code",
            "available": (hosp / "drgcodes.csv.gz").is_file(),
        },
        AXIS_PROCEDURE: {
            "counts_sql": lambda: procedure_target_counts_sql(
                procedure_relation,
                _csv(hosp / "d_icd_procedures.csv.gz")
                if (hosp / "d_icd_procedures.csv.gz").is_file()
                else None,
            ),
            "relation": procedure_relation,
            "kind": "binary",
            "entity_column": "icd_code",
            "date_column": "chartdate",
            "available": procedure_available,
        },
        AXIS_MICROBIOLOGY: {
            "counts_sql": lambda: microbiology_target_counts_sql(
                microbiology_relation
            ),
            "relation": microbiology_relation,
            "kind": "binary",
            "entity_column": "organism",
            "available": microbiology_available,
        },
    }


def assert_no_sealed_rows(frame: pd.DataFrame, *, where: str) -> None:
    """Prove the sealed split is absent instead of asserting it in prose.

    Every artifact of this stage carries ``locked_test_read: False``. That is a
    claim; this makes it a check. The sealed split is the one thing in the
    pipeline that no recomputation can restore, so the guarantee is worth a
    line of code rather than a line of documentation.
    """

    if frame.empty or "split" not in frame.columns:
        return
    present = set(str(value) for value in frame["split"].unique())
    forbidden = present - set(DEVELOPMENT_SPLITS)
    if forbidden:
        raise PermissionError(
            f"{where} returned rows from {sorted(forbidden)}; this stage may "
            "only read the development splits."
        )


def available_markers(
    connection,
    feature_glob: str,
    marker_ids: Sequence[str],
) -> list[str]:
    """Intersect the requested panel with the columns the files actually have.

    A panel and a feature set can legitimately disagree: markers added after
    an extraction exist in the panel but not yet in the parquet files. Binding
    them anyway fails the whole axis on the first missing column, which turns
    a partial result into no result.
    """

    try:
        present = {
            str(name)
            for name in connection.execute(
                f"SELECT * FROM {_feature_parquet(feature_glob)} LIMIT 0"
            ).fetchdf().columns
        }
    except Exception:  # noqa: BLE001 - an unreadable feature set is reported upstream
        return []
    return [name for name in marker_ids if name in present]


def _screen_one_axis(
    connection,
    *,
    axis: str,
    policy: BroadAxisPolicy,
    source: Mapping[str, object],
    master_relation: str,
    feature_glob: str,
    marker_ids: Sequence[str],
    progress: Optional[Callable[[str], None]],
    markers_prevalidated: bool = False,
    checkpoint_root: Optional[Path] = None,
    checkpoint_key: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    """Enumerate, extract and screen one axis.

    The two target kinds need different comparison groups, and conflating them
    is how a screen silently tests nothing:

    * A **continuous** target compares the marker against the measured value,
      among the ECGs that have such a measurement. ECGs without one are not
      controls; they are simply not informative about that quantity.
    * A **binary** target compares the marker between ECGs whose admission
      carries the entity and every other ECG of the same split. Emitting only
      the ECGs that carry it -- as the value table does -- leaves a case group
      and no controls at all.
    """

    counts = connection.execute(source["counts_sql"]()).fetchdf()
    targets = enumerate_targets(
        counts,
        policy,
        entity_column="entity_id",
        label_column="entity_label",
    )
    eligible = targets.loc[targets["eligible_for_screen"].astype(bool)]
    coverage = axis_coverage(targets)
    if eligible.empty:
        return targets, pd.DataFrame(), coverage

    entity_ids = [str(value) for value in eligible["entity_id"].tolist()]
    labels = {
        str(row.entity_id): str(row.entity_label) for row in eligible.itertuples()
    }
    if progress:
        progress(
            f"{axis}: {len(entity_ids):,} Ziele ueber der Schwelle, "
            f"{len(targets) - len(entity_ids):,} unterpowert"
        )

    if not markers_prevalidated:
        marker_ids = available_markers(connection, feature_glob, marker_ids)
    if not marker_ids:
        return targets, pd.DataFrame(), coverage
    splits = tuple(DEVELOPMENT_SPLITS)

    if policy.target_kind == "continuous":
        values_sql = continuous_axis_values_sql(
            str(source["relation"]),
            master_relation,
            axis=axis,
            entity_ids=entity_ids,
            time_column=str(source.get("time_column", "charttime")),
        )
        # The value table is built once. Measured on the laboratory axis:
        # 16,527,138 rows in 4.6 s. Rebuilding it per block would be the only
        # part of this that blocking makes more expensive.
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_AXIS_VALUES_TABLE} AS
            SELECT v.*, m.ecg_time
            FROM ({values_sql}) v
            JOIN {master_relation} m USING (study_id)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY v.subject_id, v.target_id, v.temporal_relation
                ORDER BY m.ecg_time, v.study_id
            ) = 1
            """
        )
        row_count = int(
            connection.execute(
                f"SELECT count(*) FROM {_AXIS_VALUES_TABLE} "
                f"WHERE split IN {splits}"
            ).fetchone()[0]
        )
        if row_count == 0:
            return targets, pd.DataFrame(), coverage
        blocks = _marker_blocks(marker_ids, row_count)
        records: list[dict] = []
        for index, block in enumerate(blocks, start=1):
            block_key = _axis_block_key(
                checkpoint_key, axis, "continuous", block, row_count, entity_ids
            )
            cached = _load_axis_block(checkpoint_root, axis, "continuous", index, block_key)
            if cached is not None:
                records.extend(cached)
                if progress:
                    progress(f"{axis}: Markerblock {index}/{len(blocks)} liegt bereits vor")
                continue
            if progress:
                progress(
                    f"{axis}: Messwerte und Merkmale verknuepfen "
                    f"(Markerblock {index}/{len(blocks)})"
                )
            columns = ", ".join(f'f."{name}"' for name in block)
            frame = connection.execute(
                f"""
                SELECT
                    v.subject_id,
                    v.study_id,
                    v.ecg_time,
                    v.target_id,
                    v.temporal_relation,
                    v.split,
                    v.value,
                    {columns}
                FROM {_AXIS_VALUES_TABLE} v
                JOIN {_feature_parquet(feature_glob)} f USING (study_id)
                WHERE v.split IN {splits}
                """
            ).fetchdf()
            assert_no_sealed_rows(frame, where=f"{axis} value join")
            block_records = (
                []
                if frame.empty
                else list(
                    _screen_continuous(
                        frame,
                        axis,
                        policy,
                        labels,
                        block,
                        already_indexed=True,
                    )
                )
            )
            del frame
            records.extend(block_records)
            _save_axis_block(checkpoint_root, axis, "continuous", index, block_key, block_records)
            if progress:
                progress(f"{axis}: Markerblock {index}/{len(blocks)} gesichert")
        connection.execute(f"DROP TABLE IF EXISTS {_AXIS_VALUES_TABLE}")
        if not records:
            return targets, pd.DataFrame(), coverage
        return (
            targets,
            pd.DataFrame(_ordered_records(records, marker_ids)),
            coverage,
        )

    # Binary: every ECG of the split participates, presence decides the group.
    presence_sql = binary_axis_values_sql(
        str(source["relation"]),
        master_relation,
        axis=axis,
        entity_column=str(source["entity_column"]),
        entity_ids=entity_ids,
        date_column=source.get("date_column"),
    )
    presence = connection.execute(
        f"SELECT DISTINCT study_id, target_id, temporal_relation "
        f"FROM ({presence_sql})"
    ).fetchdf()
    if presence.empty:
        return targets, pd.DataFrame(), coverage
    cohort_rows = int(
        connection.execute(
            f"SELECT count(*) FROM {_INDEX_COHORT_TABLE}"
        ).fetchone()[0]
    )
    if cohort_rows == 0:
        return targets, pd.DataFrame(), coverage
    blocks = _marker_blocks(marker_ids, cohort_rows)
    records = []
    for index, block in enumerate(blocks, start=1):
        block_key = _axis_block_key(
            checkpoint_key, axis, "binary", block, cohort_rows, entity_ids
        )
        cached = _load_axis_block(checkpoint_root, axis, "binary", index, block_key)
        if cached is not None:
            records.extend(cached)
            if progress:
                progress(f"{axis}: Markerblock {index}/{len(blocks)} liegt bereits vor")
            continue
        if progress:
            progress(
                f"{axis}: Merkmalsrahmen wird geladen "
                f"(Markerblock {index}/{len(blocks)})"
            )
        # Preserve the original outer checkpoint identity and marker order.
        # Only the transient frame/rank arrays are divided under a runtime cap.
        block_records = []
        batches = _binary_marker_batches(block, cohort_rows)
        for batch_index, batch in enumerate(batches, start=1):
            batch_root = (Path(checkpoint_root) / 'binary_batches' / f'{axis}-{index:04d}'
                          if checkpoint_root is not None and len(batches) > 1 else None)
            batch_key = content_hash({'outer_block_key':block_key, 'markers':list(batch)})
            cached_batch = _load_axis_block(batch_root, axis, 'binary', batch_index, batch_key)
            if cached_batch is not None:
                block_records.extend(cached_batch)
                if progress:
                    progress(f"{axis}: Markerblock {index}/{len(blocks)}, "
                             f"Speicherpaket {batch_index}/{len(batches)} liegt bereits vor")
                continue
            if progress and len(batches) > 1:
                progress(f"{axis}: Markerblock {index}/{len(blocks)}, "
                         f"Speicherpaket {batch_index}/{len(batches)}")
            columns = ", ".join(f'f."{name}"' for name in batch)
            cohort = connection.execute(
                f"""
                SELECT i.study_id, i.subject_id, i.ecg_time, i.split, {columns}
                FROM {_INDEX_COHORT_TABLE} i
                JOIN {_feature_parquet(feature_glob)} f USING (study_id)
                """
            ).fetchdf()
            assert_no_sealed_rows(cohort, where=f"{axis} cohort")
            batch_records = []
            if not cohort.empty:
                batch_records = _screen_binary(
                    cohort, presence, axis, policy, labels, batch,
                    already_indexed=True,
                )
            del cohort
            _save_axis_block(batch_root, axis, 'binary', batch_index, batch_key, batch_records)
            block_records.extend(batch_records)
            if progress and len(batches) > 1:
                progress(f"{axis}: Markerblock {index}/{len(blocks)}, "
                         f"Speicherpaket {batch_index}/{len(batches)} gesichert")
        records.extend(block_records)
        _save_axis_block(checkpoint_root, axis, "binary", index, block_key, block_records)
        if progress:
            progress(f"{axis}: Markerblock {index}/{len(blocks)} gesichert")
    if not records:
        return targets, pd.DataFrame(), coverage
    return targets, pd.DataFrame(_ordered_records(records, marker_ids)), coverage


# Finished marker blocks of one axis, kept across a pause. The sweep ran
# without any intermediate state: a memory pause late in an axis threw away
# every block already screened. The key binds the stage key (inputs, markers,
# policies), the scientific config, the feature and index files, the axis
# targets and the block itself; anything else recomputes.
def _axis_block_key(stage_key, axis, kind, block, rows, entity_ids) -> str:
    return content_hash({
        "stage": str(stage_key),
        "axis": str(axis),
        "kind": kind,
        "markers": list(block),
        "rows": int(rows),
        "targets": sorted(str(item) for item in (entity_ids or ())),
    })


def _axis_block_path(root, axis, kind, index) -> Optional[Path]:
    if root is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(axis))
    return Path(root) / f"{safe}-{kind}-{int(index):04d}.pkl"


def _load_axis_block(root, axis, kind, index, key):
    import pickle

    path = _axis_block_path(root, axis, kind, index)
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    records = payload.get("records")
    return list(records) if isinstance(records, list) else None


def _save_axis_block(root, axis, kind, index, key, records) -> None:
    import os
    import pickle

    path = _axis_block_path(root, axis, kind, index)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"key": key, "records": list(records)}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _record(
    *,
    axis: str,
    policy: BroadAxisPolicy,
    target_id: str,
    label: str,
    marker_id: str,
    split: str,
    n_observations: int,
    effect: float,
    p_value: float,
    n_positive: int,
    n_negative: int,
    temporal_relation: str = RELATION_UNDATED,
) -> dict:
    return {
        "axis": axis,
        "target_id": str(target_id),
        "target_label": label,
        "feature_id": marker_id,
        "evaluation_split": split,
        "n_observations": int(n_observations),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "effect": float(effect),
        "p_value": float(p_value),
        "target_kind": policy.target_kind,
        "reference_tier": policy.reference_tier,
        "claim_ceiling": policy.claim_ceiling,
        # Without these two a reader cannot tell whether the target preceded
        # the ECG, followed it, or was assigned at discharge -- which decides
        # whether a hit reads as diagnostic, prognostic or consequential.
        "temporal_role": policy.temporal_role,
        "temporal_relation": temporal_relation,
        "inference_status": inference_status(n_positive, n_negative),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Midranks, so ties are handled the way Mann-Whitney requires.

    Delegated to scipy rather than looped in Python: at 640,000 rows the
    hand-written tie walk cost 259 ms per marker, which for 1,713 markers and
    two splits was fifteen minutes of pure interpreter overhead per axis.
    """

    from scipy import stats

    return np.asarray(stats.rankdata(values, method="average"), dtype=float)


def _tie_term(values: np.ndarray) -> float:
    """Sum of ``t^3 - t`` over tie groups.

    Computed once per marker and split rather than once per test: every ECG of
    the split is either a case or a control for every target, so the pooled
    sample -- and therefore this term -- does not depend on the target at all.
    Recomputing it per test meant an O(n log n) unique() inside the innermost
    loop.
    """

    if len(values) < 2:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(np.sum(counts.astype(np.float64) ** 3 - counts))


def _rank_auc(
    ranks: np.ndarray,
    *,
    case_mask: np.ndarray,
    n_cases: int,
    n_controls: int,
    tie_term: float,
) -> tuple[float, float]:
    """Mann-Whitney AUC and tie-corrected p-value from precomputed midranks.

    The ranks were computed over cases and controls together, which is exactly
    the pooled ranking the statistic is defined on, so this is the same number
    the pairwise form gives -- only without re-sorting for every target.
    """

    if n_cases < 1 or n_controls < 1:
        return float("nan"), float("nan")
    total = n_cases + n_controls
    rank_sum = float(np.sum(ranks[case_mask]))
    statistic = rank_sum - n_cases * (n_cases + 1) / 2.0
    auc = statistic / (n_cases * n_controls)

    variance = (
        n_cases
        * n_controls
        / 12.0
        * ((total + 1) - tie_term / (total * (total - 1.0)))
    )
    if not np.isfinite(variance) or variance <= 0:
        return float(auc), float("nan")
    mean = n_cases * n_controls / 2.0
    # Same continuity correction as ``statistics.fast_mann_whitney_auc``, so
    # this screen and the phenotype screen report the same p-value for the
    # same data instead of differing in the fourth digit.
    centered = abs(statistic - mean)
    z = max(0.0, centered - 0.5) / math.sqrt(variance)
    from scipy import stats

    p_value = float(2.0 * stats.norm.sf(z))
    return float(auc), p_value


# Cells (rows x markers) allowed in one materialised block.
#
# Measured on the laboratory axis at full size -- 13,232,238 rows after the
# split filter, 1,713 markers:
#
# ====== ========= ========== ==================
# Block  Dauer     Rahmen     hochgerechnet
# ====== ========= ========== ==================
#    10     12.3 s    2.61 GB   172 Bloecke, 35 min
#    25     23.6 s    4.20 GB    69 Bloecke, 27 min
#    60     69.2 s    7.90 GB    29 Bloecke, 33 min
# ====== ========= ========== ==================
#
# The total time barely moves, so the block size is chosen purely for memory.
# Unblocked this frame would have been 13,232,238 x 1,713 x 8 bytes = 181 GB
# for the laboratory axis alone (16,527,138 value rows before the split
# filter, 226 GB) on a machine with 32 GB. The stage had never run, so nobody
# had met it.
BROAD_AXIS_CELL_BUDGET = 150_000_000
_AXIS_VALUES_TABLE = "vcg_broad_axis_values"
_INDEX_COHORT_TABLE = "vcg_broad_index_cohort"


def _marker_blocks(
    marker_ids: Sequence[str], row_count: int
) -> tuple[tuple[str, ...], ...]:
    """Split the markers so one materialised frame stays bounded.

    Exact, not approximate: ``_screen_continuous`` and ``_screen_binary`` both
    compute every record from a single marker column plus the target and split
    columns, so which markers share a frame cannot change any result.
    """

    if not marker_ids:
        return ()
    per_block = max(1, int(BROAD_AXIS_CELL_BUDGET // max(1, int(row_count))))
    ordered = tuple(marker_ids)
    return tuple(
        ordered[start : start + per_block]
        for start in range(0, len(ordered), per_block)
    )


def _binary_marker_batches(marker_ids, row_count):
    """Smaller transient binary frames; published outer blocks stay unchanged."""
    import os
    raw = os.environ.get("VCG_JOB_MEMORY_GB")
    if raw is None:
        return (tuple(marker_ids),) if marker_ids else ()
    memory_gb = float(raw)
    if not math.isfinite(memory_gb) or memory_gb <= 0:
        raise ValueError("VCG_JOB_MEMORY_GB must be finite and positive")
    # Binary screening keeps cohort, split copies, finite masks and rank arrays.
    # Five million cells per granted GiB bound each transient materialization.
    cells = min(BROAD_AXIS_CELL_BUDGET, max(1, int(memory_gb * 5_000_000)))
    width = max(1, cells // max(1, int(row_count)))
    markers = tuple(marker_ids)
    return tuple(markers[i:i + width] for i in range(0, len(markers), width))


def _ordered_records(
    records: Sequence[dict], marker_ids: Sequence[str]
) -> list[dict]:
    """Put blocked records back into the order a single frame produced.

    The set of records does not depend on blocking; the order does, because
    the marker loop moved outside the target loop. Downstream FDR sorts on its
    own, but a screen whose row order changes with an internal batch size is
    not reproducible, and reproducibility is cheaper to keep than to argue
    about later.
    """

    marker_rank = {name: index for index, name in enumerate(marker_ids)}
    split_rank = {name: index for index, name in enumerate(DEVELOPMENT_SPLITS)}
    return sorted(
        records,
        key=lambda item: (
            str(item.get("target_id", "")),
            str(item.get("temporal_relation", "")),
            split_rank.get(str(item.get("evaluation_split", "")), 99),
            marker_rank.get(str(item.get("feature_id", "")), 10**9),
        ),
    )


def _missingness_groups(
    matrix: np.ndarray,
    marker_ids: Sequence[str],
) -> list[tuple[np.ndarray, list[int]]]:
    """Group markers that are observed on exactly the same recordings.

    What this saves is the *target* ranking, not the marker rankings: each
    marker is a different column and still has to be ranked once.  Markers
    sharing a finite pattern share the same usable row set, so the target is
    ranked once per pattern instead of once per marker.  Measured on the real
    store, 1,821 numeric columns fall into 153 distinct patterns.

    Measured end to end at 300 markers over 20,000 rows with 25 patterns: 0.80 s
    before, 0.62 s after -- a quarter off, with a largest rho difference of
    exactly zero.  That is the honest size of it; the bulk of the remaining
    cost is the per-marker ranking, which no grouping can remove.
    """

    grouped: dict[bytes, list[int]] = {}
    for position in range(matrix.shape[1]):
        key = np.isfinite(matrix[:, position]).tobytes()
        grouped.setdefault(key, []).append(position)
    return [
        (np.frombuffer(key, dtype=bool), positions)
        for key, positions in grouped.items()
    ]


def _cross_sectional_index(
    frame: pd.DataFrame,
    *,
    endpoint_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Select one deterministic ECG per patient and endpoint.

    Broad cross-sectional tests require independent patients, not recordings.
    Continuous endpoints use the earliest ECG at which that endpoint is
    observed; binary endpoints use the same outcome-blind earliest ECG for
    every endpoint.  Repeated recordings remain available to the dedicated
    within-subject stages.

    Small unit-level callers predating patient identifiers are kept usable;
    the production query always supplies ``subject_id``, ``ecg_time`` and
    ``study_id``.
    """

    if "subject_id" not in frame.columns:
        return frame
    if frame["subject_id"].isna().any():
        raise ValueError("cross_sectional_screen_requires_subject_id")
    keys = [
        name
        for name in (*endpoint_columns, "split", "subject_id")
        if name in frame.columns
    ]
    if not frame.duplicated(keys, keep=False).any():
        return frame
    order = [
        *keys,
        *(
            name
            for name in ("ecg_time", "study_id")
            if name in frame.columns and name not in keys
        ),
    ]
    return (
        frame.sort_values(order, kind="mergesort")
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )


def _screen_continuous(
    frame: pd.DataFrame,
    axis: str,
    policy: BroadAxisPolicy,
    labels: Mapping[str, str],
    marker_ids: Sequence[str],
    *,
    already_indexed: bool = False,
) -> list[dict]:
    from scipy import stats as scipy_stats

    if "temporal_relation" not in frame.columns:
        frame = frame.assign(temporal_relation=RELATION_UNDATED)
    if not already_indexed:
        frame = _cross_sectional_index(
            frame,
            endpoint_columns=("target_id", "temporal_relation"),
        )
    records: list[dict] = []
    columns = list(marker_ids)
    for (target_id, relation), block in frame.groupby(
        ["target_id", "temporal_relation"], sort=True
    ):
        for split in DEVELOPMENT_SPLITS:
            scoped = block.loc[block["split"] == split]
            if len(scoped) < MINIMUM_CASES:
                continue
            target_values = pd.to_numeric(
                scoped["value"], errors="coerce"
            ).to_numpy(dtype=float)
            target_finite = np.isfinite(target_values)
            # One conversion for the whole block. ``pd.to_numeric`` used to run
            # inside the marker loop, once per (target, split, marker), on a
            # column that Parquet already delivers as float64 -- and calling it
            # per column here instead would simply move the same overhead.
            marker_block = scoped[columns]
            try:
                matrix = marker_block.to_numpy(dtype=float)
            except (TypeError, ValueError):
                matrix = marker_block.apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)
            for marker_finite, positions in _missingness_groups(matrix, columns):
                usable = marker_finite & target_finite
                n_usable = int(usable.sum())
                if n_usable < MINIMUM_CASES:
                    continue
                if n_usable / max(1, len(scoped)) < MINIMUM_MARKER_COVERAGE:
                    continue
                if n_usable < 10:
                    continue
                target_ranks = pd.Series(target_values[usable]).rank().to_numpy()
                target_ranks -= target_ranks.mean()
                target_sum_squares = float((target_ranks**2).sum())
                if target_sum_squares <= 0:
                    continue
                marker_ranks = np.asarray(
                    scipy_stats.rankdata(
                        matrix[np.ix_(usable, positions)],
                        method="average",
                        axis=0,
                    ),
                    dtype=float,
                )
                marker_ranks -= marker_ranks.mean(axis=0, keepdims=True)
                for column, position in enumerate(positions):
                    ranks = marker_ranks[:, column]
                    denominator = math.sqrt(
                        float((ranks**2).sum()) * target_sum_squares
                    )
                    if denominator <= 0:
                        continue
                    rho = float((ranks * target_ranks).sum() / denominator)
                    rho = max(-0.999999, min(0.999999, rho))
                    statistic = rho * math.sqrt(
                        (n_usable - 2) / (1.0 - rho * rho)
                    )
                    p_value = float(
                        2.0 * scipy_stats.t.sf(abs(statistic), df=n_usable - 2)
                    )
                    if not np.isfinite(p_value):
                        continue
                    records.append(
                        _record(
                            axis=axis,
                            policy=policy,
                            target_id=str(target_id),
                            label=labels.get(str(target_id), str(target_id)),
                            marker_id=columns[position],
                            split=split,
                            n_observations=n_usable,
                            effect=rho,
                            p_value=p_value,
                            n_positive=n_usable,
                            n_negative=n_usable,
                            temporal_relation=str(relation),
                        )
                    )
    return _ordered_records(records, columns)


def _screen_binary(
    cohort: pd.DataFrame,
    presence: pd.DataFrame,
    axis: str,
    policy: BroadAxisPolicy,
    labels: Mapping[str, str],
    marker_ids: Sequence[str],
    *,
    already_indexed: bool = False,
) -> list[dict]:
    if not already_indexed:
        cohort = _cross_sectional_index(cohort)
    records: list[dict] = []
    # One group per (entity, temporal relation): a procedure done before the
    # ECG and the same procedure done after it are separate findings.
    if "temporal_relation" not in presence.columns:
        # An axis whose source carries no date still has to screen; it just
        # cannot claim a direction.
        presence = presence.assign(temporal_relation=RELATION_UNDATED)
    present_by_target = {
        (str(target_id), str(relation)): {
            int(value) for value in block["study_id"].tolist()
        }
        for (target_id, relation), block in presence.groupby(
            ["target_id", "temporal_relation"], sort=True
        )
    }
    for split in DEVELOPMENT_SPLITS:
        scoped = cohort.loc[cohort["split"] == split]
        if len(scoped) < 2 * MINIMUM_CASES:
            continue
        study_ids = scoped["study_id"].to_numpy()
        # One dict lookup per present study instead of one per cohort row:
        # building the mask by iterating all 600,000 study ids for each of 671
        # targets was 400 million Python-level membership tests.
        position_of = {
            int(value): index for index, value in enumerate(study_ids)
        }
        # Rank each marker once per split, not once per (target, marker).
        # The Mann-Whitney AUC is a function of the case ranks alone, so one
        # sort per marker replaces one sort per target: at 671 binary targets
        # and 1,713 markers that is 3,426 sorts instead of 1.15 million, and it
        # is the same number, not an approximation.
        marker_ranks: dict[str, np.ndarray] = {}
        marker_finite: dict[str, np.ndarray] = {}
        marker_tie: dict[str, float] = {}
        for marker_id in marker_ids:
            values = pd.to_numeric(
                scoped[marker_id], errors="coerce"
            ).to_numpy(dtype=float)
            finite = np.isfinite(values)
            marker_finite[marker_id] = finite
            ranks = np.full(len(values), 0.0, dtype=float)
            if finite.any():
                ranks[finite] = _average_ranks(values[finite])
                marker_tie[marker_id] = _tie_term(values[finite])
            else:
                marker_tie[marker_id] = 0.0
            marker_ranks[marker_id] = ranks
        for (target_id, relation), present in present_by_target.items():
            if not present:
                continue
            positions = [
                position_of[study] for study in present if study in position_of
            ]
            if len(positions) < MINIMUM_CASES:
                continue
            mask = np.zeros(len(study_ids), dtype=bool)
            mask[positions] = True
            if int((~mask).sum()) < MINIMUM_CASES:
                continue
            for marker_id in marker_ids:
                finite = marker_finite[marker_id]
                case_mask = mask & finite
                control_mask = (~mask) & finite
                n_cases = int(case_mask.sum())
                n_controls = int(control_mask.sum())
                if n_cases < MINIMUM_CASES or n_controls < MINIMUM_CASES:
                    continue
                if finite.mean() < MINIMUM_MARKER_COVERAGE:
                    continue
                auc, p_value = _rank_auc(
                    marker_ranks[marker_id],
                    case_mask=case_mask,
                    n_cases=n_cases,
                    n_controls=n_controls,
                    tie_term=marker_tie[marker_id],
                )
                if not np.isfinite(p_value):
                    continue
                records.append(
                    _record(
                        axis=axis,
                        policy=policy,
                        target_id=str(target_id),
                        label=labels.get(str(target_id), str(target_id)),
                        marker_id=marker_id,
                        split=split,
                        n_observations=n_cases + n_controls,
                        effect=auc,
                        p_value=p_value,
                        n_positive=n_cases,
                        n_negative=n_controls,
                        temporal_relation=relation,
                    )
                )
    return records


def _spearman(values: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Rank correlation with a tie-aware asymptotic p-value."""

    n = len(values)
    if n < 10:
        return float("nan"), float("nan")
    ranks_a = pd.Series(values).rank().to_numpy()
    ranks_b = pd.Series(target).rank().to_numpy()
    ranks_a -= ranks_a.mean()
    ranks_b -= ranks_b.mean()
    denominator = math.sqrt(float((ranks_a**2).sum()) * float((ranks_b**2).sum()))
    if denominator <= 0:
        return float("nan"), float("nan")
    rho = float((ranks_a * ranks_b).sum() / denominator)
    rho = max(-0.999999, min(0.999999, rho))
    statistic = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    from scipy import stats

    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - 2))
    return rho, p_value


def _axis_source_fingerprint(master: Path, feature_glob: str) -> dict:
    from .storage import feature_manifest_path, file_sha256

    manifest = feature_manifest_path(feature_glob)
    return {
        "master_index_sha256": file_sha256(master) if Path(master).is_file() else None,
        "feature_manifest_sha256": file_sha256(manifest) if manifest.is_file() else None,
    }


def run_broad_axis_screen(
    config: MimicProjectConfig,
    *,
    run_id: str,
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel"),
    feature_glob: Optional[str] = None,
    axes: Optional[tuple[str, ...]] = None,
    max_markers: Optional[int] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Screen every marker against every frequent entity of each axis."""

    import duckdb

    run_root = config.runs_root / run_id / "broad_axes"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "broad_axes.manifest.json"

    policies = {
        policy.axis: policy
        for policy in DEFAULT_POLICIES
        if axes is None or policy.axis in axes
    }
    # The floors are part of the preregistration, not a tuning knob.
    assert_policies_prespecified(
        DEFAULT_POLICIES,
        expected=dict(PRESPECIFIED_FLOORS),
    )
    marker_ids = list(screening_marker_ids(panel_ids))
    if max_markers is not None:
        marker_ids = marker_ids[: int(max_markers)]
    if not marker_ids:
        raise ValueError("No screenable markers in the selected panels")

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": marker_ids,
            "axes": sorted(policies),
            "floors": dict(PRESPECIFIED_FLOORS),
            "train_alpha": TRAIN_FDR_ALPHA,
            "validation_alpha": VALIDATION_FDR_ALPHA,
            "minimum_absolute_rho": MINIMUM_ABSOLUTE_RHO,
            "minimum_absolute_auc_margin": MINIMUM_ABSOLUTE_AUC_MARGIN,
            "broad_event_screen_version": BROAD_EVENT_SCREEN_VERSION,
            "adjustment_version": BROAD_AXIS_ADJUSTMENT_VERSION,
            "primary_adjustment": PRIMARY_COVARIATES,
            "sensitivity_adjustment": SENSITIVITY_COVARIATES,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, BROAD_AXIS_STAGE_VERSION):
            return previous

    master = config.master_root / "ecg_master_index.parquet"
    sources = axis_sources(config)
    target_frames: list[pd.DataFrame] = []
    result_frames: list[pd.DataFrame] = []
    coverages: list[Mapping[str, object]] = []
    skipped: list[dict] = []

    from .storage import isolated_analysis_connection

    with isolated_analysis_connection(config) as connection:
        # One outcome-blind index ECG per patient, materialized once and
        # reused by every binary axis and marker block.  Previously the same
        # cohort was transferred and re-sorted for every block.
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE {_INDEX_COHORT_TABLE} AS
            SELECT
                f.study_id,
                m.subject_id,
                m.ecg_time,
                m.split
            FROM (
                SELECT DISTINCT study_id
                FROM {_feature_parquet(resolved_glob)}
            ) f
            JOIN {_parquet(master)} m USING (study_id)
            WHERE m.split IN {tuple(DEVELOPMENT_SPLITS)}
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY m.subject_id
                ORDER BY m.ecg_time, f.study_id
            ) = 1
            """
        )
        # Every axis uses the same feature store and marker panel. Resolve
        # the wide Parquet schema once; repeating this metadata bind per axis
        # cannot alter the intersection and is costly over many partitions.
        available_marker_ids = available_markers(
            connection,
            resolved_glob,
            marker_ids,
        )
        for axis, policy in sorted(
            policies.items(), key=lambda item: axis_priority(item[0])
        ):
            source = sources.get(axis)
            if source is None or not source.get("available"):
                skipped.append(
                    {
                        "axis": axis,
                        "status": "source_unavailable",
                        "error_type": "SourceUnavailable",
                        "message": str((source or {}).get("reason", "")),
                    }
                )
                continue
            if progress:
                progress(f"Breitachse {axis}: Ziele werden ermittelt")
            try:
                targets, results, coverage = _screen_one_axis(
                    connection,
                    axis=axis,
                    policy=policy,
                    source=source,
                    master_relation=_parquet(master),
                    feature_glob=resolved_glob,
                    marker_ids=available_marker_ids,
                    progress=progress,
                    markers_prevalidated=True,
                    checkpoint_root=run_root / ".axis_blocks",
                    checkpoint_key=content_hash({
                        "input_hash": input_hash,
                        "config_hash": config_hash,
                        "sources": _axis_source_fingerprint(master, resolved_glob),
                    }),
                )
            except Exception as error:  # noqa: BLE001 - one axis must not stop the sweep
                skipped.append(
                    {
                        "axis": axis,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue
            if "axis" not in targets.columns:
                targets.insert(0, "axis", axis)
            target_frames.append(targets)
            coverages.append(coverage)
            if not results.empty:
                result_frames.append(results)

    targets_all = (
        pd.concat(target_frames, ignore_index=True)
        if target_frames
        else pd.DataFrame()
    )
    results_all = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )

    replicated = pd.DataFrame()
    if not results_all.empty:
        # Each axis is its own family, so a new axis does not raise the bar
        # for the phenotype screen or the curated associations.
        # The temporal relation is part of the family: screening "before" and
        # "after" together would let one direction borrow the other's
        # multiplicity budget.
        corrected = apply_fdr_by_family(
            results_all,
            family_columns=("axis", "temporal_relation", "evaluation_split"),
        )
        train_hits = corrected.loc[
            corrected["evaluation_split"].eq("train")
            & corrected["q_value"].le(TRAIN_FDR_ALPHA)
        ]
        selected = set(
            zip(
                train_hits["axis"],
                train_hits["target_id"],
                train_hits["temporal_relation"],
                train_hits["feature_id"],
            )
        )
        validation = corrected.loc[
            corrected["evaluation_split"].eq("validation")
        ].copy()
        validation["selected_on_train"] = [
            key in selected
            for key in zip(
                validation["axis"],
                validation["target_id"],
                validation["temporal_relation"],
                validation["feature_id"],
            )
        ]
        candidates = validation.loc[validation["selected_on_train"]]
        if not candidates.empty:
            candidates = apply_fdr_by_family(
                candidates.drop(columns=["q_value", "fdr_significant"], errors="ignore"),
                family_columns=("axis", "temporal_relation"),
            )
            replicated = candidates.loc[
                candidates["q_value"].le(VALIDATION_FDR_ALPHA)
            ].copy()
            # Direction must agree, or "replicated" means nothing.
            train_direction = {
                (
                    row.axis,
                    row.target_id,
                    row.temporal_relation,
                    row.feature_id,
                ): float(row.effect)
                for row in train_hits.itertuples()
            }
            replicated["train_effect"] = [
                train_direction.get(key, float("nan"))
                for key in zip(
                    replicated["axis"],
                    replicated["target_id"],
                    replicated["temporal_relation"],
                    replicated["feature_id"],
                )
            ]
            replicated["direction_consistent"] = [
                bool(
                    np.isfinite(train)
                    and np.isfinite(validation_effect)
                    and (train - _neutral(kind)) * (validation_effect - _neutral(kind))
                    > 0
                )
                for train, validation_effect, kind in zip(
                    replicated["train_effect"],
                    replicated["effect"],
                    replicated["target_kind"],
                )
            ]
            replicated = replicated.loc[replicated["direction_consistent"]]
            # The prespecified effect floor, applied last and recorded as its
            # own column rather than silently dropping rows: a hit that
            # replicated but is too small to matter is a fact about the
            # screen, and hiding it would make the counts unreadable.
            replicated["meets_effect_floor"] = [
                meets_effect_floor(effect, kind)
                and meets_effect_floor(train, kind)
                for effect, train, kind in zip(
                    replicated["effect"],
                    replicated["train_effect"],
                    replicated["target_kind"],
                )
            ]
            replicated["effect_floor"] = [
                _effect_floor(kind) for kind in replicated["target_kind"]
            ]
        results_all = corrected

    coverage_report = (
        combined_coverage_report(coverages) if coverages else pd.DataFrame()
    )
    # These discovery intermediates are no longer used. Retain only the actual
    # output frames and candidate list while models allocate their designs.
    result_frames.clear()
    target_frames.clear()
    results = targets = None
    if not results_all.empty:
        del train_hits, selected, validation, candidates
        if 'train_direction' in locals():
            train_direction.clear()
    import gc
    gc.collect()
    adjusted = _adjust_replicated_hits(
        config,
        replicated=replicated,
        feature_glob=resolved_glob,
        sources=sources,
        progress=progress,
        checkpoint_root=run_root / '.adjustment_packets',
        checkpoint_input_hash=input_hash,
    )
    replicated = _annotate_adjusted_confirmation(replicated, adjusted)
    family = (
        screen_family_size(targets_all, n_markers=len(marker_ids))
        if not targets_all.empty
        else {}
    )

    paths = {
        "targets": run_root / "broad_axis_targets.parquet",
        "screening": run_root / "broad_axis_screening.parquet",
        "replicated": run_root / "broad_axis_replicated_hits.parquet",
        "adjusted": run_root / "broad_axis_adjusted_results.parquet",
        "coverage": run_root / "broad_axis_coverage.parquet",
    }
    atomic_write_parquet(targets_all, paths["targets"])
    atomic_write_parquet(results_all, paths["screening"])
    atomic_write_parquet(replicated, paths["replicated"])
    atomic_write_parquet(adjusted, paths["adjusted"])
    atomic_write_parquet(coverage_report, paths["coverage"])

    summary = {
        "run_id": run_id,
        "axes_screened": sorted(policies),
        "axes_skipped": skipped,
        "patient_index_policy": (
            "continuous=earliest_endpoint_observed_ecg_per_subject_relation;"
            "binary=outcome_blind_earliest_ecg_per_subject"
        ),
        "future_measurement_policy": (
            "post_ecg_values_are_labeled_after_ecg_and_are_not_predictor_inputs"
        ),
        "n_markers": len(marker_ids),
        "n_targets_total": int(len(targets_all)),
        "n_targets_eligible": (
            int(targets_all["eligible_for_screen"].astype(bool).sum())
            if not targets_all.empty
            else 0
        ),
        "n_tests": int(len(results_all)),
        "n_replicated_hits": int(len(replicated)),
        "n_replicated_hits_above_effect_floor": (
            int(replicated["meets_effect_floor"].astype(bool).sum())
            if "meets_effect_floor" in replicated
            else 0
        ),
        "n_adjusted_models": int(len(adjusted)),
        "n_adjusted_confirmed_hits": (
            int(replicated["adjusted_confirmed"].astype(bool).sum())
            if "adjusted_confirmed" in replicated
            else 0
        ),
        "adjustment_version": BROAD_AXIS_ADJUSTMENT_VERSION,
        "primary_adjustment": list(PRIMARY_COVARIATES),
        "sensitivity_adjustment": list(SENSITIVITY_COVARIATES),
        "minimum_absolute_rho": MINIMUM_ABSOLUTE_RHO,
        "minimum_absolute_auc_margin": MINIMUM_ABSOLUTE_AUC_MARGIN,
        "effect_floor_prespecified": True,
        "family": dict(family),
        "train_fdr_alpha": TRAIN_FDR_ALPHA,
        "validation_fdr_alpha": VALIDATION_FDR_ALPHA,
        "fdr_family_definition": "per_axis_not_pooled_across_axes",
        "splits_used": list(DEVELOPMENT_SPLITS),
        "locked_test_read": False,
        "note_en": (
            "Frequency is the only inclusion rule and it was fixed before the "
            "screen. Entities below the floor are reported as underpowered, "
            "not dropped. The ICU axis carries invasively measured quantities, "
            "so a hit there is a surrogate question rather than an association."
        ),
        "broad_axis_stage_version": BROAD_AXIS_STAGE_VERSION,
        "operational_config_hash": operational_config_hash(config),
    }
    summary_path = run_root / "broad_axis_summary.json"
    atomic_write_json(summary_path, summary)

    output_paths = tuple(
        str(path.resolve()) for path in [*paths.values(), summary_path]
    )
    manifest = StageManifest.complete(
        stage="screen-broad-axes",
        stage_version=BROAD_AXIS_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=output_paths,
        row_counts={
            "targets": int(len(targets_all)),
            "screening": int(len(results_all)),
            "replicated": int(len(replicated)),
            "adjusted": int(len(adjusted)),
        },
        status=(
            "complete"
            if not skipped and not targets_all.empty
            else "incomplete_axis_failures"
        ),
        notes=(
            f"broad_event_screen_version={BROAD_EVENT_SCREEN_VERSION}",
            "fdr_family=per_axis",
            "locked_test_read=False",
            f"adjustment_version={BROAD_AXIS_ADJUSTMENT_VERSION}",
            'prepared_nullable_design_encoding=numpy_float64_v1; legacy_dtype_failure_packets_preserved',
        ),
        output_integrity=capture_output_integrity(
            tuple(Path(path) for path in output_paths)
        ),
    )
    manifest.save(manifest_path)
    # Preserve both screening and adjustment checkpoints after publication.
    if skipped or targets_all.empty:
        details = "; ".join(
            f"{item['axis']}:{item['status']}:{item.get('error_type', '')}"
            for item in skipped
        ) or "no eligible axis produced targets"
        raise RuntimeError(
            "Broad-axis screen incomplete; partial artifacts and a structured "
            f"manifest were written for diagnosis: {details}"
        )
    return manifest


def _neutral(target_kind: object) -> float:
    """The value an effect takes when there is nothing to find."""

    return 0.5 if str(target_kind) == "binary" else 0.0


__all__ = [
    "BROAD_AXIS_STAGE_VERSION",
    "DEVELOPMENT_SPLITS",
    "PRESPECIFIED_FLOORS",
    "TRAIN_FDR_ALPHA",
    "VALIDATION_FDR_ALPHA",
    "assert_no_sealed_rows",
    "axis_sources",
    "run_broad_axis_screen",
    "screening_marker_ids",
]
