"""Pipeline stage: measurement repeatability and within-patient trajectories.

Assembles the replicate-pair set and the trajectory frame from the extracted
features, then writes the repeatability table that the criterion catalog's open
limitation has been waiting for ("technische Wiederholbarkeit muss separat
quantifiziert werden, bevor ein Freeze erlaubt ist").

Runs on train and validation only. Every artifact carries the estimand
explicitly, because the numbers here are upper bounds on measurement error
rather than device precision: two recordings minutes apart still contain real
physiological movement, and nothing in this data can separate the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .panels import default_feature_panels
from .repeatability import (
    MINIMUM_RELIABLE_ICC,
    REPEATABILITY_VERSION,
    ReplicatePairPolicy,
    repeatability_table,
    repeatability_table_from_parquet,
    summarize_repeatability,
)
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    development_feature_glob,
    operational_config_hash,
    scientific_config_hash,
)
from .trajectory import (
    TRAJECTORY_VERSION,
    TrajectoryOptions,
    build_replicate_pairs,
    build_trajectory_frame,
    observation_intensity_diagnostics,
    observation_intensity_weights,
)


REPEATABILITY_STAGE_VERSION = "1.2.0"


def run_repeatability_analysis(
    config: MimicProjectConfig,
    *,
    run_id: str,
    feature_glob: Optional[str] = None,
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel"),
    policy: ReplicatePairPolicy = ReplicatePairPolicy(),
    options: TrajectoryOptions = TrajectoryOptions(),
    max_markers: Optional[int] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Produce replicate pairs, repeatability statistics and trajectories."""

    run_root = config.runs_root / run_id / "repeatability"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "repeatability.manifest.json"

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    marker_ids: list[str] = []
    for panel_id in panel_ids:
        panel = panels.get(panel_id)
        if panel is None:
            raise KeyError(f"Unknown feature panel: {panel_id}")
        marker_ids.extend(panel.feature_ids)
    marker_ids = list(dict.fromkeys(marker_ids))
    if max_markers is not None:
        marker_ids = marker_ids[: int(max_markers)]
    if not marker_ids:
        raise ValueError("No markers selected for the repeatability stage")

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "markers": marker_ids,
            "policy": policy.to_dict(),
            "repeatability_version": REPEATABILITY_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, REPEATABILITY_STAGE_VERSION):
            return previous

    # A directory of part files, not one wide table: the pair artifact carries
    # two columns per marker (2 x 1,753 = 3,506) and building it in one query
    # made DuckDB hold both self-join build sides at 9 GB each.
    pairs_path = run_root / "replicate_pairs"
    build_replicate_pairs(
        config,
        feature_glob=resolved_glob,
        marker_ids=tuple(marker_ids),
        output_path=pairs_path,
        policy=policy,
        progress=progress,
    )
    # The pair artifact carries two columns per marker -- 2 x 1,753 = 3,506,
    # the widest table this project writes. Reading it whole put every pair
    # times every marker into memory at once. The row count comes from the
    # footer instead, which costs nothing.
    import pyarrow.parquet as pq

    from .trajectory import replicate_pair_parts

    parts = replicate_pair_parts(pairs_path)
    n_pairs = int(pq.ParquetFile(parts[0]).metadata.num_rows) if parts else 0
    # The key columns are identical in every part, so the summary statistics
    # come from the first one instead of from the whole artifact.
    pair_keys = (
        pq.read_table(
            parts[0],
            columns=["subject_id", "intervening_intervention_checked"],
        ).to_pandas()
        if parts
        else pd.DataFrame(
            columns=["subject_id", "intervening_intervention_checked"]
        )
    )
    if progress:
        progress(
            f"Wiederholbarkeit: {n_pairs:,} Replikatpaare, "
            f"{len(marker_ids):,} Marker auswerten"
        )
    table = repeatability_table_from_parquet(pairs_path, marker_ids)
    repeatability_path = run_root / "marker_repeatability.parquet"
    atomic_write_parquet(table, repeatability_path)

    trajectory_path = run_root / "marker_trajectories.parquet"
    build_trajectory_frame(
        config,
        feature_glob=resolved_glob,
        marker_ids=tuple(marker_ids),
        output_path=trajectory_path,
        options=options,
        progress=progress,
    )
    # The trajectory file holds one row per subject and marker -- about 240
    # million rows for the full cohort. Only its row count and one ECG count
    # per subject are needed here, so neither is read into pandas whole.
    n_trajectory_rows = int(pq.ParquetFile(trajectory_path).metadata.num_rows)
    if n_trajectory_rows:
        import duckdb

        with duckdb.connect() as reader:
            reader.execute(
                f"SET memory_limit='{max(1, int(config.duckdb_memory_limit_gb))}GB'"
            )
            counts = reader.execute(
                "SELECT subject_id, first(n_ecgs) AS n_ecgs "
                f"FROM read_parquet('{str(trajectory_path).replace(chr(92), '/')}') "
                "GROUP BY subject_id ORDER BY subject_id"
            ).df().set_index("subject_id")["n_ecgs"]
    else:
        counts = pd.Series(dtype=float)
    intensity = observation_intensity_weights(counts.to_numpy())
    summary = {
        "run_id": run_id,
        "n_replicate_pairs": int(n_pairs),
        "n_subjects_with_pairs": (
            int(pair_keys["subject_id"].nunique()) if not pair_keys.empty else 0
        ),
        "intervening_intervention_checked": (
            bool(pair_keys["intervening_intervention_checked"].all())
            if not pair_keys.empty
            and "intervening_intervention_checked" in pair_keys.columns
            else False
        ),
        "replicate_policy": policy.to_dict(),
        "n_trajectory_rows": n_trajectory_rows,
        "n_subjects_with_trajectory": int(len(counts)),
        "observation_intensity": {
            key: value
            for key, value in intensity.items()
            if key != "weights"
        },
        "minimum_reliable_icc": MINIMUM_RELIABLE_ICC,
        **summarize_repeatability(table),
    }
    summary_path = run_root / "repeatability_summary.json"
    atomic_write_json(summary_path, summary)

    # The replicate pairs are a directory of part files, one per marker
    # block; the integrity record is taken per part. Listing the directory
    # itself ended the first production-path preview after 26 minutes of
    # trajectory work with "Stage output file is missing".
    output_paths = (
        *(str(part.resolve()) for part in replicate_pair_parts(pairs_path)),
        str(repeatability_path.resolve()),
        str(trajectory_path.resolve()),
        str(summary_path.resolve()),
    )
    manifest = StageManifest.complete(
        stage="repeatability",
        stage_version=REPEATABILITY_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=output_paths,
        row_counts={
            "replicate_pairs": int(n_pairs),
            "marker_repeatability": int(len(table)),
            "marker_trajectories": n_trajectory_rows,
        },
        status=(
            "complete"
            if n_pairs > 0
            else "complete_without_replicate_pairs"
        ),
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"repeatability_version={REPEATABILITY_VERSION}",
            f"trajectory_version={TRAJECTORY_VERSION}",
            "estimand=upper_bound_on_measurement_error",
            "locked_test_read=False",
        ),
        output_integrity=capture_output_integrity(
            tuple(Path(path) for path in output_paths)
        ),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "REPEATABILITY_STAGE_VERSION",
    "run_repeatability_analysis",
]
