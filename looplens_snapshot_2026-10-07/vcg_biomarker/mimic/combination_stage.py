"""Pipeline stage: how deep marker combinations are worth taking.

Runs the adaptive beam search of :mod:`combination_search` for every endpoint
and writes the depth curve -- what each additional component actually bought,
measured on patients the selection never saw.

Placed last in the guided run on purpose. It is the only stage whose runtime is
open-ended by design, and it needs the extracted features that everything
before it produces. Because it writes after every depth level, stopping it at
any moment leaves a complete result for every depth already finished; nothing
downstream waits on it.

Stopping it early is therefore cheap, and there are three ways to do it:

* let it stop itself, which is the default -- the search ends when extra depth
  stops buying held-out discrimination;
* give it a wall-clock or evaluation budget;
* drop a stop file into the run directory, which is the mode for "let it grind
  and I will say when".

The sealed test is not read here, and depth alone can never promote a result:
anything past three components is reported as a composite score rather than a
criterion, because that is what it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .combination_search import (
    DEEP_COMBINATION_VERSION,
    DepthLevelResult,
    SearchBudget,
    beam_search_combinations,
    depth_profile as depth_profile,
    recommended_depth,
    search_cost_comparison,
)
from .config import MimicProjectConfig
from .feature_roles import is_combination_candidate
from .panels import _feature_family, default_feature_panels
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    development_feature_glob,
    feature_manifest_path,
    feature_parquet_clause,
    input_artifact_fingerprints,
    operational_config_hash,
    parquet_columns,
    scientific_config_hash,
)


from .refinement import (  # noqa: F401 - re-exported for existing callers
    DEFAULT_STOP_MODE,
    STOP_FILE_NAME,
    STOP_MODES,
    STOP_MODE_AFTER_ENDPOINT,
    STOP_MODE_AFTER_SWEEP,
    STOP_MODE_IMMEDIATE,
    clear_stop,
    request_stop,
    requested_stop_mode,
    stop_file_requested,
)


COMBINATION_STAGE_VERSION = "1.3.0"

# The stop file, the stop modes and the live state now live in
# ``refinement``, because the marker bootstrap is open-ended in the same way
# and a user should not have to learn two controls for one activity. The names
# are re-exported so existing callers and tests keep working, and the file
# names are unchanged so a stop written by a new GUI still reaches a search
# started from older code.
#
# Three ways to end an open-ended search, because they trade different things.
# The search is endpoint-major: it takes one endpoint down to its own stopping
# depth before moving to the next. Aborting in the middle therefore leaves the
# first endpoints fully searched and the rest with no combinations at all --
# an inhomogeneous result set in which "no combination found" and "never
# looked" are indistinguishable. Only ``after_sweep`` avoids that, which is
# why it is the default.
LIVE_FILE_NAME = "deep_search_live.json"

# Development splits only. Named rather than implied, because this stage is the
# one a user is most likely to run by hand, repeatedly, at odd hours.
DEVELOPMENT_SPLITS: tuple[str, ...] = ("train",)





def read_live_state(run_root: Path) -> Optional[dict]:
    """Current endpoint, current depth and the estimate for the next level."""

    path = Path(run_root) / LIVE_FILE_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def describe_live_state(state: Optional[Mapping[str, object]]) -> str:
    """One readable line for a status bar."""

    if not state:
        return "Tiefensuche: kein Live-Status verfügbar."
    endpoint = state.get("endpoint_id") or "?"
    position = state.get("endpoint_position") or 0
    total = state.get("endpoints_total") or 0
    depth = state.get("current_depth") or 0
    next_seconds = state.get("estimated_next_level_seconds")
    next_seconds_value = (
        float(str(next_seconds)) if next_seconds is not None else None
    )
    estimate = (
        f"nächste Stufe ca. {next_seconds_value:.0f} s"
        if next_seconds_value is not None and next_seconds_value > 0
        else "nächste Stufe noch nicht abschätzbar"
    )
    pending = state.get("stop_mode_pending")
    suffix = f" | Abbruch angefordert: {pending}" if pending else ""
    return (
        f"Endpunkt {position}/{total} ({endpoint}) | Tiefe {depth} | {estimate}{suffix}"
    )


def aligned_out_of_fold_scores(
    frame: pd.DataFrame,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    marker_ids: Sequence[str],
    *,
    folds: int = 5,
    random_seed: int = 20260814,
    minimum_available: float = 0.80,
) -> dict[str, np.ndarray]:
    """Turn raw marker values into direction-aligned, out-of-fold z-scores.

    Each marker is standardised and oriented on the fitting folds and scored on
    the held-out fold, so a marker never contributes a value that was scaled
    using its own row. Markers whose direction is unstable across folds are
    dropped: a marker that points one way in three folds and the other way in
    two has no direction to align, and averaging it into a composite adds noise
    with a sign.
    """

    labels = np.asarray(labels, dtype=np.int8)
    subjects = np.asarray(subject_ids)
    if np.unique(labels).size < 2:
        return {}
    try:
        splitter = StratifiedGroupKFold(
            n_splits=int(folds),
            shuffle=True,
            random_state=int(random_seed),
        )
        split_pairs = list(
            splitter.split(np.zeros(len(labels)), labels, groups=subjects)
        )
    except ValueError:
        return {}

    aligned: dict[str, np.ndarray] = {}
    for marker_id in marker_ids:
        if marker_id not in frame.columns:
            continue
        values = pd.to_numeric(frame[marker_id], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isfinite(values).mean() < minimum_available:
            continue
        scores: np.ndarray = np.full(len(values), np.nan, dtype=float)
        directions: list[int] = []
        for fit_index, holdout_index in split_pairs:
            fit_valid = np.isfinite(values[fit_index])
            holdout_valid = np.isfinite(values[holdout_index])
            fit_labels = labels[fit_index][fit_valid]
            if np.unique(fit_labels).size < 2 or not holdout_valid.any():
                continue
            fit_values = values[fit_index][fit_valid]
            deviation = float(np.std(fit_values, ddof=1))
            if not np.isfinite(deviation) or deviation <= 0:
                continue
            direction = (
                1 if roc_auc_score(fit_labels, fit_values) >= 0.5 else -1
            )
            directions.append(direction)
            scores[holdout_index[holdout_valid]] = direction * (
                (values[holdout_index][holdout_valid] - float(np.mean(fit_values)))
                / deviation
            )
        if not directions:
            continue
        dominant = max(directions.count(1), directions.count(-1))
        if dominant < int(np.ceil(0.80 * len(directions))):
            continue
        if not np.isfinite(scores).any():
            continue
        aligned[marker_id] = scores
    return aligned


def _load_endpoint_frame(
    config: MimicProjectConfig,
    *,
    feature_glob: str,
    endpoint_id: str,
    marker_ids: Sequence[str],
    max_rows: Optional[int],
) -> pd.DataFrame:
    labels_path = (
        config.labels_root / "clinical_labels_wide.parquet"
    ).as_posix().replace("'", "''")
    feature_sql = feature_glob.replace("\\", "/").replace("'", "''")
    state_column = '"' + f"{endpoint_id}_state".replace('"', '""') + '"'
    splits = ", ".join(f"'{value}'" for value in DEVELOPMENT_SPLITS)
    limit = f" LIMIT {int(max_rows)}" if max_rows else ""
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True, read_only=True) as connection:
        available = parquet_columns(connection, feature_sql)
        selected = tuple(
            marker_id for marker_id in marker_ids if marker_id in available
        )
        if not selected:
            return pd.DataFrame()
        marker_select = ",\n            ".join(
            "f.\"" + marker_id.replace('"', '""') + "\""
            for marker_id in selected
        )
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
        query = f"""
            SELECT
                f.subject_id,
                f.study_id,
                {marker_select},
                CASE WHEN l.{state_column} = 'positive' THEN 1 ELSE 0 END
                    AS label
            FROM {read_features} f
            JOIN read_parquet('{labels_path}') l
              USING (subject_id, study_id)
            WHERE f.split IN ({splits})
              AND l.split IN ({splits})
              AND l.{state_column} IN ('positive', 'negative')
              AND f.extraction_status = 'ok'
            ORDER BY f.subject_id, f.study_id
            {limit}
        """
        return connection.execute(query).fetchdf()


def available_endpoints(config: MimicProjectConfig) -> tuple[str, ...]:
    """Endpoints that have a usable state column in the label table."""

    from .storage import isolated_analysis_connection

    labels_path = (config.labels_root / "clinical_labels_wide.parquet").as_posix()
    with isolated_analysis_connection(config) as connection:
        columns = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{labels_path}')"
        ).fetchdf()["column_name"].tolist()
    return tuple(
        column[: -len("_state")]
        for column in columns
        if column.endswith("_state")
    )


def run_deep_combination_search(
    config: MimicProjectConfig,
    *,
    run_id: str,
    endpoint_ids: Optional[tuple[str, ...]] = None,
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel"),
    feature_glob: Optional[str] = None,
    max_depth: Optional[int] = 12,
    patience: Optional[int] = 2,
    minimum_increment: float = 0.002,
    beam_width: int = 8,
    seconds_per_endpoint: Optional[float] = None,
    run_until_stopped: bool = False,
    max_rows_per_endpoint: Optional[int] = None,
    minimum_positive: int = 100,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Search combination depth adaptively for every endpoint.

    ``run_until_stopped`` removes both the depth cap and the automatic stopping
    rule, leaving the stop file, the time budget and the supply of unused
    marker families as the only limits. Results are written after every depth
    level, so ending it is always safe.
    """

    run_root = config.runs_root / run_id / "deep_combinations"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = run_root / "deep_combinations.manifest.json"
    # Starting the stage *is* the intent to search. A stop file left from an
    # earlier run would otherwise end this one before it began -- silently,
    # since a stop is a legitimate outcome rather than an error.
    try:
        (run_root / STOP_FILE_NAME).unlink(missing_ok=True)
    except OSError:
        pass

    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    marker_ids: list[str] = []
    for panel_id in panel_ids:
        panel = panels.get(panel_id)
        if panel is None:
            raise KeyError(f"Unknown feature panel: {panel_id}")
        marker_ids.extend(panel.feature_ids)
    marker_ids = [
        marker_id
        for marker_id in dict.fromkeys(marker_ids)
        if is_combination_candidate(marker_id)
    ]
    if not marker_ids:
        raise ValueError("No combination-eligible markers in the chosen panels")

    resolved_glob = feature_glob or development_feature_glob(
        config.features_root, panel_ids
    )
    if endpoint_ids is None:
        endpoint_ids = available_endpoints(config)
    endpoint_ids = tuple(dict.fromkeys(endpoint_ids))
    if not endpoint_ids:
        raise ValueError("No endpoints available for the combination search")

    if run_until_stopped:
        max_depth = None
        patience = None
    budget_template = SearchBudget(
        max_depth=max_depth,
        patience=patience,
        minimum_increment=minimum_increment,
        seconds=seconds_per_endpoint,
    )

    input_hash = content_hash(
        {
            "run_id": run_id,
            "feature_glob": resolved_glob,
            "upstream_artifacts": input_artifact_fingerprints(
                {
                    "feature_manifest": feature_manifest_path(resolved_glob),
                    "labels_manifest": (
                        config.labels_root / "labels.manifest.json"
                    ),
                }
            ),
            "markers": marker_ids,
            "endpoints": list(endpoint_ids),
            "budget": budget_template.to_dict(),
            "beam_width": beam_width,
            "deep_combination_version": DEEP_COMBINATION_VERSION,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            COMBINATION_STAGE_VERSION,
        ):
            return previous

    profile_rows: list[dict] = []
    summary_rows: list[dict] = []
    skipped: list[dict] = []
    stopped_early = False
    stop_mode_seen: Optional[str] = None
    # Set once a stop asks the sweep to finish shallow: every remaining
    # endpoint is then searched only this deep, so all of them end up with a
    # comparable result instead of half the list having none at all.
    sweep_depth_cap: Optional[int] = None
    live_path = run_root / LIVE_FILE_NAME

    def publish(
        *,
        endpoint_id: str,
        position: int,
        depth: int,
        level_seconds: Optional[float],
        next_seconds: Optional[float],
        phase: str,
    ) -> None:
        try:
            atomic_write_json(
                live_path,
                {
                    "run_id": run_id,
                    "endpoint_id": endpoint_id,
                    "endpoint_position": position,
                    "endpoints_total": len(endpoint_ids),
                    "endpoints_completed": len(summary_rows),
                    "current_depth": int(depth),
                    "last_level_seconds": level_seconds,
                    "estimated_next_level_seconds": next_seconds,
                    "phase": phase,
                    "stop_mode_pending": stop_mode_seen,
                    "sweep_depth_cap": sweep_depth_cap,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "deep_combination_version": DEEP_COMBINATION_VERSION,
                },
            )
        except OSError:
            # A status file that cannot be written must never stop the search.
            pass

    for position, endpoint_id in enumerate(endpoint_ids, start=1):
        mode = requested_stop_mode(run_root)
        if mode is not None:
            stop_mode_seen = mode
            stopped_early = True
            if mode in (STOP_MODE_IMMEDIATE, STOP_MODE_AFTER_ENDPOINT):
                # Both have already finished whatever they promised to finish
                # by the time control is back here.
                if progress:
                    progress(
                        f"Abbruch ({mode}) vor {endpoint_id} - "
                        f"{position - 1}/{len(endpoint_ids)} ausgewertet"
                    )
                break
            if sweep_depth_cap is None:
                # Finish the sweep, but only as deep as the shallowest
                # endpoint already reached, so the remaining ones stay
                # comparable and the rest of the run stays short.
                reached = [
                    int(row.get("best_depth", 0) or 0) for row in summary_rows
                ]
                sweep_depth_cap = max(2, min(reached) if reached else 2)
                if progress:
                    progress(
                        f"Abbruch nach Durchlauf angefordert: die restlichen "
                        f"{len(endpoint_ids) - position + 1} Endpunkte werden "
                        f"noch bis Tiefe {sweep_depth_cap} ausgewertet, damit "
                        "keiner ohne Kombinationen bleibt"
                    )
        if progress:
            progress(
                f"Tiefensuche {position}/{len(endpoint_ids)}: {endpoint_id}"
            )
        publish(
            endpoint_id=endpoint_id,
            position=position,
            depth=0,
            level_seconds=None,
            next_seconds=None,
            phase="loading",
        )
        try:
            frame = _load_endpoint_frame(
                config,
                feature_glob=resolved_glob,
                endpoint_id=endpoint_id,
                marker_ids=tuple(marker_ids),
                max_rows=max_rows_per_endpoint,
            )
        except Exception as error:  # noqa: BLE001 - one endpoint must not stop the stage
            skipped.append(
                {
                    "endpoint_id": endpoint_id,
                    "reason": f"load_failed: {error}",
                }
            )
            continue
        if frame.empty or "subject_id" not in frame.columns:
            skipped.append({"endpoint_id": endpoint_id, "reason": "no_rows"})
            continue
        labels = frame["label"].to_numpy(dtype=np.int8)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if min(positives, negatives) < int(minimum_positive):
            skipped.append(
                {
                    "endpoint_id": endpoint_id,
                    "reason": "underpowered_here_not_uninformative",
                    "n_positive": positives,
                    "n_negative": negatives,
                }
            )
            continue

        aligned = aligned_out_of_fold_scores(
            frame,
            labels,
            frame["subject_id"].to_numpy(),
            marker_ids,
        )
        if len(aligned) < 2:
            skipped.append(
                {"endpoint_id": endpoint_id, "reason": "no_alignable_markers"}
            )
            continue

        effective_depth = (
            sweep_depth_cap
            if sweep_depth_cap is not None
            else max_depth
        )

        def level_stop_requested() -> bool:
            # Only an immediate stop ends the search between levels. The other
            # two modes have promised to finish something first.
            return requested_stop_mode(run_root) == STOP_MODE_IMMEDIATE

        budget = SearchBudget(
            max_depth=effective_depth,
            patience=patience,
            minimum_increment=minimum_increment,
            seconds=seconds_per_endpoint,
            stop_requested=level_stop_requested,
        )
        level_seconds: list[float] = []

        def record_level(level: DepthLevelResult) -> None:
            profile_rows.append({"endpoint_id": endpoint_id, **level.to_dict()})
            # Written after every level: stopping the search is always safe.
            atomic_write_parquet(
                pd.DataFrame(profile_rows),
                run_root / "depth_profile.parquet",
            )
            if level.depth > 1:
                level_seconds.append(float(level.elapsed_seconds))
            # Each level costs beam_width x representatives evaluations, so
            # the level just finished is the best estimate for the next one.
            publish(
                endpoint_id=endpoint_id,
                position=position,
                depth=int(level.depth),
                level_seconds=float(level.elapsed_seconds),
                next_seconds=(
                    float(np.median(level_seconds[-3:]))
                    if level_seconds
                    else None
                ),
                phase="searching",
            )

        levels = beam_search_combinations(
            aligned,
            labels,
            frame["subject_id"].to_numpy(),
            family_for_feature=_feature_family,
            budget=budget,
            beam_width=int(beam_width),
            worker_count=min(2, max(1, int(getattr(config, "workers", 1)))),
            on_depth_complete=record_level,
            progress=(
                (lambda message: progress(f"  {endpoint_id}: {message}"))
                if progress
                else None
            ),
        )
        if not levels:
            skipped.append(
                {"endpoint_id": endpoint_id, "reason": "search_returned_nothing"}
            )
            continue
        summary_rows.append(
            {
                "endpoint_id": endpoint_id,
                "n_rows": int(len(frame)),
                "n_positive": positives,
                "n_markers_aligned": len(aligned),
                **recommended_depth(levels),
            }
        )
        if requested_stop_mode(run_root) == STOP_MODE_AFTER_ENDPOINT:
            stop_mode_seen = STOP_MODE_AFTER_ENDPOINT
            stopped_early = True
            if progress:
                progress(
                    f"Abbruch nach Endpunkt: {endpoint_id} vollständig, "
                    f"{position}/{len(endpoint_ids)} ausgewertet"
                )
            break

    profile = pd.DataFrame(profile_rows)
    summary = pd.DataFrame(summary_rows)
    profile_path = run_root / "depth_profile.parquet"
    summary_path = run_root / "depth_summary.parquet"
    atomic_write_parquet(profile, profile_path)
    atomic_write_parquet(summary, summary_path)

    cost_path = run_root / "search_cost_comparison.parquet"
    atomic_write_parquet(search_cost_comparison(len(marker_ids)), cost_path)

    report = {
        "run_id": run_id,
        "endpoints_requested": len(endpoint_ids),
        "endpoints_searched": int(len(summary)),
        "endpoints_skipped": len(skipped),
        "skipped": skipped,
        "budget": budget_template.to_dict(),
        "beam_width": int(beam_width),
        "stopped_by_stop_file": stopped_early,
        "stop_mode": stop_mode_seen,
        "sweep_depth_cap_after_stop": sweep_depth_cap,
        # The homogeneity of the result set, stated rather than left to be
        # inferred: an endpoint that was never reached and one where no
        # combination beat its best component look identical in the tables.
        "sweep_complete": bool(
            len(summary_rows) + len(skipped) == len(endpoint_ids)
        ),
        "endpoints_never_reached": [
            endpoint_id
            for endpoint_id in endpoint_ids
            if endpoint_id
            not in {str(row["endpoint_id"]) for row in summary_rows}
            and endpoint_id not in {str(row["endpoint_id"]) for row in skipped}
        ],
        "depth_reached_by_endpoint": {
            str(row["endpoint_id"]): int(row.get("best_depth", 0) or 0)
            for row in summary_rows
        },
        "minimum_depth_reached": (
            int(min(int(row.get("best_depth", 0) or 0) for row in summary_rows))
            if summary_rows
            else 0
        ),
        "splits_used": list(DEVELOPMENT_SPLITS),
        "locked_test_read": False,
        "measurement_role": "inner_selection_not_independent_confirmation",
        "evidence_stage": "exploratory",
        "clinical_claim_allowed": False,
        "median_recommended_depth": (
            float(summary["recommended_depth"].median())
            if not summary.empty
            else float("nan")
        ),
        "note_en": (
            "Depth is tuned on train-only inner selection data. Legacy confirmation "
            "columns are not independent validation. Depth describes formula "
            "complexity only; no depth or AUC authorizes a clinical claim."
        ),
        "deep_combination_version": DEEP_COMBINATION_VERSION,
    }
    report_path = run_root / "deep_combination_summary.json"
    atomic_write_json(report_path, report)

    output_paths = (
        str(profile_path.resolve()),
        str(summary_path.resolve()),
        str(cost_path.resolve()),
        str(report_path.resolve()),
    )
    manifest = StageManifest.complete(
        stage="deep-combination-search",
        stage_version=COMBINATION_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=output_paths,
        row_counts={
            "depth_profile": int(len(profile)),
            "depth_summary": int(len(summary)),
        },
        status=(
            "complete"
            if not stopped_early and not summary.empty
            else "stopped_by_user_sweep_complete"
            if stopped_early and report["sweep_complete"]
            else "stopped_by_user_partial_sweep"
            if stopped_early
            else "complete_without_searchable_endpoints"
        ),
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"deep_combination_version={DEEP_COMBINATION_VERSION}",
            "depth_selection=inner_selection_gain_not_independent_confirmation",
            "locked_test_read=False",
        ),
        output_integrity=capture_output_integrity(
            tuple(Path(path) for path in output_paths)
        ),
    )
    manifest.save(manifest_path)
    # A finished run must not leave a live file that says "searching".
    try:
        live_path.unlink(missing_ok=True)
    except OSError:
        pass
    return manifest


__all__ = [
    "COMBINATION_STAGE_VERSION",
    "DEVELOPMENT_SPLITS",
    "STOP_FILE_NAME",
    "aligned_out_of_fold_scores",
    "available_endpoints",
    "run_deep_combination_search",
    "stop_file_requested",
]
