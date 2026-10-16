from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .feature_extraction import feature_implementation_hash
from .models import ResultQuery
from .phenome import PHENOME_STAGE_VERSION
from .storage import DuckDBCatalog, StageManifest
from .user_guidance import annotate_result_display
from .validation import VALIDATION_STAGE_VERSION
from .validation_provenance import validation_implementation_hash


RESULT_QUERY_VERSION = "1.2.0"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultRunDescriptor:
    run_id: str
    validation_status: str
    validation_stage_version: str
    validation_compatible: bool
    analysis_completeness: str
    phenome_status: str
    phenome_compatible: bool
    updated_at: str
    run_root: Path

    @property
    def complete_compatible(self) -> bool:
        return (
            self.validation_status == "complete"
            and self.validation_compatible
            and self.analysis_completeness == "complete_snapshot"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_root"] = str(self.run_root)
        data["complete_compatible"] = self.complete_compatible
        return data


@dataclass(frozen=True)
class ResultPage:
    run_id: str
    perspective: str
    artifact_path: Optional[Path]
    rows: pd.DataFrame
    total_count: int
    offset: int
    limit: int
    snapshot_hash: str
    analysis_completeness: str
    status: str
    message_de: str = ""
    unsupported_filters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "perspective": self.perspective,
            "artifact_path": (
                str(self.artifact_path)
                if self.artifact_path is not None
                else None
            ),
            "total_count": self.total_count,
            "offset": self.offset,
            "limit": self.limit,
            "snapshot_hash": self.snapshot_hash,
            "analysis_completeness": self.analysis_completeness,
            "status": self.status,
            "message_de": self.message_de,
            "unsupported_filters": list(self.unsupported_filters),
        }


class ResultRepository:
    """Run-scoped, read-only result access shared by GUI and exports."""

    _ARTIFACTS = {
        "pathologies": "pathology_marker_explorer.parquet",
        "biomarkers": "marker_pathology_explorer.parquet",
        "criteria": "criterion_candidates.parquet",
        "evidence": "guideline/candidate_evidence.parquet",
        "discovery": "phenome/new_associations_explorer.parquet",
    }

    def __init__(
        self,
        config: MimicProjectConfig,
        *,
        expected_feature_code_hash: Optional[str] = None,
        expected_validation_code_hash: Optional[str] = None,
    ):
        self.config = config
        self.expected_feature_code_hash = (
            expected_feature_code_hash or feature_implementation_hash()
        )
        self.expected_validation_code_hash = (
            expected_validation_code_hash or validation_implementation_hash()
        )
        self._available_runs_cache: Optional[tuple[ResultRunDescriptor, ...]] = None
        self._available_runs_fingerprint: Optional[tuple] = None

    def _runs_fingerprint(self) -> tuple:
        # Cheap (stat-only, no parquet/DuckDB reads) fingerprint of exactly
        # the files available_runs() reads per run directory -- used to
        # skip the expensive per-run DuckDB scan below when nothing has
        # actually changed since the last call.
        if not self.config.runs_root.is_dir():
            return ()
        entries = []
        for run_root in sorted(self.config.runs_root.iterdir()):
            if not run_root.is_dir():
                continue
            for name in (
                "validate.manifest.json",
                "development_provenance.json",
                "phenome/phenome.manifest.json",
                "screening_results.parquet",
            ):
                path = run_root / name
                try:
                    stat = path.stat()
                    entries.append((run_root.name, name, stat.st_mtime_ns, stat.st_size))
                except OSError:
                    entries.append((run_root.name, name, None, None))
        return tuple(entries)

    def available_runs(
        self,
        *,
        on_run_processed: Optional[Callable[[], None]] = None,
    ) -> tuple[ResultRunDescriptor, ...]:
        fingerprint = self._runs_fingerprint()
        if (
            self._available_runs_cache is not None
            and fingerprint == self._available_runs_fingerprint
        ):
            return self._available_runs_cache
        if not self.config.runs_root.is_dir():
            self._available_runs_cache = ()
            self._available_runs_fingerprint = fingerprint
            return ()
        descriptors: list[ResultRunDescriptor] = []
        for run_root in self.config.runs_root.iterdir():
            if not run_root.is_dir():
                continue
            validate_path = run_root / "validate.manifest.json"
            if not validate_path.is_file():
                continue
            try:
                validation = StageManifest.load(validate_path)
            except (OSError, TypeError, ValueError):
                continue
            provenance = _read_json(
                run_root / "development_provenance.json"
            )
            feature_integrity = provenance.get("feature_integrity", {})
            validation_compatible = bool(
                validation.stage_version == VALIDATION_STAGE_VERSION
                and provenance.get("code_hash")
                == self.expected_validation_code_hash
                and isinstance(feature_integrity, dict)
                and feature_integrity.get("source_code_hash")
                == self.expected_feature_code_hash
            )
            phenome_path = run_root / "phenome" / "phenome.manifest.json"
            try:
                phenome = (
                    StageManifest.load(phenome_path)
                    if phenome_path.is_file()
                    else None
                )
            except (OSError, TypeError, ValueError):
                phenome = None
            completeness = _analysis_completeness(
                run_root / "screening_results.parquet",
                self.config,
            )
            updated = max(
                path.stat().st_mtime
                for path in (validate_path, phenome_path)
                if path.is_file()
            )
            descriptors.append(
                ResultRunDescriptor(
                    run_id=run_root.name,
                    validation_status=validation.status,
                    validation_stage_version=validation.stage_version,
                    validation_compatible=validation_compatible,
                    analysis_completeness=completeness,
                    phenome_status=(phenome.status if phenome else "not_run"),
                    phenome_compatible=(
                        phenome is not None
                        and phenome.stage_version == PHENOME_STAGE_VERSION
                    ),
                    updated_at=pd.Timestamp(updated, unit="s", tz="UTC").isoformat(),
                    run_root=run_root,
                )
            )
            if on_run_processed is not None:
                on_run_processed()
        result = tuple(
            sorted(
                descriptors,
                key=lambda item: (item.updated_at, item.run_id),
                reverse=True,
            )
        )
        self._available_runs_cache = result
        self._available_runs_fingerprint = fingerprint
        return result

    def default_run(
        self,
        *,
        descriptors: Optional[tuple[ResultRunDescriptor, ...]] = None,
    ) -> Optional[ResultRunDescriptor]:
        return next(
            (
                item
                for item in (
                    descriptors if descriptors is not None else self.available_runs()
                )
                if item.complete_compatible
            ),
            None,
        )

    def query(self, query: ResultQuery) -> ResultPage:
        descriptor = self._resolve_run(query)
        if descriptor is None:
            return _empty_page(
                query,
                status="no_compatible_run",
                message_de=(
                    "Es gibt noch keinen vollständigen, mit dem aktuellen "
                    "Code kompatiblen Validation-Lauf."
                ),
            )
        if (
            not descriptor.validation_compatible
            and not query.include_historical
        ):
            return _empty_page(
                query,
                run_id=descriptor.run_id,
                status="incompatible_historical_run",
                message_de=(
                    "Dieser Lauf stammt aus einer älteren Ergebnisversion. "
                    "Aktivieren Sie historische Ergebnisse nur zur "
                    "Hypothesenansicht."
                ),
            )
        if query.complete_runs_only and not descriptor.complete_compatible:
            return _empty_page(
                query,
                run_id=descriptor.run_id,
                status="partial_or_incompatible_run",
                message_de=(
                    "Der gewählte Lauf ist eine Vorschau, unvollständig oder "
                    "nicht mit dem aktuellen Auswertungscode kompatibel."
                ),
            )
        artifact = descriptor.run_root / self._ARTIFACTS[query.perspective]
        manifest = self._manifest_for_perspective(
            descriptor.run_root,
            query.perspective,
        )
        if manifest is None or not artifact.is_file():
            return _empty_page(
                query,
                run_id=descriptor.run_id,
                status="artifact_not_available",
                message_de=_missing_artifact_message(query.perspective),
            )
        if str(artifact.resolve()) not in {
            str(Path(path).resolve()) for path in manifest.output_paths
        }:
            return _empty_page(
                query,
                run_id=descriptor.run_id,
                status="artifact_not_in_manifest",
                message_de=(
                    "Die Datei gehört nicht zum versiegelten Manifest dieses "
                    "Laufs und wird deshalb nicht angezeigt."
                ),
            )
        rows, total, unsupported = self._execute_query(artifact, query)
        rows = _ensure_friendly_columns(rows)
        return ResultPage(
            run_id=descriptor.run_id,
            perspective=query.perspective,
            artifact_path=artifact,
            rows=rows,
            total_count=total,
            offset=query.offset,
            limit=query.limit,
            snapshot_hash=manifest.input_hash,
            analysis_completeness=descriptor.analysis_completeness,
            status="ok" if total else "empty",
            message_de=(
                "Keine Ergebnisse entsprechen den aktiven Filtern."
                if not total
                else ""
            ),
            unsupported_filters=unsupported,
        )

    def _resolve_run(
        self,
        query: ResultQuery,
    ) -> Optional[ResultRunDescriptor]:
        available = self.available_runs()
        if query.run_id:
            return next(
                (item for item in available if item.run_id == query.run_id),
                None,
            )
        return next(
            (item for item in available if item.complete_compatible),
            None,
        )

    @staticmethod
    def _manifest_for_perspective(
        run_root: Path,
        perspective: str,
    ) -> Optional[StageManifest]:
        path = (
            run_root / "phenome" / "phenome.manifest.json"
            if perspective == "discovery"
            else run_root / "validate.manifest.json"
        )
        try:
            return StageManifest.load(path)
        except (OSError, TypeError, ValueError):
            return None

    def _execute_query(
        self,
        artifact: Path,
        query: ResultQuery,
    ) -> tuple[pd.DataFrame, int, tuple[str, ...]]:
        catalog = DuckDBCatalog(
            self.config.catalog_path,
            memory_limit_gb=self.config.duckdb_memory_limit_gb,
            threads=min(2, self.config.workers),
        )
        source = _sql_path(artifact)
        with catalog.connect(isolated=True) as connection:
            schema = {
                str(row[0])
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{source}')"
                ).fetchall()
            }
            where, parameters, unsupported = _where_clause(query, schema)
            count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{source}') {where}",
                    parameters,
                ).fetchone()[0]
            )
            sort_column = _sort_column(query, schema)
            order = "DESC" if query.sort_descending else "ASC"
            sql = (
                f"SELECT * FROM read_parquet('{source}') {where} "
                f'ORDER BY "{sort_column}" {order} NULLS LAST '
                "LIMIT ? OFFSET ?"
            )
            page_parameters = [*parameters, int(query.limit), int(query.offset)]
            rows = connection.execute(sql, page_parameters).fetchdf()
        return rows, count, tuple(unsupported)


def _where_clause(
    query: ResultQuery,
    schema: set[str],
) -> tuple[str, list[Any], list[str]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    unsupported: list[str] = []

    def in_filter(name: str, values: tuple[str, ...], label: str) -> None:
        if not values:
            return
        if name not in schema:
            unsupported.append(label)
            return
        placeholders = ",".join("?" for _ in values)
        clauses.append(f'CAST("{name}" AS VARCHAR) IN ({placeholders})')
        parameters.extend(str(value) for value in values)

    endpoint_column = "endpoint_id" if "endpoint_id" in schema else "target_id"
    in_filter(endpoint_column, query.endpoint_ids, "klinisches Ziel")
    in_filter(
        "pathology_group_id",
        query.pathology_group_ids,
        "Pathologiegruppe",
    )
    in_filter("panel_id", query.panel_ids, "Markerpanel")
    in_filter("feature_id", query.feature_ids, "Biomarker")
    family_column = (
        "conceptual_family"
        if "conceptual_family" in schema
        else "feature_family"
    )
    in_filter(family_column, query.feature_families, "Markerfamilie")
    candidate_column = (
        "candidate_type"
        if "candidate_type" in schema
        else "discovery_class"
        if "discovery_class" in schema
        else "candidate_type"
    )
    in_filter(candidate_column, query.candidate_types, "Ergebnisbereich")
    baseline_column = (
        "baseline_registry_id"
        if "baseline_registry_id" in schema
        else "cardiac_scope_id"
        if "cardiac_scope_id" in schema
        else "baseline_registry_id"
    )
    in_filter(baseline_column, query.baseline_ids, "Pathologieart")
    evidence_column = (
        "evidence_grade"
        if "evidence_grade" in schema
        else "stage"
        if "stage" in schema
        else "status_id"
    )
    in_filter(evidence_column, query.evidence_grades, "Evidenzstufe")

    if query.validation_only and "evaluation_split" in schema:
        clauses.append('CAST("evaluation_split" AS VARCHAR) = ?')
        parameters.append("validation")
    if query.complete_runs_only and "analysis_completeness" in schema:
        clauses.append('CAST("analysis_completeness" AS VARCHAR) = ?')
        parameters.append("complete_snapshot")
    if not query.include_historical:
        if "historical_pre_audit_result" in schema:
            clauses.append(
                'COALESCE(CAST("historical_pre_audit_result" AS BOOLEAN), false) = false'
            )
        elif "historical_implementation" in schema:
            clauses.append(
                'COALESCE(CAST("historical_implementation" AS BOOLEAN), false) = false'
            )

    _minimum_filter(
        clauses,
        parameters,
        unsupported,
        query.minimum_sensitivity,
        schema,
        (
            "sensitivity_at_95_specificity",
            "sensitivity",
        ),
        "Mindest-Sensitivität",
    )
    _minimum_filter(
        clauses,
        parameters,
        unsupported,
        query.minimum_specificity,
        schema,
        (
            "operating_specificity",
            "specificity",
        ),
        "Mindest-Spezifität",
    )
    _minimum_filter(
        clauses,
        parameters,
        unsupported,
        query.minimum_signal_quality,
        schema,
        ("mean_signal_quality",),
        "Mindest-Signalqualität",
    )
    if query.maximum_q_value is not None:
        q_column = next(
            (
                item
                for item in (
                    "q_value",
                    "adjusted_q_value",
                    "validation_q_value",
                )
                if item in schema
            ),
            None,
        )
        if q_column is None:
            unsupported.append("maximale FDR-q")
        else:
            clauses.append(f'CAST("{q_column}" AS DOUBLE) <= ?')
            parameters.append(float(query.maximum_q_value))

    if query.search_text.strip():
        search_columns = [
            item
            for item in (
                "endpoint_name_de",
                "endpoint_definition_de",
                "pathology_group",
                "feature_name_de",
                "feature_id",
                "endpoint_id",
                "target_name",
                "target_id",
                "conceptual_family",
                "display_name_de",
                "scope_display_de",
                "discovery_domain",
                "cardiac_scope_de",
                "status_de",
                "followup_priority_de",
            )
            if item in schema
        ]
        if search_columns:
            needle = f"%{query.search_text.strip().casefold()}%"
            clauses.append(
                "("
                + " OR ".join(
                    f'LOWER(COALESCE(CAST("{column}" AS VARCHAR), \'\')) LIKE ?'
                    for column in search_columns
                )
                + ")"
            )
            parameters.extend(needle for _ in search_columns)
        else:
            unsupported.append("Freitextsuche")
    return (
        ("WHERE " + " AND ".join(clauses)) if clauses else "",
        parameters,
        unsupported,
    )


def _minimum_filter(
    clauses: list[str],
    parameters: list[Any],
    unsupported: list[str],
    value: Optional[float],
    schema: set[str],
    candidates: tuple[str, ...],
    label: str,
) -> None:
    if value is None:
        return
    column = next((item for item in candidates if item in schema), None)
    if column is None:
        unsupported.append(label)
        return
    clauses.append(f'CAST("{column}" AS DOUBLE) >= ?')
    parameters.append(float(value))


def _sort_column(query: ResultQuery, schema: set[str]) -> str:
    requested = str(query.sort_by or "")
    allowed = {
        "evidence_rank",
        "roc_auc",
        "average_precision",
        "sensitivity_at_95_specificity",
        "operating_specificity",
        "q_value",
        "adjusted_q_value",
        "validation_q_value",
        "association_strength",
        "statistical_strength",
        "endpoint_name_de",
        "feature_name_de",
        "target_name",
        "status_de",
    }
    if requested in allowed and requested in schema:
        return requested
    for fallback in (
        "evidence_rank",
        "association_strength",
        "statistical_strength",
        "roc_auc",
        "endpoint_name_de",
        "target_name",
        "feature_id",
    ):
        if fallback in schema:
            return fallback
    return sorted(schema)[0]


def _analysis_completeness(
    artifact: Path,
    config: MimicProjectConfig,
) -> str:
    if not artifact.is_file():
        return "missing"
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=1,
    )
    source = _sql_path(artifact)
    try:
        with catalog.connect(isolated=True) as connection:
            schema = {
                str(row[0])
                for row in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{source}')"
                ).fetchall()
            }
            if "analysis_completeness" not in schema:
                return "historical_unknown"
            row = connection.execute(
                f'SELECT DISTINCT CAST("analysis_completeness" AS VARCHAR) '
                f"FROM read_parquet('{source}') LIMIT 2"
            ).fetchall()
        values = {str(item[0]) for item in row}
        return next(iter(values)) if len(values) == 1 else "mixed_invalid"
    except Exception as exc:
        LOGGER.warning(
            "Result artifact could not be inspected: %s (%s)",
            artifact,
            exc,
        )
        return "unreadable"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _ensure_friendly_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or (
        "endpoint_name_de" in frame and "feature_name_de" in frame
    ):
        return frame
    return pd.DataFrame(
        annotate_result_display(frame.to_dict(orient="records"))
    )


def _empty_page(
    query: ResultQuery,
    *,
    run_id: str = "",
    status: str,
    message_de: str,
) -> ResultPage:
    return ResultPage(
        run_id=run_id,
        perspective=query.perspective,
        artifact_path=None,
        rows=pd.DataFrame(),
        total_count=0,
        offset=query.offset,
        limit=query.limit,
        snapshot_hash="",
        analysis_completeness="unavailable",
        status=status,
        message_de=message_de,
    )


def _missing_artifact_message(perspective: str) -> str:
    return {
        "pathologies": "Für diesen Lauf wurden noch keine Pathologie-Ergebnisse erzeugt.",
        "biomarkers": "Für diesen Lauf wurden noch keine Biomarker-Ergebnisse erzeugt.",
        "criteria": "Für diesen Lauf gibt es noch keine Kriterienkandidaten.",
        "discovery": "Der breite Zusammenhangsscan wurde für diesen Lauf noch nicht ausgeführt.",
    }[perspective]


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


__all__ = [
    "RESULT_QUERY_VERSION",
    "ResultPage",
    "ResultRepository",
    "ResultRunDescriptor",
]
