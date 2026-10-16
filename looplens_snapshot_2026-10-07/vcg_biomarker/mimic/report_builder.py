from __future__ import annotations

import base64
import html
import io
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .config import MimicProjectConfig
from .feature_roles import feature_analysis_role, is_candidate_biomarker
from .models import ReportSpec, ReportType
from .presentation import comparison_effect_frame, comparison_effect_labels
from .reports import (
    MIMIC_DUA_NOTICE,
    generate_run_reports,
    resolve_acceptance_audit_paths,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
)
from .user_guidance import annotate_result_frame


CONFIGURED_REPORT_VERSION = "1.13.0"
EXCEL_MAX_TOTAL_CELLS = 1_000_000
EXCEL_MAX_ROWS_PER_TABLE = 10_000
EXCEL_MIN_ROWS_PER_TABLE = 25
REPORTABLE_PHENOME_TABLES = frozenset(
    {
        "association_domain_definitions",
        "association_domain_replicated_hits",
        "broad_discovery_replicated_hits",
        "ccsr_audit_summary",
        "ccsr_summary",
        "future_phenome_label_summary",
        "future_phenome_replicated_hits",
        "phenome_diagnostics",
        "phenome_adjusted_confirmation",
        "phenome_discovery_status",
        "phenome_label_summary",
        "phenome_promotion_candidates",
        "phenome_replicated_hits",
        "phenotype_summary",
        "broad_discovery_ranked",
        "new_associations_explorer",
    }
)
RESEARCH_NOTICE_DE = (
    "Forschungsplattform für klinische Referenzvergleiche. Die dargestellten "
    "Marker sind keine Einzelfalldiagnose und keine Behandlungsempfehlung."
)
RESEARCH_NOTICE_EN = (
    "Research platform for clinical reference comparisons. Displayed markers "
    "are not patient-level diagnoses or treatment recommendations."
)

TABLE_TITLES_DE = {
    "run_manifest": "Lauf und Reproduzierbarkeit",
    "clinical_utility_summary": "Evidenzstufe und klinischer Nutzen",
    "clinical_utility_gates": "Harte Gates der klinischen Nutzenbewertung",
    "data_inventory": "Verfügbare Datenquellen",
    "ecg_master_index_summary": "ECG-Masterindex",
    "label_summary": "Klinische Referenzgruppen",
    "split_summary": "Patientengetrennte Datenaufteilung",
    "screening_results": "Vollständige Biomarker-Prüfung",
    "pathology_marker_explorer": "Pathologien und ihre Biomarker",
    "marker_pathology_explorer": "Biomarker und zugehörige Pathologien",
    "endpoint_summary_explorer": "Übersicht klinischer Ziele",
    "criterion_candidates": "Kandidaten für ECG-/VCG-Kriterien",
    "criterion_thresholds": "Trainingsbasierte Grenzwerte und Grauzonen",
    "classical_baseline_results": "Klassische ECG-Regeln",
    "machine_report_baseline_results": "Maschinelle ECG-Befunde",
    "vcg_vs_classical_comparison": "VCG im Vergleich zu klassischen ECG-Regeln",
    "combination_results": "Kleine interpretierbare Markerkombinationen",
    "overfitting_stability_report": "Stabilität und Überanpassung",
    "feature_failure_report": "Nicht auswertbare Marker und Qualitätsprobleme",
    "validation_failure_report": "Nicht auswertbare klinische Vergleiche",
    "review_candidates": "Diskordante Fälle für lokale Prüfung",
    "discordance_report": "Zusammenfassung diskordanter Ergebnisse",
    "final_test_report": "Versiegelter Teststatus",
    "report_provenance": "Snapshot, Versionen und Testschutz",
    "phenome_adjusted_confirmation": "Adjustierte Bestätigung neuer Zusammenhänge",
    "phenome_discovery_status": "Evidenzstatus neuer Pathologie-Zusammenhänge",
    "phenome_new_associations_explorer": "Neue Pathologie- und Kontextzusammenhänge",
    "phenome_broad_discovery_ranked": "Breiter explorativer Scan",
    "phenome_replicated_hits": "In Validation replizierte Phenome-Treffer",
}

TABLE_TITLES_EN = {
    "clinical_utility_summary": "Evidence level and clinical utility",
    "clinical_utility_gates": "Clinical-utility hard gates",
    "criterion_candidates": "ECG/VCG criterion candidates",
    "criterion_thresholds": "Train-derived thresholds and uncertainty bands",
    "pathology_marker_explorer": "Pathologies and associated biomarkers",
    "marker_pathology_explorer": "Biomarkers and associated pathologies",
    "endpoint_summary_explorer": "Clinical target overview",
    "phenome_new_associations_explorer": "New pathology and context associations",
    "phenome_adjusted_confirmation": "Adjusted confirmation of new associations",
    "phenome_discovery_status": "Evidence status of pathology associations",
}

FIGURE_TITLES_DE = {
    "marker_ranking": "Beste formal auswertbare Validation-Marker",
    "endpoint_heatmap": "Biomarker × klinisches Ziel",
    "cohort_counts": "Beurteilbare Fälle und Kontrollen",
    "baseline_delta_forest": "VCG gegen klassische ECG-Regeln (Baseline Delta Forest)",
    "criterion_candidate_performance": "Kriterienkandidaten: Sensitivität und Spezifität",
    "new_association_status": "Neue Zusammenhänge nach Evidenzstatus",
    "validation_roc_pr": "ROC- und PR-Kurven (Validation Roc Pr)",
    "score_distribution": "Score-Verteilungen (Score Distribution)",
    "operating_points": "Grenzwerte und Betriebspunkte (Operating Points)",
    "sensitivity_stability": "Robustheit der Ergebnisse (Sensitivity Stability)",
    "adjusted_utility": "Adjustierte Modelle und Nutzen (Adjusted Utility)",
}


def _scientific_mimic_hash() -> str:
    from .code_scope import scientific_source_hash

    return scientific_source_hash(Path(__file__).resolve().parent)


def build_configured_report(
    config: MimicProjectConfig,
    spec: ReportSpec,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    run_root = config.runs_root / spec.run_id
    if not run_root.is_dir():
        raise FileNotFoundError(f"Validation run does not exist: {run_root}")
    canonical_root = run_root / "reports"
    canonical_manifest = canonical_root / "report.manifest.json"
    if not canonical_manifest.is_file():
        if progress:
            progress("Kanonische Run-Tabellen werden erzeugt")
        generate_run_reports(
            config,
            spec.run_id,
            force=False,
            progress=progress,
        )

    source_paths = _select_source_paths(run_root, canonical_root, spec)
    acceptance_json_path, _ = resolve_acceptance_audit_paths(run_root)
    input_hash = content_hash(
        {
            "spec": spec.to_dict(),
            "sources": {
                name: file_fingerprint(path)
                for name, path in source_paths.items()
            },
            "acceptance_audit_json": (
                file_fingerprint(acceptance_json_path)
                if acceptance_json_path is not None
                else None
            ),
            # Run machinery (code_scope) cannot change a report value.
            "code_hash": _scientific_mimic_hash(),
        }
    )
    config_hash = scientific_config_hash(config)
    build_root = canonical_root / "builds"
    build_root.mkdir(parents=True, exist_ok=True)
    output_root = build_root / spec.report_id
    manifest_path = output_root / "report_build.manifest.json"
    if not force and manifest_path.is_file():
        existing = StageManifest.load(manifest_path)
        if existing.matches(
            input_hash,
            config_hash,
            CONFIGURED_REPORT_VERSION,
        ):
            return existing

    published_output_root = output_root
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{spec.report_id}.",
            suffix=".staging",
            dir=build_root,
        )
    )
    output_root = staging_root

    if progress:
        progress("Berichtstabellen filtern und anonymisieren")
    tables = {
        name: _prepare_table(pd.read_parquet(path), name=name, spec=spec)
        for name, path in source_paths.items()
    }
    tables = {
        name: frame
        for name, frame in tables.items()
        if not frame.empty and _table_allowed(name, spec)
    }
    provenance = _report_provenance(
        config,
        spec,
        source_paths,
        input_hash,
        tables,
    )
    tables["report_provenance"] = pd.DataFrame([provenance])

    output_paths: list[str] = []
    if "parquet" in spec.formats:
        parquet_root = output_root / "parquet"
        for name, frame in tables.items():
            path = parquet_root / f"{name}.parquet"
            atomic_write_parquet(_parquet_safe(frame), path)
            output_paths.append(str(path.resolve()))
    if "csv" in spec.formats:
        csv_root = output_root / "csv"
        for name, frame in tables.items():
            path = csv_root / f"{name}.csv"
            _atomic_write_csv(frame, path)
            output_paths.append(str(path.resolve()))

    if progress:
        progress("Visualisierungen für den Bericht erzeugen")
    figures = _build_figures(tables, spec, provenance)
    if "xlsx" in spec.formats:
        excel_path = output_root / f"{spec.report_id}.xlsx"
        _atomic_write_excel(tables, excel_path)
        output_paths.append(str(excel_path.resolve()))
    if "html" in spec.formats:
        html_path = output_root / f"{spec.report_id}.html"
        atomic_write_text(
            html_path,
            _render_self_contained_html(spec, tables, figures, provenance),
        )
        output_paths.append(str(html_path.resolve()))
    if "pdf" in spec.formats:
        if progress:
            progress("Druckbaren A4-PDF-Bericht rendern")
        pdf_path = output_root / f"{spec.report_id}.pdf"
        _atomic_write_pdf(pdf_path, spec, tables, figures, provenance)
        output_paths.append(str(pdf_path.resolve()))

    output_paths = [
        str(
            (
                published_output_root
                / Path(path).relative_to(staging_root.resolve())
            ).resolve()
        )
        for path in output_paths
    ]
    metadata_path = output_root / "report_spec.json"
    atomic_write_json(
        metadata_path,
        {
            "report_spec": spec.to_dict(),
            "provenance": provenance,
            "source_tables": {
                name: str(path.resolve()) for name, path in source_paths.items()
            },
            "output_paths": output_paths,
        },
    )
    output_paths.append(
        str(
            (
                published_output_root
                / metadata_path.relative_to(staging_root)
            ).resolve()
        )
    )
    _close_figures(figures)

    manifest = StageManifest.complete(
        stage=f"report-build:{spec.run_id}:{spec.report_id}",
        stage_version=CONFIGURED_REPORT_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts={name: len(frame) for name, frame in tables.items()},
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            MIMIC_DUA_NOTICE,
            _research_notice(spec.language),
            (
                "Patient identifiers included by explicit local opt-in."
                if spec.include_case_ids
                else "Patient identifiers excluded by default."
            ),
            (
                "Configured report exports contain aggregate and replicated "
                "phenome results; complete screening tables remain in the "
                "canonical run artifacts."
            ),
            (
                "Excel is a documented preview constrained by a cell budget; "
                "CSV and Parquet contain every selected report row."
            ),
        ),
    )
    manifest.save(staging_root / "report_build.manifest.json")
    _publish_report_directory(staging_root, published_output_root)
    DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    ).register_stage(manifest, manifest_path)
    return manifest


def _publish_report_directory(staging_root: Path, output_root: Path) -> None:
    build_root = output_root.parent.resolve()
    if staging_root.resolve().parent != build_root:
        raise ValueError("Report staging directory is outside the build root")
    backup_root: Optional[Path] = None
    if output_root.exists():
        backup_root = output_root.with_name(
            f".{output_root.name}.{uuid.uuid4().hex}.backup"
        )
        os.replace(output_root, backup_root)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        if (
            backup_root is not None
            and backup_root.exists()
            and not output_root.exists()
        ):
            os.replace(backup_root, output_root)
        raise
    if backup_root is not None and backup_root.exists():
        _remove_staged_report_tree(backup_root, output_root.parent)


def _remove_staged_report_tree(path: Path, build_root: Path) -> None:
    resolved = path.resolve()
    if (
        resolved.parent != build_root.resolve()
        or not resolved.name.startswith(".")
    ):
        raise ValueError(
            f"Refusing to remove report tree outside build staging: {resolved}"
        )
    shutil.rmtree(resolved)


def _select_source_paths(
    run_root: Path,
    canonical_root: Path,
    spec: ReportSpec,
) -> dict[str, Path]:
    candidates = {
        path.stem: path
        for path in sorted(canonical_root.glob("*.parquet"))
        if path.is_file()
    }
    for path in sorted((run_root / "phenome").glob("*.parquet")):
        if path.stem not in REPORTABLE_PHENOME_TABLES:
            continue
        key = (
            path.stem
            if path.stem.startswith("phenome_")
            else f"phenome_{path.stem}"
        )
        candidates[key] = path
    for path in sorted(
        (run_root / "group_vcg").glob(
            "endpoint=*/split=*/group_vcg_*.parquet"
        )
    ):
        if (
            path.name == "group_vcg_case_audit.parquet"
            and not spec.include_case_appendix
        ):
            continue
        endpoint = path.parent.parent.name.removeprefix("endpoint=")
        split = path.parent.name.removeprefix("split=")
        key = f"group_vcg_{endpoint}_{split}_{path.stem}"
        candidates[key] = path
    if not spec.include_case_appendix:
        candidates.pop("review_candidates", None)
        candidates.pop("discordance_report", None)
    if not candidates:
        raise FileNotFoundError(
            f"No canonical report tables found in {canonical_root}"
        )
    return candidates


def _prepare_table(
    frame: pd.DataFrame,
    *,
    name: str,
    spec: ReportSpec,
) -> pd.DataFrame:
    result = frame.copy()
    if spec.endpoint_ids and "endpoint_id" in result:
        result = result.loc[
            result["endpoint_id"].astype(str).isin(spec.endpoint_ids)
        ]
    if spec.panel_ids and "panel_id" in result:
        result = result.loc[result["panel_id"].astype(str).isin(spec.panel_ids)]
    if spec.feature_ids:
        feature_mask = pd.Series(False, index=result.index)
        feature_columns_found = False
        for column in ("feature_id", "vcg_feature_id", "marker_id"):
            if column in result:
                feature_columns_found = True
                feature_mask |= result[column].astype(str).isin(spec.feature_ids)
        for column in (
            "component_features",
            "component_ids",
            "components",
        ):
            if column in result:
                feature_columns_found = True
                feature_mask |= result[column].map(
                    lambda value: bool(
                        set(_feature_members(value)).intersection(spec.feature_ids)
                    )
                )
        if feature_columns_found:
            result = result.loc[feature_mask]
    if not spec.include_case_ids:
        protected = {
            "subject_id",
            "study_id",
            "hadm_id",
            "ed_stay_id",
            "waveform_path",
            "source_path",
            "patient_id",
            "ecg_id",
        }
        result = result.drop(
            columns=[column for column in result if column in protected],
            errors="ignore",
        )
    if not spec.include_case_appendix and name.startswith("review_"):
        return pd.DataFrame()
    if "feature_id" in result:
        result["feature_analysis_role"] = result["feature_id"].map(
            feature_analysis_role
        )
    if "vcg_feature_id" in result:
        result["vcg_feature_analysis_role"] = result["vcg_feature_id"].map(
            feature_analysis_role
        )
    result = annotate_result_frame(result)
    return result.reset_index(drop=True)


def _table_allowed(name: str, spec: ReportSpec) -> bool:
    section_map = {
        "summary": (
            "run_manifest",
            "report_metadata",
            "final_test_report",
            "clinical_utility_summary",
        ),
        "methods": (
            "baseline_registry",
            "methodology_registry",
            "run_manifest",
        ),
        "cohort": (
            "data_inventory",
            "ecg_master_index_summary",
            "label_summary",
            "split_summary",
            "validation_label_counts",
        ),
        "performance": (
            "screening_results",
            "combination_results",
            "adjusted_model_results",
            "all_ecg_clustered_results",
            "performance_curve_points",
            "score_distribution_bins",
            "operating_point_results",
            "pathology_marker_explorer",
            "marker_pathology_explorer",
            "endpoint_summary_explorer",
            "criterion_candidates",
            "criterion_thresholds",
            "group_vcg_",
        ),
        "baselines": (
            "classical_baseline_results",
            "machine_report_baseline_results",
            "vcg_vs_classical_comparison",
            "descriptive_validation_ranking",
            "ptb_transfer_matrix",
        ),
        "stability": (
            "overfitting_stability_report",
            "label_window_sensitivity_summary",
            "decision_curve_results",
        ),
        "review": ("review_candidates", "discordance_report"),
        "discovery": ("phenome_",),
        "limitations": (
            "feature_failure_report",
            "validation_failure_report",
            "review_candidate_diagnostics",
            "clinical_utility_gates",
            "clinical_utility_summary",
        ),
        "provenance": (
            "run_manifest",
            "report_metadata",
            "acceptance_audit",
            "clinical_utility_summary",
        ),
    }
    if name.startswith("phenome_"):
        return "discovery" in spec.sections or spec.report_type == ReportType.STUDY
    allowed_prefixes = tuple(
        prefix
        for section in spec.sections
        for prefix in section_map.get(section, ())
    )
    return (
        not allowed_prefixes
        or name in allowed_prefixes
        or any(name.startswith(prefix) for prefix in allowed_prefixes)
    )


def _report_provenance(
    config: MimicProjectConfig,
    spec: ReportSpec,
    sources: dict[str, Path],
    input_hash: str,
    tables: dict[str, pd.DataFrame],
) -> dict:
    run_root = config.runs_root / spec.run_id
    development_path = run_root / "development_provenance.json"
    freeze_path = run_root / "freeze_manifest.json"
    lock_path = run_root / "test_lock.json"
    acceptance_path, _ = resolve_acceptance_audit_paths(run_root)
    development = _read_json(development_path)
    validation_manifest = _read_json(run_root / "validate.manifest.json")
    freeze = _read_json(freeze_path)
    lock = _read_json(lock_path)
    acceptance = (
        _read_json(acceptance_path)
        if acceptance_path is not None
        else {}
    )
    test_status = "sealed_not_evaluated"
    if lock.get("consumed_at"):
        test_status = "locked_test_consumed"
    feature_scope = str(development.get("feature_scope", "unknown"))
    development_complete = (
        feature_scope == "development_only"
        and validation_manifest.get("status") == "complete"
        and development.get("locked_test_labels_read") is False
    )
    if test_status == "locked_test_consumed":
        analysis_status = "locked_test_final"
    elif freeze.get("freeze_hash"):
        analysis_status = "frozen_development"
    elif development_complete:
        analysis_status = "complete_development"
    else:
        analysis_status = "partial_preview"
    evaluation_splits = sorted(
        {
            str(value)
            for frame in tables.values()
            if "evaluation_split" in frame
            for value in frame["evaluation_split"].dropna().unique()
        }
    )
    power_statuses = sorted(
        {
            str(value)
            for frame in tables.values()
            for column in ("inference_status", "power_status")
            if column in frame
            for value in frame[column].dropna().unique()
        }
    )
    exploratory_content = any(
        (
            frame["exploratory_score"].fillna(False).astype(bool).any()
            if "exploratory_score" in frame
            else False
        )
        or (
            frame["panel_id"]
            .astype(str)
            .isin(("sentinel", "discovery_reserve"))
            .any()
            if "panel_id" in frame
            else False
        )
        for frame in tables.values()
    )
    feature_snapshot = development.get("feature_snapshot", {})
    return {
        "report_id": spec.report_id,
        "run_id": spec.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_type": spec.report_type.value,
        "language": spec.language,
        "input_hash": input_hash,
        "snapshot_hash": content_hash(
            {
                name: file_fingerprint(path)
                for name, path in sources.items()
            }
        ),
        "feature_snapshot_hash": feature_snapshot.get("snapshot_hash", ""),
        "analysis_status": analysis_status,
        "artifact_completeness": (
            "complete" if development_complete else "partial"
        ),
        "feature_scope": feature_scope,
        "evaluation_splits": ",".join(evaluation_splits) or "not_recorded",
        "power_statuses": ",".join(power_statuses) or "not_recorded",
        "exploration_grade": (
            "contains_exploratory_results"
            if exploratory_content
            else "prespecified_only"
        ),
        "development_code_hash": development.get("code_hash", ""),
        "freeze_hash": freeze.get("freeze_hash", lock.get("freeze_hash", "")),
        "locked_test_status": test_status,
        "acceptance_status": acceptance.get("status", "not_run"),
        "acceptance_audit_hash": acceptance.get("audit_hash", ""),
        "acceptance_locked_test_outcomes_read": acceptance.get(
            "locked_test_outcomes_read"
        ),
        "acceptance_ptb_fold_10_outcomes_read": acceptance.get(
            "ptb_fold_10_outcomes_read"
        ),
        "mimic_releases": json.dumps(
            {
                "mimic_iv": config.mimic_iv_release,
                "mimic_iv_ed": config.mimic_ed_release,
                "mimic_iv_ecg": config.mimic_ecg_release,
                "mimic_iv_echo": config.mimic_echo_release,
            },
            sort_keys=True,
        ),
        "patient_ids_included": spec.include_case_ids,
        "case_appendix_included": spec.include_case_appendix,
        "dua_notice": MIMIC_DUA_NOTICE,
        "research_notice": _research_notice(spec.language),
    }


def _formal_candidate_marker_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    if "feature_id" in result:
        result = result.loc[
            result["feature_id"].map(is_candidate_biomarker)
        ]
    if "inference_status" in result:
        result = result.loc[
            result["inference_status"].astype(str).eq("formal_inference")
        ]
    return result


def _feature_members(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return tuple(str(item) for item in value if str(item).strip())
    text = str(value or "").strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, list):
            return tuple(str(item) for item in decoded if str(item).strip())
    for separator in ("|", ";", ",", "+"):
        if separator in text:
            return tuple(
                token.strip()
                for token in text.split(separator)
                if token.strip()
            )
    return (text,)


def _build_figures(
    tables: dict[str, pd.DataFrame],
    spec: ReportSpec,
    provenance: dict,
) -> list[tuple[str, object]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[tuple[str, object]] = []
    screening = tables.get("screening_results", pd.DataFrame())
    if not screening.empty:
        validation = screening
        if "evaluation_split" in validation:
            preferred = validation.loc[
                validation["evaluation_split"].astype(str).eq("validation")
            ]
            if not preferred.empty:
                validation = preferred
        validation = _formal_candidate_marker_rows(validation)
        auc_column = _first_column(validation, ("roc_auc", "auc_roc", "auc"))
        ap_column = _first_column(
            validation,
            ("average_precision", "auc_pr", "pr_auc"),
        )
        if auc_column and "feature_id" in validation:
            ranked = validation.dropna(subset=[auc_column]).nlargest(
                min(20, len(validation)),
                auc_column,
            )
            fig, axis = plt.subplots(figsize=(8.27, 5.5))
            labels = (
                ranked["endpoint_id"].astype(str)
                + " | "
                + ranked["feature_id"].astype(str)
                if "endpoint_id" in ranked
                else ranked["feature_id"].astype(str)
            )
            y = np.arange(len(ranked))
            axis.barh(y, ranked[auc_column], color="#087e8b")
            axis.set_yticks(y, labels)
            axis.invert_yaxis()
            axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
            axis.set_xlim(0.45, 1.0)
            axis.set_xlabel("ROC-AUC")
            axis.set_title(
                _text(
                    spec,
                    "Formal auswertbare Validation-Marker",
                    "Formally evaluable validation markers",
                )
            )
            axis.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            figures.append(("marker_ranking", fig))
        if (
            auc_column
            and ap_column
            and "endpoint_id" in validation
            and "feature_id" in validation
        ):
            top = validation.dropna(subset=[auc_column]).nlargest(
                min(300, len(validation)),
                auc_column,
            )
            pivot = top.pivot_table(
                index="feature_id",
                columns="endpoint_id",
                values=auc_column,
                aggfunc="max",
            )
            pivot = pivot.loc[
                pivot.max(axis=1).sort_values(ascending=False).head(30).index
            ]
            if not pivot.empty:
                fig, axis = plt.subplots(figsize=(8.27, 7.4))
                image = axis.imshow(
                    pivot.fillna(0.5).to_numpy(),
                    aspect="auto",
                    cmap="viridis",
                    vmin=0.5,
                    vmax=max(0.8, float(np.nanmax(pivot.to_numpy()))),
                )
                axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
                axis.set_xticks(
                    np.arange(len(pivot.columns)),
                    pivot.columns,
                    rotation=65,
                    ha="right",
                )
                axis.set_title(
                    _text(
                        spec,
                        "Marker × Endpunkt: ROC-AUC",
                        "Marker × endpoint: ROC-AUC",
                    )
                )
                fig.colorbar(image, ax=axis, label="ROC-AUC")
                fig.tight_layout()
                figures.append(("endpoint_heatmap", fig))

    labels = tables.get("label_summary", pd.DataFrame())
    endpoint_column = _first_column(labels, ("label_id", "endpoint_id"))
    positive_column = _first_column(
        labels,
        ("positive_count", "n_positive", "positive"),
    )
    negative_column = _first_column(
        labels,
        ("negative_count", "n_negative", "negative"),
    )
    if endpoint_column and positive_column and negative_column:
        cohort = labels.copy()
        cohort[positive_column] = pd.to_numeric(
            cohort[positive_column], errors="coerce"
        )
        cohort[negative_column] = pd.to_numeric(
            cohort[negative_column], errors="coerce"
        )
        cohort = cohort.dropna(subset=[positive_column, negative_column]).head(25)
        if not cohort.empty:
            fig, axis = plt.subplots(figsize=(8.27, 6.0))
            y = np.arange(len(cohort))
            axis.barh(
                y,
                cohort[negative_column],
                color="#b8c4c8",
                label=_text(spec, "Negativ", "Negative"),
            )
            axis.barh(
                y,
                cohort[positive_column],
                left=cohort[negative_column],
                color="#d1495b",
                label=_text(spec, "Positiv", "Positive"),
            )
            axis.set_yticks(y, cohort[endpoint_column].astype(str))
            axis.invert_yaxis()
            axis.set_xlabel(_text(spec, "Beurteilbare Personen", "Assessable subjects"))
            axis.set_title(_text(spec, "Endpunkt-Kohorten", "Endpoint cohorts"))
            axis.legend()
            axis.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            figures.append(("cohort_counts", fig))

    comparisons = tables.get("vcg_vs_classical_comparison", pd.DataFrame())
    comparison = comparison_effect_frame(comparisons)
    if not comparison.empty:
        comparison = comparison.reindex(
            comparison["effect_value"].abs().sort_values(ascending=False).index
        ).head(25)
        fig, axis = plt.subplots(figsize=(8.27, 6.0))
        y = np.arange(len(comparison))
        values = comparison["effect_value"].to_numpy(dtype=float)
        low = comparison["effect_ci_low"].to_numpy(dtype=float)
        high = comparison["effect_ci_high"].to_numpy(dtype=float)
        finite_ci = np.isfinite(low) & np.isfinite(high)
        errors = np.vstack(
            (
                np.where(finite_ci, np.maximum(0.0, values - low), 0.0),
                np.where(finite_ci, np.maximum(0.0, high - values), 0.0),
            )
        )
        axis.errorbar(
            values,
            y,
            xerr=errors,
            fmt="o",
            color="#087e8b",
            ecolor="#5f6b6d",
            capsize=2,
        )
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_yticks(y, comparison_effect_labels(comparison))
        axis.invert_yaxis()
        axis.set_xlabel(
            _text(
                spec,
                "Gepaarte Differenz; positiv spricht fuer VCG",
                "Paired difference; positive favors VCG",
            )
        )
        axis.set_title(
            _text(
                spec,
                "VCG gegen klassische EKG-Referenz",
                "VCG versus classical ECG reference",
            )
        )
        axis.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        figures.append(("baseline_delta_forest", fig))

    criteria = tables.get("criterion_candidates", pd.DataFrame())
    if {"sensitivity", "specificity"}.issubset(criteria) and not criteria.empty:
        values = criteria.copy()
        values["sensitivity"] = pd.to_numeric(
            values["sensitivity"], errors="coerce"
        )
        values["specificity"] = pd.to_numeric(
            values["specificity"], errors="coerce"
        )
        values = values.dropna(subset=["sensitivity", "specificity"])
        if not values.empty:
            evidence = _string_series(values, "evidence_grade")
            evidence_codes, evidence_labels = pd.factorize(
                evidence.replace("", "not_recorded"), sort=True
            )
            auc = pd.to_numeric(
                values.get("roc_auc", pd.Series(np.nan, index=values.index)),
                errors="coerce",
            )
            marker_sizes = 30.0 + 90.0 * np.clip(
                auc.fillna(0.5).to_numpy(dtype=float) - 0.5,
                0.0,
                0.5,
            ) / 0.5
            fig, axis = plt.subplots(figsize=(8.27, 5.4))
            scatter = axis.scatter(
                values["specificity"],
                values["sensitivity"],
                c=evidence_codes,
                cmap="tab10",
                s=marker_sizes,
                alpha=0.78,
                edgecolors="white",
                linewidths=0.5,
            )
            axis.axvline(0.90, color="#778388", linestyle="--", linewidth=1)
            axis.axvline(0.95, color="#778388", linestyle=":", linewidth=1)
            axis.set_xlim(0.45, 1.01)
            axis.set_ylim(0.0, 1.01)
            axis.set_xlabel(_text(spec, "Spezifität", "Specificity"))
            axis.set_ylabel(_text(spec, "Sensitivität", "Sensitivity"))
            axis.set_title(
                _text(
                    spec,
                    "Kriterienkandidaten am vorgegebenen Betriebspunkt",
                    "Criterion candidates at their prespecified operating point",
                )
            )
            handles, _ = scatter.legend_elements(prop="colors")
            if handles:
                axis.legend(
                    handles,
                    list(evidence_labels),
                    title=_text(spec, "Evidenz", "Evidence"),
                    fontsize=7,
                    loc="lower left",
                )
            axis.grid(alpha=0.2)
            fig.tight_layout()
            figures.append(("criterion_candidate_performance", fig))

    new_associations = tables.get(
        "phenome_new_associations_explorer", pd.DataFrame()
    )
    if not new_associations.empty and "status_de" in new_associations:
        status_counts = (
            new_associations["status_de"]
            .fillna(_text(spec, "nicht eingeordnet", "not classified"))
            .astype(str)
            .value_counts()
            .head(12)
            .sort_values()
        )
        if not status_counts.empty:
            fig, axis = plt.subplots(figsize=(8.27, 4.8))
            axis.barh(
                np.arange(len(status_counts)),
                status_counts.to_numpy(),
                color="#087e8b",
            )
            axis.set_yticks(
                np.arange(len(status_counts)), status_counts.index.astype(str)
            )
            axis.set_xlabel(_text(spec, "Marker-Ziel-Paare", "Marker-target pairs"))
            axis.set_title(
                _text(
                    spec,
                    "Neue Zusammenhänge nach Replikation und Adjustierung",
                    "New associations after replication and adjustment",
                )
            )
            axis.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            figures.append(("new_association_status", fig))

    transfer = tables.get("ptb_transfer_matrix", pd.DataFrame())
    transfer_columns = {
        "ptb_validation_rank_biserial",
        "mimic_validation_rank_biserial",
    }
    if not transfer.empty and transfer_columns.issubset(transfer):
        values = transfer.copy()
        values["ptb_validation_rank_biserial"] = pd.to_numeric(
            values["ptb_validation_rank_biserial"],
            errors="coerce",
        )
        values["mimic_validation_rank_biserial"] = pd.to_numeric(
            values["mimic_validation_rank_biserial"],
            errors="coerce",
        )
        values = values.dropna(
            subset=list(transfer_columns)
        ).head(500)
        if not values.empty:
            agreement = values.get(
                "train_direction_agreement",
                pd.Series(pd.NA, index=values.index),
            )
            colors = [
                (
                    "#778388"
                    if pd.isna(item)
                    else "#16803a"
                    if bool(item)
                    else "#d97706"
                )
                for item in agreement
            ]
            score_mask = values.get(
                "transfer_type",
                pd.Series("shared_single_marker", index=values.index),
            ).astype(str).eq("ptb_frozen_score")
            fig, axis = plt.subplots(figsize=(8.27, 5.2))
            for selected, marker, label in (
                (
                    ~score_mask,
                    "o",
                    _text(spec, "Einzelmarker", "Single marker"),
                ),
                (
                    score_mask,
                    "s",
                    _text(spec, "Eingefrorener Score", "Frozen score"),
                ),
            ):
                if not selected.any():
                    continue
                axis.scatter(
                    values.loc[
                        selected,
                        "ptb_validation_rank_biserial",
                    ],
                    values.loc[
                        selected,
                        "mimic_validation_rank_biserial",
                    ],
                    c=np.asarray(colors, dtype=object)[selected.to_numpy()],
                    marker=marker,
                    s=28,
                    alpha=0.8,
                    label=label,
                )
            axis.axhline(0, color="#778388", linewidth=1, linestyle="--")
            axis.axvline(0, color="#778388", linewidth=1, linestyle="--")
            axis.set_xlim(-1.02, 1.02)
            axis.set_ylim(-1.02, 1.02)
            axis.set_xlabel(
                _text(
                    spec,
                    "PTB-XL roher rank-biserialer Effekt",
                    "PTB-XL raw rank-biserial effect",
                )
            )
            axis.set_ylabel(
                _text(
                    spec,
                    "MIMIC roher rank-biserialer Effekt",
                    "MIMIC raw rank-biserial effect",
                )
            )
            axis.set_title(
                _text(
                    spec,
                    "Richtungstransfer ohne Pooling der Referenzendpunkte",
                    "Direction transfer without pooling reference endpoints",
                )
            )
            axis.text(
                0.01,
                0.01,
                _text(
                    spec,
                    "Gruen: gleiche Train-Richtung; Orange: Gegenrichtung; "
                    "Grau: nicht beurteilbar. Keine AUC-Differenz.",
                    "Green: same train direction; orange: opposite; "
                    "gray: unavailable. Not an AUC difference.",
                ),
                transform=axis.transAxes,
                fontsize=7,
                va="bottom",
            )
            axis.legend(loc="best")
            axis.grid(alpha=0.2)
            fig.tight_layout()
            figures.append(("ptb_mimic_direction_transfer", fig))

    figures.extend(_validation_detail_figures(tables, spec, plt))
    figures.extend(_group_vcg_figures(tables, spec, plt))

    phenome = tables.get("phenome_train_screening", pd.DataFrame())
    phenome_plot_scope = "all_train_tests"
    if phenome.empty:
        phenome = tables.get("phenome_replicated_hits", pd.DataFrame())
        phenome_plot_scope = "replicated_hits"
    if "feature_id" in phenome:
        phenome = phenome.loc[
            phenome["feature_id"].map(is_candidate_biomarker)
        ]
    effect_column = _first_column(
        phenome,
        ("rank_biserial", "effect_size", "validation_effect"),
    )
    if not phenome.empty and "p_value" in phenome and effect_column:
        values = phenome.copy()
        values["p_value"] = pd.to_numeric(values["p_value"], errors="coerce")
        values[effect_column] = pd.to_numeric(
            values[effect_column], errors="coerce"
        )
        values = values.loc[values["p_value"].gt(0)].head(100_000)
        if not values.empty:
            fig, axis = plt.subplots(figsize=(8.27, 5.2))
            x = np.arange(len(values))
            colors = (
                pd.factorize(values.get("body_system", "phenome"))[0]
                if "body_system" in values
                else np.zeros(len(values))
            )
            axis.scatter(
                x,
                -np.log10(values["p_value"]),
                c=colors,
                cmap="tab20",
                s=6,
                alpha=0.65,
            )
            axis.set_xlabel(_text(spec, "Phänotyp-Marker-Test", "Phenotype-marker test"))
            axis.set_ylabel("-log10(p)")
            axis.set_title(
                _text(
                    spec,
                    (
                        "Replizierte explorative Phenome-Treffer"
                        if phenome_plot_scope == "replicated_hits"
                        else "Explorativer Phenome-Scan"
                    ),
                    (
                        "Replicated exploratory phenome hits"
                        if phenome_plot_scope == "replicated_hits"
                        else "Exploratory phenome scan"
                    ),
                )
            )
            axis.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            figures.append(("phenome_manhattan", fig))

    broad = tables.get(
        "phenome_broad_discovery_replicated_hits",
        pd.DataFrame(),
    )
    if "feature_id" in broad:
        broad = broad.loc[
            broad["feature_id"].map(is_candidate_biomarker)
        ]
    if {
        "validation_effect",
        "validation_q_value",
        "discovery_domain",
        "discovery_class",
    }.issubset(broad):
        values = broad.copy()
        values["validation_effect"] = pd.to_numeric(
            values["validation_effect"],
            errors="coerce",
        )
        values["validation_q_value"] = pd.to_numeric(
            values["validation_q_value"],
            errors="coerce",
        )
        values = values.loc[
            values["validation_effect"].notna()
            & values["validation_q_value"].gt(0)
        ].sort_values("validation_q_value").head(250)
        if not values.empty:
            domains = values["discovery_domain"].fillna("unknown").astype(str)
            codes, labels = pd.factorize(domains, sort=True)
            fig, axis = plt.subplots(figsize=(8.27, 4.8))
            scatter = axis.scatter(
                values["validation_effect"],
                -np.log10(values["validation_q_value"]),
                c=codes,
                cmap="tab10",
                s=24,
                alpha=0.75,
            )
            handles, _ = scatter.legend_elements()
            axis.legend(
                handles,
                list(labels),
                title=_text(spec, "Domäne", "Domain"),
                fontsize=7,
                loc="best",
            )
            axis.axvline(0, color="#333333", linewidth=1)
            axis.set_xlabel(
                _text(
                    spec,
                    "Validierungseffekt",
                    "Validation effect",
                )
            )
            axis.set_ylabel("-log10(q)")
            axis.set_title(
                _text(
                    spec,
                    "Replizierte breite Treffer, getrennt nach Domäne",
                    "Replicated broad hits separated by domain",
                )
            )
            axis.grid(alpha=0.2)
            fig.tight_layout()
            figures.append(("broad_discovery_domains", fig))
    _annotate_figure_provenance(figures, provenance)
    return figures


def _group_vcg_figures(
    tables: dict[str, pd.DataFrame],
    spec: ReportSpec,
    plt,
) -> list[tuple[str, object]]:
    figures: list[tuple[str, object]] = []
    summary_items = [
        (name, frame)
        for name, frame in tables.items()
        if name.startswith("group_vcg_")
        and name.endswith("_group_vcg_summary")
        and not frame.empty
    ]
    for name, summary in summary_items:
        required = {
            "label_state",
            "sample_index",
            "time_ms",
            "median_x_mV",
            "median_y_mV",
            "median_z_mV",
        }
        if not required.issubset(summary):
            continue
        groups = {
            state: (
                summary.loc[summary["label_state"].astype(str).eq(state)]
                .sort_values("sample_index")
                .reset_index(drop=True)
            )
            for state in ("negative", "positive")
        }
        if any(frame.empty for frame in groups.values()):
            continue
        endpoint = str(
            summary["endpoint_id"].iloc[0]
            if "endpoint_id" in summary
            else name
        )
        split = str(
            summary["split"].iloc[0]
            if "split" in summary
            else "development"
        )
        difference_name = name.replace(
            "_group_vcg_summary",
            "_group_vcg_difference",
        )
        difference = tables.get(difference_name, pd.DataFrame())
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.0))
        projections = (
            ("x", "y", "Frontal XY", "X (mV)", "Y (mV)"),
            ("x", "z", "Sagittal XZ", "X (mV)", "Z (mV)"),
            ("y", "z", "Transversal YZ", "Y (mV)", "Z (mV)"),
        )
        colors = {"negative": "#c63546", "positive": "#087e8b"}
        labels = {
            "negative": _text(spec, "negativ", "negative"),
            "positive": _text(spec, "positiv", "positive"),
        }
        for axis, (
            horizontal,
            vertical,
            title,
            horizontal_label,
            vertical_label,
        ) in zip(axes.flat[:3], projections):
            for state in ("negative", "positive"):
                group = groups[state]
                axis.plot(
                    group[f"median_{horizontal}_mV"],
                    group[f"median_{vertical}_mV"],
                    color=colors[state],
                    linewidth=1.7,
                    label=labels[state],
                )
                low_columns = (
                    f"{horizontal}_ci_low_mV",
                    f"{vertical}_ci_low_mV",
                )
                high_columns = (
                    f"{horizontal}_ci_high_mV",
                    f"{vertical}_ci_high_mV",
                )
                if set(low_columns + high_columns).issubset(group):
                    axis.plot(
                        group[low_columns[0]],
                        group[low_columns[1]],
                        color=colors[state],
                        linewidth=0.6,
                        alpha=0.35,
                        linestyle=":",
                    )
                    axis.plot(
                        group[high_columns[0]],
                        group[high_columns[1]],
                        color=colors[state],
                        linewidth=0.6,
                        alpha=0.35,
                        linestyle=":",
                    )
                zero_index = int(
                    np.argmin(
                        np.abs(group["time_ms"].to_numpy(dtype=float))
                    )
                )
                axis.scatter(
                    [group[f"median_{horizontal}_mV"].iloc[zero_index]],
                    [group[f"median_{vertical}_mV"].iloc[zero_index]],
                    color=colors[state],
                    s=15,
                    zorder=4,
                )
            axis.set_title(title)
            axis.set_xlabel(horizontal_label)
            axis.set_ylabel(vertical_label)
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=7)

        delta_axis = axes.flat[3]
        if {
            "time_ms",
            "delta_magnitude_mV",
        }.issubset(difference):
            difference = difference.sort_values("sample_index")
            delta_axis.plot(
                difference["time_ms"],
                difference["delta_magnitude_mV"],
                color="#c89419",
                linewidth=1.6,
            )
            if {
                "delta_magnitude_ci_low_mV",
                "delta_magnitude_ci_high_mV",
            }.issubset(difference):
                delta_axis.fill_between(
                    difference["time_ms"].to_numpy(dtype=float),
                    difference[
                        "delta_magnitude_ci_low_mV"
                    ].to_numpy(dtype=float),
                    difference[
                        "delta_magnitude_ci_high_mV"
                    ].to_numpy(dtype=float),
                    color="#c89419",
                    alpha=0.20,
                    linewidth=0,
                )
            delta_axis.axvline(0.0, color="#555555", linewidth=0.8)
            delta_axis.set_xlabel(
                _text(spec, "Zeit relativ zur R-Zacke (ms)", "Time from R peak (ms)")
            )
            delta_axis.set_ylabel(
                _text(
                    spec,
                    "|positiv - negativ| (mV)",
                    "|positive - negative| (mV)",
                )
            )
            delta_axis.set_title(
                _text(
                    spec,
                    "Räumlicher Differenzbetrag",
                    "Spatial difference magnitude",
                )
            )
            delta_axis.grid(alpha=0.2)
        else:
            delta_axis.text(
                0.5,
                0.5,
                _text(
                    spec,
                    "Differenztabelle nicht verfügbar",
                    "Difference table unavailable",
                ),
                ha="center",
                va="center",
                transform=delta_axis.transAxes,
            )
            delta_axis.set_axis_off()
        fig.suptitle(
            _text(
                spec,
                f"Gruppen-VCG: {endpoint} · {split}",
                f"Group VCG: {endpoint} · {split}",
            )
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figures.append((f"group_vcg_{endpoint}_{split}", fig))
    return figures


def _annotate_figure_provenance(
    figures: list[tuple[str, object]],
    provenance: dict,
) -> None:
    snapshot = (
        provenance.get("feature_snapshot_hash")
        or provenance.get("snapshot_hash")
        or "not-recorded"
    )
    context = (
        f"Snapshot {str(snapshot)[:16]} | "
        f"Split {provenance.get('evaluation_splits', 'not_recorded')} | "
        f"{provenance.get('artifact_completeness', 'unknown')} | "
        f"Acceptance {provenance.get('acceptance_status', 'not_run')} | "
        f"Power {provenance.get('power_statuses', 'not_recorded')} | "
        f"{provenance.get('exploration_grade', 'unknown')}"
    )
    for _identifier, figure in figures:
        figure.text(
            0.5,
            0.006,
            context,
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#52616b",
        )
        figure.subplots_adjust(
            bottom=max(float(figure.subplotpars.bottom), 0.10)
        )


def _validation_detail_figures(
    tables: dict[str, pd.DataFrame],
    spec: ReportSpec,
    plt,
) -> list[tuple[str, object]]:
    figures: list[tuple[str, object]] = []
    curves = _preferred_validation_rows(
        tables.get("performance_curve_points", pd.DataFrame())
    )
    if {"curve_type", "x", "y"}.issubset(curves):
        kind = _string_series(curves, "point_kind")
        continuous = curves.loc[kind.eq("continuous_marker_curve")].copy()
        group_columns = [
            name
            for name in ("endpoint_id", "feature_id")
            if name in continuous
        ]
        groups = (
            list(continuous.groupby(group_columns, sort=True))[:6]
            if group_columns and not continuous.empty
            else []
        )
        if groups:
            fig, axes = plt.subplots(1, 2, figsize=(8.27, 3.9))
            axes[0].plot([0, 1], [0, 1], "--", color="#778388", linewidth=1)
            for index, (key, group) in enumerate(groups):
                key_values = key if isinstance(key, tuple) else (key,)
                label = " | ".join(str(value) for value in key_values)
                color = _figure_color(index)
                for curve_type, axis in (
                    ("roc", axes[0]),
                    ("precision_recall", axes[1]),
                ):
                    points = group.loc[
                        group["curve_type"].astype(str).eq(curve_type)
                    ]
                    if "point_order" in points:
                        points = points.sort_values("point_order")
                    axis.plot(
                        pd.to_numeric(points["x"], errors="coerce"),
                        pd.to_numeric(points["y"], errors="coerce"),
                        color=color,
                        linewidth=1.5,
                        label=label,
                    )
            references = curves.loc[kind.eq("fixed_reference_point")]
            for curve_type, axis in (
                ("roc", axes[0]),
                ("precision_recall", axes[1]),
            ):
                points = references.loc[
                    references["curve_type"].astype(str).eq(curve_type)
                ]
                axis.scatter(
                    pd.to_numeric(points.get("x"), errors="coerce"),
                    pd.to_numeric(points.get("y"), errors="coerce"),
                    marker="x",
                    color="#20272b",
                    s=34,
                    label=(
                        _text(spec, "Klassische feste Regel", "Fixed classical rule")
                        if not points.empty
                        else None
                    ),
                )
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.grid(alpha=0.2)
            axes[0].set(
                xlabel="1 - specificity",
                ylabel="sensitivity",
                title="ROC",
            )
            axes[1].set(
                xlabel="recall",
                ylabel="precision",
                title=_text(spec, "Precision-Recall", "Precision-recall"),
            )
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=2,
                    fontsize=7,
                )
                fig.subplots_adjust(bottom=0.25)
            fig.suptitle(
                _text(
                    spec,
                    "Validation: kontinuierliche Marker und feste Baselines",
                    "Validation: continuous markers and fixed baselines",
                )
            )
            fig.tight_layout()
            figures.append(("validation_roc_pr", fig))

    distributions = _preferred_validation_rows(
        tables.get("score_distribution_bins", pd.DataFrame())
    )
    distribution_required = {
        "endpoint_id",
        "feature_id",
        "label",
        "bin_center",
        "fraction_within_class",
    }
    if distribution_required.issubset(distributions) and not distributions.empty:
        ranked = distributions.assign(
            _rank=_string_series(distributions, "selection_role")
            .map(
                {
                    "prespecified_primary_marker": 0,
                    "train_primary_core_representative": 1,
                    "train_sentinel_representative": 2,
                }
            )
            .fillna(9)
        ).sort_values(["_rank", "endpoint_id", "feature_id"])
        selection = ranked.iloc[0]
        selected = ranked.loc[
            ranked["endpoint_id"].astype(str).eq(str(selection["endpoint_id"]))
            & ranked["feature_id"].astype(str).eq(str(selection["feature_id"]))
        ]
        fig, axis = plt.subplots(figsize=(8.27, 4.2))
        for label, color, caption in (
            (0, "#4c78a8", _text(spec, "Negativ", "Negative")),
            (1, "#d1495b", _text(spec, "Positiv", "Positive")),
        ):
            values = selected.loc[selected["label"].eq(label)].sort_values(
                "bin_center"
            )
            axis.plot(
                pd.to_numeric(values["bin_center"], errors="coerce"),
                pd.to_numeric(
                    values["fraction_within_class"],
                    errors="coerce",
                ),
                color=color,
                linewidth=2,
                label=caption,
            )
        axis.set_xlabel("Direction-aligned train z-score")
        axis.set_ylabel(_text(spec, "Anteil je Klasse", "Fraction within class"))
        axis.set_title(
            f"{selection['endpoint_id']} | {selection['feature_id']}"
        )
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()
        figures.append(("score_distribution", fig))

    operating = _preferred_validation_rows(
        tables.get("operating_point_results", pd.DataFrame())
    )
    if {
        "specificity",
        "sensitivity",
        "operating_point",
    }.issubset(operating) and not operating.empty:
        x = pd.to_numeric(operating["specificity"], errors="coerce")
        y = pd.to_numeric(operating["sensitivity"], errors="coerce")
        valid = x.notna() & y.notna()
        operating = operating.loc[valid].head(500)
        fig, axis = plt.subplots(figsize=(8.27, 4.6))
        for index, (name, group) in enumerate(
            operating.groupby("operating_point", sort=True)
        ):
            axis.scatter(
                pd.to_numeric(group["specificity"], errors="coerce"),
                pd.to_numeric(group["sensitivity"], errors="coerce"),
                color=_figure_color(index),
                marker="x" if name == "fixed_reference_point" else "o",
                s=30,
                label=str(name),
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Specificity")
        axis.set_ylabel("Sensitivity")
        axis.set_title(
            _text(spec, "Vorab definierte Betriebspunkte", "Prespecified operating points")
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        fig.tight_layout()
        figures.append(("operating_points", fig))

    stability = tables.get("overfitting_stability_report", pd.DataFrame())
    if "analysis_type" in stability:
        stability = stability.loc[
            ~stability["analysis_type"].astype(str).eq("single_marker")
        ].copy()
        value_column = _first_column(
            stability,
            ("validation_auc", "roc_auc"),
        )
        if value_column:
            stability[value_column] = pd.to_numeric(
                stability[value_column],
                errors="coerce",
            )
            stability = stability.dropna(subset=[value_column]).head(30)
            if not stability.empty:
                fig, axis = plt.subplots(figsize=(8.27, 5.2))
                y = np.arange(len(stability))
                axis.scatter(
                    stability[value_column],
                    y,
                    color="#d1495b",
                    s=24,
                )
                axis.axvline(0.5, color="#778388", linestyle="--", linewidth=1)
                labels = stability.apply(_sensitivity_label, axis=1)
                axis.set_yticks(y, labels)
                axis.invert_yaxis()
                axis.set_xlabel("ROC-AUC")
                axis.set_title(
                    _text(
                        spec,
                        "Zeitfenster-, Transformations- und Subgruppenstabilität",
                        "Window, transform, and subgroup stability",
                    )
                )
                axis.grid(axis="x", alpha=0.2)
                fig.tight_layout()
                figures.append(("sensitivity_stability", fig))

    adjusted = _preferred_validation_rows(
        tables.get("adjusted_model_results", pd.DataFrame())
    )
    decision = _preferred_validation_rows(
        tables.get("decision_curve_results", pd.DataFrame())
    )
    has_adjusted = {
        "calibration_slope",
        "brier_score",
    }.issubset(adjusted) and not adjusted.empty
    has_decision = {
        "model_id",
        "threshold_probability",
        "net_benefit",
    }.issubset(decision) and not decision.empty
    if has_adjusted or has_decision:
        fig, axes = plt.subplots(1, 2, figsize=(8.27, 3.9))
        if has_adjusted:
            axes[0].scatter(
                pd.to_numeric(adjusted["calibration_slope"], errors="coerce"),
                pd.to_numeric(adjusted["brier_score"], errors="coerce"),
                color="#087e8b",
                s=28,
            )
            axes[0].axvline(1, color="#778388", linestyle="--", linewidth=1)
        axes[0].set(
            xlabel="Calibration slope",
            ylabel="Brier score",
            title=_text(spec, "Kalibration", "Calibration"),
        )
        if has_decision:
            for index, (model_id, group) in enumerate(
                decision.groupby("model_id", sort=True)
            ):
                group = group.sort_values("threshold_probability")
                axes[1].plot(
                    pd.to_numeric(
                        group["threshold_probability"],
                        errors="coerce",
                    ),
                    pd.to_numeric(group["net_benefit"], errors="coerce"),
                    color=_figure_color(index),
                    linewidth=1.5,
                    label=str(model_id),
                )
        axes[1].set(
            xlabel="Threshold probability",
            ylabel="Net benefit",
            title=_text(spec, "Decision Curve", "Decision curve"),
        )
        for axis in axes:
            axis.grid(alpha=0.2)
        if has_decision:
            axes[1].legend(fontsize=7)
        fig.tight_layout()
        figures.append(("adjusted_utility", fig))
    return figures


def _preferred_validation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "evaluation_split" not in frame:
        return frame
    validation = frame.loc[
        frame["evaluation_split"].astype(str).eq("validation")
    ]
    return validation if not validation.empty else frame


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[column].astype(str)


def _sensitivity_label(row: pd.Series) -> str:
    parts = [
        str(row.get("endpoint_id", "")),
        str(row.get("analysis_type", "")),
    ]
    for column in ("transform_method", "window_id", "subgroup_id"):
        value = row.get(column)
        if pd.notna(value) and str(value):
            parts.append(str(value))
    return " | ".join(part for part in parts if part)[:100]


def _figure_color(index: int) -> str:
    palette = (
        "#087e8b",
        "#d1495b",
        "#4c78a8",
        "#f0a202",
        "#287a45",
        "#8f5b9f",
    )
    return palette[index % len(palette)]


def _report_table_title(spec: ReportSpec, identifier: str) -> str:
    if spec.language == "de":
        return TABLE_TITLES_DE.get(
            identifier, identifier.replace("_", " ").title()
        )
    return TABLE_TITLES_EN.get(
        identifier, identifier.replace("_", " ").title()
    )


def _report_figure_title(spec: ReportSpec, identifier: str) -> str:
    if spec.language == "de":
        return FIGURE_TITLES_DE.get(
            identifier, identifier.replace("_", " ").title()
        )
    return identifier.replace("_", " ").title()


def _report_preview(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
    maximum_columns: int,
) -> pd.DataFrame:
    preferred = (
        "clinical_target_de",
        "endpoint_name_de",
        "target_name",
        "clinical_category_de",
        "biomarker_name_de",
        "feature_name_de",
        "display_name_de",
        "status_de",
        "stage_de",
        "evidence_grade_de",
        "formula",
        "specificity_95_threshold",
        "specificity_95_gray_zone",
        "sensitivity",
        "specificity",
        "roc_auc",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "validation_auc",
        "validation_q_value",
        "adjusted_q_value",
        "interpretation_de",
        "endpoint_id",
        "feature_id",
        "criterion_id",
    )
    columns = [column for column in preferred if column in frame]
    columns.extend(column for column in frame if column not in columns)
    return frame.loc[:, columns[:maximum_columns]].head(maximum_rows).copy()


def _report_summary_metrics(
    tables: dict[str, pd.DataFrame],
    provenance: dict,
) -> tuple[tuple[str, str], ...]:
    criteria = tables.get("criterion_candidates", pd.DataFrame())
    confirmed_criteria = 0
    if not criteria.empty:
        stage = _string_series(criteria, "stage")
        confirmed_criteria = int(
            stage.isin(
                (
                    "validation_replicated",
                    "frozen",
                    "locked_test_validated",
                    "external_validated",
                )
            ).sum()
        )
    endpoint_ids: set[str] = set()
    for frame in tables.values():
        for column in ("endpoint_id", "label_id"):
            if column in frame:
                endpoint_ids.update(
                    str(value)
                    for value in frame[column].dropna().unique()
                )
                break
    discovery = tables.get(
        "phenome_new_associations_explorer", pd.DataFrame()
    )
    replicated_associations = 0
    if not discovery.empty:
        status = _string_series(discovery, "status_id")
        replicated_associations = int(
            status.isin(("replicated", "replicated_association")).sum()
        )
    failures = tables.get("validation_failure_report", pd.DataFrame())
    blockers = 0
    if not failures.empty:
        status = _string_series(failures, "status")
        blockers = int(
            (~status.isin(("", "ok", "not_available"))).sum()
        )
    return (
        ("klinische Ziele im Bericht", f"{len(endpoint_ids):,}"),
        ("Kriterienkandidaten", f"{len(criteria):,}"),
        ("in Validation bestätigte Kriterien", f"{confirmed_criteria:,}"),
        ("replizierte neue Zusammenhänge", f"{replicated_associations:,}"),
        ("ausgewiesene Analyseblocker", f"{blockers:,}"),
        ("Artefaktstatus", str(provenance.get("artifact_completeness", ""))),
    )


def _render_executive_summary_html(
    spec: ReportSpec,
    tables: dict[str, pd.DataFrame],
    provenance: dict,
) -> str:
    heading = _text(spec, "Ergebnis auf einen Blick", "Results at a glance")
    cards = "".join(
        "<div class=\"metric\">"
        f"<strong>{html.escape(value)}</strong>"
        f"<span>{html.escape(label)}</span></div>"
        for label, value in _report_summary_metrics(tables, provenance)
    )
    return f"<section><h2>{html.escape(heading)}</h2><div class=\"metrics\">{cards}</div></section>"


def _render_self_contained_html(
    spec: ReportSpec,
    tables: dict[str, pd.DataFrame],
    figures: list[tuple[str, object]],
    provenance: dict,
) -> str:
    title = _report_title(spec)
    figure_html = []
    for identifier, figure in figures:
        payload = io.BytesIO()
        figure.savefig(payload, format="png", dpi=150, bbox_inches="tight")
        encoded = base64.b64encode(payload.getvalue()).decode("ascii")
        display_title = _report_figure_title(spec, identifier)
        figure_html.append(
            f"<section><h2>{html.escape(display_title)}</h2>"
            f"<p class=\"artifact-id\">Abbildung: {html.escape(identifier)}</p>"
            f"<img alt=\"{html.escape(identifier)}\" "
            f"src=\"data:image/png;base64,{encoded}\"></section>"
        )
    table_html = []
    for name, frame in tables.items():
        preview = _report_preview(frame, maximum_rows=100, maximum_columns=16)
        display_title = _report_table_title(spec, name)
        table_html.append(
            f"<section><h2>{html.escape(display_title)}</h2>"
            f"<p class=\"artifact-id\">Artefakt: {html.escape(name)}</p>"
            f"<p>{len(frame):,} Zeilen; Vorschau auf 100 Zeilen und "
            f"16 Spalten begrenzt. Vollständige Tabellen stehen in Parquet/CSV.</p>"
            f"{preview.to_html(index=False, border=0, escape=True)}</section>"
        )
    summary_html = _render_executive_summary_html(spec, tables, provenance)
    return f"""<!doctype html>
<html lang="{html.escape(spec.language)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
@page {{ size: A4; margin: 16mm; }}
body {{ font-family: Arial, sans-serif; color: #182226; margin: 0 auto;
       max-width: 1180px; padding: 28px; line-height: 1.4; }}
h1 {{ font-size: 28px; margin-bottom: 6px; }}
h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #ccd5d8;
      padding-bottom: 5px; }}
.notice {{ border-left: 5px solid #d1495b; padding: 10px 14px;
           background: #f7f9fa; }}
.meta {{ color: #52616b; font-size: 13px; }}
.artifact-id {{ color: #64747b; font: 11px monospace; margin-top: -8px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px; margin: 18px 0 26px; }}
.metric {{ border-left: 4px solid #087e8b; padding: 9px 12px;
           background: #f2f6f7; }}
.metric strong {{ display: block; font-size: 22px; }}
.metric span {{ color: #44545b; font-size: 12px; }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 11px;
         display: block; overflow-x: auto; }}
th, td {{ border-bottom: 1px solid #d9e0e2; padding: 5px 7px;
          text-align: left; vertical-align: top; }}
th {{ background: #edf2f3; position: sticky; top: 0; }}
@media print {{ body {{ padding: 0; }} section {{ break-inside: avoid; }}
               table {{ font-size: 8px; }} }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="meta">Run {html.escape(spec.run_id)} | Snapshot
{html.escape(str(provenance['snapshot_hash']))} |
Status: {html.escape(str(provenance['analysis_status']))} |
Vollständigkeit: {html.escape(str(provenance['artifact_completeness']))} |
Splits: {html.escape(str(provenance['evaluation_splits']))} |
Locked test: {html.escape(str(provenance['locked_test_status']))} |
Acceptance: {html.escape(str(provenance['acceptance_status']))}</p>
<p class="notice">{html.escape(_research_notice(spec.language))}<br>
{html.escape(MIMIC_DUA_NOTICE)}</p>
{summary_html}
{''.join(figure_html)}
{''.join(table_html)}
</body>
</html>"""


def _atomic_write_pdf(
    path: Path,
    spec: ReportSpec,
    tables: dict[str, pd.DataFrame],
    figures: list[tuple[str, object]],
    provenance: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.pdf",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        with PdfPages(temporary_name) as pdf:
            summary_metrics = _report_summary_metrics(tables, provenance)
            cover = plt.figure(figsize=(8.27, 11.69))
            cover.text(0.08, 0.90, _report_title(spec), fontsize=24, weight="bold")
            cover.text(0.08, 0.85, f"Run: {spec.run_id}", fontsize=13)
            cover.text(
                0.08,
                0.81,
                f"Snapshot: {provenance['snapshot_hash']}",
                fontsize=8,
                family="monospace",
            )
            cover.text(
                0.08,
                0.76,
                _research_notice(spec.language),
                fontsize=11,
                wrap=True,
            )
            cover.text(0.08, 0.69, MIMIC_DUA_NOTICE, fontsize=9, wrap=True)
            cover.text(
                0.08,
                0.62,
                f"Locked test: {provenance['locked_test_status']}",
                fontsize=11,
            )
            cover.text(
                0.08,
                0.57,
                (
                    f"Status: {provenance['analysis_status']} | "
                    f"completeness: {provenance['artifact_completeness']}"
                ),
                fontsize=10,
            )
            cover.text(
                0.08,
                0.53,
                (
                    f"Splits: {provenance['evaluation_splits']} | "
                    f"Power: {provenance['power_statuses']} | "
                    f"{provenance['exploration_grade']}"
                ),
                fontsize=8,
                wrap=True,
            )
            cover.text(
                0.08,
                0.49,
                (
                    f"Acceptance: {provenance['acceptance_status']} | "
                    "locked outcomes read: "
                    f"{provenance['acceptance_locked_test_outcomes_read']} | "
                    "PTB fold 10 read: "
                    f"{provenance['acceptance_ptb_fold_10_outcomes_read']}"
                ),
                fontsize=8,
                wrap=True,
            )
            metric_y = 0.42
            for label, value in summary_metrics:
                cover.text(
                    0.08,
                    metric_y,
                    f"{value}  {label}",
                    fontsize=10,
                    weight="bold" if metric_y == 0.42 else "normal",
                )
                metric_y -= 0.035
            cover.text(
                0.08,
                0.08,
                datetime.now(timezone.utc).isoformat(),
                fontsize=8,
            )
            pdf.savefig(cover)
            plt.close(cover)
            for _, figure in figures:
                pdf.savefig(figure)
            for name, frame in tables.items():
                if name == "report_provenance":
                    continue
                page = plt.figure(figsize=(8.27, 11.69))
                page.text(
                    0.06,
                    0.95,
                    _report_table_title(spec, name),
                    fontsize=16,
                    weight="bold",
                )
                page.text(
                    0.06,
                    0.925,
                    f"Artefakt: {name}",
                    fontsize=6.5,
                    family="monospace",
                    color="#64747b",
                )
                preview = _report_preview(
                    frame, maximum_rows=28, maximum_columns=8
                )
                if preview.empty:
                    page.text(0.06, 0.88, "No rows available.", fontsize=10)
                else:
                    preview = preview.map(_short_cell)
                    axis = page.add_axes((0.04, 0.08, 0.92, 0.82))
                    axis.axis("off")
                    table = axis.table(
                        cellText=preview.to_numpy(),
                        colLabels=preview.columns,
                        cellLoc="left",
                        colLoc="left",
                        loc="upper left",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(5.7)
                    table.scale(1.0, 1.25)
                    page.text(
                        0.06,
                        0.91,
                        f"{len(frame):,} rows; preview limited to 28.",
                        fontsize=8,
                    )
                pdf.savefig(page)
                plt.close(page)
        atomic_replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_write_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    excel_tables = _excel_export_tables(tables)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.xlsx",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        with pd.ExcelWriter(temporary_name, engine="openpyxl") as writer:
            used: set[str] = set()
            for name, frame in excel_tables.items():
                for chunk_name, chunk in _excel_table_chunks(name, frame):
                    sheet = _unique_sheet_name(chunk_name, used)
                    chunk.to_excel(
                        writer,
                        sheet_name=sheet,
                        index=False,
                    )
        atomic_replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _excel_export_tables(
    tables: dict[str, pd.DataFrame],
    *,
    max_total_cells: int = EXCEL_MAX_TOTAL_CELLS,
    max_rows_per_table: int = EXCEL_MAX_ROWS_PER_TABLE,
    min_rows_per_table: int = EXCEL_MIN_ROWS_PER_TABLE,
) -> dict[str, pd.DataFrame]:
    if max_total_cells < 1:
        raise ValueError("max_total_cells must be positive")
    if max_rows_per_table < 1:
        raise ValueError("max_rows_per_table must be positive")
    if min_rows_per_table < 1:
        raise ValueError("min_rows_per_table must be positive")
    if not tables:
        return {
            "export_index": pd.DataFrame(
                columns=(
                    "table_name",
                    "source_rows",
                    "workbook_rows",
                    "omitted_rows",
                    "selection_rule",
                    "complete_formats",
                )
            )
        }

    names = list(tables)
    column_counts = {
        name: max(1, len(tables[name].columns))
        for name in names
    }
    minimum_one_row_cost = sum(
        column_counts[name]
        for name in names
        if len(tables[name]) > 0
    )
    if minimum_one_row_cost > max_total_cells:
        raise ValueError(
            "max_total_cells is too small to include one row per table"
        )
    minimum = min_rows_per_table
    minimum_cost = sum(
        min(len(tables[name]), max_rows_per_table, minimum)
        * column_counts[name]
        for name in names
    )
    if minimum_cost > max_total_cells:
        minimum = max(
            1,
            max_total_cells // max(1, sum(column_counts.values())),
        )
    allocated = {
        name: min(len(tables[name]), max_rows_per_table, minimum)
        for name in names
    }
    used_cells = sum(
        allocated[name] * column_counts[name]
        for name in names
    )
    remaining_cells = max(0, max_total_cells - used_cells)
    active = [
        name
        for name in names
        if allocated[name] < min(len(tables[name]), max_rows_per_table)
    ]
    while remaining_cells > 0 and active:
        share = max(1, remaining_cells // len(active))
        progressed = False
        next_active: list[str] = []
        for name in active:
            columns = column_counts[name]
            capacity = (
                min(len(tables[name]), max_rows_per_table)
                - allocated[name]
            )
            extra = min(capacity, share // columns)
            if extra < 1 and remaining_cells >= columns:
                extra = 1
            if extra > 0 and extra * columns <= remaining_cells:
                allocated[name] += extra
                remaining_cells -= extra * columns
                progressed = True
            if allocated[name] < min(
                len(tables[name]),
                max_rows_per_table,
            ):
                next_active.append(name)
        if not progressed:
            break
        active = next_active

    index_rows = []
    export_tables: dict[str, pd.DataFrame] = {}
    for name in names:
        source_rows = len(tables[name])
        workbook_rows = allocated[name]
        index_rows.append(
            {
                "table_name": name,
                "source_rows": source_rows,
                "workbook_rows": workbook_rows,
                "omitted_rows": source_rows - workbook_rows,
                "selection_rule": (
                    "complete"
                    if source_rows == workbook_rows
                    else "source_order_head_preview"
                ),
                "complete_formats": "CSV; Parquet",
            }
        )
        export_tables[name] = tables[name].head(workbook_rows)
    return {
        "export_index": pd.DataFrame(index_rows),
        **export_tables,
    }


def _excel_table_chunks(
    name: str,
    frame: pd.DataFrame,
    *,
    max_data_rows: int = 1_048_575,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    if max_data_rows < 1:
        raise ValueError("max_data_rows must be positive")
    if len(frame) <= max_data_rows:
        return ((name, frame),)
    chunk_count = (len(frame) + max_data_rows - 1) // max_data_rows
    width = max(2, len(str(chunk_count)))
    return tuple(
        (
            f"{name}_part_{part_index + 1:0{width}d}",
            frame.iloc[
                part_index * max_data_rows:
                min((part_index + 1) * max_data_rows, len(frame))
            ],
        )
        for part_index in range(chunk_count)
    )


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp.csv",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary_name, index=False)
        atomic_replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        non_null = result[column].dropna()
        if non_null.empty:
            continue
        if non_null.map(
            lambda value: isinstance(value, (dict, list, tuple, set))
        ).any():
            result[column] = result[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (dict, list, tuple, set))
                    else value
                )
            )
    return result


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    return next((name for name in names if name in frame), None)


def _unique_sheet_name(name: str, used: set[str]) -> str:
    base = "".join(
        character if character not in r"[]:*?/\\" else "_"
        for character in name
    )[:31] or "table"
    candidate = base
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _short_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return text if len(text) <= 42 else f"{text[:39]}..."


def _close_figures(figures: list[tuple[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for _, figure in figures:
        plt.close(figure)


def _report_title(spec: ReportSpec) -> str:
    type_names = {
        ReportType.STUDY: ("Studienbericht", "Study report"),
        ReportType.ENDPOINT: ("Endpunktbericht", "Endpoint report"),
        ReportType.MARKER: ("Markerbericht", "Marker report"),
        ReportType.CASE: ("Fallbericht", "Case report"),
        ReportType.JOB_QC: ("Job- und Qualitätsbericht", "Job and quality report"),
    }
    return _text(spec, *type_names[spec.report_type])


def _research_notice(language: str) -> str:
    return RESEARCH_NOTICE_DE if language == "de" else RESEARCH_NOTICE_EN


def _text(spec: ReportSpec, german: str, english: str) -> str:
    return german if spec.language == "de" else english
