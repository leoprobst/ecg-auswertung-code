from __future__ import annotations

import json
import inspect
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .label_registry import LABEL_RULE_VERSION, default_label_definitions
from .models import LabelDefinition, TargetKind
from .selection_bias import (
    MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION,
    endpoint_verification_reachability,
)
from .run_log import profile_query
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_content_fingerprint,
)


LABEL_STAGE_VERSION = "2.27.0"
WINDOW_SENSITIVITY_VERSION = "1.2.0"
DEVELOPMENT_LABEL_ATTESTATION_VERSION = "1.0.0"
DEVELOPMENT_LABEL_ATTESTATION_NAME = "development_label_attestation.json"


@dataclass(frozen=True)
class SQLLabelRule:
    state: str
    eligibility: str
    confidence: str
    time_delta_hours: str = "NULL::DOUBLE"
    time_precision: str = "'source_specific'"
    evidence_ids: str = "'[]'"
    evidence_summary: str = "''"


def label_implementation_payload() -> dict:
    definitions = default_label_definitions()
    rules = _label_rules()
    return {
        "label_stage_version": LABEL_STAGE_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "window_sensitivity_version": WINDOW_SENSITIVITY_VERSION,
        "definitions": [definition.to_dict() for definition in definitions],
        "sql_rules": {
            label_id: {
                "state": rule.state,
                "eligibility": rule.eligibility,
                "confidence": rule.confidence,
                "time_delta_hours": rule.time_delta_hours,
                "time_precision": rule.time_precision,
                "evidence_ids": rule.evidence_ids,
                "evidence_summary": rule.evidence_summary,
            }
            for label_id, rule in sorted(rules.items())
        },
        "query_implementations": {
            function.__name__: inspect.getsource(function)
            for function in (
                _long_label_query,
                _wide_label_query,
                _index_cohort_query,
                _window_sensitivity_cohort_query,
                _reference_blind_window_index_query,
                _window_profile_select,
                _parquet_relation,
            )
        },
    }


def label_implementation_hash() -> str:
    return content_hash(label_implementation_payload())


def build_clinical_labels(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    context_path = config.evidence_root / "clinical_context_features.parquet"
    if not context_path.exists():
        raise FileNotFoundError(
            f"Clinical context is missing: {context_path}. Run build-evidence first."
        )
    definitions = default_label_definitions()
    rules = _label_rules()
    implementation_hash = label_implementation_hash()
    input_hash = content_hash(
        {
            "context": file_content_fingerprint(context_path),
            "definitions": [definition.to_dict() for definition in definitions],
            "implemented_rules": sorted(rules),
            "label_implementation_hash": implementation_hash,
        }
    )
    config_hash = scientific_config_hash(config)
    manifest_path = config.labels_root / "labels.manifest.json"
    definitions_path = config.labels_root / "label_definitions.json"
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, LABEL_STAGE_VERSION):
            artifact_integrity = verify_clinical_label_artifacts(config)
            summary_json_path = config.labels_root / "label_summary.json"
            summary_payload = json.loads(
                summary_json_path.read_text(encoding="utf-8")
            )
            if summary_payload.get("artifact_integrity") != artifact_integrity:
                summary_payload["artifact_integrity"] = artifact_integrity
                atomic_write_json(summary_json_path, summary_payload)
            # Reusing finished labels unchanged: the attestation has to cover
            # exactly them, so a superseded one is fatal here.
            verify_development_label_attestation(config)
            return previous
        if development_label_attestation_path(config).is_file():
            # Still current: the labels were attested and nothing they depend
            # on has moved, so reuse them. Superseded: fall through and
            # rebuild, which is what attest-labels will then attest.
            still_current = verify_development_label_attestation(
                config, require_current=False
            )
            if still_current is not None and (
                previous.status == "complete"
                and all(
                    Path(path).exists()
                    for path in previous.output_paths
                )
            ):
                return previous
            if still_current is None and progress:
                progress(
                    "Label-Attestierung ist ueberholt; Zielgruppen werden "
                    "neu gebildet"
                )

    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
        spill_limit_gb=LABEL_BUILD_SPILL_GIB,
    )
    long_path = config.labels_root / "clinical_label_evidence.parquet"
    wide_path = config.labels_root / "clinical_labels_wide.parquet"
    cohort_path = config.labels_root / "endpoint_index_cohorts.parquet"
    summary_path = config.labels_root / "label_summary.parquet"
    reachability_path = (
        config.labels_root / "endpoint_verification_reachability.parquet"
    )
    split_summary_path = config.labels_root / "split_summary.parquet"
    window_cohort_path = (
        config.labels_root / "endpoint_window_sensitivity_cohorts.parquet"
    )
    window_summary_path = (
        config.labels_root / "label_window_sensitivity_summary.parquet"
    )
    summary_json_path = config.labels_root / "label_summary.json"
    attestation_path = (
        config.labels_root / DEVELOPMENT_LABEL_ATTESTATION_NAME
    )
    sealed_root = config.labels_root / "sealed"
    sealed_root.mkdir(parents=True, exist_ok=True)
    locked_long_path = sealed_root / "locked_clinical_label_evidence.parquet"
    locked_wide_path = sealed_root / "locked_clinical_labels_wide.parquet"
    locked_cohort_path = sealed_root / "locked_endpoint_index_cohorts.parquet"
    locked_window_cohort_path = (
        sealed_root / "locked_endpoint_window_sensitivity_cohorts.parquet"
    )
    seal_path = sealed_root / "locked_test_seal.json"
    staging_path = (
        config.labels_root / f".all_label_evidence.{os.getpid()}.parquet"
    )
    window_staging_path = (
        config.labels_root
        / f".all_window_sensitivity_cohorts.{os.getpid()}.parquet"
    )

    try:
        with catalog.connect() as connection:
            for statement in label_build_session_options(config):
                connection.execute(statement)
            if progress:
                progress("Erzeuge transparente vierstufige klinische Labels")
            all_count = _copy_query_atomic(
                connection,
                _long_label_query(context_path, definitions, rules),
                staging_path,
            )
            staging_sql = _sql_path(staging_path)
            long_count = _copy_query_atomic(
                connection,
                (
                    f"SELECT * FROM read_parquet('{staging_sql}') "
                    "WHERE split IN ('train', 'validation')"
                ),
                long_path,
            )
            locked_long_count = _copy_query_atomic(
                connection,
                (
                    f"SELECT * FROM read_parquet('{staging_sql}') "
                    "WHERE split = 'locked_test'"
                ),
                locked_long_path,
            )

            if progress:
                progress("Erzeuge getrennte ECG-weite Labelansichten")
            wide_count = _copy_query_atomic(
                connection,
                _wide_label_query(long_path, definitions),
                wide_path,
            )
            locked_wide_count = _copy_query_atomic(
                connection,
                _wide_label_query(locked_long_path, definitions),
                locked_wide_path,
            )

            if progress:
                progress("Waehle getrennte primaere Index-ECGs")
            cohort_count = _copy_query_atomic(
                connection,
                _index_cohort_query(long_path),
                cohort_path,
            )
            locked_cohort_count = _copy_query_atomic(
                connection,
                _index_cohort_query(locked_long_path),
                locked_cohort_path,
            )
            if progress:
                progress(
                    "Erzeuge versiegelte Troponin-, Kalium- und Echo-"
                    "Zeitfensterkohorten"
                )
            all_window_count = _copy_query_atomic(
                connection,
                _window_sensitivity_cohort_query(context_path, rules),
                window_staging_path,
            )
            window_staging_sql = _sql_path(window_staging_path)
            window_count = _copy_query_atomic(
                connection,
                (
                    f"SELECT * FROM read_parquet('{window_staging_sql}') "
                    "WHERE split IN ('train', 'validation')"
                ),
                window_cohort_path,
            )
            locked_window_count = _copy_query_atomic(
                connection,
                (
                    f"SELECT * FROM read_parquet('{window_staging_sql}') "
                    "WHERE split = 'locked_test'"
                ),
                locked_window_cohort_path,
            )
            window_summary = connection.execute(
                _window_sensitivity_summary_query(window_cohort_path)
            ).fetchdf()

            if progress:
                progress("Berechne Entwicklungs-QC ohne Locked-Test-Praevalenzen")
            label_summary = connection.execute(
                _label_summary_query(long_path, cohort_path)
            ).fetchdf()
            split_summary = connection.execute(
                _split_summary_query(cohort_path)
            ).fetchdf()
    finally:
        for temporary_path in (staging_path, window_staging_path):
            if temporary_path.exists():
                resolved = temporary_path.resolve()
                if not resolved.is_relative_to(config.labels_root.resolve()):
                    raise RuntimeError(
                        "Refusing to remove staging file outside label root: "
                        f"{resolved}"
                    )
                resolved.unlink()

    atomic_write_parquet(label_summary, summary_path)
    # Whether a diagnostic-accuracy claim is reachable at all is fixed by how
    # often the reference was obtained, not by anything the analysis does.
    # Written next to the summary so that "not confirmed" can be read as "the
    # reference was never obtained" where that is what it means.
    reachability = endpoint_verification_reachability(
        label_summary.rename(
            columns={"unavailable_count": "unavailable_ecg_count"}
        ),
        minimum_verification_rate=MINIMUM_VERIFICATION_RATE_FOR_CONFIRMATION,
    )
    atomic_write_parquet(reachability, reachability_path)
    atomic_write_parquet(split_summary, split_summary_path)
    atomic_write_parquet(window_summary, window_summary_path)
    atomic_write_json(
        definitions_path,
        {
            "rule_version": LABEL_RULE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "definitions": [
                definition.to_dict()
                for definition in definitions
            ],
        },
    )
    artifact_integrity = verify_clinical_label_artifacts(config)
    summary_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": LABEL_RULE_VERSION,
        "label_count": len(definitions),
        "implemented_rule_count": len(set(rules) & {d.label_id for d in definitions}),
        "labels": _summary_records(label_summary),
        "splits": _summary_records(split_summary),
        "artifact_integrity": artifact_integrity,
        "locked_test": {
            "status": "sealed",
            "prevalence_reported": False,
            "association_view_registered": False,
        },
        "scientific_caveat": (
            "retrospective_proxy labels are composite reference endpoints, "
            "not direct diagnoses or angiographic ground truth; MIMIC-IV-ECG "
            "device clocks were not externally synchronized, so encounter "
            "overlap and narrow EHR windows remain temporal sensitivity "
            "assumptions rather than minute-resolved ground truth"
        ),
    }
    atomic_write_json(summary_json_path, summary_payload)
    # Audit 5.3.9: the measurement sensitivities as a fixed part of every
    # label build, on the development splits only.
    sensitivity_path = config.labels_root / "label_measurement_sensitivity.json"
    atomic_write_json(sensitivity_path, measurement_sensitivity_summary(config))
    atomic_write_json(
        seal_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "sealed",
            "stage_version": LABEL_STAGE_VERSION,
            "rule_version": LABEL_RULE_VERSION,
            "row_count": locked_long_count,
            "ecg_count": locked_wide_count,
            "endpoint_index_row_count": locked_cohort_count,
            "window_sensitivity_row_count": locked_window_count,
            "prevalence_disclosed": False,
            "duckdb_view_registered": False,
            "access_policy": "evaluate-test only after immutable freeze",
        },
    )
    _write_development_label_attestation(
        config,
        mode="generated_by_current_implementation",
        implementation_hash=implementation_hash,
        comparisons={
            "status": "generated_current",
            "all_exact": True,
            "development_row_counts": {
                "clinical_label_evidence": long_count,
                "clinical_labels_wide": wide_count,
                "endpoint_index_cohorts": cohort_count,
                "endpoint_window_sensitivity_cohorts": window_count,
            },
        },
    )

    manifest = StageManifest.complete(
        stage="build-labels",
        stage_version=LABEL_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(long_path.resolve()),
            str(wide_path.resolve()),
            str(cohort_path.resolve()),
            str(summary_path.resolve()),
            str(split_summary_path.resolve()),
            str(window_cohort_path.resolve()),
            str(window_summary_path.resolve()),
            str(summary_json_path.resolve()),
            str(sensitivity_path.resolve()),
            str(definitions_path.resolve()),
            str(attestation_path.resolve()),
            str(locked_long_path.resolve()),
            str(locked_wide_path.resolve()),
            str(locked_cohort_path.resolve()),
            str(locked_window_cohort_path.resolve()),
            str(seal_path.resolve()),
        ),
        row_counts={
            "clinical_label_evidence": long_count,
            "clinical_labels_wide": wide_count,
            "endpoint_index_cohorts": cohort_count,
            "locked_clinical_label_evidence": locked_long_count,
            "locked_clinical_labels_wide": locked_wide_count,
            "locked_endpoint_index_cohorts": locked_cohort_count,
            "endpoint_window_sensitivity_cohorts": window_count,
            "locked_endpoint_window_sensitivity_cohorts": locked_window_count,
            "all_window_sensitivity_rows": all_window_count,
            "label_window_sensitivity_summary": len(window_summary),
            "all_label_rows": all_count,
            "label_summary": len(label_summary),
            "split_summary": len(split_summary),
        },
        notes=(f"operational_config_hash={operational_config_hash(config)}",),
        output_integrity=capture_output_integrity(
            (
                long_path,
                wide_path,
                cohort_path,
                summary_path,
                split_summary_path,
                window_cohort_path,
                window_summary_path,
                summary_json_path,
                definitions_path,
                attestation_path,
                locked_long_path,
                locked_wide_path,
                locked_cohort_path,
                locked_window_cohort_path,
                seal_path,
            )
        ),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    for name, path in (
        ("clinical_label_evidence", long_path),
        ("clinical_labels_wide", wide_path),
        ("endpoint_index_cohorts", cohort_path),
        ("label_summary", summary_path),
        ("split_summary", split_summary_path),
        ("endpoint_window_sensitivity_cohorts", window_cohort_path),
        ("label_window_sensitivity_summary", window_summary_path),
    ):
        catalog.create_parquet_view(name, path)
    return manifest


def _label_rules() -> dict[str, SQLLabelRule]:
    rules: dict[str, SQLLabelRule] = {}

    rules["troponin_positive_any"] = SQLLabelRule(
        state="""
            CASE
                WHEN troponin_interval_state_standard IN ('positive', 'negative', 'indeterminate')
                    THEN troponin_interval_state_standard
                ELSE 'unavailable'
            END
        """,
        eligibility=_availability_eligibility(
            "troponin_interval_state_standard IN ('positive', 'negative')"
        ),
        confidence=_confidence(
            "troponin_interval_state_standard IN ('positive', 'negative')", high="'high'", fallback="'none'"
        ),
        time_delta_hours=(
            "CASE WHEN troponin_interval_state_standard = 'positive' "
            "THEN troponin_interval_positive_delta_hours_standard "
            "ELSE troponin_nearest_delta_hours END"
        ),
        time_precision="'timestamp'",
        evidence_ids=(
            "CASE WHEN troponin_interval_state_standard = 'positive' "
            "THEN "
            + _single_evidence_id_json(
                "troponin_interval_positive_labevent_id_standard",
                "labevent",
            )
            + " ELSE "
            + _single_evidence_id_json(
                "troponin_nearest_labevent_id",
                "labevent",
            )
            + " END"
        ),
        evidence_summary="""
            CONCAT(
                'interval state=', COALESCE(troponin_interval_state_standard, 'unavailable'),
                '; exact max URL ratio=',
                COALESCE(ROUND(troponin_max_url_ratio, 3)::VARCHAR, 'missing'),
                '; measurements=', COALESCE(troponin_count_standard::VARCHAR, '0')
            )
        """,
    )
    rules["acute_troponin_change_proxy"] = SQLLabelRule(
        state="""
            CASE
                WHEN acute_troponin_change_proxy_value THEN 'positive'
                WHEN troponin_serial_assessed THEN 'negative'
                WHEN troponin_count_standard > 0 THEN 'indeterminate'
                ELSE 'unavailable'
            END
        """,
        eligibility=_availability_eligibility(
            "COALESCE(troponin_serial_assessed, FALSE)"
        ),
        confidence=_confidence(
            "COALESCE(troponin_serial_assessed, FALSE)",
            high="'high'",
            fallback="'none'",
        ),
        time_delta_hours="troponin_event_delta_hours",
        time_precision="'timestamp'",
        evidence_ids=_paired_evidence_ids_json(
            "qualifying_first_labevent_id",
            "qualifying_last_labevent_id",
            "labevent",
        ),
        evidence_summary="""
            CONCAT(
                'serial assessed=', COALESCE(troponin_serial_assessed::VARCHAR, 'false'),
                '; max relative delta=',
                COALESCE(ROUND(troponin_max_relative_delta, 3)::VARCHAR, 'missing')
            )
        """,
    )
    # Audit 5.2.3: the rise from "not detectable" as its own phenotype. The
    # first same-assay result of the serial window is reported only as below
    # a detection limit at or below the URL, and a later one is certainly
    # above the URL. Negatives start undetectable and cannot exceed the URL
    # later; every other series is outside the phenotype's population.
    rules["troponin_rise_from_undetectable"] = SQLLabelRule(
        state="""
            CASE
                WHEN troponin_undetectable_rise_state = 'rise_from_undetectable'
                    THEN 'positive'
                WHEN troponin_undetectable_rise_state = 'undetectable_without_rise'
                    THEN 'negative'
                WHEN troponin_undetectable_rise_state = 'indeterminate'
                    THEN 'indeterminate'
                ELSE 'unavailable'
            END
        """,
        eligibility=_availability_eligibility(
            "troponin_undetectable_rise_state IN "
            "('rise_from_undetectable', 'undetectable_without_rise')"
        ),
        confidence=_confidence(
            "troponin_undetectable_rise_state IN "
            "('rise_from_undetectable', 'undetectable_without_rise')",
            high="'high'",
            fallback="'none'",
        ),
        time_delta_hours="troponin_event_delta_hours",
        time_precision="'timestamp'",
        evidence_ids=_paired_evidence_ids_json(
            "qualifying_first_labevent_id",
            "qualifying_last_labevent_id",
            "labevent",
        ),
        evidence_summary="""
            CONCAT(
                'undetectable-start state=',
                COALESCE(troponin_undetectable_rise_state, 'unavailable')
            )
        """,
    )
    demand_context = (
        "COALESCE(dx_sepsis, FALSE) OR COALESCE(dx_shock, FALSE) "
        "OR COALESCE(dx_hypoxia_respiratory_failure, FALSE) "
        "OR COALESCE(dx_atrial_fibrillation_flutter, FALSE) "
        "OR COALESCE(dx_ventricular_tachycardia, FALSE) "
        "OR COALESCE(dx_anemia, FALSE) "
        "OR COALESCE(dx_acute_renal_failure, FALSE) "
        "OR COALESCE(dx_chronic_kidney_disease, FALSE) "
        "OR COALESCE(hemoglobin_nearest < 8, FALSE) "
        "OR COALESCE(lactate_nearest > 2, FALSE)"
    )
    acute_coronary_hard_negative = (
        "COALESCE(troponin_serial_assessed, FALSE) AND (("
        "NOT COALESCE(acute_troponin_change_proxy_value, FALSE) "
        "AND NOT COALESCE(dx_acute_mi, FALSE) "
        "AND NOT COALESCE(proc_pci, FALSE) "
        "AND NOT COALESCE(proc_coronary_angiography, FALSE)) OR ("
        "COALESCE(acute_troponin_change_proxy_value, FALSE) "
        "AND (" + demand_context + ") "
        "AND NOT COALESCE(dx_acute_mi, FALSE) "
        "AND NOT COALESCE(proc_pci, FALSE) "
        "AND NOT COALESCE(proc_coronary_angiography, FALSE)))"
    )
    acute_coronary_partial_evidence = (
        "(troponin_count_standard > 0 "
        "OR COALESCE(dx_acute_mi, FALSE) "
        "OR COALESCE(proc_pci, FALSE) "
        "OR COALESCE(proc_coronary_angiography, FALSE)) "
        "AND NOT (" + acute_coronary_hard_negative + ")"
    )
    rules["acute_mi_high_specificity"] = _composite_rule(
        positive=(
            "COALESCE(acute_troponin_change_proxy_value, FALSE) "
            "AND COALESCE(dx_acute_mi, FALSE) "
            "AND COALESCE(proc_pci, FALSE) AND pci_day_delta = 0"
        ),
        assessed="COALESCE(troponin_serial_assessed, FALSE)",
        negative=acute_coronary_hard_negative,
        indeterminate=acute_coronary_partial_evidence,
        summary=(
            "'acute injury + acute-MI ICD + same-calendar-day PCI; "
            "controls require serial troponin without coronary evidence or "
            "a non-coronary demand context; procedure time precision=date'"
        ),
        positive_confidence="'medium'",
        time_delta_hours="troponin_event_delta_hours",
        time_precision="'mixed_timestamp_and_procedure_date'",
        evidence_ids=_paired_evidence_ids_json(
            "qualifying_first_labevent_id",
            "qualifying_last_labevent_id",
            "labevent",
        ),
    )
    rules["acute_mi_probable"] = _composite_rule(
        positive=(
            "COALESCE(acute_troponin_change_proxy_value, FALSE) "
            "AND COALESCE(dx_acute_mi, FALSE) AND ("
            "(COALESCE(proc_pci, FALSE) AND pci_day_delta BETWEEN 0 AND 3) "
            "OR (COALESCE(proc_coronary_angiography, FALSE) "
            "AND cath_day_delta BETWEEN 0 AND 3) "
            "OR (COALESCE(med_antiplatelet, FALSE) "
            "AND COALESCE(med_heparin, FALSE)))"
        ),
        assessed="COALESCE(troponin_serial_assessed, FALSE)",
        negative=acute_coronary_hard_negative,
        indeterminate=acute_coronary_partial_evidence,
        summary=(
            "'acute injury + acute-MI ICD + cath/PCI/ACS treatment context; "
            "partial composites are indeterminate'"
        ),
        positive_confidence="'medium'",
        time_delta_hours="troponin_event_delta_hours",
        time_precision="'mixed_timestamp_and_procedure_date'",
        evidence_ids=_paired_evidence_ids_json(
            "qualifying_first_labevent_id",
            "qualifying_last_labevent_id",
            "labevent",
        ),
    )
    rules["omi_revascularization_proxy"] = _composite_rule(
        positive=(
            "COALESCE(acute_troponin_change_proxy_value, FALSE) "
            "AND COALESCE(proc_pci, FALSE) AND pci_day_delta BETWEEN 0 AND 1 "
            "AND (COALESCE(dx_acute_mi, FALSE) OR "
            "REGEXP_MATCHES(LOWER(COALESCE(chief_complaint, '')), "
            "'chest pain|chest pressure|angina'))"
        ),
        assessed="COALESCE(troponin_serial_assessed, FALSE)",
        negative=acute_coronary_hard_negative,
        indeterminate=acute_coronary_partial_evidence,
        summary=(
            "'revascularization proxy; partial composites are indeterminate; "
            "no angiographic occlusion data'"
        ),
        positive_confidence="'medium'",
        time_delta_hours="troponin_event_delta_hours",
        time_precision="'mixed_timestamp_and_procedure_date'",
        evidence_ids=_paired_evidence_ids_json(
            "qualifying_first_labevent_id",
            "qualifying_last_labevent_id",
            "labevent",
        ),
    )
    rules["myocardial_injury_demand_context"] = _composite_rule(
        positive=(
            "troponin_interval_state_standard = 'positive' AND (" + demand_context + ") "
            "AND NOT COALESCE(proc_pci, FALSE)"
        ),
        assessed="troponin_interval_state_standard IN ('positive', 'negative')",
        summary="'myocardial injury with demand context; not a type-2 MI diagnosis'",
        positive_confidence="'medium'",
        time_delta_hours="troponin_nearest_delta_hours",
        time_precision="'mixed_timestamp_and_encounter'",
        evidence_ids=_single_evidence_id_json(
            "troponin_max_url_labevent_id",
            "labevent",
        ),
    )

    rules.update(_echo_rules())
    rules.update(_electrolyte_and_qt_rules())
    rules.update(_af_rules())
    rules.update(_conduction_progression_rules())
    rules.update(_secondary_rules())
    return rules


def _echo_rules() -> dict[str, SQLLabelRule]:
    available = "COALESCE(has_tte_30d, FALSE)"

    def echo_rule(
        positive: str,
        negative: str,
        *,
        indeterminate: str = "FALSE",
        summary: str,
    ) -> SQLLabelRule:
        positive_bool = f"COALESCE(({positive}), FALSE)"
        negative_bool = f"COALESCE(({negative}), FALSE)"
        indeterminate_bool = f"COALESCE(({indeterminate}), FALSE)"
        assessed = (
            f"({positive_bool} OR {negative_bool} OR {indeterminate_bool})"
        )
        evaluable = (
            f"({available}) AND ({assessed}) "
            f"AND NOT {indeterminate_bool} "
            f"AND NOT ({positive_bool} AND {negative_bool})"
        )
        return SQLLabelRule(
            state=f"""
                CASE
                    WHEN NOT ({available}) THEN 'unavailable'
                    WHEN NOT ({assessed}) THEN 'unavailable'
                    WHEN {indeterminate_bool}
                      OR ({positive_bool} AND {negative_bool}) THEN 'indeterminate'
                    WHEN {positive_bool} THEN 'positive'
                    WHEN {negative_bool} THEN 'negative'
                    ELSE 'indeterminate'
                END
            """,
            eligibility=_availability_eligibility(evaluable),
            confidence=f"""
                CASE
                    WHEN NOT ({evaluable}) THEN 'none'
                    WHEN has_tte_7d THEN 'high'
                    WHEN has_tte_30d THEN 'medium'
                    ELSE 'none'
                END
            """,
            time_delta_hours="nearest_tte_delta_hours",
            time_precision="'timestamp'",
            evidence_ids=(
                "CASE WHEN nearest_tte_measurement_id IS NULL THEN '[]' "
                "ELSE '[\"' || nearest_tte_measurement_id::VARCHAR || '\"]' END"
            ),
            evidence_summary=summary,
        )

    lower_wall = "LOWER(COALESCE(lv_wall_category, ''))"
    # Audit D5: ASE 2015 defines LVH by mass. A reported LVMI wins; otherwise
    # the Devereux mass reconstructed in the evidence context over the echo
    # BSA. The strict endpoint no longer reads the wall-category text, which
    # compared walls >= 1.5 cm against < 1.2 cm (extreme groups) and counted
    # eccentric LVH as a control.
    lvmi = "COALESCE(lv_mass_index, lv_mass_index_devereux_g_m2)"
    mass_positive = (
        f"({lvmi} IS NOT NULL AND ("
        f"(sex = 'M' AND {lvmi} > 115) OR "
        f"(sex = 'F' AND {lvmi} > 95)))"
    )
    mass_negative = (
        f"({lvmi} IS NOT NULL AND ("
        f"(sex = 'M' AND {lvmi} <= 115) OR "
        f"(sex = 'F' AND {lvmi} <= 95)))"
    )
    strict_positive = mass_positive
    strict_negative = mass_negative
    sensitive_positive = (
        f"{mass_positive} OR {lower_wall} LIKE '%mod symmetric%' OR "
        f"{lower_wall} LIKE '%severe symmetric%' OR "
        f"{lower_wall} LIKE '%mild symmetric%'"
    )
    sensitive_negative = f"{mass_negative} OR {lower_wall} LIKE '%normal%'"
    lvmi_summary = f"COALESCE(ROUND({lvmi}, 1)::VARCHAR, 'missing')"
    rules = {
        "echo_lvh_strict": echo_rule(
            strict_positive,
            strict_negative,
            summary=f"""
                CONCAT('LVMI=', {lvmi_summary},
                       ' g/m2; source=', CASE WHEN lv_mass_index IS NOT NULL
                       THEN 'reported' ELSE 'devereux_linear' END,
                       '; sex=', COALESCE(sex, 'missing'))
            """,
        ),
        "echo_lvh_sensitive": echo_rule(
            sensitive_positive,
            sensitive_negative,
            summary=f"""
                CONCAT('LVH sensitive; LVMI=', {lvmi_summary},
                       '; category=', COALESCE(lv_wall_category, 'missing'))
            """,
        ),
        # ASE geometry against normal geometry. Remodeling without LVH is
        # neither case nor control here.
        "lvh_concentric_ase": echo_rule(
            "lv_geometry_ase = 'concentric_hypertrophy'",
            "lv_geometry_ase = 'normal_geometry'",
            summary=f"""
                CONCAT('geometry=', COALESCE(lv_geometry_ase, 'missing'),
                       '; LVMI=', {lvmi_summary})
            """,
        ),
        "lvh_eccentric_ase": echo_rule(
            "lv_geometry_ase = 'eccentric_hypertrophy'",
            "lv_geometry_ase = 'normal_geometry'",
            summary=f"""
                CONCAT('geometry=', COALESCE(lv_geometry_ase, 'missing'),
                       '; LVMI=', {lvmi_summary})
            """,
        ),
        "lv_wall_thickening": echo_rule(
            (
                "septal_thickness_cm >= 1.2 "
                "OR posterior_wall_thickness_cm >= 1.2 "
                f"OR ({lower_wall} <> '' AND {lower_wall} NOT LIKE '%normal%')"
            ),
            (
                "septal_thickness_cm < 1.2 "
                "AND posterior_wall_thickness_cm < 1.2 "
                f"OR {lower_wall} LIKE '%normal%'"
            ),
            summary="""
                CONCAT('septum=', COALESCE(septal_thickness_cm::VARCHAR, 'missing'),
                       '; posterior=',
                       COALESCE(posterior_wall_thickness_cm::VARCHAR, 'missing'))
            """,
        ),
        "lv_dilatation_proxy": echo_rule(
            (
                "lv_internal_diameter_diastole_cm IS NOT NULL AND ("
                "(sex = 'M' AND lv_internal_diameter_diastole_cm > 5.8) OR "
                "(sex = 'F' AND lv_internal_diameter_diastole_cm > 5.2))"
            ),
            (
                "lv_internal_diameter_diastole_cm IS NOT NULL AND ("
                "(sex = 'M' AND lv_internal_diameter_diastole_cm <= 5.8) OR "
                "(sex = 'F' AND lv_internal_diameter_diastole_cm <= 5.2))"
            ),
            summary="""
                CONCAT('LVIDd cm=',
                       COALESCE(lv_internal_diameter_diastole_cm::VARCHAR,
                                'missing'),
                       '; sex=', COALESCE(sex, 'missing'))
            """,
        ),
        "concentric_remodeling_proxy": echo_rule(
            "relative_wall_thickness > 0.42",
            "relative_wall_thickness <= 0.42",
            summary="""
                CONCAT('relative wall thickness=',
                       COALESCE(ROUND(relative_wall_thickness, 3)::VARCHAR, 'missing'))
            """,
        ),
    }

    for label_id, positive, negative in (
        ("reduced_lvef_strict", "preferred_lvef < 40", "preferred_lvef >= 40"),
        ("reduced_lvef", "preferred_lvef < 50", "preferred_lvef >= 50"),
        ("preserved_lvef", "preferred_lvef >= 50", "preferred_lvef < 50"),
    ):
        rules[label_id] = echo_rule(
            positive,
            negative,
            indeterminate="COALESCE(lvef_conflict_gt_10, FALSE)",
            summary="""
                CONCAT('preferred LVEF=',
                       COALESCE(ROUND(preferred_lvef, 1)::VARCHAR, 'missing'),
                       '; method=', COALESCE(preferred_lvef_method, 'missing'),
                       '; >10pp conflict=',
                       COALESCE(lvef_conflict_gt_10::VARCHAR, 'false'))
            """,
        )

    la = "LOWER(COALESCE(la_size_category, ''))"
    ra = "LOWER(COALESCE(ra_size_category, ''))"
    rv_size = "LOWER(COALESCE(rv_size_category, ''))"
    rv_function = "LOWER(COALESCE(rv_function_category, ''))"
    # Discard an uncertain text channel, not an independent numeric measurement.
    def uncertain_category(expression: str) -> str:
        return (f"regexp_matches({expression}, "
                "'(^|[^a-z])(no|not|without|possible|possibly|suspected|uncertain|cannot)([^a-z]|$)')")

    rv_size_uncertain = uncertain_category(rv_size)
    la_uncertain = uncertain_category(la)
    ra_uncertain = uncertain_category(ra)
    la = f"CASE WHEN {la_uncertain} THEN '' ELSE {la} END"
    ra = f"CASE WHEN {ra_uncertain} THEN '' ELSE {ra} END"
    rv_function_uncertain = uncertain_category(rv_function)
    rv_size = f"CASE WHEN {rv_size_uncertain} THEN '' ELSE {rv_size} END"
    rv_function = f"CASE WHEN {rv_function_uncertain} THEN '' ELSE {rv_function} END"
    rules["echo_lae"] = echo_rule(
        (
            "la_volume_index > 34 OR "
            f"{la} LIKE '%lae%' OR {la} LIKE '%enlarg%' OR "
            f"{la} LIKE '%mild lavi%' OR {la} LIKE '%mod lavi%' OR "
            f"{la} LIKE '%severe lavi%' OR {la} LIKE '%dilated%' OR "
            f"{la} LIKE '%elongat%'"
        ),
        (
            "la_volume_index <= 34 OR "
            f"{la} LIKE '%normal lavi%' OR {la} LIKE '%nl (%'"
        ),
        indeterminate=f"la_volume_index IS NULL AND {la_uncertain}",
        summary="""
            CONCAT('LAVI=', COALESCE(la_volume_index::VARCHAR, 'missing'),
                   '; method=', COALESCE(la_volume_index_method, 'missing'),
                   '; category=', COALESCE(la_size_category, 'missing'))
        """,
    )
    rules["echo_rae"] = echo_rule(
        f"{ra} LIKE '%rae%' OR {ra} LIKE '%enlarg%'",
        f"{ra} LIKE '%normal%' OR {ra} LIKE '%nl (%' OR {ra} LIKE 'nl %'",
        indeterminate=ra_uncertain,
        summary="COALESCE(ra_size_category, 'missing')",
    )
    rules["echo_rv_dilation"] = echo_rule(
        (
            "rv_basal_diameter_cm >= 4.2 OR "
            f"{rv_size} LIKE '%rve%' OR {rv_size} LIKE '%dilat%'"
        ),
        (
            "rv_basal_diameter_cm < 4.2 OR "
            f"{rv_size} LIKE '%normal%' OR {rv_size} LIKE 'nl %' "
            f"OR {rv_size} LIKE '%cavity small%'"
        ),
        indeterminate=f"rv_basal_diameter_cm IS NULL AND {rv_size_uncertain}",
        summary="""
            CONCAT('RV basal diameter=',
                   COALESCE(rv_basal_diameter_cm::VARCHAR, 'missing'),
                   '; category=', COALESCE(rv_size_category, 'missing'))
        """,
    )
    rules["echo_rv_systolic_dysfunction"] = echo_rule(
        (
            "tapse_cm < 1.7 OR "
            f"{rv_function} LIKE '%hypo%' OR "
            f"{rv_function} LIKE '%depress%' OR "
            f"{rv_function} LIKE '%dysfunction%'"
        ),
        (
            "tapse_cm >= 1.7 OR "
            f"{rv_function} LIKE '%nl rv function%' OR "
            f"{rv_function} LIKE '%normal function%' OR "
            f"{rv_function} LIKE '%hyperdynamic%'"
        ),
        indeterminate=f"tapse_cm IS NULL AND {rv_function_uncertain}",
        summary="""
            CONCAT('TAPSE=', COALESCE(tapse_cm::VARCHAR, 'missing'),
                   '; category=', COALESCE(rv_function_category, 'missing'))
        """,
    )
    rules["elevated_rvsp_or_pasp"] = echo_rule(
        "derived_rvsp_or_pasp_mmhg > 35",
        "derived_rvsp_or_pasp_mmhg <= 35",
        summary="""
            CONCAT('RVSP/PASP=', COALESCE(
                ROUND(derived_rvsp_or_pasp_mmhg, 1)::VARCHAR, 'missing'),
                '; method=', COALESCE(rvsp_or_pasp_method, 'missing'))
        """,
    )
    phtn = "LOWER(COALESCE(pulmonary_hypertension_category, ''))"
    rules["echo_pulmonary_hypertension"] = echo_rule(
        (
            f"{phtn} LIKE '%mild%' OR {phtn} LIKE '%mod%' "
            f"OR {phtn} LIKE '%severe%'"
        ),
        (
            f"{phtn} LIKE '%normal%' OR {phtn} LIKE '%high normal%' "
            f"OR {phtn} LIKE '%unlikely%'"
        ),
        indeterminate=(
            f"{phtn} LIKE '%indeterminate%' "
            f"OR {phtn} LIKE '%borderline%'"
            f" OR regexp_matches({phtn}, "
            "'(^|[^a-z])(not|no|without|possible|possibly|suspected|uncertain|cannot)([^a-z]|$)')"
        ),
        summary="COALESCE(pulmonary_hypertension_category, 'missing')",
    )
    # Blank/unvisualized segments cannot establish an independent negative.
    # The broader documentation endpoint is separately named and is a proxy.
    # Require complete explicitly normal recorded coverage for the clinical
    # arm; a normal isolated segment cannot exclude abnormalities elsewhere.
    rwma_complete_negative = (
        "COALESCE(rwma_complete_normal_coverage, FALSE) "
        "AND NOT COALESCE(rwma_reference_conflict, FALSE)"
    )
    rules["regional_wall_motion_abnormality"] = echo_rule(
        "rwma_abnormal_segments > 0",
        rwma_complete_negative,
        indeterminate=(
            "COALESCE(rwma_reference_conflict, FALSE) OR "
            "(rwma_reported_segments > 0 AND rwma_abnormal_segments = 0 "
            f"AND NOT ({rwma_complete_negative}))"
        ),
        summary="""
            CONCAT('abnormal segments=', COALESCE(rwma_abnormal_segments::VARCHAR, '0'),
                   '; explicitly normal=', COALESCE(rwma_normal_segments::VARCHAR, '0'),
                   '; not visualised=', COALESCE(rwma_not_visualized_segments::VARCHAR, '0'),
                   '; segments in report=', COALESCE(rwma_reported_segments::VARCHAR, '0'),
                   '; negative requires complete explicitly normal coverage')
        """,
    )
    rules["regional_wall_motion_abnormality_explicit"] = echo_rule(
        "rwma_abnormal_segments > 0",
        "rwma_normal_segments > 0 AND rwma_abnormal_segments = 0",
        summary="""
            CONCAT('abnormal segments=', COALESCE(rwma_abnormal_segments::VARCHAR, '0'),
                   '; explicitly normal=', COALESCE(rwma_normal_segments::VARCHAR, '0'),
                   '; strict arm: every negative names a normal segment')
        """,
    )
    rules["regional_wall_motion_documentation_proxy"] = echo_rule(
        "rwma_abnormal_segments > 0",
        "rwma_reported_segments > 0 AND rwma_abnormal_segments = 0",
        summary="'Documentation phenotype only; absence of a recorded abnormality does not exclude disease'",
    )

    valve_ids = (
        "aortic_stenosis",
        "aortic_regurgitation",
        "mitral_stenosis",
        "mitral_regurgitation",
        "tricuspid_regurgitation",
    )
    valve_reference_limits = {
        "aortic_regurgitation": "; onset_and_cause=unverified; reported_severity_not_acute_AR_diagnosis",
        "mitral_regurgitation": (
            "; primary_secondary_atrial_mechanism=unverified"
            "; prolapse_and_annular_disjunction=unverified"
            "; reported_severity_not_arrhythmogenic_MVP_diagnosis"
        ),
        "mitral_stenosis": (
            "; rheumatic_or_calcific_cause=unverified"
            "; reported_severity_not_measured_LA_pressure_or_electromechanical_delay"
        ),
    }
    for valve in valve_ids:
        severity = f"{valve}_severity"
        rules[valve] = echo_rule(
            f"{severity} >= 2",
            f"{severity} < 2",
            summary=f"""
                CONCAT('severity=', COALESCE({severity}::VARCHAR, 'missing'),
                       '; raw=', COALESCE({valve}_raw, 'missing'),
                       '{valve_reference_limits.get(valve, "")}')
            """,
        )
    all_severities = ", ".join(f"{valve}_severity" for valve in valve_ids)
    graded_count = " + ".join(
        f"CASE WHEN {valve}_severity IS NOT NULL THEN 1 ELSE 0 END"
        for valve in valve_ids
    )
    any_graded = " OR ".join(
        f"{valve}_severity IS NOT NULL" for valve in valve_ids
    )

    def composite_valve_rule(threshold: int) -> SQLLabelRule:
        """Positive if any graded valve reaches the threshold, negative if none does.

        The negative arm used to be ``AND``-joined over all five valve
        severities. In SQL ``NULL < 2`` is NULL, so that required **every**
        valve to have been graded, and echocardiograms grade the valves they
        were asked about. Measured on the 206,488 studies in this project:
        819 have all five graded and all below the threshold, against 147,249
        that have at least one graded valve and none reaching it. The endpoint
        therefore ran with 301 usable negatives against 9,400 positives -- a
        96.9 % prevalence that the power gate still passed as eligible.

        The rule now asks what the reference standard can actually answer: was
        any valve assessed, and did any assessed valve reach the threshold. A
        study where the aortic valve was never graded cannot exclude aortic
        disease, so the number of graded valves travels in the summary and can
        be used to restrict a sensitivity analysis to fully assessed studies.
        """

        positive = " OR ".join(
            f"{valve}_severity >= {threshold}" for valve in valve_ids
        )
        negative = (
            f"({any_graded}) AND NOT ("
            + " OR ".join(
                f"COALESCE({valve}_severity, 0) >= {threshold}"
                for valve in valve_ids
            )
            + ")"
        )
        return echo_rule(
            positive,
            negative,
            summary=(
                "CONCAT('valve severities=', CONCAT_WS(',', "
                f"{all_severities}), '; graded valves=', ({graded_count})::VARCHAR"
                f", '/{len(valve_ids)}')"
            ),
        )

    rules["significant_valvular_disease"] = composite_valve_rule(2)
    for valve in valve_ids:
        severity = f"{valve}_severity"
        rules[f"{valve}_severe"] = echo_rule(
            f"{severity} = 3",
            f"{severity} < 3",
            summary=f"""
                CONCAT('severity=', COALESCE({severity}::VARCHAR, 'missing'),
                       '; raw=', COALESCE({valve}_raw, 'missing'),
                       '{valve_reference_limits.get(valve, "")}')
            """,
        )
    rules["significant_valvular_disease_severe"] = composite_valve_rule(3)

    # Quantitative diastolic, aortic-haemodynamic and linear-geometry
    # endpoints. Thresholds follow the usual echocardiographic conventions
    # (ASE/EACVI): E/A <0.8 with a long deceleration time is the impaired-
    # relaxation pattern, E/A >2 with a short one the restrictive pattern,
    # peak aortic velocity 3.0 and 4.0 m/s the moderate and severe stenosis
    # cut-offs, E/e-prime >14 the elevated-filling-pressure surrogate.
    #
    # The negative arm deliberately requires both components to be measured:
    # "not restrictive" is only meaningful when the filling pattern was
    # actually assessable, and treating an unmeasured study as negative would
    # inflate specificity exactly where the reference standard is weakest.
    rules["diastolic_relaxation_abnormality"] = echo_rule(
        "mitral_e_a_ratio < 0.8 AND mitral_e_deceleration_time_ms > 200",
        "mitral_e_a_ratio BETWEEN 0.8 AND 2.0 "
        "AND mitral_e_deceleration_time_ms BETWEEN 160 AND 200",
        summary=(
            "CONCAT('E/A=', COALESCE(ROUND(mitral_e_a_ratio, 2)::VARCHAR, "
            "'missing'), '; DT=', COALESCE("
            "ROUND(mitral_e_deceleration_time_ms, 0)::VARCHAR, 'missing'), "
            "'; measurement_pattern_only; echo_rhythm_applicability=unverified; not_HFpEF_diagnosis')"
        ),
    )
    rules["restrictive_filling_pattern"] = echo_rule(
        "mitral_e_a_ratio > 2.0 AND mitral_e_deceleration_time_ms < 160",
        "mitral_e_a_ratio IS NOT NULL "
        "AND mitral_e_deceleration_time_ms IS NOT NULL "
        "AND NOT (mitral_e_a_ratio > 2.0 "
        "AND mitral_e_deceleration_time_ms < 160)",
        summary=(
            "CONCAT('E/A=', COALESCE(ROUND(mitral_e_a_ratio, 2)::VARCHAR, "
            "'missing'), '; DT=', COALESCE("
            "ROUND(mitral_e_deceleration_time_ms, 0)::VARCHAR, 'missing'), "
            "'; measurement_pattern_only; echo_rhythm_applicability=unverified; not_HFpEF_diagnosis')"
        ),
    )
    rules["elevated_filling_pressure_proxy"] = echo_rule(
        "mitral_e_e_prime_ratio > 14",
        "mitral_e_e_prime_ratio <= 14",
        summary=(
            "CONCAT('E/e-prime=', COALESCE("
            "ROUND(mitral_e_e_prime_ratio, 1)::VARCHAR, 'missing'), "
            "'; measurement_proxy_only; echo_rhythm_applicability=unverified; not_HFpEF_diagnosis')"
        ),
    )
    for label_id, threshold in (
        ("severe_aortic_stenosis_velocity", 4.0),
        ("moderate_aortic_stenosis_velocity", 3.0),
    ):
        rules[label_id] = echo_rule(
            f"aortic_peak_velocity_m_per_s >= {threshold}",
            f"aortic_peak_velocity_m_per_s < {threshold}",
            summary=(
                "CONCAT('AV Vmax=', COALESCE("
                "ROUND(aortic_peak_velocity_m_per_s, 2)::VARCHAR, 'missing'), "
                f"' m/s; threshold={threshold}; velocity_threshold_only; "
                "below_threshold_does_not_exclude_anatomic_severe_AS; "
                "flow_and_rhythm_applicability=unverified')"
            ),
        )
    for label_id, column, threshold, unit in (
        ("la_dilatation_linear", "la_anteroposterior_dimension_cm", 4.0, "cm"),
        ("ra_dilatation_linear", "ra_length_cm", 5.3, "cm"),
        ("concentric_remodeling_rwt", "relative_wall_thickness_echo", 0.42, ""),
        ("aortic_root_dilatation", "aortic_sinus_diameter_cm", 4.0, "cm"),
    ):
        rules[label_id] = echo_rule(
            f"{column} > {threshold}",
            f"{column} <= {threshold}",
            summary=(
                f"CONCAT('{column}=', COALESCE("
                f"ROUND({column}, 2)::VARCHAR, 'missing'), "
                f"' {unit}; threshold={threshold}')"
            ),
        )
    return rules


def _electrolyte_and_qt_rules() -> dict[str, SQLLabelRule]:
    from .lab_intervals import interval_comparison_sql as compare

    potassium_normal = (
        compare("potassium", ">=", "3.5") + " AND " + compare("potassium", "<=", "5.5")
    )
    def measured_state(
        value: str,
        measured: str,
        positive: str,
        negative: str,
        summary: str,
        *,
        time_delta_hours: str = "NULL::DOUBLE",
        time_precision: str = "'timestamp'",
        evidence_ids: str = "'[]'",
    ) -> SQLLabelRule:
        if value in ("potassium_nearest", "magnesium_nearest", "total_calcium_nearest"):
            group = value.removesuffix("_nearest")
            summary = (
                f"CONCAT(({summary}), '; qualifier=', COALESCE({group}_nearest_censoring, 'exact_or_missing'), "
                f"'; bound=', COALESCE({group}_nearest_bound::VARCHAR, 'missing'))"
            )
        measured_bool = f"COALESCE(({measured}), FALSE)"
        positive_bool = f"COALESCE(({positive}), FALSE)"
        negative_bool = f"COALESCE(({negative}), FALSE)"
        return SQLLabelRule(
            state=f"""
                CASE
                    WHEN NOT {measured_bool} THEN 'unavailable'
                    WHEN {positive_bool} THEN 'positive'
                    WHEN {negative_bool} THEN 'negative'
                    ELSE 'indeterminate'
                END
            """,
            eligibility=_availability_eligibility(
                f"{measured_bool} AND ({positive_bool} OR {negative_bool})"
            ),
            confidence=(
                f"CASE WHEN {measured_bool} "
                f"AND ({positive_bool} OR {negative_bool}) "
                "THEN 'high' ELSE 'none' END"
            ),
            time_delta_hours=time_delta_hours,
            time_precision=time_precision,
            evidence_ids=evidence_ids,
            evidence_summary=summary,
        )

    rules = {
        "qtc_gt_480": measured_state(
            "machine_qtc_fridericia_ms",
            "machine_qtc_fridericia_ms IS NOT NULL",
            "machine_qtc_fridericia_ms > 480",
            "machine_qtc_fridericia_ms <= 480",
            "CONCAT('machine QTc Fridericia=', ROUND(machine_qtc_fridericia_ms, 1))",
            time_delta_hours="0.0",
            time_precision="'ecg_derived'",
        ),
        "qtc_gt_500": measured_state(
            "machine_qtc_fridericia_ms",
            "machine_qtc_fridericia_ms IS NOT NULL",
            "machine_qtc_fridericia_ms > 500",
            "machine_qtc_fridericia_ms <= 500",
            "CONCAT('machine QTc Fridericia=', ROUND(machine_qtc_fridericia_ms, 1))",
            time_delta_hours="0.0",
            time_precision="'ecg_derived'",
        ),
        "hypokalemia_near_ecg": measured_state(
            "potassium_nearest",
            "potassium_count_6h > 0",
            compare("potassium", "<", "3.5"),
            potassium_normal,
            "CONCAT('K=', potassium_nearest, '; delta_h=', potassium_delta_hours)",
            time_delta_hours="potassium_delta_hours",
            evidence_ids=_single_evidence_id_json(
                "potassium_nearest_labevent_id",
                "labevent",
            ),
        ),
        "severe_hypokalemia_near_ecg": measured_state(
            "potassium_nearest",
            "potassium_count_6h > 0",
            compare("potassium", "<", "3.0"),
            compare("potassium", ">=", "3.0"),
            "CONCAT('K=', potassium_nearest, '; delta_h=', potassium_delta_hours)",
            time_delta_hours="potassium_delta_hours",
            evidence_ids=_single_evidence_id_json(
                "potassium_nearest_labevent_id",
                "labevent",
            ),
        ),
        "hyperkalemia_near_ecg": measured_state(
            "potassium_nearest",
            "potassium_count_6h > 0",
            compare("potassium", ">", "5.5"),
            potassium_normal,
            "CONCAT('K=', potassium_nearest, '; delta_h=', potassium_delta_hours)",
            time_delta_hours="potassium_delta_hours",
            evidence_ids=_single_evidence_id_json(
                "potassium_nearest_labevent_id",
                "labevent",
            ),
        ),
        "severe_hyperkalemia_near_ecg": measured_state(
            "potassium_nearest",
            "potassium_count_6h > 0",
            compare("potassium", ">=", "6.5"),
            compare("potassium", "<", "6.5"),
            "CONCAT('K=', potassium_nearest, '; delta_h=', potassium_delta_hours)",
            time_delta_hours="potassium_delta_hours",
            evidence_ids=_single_evidence_id_json(
                "potassium_nearest_labevent_id",
                "labevent",
            ),
        ),
        "hypomagnesemia_near_ecg": measured_state(
            "magnesium_nearest",
            "magnesium_count_6h > 0 AND magnesium_reference_lower IS NOT NULL",
            compare("magnesium", "<", "magnesium_reference_lower"),
            compare("magnesium", ">=", "magnesium_reference_lower"),
            """
                CONCAT('Mg=', magnesium_nearest,
                       '; lower reference=', magnesium_reference_lower,
                       '; unit=', magnesium_unit)
            """,
            evidence_ids=_single_evidence_id_json(
                "magnesium_nearest_labevent_id",
                "labevent",
            ),
        ),
        "hypocalcemia_near_ecg": measured_state(
            "total_calcium_nearest",
            (
                "(total_calcium_count_6h > 0 "
                "AND total_calcium_reference_lower IS NOT NULL) OR "
                "(ionized_calcium_count_6h > 0 "
                "AND ionized_calcium_reference_lower IS NOT NULL)"
            ),
            (
                "(total_calcium_count_6h > 0 "
                "AND total_calcium_reference_lower IS NOT NULL "
                f"AND {compare('total_calcium', '<', 'total_calcium_reference_lower')}) OR "
                "(ionized_calcium_count_6h > 0 "
                "AND ionized_calcium_reference_lower IS NOT NULL "
                f"AND {compare('ionized_calcium', '<', 'ionized_calcium_reference_lower')})"
            ),
            (
                "(total_calcium_count_6h IS NULL "
                "OR total_calcium_count_6h = 0 "
                "OR total_calcium_reference_lower IS NULL "
                f"OR {compare('total_calcium', '>=', 'total_calcium_reference_lower')}) "
                "AND (ionized_calcium_count_6h IS NULL "
                "OR ionized_calcium_count_6h = 0 "
                "OR ionized_calcium_reference_lower IS NULL "
                f"OR {compare('ionized_calcium', '>=', 'ionized_calcium_reference_lower')})"
            ),
            """
                CONCAT(
                    'total Ca=', COALESCE(total_calcium_nearest::VARCHAR, 'missing'),
                    '; total lower=',
                    COALESCE(total_calcium_reference_lower::VARCHAR, 'missing'),
                    '; ionized Ca=',
                    COALESCE(ionized_calcium_nearest::VARCHAR, 'missing'),
                    '; ionized lower=',
                    COALESCE(ionized_calcium_reference_lower::VARCHAR, 'missing'),
                    '; total unit=', COALESCE(total_calcium_unit, 'missing'),
                    '; ionized unit=', COALESCE(ionized_calcium_unit, 'missing')
                )
            """,
            evidence_ids=(
                "CASE "
                "WHEN total_calcium_nearest_labevent_id IS NOT NULL THEN "
                "'[\"labevent:' || "
                "total_calcium_nearest_labevent_id::VARCHAR || '\"]' "
                "WHEN ionized_calcium_nearest_labevent_id IS NOT NULL THEN "
                "'[\"labevent:' || "
                "ionized_calcium_nearest_labevent_id::VARCHAR || '\"]' "
                "ELSE '[]' END"
            ),
        ),
    }
    return rules


def _af_rules() -> dict[str, SQLLabelRule]:
    # Audit D9: prior AF is AF evidenced before the ECG -- this ECG's report,
    # an earlier AF report, or an AF-coded admission before the index one.
    # The index admission's own code has no time; it neither proves prior AF
    # nor rules out new onset.
    current_or_prior = (
        "COALESCE(current_report_af_state = 'positive', FALSE) "
        "OR (first_af_report_time IS NOT NULL AND first_af_report_time <= ecg_time) "
        "OR COALESCE(af_dx_prior_admission, FALSE)"
    )
    index_code_only = "COALESCE(af_dx_index_admission, FALSE)"
    rules = {}
    for label_id, days in (
        ("new_onset_af_30d", 30),
        ("new_onset_af_90d", 90),
        ("new_onset_af_1y", 365),
    ):
        horizon_hours = days * 24
        within = (
            f"next_af_event_delta_hours > 0 "
            f"AND next_af_event_delta_hours <= {horizon_hours}"
        )
        state = f"""
            CASE
                WHEN ({current_or_prior}) THEN 'indeterminate'
                WHEN {within} THEN 'positive'
                -- AF coded in the index admission without a timed event:
                -- onset before or after the ECG cannot be told apart.
                WHEN {index_code_only} THEN 'indeterminate'
                WHEN death_delta_days BETWEEN 0 AND {days - 1}
                    THEN 'indeterminate'
                WHEN followup_hours >= {horizon_hours}
                    THEN 'negative'
                ELSE 'indeterminate'
            END
        """
        eligibility = f"""
            CASE
                WHEN ({current_or_prior}) THEN 'excluded'
                WHEN {within} THEN 'eligible'
                WHEN {index_code_only} THEN 'insufficient_evidence'
                WHEN death_delta_days BETWEEN 0 AND {days - 1}
                    THEN 'insufficient_evidence'
                WHEN followup_hours >= {horizon_hours} THEN 'eligible'
                ELSE 'insufficient_evidence'
            END
        """
        rules[label_id] = SQLLabelRule(
            state=state,
            eligibility=eligibility,
            confidence=(
                f"CASE WHEN ({current_or_prior}) THEN 'none' "
                f"WHEN {within} "
                " AND next_af_event_source = 'machine_report' THEN 'high' "
                f"WHEN {within} THEN 'medium' "
                f"WHEN {index_code_only} THEN 'none' "
                f"WHEN death_delta_days BETWEEN 0 AND {days - 1} THEN 'none' "
                f"WHEN followup_hours >= {horizon_hours} THEN 'medium' "
                "ELSE 'low' END"
            ),
            time_delta_hours="next_af_event_delta_hours",
            time_precision=(
                "CASE WHEN next_af_event_source = 'machine_report' "
                "THEN 'timestamp' WHEN next_af_event_source = 'icd_admission' "
                "THEN 'encounter_start_proxy' ELSE 'followup_interval' END"
            ),
            evidence_ids=(
                "CASE "
                "WHEN next_af_event_source = 'machine_report' "
                "AND first_af_report_study_id IS NOT NULL "
                "THEN '[\"study:' || first_af_report_study_id::VARCHAR || '\"]' "
                "WHEN next_af_event_source = 'icd_admission' "
                "AND next_af_dx_hadm_id IS NOT NULL "
                "THEN '[\"hadm:' || next_af_dx_hadm_id::VARCHAR || '\"]' "
                "ELSE '[]' END"
            ),
            evidence_summary=f"""
                CONCAT('next AF event delta_h=',
                       COALESCE(next_af_event_delta_hours::VARCHAR, 'none'),
                       '; source=',
                       COALESCE(next_af_event_source, 'none'),
                       '; index admission AF code=',
                       COALESCE(af_dx_index_admission::VARCHAR, 'false'),
                       '; follow-up_h=', COALESCE(followup_hours::VARCHAR, 'missing'),
                       '; horizon_days={days}')
            """,
        )
    return rules


def measurement_sensitivity_summary(config) -> dict:
    """Hemolysis, censoring, duplicate and follow-up counts (audit 5.3.9).

    Development splits (train, validation) only; the locked test is not
    read. Each block reports its own source problem instead of failing the
    stage, so a missing optional table is visible rather than fatal.
    """

    import duckdb

    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    development = (
        f"SELECT subject_id, study_id FROM read_parquet('{master}') "
        "WHERE split IN ('train', 'validation')"
    )
    queries = {
        "hemolysis_excluded_results_by_lab_group": f"""
            SELECT l.lab_group, COUNT(*) AS results
            FROM read_parquet('{_sql_path(config.evidence_root / "labs.parquet")}') l
            WHERE l.value_status = 'hemolysis_reported'
              AND l.subject_id IN (SELECT subject_id FROM ({development}))
            GROUP BY l.lab_group ORDER BY l.lab_group
        """,
        "troponin_interval_states": f"""
            SELECT t.serial_interval_state, t.undetectable_rise_state,
                   COUNT(*) AS ecgs
            FROM read_parquet('{_sql_path(config.evidence_root / "ecg_troponin_serial.parquet")}') t
            JOIN ({development}) d USING (subject_id, study_id)
            GROUP BY ALL ORDER BY ALL
        """,
        "same_subject_same_time_ecg_groups": f"""
            SELECT COUNT(*) AS groups, COALESCE(SUM(n), 0) AS ecgs
            FROM (
                SELECT subject_id, ecg_time, COUNT(*) AS n
                FROM read_parquet('{master}')
                WHERE split IN ('train', 'validation') AND ecg_time IS NOT NULL
                GROUP BY subject_id, ecg_time HAVING COUNT(*) > 1
            )
        """,
        "vital_status_followup_below_30_days_without_death": f"""
            SELECT COUNT(*) AS ecgs
            FROM read_parquet('{_sql_path(config.evidence_root / "clinical_context_features.parquet")}') c
            WHERE c.split IN ('train', 'validation')
              AND c.death_delta_days IS NULL
              AND c.vital_status_followup_hours < 24 * 30
        """,
    }
    summary: dict = {
        "scope": "development_splits_only",
        "notes": {
            "hemolysis": "potassium, magnesium and phosphate results with a reported hemolysis are not measurements (audit D2)",
            "censoring": "troponin series decided over the whole interval of censored results (audit D3, 5.2.3)",
            "duplicates": "same subject and timestamp; identical signals are removed later by signal hash (audit X8)",
            "follow_up": "no death and less than 30 days of registry-covered follow-up: mortality endpoints indeterminate (audit D4)",
        },
    }
    with duckdb.connect() as connection:
        for name, query in queries.items():
            try:
                summary[name] = connection.execute(query).df().to_dict(orient="records")
            except (duckdb.Error, OSError) as exc:
                summary[name] = {"status": "source_unavailable", "error": str(exc)[:300]}
    return summary


def _conduction_progression_rules() -> dict[str, SQLLabelRule]:
    """Audit 5.2.5: new high-grade AV block, new LBBB or a first permanent
    pacemaker within one year of the ECG. Same construction as new-onset AF
    (D9): prior evidence excludes, untimed index-admission evidence and
    deaths inside the horizon are indeterminate, negatives need documented
    observation through the horizon."""

    current_or_prior = (
        "COALESCE(current_report_lbbb_state = 'positive', FALSE) "
        "OR COALESCE(current_report_pacemaker_state = 'positive', FALSE) "
        "OR COALESCE(conduction_prior_evidence, FALSE)"
    )
    index_untimed = "COALESCE(conduction_index_untimed, FALSE)"
    days = 365
    horizon_hours = days * 24
    within = (
        f"next_conduction_event_delta_hours > 0 "
        f"AND next_conduction_event_delta_hours <= {horizon_hours}"
    )
    # A device-in-situ code in a later admission within the horizon without a
    # timed implantation: the device arrived at an unknown time.
    device_untimed = (
        f"COALESCE(next_device_status_delta_hours BETWEEN 0 AND {horizon_hours}, FALSE)"
    )
    state = f"""
        CASE
            WHEN ({current_or_prior}) THEN 'indeterminate'
            WHEN {within} THEN 'positive'
            WHEN {index_untimed} THEN 'indeterminate'
            WHEN {device_untimed} THEN 'indeterminate'
            WHEN death_delta_days BETWEEN 0 AND {days - 1} THEN 'indeterminate'
            WHEN followup_hours >= {horizon_hours} THEN 'negative'
            ELSE 'indeterminate'
        END
    """
    eligibility = f"""
        CASE
            WHEN ({current_or_prior}) THEN 'excluded'
            WHEN {within} THEN 'eligible'
            WHEN {index_untimed} THEN 'insufficient_evidence'
            WHEN {device_untimed} THEN 'insufficient_evidence'
            WHEN death_delta_days BETWEEN 0 AND {days - 1}
                THEN 'insufficient_evidence'
            WHEN followup_hours >= {horizon_hours} THEN 'eligible'
            ELSE 'insufficient_evidence'
        END
    """
    return {
        "conduction_progression_1y": SQLLabelRule(
            state=state,
            eligibility=eligibility,
            confidence=(
                f"CASE WHEN ({current_or_prior}) THEN 'none' "
                f"WHEN {within} AND next_conduction_event_source = "
                "'pacemaker_procedure' THEN 'high' "
                f"WHEN {within} THEN 'medium' "
                f"WHEN {index_untimed} THEN 'none' "
                f"WHEN death_delta_days BETWEEN 0 AND {days - 1} THEN 'none' "
                f"WHEN followup_hours >= {horizon_hours} THEN 'medium' "
                "ELSE 'low' END"
            ),
            time_delta_hours="next_conduction_event_delta_hours",
            time_precision=(
                "CASE WHEN next_conduction_event_source = 'pacemaker_procedure' "
                "THEN 'date' WHEN next_conduction_event_source = 'icd_admission' "
                "THEN 'encounter_start_proxy' ELSE 'followup_interval' END"
            ),
            evidence_ids="'[]'",
            evidence_summary=f"""
                CONCAT('next conduction event delta_h=',
                       COALESCE(next_conduction_event_delta_hours::VARCHAR, 'none'),
                       '; source=', COALESCE(next_conduction_event_source, 'none'),
                       '; untimed index evidence=',
                       COALESCE(conduction_index_untimed::VARCHAR, 'false'),
                       '; follow-up_h=', COALESCE(followup_hours::VARCHAR, 'missing'),
                       '; horizon_days={days}')
            """,
        )
    }


def _secondary_rules() -> dict[str, SQLLabelRule]:
    right_heart_context = (
        "(COALESCE(dx_pulmonary_embolism, FALSE) "
        "OR COALESCE(dx_pulmonary_hypertension, FALSE)) "
        "AND COALESCE(has_tte_7d, FALSE)"
    )
    rv_size_positive = (
        "(rv_basal_diameter_cm >= 4.2 "
        "OR LOWER(COALESCE(rv_size_category, '')) LIKE '%rve%' "
        "OR LOWER(COALESCE(rv_size_category, '')) LIKE '%dilat%')"
    )
    rv_size_negative = (
        "(rv_basal_diameter_cm < 4.2 "
        "OR LOWER(COALESCE(rv_size_category, '')) LIKE '%normal%' "
        "OR LOWER(COALESCE(rv_size_category, '')) LIKE 'nl %' "
        "OR LOWER(COALESCE(rv_size_category, '')) LIKE '%cavity small%')"
    )
    rv_function_positive = (
        "(tapse_cm < 1.7 "
        "OR LOWER(COALESCE(rv_function_category, '')) LIKE '%hypo%' "
        "OR LOWER(COALESCE(rv_function_category, '')) LIKE '%depress%' "
        "OR LOWER(COALESCE(rv_function_category, '')) LIKE '%dysfunction%')"
    )
    rv_function_negative = (
        "(tapse_cm >= 1.7 "
        "OR LOWER(COALESCE(rv_function_category, '')) "
        "LIKE '%nl rv function%' "
        "OR LOWER(COALESCE(rv_function_category, '')) "
        "LIKE '%normal function%' "
        "OR LOWER(COALESCE(rv_function_category, '')) LIKE '%hyperdynamic%')"
    )
    rules = {
        "old_mi_icd": _history_rule(
            "history_old_mi_before_ecg",
            "old-MI ICD at or before index ECG",
        ),
        "mi_or_revascularization_history": _history_rule(
            "history_mi_or_revascularization_before_ecg",
            "MI/PCI/CABG evidence at or before index ECG",
        ),
        "heart_failure_icd": _binary_context_rule(
            "dx_heart_failure", "heart-failure ICD in index encounter"
        ),
        "elevated_ntprobnp": SQLLabelRule(
            state="""
                CASE
                    WHEN ntprobnp_max_url_ratio > 1 THEN 'positive'
                    WHEN ntprobnp_max_url_ratio <= 1 THEN 'negative'
                    ELSE 'unavailable'
                END
            """,
            eligibility=_availability_eligibility(
                "ntprobnp_max_url_ratio IS NOT NULL"
            ),
            confidence=_confidence(
                "ntprobnp_max_url_ratio IS NOT NULL",
                high="'high'",
                fallback="'none'",
            ),
            evidence_summary=(
                "CONCAT('NT-proBNP URL ratio=', "
                "COALESCE(ntprobnp_max_url_ratio::VARCHAR, 'missing'))"
            ),
            time_delta_hours="ntprobnp_max_url_delta_hours",
            time_precision="'timestamp'",
            evidence_ids=_single_evidence_id_json(
                "ntprobnp_max_url_labevent_id",
                "labevent",
            ),
        ),
        "acute_decompensated_heart_failure_proxy": _composite_rule(
            positive=(
                "COALESCE(dx_heart_failure, FALSE) "
                "AND ntprobnp_max_url_ratio > 1 "
                "AND (COALESCE(med_diuretic, FALSE) OR "
                "REGEXP_MATCHES(LOWER(COALESCE(chief_complaint, '')), "
                "'dyspnea|shortness of breath'))"
            ),
            assessed=(
                "hadm_id IS NOT NULL "
                "AND ntprobnp_max_url_ratio IS NOT NULL"
            ),
            negative=(
                "hadm_id IS NOT NULL "
                "AND ntprobnp_max_url_ratio <= 1 "
                "AND NOT COALESCE(dx_heart_failure, FALSE)"
            ),
            indeterminate=(
                "hadm_id IS NOT NULL AND ("
                "COALESCE(dx_heart_failure, FALSE) "
                "OR ntprobnp_max_url_ratio > 1)"
            ),
            summary=(
                "'HF ICD + elevated NT-proBNP + dyspnea/diuretic context; "
                "controls require measured non-elevated NT-proBNP and no HF ICD'"
            ),
            positive_confidence="'medium'",
            time_delta_hours="ntprobnp_max_url_delta_hours",
            time_precision="'timestamp_and_encounter'",
            evidence_ids=_single_evidence_id_json(
                "ntprobnp_max_url_labevent_id",
                "labevent",
            ),
        ),
        "pulmonary_embolism_icd": _binary_context_rule(
            "dx_pulmonary_embolism", "PE ICD in index encounter"
        ),
        "pulmonary_hypertension_icd": _binary_context_rule(
            "dx_pulmonary_hypertension", "PH ICD in index encounter"
        ),
        "right_heart_strain_proxy": _composite_rule(
            positive=(
                right_heart_context
                + " AND ("
                + rv_size_positive
                + " OR "
                + rv_function_positive
                + ")"
            ),
            assessed=(
                right_heart_context
                + " AND ("
                + rv_size_positive
                + " OR "
                + rv_size_negative
                + " OR "
                + rv_function_positive
                + " OR "
                + rv_function_negative
                + ")"
            ),
            negative=(
                right_heart_context
                + " AND "
                + rv_size_negative
                + " AND "
                + rv_function_negative
            ),
            summary=(
                "'PE/PH context plus near-event RV dilation/dysfunction; "
                "controls require assessed normal RV size and function'"
            ),
            positive_confidence="'medium'",
            time_delta_hours="nearest_tte_delta_hours",
            time_precision="'timestamp'",
            evidence_ids=(
                "CASE WHEN nearest_tte_measurement_id IS NULL THEN '[]' "
                "ELSE '[\"' || nearest_tte_measurement_id::VARCHAR || '\"]' END"
            ),
        ),
        "cardiomyopathy_icd": _binary_context_rule(
            "dx_cardiomyopathy", "cardiomyopathy ICD in index encounter"
        ),
        "pericarditis_icd": _binary_context_rule(
            "dx_pericarditis", "pericarditis ICD in index encounter"
        ),
        "pericardial_effusion": _binary_context_rule(
            "dx_pericardial_effusion",
            "pericardial-effusion ICD in index encounter",
        ),
        "cardiac_amyloidosis": _binary_context_rule(
            "dx_systemic_amyloidosis",
            "systemic-amyloidosis ICD in index encounter",
        ),
        "hypertrophic_cardiomyopathy_icd": _binary_context_rule(
            "dx_hypertrophic_cardiomyopathy",
            "hypertrophic-cardiomyopathy ICD in index encounter",
        ),
        "copd_icd": _binary_context_rule(
            "dx_copd", "COPD ICD in index encounter"
        ),
        # Positives come from the structured record alone -- a device status
        # code or an implantation procedure -- so the endpoint is not read off
        # the tracing the markers are computed from.
        #
        # The controls are the hard part. Measured over the cohort, 19.372
        # recordings whose machine report says paced carry no device code in
        # the index encounter, because the device was implanted years earlier.
        # Calling those negative would put 19.372 paced tracings into the
        # control group of an endpoint about pacing. They are withheld as
        # indeterminate instead: the report is allowed to remove a control,
        # never to create a case.
        "pacemaker_carrier": SQLLabelRule(
            state="""
                CASE
                    WHEN hadm_id IS NULL THEN 'unavailable'
                    WHEN COALESCE(dx_cardiac_device_status, FALSE)
                      OR COALESCE(proc_pacemaker_or_icd, FALSE) THEN 'positive'
                    WHEN current_report_pacemaker_state = 'positive'
                        THEN 'indeterminate'
                    ELSE 'negative'
                END
            """,
            eligibility="""
                CASE
                    WHEN hadm_id IS NULL THEN 'unavailable'
                    WHEN current_report_pacemaker_state = 'positive'
                     AND NOT (
                        COALESCE(dx_cardiac_device_status, FALSE)
                        OR COALESCE(proc_pacemaker_or_icd, FALSE)
                     ) THEN 'insufficient_evidence'
                    ELSE 'eligible'
                END
            """,
            confidence="""
                CASE
                    WHEN COALESCE(dx_cardiac_device_status, FALSE)
                      OR COALESCE(proc_pacemaker_or_icd, FALSE) THEN 'medium'
                    WHEN hadm_id IS NOT NULL THEN 'low'
                    ELSE 'none'
                END
            """,
            time_precision="'encounter'",
            evidence_summary=(
                "'device status code or pacemaker/ICD procedure in the index "
                "encounter; paced tracings without either are withheld'"
            ),
        ),
        # Read from ``first_af_dx_time`` and not from ``first_af_event_time``.
        # The latter is the earlier of the coded diagnosis and the *machine
        # report*, and a machine report is a reading of the same tracing the
        # markers are computed from -- an endpoint built on it would report
        # agreement between two readings rather than a finding. The
        # difference is not hypothetical: ``first_af_event_source`` records
        # which of the two won, and the report wins often.
        "prevalent_atrial_fibrillation": _history_rule(
            "first_af_dx_time IS NOT NULL AND first_af_dx_time <= ecg_time",
            "AF/AFlutter ICD dated at or before the index ECG",
        ),
        "myocarditis_icd": _composite_rule(
            positive=(
                "COALESCE(dx_myocarditis, FALSE) "
                "AND troponin_max_url_ratio > 1"
            ),
            assessed="hadm_id IS NOT NULL",
            negative=(
                "hadm_id IS NOT NULL "
                "AND NOT COALESCE(dx_myocarditis, FALSE)"
            ),
            summary=(
                "'myocarditis ICD plus myocardial injury context; "
                "ICD-only cases are indeterminate'"
            ),
            positive_confidence="'medium'",
        ),
        "endocarditis_high_confidence_proxy": _composite_rule(
            positive=(
                "COALESCE(dx_endocarditis, FALSE) "
                "AND COALESCE(typical_endocarditis_organism, FALSE) "
                "AND (COALESCE(proc_valve_procedure, FALSE) "
                "OR COALESCE(med_endocarditis_antibiotic_context, FALSE))"
            ),
            assessed="hadm_id IS NOT NULL",
            negative=(
                "hadm_id IS NOT NULL "
                "AND NOT COALESCE(dx_endocarditis, FALSE)"
            ),
            summary=(
                "'same-encounter endocarditis ICD + typical organism "
                "+ antibiotic/valve-procedure context; ICD-only or partial "
                "evidence is indeterminate'"
            ),
        ),
        "vt_vf_icd_or_event": _binary_context_rule(
            (
                "COALESCE(dx_ventricular_tachycardia, FALSE) "
                "OR COALESCE(dx_ventricular_fibrillation, FALSE) "
                "OR COALESCE(proc_cardioversion_or_defibrillation, FALSE)"
            ),
            "VT/VF ICD or cardioversion/defibrillation",
        ),
        "cardiac_arrest": _binary_context_rule(
            "dx_cardiac_arrest", "cardiac-arrest ICD in index encounter"
        ),
        "in_hospital_mortality": SQLLabelRule(
            state="""
                CASE
                    WHEN hospital_expire_flag IS NULL THEN 'unavailable'
                    WHEN hospital_expire_flag = 1 THEN 'positive'
                    WHEN hospital_expire_flag = 0 THEN 'negative'
                    ELSE 'indeterminate'
                END
            """,
            eligibility=_availability_eligibility(
                "hospital_expire_flag IS NOT NULL"
            ),
            confidence=(
                "CASE WHEN hospital_expire_flag IS NOT NULL THEN 'high' "
                "ELSE 'none' END"
            ),
            evidence_summary=(
                "CONCAT('hospital_expire_flag=', "
                "COALESCE(hospital_expire_flag::VARCHAR, 'missing'))"
            ),
        ),
        # Audit D4: survival is ascertained like death -- through the
        # registry window (last discharge + 365 d), not through the next
        # contact. A death after day 30 proves survival to day 30.
        "mortality_30d": SQLLabelRule(
            state="""
                CASE
                    WHEN death_delta_days BETWEEN 0 AND 30 THEN 'positive'
                    WHEN death_delta_days > 30 THEN 'negative'
                    WHEN death_delta_days IS NULL
                     AND vital_status_followup_hours >= 30 * 24
                        THEN 'negative'
                    ELSE 'indeterminate'
                END
            """,
            eligibility=(
                "CASE WHEN death_delta_days >= 0 "
                "OR (death_delta_days IS NULL "
                "AND vital_status_followup_hours >= 30 * 24) THEN 'eligible' "
                "ELSE 'insufficient_evidence' END"
            ),
            confidence=(
                "CASE WHEN death_delta_days >= 0 THEN 'high' "
                "WHEN death_delta_days IS NULL "
                "AND vital_status_followup_hours >= 30 * 24 THEN 'medium' "
                "ELSE 'low' END"
            ),
            time_delta_hours="death_delta_days * 24.0",
            evidence_summary=(
                "CONCAT('death delta days=', "
                "COALESCE(death_delta_days::VARCHAR, 'none'))"
            ),
        ),
        "qt_prolonging_drug_exposure": _binary_exposure_rule(
            "med_qt_prolonging_drug", "QT-prolonging drug administration/order"
        ),
        "antiarrhythmic_exposure_near_ecg": _binary_exposure_rule(
            "med_antiarrhythmic_class_i_iii",
            "Class I/III antiarrhythmic administration/order",
        ),
        "digoxin_exposure_near_ecg": _binary_exposure_rule(
            "med_digoxin", "digoxin administration/order"
        ),
    }
    return rules


def _long_label_query(
    context_path: Path,
    definitions: tuple[LabelDefinition, ...],
    rules: dict[str, SQLLabelRule],
    *,
    split_values: Optional[tuple[str, ...]] = None,
) -> str:
    definition_ids = {definition.label_id for definition in definitions}
    missing_rules = sorted(definition_ids - set(rules))
    extra_rules = sorted(set(rules) - definition_ids)
    if missing_rules or extra_rules:
        raise RuntimeError(
            "Label registry and SQL implementation differ; "
            f"missing rules={missing_rules}, extra rules={extra_rules}"
        )
    values = []
    for definition in definitions:
        sources = ",".join(definition.source_tables).replace("'", "''")
        index_policy = definition.index_policy.replace("'", "''")
        values.append(
            "("
            f"'{definition.label_id}', "
            f"'{definition.target_kind.value}', "
            f"'{definition.panel}', "
            f"'{sources}', "
            f"'{index_policy}', "
            f"{_optional_float_sql(definition.window_before_hours)}, "
            f"{_optional_float_sql(definition.window_after_hours)}"
            ")"
        )
    label_values = ", ".join(values)
    state_cases = []
    eligibility_cases = []
    confidence_cases = []
    delta_cases = []
    precision_cases = []
    evidence_id_cases = []
    summary_cases = []
    for definition in definitions:
        rule = rules[definition.label_id]
        label = definition.label_id
        state_cases.append(f"WHEN '{label}' THEN ({rule.state})")
        eligibility_cases.append(
            f"WHEN '{label}' THEN ({rule.eligibility})"
        )
        confidence_cases.append(f"WHEN '{label}' THEN ({rule.confidence})")
        delta_cases.append(
            f"WHEN '{label}' THEN ({rule.time_delta_hours})"
        )
        precision_cases.append(
            f"WHEN '{label}' THEN ({rule.time_precision})"
        )
        evidence_id_cases.append(
            f"WHEN '{label}' THEN ({rule.evidence_ids})"
        )
        summary_cases.append(
            f"WHEN '{label}' THEN ({rule.evidence_summary})"
        )

    context_relation = _parquet_relation(
        context_path,
        split_values=split_values,
    )
    return f"""
        WITH labels(
            label_id,
            target_kind,
            panel,
            source_tables,
            index_policy,
            window_before_hours,
            window_after_hours
        ) AS (
            VALUES {label_values}
        )
        SELECT
            c.subject_id,
            c.study_id,
            c.ecg_time,
            c.split,
            l.label_id,
            l.target_kind,
            l.panel,
            l.index_policy,
            l.window_before_hours,
            l.window_after_hours,
            CASE l.label_id {' '.join(state_cases)} ELSE 'unavailable' END
                AS label_state,
            CASE l.label_id {' '.join(eligibility_cases)}
                ELSE 'insufficient_evidence' END AS eligibility_state,
            CASE l.label_id {' '.join(confidence_cases)} ELSE 'none' END
                AS confidence,
            CAST(
                CASE l.label_id {' '.join(delta_cases)} ELSE NULL END
                AS DOUBLE
            ) AS time_delta_hours,
            CASE l.label_id {' '.join(precision_cases)}
                ELSE 'source_specific' END AS time_precision,
            CASE l.label_id {' '.join(evidence_id_cases)} ELSE '[]' END
                AS evidence_ids_json,
            CASE l.label_id {' '.join(summary_cases)} ELSE '' END
                AS evidence_summary,
            CASE
                WHEN l.source_tables = 'machine_measurements'
                    THEN 'same_ecg_study_id_linkage'
                ELSE 'ehr_device_clock_not_externally_synchronized'
            END AS temporal_evidence_scope,
            FALSE AS ehr_time_alignment_confirmatory_eligible,
            l.source_tables,
            '{LABEL_RULE_VERSION}' AS rule_version
        FROM {context_relation} c
        CROSS JOIN labels l
    """


def _wide_label_query(
    long_path: Path,
    definitions: tuple[LabelDefinition, ...],
) -> str:
    states = ",\n            ".join(
        (
            f"MAX(label_state) FILTER (WHERE label_id = '{definition.label_id}') "
            f"AS {definition.label_id}_state"
        )
        for definition in definitions
    )
    path = _sql_path(long_path)
    return f"""
        SELECT
            subject_id,
            study_id,
            MIN(ecg_time) AS ecg_time,
            MIN(split) AS split,
            {states}
        FROM read_parquet('{path}')
        GROUP BY subject_id, study_id
    """


def _index_cohort_query(long_path: Path) -> str:
    """One index ECG per patient and endpoint (audit X1).

    Cross-sectional endpoints (ECG against a near-simultaneous echo, lab or
    coded diagnosis) take the first *verified* ECG: the first one with a
    decided reference, chosen by the reference's existence and blind to its
    result -- the STARD form. The earlier rule took the first ECG at all and
    then dropped the patient when that ECG had no reference, which kept 22-78 %
    of the verified patients and lost positives disproportionately, because
    disease appears later in a patient's record. Patients never verified keep
    their first ECG, so the verification-bias analysis still sees them.

    Prognostic endpoints (``predictive``: new-onset AF, mortality) keep the
    first ECG as time zero; choosing it by follow-up would make time zero
    depend on the future.
    """
    path = _sql_path(long_path)
    return f"""
        WITH candidates AS (
            SELECT
                *,
                eligibility_state = 'eligible'
                    AND label_state IN ('positive', 'negative') AS reference_verified,
                target_kind = 'predictive' AS time_zero_design
            FROM read_parquet('{path}')
            WHERE eligibility_state <> 'excluded'
        ),
        ranked AS (
            SELECT
                *,
                BOOL_OR(reference_verified) OVER (
                    PARTITION BY subject_id, label_id
                ) AS any_reference_verified,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id, label_id
                    ORDER BY
                        CASE
                            WHEN time_zero_design OR reference_verified THEN 0
                            ELSE 1
                        END,
                        ecg_time,
                        study_id
                ) AS index_rank
            FROM candidates
        )
        SELECT
            * EXCLUDE(
                index_rank,
                reference_verified,
                time_zero_design,
                any_reference_verified
            ),
            CASE
                WHEN time_zero_design THEN 'first_nonexcluded_ecg_time_zero'
                WHEN any_reference_verified
                    THEN 'first_verified_ecg_reference_result_blind'
                ELSE 'first_nonexcluded_ecg_never_verified'
            END AS index_selection_strategy
        FROM ranked
        WHERE index_rank = 1
    """


def _window_sensitivity_cohort_query(
    context_path: Path,
    rules: dict[str, SQLLabelRule],
    *,
    split_values: Optional[tuple[str, ...]] = None,
) -> str:
    context_relation = _parquet_relation(
        context_path,
        split_values=split_values,
    )
    selections: list[str] = []
    echo_labels = (
        "echo_lvh_strict",
        "reduced_lvef",
        "echo_lae",
        "echo_rv_dilation",
        "echo_rv_systolic_dysfunction",
    )
    echo_profiles = (
        ("strict_7d", "has_tte_7d", -24 * 7, 24 * 7),
        ("standard_30d", "has_tte_30d", -24 * 30, 24 * 30),
        ("sensitive_90d", "has_tte_90d", -24 * 90, 24 * 90),
    )
    standard_availability = "COALESCE(has_tte_30d, FALSE)"
    for label_id in echo_labels:
        rule = rules[label_id]
        for profile, availability_column, before, after in echo_profiles:
            availability = f"COALESCE({availability_column}, FALSE)"
            state = rule.state.replace(standard_availability, availability)
            eligibility = rule.eligibility.replace(
                standard_availability,
                availability,
            )
            confidence = (
                f"CASE WHEN ({eligibility}) <> 'eligible' THEN 'none' "
                "WHEN COALESCE(has_tte_7d, FALSE) THEN 'high' "
                "WHEN COALESCE(has_tte_30d, FALSE) THEN 'medium' "
                "ELSE 'low' END"
            )
            selections.append(
                _window_profile_select(
                    context_relation,
                    label_id=label_id,
                    profile=profile,
                    before_hours=before,
                    after_hours=after,
                    state=state,
                    eligibility=eligibility,
                    confidence=confidence,
                    time_delta_hours=rule.time_delta_hours,
                    evidence_ids=rule.evidence_ids,
                )
            )

    troponin_profiles = (
        (
            "strict_-3h_+6h",
            -3,
            6,
            "troponin_max_url_ratio_strict",
            "troponin_count_strict",
            "troponin_nearest_delta_hours_strict",
            "troponin_nearest_labevent_id_strict",
        ),
        (
            "standard_-6h_+12h",
            -6,
            12,
            "troponin_max_url_ratio",
            "troponin_count_standard",
            "troponin_nearest_delta_hours",
            "troponin_nearest_labevent_id",
        ),
        (
            "sensitive_24h",
            -24,
            24,
            "troponin_max_url_ratio_24h",
            "troponin_count_24h",
            "troponin_nearest_delta_hours_24h",
            "troponin_nearest_labevent_id_24h",
        ),
    )
    for (
        profile,
        before,
        after,
        ratio,
        count,
        delta,
        evidence_id,
    ) in troponin_profiles:
        suffix = {"strict_-3h_+6h": "strict", "standard_-6h_+12h": "standard",
                  "sensitive_24h": "24h"}[profile]
        state = f"COALESCE(troponin_interval_state_{suffix}, 'unavailable')"
        eligibility = _availability_eligibility(
            f"{state} IN ('positive', 'negative')"
        )
        delta = (f"CASE WHEN {state} = 'positive' THEN "
                 f"troponin_interval_positive_delta_hours_{suffix} ELSE {delta} END")
        evidence_id = (f"CASE WHEN {state} = 'positive' THEN "
                       f"troponin_interval_positive_labevent_id_{suffix} ELSE {evidence_id} END")
        selections.append(
            _window_profile_select(
                context_relation,
                label_id="troponin_positive_any",
                profile=profile,
                before_hours=before,
                after_hours=after,
                state=state,
                eligibility=eligibility,
                confidence=(
                    f"CASE WHEN ({eligibility}) = 'eligible' "
                    "THEN 'high' ELSE 'none' END"
                ),
                time_delta_hours=delta,
                evidence_ids=_single_evidence_id_json(
                    evidence_id,
                    "labevent",
                ),
            )
        )

    potassium_profiles = (
        (
            "strict_2h",
            -2,
            2,
            "potassium_nearest_2h",
            "potassium_count_2h",
            "potassium_delta_hours_2h",
            "potassium_nearest_labevent_id_2h",
        ),
        (
            "standard_6h",
            -6,
            6,
            "potassium_nearest",
            "potassium_count_6h",
            "potassium_delta_hours",
            "potassium_nearest_labevent_id",
        ),
        (
            "sensitive_12h",
            -12,
            12,
            "potassium_nearest_12h",
            "potassium_count_12h",
            "potassium_delta_hours_12h",
            "potassium_nearest_labevent_id_12h",
        ),
        (
            "sensitive_24h",
            -24,
            24,
            "potassium_nearest_24h",
            "potassium_count_24h",
            "potassium_delta_hours_24h",
            "potassium_nearest_labevent_id_24h",
        ),
    )
    potassium_rules = {
        "hypokalemia_near_ecg": (
            "{value} < 3.5",
            "{value} BETWEEN 3.5 AND 5.5",
        ),
        "severe_hypokalemia_near_ecg": (
            "{value} < 3.0",
            "{value} >= 3.0",
        ),
        "hyperkalemia_near_ecg": (
            "{value} > 5.5",
            "{value} BETWEEN 3.5 AND 5.5",
        ),
        "severe_hyperkalemia_near_ecg": (
            "{value} >= 6.5",
            "{value} < 6.5",
        ),
    }
    for label_id, (positive_template, negative_template) in (
        potassium_rules.items()
    ):
        for (
            profile,
            before,
            after,
            value,
            count,
            delta,
            evidence_id,
        ) in potassium_profiles:
            measured = f"COALESCE({count}, 0) > 0 AND {value} IS NOT NULL"
            positive = positive_template.format(value=value)
            negative = negative_template.format(value=value)
            state = (
                f"CASE WHEN NOT ({measured}) THEN 'unavailable' "
                f"WHEN {positive} THEN 'positive' "
                f"WHEN {negative} THEN 'negative' "
                "ELSE 'indeterminate' END"
            )
            eligibility = _availability_eligibility(
                f"({measured}) AND (({positive}) OR ({negative}))"
            )
            selections.append(
                _window_profile_select(
                    context_relation,
                    label_id=label_id,
                    profile=profile,
                    before_hours=before,
                    after_hours=after,
                    state=state,
                    eligibility=eligibility,
                    confidence=(
                        f"CASE WHEN ({eligibility}) = 'eligible' "
                        "THEN 'high' ELSE 'none' END"
                    ),
                    time_delta_hours=delta,
                    evidence_ids=_single_evidence_id_json(
                        evidence_id,
                        "labevent",
                    ),
                )
            )

    union_sql = "\nUNION ALL\n".join(selections)
    return _reference_blind_window_index_query(union_sql)


def _reference_blind_window_index_query(candidates_sql: str) -> str:
    """Choose one index per patient/endpoint, shared by every time window.

    As in the main index cohort (audit X1) the first ECG verified in any
    window profile comes first, blind to the reference result; patients never
    verified keep their first ECG. Only records excluded in every profile
    leave the prespecified index universe. All profile states at the selected
    recording survive for availability QC.
    """
    return f"""
        WITH candidates AS (
            {candidates_sql}
        ),
        ranked AS (
            SELECT
                subject_id, label_id, study_id, ecg_time,
                ROW_NUMBER() OVER (
                    PARTITION BY subject_id, label_id
                    ORDER BY
                        CASE WHEN BOOL_OR(
                            eligibility_state = 'eligible'
                            AND label_state IN ('positive', 'negative')
                        ) THEN 0 ELSE 1 END,
                        ecg_time,
                        study_id
                ) AS index_rank
            FROM candidates
            GROUP BY subject_id, label_id, study_id, ecg_time
            HAVING BOOL_OR(eligibility_state <> 'excluded')
        )
        SELECT
            c.*,
            'first_verified_ecg_reference_result_blind_shared_windows' AS index_selection_strategy
        FROM candidates c
        JOIN ranked r USING (subject_id, label_id, study_id)
        WHERE r.index_rank = 1
          AND c.ecg_time IS NOT DISTINCT FROM r.ecg_time
    """


def _window_profile_select(
    context_relation: str,
    *,
    label_id: str,
    profile: str,
    before_hours: float,
    after_hours: float,
    state: str,
    eligibility: str,
    confidence: str,
    time_delta_hours: str,
    evidence_ids: str,
) -> str:
    return f"""
        SELECT
            subject_id,
            study_id,
            ecg_time,
            split,
            '{label_id}' AS label_id,
            '{profile}' AS window_profile,
            {float(before_hours)}::DOUBLE AS window_before_hours,
            {float(after_hours)}::DOUBLE AS window_after_hours,
            ({state}) AS label_state,
            ({eligibility}) AS eligibility_state,
            ({confidence}) AS confidence,
            CAST(({time_delta_hours}) AS DOUBLE) AS time_delta_hours,
            'timestamp' AS time_precision,
            ({evidence_ids}) AS evidence_ids_json,
            '{WINDOW_SENSITIVITY_VERSION}' AS window_rule_version
        FROM {context_relation}
    """


def _window_sensitivity_summary_query(path: Path) -> str:
    source = _sql_path(path)
    return f"""
        SELECT
            label_id,
            window_profile,
            window_before_hours,
            window_after_hours,
            split,
            COUNT(*) AS index_ecg_count,
            COUNT(DISTINCT subject_id) AS subject_count,
            COUNT(*) FILTER (WHERE label_state = 'positive' AND eligibility_state = 'eligible')
                AS positive_subject_count,
            COUNT(*) FILTER (WHERE label_state = 'negative' AND eligibility_state = 'eligible')
                AS negative_subject_count,
            COUNT(*) FILTER (WHERE label_state = 'unavailable') AS unavailable_subject_count,
            COUNT(*) FILTER (WHERE label_state = 'indeterminate') AS indeterminate_subject_count,
            COUNT(*) FILTER (WHERE eligibility_state <> 'eligible') AS ineligible_subject_count,
            (positive_subject_count + negative_subject_count)::DOUBLE / NULLIF(COUNT(*), 0)
                AS reference_available_fraction,
            positive_subject_count::DOUBLE
                / NULLIF(positive_subject_count + negative_subject_count, 0)
                AS prevalence,
            MEDIAN(ABS(time_delta_hours)) FILTER (
                WHERE eligibility_state = 'eligible' AND label_state IN ('positive', 'negative')
            ) AS median_absolute_delta_hours,
            '{WINDOW_SENSITIVITY_VERSION}' AS window_rule_version
        FROM read_parquet('{source}')
        GROUP BY
            label_id,
            window_profile,
            window_before_hours,
            window_after_hours,
            split
        ORDER BY label_id, window_before_hours, split
    """


def _label_summary_query(long_path: Path, cohort_path: Path) -> str:
    path = _sql_path(long_path)
    cohort = _sql_path(cohort_path)
    return f"""
        WITH raw AS (
            SELECT
                label_id,
                MIN(target_kind) AS target_kind,
                MIN(panel) AS panel,
                COUNT(*) AS ecg_count,
                COUNT(DISTINCT subject_id) AS subject_count,
                COUNT(*) FILTER (WHERE label_state = 'indeterminate')
                    AS indeterminate_count,
                COUNT(*) FILTER (WHERE label_state = 'unavailable')
                    AS unavailable_count,
                COUNT(*) FILTER (WHERE eligibility_state = 'excluded')
                    AS excluded_count,
                MIN(source_tables) AS source_tables,
                MIN(index_policy) AS index_policy,
                MIN(window_before_hours) AS window_before_hours,
                MIN(window_after_hours) AS window_after_hours,
                MIN(rule_version) AS rule_version
            FROM read_parquet('{path}')
            GROUP BY label_id
        ),
        per_split AS (
            SELECT
                label_id,
                MIN(minority) AS smallest_split_minority_subject_count
            FROM (
                SELECT
                    label_id,
                    split,
                    LEAST(
                        COUNT(DISTINCT subject_id) FILTER (
                            WHERE label_state = 'positive'
                        ),
                        COUNT(DISTINCT subject_id) FILTER (
                            WHERE label_state = 'negative'
                        )
                    ) AS minority
                FROM read_parquet('{cohort}')
                WHERE split IN ('train', 'validation')
                GROUP BY label_id, split
            )
            GROUP BY label_id
        ),
        selected AS (
            SELECT
                label_id,
                COUNT(*) FILTER (WHERE label_state = 'positive')
                    AS positive_ecg_count,
                COUNT(DISTINCT subject_id) FILTER (
                    WHERE label_state = 'positive'
                ) AS positive_subject_count,
                COUNT(*) FILTER (WHERE label_state = 'negative')
                    AS negative_ecg_count,
                COUNT(DISTINCT subject_id) FILTER (
                    WHERE label_state = 'negative'
                ) AS negative_subject_count
            FROM read_parquet('{cohort}')
            GROUP BY label_id
        )
        SELECT
            r.*,
            COALESCE(s.positive_ecg_count, 0) AS positive_ecg_count,
            COALESCE(s.positive_subject_count, 0) AS positive_subject_count,
            COALESCE(s.negative_ecg_count, 0) AS negative_ecg_count,
            COALESCE(s.negative_subject_count, 0) AS negative_subject_count,
            s.positive_subject_count::DOUBLE
                / NULLIF(
                    s.positive_subject_count + s.negative_subject_count,
                    0
                ) AS subject_prevalence,
            COALESCE(p.smallest_split_minority_subject_count, 0)
                AS smallest_split_minority_subject_count,
            -- Power was judged on class sizes pooled over train and validation,
            -- which says nothing about whether either half can carry an
            -- estimate on its own. The screen replicates on validation, so the
            -- binding constraint is the smaller half.
            CASE
                WHEN COALESCE(s.positive_subject_count, 0) < 30
                  OR COALESCE(s.negative_subject_count, 0) < 30
                    THEN 'no_inference'
                WHEN s.positive_subject_count < 100
                  OR s.negative_subject_count < 100
                  OR COALESCE(p.smallest_split_minority_subject_count, 0) < 100
                    THEN 'underpowered_descriptive'
                ELSE 'formal_analysis_eligible'
            END AS power_status,
            -- Prevalence was not looked at at all, and absolute counts hid the
            -- consequence: significant_valvular_disease ran at a prevalence of
            -- 0.969 -- 9,400 positives against 301 negatives -- and was still
            -- reported as eligible for formal analysis, because 301 clears the
            -- hundred-subject floor.
            --
            -- The check is deliberately one-sided. A low prevalence is what a
            -- rare disease looks like: severe aortic regurgitation at 0.017
            -- with 238 positives against 14,019 negatives is a well-specified
            -- endpoint, and the absolute-count rule above already protects it.
            -- A high one is different. The reference standards here are ordered
            -- on suspicion -- an echocardiogram, a troponin -- so a positive
            -- rate above 90 % says more about who was tested than about any
            -- population, and the negative arm is whatever survived that
            -- selection. Such an endpoint needs the selection-bias bounds, not
            -- an accuracy claim.
            CASE
                WHEN COALESCE(s.positive_subject_count, 0)
                   + COALESCE(s.negative_subject_count, 0) = 0
                    THEN 'not_assessable'
                WHEN s.positive_subject_count::DOUBLE
                     / NULLIF(
                         s.positive_subject_count + s.negative_subject_count, 0
                     ) > 0.90
                    THEN 'high_prevalence_verification_conditioned'
                ELSE 'prevalence_supports_inference'
            END AS prevalence_status
        FROM raw r
        LEFT JOIN selected s USING (label_id)
        LEFT JOIN per_split p USING (label_id)
        ORDER BY panel, label_id
    """


def _split_summary_query(cohort_path: Path) -> str:
    path = _sql_path(cohort_path)
    return f"""
        SELECT
            label_id,
            split,
            COUNT(DISTINCT subject_id) FILTER (
                WHERE label_state = 'positive'
            ) AS positive_subject_count,
            COUNT(DISTINCT subject_id) FILTER (
                WHERE label_state = 'negative'
            ) AS negative_subject_count,
            COUNT(DISTINCT subject_id) AS total_subject_count,
            positive_subject_count::DOUBLE
                / NULLIF(positive_subject_count + negative_subject_count, 0)
                AS prevalence
        FROM read_parquet('{path}')
        GROUP BY label_id, split
        ORDER BY label_id, split
    """


def _composite_rule(
    *,
    positive: str,
    assessed: str,
    negative: Optional[str] = None,
    indeterminate: str = "FALSE",
    summary: str,
    positive_confidence: str = "'high'",
    time_delta_hours: str = "NULL::DOUBLE",
    time_precision: str = "'source_specific'",
    evidence_ids: str = "'[]'",
) -> SQLLabelRule:
    negative_expression = (
        negative
        if negative is not None
        else f"({assessed}) AND NOT COALESCE(({positive}), FALSE)"
    )
    positive_value = f"COALESCE(({positive}), FALSE)"
    assessed_value = f"COALESCE(({assessed}), FALSE)"
    negative_value = f"COALESCE(({negative_expression}), FALSE)"
    indeterminate_value = f"COALESCE(({indeterminate}), FALSE)"
    return SQLLabelRule(
        state=f"""
            CASE
                WHEN {positive_value} AND {negative_value} THEN 'indeterminate'
                WHEN {positive_value} THEN 'positive'
                WHEN {indeterminate_value} THEN 'indeterminate'
                WHEN NOT {assessed_value} THEN 'unavailable'
                WHEN {negative_value} THEN 'negative'
                ELSE 'indeterminate'
            END
        """,
        eligibility=f"""
            CASE
                WHEN {positive_value} AND NOT {negative_value}
                    THEN 'eligible'
                WHEN {assessed_value}
                 AND {negative_value}
                 AND NOT {positive_value}
                 AND NOT {indeterminate_value}
                    THEN 'eligible'
                ELSE 'insufficient_evidence'
            END
        """,
        confidence=f"""
            CASE
                WHEN {positive_value} AND {negative_value} THEN 'low'
                WHEN {positive_value} THEN {positive_confidence}
                WHEN NOT {assessed_value} THEN 'none'
                WHEN {indeterminate_value} THEN 'low'
                WHEN {negative_value} THEN 'medium'
                ELSE 'low'
            END
        """,
        time_delta_hours=time_delta_hours,
        time_precision=time_precision,
        evidence_ids=evidence_ids,
        evidence_summary=summary,
    )


def _binary_context_rule(expression: str, summary: str) -> SQLLabelRule:
    normalized = (
        expression
        if any(token in expression for token in ("(", " ", "OR", "AND"))
        else f"COALESCE({expression}, FALSE)"
    )
    return SQLLabelRule(
        state=f"""
            CASE
                WHEN hadm_id IS NULL THEN 'unavailable'
                WHEN ({normalized}) THEN 'positive'
                ELSE 'negative'
            END
        """,
        eligibility=_availability_eligibility("hadm_id IS NOT NULL"),
        confidence=(
            f"CASE WHEN ({normalized}) THEN 'low' "
            "WHEN hadm_id IS NOT NULL THEN 'medium' ELSE 'none' END"
        ),
        time_precision="'encounter'",
        evidence_summary=f"'{summary.replace(chr(39), chr(39) * 2)}'",
    )


def _history_rule(expression: str, summary: str) -> SQLLabelRule:
    assessed = (
        "first_encounter_time IS NOT NULL "
        "AND first_encounter_time <= ecg_time"
    )
    return SQLLabelRule(
        state=f"""
            CASE
                WHEN COALESCE({expression}, FALSE) THEN 'positive'
                WHEN {assessed} THEN 'negative'
                ELSE 'unavailable'
            END
        """,
        eligibility=_availability_eligibility(
            f"COALESCE({expression}, FALSE) OR ({assessed})"
        ),
        confidence=f"""
            CASE
                WHEN COALESCE({expression}, FALSE) THEN 'medium'
                WHEN {assessed} THEN 'low'
                ELSE 'none'
            END
        """,
        time_precision="'mixed_date_and_encounter'",
        evidence_summary=f"'{summary.replace(chr(39), chr(39) * 2)}'",
    )


def _binary_exposure_rule(expression: str, summary: str) -> SQLLabelRule:
    return SQLLabelRule(
        state=f"""
            CASE
                WHEN COALESCE({expression}, FALSE) THEN 'positive'
                WHEN COALESCE(medication_assessed, FALSE) THEN 'negative'
                ELSE 'unavailable'
            END
        """,
        eligibility=_availability_eligibility(
            f"COALESCE({expression}, FALSE) "
            "OR COALESCE(medication_assessed, FALSE)"
        ),
        confidence=f"""
            CASE
                WHEN COALESCE({expression}, FALSE) THEN 'medium'
                WHEN COALESCE(medication_assessed, FALSE) THEN 'low'
                ELSE 'none'
            END
        """,
        time_precision="'timestamp_or_active_interval'",
        evidence_summary=f"'{summary.replace(chr(39), chr(39) * 2)}'",
    )


def _availability_eligibility(condition: str) -> str:
    return (
        f"CASE WHEN ({condition}) THEN 'eligible' "
        "ELSE 'insufficient_evidence' END"
    )


def _confidence(
    condition: str,
    *,
    high: str,
    fallback: str,
) -> str:
    return f"CASE WHEN ({condition}) THEN {high} ELSE {fallback} END"


def _single_evidence_id_json(column: str, prefix: str) -> str:
    return (
        f"CASE WHEN {column} IS NULL THEN '[]' "
        f"ELSE '[\"{prefix}:' || {column}::VARCHAR || '\"]' END"
    )


def _paired_evidence_ids_json(
    first_column: str,
    second_column: str,
    prefix: str,
) -> str:
    return (
        "CASE "
        f"WHEN {first_column} IS NULL AND {second_column} IS NULL THEN '[]' "
        f"WHEN {second_column} IS NULL OR {second_column} = {first_column} "
        f"THEN '[\"{prefix}:' || {first_column}::VARCHAR || '\"]' "
        f"WHEN {first_column} IS NULL "
        f"THEN '[\"{prefix}:' || {second_column}::VARCHAR || '\"]' "
        f"ELSE '[\"{prefix}:' || {first_column}::VARCHAR || "
        f"'\",\"{prefix}:' || {second_column}::VARCHAR || '\"]' END"
    )


# The canonical label build over every split (50.6 million rows in the
# current evidence) spills. With insertion order preserved and the
# per-connection spill cap of 16 GiB introduced on 2026-09-16, the first
# rebuild after that date (pilot A123, 2026-09-19) ended with DuckDB's
# "failed to offload data block (16.0 GiB/16.0 GiB used)". Row order is not
# part of the label contract -- every consumer joins by keys and the
# attestation compares rule results per label -- so the build takes the
# same session settings the evidence build already uses, plus a spill
# allowance of its own below the work root, where the free space is.
LABEL_BUILD_SPILL_GIB = 48


def label_build_session_options(config: MimicProjectConfig) -> tuple[str, ...]:
    spill_root = config.work_root / "tmp" / "labels_duckdb"
    spill_root.mkdir(parents=True, exist_ok=True)
    return (
        "SET preserve_insertion_order=false",
        f"SET temp_directory='{_sql_path(spill_root)}'",
        f"SET max_temp_directory_size='{LABEL_BUILD_SPILL_GIB}GiB'",
    )


def _copy_query_atomic(connection, query: str, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    temporary_sql = _sql_path(temporary)
    with profile_query(f"labels:{output_path.name}") as profile:
        copy_result = connection.execute(
            f"COPY ({query}) TO '{temporary_sql}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
        ).fetchone()
        if copy_result is None:
            raise RuntimeError("DuckDB COPY returned no row count")
        count = int(copy_result[0])
        atomic_replace(temporary, output_path)
        profile["rows"] = count
        return count


def _summary_records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.where(pd.notna(frame), None)
    return frame.to_dict(orient="records")


def _optional_float_sql(value: Optional[float]) -> str:
    return (
        "NULL::DOUBLE"
        if value is None
        else f"{float(value)}::DOUBLE"
    )


def development_label_attestation_path(
    config: MimicProjectConfig,
) -> Path:
    return config.labels_root / DEVELOPMENT_LABEL_ATTESTATION_NAME


def attest_development_label_artifacts(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    attestation_path = development_label_attestation_path(config)
    if attestation_path.exists() and not force:
        current = verify_development_label_attestation(
            config, require_current=False
        )
        if current is not None:
            return current
        if progress:
            progress(
                "Vorhandene Label-Attestierung ist ueberholt; sie wird neu "
                "erstellt"
            )

    context_path = config.evidence_root / "clinical_context_features.parquet"
    artifacts = _development_label_artifact_paths(config)
    required = {"context": context_path, **artifacts}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Development label attestation inputs are missing: "
            + ", ".join(missing)
        )
    for path in artifacts.values():
        _assert_unsealed_development_path(config, path)

    audit_root = (
        config.labels_root / "audits" / "current-code-development-v1"
    )
    temp_root = audit_root / "tmp"
    audit_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    rules = _label_rules()
    definitions = default_label_definitions()
    long_path = artifacts["clinical_label_evidence"]
    long_sql = _sql_path(long_path)
    implementation_hash = label_implementation_hash()
    audit_input_hash = content_hash(
        {
            "implementation_hash": implementation_hash,
            "clinical_context": file_content_fingerprint(context_path),
            "canonical_long": file_content_fingerprint(long_path),
            "development_splits": ("train", "validation"),
        }
    )
    progress_path = audit_root / "long_rule_progress.json"
    long_results = _load_label_audit_progress(
        progress_path,
        audit_input_hash=audit_input_hash,
    )
    completed_labels = {
        str(item.get("label_id"))
        for item in long_results
        if item.get("exact") is True
    }

    audit_catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=1,
        threads=1,
    )
    with audit_catalog.connect(isolated=True) as connection:
        connection.execute(
            f"SET temp_directory='{_sql_path(temp_root)}'"
        )
        connection.execute("SET preserve_insertion_order=FALSE")
        splits = {
            str(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT split FROM read_parquet('{long_sql}')"
            ).fetchall()
        }
        if not splits or not splits <= {"train", "validation"}:
            raise RuntimeError(
                "Development label attestation refuses non-development splits: "
                f"{sorted(splits)}"
            )

        for index, definition in enumerate(definitions, start=1):
            if progress:
                progress(
                    "Pruefe Labelregel "
                    f"{index:,}/{len(definitions):,}: {definition.label_id}"
                )
            if definition.label_id in completed_labels:
                continue
            current_query = _long_label_query(
                context_path,
                (definition,),
                {definition.label_id: rules[definition.label_id]},
                split_values=("train", "validation"),
            )
            canonical_query = (
                f"SELECT * FROM read_parquet('{long_sql}') "
                "WHERE label_id = "
                f"'{definition.label_id.replace(chr(39), chr(39) * 2)}'"
            )
            comparison = _exact_query_comparison(
                connection,
                current_query=current_query,
                canonical_query=canonical_query,
            )
            comparison["label_id"] = definition.label_id
            long_results.append(comparison)
            atomic_write_json(
                progress_path,
                {
                    "audit_input_hash": audit_input_hash,
                    "status": "in_progress",
                    "scope": "train_validation_only",
                    "long_rule_results": long_results,
                    "locked_test_read_for_attestation": False,
                },
            )
            if not comparison["exact"]:
                _write_failed_label_audit(
                    audit_root,
                    implementation_hash=implementation_hash,
                    long_results=long_results,
                    reason=(
                        "Current label rule differs from the canonical "
                        f"development artifact for {definition.label_id}"
                    ),
                )
                raise RuntimeError(
                    "Development label attestation failed for "
                    f"{definition.label_id}: {comparison}"
                )

        derived_queries = {
            "clinical_labels_wide": _wide_label_query(
                long_path,
                definitions,
            ),
            "endpoint_index_cohorts": _index_cohort_query(long_path),
            "endpoint_window_sensitivity_cohorts": (
                _window_sensitivity_cohort_query(
                    context_path,
                    rules,
                    split_values=("train", "validation"),
                )
            ),
        }
        derived_results: dict[str, dict] = {}
        for index, (name, query) in enumerate(
            derived_queries.items(),
            start=1,
        ):
            if progress:
                progress(
                    "Pruefe abgeleitete Labelansicht "
                    f"{index:,}/{len(derived_queries):,}: {name}"
                )
            candidate_path = audit_root / f"{name}.candidate.parquet"
            _copy_query_atomic(connection, query, candidate_path)
            comparison = _exact_file_comparison(
                connection,
                candidate_path,
                artifacts[name],
            )
            comparison["candidate"] = file_content_fingerprint(
                candidate_path
            )
            derived_results[name] = comparison
            if not comparison["exact"]:
                _write_failed_label_audit(
                    audit_root,
                    implementation_hash=implementation_hash,
                    long_results=long_results,
                    derived_results=derived_results,
                    reason=(
                        "Current derived-label query differs from the "
                        f"canonical development artifact for {name}"
                    ),
                )
                raise RuntimeError(
                    "Development label attestation failed for "
                    f"{name}: {comparison}"
                )

    comparisons = {
        "status": "exact_development_rebuild",
        "all_exact": True,
        "comparison_operator": "EXCEPT ALL in both directions",
        "long_rule_results": long_results,
        "derived_results": derived_results,
        "locked_test_rows_materialized": False,
    }
    attestation = _write_development_label_attestation(
        config,
        mode="legacy_artifact_exact_rebuild",
        implementation_hash=implementation_hash,
        comparisons=comparisons,
    )
    atomic_write_json(
        progress_path,
        {
            "audit_input_hash": audit_input_hash,
            "status": "complete",
            "scope": "train_validation_only",
            "long_rule_results": long_results,
            "locked_test_read_for_attestation": False,
        },
    )
    atomic_write_json(
        audit_root / "audit_status.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "verified",
            "attestation_hash": attestation["attestation_hash"],
            "label_implementation_hash": implementation_hash,
            "all_exact": True,
            "locked_test_read_for_attestation": False,
        },
    )
    return attestation


def verify_development_label_attestation(
    config: MimicProjectConfig,
    *,
    require_current: bool = True,
) -> Optional[dict]:
    """The attestation payload, or None when it has been superseded.

    ``require_current=True`` is for callers that are about to *rely* on the
    labels -- validation, freezing, anything downstream. For them a superseded
    attestation is fatal, because it no longer covers what they would read.

    ``require_current=False`` is for the two stages that can do something
    about it: the label build, which rebuilds, and the attestation stage,
    which re-attests.
    """

    path = development_label_attestation_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            "Development label attestation is missing: "
            f"{path}. Run attest-labels before formal validation."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_hash = str(payload.get("attestation_hash", ""))
    hash_payload = dict(payload)
    hash_payload.pop("attestation_hash", None)
    if not recorded_hash or content_hash(hash_payload) != recorded_hash:
        raise RuntimeError(
            "Development label attestation hash verification failed"
        )
    if payload.get("status") != "verified":
        raise RuntimeError("Development label attestation is not verified")
    if payload.get("scope") != "train_validation_only":
        raise RuntimeError(
            "Development label attestation has an invalid analysis scope"
        )
    if payload.get("locked_test_read_for_attestation") is not False:
        raise RuntimeError(
            "Development label attestation does not prove locked-test isolation"
        )
    if payload.get("locked_test_semantics_attested") is not False:
        raise RuntimeError(
            "Development attestation must not claim locked-test semantics"
        )
    if not payload.get("comparisons", {}).get("all_exact"):
        raise RuntimeError(
            "Development label attestation contains non-exact comparisons"
        )

    # Everything above says whether the attestation can be trusted at all: a
    # forged or tampered file, a claim about the locked test, a non-exact
    # comparison. Those are failures and stay failures.
    #
    # What follows says something different -- whether it still *describes*
    # the labels on disk. An attestation is a statement about one
    # implementation over one set of inputs, and both move on: rules get
    # refined, the clinical context gets rebuilt. That is the pipeline working
    # as intended, not a breach, and the answer to it is to rebuild the labels
    # and attest them again. Reporting it as superseded lets callers do that;
    # raising would leave no way forward at all, because the labels cannot be
    # rebuilt while the stale attestation forbids it and cannot be re-attested
    # before they are rebuilt.
    superseded: list[str] = []
    if payload.get("label_implementation_hash") != label_implementation_hash():
        superseded.append("label implementation")
    expected_context = file_content_fingerprint(
        config.evidence_root / "clinical_context_features.parquet"
    )
    if payload.get("clinical_context") != expected_context:
        superseded.append("clinical context")
    expected_artifacts = {
        name: file_content_fingerprint(artifact_path)
        for name, artifact_path in _development_label_artifact_paths(
            config
        ).items()
    }
    if payload.get("development_artifacts") != expected_artifacts:
        superseded.append("label artifacts")
    for fingerprint in expected_artifacts.values():
        artifact_path = Path(str(fingerprint["path"]))
        _assert_unsealed_development_path(config, artifact_path)
    if superseded:
        if require_current:
            raise RuntimeError(
                "Development label attestation is superseded ("
                + ", ".join(superseded)
                + "); rebuild the labels and run attest-labels again"
            )
        return None
    return payload


def _write_development_label_attestation(
    config: MimicProjectConfig,
    *,
    mode: str,
    implementation_hash: str,
    comparisons: dict,
) -> dict:
    artifacts = _development_label_artifact_paths(config)
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    context_path = config.evidence_root / "clinical_context_features.parquet"
    if not context_path.is_file():
        missing.append(str(context_path))
    if missing:
        raise FileNotFoundError(
            "Cannot attest missing development label artifacts: "
            + ", ".join(missing)
        )
    for path in artifacts.values():
        _assert_unsealed_development_path(config, path)
    payload = {
        "attestation_version": DEVELOPMENT_LABEL_ATTESTATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "scope": "train_validation_only",
        "source_splits": ["train", "validation"],
        "mode": mode,
        "label_stage_version": LABEL_STAGE_VERSION,
        "label_rule_version": LABEL_RULE_VERSION,
        "label_implementation_hash": implementation_hash,
        "clinical_context": file_content_fingerprint(context_path),
        "development_artifacts": {
            name: file_content_fingerprint(path)
            for name, path in artifacts.items()
        },
        "comparisons": comparisons,
        "locked_test_read_for_attestation": False,
        "locked_test_semantics_attested": False,
        "sealed_artifacts_referenced": False,
    }
    payload["attestation_hash"] = content_hash(payload)
    atomic_write_json(development_label_attestation_path(config), payload)
    return payload


def _development_label_artifact_paths(
    config: MimicProjectConfig,
) -> dict[str, Path]:
    return {
        "clinical_label_evidence": (
            config.labels_root / "clinical_label_evidence.parquet"
        ),
        "clinical_labels_wide": (
            config.labels_root / "clinical_labels_wide.parquet"
        ),
        "endpoint_index_cohorts": (
            config.labels_root / "endpoint_index_cohorts.parquet"
        ),
        "endpoint_window_sensitivity_cohorts": (
            config.labels_root
            / "endpoint_window_sensitivity_cohorts.parquet"
        ),
        "label_definitions": config.labels_root / "label_definitions.json",
    }


def _assert_unsealed_development_path(
    config: MimicProjectConfig,
    path: Path,
) -> None:
    resolved = path.resolve()
    sealed_root = (config.labels_root / "sealed").resolve()
    if resolved == sealed_root or resolved.is_relative_to(sealed_root):
        raise RuntimeError(
            "Development label attestation refuses sealed artifacts: "
            f"{resolved}"
        )


def _exact_file_comparison(
    connection,
    current_path: Path,
    canonical_path: Path,
) -> dict:
    return _exact_query_comparison(
        connection,
        current_query=(
            f"SELECT * FROM read_parquet('{_sql_path(current_path)}')"
        ),
        canonical_query=(
            f"SELECT * FROM read_parquet('{_sql_path(canonical_path)}')"
        ),
    )


def _exact_query_comparison(
    connection,
    *,
    current_query: str,
    canonical_query: str,
) -> dict:
    current_relation = (
        f"SELECT * FROM ({current_query}) current_relation"
    )
    canonical_relation = (
        f"SELECT * FROM ({canonical_query}) canonical_relation"
    )
    current_schema = [
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            f"DESCRIBE {current_relation}"
        ).fetchall()
    ]
    canonical_schema = [
        (str(row[0]), str(row[1]))
        for row in connection.execute(
            f"DESCRIBE {canonical_relation}"
        ).fetchall()
    ]
    if current_schema != canonical_schema:
        return {
            "exact": False,
            "schema_exact": False,
            "current_schema": current_schema,
            "canonical_schema": canonical_schema,
            "current_minus_canonical": None,
            "canonical_minus_current": None,
        }
    current_count = int(
        connection.execute(
            f"SELECT count(*) FROM ({current_relation}) rows"
        ).fetchone()[0]
    )
    canonical_count = int(
        connection.execute(
            f"SELECT count(*) FROM ({canonical_relation}) rows"
        ).fetchone()[0]
    )
    current_minus = int(
        connection.execute(
            "SELECT count(*) FROM ("
            f"{current_relation} EXCEPT ALL {canonical_relation}"
            ") differences"
        ).fetchone()[0]
    )
    canonical_minus = int(
        connection.execute(
            "SELECT count(*) FROM ("
            f"{canonical_relation} EXCEPT ALL {current_relation}"
            ") differences"
        ).fetchone()[0]
    )
    return {
        "exact": (
            current_count == canonical_count
            and current_minus == 0
            and canonical_minus == 0
        ),
        "schema_exact": True,
        "row_count": current_count,
        "canonical_row_count": canonical_count,
        "current_minus_canonical": current_minus,
        "canonical_minus_current": canonical_minus,
    }


def _write_failed_label_audit(
    audit_root: Path,
    *,
    implementation_hash: str,
    long_results: list[dict],
    reason: str,
    derived_results: Optional[dict[str, dict]] = None,
) -> None:
    atomic_write_json(
        audit_root / "failed_audit.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "scope": "train_validation_only",
            "reason": reason,
            "label_implementation_hash": implementation_hash,
            "long_rule_results": long_results,
            "derived_results": derived_results or {},
            "locked_test_read_for_attestation": False,
        },
    )


def _load_label_audit_progress(
    path: Path,
    *,
    audit_input_hash: str,
) -> list[dict]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if payload.get("audit_input_hash") != audit_input_hash:
        return []
    results = payload.get("long_rule_results")
    if not isinstance(results, list):
        return []
    return [
        dict(item)
        for item in results
        if isinstance(item, dict)
        and item.get("exact") is True
        and item.get("label_id")
    ]


def verify_clinical_label_artifacts(
    config: MimicProjectConfig,
) -> dict[str, int | str]:
    root = config.labels_root
    paths = {
        "development_long": root / "clinical_label_evidence.parquet",
        "development_wide": root / "clinical_labels_wide.parquet",
        "development_cohort": root / "endpoint_index_cohorts.parquet",
        "locked_long": root / "sealed" / "locked_clinical_label_evidence.parquet",
        "locked_wide": root / "sealed" / "locked_clinical_labels_wide.parquet",
        "locked_cohort": root / "sealed" / "locked_endpoint_index_cohorts.parquet",
        "definitions": root / "label_definitions.json",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Clinical label artifact integrity failed; missing="
            + ", ".join(missing)
        )

    definition_payload = json.loads(
        paths["definitions"].read_text(encoding="utf-8")
    )
    definition_version = str(definition_payload.get("rule_version", ""))
    if definition_version != LABEL_RULE_VERSION:
        raise RuntimeError(
            "Clinical label definitions are stale: "
            f"expected {LABEL_RULE_VERSION}, found {definition_version or 'missing'}"
        )
    expected_ids = {
        definition.label_id
        for definition in default_label_definitions()
    }
    definition_rows = definition_payload.get("definitions", [])
    actual_ids = {
        str(item.get("label_id", ""))
        for item in definition_rows
        if isinstance(item, dict)
    }
    stale_definition_rows = []
    for item in definition_rows:
        if not isinstance(item, dict):
            stale_definition_rows.append("<non-object>")
        elif item.get("rule_version") != LABEL_RULE_VERSION:
            stale_definition_rows.append(str(item.get("label_id", "missing")))
    if actual_ids != expected_ids or stale_definition_rows:
        raise RuntimeError(
            "Clinical label definitions do not match the active registry: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}, "
            f"stale={sorted(stale_definition_rows)}"
        )

    dev_long = _sql_path(paths["development_long"])
    dev_wide = _sql_path(paths["development_wide"])
    dev_cohort = _sql_path(paths["development_cohort"])
    locked_long = _sql_path(paths["locked_long"])
    locked_wide = _sql_path(paths["locked_wide"])
    locked_cohort = _sql_path(paths["locked_cohort"])
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        def count(query: str) -> int:
            return int(connection.execute(query).fetchone()[0])

        diagnostics: dict[str, int | str] = {
            "status": "verified",
            "rule_version": LABEL_RULE_VERSION,
            "invalid_development_state_eligibility": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_long}')
                WHERE eligibility_state = 'eligible'
                  AND label_state NOT IN ('positive', 'negative')
                """
            ),
            "invalid_locked_state_eligibility": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{locked_long}')
                WHERE eligibility_state = 'eligible'
                  AND label_state NOT IN ('positive', 'negative')
                """
            ),
            "development_duplicate_subject_label": count(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT subject_id, label_id, COUNT(*) AS n
                    FROM read_parquet('{dev_cohort}')
                    GROUP BY ALL
                    HAVING n > 1
                )
                """
            ),
            "locked_duplicate_subject_label": count(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT subject_id, label_id, COUNT(*) AS n
                    FROM read_parquet('{locked_cohort}')
                    GROUP BY ALL
                    HAVING n > 1
                )
                """
            ),
            "development_split_contamination": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_cohort}')
                WHERE split NOT IN ('train', 'validation')
                """
            ),
            "locked_split_contamination": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{locked_cohort}')
                WHERE split <> 'locked_test'
                """
            ),
            "subject_split_overlap": count(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT subject_id
                    FROM read_parquet('{dev_wide}')
                    INTERSECT
                    SELECT DISTINCT subject_id
                    FROM read_parquet('{locked_wide}')
                )
                """
            ),
            "development_rule_version_mismatch": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_long}')
                WHERE rule_version <> '{LABEL_RULE_VERSION}'
                   OR rule_version IS NULL
                """
            ),
            "locked_rule_version_mismatch": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{locked_long}')
                WHERE rule_version <> '{LABEL_RULE_VERSION}'
                   OR rule_version IS NULL
                """
            ),
            "missing_positive_reference_ids": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_long}')
                WHERE label_state = 'positive'
                  AND label_id IN (
                      'elevated_ntprobnp',
                      'acute_decompensated_heart_failure_proxy',
                      'right_heart_strain_proxy'
                  )
                  AND (
                      evidence_ids_json IS NULL
                      OR evidence_ids_json = '[]'
                  )
                """
            ),
            "reference_delta_outside_declared_window": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_long}')
                WHERE label_state = 'positive'
                  AND (
                      (
                          label_id IN (
                              'elevated_ntprobnp',
                              'acute_decompensated_heart_failure_proxy'
                          )
                          AND (
                              time_delta_hours IS NULL
                              OR time_delta_hours < -6
                              OR time_delta_hours > 12
                          )
                      )
                      OR (
                          label_id = 'right_heart_strain_proxy'
                          AND (
                              time_delta_hours IS NULL
                              OR ABS(time_delta_hours) > 168
                          )
                      )
                  )
                """
            ),
            "window_metadata_mismatch": count(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{dev_long}')
                WHERE (
                    label_id = 'right_heart_strain_proxy'
                    AND (
                        window_before_hours IS DISTINCT FROM -168.0
                        OR window_after_hours IS DISTINCT FROM 168.0
                    )
                ) OR (
                    label_id = 'mortality_30d'
                    AND (
                        window_before_hours IS DISTINCT FROM 0.0
                        OR window_after_hours IS DISTINCT FROM 720.0
                    )
                ) OR (
                    label_id = 'heart_failure_icd'
                    AND (
                        window_before_hours IS NOT NULL
                        OR window_after_hours IS NOT NULL
                    )
                )
                """
            ),
        }

    violations = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"status", "rule_version"} and value != 0
    }
    if violations:
        raise RuntimeError(
            "Clinical label artifact integrity failed: "
            + ", ".join(f"{key}={value}" for key, value in violations.items())
        )
    return diagnostics


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _parquet_relation(
    path: Path | str,
    *,
    split_values: Optional[tuple[str, ...]] = None,
) -> str:
    source = f"read_parquet('{_sql_path(path)}')"
    if not split_values:
        return source
    values = ", ".join(
        "'" + str(value).replace("'", "''") + "'" for value in split_values
    )
    return f"(SELECT * FROM {source} WHERE split IN ({values}))"
