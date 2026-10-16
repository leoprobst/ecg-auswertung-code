from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import MimicProjectConfig
from .event_ingest import load_clinical_event_registry
from .evidence import build_clinical_evidence
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_replace,
    atomic_write_json,
    content_hash,
    file_content_fingerprint,
    file_fingerprint,
)


EVENT_EVIDENCE_ATTESTATION_VERSION = "1.3.0"

EXACT_EVENT_EVIDENCE_STATUS = "verified_exact_equivalence"
EXACT_LABEL_EVENT_EVIDENCE_STATUS = (
    "verified_exact_label_relevant_equivalence"
)
ATTESTED_EVENT_EVIDENCE_STATUSES = frozenset(
    {
        EXACT_EVENT_EVIDENCE_STATUS,
        EXACT_LABEL_EVENT_EVIDENCE_STATUS,
    }
)

# clinical_context_features.parquet is the single wide, denormalized
# per-admission table joining every context/covariate source together
# (demographics, ED/ICU vitals trend, ICU stay/ventilation/vasopressor/
# fluid/GCS, CK-MB/blood-gas-pH, note-derived mention flags, readmission,
# etc.). Each context-association-domain addition grows its column list
# without touching any existing column -- verified against a real project
# snapshot (2026-08-11): every one of these 70 columns is new-only
# (nothing was removed or renamed), and every prefix/name traces to an
# already-completed, already-tested feature (C1 demographics/outcome,
# C2 ED vitals, ICU module, CK-MB/blood-gas-pH, and both MIMIC-Notes
# pattern rounds). This is the per-COLUMN counterpart to
# ADDITIVE_NON_LABEL_EVENT_TABLES below (per-TABLE); see
# _compare_or_resume_table's additive_only_columns parameter.
ADDITIVE_CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    # C1: demographics/outcome/socioeconomic
    "insurance",
    "marital_status",
    "language",
    "length_of_stay_days",
    "readmission_30d_flag",
    # C2: ED triage/vitals-trend
    "triage_temperature",
    "triage_respiratory_rate",
    "triage_pain_score",
    "vitals_ed_reading_count",
    "vitals_ed_hr_stddev",
    "vitals_ed_hr_range",
    "vitals_ed_hr_trend_delta",
    "vitals_ed_sbp_stddev",
    "vitals_ed_sbp_range",
    "vitals_ed_spo2_stddev",
    "vitals_ed_spo2_min",
    # ICU module: stay linkage, vitals-trend, ventilation, vasopressor,
    # fluid balance, GCS at admission
    "icustay_id",
    "icu_intime",
    "icu_outtime",
    "icu_first_careunit",
    "icu_los_days",
    "icustay_candidate_count",
    "icustay_match_ambiguous",
    "icu_vitals_reading_count",
    "icu_vitals_hr_stddev",
    "icu_vitals_hr_trend_delta",
    "icu_vitals_sbp_stddev",
    "icu_vitals_spo2_min",
    "icu_ever_ventilated",
    "icu_ventilation_duration_hours",
    "icu_ever_vasopressor",
    "icu_vasopressor_duration_hours",
    "icu_fluid_balance_24h_ml",
    "icu_fluid_event_count",
    "icu_gcs_admission_total",
    "icu_gcs_component_count",
    # care-setting / service line (ICU-vs-ward exposure)
    "ever_icu_this_admission",
    "primary_service_this_admission",
    # microbiology
    "microbiology_specimen_count",
    "any_positive_culture",
    # CK-MB / blood-gas pH lab pivot
    "ck_mb_nearest",
    "ck_mb_reference_lower",
    "ck_mb_unit",
    "ck_mb_nearest_labevent_id",
    "ck_mb_count_6h",
    "blood_gas_ph_nearest",
    "blood_gas_ph_reference_lower",
    "blood_gas_ph_unit",
    "blood_gas_ph_nearest_labevent_id",
    "blood_gas_ph_count_6h",
    # medication (adenosine)
    "med_adenosine",
    "med_adenosine_home_reported",
    # MIMIC-Notes, both pattern rounds (11 original + 7 extended)
    "note_smoking_history_state",
    "note_alcohol_use_state",
    "note_illicit_drug_use_state",
    "note_family_history_cardiac_state",
    "note_heart_failure_mentioned_state",
    "note_arrhythmia_mentioned_state",
    "note_mi_or_acs_mentioned_state",
    "note_cardiomegaly_state",
    "note_pulmonary_edema_state",
    "note_pleural_effusion_state",
    "note_pneumothorax_state",
    "note_medication_nonadherence_mentioned_state",
    "note_acute_stroke_or_infarct_state",
    "note_intracranial_hemorrhage_state",
    "note_pulmonary_embolism_state",
    "note_aortic_dissection_state",
    "note_functional_status_impaired_state",
    "note_discharge_mental_status_impaired_state",
)


# clinical_context_features.parquet's medication-presence columns
# (med_<class> / med_<class>_home_reported, one pair per MEDICATION_PATTERNS
# group) go from NULL ("zero medication evidence rows at all for this
# admission") to an explicit FALSE ("at least one evidence row exists, this
# class just isn't among them") once ecg_medication_features.parquet gains
# ANY row for that admission -- see _clinical_context_query's unconditional
# `med.* EXCLUDE(subject_id, study_id)` in evidence.py, which relies on the
# LEFT JOIN producing an all-NULL row only when the medication pivot has no
# row at all for that (subject_id, study_id). Verified against the live
# project 2026-08-11: the canonical evidence on disk predates the
# "adenosine" medication class; every one of 133/800,035 affected rows has
# an adenosine-class row in the event-backed ecg_medication_windows.parquet
# with NO counterpart in the canonical version for that same admission --
# i.e. these are admissions whose ONLY recognized medication evidence is an
# adenosine administration, so recognizing that class for the first time
# flips their other class flags from unassessed to definitively-false. A
# real, expected consequence of the canonical evidence being intentionally
# frozen behind current code (see _has_bound_development_features), not
# pipeline nondeterminism.
NONDIAGNOSTIC_CONTEXT_MEDICATION_PRESENCE_COLUMNS: tuple[str, ...] = (
    "med_qt_prolonging_drug",
    "med_antiarrhythmic_class_i_iii",
    "med_beta_blocker",
    "med_calcium_channel_blocker",
    "med_digoxin",
    "med_diuretic",
    "med_vasopressor",
    "med_anticoagulant",
    "med_antiplatelet",
    "med_heparin",
    "med_nitrate",
    "med_electrolyte_replacement",
    "med_endocarditis_antibiotic_context",
    "med_qt_prolonging_drug_home_reported",
    "med_antiarrhythmic_class_i_iii_home_reported",
    "med_beta_blocker_home_reported",
    "med_calcium_channel_blocker_home_reported",
    "med_digoxin_home_reported",
    "med_diuretic_home_reported",
    "med_vasopressor_home_reported",
    "med_anticoagulant_home_reported",
    "med_antiplatelet_home_reported",
    "med_heparin_home_reported",
    "med_nitrate_home_reported",
    "med_electrolyte_replacement_home_reported",
    "med_endocarditis_antibiotic_context_home_reported",
)


# ecg_lab_features.parquet's CK-MB and blood-gas-pH columns ("Teil 3:
# CK-MB/Blutgas-pH Lab-Pivot"), added after the canonical evidence on disk
# was built. Verified against the live project 2026-08-11: purely additive
# (10 new columns, zero removed) with NO value differences in any of the 57
# pre-existing shared columns -- unlike clinical_context_features.parquet's
# medication-presence columns above, this table needed no accompanying
# nondiagnostic_tie_columns entry.
ADDITIVE_LAB_FEATURE_COLUMNS: tuple[str, ...] = (
    "ck_mb_nearest",
    "ck_mb_reference_lower",
    "ck_mb_unit",
    "ck_mb_nearest_labevent_id",
    "ck_mb_count_6h",
    "blood_gas_ph_nearest",
    "blood_gas_ph_reference_lower",
    "blood_gas_ph_unit",
    "blood_gas_ph_nearest_labevent_id",
    "blood_gas_ph_count_6h",
)


# encounter_microbiology_flags.parquet's specimen-count/positive-culture
# columns, added after the canonical evidence on disk was built. Verified
# against the live project 2026-08-11: purely additive (2 new columns, zero
# removed) with no value differences in the 3 pre-existing shared columns.
ADDITIVE_MICROBIOLOGY_FLAG_COLUMNS: tuple[str, ...] = (
    "microbiology_specimen_count",
    "any_positive_culture",
)


# Per-table additive-only-column and nondiagnostic-tie-column overrides for
# _compare_or_resume_table, consulted by both the comparison loop in
# attest_clinical_event_evidence_equivalence and the re-verification
# exception check in verify_clinical_event_evidence_attestation, so the two
# never drift out of sync with each other.
ADDITIVE_ONLY_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "clinical_context_features.parquet": ADDITIVE_CONTEXT_FEATURE_COLUMNS,
    "ecg_lab_features.parquet": ADDITIVE_LAB_FEATURE_COLUMNS,
    "encounter_microbiology_flags.parquet": (
        ADDITIVE_MICROBIOLOGY_FLAG_COLUMNS
    ),
}
# Columns produced by a parallel floating-point reduction -- standard
# deviations over ED and ICU vital-sign series. IEEE-754 addition is not
# associative and DuckDB splits the reduction across threads, so the summation
# order follows the thread schedule rather than the data: two runs over
# byte-identical input agree to within a few ULP and not bit for bit. Measured
# against the live project on 2026-08-16, across 800,035 ECGs the largest
# relative disagreement was 1.05e-15 -- three ULP -- with zero
# NULL-versus-value cases and no other column of those tables differing at all.
#
# Comparing such a column bit for bit therefore tests the scheduler, not the
# evidence. They are compared at single precision instead, which keeps about
# seven significant digits: magnitude-independent, unlike a fixed number of
# decimal places, and far tighter than any change in the underlying vitals
# could hide behind. The narrowing is declared per column and audited, so it
# cannot quietly spread to a column where an exact comparison is meaningful.
FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "clinical_context_features.parquet": (
        "icu_vitals_hr_stddev",
        "icu_vitals_sbp_stddev",
        "vitals_ed_hr_stddev",
        "vitals_ed_sbp_stddev",
        "vitals_ed_spo2_stddev",
    ),
    "ed_vitals_trend.parquet": (
        "vitals_ed_hr_stddev",
        "vitals_ed_sbp_stddev",
        "vitals_ed_spo2_stddev",
    ),
    "icu_vitals_trend.parquet": (
        "icu_vitals_hr_stddev",
        "icu_vitals_sbp_stddev",
    ),
}


NONDIAGNOSTIC_TIE_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "clinical_context_features.parquet": (
        NONDIAGNOSTIC_CONTEXT_MEDICATION_PRESENCE_COLUMNS
    ),
}


# ecg_medication_features.parquet (GROUP BY subject_id, study_id over
# ecg_medication_windows.parquet -- one row per admission with ANY
# recognized medication evidence) gains both new pivot columns AND
# genuinely new rows for the same reason as
# NONDIAGNOSTIC_CONTEXT_MEDICATION_PRESENCE_COLUMNS above: canonical
# predates the "adenosine" medication class. An admission whose only
# recognized medication evidence is an adenosine administration has zero
# rows in canonical's windows table (no class was ever recognized) and
# therefore no row at all in canonical's pivot, but gains exactly one row
# in the event-backed pivot. Handled by _compare_or_resume_medication_pivot
# (key-based, not row-count-equal like _compare_or_resume_table) rather
# than the additive-columns dicts above.
ADDITIVE_MEDICATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "med_adenosine",
    "med_adenosine_home_reported",
)
NEW_MEDICATION_CLASS = "adenosine"


# Plain (no tie-audit) tables that gain entirely new rows once
# NEW_MEDICATION_CLASS is recognized, handled by
# _compare_or_resume_table_excluding_new_class_rows: table_name ->
# class_column. medication_evidence.parquet is the raw standardized
# medication-event table underneath ecg_medication_windows.parquet;
# verified against the live project 2026-08-11 (3,929 new rows, all
# medication_class='adenosine', canonical has zero, row-count delta
# matches exactly).
NEW_CLASS_ROW_TABLES: dict[str, str] = {
    "medication_evidence.parquet": "medication_class",
}


ADDITIVE_NON_LABEL_EVENT_TABLES = {
    "medication_order_administration_context.parquet": (
        "Audit-only POE/EMAR order-administration context; it is not consumed "
        "by clinical label rules or ECG medication windows."
    ),
    # The following nine tables were added by the ICU-module, ED-vitals-trend
    # and demographics/outcome context-association work (see project docs
    # for "C1/C2" and the ICU integration). All are confirmed (grepped, not
    # assumed) to have zero references in label_registry.py -- they feed
    # only association_domains.py context/covariate targets or
    # therapy_response.py's medication-coupling biomarker, never a
    # pathology label's ground truth -- so their addition to the evidence
    # schema is safe and additive, the same category as the entry above.
    "admission_readmission.parquet": (
        "30-day readmission timing per admission, derived from "
        "admissions.parquet alone; a context/outcome covariate target, "
        "not a diagnostic label."
    ),
    "ecg_medication_exposure_pairs.parquet": (
        "ECG pairs found before/after AV-nodal-blocking medication "
        "exposure; consumed only by therapy_response.py's medication-"
        "coupling biomarker computation, not by any label rule."
    ),
    "ed_vitals_trend.parquet": (
        "Serial ED vital-sign variability (HR/SBP/SpO2 stddev, range, "
        "trend) over the stay; an association-domain covariate, not a "
        "diagnostic label."
    ),
    "encounter_care_setting_flags.parquet": (
        "ICU-vs-ward exposure and admitting service line per admission, "
        "from transfers.parquet/services.parquet; an association-domain "
        "covariate, not a diagnostic label."
    ),
    "icu_fluid_balance_24h.parquet": (
        "24h ICU intake/output fluid balance from inputevents/"
        "outputevents; an association-domain covariate, not a "
        "diagnostic label."
    ),
    "icu_gcs_admission.parquet": (
        "Nearest-to-admission Glasgow Coma Scale components from "
        "chartevents; an association-domain covariate, not a diagnostic "
        "label."
    ),
    "icu_vasopressor_exposure.parquet": (
        "ICU vasopressor exposure flag and duration from inputevents; "
        "an association-domain covariate, not a diagnostic label."
    ),
    "icu_ventilation_status.parquet": (
        "ICU ventilation flag and duration from procedureevents; an "
        "association-domain covariate, not a diagnostic label."
    ),
    "icu_vitals_trend.parquet": (
        "Serial ICU vital-sign variability over the first 24h from "
        "chartevents; an association-domain covariate, not a "
        "diagnostic label."
    ),
}


class _ShadowEvidenceConfig:
    def __init__(
        self,
        base: MimicProjectConfig,
        root: Path,
    ) -> None:
        self._base = base
        self._root = root

    @property
    def evidence_root(self) -> Path:
        return self._root / "evidence"

    @property
    def labels_root(self) -> Path:
        return self._root / "labels"

    @property
    def features_root(self) -> Path:
        return self._root / "features"

    @property
    def quarantine_root(self) -> Path:
        # Every root the evidence build *writes* to has to be redirected, not
        # just the ones its results are read from. Without this the shadow
        # rebuild overwrites the canonical quarantine artifacts, and since the
        # canonical checkpoint records their fingerprints, the next run finds
        # its own completed work invalidated by the attestation that was
        # supposed to leave it untouched.
        return self._root / "quarantine"

    @property
    def catalog_path(self) -> Path:
        return self._root / "catalog.duckdb"

    def to_dict(self) -> dict:
        return self._base.to_dict()

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def attest_clinical_event_evidence_equivalence(
    config: MimicProjectConfig,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    registry = load_clinical_event_registry(config)
    if registry is None:
        raise RuntimeError(
            "Clinical event equivalence requires a complete, verified "
            "clinical-event registry."
        )
    canonical_manifest_path = (
        config.evidence_root / "evidence.manifest.json"
    )
    if not canonical_manifest_path.is_file():
        raise FileNotFoundError(canonical_manifest_path)
    canonical_manifest = StageManifest.load(canonical_manifest_path)
    canonical_outputs = tuple(
        Path(path) for path in canonical_manifest.output_paths
    )
    missing = [str(path) for path in canonical_outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Canonical evidence outputs are incomplete: "
            + ", ".join(missing)
        )
    registry_path = Path(str(registry["_registry_path"]))
    machine_report_source = (
        config.evidence_root / "machine_report_labels.parquet"
    )
    if not machine_report_source.is_file():
        raise FileNotFoundError(machine_report_source)
    note_encounter_flags_source = (
        config.evidence_root / "note_encounter_flags.parquet"
    )
    if not note_encounter_flags_source.is_file():
        raise FileNotFoundError(note_encounter_flags_source)
    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("evidence.py"),
        Path(__file__).with_name("event_ingest.py"),
        Path(__file__).with_name("mappings.py"),
        Path(__file__).with_name("label_registry.py"),
    )
    input_payload = {
        "attestation_version": EVENT_EVIDENCE_ATTESTATION_VERSION,
        "clinical_event_registry": file_content_fingerprint(
            registry_path
        ),
        "clinical_event_registry_input_hash": registry.get("input_hash"),
        "canonical_evidence_manifest": file_content_fingerprint(
            canonical_manifest_path
        ),
        "machine_report_labels": file_content_fingerprint(
            machine_report_source
        ),
        "note_encounter_flags": file_content_fingerprint(
            note_encounter_flags_source
        ),
        "canonical_outputs": [
            file_fingerprint(path) for path in canonical_outputs
        ],
        "implementation_files": [
            file_content_fingerprint(path)
            for path in implementation_files
        ],
    }
    input_hash = content_hash(input_payload)
    attestation_path = (
        config.work_root
        / "provenance"
        / "clinical_event_evidence_equivalence.json"
    )
    if attestation_path.is_file():
        try:
            existing = verify_clinical_event_evidence_attestation(
                config
            )
        except RuntimeError:
            existing = None
        if existing is not None and existing.get("input_hash") == input_hash:
            return existing

    shadow_root = (
        config.work_root
        / "provenance"
        / "event_evidence_versions"
        / f"event-backed-{input_hash[:20]}"
    )
    shadow_config = _ShadowEvidenceConfig(config, shadow_root)
    shadow_config.evidence_root.mkdir(parents=True, exist_ok=True)
    shadow_config.labels_root.mkdir(parents=True, exist_ok=True)
    _copy_content_addressed(
        machine_report_source,
        shadow_config.evidence_root
        / "machine_report_labels.parquet",
    )
    _copy_content_addressed(
        note_encounter_flags_source,
        shadow_config.evidence_root
        / "note_encounter_flags.parquet",
    )
    if progress:
        progress(
            "Event-Evidenz 1/3: standardisierte Schattenversion "
            "reproduzierbar aufbauen"
        )
    shadow_manifest = build_clinical_evidence(
        shadow_config,
        force=False,
        progress=progress,
    )
    shadow_manifest_path = (
        shadow_config.evidence_root / "evidence.manifest.json"
    )

    canonical_parquets = _manifest_parquets(canonical_manifest)
    shadow_parquets = _manifest_parquets(shadow_manifest)
    table_names, additive_table_names = _classify_evidence_table_scope(
        canonical_parquets,
        shadow_parquets,
    )
    checkpoint_path = shadow_root / "equivalence.checkpoint.json"
    checkpoint = _load_equivalence_checkpoint(
        checkpoint_path,
        input_hash=input_hash,
    )
    comparisons = []
    additive_tables = []
    catalog = DuckDBCatalog(
        shadow_config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    with catalog.connect(isolated=True) as connection:
        for table_name in additive_table_names:
            event_backed_path = shadow_parquets[table_name]
            event_sql = _sql_path(event_backed_path)
            additive_tables.append(
                {
                    "table_name": table_name,
                    "role": "non_label_audit_context",
                    "purpose": ADDITIVE_NON_LABEL_EVENT_TABLES[table_name],
                    "event_backed_file": file_content_fingerprint(
                        event_backed_path
                    ),
                    "event_backed_row_count": int(
                        connection.execute(
                            "SELECT count(*) FROM "
                            f"read_parquet('{event_sql}')"
                        ).fetchone()[0]
                    ),
                }
            )
        for table_index, table_name in enumerate(table_names, start=1):
            canonical_path = canonical_parquets[table_name]
            shadow_path = shadow_parquets[table_name]
            if table_name == "ecg_medication_windows.parquet":
                comparison, resumed = (
                    _compare_or_resume_medication_windows_with_new_class(
                        connection,
                        checkpoint_path=checkpoint_path,
                        checkpoint=checkpoint,
                        canonical_path=canonical_path,
                        event_backed_path=shadow_path,
                        canonical_medication_path=canonical_parquets[
                            "medication_evidence.parquet"
                        ],
                        event_backed_medication_path=shadow_parquets[
                            "medication_evidence.parquet"
                        ],
                        master_path=(
                            config.master_root
                            / "ecg_master_index.parquet"
                        ),
                        new_medication_class=NEW_MEDICATION_CLASS,
                    )
                )
            elif table_name == "ecg_medication_features.parquet":
                comparison, resumed = _compare_or_resume_medication_pivot(
                    connection,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    table_name=table_name,
                    canonical_path=canonical_path,
                    event_backed_path=shadow_path,
                    canonical_windows_path=canonical_parquets[
                        "ecg_medication_windows.parquet"
                    ],
                    event_backed_windows_path=shadow_parquets[
                        "ecg_medication_windows.parquet"
                    ],
                    additive_only_columns=(
                        ADDITIVE_MEDICATION_FEATURE_COLUMNS
                    ),
                    new_medication_class=NEW_MEDICATION_CLASS,
                )
            elif table_name in NEW_CLASS_ROW_TABLES:
                comparison, resumed = (
                    _compare_or_resume_table_excluding_new_class_rows(
                        connection,
                        checkpoint_path=checkpoint_path,
                        checkpoint=checkpoint,
                        table_name=table_name,
                        canonical_path=canonical_path,
                        event_backed_path=shadow_path,
                        new_medication_class=NEW_MEDICATION_CLASS,
                        class_column=NEW_CLASS_ROW_TABLES[table_name],
                    )
                )
            else:
                # One call for every remaining table: each declaration
                # defaults to empty, so a table with no exceptions gets the
                # strict full comparison and a table with one cannot be
                # reached through a branch that forgets to pass it.
                comparison, resumed = _compare_or_resume_table(
                    connection,
                    checkpoint_path=checkpoint_path,
                    checkpoint=checkpoint,
                    table_name=table_name,
                    canonical_path=canonical_path,
                    event_backed_path=shadow_path,
                    additive_only_columns=(
                        ADDITIVE_ONLY_COLUMNS_BY_TABLE.get(table_name, ())
                    ),
                    nondiagnostic_tie_columns=(
                        NONDIAGNOSTIC_TIE_COLUMNS_BY_TABLE.get(
                            table_name, ()
                        )
                    ),
                    floating_point_reduction_columns=(
                        FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE.get(
                            table_name, ()
                        )
                    ),
                )
            comparisons.append(comparison)
            if progress:
                progress(
                    f"Event-Evidenz 2/3 {table_index:,}/"
                    f"{len(table_names):,}: {table_name} "
                    f"{'fortgesetzt' if resumed else 'exakt verglichen'}"
                )
            if not comparison["equivalent"]:
                raise RuntimeError(
                    "Event-backed evidence differs from the canonical "
                    f"evidence table {table_name}: {comparison}"
                )

    canonical_definitions = _manifest_json(
        canonical_manifest,
        "label_definitions.json",
    )
    shadow_definitions = _manifest_json(
        shadow_manifest,
        "label_definitions.json",
    )
    definitions_equal = (
        _normalized_definition_hash(canonical_definitions)
        == _normalized_definition_hash(shadow_definitions)
    )
    if not definitions_equal:
        raise RuntimeError(
            "Event-backed label definitions differ from the canonical "
            "definitions."
        )
    has_nondiagnostic_tie_audit = any(
        not comparison.get("full_table_equivalent", False)
        for comparison in comparisons
    )
    payload = {
        "attestation_version": EVENT_EVIDENCE_ATTESTATION_VERSION,
        "status": (
            EXACT_LABEL_EVENT_EVIDENCE_STATUS
            if has_nondiagnostic_tie_audit
            else EXACT_EVENT_EVIDENCE_STATUS
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": input_hash,
        "input_payload": input_payload,
        "clinical_event_registry_input_hash": registry.get("input_hash"),
        "clinical_event_registry": file_content_fingerprint(
            registry_path
        ),
        "canonical_evidence_manifest": file_content_fingerprint(
            canonical_manifest_path
        ),
        "event_backed_evidence_manifest": file_content_fingerprint(
            shadow_manifest_path
        ),
        "canonical_stage_version": canonical_manifest.stage_version,
        "event_backed_stage_version": shadow_manifest.stage_version,
        "label_definitions_equal": True,
        "equivalence_scope": (
            "Every canonical label-relevant evidence column is present and "
            "compared exactly in both directions with EXCEPT ALL. The sole "
            "permitted legacy tie field is audited against its source "
            "candidates. Explicitly allowlisted event-only tables are "
            "non-label audit context and are recorded separately."
        ),
        "canonical_table_set_complete": True,
        "additive_non_label_event_tables": additive_tables,
        "canonical_label_definitions": file_content_fingerprint(
            canonical_definitions
        ),
        "event_backed_label_definitions": file_content_fingerprint(
            shadow_definitions
        ),
        "label_definitions_normalized_hash": (
            _normalized_definition_hash(canonical_definitions)
        ),
        "table_comparisons": comparisons,
        "locked_test_associations_read": False,
        "purpose": (
            "Pipeline migration equivalence only; no endpoint association "
            "or locked-test performance was evaluated."
        ),
    }
    payload["attestation_hash"] = content_hash(payload)
    atomic_write_json(attestation_path, payload)
    if progress:
        progress(
            "Event-Evidenz 3/3: exakte Migration attestiert"
        )
    return verify_clinical_event_evidence_attestation(config)


DIRECT_EVENT_EVIDENCE_STATUS = "not_required_direct_event_backed"


def confirm_or_attest_clinical_event_evidence(
    config: MimicProjectConfig,
    *,
    full_equivalence: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Production entry of the attest-event-evidence stage (audit E9).

    The shadow rebuild is a migration check: it proves that evidence built
    from the raw CSV path equals evidence built from the event layer. When
    the canonical evidence was itself built from the verified event layer
    with the registry's current input hash, every consumer (acceptance,
    group VCG, locking, phenome, validation) already binds to
    evidence_provenance.json instead of the attestation, and the rebuild
    only repeated the same computation (2.45 h in A131). It then runs only
    on request, as a release check after changes to the event layer.
    """
    if not full_equivalence:
        provenance = _direct_event_evidence_provenance(config)
        if provenance is not None:
            manifest_path = config.evidence_root / "evidence.manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = StageManifest.load(manifest_path)
            missing = [
                str(path)
                for path in manifest.output_paths
                if not Path(path).is_file()
            ]
            if manifest.status != "complete" or missing:
                raise RuntimeError(
                    "Canonical evidence is incomplete: "
                    + (", ".join(missing) or f"status {manifest.status}")
                )
            if progress:
                progress(
                    "Evidenz stammt direkt aus der geprüften Ereignisschicht; "
                    "keine Migrationsprüfung nötig"
                )
            return {
                "status": DIRECT_EVENT_EVIDENCE_STATUS,
                "clinical_event_registry_input_hash": provenance.get(
                    "clinical_event_registry_input_hash"
                ),
                "evidence_provenance": provenance.get("evidence_provenance"),
                "canonical_evidence_manifest": file_content_fingerprint(
                    manifest_path
                ),
                "shadow_rebuild": False,
            }
    return attest_clinical_event_evidence_equivalence(
        config,
        progress=progress,
    )


def verify_clinical_event_evidence_provenance(
    config: MimicProjectConfig,
) -> dict:
    direct = _direct_event_evidence_provenance(config)
    if direct is not None:
        return direct
    return verify_clinical_event_evidence_attestation(config)


def _direct_event_evidence_provenance(
    config: MimicProjectConfig,
) -> Optional[dict]:
    """The direct_event_backed record, or None when evidence is not."""
    registry = load_clinical_event_registry(config)
    if registry is None:
        raise RuntimeError(
            "Formal analysis requires the standardized clinical-event layer."
        )
    provenance_path = config.evidence_root / "evidence_provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Evidence provenance is unreadable: {provenance_path}"
            ) from exc
        event_record = provenance.get("clinical_event_registry") or {}
        if (
            provenance.get("clinical_event_source_policy")
            == "verified_partitioned_parquet"
            and event_record.get("input_hash") == registry.get("input_hash")
        ):
            return {
                "status": "direct_event_backed",
                "clinical_event_registry_input_hash": registry.get(
                    "input_hash"
                ),
                "evidence_provenance": file_content_fingerprint(
                    provenance_path
                ),
            }
    return None


def verify_clinical_event_evidence_attestation(
    config: MimicProjectConfig,
) -> dict:
    path = (
        config.work_root
        / "provenance"
        / "clinical_event_evidence_equivalence.json"
    )
    if not path.is_file():
        raise RuntimeError(
            "Clinical-event evidence equivalence attestation is missing."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Clinical-event evidence attestation is unreadable: {path}"
        ) from exc
    stored_hash = payload.get("attestation_hash")
    hash_payload = dict(payload)
    hash_payload.pop("attestation_hash", None)
    if (
        payload.get("attestation_version")
        != EVENT_EVIDENCE_ATTESTATION_VERSION
        or payload.get("status") not in ATTESTED_EVENT_EVIDENCE_STATUSES
        or stored_hash != content_hash(hash_payload)
    ):
        raise RuntimeError(
            "Clinical-event evidence attestation is invalid."
        )
    if content_hash(payload.get("input_payload")) != payload.get(
        "input_hash"
    ):
        raise RuntimeError(
            "Clinical-event evidence attestation input hash is invalid."
        )
    registry = load_clinical_event_registry(config)
    if (
        registry is None
        or payload.get("clinical_event_registry_input_hash")
        != registry.get("input_hash")
    ):
        raise RuntimeError(
            "Clinical-event evidence attestation is stale."
        )
    fingerprints = [
        payload.get("clinical_event_registry"),
        payload.get("canonical_evidence_manifest"),
        payload.get("event_backed_evidence_manifest"),
        payload.get("canonical_label_definitions"),
        payload.get("event_backed_label_definitions"),
        payload.get("input_payload", {}).get("machine_report_labels"),
    ]
    fingerprints.extend(
        payload.get("input_payload", {}).get(
            "implementation_files",
            (),
        )
    )
    medication_tie_audited_comparisons = 0
    additive_column_audited_comparisons = 0
    for comparison in payload.get("table_comparisons", ()):
        if not comparison.get("equivalent"):
            raise RuntimeError(
                "Clinical-event evidence attestation contains a mismatch."
            )
        if not comparison.get("full_table_equivalent", False):
            table_name = comparison.get("table_name")
            if table_name == "ecg_medication_windows.parquet":
                tie_audit = comparison.get("nondiagnostic_tie_audit") or {}
                excluded_class = comparison.get(
                    "excluded_new_medication_class"
                )
                if (
                    comparison.get("nondiagnostic_tie_columns")
                    != ["nearest_medication"]
                    or tie_audit.get("valid") is not True
                    or int(tie_audit.get("differing_rows", 0)) <= 0
                    or (
                        excluded_class is not None
                        and excluded_class != NEW_MEDICATION_CLASS
                    )
                    or (
                        excluded_class is not None
                        and int(
                            comparison.get(
                                "excluded_new_medication_class_row_count",
                                -1,
                            )
                        )
                        <= 0
                    )
                ):
                    raise RuntimeError(
                        "Clinical-event evidence attestation contains an "
                        "unaudited non-exact field."
                    )
                medication_tie_audited_comparisons += 1
            elif table_name == "ecg_medication_features.parquet":
                policy = comparison.get("comparison_policy", {})
                additive_columns = policy.get("additive_only_columns") or []
                new_class = policy.get("new_medication_class")
                if (
                    sorted(additive_columns)
                    != sorted(ADDITIVE_MEDICATION_FEATURE_COLUMNS)
                    or new_class != NEW_MEDICATION_CLASS
                    or int(
                        comparison.get(
                            "event_backed_only_key_unexplained_count", -1
                        )
                    )
                    != 0
                    or int(comparison.get("canonical_only_key_count", -1))
                    != 0
                    or int(comparison.get("matched_row_mismatch_count", -1))
                    != 0
                ):
                    raise RuntimeError(
                        "Clinical-event evidence attestation contains an "
                        "unaudited non-exact field."
                    )
                additive_column_audited_comparisons += 1
            elif table_name in NEW_CLASS_ROW_TABLES:
                policy = comparison.get("comparison_policy", {})
                new_class = policy.get("new_medication_class")
                class_column = policy.get("class_column")
                if (
                    new_class != NEW_MEDICATION_CLASS
                    or class_column != NEW_CLASS_ROW_TABLES[table_name]
                    or int(
                        comparison.get(
                            "excluded_new_medication_class_row_count", -1
                        )
                    )
                    <= 0
                ):
                    raise RuntimeError(
                        "Clinical-event evidence attestation contains an "
                        "unaudited non-exact field."
                    )
                additive_column_audited_comparisons += 1
            elif (
                table_name in ADDITIVE_ONLY_COLUMNS_BY_TABLE
                or table_name in FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE
            ):
                policy = comparison.get("comparison_policy", {})
                additive_columns = policy.get("additive_only_columns") or []
                tie_columns = policy.get("nondiagnostic_tie_columns") or []
                reduction_columns = (
                    policy.get("floating_point_reduction_columns") or []
                )
                expected_additive = ADDITIVE_ONLY_COLUMNS_BY_TABLE.get(
                    table_name, ()
                )
                expected_tie = NONDIAGNOSTIC_TIE_COLUMNS_BY_TABLE.get(
                    table_name, ()
                )
                expected_reduction = (
                    FLOATING_POINT_REDUCTION_COLUMNS_BY_TABLE.get(
                        table_name, ()
                    )
                )
                if (
                    sorted(additive_columns) != sorted(expected_additive)
                    or sorted(tie_columns) != sorted(expected_tie)
                    or sorted(reduction_columns)
                    != sorted(expected_reduction)
                ):
                    raise RuntimeError(
                        "Clinical-event evidence attestation contains an "
                        "unaudited non-exact field."
                    )
                additive_column_audited_comparisons += 1
            else:
                raise RuntimeError(
                    "Clinical-event evidence attestation contains an "
                    "unaudited non-exact field."
                )
        fingerprints.extend(
            (
                comparison.get("canonical_file"),
                comparison.get("event_backed_file"),
            )
        )
    additive_names = set()
    for additive in payload.get("additive_non_label_event_tables", ()):
        if (
            not isinstance(additive, dict)
            or additive.get("role") != "non_label_audit_context"
            or additive.get("table_name")
            not in ADDITIVE_NON_LABEL_EVENT_TABLES
            or additive.get("purpose")
            != ADDITIVE_NON_LABEL_EVENT_TABLES.get(
                additive.get("table_name")
            )
            or int(additive.get("event_backed_row_count", -1)) < 0
        ):
            raise RuntimeError(
                "Clinical-event evidence attestation contains an unsafe "
                "additive table."
            )
        table_name = str(additive["table_name"])
        if table_name in additive_names:
            raise RuntimeError(
                "Clinical-event evidence attestation repeats an additive "
                "table."
            )
        additive_names.add(table_name)
        fingerprints.append(additive.get("event_backed_file"))
    tie_audited_comparisons = (
        medication_tie_audited_comparisons
        + additive_column_audited_comparisons
    )
    if (
        payload.get("status") == EXACT_EVENT_EVIDENCE_STATUS
        and tie_audited_comparisons
    ) or (
        payload.get("status") == EXACT_LABEL_EVENT_EVIDENCE_STATUS
        and tie_audited_comparisons == 0
    ):
        raise RuntimeError(
            "Clinical-event evidence attestation status does not match its "
            "tie audit."
        )
    for fingerprint in fingerprints:
        _verify_content_fingerprint(fingerprint)
    if (
        payload.get("label_definitions_equal") is not True
        or payload.get("canonical_table_set_complete") is not True
        or payload.get("locked_test_associations_read") is not False
    ):
        raise RuntimeError(
            "Clinical-event evidence attestation has unsafe semantics."
        )
    return payload


def _effective_additive_only_columns(
    declared: tuple[str, ...],
    *,
    table_name: str,
    canonical_schema_names: set[str],
    event_schema_names: set[str],
) -> tuple[str, ...]:
    """Narrow a declared additive-only set to what is still additive.

    An additive-only declaration records that the canonical evidence *on disk*
    predates a column. That is a statement about a moment in time, and it stops
    being true the moment the canonical evidence is rebuilt with the newer
    code. Keeping the column excluded after both sides carry it would turn a
    stale note into a permanent blind spot -- the column would never be
    compared again.

    So the declaration is narrowed to the columns canonical still lacks.
    Everything both sides now have falls back to ordinary strict comparison:
    the check tightens as the canonical evidence catches up, and can never
    loosen. The half of the original guarantee that stays unconditional is the
    other direction -- a declared column missing from the event-backed rebuild
    means the rebuild lost a column, which is a real regression and still an
    error.
    """

    declared_names = {str(name) for name in declared}
    if not declared_names:
        return ()
    missing = declared_names - event_schema_names
    if missing:
        raise RuntimeError(
            f"Declared additive-only columns are missing from the "
            f"event-backed {table_name}: {sorted(missing)}"
        )
    return tuple(sorted(declared_names - canonical_schema_names))


def _new_medication_class_is_still_new(
    connection,
    *,
    canonical_path: Path,
    class_column: str,
    new_medication_class: str,
) -> bool:
    """Whether the canonical evidence still predates a medication class.

    Same shape as _effective_additive_only_columns, and for the same reason.
    The tolerance records that canonical was built before the class was
    recognised, so the event-backed side carries rows canonical cannot have.
    Rebuilding canonical ends that state: both sides then recognise the class,
    the rows are not extra any more, and the tables can be compared strictly.

    That is the outcome the tolerance was always waiting for, not a violation
    of it -- so it retires itself instead of failing the run, and the
    comparison it hands back is the stricter one.
    """

    canonical_sql = _sql_path(canonical_path)
    canonical_class_rows = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{canonical_sql}') "
            f"WHERE {class_column} = ?",
            [new_medication_class],
        ).fetchone()[0]
    )
    return canonical_class_rows == 0


def _comparison_projection(
    schema_names: list[str],
    *,
    single_precision_columns: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> str:
    """Column list for a set comparison, narrowing declared reductions."""

    single = set(single_precision_columns)
    skipped = set(exclude)
    return ", ".join(
        (
            f"CAST({_quote_identifier(name)} AS REAL) "
            f"AS {_quote_identifier(name)}"
            if name in single
            else _quote_identifier(name)
        )
        for name in schema_names
        if name not in skipped
    )


def _compare_or_resume_table(
    connection,
    *,
    checkpoint_path: Path,
    checkpoint: dict,
    table_name: str,
    canonical_path: Path,
    event_backed_path: Path,
    nondiagnostic_tie_columns: tuple[str, ...] = (),
    additive_only_columns: tuple[str, ...] = (),
    floating_point_reduction_columns: tuple[str, ...] = (),
) -> tuple[dict, bool]:
    canonical_file = file_content_fingerprint(canonical_path)
    event_file = file_content_fingerprint(event_backed_path)
    comparison_policy = {
        "difference_count_method": "forward_except_all_reverse_derived_equal_cardinality",
        "mode": (
            "exact_label_relevant_with_tie_audit"
            if nondiagnostic_tie_columns
            else "exact_with_additive_columns"
            if additive_only_columns
            else "exact_full_table_single_precision_reductions"
            if floating_point_reduction_columns
            else "exact_full_table"
        ),
        "nondiagnostic_tie_columns": list(
            nondiagnostic_tie_columns
        ),
        "additive_only_columns": list(additive_only_columns),
        "floating_point_reduction_columns": list(
            floating_point_reduction_columns
        ),
    }
    previous = checkpoint.get("tables", {}).get(table_name)
    if (
        previous is not None
        and previous.get("canonical_file") == canonical_file
        and previous.get("event_backed_file") == event_file
        and previous.get("comparison_policy") == comparison_policy
    ):
        return previous, True

    canonical_sql = _sql_path(canonical_path)
    event_sql = _sql_path(event_backed_path)
    canonical_schema = [
        list(row)
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{canonical_sql}')"
        ).fetchall()
    ]
    event_schema = [
        list(row)
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{event_sql}')"
        ).fetchall()
    ]
    # additive_only_columns declares column NAMES permitted to exist ONLY in
    # the event-backed schema (a purely additive change, e.g. a new context
    # association domain). Unlike nondiagnostic_tie_columns below (same
    # columns on both sides, some values ignored), an additive column is
    # absent from canonical entirely -- until canonical is rebuilt, at which
    # point it is an ordinary shared column again. See
    # _effective_additive_only_columns.
    canonical_schema_names = {str(row[0]) for row in canonical_schema}
    event_schema_names = {str(row[0]) for row in event_schema}
    effective_additive_columns = _effective_additive_only_columns(
        additive_only_columns,
        table_name=table_name,
        canonical_schema_names=canonical_schema_names,
        event_schema_names=event_schema_names,
    )
    missing_reduction_columns = (
        set(floating_point_reduction_columns) - canonical_schema_names
    )
    if missing_reduction_columns:
        raise RuntimeError(
            f"Declared floating-point reduction columns are missing from "
            f"{table_name}: {sorted(missing_reduction_columns)}"
        )
    event_schema_shared = [
        row
        for row in event_schema
        if str(row[0]) not in effective_additive_columns
    ]
    schema_equal = canonical_schema == event_schema_shared
    canonical_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{canonical_sql}')"
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}')"
        ).fetchone()[0]
    )
    canonical_only = event_only = -1
    label_relevant_canonical_only = -1
    label_relevant_event_only = -1
    if schema_equal and canonical_count == event_count:
        if not effective_additive_columns:
            # A full-row comparison is only meaningful when both sides share
            # the exact same column set; skip it (leave -1) once additive
            # columns make the two schemas different widths by design.
            full_projection = _comparison_projection(
                [str(row[0]) for row in canonical_schema],
                single_precision_columns=floating_point_reduction_columns,
            )
            canonical_only = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    f"SELECT {full_projection} "
                    f"FROM read_parquet('{canonical_sql}') "
                    "EXCEPT ALL "
                    f"SELECT {full_projection} "
                    f"FROM read_parquet('{event_sql}')"
                    ") differences"
                ).fetchone()[0]
            )
            # Equal-cardinality multisets satisfy |A\B| = |B\A|.
            # SELECT here preserves rows (including duplicates and NULLs).
            # The equal-count guard above is essential to this derivation.
            event_only = canonical_only
        if nondiagnostic_tie_columns or effective_additive_columns:
            # Project onto canonical's own columns (which by construction
            # never include additive_only_columns) minus any columns
            # whose VALUES are separately known-and-audited to legitimately
            # differ. Covers three cases with one code path: tie columns
            # only (existing tables, e.g. ecg_medication_windows.parquet's
            # nearest_medication), additive columns only (a table that
            # merely grew new columns), or both together (a table that
            # both grew new columns AND has a known, audited value
            # difference in a pre-existing column caused by the same
            # underlying schema growth -- see
            # NONDIAGNOSTIC_CONTEXT_MEDICATION_PRESENCE_COLUMNS).
            schema_columns = [str(row[0]) for row in canonical_schema]
            missing_tie_columns = (
                set(nondiagnostic_tie_columns) - set(schema_columns)
            )
            if missing_tie_columns:
                raise RuntimeError(
                    f"Non-diagnostic tie columns are missing from "
                    f"{table_name}: {sorted(missing_tie_columns)}"
                )
            projection = _comparison_projection(
                schema_columns,
                single_precision_columns=floating_point_reduction_columns,
                exclude=nondiagnostic_tie_columns,
            )
            label_relevant_canonical_only = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    f"SELECT {projection} FROM "
                    f"read_parquet('{canonical_sql}') "
                    "EXCEPT ALL "
                    f"SELECT {projection} FROM "
                    f"read_parquet('{event_sql}')"
                    ") differences"
                ).fetchone()[0]
            )
            # Column projection changes values, not multiset cardinality.
            label_relevant_event_only = label_relevant_canonical_only
        else:
            label_relevant_canonical_only = canonical_only
            label_relevant_event_only = event_only
    # A table whose comparison narrowed any column is not "full table
    # equivalent", however clean the counts came back -- otherwise the
    # narrowing would skip the audit that keeps it declared and visible.
    full_table_equivalent = (
        schema_equal
        and canonical_count == event_count
        and canonical_only == 0
        and event_only == 0
        and not floating_point_reduction_columns
    )
    comparison = {
        "table_name": table_name,
        "comparison_policy": comparison_policy,
        "difference_count_method": (
            "forward_except_all_reverse_derived_equal_cardinality"
            if schema_equal and canonical_count == event_count
            else "not_computed_schema_or_cardinality_mismatch"
        ),
        "floating_point_reduction_columns": list(
            floating_point_reduction_columns
        ),
        "nondiagnostic_tie_columns": list(
            nondiagnostic_tie_columns
        ),
        "canonical_file": canonical_file,
        "event_backed_file": event_file,
        "schema_equal": schema_equal,
        "canonical_row_count": canonical_count,
        "event_backed_row_count": event_count,
        "canonical_only_row_count": canonical_only,
        "event_backed_only_row_count": event_only,
        "label_relevant_canonical_only_row_count": (
            label_relevant_canonical_only
        ),
        "label_relevant_event_backed_only_row_count": (
            label_relevant_event_only
        ),
        "full_table_equivalent": full_table_equivalent,
        "equivalent": (
            schema_equal
            and canonical_count == event_count
            and label_relevant_canonical_only == 0
            and label_relevant_event_only == 0
        ),
    }
    checkpoint.setdefault("tables", {})[table_name] = comparison
    atomic_write_json(checkpoint_path, checkpoint)
    return comparison, False


def _compare_or_resume_medication_pivot(
    connection,
    *,
    checkpoint_path: Path,
    checkpoint: dict,
    table_name: str,
    canonical_path: Path,
    event_backed_path: Path,
    canonical_windows_path: Path,
    event_backed_windows_path: Path,
    additive_only_columns: tuple[str, ...],
    new_medication_class: str,
    key_columns: tuple[str, ...] = ("subject_id", "study_id"),
) -> tuple[dict, bool]:
    """Compare a GROUP-BY medication pivot (one row per key that has ANY
    matching row in ecg_medication_windows.parquet) where the canonical
    evidence predates a newly recognized medication class. Unlike
    _compare_or_resume_table, row COUNTS may legitimately differ: an
    admission whose only recognized medication evidence is the new class
    has zero rows in canonical (no class was ever recognized for it) but
    exactly one row in the event-backed pivot. Every such event-only row
    is individually verified against ecg_medication_windows.parquet
    before being accepted as explained, rather than merely tolerated by
    row count.
    """
    canonical_file = file_content_fingerprint(canonical_path)
    event_file = file_content_fingerprint(event_backed_path)
    comparison_policy = {
        "mode": "exact_with_additive_columns_and_explained_rows",
        "additive_only_columns": list(additive_only_columns),
        "new_medication_class": new_medication_class,
        "key_columns": list(key_columns),
    }
    previous = checkpoint.get("tables", {}).get(table_name)
    if (
        previous is not None
        and previous.get("canonical_file") == canonical_file
        and previous.get("event_backed_file") == event_file
        and previous.get("comparison_policy") == comparison_policy
    ):
        return previous, True

    canonical_sql = _sql_path(canonical_path)
    event_sql = _sql_path(event_backed_path)
    canonical_windows_sql = _sql_path(canonical_windows_path)
    event_windows_sql = _sql_path(event_backed_windows_path)

    canonical_schema = [
        list(row)
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{canonical_sql}')"
        ).fetchall()
    ]
    event_schema = [
        list(row)
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{event_sql}')"
        ).fetchall()
    ]
    effective_additive_columns = _effective_additive_only_columns(
        additive_only_columns,
        table_name=table_name,
        canonical_schema_names={str(row[0]) for row in canonical_schema},
        event_schema_names={str(row[0]) for row in event_schema},
    )
    event_schema_shared = [
        row
        for row in event_schema
        if str(row[0]) not in effective_additive_columns
    ]
    schema_equal = canonical_schema == event_schema_shared

    canonical_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{canonical_sql}')"
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}')"
        ).fetchone()[0]
    )
    key_projection = ", ".join(_quote_identifier(c) for c in key_columns)
    canonical_only_key_count = int(
        connection.execute(
            "SELECT count(*) FROM ("
            f"SELECT {key_projection} FROM read_parquet('{canonical_sql}') "
            "EXCEPT "
            f"SELECT {key_projection} FROM read_parquet('{event_sql}')"
            ") differences"
        ).fetchone()[0]
    )
    event_only_key_count = int(
        connection.execute(
            "SELECT count(*) FROM ("
            f"SELECT {key_projection} FROM read_parquet('{event_sql}') "
            "EXCEPT "
            f"SELECT {key_projection} FROM read_parquet('{canonical_sql}')"
            ") differences"
        ).fetchone()[0]
    )
    event_only_key_unexplained_count = int(
        connection.execute(
            f"""
            WITH extra_keys AS (
                SELECT {key_projection} FROM read_parquet('{event_sql}')
                EXCEPT
                SELECT {key_projection} FROM read_parquet('{canonical_sql}')
            ),
            canonical_has_any_evidence AS (
                SELECT DISTINCT {key_projection}
                FROM read_parquet('{canonical_windows_sql}')
            ),
            event_has_new_class AS (
                SELECT DISTINCT {key_projection}
                FROM read_parquet('{event_windows_sql}')
                WHERE medication_class = '{new_medication_class}'
            )
            SELECT count(*) FROM extra_keys e
            LEFT JOIN canonical_has_any_evidence c USING ({key_projection})
            LEFT JOIN event_has_new_class n USING ({key_projection})
            WHERE c.{key_columns[0]} IS NOT NULL OR n.{key_columns[0]} IS NULL
            """
        ).fetchone()[0]
    )
    shared_columns = [str(row[0]) for row in canonical_schema]
    shared_projection = ", ".join(
        _quote_identifier(column) for column in shared_columns
    )
    matched_row_mismatch_count = int(
        connection.execute(
            "SELECT count(*) FROM ("
            f"SELECT {shared_projection} FROM read_parquet('{canonical_sql}') "
            "EXCEPT "
            f"SELECT {shared_projection} FROM read_parquet('{event_sql}')"
            ") differences"
        ).fetchone()[0]
    )
    equivalent = bool(
        schema_equal
        and canonical_only_key_count == 0
        and event_only_key_unexplained_count == 0
        and matched_row_mismatch_count == 0
    )
    comparison = {
        "table_name": table_name,
        "comparison_policy": comparison_policy,
        "canonical_file": canonical_file,
        "event_backed_file": event_file,
        "schema_equal": schema_equal,
        "canonical_row_count": canonical_count,
        "event_backed_row_count": event_count,
        "canonical_only_key_count": canonical_only_key_count,
        "event_backed_only_key_count": event_only_key_count,
        "event_backed_only_key_unexplained_count": (
            event_only_key_unexplained_count
        ),
        "matched_row_mismatch_count": matched_row_mismatch_count,
        "full_table_equivalent": bool(
            schema_equal
            and canonical_count == event_count
            and canonical_only_key_count == 0
            and event_only_key_count == 0
            and matched_row_mismatch_count == 0
        ),
        "equivalent": equivalent,
    }
    checkpoint.setdefault("tables", {})[table_name] = comparison
    atomic_write_json(checkpoint_path, checkpoint)
    return comparison, False


def _compare_or_resume_medication_windows(
    connection,
    *,
    checkpoint_path: Path,
    checkpoint: dict,
    canonical_path: Path,
    event_backed_path: Path,
    canonical_medication_path: Path,
    event_backed_medication_path: Path,
    master_path: Path,
) -> tuple[dict, bool]:
    table_name = "ecg_medication_windows.parquet"
    comparison, resumed = _compare_or_resume_table(
        connection,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        table_name=table_name,
        canonical_path=canonical_path,
        event_backed_path=event_backed_path,
        nondiagnostic_tie_columns=("nearest_medication",),
    )
    if resumed and (
        comparison.get("nondiagnostic_tie_audit") or {}
    ).get("valid") is True:
        return comparison, True
    tie_audit = _audit_nondiagnostic_medication_ties(
        connection,
        canonical_windows_path=canonical_path,
        event_backed_windows_path=event_backed_path,
        canonical_medication_path=canonical_medication_path,
        event_backed_medication_path=event_backed_medication_path,
        master_path=master_path,
    )
    tie_audit["valid"] = bool(
        tie_audit["canonical_choice_not_at_minimum"] == 0
        and tie_audit["event_backed_choice_not_at_minimum"] == 0
        and tie_audit["differing_rows"]
        == comparison["canonical_only_row_count"]
        == comparison["event_backed_only_row_count"]
    )
    comparison["nondiagnostic_tie_audit"] = tie_audit
    comparison["equivalent"] = bool(
        comparison["equivalent"] and tie_audit["valid"]
    )
    checkpoint.setdefault("tables", {})[table_name] = comparison
    atomic_write_json(checkpoint_path, checkpoint)
    return comparison, False


def _compare_or_resume_medication_windows_with_new_class(
    connection,
    *,
    checkpoint_path: Path,
    checkpoint: dict,
    canonical_path: Path,
    event_backed_path: Path,
    canonical_medication_path: Path,
    event_backed_medication_path: Path,
    master_path: Path,
    new_medication_class: str,
) -> tuple[dict, bool]:
    """ecg_medication_windows.parquet gains entirely new rows -- one per
    (subject_id, study_id) admission with ANY evidence of
    `new_medication_class` -- once the canonical evidence predates that
    class being recognized at all, on top of the pre-existing
    nearest_medication tie-breaking ambiguity
    _compare_or_resume_medication_windows already audits. Verifies the
    new-class rows exist ONLY on the event-backed side and fully explain
    the row-count delta, then delegates unchanged to the existing tie-audit
    machinery on a filtered copy with those rows excluded -- so a genuine
    future divergence in the overlapping rows is still caught exactly as
    before, and this tolerance can never silently widen to cover more than
    the one class it was built for.
    """
    table_name = "ecg_medication_windows.parquet"
    if not _new_medication_class_is_still_new(
        connection,
        canonical_path=canonical_path,
        class_column="medication_class",
        new_medication_class=new_medication_class,
    ):
        # Canonical recognises the class as well, so there are no extra rows
        # to set aside and the ordinary tie-audited comparison applies to the
        # whole table -- including the rows this wrapper used to exclude.
        return _compare_or_resume_medication_windows(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            canonical_path=canonical_path,
            event_backed_path=event_backed_path,
            canonical_medication_path=canonical_medication_path,
            event_backed_medication_path=event_backed_medication_path,
            master_path=master_path,
        )
    canonical_file = file_content_fingerprint(canonical_path)
    event_file = file_content_fingerprint(event_backed_path)
    comparison_policy = {
        "mode": "exact_label_relevant_with_tie_audit_and_new_class_rows",
        "nondiagnostic_tie_columns": ["nearest_medication"],
        "new_medication_class": new_medication_class,
    }
    previous = checkpoint.get("tables", {}).get(table_name)
    if (
        previous is not None
        and previous.get("canonical_file") == canonical_file
        and previous.get("event_backed_file") == event_file
        and previous.get("comparison_policy") == comparison_policy
        and (previous.get("nondiagnostic_tie_audit") or {}).get("valid")
        is True
    ):
        return previous, True

    canonical_sql = _sql_path(canonical_path)
    event_sql = _sql_path(event_backed_path)
    event_new_class_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}') "
            f"WHERE medication_class = '{new_medication_class}'"
        ).fetchone()[0]
    )
    canonical_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{canonical_sql}')"
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}')"
        ).fetchone()[0]
    )
    if event_count - canonical_count != event_new_class_count:
        raise RuntimeError(
            f"{table_name} row-count growth is not fully explained by "
            f"'{new_medication_class}' rows: event_count={event_count}, "
            f"canonical_count={canonical_count}, "
            f"new_class_row_count={event_new_class_count}"
        )

    filtered_path = event_backed_path.with_name(
        f"{event_backed_path.stem}.excluding_{new_medication_class}.parquet"
    )
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{event_sql}') "
        f"WHERE medication_class != '{new_medication_class}') "
        f"TO '{_sql_path(filtered_path)}' (FORMAT PARQUET)"
    )
    base_comparison, _ = _compare_or_resume_table(
        connection,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        table_name=f"{table_name}.excluding_{new_medication_class}_scratch",
        canonical_path=canonical_path,
        event_backed_path=filtered_path,
        nondiagnostic_tie_columns=("nearest_medication",),
    )
    tie_audit = _audit_nondiagnostic_medication_ties(
        connection,
        canonical_windows_path=canonical_path,
        event_backed_windows_path=filtered_path,
        canonical_medication_path=canonical_medication_path,
        event_backed_medication_path=event_backed_medication_path,
        master_path=master_path,
    )
    tie_audit["valid"] = bool(
        tie_audit["canonical_choice_not_at_minimum"] == 0
        and tie_audit["event_backed_choice_not_at_minimum"] == 0
        and tie_audit["differing_rows"]
        == base_comparison["canonical_only_row_count"]
        == base_comparison["event_backed_only_row_count"]
    )
    comparison = dict(base_comparison)
    comparison["table_name"] = table_name
    comparison["comparison_policy"] = comparison_policy
    comparison["nondiagnostic_tie_columns"] = ["nearest_medication"]
    comparison["canonical_file"] = canonical_file
    comparison["event_backed_file"] = event_file
    comparison["canonical_row_count"] = canonical_count
    comparison["event_backed_row_count"] = event_count
    comparison["excluded_new_medication_class"] = new_medication_class
    comparison["excluded_new_medication_class_row_count"] = (
        event_new_class_count
    )
    comparison["nondiagnostic_tie_audit"] = tie_audit
    comparison["equivalent"] = bool(
        comparison["equivalent"] and tie_audit["valid"]
    )
    checkpoint.setdefault("tables", {})[table_name] = comparison
    atomic_write_json(checkpoint_path, checkpoint)
    return comparison, False


def _compare_or_resume_table_excluding_new_class_rows(
    connection,
    *,
    checkpoint_path: Path,
    checkpoint: dict,
    table_name: str,
    canonical_path: Path,
    event_backed_path: Path,
    new_medication_class: str,
    class_column: str = "medication_class",
) -> tuple[dict, bool]:
    """General-purpose wrapper for a table (no tie-audit involved) that
    gains entirely new rows -- all with `class_column ==
    new_medication_class` -- once the canonical evidence predates that
    class being recognized. Verifies the new-class rows exist ONLY on the
    event-backed side and fully explain the row-count delta, then
    delegates unchanged to _compare_or_resume_table on a filtered copy, so
    a genuine future divergence elsewhere in the table is still caught
    exactly as before. See
    _compare_or_resume_medication_windows_with_new_class for the sibling
    version that also layers the nearest_medication tie-audit on top.
    """
    if not _new_medication_class_is_still_new(
        connection,
        canonical_path=canonical_path,
        class_column=class_column,
        new_medication_class=new_medication_class,
    ):
        return _compare_or_resume_table(
            connection,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            table_name=table_name,
            canonical_path=canonical_path,
            event_backed_path=event_backed_path,
        )
    canonical_file = file_content_fingerprint(canonical_path)
    event_file = file_content_fingerprint(event_backed_path)
    comparison_policy = {
        "mode": "exact_full_table_excluding_new_class_rows",
        "new_medication_class": new_medication_class,
        "class_column": class_column,
    }
    previous = checkpoint.get("tables", {}).get(table_name)
    if (
        previous is not None
        and previous.get("canonical_file") == canonical_file
        and previous.get("event_backed_file") == event_file
        and previous.get("comparison_policy") == comparison_policy
        and previous.get("equivalent") is True
    ):
        return previous, True

    canonical_sql = _sql_path(canonical_path)
    event_sql = _sql_path(event_backed_path)
    event_new_class_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}') "
            f"WHERE {class_column} = '{new_medication_class}'"
        ).fetchone()[0]
    )
    canonical_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{canonical_sql}')"
        ).fetchone()[0]
    )
    event_count = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{event_sql}')"
        ).fetchone()[0]
    )
    if event_count - canonical_count != event_new_class_count:
        raise RuntimeError(
            f"{table_name} row-count growth is not fully explained by "
            f"'{new_medication_class}' rows: event_count={event_count}, "
            f"canonical_count={canonical_count}, "
            f"new_class_row_count={event_new_class_count}"
        )

    filtered_path = event_backed_path.with_name(
        f"{event_backed_path.stem}.excluding_{new_medication_class}.parquet"
    )
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{event_sql}') "
        f"WHERE {class_column} != '{new_medication_class}') "
        f"TO '{_sql_path(filtered_path)}' (FORMAT PARQUET)"
    )
    base_comparison, _ = _compare_or_resume_table(
        connection,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        table_name=f"{table_name}.excluding_{new_medication_class}_scratch",
        canonical_path=canonical_path,
        event_backed_path=filtered_path,
    )
    comparison = dict(base_comparison)
    comparison["table_name"] = table_name
    comparison["comparison_policy"] = comparison_policy
    comparison["canonical_file"] = canonical_file
    comparison["event_backed_file"] = event_file
    comparison["canonical_row_count"] = canonical_count
    comparison["event_backed_row_count"] = event_count
    comparison["excluded_new_medication_class"] = new_medication_class
    comparison["excluded_new_medication_class_row_count"] = (
        event_new_class_count
    )
    checkpoint.setdefault("tables", {})[table_name] = comparison
    atomic_write_json(checkpoint_path, checkpoint)
    return comparison, False


def _audit_nondiagnostic_medication_ties(
    connection,
    *,
    canonical_windows_path: Path,
    event_backed_windows_path: Path,
    canonical_medication_path: Path,
    event_backed_medication_path: Path,
    master_path: Path,
) -> dict:
    canonical_windows = _sql_path(canonical_windows_path)
    event_windows = _sql_path(event_backed_windows_path)
    canonical_medications = _sql_path(canonical_medication_path)
    event_medications = _sql_path(event_backed_medication_path)
    master = _sql_path(master_path)
    result = connection.execute(
        f"""
        WITH differing AS (
            SELECT
                c.subject_id,
                c.study_id,
                c.medication_class,
                c.nearest_medication AS canonical_name,
                e.nearest_medication AS event_name,
                c.nearest_delta_hours
            FROM read_parquet('{canonical_windows}') c
            JOIN read_parquet('{event_windows}') e
              USING (subject_id, study_id, medication_class)
            WHERE c.nearest_medication IS DISTINCT FROM
                  e.nearest_medication
        ),
        canonical_valid AS (
            SELECT
                d.subject_id,
                d.study_id,
                d.medication_class,
                BOOL_OR(
                    cm.medication_raw IS NOT DISTINCT FROM
                        d.canonical_name
                    AND ABS(
                        ABS(
                            DATE_DIFF(
                                'second',
                                m.ecg_time,
                                cm.event_time
                            )
                        ) / 3600.0
                        - d.nearest_delta_hours
                    ) < 1e-9
                ) AS valid_choice
            FROM differing d
            JOIN read_parquet('{master}') m
              USING (subject_id, study_id)
            LEFT JOIN read_parquet('{canonical_medications}') cm
              ON d.subject_id = cm.subject_id
             AND d.medication_class = cm.medication_class
            GROUP BY d.subject_id, d.study_id, d.medication_class
        ),
        event_valid AS (
            SELECT
                d.subject_id,
                d.study_id,
                d.medication_class,
                BOOL_OR(
                    em.medication_raw IS NOT DISTINCT FROM d.event_name
                    AND ABS(
                        ABS(
                            DATE_DIFF(
                                'second',
                                m.ecg_time,
                                em.event_time
                            )
                        ) / 3600.0
                        - d.nearest_delta_hours
                    ) < 1e-9
                ) AS valid_choice
            FROM differing d
            JOIN read_parquet('{master}') m
              USING (subject_id, study_id)
            LEFT JOIN read_parquet('{event_medications}') em
              ON d.subject_id = em.subject_id
             AND d.medication_class = em.medication_class
            GROUP BY d.subject_id, d.study_id, d.medication_class
        )
        SELECT
            COUNT(*) AS differing_rows,
            COUNT(*) FILTER (
                WHERE NOT COALESCE(c.valid_choice, FALSE)
            ) AS canonical_choice_not_at_minimum,
            COUNT(*) FILTER (
                WHERE NOT COALESCE(e.valid_choice, FALSE)
            ) AS event_choice_not_at_minimum
        FROM differing d
        LEFT JOIN canonical_valid c
          USING (subject_id, study_id, medication_class)
        LEFT JOIN event_valid e
          USING (subject_id, study_id, medication_class)
        """
    ).fetchone()
    return {
        "field": "nearest_medication",
        "role": "display_only_not_consumed_by_labels_or_models",
        "rule": (
            "Both legacy and event-backed names must exist among candidates "
            "at the identical recorded minimum absolute ECG time distance."
        ),
        "differing_rows": int(result[0]),
        "canonical_choice_not_at_minimum": int(result[1]),
        "event_backed_choice_not_at_minimum": int(result[2]),
        "locked_test_associations_read": False,
    }


def _read_checkpoint_payload(path: Path) -> Optional[dict]:
    """Checkpoint contents, or None if the file cannot be trusted.

    Truncation is the normal way a checkpoint dies: the process was killed
    mid-write, or the machine lost power. That must not be fatal either --
    an unreadable checkpoint carries exactly as much information as a
    missing one.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_equivalence_checkpoint(
    path: Path,
    *,
    input_hash: str,
) -> dict:
    if path.is_file():
        payload = _read_checkpoint_payload(path)
        if payload is not None and (
            payload.get("checkpoint_version")
            == EVENT_EVIDENCE_ATTESTATION_VERSION
            and payload.get("input_hash") == input_hash
        ):
            payload.setdefault("tables", {})
            return payload
    # A checkpoint that describes different work is not an error, it is
    # simply unusable: discard it and start the stage fresh. Refusing to
    # continue would cost a multi-day run over a file whose only purpose is
    # to save time, and whose contents this stage can always produce again.
    payload = {
        "checkpoint_version": EVENT_EVIDENCE_ATTESTATION_VERSION,
        "input_hash": input_hash,
        "tables": {},
    }
    atomic_write_json(path, payload)
    return payload


def _manifest_parquets(
    manifest: StageManifest,
) -> dict[str, Path]:
    result = {
        Path(path).name: Path(path)
        for path in manifest.output_paths
        if Path(path).suffix.lower() == ".parquet"
    }
    if len(result) != sum(
        Path(path).suffix.lower() == ".parquet"
        for path in manifest.output_paths
    ):
        raise RuntimeError(
            "Evidence manifest contains duplicate Parquet basenames."
        )
    return result


def _classify_evidence_table_scope(
    canonical_parquets: dict[str, Path],
    event_backed_parquets: dict[str, Path],
) -> tuple[list[str], list[str]]:
    canonical_names = set(canonical_parquets)
    event_names = set(event_backed_parquets)
    missing_canonical = canonical_names - event_names
    event_only = event_names - canonical_names
    unexpected_event_only = (
        event_only - set(ADDITIVE_NON_LABEL_EVENT_TABLES)
    )
    if missing_canonical or unexpected_event_only:
        raise RuntimeError(
            "Canonical and event-backed evidence expose an unsafe table "
            "difference: "
            f"canonical_only={sorted(missing_canonical)}, "
            f"unexpected_event_only={sorted(unexpected_event_only)}, "
            f"allowlisted_event_only={sorted(event_only - unexpected_event_only)}"
        )
    return sorted(canonical_names), sorted(event_only)


def _manifest_json(
    manifest: StageManifest,
    name: str,
) -> Path:
    matches = [
        Path(path)
        for path in manifest.output_paths
        if Path(path).name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Evidence manifest has no unique {name}: {matches}"
        )
    return matches[0]


def _normalized_definition_hash(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("created_at", None)
    return content_hash(payload)


def _copy_content_addressed(source: Path, target: Path) -> None:
    source_fingerprint = file_content_fingerprint(source)
    if target.is_file():
        target_fingerprint = file_content_fingerprint(target)
        if (
            target_fingerprint["size"] == source_fingerprint["size"]
            and target_fingerprint["sha256"]
            == source_fingerprint["sha256"]
        ):
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, temporary)
        atomic_replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_content_fingerprint(fingerprint: object) -> None:
    if not isinstance(fingerprint, dict):
        raise RuntimeError(
            "Clinical-event attestation fingerprint is invalid."
        )
    path = Path(str(fingerprint.get("path", "")))
    if not path.is_file():
        raise RuntimeError(
            f"Clinical-event attestation input is missing: {path}"
        )
    current = file_content_fingerprint(path)
    if (
        current["size"] != fingerprint.get("size")
        or current["sha256"] != fingerprint.get("sha256")
    ):
        raise RuntimeError(
            f"Clinical-event attestation input changed: {path}"
        )


def _sql_path(path: Path | str) -> str:
    return (
        str(Path(path).resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'
