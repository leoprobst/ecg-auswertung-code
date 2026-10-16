"""MIMIC-IV-Notes -> encounter-level clinical-mention flags.

Structurally mirrors feature_extraction.py's build_machine_report_labels():
row-by-row Python regex parsing (inherently procedural, not DuckDB-SQL-
expressible, because of the negation/uncertainty heuristic in
note_parser.ClinicalNoteParser), written via atomic_write_parquet,
versioned via StageManifest.

Unlike machine reports (one row per ECG study), discharge summaries and
radiology reports are not aligned to a single ECG -- they are aggregated
to *admission* level (subject_id, hadm_id), analogous to
encounter_microbiology_flags/encounter_care_setting_flags in evidence.py:
a label's encounter-level state is the highest-priority state (positive >
uncertain > negative > not_mentioned) among all notes for that admission.
Admissions with none of the relevant note/report kinds are not included
in the output at all -- evidence.py's LEFT JOIN then leaves those
columns NULL, keeping "no note available" distinct from "note available,
topic not mentioned" (NoteMentionState.NOT_MENTIONED).

Covers, per admission: discharge-summary risk factors/diagnosis
mentions/medication non-adherence, discharge functional/mental status,
and three radiology-report modality groups (chest X-ray, CT/MR head,
CTA chest) -- each modality group scanned only against reports of that
modality (radiology_detail.parquet's exam_name), so a chest X-ray
pattern is never evaluated against a CT head report or vice versa.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .note_parser import (
    DISCHARGE_SECTION_PATTERN,
    RADIOLOGY_SECTION_PATTERN,
    ClinicalNoteParser,
    NOTE_PATTERNS,
    NoteLabel,
    NoteMentionState,
    classify_activity_status,
    classify_mental_status,
    split_sections,
)
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_fingerprint,
)


NOTE_PARSER_VERSION = "2.2.0"
NOTE_COVERAGE_FILENAME = "note_coverage.json"
# Domains in association_domains.py whose evidence can only come from
# MIMIC-IV-Note. When the note sources are absent these produce zero rows,
# which is indistinguishable from "screened, no association" unless the
# coverage state is carried alongside the results.
NOTE_DEPENDENT_DOMAIN_PREFIX = "note_"


NOTE_BATCH_ROWS = 20_000


def _iter_note_rows(path: Path, columns: list[str], keep=None):
    """Yield note rows in file order without loading every text at once.

    Reading all 331,793 discharge texts into one frame needed about 8.5 GB and,
    on a machine with 9-12 GB free, tripped the supervisor's memory pause on
    every resume; the stage restarts from scratch, so it never finished.
    Batches keep the peak near one batch of texts. Order and values are those
    of the whole-frame read: only rows with a hadm_id are kept, as before.
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=NOTE_BATCH_ROWS, columns=columns):
        frame = batch.to_pandas()
        frame = frame[frame["hadm_id"].notna()]
        if keep is not None:
            frame = frame[keep(frame)]
        yield from frame.itertuples(index=False)


def _count_note_rows(path: Path, keep=None) -> int:
    import pyarrow.parquet as pq

    columns = ["hadm_id"] if keep is None else ["note_id", "hadm_id"]
    total = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=NOTE_BATCH_ROWS * 10, columns=columns):
        frame = batch.to_pandas()
        frame = frame[frame["hadm_id"].notna()]
        total += int(len(frame) if keep is None else keep(frame).sum())
    return total


def note_coverage_path(config: MimicProjectConfig) -> Path:
    return config.evidence_root / NOTE_COVERAGE_FILENAME


def read_note_coverage(config: MimicProjectConfig) -> dict:
    """Return the last parse-notes coverage state, or an explicit unknown.

    A missing coverage artifact is reported as ``not_run`` rather than as an
    absence of findings: callers must be able to tell "no note evidence was
    ever produced" from "note evidence was produced and contained nothing".
    """

    path = note_coverage_path(config)
    if not path.is_file():
        return {
            "status": "not_run",
            "assessed": False,
            "present_tables": [],
            "missing_tables": list(_EXPECTED_NOTE_SOURCES),
            "note_encounter_rows": 0,
            "note_mention_rows": 0,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "status": "unreadable",
            "assessed": False,
            "present_tables": [],
            "missing_tables": list(_EXPECTED_NOTE_SOURCES),
            "note_encounter_rows": 0,
            "note_mention_rows": 0,
        }
    payload["assessed"] = bool(payload.get("status") == "complete")
    return payload


_EXPECTED_NOTE_SOURCES = ("discharge", "radiology", "radiology_detail")

_RISK_FACTOR_LABELS = ("smoking_history", "alcohol_use", "illicit_drug_use")
_FAMILY_HISTORY_LABELS = ("family_history_cardiac",)
_DIAGNOSIS_LABELS = (
    "heart_failure_mentioned",
    "arrhythmia_mentioned",
    "mi_or_acs_mentioned",
)
_MEDICATION_ADHERENCE_LABELS = ("medication_nonadherence_mentioned",)
_CHEST_XRAY_LABELS = (
    "cardiomegaly",
    "pulmonary_edema",
    "pleural_effusion",
    "pneumothorax",
)
_HEAD_IMAGING_LABELS = ("acute_stroke_or_infarct", "intracranial_hemorrhage")
_CTA_CHEST_LABELS = ("pulmonary_embolism", "aortic_dissection")
# Not driven by NOTE_PATTERNS/ClinicalNoteParser -- see note_parser.py's
# classify_activity_status()/classify_mental_status() docstrings.
_FUNCTIONAL_STATUS_LABELS = (
    "functional_status_impaired",
    "discharge_mental_status_impaired",
)
_ALL_LABELS = (
    _RISK_FACTOR_LABELS
    + _FAMILY_HISTORY_LABELS
    + _DIAGNOSIS_LABELS
    + _MEDICATION_ADHERENCE_LABELS
    + _CHEST_XRAY_LABELS
    + _HEAD_IMAGING_LABELS
    + _CTA_CHEST_LABELS
    + _FUNCTIONAL_STATUS_LABELS
)

_STATE_PRIORITY = {
    NoteMentionState.POSITIVE: 3,
    NoteMentionState.UNCERTAIN: 2,
    NoteMentionState.NEGATIVE: 1,
    NoteMentionState.NOT_MENTIONED: 0,
}

_ACTIVITY_STATUS_FIELD_RE = re.compile(r"Activity Status:\s*([^\n]+)", re.IGNORECASE)
_MENTAL_STATUS_FIELD_RE = re.compile(r"Mental Status:\s*([^\n]+)", re.IGNORECASE)


def _subset_parser(label_ids: tuple[str, ...]) -> ClinicalNoteParser:
    return ClinicalNoteParser({label: NOTE_PATTERNS[label] for label in label_ids})


_RISK_FACTOR_PARSER = _subset_parser(_RISK_FACTOR_LABELS)
_FAMILY_HISTORY_PARSER = _subset_parser(_FAMILY_HISTORY_LABELS)
_DIAGNOSIS_PARSER = _subset_parser(_DIAGNOSIS_LABELS)
_MEDICATION_ADHERENCE_PARSER = _subset_parser(_MEDICATION_ADHERENCE_LABELS)
_CHEST_XRAY_PARSER = _subset_parser(_CHEST_XRAY_LABELS)
_HEAD_IMAGING_PARSER = _subset_parser(_HEAD_IMAGING_LABELS)
_CTA_CHEST_PARSER = _subset_parser(_CTA_CHEST_LABELS)


def _parse_discharge_note(text: str) -> dict[str, NoteLabel]:
    sections = split_sections(text, DISCHARGE_SECTION_PATTERN)
    social_history = sections.get("social history", "")
    family_history = sections.get("family history", "")
    diagnosis_text = "\n".join(
        part
        for part in (
            sections.get("discharge diagnosis", ""),
            sections.get("brief hospital course", ""),
        )
        if part
    )
    adherence_text = "\n".join(
        part
        for part in (
            sections.get("history of present illness", ""),
            sections.get("brief hospital course", ""),
            social_history,
        )
        if part
    )
    labels: dict[str, NoteLabel] = {}
    labels.update(_RISK_FACTOR_PARSER.parse(social_history))
    labels.update(_FAMILY_HISTORY_PARSER.parse(family_history))
    labels.update(_DIAGNOSIS_PARSER.parse(diagnosis_text))
    labels.update(_MEDICATION_ADHERENCE_PARSER.parse(adherence_text))

    discharge_condition = sections.get("discharge condition", "")
    activity_match = _ACTIVITY_STATUS_FIELD_RE.search(discharge_condition)
    activity_value = activity_match.group(1).strip() if activity_match else ""
    labels["functional_status_impaired"] = NoteLabel(
        "functional_status_impaired",
        classify_activity_status(activity_value),
        activity_value,
        "activity_status_field",
    )
    mental_match = _MENTAL_STATUS_FIELD_RE.search(discharge_condition)
    mental_value = mental_match.group(1).strip() if mental_match else ""
    labels["discharge_mental_status_impaired"] = NoteLabel(
        "discharge_mental_status_impaired",
        classify_mental_status(mental_value),
        mental_value,
        "mental_status_field",
    )
    return labels


def _parse_radiology_impression(
    text: str, parser: ClinicalNoteParser
) -> dict[str, NoteLabel]:
    sections = split_sections(text, RADIOLOGY_SECTION_PATTERN)
    scope = sections.get("impression") or sections.get("findings") or ""
    return parser.parse(scope)


def _absorb(
    admission_states: dict[tuple[int, int], dict[str, NoteMentionState]],
    mentions: list[dict],
    subject_id: int,
    hadm_id: int,
    note_kind: str,
    parsed: dict[str, NoteLabel],
) -> None:
    bucket = admission_states.setdefault((subject_id, hadm_id), {})
    for label_id, label in parsed.items():
        previous_state = bucket.get(label_id, NoteMentionState.NOT_MENTIONED)
        if _STATE_PRIORITY[label.state] > _STATE_PRIORITY[previous_state]:
            bucket[label_id] = label.state
        if label.state != NoteMentionState.NOT_MENTIONED:
            mentions.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "note_kind": note_kind,
                    **label.to_dict(),
                }
            )


def build_note_labels(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    discharge_path = config.standardized_root / "discharge.parquet"
    radiology_path = config.standardized_root / "radiology.parquet"
    radiology_detail_path = config.standardized_root / "radiology_detail.parquet"
    output_path = config.evidence_root / "note_encounter_flags.parquet"
    mentions_path = config.evidence_root / "note_mentions.parquet"
    manifest_path = config.evidence_root / "notes.manifest.json"

    input_hash = content_hash(
        {
            "discharge": (
                file_fingerprint(discharge_path) if discharge_path.exists() else None
            ),
            "radiology": (
                file_fingerprint(radiology_path) if radiology_path.exists() else None
            ),
            "radiology_detail": (
                file_fingerprint(radiology_detail_path)
                if radiology_detail_path.exists()
                else None
            ),
            "parser_version": NOTE_PARSER_VERSION,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, NOTE_PARSER_VERSION):
            return previous

    admission_states: dict[tuple[int, int], dict[str, NoteMentionState]] = {}
    mentions: list[dict] = []

    if discharge_path.exists():
        total = _count_note_rows(discharge_path)
        rows = _iter_note_rows(discharge_path, ["subject_id", "hadm_id", "text"])
        for index, row in enumerate(rows, start=1):
            _absorb(
                admission_states,
                mentions,
                int(row.subject_id),
                int(row.hadm_id),
                "discharge",
                _parse_discharge_note(row.text),
            )
            if progress and index % 50_000 == 0:
                progress(f"Entlassungsbriefe {index:,}/{total:,}")

    if radiology_path.exists() and radiology_detail_path.exists():
        # Only exam names are used; filtering while reading avoids holding
        # every detail field of every radiology report in memory.
        detail = pd.read_parquet(
            radiology_detail_path,
            columns=["note_id", "field_name", "field_value"],
            filters=[("field_name", "==", "exam_name")],
        )
        exam_name = detail["field_value"].astype(str)
        is_exam_name = detail["field_name"] == "exam_name"
        # "%CHEST%" alone also catches CT/CTA/MRA/MRI chest and chest
        # fluoroscopy/ultrasound studies (real-data finding) -- excluded
        # here so this pass only sees genuine plain-film chest X-rays;
        # CT/CTA studies get their own dedicated passes below.
        not_plain_film = (
            exam_name.str.contains(r"^CT\s", case=False, na=False, regex=True)
            | exam_name.str.contains(r"^CTA", case=False, na=False, regex=True)
            | exam_name.str.contains(r"^MR", case=False, na=False, regex=True)
            | exam_name.str.contains("FLUORO", case=False, na=False, regex=False)
            | exam_name.str.contains("U.S.", case=False, na=False, regex=False)
        )
        chest_xray_ids = set(
            detail.loc[
                is_exam_name
                & exam_name.str.contains("CHEST", case=False, na=False, regex=False)
                & ~not_plain_film,
                "note_id",
            ]
        )
        head_imaging_ids = set(
            detail.loc[
                is_exam_name
                & (
                    exam_name.str.contains("CT HEAD", case=False, na=False, regex=False)
                    | exam_name.str.contains("MR HEAD", case=False, na=False, regex=False)
                ),
                "note_id",
            ]
        )
        cta_chest_ids = set(
            detail.loc[
                is_exam_name
                & exam_name.str.contains("CTA CHEST", case=False, na=False, regex=False),
                "note_id",
            ]
        )

        del detail, exam_name, is_exam_name, not_plain_film
        for note_kind, ids, parser in (
            ("radiology_chest_xray", chest_xray_ids, _CHEST_XRAY_PARSER),
            ("radiology_head", head_imaging_ids, _HEAD_IMAGING_PARSER),
            ("radiology_cta_chest", cta_chest_ids, _CTA_CHEST_PARSER),
        ):
            # One streaming pass per note kind keeps the original order:
            # all chest films in file order, then head, then CTA chest.
            keep = lambda frame, ids=ids: frame["note_id"].isin(ids)
            total = _count_note_rows(radiology_path, keep)
            subset = _iter_note_rows(
                radiology_path, ["note_id", "subject_id", "hadm_id", "text"], keep
            )
            for index, row in enumerate(subset, start=1):
                _absorb(
                    admission_states,
                    mentions,
                    int(row.subject_id),
                    int(row.hadm_id),
                    note_kind,
                    _parse_radiology_impression(row.text, parser),
                )
                if progress and index % 50_000 == 0:
                    progress(f"{note_kind} {index:,}/{total:,}")

    output_columns = ["subject_id", "hadm_id"] + [
        f"note_{label_id}_state" for label_id in _ALL_LABELS
    ]
    rows = []
    for (subject_id, hadm_id), bucket in admission_states.items():
        row = {"subject_id": subject_id, "hadm_id": hadm_id}
        for label_id in _ALL_LABELS:
            row[f"note_{label_id}_state"] = bucket.get(
                label_id, NoteMentionState.NOT_MENTIONED
            ).value
        rows.append(row)
    output_frame = (
        pd.DataFrame(rows, columns=output_columns)
        if rows
        else pd.DataFrame(columns=output_columns)
    )

    mention_columns = [
        "subject_id",
        "hadm_id",
        "note_kind",
        "label_id",
        "state",
        "matched_clause",
        "matched_pattern",
    ]
    mention_frame = (
        pd.DataFrame(mentions, columns=mention_columns)
        if mentions
        else pd.DataFrame(columns=mention_columns)
    )

    atomic_write_parquet(output_frame, output_path)
    atomic_write_parquet(mention_frame, mentions_path)
    available_sources = tuple(
        name
        for name, path in (
            ("discharge", discharge_path),
            ("radiology", radiology_path),
            ("radiology_detail", radiology_detail_path),
        )
        if path.is_file()
    )
    expected_sources = _EXPECTED_NOTE_SOURCES
    missing_sources = tuple(
        name for name in expected_sources if name not in available_sources
    )
    if not available_sources and config.mimic_note_root is None:
        coverage_status = "not_configured"
    elif not available_sources:
        coverage_status = "configured_but_not_ingested"
    elif missing_sources:
        coverage_status = "partial"
    else:
        coverage_status = "complete"
    output_paths = (
        str(output_path.resolve()),
        str(mentions_path.resolve()),
    )

    coverage_payload = {
        "stage": "parse-notes",
        "stage_version": NOTE_PARSER_VERSION,
        "status": coverage_status,
        "configured_root": (
            str(config.mimic_note_root)
            if config.mimic_note_root is not None
            else None
        ),
        "present_tables": list(available_sources),
        "missing_tables": list(missing_sources),
        "note_encounter_rows": int(len(output_frame)),
        "note_mention_rows": int(len(mention_frame)),
        "zero_rows_are_not_evidence_of_assessed_absence": True,
    }
    atomic_write_json(note_coverage_path(config), coverage_payload)

    # Only a genuinely complete note ingest may report "complete". Any other
    # state leaves the stage re-runnable, so that configuring or ingesting the
    # note sources later actually re-executes the parse instead of being
    # short-circuited by the empty artifact written here.
    manifest = StageManifest.complete(
        stage="parse-notes",
        stage_version=NOTE_PARSER_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=output_paths,
        row_counts={
            "note_encounters": len(output_frame),
            "note_mentions": len(mention_frame),
        },
        status=(
            "complete"
            if coverage_status == "complete"
            else f"incomplete_note_sources_{coverage_status}"
        ),
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            f"notes_coverage_status={coverage_status}",
            "notes_coverage_present=" + ",".join(available_sources),
            "notes_coverage_missing=" + ",".join(missing_sources),
            "zero_rows_are_not_evidence_of_assessed_absence=True",
        ),
        output_integrity=capture_output_integrity(output_paths),
    )
    manifest.save(manifest_path)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    catalog.register_stage(manifest, manifest_path)
    catalog.create_parquet_view("note_encounter_flags", output_path)
    catalog.create_parquet_view("note_mentions", mentions_path)
    return manifest
