"""Select short, locally annotated ECGs for blinded practice sessions.

Challenge annotations describe ECG readings, not independently verified
clinical diagnoses. PTB-XL is excluded because its held-out fold is sealed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from ..mimic.external_cohorts import (
    ROLE_INDEPENDENT_EXTERNAL,
    ExternalRecord,
    parse_external_header,
    source_by_directory,
)


@dataclass(frozen=True)
class PracticeCase:
    record: ExternalRecord
    header: Path


def choose_practice_case(
    root: Path,
    *,
    rng: random.Random | None = None,
    excluded_headers: frozenset[Path] = frozenset(),
    attempts: int = 80,
) -> PracticeCase:
    """Sample source and subfolder first; enrich for non-sinus-only labels.

    This is deliberately *not* a prevalence-representative population sample.
    Only headers are searched; the waveform is loaded after selection.
    """
    chooser = rng or random.SystemRandom()
    base = Path(root)
    sources = [
        (directory, source)
        for directory in base.iterdir() if directory.is_dir()
        if (source := source_by_directory(directory.name)) is not None
        and source.role == ROLE_INDEPENDENT_EXTERNAL
    ] if base.is_dir() else []
    if not sources:
        raise FileNotFoundError("Keine unabhängigen Challenge-Trainingsquellen vorhanden")
    fallback: PracticeCase | None = None
    for _ in range(attempts):
        directory, source = chooser.choice(sources)
        groups = [item for item in directory.iterdir() if item.is_dir()]
        group = chooser.choice(groups) if groups else directory
        headers = [item for item in group.glob("*.hea") if item.resolve() not in excluded_headers]
        if not headers:
            continue
        header = chooser.choice(headers)
        record = parse_external_header(header, source.source_id)
        if (record is None or record.n_leads != 12
                or not 5 <= record.duration_seconds <= 30
                or not record.labels and not record.unmapped_codes):
            continue
        candidate = PracticeCase(record=record, header=header.resolve())
        if any(label not in {"sinus_rhythm", "sinus_arrhythmia"} for label in record.labels):
            return candidate
        fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError("Keine geeignete kurze, annotierte 12-Kanal-Aufnahme gefunden")


def practice_solution(case: PracticeCase) -> dict[str, object]:
    record = case.record
    return {
        "source": record.source_id,
        "record_id": record.record_id,
        "age": record.age,
        "sex": record.sex or "unbekannt",
        "ecg_annotation_labels": list(record.labels),
        "unmapped_snomed_codes": list(record.unmapped_codes),
        "reference_kind": "expert_ecg_annotation_not_independent_disease_reference",
        "annotation_scope": "Nur ausdrücklich genannte Labels; fehlende Labels sind nicht negativ",
    }
