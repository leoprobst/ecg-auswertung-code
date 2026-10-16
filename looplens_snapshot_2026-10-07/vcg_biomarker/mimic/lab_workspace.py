"""Local, versioned expert-review overlays. Never writes production results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import math
import uuid

from .storage import atomic_write_json

LAB_SCHEMA_VERSION = "1.0.0"
REPORT_SECTIONS = (
    "Qualität", "Rhythmus / Frequenz", "Achsen", "Überleitung", "QRS",
    "Repolarisation", "Vergleich", "Zusammenfassung",
)


def report_text_digest(sections: dict) -> str:
    """Bind an explicit conflict review to the exact report text, not just a click."""
    return hashlib.sha256(json.dumps(sections, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def unresolved_review_conflicts(reviews: dict, sections: dict, analysis_id: str | None = None) -> list[str]:
    digest = report_text_digest(sections)
    unresolved = []
    for name, review in reviews.items():
        if not isinstance(review, dict):
            unresolved.append(name)
            continue
        stale = (analysis_id is not None and review.get('decision') in ('accepted', 'edited')
                 and (not analysis_id or review.get('analysis_id') != analysis_id))
        if not review.get('requires_text_reconciliation') and not stale:
            continue
        proof = review.get('text_reconciliation')
        if (not isinstance(proof, dict) or proof.get('sections_sha256') != digest
                or (stale and (not analysis_id or proof.get('analysis_id') != analysis_id))
                or not all(isinstance(proof.get(key), str) and proof[key].strip()
                           for key in ('author', 'reason', 'reviewed_at'))):
            unresolved.append(name)
    return unresolved


def unresolved_override_conflicts(overrides: dict, sections: dict, analysis_id: str) -> list[str]:
    # Overrides refer to a recorded automatic value; retain that origin on re-review.
    reviews = {name: dict(value, decision='accepted') if isinstance(value, dict) else value
               for name, value in overrides.items()}
    return unresolved_review_conflicts(reviews, sections, analysis_id)


@dataclass(frozen=True)
class SignalReference:
    case_id: str
    source_sha256: str
    sampling_rate_hz: float
    sample_count: int
    processing: str = "native_raw"

    def __post_init__(self):
        if not math.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0:
            raise ValueError("Invalid sampling rate")
        if self.sample_count < 1 or not self.source_sha256:
            raise ValueError("Signal identity and sample count required")

    def sample(self, seconds: float) -> int:
        if not math.isfinite(seconds) or seconds < 0 or seconds > (self.sample_count - 1) / self.sampling_rate_hz:
            raise ValueError("Time outside source signal")
        return min(self.sample_count - 1, round(seconds * self.sampling_rate_hz))


@dataclass(frozen=True)
class MeasurementAnnotation:
    signal: SignalReference
    lead: str
    start_sample: int
    end_sample: int
    amplitude_start_mv: float = 0.0
    amplitude_end_mv: float = 0.0
    method: str = "manual_caliper"
    scope: str = "single_lead"
    beat_id: str = ""
    author: str = "local_user"

    def __post_init__(self):
        if not 0 <= self.start_sample <= self.end_sample < self.signal.sample_count:
            raise ValueError("Invalid measurement interval")
        if not all(math.isfinite(v) for v in (self.amplitude_start_mv, self.amplitude_end_mv)):
            raise ValueError("Invalid amplitude")

    @property
    def duration_ms(self):
        return 1000 * (self.end_sample - self.start_sample) / self.signal.sampling_rate_hz

    @property
    def delta_mv(self):
        return self.amplitude_end_mv - self.amplitude_start_mv


def resolve_powerline(choice: str, metadata: dict) -> float:
    """Unknown acquisition country is NOT evidence for 50 or 60 Hz."""
    if choice == "off":
        return 0.0
    if choice in ("50", "60"):
        return float(choice)
    if choice != "metadata":
        raise ValueError("Unknown powerline option")
    value = metadata.get("powerline_hz")
    if value is None:
        value = metadata.get("source_metadata", {}).get("powerline_hz")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if value in (50.0, 60.0) else 0.0


@dataclass(frozen=True)
class ReportRevision:
    case_id: str
    source_sha256: str
    author: str
    status: str = "draft"
    sections: dict = field(default_factory=dict)
    manual_overrides: dict = field(default_factory=dict)
    suggestion_review: dict = field(default_factory=dict)
    measurements: tuple = ()
    analysis_id: str = ""
    parent_revision_id: str = ""
    schema_version: str = LAB_SCHEMA_VERSION
    revision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LabReviewStore:
    def __init__(self, cases_root: Path, case_id: str):
        # Only a direct child of the case library is ever writable.
        if not case_id.startswith("case-") or any(c in case_id for c in "/\\:") or ".." in case_id:
            raise ValueError("Unsafe case identifier")
        self.case_id = case_id
        self.root = (Path(cases_root) / case_id / "expert_review").resolve()
        if not self.root.is_relative_to(Path(cases_root).resolve()):
            raise ValueError("Review path escapes case library")

    def reports(self) -> list[dict]:
        paths = sorted((self.root / "reports").glob("*.json"))
        remaining = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        ordered = []
        parent = ""
        while remaining:
            children = [item for item in remaining if item["parent_revision_id"] == parent]
            if len(children) != 1:
                raise ValueError("Ambiguous or broken report revision chain")
            current = children[0]
            ordered.append(current)
            remaining.remove(current)
            parent = current["revision_id"]
        return ordered

    def patient_link(self, source_sha256: str) -> dict:
        revisions = sorted((self.root / "patient_links").glob("*.json"))
        path = revisions[-1] if revisions else self.root / "patient_link.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("source_sha256") != source_sha256:
            raise ValueError("Patient link belongs to another signal")
        return data

    def save_patient_link(self, source_sha256: str, namespace: str, patient_id: str, evidence: str, author: str):
        if not all(str(value).strip() for value in (source_sha256, namespace, patient_id, evidence, author)):
            raise ValueError("Verified source namespace, patient ID, evidence and reviewer required")
        target = self.root / "patient_links" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, {"source_sha256": source_sha256, "source": namespace.strip(), "patient_id": patient_id.strip(), "evidence": evidence, "author": author, "created_at": datetime.now(timezone.utc).isoformat(), "method": "manual_verified_link"})

    def save_report(self, report: ReportRevision) -> Path:
        # Reuse the existing OS-held lock; released even after process death.
        from .pilot_lab_worker import single_lab_lock
        with single_lab_lock(self.root / "report_write_lock"):
            return self._save_report_unlocked(report)

    def _save_report_unlocked(self, report: ReportRevision) -> Path:
        if report.case_id != self.case_id or not report.author.strip() or not report.source_sha256:
            raise ValueError("Case identity and reviewer required")
        from .storage import content_hash
        for criterion, review in report.suggestion_review.items():
            if not isinstance(review, dict):
                raise ValueError('Invalid review record')
            if 'evidence_snapshot' not in review and 'evidence_sha256' not in review:
                continue  # Legacy records remain explicit legacy, not upgraded proof.
            evidence = review.get('evidence_snapshot')
            if not isinstance(evidence, dict) or content_hash(evidence) != review.get('evidence_sha256'):
                raise ValueError('Review evidence hash mismatch')
            reference = evidence.get('signal_reference')
            if (not isinstance(reference, dict) or reference.get('case_id') != report.case_id
                    or reference.get('source_sha256') != report.source_sha256
                    or evidence.get('criterion_id') != criterion
                    or evidence.get('analysis_id') != review.get('analysis_id')):
                raise ValueError('Review evidence identity mismatch')
        if not report.revision_id or not all(c.isalnum() or c in "-_" for c in report.revision_id):
            raise ValueError("Unsafe revision identifier")
        datetime.fromisoformat(report.created_at)
        previous = self.reports()
        latest = previous[-1] if previous else None
        if any(item["revision_id"] == report.revision_id for item in previous):
            raise ValueError("Existing revision cannot be overwritten")
        if latest and latest["source_sha256"] != report.source_sha256:
            raise ValueError("Source signal changed")
        if report.parent_revision_id != (latest["revision_id"] if latest else ""):
            raise ValueError("Review changed; reload latest revision")
        allowed = {"draft": {"draft", "reviewed"}, "reviewed": {"draft", "reviewed", "completed"}, "completed": {"draft"}}
        if report.status not in allowed[latest["status"] if latest else "draft"]:
            raise ValueError("Complete only a reviewed report; reopen completed reports as a new draft")
        if report.status != "draft" and not str(report.sections.get("Zusammenfassung", "")).strip():
            raise ValueError("Summary required for review")
        if report.status != "draft" and unresolved_review_conflicts(report.suggestion_review, report.sections, report.analysis_id):
            raise ValueError("Unresolved suggestion changes: reconcile report text before review/completion")
        if report.status != "draft" and unresolved_override_conflicts(report.manual_overrides, report.sections, report.analysis_id):
            raise ValueError("Unresolved manual override provenance: reconcile with current analysis before review/completion")
        if latest and report.status == "completed":
            if (report.sections != latest["sections"] or list(report.measurements) != latest["measurements"] or report.analysis_id != latest["analysis_id"]
                    or report.manual_overrides != latest.get("manual_overrides", {}) or report.suggestion_review != latest.get("suggestion_review", {})):
                raise ValueError("Changed content must be reviewed before completion")
        target = self.root / "reports" / f"{report.created_at.replace(':', '')}_{report.revision_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, asdict(report))
        return target

    def save_annotations(self, signal: SignalReference, measurements: list[dict], fiducials: dict, author: str) -> Path:
        if signal.case_id != self.case_id or not author.strip():
            raise ValueError("Signal identity and author required")
        for sample in fiducials.values():
            if not isinstance(sample, int) or not 0 <= sample < signal.sample_count:
                raise ValueError("Fiducial outside native signal")
        for item in measurements:
            if item.get("signal") != asdict(signal):
                raise ValueError("Measurement refers to another signal or processing path")
            measurement = MeasurementAnnotation(signal, item["lead"], item["start_sample"], item["end_sample"], item["amplitude_start_mv"], item["amplitude_end_mv"])
            if measurement.lead not in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"):
                raise ValueError("Unknown ECG lead")
        target = self.root / "annotations" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, {"schema_version": LAB_SCHEMA_VERSION, "signal": asdict(signal), "measurements": measurements, "fiducials": fiducials, "author": author})
        return target
