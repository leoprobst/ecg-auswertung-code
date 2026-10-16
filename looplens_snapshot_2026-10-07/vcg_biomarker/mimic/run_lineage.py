"""Analyses identify themselves; nobody has to name them.

The run identifier used to be a text field with a warning attached: *change it
when markers, evidence or configuration have changed, otherwise new results
mix with old ones.*  That put the hardest question in the project on the
person least able to answer it.  The software already computes, to the byte,
whether a change alters results -- that is what the stage hashes are for --
while a human has to remember which of fifty edits mattered.  Getting it wrong
in the careless direction silently blends two different studies; getting it
wrong in the careful direction throws away weeks of valid computation.

So the identifier is derived, not typed.

**Identity comes from every scientific input the pipeline trusts.**  A run is
the same run when the feature implementation, the complete analysis backend,
the scientific configuration and the panel content all match.  In particular,
changing a label, evidence window, statistical estimator or validation gate
must create a new generation just as surely as changing a marker formula.
An identity key that omits something the result depends on is the most
dangerous defect this project can have.

**Explanation comes from a decomposition that is allowed to be incomplete.**
Alongside the identity, component hashes are recorded -- markers, delineation,
signal path, configuration, panels -- purely so a human can be told *what*
changed between two generations.  These never decide identity.  If the
decomposition misses something, a wrong sentence appears in the interface; if
identity missed something, two studies would silently merge.

**Naming.**  ``A007-2026-08-20-a3f9c2``:

* ``A007`` -- the generation, monotonically increasing.  What a person says
  out loud: "the seventh analysis".
* ``2026-08-20`` -- when this generation first appeared.
* ``a3f9c2`` -- the first six characters of the identity fingerprint, so a
  directory on disk can be matched to a registry entry by eye.

Sortable, recognisable, stable, and collision-free for the same reason the
fingerprint is.

**Notes live in the registry, not in the run.**  A note is not a result: it
must stay writable after a run is frozen, and writing into a frozen run
directory is exactly what the freeze exists to prevent.

**Old runs stay readable.**  Directories that predate this scheme are adopted
as legacy entries rather than ignored, so nothing that was computed becomes
invisible because the naming changed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .storage import atomic_write_json, content_hash, source_tree_hash


RUN_LINEAGE_VERSION = "1.2.0"
REGISTRY_FILENAME = "analysis_lineage.json"

# A007-2026-08-20-a3f9c2
GENERATED_RUN_ID = re.compile(
    r"^A(?P<generation>\d{3,})-(?P<date>\d{4}-\d{2}-\d{2})-(?P<short>[0-9a-f]{6})$"
)

COMPONENT_LABELS_DE: dict[str, str] = {
    "analysis": "Analyse-Backend",
    "markers": "Marker-Code",
    "delineation": "Delineation",
    "signal": "Signalaufbereitung",
    "config": "Projektkonfiguration",
    "panels": "Marker-Panels",
    "operational": "Laufsteuerung (nicht identitätsbildend)",
}


def _short(fingerprint: str) -> str:
    return str(fingerprint)[:6]


def format_run_id(generation: int, created: str, fingerprint: str) -> str:
    date = str(created)[:10]
    return f"A{int(generation):03d}-{date}-{_short(fingerprint)}"


@dataclass(frozen=True)
class RunFingerprint:
    """What makes this analysis the analysis it is."""

    fingerprint: str
    components: dict[str, str] = field(default_factory=dict)

    def changed_components(self, other: "RunFingerprint") -> tuple[str, ...]:
        """Which named components differ -- for explanation only."""

        names = set(self.components) | set(other.components)
        return tuple(
            sorted(
                name
                for name in names
                if self.components.get(name) != other.components.get(name)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "components": dict(self.components),
        }


def compute_fingerprint(config: Any) -> RunFingerprint:
    """Derive the identity of the analysis this configuration would produce.

    Imports are local because this module is loaded by the interface at
    startup, while the feature and panel machinery pulls in the whole marker
    catalog.
    """

    from ..core.delineation.benchmark_suite import (
        delineation_implementation_hash,
    )
    from .feature_extraction import feature_implementation_hash
    from .panels import default_feature_panels
    from .presets import STUDY_PRESET_ID, get_analysis_preset
    from .storage import file_sha256, scientific_config_hash

    package_root = Path(__file__).resolve().parents[1]
    core_root = package_root / "core"
    mimic_root = Path(__file__).resolve().parent
    panels = default_feature_panels()
    panel_hash = content_hash(
        {panel.panel_id: sorted(panel.feature_ids) for panel in panels}
    )
    code_hash = feature_implementation_hash()
    # Stage-level manifests already prevent stale reuse, but the run identity
    # is also what isolates result discovery in the GUI.  Hashing only marker
    # code used to let a changed endpoint, evidence window or estimator keep
    # the same run id.  Cover the complete non-GUI analysis backend so no two
    # scientific methods can share an apparent analysis generation.
    # Run machinery (supervisor, queue, telemetry, gates; see code_scope) is
    # excluded: fixing it must not turn finished pilots into a new analysis.
    from .code_scope import operational_source_hash, scientific_source_hash

    analysis_hash = scientific_source_hash(mimic_root)
    config_digest = scientific_config_hash(config)

    # Identity: all scientific code and inputs that can change a result.
    identity = content_hash(
        {
            "lineage_version": RUN_LINEAGE_VERSION,
            "feature_implementation_hash": code_hash,
            "analysis_implementation_hash": analysis_hash,
            "scientific_config_hash": config_digest,
            "panel_hash": panel_hash,
            "preset_id": get_analysis_preset(STUDY_PRESET_ID).preset_id,
        }
    )

    # Explanation only.  Never consulted for identity, and deliberately
    # permitted to be incomplete.
    components = {
        "analysis": analysis_hash,
        "markers": source_tree_hash(core_root / "biomarkers"),
        "delineation": delineation_implementation_hash(),
        "signal": content_hash(
            {
                name: file_sha256(core_root / name)
                for name in ("preprocessing.py", "vcg_transform.py")
                if (core_root / name).is_file()
            }
        ),
        "config": config_digest,
        "panels": panel_hash,
        "operational": operational_source_hash(mimic_root),
    }
    return RunFingerprint(fingerprint=identity, components=components)


@dataclass
class AnalysisRecord:
    """One generation of the analysis."""

    run_id: str
    generation: int
    fingerprint: str
    created_at: str
    components: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    legacy: bool = False
    changed_from_previous: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generation": int(self.generation),
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "components": dict(self.components),
            "notes": self.notes,
            "legacy": bool(self.legacy),
            "changed_from_previous": list(self.changed_from_previous),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisRecord":
        return cls(
            run_id=str(payload.get("run_id", "")),
            generation=int(payload.get("generation", 0) or 0),
            fingerprint=str(payload.get("fingerprint", "")),
            created_at=str(payload.get("created_at", "")),
            components=dict(payload.get("components") or {}),
            notes=str(payload.get("notes", "")),
            legacy=bool(payload.get("legacy", False)),
            changed_from_previous=tuple(
                str(item) for item in (payload.get("changed_from_previous") or [])
            ),
        )

    def describe_change_de(self) -> str:
        if self.legacy:
            return "Aus der Zeit vor der automatischen Benennung."
        if not self.changed_from_previous:
            return "Erste Analyse dieses Projekts."
        labels = [
            COMPONENT_LABELS_DE.get(name, name)
            for name in self.changed_from_previous
        ]
        return "Neu, weil sich geändert hat: " + ", ".join(labels) + "."


class AnalysisLineage:
    """The registry of analyses, keyed by what makes them different."""

    def __init__(self, runs_root: Path):
        self.runs_root = Path(runs_root)
        self.path = self.runs_root / REGISTRY_FILENAME
        self.records: list[AnalysisRecord] = []
        self._load()

    # ------------------------------------------------------------- storage
    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.records = []
            return
        if not isinstance(payload, dict):
            self.records = []
            return
        self.records = [
            AnalysisRecord.from_dict(item)
            for item in (payload.get("analyses") or [])
            if isinstance(item, dict)
        ]

    def save(self) -> Path:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.path,
            {
                "lineage_version": RUN_LINEAGE_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "analyses": [record.to_dict() for record in self.records],
            },
        )
        return self.path

    # ------------------------------------------------------------- queries
    def by_run_id(self, run_id: str) -> Optional[AnalysisRecord]:
        for record in self.records:
            if record.run_id == str(run_id):
                return record
        return None

    def by_fingerprint(self, fingerprint: str) -> Optional[AnalysisRecord]:
        for record in self.records:
            if record.fingerprint and record.fingerprint == str(fingerprint):
                return record
        return None

    @property
    def latest(self) -> Optional[AnalysisRecord]:
        generated = [item for item in self.records if not item.legacy]
        if not generated:
            return None
        return max(generated, key=lambda item: item.generation)

    def next_generation(self) -> int:
        if not self.records:
            return 1
        return max(item.generation for item in self.records) + 1

    # ------------------------------------------------------------ mutation
    def adopt_existing_directories(self) -> tuple[AnalysisRecord, ...]:
        """Register run directories that predate this scheme.

        Nothing that was computed may become invisible because the naming
        changed, so unknown directories are adopted rather than ignored. They
        carry no fingerprint: what produced them cannot be reconstructed, and
        claiming one would be worse than admitting the gap.
        """

        if not self.runs_root.is_dir():
            return ()
        known = {record.run_id for record in self.records}
        adopted: list[AnalysisRecord] = []
        for path in sorted(self.runs_root.iterdir()):
            if not path.is_dir() or path.name in known:
                continue
            if GENERATED_RUN_ID.match(path.name):
                # Generated name without a registry entry: the registry was
                # lost or moved. Adopt it, still without inventing a
                # fingerprint.
                pass
            try:
                created = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                created = ""
            adopted.append(
                AnalysisRecord(
                    run_id=path.name,
                    generation=0,
                    fingerprint="",
                    created_at=created,
                    legacy=True,
                )
            )
        if adopted:
            self.records.extend(adopted)
            self.save()
        return tuple(adopted)

    def resolve(self, fingerprint: RunFingerprint) -> AnalysisRecord:
        """The analysis this configuration belongs to, creating it if new."""

        existing = self.by_fingerprint(fingerprint.fingerprint)
        if existing is not None:
            return existing

        previous = self.latest
        changed: tuple[str, ...] = ()
        if previous is not None:
            changed = RunFingerprint(
                fingerprint=previous.fingerprint,
                components=previous.components,
            ).changed_components(fingerprint)

        created = datetime.now(timezone.utc).isoformat()
        generation = self.next_generation()
        record = AnalysisRecord(
            run_id=format_run_id(generation, created, fingerprint.fingerprint),
            generation=generation,
            fingerprint=fingerprint.fingerprint,
            created_at=created,
            components=dict(fingerprint.components),
            changed_from_previous=changed,
        )
        self.records.append(record)
        self.save()
        return record

    def set_notes(self, run_id: str, notes: str) -> Optional[AnalysisRecord]:
        """Attach free text to an analysis.

        Works for frozen runs too: notes live here rather than in the run
        directory precisely so that a finished analysis can still be annotated
        without touching its results.
        """

        record = self.by_run_id(run_id)
        if record is None:
            return None
        record.notes = str(notes)
        self.save()
        return record


def resolve_current_analysis(config: Any) -> tuple[AnalysisRecord, RunFingerprint]:
    """Which analysis the current configuration would start or continue."""

    fingerprint = compute_fingerprint(config)
    lineage = AnalysisLineage(config.runs_root)
    lineage.adopt_existing_directories()
    return lineage.resolve(fingerprint), fingerprint


def analysis_is_frozen(runs_root: Path, run_id: str) -> bool:
    from .run_integrity import immutable_development_run_markers

    return bool(immutable_development_run_markers(Path(runs_root) / str(run_id)))


__all__ = [
    "COMPONENT_LABELS_DE",
    "GENERATED_RUN_ID",
    "REGISTRY_FILENAME",
    "RUN_LINEAGE_VERSION",
    "AnalysisLineage",
    "AnalysisRecord",
    "RunFingerprint",
    "analysis_is_frozen",
    "compute_fingerprint",
    "format_run_id",
    "resolve_current_analysis",
]
