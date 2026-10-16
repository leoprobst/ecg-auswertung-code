"""Find a stage's broken assumptions before it costs a week.

The ``therapy-response`` stage had never run. The first time it did, it failed
on ``BinderException: Referenced column "heart_rate_bpm" not found`` -- a
catalog marker whose role is ``ecg_context_comparator``, which keeps it out of
every panel, so the extraction never writes it. The query had been wrong since
it was written; nothing had ever asked it a question.

Three stages are still in that position, and each one is discovered the same
way today: the run reaches it after days of work and stops.

This module asks the questions early. Two checks, both cheap and both aimed at
failure modes that actually happened:

* **Column contract.** Every catalog marker name a stage's SQL mentions must
  exist in the extracted features. Comparing against the *catalog* rather than
  against every identifier is what makes this precise: ``heart_rate_bpm`` is a
  real marker that is simply not written, while ``subject_id`` or ``count``
  are not markers at all and never produce noise.

* **Materialisation size.** A ``fetchdf`` that selects every marker for the
  whole cohort is 640,104 x 1,753 x 8 bytes before pandas overhead. One such
  call sits in a stage that has never run. Estimating it is arithmetic; being
  surprised by it costs a day.

What this cannot do is prove a stage correct. It proves that the columns it
names exist and that its result will fit, which is exactly the ground the last
two failures stood on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .storage import parquet_columns


PREFLIGHT_VERSION = "1.1.0"

# Modules whose stages read the extracted features.  The stages that had
# never run are the reason this exists.
#
# Every module a stage *reads through*, not just the one it is named after.
# The first version listed ``repeatability`` alone and therefore checked a
# file that issues no queries at all: the reads live in ``trajectory.py``,
# and that is where the widest one in the project sits -- a self-join
# carrying 2 x 1,753 marker columns, with ``union_by_name`` on both sides.
# A checker looking at the wrong file reports "ok" and is worse than none.
STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "therapy-response": ("therapy_response",),
    "physiological-gap-markers": ("physiological_gap_models",),
    "repeatability": ("repeatability_stage", "repeatability", "trajectory"),
    "validate": ("validation",),
    "discover-phenome": ("phenome",),
    "screen-broad-axes": ("broad_axis_stage",),
    "deep-combination-search": ("combination_stage",),
}

# Identifiers that look like markers but are structural.  Kept short on
# purpose: the catalog comparison already excludes almost everything.
_STRUCTURAL = frozenset(
    {
        "subject_id",
        "study_id",
        "split",
        "label_id",
        "label_state",
        "ecg_time",
        "extraction_status",
        "confidence",
    }
)

_IDENTIFIER = re.compile(r"\b([a-z][a-z0-9_]{3,})\b")
# ``... AS heart_rate_bpm`` produces a name; it does not require one.
# Without this the check flagged the very stage whose fix introduced the
# alias, which is the kind of false positive that gets a checker ignored.
_ALIAS = re.compile(r"\bAS\s+([a-z][a-z0-9_]{3,})\b", re.IGNORECASE)
# A column list built by joining the marker ids is the shape that
# materialises the whole catalog.  ``f.{feature_column}`` -- one marker at
# a time, as validation does -- is not, and reporting both the same way
# would be a constant wearing a measurement's clothes.
_WIDE_COLUMN_LIST = re.compile(
    # ``[^)]`` looked tighter and was wrong: the column expression itself
    # contains brackets (``f"f.{_quote_identifier(item)}"``), so the class
    # stopped before reaching the ``for``.  A lazy any-character span crosses
    # them.
    # The collection names that actually appear. ``marker_ids`` was
    # missing, which is why the widest build in the project -- the
    # trajectory self-join, 2 x 1,753 columns -- went unreported while
    # narrower ones were flagged.
    r"join\([\s\S]{0,200}?for \w+ in (?:selected|feature_ids|marker_ids|feature_columns)\b",
    re.DOTALL,
)
_STRING_LITERAL = re.compile(
    r'"""(.*?)"""|\'\'\'(.*?)\'\'\'|"([^"\n]*)"|\'([^\'\n]*)\'',
    re.DOTALL,
)
_MATERIALISE = re.compile(r"\.fetchdf\(\)|\.fetch_df\(\)|\.df\(\)")


@dataclass
class StageFinding:
    stage: str
    module: str
    missing_columns: tuple[str, ...] = ()
    materialisation_sites: int = 0
    worst_case_materialisation_gb: Optional[float] = None
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "module": self.module,
            "ok": self.ok,
            "missing_columns": list(self.missing_columns),
            "materialisation_sites": int(self.materialisation_sites),
            "worst_case_materialisation_gb": self.worst_case_materialisation_gb,
            "notes": list(self.notes),
        }


@dataclass
class PreflightReport:
    findings: list[StageFinding] = field(default_factory=list)
    feature_column_count: int = 0
    catalog_marker_count: int = 0
    feature_schema_source: str = ""
    feature_glob: str = ""

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preflight_version": PREFLIGHT_VERSION,
            "ok": self.ok,
            "feature_column_count": self.feature_column_count,
            "catalog_marker_count": self.catalog_marker_count,
            "feature_schema_source": self.feature_schema_source,
            "feature_glob": self.feature_glob,
            "findings": [item.to_dict() for item in self.findings],
        }


def _sql_literals(source: str) -> list[str]:
    """Every string literal, where the SQL lives."""

    out: list[str] = []
    for match in _STRING_LITERAL.finditer(source):
        for group in match.groups():
            if group and len(group) > 20:
                out.append(group)
    return out


def catalog_marker_names() -> frozenset[str]:
    """Every name the marker catalog knows, implemented or not."""

    from ..core.biomarkers import BiomarkerRegistry
    from ..core import biomarkers  # noqa: F401  (registers the catalog)

    return frozenset(BiomarkerRegistry.available(include_metadata_only=True))


def marker_names_referenced(source: str, catalog: Iterable[str]) -> set[str]:
    """Catalog marker names a module's SQL requires from the features.

    Aliases are collected across the **whole module**, not per literal. A
    stage can produce a name in one fragment and read it from a CTE in
    another -- therapy_response builds ``m.machine_heart_rate_bpm AS
    heart_rate_bpm`` in one variable and then selects ``fb.heart_rate_bpm``
    from the CTE that fragment defines. Matching aliases only within the same
    string flagged that as missing, which is exactly the false positive that
    teaches people to ignore a checker.

    The cost is that a module which both produces and genuinely requires the
    same name would be let through. That trade is deliberate: a checker with
    false alarms gets switched off, and then it catches nothing at all.
    """

    known = set(catalog) - _STRUCTURAL
    produced = set(_ALIAS.findall(source))
    referenced: set[str] = set()
    for literal in _sql_literals(source):
        if "select" not in literal.lower():
            continue
        for name in _IDENTIFIER.findall(literal):
            if name in known and name not in produced:
                referenced.add(name)
    return referenced


def check_stage(
    stage: str,
    module_name: str | tuple[str, ...],
    *,
    feature_columns: set[str],
    catalog: Iterable[str],
    package_root: Optional[Path] = None,
    cohort_rows: Optional[int] = None,
) -> StageFinding:
    root = package_root or Path(__file__).resolve().parent
    names = (module_name,) if isinstance(module_name, str) else tuple(module_name)
    label = "+".join(names)
    parts: list[str] = []
    missing_modules: list[str] = []
    for name in names:
        path = root / f"{name}.py"
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        else:
            missing_modules.append(str(path))
    if not parts:
        return StageFinding(
            stage=stage,
            module=label,
            notes=tuple(f"Modul {item} fehlt" for item in missing_modules),
        )
    # Concatenated on purpose: an alias declared in one module and used in
    # another must still count as declared.
    source = "\n".join(parts)
    referenced = marker_names_referenced(source, catalog)
    missing = tuple(sorted(name for name in referenced if name not in feature_columns))

    sites = len(_MATERIALISE.findall(source))
    estimate: Optional[float] = None
    notes: list[str] = []
    wide = bool(_WIDE_COLUMN_LIST.search(source))
    if wide and sites:
        # What a source file can show is that the *path* exists, not that the
        # run takes it: ``validate`` has this shape and the guided run never
        # reaches it, because a shared feature matrix replaces the select.
        # Saying so is the whole value here; a number would only look like a
        # measurement. Measured once already: the gap stage was assumed to
        # materialise 1,753 markers and materialises 80.
        notes.append(
            "breite Spaltenliste aus feature_ids vorhanden -- ob der Zweig "
            "genommen wird, entscheidet sich zur Laufzeit"
        )
        if cohort_rows:
            estimate = round(
                cohort_rows * len(feature_columns) * 8 / 1024**3, 2
            )
    elif sites:
        notes.append(
            f"{sites} Materialisierung(en), Spaltenliste nicht aus feature_ids"
        )
    return StageFinding(
        stage=stage,
        module=label,
        missing_columns=missing,
        materialisation_sites=sites,
        worst_case_materialisation_gb=estimate,
        notes=tuple(notes),
    )


def run_preflight(
    config: Any,
    *,
    panel_key: str = (
        "primary_core_sentinel_discovery_reserve_classical_criteria"
    ),
    stages: Optional[Iterable[str]] = None,
    cohort_rows: Optional[int] = None,
) -> PreflightReport:
    """Check the stages that still have to run against the real artifacts."""

    base = (
        Path(config.features_root)
        / panel_key
        / "development"
        / "split=train"
    )
    probe = str(base / "part-*.parquet").replace("\\", "/")
    parquet_files = tuple(base.glob("part-*.parquet"))
    if parquet_files:
        from .storage import isolated_analysis_connection
        with isolated_analysis_connection(config) as connection:
            connection.execute(
                "SET memory_limit='2GB'; SET threads=1; "
                "SET enable_progress_bar=false;"
            )
            feature_columns = set(parquet_columns(connection, probe))
        schema_source = "existing_feature_parquet"
    else:
        # A pre-extraction preflight must not require the artifact it is
        # meant to protect. Resolve the exact registered panel union instead;
        # the later extraction manifest still verifies the materialized
        # schema. Unknown keys fail closed rather than silently using all
        # catalog markers.
        from .panels import default_feature_panels

        panels = default_feature_panels()
        matching_panel_ids: tuple[str, ...] | None = None
        for width in range(1, len(panels) + 1):
            from itertools import permutations

            for selection in permutations(
                (panel.panel_id for panel in panels),
                width,
            ):
                if "_".join(selection) == panel_key:
                    matching_panel_ids = selection
                    break
            if matching_panel_ids is not None:
                break
        if matching_panel_ids is None:
            raise ValueError(f"Unknown feature panel key: {panel_key}")
        by_id = {panel.panel_id: panel for panel in panels}
        feature_columns = {
            feature_id
            for panel_id in matching_panel_ids
            for feature_id in by_id[panel_id].feature_ids
        }
        schema_source = "registered_panel_contract_before_extraction"

    catalog = catalog_marker_names()
    selected = list(stages) if stages is not None else list(STAGE_MODULES)
    report = PreflightReport(
        feature_column_count=len(feature_columns),
        catalog_marker_count=len(catalog),
        feature_schema_source=schema_source,
        feature_glob=probe,
    )
    for stage in selected:
        module_name = STAGE_MODULES.get(stage)
        if module_name is None:
            continue
        report.findings.append(
            check_stage(
                stage,
                module_name,
                feature_columns=feature_columns,
                catalog=catalog,
                cohort_rows=cohort_rows,
            )
        )
    return report


def render_report(report: PreflightReport) -> str:
    lines = [
        f"Merkmalsspalten: {report.feature_column_count:,}   "
        f"Katalog-Marker: {report.catalog_marker_count:,}",
        f"Schemasource: {getattr(report, 'feature_schema_source', '')}",
        "",
        f"{'Stufe':28s} {'Zustand':10s} Befund",
    ]
    for item in report.findings:
        state = "ok" if item.ok else "FEHLER"
        detail = ""
        if item.missing_columns:
            detail = "fehlende Spalten: " + ", ".join(item.missing_columns)
        elif item.worst_case_materialisation_gb:
            detail = (
                f"obere Schranke {item.worst_case_materialisation_gb} GB, "
                f"falls alle {report.feature_column_count:,} Spalten "
                f"selektiert wuerden ({item.materialisation_sites} Stellen) "
                "-- keine Messung"
            )
        elif item.notes:
            detail = "; ".join(item.notes)
        lines.append(f"{item.stage:28s} {state:10s} {detail}")
    return "\n".join(lines)


__all__ = [
    "PREFLIGHT_VERSION",
    "STAGE_MODULES",
    "PreflightReport",
    "StageFinding",
    "catalog_marker_names",
    "check_stage",
    "marker_names_referenced",
    "render_report",
    "run_preflight",
]
