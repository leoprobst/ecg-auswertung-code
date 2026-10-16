from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..delineation.base import RecordDelineation


@dataclass
class BiomarkerResult:
    name: str
    value: Optional[float]
    unit: str
    category: str
    confidence: float = 1.0
    per_lead: Optional[dict[str, float]] = None
    description: str = ""
    references: list[str] = field(default_factory=list)
    computation_notes: str = ""
    family: str = ""
    formula: str = ""
    implementation_status: str = "implemented"
    pathological_direction: str = "data_driven"
    pathological_note: str = ""
    measurement_windows: Optional[dict] = None


@dataclass
class BiomarkerDefinition:
    name: str
    family: str
    description: str
    formula: str
    unit: str
    category: str = "vcg"
    source: str = ""
    source_index: Optional[int] = None
    normal_range: tuple[Optional[float], Optional[float]] = (None, None)
    pathological_direction: str = "data_driven"
    pathological_note: str = (
        "No fixed clinical cutoff is defined for this exploratory biomarker. "
        "Pathologic thresholds must be learned per pathology and validated."
    )
    implementation_status: str = "candidate"
    aliases: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    notes: str = ""


class BiomarkerBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def unit(self) -> str: ...

    @property
    @abstractmethod
    def category(self) -> str: ...

    @property
    def references(self) -> list[str]:
        return []

    def normal_range(self, age: Optional[int] = None, sex: Optional[str] = None) -> tuple[Optional[float], Optional[float]]:
        return (None, None)

    @abstractmethod
    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult: ...


class BiomarkerRegistry:
    _biomarkers: dict[str, BiomarkerBase] = {}
    _definitions: dict[str, BiomarkerDefinition] = {}
    _definition_aliases: dict[str, str] = {}

    @classmethod
    def register(cls, biomarker: BiomarkerBase) -> None:
        cls._biomarkers[biomarker.name] = biomarker
        normal_range = biomarker.normal_range()
        existing = cls._definitions.get(biomarker.name)
        definition = BiomarkerDefinition(
            name=biomarker.name,
            family=existing.family if existing else biomarker.category,
            description=existing.description if existing and existing.description else biomarker.description,
            formula=existing.formula if existing else "",
            unit=biomarker.unit,
            category=biomarker.category,
            source=existing.source if existing else "implemented biomarker",
            source_index=existing.source_index if existing else None,
            normal_range=normal_range,
            pathological_direction=existing.pathological_direction if existing else "data_driven",
            pathological_note=existing.pathological_note if existing else (
                "Reference ranges are descriptive only; diagnostic cutoffs in cohort analyses "
                "come from pathology-specific threshold sweeps."
            ),
            implementation_status="implemented",
            aliases=existing.aliases if existing else [],
            references=biomarker.references or (existing.references if existing else []),
            notes=existing.notes if existing else "",
        )
        cls.register_definition(definition)

    @classmethod
    def register_definition(cls, definition: BiomarkerDefinition) -> None:
        existing = cls._definitions.get(definition.name)
        if existing and existing.implementation_status == "implemented" and definition.implementation_status != "implemented":
            definition.implementation_status = "implemented"
            if existing.references and not definition.references:
                definition.references = existing.references
            if existing.normal_range != (None, None):
                definition.normal_range = existing.normal_range

        cls._definitions[definition.name] = definition
        for alias in definition.aliases:
            cls._definition_aliases[alias] = definition.name

    @classmethod
    def get(cls, name: str) -> BiomarkerBase:
        if name not in cls._biomarkers:
            raise ValueError(f"Biomarker '{name}' not registered. Available: {list(cls._biomarkers.keys())}")
        return cls._biomarkers[name]

    @classmethod
    def available(cls, include_metadata_only: bool = False) -> list[str]:
        if include_metadata_only:
            return list(cls._definitions.keys())
        return list(cls._biomarkers.keys())

    @classmethod
    def by_category(cls, category: str) -> list[BiomarkerBase]:
        return [b for b in cls._biomarkers.values() if b.category == category]

    @classmethod
    def definition(cls, name: str) -> BiomarkerDefinition:
        canonical = cls._definition_aliases.get(name, name)
        if canonical not in cls._definitions:
            raise ValueError(f"Biomarker definition '{name}' not registered.")
        return cls._definitions[canonical]

    @classmethod
    def definitions(cls) -> list[BiomarkerDefinition]:
        return list(cls._definitions.values())

    @classmethod
    def search_definitions(
        cls,
        text: str = "",
        category: Optional[str] = None,
        family: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[BiomarkerDefinition]:
        needle = text.strip().lower()
        results = []
        for definition in cls._definitions.values():
            if category and category != "All" and definition.category != category:
                continue
            if family and family != "All" and definition.family != family:
                continue
            if status and status != "All" and definition.implementation_status != status:
                continue
            haystack = " ".join([
                definition.name,
                definition.family,
                definition.category,
                definition.description,
                definition.formula,
                definition.unit,
                definition.implementation_status,
                " ".join(definition.aliases),
            ]).lower()
            if needle and needle not in haystack:
                continue
            results.append(definition)
        # Do not leak full-record sources or a record-scoped primitive DAG to
        # a later direct ``biomarker.compute(...)`` call in the same thread.
        # The complete panel has been materialized at this point, so the
        # shared context is no longer needed.
        from .catalog_biomarkers import reset_feature_context

        reset_feature_context()
        return results

    @classmethod
    def compute_all(
        cls,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
        categories: Optional[list[str]] = None,
        names: Optional[list[str]] = None,
        record_ecg_12lead: Optional[np.ndarray] = None,
        record_vcg_xyz: Optional[np.ndarray] = None,
        record_vcg_wideband_xyz: Optional[np.ndarray] = None,
        record_raw_ecg_12lead: Optional[np.ndarray] = None,
        record_acquisition: Optional[dict] = None,
    ) -> list[BiomarkerResult]:
        # Catalog feature contexts are intentionally scoped to one complete
        # computation so manual fiducial revisions can never reuse stale
        # segment or representative-beat caches.
        from .catalog_biomarkers import (
            reset_feature_context,
            set_feature_context_record_sources,
        )

        reset_feature_context()
        set_feature_context_record_sources(
            record_ecg_12lead,
            record_vcg_xyz,
            record_vcg_wideband_xyz,
            record_raw_ecg_12lead,
            record_acquisition,
        )
        biomarkers: list[BiomarkerBase | BiomarkerDefinition]
        if names is not None:
            biomarkers = []
            for name in names:
                if name in cls._biomarkers:
                    biomarkers.append(cls._biomarkers[name])
                else:
                    biomarkers.append(cls.definition(name))
        elif categories is not None:
            biomarkers = []
            for cat in categories:
                biomarkers.extend(cls.by_category(cat))
        else:
            biomarkers = list(cls._biomarkers.values())

        try:
            stable_cache = getattr(delineation, "stable_beat_cache", None)
            cache_scope = (
                stable_cache()
                if callable(stable_cache)
                else nullcontext()
            )
            with cache_scope:
                return cls._compute_biomarkers(
                    biomarkers,
                    delineation,
                    ecg_12lead,
                    vcg_xyz,
                    fs,
                )
        finally:
            # The record sources set above outlive this call otherwise, and a
            # later direct ``biomarker.compute(...)`` in the same thread would
            # resolve its VCG origin from the *previous* record's full signal.
            # That produces a wrong number, not an error.
            reset_feature_context()

    @classmethod
    def _compute_biomarkers(
        cls,
        biomarkers,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray],
        vcg_xyz: Optional[np.ndarray],
        fs: int,
    ) -> list[BiomarkerResult]:
        results = []
        for b in biomarkers:
            if isinstance(b, BiomarkerDefinition):
                results.append(BiomarkerResult(
                    name=b.name,
                    value=None,
                    unit=b.unit,
                    category=b.category,
                    confidence=0.0,
                    description=b.description,
                    references=b.references,
                    computation_notes="Candidate definition is registered, but no calculator is implemented yet.",
                    family=b.family,
                    formula=b.formula,
                    implementation_status=b.implementation_status,
                    pathological_direction=b.pathological_direction,
                    pathological_note=b.pathological_note,
                ))
                continue
            try:
                result = b.compute(delineation, ecg_12lead, vcg_xyz, fs)
                try:
                    definition = cls.definition(result.name)
                    result.family = definition.family
                    result.formula = definition.formula
                    result.implementation_status = definition.implementation_status
                    result.pathological_direction = definition.pathological_direction
                    result.pathological_note = definition.pathological_note
                    if not result.description:
                        result.description = definition.description
                except Exception:
                    pass
                results.append(result)
            except Exception as e:
                results.append(BiomarkerResult(
                    name=b.name,
                    value=None,
                    unit=b.unit,
                    category=b.category,
                    confidence=0.0,
                    description=b.description,
                    references=b.references,
                    computation_notes=f"Error: {str(e)}",
                ))

        return results

    @classmethod
    def compute_to_dataframe(
        cls,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> dict[str, Optional[float]]:
        results = cls.compute_all(delineation, ecg_12lead, vcg_xyz, fs)
        return {r.name: r.value for r in results}
