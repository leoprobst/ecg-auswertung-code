from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathologyDefinition:
    name: str
    description: str
    superclass: str
    subclass: str
    scp_codes: list[str]
    references: list[str] = field(default_factory=list)

    def matches_scp_codes(self, record_scp_codes: dict[str, int], min_likelihood: int = 0) -> bool:
        for code in self.scp_codes:
            if code in record_scp_codes:
                likelihood = record_scp_codes[code]
                if likelihood >= min_likelihood:
                    return True
        return False

    def get_max_likelihood(self, record_scp_codes: dict[str, int]) -> int:
        max_lh = 0
        for code in self.scp_codes:
            if code in record_scp_codes:
                max_lh = max(max_lh, record_scp_codes[code])
        return max_lh


class PathologyBase(ABC):
    @property
    @abstractmethod
    def definition(self) -> PathologyDefinition: ...

    def is_present(self, record_scp_codes: dict[str, int], min_likelihood: int = 0) -> bool:
        return self.definition.matches_scp_codes(record_scp_codes, min_likelihood)

    def get_likelihood(self, record_scp_codes: dict[str, int]) -> int:
        return self.definition.get_max_likelihood(record_scp_codes)


class PathologyRegistry:
    _pathologies: dict[str, PathologyBase] = {}
    _superclass_index: dict[str, list[str]] = {}

    @classmethod
    def register(cls, pathology: PathologyBase) -> None:
        name = pathology.definition.name
        cls._pathologies[name] = pathology
        sc = pathology.definition.superclass
        if sc not in cls._superclass_index:
            cls._superclass_index[sc] = []
        cls._superclass_index[sc].append(name)

    @classmethod
    def get(cls, name: str) -> PathologyBase:
        if name not in cls._pathologies:
            raise ValueError(f"Pathology '{name}' not registered. Available: {list(cls._pathologies.keys())}")
        return cls._pathologies[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._pathologies.keys())

    @classmethod
    def by_superclass(cls, superclass: str) -> list[PathologyBase]:
        names = cls._superclass_index.get(superclass, [])
        return [cls._pathologies[n] for n in names]

    @classmethod
    def superclasses(cls) -> list[str]:
        return list(cls._superclass_index.keys())

    @classmethod
    def check_all(cls, record_scp_codes: dict[str, int], min_likelihood: int = 0) -> dict[str, bool]:
        results = {}
        for name, pathology in cls._pathologies.items():
            results[name] = pathology.is_present(record_scp_codes, min_likelihood)
        return results

    @classmethod
    def check_by_superclass(cls, record_scp_codes: dict[str, int], superclass: str, min_likelihood: int = 0) -> dict[str, bool]:
        results = {}
        for pathology in cls.by_superclass(superclass):
            results[pathology.definition.name] = pathology.is_present(record_scp_codes, min_likelihood)
        return results
