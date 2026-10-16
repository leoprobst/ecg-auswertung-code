from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .analysis.results import AnalysisRun


LEGACY_DIAGNOSTIC_NOTICE = (
    "Historische PTB-XL-Autodiagnose deaktiviert: Die PPV-gewichtete Kombination "
    "ist keine validierte Erkrankungswahrscheinlichkeit. Bitte den versionierten "
    "Kriterien- und Forschungsbereich im EKG-Labor verwenden. Alte Ergebnisse "
    "werden dadurch nicht nachträglich validiert."
)


class LegacyDiagnosticDisabledError(RuntimeError):
    """The retired heuristic must not produce new patient-level claims."""


@dataclass
class PathologyScore:
    pathology_name: str
    superclass: str
    probability: float
    evidence: list[dict] = field(default_factory=list)
    matched_biomarkers: list[str] = field(default_factory=list)
    best_auc: float = 0.0
    best_threshold: Optional[float] = None
    is_above_threshold: bool = False

    @property
    def confidence_label(self) -> str:
        return "UNVALIDATED_LEGACY"


@dataclass
class DiagnosticReport:
    biomarker_values: dict[str, Optional[float]]
    age: Optional[int]
    sex: Optional[str]
    pathology_scores: list[PathologyScore]
    threshold_violations: list[dict]
    normal_biomarkers: list[str]
    abnormal_biomarkers: list[str]
    borderline_biomarkers: list[str]

    @property
    def top_pathology(self) -> Optional[PathologyScore]:
        raise LegacyDiagnosticDisabledError(LEGACY_DIAGNOSTIC_NOTICE)

    @property
    def significant_pathologies(self) -> list[PathologyScore]:
        raise LegacyDiagnosticDisabledError(LEGACY_DIAGNOSTIC_NOTICE)

    @property
    def summary(self) -> str:
        return (LEGACY_DIAGNOSTIC_NOTICE +
                f"\nHistorischer Datensatz: {len(self.pathology_scores)} gespeicherte "
                "Heuristik-Einträge. Keine aktuelle Diagnose- oder Sicherheitsbewertung.")


class DiagnosticEngine:
    def __init__(self):
        self._learned_thresholds: dict[str, dict[str, dict]] = {}
        self._analysis_run: Optional[AnalysisRun] = None
        self._min_auc: float = 0.6

    @property
    def has_learned_data(self) -> bool:
        return len(self._learned_thresholds) > 0

    def learn_from_analysis(self, run: AnalysisRun, min_auc: float = 0.6):
        self._analysis_run = run
        self._min_auc = min_auc
        self._learned_thresholds.clear()

        for r in run.results:
            if r.auc < min_auc:
                continue

            bname = r.biomarker_name
            pname = r.pathology_name

            if bname not in self._learned_thresholds:
                self._learned_thresholds[bname] = {}

            direction = r.threshold_direction or ("above" if r.direction >= 0 else "below")

            self._learned_thresholds[bname][pname] = {
                "threshold": r.optimal_threshold,
                "auc": r.auc,
                "sensitivity": r.optimal_sensitivity,
                "specificity": r.optimal_specificity,
                "ppv": r.optimal_ppv,
                "npv": r.optimal_npv,
                "youden_j": r.youden_j,
                "is_significant": r.is_significant,
                "direction": direction,
                "threshold_status": r.threshold_status,
            }

    def diagnose(self, biomarker_values: dict[str, Optional[float]],
                 age: Optional[int] = None, sex: Optional[str] = None,
                 min_probability: float = 0.0) -> DiagnosticReport:
        raise LegacyDiagnosticDisabledError(LEGACY_DIAGNOSTIC_NOTICE)

    def _compute_pathology_scores(self, biomarker_values: dict,
                                   violations: list[dict],
                                   min_probability: float) -> list[PathologyScore]:
        raise LegacyDiagnosticDisabledError(LEGACY_DIAGNOSTIC_NOTICE)

    def get_biomarker_pathology_map(self) -> dict[str, list[tuple[str, float, float, str]]]:
        result = {}
        for bname, pathologies in self._learned_thresholds.items():
            entries = []
            for pname, data in pathologies.items():
                entries.append((pname, data["threshold"], data["auc"], data.get("direction", "above")))
            entries.sort(key=lambda x: x[2], reverse=True)
            result[bname] = entries
        return result
