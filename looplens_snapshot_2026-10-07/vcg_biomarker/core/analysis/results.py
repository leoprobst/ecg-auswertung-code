from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .threshold_sweep import SweepResult
from .statistics import apply_multiple_testing_correction, filter_significant_results
from .clinical_baseline import ClinicalRuleResult


@dataclass
class AnalysisRun:
    run_id: str
    biomarker_names: list[str]
    pathology_names: list[str]
    vcg_method: str
    results: list[SweepResult] = field(default_factory=list)
    correction_method: str = "fdr_bh"
    dataset_versions: dict = field(default_factory=lambda: {
        "analysis_schema": "evidence_v1",
        "ptbxl": "local",
        "mimic_iv_ecg": "ready",
        "mimic_iv_echo": "ready",
        "mimic_iv": "3.1-ready",
    })
    cohort_filter: dict = field(default_factory=dict)
    linkage_rules: dict = field(default_factory=lambda: {
        "ecg_echo_window_days": 2,
        "ecg_clinical_alignment": "subject_id plus relative timestamps",
        "status": "prepared_not_applied",
    })
    split_strategy: str = "development_only"
    comparison_baseline: dict = field(default_factory=lambda: {
        "name": "standard_ecg",
        "status": "not_configured",
    })
    analysis_mode: str = "exploratory_threshold_sweep"
    clinical_baseline_results: list[ClinicalRuleResult] = field(default_factory=list)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "clinical_baseline_results" not in self.__dict__:
            self.clinical_baseline_results = []

    def apply_corrections(self):
        self.results = apply_multiple_testing_correction(
            self.results, method=self.correction_method
        )

    def get_significant(self, min_auc: float = 0.6, min_sens: float = 0.5, min_spec: float = 0.5) -> list[SweepResult]:
        return filter_significant_results(self.results, min_auc, min_sens, min_spec)

    _SUBGROUP_EXPORT_STRATA = [
        "sex_M", "sex_F",
        "age_18-29", "age_30-49", "age_50-69", "age_70+",
    ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = r.summary_dict.copy()
            sm = r.subgroup_metrics or {}
            for stratum in self._SUBGROUP_EXPORT_STRATA:
                col_suffix = stratum.replace("-", "_")
                data = sm.get(stratum, {})
                row[f"auc_{col_suffix}"] = data.get("auc")
                row[f"threshold_{col_suffix}"] = data.get("threshold")
                row[f"sens_{col_suffix}"] = data.get("sensitivity")
                row[f"spec_{col_suffix}"] = data.get("specificity")
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(self, path: Path) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def export_excel(self, path: Path) -> None:
        df = self.to_dataframe()
        df.to_excel(path, index=False)

    def to_dict(self, include_curves: bool = True) -> dict:
        return {
            "schema": "vcg_biomarker.analysis_run.v1",
            "run_id": self.run_id,
            "biomarker_names": self.biomarker_names,
            "pathology_names": self.pathology_names,
            "vcg_method": self.vcg_method,
            "correction_method": self.correction_method,
            "dataset_versions": self.dataset_versions,
            "cohort_filter": self.cohort_filter,
            "linkage_rules": self.linkage_rules,
            "split_strategy": self.split_strategy,
            "comparison_baseline": self.comparison_baseline,
            "analysis_mode": self.analysis_mode,
            "clinical_baseline_results": [
                r.to_dict() if hasattr(r, "to_dict") else dict(r)
                for r in getattr(self, "clinical_baseline_results", [])
            ],
            "results": [r.to_json_dict(include_curves=include_curves) for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisRun":
        results = [SweepResult.from_json_dict(r) for r in data.get("results", [])]
        return cls(
            run_id=data.get("run_id", "imported"),
            biomarker_names=data.get("biomarker_names")
            or sorted({r.biomarker_name for r in results}),
            pathology_names=data.get("pathology_names")
            or sorted({r.pathology_name for r in results}),
            vcg_method=data.get("vcg_method", "unknown"),
            results=results,
            correction_method=data.get("correction_method", "fdr_bh"),
            dataset_versions=data.get("dataset_versions", {}),
            cohort_filter=data.get("cohort_filter", {}),
            linkage_rules=data.get("linkage_rules", {}),
            split_strategy=data.get("split_strategy", "development_only"),
            comparison_baseline=data.get("comparison_baseline", {}),
            analysis_mode=data.get("analysis_mode", "exploratory_threshold_sweep"),
            clinical_baseline_results=[
                ClinicalRuleResult.from_dict(r)
                for r in data.get("clinical_baseline_results", []) or []
            ],
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, run_id: str = "imported_csv",
                       vcg_method: str = "unknown") -> "AnalysisRun":
        records = df.to_dict(orient="records")
        results = [SweepResult.from_json_dict(row) for row in records]
        return cls(
            run_id=run_id,
            biomarker_names=sorted({r.biomarker_name for r in results}),
            pathology_names=sorted({r.pathology_name for r in results}),
            vcg_method=vcg_method,
            results=results,
        )

    @classmethod
    def import_csv(cls, path: Path, run_id: str = "imported_csv",
                   vcg_method: str = "unknown") -> "AnalysisRun":
        return cls.from_dataframe(pd.read_csv(path), run_id=run_id, vcg_method=vcg_method)

    def export_json(self, path: Path, include_curves: bool = True) -> None:
        path.write_text(
            json.dumps(self.to_dict(include_curves=include_curves), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def import_json(cls, path: Path) -> "AnalysisRun":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def ranked_results(self, min_auc: float = 0.0, include_combined: bool = True) -> list[SweepResult]:
        results = [r for r in self.results if r.auc >= min_auc]
        if not include_combined:
            results = [r for r in results if "+" not in r.biomarker_name]
        return sorted(
            results,
            key=lambda r: (r.stability_score, r.auc, r.youden_j, r.n_total),
            reverse=True,
        )

    def cohort_summary(self) -> dict:
        n_pairs = len(self.results)
        n_records = max((r.n_total for r in self.results), default=0)
        return {
            "run_id": self.run_id,
            "n_records_max": n_records,
            "n_biomarkers": len(self.biomarker_names),
            "n_pathologies": len(self.pathology_names),
            "n_pairs": n_pairs,
            "n_significant": sum(1 for r in self.results if r.is_significant),
            "vcg_method": self.vcg_method,
            "analysis_mode": self.analysis_mode,
            "split_strategy": self.split_strategy,
        }

    def get_biomarker_pathology_matrix(self, metric: str = "auc") -> pd.DataFrame:
        data = {}
        for r in self.results:
            if r.biomarker_name not in data:
                data[r.biomarker_name] = {}
            value = getattr(r, metric, r.auc)
            data[r.biomarker_name][r.pathology_name] = value

        return pd.DataFrame(data).T
