from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        if num == 0:
            return 0.0
        return float("inf")
    return float(num / den)


def _normalize_sex(value) -> str:
    text = str(value or "").strip().upper()
    if text in {"M", "MALE", "MAN", "1"}:
        return "M"
    if text in {"F", "FEMALE", "WOMAN", "0"}:
        return "F"
    return ""


@dataclass(frozen=True)
class ClinicalCondition:
    biomarker_name: str
    comparator: str
    threshold: Optional[float] = None
    threshold_by_sex: dict[str, float] = field(default_factory=dict)
    alternative_biomarker_names: tuple[str, ...] = ()
    display_name: str = ""

    @property
    def candidate_biomarker_names(self) -> tuple[str, ...]:
        return (self.biomarker_name, *self.alternative_biomarker_names)

    @property
    def needs_sex(self) -> bool:
        return bool(self.threshold_by_sex)

    @property
    def threshold_label(self) -> str:
        name = self.display_name or self.biomarker_name
        if self.threshold_by_sex:
            parts = [
                f"{sex} {self.comparator} {value:g}"
                for sex, value in sorted(self.threshold_by_sex.items())
            ]
            return f"{name}: " + ", ".join(parts)
        return f"{name}: {self.comparator} {self.threshold:g}"

    def available_biomarker_name(self, frame: pd.DataFrame) -> Optional[str]:
        for name in self.candidate_biomarker_names:
            if f"bm_{name}" in frame.columns:
                return name
        return None

    def missing_label(self) -> str:
        names = [f"bm_{name}" for name in self.candidate_biomarker_names]
        return " or ".join(names)

    def required_columns(self) -> list[str]:
        cols = [f"bm_{name}" for name in self.candidate_biomarker_names]
        if self.needs_sex:
            cols.append("sex")
        return cols

    def threshold_series(self, frame: pd.DataFrame) -> pd.Series:
        if self.threshold_by_sex:
            sex = frame["sex"].map(_normalize_sex)
            normalized = {str(k).upper(): float(v) for k, v in self.threshold_by_sex.items()}
            return sex.map(normalized).astype(float)
        return pd.Series(float(self.threshold), index=frame.index, dtype=float)

    def compare(self, values: pd.Series, thresholds: pd.Series) -> pd.Series:
        if self.comparator == ">":
            return values > thresholds
        if self.comparator == ">=":
            return values >= thresholds
        if self.comparator == "<":
            return values < thresholds
        if self.comparator == "<=":
            return values <= thresholds
        raise ValueError(f"Unsupported comparator: {self.comparator}")


@dataclass(frozen=True)
class ClinicalRule:
    rule_id: str
    pathology_name: str
    biomarker_name: str
    display_name: str
    comparator: str
    threshold: Optional[float] = None
    threshold_by_sex: dict[str, float] = field(default_factory=dict)
    source_label: str = ""
    source_url: str = ""
    note: str = ""
    alternative_biomarker_names: tuple[str, ...] = ()
    conditions: tuple[ClinicalCondition, ...] = ()
    reference_sensitivity: Optional[float] = None
    reference_specificity: Optional[float] = None
    reference_ppv: Optional[float] = None
    reference_npv: Optional[float] = None
    reference_accuracy: Optional[float] = None
    reference_auc: Optional[float] = None
    reference_population: str = ""
    reference_note: str = ""

    def __setstate__(self, state):
        self.__dict__.update(state)
        defaults = {
            "reference_sensitivity": None,
            "reference_specificity": None,
            "reference_ppv": None,
            "reference_npv": None,
            "reference_accuracy": None,
            "reference_balanced_accuracy": None,
            "reference_auc": None,
            "reference_population": "",
            "reference_note": "",
        }
        for key, value in defaults.items():
            if key not in self.__dict__:
                setattr(self, key, value)

    @property
    def pathology_column(self) -> str:
        return f"path_{self.pathology_name}"

    @property
    def needs_sex(self) -> bool:
        return any(condition.needs_sex for condition in self.rule_conditions)

    @property
    def rule_conditions(self) -> tuple[ClinicalCondition, ...]:
        if self.conditions:
            return self.conditions
        return (
            ClinicalCondition(
                biomarker_name=self.biomarker_name,
                comparator=self.comparator,
                threshold=self.threshold,
                threshold_by_sex=self.threshold_by_sex,
                alternative_biomarker_names=self.alternative_biomarker_names,
                display_name=self.display_name,
            ),
        )

    @property
    def threshold_label(self) -> str:
        labels = [condition.threshold_label for condition in self.rule_conditions]
        if len(labels) == 1:
            label = labels[0]
            prefix = f"{self.display_name}: "
            return label[len(prefix):] if label.startswith(prefix) else label
        return " AND ".join(labels)

    @property
    def reference_balanced_accuracy(self) -> Optional[float]:
        if self.reference_sensitivity is None or self.reference_specificity is None:
            return None
        return float((self.reference_sensitivity + self.reference_specificity) / 2.0)

    def required_columns(self) -> list[str]:
        cols = [self.pathology_column]
        for condition in self.rule_conditions:
            cols.extend(condition.required_columns())
        return list(dict.fromkeys(cols))


@dataclass
class ClinicalRuleResult:
    rule_id: str
    pathology_name: str
    biomarker_name: str
    display_name: str
    threshold_label: str
    source_label: str
    source_url: str
    note: str
    status: str
    missing_columns: list[str] = field(default_factory=list)
    n_total: int = 0
    n_evaluable: int = 0
    n_positive: int = 0
    n_negative: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    ppv: Optional[float] = None
    npv: Optional[float] = None
    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    lr_positive: Optional[float] = None
    lr_negative: Optional[float] = None
    reference_sensitivity: Optional[float] = None
    reference_specificity: Optional[float] = None
    reference_ppv: Optional[float] = None
    reference_npv: Optional[float] = None
    reference_accuracy: Optional[float] = None
    reference_balanced_accuracy: Optional[float] = None
    reference_auc: Optional[float] = None
    reference_population: str = ""
    reference_note: str = ""

    @classmethod
    def missing(cls, rule: ClinicalRule, missing_columns: list[str], n_total: int) -> "ClinicalRuleResult":
        return cls(
            rule_id=rule.rule_id,
            pathology_name=rule.pathology_name,
            biomarker_name=rule.biomarker_name,
            display_name=rule.display_name,
            threshold_label=rule.threshold_label,
            source_label=rule.source_label,
            source_url=rule.source_url,
            note=rule.note,
            status="missing_columns",
            missing_columns=missing_columns,
            n_total=n_total,
            reference_sensitivity=rule.reference_sensitivity,
            reference_specificity=rule.reference_specificity,
            reference_ppv=rule.reference_ppv,
            reference_npv=rule.reference_npv,
            reference_accuracy=rule.reference_accuracy,
            reference_balanced_accuracy=rule.reference_balanced_accuracy,
            reference_auc=rule.reference_auc,
            reference_population=rule.reference_population,
            reference_note=rule.reference_note,
        )

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "pathology_name": self.pathology_name,
            "biomarker_name": self.biomarker_name,
            "display_name": self.display_name,
            "threshold_label": self.threshold_label,
            "source_label": self.source_label,
            "source_url": self.source_url,
            "note": self.note,
            "status": self.status,
            "missing_columns": list(self.missing_columns),
            "n_total": self.n_total,
            "n_evaluable": self.n_evaluable,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "lr_positive": self.lr_positive,
            "lr_negative": self.lr_negative,
            "reference_sensitivity": self.reference_sensitivity,
            "reference_specificity": self.reference_specificity,
            "reference_ppv": self.reference_ppv,
            "reference_npv": self.reference_npv,
            "reference_accuracy": self.reference_accuracy,
            "reference_balanced_accuracy": self.reference_balanced_accuracy,
            "reference_auc": self.reference_auc,
            "reference_population": self.reference_population,
            "reference_note": self.reference_note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalRuleResult":
        return cls(
            rule_id=data.get("rule_id", ""),
            pathology_name=data.get("pathology_name", ""),
            biomarker_name=data.get("biomarker_name", ""),
            display_name=data.get("display_name", ""),
            threshold_label=data.get("threshold_label", ""),
            source_label=data.get("source_label", ""),
            source_url=data.get("source_url", ""),
            note=data.get("note", ""),
            status=data.get("status", "unknown"),
            missing_columns=list(data.get("missing_columns", []) or []),
            n_total=int(data.get("n_total", 0) or 0),
            n_evaluable=int(data.get("n_evaluable", 0) or 0),
            n_positive=int(data.get("n_positive", 0) or 0),
            n_negative=int(data.get("n_negative", 0) or 0),
            tp=int(data.get("tp", 0) or 0),
            fp=int(data.get("fp", 0) or 0),
            tn=int(data.get("tn", 0) or 0),
            fn=int(data.get("fn", 0) or 0),
            sensitivity=_optional_float(data.get("sensitivity")),
            specificity=_optional_float(data.get("specificity")),
            ppv=_optional_float(data.get("ppv")),
            npv=_optional_float(data.get("npv")),
            accuracy=_optional_float(data.get("accuracy")),
            balanced_accuracy=_optional_float(data.get("balanced_accuracy")),
            lr_positive=_optional_float(data.get("lr_positive")),
            lr_negative=_optional_float(data.get("lr_negative")),
            reference_sensitivity=_optional_float(data.get("reference_sensitivity")),
            reference_specificity=_optional_float(data.get("reference_specificity")),
            reference_ppv=_optional_float(data.get("reference_ppv")),
            reference_npv=_optional_float(data.get("reference_npv")),
            reference_accuracy=_optional_float(data.get("reference_accuracy")),
            reference_balanced_accuracy=_optional_float(data.get("reference_balanced_accuracy")),
            reference_auc=_optional_float(data.get("reference_auc")),
            reference_population=data.get("reference_population", ""),
            reference_note=data.get("reference_note", ""),
        )


def default_clinical_rules() -> list[ClinicalRule]:
    pr_alt = ("pr_interval_ms",)
    qrs_alt = ("qrs_duration_ms",)
    qtc_bazett_alt = ("qtc_bazett_ms",)
    qtc_fridericia_alt = ("qtc_fridericia_ms",)
    st_elev_alt = ("st60_elevation_max_lead / depression_min_lead",)
    lvh_meta = "Yu et al. PLoS ONE 2021 pooled ECG-LVH meta-analysis vs echo LVMI."

    return [
        ClinicalRule(
            rule_id="avb_pr_gt_200",
            pathology_name="_AVB",
            biomarker_name="pr_interval",
            display_name="PR interval",
            comparator=">",
            threshold=200.0,
            source_label="First-degree AV block: PR > 200 ms",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK448164/",
            note="Clinical ECG conduction reference; used as context, not as a final diagnosis.",
            alternative_biomarker_names=pr_alt,
            reference_note="Definition-style ECG criterion; no independent sensitivity/specificity is displayed.",
        ),
        ClinicalRule(
            rule_id="avb_marked_pr_gt_300",
            pathology_name="_AVB",
            biomarker_name="pr_interval",
            display_name="Marked PR prolongation",
            comparator=">",
            threshold=300.0,
            source_label="Marked first-degree AV block: PR > 300 ms",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK448164/",
            note="Very long PR context only; PTB-XL _AVB also contains higher-grade AV block labels.",
            alternative_biomarker_names=pr_alt,
            reference_note="Definition-style severe PR-prolongation flag; not a sensitivity/specificity benchmark.",
        ),
        ClinicalRule(
            rule_id="lngqt_qtc_bazett_sex",
            pathology_name="LNGQT",
            biomarker_name="qtc_bazett",
            display_name="QTc Bazett",
            comparator=">",
            threshold_by_sex={"M": 440.0, "F": 460.0},
            source_label="QTc upper reference threshold",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK441860/",
            note="Sex-specific prolonged-QTc screening threshold; threshold variants exist across references.",
            alternative_biomarker_names=qtc_bazett_alt,
            reference_note="Reference threshold, not an independent diagnostic-accuracy study.",
        ),
        ClinicalRule(
            rule_id="lngqt_qtc_fridericia_sex",
            pathology_name="LNGQT",
            biomarker_name="qtc_fridericia",
            display_name="QTc Fridericia",
            comparator=">",
            threshold_by_sex={"M": 440.0, "F": 460.0},
            source_label="QTc upper reference threshold",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK441860/",
            note="Sex-specific prolonged-QTc screening threshold; formula choice is shown separately.",
            alternative_biomarker_names=qtc_fridericia_alt,
            reference_note="Reference threshold, not an independent diagnostic-accuracy study.",
        ),
        ClinicalRule(
            rule_id="lngqt_qtc_bazett_gt_500",
            pathology_name="LNGQT",
            biomarker_name="qtc_bazett",
            display_name="QTc Bazett high-risk range",
            comparator=">",
            threshold=500.0,
            source_label="QTc > 500 ms higher torsade risk range",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK441860/",
            note="High-risk prolonged-QTc context; stricter than the upper-reference threshold.",
            alternative_biomarker_names=qtc_bazett_alt,
            reference_note="Risk-range threshold; not a sensitivity/specificity benchmark.",
        ),
        ClinicalRule(
            rule_id="lngqt_qtc_fridericia_gt_500",
            pathology_name="LNGQT",
            biomarker_name="qtc_fridericia",
            display_name="QTc Fridericia high-risk range",
            comparator=">",
            threshold=500.0,
            source_label="QTc > 500 ms higher torsade risk range",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK441860/",
            note="High-risk prolonged-QTc context; stricter than the upper-reference threshold.",
            alternative_biomarker_names=qtc_fridericia_alt,
            reference_note="Risk-range threshold; not a sensitivity/specificity benchmark.",
        ),
        ClinicalRule(
            rule_id="lngqt_qtc_framingham_sex",
            pathology_name="LNGQT",
            biomarker_name="qtc_framingham_ms",
            display_name="QTc Framingham",
            comparator=">",
            threshold_by_sex={"M": 440.0, "F": 460.0},
            source_label="QTc upper reference threshold",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK441860/",
            note="Alternative QTc correction available in the extended biomarker catalog.",
            reference_note="Reference threshold, not an independent diagnostic-accuracy study.",
        ),
        ClinicalRule(
            rule_id="lvh_sokolow_lyon_ge_3500",
            pathology_name="LVH",
            biomarker_name="sokolow_lyon_voltage_surrogate",
            display_name="Sokolow-Lyon voltage surrogate",
            comparator=">=",
            threshold=3500.0,
            source_label="Sokolow-Lyon LVH voltage",
            source_url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246305",
            note="Voltage-only LVH criterion; specificity is often higher than sensitivity.",
            reference_sensitivity=0.24,
            reference_specificity=0.94,
            reference_accuracy=0.28,
            reference_population="Pooled ECG-LVH meta-analysis vs echocardiographic LVMI.",
            reference_note=lvh_meta,
        ),
        ClinicalRule(
            rule_id="lvh_cornell_voltage_sex",
            pathology_name="LVH",
            biomarker_name="cornell_voltage_surrogate",
            display_name="Cornell voltage surrogate",
            comparator=">",
            threshold_by_sex={"M": 2800.0, "F": 2000.0},
            source_label="Cornell voltage LVH threshold",
            source_url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246305",
            note="Sex-specific voltage criterion; surrogate uses the app's measured ECG projection features.",
            reference_sensitivity=0.29,
            reference_specificity=0.92,
            reference_accuracy=0.67,
            reference_population="Pooled ECG-LVH meta-analysis vs echocardiographic LVMI.",
            reference_note=lvh_meta,
        ),
        ClinicalRule(
            rule_id="lvh_peguero_lo_presti_voltage_sex",
            pathology_name="LVH",
            biomarker_name="peguero_lo_presti_voltage_surrogate",
            display_name="Peguero-Lo Presti voltage surrogate",
            comparator=">=",
            threshold_by_sex={"M": 2800.0, "F": 2300.0},
            source_label="Peguero-Lo Presti LVH voltage threshold",
            source_url="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0246305",
            note="Deepest S wave plus S_V4 surrogate; available after preprocessing with the updated biomarker catalog.",
            reference_sensitivity=0.52,
            reference_specificity=0.85,
            reference_accuracy=0.69,
            reference_population="Pooled ECG-LVH meta-analysis vs echocardiographic LVMI.",
            reference_note=lvh_meta,
        ),
        ClinicalRule(
            rule_id="clbbb_qrs_wide_ge_120",
            pathology_name="CLBBB",
            biomarker_name="qrs_duration",
            display_name="Wide-QRS screen",
            comparator=">=",
            threshold=120.0,
            source_label="Bundle branch block QRS duration",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK482167/",
            note="Not morphology-specific for LBBB; only a broad QRS-duration screen.",
            alternative_biomarker_names=qrs_alt,
            reference_note="QRS duration is necessary but not morphology-specific; no standalone accuracy is displayed.",
        ),
        ClinicalRule(
            rule_id="clbbb_strauss_qrs_sex",
            pathology_name="CLBBB",
            biomarker_name="qrs_duration",
            display_name="Strict LBBB QRS-duration screen",
            comparator=">=",
            threshold_by_sex={"M": 140.0, "F": 130.0},
            source_label="Strauss strict LBBB QRS-duration criterion",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK482167/",
            note="Still lacks the required LBBB morphology checks; shown as a stricter duration-only context rule.",
            alternative_biomarker_names=qrs_alt,
            reference_note="Duration-only subset of strict LBBB criteria; morphology is not evaluated here.",
        ),
        ClinicalRule(
            rule_id="crbbb_qrs_wide_ge_120",
            pathology_name="CRBBB",
            biomarker_name="qrs_duration",
            display_name="Wide-QRS screen",
            comparator=">=",
            threshold=120.0,
            source_label="Bundle branch block QRS duration",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK507872/",
            note="Not morphology-specific for RBBB; only a broad QRS-duration screen.",
            alternative_biomarker_names=qrs_alt,
            reference_note="QRS duration is necessary but not morphology-specific; no standalone accuracy is displayed.",
        ),
        ClinicalRule(
            rule_id="irbbb_qrs_100_119",
            pathology_name="IRBBB",
            biomarker_name="qrs_duration",
            display_name="Incomplete RBBB duration screen",
            comparator=">=",
            conditions=(
                ClinicalCondition("qrs_duration", ">=", 100.0, alternative_biomarker_names=qrs_alt),
                ClinicalCondition("qrs_duration", "<", 120.0, alternative_biomarker_names=qrs_alt),
            ),
            source_label="Incomplete RBBB QRS duration range",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK507872/",
            note="Not morphology-specific for incomplete RBBB; only a QRS-duration range screen.",
            reference_note="Duration-only screen; RBBB morphology is not evaluated here.",
        ),
        ClinicalRule(
            rule_id="ivcb_qrs_wide_ge_120",
            pathology_name="IVCB",
            biomarker_name="qrs_duration",
            display_name="Wide-QRS screen",
            comparator=">=",
            threshold=120.0,
            source_label="Wide QRS duration context",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK507872/",
            note="Not morphology-specific; intended only as a broad conduction-delay reference.",
            alternative_biomarker_names=qrs_alt,
            reference_note="Broad conduction-delay threshold; no standalone diagnostic accuracy is displayed.",
        ),
        ClinicalRule(
            rule_id="wpw_short_pr_wide_qrs_screen",
            pathology_name="WPW",
            biomarker_name="pr_interval + qrs_duration",
            display_name="WPW pre-excitation interval screen",
            comparator="<",
            conditions=(
                ClinicalCondition("pr_interval", "<", 120.0, alternative_biomarker_names=pr_alt),
                ClinicalCondition("qrs_duration", ">=", 120.0, alternative_biomarker_names=qrs_alt),
            ),
            source_label="WPW pattern: short PR, wide QRS, delta wave",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK554437/",
            note="Delta-wave morphology is not computed; this is only an interval-based pre-excitation screen.",
            reference_note="Partial WPW ECG pattern only; delta wave is not evaluated.",
        ),
        ClinicalRule(
            rule_id="lae_p_duration_ge_120",
            pathology_name="LAO/LAE",
            biomarker_name="p_duration",
            display_name="P-wave duration",
            comparator=">=",
            threshold=120.0,
            source_label="LAE/IAB screen: P-wave duration >= 120 ms",
            source_url="https://pubmed.ncbi.nlm.nih.gov/41417412/",
            note="Left atrial enlargement/interatrial-block context; ECG duration alone has low sensitivity.",
            alternative_biomarker_names=("p_duration_ms",),
            reference_sensitivity=0.296,
            reference_specificity=0.811,
            reference_population="ELSA-Brasil LAE study using left atrial volume reference.",
            reference_note="P-wave duration >=120 ms showed low sensitivity and moderate-high specificity for LAE.",
        ),
        ClinicalRule(
            rule_id="rae_p_amp_ii_gt_250",
            pathology_name="RAO/RAE",
            biomarker_name="p_amplitude_ii",
            display_name="P-wave amplitude in lead II",
            comparator=">",
            threshold=250.0,
            source_label="RAE screen: P in lead II > 2.5 mm",
            source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC2244611/",
            note="Right atrial enlargement context; lead-II amplitude alone is insensitive and morphology-dependent.",
            reference_specificity=1.0,
            reference_population="CMR atrial enlargement validation; P wave in II/III/aVF >2.5 mm had high specificity.",
            reference_note="Published CMR validation reports high specificity for inferior-lead P amplitude criteria; sensitivity is poor/not shown for this single lead-II surrogate.",
        ),
        ClinicalRule(
            rule_id="sttc_st_depression_ge_50",
            pathology_name="STTC",
            biomarker_name="st_depression",
            display_name="ST depression screen",
            comparator=">=",
            threshold=50.0,
            source_label="Ischemic ST depression >= 0.5 mm in contiguous leads",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK549803/",
            note="Broad max-any-lead ST-depression screen; contiguous-lead and morphology requirements are not evaluated.",
            reference_note="Clinical threshold context only; cache value is a broad maximum, not the full guideline rule.",
        ),
        ClinicalRule(
            rule_id="isca_st_depression_ge_50",
            pathology_name="ISCA",
            biomarker_name="st_depression",
            display_name="ST depression screen",
            comparator=">=",
            threshold=50.0,
            source_label="Ischemic ST depression >= 0.5 mm in contiguous leads",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK549803/",
            note="Broad ischemia screen for anterior ischemic-change labels; no contiguous-lead localization is evaluated.",
            reference_note="Clinical threshold context only; cache value is a broad maximum, not the full guideline rule.",
        ),
        ClinicalRule(
            rule_id="isci_st_depression_ge_50",
            pathology_name="ISCI",
            biomarker_name="st_depression",
            display_name="ST depression screen",
            comparator=">=",
            threshold=50.0,
            source_label="Ischemic ST depression >= 0.5 mm in contiguous leads",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK549803/",
            note="Broad ischemia screen for inferior ischemic-change labels; no contiguous-lead localization is evaluated.",
            reference_note="Clinical threshold context only; cache value is a broad maximum, not the full guideline rule.",
        ),
        ClinicalRule(
            rule_id="nst_st_depression_ge_50",
            pathology_name="NST_",
            biomarker_name="st_depression",
            display_name="ST depression screen",
            comparator=">=",
            threshold=50.0,
            source_label="Ischemic ST depression >= 0.5 mm in contiguous leads",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK549803/",
            note="Broad ST-depression screen for non-specific ST-change labels; no contiguous-lead localization is evaluated.",
            reference_note="Clinical threshold context only; cache value is a broad maximum, not the full guideline rule.",
        ),
        ClinicalRule(
            rule_id="ami_st_elevation_ge_100",
            pathology_name="AMI",
            biomarker_name="st_elevation",
            display_name="ST elevation screen",
            comparator=">=",
            threshold=100.0,
            source_label="ST elevation >= 0.1 mV outside V2-V3",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK532281/",
            note="Broad max-any-lead ST-elevation screen; age/sex, V2-V3, reciprocal-change and contiguous-lead rules are not evaluated.",
            alternative_biomarker_names=st_elev_alt,
            reference_note="Clinical threshold context only; PTB-XL AMI labels include infarction/injury statements, not necessarily acute STEMI.",
        ),
        ClinicalRule(
            rule_id="imi_st_elevation_ge_100",
            pathology_name="IMI",
            biomarker_name="st_elevation",
            display_name="ST elevation screen",
            comparator=">=",
            threshold=100.0,
            source_label="ST elevation >= 0.1 mV outside V2-V3",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK532281/",
            note="Broad max-any-lead ST-elevation screen; no inferior-lead contiguity/localization rule is evaluated.",
            alternative_biomarker_names=st_elev_alt,
            reference_note="Clinical threshold context only; PTB-XL IMI labels include infarction/injury statements, not necessarily acute STEMI.",
        ),
        ClinicalRule(
            rule_id="lmi_st_elevation_ge_100",
            pathology_name="LMI",
            biomarker_name="st_elevation",
            display_name="ST elevation screen",
            comparator=">=",
            threshold=100.0,
            source_label="ST elevation >= 0.1 mV outside V2-V3",
            source_url="https://www.ncbi.nlm.nih.gov/books/NBK532281/",
            note="Broad max-any-lead ST-elevation screen; no lateral-lead contiguity/localization rule is evaluated.",
            alternative_biomarker_names=st_elev_alt,
            reference_note="Clinical threshold context only; PTB-XL LMI labels are not a pure acute-STEMI endpoint.",
        ),
    ]


def required_clinical_baseline_columns(rules: Optional[list[ClinicalRule]] = None) -> list[str]:
    columns: set[str] = set()
    for rule in rules or default_clinical_rules():
        columns.update(rule.required_columns())
    return sorted(columns)


def evaluate_clinical_rules(
    frame: pd.DataFrame,
    rules: Optional[list[ClinicalRule]] = None,
) -> list[ClinicalRuleResult]:
    rules = rules or default_clinical_rules()
    results: list[ClinicalRuleResult] = []
    n_total = int(len(frame))

    for rule in rules:
        conditions = rule.rule_conditions
        missing = []
        if rule.pathology_column not in frame.columns:
            missing.append(rule.pathology_column)
        used_biomarker_names = []
        for condition in conditions:
            name = condition.available_biomarker_name(frame)
            if name is None:
                missing.append(condition.missing_label())
            else:
                used_biomarker_names.append(name)
            if condition.needs_sex and "sex" not in frame.columns:
                missing.append("sex")
        missing = list(dict.fromkeys(missing))
        if missing:
            results.append(ClinicalRuleResult.missing(rule, missing, n_total))
            continue

        labels = pd.to_numeric(frame[rule.pathology_column], errors="coerce")
        condition_values: list[tuple[ClinicalCondition, pd.Series, pd.Series]] = []
        valid = labels.isin([0, 1])
        for condition in conditions:
            name = condition.available_biomarker_name(frame)
            values = pd.to_numeric(frame[f"bm_{name}"], errors="coerce")
            thresholds = condition.threshold_series(frame)
            valid = (
                valid
                & values.replace([np.inf, -np.inf], np.nan).notna()
                & thresholds.replace([np.inf, -np.inf], np.nan).notna()
            )
            condition_values.append((condition, values, thresholds))

        if not bool(valid.any()):
            results.append(ClinicalRuleResult(
                rule_id=rule.rule_id,
                pathology_name=rule.pathology_name,
                biomarker_name=" + ".join(used_biomarker_names) or rule.biomarker_name,
                display_name=rule.display_name,
                threshold_label=rule.threshold_label,
                source_label=rule.source_label,
                source_url=rule.source_url,
                note=rule.note,
                status="no_evaluable_records",
                n_total=n_total,
                reference_sensitivity=rule.reference_sensitivity,
                reference_specificity=rule.reference_specificity,
                reference_ppv=rule.reference_ppv,
                reference_npv=rule.reference_npv,
                reference_accuracy=rule.reference_accuracy,
                reference_balanced_accuracy=rule.reference_balanced_accuracy,
                reference_auc=rule.reference_auc,
                reference_population=rule.reference_population,
                reference_note=rule.reference_note,
            ))
            continue

        pred = np.ones(int(valid.sum()), dtype=bool)
        for condition, values, thresholds in condition_values:
            condition_pred = condition.compare(
                values[valid].astype(float),
                thresholds[valid].astype(float),
            ).to_numpy(dtype=bool)
            pred &= condition_pred
        truth = labels[valid].astype(int).to_numpy() == 1

        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        tn = int(np.sum(~pred & ~truth))
        fn = int(np.sum(~pred & truth))
        n_evaluable = int(len(truth))
        n_positive = int(np.sum(truth))
        n_negative = n_evaluable - n_positive

        sensitivity = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)
        ppv = _safe_div(tp, tp + fp)
        npv = _safe_div(tn, tn + fn)
        accuracy = _safe_div(tp + tn, n_evaluable)
        balanced = None
        if sensitivity is not None and specificity is not None:
            balanced = float((sensitivity + specificity) / 2.0)
        lr_positive = None if specificity is None else _safe_div(sensitivity or 0.0, 1.0 - specificity)
        lr_negative = None if specificity is None or specificity == 0 else _safe_div(1.0 - (sensitivity or 0.0), specificity)
        status = "ok" if n_positive > 0 and n_negative > 0 else "insufficient_labels"

        results.append(ClinicalRuleResult(
            rule_id=rule.rule_id,
            pathology_name=rule.pathology_name,
            biomarker_name=" + ".join(used_biomarker_names) or rule.biomarker_name,
            display_name=rule.display_name,
            threshold_label=rule.threshold_label,
            source_label=rule.source_label,
            source_url=rule.source_url,
            note=rule.note,
            status=status,
            n_total=n_total,
            n_evaluable=n_evaluable,
            n_positive=n_positive,
            n_negative=n_negative,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            accuracy=accuracy,
            balanced_accuracy=balanced,
            lr_positive=lr_positive,
            lr_negative=lr_negative,
            reference_sensitivity=rule.reference_sensitivity,
            reference_specificity=rule.reference_specificity,
            reference_ppv=rule.reference_ppv,
            reference_npv=rule.reference_npv,
            reference_accuracy=rule.reference_accuracy,
            reference_balanced_accuracy=rule.reference_balanced_accuracy,
            reference_auc=rule.reference_auc,
            reference_population=rule.reference_population,
            reference_note=rule.reference_note,
        ))

    return results
