from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


def _safe_round(value, ndigits: int = 4):
    if value is None:
        return None
    try:
        if not np.isfinite(value):
            return None
    except TypeError:
        return value
    return round(float(value), ndigits)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        if num == 0:
            return 0.0
        return float("inf")
    return float(num / den)


@dataclass
class SweepResult:
    biomarker_name: str
    pathology_name: str
    n_positive: int
    n_negative: int
    n_total: int
    auc: float
    optimal_threshold: float
    optimal_sensitivity: float
    optimal_specificity: float
    optimal_ppv: float
    optimal_npv: float
    optimal_accuracy: float
    youden_j: float
    roc_thresholds: np.ndarray
    roc_tpr: np.ndarray
    roc_fpr: np.ndarray
    p_value: float
    p_value_corrected: Optional[float] = None
    correction_method: str = ""
    ci_sensitivity: Optional[tuple[float, float]] = None
    ci_specificity: Optional[tuple[float, float]] = None
    ci_auc: Optional[tuple[float, float]] = None
    # +1: higher value → pathological (standard); -1: lower value → pathological (inverted)
    direction: float = 1.0
    threshold_direction: str = ""
    threshold_status: str = "exploratory"
    dataset: str = "PTB-XL"
    cohort_id: str = "default"
    label_source: str = "SCP statements"
    reference_standard: str = "ECG report labels"
    missing_rate: float = 0.0
    quality_summary: dict = field(default_factory=dict)
    effect_size: float = 0.0
    auprc: float = 0.0
    subgroup_metrics: dict = field(default_factory=dict)
    validation_split: str = "development"
    baseline_auc: Optional[float] = None
    normative_p5: Optional[float] = None   # P5  aus NORM-Kohorte (alle Strata)
    normative_p95: Optional[float] = None  # P95 aus NORM-Kohorte (alle Strata)

    def __post_init__(self):
        if not self.threshold_direction:
            self.threshold_direction = "above" if self.direction >= 0 else "below"
        if "+" in self.biomarker_name and self.threshold_status == "exploratory":
            self.threshold_status = "exploratory_score"

    @property
    def is_significant(self) -> bool:
        p = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        return p < 0.05

    @property
    def prevalence(self) -> float:
        return _safe_div(self.n_positive, self.n_total) if self.n_total else 0.0

    @property
    def lr_positive(self) -> float:
        return _safe_div(self.optimal_sensitivity, 1.0 - self.optimal_specificity)

    @property
    def lr_negative(self) -> float:
        return _safe_div(1.0 - self.optimal_sensitivity, self.optimal_specificity)

    @property
    def balanced_accuracy(self) -> float:
        return float((self.optimal_sensitivity + self.optimal_specificity) / 2.0)

    @property
    def baseline_delta_auc(self) -> Optional[float]:
        if self.baseline_auc is None:
            return None
        return float(self.auc - self.baseline_auc)

    @property
    def stability_score(self) -> float:
        if self.n_total <= 0:
            return 0.0
        ci_width = 1.0
        if self.ci_auc is not None:
            ci_width = max(0.0, float(self.ci_auc[1] - self.ci_auc[0]))
        precision = max(0.0, min(1.0, 1.0 - ci_width / 0.5))
        event_balance = min(1.0, min(self.n_positive, self.n_negative) / 100.0)
        sample_size = min(1.0, self.n_total / 1000.0)
        return float(0.55 * precision + 0.30 * event_balance + 0.15 * sample_size)

    def confusion_counts(self, sensitivity: Optional[float] = None,
                         specificity: Optional[float] = None) -> dict[str, int]:
        sens = self.optimal_sensitivity if sensitivity is None else sensitivity
        spec = self.optimal_specificity if specificity is None else specificity
        tp = int(round(sens * self.n_positive))
        fn = max(0, self.n_positive - tp)
        tn = int(round(spec * self.n_negative))
        fp = max(0, self.n_negative - tn)
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def predictive_values_at_prevalence(self, prevalence: float) -> tuple[float, float]:
        prev = min(max(float(prevalence), 0.0), 1.0)
        sens = self.optimal_sensitivity
        spec = self.optimal_specificity
        ppv_den = sens * prev + (1.0 - spec) * (1.0 - prev)
        npv_den = spec * (1.0 - prev) + (1.0 - sens) * prev
        ppv = (sens * prev / ppv_den) if ppv_den > 0 else 0.0
        npv = (spec * (1.0 - prev) / npv_den) if npv_den > 0 else 0.0
        return float(ppv), float(npv)

    @property
    def summary_dict(self) -> dict:
        return {
            "biomarker": self.biomarker_name,
            "pathology": self.pathology_name,
            "dataset": self.dataset,
            "cohort_id": self.cohort_id,
            "label_source": self.label_source,
            "reference_standard": self.reference_standard,
            "validation_split": self.validation_split,
            "n_total": self.n_total,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "prevalence": _safe_round(self.prevalence, 4),
            "missing_rate": _safe_round(self.missing_rate, 4),
            "auc": _safe_round(self.auc, 4),
            "auc_ci_low": _safe_round(self.ci_auc[0], 4) if self.ci_auc else None,
            "auc_ci_high": _safe_round(self.ci_auc[1], 4) if self.ci_auc else None,
            "auprc": _safe_round(self.auprc, 4),
            "threshold": _safe_round(self.optimal_threshold, 4),
            "threshold_direction": self.threshold_direction,
            "threshold_status": self.threshold_status,
            "sensitivity": _safe_round(self.optimal_sensitivity, 4),
            "sensitivity_ci_low": _safe_round(self.ci_sensitivity[0], 4) if self.ci_sensitivity else None,
            "sensitivity_ci_high": _safe_round(self.ci_sensitivity[1], 4) if self.ci_sensitivity else None,
            "specificity": _safe_round(self.optimal_specificity, 4),
            "specificity_ci_low": _safe_round(self.ci_specificity[0], 4) if self.ci_specificity else None,
            "specificity_ci_high": _safe_round(self.ci_specificity[1], 4) if self.ci_specificity else None,
            "ppv": _safe_round(self.optimal_ppv, 4),
            "npv": _safe_round(self.optimal_npv, 4),
            "lr_positive": _safe_round(self.lr_positive, 4),
            "lr_negative": _safe_round(self.lr_negative, 4),
            "accuracy": _safe_round(self.optimal_accuracy, 4),
            "balanced_accuracy": _safe_round(self.balanced_accuracy, 4),
            "youden_j": _safe_round(self.youden_j, 4),
            "effect_size": _safe_round(self.effect_size, 4),
            "stability": _safe_round(self.stability_score, 4),
            "baseline_auc": _safe_round(self.baseline_auc, 4),
            "baseline_delta_auc": _safe_round(self.baseline_delta_auc, 4),
            "norm_p5": _safe_round(self.normative_p5, 4),
            "norm_p95": _safe_round(self.normative_p95, 4),
            "p_value": _safe_round(self.p_value, 6),
            "p_corrected": _safe_round(self.p_value_corrected, 6) if self.p_value_corrected is not None else None,
            "significant": self.is_significant,
        }

    def to_json_dict(self, include_curves: bool = True) -> dict:
        data = self.summary_dict.copy()
        data.update({
            "correction_method": self.correction_method,
            "direction": self.direction,
            "quality_summary": self.quality_summary,
            "subgroup_metrics": self.subgroup_metrics,
            "normative_p5": self.normative_p5,
            "normative_p95": self.normative_p95,
            # Override the 6-decimal-place rounded values from summary_dict with full
            # precision so that tiny p-values (e.g. 1e-50) survive the roundtrip.
            # summary_dict rounds to 0.0 and from_json_dict would read that as 1.0.
            "p_value": self.p_value,
            "p_corrected": self.p_value_corrected,
        })
        if include_curves:
            data.update({
                "roc_thresholds": self.roc_thresholds.tolist(),
                "roc_tpr": self.roc_tpr.tolist(),
                "roc_fpr": self.roc_fpr.tolist(),
            })
        return data

    @classmethod
    def from_json_dict(cls, data: dict) -> "SweepResult":
        return cls(
            biomarker_name=data.get("biomarker", data.get("biomarker_name", "")),
            pathology_name=data.get("pathology", data.get("pathology_name", "")),
            n_positive=int(data.get("n_positive", 0) or 0),
            n_negative=int(data.get("n_negative", 0) or 0),
            n_total=int(data.get("n_total", 0) or 0),
            auc=float(data.get("auc", 0.0) or 0.0),
            optimal_threshold=float(data.get("threshold", data.get("optimal_threshold", 0.0)) or 0.0),
            optimal_sensitivity=float(data.get("sensitivity", data.get("optimal_sensitivity", 0.0)) or 0.0),
            optimal_specificity=float(data.get("specificity", data.get("optimal_specificity", 0.0)) or 0.0),
            optimal_ppv=float(data.get("ppv", data.get("optimal_ppv", 0.0)) or 0.0),
            optimal_npv=float(data.get("npv", data.get("optimal_npv", 0.0)) or 0.0),
            optimal_accuracy=float(data.get("accuracy", data.get("optimal_accuracy", 0.0)) or 0.0),
            youden_j=float(data.get("youden_j", 0.0) or 0.0),
            roc_thresholds=np.array(data.get("roc_thresholds", []), dtype=float),
            roc_tpr=np.array(data.get("roc_tpr", []), dtype=float),
            roc_fpr=np.array(data.get("roc_fpr", []), dtype=float),
            p_value=_parse_p_value(data.get("p_value")),
            p_value_corrected=_optional_float(data.get("p_corrected", data.get("p_value_corrected"))),
            correction_method=data.get("correction_method", ""),
            ci_sensitivity=_tuple_ci(data, "sensitivity"),
            ci_specificity=_tuple_ci(data, "specificity"),
            ci_auc=_tuple_ci(data, "auc"),
            direction=float(data.get("direction", 1.0) or 1.0),
            threshold_direction=data.get("threshold_direction", ""),
            threshold_status=data.get("threshold_status", "exploratory"),
            dataset=data.get("dataset", "PTB-XL"),
            cohort_id=data.get("cohort_id", "default"),
            label_source=data.get("label_source", "SCP statements"),
            reference_standard=data.get("reference_standard", "ECG report labels"),
            missing_rate=float(data.get("missing_rate", 0.0) or 0.0),
            quality_summary=data.get("quality_summary", {}) or {},
            effect_size=float(data.get("effect_size", 0.0) or 0.0),
            auprc=float(data.get("auprc", 0.0) or 0.0),
            subgroup_metrics=data.get("subgroup_metrics", {}) or {},
            validation_split=data.get("validation_split", "development"),
            baseline_auc=_optional_float(data.get("baseline_auc")),
            normative_p5=_optional_float(data.get("normative_p5", data.get("norm_p5"))),
            normative_p95=_optional_float(data.get("normative_p95", data.get("norm_p95"))),
        )

    def evidence_narrative(self) -> str:
        ci = ""
        if self.ci_auc:
            ci = f" (95% CI {self.ci_auc[0]:.3f}-{self.ci_auc[1]:.3f})"
        q_value = self.p_value_corrected if self.p_value_corrected is not None else self.p_value
        comparator = ""
        if self.baseline_delta_auc is not None:
            comparator = f" Compared with the configured baseline, delta AUC is {self.baseline_delta_auc:+.3f}."
        return (
            f"{self.biomarker_name} is an {self.threshold_status} candidate for {self.pathology_name} "
            f"in cohort '{self.cohort_id}' ({self.dataset}, {self.validation_split}). "
            f"The learned {self.threshold_direction}-threshold is {self.optimal_threshold:.4f}, with "
            f"AUC {self.auc:.3f}{ci}, sensitivity {self.optimal_sensitivity:.3f}, "
            f"specificity {self.optimal_specificity:.3f}, LR+ {_safe_round(self.lr_positive, 2)}, "
            f"LR- {_safe_round(self.lr_negative, 2)}, and q={q_value:.3g}. "
            f"The observed prevalence is {self.prevalence:.1%} ({self.n_positive}/{self.n_total})."
            f"{comparator} This result should be treated as research-use evidence until validated on an "
            f"independent or temporally separated cohort."
        )


def _optional_float(value) -> Optional[float]:
    if value in (None, "", "None", "nan"):
        return None
    value = float(value)
    if not np.isfinite(value):
        return None


def _parse_p_value(value) -> float:
    """Parse a p-value from JSON/CSV without the falsy-zero trap.

    The bug: ``float(0.0) or 1.0`` evaluates to 1.0 because 0.0 is falsy.
    Very small p-values (e.g. 1e-50) are rounded to 0.0 when stored with
    6 decimal places, then deserialized as 1.0 — making highly significant
    markers appear non-significant.
    """
    if value is None or value in ("", "None", "nan"):
        return 1.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(v) or v < 0:
        return 1.0
    return v
    return value


def _tuple_ci(data: dict, prefix: str) -> Optional[tuple[float, float]]:
    lo = data.get(f"{prefix}_ci_low")
    hi = data.get(f"{prefix}_ci_high")
    legacy = data.get(f"ci_{prefix}")
    if legacy is not None and len(legacy) == 2:
        return (float(legacy[0]), float(legacy[1]))
    if lo is None or hi is None:
        return None
    return (float(lo), float(hi))


def compute_roc_curve(biomarker_values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve assuming higher biomarker value → more likely pathological.

    Thresholds are sorted DESCENDING so FPR increases monotonically as threshold
    decreases (more samples classified positive). Sentinel points:
      - prepend high sentinel  → threshold above all values → FPR=0, TPR=0
      - append  low  sentinel  → threshold below all values → FPR=1, TPR=1
    """
    positive_vals = biomarker_values[labels == 1]
    negative_vals = biomarker_values[labels == 0]

    n_pos = len(positive_vals)
    n_neg = len(negative_vals)

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])

    all_vals = np.concatenate([positive_vals, negative_vals])
    thresholds = np.percentile(all_vals, np.linspace(0, 100, 101))
    thresholds = np.unique(thresholds)
    # Descending: high threshold first (FPR≈0), low threshold last (FPR≈1)
    thresholds = np.sort(thresholds)[::-1]

    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        tp = np.sum(positive_vals >= thresh)
        fp = np.sum(negative_vals >= thresh)

        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    # Sentinels: prepend (FPR=0, TPR=0) for threshold above all data,
    #            append  (FPR=1, TPR=1) for threshold below all data.
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    thresholds = np.concatenate([[thresholds[0] + 1.0], thresholds, [thresholds[-1] - 1.0]])

    return fpr, tpr, thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    return float(np.trapezoid(tpr_sorted, fpr_sorted))


def compute_auprc(biomarker_values: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return 0.0

    order = np.argsort(-biomarker_values)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)

    prevalence = n_pos / max(len(labels), 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[prevalence], precision])
    return float(np.trapezoid(precision, recall))


def compute_effect_size_values(biomarker_values: np.ndarray, labels: np.ndarray) -> float:
    positive_vals = biomarker_values[labels == 1]
    negative_vals = biomarker_values[labels == 0]
    if len(positive_vals) < 2 or len(negative_vals) < 2:
        return 0.0

    n1 = len(positive_vals)
    n2 = len(negative_vals)
    var1 = np.var(positive_vals, ddof=1)
    var2 = np.var(negative_vals, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    if pooled < 1e-10:
        return 0.0
    return float((np.mean(positive_vals) - np.mean(negative_vals)) / pooled)


def find_optimal_threshold(
    biomarker_values: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    positive_vals = biomarker_values[labels == 1]
    negative_vals = biomarker_values[labels == 0]

    best_j = -1
    best_thresh = 0
    best_sens = 0
    best_spec = 0

    for thresh in thresholds:
        tp = np.sum(positive_vals >= thresh)
        fp = np.sum(negative_vals >= thresh)
        fn = n_pos - tp
        tn = n_neg - fp

        sens = tp / n_pos if n_pos > 0 else 0
        spec = tn / n_neg if n_neg > 0 else 0

        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = thresh
            best_sens = sens
            best_spec = spec

    tp = np.sum(positive_vals >= best_thresh)
    fp = np.sum(negative_vals >= best_thresh)
    fn = n_pos - tp
    tn = n_neg - fp

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else 0

    return best_thresh, best_sens, best_spec, ppv, npv, accuracy, best_j


def compute_p_value(biomarker_values: np.ndarray, labels: np.ndarray) -> float:
    import warnings

    positive_vals = biomarker_values[labels == 1]
    negative_vals = biomarker_values[labels == 0]

    if len(positive_vals) < 2 or len(negative_vals) < 2:
        return 1.0

    # Extreme ties: all values identical across both groups → AUC=0.5, no discrimination.
    # scipy's tie correction in mannwhitneyu produces sqrt(-x) → RuntimeWarning + NaN p.
    all_vals = np.concatenate([positive_vals, negative_vals])
    if len(np.unique(all_vals)) == 1:
        return 1.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, p = sp_stats.mannwhitneyu(positive_vals, negative_vals, alternative="two-sided")
        p = float(p)
        return p if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def bootstrap_confidence_interval(
    biomarker_values: np.ndarray,
    labels: np.ndarray,
    metric_func,
    n_iterations: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    n = len(biomarker_values)
    boot_metrics = []

    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        bv = biomarker_values[idx]
        lb = labels[idx]

        if len(np.unique(lb)) < 2:
            continue

        try:
            m = metric_func(bv, lb)
            boot_metrics.append(m)
        except Exception:
            continue

    if len(boot_metrics) == 0:
        return (0.0, 0.0)

    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(boot_metrics, alpha * 100))
    upper = float(np.percentile(boot_metrics, (1 - alpha) * 100))

    return (lower, upper)


def quick_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC-only, no bootstrap — used for greedy screening sweeps."""
    valid = np.isfinite(values) & np.isfinite(labels.astype(float))
    bv = values[valid]
    lb = labels[valid]
    if len(bv) < 10 or len(np.unique(lb)) < 2:
        return 0.5
    fpr, tpr, _ = compute_roc_curve(bv, lb)
    auc = compute_auc(fpr, tpr)
    return float(max(auc, 1.0 - auc))


def threshold_sweep(
    biomarker_values: np.ndarray,
    labels: np.ndarray,
    biomarker_name: str,
    pathology_name: str,
    bootstrap_iterations: int = 1000,
) -> SweepResult:
    raw_n = len(biomarker_values)
    valid_mask = np.isfinite(biomarker_values) & np.isfinite(labels.astype(float))
    bv = biomarker_values[valid_mask]
    lb = labels[valid_mask]
    missing_rate = 1.0 - (len(bv) / raw_n) if raw_n else 0.0

    n_total = len(bv)
    n_positive = int(np.sum(lb == 1))
    n_negative = int(np.sum(lb == 0))

    if n_positive < 5 or n_negative < 5:
        return SweepResult(
            biomarker_name=biomarker_name,
            pathology_name=pathology_name,
            n_positive=n_positive,
            n_negative=n_negative,
            n_total=n_total,
            auc=0.0,
            optimal_threshold=0.0,
            optimal_sensitivity=0.0,
            optimal_specificity=0.0,
            optimal_ppv=0.0,
            optimal_npv=0.0,
            optimal_accuracy=0.0,
            youden_j=0.0,
            roc_thresholds=np.array([]),
            roc_tpr=np.array([]),
            roc_fpr=np.array([]),
            p_value=1.0,
            missing_rate=missing_rate,
            quality_summary={
                "minimum_class_size_met": False,
                "reason": "Need at least 5 positive and 5 negative records",
            },
        )

    fpr, tpr, thresholds = compute_roc_curve(bv, lb)
    auc = compute_auc(fpr, tpr)

    # Inverted biomarker: lower value → pathological (e.g. ejection fraction).
    # Flip sign so that "higher flipped value = more pathological" holds,
    # then flip the optimal threshold back to the original scale.
    direction = 1.0
    bv_analysis = bv
    if auc < 0.5:
        direction = -1.0
        bv_analysis = -bv
        fpr, tpr, thresholds = compute_roc_curve(bv_analysis, lb)
        auc = compute_auc(fpr, tpr)

    opt_thresh_analysis, sens, spec, ppv, npv, acc, youden = find_optimal_threshold(
        bv_analysis, lb, thresholds
    )
    # Threshold in original biomarker space
    opt_thresh = direction * opt_thresh_analysis

    # ROC thresholds stored in original space for interpretable display
    stored_thresholds = direction * thresholds

    p_value = compute_p_value(bv, lb)
    effect_size = compute_effect_size_values(bv, lb)
    auprc = compute_auprc(bv_analysis, lb)

    ci_auc = bootstrap_confidence_interval(
        bv_analysis, lb,
        lambda b, l: compute_auc(*compute_roc_curve(b, l)[:2]),
        n_iterations=bootstrap_iterations,
    )

    ci_sens = bootstrap_confidence_interval(
        bv_analysis, lb,
        lambda b, l: find_optimal_threshold(b, l, compute_roc_curve(b, l)[2])[1],
        n_iterations=bootstrap_iterations,
    )

    ci_spec = bootstrap_confidence_interval(
        bv_analysis, lb,
        lambda b, l: find_optimal_threshold(b, l, compute_roc_curve(b, l)[2])[2],
        n_iterations=bootstrap_iterations,
    )

    return SweepResult(
        biomarker_name=biomarker_name,
        pathology_name=pathology_name,
        n_positive=n_positive,
        n_negative=n_negative,
        n_total=n_total,
        auc=auc,
        optimal_threshold=opt_thresh,
        optimal_sensitivity=sens,
        optimal_specificity=spec,
        optimal_ppv=ppv,
        optimal_npv=npv,
        optimal_accuracy=acc,
        youden_j=youden,
        roc_thresholds=stored_thresholds,
        roc_tpr=tpr,
        roc_fpr=fpr,
        p_value=p_value,
        ci_auc=ci_auc,
        ci_sensitivity=ci_sens,
        ci_specificity=ci_spec,
        direction=direction,
        missing_rate=missing_rate,
        effect_size=effect_size,
        auprc=auprc,
        quality_summary={
            "minimum_class_size_met": True,
            "bootstrap_iterations": bootstrap_iterations,
        },
    )
