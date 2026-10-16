"""Exploratory rank regression with patient-clustered CR1 uncertainty."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import rankdata, t

from .patient_identity import known_patient_mask


def clustered_rank_association(x, y, subjects, *, minimum_patients=30):
    """Slope on standardized empirical ranks equals descriptive Spearman rho.

    CI/p refer to the rank-regression slope conditional on observed ranks,
    not an exact distribution-free population Spearman interval. Patients are
    the independent sampling units. Time-varying confounding is not adjusted.
    """
    if type(minimum_patients) is not int or minimum_patients < 3:
        raise ValueError("At least three independent patient clusters required")
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite, equal-length one-dimensional rank inputs required")
    return _evaluate(_prepare(x, subjects), y, minimum_patients)


@dataclass(frozen=True)
class _Prepared:
    n: int
    rx: np.ndarray | None
    xx: float
    codes: np.ndarray | None
    g: int


def _prepare(x, subjects):
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return _Prepared(n, None, 0., None, 0)
    rx = rankdata(x)
    rx = (rx-rx.mean())/rx.std()
    if subjects is None:
        return _Prepared(n, rx, float(rx @ rx), None, 0)
    subjects = pd.Series(subjects).reset_index(drop=True)
    if len(subjects) != n:
        raise ValueError("Patient vector must match rank input length")
    if not known_patient_mask(subjects).all():
        return _Prepared(n, rx, float(rx @ rx), None, 0)
    codes, patients = pd.factorize(subjects, sort=False)
    return _Prepared(n, rx, float(rx @ rx), codes, len(patients))


def _evaluate(prepared, y, minimum_patients):
    result = dict(association_status="non_estimable", abstention_reason="unknown_patient_identity",
        p_value=np.nan, rank_slope_ci_low=np.nan, rank_slope_ci_high=np.nan,
        rank_slope_se=np.nan, n_patients=0, spearman_rho=np.nan,
        inference_method="patient_clustered_CR1_standardized_rank_OLS_t_Gminus1",
        inference_limit="conditional_empirical_rank_regression_not_exact_spearman_inference",
        adjustment_status="unadjusted_for_time_varying_confounders",
        clinical_claim_allowed=False)
    if prepared.rx is None or np.ptp(y) == 0:
        result["abstention_reason"] = "insufficient_or_constant_values"
        return result
    rx, ry = prepared.rx, rankdata(y)
    ry = (ry-ry.mean())/ry.std()
    xx = prepared.xx
    slope = float(rx @ ry / xx)
    result["spearman_rho"] = slope
    if prepared.codes is None:
        return result
    codes, g, n = prepared.codes, prepared.g, prepared.n
    result["n_patients"] = g
    if g < minimum_patients:
        result["abstention_reason"] = "too_few_independent_patients"
        return result
    residual = ry-slope*rx
    scores = np.bincount(codes, weights=rx*residual, minlength=g)
    variance = (g/(g-1))*((n-1)/(n-2))*float(scores @ scores)/(xx*xx)
    if not np.isfinite(variance) or variance <= np.finfo(float).eps**2:
        result["abstention_reason"] = "degenerate_cluster_uncertainty"
        return result
    se = float(np.sqrt(variance))
    critical = float(t.ppf(.975, g-1))
    result.update(association_status="estimable", abstention_reason="",
        p_value=float(2*t.sf(abs(slope/se), g-1)), rank_slope_se=se,
        rank_slope_ci_low=slope-critical*se, rank_slope_ci_high=slope+critical*se)
    return result


class ClusteredRankBatch:
    """Reuse exposure ranks and cluster codes for consecutive identical masks.

    Cache exactly one prepared cohort, bounded independently of marker count.
    Different missingness always reranks/reclusters the actual complete cases.
    Instances own their inputs and belong to one analysis, never global state.
    """

    def __init__(self, x, subjects, *, minimum_patients=30):
        if type(minimum_patients) is not int or minimum_patients < 3:
            raise ValueError("At least three independent patient clusters required")
        self.x = np.array(x, dtype=float, copy=True)
        if self.x.ndim != 1:
            raise ValueError("One-dimensional exposure required")
        self.subjects = None if subjects is None else pd.Series(subjects).reset_index(drop=True).copy()
        if self.subjects is not None and len(self.subjects) != len(self.x):
            raise ValueError("Patient vector must match exposure length")
        self.finite_x = np.isfinite(self.x)
        self.minimum_patients = minimum_patients
        self._mask = None
        self._prepared = None
        self.cache_hits = 0
        self.preparations = 0

    def estimate(self, y, *, minimum_pairs=30):
        y = np.asarray(y, dtype=float)
        if y.shape != self.x.shape:
            raise ValueError("Marker vector must match exposure length")
        mask = self.finite_x & np.isfinite(y)
        if int(mask.sum()) < minimum_pairs:
            return None, mask
        if self._mask is None or not np.array_equal(mask, self._mask):
            self._prepared = _prepare(self.x[mask],
                self.subjects[mask] if self.subjects is not None else None)
            self._mask = mask.copy()
            self.preparations += 1
        else:
            self.cache_hits += 1
        return _evaluate(self._prepared, y[mask], self.minimum_patients), mask
