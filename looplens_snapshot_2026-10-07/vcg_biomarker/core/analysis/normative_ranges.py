from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


_AGE_BINS: list[tuple[int, int, str]] = [
    (18, 29, "age_18-29"),
    (30, 49, "age_30-49"),
    (50, 69, "age_50-69"),
    (70, 999, "age_70+"),
]
_MIN_N = 10  # minimum NORM records per stratum to compute percentiles


@dataclass
class NormativeStratum:
    """Percentile summary for one biomarker in one demographic stratum."""
    label: str
    n: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean: float
    std: float

    def to_dict(self) -> dict:
        return {
            "label": self.label, "n": self.n,
            "p5": self.p5, "p25": self.p25, "p50": self.p50,
            "p75": self.p75, "p95": self.p95,
            "mean": self.mean, "std": self.std,
        }


def _make_stratum(label: str, values: np.ndarray) -> Optional[NormativeStratum]:
    valid = values[np.isfinite(values)]
    if len(valid) < _MIN_N:
        return None
    return NormativeStratum(
        label=label,
        n=len(valid),
        p5=float(np.percentile(valid, 5)),
        p25=float(np.percentile(valid, 25)),
        p50=float(np.percentile(valid, 50)),
        p75=float(np.percentile(valid, 75)),
        p95=float(np.percentile(valid, 95)),
        mean=float(np.mean(valid)),
        std=float(np.std(valid)),
    )


class NormativeRangeStore:
    """
    Stores per-biomarker percentile distributions from the NORM cohort,
    stratified by sex and age group.

    Usage::

        store = NormativeRangeStore()
        store.compute(biomarker_data, age_array, sex_array, norm_mask)
        stratum = store.get("qrs_loop_area", sex="M", age=45)
        # stratum.p5 .. stratum.p95 are the reference range bounds
    """

    def __init__(self):
        # biomarker_name → list of NormativeStratum (one per stratum)
        self._data: dict[str, list[NormativeStratum]] = {}

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(
        self,
        biomarker_data: dict[str, np.ndarray],
        age_array: np.ndarray,
        sex_array: np.ndarray,
        norm_mask: np.ndarray,
    ) -> None:
        """
        Compute normative strata from the NORM sub-population.

        Args:
            biomarker_data: dict mapping biomarker name → float32 array (all records)
            age_array:  float32 array, NaN where age is unknown
            sex_array:  object (str) array, "" where sex is unknown ("M" / "F")
            norm_mask:  bool array, True for records classified as NORM
        """
        self._data.clear()
        norm_mask = np.asarray(norm_mask, dtype=bool)
        age_norm = np.asarray(age_array, dtype=np.float32)[norm_mask]
        sex_norm = np.asarray(sex_array, dtype=object)[norm_mask]

        sex_M = sex_norm == "M"
        sex_F = sex_norm == "F"

        for bname, arr in biomarker_data.items():
            arr_norm = np.asarray(arr, dtype=np.float32)[norm_mask]
            strata: list[NormativeStratum] = []

            # Overall
            s = _make_stratum("all", arr_norm)
            if s:
                strata.append(s)

            # By sex
            for mask, label in ((sex_M, "sex_M"), (sex_F, "sex_F")):
                s = _make_stratum(label, arr_norm[mask])
                if s:
                    strata.append(s)

            # By age group
            for lo, hi, label in _AGE_BINS:
                age_mask = np.isfinite(age_norm) & (age_norm >= lo) & (age_norm <= hi)
                s = _make_stratum(label, arr_norm[age_mask])
                if s:
                    strata.append(s)

            # Combined sex × age (only when n ≥ 30)
            for sex_mask, sex_label in ((sex_M, "sex_M"), (sex_F, "sex_F")):
                for lo, hi, age_label in _AGE_BINS:
                    age_mask = np.isfinite(age_norm) & (age_norm >= lo) & (age_norm <= hi)
                    combo_mask = sex_mask & age_mask
                    if combo_mask.sum() < 30:
                        continue
                    label = f"{sex_label}_{age_label}"
                    s = _make_stratum(label, arr_norm[combo_mask])
                    if s:
                        strata.append(s)

            if strata:
                self._data[bname] = strata

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self,
        biomarker: str,
        sex: Optional[str] = None,
        age: Optional[int] = None,
    ) -> Optional[NormativeStratum]:
        """
        Return the most specific stratum available for the given demographics.

        Fallback chain: sex+age → sex-only → age-only → "all".
        """
        strata = self._data.get(biomarker)
        if not strata:
            return None

        strata_by_label = {s.label: s for s in strata}

        age_label: Optional[str] = None
        if age is not None:
            for lo, hi, label in _AGE_BINS:
                if lo <= age <= hi:
                    age_label = label
                    break

        sex_label: Optional[str] = None
        if sex in ("M", "F"):
            sex_label = f"sex_{sex}"

        # Most specific → least specific
        for candidate in _candidates(sex_label, age_label):
            if candidate in strata_by_label:
                return strata_by_label[candidate]
        return None

    def get_overall(self, biomarker: str) -> Optional[NormativeStratum]:
        """Return the "all" stratum (no sex/age stratification)."""
        strata = self._data.get(biomarker)
        if not strata:
            return None
        for s in strata:
            if s.label == "all":
                return s
        return None

    def biomarkers(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> NormativeRangeStore:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NormativeRangeStore, got {type(obj)}")
        return obj


def _candidates(sex_label: Optional[str], age_label: Optional[str]) -> list[str]:
    """Priority list of stratum labels from most to least specific."""
    result = []
    if sex_label and age_label:
        result.append(f"{sex_label}_{age_label}")
    if sex_label:
        result.append(sex_label)
    if age_label:
        result.append(age_label)
    result.append("all")
    return result
