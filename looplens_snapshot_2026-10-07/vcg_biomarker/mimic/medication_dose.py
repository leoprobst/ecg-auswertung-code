"""Quantitative drug exposure: dose, route, infusion rate, actually given.

Until now the medication layer was binary — 28 feature columns, none of them
carrying a dose. For the ECG that throws away the part that matters most:
QT prolongation, AV conduction slowing and rate control are dose-dependent,
and a dose-response relationship is the single strongest piece of evidence
that an observed association is pharmacological rather than confounded by
indication. "On amiodarone yes/no" cannot show it; "QTc rose 6 ms per 100 mg"
can.

Three distinctions the sources support and this module keeps:

* **Prescribed versus given.** ``prescriptions`` records intent;
  ``emar_detail`` records administration, including an explicit
  ``complete_dose_not_given`` flag. Treating a prescription as an exposure
  counts drugs the patient never received.
* **Route.** Intravenous amiodarone and oral amiodarone have different
  time courses; pooling them blurs any temporal relationship to the ECG.
* **Bolus versus infusion.** ``inputevents`` carries a rate, which is the
  right exposure measure for a continuous drug — cumulative amount alone
  hides whether it arrived over ten minutes or ten hours.

De-identification redacts a share of dose strings to ``___``. Those rows are
counted as *dose unknown*, never as zero: a redacted dose is missing
information, and silently reading it as "no drug" would bias every
dose-response estimate towards the null.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


MEDICATION_DOSE_VERSION = "1.0.0"

# Redaction marker used throughout the de-identified free-text fields.
REDACTION_TOKEN = "___"

# Mass units normalised to milligrams. Volume and activity units are kept
# separate on purpose: millilitres of an unknown concentration are not a dose,
# and international units do not convert to mass.
_MASS_TO_MG: Mapping[str, float] = {
    "mg": 1.0,
    "milligram": 1.0,
    "milligrams": 1.0,
    "g": 1000.0,
    "gram": 1000.0,
    "grams": 1000.0,
    "gm": 1000.0,
    "mcg": 0.001,
    "ug": 0.001,
    "microgram": 0.001,
    "micrograms": 0.001,
    "ng": 1e-6,
}

_VOLUME_UNITS: frozenset[str] = frozenset({"ml", "millilitre", "milliliter", "l"})
_ACTIVITY_UNITS: frozenset[str] = frozenset({"unit", "units", "iu", "u"})

_RANGE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$")
_NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

DOSE_STATUS_NUMERIC = "numeric"
DOSE_STATUS_RANGE_MIDPOINT = "range_midpoint"
DOSE_STATUS_REDACTED = "redacted"
DOSE_STATUS_UNPARSEABLE = "unparseable"
DOSE_STATUS_MISSING = "missing"


@dataclass(frozen=True)
class ParsedDose:
    value: float
    status: str
    original: str


def parse_dose_value(text: object) -> ParsedDose:
    """Parse one free-text dose into a number, or say why it could not be.

    A range such as ``"5-10"`` becomes its midpoint and is labelled as such,
    so a downstream analysis can exclude midpoints from a dose-response fit if
    it wants a stricter exposure definition.
    """

    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ParsedDose(float("nan"), DOSE_STATUS_MISSING, "")
    raw = str(text).strip()
    if not raw:
        return ParsedDose(float("nan"), DOSE_STATUS_MISSING, raw)
    if REDACTION_TOKEN in raw:
        return ParsedDose(float("nan"), DOSE_STATUS_REDACTED, raw)
    try:
        return ParsedDose(float(raw), DOSE_STATUS_NUMERIC, raw)
    except ValueError:
        pass
    match = _RANGE_RE.match(raw)
    if match is not None:
        low, high = float(match.group(1)), float(match.group(2))
        return ParsedDose((low + high) / 2.0, DOSE_STATUS_RANGE_MIDPOINT, raw)
    numbers = _NUMBER_RE.findall(raw)
    if len(numbers) == 1:
        try:
            return ParsedDose(float(numbers[0]), DOSE_STATUS_NUMERIC, raw)
        except ValueError:
            pass
    return ParsedDose(float("nan"), DOSE_STATUS_UNPARSEABLE, raw)


def normalise_unit(unit: object) -> str:
    """Canonical lower-case unit token."""

    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    return str(unit).strip().casefold().rstrip(".")


def to_milligrams(value: float, unit: object) -> tuple[float, str]:
    """Convert a dose to milligrams where the unit permits it.

    Returns the converted value and a scale label. Volume and activity units
    are returned unconverted with their own scale, because ``10 mL`` of an
    unknown concentration and ``10 units`` of insulin are not masses and
    pretending otherwise would make a dose-response curve meaningless.
    """

    token = normalise_unit(unit)
    if not np.isfinite(value):
        return float("nan"), "unknown"
    if token in _MASS_TO_MG:
        return float(value) * _MASS_TO_MG[token], "mg"
    if token in _VOLUME_UNITS:
        return float(value), "ml"
    if token in _ACTIVITY_UNITS:
        return float(value), "units"
    return float(value), f"raw:{token}" if token else "raw:unknown"


def parse_dose_frame(
    frame: pd.DataFrame,
    *,
    value_column: str,
    unit_column: Optional[str] = None,
    prefix: str = "dose",
) -> pd.DataFrame:
    """Vectorised dose parsing with an explicit status column."""

    parsed = [parse_dose_value(value) for value in frame.get(value_column, [])]
    values = np.asarray([item.value for item in parsed], dtype=float)
    statuses = [item.status for item in parsed]
    units = (
        frame[unit_column]
        if unit_column and unit_column in frame.columns
        else pd.Series([""] * len(frame), index=frame.index)
    )
    converted = [
        to_milligrams(value, unit)
        for value, unit in zip(values, units.tolist())
    ]
    output = pd.DataFrame(index=frame.index)
    output[f"{prefix}_value"] = values
    output[f"{prefix}_status"] = statuses
    output[f"{prefix}_unit"] = [normalise_unit(unit) for unit in units.tolist()]
    output[f"{prefix}_normalised"] = [item[0] for item in converted]
    output[f"{prefix}_scale"] = [item[1] for item in converted]
    output[f"{prefix}_is_mass"] = [item[1] == "mg" for item in converted]
    # Redaction is missingness, never zero.
    output[f"{prefix}_redacted"] = [
        status == DOSE_STATUS_REDACTED for status in statuses
    ]
    output["medication_dose_version"] = MEDICATION_DOSE_VERSION
    return output


def administration_actually_given(
    complete_dose_not_given: Iterable[object],
    dose_value: Iterable[float],
) -> np.ndarray:
    """True where the record documents an administration that took place.

    ``complete_dose_not_given`` marks an order that was withheld. Counting
    those as exposure is the most common way a claims-style medication
    analysis overstates how many patients received a drug.
    """

    flags = pd.Series(list(complete_dose_not_given)).astype("string").str.casefold()
    not_given = flags.isin(("1", "true", "yes", "y")).fillna(False).to_numpy()
    doses = np.asarray(list(dose_value), dtype=float)
    return (~not_given) & (np.isfinite(doses) | ~np.isfinite(doses))


def exposure_summary(
    doses: Iterable[float],
    *,
    scales: Optional[Iterable[str]] = None,
    rates: Optional[Iterable[float]] = None,
    routes: Optional[Iterable[str]] = None,
) -> dict[str, object]:
    """Cumulative dose, peak rate and route mix inside one exposure window."""

    dose_array = np.asarray(list(doses), dtype=float)
    scale_list = list(scales) if scales is not None else []
    mass_mask = (
        np.asarray([scale == "mg" for scale in scale_list], dtype=bool)
        if scale_list
        else np.ones(len(dose_array), dtype=bool)
    )
    if len(mass_mask) != len(dose_array):
        mass_mask = np.ones(len(dose_array), dtype=bool)
    usable = np.isfinite(dose_array) & mass_mask
    rate_array = (
        np.asarray(list(rates), dtype=float)
        if rates is not None
        else np.empty(0, dtype=float)
    )
    route_series = (
        pd.Series(list(routes)).astype("string").str.upper()
        if routes is not None
        else pd.Series(dtype="string")
    )
    return {
        "n_administrations": int(len(dose_array)),
        "n_dose_known": int(usable.sum()),
        "dose_known_fraction": (
            float(usable.sum() / len(dose_array)) if len(dose_array) else 0.0
        ),
        "cumulative_dose_mg": (
            float(np.sum(dose_array[usable])) if usable.any() else float("nan")
        ),
        "max_single_dose_mg": (
            float(np.max(dose_array[usable])) if usable.any() else float("nan")
        ),
        "mean_dose_mg": (
            float(np.mean(dose_array[usable])) if usable.any() else float("nan")
        ),
        "peak_infusion_rate": (
            float(np.nanmax(rate_array))
            if rate_array.size and np.isfinite(rate_array).any()
            else float("nan")
        ),
        "any_intravenous": bool(route_series.isin(("IV", "IVPB")).any()),
        "any_oral": bool(route_series.isin(("PO", "NG", "PO/NG")).any()),
        "route_mix": "|".join(sorted(set(route_series.dropna().tolist()))[:6]),
        "dose_measurable": bool(usable.any()),
        "medication_dose_version": MEDICATION_DOSE_VERSION,
    }


def dose_response_levels(
    doses: Iterable[float],
    *,
    quantiles: Sequence[float] = (0.5,),
    labels: Sequence[str] = ("low", "high"),
) -> dict[str, object]:
    """Prespecified dose categories, with unknown dose kept as its own level.

    Patients whose dose was redacted or unparseable are labelled ``unknown``
    rather than pooled into the low group, so a dose-response gradient cannot
    be manufactured out of missingness.
    """

    values = np.asarray(list(doses), dtype=float)
    known = np.isfinite(values)
    levels = np.full(len(values), "unknown", dtype=object)
    if known.sum() < 10:
        return {
            "status": "not_evaluable",
            "levels": levels,
            "n_known": int(known.sum()),
        }
    if len(labels) != len(quantiles) + 1:
        raise ValueError("labels must have exactly one more entry than quantiles")
    cuts = np.quantile(values[known], list(quantiles))
    assigned = np.digitize(values[known], cuts, right=False)
    levels[known] = np.asarray(labels, dtype=object)[
        np.clip(assigned, 0, len(labels) - 1)
    ]
    return {
        "status": "evaluable",
        "levels": levels,
        "cut_points": [float(value) for value in np.atleast_1d(cuts)],
        "quantiles": list(quantiles),
        "n_known": int(known.sum()),
        "n_unknown": int((~known).sum()),
        "unknown_kept_separate": True,
        "medication_dose_version": MEDICATION_DOSE_VERSION,
    }


def dose_response_trend(
    doses: Iterable[float],
    response: Iterable[float],
    *,
    subject_ids: Optional[Iterable[object]] = None,
) -> dict[str, object]:
    """Linear dose-response slope with subject-clustered robust errors.

    A monotone gradient across dose is what separates a pharmacological effect
    from confounding by indication: sicker patients get the drug, but there is
    no reason for them to get systematically *more* of it in proportion to an
    ECG change unless the drug is doing something.
    """

    import statsmodels.api as sm

    dose_array = np.asarray(list(doses), dtype=float)
    response_array = np.asarray(list(response), dtype=float)
    usable = np.isfinite(dose_array) & np.isfinite(response_array)
    base: dict[str, object] = {
        "status": "not_evaluable",
        "n": int(usable.sum()),
        "slope_per_unit_dose": float("nan"),
        "p_value": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "medication_dose_version": MEDICATION_DOSE_VERSION,
    }
    if usable.sum() < 30 or np.unique(dose_array[usable]).size < 3:
        base["failure_reason"] = "insufficient_dose_variation"
        return base
    design = sm.add_constant(dose_array[usable])
    model = sm.OLS(response_array[usable], design)
    if subject_ids is not None:
        groups = pd.Series(list(subject_ids))[usable].to_numpy()
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        inference = "ols_subject_cluster_robust"
    else:
        result = model.fit(cov_type="HC3")
        inference = "ols_heteroskedasticity_robust"
    confidence = result.conf_int()
    base.update(
        {
            "status": "evaluable",
            "failure_reason": "",
            "slope_per_unit_dose": float(result.params[1]),
            "std_error": float(result.bse[1]),
            "p_value": float(result.pvalues[1]),
            "ci_low": float(confidence[1][0]),
            "ci_high": float(confidence[1][1]),
            "inference_method": inference,
            "interpretation": (
                "a monotone gradient supports a pharmacological effect; it "
                "does not by itself establish one, because dose also tracks "
                "severity"
            ),
        }
    )
    return base


__all__ = [
    "DOSE_STATUS_MISSING",
    "DOSE_STATUS_NUMERIC",
    "DOSE_STATUS_RANGE_MIDPOINT",
    "DOSE_STATUS_REDACTED",
    "DOSE_STATUS_UNPARSEABLE",
    "MEDICATION_DOSE_VERSION",
    "ParsedDose",
    "REDACTION_TOKEN",
    "administration_actually_given",
    "dose_response_levels",
    "dose_response_trend",
    "exposure_summary",
    "normalise_unit",
    "parse_dose_frame",
    "parse_dose_value",
    "to_milligrams",
]
