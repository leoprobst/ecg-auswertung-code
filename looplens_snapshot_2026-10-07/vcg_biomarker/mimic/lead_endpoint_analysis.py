"""Adapter from the reference-blind validation universe to paired lead models."""
from __future__ import annotations

import pandas as pd
import numpy as np

from .lead_comparison import compare_leads
from .lead_measurements import empty_lead_measurements
from .run_log import profiled_phase

LEAD_ENDPOINT_ADAPTER_VERSION = "1.2.0"
LEAD_FAMILIES = ("qrs", "p", "st", "t")
LEAD_COVARIATES = ("lead_adjust_age", "lead_adjust_sex_male", "lead_adjust_heart_rate",
                   "lead_adjust_quality", "lead_adjust_ed", "lead_adjust_inpatient_without_ed")
LEAD_INPUT_COLUMNS = (*empty_lead_measurements(), "context_heart_rate_bpm", "signal_quality_score", "extraction_status")
LEAD_CONTEXT_COLUMNS = ("subject_id", "split", "label_state", "eligibility_state",
                        "reference_kind", "age", "sex", "ed_stay_id", "hadm_id")


def _project_lead_inputs(universe: pd.DataFrame) -> pd.DataFrame:
    """Copy only consumed inputs; preserve rows, index, missingness and order."""
    required = set(LEAD_INPUT_COLUMNS) | set(LEAD_CONTEXT_COLUMNS)
    return universe.loc[:, [name for name in universe.columns if name in required]].copy()


def prepare_lead_frame(universe: pd.DataFrame, *, lead_only: bool = False) -> pd.DataFrame:
    # Voltage/mass also uses this covariate preparation and needs its own
    # echo/voltage inputs. Projection is opt-in only for the lead consumer.
    output = _project_lead_inputs(universe) if lead_only else universe.copy()
    def numeric(name):
        return pd.to_numeric(output.get(name, pd.Series(np.nan, index=output.index)), errors="coerce")
    output["reference_eligible"] = output.eligibility_state.eq("eligible")
    output["lead_adjust_age"] = numeric("age")
    output["lead_adjust_sex_male"] = output.get("sex", pd.Series(None, index=output.index, dtype=object)).astype("string").str.upper().map({"M": 1., "F": 0.})
    output["lead_adjust_heart_rate"] = numeric("context_heart_rate_bpm")
    output["lead_adjust_quality"] = numeric("signal_quality_score")
    # Reference category is unlinked to ED/admission, NOT proven outpatient.
    if {"ed_stay_id", "hadm_id"} <= set(output):
        ed = output.ed_stay_id.notna()
        output["lead_adjust_ed"] = ed.astype(float)
        output["lead_adjust_inpatient_without_ed"] = (output.hadm_id.notna() & ~ed).astype(float)
    else:
        output["lead_adjust_ed"] = np.nan
        output["lead_adjust_inpatient_without_ed"] = np.nan
    if "extraction_status" not in output:
        output["extraction_status"] = "unavailable"
    return output


@profiled_phase("lead_comparison_models_and_bootstrap")
def analyze_endpoint_leads(universe: pd.DataFrame, *, endpoint_id: str, iterations: int,
                           seed: int, preview: bool) -> tuple[list[dict], list[dict]]:
    frame = prepare_lead_frame(universe, lead_only=True)
    reference = sorted(set(frame.get("reference_kind", pd.Series("unknown", index=frame.index)).dropna().astype(str)))
    rows, models = [], []
    for family in LEAD_FAMILIES:
        result = compare_leads(frame, endpoint_id=endpoint_id, cohort_id="mimic_development",
            reference_kind="|".join(reference) or "unknown", family=family,
            covariates=LEAD_COVARIATES, iterations=iterations, seed=seed, confirm_association=not preview)
        table = result.results
        table["reference_alignment_confirmed"] = False  # no independent clock validation from this adapter
        table["care_setting_reference_category"] = "no_linked_ed_or_admission_not_proven_outpatient"
        table["sampling_scope"] = "pilot_enriched_descriptive" if preview else "development_index_cohort"
        table["calibrated_probability_claim_allowed"] = False
        table["adapter_version"] = LEAD_ENDPOINT_ADAPTER_VERSION
        rows.extend(table.to_dict("records"))
        models.append({"endpoint_id": endpoint_id, "family": family, "models": result.models,
                       "covariates": list(LEAD_COVARIATES), "adapter_version": LEAD_ENDPOINT_ADAPTER_VERSION})
    return rows, models
