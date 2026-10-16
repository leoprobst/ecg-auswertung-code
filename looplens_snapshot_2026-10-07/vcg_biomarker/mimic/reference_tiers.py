"""Reference-standard quality, graded — and a sampling plan to check it.

Diagnostic accuracy is only ever accuracy *against something*. This project's
63 endpoints are anchored to very different somethings: a quantitative echo
measurement taken near the ECG is not the same class of evidence as a single
billing code attached to an admission, yet both currently flow into the same
metrics with the same wording.

Three tiers, following the usual reference-standard hierarchy:

* **Tier A** — an observed, temporally eligible quantitative measurement.
  Measurement error, temporal mismatch and selective verification still matter.
* **Tier B** — agreement of several independent sources (e.g. an ICD code plus
  a matching medication plus a matching laboratory value).
* **Tier C** — one administrative code or one text mention. Sensitivity of
  such phenotypes is often well below 0.8, and the error is frequently
  differential with respect to how sick or how observed a patient was.

Grading matters because imperfect reference standards bias accuracy estimates
in a direction that depends on the error structure — usually downward for
sensitivity, but upward for a marker correlated with whatever drives the coding
itself. A Tier C endpoint therefore cannot carry the same claim as a Tier A
endpoint, no matter how good the AUC looks.

The module draws disjoint probability samples of discordant and concordant
cases. Selection metadata is coordinator-only; reviewer packets exclude the
stratum as well as labels and marker values. Unweighted adjudicated accuracy
must not be extrapolated to the source population.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


REFERENCE_TIER_VERSION = "2.2.0"

TIER_A = "A_quantitative_or_adjudicated"
TIER_B = "B_multi_source_concordant"
TIER_C = "C_single_administrative_or_text"
TIER_UNKNOWN = "unknown"

VALID_TIERS: frozenset[str] = frozenset({TIER_A, TIER_B, TIER_C, TIER_UNKNOWN})

# Highest claim each tier can support on its own.
TIER_CLAIM_CEILING: Mapping[str, str] = {
    TIER_A: "diagnostic_accuracy",
    TIER_B: "diagnostic_accuracy_with_reference_uncertainty",
    TIER_C: "association_only_not_accuracy",
    TIER_UNKNOWN: "association_only_not_accuracy",
}

# Source tables that constitute a quantitative, timed measurement.
_QUANTITATIVE_SOURCES: frozenset[str] = frozenset(
    {
        # MIMIC-IV-ECHO structured measurements arrive under the source file
        # name; missing it silently demoted every echo-anchored endpoint --
        # the largest Tier A group in this registry -- to "unknown".
        "structured-measurement",
        "echo_measurements",
        "echo_measurements_wide",
        "labevents",
        "omr",
        "chartevents",
        "vitalsign",
        "triage",
    }
)
# Sources that are administrative or free text only.
_ADMINISTRATIVE_SOURCES: frozenset[str] = frozenset(
    {
        "diagnoses_icd",
        "procedures_icd",
        "ed_diagnosis",
        "discharge",
        "radiology",
        "services",
        "admissions",
        "edstays",
        "patients",
    }
)
# Dispensing and administration records: real evidence of exposure, but not a
# quantitative measurement of a physiological state.
_EXPOSURE_SOURCES: frozenset[str] = frozenset(
    {
        "emar",
        "prescriptions",
        "pharmacy",
        "pyxis",
        "medrecon",
        "microbiologyevents",
    }
)
# Same-recording device output: not an independent reference at all.
_SAME_RECORDING_SOURCES: frozenset[str] = frozenset({"machine_measurements"})


@dataclass(frozen=True)
class AdjudicationSamplePolicy:
    """How many of each stratum a blinded review should cover."""

    positives: int = 50
    negatives: int = 50
    uncertain: int = 50
    discordant: int = 100
    reviewers: int = 2
    random_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.reviewers < 2:
            raise ValueError(
                "A single reviewer cannot produce an agreement statistic"
            )
        for name in ("positives", "negatives", "uncertain", "discordant"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "positives": self.positives,
            "negatives": self.negatives,
            "uncertain": self.uncertain,
            "discordant": self.discordant,
            "reviewers": self.reviewers,
            "random_seed": self.random_seed,
        }


def classify_reference_tier(
    source_tables: Iterable[str],
    *,
    confidence: str = "",
    independent_source_count: Optional[int] = None,
    evidence_available: bool = False,
    temporally_eligible: bool = False,
    sources_concordant: bool = False,
) -> str:
    """Grade one endpoint's reference standard.

    ``source_tables`` is the endpoint's declared provenance. A same-recording
    device source can never raise the tier: it is the ECG describing itself.
    """

    sources = {str(item).strip() for item in source_tables if str(item).strip()}
    if not sources:
        return TIER_UNKNOWN
    informative = sources - _SAME_RECORDING_SOURCES - {"d_labitems", "d_items"}
    if not informative:
        # Only the recorder's own measurement of the same waveform.
        return TIER_UNKNOWN
    quantitative = informative & _QUANTITATIVE_SOURCES
    administrative = informative & _ADMINISTRATIVE_SOURCES
    exposure = informative & _EXPOSURE_SOURCES
    unclassified = informative - (
        _QUANTITATIVE_SOURCES | _ADMINISTRATIVE_SOURCES | _EXPOSURE_SOURCES
    )
    distinct = (
        int(independent_source_count)
        if independent_source_count is not None
        else 1
    )
    if quantitative and evidence_available and temporally_eligible:
        return TIER_A
    if distinct >= 2 and evidence_available and sources_concordant:
        return TIER_B
    if administrative or exposure:
        return TIER_C
    if unclassified:
        # An unrecognised source is not evidence of quality either way; it is
        # reported as ungraded rather than assumed adequate.
        return TIER_UNKNOWN
    return TIER_UNKNOWN


def annotate_reference_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Observed reference metadata; declarations alone cannot prove validity.

    Quantitative references need a measured evidence ID and a timed, known
    label. This does not assert absence of measurement or verification bias.
    Multiple table names do not establish source independence/concordance.
    """
    result = frame.copy()
    # Never materialize the wide feature matrix as Python row dictionaries.
    # Decode only the narrow provenance columns and classify repeated source
    # combinations once. This path runs for every endpoint, not every marker.
    def column(name: str, default: object) -> pd.Series:
        return frame[name] if name in frame else pd.Series(default, index=frame.index)

    def has_evidence(value: object) -> bool:
        try:
            evidence = json.loads(str(value))
        except (TypeError, ValueError):
            return False
        return isinstance(evidence, list) and any(str(item).strip() for item in evidence if item is not None)

    available = column("evidence_ids_json", "[]").map(has_evidence) & column("label_state", "unknown").isin(["positive", "negative"])
    delta = pd.to_numeric(column("time_delta_hours", np.nan), errors="coerce")
    timed = np.isfinite(delta) & column("time_precision", "").eq("timestamp")
    if "eligibility" in frame:
        timed &= frame["eligibility"].eq("eligible")
    if "eligibility_state" in frame:
        timed &= frame["eligibility_state"].eq("eligible")
    source_names = column("source_tables", "").fillna("").astype(str)
    descriptors = {}
    for source in source_names.unique():
        sources = tuple(item.strip() for item in re.split(r"[,|]", source) if item.strip())
        same_ecg = bool(set(sources) & _SAME_RECORDING_SOURCES) and not bool(set(sources) - _SAME_RECORDING_SOURCES)
        descriptors[source] = (classify_reference_tier(sources),
            classify_reference_tier(sources, evidence_available=True, temporally_eligible=True), same_ecg)
    tier = source_names.map(lambda name: descriptors[name][0])
    observed = available & timed
    tier.loc[observed] = source_names.loc[observed].map(lambda name: descriptors[name][1])
    result["reference_tier"] = tier
    result["reference_kind"] = np.select(
        [source_names.map(lambda name: descriptors[name][2]), tier.eq(TIER_A), tier.eq(TIER_C)],
        ["same_ecg_device", "timed_quantitative", "administrative_or_text"], default="unverified_or_unknown")
    result["reference_evidence_available"] = available
    result["reference_time_eligible"] = timed
    from .result_contract import true
    result["reference_alignment_confirmed"] = column("ehr_time_alignment_confirmatory_eligible", False).map(true)
    result["independent_reference_modality_available"] = (
        available & source_names.map(lambda name: descriptors[name][1] == TIER_A)
    )
    result["reference_clock_status"] = np.where(
        result["reference_alignment_confirmed"], "confirmed",
        "unverified_device_clock",
    )
    # An independent modality does not establish a reliable treatment/lab
    # time window. Keep the existing strict confirmation gate unchanged.
    result["reference_timing_claim_allowed"] = (
        available & timed & result["reference_alignment_confirmed"]
    )
    result["reference_claim_ceiling"] = tier.map(TIER_CLAIM_CEILING)
    result["reference_tier_version"] = REFERENCE_TIER_VERSION
    return result


def reference_tier_table(
    definitions: Iterable[object],
) -> pd.DataFrame:
    """Tier every registered endpoint, with its claim ceiling."""

    rows: list[dict[str, object]] = []
    for definition in definitions:
        source_tables = tuple(getattr(definition, "source_tables", ()) or ())
        target_kind = getattr(definition, "target_kind", None)
        target_value = getattr(target_kind, "value", str(target_kind or ""))
        tier = classify_reference_tier(source_tables)
        if target_value == "analytical_concordance":
            tier = TIER_UNKNOWN
        rows.append(
            {
                "label_id": str(getattr(definition, "label_id", "")),
                "display_name": str(getattr(definition, "display_name", "")),
                "target_kind": target_value,
                "source_tables": "|".join(source_tables),
                "reference_tier": tier,
                "claim_ceiling": TIER_CLAIM_CEILING.get(
                    tier,
                    TIER_CLAIM_CEILING[TIER_UNKNOWN],
                ),
                "accuracy_claim_supported": tier in (TIER_A, TIER_B),
                "reference_tier_version": REFERENCE_TIER_VERSION,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "label_id",
                "display_name",
                "target_kind",
                "source_tables",
                "reference_tier",
                "claim_ceiling",
                "accuracy_claim_supported",
                "reference_tier_version",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["reference_tier", "label_id"],
        kind="mergesort",
        ignore_index=True,
    )


def source_concordance(
    frame: pd.DataFrame,
    source_columns: Sequence[str],
) -> pd.DataFrame:
    """Per row: how many sources fired, and whether they disagree.

    Each source column holds 1, 0 or NaN. Disagreement between sources is not
    noise to be voted away -- it locates precisely the cases where the
    reference standard is doing the least work, which is where blinded review
    buys the most.
    """

    missing = sorted(set(source_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Concordance is missing source columns: {missing}")
    matrix = frame.loc[:, list(source_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    observed = matrix.notna()
    positives = (matrix == 1) & observed
    negatives = (matrix == 0) & observed
    output = pd.DataFrame(index=frame.index)
    output["sources_observed"] = observed.sum(axis=1).astype(int)
    output["sources_positive"] = positives.sum(axis=1).astype(int)
    output["sources_negative"] = negatives.sum(axis=1).astype(int)
    output["concordance_state"] = np.select(
        [
            output["sources_observed"] == 0,
            (output["sources_positive"] > 0) & (output["sources_negative"] > 0),
            (output["sources_positive"] > 0) & (output["sources_negative"] == 0),
            (output["sources_negative"] > 0) & (output["sources_positive"] == 0),
        ],
        ["unobserved", "discordant", "concordant_positive", "concordant_negative"],
        default="unobserved",
    )
    output["assigned_tier"] = np.select(
        [
            output["concordance_state"].eq("discordant"),
            (output["sources_observed"] >= 2)
            & output["concordance_state"].str.startswith("concordant"),
            output["sources_observed"] == 1,
        ],
        [TIER_C, TIER_B, TIER_C],
        default=TIER_UNKNOWN,
    )
    return output


def draw_adjudication_sample(
    frame: pd.DataFrame,
    *,
    endpoint_id: str,
    state_column: str = "label_state",
    concordance_column: Optional[str] = None,
    identifier_columns: Sequence[str] = ("subject_id", "study_id"),
    policy: AdjudicationSamplePolicy = AdjudicationSamplePolicy(),
) -> pd.DataFrame:
    """Draw a stratified reproducible coordinator-only selection manifest.

    Deterministic given the identifiers and the seed, so the same sample can
    be regenerated for a second reviewer without storing it. The output
    carries no label and no marker value: the reviewer must not see what the
    pipeline decided, or the agreement statistic measures nothing.
    """

    missing = sorted(set(identifier_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Adjudication sample is missing columns: {missing}")
    if state_column not in frame.columns:
        raise ValueError(f"Missing state column: {state_column}")
    working = frame.copy()
    if working.duplicated(list(identifier_columns)).any():
        raise ValueError("Adjudication input must contain one row per case")
    states = working[state_column].astype("string").str.casefold()
    strata: dict[str, pd.Series] = {
        "positive": states.eq("positive").fillna(False),
        "negative": states.eq("negative").fillna(False),
        "uncertain": states.isin(
            ("indeterminate", "uncertain", "unavailable")
        ).fillna(False),
    }
    if concordance_column and concordance_column in working.columns:
        strata["discordant"] = (
            working[concordance_column].astype("string").eq("discordant")
        ).fillna(False)
    else:
        strata["discordant"] = pd.Series(False, index=working.index)
    # Disjoint strata: no patient is sampled twice; inclusion probability is
    # exactly the stratum sampling fraction, not an overlapping approximation.
    for name in ("positive", "negative", "uncertain"):
        strata[name] &= ~strata["discordant"]
    quotas = {
        "positive": policy.positives,
        "negative": policy.negatives,
        "uncertain": policy.uncertain,
        "discordant": policy.discordant,
    }
    selected: list[pd.DataFrame] = []
    for stratum, mask in strata.items():
        quota = int(quotas.get(stratum, 0))
        if quota <= 0:
            continue
        candidates = working.loc[mask.to_numpy()].copy()
        if candidates.empty:
            continue
        candidates["__order"] = [
            _deterministic_rank(row, identifier_columns, policy.random_seed, stratum)
            for _, row in candidates.iterrows()
        ]
        candidates = candidates.sort_values("__order", kind="mergesort")
        chosen = candidates.head(quota).drop(columns="__order")
        chosen = chosen.loc[:, list(identifier_columns)].copy()
        chosen["review_stratum"] = stratum
        chosen["stratum_population_n"] = len(candidates)
        chosen["selection_probability"] = min(1.0, quota / len(candidates))
        selected.append(chosen)
    if not selected:
        return pd.DataFrame(
            columns=[
                *identifier_columns,
                "review_stratum",
                "endpoint_id",
                "review_case_id",
                "reviewer_slot",
                "blinded",
            ]
        )
    sample = pd.concat(selected, ignore_index=True)
    sample["endpoint_id"] = str(endpoint_id)
    sample["review_case_id"] = [
        _case_identifier(row, identifier_columns, endpoint_id)
        for _, row in sample.iterrows()
    ]
    expanded = []
    for slot in range(1, policy.reviewers + 1):
        copy = sample.copy()
        copy["reviewer_slot"] = slot
        expanded.append(copy)
    output = pd.concat(expanded, ignore_index=True)
    output["blinded"] = False
    output["pipeline_label_withheld"] = False
    output["access_role"] = "coordinator_only"
    output["adjudication_version"] = REFERENCE_TIER_VERSION
    return output.sort_values(
        ["reviewer_slot", "review_stratum", "review_case_id"],
        kind="mergesort",
        ignore_index=True,
    )


def blinded_reviewer_packet(selection: pd.DataFrame) -> pd.DataFrame:
    """Only opaque case IDs reach reviewers; the coordinator resolves signals.

    Sorting by case ID, not source stratum, also avoids order leaking labels.
    Selection probabilities remain in the separate manifest for weighted audit.
    """
    columns = ["review_case_id", "reviewer_slot", "endpoint_id"]
    packet = selection.loc[:, columns].copy()
    packet["blinded"] = True
    packet["pipeline_label_withheld"] = True
    return packet.sort_values(["reviewer_slot", "review_case_id"], ignore_index=True)


def adjudication_agreement(
    reviews: pd.DataFrame,
    *,
    case_column: str = "review_case_id",
    reviewer_column: str = "reviewer_slot",
    verdict_column: str = "reviewer_verdict",
) -> dict[str, object]:
    """Cohen's kappa between two reviewers over the adjudicated cases."""

    required = {case_column, reviewer_column, verdict_column}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError(f"Adjudication results are missing columns: {missing}")
    from .patient_identity import known_patient_mask
    if (not known_patient_mask(reviews[case_column]).all()
            or not known_patient_mask(reviews[reviewer_column]).all()):
        raise ValueError("Known case and reviewer identities required")
    if reviews.duplicated([case_column, reviewer_column]).any():
        raise ValueError("Duplicate case/reviewer verdict; no implicit first-result selection")
    common = {
        "n_cases_submitted": int(reviews[case_column].nunique()),
        "n_reviewers": int(reviews[reviewer_column].nunique()),
        "agreement_scope": "unweighted_selected_review_sample_not_population_accuracy",
        "clinical_claim_allowed": False,
        "adjudication_version": REFERENCE_TIER_VERSION,
    }
    if common["n_reviewers"] != 2:
        return {**common, "status": "not_evaluable", "failure_reason": "cohen_kappa_requires_exactly_two_reviewers",
                "n_cases": 0, "cohen_kappa": float("nan"), "raw_agreement": float("nan")}
    working = reviews.copy()
    working[verdict_column] = working[verdict_column].astype("string").str.strip().replace("", pd.NA)
    pivot = working.pivot(
        index=case_column,
        columns=reviewer_column,
        values=verdict_column,
    ).dropna()
    if pivot.shape[1] < 2 or len(pivot) < 2:
        return {
            **common,
            "status": "not_evaluable",
            "failure_reason": "insufficient_complete_review_pairs",
            "n_cases": int(len(pivot)),
            "cohen_kappa": float("nan"),
            "raw_agreement": float("nan"),
        }
    first = pivot.iloc[:, 0].astype(str).to_numpy()
    second = pivot.iloc[:, 1].astype(str).to_numpy()
    categories = sorted(set(first) | set(second))
    observed = float(np.mean(first == second))
    expected = 0.0
    for category in categories:
        expected += (
            float(np.mean(first == category))
            * float(np.mean(second == category))
        )
    kappa = (
        (observed - expected) / (1.0 - expected)
        if 1.0 - expected > 1e-12
        else float("nan")
    )
    return {
        **common,
        "status": "evaluable" if np.isfinite(kappa) else "not_evaluable",
        "failure_reason": "" if np.isfinite(kappa) else "kappa_undefined_constant_verdict",
        "n_cases_incomplete": common["n_cases_submitted"]-int(len(pivot)),
        "n_cases": int(len(pivot)),
        "n_categories": len(categories),
        "raw_agreement": observed,
        "cohen_kappa": float(kappa),
        "adjudication_version": REFERENCE_TIER_VERSION,
    }


def _deterministic_rank(
    row: pd.Series,
    identifier_columns: Sequence[str],
    seed: int,
    stratum: str,
) -> str:
    payload = "|".join(str(row[column]) for column in identifier_columns)
    return hashlib.sha256(
        f"{seed}:{stratum}:{payload}".encode("utf-8")
    ).hexdigest()


def _case_identifier(
    row: pd.Series,
    identifier_columns: Sequence[str],
    endpoint_id: str,
) -> str:
    payload = "|".join(str(row[column]) for column in identifier_columns)
    return hashlib.sha256(
        f"{endpoint_id}:{payload}".encode("utf-8")
    ).hexdigest()[:16]


__all__ = [
    "AdjudicationSamplePolicy",
    "REFERENCE_TIER_VERSION",
    "TIER_A",
    "TIER_B",
    "TIER_C",
    "TIER_CLAIM_CEILING",
    "TIER_UNKNOWN",
    "VALID_TIERS",
    "adjudication_agreement",
    "classify_reference_tier",
    "draw_adjudication_sample",
    "reference_tier_table",
    "source_concordance",
]
