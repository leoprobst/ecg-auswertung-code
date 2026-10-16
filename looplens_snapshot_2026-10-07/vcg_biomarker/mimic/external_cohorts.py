"""External ECG cohorts: PhysioNet/CinC Challenge 2021 training collection.

MIMIC-IV is one hospital. The project's own hardening document states that no
amount of internal replication can establish external generalisability, and
until now the only external set was PTB-XL — which this project already used
for development, so it cannot serve as an independent test.

The Challenge 2021 collection supplies what was missing: 88,173 twelve-lead
recordings from seven sources across four countries and several recorder
generations, with expert ECG interpretation labels as SNOMED-CT codes.

What it can and cannot do is worth stating precisely, because the two are
easy to conflate:

* **It can validate measurement.** Do the markers compute at all on other
  devices, sampling rates and populations? With what coverage, what
  abstention rate, what distribution shift? This is unambiguous external
  analytical validation, and it is the part the project needs most.
* **It can validate against expert ECG reading.** The labels are
  interpretations *of the ECG*, made by readers at those sites.
* **It cannot validate against echo or laboratory reference standards.**
  Agreement with another reader is not diagnostic accuracy against an
  independent measurement of the heart. In the project's own tier language
  these labels are Tier C, and results carry that ceiling.

Two members of the collection are deliberately routed elsewhere:

* **PTB-XL** is already this project's development transfer set. Including it
  in the independent arm would report development data as external
  validation. It is retained, flagged, and reported separately.
* **St Petersburg INCART** is 257 Hz, thirty-minute recordings — Holter-like,
  not ten-second diagnostic ECGs. It is not comparable for a diagnostic
  claim, and is routed to the delineation benchmark, where long annotated
  recordings are exactly what is wanted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from .patient_identity import known_patient_mask


EXTERNAL_COHORT_VERSION = "1.4.0"

# Role of each source in this project's evidence structure.
ROLE_INDEPENDENT_EXTERNAL = "independent_external"
ROLE_DEVELOPMENT_REUSED = "development_reused_not_independent"
ROLE_DELINEATION_BENCHMARK = "delineation_benchmark_only"


@dataclass(frozen=True)
class ExternalSource:
    source_id: str
    display_name: str
    country: str
    directory_names: tuple[str, ...]
    role: str
    expected_sampling_rate_hz: Optional[int] = None
    notes: str = ""
    acquisition_upper_bandwidth_hz: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "display_name": self.display_name,
            "country": self.country,
            "role": self.role,
            "expected_sampling_rate_hz": self.expected_sampling_rate_hz,
            "acquisition_upper_bandwidth_hz": (
                self.acquisition_upper_bandwidth_hz
            ),
            "notes": self.notes,
            "independent": self.role == ROLE_INDEPENDENT_EXTERNAL,
        }


EXTERNAL_SOURCES: tuple[ExternalSource, ...] = (
    ExternalSource(
        "cpsc_2018",
        "CPSC 2018",
        "China",
        ("cpsc_2018",),
        ROLE_INDEPENDENT_EXTERNAL,
        500,
        "China Physiological Signal Challenge, 11 hospitals.",
    ),
    ExternalSource(
        "cpsc_2018_extra",
        "CPSC 2018 extra",
        "China",
        ("cpsc_2018_extra",),
        ROLE_INDEPENDENT_EXTERNAL,
        500,
        "Unused CPSC data; same acquisition programme.",
    ),
    ExternalSource(
        "georgia",
        "Georgia 12-lead",
        "USA",
        ("georgia",),
        ROLE_INDEPENDENT_EXTERNAL,
        500,
        "Emory University; a US population other than MIMIC's.",
    ),
    ExternalSource(
        "chapman_shaoxing",
        "Chapman-Shaoxing",
        "China",
        ("chapman_shaoxing",),
        ROLE_INDEPENDENT_EXTERNAL,
        500,
        "Chapman University and Shaoxing People's Hospital.",
    ),
    ExternalSource(
        "ningbo",
        "Ningbo First Hospital",
        "China",
        ("ningbo",),
        ROLE_INDEPENDENT_EXTERNAL,
        500,
        "Largest single source in the collection.",
    ),
    ExternalSource(
        "ptb",
        "PTB Diagnostic",
        "Germany",
        ("ptb",),
        ROLE_INDEPENDENT_EXTERNAL,
        1000,
        "Older PTB Diagnostic database; distinct from PTB-XL.",
    ),
    ExternalSource(
        "ptb_xl",
        "PTB-XL",
        "Germany",
        ("ptb-xl", "ptb_xl"),
        ROLE_DEVELOPMENT_REUSED,
        500,
        "Already used as this project's frozen transfer set. Reporting it as "
        "external validation would present development data as independent.",
    ),
    ExternalSource(
        "st_petersburg_incart",
        "St Petersburg INCART",
        "Russia",
        ("st_petersburg_incart", "incartdb", "incart"),
        ROLE_DELINEATION_BENCHMARK,
        257,
        "257 Hz, thirty-minute recordings. Not comparable to ten-second "
        "diagnostic ECGs; used for boundary benchmarking instead.",
    ),
)


def source_acquisition_upper_bandwidth_hz(source_id: str) -> Optional[float]:
    """Return a documented analogue front-end corner, never ADC Nyquist.

    The Challenge headers specify sampling rates but not a validated analogue
    acquisition bandwidth.  Consequently the registry deliberately contains
    ``None`` until primary device documentation can support a value.
    """

    normalized = str(source_id).strip().casefold()
    for source in EXTERNAL_SOURCES:
        if source.source_id.casefold() == normalized:
            return source.acquisition_upper_bandwidth_hz
    return None


# Mains frequency is an acquisition-site property, not a global analysis
# switch.  In particular, the Georgia/USA arm must never inherit the 50 Hz
# default used by the Chinese and German sources.
POWERLINE_HZ_BY_SOURCE: Mapping[str, float] = {
    "cpsc_2018": 50.0,
    "cpsc_2018_extra": 50.0,
    "georgia": 60.0,
    "chapman_shaoxing": 50.0,
    "ningbo": 50.0,
    "ptb": 50.0,
    "ptb_xl": 50.0,
    "st_petersburg_incart": 50.0,
}


def source_powerline_hz(source_id: str) -> float:
    try:
        return float(POWERLINE_HZ_BY_SOURCE[str(source_id)])
    except KeyError as exc:
        raise ValueError(
            f"No prespecified mains frequency for external source {source_id!r}"
        ) from exc


def source_by_directory(name: str) -> Optional[ExternalSource]:
    token = str(name).strip().casefold()
    for source in EXTERNAL_SOURCES:
        if token in {item.casefold() for item in source.directory_names}:
            return source
    return None


def independent_source_ids() -> tuple[str, ...]:
    return tuple(
        source.source_id
        for source in EXTERNAL_SOURCES
        if source.role == ROLE_INDEPENDENT_EXTERNAL
    )


# --- SNOMED-CT interpretation labels ---------------------------------------

# Curated subset of the 133 codes present in the collection, restricted to
# concepts this project actually screens for. Deliberately not exhaustive:
# an unmapped code is reported as unmapped, never silently treated as absent.
SNOMED_LABELS: Mapping[str, str] = {
    "426783006": "sinus_rhythm",
    "426177001": "sinus_bradycardia",
    "427084000": "sinus_tachycardia",
    "427393009": "sinus_arrhythmia",
    "164889003": "atrial_fibrillation",
    "164890007": "atrial_flutter",
    "270492004": "first_degree_av_block",
    "59118001": "right_bundle_branch_block",
    "713427006": "complete_right_bundle_branch_block",
    "713426002": "incomplete_right_bundle_branch_block",
    "164909002": "left_bundle_branch_block",
    "445118002": "left_anterior_fascicular_block",
    "445211001": "left_posterior_fascicular_block",
    "698252002": "nonspecific_intraventricular_conduction_delay",
    "39732003": "left_axis_deviation",
    "47665007": "right_axis_deviation",
    "55827005": "left_ventricular_hypertrophy",
    "164873001": "left_ventricular_hypertrophy_voltage",
    "266249003": "right_ventricular_hypertrophy",
    "251146004": "low_qrs_voltage",
    "111975006": "prolonged_qt_interval",
    "164934002": "t_wave_abnormal",
    "59931005": "t_wave_inversion",
    "164930006": "st_elevation",
    "429622005": "st_depression",
    "428750005": "nonspecific_st_t_abnormality",
    "55930002": "st_t_change",
    "164865005": "myocardial_infarction",
    "164861001": "myocardial_ischemia",
    "164917005": "q_wave_abnormal",
    "284470004": "premature_atrial_contraction",
    "164884008": "ventricular_ectopic_beats",
    "427172004": "premature_ventricular_contraction",
    "10370003": "pacing_rhythm",
    "164947007": "prolonged_pr_interval",
    "89792004": "right_ventricular_hypertrophy_alt",
    "413444003": "acute_myocardial_ischemia",
    "426627000": "bradycardia",
    "17338001": "ventricular_premature_beats",
}

# How an external interpretation label relates to a project endpoint. The
# reference standard differs -- an ECG reader's LVH is not echo LVH -- so each
# pairing carries the comparison kind explicitly.
COMPARISON_ECG_READING = "expert_ecg_reading_not_independent_measurement"

EXTERNAL_ENDPOINT_MAP: Mapping[str, tuple[str, ...]] = {
    # Project endpoint -> external interpretation labels that address it.
    "echo_lvh_strict": (
        "left_ventricular_hypertrophy",
        "left_ventricular_hypertrophy_voltage",
    ),
    "echo_lvh_sensitive": (
        "left_ventricular_hypertrophy",
        "left_ventricular_hypertrophy_voltage",
    ),
    "acute_mi_probable": ("myocardial_infarction", "myocardial_ischemia"),
    "acute_mi_high_specificity": ("myocardial_infarction",),
    "new_onset_af_30d": ("atrial_fibrillation", "atrial_flutter"),
    "qtc_gt_480": ("prolonged_qt_interval",),
    "qtc_gt_500": ("prolonged_qt_interval",),
}


def map_snomed_codes(codes: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split SNOMED codes into mapped label names and unmapped codes."""

    mapped: list[str] = []
    unmapped: list[str] = []
    for code in codes:
        token = str(code).strip()
        if not token:
            continue
        name = SNOMED_LABELS.get(token)
        if name is None:
            unmapped.append(token)
        else:
            mapped.append(name)
    return tuple(dict.fromkeys(mapped)), tuple(dict.fromkeys(unmapped))


# --- header parsing ---------------------------------------------------------

_DX_RE = re.compile(r"^#\s*Dx:\s*(.+)$", re.MULTILINE)
_AGE_RE = re.compile(r"^#\s*Age:\s*(.+)$", re.MULTILINE)
_SEX_RE = re.compile(r"^#\s*Sex:\s*(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class ExternalRecord:
    record_id: str
    source_id: str
    header_path: str
    n_leads: int
    sampling_rate_hz: int
    n_samples: int
    duration_seconds: float
    age: Optional[float]
    sex: str
    labels: tuple[str, ...]
    unmapped_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "header_path": self.header_path,
            "n_leads": self.n_leads,
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_samples": self.n_samples,
            "duration_seconds": self.duration_seconds,
            "age": self.age,
            "sex": self.sex,
            "labels": "|".join(self.labels),
            "unmapped_codes": "|".join(self.unmapped_codes),
            "n_labels": len(self.labels),
            "external_cohort_version": EXTERNAL_COHORT_VERSION,
        }


def parse_external_header(path: Path, source_id: str) -> Optional[ExternalRecord]:
    """Parse one WFDB header into a record descriptor."""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    fields = lines[0].split()
    if len(fields) < 4:
        return None
    try:
        n_leads = int(fields[1])
        sampling_rate = int(float(fields[2]))
        n_samples = int(fields[3])
    except (TypeError, ValueError):
        return None
    age_match = _AGE_RE.search(text)
    age: Optional[float] = None
    if age_match:
        try:
            parsed = float(age_match.group(1).strip())
            age = parsed if np.isfinite(parsed) else None
        except ValueError:
            age = None
    sex_match = _SEX_RE.search(text)
    sex = sex_match.group(1).strip() if sex_match else ""
    dx_match = _DX_RE.search(text)
    codes = (
        [item.strip() for item in dx_match.group(1).split(",")]
        if dx_match
        else []
    )
    labels, unmapped = map_snomed_codes(codes)
    return ExternalRecord(
        record_id=str(fields[0]),
        source_id=source_id,
        header_path=str(Path(path).resolve()),
        n_leads=n_leads,
        sampling_rate_hz=sampling_rate,
        n_samples=n_samples,
        duration_seconds=(
            float(n_samples) / sampling_rate if sampling_rate else float("nan")
        ),
        age=age,
        sex=sex.upper()[:1] if sex else "",
        labels=labels,
        unmapped_codes=unmapped,
    )


def inventory_external_collection(
    root: Path | str,
    *,
    limit_per_source: Optional[int] = None,
) -> pd.DataFrame:
    """Enumerate every record in the collection with its source and labels."""

    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"External collection root not found: {base}")
    rows: list[dict[str, object]] = []
    for directory in sorted(item for item in base.iterdir() if item.is_dir()):
        source = source_by_directory(directory.name)
        if source is None:
            continue
        # Some Challenge conversions discarded patient keys. A documented
        # source-specific sidecar can restore them without guessing IDs from
        # unrelated record names. It must not contain labels from MIMIC.
        sidecar = directory / "reference_contract.parquet"
        references = pd.read_parquet(sidecar) if sidecar.is_file() else pd.DataFrame()
        lookup = {}
        if not references.empty:
            required = {"record_id", "reference_contract_source"}
            if not required <= set(references) or references.record_id.duplicated().any():
                raise ValueError(f"Invalid or ambiguous external reference contract: {sidecar}")
            if references.reference_contract_source.fillna("").astype(str).str.strip().eq("").any():
                raise ValueError(f"Reference contract needs provenance: {sidecar}")
            if "patient_id" in references:
                references.loc[~known_patient_mask(references.patient_id), "patient_id"] = None
            lookup = {str(row["record_id"]): row for row in references.to_dict("records")}
        count = 0
        for header in sorted(directory.rglob("*.hea")):
            record = parse_external_header(header, source.source_id)
            if record is None:
                continue
            payload = record.to_dict()
            reference = lookup.get(str(record.record_id), {})
            payload.update({str(key): value for key, value in reference.items()
                if key in {"patient_id", "reference_contract_source", "annotation_exhaustive"}
                or str(key).startswith(("label_state_", "label_coverage_"))})
            identity_known = bool(pd.notna(payload.get("patient_id")))
            payload["patient_identity_status"] = "source_contract" if identity_known else "unknown"
            payload["reference_contract_available"] = bool(reference)
            payload.update(
                {
                    "source_role": source.role,
                    "source_country": source.country,
                    "source_display_name": source.display_name,
                    "independent_external": (
                        source.role == ROLE_INDEPENDENT_EXTERNAL
                    ),
                }
            )
            rows.append(payload)
            count += 1
            if limit_per_source is not None and count >= int(limit_per_source):
                break
    if not rows:
        return pd.DataFrame(
            columns=[
                "record_id",
                "source_id",
                "source_role",
                "independent_external",
                "sampling_rate_hz",
                "duration_seconds",
                "labels",
            ]
        )
    return pd.DataFrame(rows)


def collection_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Per-source counts, sampling rates and label coverage."""

    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "n_records",
                "role",
                "independent_external",
                "median_duration_seconds",
                "labelled_share",
            ]
        )
    rows: list[dict[str, object]] = []
    for source_id, group in inventory.groupby("source_id", sort=True):
        rows.append(
            {
                "source_id": str(source_id),
                "display_name": str(group["source_display_name"].iloc[0]),
                "country": str(group["source_country"].iloc[0]),
                "role": str(group["source_role"].iloc[0]),
                "independent_external": bool(
                    group["independent_external"].iloc[0]
                ),
                "n_records": int(len(group)),
                "sampling_rates_hz": "|".join(
                    str(value)
                    for value in sorted(group["sampling_rate_hz"].unique())
                ),
                "median_duration_seconds": float(
                    group["duration_seconds"].median()
                ),
                "labelled_share": float(group["n_labels"].gt(0).mean()),
                "median_labels_per_record": float(group["n_labels"].median()),
                "unmapped_code_share": float(
                    group["unmapped_codes"].astype(str).str.len().gt(0).mean()
                ),
                "external_cohort_version": EXTERNAL_COHORT_VERSION,
            }
        )
    return pd.DataFrame(rows)


def eligible_for_diagnostic_validation(
    inventory: pd.DataFrame,
    *,
    minimum_duration_seconds: float = 8.0,
    maximum_duration_seconds: float = 60.0,
    required_leads: int = 12,
) -> pd.DataFrame:
    """Restrict to records comparable with MIMIC's ten-second 12-lead ECGs.

    Duration and lead count are eligibility criteria, not quality filters: a
    thirty-minute Holter recording and a ten-second diagnostic ECG answer
    different questions, and pooling them would make an external accuracy
    figure uninterpretable.
    """

    if inventory.empty:
        output = inventory.copy()
        output["eligible_for_analytical_validation"] = pd.Series(False, index=output.index, dtype=bool)
        output["eligible_for_diagnostic_validation"] = pd.Series(False, index=output.index, dtype=bool)
        output["ineligibility_reason"] = pd.Series(index=output.index, dtype=str)
        return output
    output = inventory.copy()
    from .result_contract import true
    source_role_eligible = output["independent_external"].map(true)
    output["eligible_for_analytical_validation"] = (
        source_role_eligible
        & output["n_leads"].eq(int(required_leads))
        & output["duration_seconds"].between(
            float(minimum_duration_seconds),
            float(maximum_duration_seconds),
        )
    )
    output["eligible_for_diagnostic_validation"] = output["eligible_for_analytical_validation"] & output["n_labels"].gt(0)
    output["patient_independence_established_by_source_role"] = False
    output["ineligibility_reason"] = np.select(
        [
            ~source_role_eligible,
            output["n_leads"].ne(int(required_leads)),
            ~output["duration_seconds"].between(
                float(minimum_duration_seconds),
                float(maximum_duration_seconds),
            ),
            output["n_labels"].eq(0),
        ],
        [
            "source_outside_external_validation_role",
            "lead_count_mismatch",
            "duration_not_comparable",
            "no_mapped_label",
        ],
        default="",
    )
    return output


def external_claim_scope(inventory: pd.DataFrame) -> Mapping[str, object]:
    """What this collection does and does not license as a claim."""

    eligible = eligible_for_diagnostic_validation(inventory)
    annotation = eligible.loc[
        eligible["eligible_for_diagnostic_validation"].astype(bool)
    ]
    analytical = eligible.loc[eligible["eligible_for_analytical_validation"]]
    return {
        "n_records_total": int(len(inventory)),
        "n_records_analytical_role_eligible_before_copy_checks": int(len(analytical)),
        "n_records_annotation_role_eligible_before_copy_checks": int(len(annotation)),
        "source_roles_prove_patient_independence": False,
        "n_verified_independent_patients": None,
        "external_validation_role_sources": sorted(
            str(value) for value in annotation.get("source_id", pd.Series(dtype=str)).unique()
        ),
        "countries": sorted(
            str(value) for value in analytical.get("source_country", pd.Series(dtype=str)).unique()
        ),
        "reference_standard": COMPARISON_ECG_READING,
        "reference_tier": "C_single_administrative_or_text",
        "supports_external_analytical_validation": bool(len(analytical)),
        "supports_external_agreement_with_expert_reading": bool(len(annotation)),
        "capability_flags_are_not_successful_validation_results": True,
        "supports_external_accuracy_against_echo_or_laboratory": False,
        "development_reused_sources_excluded": [
            source.source_id
            for source in EXTERNAL_SOURCES
            if source.role == ROLE_DEVELOPMENT_REUSED
        ],
        "delineation_only_sources": [
            source.source_id
            for source in EXTERNAL_SOURCES
            if source.role == ROLE_DELINEATION_BENCHMARK
        ],
        "external_cohort_version": EXTERNAL_COHORT_VERSION,
    }


def source_registry_table() -> pd.DataFrame:
    """The source registry, for preregistration and reporting."""

    return pd.DataFrame([source.to_dict() for source in EXTERNAL_SOURCES])


__all__ = [
    "COMPARISON_ECG_READING",
    "EXTERNAL_COHORT_VERSION",
    "EXTERNAL_ENDPOINT_MAP",
    "EXTERNAL_SOURCES",
    "ExternalRecord",
    "ExternalSource",
    "ROLE_DELINEATION_BENCHMARK",
    "ROLE_DEVELOPMENT_REUSED",
    "ROLE_INDEPENDENT_EXTERNAL",
    "SNOMED_LABELS",
    "collection_summary",
    "eligible_for_diagnostic_validation",
    "external_claim_scope",
    "independent_source_ids",
    "inventory_external_collection",
    "map_snomed_codes",
    "parse_external_header",
    "source_by_directory",
    "source_registry_table",
]
