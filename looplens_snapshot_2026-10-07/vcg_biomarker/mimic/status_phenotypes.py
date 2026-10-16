"""The diagnosis codes the broad scan never sees, and how to screen them.

Measured on this project's own data: of 28,583 distinct ICD codes, 72.7 % map
to a PheCodeX phenotype. That leaves **15.6 % of all 6.36 million diagnosis
rows** with no phenotype at all, and they never enter the broad scan.

The dropped set is not noise. Ranked by frequency it is dominated by ICD-10 Z
and ICD-9 V codes -- "factors influencing health status and contact with
health services" -- and several of them are among the most informative fields
in the whole database for this project:

* ``Z79.01`` / ``V58.61`` long-term anticoagulant use (30,957 + 22,635 rows):
  a strong proxy for atrial fibrillation, mechanical valves and thromboembolic
  disease, and the single best surrogate this cohort has for a rhythm history
  that predates the record.
* ``Z79.02`` / ``V58.66`` long-term antiplatelet use: coronary disease.
* ``Z79.4`` long-term insulin use: diabetes severity, not just presence.
* ``Z66`` / ``V49.86`` do-not-resuscitate status (18,899 + 10,530 rows): a
  goals-of-care marker and one of the strongest confounders of any mortality
  endpoint in an intensive-care population. Omitting it does not make it
  stop confounding.

PheCodeX deliberately excludes these because it is a *disease* taxonomy and a
status code is not a disease. That is a reasonable design decision for its own
purpose and the wrong one here: exposure, treatment status and goals of care
are exactly the context this project wants to screen ECG markers against.

They are therefore given their own arm with their own semantics rather than
being forced into a disease phenotype. Two properties follow and are enforced:

* a status code is a **statement about care, not about pathology**, so a hit
  is an association with treatment or documentation practice until shown
  otherwise;
* several of them are **downstream of the ECG** -- anticoagulation is started
  because atrial fibrillation was seen -- so their temporal role must be
  declared and they must never be read as predictors of the finding that
  caused them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


STATUS_PHENOTYPE_VERSION = "1.0.0"

# What the code says about the patient's relationship to care.
ROLE_TREATMENT_STATUS = "ongoing_treatment_status"
ROLE_GOALS_OF_CARE = "goals_of_care"
ROLE_DEVICE_PRESENT = "implanted_device_status"
ROLE_PROCEDURE_HISTORY = "procedure_history"
ROLE_EXPOSURE_OR_CONTACT = "exposure_or_contact"
ROLE_SOCIAL_CONTEXT = "social_or_administrative_context"

# Whether the status plausibly precedes or follows the ECG finding.
TEMPORAL_PRECEDES = "plausibly_precedes_ecg"
TEMPORAL_FOLLOWS = "plausibly_follows_ecg_finding"
TEMPORAL_AMBIGUOUS = "temporally_ambiguous"


@dataclass(frozen=True)
class StatusPhenotype:
    status_id: str
    display_name_de: str
    role: str
    temporal_role: str
    icd9_prefixes: tuple[str, ...]
    icd10_prefixes: tuple[str, ...]
    rationale_de: str

    def matches(self, code: str, version: int) -> bool:
        normalized = str(code).upper().replace(".", "").strip()
        prefixes = self.icd9_prefixes if int(version) == 9 else self.icd10_prefixes
        return any(normalized.startswith(prefix) for prefix in prefixes)

    def to_dict(self) -> dict[str, object]:
        return {
            "status_id": self.status_id,
            "display_name_de": self.display_name_de,
            "role": self.role,
            "temporal_role": self.temporal_role,
            "icd9_prefixes": "|".join(self.icd9_prefixes),
            "icd10_prefixes": "|".join(self.icd10_prefixes),
            "rationale_de": self.rationale_de,
            "reference_tier": "C_single_administrative_or_text",
            "claim_ceiling": "association_with_care_context_not_pathology",
            "status_phenotype_version": STATUS_PHENOTYPE_VERSION,
        }


STATUS_PHENOTYPES: tuple[StatusPhenotype, ...] = (
    StatusPhenotype(
        "long_term_anticoagulant_use",
        "Dauerhafte Antikoagulation",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_FOLLOWS,
        ("V5861", "V5863"),
        ("Z7901",),
        "Starker Surrogatmarker fuer Vorhofflimmern, mechanische Klappen und "
        "Thromboembolie. Folgt der Rhythmusdiagnose haeufig nach, darf also "
        "nicht als Praediktor eines AF-Befundes gelesen werden.",
    ),
    StatusPhenotype(
        "long_term_antiplatelet_use",
        "Dauerhafte Thrombozytenaggregationshemmung",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_FOLLOWS,
        ("V5866",),
        ("Z7902",),
        "Surrogat fuer koronare Herzkrankheit und Stentversorgung.",
    ),
    StatusPhenotype(
        "long_term_insulin_use",
        "Dauerhafte Insulintherapie",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V5867",),
        ("Z794",),
        "Diabetesschwere statt blosser Diagnose; relevant fuer autonome "
        "Neuropathie und Repolarisation.",
    ),
    StatusPhenotype(
        "long_term_steroid_use",
        "Dauerhafte Steroidtherapie",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V5865",),
        ("Z7952", "Z7951"),
        "Elektrolyt- und Volumeneffekte mit direkter EKG-Relevanz.",
    ),
    StatusPhenotype(
        "do_not_resuscitate_status",
        "Do-not-resuscitate-Status",
        ROLE_GOALS_OF_CARE,
        TEMPORAL_AMBIGUOUS,
        ("V4986",),
        ("Z66",),
        "Einer der staerksten Confounder jedes Mortalitaetsendpunkts in einer "
        "Intensivpopulation. Ihn wegzulassen laesst ihn nicht aufhoeren zu "
        "confounden.",
    ),
    StatusPhenotype(
        "palliative_care_encounter",
        "Palliativversorgung",
        ROLE_GOALS_OF_CARE,
        TEMPORAL_AMBIGUOUS,
        ("V667", "V577"),
        ("Z515",),
        "Behandlungsziel statt Pathologie; verzerrt jede Outcome-Analyse, die "
        "es nicht beruecksichtigt.",
    ),
    StatusPhenotype(
        "cardiac_device_present",
        "Herzschrittmacher oder Defibrillator vorhanden",
        ROLE_DEVICE_PRESENT,
        TEMPORAL_PRECEDES,
        ("V450", "V533"),
        ("Z950", "Z954", "Z9581", "Z4501"),
        "Geraetestatus veraendert die Erregungsausbreitung direkt und ist ein "
        "Ausschlusskriterium fuer viele Morphologiemarker.",
    ),
    StatusPhenotype(
        "coronary_revascularisation_history",
        "Zustand nach Revaskularisation",
        ROLE_PROCEDURE_HISTORY,
        TEMPORAL_PRECEDES,
        ("V4581", "V4582"),
        ("Z951", "Z955"),
        "Narbe und veraenderte Aktivierung nach Bypass oder Stent.",
    ),
    StatusPhenotype(
        "transplant_status",
        "Zustand nach Organtransplantation",
        ROLE_PROCEDURE_HISTORY,
        TEMPORAL_PRECEDES,
        ("V420", "V421", "V427"),
        ("Z940", "Z941", "Z944", "Z9483"),
        "Denerviertes Herz oder Immunsuppression, beides mit eigener "
        "EKG-Signatur.",
    ),
    StatusPhenotype(
        "dialysis_dependence",
        "Dialysepflicht",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V451", "V560"),
        ("Z992", "Z4931", "Z4901"),
        "Elektrolytschwankungen im Dialyserhythmus sind eine der staerksten "
        "nicht-kardialen EKG-Determinanten ueberhaupt.",
    ),
    StatusPhenotype(
        "ventilator_dependence",
        "Beatmungsabhaengigkeit",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V461",),
        ("Z9911",),
        "Intrathorakaler Druck und Sedierung veraendern Achse und Amplitude.",
    ),
    StatusPhenotype(
        "chemotherapy_encounter",
        "Laufende Chemotherapie",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V581", "V5811"),
        ("Z5111", "Z5112"),
        "Kardiotoxizitaet von Anthrazyklinen und Tyrosinkinasehemmern ist ein "
        "eigenes Forschungsfeld mit klarer EKG-Fragestellung.",
    ),
    StatusPhenotype(
        "radiotherapy_encounter",
        "Laufende Strahlentherapie",
        ROLE_TREATMENT_STATUS,
        TEMPORAL_PRECEDES,
        ("V580", "V661"),
        ("Z510",),
        "Mediastinale Bestrahlung fuehrt zu Perikard- und Reizleitungsschaeden.",
    ),
    StatusPhenotype(
        "tobacco_use_status",
        "Tabakkonsum oder -abhaengigkeit dokumentiert",
        ROLE_SOCIAL_CONTEXT,
        TEMPORAL_PRECEDES,
        ("3051", "V1582"),
        ("Z720", "F17", "Z87891"),
        "Wichtiger kardiovaskulaerer Risikofaktor, der in den Freitextnotizen "
        "stark redigiert ist und hier ueber Codes ergaenzt wird.",
    ),
    StatusPhenotype(
        "alcohol_use_status",
        "Alkoholkonsum oder -abhaengigkeit dokumentiert",
        ROLE_SOCIAL_CONTEXT,
        TEMPORAL_PRECEDES,
        ("3039", "3050"),
        ("F10", "Z721"),
        "Holiday-Heart-Phaenomen, Kardiomyopathie und Elektrolytstoerungen.",
    ),
    StatusPhenotype(
        "infection_contact_or_carrier",
        "Infektionskontakt oder Traegerstatus",
        ROLE_EXPOSURE_OR_CONTACT,
        TEMPORAL_AMBIGUOUS,
        ("V01", "V02"),
        ("Z20", "Z22"),
        "Erfasst unter anderem den COVID-Kontaktcode, der in dieser Kohorte "
        "der haeufigste ungemappte Code ueberhaupt ist.",
    ),
    StatusPhenotype(
        "body_mass_index_documented",
        "BMI-Kategorie dokumentiert",
        ROLE_SOCIAL_CONTEXT,
        TEMPORAL_PRECEDES,
        ("V850",),
        ("Z68",),
        "Koerperbau beeinflusst Amplituden direkt; die Z68-Kategorie ist oft "
        "vorhanden, wo eine Messung fehlt.",
    ),
    StatusPhenotype(
        "homelessness_or_housing_instability",
        "Wohnungslosigkeit oder instabile Wohnsituation",
        ROLE_SOCIAL_CONTEXT,
        TEMPORAL_PRECEDES,
        ("V600",),
        ("Z590",),
        "Sozialer Determinant mit starkem Einfluss auf Versorgungsintensitaet "
        "und damit auf jede Beobachtungsselektion.",
    ),
)


def status_phenotype_registry() -> pd.DataFrame:
    """The status-code arm as a reviewable table."""

    return pd.DataFrame([item.to_dict() for item in STATUS_PHENOTYPES])


def map_status_codes(
    codes: Iterable[tuple[str, int]],
) -> dict[str, list[str]]:
    """Map (code, version) pairs to status phenotype ids."""

    result: dict[str, list[str]] = {}
    for code, version in codes:
        for phenotype in STATUS_PHENOTYPES:
            if phenotype.matches(code, version):
                result.setdefault(phenotype.status_id, []).append(str(code))
    return result


def status_flags_sql(alias: str = "dx") -> str:
    """SQL CASE expressions producing one boolean per status phenotype."""

    fragments: list[str] = []
    for phenotype in STATUS_PHENOTYPES:
        conditions: list[str] = []
        if phenotype.icd9_prefixes:
            joined = " OR ".join(
                f"REPLACE(UPPER({alias}.icd_code), '.', '') LIKE '{prefix}%'"
                for prefix in phenotype.icd9_prefixes
            )
            conditions.append(f"({alias}.icd_version = 9 AND ({joined}))")
        if phenotype.icd10_prefixes:
            joined = " OR ".join(
                f"REPLACE(UPPER({alias}.icd_code), '.', '') LIKE '{prefix}%'"
                for prefix in phenotype.icd10_prefixes
            )
            conditions.append(f"({alias}.icd_version = 10 AND ({joined}))")
        expression = " OR ".join(conditions) if conditions else "FALSE"
        fragments.append(
            f"BOOL_OR({expression}) AS status_{phenotype.status_id}"
        )
    return ",\n            ".join(fragments)


def unmapped_code_census(
    diagnoses: pd.DataFrame,
    mapped_codes: Iterable[tuple[str, int]],
    *,
    code_column: str = "icd_code",
    version_column: str = "icd_version",
) -> pd.DataFrame:
    """Which codes reach no phenotype at all, ranked by how often they occur.

    Produced so that "not screened" is a reported quantity rather than an
    invisible property of the mapping table. A taxonomy that silently drops a
    sixth of the diagnosis rows should have to say so.
    """

    if diagnoses.empty:
        return pd.DataFrame(
            columns=[code_column, version_column, "n", "covered", "share"]
        )
    counts = (
        diagnoses.groupby([version_column, code_column])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False, kind="mergesort")
    )
    covered = {
        (str(code), int(version)) for code, version in mapped_codes
    }
    status_covered = set()
    for _, row in counts.iterrows():
        code = str(row[code_column])
        version = int(row[version_column])
        for phenotype in STATUS_PHENOTYPES:
            if phenotype.matches(code, version):
                status_covered.add((code, version))
                break
    counts["covered_by_disease_taxonomy"] = [
        (str(row[code_column]), int(row[version_column])) in covered
        for _, row in counts.iterrows()
    ]
    counts["covered_by_status_arm"] = [
        (str(row[code_column]), int(row[version_column])) in status_covered
        for _, row in counts.iterrows()
    ]
    counts["covered"] = (
        counts["covered_by_disease_taxonomy"] | counts["covered_by_status_arm"]
    )
    total = float(counts["n"].sum())
    counts["share"] = counts["n"] / max(total, 1.0)
    counts["status_phenotype_version"] = STATUS_PHENOTYPE_VERSION
    return counts.reset_index(drop=True)


def coverage_summary(census: pd.DataFrame) -> Mapping[str, object]:
    """Headline numbers for the report layer."""

    if census.empty:
        return {
            "codes_total": 0,
            "rows_total": 0,
            "row_coverage": float("nan"),
            "status_phenotype_version": STATUS_PHENOTYPE_VERSION,
        }
    rows_total = float(census["n"].sum())
    covered_rows = float(census.loc[census["covered"], "n"].sum())
    disease_rows = float(
        census.loc[census["covered_by_disease_taxonomy"], "n"].sum()
    )
    status_rows = float(
        census.loc[
            census["covered_by_status_arm"]
            & ~census["covered_by_disease_taxonomy"],
            "n",
        ].sum()
    )
    uncovered = census.loc[~census["covered"]].head(25)
    return {
        "codes_total": int(len(census)),
        "codes_uncovered": int((~census["covered"]).sum()),
        "rows_total": int(rows_total),
        "row_coverage": covered_rows / max(rows_total, 1.0),
        "row_coverage_disease_taxonomy": disease_rows / max(rows_total, 1.0),
        "row_coverage_added_by_status_arm": status_rows / max(rows_total, 1.0),
        "largest_uncovered_codes": [
            {"icd_code": str(row["icd_code"]), "n": int(row["n"])}
            for _, row in uncovered.iterrows()
        ],
        "interpretation_de": (
            "Nicht abgedeckte Codes erreichen den breiten Scan nicht. Das ist "
            "keine geprüfte Abwesenheit einer Assoziation, sondern eine "
            "Luecke in der Taxonomie."
        ),
        "status_phenotype_version": STATUS_PHENOTYPE_VERSION,
    }


def status_association_definitions() -> list[dict[str, object]]:
    """Status phenotypes as context-association definitions.

    Emitted in the same shape the association-domain screen consumes, with the
    temporal role carried through so a status that follows the ECG finding
    cannot be presented as a predictor of it.
    """

    definitions: list[dict[str, object]] = []
    for phenotype in STATUS_PHENOTYPES:
        definitions.append(
            {
                "association_id": f"status_{phenotype.status_id}",
                "display_name": phenotype.display_name_de,
                "domain": f"care_status_{phenotype.role}",
                "target_kind": "binary",
                "value_sql": (
                    f"CASE WHEN status_{phenotype.status_id} THEN 1.0 ELSE 0.0 END"
                ),
                "eligibility_sql": (
                    f"status_{phenotype.status_id} IS NOT NULL"
                ),
                "temporal_role": phenotype.temporal_role,
                "index_policy": "first eligible ECG, target-blind",
                "notes": phenotype.rationale_de,
                "reference_tier": "C_single_administrative_or_text",
                "claim_ceiling": (
                    "association_with_care_context_not_pathology"
                ),
                "usable_as_predictor": (
                    phenotype.temporal_role != TEMPORAL_FOLLOWS
                ),
            }
        )
    return definitions


__all__ = [
    "ROLE_DEVICE_PRESENT",
    "ROLE_EXPOSURE_OR_CONTACT",
    "ROLE_GOALS_OF_CARE",
    "ROLE_PROCEDURE_HISTORY",
    "ROLE_SOCIAL_CONTEXT",
    "ROLE_TREATMENT_STATUS",
    "STATUS_PHENOTYPES",
    "STATUS_PHENOTYPE_VERSION",
    "StatusPhenotype",
    "TEMPORAL_AMBIGUOUS",
    "TEMPORAL_FOLLOWS",
    "TEMPORAL_PRECEDES",
    "coverage_summary",
    "map_status_codes",
    "status_association_definitions",
    "status_flags_sql",
    "status_phenotype_registry",
    "unmapped_code_census",
]
