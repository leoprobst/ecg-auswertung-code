"""MIMIC-IV-Notes (discharge summaries, radiology reports) text mining --
structurally mirrors report_parser.py's clause-splitting and negation/
uncertainty heuristic, which is generic and not ECG-specific.

NOTE_PATTERNS covers a curated set of 16 signals calibrated against real
discharge.csv/radiology.csv samples (~15k discharge notes, ~4.5k CT/MR
head reports, ~850 CTA chest reports, ~1.1k chest X-ray reports):
radiology heart/lung findings, cardiovascular risk factors, free-text
diagnosis mentions, medication non-adherence, and head/CTA-chest imaging
findings (stroke, intracranial hemorrhage, pulmonary embolism, aortic
dissection). Two further signals -- functional/mental status at
discharge -- are extracted separately via classify_activity_status()/
classify_mental_status() (see their docstrings for why). This is a real
but still deliberately non-exhaustive set -- e.g. no CT abdomen/pelvis
or ultrasound vocabulary, no non-English notes, no medication-list
structured data beyond free-text mentions -- curated the same way as the
chartevents itemid list and extendable the same way, not a one-time
"complete" NLP effort.

split_sections() only splits on a curated allow-list of top-level
headers (DISCHARGE_SECTION_HEADERS / RADIOLOGY_SECTION_HEADERS) rather
than any capitalized "Word:" line. MIMIC-IV discharge summaries nest
sub-headers inside some top-level sections (e.g. "Discharge Diagnosis:"
is immediately followed by "Primary:" / "Secondary:" sub-headers) --
splitting on every colon-terminated line would treat those sub-headers
as new top-level boundaries and truncate the parent section to empty
content, silently losing it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable


# Top-level discharge-summary headers, in the order MIMIC-IV discharge
# summaries typically present them. Only these split text into sections --
# sub-headers nested inside a section (e.g. "Primary:"/"Secondary:" inside
# "Discharge Diagnosis:") are left as part of that section's content.
DISCHARGE_SECTION_HEADERS: tuple[str, ...] = (
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
)

# Top-level radiology-report headers.
RADIOLOGY_SECTION_HEADERS: tuple[str, ...] = (
    "EXAMINATION",
    "INDICATION",
    "TECHNIQUE",
    "COMPARISON",
    "FINDINGS",
    "IMPRESSION",
)

DISCHARGE_SECTION_PATTERN = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(h) for h in DISCHARGE_SECTION_HEADERS) + r"):[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)
RADIOLOGY_SECTION_PATTERN = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(h) for h in RADIOLOGY_SECTION_HEADERS) + r"):[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def split_sections(text: str | None, pattern: re.Pattern = DISCHARGE_SECTION_PATTERN) -> dict[str, str]:
    """Split `text` into {header.lower(): content} using `pattern` to find
    section boundaries. Headers may have content on the same line
    ("IMPRESSION:  No acute process.") and/or on following lines; both are
    captured. Only boundaries matched by `pattern` count as new sections --
    pass RADIOLOGY_SECTION_PATTERN for radiology reports.
    """
    text = str(text or "")
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        header = match.group(1).strip().lower()
        same_line = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rest = text[start:end].strip()
        content = (same_line + "\n" + rest).strip() if rest else same_line
        if content:
            sections.setdefault(header, content)
    return sections


class NoteMentionState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"


@dataclass(frozen=True)
class NoteLabel:
    label_id: str
    state: NoteMentionState
    matched_clause: str = ""
    matched_pattern: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        return data


# Curated against real discharge.csv/radiology.csv samples -- see module
# docstring. Shape matches report_parser.py's REPORT_PATTERNS:
# {label_id: (regex, ...)}. Grouped by which note section each label is
# meant to be scanned against (see note_labels.py for the section
# extraction -> label wiring):
#   chest X-ray (radiology.parquet, "impression" section, with "findings"
#   as fallback when no separate impression is present):
#     cardiomegaly, pulmonary_edema, pleural_effusion, pneumothorax
#   risk factors (discharge.parquet, "social history" / "family history"):
#     smoking_history, alcohol_use, illicit_drug_use, family_history_cardiac
#   diagnosis mentions (discharge.parquet, "discharge diagnosis" /
#   "brief hospital course"):
#     heart_failure_mentioned, arrhythmia_mentioned, mi_or_acs_mentioned
#   medication adherence (discharge.parquet, "history of present illness"
#   + "brief hospital course" + "social history"):
#     medication_nonadherence_mentioned
#   CT/MR head (radiology.parquet, "impression" with "findings" fallback):
#     acute_stroke_or_infarct, intracranial_hemorrhage
#   CTA chest (radiology.parquet, "impression" with "findings" fallback):
#     pulmonary_embolism, aortic_dissection
NOTE_PATTERNS: dict[str, tuple[str, ...]] = {
    "cardiomegaly": (
        r"\bcardiomegaly\b",
    ),
    "pulmonary_edema": (
        r"\bpulmonary edema\b",
        r"\bpulmonary vascular congestion\b",
        r"\bvascular congestion\b",
    ),
    "pleural_effusion": (
        r"\bpleural effusions?\b",
    ),
    "pneumothorax": (
        r"\bpneumothora(?:x|ces)\b",
    ),
    "smoking_history": (
        r"\bsmok(?:e[ds]?|ing)\b",
        r"\btobacco\b",
        r"\bpack[\s-]?year",
    ),
    "alcohol_use": (
        r"\balcohol\b",
        r"\bEtOH\b",
    ),
    "illicit_drug_use": (
        r"\bIVDU\b",
        r"\bcocaine\b",
        r"\bheroin\b",
        r"\billicit drug",
        r"\bsubstance abuse\b",
        r"\bdrug use\b",
    ),
    "family_history_cardiac": (
        r"\bheart disease\b",
        r"\bmyocardial infarction\b",
        r"\bsudden (?:cardiac )?death\b",
        r"\bcardiomyopathy\b",
        r"\bcoronary artery disease\b",
    ),
    "heart_failure_mentioned": (
        r"\bheart failure\b",
        r"\bCHF\b",
        r"\bcardiomyopathy\b",
    ),
    "arrhythmia_mentioned": (
        r"\batrial fibrillation\b",
        r"\batrial flutter\b",
        r"\bventricular tachycardia\b",
        r"\barrhythmia\b",
    ),
    "mi_or_acs_mentioned": (
        r"\bmyocardial infarction\b",
        r"\bNSTEMI\b",
        r"\bSTEMI\b",
        r"\bacute coronary syndrome\b",
    ),
    "medication_nonadherence_mentioned": (
        r"\bnon[- ]?complian\w*\b",
        r"\bmedication[s]?\s+non[- ]?adherence\b",
        r"\bmissed\s+(?:several\s+|his\s+|her\s+)?(?:doses|medications?)\b",
        r"\bran out of\s+(?:his\s+|her\s+|their\s+)?medications?\b",
        # negative lookahead excludes structured "Stopped Medications:"
        # list headers (clinician-directed discontinuation), which the
        # bare phrase would otherwise misfire on -- see module docstring.
        r"\bstopped\s+(?:taking\s+)?(?:his\s+|her\s+|their\s+)?medications?\b(?!\s*:)",
        r"\bnot taking\s+(?:his\s+|her\s+|their\s+)?medications?\b",
        r"\bself[- ]?discontinu\w*",
        r"\bpoor(?:ly)?\s+adherent\b",
    ),
    "acute_stroke_or_infarct": (
        r"\bacute infarct",
        r"\bischemic strok",
        r"\bacute ischemi",
        r"\bcerebral infarct",
        r"\bacute cva\b",
    ),
    "intracranial_hemorrhage": (
        r"\bintracranial hemorrhage",
        r"\bintraparenchymal hemorrhage",
        r"\bsubarachnoid hemorrhage",
        r"\bsubdural hematoma",
        r"\bepidural hematoma",
        r"\bintraventricular hemorrhage",
        r"\bhemorrhagic transformation",
    ),
    "pulmonary_embolism": (
        r"\bpulmonary embolus",
        r"\bpulmonary emboli",
        r"\bpulmonary embolism",
        r"\bfilling defect.{0,20}pulmonary arter",
    ),
    "aortic_dissection": (
        r"\baortic dissection",
        r"\baortic syndrome",
        r"\bintramural hematoma\b",
    ),
}

# Reused as-is from report_parser.py -- negation/uncertainty phrasing is
# general clinical-text convention, not specific to ECG machine reports.
NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bdenies\b",
    r"\bdenied\b",
)
UNCERTAINTY_PATTERNS = (
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bcannot rule out\b",
    r"\bconsider\b",
    r"\bsuggest(?:s|ive of)?\b",
    r"\bquestion of\b",
    r"\bmay represent\b",
)


class ClinicalNoteParser:
    def __init__(
        self,
        patterns: dict[str, tuple[str, ...]] = NOTE_PATTERNS,
    ):
        self._patterns = {
            label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in values)
            for label, values in patterns.items()
        }
        self._negations = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in NEGATION_PATTERNS
        )
        self._uncertainties = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in UNCERTAINTY_PATTERNS
        )

    @property
    def label_ids(self) -> tuple[str, ...]:
        return tuple(self._patterns)

    def parse(self, note_text: str | None) -> dict[str, NoteLabel]:
        text = str(note_text or "").strip()
        clauses = _split_clauses(text)
        output = {}
        for label_id, patterns in self._patterns.items():
            mentions = []
            for clause in clauses:
                for pattern in patterns:
                    match = pattern.search(clause)
                    if match:
                        mentions.append(
                            (
                                _mention_state(
                                    clause,
                                    match.start(),
                                    self._negations,
                                    self._uncertainties,
                                ),
                                clause.strip(),
                                pattern.pattern,
                            )
                        )
            if not mentions:
                output[label_id] = NoteLabel(
                    label_id,
                    NoteMentionState.NOT_MENTIONED,
                )
                continue
            state, clause, pattern = _resolve_mentions(mentions)
            output[label_id] = NoteLabel(label_id, state, clause, pattern)
        return output


def _split_clauses(text: str) -> list[str]:
    return [
        clause.strip()
        for clause in re.split(r"\||[\r\n]+|(?<=[.;])\s+", text)
        if clause.strip()
    ]


def _mention_state(
    clause: str,
    match_start: int,
    negations: Iterable[re.Pattern],
    uncertainties: Iterable[re.Pattern],
) -> NoteMentionState:
    prefix_start = max(0, match_start - 45)
    prefix = clause[prefix_start:match_start]
    negation_matches = [
        match
        for pattern in negations
        for match in pattern.finditer(prefix)
        if match.end() >= len(prefix) - 30
    ]
    for negation in negation_matches:
        bridge = prefix[negation.end():]
        causal_context = re.search(
            r"\b(?:because\s+of|due\s+to|secondary\s+to)\b",
            bridge,
            re.IGNORECASE,
        )
        if causal_context is None:
            return NoteMentionState.NEGATIVE
    if any(pattern.search(clause) for pattern in uncertainties):
        return NoteMentionState.UNCERTAIN
    return NoteMentionState.POSITIVE


def _resolve_mentions(
    mentions: list[tuple[NoteMentionState, str, str]],
) -> tuple[NoteMentionState, str, str]:
    priority = {
        NoteMentionState.POSITIVE: 3,
        NoteMentionState.UNCERTAIN: 2,
        NoteMentionState.NEGATIVE: 1,
        NoteMentionState.NOT_MENTIONED: 0,
    }
    return max(mentions, key=lambda item: priority[item[0]])


# Functional/mental status at discharge is a short, templated fill-in-
# the-blank field ("Activity Status: Ambulatory - Independent." /
# "Mental Status: Clear and coherent."), not free-flowing prose with
# negation phrasing -- calibrated against real discharge.csv samples
# (~76% of discharge summaries have these sub-fields nested inside the
# "Discharge Condition" section). ClinicalNoteParser's negation/
# uncertainty heuristic is built for prose ("no evidence of X") and
# would not fit this field's semantics (the absence of an impairment
# keyword IS the confirmed-negative signal, there is no "no"/"denies"
# to detect) -- so these use dedicated small classifiers instead of
# NOTE_PATTERNS/ClinicalNoteParser.
_ACTIVITY_IMPAIRED_RE = re.compile(
    r"\bassistance\b|\bassist\b|\bbedbound\b|\bbedrest\b|\bwheelchair\b|"
    r"non-?ambulatory|not ambulatory|\bsupervision\b|"
    r"non-?weight bearing",
    re.IGNORECASE,
)
_ACTIVITY_INDEPENDENT_RE = re.compile(r"\bindependent\b", re.IGNORECASE)
_MENTAL_STATUS_IMPAIRED_RE = re.compile(
    r"\bconfused\b|\bdisoriented\b|\blethargic\b|\bobtund|\bsedated\b|"
    r"\bintubated\b",
    re.IGNORECASE,
)
_MENTAL_STATUS_CLEAR_RE = re.compile(
    r"clear and coherent|alert and (?:interactive|oriented)|"
    r"awake,?\s*alert|alert,?\s*(?:and\s*)?oriented|"
    r"\bA(?:&O|AO|O)x\d|normal level of (?:arousal|alertness)",
    re.IGNORECASE,
)


def classify_activity_status(value: str | None) -> NoteMentionState:
    text = str(value or "").strip()
    if not text:
        return NoteMentionState.NOT_MENTIONED
    if _ACTIVITY_IMPAIRED_RE.search(text):
        return NoteMentionState.POSITIVE
    if _ACTIVITY_INDEPENDENT_RE.search(text):
        return NoteMentionState.NEGATIVE
    return NoteMentionState.UNCERTAIN


def classify_mental_status(value: str | None) -> NoteMentionState:
    text = str(value or "").strip()
    if not text:
        return NoteMentionState.NOT_MENTIONED
    if _MENTAL_STATUS_IMPAIRED_RE.search(text):
        return NoteMentionState.POSITIVE
    if _MENTAL_STATUS_CLEAR_RE.search(text):
        return NoteMentionState.NEGATIVE
    return NoteMentionState.UNCERTAIN
