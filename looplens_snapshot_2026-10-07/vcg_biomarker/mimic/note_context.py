"""Context resolution for clinical note mentions, and the leakage rule.

Three things decide whether a phrase in a note means what a naive keyword
match takes it to mean, and the existing parser only handled the first:

1. **Negation** — "no evidence of pulmonary edema". Already covered.
2. **Experiencer** — "father had an MI at 50" says nothing about this
   patient. Family-history phrasing is common in exactly the sections that
   mention cardiac disease most often, so ignoring it inflates prevalence.
3. **Temporality** — "history of atrial fibrillation" and "developed atrial
   fibrillation on day 4" are different facts. A historical mention is a
   covariate; an incident one is an outcome. Conflating them turns a
   prediction target into something already true at baseline.

Beyond parsing there is a harder problem, and it is the reason this module
exists at all: **a discharge summary is written at the end of the admission.**
Treating anything in it as available at the time of an ECG recorded on day 1
is leakage — the note already knows the outcome. Radiology reports carry their
own timestamp and can be used, but only for reports that precede the ECG.

``note_availability_state`` enforces that distinction, and it is deliberately
strict: a note whose timestamp is unknown counts as unavailable rather than
being given the benefit of the doubt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


NOTE_CONTEXT_VERSION = "1.0.0"

EXPERIENCER_PATIENT = "patient"
EXPERIENCER_OTHER = "other_person"

TEMPORALITY_CURRENT = "current"
TEMPORALITY_HISTORICAL = "historical"
TEMPORALITY_HYPOTHETICAL = "hypothetical"

# Phrases that assign a finding to somebody other than the patient. Matched in
# the clause preceding the finding, which is where the attribution sits in
# English clinical prose.
FAMILY_EXPERIENCER_PATTERNS: tuple[str, ...] = (
    r"\bfamily history\b",
    r"\bfather\b",
    r"\bmother\b",
    r"\bbrother\b",
    r"\bsister\b",
    r"\bsibling\b",
    r"\bparents?\b",
    r"\bgrandfather\b",
    r"\bgrandmother\b",
    r"\bson\b",
    r"\bdaughter\b",
    r"\baunt\b",
    r"\buncle\b",
    r"\bcousin\b",
    r"\bmaternal\b",
    r"\bpaternal\b",
    r"\brelatives?\b",
)

# Phrases that place a finding before this encounter.
HISTORICAL_PATTERNS: tuple[str, ...] = (
    r"\bhistory of\b",
    r"\bh/o\b",
    r"\bpmh\b",
    r"\bpast medical history\b",
    r"\bprior\b",
    r"\bpreviously\b",
    r"\bin the past\b",
    r"\bstatus post\b",
    r"\bs/p\b",
    r"\bchronic\b",
    r"\blongstanding\b",
    r"\byears? ago\b",
    r"\bmonths? ago\b",
    r"\bremote\b",
)

# Phrases that make a finding conditional rather than asserted.
HYPOTHETICAL_PATTERNS: tuple[str, ...] = (
    r"\bif\b",
    r"\bshould\b",
    r"\bin case of\b",
    r"\bto rule out\b",
    r"\breturn (?:if|for)\b",
    r"\bcall (?:if|for)\b",
    r"\bwatch for\b",
    r"\brisk of\b",
    r"\bprophylaxis\b",
)

# How far back the attribution scan reaches. Long enough to catch the clause
# subject, short enough that an unrelated earlier sentence does not bleed in.
_CONTEXT_WINDOW_CHARACTERS = 90


@dataclass(frozen=True)
class MentionContext:
    experiencer: str
    temporality: str
    experiencer_cue: str
    temporality_cue: str

    def to_dict(self) -> dict[str, str]:
        return {
            "experiencer": self.experiencer,
            "temporality": self.temporality,
            "experiencer_cue": self.experiencer_cue,
            "temporality_cue": self.temporality_cue,
        }

    @property
    def counts_for_patient(self) -> bool:
        return self.experiencer == EXPERIENCER_PATIENT

    @property
    def is_incident(self) -> bool:
        return (
            self.experiencer == EXPERIENCER_PATIENT
            and self.temporality == TEMPORALITY_CURRENT
        )


def _compiled(patterns: Iterable[str]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_FAMILY = _compiled(FAMILY_EXPERIENCER_PATTERNS)
_HISTORICAL = _compiled(HISTORICAL_PATTERNS)
_HYPOTHETICAL = _compiled(HYPOTHETICAL_PATTERNS)


def resolve_mention_context(
    text: str,
    match_start: int,
    *,
    section_name: str = "",
    window: int = _CONTEXT_WINDOW_CHARACTERS,
) -> MentionContext:
    """Attribute one mention to an experiencer and a point in time.

    The section header is authoritative when it is explicit: a finding inside
    "Family History" belongs to a relative regardless of local phrasing, and a
    finding inside "Past Medical History" is historical by construction.
    """

    body = str(text or "")
    start = max(0, int(match_start) - int(window))
    prefix = body[start : int(match_start)]
    section = str(section_name or "").casefold()

    experiencer = EXPERIENCER_PATIENT
    experiencer_cue = ""
    if "family history" in section:
        experiencer = EXPERIENCER_OTHER
        experiencer_cue = "section:family_history"
    else:
        for pattern in _FAMILY:
            found = pattern.search(prefix)
            if found is not None:
                experiencer = EXPERIENCER_OTHER
                experiencer_cue = found.group(0)
                break

    temporality = TEMPORALITY_CURRENT
    temporality_cue = ""
    if "past medical history" in section or "history" in section:
        temporality = TEMPORALITY_HISTORICAL
        temporality_cue = f"section:{section.strip()[:40]}"
    else:
        for pattern in _HYPOTHETICAL:
            found = pattern.search(prefix)
            if found is not None:
                temporality = TEMPORALITY_HYPOTHETICAL
                temporality_cue = found.group(0)
                break
        if temporality == TEMPORALITY_CURRENT:
            for pattern in _HISTORICAL:
                found = pattern.search(prefix)
                if found is not None:
                    temporality = TEMPORALITY_HISTORICAL
                    temporality_cue = found.group(0)
                    break
    return MentionContext(
        experiencer=experiencer,
        temporality=temporality,
        experiencer_cue=experiencer_cue,
        temporality_cue=temporality_cue,
    )


# --- leakage control ------------------------------------------------------

AVAILABILITY_AVAILABLE = "available_before_ecg"
AVAILABILITY_FUTURE = "written_after_ecg"
AVAILABILITY_UNKNOWN = "timestamp_unknown"
AVAILABILITY_END_OF_STAY = "end_of_encounter_document"

# Note kinds written at the end of an admission. Their content summarises the
# whole stay, so nothing in them can be an ECG-time predictor.
END_OF_ENCOUNTER_NOTE_KINDS: frozenset[str] = frozenset(
    {"discharge", "discharge_summary", "dc_summary"}
)


def note_availability_state(
    note_kind: Iterable[str],
    note_time: Iterable[object],
    ecg_time: Iterable[object],
) -> pd.Series:
    """Classify each note as usable at ECG time, or not.

    A discharge summary is never usable as a predictor, whatever its recorded
    timestamp says. Everything else needs a timestamp strictly before the ECG;
    a missing timestamp is treated as unusable, because the alternative --
    assuming it came first -- is exactly how outcome information leaks in.
    """

    kinds = pd.Series(list(note_kind), dtype="object").astype("string").str.casefold()
    note_times = pd.to_datetime(pd.Series(list(note_time)), errors="coerce")
    ecg_times = pd.to_datetime(pd.Series(list(ecg_time)), errors="coerce")
    state = pd.Series(AVAILABILITY_UNKNOWN, index=kinds.index, dtype="object")
    end_of_stay = kinds.isin(END_OF_ENCOUNTER_NOTE_KINDS).fillna(False)
    known = note_times.notna() & ecg_times.notna()
    state.loc[known & (note_times < ecg_times)] = AVAILABILITY_AVAILABLE
    state.loc[known & (note_times >= ecg_times)] = AVAILABILITY_FUTURE
    state.loc[end_of_stay.to_numpy()] = AVAILABILITY_END_OF_STAY
    return state


def assert_no_predictor_leakage(
    frame: pd.DataFrame,
    *,
    availability_column: str = "note_availability",
    role_column: str = "note_role",
) -> None:
    """Fail when a note used as a predictor was not available at ECG time.

    ``note_role`` is ``"predictor"`` or ``"outcome_or_context"``. Notes used
    to define an outcome may of course come from later in the stay; notes used
    to predict must not.
    """

    if frame.empty:
        return
    missing = sorted({availability_column, role_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Leakage check is missing columns: {missing}")
    predictors = frame.loc[
        frame[role_column].astype("string").str.casefold().eq("predictor")
    ]
    offending = predictors.loc[
        ~predictors[availability_column].isin([AVAILABILITY_AVAILABLE])
    ]
    if not offending.empty:
        states = (
            offending[availability_column].value_counts().to_dict()
        )
        raise AssertionError(
            "Notes used as predictors were not available at ECG time: "
            f"{states}. A discharge summary describes the outcome of the "
            "stay; using it to predict an event inside that stay is leakage."
        )


def annotate_note_usage(
    frame: pd.DataFrame,
    *,
    note_kind_column: str = "note_kind",
    note_time_column: str = "note_time",
    ecg_time_column: str = "ecg_time",
    role: str = "outcome_or_context",
) -> pd.DataFrame:
    """Attach availability state and role to a note-mention frame."""

    output = frame.copy()
    output["note_availability"] = note_availability_state(
        output.get(note_kind_column, pd.Series("", index=output.index)),
        output.get(note_time_column, pd.Series(pd.NaT, index=output.index)),
        output.get(ecg_time_column, pd.Series(pd.NaT, index=output.index)),
    ).to_numpy()
    output["note_role"] = str(role)
    output["usable_as_ecg_time_predictor"] = output["note_availability"].eq(
        AVAILABILITY_AVAILABLE
    )
    output["note_context_version"] = NOTE_CONTEXT_VERSION
    return output


# --- validation sample ----------------------------------------------------


def draw_note_validation_sample(
    mentions: pd.DataFrame,
    *,
    per_label: int = 25,
    random_seed: int = 20260812,
    label_column: str = "label_id",
    state_column: str = "state",
) -> pd.DataFrame:
    """Stratified sample of note mentions for manual annotation.

    Every curated pattern set is a hypothesis about how clinicians write. The
    only way to find out whether ``NOTE_PATTERNS`` actually captures what it
    claims is to read a sample and compare. Stratified by label and by
    resolved state so that negatives and uncertain mentions -- where pattern
    matching fails most often -- are represented rather than swamped.
    """

    if mentions.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                label_column,
                state_column,
                "matched_clause",
                "annotation_state",
                "annotation_experiencer",
                "annotation_temporality",
            ]
        )
    for column in (label_column, state_column):
        if column not in mentions.columns:
            raise ValueError(f"Note mentions are missing column: {column}")
    generator = np.random.default_rng(int(random_seed))
    parts: list[pd.DataFrame] = []
    for (label, state), group in mentions.groupby(
        [label_column, state_column],
        sort=True,
    ):
        take = min(int(per_label), len(group))
        if take <= 0:
            continue
        indices = generator.choice(len(group), size=take, replace=False)
        parts.append(group.iloc[np.sort(indices)])
    if not parts:
        return mentions.head(0).copy()
    sample = pd.concat(parts, ignore_index=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    # Blank annotation columns for the human reviewer; the pipeline's own
    # verdict stays in ``state`` so agreement can be computed afterwards.
    sample["annotation_state"] = ""
    sample["annotation_experiencer"] = ""
    sample["annotation_temporality"] = ""
    sample["annotation_version"] = NOTE_CONTEXT_VERSION
    return sample


def note_pattern_validation_metrics(
    annotated: pd.DataFrame,
    *,
    state_column: str = "state",
    annotation_column: str = "annotation_state",
) -> dict[str, object]:
    """Precision/recall of the pattern set against manual annotation."""

    required = {state_column, annotation_column}
    missing = sorted(required - set(annotated.columns))
    if missing:
        raise ValueError(f"Annotated sample is missing columns: {missing}")
    working = annotated.copy()
    predicted = working[state_column].astype("string").str.casefold()
    truth = working[annotation_column].astype("string").str.casefold()
    scored = working.loc[truth.notna() & truth.ne("")]
    if scored.empty:
        return {
            "status": "not_annotated",
            "n_annotated": 0,
            "precision": float("nan"),
            "recall": float("nan"),
        }
    predicted = predicted.loc[scored.index].eq("positive")
    truth_positive = truth.loc[scored.index].eq("positive")
    true_positive = int((predicted & truth_positive).sum())
    false_positive = int((predicted & ~truth_positive).sum())
    false_negative = int((~predicted & truth_positive).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else float("nan")
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    return {
        "status": "evaluable",
        "n_annotated": int(len(scored)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "note_context_version": NOTE_CONTEXT_VERSION,
    }


__all__ = [
    "AVAILABILITY_AVAILABLE",
    "AVAILABILITY_END_OF_STAY",
    "AVAILABILITY_FUTURE",
    "AVAILABILITY_UNKNOWN",
    "EXPERIENCER_OTHER",
    "EXPERIENCER_PATIENT",
    "MentionContext",
    "NOTE_CONTEXT_VERSION",
    "TEMPORALITY_CURRENT",
    "TEMPORALITY_HISTORICAL",
    "TEMPORALITY_HYPOTHETICAL",
    "annotate_note_usage",
    "assert_no_predictor_leakage",
    "draw_note_validation_sample",
    "note_availability_state",
    "note_pattern_validation_metrics",
    "resolve_mention_context",
]
