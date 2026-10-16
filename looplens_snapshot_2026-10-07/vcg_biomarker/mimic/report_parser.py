from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable


class ReportMentionState(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    NOT_MENTIONED = "not_mentioned"


@dataclass(frozen=True)
class ReportLabel:
    label_id: str
    state: ReportMentionState
    matched_clause: str = ""
    matched_pattern: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["state"] = self.state.value
        return data


REPORT_PATTERNS: dict[str, tuple[str, ...]] = {
    "af": (
        r"\batrial fibrillation\b",
        r"\batrial flutter\b",
        r"\ba[\s-]?fib\b",
    ),
    "lbbb": (r"\bleft bundle branch block\b", r"\blbbb\b"),
    "rbbb": (r"\bright bundle branch block\b", r"\brbbb\b"),
    "st_elevation": (r"\bst elevation\b", r"\belevated st\b"),
    "st_depression": (r"\bst depression\b", r"\bdepressed st\b"),
    "ischemia": (r"\bisch(?:a?e)mi(?:a|c)\b",),
    "infarct": (r"\binfarct(?:ion)?\b", r"\bmyocardial infarction\b"),
    "lvh": (r"\bleft ventricular hypertrophy\b", r"\blvh\b"),
    "rvh": (r"\bright ventricular hypertrophy\b", r"\brvh\b"),
    "lae": (r"\bleft atrial (?:enlargement|abnormality)\b", r"\blae\b"),
    "rae": (r"\bright atrial (?:enlargement|abnormality)\b", r"\brae\b"),
    "qt_prolongation": (r"\bprolonged qt(?:c)?\b", r"\bqt(?:c)? prolongation\b"),
    # Measured over all 800.035 machine reports: the three literal phrases
    # this used to list -- "paced rhythm", "pacemaker", "ventricular pacing" --
    # matched 28.858 of 46.419 pacing mentions and missed 17.561, among them
    # "demand pacing" (6.608), "atrial pacing" (4.691) and "demand atrial
    # pacing" (2.695). That is not a cosmetic gap: this label is what excludes
    # paced recordings from the voltage, Q-wave and ischaemia criteria, so
    # roughly 14.000 paced recordings were being read by rules that do not
    # apply to them. The stems below match 45.127 of the 46.419 and nothing
    # outside the pacing family.
    #
    # "pacer" is deliberately absent: the remaining 1.292 are "pacer detection
    # suspended due to external noise", which reports a failure to detect a
    # pacemaker rather than the presence of one.
    "pacemaker": (r"\bpacemaker\b", r"\bpacing\b", r"\bpaced\b"),
    "low_voltage": (r"\blow voltage\b", r"\blow qrs voltage\b"),
}

NEGATION_PATTERNS = (
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnegative for\b",
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


class MachineReportParser:
    def __init__(
        self,
        patterns: dict[str, tuple[str, ...]] = REPORT_PATTERNS,
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

    def parse(self, report: str | None) -> dict[str, ReportLabel]:
        text = str(report or "").strip()
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
                output[label_id] = ReportLabel(
                    label_id,
                    ReportMentionState.NOT_MENTIONED,
                )
                continue
            state, clause, pattern = _resolve_mentions(mentions)
            output[label_id] = ReportLabel(label_id, state, clause, pattern)
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
) -> ReportMentionState:
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
            return ReportMentionState.NEGATIVE
    if any(pattern.search(clause) for pattern in uncertainties):
        return ReportMentionState.UNCERTAIN
    return ReportMentionState.POSITIVE


def _resolve_mentions(
    mentions: list[tuple[ReportMentionState, str, str]],
) -> tuple[ReportMentionState, str, str]:
    priority = {
        ReportMentionState.POSITIVE: 3,
        ReportMentionState.UNCERTAIN: 2,
        ReportMentionState.NEGATIVE: 1,
        ReportMentionState.NOT_MENTIONED: 0,
    }
    return max(mentions, key=lambda item: priority[item[0]])
