"""Specimen-quality rules for laboratory results (audit D2: hemolysis).

MIMIC-IV writes hemolysis into ``labevents.comments`` of the affected result
row, e.g. "HEMOLYSIS FALSELY ELEVATES K", "SLIGHTLY HEMOLYZED SPECIMEN",
"SPECIMEN NOT HEMOLYZED". Measured on the event layer: 114,795 serum
potassium results carry such a comment; 20.4 % of the positive index ECGs of
``hyperkalemia_near_ecg`` rested on one (audit D2). Red-cell lysis releases
potassium, magnesium and phosphate, so these values are pseudo-elevations and
are treated as unusable, not as measurements (Wilson et al. 2020,
PMID 33207176). Any other analyte is unusable only when the laboratory wrote
on that very row that hemolysis falsified it ("falsely elevates this test",
seen on bicarbonate); a bare "hemolyzed specimen" on a coagulation time is
kept and flagged.
"""
from __future__ import annotations

from .mappings import LAB_ITEM_GROUPS

HEMOLYSIS_RULE_VERSION = "1.0.0"
HEMOLYSIS_SENSITIVE_GROUPS = ("potassium", "magnesium", "phosphate")
HEMOLYSIS_SENSITIVE_ITEMIDS = tuple(
    sorted(
        itemid
        for group in HEMOLYSIS_SENSITIVE_GROUPS
        for itemid in LAB_ITEM_GROUPS[group]
    )
)
# Negations are removed before the positive test: "specimen not hemolyzed",
# "no hemolysis present", "not hemolysed".
_NEGATION = r"(specimen (was |is )?)?(not|no) (hemoly|haemoly)[a-z]*( present)?"
_FALSIFICATION = r"(hemoly|haemoly)[a-z]*\s+(may\s+)?falsely"


def _lowered(comments_sql: str) -> str:
    return f"lower(COALESCE(CAST({comments_sql} AS VARCHAR), ''))"


def hemolysis_state_sql(comments_sql: str) -> str:
    """'reported', 'explicitly_absent' or NULL for one result row."""
    lowered = _lowered(comments_sql)
    remainder = f"regexp_replace({lowered}, '{_NEGATION}', '', 'g')"
    return (
        f"CASE WHEN regexp_matches({remainder}, '(hemoly|haemoly)') "
        "THEN 'reported' "
        f"WHEN regexp_matches({lowered}, '(hemoly|haemoly)') "
        "THEN 'explicitly_absent' END"
    )


def hemolysis_invalidates_sql(comments_sql: str, itemid_sql: str) -> str:
    """TRUE when the reported hemolysis makes this result unusable."""
    lowered = _lowered(comments_sql)
    sensitive = ", ".join(str(item) for item in HEMOLYSIS_SENSITIVE_ITEMIDS)
    return (
        f"COALESCE(({hemolysis_state_sql(comments_sql)}) = 'reported' AND ("
        f"TRY_CAST({itemid_sql} AS INTEGER) IN ({sensitive}) "
        f"OR regexp_matches({lowered}, '{_FALSIFICATION}')), FALSE)"
    )


def without_hemolyzed_results(relation_sql: str) -> str:
    """Lab relation without hemolysis-invalidated results (needs comments)."""
    return (
        f"(SELECT * FROM {relation_sql} "
        f"WHERE NOT {hemolysis_invalidates_sql('comments', 'itemid')})"
    )
