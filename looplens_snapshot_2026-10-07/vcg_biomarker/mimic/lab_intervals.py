"""SQL expressions for assay bounds; a bound is never an exact measurement."""
from __future__ import annotations


def censored_result_sql(value_sql: str | None = None) -> tuple[str, str]:
    number = r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:e[+-]?[0-9]+)?)"
    operator = r"(<=|>=|<|>|less than or equal to|greater than or equal to|less than|greater than)"
    pattern = rf"^\s*{operator}\s*{number}"
    qualifier = rf"^\s*{operator}"
    comment = "LOWER(COALESCE(comments, ''))"
    text = comment if value_sql is None else (
        f"CASE WHEN regexp_matches(LOWER(COALESCE({value_sql}, '')), '{qualifier}') "
        f"THEN LOWER({value_sql}) ELSE {comment} END"
    )
    op = f"regexp_extract({text}, '{pattern}', 1)"
    bound = f"TRY_CAST(regexp_extract({text}, '{pattern}', 2) AS DOUBLE)"
    censor = (
        f"CASE WHEN isfinite({bound}) THEN CASE {op} "
        "WHEN '<' THEN 'below_limit' WHEN 'less than' THEN 'below_limit' "
        "WHEN '<=' THEN 'at_or_below_limit' "
        "WHEN 'less than or equal to' THEN 'at_or_below_limit' "
        "WHEN '>' THEN 'above_limit' WHEN 'greater than' THEN 'above_limit' "
        "WHEN '>=' THEN 'at_or_above_limit' "
        "WHEN 'greater than or equal to' THEN 'at_or_above_limit' END END"
    )
    numeric = (f"CASE WHEN {censor} IS NOT NULL THEN {bound} "
               f"WHEN regexp_matches({text}, '{qualifier}') THEN NULL::DOUBLE "
               "ELSE TRY_CAST(valuenum AS DOUBLE) END")
    return numeric, censor


def upper_reference_state_sql(bound: str, censor: str, reference: str) -> str:
    """Classify the whole measurement interval against x > upper reference."""
    return f"""CASE
        WHEN NOT COALESCE(isfinite({bound}) AND isfinite({reference}) AND {reference} > 0, FALSE)
            THEN 'indeterminate'
        WHEN {censor} IS NULL THEN CASE WHEN {bound} > {reference}
            THEN 'positive' ELSE 'negative' END
        WHEN ({censor} = 'above_limit' AND {bound} >= {reference})
          OR ({censor} = 'at_or_above_limit' AND {bound} > {reference}) THEN 'positive'
        WHEN {censor} IN ('below_limit', 'at_or_below_limit')
          AND {bound} <= {reference} THEN 'negative'
        ELSE 'indeterminate' END"""


LAB_WINDOWS = {"strict": "delta_hours BETWEEN -3 AND 6",
               "standard": "delta_hours BETWEEN -6 AND 12", "24h": "TRUE"}


def interval_comparison_sql(group: str, operator: str, threshold: str) -> str:
    """True only if every value allowed by the reported bound satisfies a rule."""
    if operator not in ("<", "<=", ">", ">="):
        raise ValueError("Unsupported interval comparison")
    bound, censor = f"{group}_nearest_bound", f"{group}_nearest_censoring"
    direction = "below" if operator.startswith("<") else "above"
    strict = "<" if direction == "below" else ">"
    open_test = f"{bound} {operator} ({threshold})"
    if operator == strict:
        open_test = f"({bound} {strict} ({threshold}) OR {bound} = ({threshold}))"
    return f"""COALESCE((isfinite({bound}) AND isfinite({threshold}) AND (
        ({censor} IS NULL AND {bound} {operator} ({threshold})) OR
        ({censor} = '{direction}_limit' AND {open_test}) OR
        ({censor} = 'at_or_{direction}_limit' AND {bound} {operator} ({threshold}))
    )), FALSE)"""


def interval_window_aggregates(nearest_order: str) -> str:
    expressions = []
    for suffix, window in LAB_WINDOWS.items():
        for state in ("positive", "negative", "indeterminate"):
            expressions.append(
                f"COUNT(*) FILTER (WHERE ({window}) AND value_status = 'usable' "
                f"AND upper_reference_state = '{state}') AS interval_{state}_count_{suffix}"
            )
        for field in ("labevent_id", "delta_hours"):
            expressions.append(
                f"ARG_MIN({field}, {nearest_order}) FILTER (WHERE ({window}) "
                "AND value_status = 'usable' AND upper_reference_state = 'positive') "
                f"AS interval_positive_{field}_{suffix}"
            )
    return ",\n".join(expressions) + ","


def troponin_interval_pivot() -> str:
    expressions = []
    where = "lab_group IN ('troponin_i', 'troponin_t')"
    for suffix in LAB_WINDOWS:
        counts = {state: f"COALESCE(SUM(interval_{state}_count_{suffix}) FILTER (WHERE {where}), 0)"
                  for state in ("positive", "negative", "indeterminate")}
        expressions.append(
            f"CASE WHEN {counts['positive']} > 0 THEN 'positive' "
            f"WHEN {counts['indeterminate']} > 0 THEN 'indeterminate' "
            f"WHEN {counts['negative']} > 0 THEN 'negative' ELSE 'unavailable' END "
            f"AS troponin_interval_state_{suffix}"
        )
        for field in ("labevent_id", "delta_hours"):
            expressions.append(
                f"ARG_MIN(interval_positive_{field}_{suffix}, "
                f"STRUCT_PACK(distance := ABS(interval_positive_delta_hours_{suffix}), "
                f"event_id := interval_positive_labevent_id_{suffix})) FILTER (WHERE {where}) "
                f"AS troponin_interval_positive_{field}_{suffix}"
            )
    return ",\n".join(expressions) + ","
