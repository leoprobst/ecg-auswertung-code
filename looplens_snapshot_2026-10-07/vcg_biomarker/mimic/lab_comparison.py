"""Descriptive, paired local measurements; never a diagnostic change classifier."""
from __future__ import annotations

import math


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def paired_intervals(current: dict, comparator: dict) -> list[dict]:
    """Require a declared common measurement protocol before computing deltas.

    No chronological ordering, statistical significance or measurement-error
    confidence interval can be inferred from two records.
    """
    def index(payload):
        delineation = payload.get('delineation')
        rows = delineation.get("interval_measurements", []) if isinstance(delineation, dict) else []
        result = {}
        duplicates = set()
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not isinstance(row.get("interval"), str) or not row['interval']:
                continue
            name = row["interval"]
            if name in result:
                duplicates.add(name)
            result[name] = row
        for name in duplicates:
            result[name] = {"status": "ambiguous_duplicate"}
        return result

    left, right = index(current), index(comparator)
    protocol = current.get("measurement_protocol")
    compatible = (isinstance(protocol, dict)
                  and all(isinstance(protocol.get(k), str) and bool(protocol[k].strip())
                          for k in ('schema_version', 'implementation_hash'))
                  and protocol == comparator.get("measurement_protocol"))
    processing = current.get("lab_processing", {})
    other = comparator.get("lab_processing", {})
    processing = processing if isinstance(processing, dict) else {}
    other = other if isinstance(other, dict) else {}
    keys = ("analysis_sampling_rate_hz", "powerline_hz")
    compatible = compatible and all(k in processing and k in other and processing[k] == other[k] for k in keys)
    compatible = compatible and all(
        _finite(p.get('analysis_sampling_rate_hz')) is not None
        and p['analysis_sampling_rate_hz'] > 0
        and _finite(p.get('powerline_hz')) is not None
        and p['powerline_hz'] in (0, 50, 60) for p in (processing, other))
    rows = []
    for name in sorted(left.keys() | right.keys()):
        a, b = left.get(name, {}), right.get(name, {})
        x, y = _finite(a.get("value_ms")), _finite(b.get("value_ms"))
        reason = "descriptive_only"
        if not compatible:
            reason = "unknown_or_different_measurement_protocol"
        elif a.get("measurement_method") != b.get("measurement_method"):
            reason = "different_measurement_method"
        elif a.get("status") != "measurable" or b.get("status") != "measurable" or x is None or y is None:
            reason = "missing_or_not_measurable"
        rows.append({"measurement": name, "unit": "ms", "current": x,
                     "comparator": y, "delta_current_minus_comparator": x-y if reason == "descriptive_only" else None,
                     "status": reason, "current_analysis_id": current.get("analysis_id"),
                     "comparator_analysis_id": comparator.get("analysis_id")})
    return rows
