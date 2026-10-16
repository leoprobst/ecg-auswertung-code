"""Descriptive P-wave measurements from already accepted delineation beats."""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Mapping
import math


def accepted_p_wave_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Do not promote candidates or other beats into the representative interval."""
    delineation = payload.get("delineation", {})
    inventory = payload.get("beat_inventory", [])
    if not isinstance(delineation, Mapping) or not isinstance(inventory, list):
        return {"status": "unavailable", "reason": "Keine Schlagevidenz gespeichert"}
    accepted = {
        item.get("beat_index"): item
        for item in inventory if isinstance(item, Mapping)
        and type(item.get("beat_index")) is int and item.get("accepted") is True
    }
    clusters = Counter(
        item.get("morphology_cluster_id") for item in accepted.values()
        if isinstance(item.get("morphology_cluster_id"), (str, int))
        and not isinstance(item.get("morphology_cluster_id"), bool)
    )
    if not clusters:
        return {"status": "unavailable", "reason": "Kein akzeptierter Morphologiecluster"}
    dominant = sorted(clusters, key=lambda key: (-clusters[key], str(key)))[0]
    segments = delineation.get("accepted_beat_wave_segments", [])
    if not isinstance(segments, list):
        return {"status": "unavailable", "reason": "Keine schlagweisen Wellengrenzen gespeichert"}
    values = {}
    for segment in segments:
        if not isinstance(segment, Mapping) or segment.get("segment") != "p_wave" or segment.get("status") != "measurable":
            continue
        index = segment.get("beat_index")
        if type(index) is not int or index not in accepted or accepted[index].get("morphology_cluster_id") != dominant:
            continue
        duration = segment.get("duration_ms")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            continue
        values[index] = float(duration)
    if not values:
        return {"status": "unavailable", "reason": "Keine auswertbare P-Welle im Hauptcluster"}
    return {
        "status": "descriptive", "median_ms": float(median(values.values())),
        "n_beats": len(values), "n_accepted_cluster_beats": clusters[dominant],
        "cluster_id": dominant,
        "reason": "Schlagweiser Median aus akzeptierten P-Segmenten; kein Ersatz für das Intervall des repräsentativen Schlags",
    }
