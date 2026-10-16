"""Declared data dependencies, deterministic ordering, no implicit parallelism.

The graph exposes safe scheduling opportunities; resource/benchmark gates still
decide whether independent nodes may execute concurrently.
"""
VERSION = "development-workflow-dag-1.0.0"


def dependency_graph(stages):
    if len(stages) != len(set(stages)):
        raise ValueError("Duplicate workflow stage ids")
    selected = set(stages)
    features = "extract-features" if "extract-features" in selected else "extract-features-pilot-1000"
    validation = "validate" if "validate" in selected else "validate-pilot-1000"
    deps = {
        "inventory": (), "genetic-feasibility": ("inventory",), "benchmark": ("inventory",),
        "ingest": ("inventory", "benchmark"), "ingest-events": ("ingest",),
        "build-index": ("ingest", "ingest-events"), "parse-reports": ("build-index",),
        "parse-notes": ("build-index",), "build-evidence": ("parse-reports", "parse-notes"),
        "attest-event-evidence": ("build-evidence",), "build-labels": ("attest-event-evidence",),
        "attest-labels": ("build-labels",), "validate-delineation": ("inventory",),
        "audit-biomarkers": ("validate-delineation", "attest-labels"),
        "extract-features-pilot-1000": ("audit-biomarkers",),
        "validate-pilot-1000": ("extract-features-pilot-1000",),
        "extract-features-pilot-10000": ("validate-pilot-1000",),
        "validate-pilot-10000": ("extract-features-pilot-10000",),
        "pilot-go-no-go": ("validate-pilot-1000", "validate-pilot-10000"),
        "extract-features": ("validate-pilot-1000", "validate-pilot-10000"),
        "therapy-response": (features, "attest-labels"),
        "physiological-gap-markers": ("therapy-response",),
        "marker-redundancy": (features, "physiological-gap-markers"),
        "repeatability": (features,), "validate": ("marker-redundancy", "repeatability"),
        "external-validation": (validation,), "discover-phenome": (validation,),
        "control-panel": (validation,), "segment-availability": (features, "attest-labels"),
        "ext-icd-crosscheck": ("attest-labels",), "reference-ranges": (features,),
        "transform-agreement": (features,), "population-cohorts": ("inventory",),
        "two-stage-selection": (validation,), "screen-broad-axes": (validation,),
        "within-subject-change": (features, "attest-labels"),
        "voltage-mass-discordance": (features, "attest-labels"),
        "time-to-event": (features, "attest-labels"), "occlusion-yield": (features, "attest-labels"),
        "thesis-tests": (features, "attest-labels", "reference-ranges"),
        "score-discovery": (validation, "screen-broad-axes", "marker-redundancy"),
        "reporting-guideline": (), "report": ("reporting-guideline",), "audit": ("report",),
    }
    unknown = selected - set(deps)
    if unknown:
        raise ValueError(f"Undeclared workflow stages: {sorted(unknown)}")
    # Final reporting is a barrier after every required analysis, including the
    # bounded score pass; a separate deep search is not part of this graph.
    deps["reporting-guideline"] = tuple(s for s in stages if s not in {"reporting-guideline", "report", "audit"})
    return {s: tuple(d for d in deps[s] if d in selected and d != s) for s in stages}


def ordered_stages(stages):
    graph = dependency_graph(stages)
    remaining, result = list(stages), []
    completed = set()
    while remaining:
        next_stage = next((s for s in remaining if set(graph[s]) <= completed), None)
        if next_stage is None:
            raise ValueError("Cyclic workflow dependencies")
        result.append(next_stage)
        completed.add(next_stage)
        remaining.remove(next_stage)
    return tuple(result)


def critical_path_seconds(stages, durations):
    graph = dependency_graph(stages)
    finish = {}
    for stage in ordered_stages(stages):
        duration = durations.get(stage)
        if duration is None or any(finish[d] is None for d in graph[stage]):
            finish[stage] = None
        else:
            if duration < 0:
                raise ValueError("Negative stage duration")
            finish[stage] = float(duration) + max((finish[d] for d in graph[stage]), default=0.)
    return None if any(v is None for v in finish.values()) else max(finish.values(), default=0.)
