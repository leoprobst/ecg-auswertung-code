"""Portable, bounded extraction intermediates; no clinical classification.

Templates are real, quality-selected cluster exemplars, never mixtures of
different morphologies. Float32 storage is explicitly lossy; hashes bind the
stored representation, not the original acquisition. Raw recordings remain
necessary for alternative preprocessing and arbitrary new signal markers.
"""

from __future__ import annotations

import base64
import hashlib
import json
import zlib
from typing import Any

import numpy as np

VERSION = "1.1.0"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Measured on the A117 10k pilot: a ``metadata_only`` capsule was 35 KB, all
# of it the beat list, and most of that the same source and evidence strings
# repeated for every fiducial of every beat -- ten fiducials, nine beats,
# "fusion_consensus:neurokit+multilead+vcg" ninety times. At 800.000
# recordings that is 28 GB of JSON copied into every analysis cache. Version
# 1.1.0 interns those strings once per capsule, rounds the per-fiducial floats
# to what the delineator can resolve, and stores each fiducial as one array in
# the order ``POINT_FIELDS`` names -- 35 KB becomes about 10 KB, and
# ``resolve_point`` gives the readable dict back.
CONFIDENCE_DECIMALS = 4
MILLISECOND_DECIMALS = 3
POINT_FIELDS = (
    "sample",
    "confidence",
    "uncertainty_ms",
    "conflict_ms",
    "source",
    "evidence_sources",
)
POINTS = (
    "p_onset",
    "p_peak",
    "p_offset",
    "r_onset",
    "r_peak",
    "r_offset",
    "j_point",
    "t_onset",
    "t_peak",
    "t_offset",
)
MAX_TEMPLATE_BYTES = 4 * 1024 * 1024


def _number(value, decimals=None):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    if not np.isfinite(value):
        return None
    return value if decimals is None else round(value, decimals)


def _intern(vocabulary, value):
    """Index of ``value`` in the capsule's vocabulary, appended on first sight."""
    value = str(value)
    try:
        return vocabulary.index(value)
    except ValueError:
        vocabulary.append(value)
        return len(vocabulary) - 1


def resolve_point(capsule, point):
    """A fiducial as a readable dict, strings restored from the vocabulary.

    Accepts the 1.1.0 array form and the 1.0.0 dict form alike, so a reader
    can open either generation of feature partition.
    """
    if capsule.get("version", "1.0.0") not in {"1.0.0", "1.1.0"}:
        raise ValueError("unsupported_measurement_capsule_version")
    if point is None:
        return None
    if isinstance(point, (list, tuple)):
        fields = capsule.get("point_fields")
        if (capsule.get("version") != "1.1.0"
                or not isinstance(fields, (list, tuple))
                or len(fields) != len(POINT_FIELDS)
                or any(not isinstance(field, str) for field in fields)
                or set(fields) != set(POINT_FIELDS)
                or len(point) != len(fields)):
            raise ValueError("invalid_capsule_point_schema")
        point = dict(zip(fields, point))
    if not isinstance(point, dict):
        raise ValueError("invalid_capsule_point")
    vocabulary = capsule.get("fiducial_vocabulary", {})
    sources = vocabulary.get("sources", [])
    evidence = vocabulary.get("evidence_sources", [])
    resolved = dict(point)
    source = point.get("source")
    def decoded(item, values):
        if isinstance(item, bool):
            raise ValueError("invalid_capsule_vocabulary_index")
        if isinstance(item, int):
            if not 0 <= item < len(values):
                raise ValueError("invalid_capsule_vocabulary_index")
            item = values[item]
        if item is not None and not isinstance(item, str):
            raise ValueError("invalid_capsule_vocabulary_value")
        return item

    resolved["source"] = decoded(source, sources)
    evidence_ids = point.get("evidence_sources", [])
    if not isinstance(evidence_ids, (list, tuple)):
        raise ValueError("invalid_capsule_evidence_sources")
    resolved["evidence_sources"] = [
        decoded(item, evidence) for item in evidence_ids
    ]
    sample = resolved.get("sample")
    if (isinstance(sample, bool) or not isinstance(sample, (int, np.integer))
            or sample < 0):
        raise ValueError("invalid_capsule_sample")
    if "signal_samples" in capsule and sample >= capsule["signal_samples"]:
        raise ValueError("capsule_sample_out_of_bounds")
    return resolved


def point_sample(point):
    """The sample index of a stored fiducial in either form, or None."""
    if point is None:
        return None
    if isinstance(point, (list, tuple)):
        return int(point[0])
    return int(point["sample"])


def measurement_capsule(
    signal,
    delineation,
    fs,
    *,
    source,
    record_id,
    raw_reference,
    processing,
    max_clusters=8,
):
    """Reuse already prepared 12-lead morphology; no loading or delineation.

    Cluster count is bounded in the archive only, not in biomarker computation.
    Unknown cluster membership is never pooled into an assumed sinus template.
    All original beat boundaries remain available even for excluded exemplars.
    """
    x = np.asarray(signal)
    if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("invalid_capsule_signal_or_sampling_rate")
    if not isinstance(max_clusters, int) or not 1 <= max_clusters <= 32:
        raise ValueError("invalid_cluster_storage_budget")
    output = dict(
        version=VERSION,
        source=str(source),
        record_id=str(record_id),
        raw_reference=str(raw_reference),
        sampling_rate_hz=float(fs),
        signal_samples=len(x),
        lead_order=list(LEADS),
        unit="mV",
        processing=processing,
        clinical_claim_allowed=False,
        coordinate_system="original_record_samples",
        template_semantics="observed_cluster_exemplar_not_median",
        point_fields=list(POINT_FIELDS),
        fiducial_vocabulary={"sources": [], "evidence_sources": []},
        beats=[],
        templates=[],
        omitted_clusters=[],
    )
    sources = output["fiducial_vocabulary"]["sources"]
    evidence = output["fiducial_vocabulary"]["evidence_sources"]
    output["lead_quality_descriptors"] = {
        "finite_fraction": np.mean(np.isfinite(x), axis=0).tolist()
        if len(x)
        else [None] * 12,
        "adjacent_equal_fraction": np.mean(np.diff(x, axis=0) == 0, axis=0).tolist()
        if len(x) > 1
        else [None] * 12,
        "interpretation": "technical_descriptors_not_independent_noise_calibration",
    }
    groups: dict[int, list[tuple[float, int, dict[str, Any]]]] = {}
    for index, beat in enumerate(delineation.beats):
        points = {}
        for name in POINTS:
            point = getattr(beat, name, None)
            points[name] = (
                None
                if point is None
                else [
                    int(point.sample),
                    _number(point.confidence, CONFIDENCE_DECIMALS),
                    _number(getattr(point, "uncertainty_ms", None), MILLISECOND_DECIMALS),
                    _number(getattr(point, "conflict_ms", None), MILLISECOND_DECIMALS),
                    _intern(sources, point.source),
                    [
                        _intern(evidence, item)
                        for item in getattr(point, "evidence_sources", ())
                    ],
                ]
            )
        cluster = getattr(beat, "morphology_cluster_id", None)
        cluster = None if cluster is None else int(cluster)
        reasons = list(getattr(beat, "review_reasons", ()))
        conflict = bool(getattr(beat, "boundary_conflict", False))
        if conflict:
            reasons.append("boundary_conflict")
        if cluster is None:
            reasons.append("unknown_morphology_cluster")
        confidence_method = getattr(beat, "segment_confidence", None)
        confidence = (
            _number(confidence_method("qrs")) if callable(confidence_method) else None
        )
        entry = dict(
            beat_index=index,
            cluster_id=cluster,
            points=points,
            beat_type=str(getattr(beat, "beat_type", "unknown")),
            beat_type_source=str(getattr(beat, "beat_type_source", "")),
            morphology_outlier=bool(getattr(beat, "morphology_outlier", False)),
            qrs_confidence=confidence,
            exclusion_reasons=reasons,
            stored_as_exemplar=False,
        )
        output["beats"].append(entry)
        q, end = point_sample(points["r_onset"]), point_sample(points["r_offset"])
        if (
            cluster is not None
            and not conflict
            and confidence is not None
            and confidence >= 0.7
            and q is not None
            and end is not None
            and 0 <= q < end < len(x)
        ):
            groups.setdefault(int(cluster), []).append((confidence, index, entry))
    used_bytes = 0
    for rank, cluster in enumerate(sorted(groups)):
        if rank >= max_clusters:
            output["omitted_clusters"].append(
                dict(cluster_id=cluster, reason="cluster_budget")
            )
            continue
        _, index, entry = sorted(groups[cluster], key=lambda item: (-item[0], item[1]))[
            0
        ]
        points = entry["points"]
        r = point_sample(points["r_peak"])
        p_onset = point_sample(points["p_onset"])
        t_offset = point_sample(points["t_offset"])
        left = min(p_onset if p_onset is not None else r, r - round(0.3 * fs))
        right = max(t_offset if t_offset is not None else r, r + round(0.6 * fs)) + 1
        start, stop = max(0, left), min(len(x), right)
        # Corrupt distant fiducials must not cause an unbounded signal copy.
        if stop <= start or stop - start > round(3 * fs):
            output["omitted_clusters"].append(
                dict(cluster_id=cluster, reason="invalid_template_extent")
            )
            continue
        values = np.asarray(x[start:stop], dtype="<f4")
        if (
            not np.isfinite(values).all()
            or used_bytes + values.nbytes > MAX_TEMPLATE_BYTES
        ):
            output["omitted_clusters"].append(
                dict(cluster_id=cluster, reason="nonfinite_or_storage_budget")
            )
            continue
        raw = values.tobytes()
        encoded = base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")
        used_bytes += values.nbytes
        entry["stored_as_exemplar"] = True
        output["templates"].append(
            dict(
                cluster_id=cluster,
                beat_index=index,
                start_sample=start,
                shape=list(values.shape),
                dtype="<f4",
                encoding="zlib_base64",
                sha256=hashlib.sha256(raw).hexdigest(),
                data=encoded,
                boundary_clipped=(start != left or stop != right),
            )
        )
    output["template_uncompressed_bytes"] = used_bytes
    output["status"] = "available" if output["templates"] else "metadata_only"
    return output


def decode_template(template):
    """Bounded decompression with shape, byte length and digest validation."""
    shape = template.get("shape")
    if (
        template.get("dtype") != "<f4"
        or template.get("encoding") != "zlib_base64"
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n <= 0 for n in shape)
        or shape[1] != 12
    ):
        raise ValueError("invalid_template_contract")
    size = shape[0] * shape[1] * 4
    if size > MAX_TEMPLATE_BYTES or len(template["data"]) > MAX_TEMPLATE_BYTES * 2:
        raise ValueError("template_budget_exceeded")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(
        base64.b64decode(template["data"], validate=True), size + 1
    )
    if (
        len(raw) != size
        or not decoder.eof
        or decoder.unused_data
        or hashlib.sha256(raw).hexdigest() != template["sha256"]
    ):
        raise ValueError("template_integrity_failure")
    values = np.frombuffer(raw, dtype="<f4").reshape(shape)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_template")
    return values


def capsule_json(*args, **kwargs):
    return json.dumps(
        measurement_capsule(*args, **kwargs), allow_nan=False, separators=(",", ":")
    )
