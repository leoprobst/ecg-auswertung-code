"""Run the marker pipeline on external cohorts and report the site shift.

Three questions, answered separately because they license different claims:

1. **Does the measurement survive the move?** Coverage, abstention rate and
   failure modes on other recorders, populations and sampling rates. A marker
   that only computes on MIMIC hardware has no external validity regardless
   of its AUC.
2. **How far does the marker distribution shift between sites?** Reported as
   a standardised difference against the development cohort. Neither a large
   nor a small marginal shift establishes calibration or threshold validity;
   those require direct outcome-bound validation.
3. **Does the marker agree with expert ECG reading elsewhere?** Reported per
   site and pooled, and labelled as agreement with a reader rather than
   accuracy against an independent measurement.

Frozen artifacts only: directions, thresholds and score coefficients come
from the development run and are applied unchanged. Nothing here refits, and
nothing here touches the locked test.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .external_cohorts import (
    COMPARISON_ECG_READING,
    EXTERNAL_COHORT_VERSION,
    EXTERNAL_ENDPOINT_MAP,
    eligible_for_diagnostic_validation,
    external_claim_scope,
    inventory_external_collection,
    source_powerline_hz,
    source_acquisition_upper_bandwidth_hz,
)
from .statistics import MarkerModel
from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_sha256,
)
from .statistics import (
    classification_metrics,
    evaluate_marker,
    inference_status,
    marker_bootstrap_intervals,
)


EXTERNAL_VALIDATION_VERSION = "1.6.0"

# Legacy constant name: an exploratory distribution-screening threshold,
# never evidence that a clinical operating point can or cannot transfer.
MAXIMUM_TRANSFERABLE_SHIFT = 0.5


def extraction_summary(features: pd.DataFrame) -> dict[str, object]:
    """Separate completed measurements, copy exclusions and unavailable results.

    A processed row is not necessarily a successful extraction. Proven copies
    are deliberately excluded, not technical failures; unidentified failures
    remain in the measurement denominator.
    """
    from .external_identity_audit import measurement_representative_mask

    representative = measurement_representative_mask(features)
    status = features.get("extraction_status", pd.Series("unknown", index=features.index))
    n_selected = len(features)
    n_eligible = int(representative.sum())
    n_ok = int((representative & status.eq("ok")).sum())
    return {
        "n_records_selected": n_selected,
        "n_records_processed": n_selected,
        "n_records_extracted": n_ok,
        "n_records_exact_copy_excluded": int((~representative).sum()),
        "n_records_measurement_denominator": n_eligible,
        "n_records_without_successful_extraction": n_eligible - n_ok,
        "extraction_status_counts_before_copy_filter": {
            str(key): int(value) for key, value in status.fillna("unknown").value_counts().items()
        },
        "extraction_success_rate": n_ok / n_eligible if n_eligible else None,
        "extraction_success_rate_status": "computed" if n_eligible else "no_measurement_denominator",
    }


@dataclass(frozen=True)
class ExternalExtractionOptions:
    marker_ids: tuple[str, ...]
    max_records_per_source: Optional[int] = None
    powerline_hz: Optional[float] = None
    powerline_hz_by_source: Mapping[str, float] = field(default_factory=dict)
    minimum_signal_quality: float = 0.0
    workers: int = 1
    target_sampling_rate_hz: int = 500

    def __post_init__(self) -> None:
        if not self.marker_ids:
            raise ValueError("At least one marker must be requested")
        if self.workers < 1:
            raise ValueError("workers must be at least one")
        if not 0.0 <= float(self.minimum_signal_quality) <= 1.0:
            raise ValueError("minimum_signal_quality must be between zero and one")
        if int(self.target_sampling_rate_hz) < 200:
            raise ValueError("target_sampling_rate_hz must be at least 200")


@dataclass(frozen=True)
class FrozenExternalMarker:
    endpoint_id: str
    panel_id: str
    criterion_id: str
    model: MarkerModel
    marker_model_hash: str


@dataclass(frozen=True)
class FrozenExternalContract:
    development_run_id: str
    freeze_hash: str
    shortlist_hash: str
    markers: tuple[FrozenExternalMarker, ...]
    tracked_file_hashes_verified: bool = True

    @property
    def marker_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.model.feature_id for item in self.markers))

    @property
    def development_reference(self) -> dict[str, tuple[float, float]]:
        # The frozen train mean/SD are part of MarkerModel. If one marker was
        # shortlisted for several endpoints, require the stored reference to
        # be identical rather than silently selecting one endpoint.
        result: dict[str, tuple[float, float]] = {}
        for item in self.markers:
            reference = (item.model.mean, item.model.standard_deviation)
            previous = result.setdefault(item.model.feature_id, reference)
            if not np.allclose(previous, reference, rtol=0.0, atol=0.0):
                raise RuntimeError(
                    "Frozen development references disagree across endpoints "
                    f"for {item.model.feature_id!r}."
                )
        return result


def load_frozen_external_contract(
    development_run_root: Path | str,
) -> FrozenExternalContract:
    """Load only immutable, hash-bound hypotheses for external evaluation."""

    root = Path(development_run_root)
    paths = {
        "freeze": root / "freeze_manifest.json",
        "lock": root / "test_lock.json",
        "models": root / "trained_marker_models.json",
        "shortlist": root / "locked_test_marker_shortlist.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "External validation requires a completed immutable development "
            f"freeze; missing: {missing}"
        )
    freeze = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    payload = freeze.get("payload")
    freeze_hash = str(freeze.get("freeze_hash", ""))
    if (
        not isinstance(payload, dict)
        or not freeze_hash
        or content_hash(payload) != freeze_hash
        or str(lock.get("freeze_hash", "")) != freeze_hash
    ):
        raise RuntimeError("External validation refused: freeze hash is invalid")
    if lock.get("invalidated_at") is not None:
        raise RuntimeError("External validation refused: freeze was invalidated")
    run_id = str(freeze.get("run_id", ""))
    if not run_id or str(lock.get("run_id", "")) != run_id:
        raise RuntimeError("External validation refused: freeze run ID mismatch")

    tracked = dict(payload.get("tracked_file_sha256", {}))
    required_hashes = {
        "trained_marker_models": paths["models"],
        "locked_test_marker_shortlist": paths["shortlist"],
    }
    for key, path in required_hashes.items():
        expected = str(tracked.get(key, ""))
        if not expected or file_sha256(path) != expected:
            raise RuntimeError(
                "External validation refused: frozen artifact changed: "
                f"{key}"
            )

    shortlist = json.loads(paths["shortlist"].read_text(encoding="utf-8"))
    entries = shortlist.get("entries")
    if (
        not isinstance(entries, list)
        or shortlist.get("locked_test_read") is not False
        or str(shortlist.get("development_run_id", "")) != run_id
    ):
        raise RuntimeError("External validation shortlist is invalid")
    shortlist_hash = content_hash(
        {
            "shortlist_version": shortlist.get("shortlist_version"),
            "entries": entries,
        }
    )
    if shortlist_hash != str(shortlist.get("shortlist_hash", "")):
        raise RuntimeError("External validation shortlist hash is invalid")

    models_payload = json.loads(paths["models"].read_text(encoding="utf-8"))
    lookup = {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("panel_id", "")),
            str(row.get("feature_id", "")),
        ): dict(row)
        for row in models_payload.get("models", ())
    }
    markers: list[FrozenExternalMarker] = []
    for entry in entries:
        key = (
            str(entry.get("endpoint_id", "")),
            str(entry.get("panel_id", "")),
            str(entry.get("feature_id", "")),
        )
        raw_model = lookup.get(key)
        if raw_model is None or content_hash(raw_model) != str(
            entry.get("marker_model_hash", "")
        ):
            raise RuntimeError(
                "External validation refused: shortlisted model is missing or "
                f"changed for {key}."
            )
        markers.append(
            FrozenExternalMarker(
                endpoint_id=key[0],
                panel_id=key[1],
                criterion_id=str(entry.get("criterion_id", "")),
                model=MarkerModel.from_dict(raw_model),
                marker_model_hash=str(entry["marker_model_hash"]),
            )
        )
    if not markers:
        raise RuntimeError("External validation freeze has no eligible markers")
    return FrozenExternalContract(
        development_run_id=run_id,
        freeze_hash=freeze_hash,
        shortlist_hash=shortlist_hash,
        markers=tuple(markers),
    )


def extract_external_features(
    inventory: pd.DataFrame,
    options: ExternalExtractionOptions,
    *,
    progress: Optional[Callable[[str], None]] = None,
    known_copy_representatives: Optional[Mapping[str, tuple[str, str, bool]]] = None,
    measurement_archive_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute the requested markers on external records.

    Uses exactly the same preprocessing, delineation and biomarker code as the
    development pipeline -- that identity is the whole point, and is why the
    powerline frequency is the only acquisition parameter allowed to differ
    (50 Hz in Europe and China, 60 Hz in North America).

    A record that fails is recorded with its failure reason rather than
    dropped: the external abstention rate is a result, not an inconvenience.
    """

    from ..core.biomarkers import BiomarkerRegistry
    from ..core.delineation.fusion_delineator import FusionDelineator
    from ..core.external_loader import load_external_ecg_result
    from ..core.preprocessing import (
        construct_robust_morphology_record,
        detect_r_peaks_multilead,
        prepare_signal_paths,
        resample_signal,
    )
    from ..core.vcg_transform import VCGTransformRegistry
    from .signals import assess_signal_quality

    eligible = inventory.loc[
        inventory.get(
            "eligible_for_analytical_validation",
            inventory["eligible_for_diagnostic_validation"] if "eligible_for_diagnostic_validation" in inventory else pd.Series(True, index=inventory.index),
        ).astype(bool)
    ]
    if options.max_records_per_source is not None:
        eligible = (
            eligible.groupby("source_id", sort=False, group_keys=False)
            .head(int(options.max_records_per_source))
            .copy()
        )
    columns = tuple(eligible.columns)
    records = (dict(zip(columns, values)) for values in eligible.itertuples(index=False, name=None))

    def extract_one(record: Mapping[str, object]) -> dict[str, object]:
        source_id = str(record["source_id"])
        powerline = float(
            options.powerline_hz_by_source.get(
                source_id,
                options.powerline_hz
                if options.powerline_hz is not None
                else source_powerline_hz(source_id),
            )
        )
        base: dict[str, object] = {
            "record_id": record["record_id"],
            "source_id": source_id,
            "source_country": record.get("source_country", ""),
            "sampling_rate_hz": record["sampling_rate_hz"],
            "age": record.get("age"),
            "sex": record.get("sex", ""),
            "labels": record.get("labels", ""),
            "patient_id": record.get("patient_id"),
            "source_role": record.get("source_role", "unknown"),
            "reference_contract_source": record.get("reference_contract_source"),
            "patient_identity_status": record.get("patient_identity_status", "unknown"),
            "eligible_for_diagnostic_validation": record.get("eligible_for_diagnostic_validation", True),
            "eligible_for_analytical_validation": record.get("eligible_for_analytical_validation", True),
            "fingerprint_status": "failed",
            "waveform_sha256": None,
            "annotation_exhaustive": record.get("annotation_exhaustive", False),
            "powerline_hz": powerline,
            "external_validation_version": EXTERNAL_VALIDATION_VERSION,
        }
        # Explicit endpoint-specific reference states survive extraction.
        # Missing annotations are not silently reconstructed as negatives.
        base.update({str(key): value for key, value in record.items()
                     if str(key).startswith(("label_state_", "label_coverage_"))})
        header = Path(str(record["header_path"]))
        try:
            loaded = load_external_ecg_result(header.with_suffix(""))
            original_fs = float(loaded.sampling_rate_hz)
            signal = np.asarray(loaded.signal_millivolts, dtype=float)
            from .external_identity_audit import waveform_fingerprint
            # Reuse the already loaded original samples; no second file read.
            base["waveform_sha256"] = waveform_fingerprint(signal, original_fs)
            base["fingerprint_status"] = "measured"
            known_copy = (known_copy_representatives or {}).get(str(base["waveform_sha256"]))
            if known_copy is not None and (known_copy[2] or (source_id, str(record["record_id"])) != known_copy[:2]):
                base["extraction_status"] = "excluded_exact_copy"
                base["extraction_exclusion_reason"] = "hash_matched_historical_measurement_copy"
                return base
            fs = int(options.target_sampling_rate_hz)
            if not np.isclose(original_fs, fs, rtol=0.0, atol=1e-9):
                signal = resample_signal(signal, original_fs, fs)
            base["original_sampling_rate_hz"] = original_fs
            base["analysis_sampling_rate_hz"] = fs
            signal_quality = assess_signal_quality(
                signal,
                fs,
                expected_powerline_hz=powerline,
            )
            base.update(
                {
                    f"signal_quality_{key}": value
                    for key, value in signal_quality.to_dict().items()
                }
            )
            if (
                not signal_quality.usable
                or signal_quality.score < float(options.minimum_signal_quality)
            ):
                base["extraction_status"] = "abstained_signal_qc"
                base["failure_type"] = "SignalQualityAbstention"
                base["failure_message"] = "|".join(
                    signal_quality.reasons
                    or (
                        "below_configured_minimum_signal_quality",
                    )
                )[:300]
                return base
            paths = prepare_signal_paths(
                signal,
                fs,
                powerline=powerline,
            )
            r_peaks = detect_r_peaks_multilead(paths["delineation"], fs)
            if len(r_peaks) < 2:
                raise ValueError(f"only {len(r_peaks)} R peak(s) detected")
            vcg_delineation = VCGTransformRegistry.transform(
                paths["delineation"],
                "kors",
            )
            delineation = FusionDelineator().delineate_record(
                ecg_12lead=paths["delineation"],
                r_peaks=r_peaks,
                fs=fs,
                ecg_id=int(
                    hashlib.sha256(
                        str(record["record_id"]).encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                ),
                vcg_xyz=vcg_delineation.xyz,
            )
            if not delineation.beats:
                raise ValueError("no beats could be delineated")
            robust, _ = construct_robust_morphology_record(
                paths["morphology"],
                delineation,
            )
            vcg = VCGTransformRegistry.transform(robust, "kors")
            record_vcg = VCGTransformRegistry.transform(
                paths["morphology"],
                "kors",
            )
            from ..core.measurement_capsule import capsule_json
            from ..core.measurement_robustness import angle_sensitivity
            base["context_measurement_capsule_json"] = capsule_json(
                paths["morphology"], delineation, fs, source=source_id,
                record_id=record["record_id"], raw_reference=str(header),
                processing={"signal_path": "record_morphology", "powerline_hz": powerline,
                            "original_sampling_rate_hz": original_fs,
                            "baseline": "prepared_signal_path_no_additional_capsule_correction"})
            base["context_measurement_sensitivity_json"] = json.dumps(angle_sensitivity(
                paths["morphology"], delineation, fs, source=source_id, record_id=record["record_id"]),
                allow_nan=False, separators=(",", ":"))
            acquisition_upper_hz = source_acquisition_upper_bandwidth_hz(
                source_id
            )
            wideband_supported = bool(
                acquisition_upper_hz is not None
                and float(acquisition_upper_hz) + 1e-9 >= 150.0
            )
            base["acquisition_upper_bandwidth_hz"] = (
                float(acquisition_upper_hz)
                if acquisition_upper_hz is not None
                else np.nan
            )
            base["hf_qrs_effective_band_hz"] = (
                "80-150" if wideband_supported else ""
            )
            base["hf_qrs_acquisition_supported"] = wideband_supported
            base["hf_qrs_acquisition_status"] = (
                "supported_80_150_hz"
                if wideband_supported
                else "missing_or_insufficient_acquisition_bandwidth"
            )
            results = BiomarkerRegistry.compute_all(
                delineation=delineation,
                ecg_12lead=robust,
                vcg_xyz=vcg.xyz,
                fs=fs,
                names=list(options.marker_ids),
                record_ecg_12lead=paths["morphology"],
                record_raw_ecg_12lead=signal,
                record_acquisition={"original_sampling_rate_hz": original_fs,
                                    "upper_bandwidth_hz": acquisition_upper_hz},
                record_vcg_xyz=record_vcg.xyz,
                record_vcg_wideband_xyz=(
                    record_vcg.xyz if wideband_supported else None
                ),
            )
            for result in results:
                base[result.name] = result.value
            from .classical import ClassicalECGCriteria
            from .lead_measurements import LEAD_MEASUREMENT_VERSION
            lead_measurements: dict[str, float] = {}
            ClassicalECGCriteria().compute(paths["diagnostic"], delineation,
                measurement_sink=lead_measurements, measurements_only=True)
            base.update(lead_measurements)
            base["lead_measurement_version"] = LEAD_MEASUREMENT_VERSION
            base["lead_measurement_scope"] = "diagnostic_12lead_shared_fiducials"
            base["extraction_status"] = "ok"
            base["n_beats"] = int(delineation.n_beats)
            base["quality_score"] = float(
                getattr(delineation, "quality_score", float("nan"))
            )
        except Exception as exc:  # noqa: BLE001 - failure reason is a result
            base["extraction_status"] = "failed"
            base["failure_type"] = type(exc).__name__
            base["failure_message"] = str(exc)[:300]
        return base

    total = len(eligible)
    bounded_workers = max(1, min(int(options.workers), total or 1))
    if bounded_workers == 1:
        iterator = map(extract_one, records)
    else:
        executor = ThreadPoolExecutor(
            max_workers=bounded_workers,
            thread_name_prefix="vcg-external",
        )
        from .bounded_executor import adaptive_ordered_map
        iterator = adaptive_ordered_map(executor, extract_one, records, max_workers=bounded_workers)
    rows: list[dict[str, object]] = []
    archive_buffer: list[dict[str, object]] = []
    archive_paths: list[Path] = []
    archive_root = Path(measurement_archive_root) if measurement_archive_root is not None else None
    if archive_root is not None:
        archive_root.mkdir(parents=True, exist_ok=True)

    def flush_measurements():
        if archive_buffer and archive_root is not None:
            path = archive_root / f"external_measurements_part-{len(archive_paths):05d}.parquet"
            atomic_write_parquet(pd.DataFrame(archive_buffer), path)
            archive_paths.append(path)
            archive_buffer.clear()

    try:
        for position, base in enumerate(iterator, start=1):
            if archive_root is not None and "context_measurement_capsule_json" in base:
                archive_buffer.append({"source_id": base["source_id"], "record_id": base["record_id"],
                    "context_measurement_capsule_json": base.pop("context_measurement_capsule_json"),
                    "context_measurement_sensitivity_json": base.pop("context_measurement_sensitivity_json", "")})
                base["context_measurement_partition"] = f"external_measurements_part-{len(archive_paths):05d}.parquet"
                if len(archive_buffer) >= 256:
                    flush_measurements()
            rows.append(base)
            if progress is not None and position % 250 == 0:
                progress(f"Externe Merkmalsextraktion {position:,}/{total:,}")
    finally:
        if bounded_workers > 1:
            iterator.close()
            executor.shutdown(wait=True, cancel_futures=True)
    flush_measurements()
    from .external_identity_audit import duplicate_contract
    result = duplicate_contract(pd.DataFrame(rows)) if rows else pd.DataFrame(rows)
    result.attrs["measurement_archive_paths"] = [str(path) for path in archive_paths]
    return result


def extraction_coverage(features: pd.DataFrame, marker_ids: Iterable[str]) -> pd.DataFrame:
    """Per source and marker: how often the marker could be computed at all."""

    if features.empty:
        return pd.DataFrame(
            columns=["source_id", "marker_id", "n_records", "coverage"]
        )
    marker_ids = tuple(dict.fromkeys(marker_ids))
    rows: list[dict[str, object]] = []
    for source_id, group in features.groupby("source_id", sort=True):
        from .external_identity_audit import measurement_representative_mask
        n_before_copy_filter = len(group)
        group = group.loc[measurement_representative_mask(group)]
        ok = group.loc[group["extraction_status"].eq("ok")]
        status_counts = group.extraction_status.value_counts()
        for marker_id in marker_ids:
            if marker_id not in group.columns:
                continue
            values = pd.to_numeric(ok[marker_id], errors="coerce")
            n_measurable = int(np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).sum())
            coverage = float(n_measurable / len(group)) if len(group) else np.nan
            rows.append(
                {
                    "source_id": str(source_id),
                    "marker_id": str(marker_id),
                    "n_records": int(len(group)),
                    "n_selected_before_copy_filter": n_before_copy_filter,
                    "n_exact_copy_excluded": n_before_copy_filter - len(group),
                    "n_extracted": int(len(ok)),
                    "n_signal_qc_abstained": int(status_counts.get("abstained_signal_qc", 0)),
                    "n_pipeline_failed": int(status_counts.get("failed", 0)),
                    "extraction_rate": float(len(ok) / len(group)) if len(group) else np.nan,
                    "n_measurable": n_measurable,
                    "coverage": coverage,
                    "abstention_rate": 1.0 - coverage,
                    "post_qc_marker_missingness": 1.0 - n_measurable / len(ok) if len(ok) else np.nan,
                    "external_validation_version": EXTERNAL_VALIDATION_VERSION,
                }
            )
    return pd.DataFrame(rows)


def distribution_shift(
    external: pd.DataFrame,
    development_reference: Mapping[str, tuple[float, float]],
    marker_ids: Iterable[str],
) -> pd.DataFrame:
    """Standardised difference of each marker against the development cohort.

    ``development_reference`` maps marker id to (mean, sd) measured on the
    development split. The standardised difference says how far a site sits
    from the population the thresholds were learned on, which is exactly the
    a descriptive warning, not proof that a fixed operating point can travel.
    """

    marker_ids = tuple(dict.fromkeys(marker_ids))
    rows: list[dict[str, object]] = []
    for source_id, group in external.groupby("source_id", sort=True):
        from .external_identity_audit import measurement_representative_mask
        group = group.loc[measurement_representative_mask(group)]
        ok = group.loc[group["extraction_status"].eq("ok")]
        for marker_id in marker_ids:
            if marker_id not in ok.columns or marker_id not in development_reference:
                continue
            values = pd.to_numeric(ok[marker_id], errors="coerce")
            values = values.loc[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
            if len(values) < 20:
                continue
            dev_mean, dev_sd = development_reference[marker_id]
            external_mean = float(values.mean())
            external_sd = float(values.std(ddof=1))
            pooled = float(np.sqrt((dev_sd ** 2 + external_sd ** 2) / 2.0))
            shift = (
                float((external_mean - dev_mean) / pooled)
                if np.isfinite(pooled) and pooled > 1e-12
                else float("nan")
            )
            rows.append(
                {
                    "source_id": str(source_id),
                    "marker_id": str(marker_id),
                    "n": int(len(values)),
                    "development_mean": float(dev_mean),
                    "development_sd": float(dev_sd),
                    "external_mean": external_mean,
                    "external_sd": external_sd,
                    "standardised_shift": shift,
                    "distribution_shift_within_screening_tolerance": bool(
                        np.isfinite(shift)
                        and abs(shift) <= MAXIMUM_TRANSFERABLE_SHIFT
                    ),
                    "operating_point_transferable": False,
                    "operating_point_transferability_status": "not_established_by_marginal_distribution",
                    "clinical_claim_allowed": False,
                    "external_validation_version": EXTERNAL_VALIDATION_VERSION,
                }
            )
    return pd.DataFrame(rows)


def external_agreement(
    features: pd.DataFrame,
    *,
    marker_id: str,
    endpoint_id: str,
    direction: int,
    threshold: Optional[float] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    bootstrap_iterations: int = 0,
    random_seed: int = 20260812,
) -> pd.DataFrame:
    """Marker performance against external expert ECG reading, per site.

    The label is another reader's interpretation of the same recording, so the
    result is agreement with expert reading. It is reported per site and
    pooled, never as accuracy against an independent measurement of the heart.
    """

    label_names = EXTERNAL_ENDPOINT_MAP.get(endpoint_id, ())
    if not label_names:
        return pd.DataFrame(
            columns=["source_id", "endpoint_id", "marker_id", "status"]
        )
    from .result_contract import true
    analytical_source_counts = features.groupby("source_id").size().to_dict()
    if "eligible_for_diagnostic_validation" in features:
        ok = features.loc[features.eligible_for_diagnostic_validation.map(true)].copy()
    else:
        ok = features.copy()
    if marker_id not in ok.columns:
        return pd.DataFrame(
            columns=["source_id", "endpoint_id", "marker_id", "status"]
        )
    label_series = ok["labels"].astype("string").fillna("")
    positive = pd.Series(False, index=ok.index)
    for name in label_names:
        positive |= label_series.str.split("|").apply(lambda parts: name in parts)
    # Absence of a diagnosis in a multi-label header is not a negative
    # reference unless the source explicitly attests exhaustive annotation.
    from .result_contract import true
    exhaustive = ok.get(f"label_coverage_{endpoint_id}",
        ok.get("annotation_exhaustive", pd.Series(False, index=ok.index))).map(true)
    explicit = ok.get(f"label_state_{endpoint_id}", pd.Series("unknown", index=ok.index)).astype(str)
    negative = explicit.eq("negative") | (exhaustive & ~positive)
    positive |= explicit.eq("positive")
    conflict = positive & explicit.eq("negative")
    ok["reference_conflict"] = conflict
    ok["external_label"] = np.where(conflict, np.nan, np.where(positive, 1.0, np.where(negative, 0.0, np.nan)))
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(source_id), group)
        for source_id, group in ok.groupby("source_id", sort=True)
    ]
    groups.append(("__pooled__", ok))
    copy_scope = ("|".join(sorted(ok["exact_copy_evidence_scope"].dropna().astype(str).unique()))
                  if "exact_copy_evidence_scope" in ok else "selected_extraction_only_not_patient_identity")
    for source_id, group in groups:
        n_before_copy_filter = len(group)
        from .external_identity_audit import measurement_representative_mask
        group = group.loc[measurement_representative_mask(group)]
        analytical_selected = len(features) if source_id == "__pooled__" else int(analytical_source_counts.get(source_id, 0))
        values = pd.to_numeric(group[marker_id], errors="coerce").to_numpy()
        labels = group["external_label"].to_numpy(dtype=float)
        known = np.isfinite(labels)
        values = np.where(group["extraction_status"].eq("ok"), values, np.nan)
        usable = np.isfinite(values) & known
        record: dict[str, object] = {
            "source_id": source_id,
            "endpoint_id": endpoint_id,
            "marker_id": marker_id,
            "external_labels": "|".join(label_names),
            "reference_standard": COMPARISON_ECG_READING,
            "reference_tier": "C_single_administrative_or_text",
            "claim_ceiling": "agreement_with_expert_ecg_reading",
            "n": int(usable.sum()),
            "n_positive": int(labels[usable].sum()),
            "n_negative": int((1 - labels[usable]).sum()),
            "n_selected": len(group),
            "n_selected_before_copy_filter": n_before_copy_filter,
            "n_signal_audit_selected_before_annotation_filter": analytical_selected,
            "n_analytical_only_excluded_from_annotation": analytical_selected - n_before_copy_filter,
            "n_exact_copy_excluded": n_before_copy_filter - len(group),
            "exact_copy_scope": copy_scope,
            "n_reference_known": int(known.sum()),
            "n_reference_unknown": int((~known).sum()),
            "n_reference_conflicting": int(group["reference_conflict"].sum()),
            "n_extraction_ok": int(group["extraction_status"].eq("ok").sum()),
            "n_marker_observed": int(np.isfinite(values).sum()),
            "measurement_yield": float(np.isfinite(values).mean()) if len(values) else np.nan,
            "reference_kind": "same_ecg_expert",
            "clinical_claim_allowed": False,
            "pooled_descriptive_only": source_id == "__pooled__",
            "external_validation_version": EXTERNAL_VALIDATION_VERSION,
        }
        if usable.sum() < 30 or np.unique(labels[usable]).size < 2:
            record["status"] = "not_evaluable"
            rows.append(record)
            continue
        frozen_thresholds = {
            str(name): float(value)
            for name, value in dict(thresholds or {}).items()
        }
        if threshold is not None:
            frozen_thresholds.setdefault("external", float(threshold))
        metrics = evaluate_marker(
            values[known],
            labels[known],
            direction=int(direction),
            thresholds=frozen_thresholds,
        )
        roc_auc = metrics.get("roc_auc")
        record.update(
            {
                "status": "evaluable",
                "roc_auc": roc_auc,
                # evaluate_marker already applies the frozen direction, so
                # roc_auc is the direction-aligned value. The mirrored form is
                # kept for readers comparing against the development tables.
                "aligned_auc": roc_auc,
                "raw_auc": (
                    float(1.0 - roc_auc)
                    if roc_auc is not None
                    and np.isfinite(roc_auc)
                    and int(direction) < 0
                    else roc_auc
                ),
                "rank_biserial": metrics.get("rank_biserial"),
                "average_precision": metrics.get("average_precision"),
                "p_value": metrics.get("p_value"),
                "assessability_gate_passed": metrics.get(
                    "assessability_gate_passed"
                ),
                "test_yield": metrics.get("test_yield"),
                "inference_status": inference_status(
                    int(record["n_positive"]),
                    int(record["n_negative"]),
                ),
            }
        )
        for threshold_name, frozen_threshold in frozen_thresholds.items():
            predicted = (
                int(direction) * values[usable] >= float(frozen_threshold)
            )
            fixed = classification_metrics(labels[usable], predicted)
            prefix = f"{threshold_name}_frozen"
            record.update(
                {
                    f"{prefix}_threshold_aligned": float(frozen_threshold),
                    f"{prefix}_threshold_original": float(
                        int(direction) * frozen_threshold
                    ),
                    f"{prefix}_sensitivity": fixed.get("sensitivity"),
                    f"{prefix}_specificity": fixed.get("specificity"),
                    f"{prefix}_balanced_accuracy": fixed.get(
                        "balanced_accuracy"
                    ),
                    "threshold_source": "development_frozen_not_refitted",
                }
            )
        patient_ids = group.get("patient_id", pd.Series(index=group.index, dtype=object))
        used_patients = patient_ids.loc[usable]
        from .patient_identity import known_patient_mask
        patient_identity_known = bool(known_patient_mask(used_patients).all())
        record["n_unique_patients"] = int(used_patients.nunique()) if patient_identity_known and source_id != "__pooled__" else None
        if source_id == "__pooled__" and patient_identity_known:
            record["n_unique_source_patient_keys_not_proven_distinct_people"] = int(group.loc[usable, ["source_id", "patient_id"]].drop_duplicates().shape[0])
        record["patient_identity_status"] = "known" if patient_identity_known else "unknown_no_patient_level_inference"
        if not patient_identity_known or source_id == "__pooled__":
            record["inference_status"] = "descriptive_only"
            record["p_value"] = np.nan
        elif used_patients.duplicated().any():
            # The record-level rank-test p-value assumes independent records.
            # Patient-cluster bootstrap CIs below remain valid for repeats.
            record["p_value"] = np.nan
            record["inference_status"] = "patient_cluster_ci_only" if bootstrap_iterations > 0 else "descriptive_only"
        if bootstrap_iterations > 0 and source_id != "__pooled__" and patient_identity_known:
            intervals = marker_bootstrap_intervals(
                patient_ids.loc[usable].astype(str).to_numpy(),
                values[usable],
                labels[usable],
                direction=int(direction),
                thresholds={},
                iterations=int(bootstrap_iterations),
                seed=int(random_seed),
            )
            record.update(
                {
                    key: value
                    for key, value in intervals.items()
                    if key.endswith(("_ci_low", "_ci_high"))
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def site_consistency(agreement: pd.DataFrame) -> Mapping[str, object]:
    """Whether a result holds across sites or rests on one of them."""

    if agreement.empty:
        return {"status": "not_evaluated", "sites_evaluable": 0}
    per_site = agreement.loc[
        agreement["source_id"].ne("__pooled__")
        & agreement["status"].eq("evaluable")
    ]
    if per_site.empty:
        return {"status": "not_evaluated", "sites_evaluable": 0}
    aucs = pd.to_numeric(per_site["aligned_auc"], errors="coerce").dropna()
    return {
        "status": "evaluated",
        "sites_evaluable": int(len(per_site)),
        "min_aligned_auc": float(aucs.min()) if len(aucs) else float("nan"),
        "max_aligned_auc": float(aucs.max()) if len(aucs) else float("nan"),
        "auc_range": float(aucs.max() - aucs.min()) if len(aucs) else float("nan"),
        "median_aligned_auc": float(aucs.median()) if len(aucs) else float("nan"),
        "consistent_across_sites": bool(
            len(aucs) >= 2 and float(aucs.max() - aucs.min()) <= 0.10
        ),
        "sites_above_chance": int((aucs > 0.5).sum()),
        "interpretation": (
            "a wide spread across sites means the pooled figure describes no "
            "single site; it is a mixture, not a transferable estimate"
        ),
        "external_validation_version": EXTERNAL_VALIDATION_VERSION,
    }


def run_external_validation(
    collection_root: Path | str,
    *,
    frozen_contract: FrozenExternalContract,
    output_root: Path,
    identity_manifest: Optional[Path] = None,
    identity_manifest_sha256: Optional[str] = None,
    marker_ids: Optional[tuple[str, ...]] = None,
    max_records_per_source: Optional[int] = None,
    powerline_hz: Optional[float] = None,
    minimum_signal_quality: float = 0.0,
    workers: int = 1,
    bootstrap_iterations: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Full external run: inventory, extraction, coverage, shift, agreement."""

    selected_ids = tuple(marker_ids or frozen_contract.marker_ids)
    unsealed = sorted(set(selected_ids) - set(frozen_contract.marker_ids))
    if unsealed:
        raise PermissionError(
            "External validation may only evaluate hypotheses in the immutable "
            f"development shortlist; not frozen: {unsealed}"
        )
    selected_markers = tuple(
        item
        for item in frozen_contract.markers
        if item.model.feature_id in selected_ids
    )
    if not selected_markers:
        raise ValueError("No frozen external marker remains after filtering")
    # Keep the caller's order while removing duplicates for extraction.
    selected_ids = tuple(dict.fromkeys(selected_ids))
    development_reference = frozen_contract.development_reference

    if (identity_manifest is None) != (identity_manifest_sha256 is None):
        raise ValueError("Copy evidence requires both manifest and frozen SHA-256")
    copy_evidence = None
    copy_lookup = None
    if identity_manifest is not None:
        from .external_copy_evidence import load_known_copy_evidence
        copy_evidence = load_known_copy_evidence(identity_manifest, identity_manifest_sha256)
        representatives = copy_evidence.representatives
        copy_lookup = dict(zip(representatives.index.astype(str), zip(
            representatives.source_id.astype(str), representatives.record_id.astype(str),
            representatives.copy_overlaps_development_source.astype(bool))))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Externe Kohorten: Sammlung inventarisieren")
    inventory = eligible_for_diagnostic_validation(
        inventory_external_collection(collection_root)
    )
    inventory_path = output_root / "external_inventory.parquet"
    atomic_write_parquet(inventory, inventory_path)
    scope = external_claim_scope(inventory)

    if progress:
        progress("Externe Kohorten: Merkmale mit identischem Code berechnen")
    features = extract_external_features(
        inventory,
        ExternalExtractionOptions(
            marker_ids=selected_ids,
            max_records_per_source=max_records_per_source,
            powerline_hz=powerline_hz,
            minimum_signal_quality=minimum_signal_quality,
            workers=workers,
        ),
        progress=progress,
        known_copy_representatives=copy_lookup,
        measurement_archive_root=output_root,
    )
    measurement_archive_paths = [Path(path) for path in features.attrs.get("measurement_archive_paths", [])]
    features_path = output_root / "external_features.parquet"
    if copy_evidence is not None:
        from .external_copy_evidence import apply_known_copy_evidence
        features = apply_known_copy_evidence(features, copy_evidence)
    atomic_write_parquet(features, features_path)

    coverage = extraction_coverage(features, selected_ids)
    coverage_path = output_root / "external_coverage.parquet"
    atomic_write_parquet(coverage, coverage_path)

    shift = (
        distribution_shift(features, development_reference, selected_ids)
        if development_reference
        else pd.DataFrame()
    )
    if not shift.empty:
        shift_path = output_root / "external_distribution_shift.parquet"
        atomic_write_parquet(shift, shift_path)

    agreement_frames: list[pd.DataFrame] = []
    consistency: dict[str, object] = {}
    for frozen_marker in selected_markers:
        endpoint_id = frozen_marker.endpoint_id
        marker_id = frozen_marker.model.feature_id
        frame = external_agreement(
            features,
            marker_id=marker_id,
            endpoint_id=endpoint_id,
            direction=frozen_marker.model.direction,
            thresholds=frozen_marker.model.thresholds,
            bootstrap_iterations=bootstrap_iterations,
        )
        if not frame.empty:
            frame["panel_id"] = frozen_marker.panel_id
            frame["criterion_id"] = frozen_marker.criterion_id
            frame["marker_model_hash"] = frozen_marker.marker_model_hash
            frame["development_run_id"] = (
                frozen_contract.development_run_id
            )
            frame["freeze_hash"] = frozen_contract.freeze_hash
            agreement_frames.append(frame)
            consistency[f"{endpoint_id}:{marker_id}"] = site_consistency(frame)
    agreement = (
        pd.concat(agreement_frames, ignore_index=True)
        if agreement_frames
        else pd.DataFrame()
    )
    if not agreement.empty:
        agreement_path = output_root / "external_agreement.parquet"
        atomic_write_parquet(agreement, agreement_path)
    output_files = [inventory_path, features_path, coverage_path, *measurement_archive_paths]
    from .cohort_value import cohort_value_table
    value_table = cohort_value_table(features, cohort_column="source_id", marker_ids=tuple(selected_ids), agreement=agreement)
    value_path = output_root / "cohort_added_value.parquet"
    atomic_write_parquet(value_table, value_path)
    output_files.append(value_path)
    if not shift.empty:
        output_files.append(shift_path)
    if not agreement.empty:
        output_files.append(agreement_path)
    extraction_counts = extraction_summary(features)
    result = {
        "claim_scope": scope,
        **extraction_counts,
        "coverage": coverage,
        "cohort_added_value": value_table,
        "distribution_shift": shift,
        "agreement": agreement,
        "site_consistency": consistency,
        "locked_test_read": False,
        "refitted_on_external_data": False,
        "directions_thresholds_scaling_source": "immutable_development_freeze",
        "development_run_id": frozen_contract.development_run_id,
        "freeze_hash": frozen_contract.freeze_hash,
        "shortlist_hash": frozen_contract.shortlist_hash,
        "marker_count": len(selected_ids),
        "hypothesis_count": len(selected_markers),
        "external_cohort_version": EXTERNAL_COHORT_VERSION,
        "external_validation_version": EXTERNAL_VALIDATION_VERSION,
    }
    manifest_payload = {
        "measurement_archive": {
            "files": {path.name: file_sha256(path) for path in measurement_archive_paths},
            "identity_keys": ["source_id", "record_id"], "buffer_records": 256,
            "clinical_claim_allowed": False,
        },
        "extraction_summary": extraction_counts,
        "identity_manifest_sha256": copy_evidence.manifest_sha256 if copy_evidence is not None else None,
        "external_validation_version": EXTERNAL_VALIDATION_VERSION,
        "development_run_id": frozen_contract.development_run_id,
        "freeze_hash": frozen_contract.freeze_hash,
        "shortlist_hash": frozen_contract.shortlist_hash,
        "selected_marker_ids": list(selected_ids),
        "selected_hypotheses": [
            {
                "endpoint_id": item.endpoint_id,
                "panel_id": item.panel_id,
                "feature_id": item.model.feature_id,
                "criterion_id": item.criterion_id,
                "marker_model_hash": item.marker_model_hash,
            }
            for item in selected_markers
        ],
        "source_powerline_policy": (
            "explicit_global_override"
            if powerline_hz is not None
            else "prespecified_per_source"
        ),
        "minimum_signal_quality": float(minimum_signal_quality),
        "workers": int(workers),
        "locked_test_read": False,
        "refitted_on_external_data": False,
        "output_sha256": {
            path.name: file_sha256(path) for path in output_files
        },
    }
    manifest_payload["manifest_hash"] = content_hash(manifest_payload)
    manifest_path = output_root / "external_validation.manifest.json"
    atomic_write_json(manifest_path, manifest_payload)
    result["manifest_path"] = str(manifest_path.resolve())
    result["manifest_hash"] = manifest_payload["manifest_hash"]
    return result


__all__ = [
    "EXTERNAL_VALIDATION_VERSION",
    "ExternalExtractionOptions",
    "FrozenExternalContract",
    "FrozenExternalMarker",
    "MAXIMUM_TRANSFERABLE_SHIFT",
    "distribution_shift",
    "external_agreement",
    "extract_external_features",
    "extraction_coverage",
    "load_frozen_external_contract",
    "run_external_validation",
    "site_consistency",
]
