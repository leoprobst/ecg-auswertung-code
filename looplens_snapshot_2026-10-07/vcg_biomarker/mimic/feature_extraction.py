from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

from ..core.biomarkers import BiomarkerRegistry
from ..core.delineation.fusion_delineator import FusionDelineator
from ..core.preprocessing import (
    construct_robust_morphology_record,
    detect_r_peaks_multilead,
)
from ..core.vcg_transform import VCGTransformRegistry
from .classical import ClassicalECGCriteria
from .config import MimicProjectConfig
from .panels import (
    TRANSFORM_INDEPENDENT_FEATURE_IDS,
    default_feature_panels,
    freeze_panel_registry,
)
from .report_parser import MachineReportParser, ReportMentionState
from .phenotypes import prepare_phenotype_resources
from .feature_record import (
    _extract_record_batch_worker,
    _extract_record_worker,
    _initialize_feature_worker,
)
from .resources import recommend_dynamic_workers
from .signals import MimicWaveformLoader, compute_vcg_sensitivity_transforms
from .storage import (
    DuckDBCatalog,
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    canonical_json,
    capture_output_integrity,
    content_hash,
    feature_parquet_clause,
    file_fingerprint,
    file_sha256,
    operational_config_hash,
    optional_column_select,
    relation_columns,
    scientific_config_hash,
    source_tree_hash,
)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _WindowsFileBasicInfo(ctypes.Structure):
        _fields_ = (
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
        )

    _WINDOWS_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WINDOWS_CREATE_FILE = _WINDOWS_KERNEL32.CreateFileW
    _WINDOWS_CREATE_FILE.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _WINDOWS_CREATE_FILE.restype = wintypes.HANDLE
    _WINDOWS_GET_FILE_INFORMATION = (
        _WINDOWS_KERNEL32.GetFileInformationByHandleEx
    )
    _WINDOWS_GET_FILE_INFORMATION.argtypes = (
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _WINDOWS_GET_FILE_INFORMATION.restype = wintypes.BOOL
    _WINDOWS_CLOSE_HANDLE = _WINDOWS_KERNEL32.CloseHandle
    _WINDOWS_CLOSE_HANDLE.argtypes = (wintypes.HANDLE,)
    _WINDOWS_CLOSE_HANDLE.restype = wintypes.BOOL


# 3.0.0 is a new analysis generation, not an increment: the marker code
# changed, so every recording has to be extracted again.
#
#   * the two beat-to-beat dispersions were repaired -- one of them had raised
#     KeyError on 328.358 of 630.044 records and produced nothing at all
#   * amplitude ratios are capped where they are formed (1,26e9 -> 50)
#   * durations abstain instead of coming back negative
#   * the classical criteria are extracted as criteria, in their own panel
#   * heart rate travels as a context column again
#   * short signal dropouts are repaired before the quality gate rejects them
#   * 36 new markers: rotation invariants, Frenet-Serret, linking and writhe,
#     level-3 path signatures, respiration, tremor, and the atrial loop
# 3.2.0 implements the published Direct-8-Lead Integral angle literally as
# the magnitude-weighted dual integral of pairwise angles (and includes the
# J-to-Tend interval).  The former nonlinear shortcut was not equivalent, so
# no 3.1 feature partition may be reused.
# 3.3.0 repairs two bounded marker definitions found by the A017 pilot:
# T-loop late area is now an absolute additive areal-sweep fraction and the TQ
# exposure is divided by the RR interval that actually brackets that window.
# Existing 3.2 partitions therefore cannot be adopted.
# 3.4.0 uses the closed polygon perimeter for fixed-plane and best-fit-plane
# circularity, consistent with Shoelace area. Open path-length markers remain
# unchanged. This also changes dependent composites; require a new freeze and
# extraction, never adopt 3.3 pilot partitions or thresholds.
# 3.5.0 persists the prespecified per-lead classical primitives for a dedicated
# lead-utility analysis. They are not silently added to the broad marker panel.
# 3.6.0 adds four beat-to-beat morphology-variability markers to the discovery
# reserve and admits three markers of the catalog's Beat-Variabilitaet family
# that a category rule had routed to rhythm context, where they belonged to no
# panel and were therefore never computed. The curated panels are byte-identical
# -- only the reserve grew -- but the extracted column set changed, so a 3.5
# partition cannot be adopted.
# 3.7.0 adds three markers taken from named 2024-2026 publications, all into
# the discovery reserve: the terminal P-wave progression across V4-V6 that
# carried a visual four-criterion model to AUC 0,905 against invasive atrial
# voltage mapping, the signed apicobasal component of the ventricular gradient
# (computed on every recording since the GEH work, but only ever written into
# provenance metadata where no analysis could screen it), and a late-QRS speed
# uniformity as an openly speculative guess. The curated panels are
# byte-identical; only the reserve grew, and the extracted column set changed,
# so a 3.6 partition cannot be adopted.
# 3.49.0 implements audit 2026-09-23 S3-S6: missing leads no longer read as
# negative voltage criteria, Q waves from the lead's own onset with a visible
# depth in V2/V3, RBBB lateral S duration and LBBB R-peak time, and recordings
# with suspected limb-lead reversal excluded (``lead_placement_excluded``).
# 3.50.0 adds the audit 5.1 reference scores (TaVR, iCEB, Tp-e/QT, device
# frontal QRS-T angle with the recorder's T axis, 12-lead voltage, Romhilt-
# Estes, Daniel, Macruz, diffuse ST pattern out of the lab) and the eight
# markers of the 23.09. theses (T6 J-Tpeak, T10 secondary-cluster gradient,
# T15 PR-segment vector) in the discovery reserve.
# 3.51.0 fixes three delineation defects found on rhythms without clear wave
# separation: the consensus beat took its P/T status from the medoid alone
# (P lost in 23 of 177 LUDB records), the QRS detector accepted P and T
# waves in slow ventricular rhythms (AV block), and the rate was the median
# RR, i.e. the coupling interval in a bigeminy; it is now the mean RR.
# Also in 3.51.0: a definite recorder AF/flutter statement replaces the P
# wave (conducted flutter measured as P), the T window stops before the next
# P wave at fast rates, and the recorder's LBBB statement no longer blanks
# the bundle-branch patterns themselves (only pacing does).
# 3.51.1 writes every non-identifier integer column as double, so all
# partitions share one schema and full reads avoid union_by_name.
FEATURE_STAGE_VERSION = "3.51.1"
FEATURE_ARTIFACT_INDEX_VERSION = "1.4.0"
FEATURE_VERIFICATION_ATTESTATION_VERSION = "1.0.0"
FEATURE_VERIFICATION_ATTESTATION_NAME = (
    ".feature_verification_attestation.json"
)
# Partitions are assigned once per study and then kept.  See
# ``_assign_cohort_partitions`` for why positional chunking had to go.
PARTITION_ASSIGNMENT_VERSION = "sticky-1.0.0"
FEATURE_RESUME_CHECKPOINT_VERSION = "2.0.0"
REPORT_PARSER_VERSION = "1.3.0"
CAPPED_SAMPLING_VERSION = "endpoint_index_outcome_blind_v4"


@dataclass(frozen=True)
class FeatureExtractionOptions:
    panel_ids: tuple[str, ...] = ("primary_core", "sentinel")
    include_locked_test: bool = False
    locked_test_only: bool = False
    freeze_run_id: Optional[str] = None
    max_records: Optional[int] = None
    partition_size: int = 250
    micro_batch_size: int = 25
    force: bool = False
    execution_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.partition_size < 1:
            raise ValueError("partition_size must be at least one")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be at least one")
        if self.include_locked_test and not self.freeze_run_id:
            raise ValueError(
                "Union extraction with locked_test is retired. Locked-test "
                "waveforms may only be extracted separately for an explicitly "
                "frozen run."
            )
        if self.locked_test_only and not self.freeze_run_id:
            raise ValueError(
                "locked_test_only requires freeze_run_id from an immutable "
                "development freeze."
            )
        if (self.locked_test_only or self.include_locked_test) and (
            self.max_records is not None
        ):
            raise ValueError(
                "A locked-test extraction must cover its complete frozen cohort; "
                "max_records is forbidden."
            )


def extract_mimic_features(
    config: MimicProjectConfig,
    options: FeatureExtractionOptions = FeatureExtractionOptions(),
    *,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[..., None]] = None,
) -> StageManifest:
    locked_test_only = bool(
        options.locked_test_only or options.include_locked_test
    )
    cohort_path = config.labels_root / "clinical_label_evidence.parquet"
    endpoint_index_path = config.labels_root / "endpoint_index_cohorts.parquet"
    phenotype_diagnosis_path = (
        config.standardized_root / "diagnoses_icd.parquet"
    )
    if not locked_test_only and not cohort_path.exists():
        raise FileNotFoundError(
            f"Clinical label evidence is missing: {cohort_path}. "
            "Run build-labels first."
        )
    if (
        not locked_test_only
        and options.max_records is not None
        and not endpoint_index_path.exists()
    ):
        raise FileNotFoundError(
            "A capped development pilot requires the formal endpoint index: "
            f"{endpoint_index_path}. Run build-labels first."
        )
    selection = resolve_feature_panel_selection(config, options)
    panel_payload = selection["panel_payload"]
    feature_ids = selection["feature_ids"]
    vcg_feature_ids = selection["vcg_feature_ids"]
    classical_feature_ids = selection["classical_feature_ids"]
    sensitivity_feature_ids = selection["sensitivity_feature_ids"]

    feature_code_hash = feature_implementation_hash()
    phenotype_resources = (
        prepare_phenotype_resources(config)
        if not locked_test_only and phenotype_diagnosis_path.exists()
        else None
    )
    freeze_context = (
        _locked_test_freeze_context(
            config,
            run_id=str(options.freeze_run_id),
            feature_code_hash=feature_code_hash,
            panel_hash=str(panel_payload["panel_hash"]),
        )
        if locked_test_only
        else None
    )
    config_hash = scientific_config_hash(config)
    panel_key = "_".join(options.panel_ids)
    if locked_test_only:
        if freeze_context is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("Locked-test extraction requires freeze context")
        scope = f"locked-test-{freeze_context['freeze_hash'][:12]}"
    elif options.max_records is not None:
        scope = (
            f"smoke-development-{int(options.max_records)}"
        )
    else:
        scope = "development"
    output_root = config.features_root / panel_key / scope
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "features.manifest.json"

    # The cohort is loaded before the manifest check on purpose.  The stage
    # key used to carry ``file_fingerprint`` of clinical_label_evidence.parquet,
    # ecg_master_index.parquet and diagnoses_icd.parquet.  Those fingerprints
    # default to size+mtime, so *rebuilding* the labels invalidated the whole
    # extraction even when the produced bytes were identical -- and a label
    # rule change did so with certainty, although no marker depends on a label.
    # Markers depend on the cohort only through the records it selects, so that
    # is what the key is built from now.  The extra cost is one DuckDB query.
    if progress:
        progress("Bestimme die Vereinigungsmenge klinisch auswertbarer ECGs")
    cohort = _load_feature_cohort(
        config,
        options,
        phenotype_mapping_path=(
            phenotype_resources.phecodex_mapping_path
            if phenotype_resources is not None
            else None
        ),
    )
    total = len(cohort)
    if total == 0:
        raise RuntimeError("No eligible ECGs were selected for feature extraction")
    record_payloads = _cohort_record_payloads(cohort)

    stage_input_payload = feature_stage_input_payload(
        record_payloads,
        panel_hash=str(panel_payload["panel_hash"]),
        feature_code_hash=feature_code_hash,
        options=options,
        sensitivity_vcg_methods=tuple(config.sensitivity_vcg_methods),
        freeze_hash=(freeze_context["freeze_hash"] if freeze_context else None),
    )
    input_hash = content_hash(stage_input_payload)
    timing_contract = None
    if not locked_test_only:
        import platform
        import psutil
        from .runtime_calibration import build_feature_timing_contract
        from .run_log import active_log
        timing_contract = build_feature_timing_contract(
            stage_input_payload, scientific_config_sha256=config_hash,
            scheduler_sha256=file_sha256(Path(__file__)),
            resource_payload={"operational_config_sha256": operational_config_hash(config),
                "machine": platform.node(), "platform": platform.platform(),
                "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "total_ram_bytes": psutil.virtual_memory().total},
        )
        timing_log = active_log()
        if timing_log is not None:
            timing_log.feature_runtime_contract = dict(timing_contract)
            timing_log.write("runtime_contract", contract=timing_contract)
        if progress:
            progress("Feature-Laufzeitvertrag fuer die ausgewaehlte Kohorte erfasst")
    if not options.force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, FEATURE_STAGE_VERSION):
            return previous
    else:
        # Without a manifest there is no matches() call and so no reuse record;
        # the pilot gate then filed a genuine fresh extraction as "cache
        # unknown" and refused its throughput (A150 pilots, 01.10.2026).
        from .run_log import note_reuse

        note_reuse("extract-features", reused=False,
                   reason="force_requested" if options.force else "no_previous_manifest")

    partition_identity = {
        "feature_code_hash": feature_code_hash,
        "panel_hash": panel_payload["panel_hash"],
        "config_hash": config_hash,
        "feature_ids": tuple(feature_ids),
        "sensitivity_vcg_methods": tuple(config.sensitivity_vcg_methods),
        "locked_test_only": locked_test_only,
        "freeze_hash": (
            freeze_context["freeze_hash"] if freeze_context else None
        ),
    }
    chunks = _assign_cohort_partitions(
        cohort,
        output_root=output_root,
        partition_size=options.partition_size,
        force=bool(options.force),
        progress=progress,
    )
    legacy_partitions_are_current = _legacy_partitions_are_current(
        output_root,
        feature_code_hash=feature_code_hash,
        panel_hash=str(panel_payload["panel_hash"]),
        config_hash=config_hash,
    )
    output_paths: list[str] = []
    timing_contract_path = output_root / "feature_timing_contract.json"
    if timing_contract is not None:
        atomic_write_json(timing_contract_path, timing_contract)
        output_paths.append(str(timing_contract_path.resolve()))
    adopted_partitions = 0
    success_count = failure_count = skipped_partitions = 0
    calculator_failure_count = calculator_failure_ecg_count = 0
    started = time.monotonic()
    task_config = build_feature_task_config(
        config,
        vcg_feature_ids=vcg_feature_ids,
        sensitivity_feature_ids=sensitivity_feature_ids,
        classical_feature_ids=classical_feature_ids,
    )

    from .resource_grants import worker_ceiling, granted_workers, GRANT_ENV
    # One pool for the whole stage (audit 2026-09-23, E1/E2/R1/R5). The pool is
    # never rebuilt: the supervisor's grant decides how many records may be in
    # flight, and a lower grant simply submits fewer records. Processes are
    # spawned lazily up to the highest admission ever reached, so an idle
    # ceiling costs neither CPU nor the 7 CPU-seconds of a cold import.
    managed = bool(os.environ.get(GRANT_ENV))
    admitted_worker_budget = worker_ceiling(max(1, int(config.workers)))
    if managed:
        # The pool bound is the machine's capacity, not the grant seen at
        # start: an expired grant at that moment (supervisor restarting) must
        # not pin the whole stage to one process. Admission still follows the
        # live grant, and processes start only when admitted.
        from .resources import cpu_token_capacity, detect_hardware
        admitted_worker_budget = max(
            admitted_worker_budget,
            cpu_token_capacity(detect_hardware()),
        )
    pool = _AdmissionPool(admitted_worker_budget, task_config)
    grant_invalid_since: Optional[float] = None
    last_grant_warning = 0.0
    allowed_workers = (
        # Acknowledge at once: the supervisor's reservation bookkeeping reads
        # the job's resident count from this acknowledgement.
        granted_workers(admitted_worker_budget, resident_workers=0)
        if managed
        else admitted_worker_budget
    )
    reported_workers = allowed_workers
    last_grant_read = time.monotonic()
    dynamic_resize_state = _DynamicWorkerHysteresis()
    completed_records = 0
    reused_records = 0
    micro_size = min(
        max(1, int(options.micro_batch_size)),
        max(1, int(options.partition_size)),
    )
    window: deque[_PartitionWork] = deque()
    in_flight: dict = {}
    chunk_iterator = iter(enumerate(chunks))
    chunks_exhausted = False

    # Concept item 7: below the grant, keep only the workers that add
    # throughput. Supervised runs only; never above the grant.
    from .throughput_governor import ThroughputGovernor

    governor = ThroughputGovernor() if managed else None

    def current_allowed() -> int:
        nonlocal allowed_workers, last_grant_read
        nonlocal grant_invalid_since, last_grant_warning
        now = time.monotonic()
        if managed and now - last_grant_read >= 1.0:
            allowed_workers = granted_workers(
                admitted_worker_budget,
                resident_workers=len(in_flight),
            )
            last_grant_read = now
            from .resource_grants import read_grant
            if read_grant(Path(os.environ[GRANT_ENV])):
                grant_invalid_since = None
            else:
                grant_invalid_since = grant_invalid_since or now
                # O3: without a live supervisor the grant expires after 8 s
                # and the stage silently runs on one worker. Say so.
                if (
                    progress
                    and now - grant_invalid_since >= 60.0
                    and now - last_grant_warning >= 600.0
                ):
                    progress(
                        "Warnung: seit "
                        f"{int(now - grant_invalid_since)} s keine gültige "
                        "Ressourcenzuteilung (Supervisor prüfen); "
                        "Extraktion läuft mit 1 Worker"
                    )
                    last_grant_warning = now
        allowed = max(1, min(int(allowed_workers), admitted_worker_budget))
        if governor is not None:
            previous_cap = governor.cap
            allowed = governor.admit(allowed)
            if progress and governor.cap != previous_cap and previous_cap is not None:
                progress(
                    f"Durchsatzregelung: {governor.cap} Worker bringen den "
                    "hoechsten Durchsatz unter der Zuteilung"
                )
        return allowed

    def prepare_partition(chunk_index, split, part_index, chunk) -> _PartitionWork:
        nonlocal completed_records, reused_records, adopted_partitions
        nonlocal success_count, failure_count, skipped_partitions
        nonlocal calculator_failure_count, calculator_failure_ecg_count
        split_root = output_root / f"split={split}"
        split_root.mkdir(parents=True, exist_ok=True)
        part_path = split_root / f"part-{part_index:06d}.parquet"
        part_manifest = split_root / f"part-{part_index:06d}.json"
        resume_root = split_root / f".part-{part_index:06d}.resume"
        resume_manifest = split_root / f".part-{part_index:06d}.resume.json"
        study_ids = [int(value) for value in chunk["study_id"]]
        expected_study_ids = set(study_ids)
        # Deliberately not keyed on the stage input hash.  A partition
        # depends on its own records and on the code that turns them into
        # markers -- not on how many other ECGs the cohort happens to hold,
        # and not on the modification time of a label file.
        part_hash = content_hash(
            {
                "assignment_version": PARTITION_ASSIGNMENT_VERSION,
                "records": [
                    json.loads(record_payloads[study_id])
                    for study_id in study_ids
                ],
                **partition_identity,
            }
        )
        work = _PartitionWork(
            chunk_index=chunk_index,
            study_ids=study_ids,
            expected_study_ids=expected_study_ids,
            part_hash=part_hash,
            part_path=part_path,
            part_manifest=part_manifest,
            resume_root=resume_root,
            resume_manifest=resume_manifest,
        )
        existing_partition = (
            _load_partition_manifest(part_manifest)
            if part_manifest.exists()
            else None
        )
        if (
            _partition_reusable_for_execution(existing_partition, options)
            and part_path.exists()
            and existing_partition is not None
            and existing_partition.get("part_hash") != part_hash
            and legacy_partitions_are_current
            and _adopt_existing_partition(
                part_path,
                part_manifest,
                part_hash=part_hash,
                expected_study_ids=study_ids,
                execution_id=options.execution_id or "",
            )
        ):
            adopted_partitions += 1
            existing_partition = _load_partition_manifest(part_manifest)
        if (
            _partition_reusable_for_execution(existing_partition, options)
            and part_path.exists()
            and existing_partition is not None
            and existing_partition.get("part_hash") == part_hash
            and _partition_file_matches(part_path, existing_partition)
        ):
            skipped_partitions += 1
            work.output_paths.append(str(part_path.resolve()))
            from .marker_robustness import write_partition_robustness
            work.output_paths.append(str(write_partition_robustness(
                part_path, source_sha256=existing_partition.get("parquet_sha256"),
            ).resolve()))
            existing = _read_single_parquet(
                part_path,
                columns=[
                    "extraction_status",
                    "biomarker_calculation_failure_count",
                ],
            )
            success_count += int((existing["extraction_status"] == "ok").sum())
            failure_count += int((existing["extraction_status"] != "ok").sum())
            calculator_failures = pd.to_numeric(
                existing["biomarker_calculation_failure_count"],
                errors="coerce",
            ).fillna(0)
            calculator_failure_count += int(calculator_failures.sum())
            calculator_failure_ecg_count += int(
                (calculator_failures > 0).sum()
            )
            completed_records += len(existing)
            reused_records += len(existing)
            _remove_resume_fragments(resume_root, resume_manifest)
            work.done = True
            return work

        resume_state, resumed = _open_resume_fragments(
            resume_root,
            resume_manifest,
            part_hash=part_hash,
            expected_study_ids=expected_study_ids,
            force=options.force,
            execution_id=options.execution_id,
        )
        completed_studies = (
            set(int(value) for value in resumed["study_id"])
            if not resumed.empty and "study_id" in resumed
            else set()
        )
        work.resume_state = resume_state
        work.pending = deque(
            row
            for row in chunk.to_dict(orient="records")
            if int(row["study_id"]) not in completed_studies
        )
        work.had_pending = bool(work.pending)
        work.completed = len(resumed)
        reused_records += work.completed
        work.success = int(
            (
                resumed.get(
                    "extraction_status",
                    pd.Series(dtype=str),
                )
                == "ok"
            ).sum()
        )
        work.failure = work.completed - work.success
        work.calculation_failures = int(
            pd.to_numeric(
                resumed.get(
                    "biomarker_calculation_failure_count",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
        del resumed
        return work

    def report_progress() -> None:
        active = [work for work in window if not work.done]
        done = min(
            completed_records + sum(work.completed for work in active),
            total,
        )
        success_now = success_count + sum(work.success for work in active)
        failure_now = failure_count + sum(work.failure for work in active)
        calculation_failures_now = calculator_failure_count + sum(
            work.calculation_failures for work in active
        )
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = max(0, done - reused_records) / elapsed
        eta = (total - done) / rate if rate > 0 else math.inf
        message = (
            f"Features {done:,}/{total:,}; OK={success_now:,}, "
            f"Fehler={failure_now:,}, Reused={reused_records:,}, "
            f"Rechnerfehler={calculation_failures_now:,}, "
            f"ETA={_format_duration(eta)}"
        )
        if progress:
            progress(message)
        if checkpoint:
            checkpoint(message, current=done, total=total)

    def commit_buffer(work: _PartitionWork) -> None:
        micro_frame = pd.DataFrame(work.buffer)
        work.buffer = []
        work.resume_state = _append_resume_fragment(
            micro_frame,
            work.resume_root,
            work.resume_manifest,
            part_hash=work.part_hash,
            execution_id=options.execution_id,
            expected_study_ids=work.expected_study_ids,
            state=work.resume_state,
        )
        work.completed += len(micro_frame)
        micro_success = int((micro_frame["extraction_status"] == "ok").sum())
        work.success += micro_success
        work.failure += len(micro_frame) - micro_success
        work.calculation_failures += int(
            pd.to_numeric(
                micro_frame.get(
                    "biomarker_calculation_failure_count",
                    pd.Series(0, index=micro_frame.index),
                ),
                errors="coerce",
            ).fillna(0).sum()
        )
        report_progress()

    def finalize_partition(work: _PartitionWork) -> None:
        nonlocal completed_records, success_count, failure_count
        nonlocal calculator_failure_count, calculator_failure_ecg_count
        nonlocal allowed_workers, reported_workers
        frame = _materialize_resume_fragments(work.resume_state)
        frame = (
            frame.drop_duplicates("study_id", keep="last")
            .set_index("study_id")
            .reindex(work.study_ids)
            .reset_index()
        )
        _assert_exact_partition_identity(
            frame,
            expected_study_ids=work.study_ids,
            context=f"partition {work.chunk_index}",
        )
        for feature_id in feature_ids:
            if feature_id not in frame:
                frame[feature_id] = np.nan
        _cast_all_null_features_to_float(frame, feature_ids)
        _cast_integer_counts_to_float(frame)
        atomic_write_parquet(frame, work.part_path)
        parquet_fingerprint = file_fingerprint(
            work.part_path,
            include_sha256=True,
        )
        from .marker_robustness import write_partition_robustness
        work.output_paths.append(str(write_partition_robustness(
            work.part_path, frame=frame, source_sha256=parquet_fingerprint["sha256"],
        ).resolve()))
        atomic_write_json(
            work.part_manifest,
            {
                "part_hash": work.part_hash,
                "assignment_version": PARTITION_ASSIGNMENT_VERSION,
                "execution_id": options.execution_id or "",
                "study_ids": work.study_ids,
                "row_count": len(frame),
                "parquet_size": parquet_fingerprint["size"],
                "parquet_sha256": parquet_fingerprint["sha256"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _remove_resume_fragments(work.resume_root, work.resume_manifest)
        work.output_paths.append(str(work.part_path.resolve()))
        success_count += int((frame["extraction_status"] == "ok").sum())
        failure_count += int((frame["extraction_status"] != "ok").sum())
        calculator_failures = pd.to_numeric(
            frame.get(
                "biomarker_calculation_failure_count",
                pd.Series(0, index=frame.index),
            ),
            errors="coerce",
        ).fillna(0)
        calculator_failure_count += int(calculator_failures.sum())
        calculator_failure_ecg_count += int((calculator_failures > 0).sum())
        completed_records += len(frame)
        work.done = True
        work.completed = work.success = work.failure = 0
        work.calculation_failures = 0
        del frame
        if work.had_pending and not managed:
            # Unsupervised runs keep their own slow CPU/RAM adaptation; it
            # changes admission only, never the pool.
            target = recommend_dynamic_workers(
                admitted_worker_budget,
                sample_interval=0.5,
                current_workers=allowed_workers,
                reserve_gb=config.os_memory_reserve_gb,
            )
            step = dynamic_resize_state.step(allowed_workers, target)
            if step is not None:
                allowed_workers = max(1, min(int(step), admitted_worker_budget))
        if progress and allowed_workers != reported_workers:
            progress(
                f"Worker angepasst: {reported_workers} -> {allowed_workers} "
                "(Zuteilung; Pool bleibt bestehen)"
            )
            reported_workers = allowed_workers
        if progress and not work.had_pending:
            report_progress()

    def fill_window() -> None:
        nonlocal chunks_exhausted
        # Two unfinished partitions are enough to keep every admitted worker
        # busy across a partition boundary; finished ones are flushed in
        # chunk order so the output list stays deterministic.
        while not chunks_exhausted and sum(
            1 for work in window if not work.done
        ) < _PARTITION_LOOKAHEAD:
            try:
                chunk_index, (split, part_index, chunk) = next(chunk_iterator)
            except StopIteration:
                chunks_exhausted = True
                return
            work = prepare_partition(chunk_index, split, part_index, chunk)
            window.append(work)
            if not work.done and not work.pending:
                finalize_partition(work)

    def flush_finished() -> None:
        while window and window[0].done:
            output_paths.extend(window.popleft().output_paths)

    def accept(work: _PartitionWork, results: list[dict]) -> None:
        work.buffer.extend(results)
        partition_drained = not work.pending and work.outstanding == 0
        if len(work.buffer) >= micro_size or (partition_drained and work.buffer):
            commit_buffer(work)
        if partition_drained and not work.buffer:
            finalize_partition(work)

    try:
        while True:
            fill_window()
            flush_finished()
            if chunks_exhausted and not window:
                break
            if pool.executor is None:
                work = next((item for item in window if item.pending), None)
                if work is None:  # pragma: no cover - invariant guard
                    raise RuntimeError("Feature scheduler has no runnable work")
                row = work.pending.popleft()
                accept(work, [_extract_record_worker(row)])
                continue
            limit = current_allowed()
            for work in window:
                while work.pending and len(in_flight) < limit:
                    row = work.pending.popleft()
                    future = pool.executor.submit(
                        _extract_record_batch_worker,
                        [row],
                    )
                    in_flight[future] = work
                    work.outstanding += 1
                if len(in_flight) >= limit:
                    break
            if not in_flight:  # pragma: no cover - invariant guard
                raise RuntimeError("Feature scheduler has no runnable work")
            finished, _ = wait(
                tuple(in_flight),
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            for future in finished:
                work = in_flight.pop(future)
                work.outstanding -= 1
                results = list(future.result())
                if governor is not None:
                    governor.note_completed(len(results))
                accept(work, results)
    finally:
        pool.close()

    _prune_orphan_partitions(output_root, output_paths)
    artifact_index_path = output_root / "partition_index.json"
    artifact_index = _write_feature_artifact_index(
        output_root,
        stage_input_hash=input_hash,
        stage_input_payload=stage_input_payload,
        source_code_hash=feature_code_hash,
        panel_hash=panel_payload["panel_hash"],
        feature_ids=feature_ids,
        # Output provenance also includes the JSON timing contract. Only data
        # partitions belong in the strict Parquet identity/integrity index.
        parquet_paths=_feature_partition_paths(output_paths),
        destination=artifact_index_path,
    )
    output_paths.append(str(artifact_index_path.resolve()))
    if progress:
        progress("Erzeuge versionsgebundenen Feature-Fehlerbericht")
    glob = output_root / "split=*" / "part-*.parquet"
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    failure_path = output_root / "feature_failure_report.parquet"
    feature_sql = str(glob).replace("\\", "/").replace("'", "''")
    with catalog.connect(isolated=True) as connection:
        read_features = feature_parquet_clause(
            connection=connection,
            sql_path=feature_sql,
        )
        failure_frame = connection.execute(
            f"""
            WITH features AS (
                SELECT
                    extraction_status,
                    error_type,
                    error_message,
                    biomarker_calculation_failure_names
                FROM {read_features}
            ),
            extraction_failures AS (
                SELECT
                    extraction_status,
                    error_type,
                    error_message,
                    COUNT(*) AS record_count
                FROM features
                WHERE extraction_status <> 'ok'
                GROUP BY extraction_status, error_type, error_message
            ),
            calculator_failures AS (
                SELECT
                    'biomarker_calculation_failed' AS extraction_status,
                    'BiomarkerCalculationError' AS error_type,
                    failure_name AS error_message,
                    COUNT(*) AS record_count
                FROM features,
                UNNEST(
                    string_split(
                        COALESCE(
                            biomarker_calculation_failure_names,
                            ''
                        ),
                        '|'
                    )
                ) AS names(failure_name)
                WHERE failure_name <> ''
                GROUP BY failure_name
            )
            SELECT * FROM extraction_failures
            UNION ALL
            SELECT * FROM calculator_failures
            ORDER BY record_count DESC
            """
        ).fetchdf()
    atomic_write_parquet(failure_frame, failure_path)
    output_paths.append(str(failure_path.resolve()))

    sensitivity_path = output_root / "transform_sensitivity_report.parquet"
    sensitivity_report = _transform_sensitivity_report(
        output_root,
        sensitivity_feature_ids,
        tuple(config.sensitivity_vcg_methods),
    )
    atomic_write_parquet(sensitivity_report, sensitivity_path)
    output_paths.append(str(sensitivity_path.resolve()))
    transform_independent_twins = int(
        (~sensitivity_report.get(
            "transform_dependent", pd.Series(dtype=bool)
        ).astype(bool)).sum()
    )

    plausibility_path = output_root / "feature_plausibility_report.parquet"
    plausibility = _plausibility_report(
        output_root,
        feature_ids,
        progress=progress,
    )
    atomic_write_parquet(plausibility, plausibility_path)
    output_paths.append(str(plausibility_path.resolve()))
    implausible_markers = int(
        (
            pd.to_numeric(
                plausibility.get("violating_values", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)
            > 0
        ).sum()
    )

    if timing_contract is not None:
        timing_contract["observed_reused_records"] = reused_records
        timing_contract["observed_selected_records"] = total
        timing_contract["observed_recomputed_records"] = total - reused_records
        atomic_write_json(timing_contract_path, timing_contract)
    manifest = StageManifest.complete(
        stage="extract-features",
        stage_version=FEATURE_STAGE_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(output_paths),
        row_counts={
            "selected_ecgs": total,
            "successful_ecgs": success_count,
            "failed_ecgs": failure_count,
            "biomarker_calculation_failures": calculator_failure_count,
            "ecgs_with_biomarker_calculation_failures": (
                calculator_failure_ecg_count
            ),
            "skipped_partitions": skipped_partitions,
            "adopted_partitions": adopted_partitions,
            "reused_records_including_resume_fragments": reused_records,
            "feature_count": len(feature_ids),
            "transform_independent_twins": transform_independent_twins,
            "plausibility_checked_markers": int(len(plausibility)),
            "implausible_markers": implausible_markers,
            "sensitivity_feature_count": (
                len(sensitivity_feature_ids)
                * len(tuple(config.sensitivity_vcg_methods))
            ),
        },
        notes=(
            f"panels={','.join(options.panel_ids)}",
            f"locked_test_features_included={locked_test_only}",
            f"locked_test_only={locked_test_only}",
            (
                "locked_test_selection=frozen_locked_cohort_only"
                if locked_test_only
                else "locked_test_selection=not_loaded"
            ),
            (
                "development_union=clinical_assessable_plus_exact_phewas_"
                "case_control_indices"
            ),
            (
                f"capped_sampling={CAPPED_SAMPLING_VERSION}"
                if options.max_records is not None
                else "capped_sampling=not_applicable_full_union"
            ),
            (
                f"freeze_hash={freeze_context['freeze_hash']}"
                if freeze_context
                else "freeze_hash=not_applicable"
            ),
            f"feature_artifact_hash={artifact_index['artifact_hash']}",
            f"feature_timing_contract_sha256={file_sha256(timing_contract_path) if timing_contract is not None else 'not_applicable'}",
            f"operational_config_hash={operational_config_hash(config)}",
        ),
    )
    manifest.save(manifest_path)
    catalog.register_stage(manifest, manifest_path)
    return manifest



def _cast_all_null_features_to_float(frame: pd.DataFrame, feature_ids) -> None:
    """A feature column with no value in this partition must still be a double.

    Pandas hands an all-``None`` object column to Arrow as the ``null`` type,
    while the same marker is ``double`` in every partition where it has a
    value. Reading the set then depends on which file comes first. Only the
    all-null case is touched; a column with values keeps the dtype it has.
    """

    for feature_id in feature_ids:
        column = frame[feature_id]
        if column.dtype == object and column.isna().all():
            frame[feature_id] = column.astype("float64")


PARTITION_INTEGER_IDENTIFIERS = frozenset({"study_id", "subject_id", "hadm_id"})


def _cast_integer_counts_to_float(frame: pd.DataFrame) -> None:
    """Write every non-identifier integer column as a double.

    A count such as ``detected_beat_count`` is ``int64`` in a partition where
    no record lacks it and ``double`` where one does. On the A147 store 133
    of 2,562 partitions differed only that way; the differing schemas forced
    every full read into DuckDB's ``union_by_name`` path, which held about
    19 GB outside its memory limit and stopped the run on a 32 GB machine
    (UMSETZUNG_2026-09-23.md). Identifiers stay integers.
    """

    for name in frame.columns:
        if name in PARTITION_INTEGER_IDENTIFIERS:
            continue
        dtype = frame[name].dtype
        if pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            frame[name] = frame[name].astype("float64")


def build_machine_report_labels(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    master_path = config.master_root / "ecg_master_index.parquet"
    output_path = config.evidence_root / "machine_report_labels.parquet"
    mentions_path = config.evidence_root / "machine_report_mentions.parquet"
    manifest_path = config.evidence_root / "machine_reports.manifest.json"
    input_hash = content_hash(
        {
            "master": file_fingerprint(master_path),
            "parser_version": REPORT_PARSER_VERSION,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.exists():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, REPORT_PARSER_VERSION):
            return previous

    parser = MachineReportParser()
    frame = pd.read_parquet(
        master_path,
        columns=[
            "subject_id",
            "study_id",
            "split",
            "machine_report",
            "machine_time_mismatch",
            "machine_time_mismatch_seconds",
            "machine_metadata_usable_for_primary",
            "machine_metadata_temporal_status",
        ],
    )
    rows = []
    mentions = []
    total = len(frame)
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        metadata_usable = bool(row.machine_metadata_usable_for_primary)
        parsed = parser.parse(
            row.machine_report if metadata_usable else ""
        )
        output = {
            "subject_id": row.subject_id,
            "study_id": row.study_id,
            "split": row.split,
            "machine_metadata_usable": metadata_usable,
            "machine_metadata_exclusion_reason": (
                "" if metadata_usable else row.machine_metadata_temporal_status
            ),
            "machine_time_mismatch_seconds": (
                row.machine_time_mismatch_seconds
            ),
        }
        for label_id, label in parsed.items():
            output[f"report_{label_id}_state"] = label.state.value
            if label.state != ReportMentionState.NOT_MENTIONED:
                mentions.append(
                    {
                        "subject_id": row.subject_id,
                        "study_id": row.study_id,
                        "split": row.split,
                        **label.to_dict(),
                    }
                )
        rows.append(output)
        if progress and index % 50_000 == 0:
            progress(f"Machine-Reports {index:,}/{total:,}")
    output_frame = pd.DataFrame(rows)
    mention_frame = pd.DataFrame(mentions)
    atomic_write_parquet(output_frame, output_path)
    atomic_write_parquet(mention_frame, mentions_path)
    manifest = StageManifest.complete(
        stage="parse-machine-reports",
        stage_version=REPORT_PARSER_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(output_path.resolve()),
            str(mentions_path.resolve()),
        ),
        row_counts={
            "ecg_reports": len(output_frame),
            "report_mentions": len(mention_frame),
        },
        notes=(f"operational_config_hash={operational_config_hash(config)}",),
        output_integrity=capture_output_integrity((output_path, mentions_path)),
    )
    manifest.save(manifest_path)
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    catalog.register_stage(manifest, manifest_path)
    catalog.create_parquet_view("machine_report_labels", output_path)
    catalog.create_parquet_view("machine_report_mentions", mentions_path)
    return manifest



MACHINE_INTERVAL_COLUMNS = (
    "machine_qrs_duration_ms",
    "machine_qt_interval_ms",
    "machine_pr_interval_ms",
    # Not an interval, but gated the same way: the recorder's T axis for the
    # device frontal QRS-T angle (audit 5.1).
    "machine_t_axis_deg",
)


def _machine_interval_select(available: set[str], *, prefix: str = "") -> str:
    """Recorder intervals gated like the axis; typed NULL when the index lacks them."""

    parts = []
    for name in MACHINE_INTERVAL_COLUMNS:
        if name in available:
            parts.append(
                f"CASE WHEN {prefix}machine_metadata_usable_for_primary "
                f"THEN {prefix}{name} ELSE NULL END AS {name}"
            )
        else:
            parts.append(f"NULL::DOUBLE AS {name}")
    return ", ".join(parts)


def _load_feature_cohort(
    config: MimicProjectConfig,
    options: FeatureExtractionOptions,
    *,
    phenotype_mapping_path: Optional[Path] = None,
) -> pd.DataFrame:
    catalog = DuckDBCatalog(
        config.catalog_path,
        memory_limit_gb=config.duckdb_memory_limit_gb,
        threads=config.workers,
        follow_grant=True,
    )
    master = _sql_path(config.master_root / "ecg_master_index.parquet")
    # Build optional-column SQL before composing the cohort CTE strings.
    # Older fixture/artifact schemas predate recorder bandwidth metadata and
    # therefore receive a disclosed NULL rather than a binder failure.
    with catalog.connect(isolated=True) as schema_connection:
        cohort_master_columns = relation_columns(
            schema_connection,
            f"read_parquet('{master}')",
        )
    machine_bandwidth_select = optional_column_select(
        cohort_master_columns,
        "machine_bandwidth",
        "VARCHAR",
        prefix="m.",
    )
    machine_filtering_select = optional_column_select(
        cohort_master_columns,
        "machine_filtering",
        "VARCHAR",
        prefix="m.",
    )
    machine_interval_select = _machine_interval_select(
        cohort_master_columns, prefix="m."
    )
    if options.locked_test_only or options.include_locked_test:
        with catalog.connect(isolated=True) as connection:
            master_columns = relation_columns(
                connection,
                f"read_parquet('{master}')",
            )
            machine_bandwidth_select = optional_column_select(
                master_columns,
                "machine_bandwidth",
                "VARCHAR",
            )
            machine_filtering_select = optional_column_select(
                master_columns,
                "machine_filtering",
                "VARCHAR",
            )
            machine_interval_select = _machine_interval_select(master_columns)
            return connection.execute(
                f"""
                SELECT
                    subject_id,
                    study_id,
                    split,
                    ecg_time,
                    waveform_path,
                    age,
                    sex,
                    CASE WHEN machine_metadata_usable_for_primary
                        THEN machine_qrs_axis_deg ELSE NULL
                    END AS machine_qrs_axis_deg,
                    {machine_interval_select},
                    CASE WHEN machine_metadata_usable_for_primary
                        THEN machine_report ELSE NULL
                    END AS machine_report,
                    machine_cart_id,
                    {machine_bandwidth_select},
                    {machine_filtering_select},
                    machine_time_mismatch,
                    machine_time_mismatch_seconds,
                    machine_metadata_temporal_status,
                    machine_metadata_usable_for_primary
                FROM read_parquet('{master}')
                WHERE split = 'locked_test'
                ORDER BY study_id
                """
            ).fetchdf()

    cohort = _sql_path(config.labels_root / "clinical_label_evidence.parquet")
    endpoint_index_path = config.labels_root / "endpoint_index_cohorts.parquet"
    if options.max_records is not None and not endpoint_index_path.is_file():
        raise FileNotFoundError(
            "A capped development pilot requires endpoint_index_cohorts.parquet"
        )
    pilot_cohort = (
        _sql_path(endpoint_index_path)
        if options.max_records is not None
        else cohort
    )
    diagnoses_path = config.standardized_root / "diagnoses_icd.parquet"
    if diagnoses_path.exists() and phenotype_mapping_path is not None:
        diagnoses = _sql_path(diagnoses_path)
        phenome_ctes = f"""
            coded_admissions AS (
                SELECT DISTINCT subject_id, hadm_id
                FROM read_parquet('{diagnoses}')
                WHERE subject_id IS NOT NULL AND hadm_id IS NOT NULL
            ),
            phenotype_index_candidates AS (
                SELECT
                    m.subject_id,
                    m.study_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.subject_id
                        ORDER BY m.ecg_time, m.study_id
                    ) AS index_rank
                FROM read_parquet('{master}') m
                JOIN coded_admissions d USING (subject_id, hadm_id)
                WHERE m.split IN ('train', 'validation')
            ),
            phenotype_feature_index AS (
                SELECT subject_id, study_id
                FROM phenotype_index_candidates
                WHERE index_rank = 1
            ),
        """
        phenome_union = f"""
                UNION
                SELECT
                    m.subject_id,
                    m.study_id,
                    m.split,
                    m.ecg_time,
                    m.waveform_path,
                    m.age,
                    m.sex,
                    CASE WHEN m.machine_metadata_usable_for_primary
                        THEN m.machine_qrs_axis_deg ELSE NULL
                    END AS machine_qrs_axis_deg,
                    {machine_interval_select},
                    CASE WHEN m.machine_metadata_usable_for_primary
                        THEN m.machine_report ELSE NULL
                    END AS machine_report,
                    m.machine_cart_id,
                    {machine_bandwidth_select},
                    {machine_filtering_select},
                    m.machine_time_mismatch,
                    m.machine_time_mismatch_seconds,
                    m.machine_metadata_temporal_status,
                    m.machine_metadata_usable_for_primary
                FROM read_parquet('{master}') m
                JOIN phenotype_feature_index p USING (subject_id, study_id)
                WHERE m.split IN ('train', 'validation')
        """
    else:
        phenome_ctes = ""
        phenome_union = ""
    split_count = 2
    per_split = (
        math.ceil(int(options.max_records) / split_count)
        if options.max_records is not None
        else None
    )
    sample_filter = (
        f"WHERE split_row_number <= {per_split}"
        if per_split is not None
        else ""
    )
    final_limit = (
        f"LIMIT {int(options.max_records)}"
        if options.max_records is not None
        else ""
    )
    pilot_ctes = ""
    pilot_join = ""
    ranking_order = "e.study_id"
    if per_split is not None:
        pilot_ctes = f""",
            clinical_pilot_rows AS (
                SELECT DISTINCT
                    m.split,
                    c.label_id,
                    c.label_state,
                    m.study_id
                FROM read_parquet('{pilot_cohort}') c
                JOIN read_parquet('{master}') m USING (subject_id, study_id)
                WHERE c.label_state IN ('positive', 'negative')
                  AND c.eligibility_state = 'eligible'
                  AND m.split IN ('train', 'validation')
            ),
            clinical_pilot_strata AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY split, label_id, label_state
                        ORDER BY
                            md5(
                                CONCAT(
                                    CAST(study_id AS VARCHAR), ':', split, ':',
                                    label_id, ':', label_state, ':',
                                    '{CAPPED_SAMPLING_VERSION}'
                                )
                            ),
                            study_id
                    ) AS stratum_rank
                FROM clinical_pilot_rows
            ),
            clinical_pilot_stratum_counts AS (
                SELECT split, COUNT(*) AS n_strata
                FROM (
                    SELECT DISTINCT split, label_id, label_state
                    FROM clinical_pilot_rows
                )
                GROUP BY split
            ),
            pilot_coverage_ecgs AS (
                SELECT DISTINCT s.split, s.study_id
                FROM clinical_pilot_strata s
                JOIN clinical_pilot_stratum_counts n USING (split)
                WHERE s.stratum_rank <= GREATEST(
                    1,
                    CAST(
                        FLOOR(
                            ({per_split} * 0.80) / NULLIF(n.n_strata, 0)
                        ) AS BIGINT
                    )
                )
            )
        """
        pilot_join = """
                LEFT JOIN pilot_coverage_ecgs p
                  ON e.split = p.split AND e.study_id = p.study_id
        """
        ranking_order = f"""
                        CASE WHEN p.study_id IS NULL THEN 1 ELSE 0 END,
                        md5(
                            CONCAT(
                                CAST(e.study_id AS VARCHAR), ':', e.split, ':',
                                '{CAPPED_SAMPLING_VERSION}'
                            )
                        ),
                        e.study_id
        """
    with catalog.connect(isolated=True) as connection:
        master_columns = relation_columns(
            connection,
            f"read_parquet('{master}')",
        )
        machine_bandwidth_select = optional_column_select(
            master_columns,
            "machine_bandwidth",
            "VARCHAR",
            prefix="m.",
        )
        machine_filtering_select = optional_column_select(
            master_columns,
            "machine_filtering",
            "VARCHAR",
            prefix="m.",
        )
        machine_interval_select = _machine_interval_select(
            master_columns, prefix="m."
        )
        frame = connection.execute(
            f"""
            WITH {phenome_ctes}
            eligible AS (
                SELECT DISTINCT
                    m.subject_id,
                    m.study_id,
                    m.split,
                    m.ecg_time,
                    m.waveform_path,
                    m.age,
                    m.sex,
                    CASE WHEN m.machine_metadata_usable_for_primary
                        THEN m.machine_qrs_axis_deg ELSE NULL
                    END AS machine_qrs_axis_deg,
                    {machine_interval_select},
                    CASE WHEN m.machine_metadata_usable_for_primary
                        THEN m.machine_report ELSE NULL
                    END AS machine_report,
                    m.machine_cart_id,
                    {machine_bandwidth_select},
                    {machine_filtering_select},
                    m.machine_time_mismatch,
                    m.machine_time_mismatch_seconds,
                    m.machine_metadata_temporal_status,
                    m.machine_metadata_usable_for_primary
                FROM read_parquet('{cohort}') c
                JOIN read_parquet('{master}') m USING (subject_id, study_id)
                WHERE c.label_state IN ('positive', 'negative')
                  AND c.eligibility_state = 'eligible'
                  AND m.split IN ('train', 'validation')
                {phenome_union}
            )
            {pilot_ctes},
            ranked AS (
                SELECT
                    e.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY e.split ORDER BY {ranking_order}
                    ) AS split_row_number
                FROM eligible e
                {pilot_join}
            )
            SELECT * EXCLUDE (split_row_number)
            FROM ranked
            {sample_filter}
            ORDER BY split, study_id
            {final_limit}
            """
        ).fetchdf()
    if frame["study_id"].isna().any():
        raise RuntimeError("Feature cohort contains ECG rows without study_id")
    duplicate = frame["study_id"].duplicated(keep=False)
    if duplicate.any():
        examples = sorted(
            frame.loc[duplicate, "study_id"].astype(int).unique()
        )[:10]
        raise RuntimeError(
            "Feature cohort contains duplicate study_id values after clinical/"
            f"phenome union: {examples}"
        )
    return frame


def _locked_test_freeze_context(
    config: MimicProjectConfig,
    *,
    run_id: str,
    feature_code_hash: str,
    panel_hash: str,
) -> dict:
    run_root = config.runs_root / run_id
    freeze_path = run_root / "freeze_manifest.json"
    lock_path = run_root / "test_lock.json"
    if not freeze_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError(
            "Locked-test feature extraction requires an immutable development "
            f"freeze for run {run_id!r}."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    freeze_hash = str(freeze.get("freeze_hash", ""))
    if not freeze_hash or lock.get("freeze_hash") != freeze_hash:
        raise RuntimeError("Freeze manifest and test lock do not agree")
    if lock.get("invalidated_at"):
        raise RuntimeError("Locked-test freeze was invalidated")
    if lock.get("evaluation_started_at") or lock.get("consumed_at"):
        raise RuntimeError(
            "Locked-test feature extraction is forbidden after test evaluation "
            "has started."
        )
    frozen_integrity = freeze.get("payload", {}).get(
        "feature_integrity",
        {},
    )
    if frozen_integrity.get("source_code_hash") != feature_code_hash:
        raise RuntimeError(
            "Current feature implementation differs from the frozen "
            "development implementation."
        )
    if frozen_integrity.get("panel_hash") != panel_hash:
        raise RuntimeError(
            "Current feature panel differs from the frozen development panel."
        )
    return {
        "freeze_hash": freeze_hash,
        "feature_code_hash": feature_code_hash,
        "panel_hash": panel_hash,
    }


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def _sql_path(path: Path | str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", "''")


def _cohort_record_payloads(cohort: pd.DataFrame) -> dict[int, str]:
    """Return the per-record extraction inputs, keyed by study_id.

    Every column the cohort query selects is an input to the extraction, so all
    of them are carried.  Values are normalised to plain strings and numbers
    because the payload is hashed, and a hash that moved with the pandas dtype
    would invalidate work for no reason.
    """

    columns = [str(name) for name in cohort.columns]
    # Keep one canonical JSON string per ECG rather than one nested Python
    # dictionary.  On a 640k-record cohort the dictionaries alone occupied
    # hundreds of megabytes (before their scalar objects); the compact form
    # preserves the byte-for-byte hashes and is decoded only for the current
    # 250-record partition.
    payloads: dict[int, str] = {}
    for record in cohort.to_dict("records"):
        study_id = int(record["study_id"])
        payloads[study_id] = canonical_json(
            {
                name: _hashable_scalar(record.get(name))
                for name in columns
            }
        )
    return payloads


def _hashable_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _cohort_selection_hash(payloads: Mapping[int, str]) -> str:
    """Content hash of the selected records, independent of file timestamps."""

    # ``canonical_json([record_1, ...])`` is exactly an opening bracket, the
    # already canonical object strings separated by commas, and a closing
    # bracket. Stream those bytes so provenance remains identical without
    # reconstructing the full nested list in memory.
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, study_id in enumerate(sorted(payloads)):
        if index:
            digest.update(b",")
        digest.update(payloads[study_id].encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def feature_stage_input_payload(
    record_payloads: Mapping[int, str],
    *,
    panel_hash: str,
    feature_code_hash: str,
    options: FeatureExtractionOptions,
    sensitivity_vcg_methods: tuple[str, ...],
    freeze_hash: Optional[str],
) -> dict[str, object]:
    """One shared provenance contract for extraction and validation."""

    locked_test_only = bool(
        options.locked_test_only or options.include_locked_test
    )
    return {
        "cohort_selection": _cohort_selection_hash(record_payloads),
        "panel_hash": str(panel_hash),
        "feature_code_hash": str(feature_code_hash),
        "locked_test_only": locked_test_only,
        "freeze_hash": freeze_hash,
        "max_records": options.max_records,
        "capped_sampling_version": (
            CAPPED_SAMPLING_VERSION
            if options.max_records is not None
            else None
        ),
        "partition_size": int(options.partition_size),
        "panel_ids": tuple(options.panel_ids),
        "sensitivity_vcg_methods": tuple(sensitivity_vcg_methods),
    }


def current_development_feature_input_hash(
    config: MimicProjectConfig,
    *,
    panel_ids: tuple[str, ...],
    partition_size: int,
    panel_hash: str,
) -> str:
    """Reconstruct the producer's exact full-development input hash."""

    options = FeatureExtractionOptions(
        panel_ids=tuple(panel_ids),
        partition_size=int(partition_size),
    )
    diagnosis_path = config.standardized_root / "diagnoses_icd.parquet"
    phenotype_resources = (
        prepare_phenotype_resources(config)
        if diagnosis_path.is_file()
        else None
    )
    cohort = _load_feature_cohort(
        config,
        options,
        phenotype_mapping_path=(
            phenotype_resources.phecodex_mapping_path
            if phenotype_resources is not None
            else None
        ),
    )
    record_payloads = _cohort_record_payloads(cohort)
    payload = feature_stage_input_payload(
        record_payloads,
        panel_hash=panel_hash,
        feature_code_hash=feature_implementation_hash(),
        options=options,
        sensitivity_vcg_methods=tuple(config.sensitivity_vcg_methods),
        freeze_hash=None,
    )
    return content_hash(payload)


def _partition_assignment_path(output_root: Path) -> Path:
    return output_root / "partition_assignment.parquet"


def _load_partition_assignment(output_root: Path) -> dict[int, tuple[str, int]]:
    path = _partition_assignment_path(output_root)
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}
    if not {"study_id", "split", "part_index"} <= set(frame.columns):
        return {}
    return {
        int(study_id): (str(split), int(part_index))
        for study_id, split, part_index in zip(
            frame["study_id"], frame["split"], frame["part_index"]
        )
    }


def _save_partition_assignment(
    output_root: Path,
    assignment: Mapping[int, tuple[str, int]],
) -> None:
    keys = sorted(assignment)
    frame = pd.DataFrame(
        {
            "study_id": pd.Series([int(key) for key in keys], dtype="int64"),
            "split": pd.Series(
                [str(assignment[key][0]) for key in keys], dtype="object"
            ),
            "part_index": pd.Series(
                [int(assignment[key][1]) for key in keys], dtype="int32"
            ),
        }
    )
    atomic_write_parquet(frame, _partition_assignment_path(output_root))


def _bootstrap_partition_assignment(
    output_root: Path,
) -> dict[int, tuple[str, int]]:
    """Recover the assignment of an extraction written before it was recorded.

    Reads only the study_id column of every existing partition, so work already
    on disk keeps its partition membership instead of being re-chunked --
    re-chunking would invalidate every partition, which is exactly what this
    scheme exists to prevent.
    """

    assignment: dict[int, tuple[str, int]] = {}
    for part_path in sorted(output_root.glob("split=*/part-*.parquet")):
        split = part_path.parent.name.split("=", 1)[-1]
        try:
            part_index = int(part_path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        try:
            identity = _read_single_parquet(part_path, columns=["study_id"])
        except (OSError, ValueError, KeyError):
            continue
        for value in identity["study_id"]:
            if pd.isna(value):
                continue
            assignment[int(value)] = (split, part_index)
    return assignment


def _assign_cohort_partitions(
    cohort: pd.DataFrame,
    *,
    output_root: Path,
    partition_size: int,
    force: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> list[tuple[str, int, pd.DataFrame]]:
    """Group the cohort into partitions that survive the cohort growing.

    Partitions used to be positional chunks of a cohort ordered by
    (split, study_id).  Inserting a single newly eligible ECG shifted every
    later chunk boundary, so a label-rule change re-extracted all 640k records
    even though no marker depends on a label.  Assignments are therefore made
    once and kept: an ECG stays in the partition it was first put into, and
    newly eligible ECGs are appended in fresh partitions.
    """

    size = max(1, int(partition_size))
    assignment = {} if force else _load_partition_assignment(output_root)
    if not assignment and not force:
        assignment = _bootstrap_partition_assignment(output_root)
        if assignment and progress:
            progress(
                "Partitionszuordnung aus vorhandenen Feature-Dateien "
                f"uebernommen ({len(assignment):,} ECGs)"
            )
    next_index = max((index for _, index in assignment.values()), default=-1) + 1

    splits = cohort["split"].astype(str).to_numpy()
    study_ids = cohort["study_id"].astype("int64").to_numpy()
    groups: dict[tuple[str, int], list[int]] = {}
    unassigned: dict[str, list[int]] = {}
    for split, study_id in zip(splits, study_ids):
        study_id = int(study_id)
        placed = assignment.get(study_id)
        if placed is not None and placed[0] == split:
            groups.setdefault(placed, []).append(study_id)
        else:
            unassigned.setdefault(split, []).append(study_id)

    for split in sorted(unassigned):
        pending = sorted(unassigned[split])
        for start in range(0, len(pending), size):
            block = pending[start : start + size]
            key = (split, next_index)
            next_index += 1
            groups[key] = block
            for study_id in block:
                assignment[study_id] = key
    if progress and unassigned:
        added = sum(len(values) for values in unassigned.values())
        progress(f"Neue Partitionen fuer {added:,} zusaetzliche ECGs angelegt")
    _save_partition_assignment(output_root, assignment)

    position_of = {int(value): index for index, value in enumerate(study_ids)}
    chunks: list[tuple[str, int, pd.DataFrame]] = []
    for key in sorted(groups):
        members = sorted(groups[key])
        positions = np.asarray([position_of[value] for value in members], dtype=int)
        chunks.append((key[0], key[1], cohort.iloc[positions]))
    return chunks


def _adopt_existing_partition(
    part_path: Path,
    part_manifest: Path,
    *,
    part_hash: str,
    expected_study_ids: Iterable[int],
    execution_id: str,
) -> bool:
    """Re-key an already computed partition instead of computing it again.

    Called only when the marker code, panels and scientific configuration are
    unchanged, so the values in the file are exactly the values this run would
    produce.  What changed is the *key*, and -- when ECGs left the cohort -- the
    membership.  Rows are then filtered out of the existing file; no waveform is
    read and no marker is recomputed.
    """

    expected = [int(value) for value in expected_study_ids]
    if not expected:
        return False
    try:
        identity = _read_single_parquet(part_path, columns=["study_id"])
    except (OSError, ValueError, KeyError):
        return False
    if identity["study_id"].isna().any():
        return False
    present = [int(value) for value in identity["study_id"]]
    if len(present) != len(set(present)):
        return False
    if not set(expected) <= set(present):
        return False
    if len(expected) != len(present):
        try:
            frame = _read_single_parquet(part_path)
        except (OSError, ValueError):
            return False
        frame = (
            frame.drop_duplicates("study_id", keep="last")
            .set_index("study_id")
            .reindex(expected)
            .reset_index()
        )
        atomic_write_parquet(frame, part_path)
        row_count = len(frame)
    else:
        if present != expected:
            try:
                frame = _read_single_parquet(part_path)
            except (OSError, ValueError):
                return False
            frame = (
                frame.drop_duplicates("study_id", keep="last")
                .set_index("study_id")
                .reindex(expected)
                .reset_index()
            )
            atomic_write_parquet(frame, part_path)
            row_count = len(frame)
        else:
            row_count = len(present)
    fingerprint = file_fingerprint(part_path, include_sha256=True)
    atomic_write_json(
        part_manifest,
        {
            "part_hash": part_hash,
            "assignment_version": PARTITION_ASSIGNMENT_VERSION,
            "execution_id": execution_id,
            "study_ids": expected,
            "row_count": int(row_count),
            "parquet_size": fingerprint["size"],
            "parquet_sha256": fingerprint["sha256"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "adopted_from_earlier_partition_scheme": True,
        },
    )
    return True


def _legacy_partitions_are_current(
    output_root: Path,
    *,
    feature_code_hash: str,
    panel_hash: str,
    config_hash: str,
) -> bool:
    """Whether partitions from the earlier scheme still hold valid markers.

    The partition key no longer contains the stage input hash, so partitions
    written before that change carry a key that cannot be recomputed.  Their
    content is still valid whenever the marker code, the panels and the
    scientific configuration are unchanged -- and all three are recorded.
    Checking them is what lets an existing extraction be adopted instead of
    repeated.
    """

    index_path = output_root / "partition_index.json"
    manifest_path = output_root / "features.manifest.json"
    if not index_path.exists() or not manifest_path.exists():
        return False
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        str(index.get("source_code_hash") or "") == str(feature_code_hash)
        and str(index.get("panel_hash") or "") == str(panel_hash)
        and str(manifest.get("config_hash") or "") == str(config_hash)
    ):
        return True
    # The recorded hash can move for a reason that leaves every marker value
    # untouched -- the record-level code being split out of this module did
    # exactly that.  An attestation is accepted in its place only when it was
    # earned by recomputing a sample and comparing every marker value; see
    # ``feature_adoption.verify_stored_partitions``.
    from .feature_adoption import adoption_attestation

    return (
        adoption_attestation(
            output_root,
            feature_code_hash=feature_code_hash,
            panel_hash=panel_hash,
            config_hash=config_hash,
        )
        is not None
    )


_DYNAMIC_RESIZE_CONSECUTIVE_SAMPLES = 3


class _DynamicWorkerHysteresis:
    """Only recommends a resize after several consecutive same-direction
    signals, and only by one worker at a time -- mirrors the
    `MEMORY_PRESSURE_SAMPLES` debounce already used for the supervisor's
    live memory-pressure pause, so a single noisy CPU-load sample cannot
    thrash the pool size."""

    def __init__(self) -> None:
        self._streak = 0
        self._direction = 0

    def step(
        self,
        current_workers: int,
        recommended_workers: int,
    ) -> Optional[int]:
        if recommended_workers > current_workers:
            direction = 1
        elif recommended_workers < current_workers:
            direction = -1
        else:
            direction = 0
        if direction == 0 or direction != self._direction:
            self._direction = direction
            self._streak = 1 if direction != 0 else 0
            return None
        self._streak += 1
        if self._streak < _DYNAMIC_RESIZE_CONSECUTIVE_SAMPLES:
            return None
        self._streak = 0
        return current_workers + direction


# Unfinished partitions prepared ahead of the one being computed. Two keep
# every admitted worker busy across a partition boundary.
_PARTITION_LOOKAHEAD = 2


@dataclass
class _PartitionWork:
    """One partition while its records are in flight."""

    chunk_index: int
    study_ids: list
    expected_study_ids: set
    part_hash: str
    part_path: Path
    part_manifest: Path
    resume_root: Path
    resume_manifest: Path
    resume_state: Optional["_ResumeFragmentState"] = None
    pending: deque = field(default_factory=deque)
    buffer: list = field(default_factory=list)
    output_paths: list = field(default_factory=list)
    outstanding: int = 0
    completed: int = 0
    success: int = 0
    failure: int = 0
    calculation_failures: int = 0
    had_pending: bool = False
    done: bool = False


class _AdmissionPool:
    """One process pool for a whole extraction; admission decides the load.

    ``ProcessPoolExecutor`` starts processes lazily and reuses idle ones, so
    the number of processes only ever reaches the highest admission actually
    used. Nothing is torn down when the admission falls: fewer records are
    submitted, and running records finish normally. Measured on A131 before
    this change (2026-09-23): a pool rebuild at almost every micro-batch, ten
    worker starts in 90 s, ~48 % of worker CPU spent on 7-s cold imports.
    """

    def __init__(self, max_workers: int, task_config: Optional[dict] = None) -> None:
        self.max_workers = max(1, int(max_workers))
        self.executor: Optional[ProcessPoolExecutor] = None
        if self.max_workers > 1:
            if task_config is None:
                self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self.executor = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    initializer=_initialize_feature_worker,
                    initargs=(task_config,),
                )
        elif task_config is not None:
            _initialize_feature_worker(task_config)

    def close(self) -> None:
        if self.executor is not None:
            # Queued records are cancelled; they are not in any fragment and
            # are recomputed on resume.
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def _prune_orphan_partitions(
    output_root: Path,
    expected_parquet_paths: Iterable[str],
) -> None:
    root = output_root.resolve()
    expected = {Path(path).resolve() for path in expected_parquet_paths}
    expected_manifests = {path.with_suffix(".json") for path in expected}
    for pattern, retained in (
        ("split=*/part-*.parquet", expected),
        ("split=*/part-*.json", expected_manifests),
    ):
        for candidate in output_root.glob(pattern):
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise RuntimeError(
                    f"Refusing to prune partition outside feature root: {resolved}"
                )
            if resolved not in retained:
                resolved.unlink()


@dataclass(frozen=True)
class _ResumeFragmentState:
    root: Path
    manifest_path: Path
    part_hash: str
    execution_id: str
    expected_study_ids: frozenset[int]
    fragments: tuple[dict, ...] = ()

    @property
    def completed_study_ids(self) -> frozenset[int]:
        return frozenset(
            int(study_id)
            for fragment in self.fragments
            for study_id in fragment["study_ids"]
        )


def _empty_resume_fragment_state(
    root: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    execution_id: Optional[str],
    expected_study_ids: set[int],
) -> _ResumeFragmentState:
    return _ResumeFragmentState(
        root=Path(root),
        manifest_path=Path(manifest_path),
        part_hash=str(part_hash),
        execution_id=execution_id or "",
        expected_study_ids=frozenset(int(value) for value in expected_study_ids),
    )


def _open_resume_fragments(
    root: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    expected_study_ids: set[int],
    force: bool = False,
    execution_id: Optional[str] = None,
) -> tuple[_ResumeFragmentState, pd.DataFrame]:
    """Open and fully verify the last atomically committed fragment set."""
    empty = _empty_resume_fragment_state(
        root,
        manifest_path,
        part_hash=part_hash,
        execution_id=execution_id,
        expected_study_ids=expected_study_ids,
    )
    if not manifest_path.is_file() or not root.is_dir() or root.is_symlink():
        return empty, pd.DataFrame()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty, pd.DataFrame()
    if not isinstance(payload, dict):
        return empty, pd.DataFrame()
    stored_hash = payload.get("checkpoint_hash")
    hash_payload = dict(payload)
    hash_payload.pop("checkpoint_hash", None)
    if stored_hash != content_hash(hash_payload):
        return empty, pd.DataFrame()
    if (
        payload.get("checkpoint_version")
        != FEATURE_RESUME_CHECKPOINT_VERSION
        or payload.get("part_hash") != part_hash
    ):
        return empty, pd.DataFrame()
    stored_execution = str(payload.get("execution_id", ""))
    if force and (not execution_id or stored_execution != execution_id):
        return empty, pd.DataFrame()
    fragments = payload.get("fragments")
    if not isinstance(fragments, list):
        return empty, pd.DataFrame()
    completed: set[int] = set()
    normalized: list[dict] = []
    root_resolved = root.resolve()
    for sequence, fragment in enumerate(fragments):
        if not isinstance(fragment, dict):
            return empty, pd.DataFrame()
        relative_path = str(fragment.get("relative_path", ""))
        if (
            int(fragment.get("sequence", -1)) != sequence
            or not re.fullmatch(
                r"fragment-\d{6}-[0-9a-f]{16}\.parquet",
                relative_path,
            )
            or Path(relative_path).name != relative_path
        ):
            return empty, pd.DataFrame()
        data_path = root / relative_path
        try:
            resolved = data_path.resolve(strict=True)
        except OSError:
            return empty, pd.DataFrame()
        if resolved.parent != root_resolved or not resolved.is_file():
            return empty, pd.DataFrame()
        try:
            fingerprint = file_fingerprint(resolved, include_sha256=True)
        except OSError:
            return empty, pd.DataFrame()
        if (
            int(fragment.get("parquet_size", -1)) != fingerprint["size"]
            or fragment.get("parquet_sha256") != fingerprint["sha256"]
        ):
            return empty, pd.DataFrame()
        try:
            fragment_ids = tuple(int(value) for value in fragment["study_ids"])
        except (KeyError, TypeError, ValueError):
            return empty, pd.DataFrame()
        if (
            len(fragment_ids) != int(fragment.get("row_count", -1))
            or len(fragment_ids) != len(set(fragment_ids))
            or completed.intersection(fragment_ids)
            or not set(fragment_ids).issubset(expected_study_ids)
        ):
            return empty, pd.DataFrame()
        completed.update(fragment_ids)
        normalized.append(dict(fragment))
    manifest_ids = {int(value) for value in payload.get("study_ids", ())}
    if (
        manifest_ids != completed
        or int(payload.get("row_count", -1)) != len(completed)
        or int(payload.get("fragment_count", -1)) != len(normalized)
    ):
        return empty, pd.DataFrame()
    state = _ResumeFragmentState(
        root=root,
        manifest_path=manifest_path,
        part_hash=part_hash,
        execution_id=stored_execution,
        expected_study_ids=frozenset(expected_study_ids),
        fragments=tuple(normalized),
    )
    try:
        frame = _materialize_resume_fragments(state)
    except Exception:
        return empty, pd.DataFrame()
    return state, frame


def _load_resume_fragments(
    root: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    expected_study_ids: set[int],
    force: bool = False,
    execution_id: Optional[str] = None,
) -> pd.DataFrame:
    """Compatibility-friendly public helper returning the committed rows."""
    _, frame = _open_resume_fragments(
        root,
        manifest_path,
        part_hash=part_hash,
        expected_study_ids=expected_study_ids,
        force=force,
        execution_id=execution_id,
    )
    return frame


def _materialize_resume_fragments(
    state: _ResumeFragmentState,
) -> pd.DataFrame:
    if not state.fragments:
        return pd.DataFrame()
    paths = [
        str((state.root / fragment["relative_path"]).resolve())
        for fragment in state.fragments
    ]
    # One read per fragment, then a row-wise concat -- never a single
    # multi-path read.  Handing Arrow the whole list makes it unify the
    # schemas, and independently written fragments do not unify:
    #
    #   * they live under a ``split=<name>`` directory, so Arrow discovers a
    #     Hive partition key named ``split`` (dictionary-encoded) that clashes
    #     with the plain string column the fragments already carry;
    #   * a marker that abstained for every study in one fragment is stored
    #     with Arrow type ``null`` there and ``double`` in the next, which
    #     Arrow refuses to merge.
    #
    # Both aborted a finished 1000-record pilot at the very last step. The
    # fragments are disjoint row sets of one table, so concatenating them is
    # what was meant all along; pandas reconciles the dtypes on the way.
    frames = [pd.read_parquet(path) for path in paths]
    if len(frames) == 1:
        frame = frames[0]
    else:
        # A marker that abstained for every study of one fragment is an
        # all-NA column there. pandas no longer lets such a column decide the
        # result dtype silently and warns on every concat; under ``-W error``
        # that warning ended the production-path preview after the first
        # partition. The columns are excluded before the concat, as pandas
        # asks, and restored from the union afterwards so that a marker
        # absent from every fragment still leaves its column in the table.
        columns = list(dict.fromkeys(
            column for fragment in frames for column in fragment.columns
        ))
        frame = pd.concat(
            [fragment.dropna(axis=1, how="all") for fragment in frames],
            ignore_index=True,
            sort=False,
        ).reindex(columns=columns)
    if "study_id" not in frame or "extraction_status" not in frame:
        raise RuntimeError("Resume fragments lack extraction identity columns")
    if frame["study_id"].isna().any() or frame["study_id"].duplicated().any():
        raise RuntimeError("Resume fragments contain invalid study identities")
    frame_ids = frozenset(int(value) for value in frame["study_id"])
    if (
        frame_ids != state.completed_study_ids
        or not frame_ids.issubset(state.expected_study_ids)
        or len(frame) != len(frame_ids)
    ):
        raise RuntimeError("Resume fragment manifest/data identity mismatch")
    return frame


def _append_resume_fragment(
    frame: pd.DataFrame,
    root: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    execution_id: Optional[str],
    expected_study_ids: set[int],
    state: Optional[_ResumeFragmentState] = None,
) -> _ResumeFragmentState:
    """Atomically append one immutable micro-batch and publish its manifest."""
    if (
        "study_id" not in frame
        or "extraction_status" not in frame
        or frame.empty
        or frame["study_id"].isna().any()
        or frame["study_id"].duplicated().any()
    ):
        raise RuntimeError(
            "Refusing to write an empty or identity-invalid resume fragment"
        )
    if state is None:
        state, _ = _open_resume_fragments(
            root,
            manifest_path,
            part_hash=part_hash,
            expected_study_ids=expected_study_ids,
            force=False,
            execution_id=execution_id,
        )
    if (
        state.root != Path(root)
        or state.manifest_path != Path(manifest_path)
        or state.part_hash != part_hash
        or state.expected_study_ids != frozenset(expected_study_ids)
    ):
        raise RuntimeError("Resume fragment state does not match this partition")
    fragment_ids = tuple(int(value) for value in frame["study_id"])
    if (
        not set(fragment_ids).issubset(expected_study_ids)
        or state.completed_study_ids.intersection(fragment_ids)
    ):
        raise RuntimeError("Resume fragment contains unexpected or duplicate studies")
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise RuntimeError(f"Unsafe resume fragment root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    sequence = len(state.fragments)
    fragment_token = content_hash(
        {
            "part_hash": part_hash,
            "execution_id": execution_id or "",
            "sequence": sequence,
            "study_ids": fragment_ids,
        }
    )[:16]
    relative_path = (
        f"fragment-{sequence:06d}-{fragment_token}.parquet"
    )
    data_path = root / relative_path
    atomic_write_parquet(frame, data_path)
    fingerprint = file_fingerprint(data_path, include_sha256=True)
    fragment = {
        "sequence": sequence,
        "relative_path": relative_path,
        "row_count": len(frame),
        "study_ids": list(fragment_ids),
        "columns": [str(column) for column in frame.columns],
        "parquet_size": fingerprint["size"],
        "parquet_sha256": fingerprint["sha256"],
    }
    fragments = state.fragments + (fragment,)
    completed = sorted(
        int(study_id)
        for item in fragments
        for study_id in item["study_ids"]
    )
    payload = {
        "checkpoint_version": FEATURE_RESUME_CHECKPOINT_VERSION,
        "part_hash": part_hash,
        "execution_id": execution_id or "",
        "fragment_count": len(fragments),
        "row_count": len(completed),
        "study_ids": completed,
        "fragments": list(fragments),
        "checkpointed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["checkpoint_hash"] = content_hash(payload)
    # Commit point: the fragment may exist before this replace, but resume
    # considers only files named by a valid manifest.
    atomic_write_json(manifest_path, payload)
    return _ResumeFragmentState(
        root=root,
        manifest_path=manifest_path,
        part_hash=part_hash,
        execution_id=execution_id or "",
        expected_study_ids=frozenset(expected_study_ids),
        fragments=fragments,
    )


def _remove_resume_fragments(root: Path, manifest_path: Path) -> None:
    """Remove only the exact job-owned fragment files after final commit."""
    if manifest_path.exists():
        manifest_path.unlink()
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Unsafe resume fragment cleanup target: {root}")
    resolved_root = root.resolve()
    if resolved_root.parent != root.parent.resolve():
        raise RuntimeError(f"Resume fragment root escaped split directory: {root}")
    for candidate in root.iterdir():
        if (
            candidate.is_file()
            and not candidate.is_symlink()
            and re.fullmatch(
                r"fragment-\d{6}-[0-9a-f]{16}\.parquet",
                candidate.name,
            )
            and candidate.resolve().parent == resolved_root
        ):
            candidate.unlink()
    try:
        root.rmdir()
    except OSError:
        # Unknown files are deliberately left in place instead of widening a
        # cleanup operation beyond the exact artifacts owned by this stage.
        pass


def _load_resume_checkpoint(
    parquet_path: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    expected_study_ids: set[int],
    force: bool = False,
    execution_id: Optional[str] = None,
) -> pd.DataFrame:
    return _load_resume_fragments(
        _compatibility_resume_fragment_root(parquet_path),
        manifest_path,
        part_hash=part_hash,
        expected_study_ids=expected_study_ids,
        force=force,
        execution_id=execution_id,
    )


def _write_resume_checkpoint(
    frame: pd.DataFrame,
    parquet_path: Path,
    manifest_path: Path,
    *,
    part_hash: str,
    execution_id: Optional[str] = None,
) -> None:
    if (
        "study_id" not in frame
        or frame["study_id"].isna().any()
        or frame["study_id"].duplicated().any()
    ):
        raise RuntimeError(
            "Refusing to write a resume checkpoint with invalid or duplicate "
            "study_id values"
        )
    expected = {int(value) for value in frame["study_id"]}
    root = _compatibility_resume_fragment_root(parquet_path)
    state, _ = _open_resume_fragments(
        root,
        manifest_path,
        part_hash=part_hash,
        expected_study_ids=expected,
        force=bool(execution_id),
        execution_id=execution_id,
    )
    pending = frame.loc[
        ~frame["study_id"].astype(int).isin(state.completed_study_ids)
    ].copy()
    if not pending.empty:
        _append_resume_fragment(
            pending,
            root,
            manifest_path,
            part_hash=part_hash,
            execution_id=execution_id,
            expected_study_ids=expected,
            state=state,
        )


def _compatibility_resume_fragment_root(parquet_path: Path) -> Path:
    return parquet_path.with_name(f"{parquet_path.name}.fragments")


def _partition_reusable_for_execution(
    manifest: Optional[dict],
    options: FeatureExtractionOptions,
) -> bool:
    if not options.force:
        return True
    return bool(
        manifest
        and options.execution_id
        and manifest.get("execution_id") == options.execution_id
    )


def _filesystem_change_time_ns(path: Path, *, stat_ctime_ns: int) -> int | None:
    if os.name != "nt":
        # POSIX ctime cannot be restored with os.utime and therefore acts as a
        # change token in addition to mtime.
        return int(stat_ctime_ns)
    handle = _WINDOWS_CREATE_FILE(
        str(path),
        0,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        return None
    try:
        info = _WindowsFileBasicInfo()
        if not _WINDOWS_GET_FILE_INFORMATION(
            handle,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            return None
        # FILE_BASIC_INFO uses 100 ns ticks. The epoch is immaterial because
        # the value is compared for exact equality only.
        change_time_ns = int(info.ChangeTime) * 100
        # Some non-NTFS/network filesystems report a zero placeholder rather
        # than an error.  That is not a mutation token and must never license
        # the persistent fast path.
        return change_time_ns if change_time_ns > 0 else None
    finally:
        _WINDOWS_CLOSE_HANDLE(handle)


def _file_state(path: Path) -> dict[str, int | None]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "change_time_ns": _filesystem_change_time_ns(
            path,
            stat_ctime_ns=int(stat.st_ctime_ns),
        ),
    }


def _json_int(value: object, default: int) -> int:
    """Read an integer JSON contract field without permissive coercion."""

    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else int(default)
    )


def _read_stable_json_descriptor(path: Path) -> tuple[dict, dict[str, object]]:
    """Read and fingerprint a small JSON file without accepting a torn read."""

    for _ in range(2):
        before = _file_state(path)
        raw = path.read_bytes()
        after = _file_state(path)
        if before != after or len(raw) != after["size"]:
            continue
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid JSON artifact: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"JSON artifact must contain an object: {path}")
        descriptor: dict[str, object] = dict(after)
        descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
        return payload, descriptor
    raise RuntimeError(f"Artifact changed while it was being read: {path}")


def _feature_attestation_binding(
    *,
    root: Path,
    index_payload: Mapping[str, object],
    index_descriptor: Mapping[str, object],
    feature_manifest_descriptor: Mapping[str, object],
) -> dict[str, object]:
    return {
        "root": str(root.resolve()),
        "complete_index": True,
        "stage_input_hash": index_payload.get("stage_input_hash"),
        "source_code_hash": index_payload.get("source_code_hash"),
        "panel_hash": index_payload.get("panel_hash"),
        "artifact_hash": index_payload.get("artifact_hash"),
        "partition_count": _json_int(index_payload.get("partition_count"), -1),
        "row_count": _json_int(index_payload.get("row_count"), -1),
        "partition_index": dict(index_descriptor),
        "feature_manifest": dict(feature_manifest_descriptor),
    }


def _valid_feature_verification_attestation(
    path: Path,
    *,
    root: Path,
    paths: Sequence[Path],
    indexed_entries: Mapping[str, Mapping[str, object]],
    index_payload: Mapping[str, object],
    index_descriptor: Mapping[str, object],
    feature_manifest_descriptor: Mapping[str, object],
) -> bool:
    """Validate the cross-process verification cache, failing closed.

    Large Parquet bytes and identity columns are skipped only when an earlier
    complete verification is bound to the exact immutable index/manifest and
    every partition plus sidecar still has the same nanosecond file identity.
    Partition sidecars are small, so their bytes remain SHA-verified on every
    process boundary.
    """

    try:
        attestation, _ = _read_stable_json_descriptor(path)
        if (
            attestation.get("schema_version")
            != FEATURE_VERIFICATION_ATTESTATION_VERSION
            or attestation.get("complete_index") is not True
        ):
            return False
        recorded_hash = attestation.get("attestation_hash")
        unsigned = dict(attestation)
        unsigned.pop("attestation_hash", None)
        if not isinstance(recorded_hash, str) or content_hash(unsigned) != recorded_hash:
            return False
        expected_binding = _feature_attestation_binding(
            root=root,
            index_payload=index_payload,
            index_descriptor=index_descriptor,
            feature_manifest_descriptor=feature_manifest_descriptor,
        )
        if attestation.get("binding") != expected_binding:
            return False

        expected_relative_paths = tuple(
            sorted(path.relative_to(root).as_posix() for path in paths)
        )
        if expected_relative_paths != tuple(sorted(indexed_entries)):
            return False
        recorded_files = attestation.get("files")
        if not isinstance(recorded_files, list):
            return False
        by_relative_path = {
            str(entry.get("relative_path")): entry
            for entry in recorded_files
            if isinstance(entry, dict)
        }
        if (
            len(by_relative_path) != len(recorded_files)
            or tuple(sorted(by_relative_path)) != expected_relative_paths
        ):
            return False

        for relative_path in expected_relative_paths:
            entry = by_relative_path[relative_path]
            partition_path = (root / relative_path).resolve()
            expected_manifest_path = partition_path.with_suffix(".json")
            expected_manifest_relative = expected_manifest_path.relative_to(
                root
            ).as_posix()
            if entry.get("manifest_relative_path") != expected_manifest_relative:
                return False
            parquet_state = entry.get("parquet")
            if not isinstance(parquet_state, dict):
                return False
            if not isinstance(parquet_state.get("change_time_ns"), int):
                return False
            if _file_state(partition_path) != parquet_state:
                return False
            indexed_entry = indexed_entries[relative_path]
            if parquet_state.get("size") != _json_int(
                indexed_entry.get("parquet_size"), -1
            ):
                return False

            partition_manifest, manifest_descriptor = (
                _read_stable_json_descriptor(expected_manifest_path)
            )
            if entry.get("manifest") != manifest_descriptor:
                return False
            if (
                partition_manifest.get("part_hash")
                != indexed_entry.get("part_hash")
                or partition_manifest.get("parquet_sha256")
                != indexed_entry.get("parquet_sha256")
                or _json_int(partition_manifest.get("parquet_size"), -1)
                != parquet_state["size"]
                or _json_int(partition_manifest.get("row_count"), -1)
                != _json_int(indexed_entry.get("row_count"), -2)
            ):
                return False
        return True
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def _write_feature_verification_attestation(
    path: Path,
    *,
    root: Path,
    index_payload: Mapping[str, object],
    index_descriptor: Mapping[str, object],
    feature_manifest_descriptor: Mapping[str, object],
    verified_file_entries: Sequence[Mapping[str, object]],
) -> None:
    """Atomically persist a cache only while every verified file is stable."""

    current_index, current_index_descriptor = _read_stable_json_descriptor(
        root / "partition_index.json"
    )
    _, current_feature_manifest_descriptor = _read_stable_json_descriptor(
        root / "features.manifest.json"
    )
    if (
        current_index != dict(index_payload)
        or current_index_descriptor != dict(index_descriptor)
        or current_feature_manifest_descriptor
        != dict(feature_manifest_descriptor)
    ):
        raise RuntimeError(
            "Feature provenance changed during integrity verification"
        )

    stable_entries: list[dict[str, object]] = []
    for raw_entry in verified_file_entries:
        entry = dict(raw_entry)
        relative_path = str(entry["relative_path"])
        partition_path = (root / relative_path).resolve()
        parquet_state = entry.get("parquet")
        if (
            not isinstance(parquet_state, dict)
            or not isinstance(parquet_state.get("change_time_ns"), int)
        ):
            raise OSError(
                "The filesystem exposes no trustworthy change token for "
                f"{partition_path}"
            )
        if _file_state(partition_path) != parquet_state:
            raise RuntimeError(
                "Feature partition changed during integrity verification: "
                f"{partition_path}"
            )
        manifest_path = root / str(entry["manifest_relative_path"])
        _, current_manifest_descriptor = _read_stable_json_descriptor(
            manifest_path
        )
        if current_manifest_descriptor != entry.get("manifest"):
            raise RuntimeError(
                "Feature partition manifest changed during integrity "
                f"verification: {manifest_path}"
            )
        stable_entries.append(entry)

    unsigned: dict[str, object] = {
        "schema_version": FEATURE_VERIFICATION_ATTESTATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete_index": True,
        "binding": _feature_attestation_binding(
            root=root,
            index_payload=index_payload,
            index_descriptor=index_descriptor,
            feature_manifest_descriptor=feature_manifest_descriptor,
        ),
        "files": stable_entries,
    }
    attestation = dict(unsigned)
    attestation["attestation_hash"] = content_hash(unsigned)
    atomic_write_json(path, attestation)


def verify_feature_artifacts(
    feature_paths: Iterable[Path | str],
    *,
    require_complete_index: bool = True,
) -> dict:
    paths = sorted({Path(path).resolve() for path in feature_paths})
    if not paths:
        raise ValueError("No feature files were supplied for integrity verification")
    roots = {path.parent.parent for path in paths}
    if len(roots) != 1:
        raise RuntimeError(
            "Feature integrity verification requires one extraction root; "
            f"got {sorted(str(root) for root in roots)}"
        )
    root = next(iter(roots))
    index_path = root / "partition_index.json"
    manifest_path = root / "features.manifest.json"
    if not index_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Feature provenance is incomplete. Expected both "
            f"{index_path} and {manifest_path}."
        )
    payload, index_descriptor = _read_stable_json_descriptor(index_path)
    _, feature_manifest_descriptor = _read_stable_json_descriptor(
        manifest_path
    )
    stage_input_payload = payload.get("stage_input_payload")
    if (
        not isinstance(stage_input_payload, dict)
        or content_hash(stage_input_payload) != payload.get("stage_input_hash")
    ):
        raise RuntimeError(
            "Feature artifact index has no valid shared input contract. "
            "Re-run feature extraction."
        )
    source_code_hash = payload.get("source_code_hash")
    if not source_code_hash:
        raise RuntimeError(
            "Feature artifact index predates source-code provenance tracking. "
            "Re-run feature extraction."
        )
    current_code_hash = feature_implementation_hash()
    if source_code_hash != current_code_hash:
        raise RuntimeError(
            "Feature artifacts were produced by a different extraction code "
            "state. Re-run feature extraction before validation."
        )
    indexed_entries = {
        str(entry["relative_path"]): entry
        for entry in payload.get("partitions", [])
    }
    indexed_paths = {
        (root / relative_path).resolve()
        for relative_path in indexed_entries
    }
    selected_paths = set(paths)
    unindexed = selected_paths - indexed_paths
    if unindexed:
        raise RuntimeError(
            "Feature files are not part of the extraction artifact index: "
            f"{sorted(str(path) for path in unindexed)[:5]}"
        )
    if require_complete_index and selected_paths != indexed_paths:
        missing = indexed_paths - selected_paths
        raise RuntimeError(
            "Feature selection does not cover the complete frozen extraction "
            f"artifact; {len(missing):,} indexed partition(s) are missing."
        )

    selected_entries = [
        indexed_entries[path.relative_to(root).as_posix()] for path in paths
    ]
    total_rows = sum(int(entry["row_count"]) for entry in selected_entries)
    selected_hash = content_hash(selected_entries)
    complete_index = selected_paths == indexed_paths
    if require_complete_index and selected_hash != payload.get("artifact_hash"):
        raise RuntimeError(
            "Feature artifact index hash does not match its partition entries"
        )

    attestation_path = root / FEATURE_VERIFICATION_ATTESTATION_NAME
    if require_complete_index and complete_index:
        attested = _valid_feature_verification_attestation(
            attestation_path,
            root=root,
            paths=paths,
            indexed_entries=indexed_entries,
            index_payload=payload,
            index_descriptor=index_descriptor,
            feature_manifest_descriptor=feature_manifest_descriptor,
        )
        if attested:
            return _feature_integrity_result(
                root=root,
                manifest_path=manifest_path,
                index_path=index_path,
                payload=payload,
                source_code_hash=str(source_code_hash),
                selected_hash=selected_hash,
                partition_count=len(paths),
                row_count=total_rows,
                complete_index=True,
                verification_mode="persistent_attestation",
                attestation_path=attestation_path,
                attestation_status="verified",
            )

    identity_frames: list[pd.DataFrame] = []
    verified_file_entries: list[dict[str, object]] = []
    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        entry = indexed_entries[relative_path]
        partition_manifest_path = path.with_suffix(".json")
        try:
            partition_manifest, partition_manifest_descriptor = (
                _read_stable_json_descriptor(partition_manifest_path)
            )
        except (FileNotFoundError, RuntimeError, OSError, ValueError):
            raise FileNotFoundError(
                f"Feature partition manifest is missing or invalid: "
                f"{partition_manifest_path}"
            ) from None
        required_manifest_fields = {
            "part_hash",
            "row_count",
            "parquet_size",
            "parquet_sha256",
        }
        if not required_manifest_fields <= set(partition_manifest):
            raise FileNotFoundError(
                f"Feature partition manifest is missing or invalid: "
                f"{partition_manifest_path}"
            )
        if entry.get("part_hash") != partition_manifest.get("part_hash"):
            raise RuntimeError(
                f"Feature partition manifest differs from its index: {path}"
            )
        if not _partition_file_matches(path, partition_manifest):
            raise RuntimeError(
                f"Feature partition failed SHA-256/size verification: {path}"
            )
        if entry.get("parquet_sha256") != partition_manifest.get(
            "parquet_sha256"
        ):
            raise RuntimeError(
                f"Feature partition checksum differs from its index: {path}"
            )
        parquet_descriptor = _file_state(path)
        identity = _read_single_parquet(
            path,
            columns=["subject_id", "study_id", "split"],
        )
        if _file_state(path) != parquet_descriptor:
            raise RuntimeError(
                f"Feature partition changed during integrity verification: {path}"
            )
        _assert_exact_partition_identity(
            identity,
            expected_study_ids=partition_manifest.get("study_ids", ()),
            context=str(path),
        )
        identity_frames.append(identity)
        verified_file_entries.append(
            {
                "relative_path": relative_path,
                "parquet": parquet_descriptor,
                "manifest_relative_path": (
                    partition_manifest_path.relative_to(root).as_posix()
                ),
                "manifest": partition_manifest_descriptor,
            }
        )

    identities = pd.concat(identity_frames, ignore_index=True)
    if identities["study_id"].duplicated().any():
        examples = sorted(
            identities.loc[
                identities["study_id"].duplicated(keep=False),
                "study_id",
            ].astype(int).unique()
        )[:10]
        raise RuntimeError(
            "Feature artifact contains duplicate study_id values across "
            f"partitions: {examples}"
        )

    attestation_status = "not_applicable"
    if require_complete_index and complete_index:
        try:
            _write_feature_verification_attestation(
                attestation_path,
                root=root,
                index_payload=payload,
                index_descriptor=index_descriptor,
                feature_manifest_descriptor=feature_manifest_descriptor,
                verified_file_entries=verified_file_entries,
            )
            attestation_status = "written"
        except OSError:
            # The attestation is an optional performance cache. A verified
            # artifact remains valid when the cache cannot be persisted.
            attestation_status = "write_failed"

    return _feature_integrity_result(
        root=root,
        manifest_path=manifest_path,
        index_path=index_path,
        payload=payload,
        source_code_hash=str(source_code_hash),
        selected_hash=selected_hash,
        partition_count=len(paths),
        row_count=total_rows,
        complete_index=complete_index,
        verification_mode="full",
        attestation_path=attestation_path,
        attestation_status=attestation_status,
    )


def _feature_integrity_result(
    *,
    root: Path,
    manifest_path: Path,
    index_path: Path,
    payload: Mapping[str, object],
    source_code_hash: str,
    selected_hash: str,
    partition_count: int,
    row_count: int,
    complete_index: bool,
    verification_mode: str,
    attestation_path: Path,
    attestation_status: str,
) -> dict:
    return {
        "status": "verified",
        "root": str(root),
        "manifest_path": str(manifest_path.resolve()),
        "index_path": str(index_path.resolve()),
        "stage_input_hash": payload.get("stage_input_hash"),
        "stage_input_payload": payload.get("stage_input_payload"),
        "source_code_hash": source_code_hash,
        "panel_hash": payload.get("panel_hash"),
        "artifact_hash": payload.get("artifact_hash"),
        "selected_artifact_hash": selected_hash,
        "partition_count": int(partition_count),
        "row_count": int(row_count),
        "complete_index": bool(complete_index),
        "verification_mode": verification_mode,
        "verification_attestation_path": str(attestation_path.resolve()),
        "verification_attestation_status": attestation_status,
    }


def _feature_partition_paths(output_paths: Iterable[str | Path]) -> list[Path]:
    """Separate partition data from JSON provenance in stage outputs."""
    return [Path(path) for path in output_paths if Path(path).suffix == ".parquet"]


def _write_feature_artifact_index(
    output_root: Path,
    *,
    stage_input_hash: str,
    stage_input_payload: Mapping[str, object],
    source_code_hash: str,
    panel_hash: str,
    feature_ids: Iterable[str],
    parquet_paths: Iterable[Path],
    destination: Path,
) -> dict:
    root = output_root.resolve()
    entries = []
    for path in sorted(Path(path).resolve() for path in parquet_paths):
        if not path.is_relative_to(root):
            continue
        partition_manifest = _load_partition_manifest(path.with_suffix(".json"))
        if partition_manifest is None or not _partition_file_matches(
            path,
            partition_manifest,
        ):
            raise RuntimeError(
                f"Cannot index incomplete or corrupt feature partition: {path}"
            )
        identity = _read_single_parquet(path, columns=["study_id"])
        _assert_exact_partition_identity(
            identity,
            expected_study_ids=partition_manifest.get("study_ids", ()),
            context=str(path),
        )
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "part_hash": partition_manifest["part_hash"],
                "row_count": int(partition_manifest["row_count"]),
                "study_count": identity["study_id"].nunique(),
                "parquet_size": int(partition_manifest["parquet_size"]),
                "parquet_sha256": partition_manifest["parquet_sha256"],
            }
        )
    payload = {
        "index_version": FEATURE_ARTIFACT_INDEX_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage_input_hash": stage_input_hash,
        "stage_input_payload": dict(stage_input_payload),
        "source_code_hash": source_code_hash,
        "panel_hash": panel_hash,
        "feature_ids_hash": content_hash(tuple(feature_ids)),
        "partition_count": len(entries),
        "row_count": sum(entry["row_count"] for entry in entries),
        "artifact_hash": content_hash(entries),
        "partitions": entries,
    }
    atomic_write_json(destination, payload)
    return payload


def _assert_exact_partition_identity(
    frame: pd.DataFrame,
    *,
    expected_study_ids: Iterable[int],
    context: str,
) -> None:
    if "study_id" not in frame:
        raise RuntimeError(f"Feature {context} has no study_id column")
    expected = tuple(int(value) for value in expected_study_ids)
    if len(expected) != len(set(expected)):
        raise RuntimeError(
            f"Expected identity list for feature {context} contains duplicates"
        )
    if frame["study_id"].isna().any():
        raise RuntimeError(f"Feature {context} contains missing study_id values")
    actual = tuple(int(value) for value in frame["study_id"])
    if len(actual) != len(set(actual)):
        duplicates = sorted(
            frame.loc[
                frame["study_id"].duplicated(keep=False),
                "study_id",
            ].astype(int).unique()
        )[:10]
        raise RuntimeError(
            f"Feature {context} contains duplicate study_id values: {duplicates}"
        )
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise RuntimeError(
            f"Feature {context} does not match its expected ECG identity set"
        )


def _read_single_parquet(
    path: Path,
    *,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    # A file handle prevents PyArrow from interpreting ``split=train`` in the
    # parent directory as a second Hive partition column.
    with Path(path).open("rb") as handle:
        return pd.read_parquet(handle, columns=columns)


def build_feature_task_config(
    config: MimicProjectConfig,
    *,
    vcg_feature_ids: Iterable[str],
    sensitivity_feature_ids: Iterable[str],
    classical_feature_ids: Iterable[str],
) -> dict:
    """The immutable per-record settings a feature worker is initialised with.

    Factored out so that a verification run can reproduce a stored partition
    with exactly the settings that produced it, rather than an approximation of
    them.  A drift here would make the comparison meaningless.
    """

    return {
        "vcg_feature_ids": list(vcg_feature_ids),
        "sensitivity_feature_ids": list(sensitivity_feature_ids),
        "classical_feature_ids": list(classical_feature_ids),
        "sensitivity_vcg_methods": tuple(config.sensitivity_vcg_methods),
        "powerline_hz": config.powerline_hz,
        "delineation_weights": {
            "neurokit": 0.4,
            "multilead": 0.35,
            "vcg": 0.25,
        },
        "delineation_conflict_threshold_ms": 20.0,
    }


def resolve_feature_panel_selection(
    config: MimicProjectConfig,
    options: "FeatureExtractionOptions",
) -> dict:
    """Return the panel-derived identity of an extraction without running it.

    Both the stage and the adoption check need the same feature ids, panel hash
    and split into VCG/classical/sensitivity markers.  Deriving them twice by
    hand is how the two would drift apart.
    """

    panel_registry_path = config.features_root / "feature_panels.json"
    panel_payload = freeze_panel_registry(panel_registry_path)
    panels = {panel.panel_id: panel for panel in default_feature_panels()}
    unknown = set(options.panel_ids) - set(panels)
    if unknown:
        raise ValueError(f"Unknown feature panels: {sorted(unknown)}")
    feature_ids: list[str] = []
    for panel_id in options.panel_ids:
        feature_ids.extend(panels[panel_id].feature_ids)
    feature_ids = list(dict.fromkeys(feature_ids))
    # ``classical_<x>_positive`` is written as a by-product of computing
    # ``classical_<x>``: one call to the criteria yields both the continuous
    # value and the rule's verdict. Now that the verdicts belong to a panel,
    # they appear in ``feature_ids`` -- but they must not be *asked for*, or
    # the extraction would look up a criterion named ``<x>_positive``, find
    # nothing, and write ``classical_<x>_positive_positive`` beside it.
    criterion_verdict_ids = [
        name
        for name in feature_ids
        if name.startswith("classical_") and name.endswith("_positive")
    ]
    vcg_feature_ids = [
        name for name in feature_ids if not name.startswith("classical_")
    ]
    classical_feature_ids = [
        name
        for name in feature_ids
        if name.startswith("classical_") and not name.endswith("_positive")
    ]
    missing_verdict_sources = sorted(
        name
        for name in criterion_verdict_ids
        if name.removesuffix("_positive") not in set(classical_feature_ids)
    )
    if missing_verdict_sources:
        raise ValueError(
            "Criterion verdicts were selected whose continuous criterion is "
            f"not extracted: {missing_verdict_sources}"
        )
    core_panel_features = set(panels["primary_core"].feature_ids)
    sensitivity_feature_ids = [
        name
        for name in vcg_feature_ids
        if name in core_panel_features
        and name not in TRANSFORM_INDEPENDENT_FEATURE_IDS
    ]
    return {
        "panel_payload": panel_payload,
        "panel_hash": str(panel_payload["panel_hash"]),
        "feature_ids": feature_ids,
        "vcg_feature_ids": vcg_feature_ids,
        "classical_feature_ids": classical_feature_ids,
        "criterion_verdict_ids": criterion_verdict_ids,
        "sensitivity_feature_ids": sensitivity_feature_ids,
    }


def _plausibility_report(
    output_root: Path,
    feature_ids: Sequence[str],
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Count the values that left the range their unit allows.

    A narrow second pass over the finished partitions rather than an
    accumulation during the run: partitions are reused and adopted, so an
    in-flight tally would only ever cover the ones this execution happened to
    recompute, and a coverage figure that changes with the resume history is
    not a check. Reading only the markers that carry a declared range keeps it
    to a few hundred of the 1.889 columns -- measured at 12,8 s for 129
    partitions, so about four minutes for the whole store against an
    extraction that runs for days.
    """

    import pyarrow.parquet as pq

    from .marker_plausibility import (
        check_frame,
        declared_ranges,
        marker_units,
        merge_reports,
    )

    ranges = declared_ranges(marker_units())
    wanted = [marker for marker in feature_ids if marker in ranges]
    if not wanted:
        return merge_reports([])
    paths = sorted(output_root.glob("split=*/part-*.parquet"))
    if not paths:
        return merge_reports([])
    reports: list[pd.DataFrame] = []
    for index, path in enumerate(paths):
        handle = pq.ParquetFile(path)
        present_names = set(handle.schema_arrow.names)
        present = [
            marker
            for marker in wanted
            if marker in present_names
        ]
        if not present:
            continue
        reports.append(
            check_frame(handle.read(columns=present).to_pandas(), ranges)
        )
        if progress and (index + 1) % 200 == 0:
            progress(
                f"Wertebereichspruefung {index + 1:,}/{len(paths):,} Partitionen"
            )
    return merge_reports(reports)


def _transform_sensitivity_report(
    output_root: Path,
    sensitivity_feature_ids: Sequence[str],
    sensitivity_methods: Sequence[str],
) -> pd.DataFrame:
    """Check that every sensitivity twin actually depends on the transform.

    ``TRANSFORM_INDEPENDENT_FEATURE_IDS`` is a measured list, and a measured
    list goes stale. A marker whose definition later stops touching the VCG
    vector would keep being computed twice and would keep reporting a
    correlation of exactly one as if that were evidence of robustness. So the
    twins are compared once at the end of the run and the answer is written
    down: identical means the transform never entered, not that the marker
    survived it.

    A partition-sized sample is enough to catch identity -- two columns that
    agree on ten thousand records to the last bit are the same computation --
    and it keeps this to one file read rather than the whole store.
    """

    import pyarrow.parquet as pq

    paths = sorted(output_root.glob("split=*/part-*.parquet"))
    rows: list[dict] = []
    if not paths or not sensitivity_feature_ids:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "method",
                "compared_values",
                "identical_values",
                "transform_dependent",
            ]
        )
    sample = paths[:: max(1, len(paths) // 24)][:24]
    available = set(pq.ParquetFile(sample[0]).schema_arrow.names)
    pairs = [
        (marker, method, f"sensitivity_{method}__{marker}")
        for marker in sensitivity_feature_ids
        for method in sensitivity_methods
        if marker in available and f"sensitivity_{method}__{marker}" in available
    ]
    if not pairs:
        return pd.DataFrame(
            columns=[
                "marker_id",
                "method",
                "compared_values",
                "identical_values",
                "transform_dependent",
            ]
        )
    columns = sorted({name for pair in pairs for name in (pair[0], pair[2])})
    frames = [
        pq.ParquetFile(path).read(columns=columns).to_pandas()
        for path in sample
    ]
    frame = pd.concat(frames, ignore_index=True)
    for marker, method, twin in pairs:
        base = pd.to_numeric(frame[marker], errors="coerce")
        other = pd.to_numeric(frame[twin], errors="coerce")
        both = base.notna() & other.notna()
        compared = int(both.sum())
        identical = int((base[both] == other[both]).sum())
        rows.append(
            {
                "marker_id": marker,
                "method": method,
                "compared_values": compared,
                "identical_values": identical,
                "transform_dependent": bool(compared and identical < compared),
            }
        )
    return pd.DataFrame(rows)


def feature_implementation_hash() -> str:
    """Hash the code that decides what a marker is.

    ``feature_extraction.py`` is deliberately absent.  It used to be listed, and
    because it also holds the stage plumbing -- partitioning, manifests, resume
    fragments, worker-pool sizing -- editing any of that moved this hash and
    invalidated every extracted recording, although no marker value changed.
    The record-level computation now lives in ``feature_record.py``, and that is
    what is hashed.
    """

    package_root = Path(__file__).resolve().parents[1]
    mimic_root = Path(__file__).resolve().parent
    dependency_files = (
        mimic_root / "classical.py",
        mimic_root / "lead_measurements.py",
        mimic_root / "feature_record.py",
        mimic_root / "panels.py",
        mimic_root / "report_parser.py",
        mimic_root / "signals.py",
    )
    return content_hash(
        {
            "mimic_dependencies": {
                path.name: file_sha256(path) for path in dependency_files
            },
            "core_tree": source_tree_hash(
                package_root / "core",
                patterns=("*.py",),
            ),
        }
    )


def _load_partition_manifest(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "part_hash",
        "row_count",
        "parquet_size",
        "parquet_sha256",
    }
    return payload if required <= set(payload) else None


_VERIFIED_PARTITION_SHA_CACHE: set[
    tuple[str, int, int, int, int, str]
] = set()


def _partition_file_matches(path: Path, payload: dict) -> bool:
    try:
        before = _file_state(path)
        expected_size = int(payload["parquet_size"])
        expected_sha = str(payload["parquet_sha256"])
        if before["size"] != expected_size:
            return False
        change_time_ns = before.get("change_time_ns")
        cache_safe = isinstance(change_time_ns, int)
        size = before["size"]
        mtime_ns = before["mtime_ns"]
        ctime_ns = before["ctime_ns"]
        if not all(
            isinstance(value, int)
            for value in (size, mtime_ns, ctime_ns)
        ):
            return False
        cache_key = (
            str(path.resolve()),
            cast(int, size),
            cast(int, mtime_ns),
            cast(int, ctime_ns),
            int(change_time_ns or 0),
            expected_sha,
        )
        if cache_safe and cache_key in _VERIFIED_PARTITION_SHA_CACHE:
            return True
        if file_sha256(path) != expected_sha:
            return False
        after = _file_state(path)
        if after != before:
            return False
        # Index creation and verification can revisit a partition in one
        # process. Re-reading every byte adds no integrity information while
        # the nanosecond file identity tuple is unchanged.
        if cache_safe:
            _VERIFIED_PARTITION_SHA_CACHE.add(cache_key)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False
