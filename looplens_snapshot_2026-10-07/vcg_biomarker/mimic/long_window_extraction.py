"""Fixed-window measurement using the existing external signal pipeline.

Explicit WFDB input, all windows of the same patient in the same split. No
diagnostic reference is inferred from file names or morphology. INCART beat
annotations are optional context, never manual P/T-boundary references.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .code15_cohort import Code15Record
from .population_cohorts import _analyze_record
from .pre_run_extensions import ExtensionPolicy, long_window_summary
from .storage import file_sha256

VERSION = "1.0.0"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def fixed_windows(
    signal,
    fs,
    *,
    source,
    subject_id,
    record_id,
    split,
    markers,
    powerline_hz,
    workers=1,
    annotation_samples=None,
    annotation_symbols=None,
):
    """At most two parallel windows; no nested BLAS pools and no full-record copies."""
    if split not in ("train", "validation") or not source or subject_id is None:
        raise ValueError("unknown_patient_or_nondevelopment_split")
    if type(workers) is not int or not 1 <= workers <= 2 or not markers:
        raise ValueError("invalid_window_workers_or_empty_markers")
    x = np.asarray(signal)
    if (
        x.ndim != 2
        or x.shape[1] != 12
        or not np.isfinite(fs)
        or fs <= 0
        or float(fs) != int(fs)
    ):
        raise ValueError("requires_twelve_leads_integer_sampling_rate")
    samples = None if annotation_samples is None else np.asarray(annotation_samples)
    symbols = None if annotation_symbols is None else np.asarray(annotation_symbols)
    if samples is not None and (
        samples.ndim != 1
        or symbols is None
        or len(samples) != len(symbols)
        or not np.issubdtype(samples.dtype, np.integer)
        or (samples < 0).any()
        or (samples >= len(x)).any()
        or (np.diff(samples) < 0).any()
    ):
        raise ValueError("invalid_beat_annotations")
    window = int(10 * fs)
    starts = range(0, len(x) - window + 1, window)

    def calculate(start):
        record = Code15Record(
            start // window,
            subject_id,
            x[start : start + window],
            fs,
            fs,
            LEADS,
            0,
            0,
            source,
        )
        row = _analyze_record(
            record,
            {},
            tuple(markers),
            tuple(markers),
            powerline_hz=powerline_hz,
            minimum_signal_quality=0.7,
        )
        result = {
            "source": source,
            "subject_id": subject_id,
            "record_id": str(record_id),
            "study_id": f"{record_id}:{start}",
            "split": split,
            "start_s": start / fs,
            "duration_s": 10.0,
            "quality": row.get("signal_quality_score", np.nan),
            "extraction_status": row.get("extraction_status"),
            "extraction_reason": row.get("failure_message", ""),
            "detected_beats": row.get("n_beats"),
            **{
                marker: row.get(marker, np.nan)
                if row.get("extraction_status") == "ok"
                else np.nan
                for marker in markers
            },
        }
        if samples is not None:
            assert symbols is not None
            labels = symbols[(samples >= start) & (samples < start + window)]
            result.update(
                annotation_count=len(labels),
                ventricular_annotation_count=int(np.isin(labels, ["V", "E"]).sum()),
                supraventricular_annotation_count=int(
                    np.isin(labels, ["A", "a", "J", "S"]).sum()
                ),
                annotation_role="beat_class_context_not_manual_fiducials",
            )
        return result

    with threadpool_limits(limits=1):
        if workers == 1:
            rows = [calculate(start) for start in starts]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(calculate, starts))
    if not rows:
        raise ValueError("record_shorter_than_ten_seconds")
    table = pd.DataFrame(rows)
    summary = long_window_summary(table, markers, ExtensionPolicy())
    return (
        table,
        summary,
        {
            "version": VERSION,
            "windows": len(table),
            "trailing_samples_not_analyzed": len(x) % window,
            "workers": workers,
            "native_threads": 1,
            "clinical_claim_allowed": False,
            "independent_patients": 1,
        },
    )


def wfdb_windows(
    path,
    *,
    source,
    subject_id,
    record_id,
    split,
    markers,
    powerline_hz,
    workers=1,
    annotation_extension=None,
):
    """Read one verified local record; preserve physical gain and standard lead order."""
    import wfdb

    path = Path(path).resolve()
    if any(
        "locked" in part.lower() or part.lower() in {"test", "test_split"}
        for part in path.parts
    ):
        raise ValueError("locked_test_input_forbidden_before_read")
    if split not in ("train", "validation") or not source or subject_id is None:
        raise ValueError("nondevelopment_split")
    root = path.with_suffix("")
    header_path = root.with_suffix(".hea")
    hashes = {str(header_path): file_sha256(header_path)}
    header = wfdb.rdheader(str(root))
    files = [(root.parent / name).resolve() for name in set(header.file_name)]
    if annotation_extension is not None:
        if not annotation_extension.isalnum():
            raise ValueError("invalid_annotation_extension")
        files.append(root.with_suffix("." + annotation_extension))
    if any(not file.is_relative_to(root.parent) for file in files):
        raise ValueError("wfdb_dependency_outside_record_directory")
    hashes.update({str(file): file_sha256(file) for file in files})
    record = wfdb.rdrecord(str(path.with_suffix("")))
    if (
        len(record.sig_name) != 12
        or set(record.sig_name) != set(LEADS)
        or any(unit.lower() != "mv" for unit in record.units)
    ):
        raise ValueError("requires_standard_twelve_leads_in_millivolts")
    signal = record.p_signal[:, [record.sig_name.index(lead) for lead in LEADS]]
    annotations = (
        None
        if annotation_extension is None
        else wfdb.rdann(str(path.with_suffix("")), annotation_extension)
    )
    table, summary, metadata = fixed_windows(
        signal,
        record.fs,
        source=source,
        subject_id=subject_id,
        record_id=record_id,
        split=split,
        markers=markers,
        powerline_hz=powerline_hz,
        workers=workers,
        annotation_samples=None if annotations is None else annotations.sample,
        annotation_symbols=None if annotations is None else annotations.symbol,
    )
    if any(file_sha256(Path(name)) != digest for name, digest in hashes.items()):
        raise RuntimeError("waveform_input_changed_during_window_analysis")
    metadata["input_sha256"] = hashes
    return table, summary, metadata
