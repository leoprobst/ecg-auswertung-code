"""Prespecified descriptive baseline/beat-count QC, not hypothesis selection."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .storage import atomic_write_json, file_sha256

VERSION = "1.0.0"
PR = "qrs_t_area_angle_direct_8lead_deg"
PRE = "qrs_t_area_angle_direct_8lead_preqrs_deg"
COUNT = "direct8_preqrs_eligible_beat_count"
FRECHET = "qrs_t_frechet_beat_median"
MAD = "qrs_t_frechet_beat_mad"
FCOUNT = "qrs_t_frechet_eligible_beat_count"
COLUMNS = ("rhythm_context", "signal_quality_tier", PR, PRE, COUNT, FRECHET, MAD, FCOUNT)


def summarize(frame: pd.DataFrame) -> dict:
    missing = sorted(set(COLUMNS) - set(frame.columns))
    if missing:
        return {"version": VERSION, "status": "unavailable", "missing_columns": missing}
    data = frame[list(COLUMNS)].copy()
    for column in COLUMNS[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data[["rhythm_context", "signal_quality_tier"]] = data[["rhythm_context", "signal_quality_tier"]].fillna("unknown")
    records = []
    for (rhythm, quality), block in data.groupby(["rhythm_context", "signal_quality_tier"], sort=True):
        for minimum in (3, 6):
            direct = block.loc[block[COUNT].ge(minimum)]
            paired = direct[[PR, PRE]].dropna()
            differences = paired[PRE] - paired[PR]
            frechet = block.loc[block[FCOUNT].ge(minimum)]
            records.append({"rhythm": str(rhythm), "quality": str(quality), "minimum_beats": minimum,
                "n_records": len(block), "direct8_qualified": len(direct),
                "pr_available": int(direct[PR].notna().sum()), "preqrs_available": int(direct[PRE].notna().sum()),
                "paired_n": len(paired), "paired_pre_minus_pr_median_deg": float(differences.median()) if len(paired) else None,
                "frechet_qualified": len(frechet), "frechet_available": int(frechet[FRECHET].notna().sum()),
                "frechet_mad_median": float(frechet[MAD].median()) if frechet[MAD].notna().any() else None})
    inconsistent = int((data[FRECHET].notna() & (~data[FCOUNT].ge(3))).sum())
    return {"version": VERSION, "status": "complete" if not inconsistent else "inconsistent_beat_counts",
        "records": records, "inconsistent_frechet_counts": inconsistent,
        "unit": "ECG_records_not_independent_patients", "scope": "partition_descriptive_QC",
        "median_pooling_allowed": False, "mad_is_measurement_ci": False,
        "morphology_cluster_proves_homogeneity": False,
        "baseline_variability_adjustment": "not_evaluated_no_baseline_variance_input",
        "clinical_claim_allowed": False}


def write_partition_robustness(path: Path, *, frame: pd.DataFrame | None = None,
                               source_sha256: str | None = None) -> Path:
    if frame is None:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        columns = [column for column in COLUMNS if column in parquet.schema_arrow.names]
        frame = parquet.read(columns=columns).to_pandas()
    result = summarize(frame)
    result["source_sha256"] = source_sha256 or file_sha256(path)
    target = path.with_suffix(".robustness.json")
    atomic_write_json(target, result)
    return target
