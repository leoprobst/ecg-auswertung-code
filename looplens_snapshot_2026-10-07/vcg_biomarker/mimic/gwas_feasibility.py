"""Inventory genetic summary statistics without manufacturing causal claims.

The local files are exposure GWAS for the spatial QRS-T angle.  They are
valuable evidence about genetic architecture, but MIMIC/CODE/SaMi carry no
matching individual genotypes and the folder contains no independent outcome
GWAS.  This stage records exactly that boundary so a future MR/PRS analysis
cannot start merely because two large ``.tsv.gz`` files exist.
"""

from __future__ import annotations

import gzip
import math
import re
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    capture_output_integrity,
    content_hash,
    file_fingerprint,
    scientific_config_hash,
)


GWAS_FEASIBILITY_VERSION = "1.0.0"
GWAS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "variant_id",
    "p_value",
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "effect_allele_frequency",
    "beta",
    "standard_error",
)
KNOWN_ANGLE_STUDIES = {
    "GCST90246318": {
        "trait": "spatial_qrst_angle",
        "ancestry": "multi_ancestry",
    },
    "GCST90246320": {
        "trait": "spatial_qrst_angle",
        "ancestry": "european",
    },
}
GWAS_PUBLICATION = {
    "doi": "10.1038/s41467-023-36997-w",
    "title": "Genetic architecture of the electrocardiographic spatial QRS-T angle",
    "catalog_accessions": tuple(KNOWN_ANGLE_STUDIES),
}


def gwas_summary_root(config: MimicProjectConfig) -> Path:
    """Resolve the optional colocated database without changing run identity."""

    return config.mimic_iv_root.parent / "gwas-summary-statistics"


def _inspect_summary_file(
    path: Path,
    *,
    fingerprint: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    match = re.search(r"GCST\d+", path.name.upper())
    accession = match.group(0) if match else ""
    known = KNOWN_ANGLE_STUDIES.get(accession, {})
    fingerprint = fingerprint or file_fingerprint(path, mode="sha256")
    record: dict[str, object] = {
        "accession": accession,
        "path": str(path.resolve()),
        "filename": path.name,
        "trait": str(known.get("trait", "unknown")),
        "ancestry": str(known.get("ancestry", "unknown")),
        "role": "exposure_gwas_summary_statistics",
        "status": "invalid",
        "reason": "",
        "size": int(fingerprint["size"]),
        "sha256": str(fingerprint["sha256"]),
    }
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            header = tuple(next(handle).rstrip("\r\n").split("\t"))
            sample = []
            for _, line in zip(range(8), handle):
                sample.append(line.rstrip("\r\n").split("\t"))
    except (OSError, EOFError, StopIteration, UnicodeError) as exc:
        record["reason"] = f"gzip_or_header_unreadable:{type(exc).__name__}"
        return record
    missing = sorted(set(GWAS_REQUIRED_COLUMNS) - set(header))
    record["columns"] = list(header)
    record["sample_rows_checked"] = len(sample)
    if missing:
        record["reason"] = "required_columns_missing:" + "|".join(missing)
        return record
    if any(len(row) != len(header) for row in sample):
        record["reason"] = "sample_row_width_mismatch"
        return record
    if not sample:
        record["reason"] = "no_data_rows"
        return record
    positions = {name: header.index(name) for name in GWAS_REQUIRED_COLUMNS}
    try:
        for row in sample:
            p_value = float(row[positions["p_value"]])
            base_pair = int(row[positions["base_pair_location"]])
            frequency = float(row[positions["effect_allele_frequency"]])
            beta = float(row[positions["beta"]])
            standard_error = float(row[positions["standard_error"]])
            if not (
                0.0 <= p_value <= 1.0
                and base_pair > 0
                and 0.0 <= frequency <= 1.0
                and math.isfinite(beta)
                and math.isfinite(standard_error)
                and standard_error > 0.0
                and row[positions["chromosome"]]
                and row[positions["effect_allele"]]
                and row[positions["other_allele"]]
            ):
                raise ValueError("summary statistic outside its valid domain")
    except (TypeError, ValueError, OverflowError):
        record["reason"] = "sample_value_domain_invalid"
        return record
    record.update(
        {
            "status": "ready",
            "reason": "",
        }
    )
    return record


def run_gwas_feasibility(
    config: MimicProjectConfig,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    """Write a machine-readable capability audit for the local GWAS files."""

    root = gwas_summary_root(config)
    output_root = config.work_root / "genetics"
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / "gwas_source_inventory.parquet"
    summary_path = output_root / "gwas_feasibility.json"
    manifest_path = output_root / "gwas_feasibility.manifest.json"

    files = sorted(root.glob("*.tsv.gz")) if root.is_dir() else []
    source_metadata = [file_fingerprint(path, mode="sha256") for path in files]
    input_hash = content_hash(
        {
            "version": GWAS_FEASIBILITY_VERSION,
            "root": str(root.resolve()),
            "sources": source_metadata,
            "required_columns": GWAS_REQUIRED_COLUMNS,
            "known_studies": KNOWN_ANGLE_STUDIES,
        }
    )
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, GWAS_FEASIBILITY_VERSION):
            return previous

    records: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        if progress:
            progress(f"GWAS-Inventar {index}/{len(files)}: {path.name}")
        records.append(
            _inspect_summary_file(path, fingerprint=source_metadata[index - 1])
        )
    inventory = pd.DataFrame(
        records,
        columns=(
            "accession",
            "path",
            "filename",
            "trait",
            "ancestry",
            "role",
            "status",
            "reason",
            "size",
            "sha256",
            "columns",
            "sample_rows_checked",
        ),
    )
    ready_accessions = {
        str(row["accession"])
        for row in records
        if row.get("status") == "ready"
    }
    expected_accessions = set(KNOWN_ANGLE_STUDIES)
    sources_complete = expected_accessions <= ready_accessions
    summary = {
        "version": GWAS_FEASIBILITY_VERSION,
        "root": str(root.resolve()),
        "root_present": root.is_dir(),
        "n_summary_files": len(files),
        "ready_accessions": sorted(ready_accessions),
        "missing_expected_accessions": sorted(expected_accessions - ready_accessions),
        "source_inventory_complete": sources_complete,
        "all_discovered_files_valid": all(
            row.get("status") == "ready" for row in records
        ),
        "n_invalid_files": sum(
            row.get("status") != "ready" for row in records
        ),
        "available_trait": "spatial_qrst_angle" if sources_complete else None,
        "available_ancestry_arms": sorted(
            {
                str(row.get("ancestry"))
                for row in records
                if row.get("status") == "ready"
            }
        ),
        "individual_genotypes_available": False,
        "independent_outcome_gwas_available": False,
        "ld_reference_available": False,
        "mendelian_randomization_status": "not_estimable",
        "polygenic_score_status": "not_estimable",
        "causal_claims_allowed": False,
        "allowed_use": [
            "genetic_architecture_context_for_spatial_qrst_angle",
            "future_protocol_and_harmonization_planning",
        ],
        "blocked_use": [
            "mendelian_randomization_without_outcome_gwas",
            "polygenic_scoring_without_individual_genotypes_and_ld_reference",
            "causal_or_guideline_claims",
        ],
        "next_required_data": [
            "preselected_independent_outcome_gwas",
            "ancestry_matched_ld_reference",
            "or_individual_level_genotypes_in_an_external_ecg_cohort",
        ],
        "claim_scope": "feasibility_inventory_only",
        "publication": GWAS_PUBLICATION,
        "summary_statistics_format": "GWAS_Catalog_harmonised_style",
    }
    atomic_write_parquet(inventory, table_path)
    atomic_write_json(summary_path, summary)
    output_paths = (table_path, summary_path)
    manifest = StageManifest.complete(
        stage="genetic-feasibility",
        stage_version=GWAS_FEASIBILITY_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=tuple(str(path.resolve()) for path in output_paths),
        row_counts={"summary_files": len(inventory)},
        notes=(
            "separate_feasibility_arm=True",
            "mendelian_randomization_run=False",
            "causal_claims_allowed=False",
        ),
        output_integrity=capture_output_integrity(output_paths),
    )
    manifest.save(manifest_path)
    return manifest


__all__ = [
    "GWAS_FEASIBILITY_VERSION",
    "GWAS_REQUIRED_COLUMNS",
    "GWAS_PUBLICATION",
    "KNOWN_ANGLE_STUDIES",
    "gwas_summary_root",
    "run_gwas_feasibility",
]
