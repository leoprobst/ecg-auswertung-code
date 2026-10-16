from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from vcg_biomarker.mimic import services, workflow
from vcg_biomarker.mimic.config import MimicProjectConfig
from vcg_biomarker.mimic.gwas_feasibility import (
    GWAS_REQUIRED_COLUMNS,
    run_gwas_feasibility,
)


def _config(tmp_path: Path) -> MimicProjectConfig:
    config = MimicProjectConfig(
        mimic_iv_root=tmp_path / "mimic-iv",
        mimic_ed_root=tmp_path / "mimic-ed",
        mimic_ecg_root=tmp_path / "mimic-ecg",
        echo_structured_path=tmp_path / "echo.csv",
        echo_study_list_path=tmp_path / "echo-study.csv",
        echo_record_list_path=tmp_path / "echo-record.csv",
        work_root=tmp_path / "work",
    )
    config.ensure_directories()
    return config


def _write_summary(path: Path, *, p_value: str = "0.01", rows: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "variant_id": "rs123",
        "p_value": p_value,
        "chromosome": "1",
        "base_pair_location": "123456",
        "effect_allele": "A",
        "other_allele": "G",
        "effect_allele_frequency": "0.42",
        "beta": "0.12",
        "standard_error": "0.03",
    }
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(GWAS_REQUIRED_COLUMNS) + "\n")
        for _ in range(rows):
            handle.write(
                "\t".join(values[column] for column in GWAS_REQUIRED_COLUMNS)
                + "\n"
            )


def test_gwas_feasibility_inventories_both_angle_studies_without_causal_claims(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "gwas-summary-statistics"
    _write_summary(source / "GCST90246318_spatial_qrst_angle_multi.tsv.gz")
    _write_summary(source / "GCST90246320_spatial_qrst_angle_EUR.tsv.gz")

    manifest = run_gwas_feasibility(config)

    assert manifest.status == "complete"
    assert manifest.row_counts == {"summary_files": 2}
    assert len(manifest.output_integrity) == 2
    inventory = pd.read_parquet(
        config.work_root / "genetics" / "gwas_source_inventory.parquet"
    )
    assert set(inventory["status"]) == {"ready"}
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    summary = json.loads(
        (config.work_root / "genetics" / "gwas_feasibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["source_inventory_complete"] is True
    assert summary["mendelian_randomization_status"] == "not_estimable"
    assert summary["polygenic_score_status"] == "not_estimable"
    assert summary["causal_claims_allowed"] is False
    assert summary["individual_genotypes_available"] is False
    assert summary["independent_outcome_gwas_available"] is False
    assert summary["ld_reference_available"] is False


def test_gwas_feasibility_rejects_invalid_domains_and_header_only_files(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source = tmp_path / "gwas-summary-statistics"
    _write_summary(
        source / "GCST90246318_spatial_qrst_angle_multi.tsv.gz",
        p_value="1.5",
    )
    _write_summary(
        source / "GCST90246320_spatial_qrst_angle_EUR.tsv.gz",
        rows=0,
    )

    run_gwas_feasibility(config)

    inventory = pd.read_parquet(
        config.work_root / "genetics" / "gwas_source_inventory.parquet"
    ).set_index("accession")
    assert inventory.loc["GCST90246318", "reason"] == "sample_value_domain_invalid"
    assert inventory.loc["GCST90246320", "reason"] == "no_data_rows"
    summary = json.loads(
        (config.work_root / "genetics" / "gwas_feasibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["source_inventory_complete"] is False
    assert summary["n_invalid_files"] == 2


def test_absent_gwas_root_is_a_complete_nonblocking_inventory(tmp_path: Path) -> None:
    config = _config(tmp_path)

    manifest = run_gwas_feasibility(config)

    assert manifest.status == "complete"
    assert manifest.row_counts == {"summary_files": 0}
    inventory = pd.read_parquet(
        config.work_root / "genetics" / "gwas_source_inventory.parquet"
    )
    assert inventory.empty
    summary = json.loads(
        (config.work_root / "genetics" / "gwas_feasibility.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["root_present"] is False
    assert summary["source_inventory_complete"] is False


def test_gwas_feasibility_is_reachable_once_before_signal_processing() -> None:
    assert "genetic-feasibility" in services.MANAGED_SERVICE_STAGE_IDS
    assert workflow.GUIDED_PREPARATION_STAGES.count("genetic-feasibility") == 1
    assert workflow.GUIDED_PREPARATION_STAGES.index("genetic-feasibility") < (
        workflow.GUIDED_PREPARATION_STAGES.index("ingest")
    )
