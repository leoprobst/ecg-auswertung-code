from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import MimicProjectConfig
from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    file_fingerprint,
    file_sha256,
)


PHENOTYPE_MAPPING_VERSION = "phecodex-commit-6c5c6f7+ccsr-2026.1"


@dataclass(frozen=True)
class PhenotypeResources:
    phecodex_mapping_path: Path
    phecodex_definitions_path: Path
    phecodex_hierarchy_path: Path
    ccsr_mapping_path: Path
    manifest_path: Path
    resource_hash: str


def package_resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def prepare_phenotype_resources(
    config: MimicProjectConfig,
    *,
    force: bool = False,
) -> PhenotypeResources:
    source_root = package_resource_root()
    phecode_root = source_root / "phecodex"
    ccsr_root = source_root / "ccsr"
    phecode_source = _read_source_manifest(phecode_root / "SOURCE.json")
    ccsr_source = _read_source_manifest(ccsr_root / "SOURCE.json")
    _verify_pinned_files(phecode_root, phecode_source)
    _verify_pinned_files(ccsr_root, ccsr_source)
    resource_hash = content_hash(
        {
            "mapping_version": PHENOTYPE_MAPPING_VERSION,
            "phecodex": phecode_source,
            "ccsr": ccsr_source,
        }
    )
    output_root = config.resources_root / "phenotypes" / resource_hash[:16]
    mapping_path = output_root / "phecodex_mapping.parquet"
    definitions_path = output_root / "phecodex_definitions.parquet"
    hierarchy_path = output_root / "phecodex_hierarchy.parquet"
    ccsr_path = output_root / "ccsr_mapping.parquet"
    manifest_path = output_root / "resource_manifest.json"
    expected = (mapping_path, definitions_path, hierarchy_path, ccsr_path)
    if (
        not force
        and manifest_path.is_file()
        and all(path.is_file() for path in expected)
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("resource_hash") == resource_hash
            and _derived_outputs_match(expected, payload)
        ):
            return PhenotypeResources(
                mapping_path,
                definitions_path,
                hierarchy_path,
                ccsr_path,
                manifest_path,
                resource_hash,
            )

    output_root.mkdir(parents=True, exist_ok=True)
    flat = pd.read_csv(phecode_root / "phecodeX_R_map.csv", dtype=str)
    flat["icd_code"] = flat["code"].map(_normalize_icd)
    flat["icd_version"] = (
        flat["vocabulary_id"].str.upper().map(
            {"ICD9CM": 9, "ICD10CM": 10}
        )
    )
    flat = flat.rename(columns={"phecode": "source_phenotype_id"})

    hierarchy = pd.read_csv(
        phecode_root / "phecodeX_R_rollup_map.csv",
        dtype=str,
    ).rename(
        columns={
            "code": "source_phenotype_id",
            "phecode_unrolled": "phenotype_id",
        }
    )
    expanded = flat.merge(
        hierarchy,
        on="source_phenotype_id",
        how="left",
        validate="many_to_many",
    )
    expanded["phenotype_id"] = expanded["phenotype_id"].fillna(
        expanded["source_phenotype_id"]
    )
    mapping = (
        expanded[
            [
                "icd_version",
                "icd_code",
                "phenotype_id",
                "source_phenotype_id",
            ]
        ]
        .dropna(subset=["icd_version", "icd_code", "phenotype_id"])
        .drop_duplicates()
        .sort_values(["icd_version", "icd_code", "phenotype_id"])
        .reset_index(drop=True)
    )
    mapping["icd_version"] = mapping["icd_version"].astype("int8")

    labels = pd.read_csv(
        phecode_root / "phecodeX_R_labels.csv",
        dtype={
            "phenotype": str,
            "description": str,
            "group": str,
            "color": str,
        },
    ).rename(
        columns={
            "phenotype": "phenotype_id",
            "description": "display_name",
            "group": "body_system",
        }
    )
    sex = pd.read_csv(phecode_root / "phecodeX_R_sex.csv", dtype=str)
    sex = sex.rename(columns={"phecode": "phenotype_id"})
    definitions = labels.merge(sex, on="phenotype_id", how="left")
    definitions["mapping_system"] = "PhecodeX"
    definitions["mapping_version"] = PHENOTYPE_MAPPING_VERSION
    definitions = definitions.sort_values("phenotype_id").reset_index(drop=True)

    hierarchy_output = hierarchy.copy()
    hierarchy_output["is_self"] = (
        hierarchy_output["source_phenotype_id"]
        == hierarchy_output["phenotype_id"]
    )
    hierarchy_output = hierarchy_output.drop_duplicates().sort_values(
        ["source_phenotype_id", "phenotype_id"]
    )

    ccsr_source_path = (
        ccsr_root / "DXCCSR-v2026-1" / "DXCCSR_v2026-1.csv"
    )
    ccsr_raw = pd.read_csv(ccsr_source_path, dtype=str)
    ccsr_raw.columns = [_clean_ccsr_text(column) for column in ccsr_raw.columns]
    code_column = "ICD-10-CM CODE"
    category_columns = [
        column
        for column in ccsr_raw
        if column.startswith("CCSR CATEGORY ")
        and not column.endswith(" DESCRIPTION")
    ]
    ccsr_rows = []
    for category_column in category_columns:
        suffix = category_column.removeprefix("CCSR CATEGORY ")
        description_column = f"CCSR CATEGORY {suffix} DESCRIPTION"
        part = ccsr_raw[[code_column, category_column, description_column]].copy()
        part.columns = ["icd_code", "ccsr_category", "ccsr_description"]
        ccsr_rows.append(part)
    ccsr_mapping = pd.concat(ccsr_rows, ignore_index=True)
    for column in ccsr_mapping:
        ccsr_mapping[column] = ccsr_mapping[column].map(_clean_ccsr_text)
    ccsr_mapping["icd_code"] = ccsr_mapping["icd_code"].map(_normalize_icd)
    ccsr_mapping = (
        ccsr_mapping.loc[
            ccsr_mapping["icd_code"].ne("")
            & ccsr_mapping["ccsr_category"].ne("")
        ]
        .drop_duplicates()
        .sort_values(["icd_code", "ccsr_category"])
        .reset_index(drop=True)
    )
    ccsr_mapping["body_system_code"] = ccsr_mapping[
        "ccsr_category"
    ].str[:3]
    ccsr_mapping["mapping_version"] = "2026.1"

    atomic_write_parquet(mapping, mapping_path)
    atomic_write_parquet(definitions, definitions_path)
    atomic_write_parquet(hierarchy_output, hierarchy_path)
    atomic_write_parquet(ccsr_mapping, ccsr_path)
    output_fingerprints = {
        path.name: {
            "size": fingerprint["size"],
            "sha256": fingerprint["sha256"],
        }
        for path in expected
        for fingerprint in (
            file_fingerprint(path, include_sha256=True),
        )
    }
    atomic_write_json(
        manifest_path,
        {
            "mapping_version": PHENOTYPE_MAPPING_VERSION,
            "resource_hash": resource_hash,
            "phecodex_source": phecode_source,
            "ccsr_source": ccsr_source,
            "row_counts": {
                "phecodex_mapping": len(mapping),
                "phecodex_definitions": len(definitions),
                "phecodex_hierarchy": len(hierarchy_output),
                "ccsr_mapping": len(ccsr_mapping),
            },
            "outputs": [str(path.resolve()) for path in expected],
            "output_fingerprints": output_fingerprints,
        },
    )
    return PhenotypeResources(
        mapping_path,
        definitions_path,
        hierarchy_path,
        ccsr_path,
        manifest_path,
        resource_hash,
    )


def _read_source_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Pinned phenotype source manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_pinned_files(root: Path, payload: dict) -> None:
    for relative_path, expected_hash in payload.get("files", {}).items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Pinned phenotype resource missing: {path}")
        actual = file_sha256(path)
        if actual.lower() != str(expected_hash).lower():
            raise RuntimeError(
                f"Pinned phenotype resource hash mismatch: {path}"
            )


def _derived_outputs_match(paths: tuple[Path, ...], payload: dict) -> bool:
    fingerprints = payload.get("output_fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    for path in paths:
        expected = fingerprints.get(path.name)
        if not isinstance(expected, dict):
            return False
        try:
            actual = file_fingerprint(path, include_sha256=True)
        except OSError:
            return False
        if (
            int(expected.get("size", -1)) != int(actual["size"])
            or str(expected.get("sha256", "")).lower()
            != str(actual["sha256"]).lower()
        ):
            return False
    return True


def _normalize_icd(value: object) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .upper()
        .replace(".", "")
        .replace("'", "")
        .replace(" ", "")
    )


def _clean_ccsr_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().strip("'").strip()
