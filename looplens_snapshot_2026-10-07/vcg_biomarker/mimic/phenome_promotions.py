from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .config import MimicProjectConfig
from .feature_roles import (
    METHODOLOGICAL_CONTROL_FEATURE_IDS,
    feature_analysis_role,
)
from .phenome import _promotion_candidates
from .storage import (
    StageManifest,
    atomic_write_json,
    atomic_write_parquet,
    content_hash,
    operational_config_hash,
    scientific_config_hash,
    file_content_fingerprint,
)


PHENOME_PROMOTION_REFRESH_VERSION = "1.2.0"
PROMOTION_ROLE_POLICY_VERSION = "candidate-biomarkers-only-v1.0.0"


def refresh_phenome_promotions(
    config: MimicProjectConfig,
    run_id: str,
    *,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> StageManifest:
    phenome_root = config.runs_root / run_id / "phenome"
    source_manifest_path = phenome_root / "phenome.manifest.json"
    replicated_path = phenome_root / "phenome_replicated_hits.parquet"
    adjusted_path = phenome_root / "phenome_adjusted_confirmation.parquet"
    output_path = phenome_root / "phenome_promotion_candidates.parquet"
    output_json_path = phenome_root / "promotion_candidates.json"
    refresh_manifest_path = (
        phenome_root / "promotion_refresh.manifest.json"
    )

    source_manifest = StageManifest.load(source_manifest_path)
    adjustment_required = _source_requires_adjustment(source_manifest)
    _verify_source_phenome_manifest(
        source_manifest,
        replicated_path,
        adjusted_path=adjusted_path,
        adjustment_required=adjustment_required,
    )
    snapshot_hash = _feature_snapshot_hash(source_manifest)
    input_hash = content_hash(
        _promotion_refresh_input_payload(
            source_manifest_path,
            replicated_path,
            adjusted_path=adjusted_path if adjusted_path.is_file() else None,
            snapshot_hash=snapshot_hash,
        )
    )
    config_hash = scientific_config_hash(config)

    if not force and refresh_manifest_path.is_file():
        previous = StageManifest.load(refresh_manifest_path)
        if previous.matches(
            input_hash,
            config_hash,
            PHENOME_PROMOTION_REFRESH_VERSION,
        ):
            try:
                verify_phenome_promotion_refresh(phenome_root.parent)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                pass
            else:
                return previous

    if progress:
        progress("PheWAS-Promotionen 1/3: replizierte Treffer prüfen")
    replicated = pd.read_parquet(replicated_path)
    _verify_replicated_schema(replicated)
    eligible = _adjustment_eligible_hits(
        replicated,
        adjusted_path if adjusted_path.is_file() else None,
    )
    excluded_top_roles = _excluded_top_candidate_roles(eligible)

    promotions = _promotion_candidates(
        eligible,
        snapshot_hash=snapshot_hash,
    )
    records = [candidate.to_dict() for candidate in promotions]
    output = pd.DataFrame(records)
    invalid_roles = _promotion_role_counts(output)
    if invalid_roles:
        raise RuntimeError(
            "Promotion refresh still contains non-biomarker features: "
            f"{invalid_roles}"
        )

    if progress:
        progress("PheWAS-Promotionen 2/3: Kandidaten atomar ersetzen")
    atomic_write_parquet(output, output_path)
    atomic_write_json(
        output_json_path,
        {
            "status": "draft_hypotheses_only",
            "locked_test_scanned": False,
            "promotion_refresh_version": (
                PHENOME_PROMOTION_REFRESH_VERSION
            ),
            "adjustment_required": adjustment_required,
            "adjustment_enforced": adjusted_path.is_file(),
            "legacy_unadjusted_source": not adjusted_path.is_file(),
            "promotion_refresh_input_hash": input_hash,
            "promotion_role_policy": _promotion_role_policy(),
            "source_phenome_input_hash": source_manifest.input_hash,
            "source_feature_snapshot_hash": snapshot_hash,
            "excluded_top_candidate_roles": excluded_top_roles,
            "candidates": records,
        },
    )

    manifest = StageManifest.complete(
        stage="refresh-phenome-promotions",
        stage_version=PHENOME_PROMOTION_REFRESH_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(
            str(output_path.resolve()),
            str(output_json_path.resolve()),
        ),
        row_counts={
            "promotion_candidates": len(output),
            "excluded_methodological_top_candidates": int(
                excluded_top_roles.get(
                    "methodological_qc_control",
                    0,
                )
            ),
            "excluded_classical_top_candidates": int(
                excluded_top_roles.get(
                    "classical_ecg_baseline",
                    0,
                )
            ),
            "invalid_output_feature_roles": 0,
        },
        notes=(
            f"operational_config_hash={operational_config_hash(config)}",
            "scope=development_only",
            "locked_test_scanned=false",
            f"adjustment_required={str(adjustment_required).lower()}",
            f"adjustment_enforced={str(adjusted_path.is_file()).lower()}",
            f"source_phenome_input_hash={source_manifest.input_hash}",
            f"feature_snapshot_hash={snapshot_hash}",
            f"role_policy={PROMOTION_ROLE_POLICY_VERSION}",
        ),
    )
    manifest.save(refresh_manifest_path)
    verification = verify_phenome_promotion_refresh(phenome_root.parent)
    if verification["valid"] is not True:
        raise RuntimeError(
            f"Promotion refresh verification failed: {verification}"
        )
    if progress:
        progress("PheWAS-Promotionen 3/3: Refresh-Manifest verifiziert")
    return manifest


def verify_phenome_promotion_refresh(run_root: Path) -> dict[str, Any]:
    phenome_root = Path(run_root) / "phenome"
    source_manifest_path = phenome_root / "phenome.manifest.json"
    replicated_path = phenome_root / "phenome_replicated_hits.parquet"
    adjusted_path = phenome_root / "phenome_adjusted_confirmation.parquet"
    output_path = phenome_root / "phenome_promotion_candidates.parquet"
    output_json_path = phenome_root / "promotion_candidates.json"
    refresh_manifest_path = (
        phenome_root / "promotion_refresh.manifest.json"
    )

    source_manifest = StageManifest.load(source_manifest_path)
    refresh_manifest = StageManifest.load(refresh_manifest_path)
    adjustment_required = _source_requires_adjustment(source_manifest)
    _verify_source_phenome_manifest(
        source_manifest,
        replicated_path,
        adjusted_path=adjusted_path,
        adjustment_required=adjustment_required,
    )
    snapshot_hash = _feature_snapshot_hash(source_manifest)
    expected_input_hash = content_hash(
        _promotion_refresh_input_payload(
            source_manifest_path,
            replicated_path,
            adjusted_path=adjusted_path if adjusted_path.is_file() else None,
            snapshot_hash=snapshot_hash,
        )
    )
    promotion_payload = json.loads(
        output_json_path.read_text(encoding="utf-8")
    )
    candidates = pd.read_parquet(output_path)
    role_counts = _promotion_role_counts(candidates)
    parquet_records = _candidate_feature_map(candidates)
    json_records = _candidate_feature_map(
        pd.DataFrame(promotion_payload.get("candidates", ()))
    )
    notes = {str(note).lower() for note in refresh_manifest.notes}
    required_outputs = {
        str(output_path.resolve()),
        str(output_json_path.resolve()),
    }

    valid = (
        source_manifest.status == "complete"
        and refresh_manifest.status == "complete"
        and refresh_manifest.stage == "refresh-phenome-promotions"
        and refresh_manifest.stage_version
        == PHENOME_PROMOTION_REFRESH_VERSION
        and refresh_manifest.input_hash == expected_input_hash
        and required_outputs
        == {str(Path(path).resolve()) for path in refresh_manifest.output_paths}
        and all(Path(path).is_file() for path in refresh_manifest.output_paths)
        and promotion_payload.get("locked_test_scanned") is False
        and promotion_payload.get("promotion_refresh_version")
        == PHENOME_PROMOTION_REFRESH_VERSION
        and promotion_payload.get("adjustment_required")
        is adjustment_required
        and promotion_payload.get("adjustment_enforced")
        is adjusted_path.is_file()
        and promotion_payload.get("promotion_refresh_input_hash")
        == refresh_manifest.input_hash
        and promotion_payload.get("promotion_role_policy")
        == _promotion_role_policy()
        and promotion_payload.get("source_phenome_input_hash")
        == source_manifest.input_hash
        and promotion_payload.get("source_feature_snapshot_hash")
        == snapshot_hash
        and "scope=development_only" in notes
        and "locked_test_scanned=false" in notes
        and not role_counts
        and parquet_records == json_records
        and int(
            refresh_manifest.row_counts.get(
                "promotion_candidates",
                -1,
            )
        )
        == len(candidates)
        and int(
            refresh_manifest.row_counts.get(
                "invalid_output_feature_roles",
                -1,
            )
        )
        == 0
    )
    return {
        "valid": bool(valid),
        "manifest_status": refresh_manifest.status,
        "stage_version": refresh_manifest.stage_version,
        "input_hash_matches": (
            refresh_manifest.input_hash == expected_input_hash
        ),
        "candidate_count": len(candidates),
        "invalid_feature_role_counts": role_counts,
        "excluded_methodological_top_candidates": int(
            refresh_manifest.row_counts.get(
                "excluded_methodological_top_candidates",
                0,
            )
        ),
        "locked_test_scanned": promotion_payload.get(
            "locked_test_scanned"
        ),
        "adjustment_required": adjustment_required,
        "adjustment_enforced": adjusted_path.is_file(),
        "manifest_path": str(refresh_manifest_path.resolve()),
    }


def _promotion_refresh_input_payload(
    source_manifest_path: Path,
    replicated_path: Path,
    *,
    adjusted_path: Optional[Path] = None,
    snapshot_hash: str,
) -> dict[str, Any]:
    return {
        "stage_version": PHENOME_PROMOTION_REFRESH_VERSION,
        "source_manifest": file_content_fingerprint(
            source_manifest_path
        ),
        "replicated_hits": file_content_fingerprint(replicated_path),
        "adjusted_confirmation": (
            file_content_fingerprint(adjusted_path)
            if adjusted_path is not None
            else None
        ),
        "feature_snapshot_hash": snapshot_hash,
        "promotion_role_policy": _promotion_role_policy(),
    }


def _promotion_role_policy() -> dict[str, Any]:
    return {
        "version": PROMOTION_ROLE_POLICY_VERSION,
        "allowed_role": "candidate_biomarker",
        "methodological_control_feature_ids": sorted(
            METHODOLOGICAL_CONTROL_FEATURE_IDS
        ),
        "classical_feature_prefix": "classical_",
    }


def _verify_source_phenome_manifest(
    manifest: StageManifest,
    replicated_path: Path,
    *,
    adjusted_path: Optional[Path] = None,
    adjustment_required: bool = False,
) -> None:
    notes = {str(note).lower() for note in manifest.notes}
    if (
        manifest.status != "complete"
        or "locked_test_scanned=false" not in notes
        or not replicated_path.is_file()
        or str(replicated_path.resolve())
        not in {
            str(Path(path).resolve()) for path in manifest.output_paths
        }
        or (
            adjustment_required
            and (
                adjusted_path is None
                or not adjusted_path.is_file()
                or str(adjusted_path.resolve())
                not in {
                    str(Path(path).resolve())
                    for path in manifest.output_paths
                }
            )
        )
    ):
        raise RuntimeError(
            "A complete development-only PheWAS manifest is required"
        )


def _source_requires_adjustment(manifest: StageManifest) -> bool:
    return "promotion_requires_adjusted_confirmation=true" in {
        str(note).casefold() for note in manifest.notes
    }


def _adjustment_eligible_hits(
    replicated: pd.DataFrame,
    adjusted_path: Optional[Path],
) -> pd.DataFrame:
    if adjusted_path is None:
        # Legacy manifests remain reproducible, but their JSON/manifest is
        # explicitly marked as unadjusted and cannot masquerade as a new run.
        return replicated.copy()
    adjusted = pd.read_parquet(adjusted_path)
    keys = ["phenotype_id", "feature_id", "panel_id"]
    required = set(keys) | {"adjustment_robust"}
    missing = sorted(required - set(adjusted.columns))
    if missing:
        raise RuntimeError(
            "Adjusted PheWAS confirmation is missing columns: "
            f"{missing}"
        )
    if not set(keys).issubset(replicated.columns):
        raise RuntimeError(
            "Replicated PheWAS table lacks panel-aware adjustment keys"
        )
    if adjusted.duplicated(keys).any():
        raise RuntimeError(
            "Adjusted PheWAS confirmation contains duplicate keys"
        )
    eligible_keys = adjusted.loc[
        adjusted["adjustment_robust"].fillna(False).astype(bool),
        keys,
    ]
    if eligible_keys.empty:
        return replicated.iloc[0:0].copy()
    return replicated.merge(
        eligible_keys,
        on=keys,
        how="inner",
        validate="one_to_one",
    )


def _feature_snapshot_hash(manifest: StageManifest) -> str:
    prefix = "feature_snapshot_hash="
    values = [
        str(note)[len(prefix) :]
        for note in manifest.notes
        if str(note).startswith(prefix)
    ]
    if len(values) != 1 or len(values[0]) != 64:
        raise RuntimeError(
            "PheWAS manifest has no unambiguous feature snapshot hash"
        )
    return values[0]


def _verify_replicated_schema(frame: pd.DataFrame) -> None:
    required = {"phenotype_id", "feature_id", "q_value", "roc_auc"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Replicated PheWAS table is missing columns: {sorted(missing)}"
        )


def _excluded_top_candidate_roles(
    replicated: pd.DataFrame,
) -> dict[str, int]:
    if replicated.empty:
        return {}
    _verify_replicated_schema(replicated)
    counts: dict[str, int] = {}
    for _, group in replicated.groupby("phenotype_id", sort=True):
        best = group.sort_values(
            ["q_value", "roc_auc"],
            ascending=[True, False],
        ).iloc[0]
        role = feature_analysis_role(best["feature_id"])
        if role == "candidate_biomarker":
            continue
        counts[role] = counts.get(role, 0) + 1
    return counts


def _promotion_role_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame.empty or "feature_ids" not in frame:
        return counts
    for feature_ids in frame["feature_ids"]:
        if isinstance(feature_ids, str):
            values = (feature_ids,)
        else:
            values = tuple(feature_ids)
        for feature_id in values:
            role = feature_analysis_role(feature_id)
            if role == "candidate_biomarker":
                continue
            counts[role] = counts.get(role, 0) + 1
    return counts


def _candidate_feature_map(
    frame: pd.DataFrame,
) -> dict[str, tuple[str, ...]]:
    if frame.empty:
        return {}
    required = {"candidate_id", "feature_ids"}
    if not required.issubset(frame.columns):
        return {}
    result = {}
    for row in frame.itertuples(index=False):
        candidate_id = str(getattr(row, "candidate_id"))
        feature_ids = getattr(row, "feature_ids")
        if isinstance(feature_ids, str):
            values = (feature_ids,)
        else:
            values = tuple(str(value) for value in feature_ids)
        result[candidate_id] = values
    return result
