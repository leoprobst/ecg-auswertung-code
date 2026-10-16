"""The same fail-closed row scope for direct atlas parquet projections."""
from __future__ import annotations

from pathlib import Path

from ...mimic.storage import StageManifest, verify_output_integrity


def validation_output_assurance(
    manifest: StageManifest, path: Path, run_id: str,
) -> str | None:
    """Return the assurance available for a completed Validation output.

    The production validation writer for the current run declares outputs but
    does not persist per-output hashes. Those files are usable after the
    complete manifest, but must be labelled ``manifest_only``. If a future or
    synthetic manifest *does* declare hashes, a missing/bad hash fails closed.
    """
    if (manifest.status != "complete"
            or manifest.stage not in {"validate", f"validate:{run_id}"}
            or not path.is_file()):
        return None
    resolved = str(path.resolve())
    if resolved not in {str(Path(item).resolve()) for item in manifest.output_paths}:
        return None
    if not manifest.output_integrity:
        return "manifest_only"
    records = [record for record in manifest.output_integrity
               if str(Path(str(record.get("path", ""))).resolve()) == resolved]
    if len(records) == 1 and verify_output_integrity(records, mode="sha256"):
        return "sha256"
    return None


def compatible_validation_clauses(schema: set[str]) -> list[str]:
    """Mirror the ResultRepository defaults when a bounded aggregate re-reads its artifact."""
    clauses: list[str] = []
    if "evaluation_split" in schema:
        clauses.append("CAST(evaluation_split AS VARCHAR) = 'validation'")
    if "analysis_completeness" in schema:
        clauses.append("CAST(analysis_completeness AS VARCHAR) = 'complete_snapshot'")
    if "historical_pre_audit_result" in schema:
        clauses.append("COALESCE(TRY_CAST(historical_pre_audit_result AS BOOLEAN), false) = false")
    elif "historical_implementation" in schema:
        clauses.append("COALESCE(TRY_CAST(historical_implementation AS BOOLEAN), false) = false")
    return clauses
