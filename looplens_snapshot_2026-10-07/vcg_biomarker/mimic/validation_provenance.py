from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage import (
    atomic_write_json,
    content_hash,
    file_sha256,
)


VALIDATION_IMPLEMENTATION_SCOPE_VERSION = "mimic-validation-science-v1.11.0"
LOCKED_EVALUATION_SCOPE_VERSION = "mimic-locked-evaluation-v1.1.0"
LEGACY_ATTESTATION_VERSION = "mimic-validation-legacy-attestation-v1.0.0"
LEGACY_VALIDATION_STAGE_VERSION = "2.9.0"
LEGACY_VALIDATION_INPUT_HASHES = frozenset(
    {
        "acbf74444bb6505e6e8cb156190f85d63270d0131addf1257287c9f369531d86",
    }
)
LEGACY_VALIDATION_BROAD_CODE_HASHES = frozenset(
    {
        "9c110484915a3b4bf8a5fc56dca5a0bbde9a17cf85c21e4fda2a361321890225",
    }
)
LEGACY_NORMALIZED_VALIDATION_SOURCE_SHA256 = (
    "e64b0a735c7c4371b5538d8fc440be67250eefaba1cf9443c753a6ae4fd40e86"
)

_VALIDATION_SCOPE_PATHS = (
    "core/preprocessing.py",
    "core/vcg_transform.py",
    "mimic/adjusted_models.py",
    "mimic/covariate_adjusted_roc.py",
    "mimic/broad_axis_adjustment.py",
    "mimic/mechanism_hypotheses.py",
    "mimic/research_coverage.py",
    "mimic/mechanism_echo_context.py",
    "mimic/extended_mechanism_models.py",
    "mimic/extended_mechanism_workflow.py",
    "mimic/pr_calibration_contract.py",
    "mimic/mechanism_models.py",
    "mimic/mechanism_endpoint.py",
    "mimic/lead_comparison.py",
    "mimic/design_diagnostics.py",
    "mimic/lead_confirmation.py",
    "mimic/reference_tiers.py",
    "mimic/lead_endpoint_analysis.py",
    "mimic/lead_measurements.py",
    "mimic/marker_process_pool.py",
    "mimic/hf_qrs_quality.py",
    "mimic/voltage_mass.py",
    "mimic/voltage_mass_models.py",
    "mimic/voltage_mass_endpoint.py",
    "mimic/voltage_mass_confirmation.py",
    "mimic/baseline_registry.py",
    "mimic/classical.py",
    "mimic/config.py",
    "mimic/criteria.py",
    "mimic/event_equivalence.py",
    "mimic/feature_extraction.py",
    "mimic/feature_roles.py",
    "mimic/label_registry.py",
    "mimic/labels.py",
    "mimic/methodology_registry.py",
    "mimic/models.py",
    "mimic/optimization.py",
    "mimic/panels.py",
    "mimic/phenotypes.py",
    "mimic/ptb_transfer.py",
    "mimic/snapshots.py",
    "mimic/signals.py",
    "mimic/statistics.py",
    "mimic/storage.py",
    "mimic/transfer.py",
    "mimic/validation.py",
    "mimic/checkpoint_codec.py",
    "mimic/cross_fitting.py",
    "mimic/delineation_sensitivity.py",
    "mimic/fiducial_limitation.py",
    "mimic/partitioned_records.py",
    "mimic/projection_hypotheses.py",
    "mimic/reference_clock_stress.py",
    "mimic/refinement.py",
    "mimic/resources.py",
    "mimic/resource_grants.py",
    "mimic/spill_budget.py",
    "mimic/result_contract.py",
    "mimic/result_explorer.py",
    "mimic/result_finalization.py",
    "mimic/run_integrity.py",
    "mimic/run_log.py",
    "mimic/selection_bias.py",
    "mimic/validation_provenance.py",
)
_VALIDATION_SCOPE_DIRECTORIES = (
    "core/biomarkers",
    "core/delineation",
    "mimic/resources/baselines",
    "mimic/resources/methodology",
)


def validation_implementation_manifest() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    files = _scoped_files(package_root)
    fingerprints = {}
    for relative, path in files.items():
        fingerprints[relative] = {
            "sha256": (
                _normalized_validation_source_sha256(path)
                if relative == "mimic/validation.py"
                else file_sha256(path)
            ),
            "size": (
                None
                if relative == "mimic/validation.py"
                else path.stat().st_size
            ),
            "normalization": (
                "validation_provenance_assignment_only"
                if relative == "mimic/validation.py"
                else "exact_bytes"
            ),
        }
    payload = {
        "scope_version": VALIDATION_IMPLEMENTATION_SCOPE_VERSION,
        "files": fingerprints,
    }
    payload["implementation_hash"] = content_hash(payload)
    return payload


def validation_implementation_hash() -> str:
    return str(validation_implementation_manifest()["implementation_hash"])


def locked_evaluation_implementation_hash() -> str:
    package_root = Path(__file__).resolve().parents[1]
    validation_manifest = validation_implementation_manifest()
    locking_path = package_root / "mimic" / "locking.py"
    payload = {
        "scope_version": LOCKED_EVALUATION_SCOPE_VERSION,
        "development_validation_hash": validation_manifest[
            "implementation_hash"
        ],
        "locking_sha256": file_sha256(locking_path),
    }
    return content_hash(payload)


def legacy_validation_attestation_path(run_root: Path) -> Path:
    return run_root / "validation_code_scope_attestation.json"


def attest_legacy_validation_code(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    provenance_path = run_root / "development_provenance.json"
    manifest_path = run_root / "validate.manifest.json"
    provenance = _read_json(provenance_path)
    manifest = _read_json(manifest_path)
    input_payload = provenance.get("validation_input_payload")
    if not isinstance(input_payload, dict):
        raise RuntimeError("Legacy validation has no immutable input payload")
    if (
        manifest.get("status") != "complete"
        or manifest.get("stage_version") != LEGACY_VALIDATION_STAGE_VERSION
        or manifest.get("input_hash") not in LEGACY_VALIDATION_INPUT_HASHES
        or provenance.get("input_hash") != manifest.get("input_hash")
        or content_hash(input_payload) != manifest.get("input_hash")
        or provenance.get("code_hash")
        not in LEGACY_VALIDATION_BROAD_CODE_HASHES
        or input_payload.get("code_hash") != provenance.get("code_hash")
        or provenance.get("locked_test_labels_read") is not False
    ):
        raise RuntimeError(
            "Validation run is not an approved legacy migration candidate"
        )
    current = validation_implementation_manifest()
    normalized_validation_hash = current["files"]["mimic/validation.py"][
        "sha256"
    ]
    if (
        not LEGACY_NORMALIZED_VALIDATION_SOURCE_SHA256
        or normalized_validation_hash
        != LEGACY_NORMALIZED_VALIDATION_SOURCE_SHA256
    ):
        raise RuntimeError(
            "Legacy validation algorithm source no longer matches its "
            "reviewed migration anchor"
        )
    created_at = datetime.fromisoformat(str(provenance["created_at"]))
    package_root = Path(__file__).resolve().parents[1]
    scoped_files = _scoped_files(package_root)
    changed_scientific_files = []
    for relative, path in scoped_files.items():
        if relative == "mimic/validation.py":
            continue
        modified_at = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=created_at.tzinfo,
        )
        if modified_at > created_at:
            changed_scientific_files.append(relative)
    if changed_scientific_files:
        raise RuntimeError(
            "Scientific validation files changed after the legacy run: "
            f"{changed_scientific_files}"
        )
    missing_outputs = [
        path
        for path in manifest.get("output_paths", ())
        if not Path(str(path)).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Legacy validation outputs are incomplete: {missing_outputs}"
        )
    post_validation_non_scientific_changes = sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if path not in scoped_files.values()
        and datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=created_at.tzinfo,
        )
        > created_at
    )
    payload = {
        "attestation_version": LEGACY_ATTESTATION_VERSION,
        "status": "verified_legacy_scientific_scope_unchanged",
        "run_id": provenance.get("run_id"),
        "validation_created_at": provenance.get("created_at"),
        "legacy_stage_version": manifest.get("stage_version"),
        "legacy_input_hash": manifest.get("input_hash"),
        "legacy_broad_code_hash": provenance.get("code_hash"),
        "validation_implementation": current,
        "legacy_normalized_validation_source_sha256": (
            normalized_validation_hash
        ),
        "changed_scientific_files": changed_scientific_files,
        "post_validation_non_scientific_changes": (
            post_validation_non_scientific_changes
        ),
        "locked_test_labels_read": False,
        "locked_test_outcomes_read": False,
        "migration_reason": (
            "The legacy run hashed the complete application tree. The "
            "attestation limits compatibility to the reviewed validation "
            "algorithm and dependencies; GUI, reporting and PheWAS changes "
            "cannot alter train/validation marker estimates or thresholds."
        ),
        "source_paths": {
            "development_provenance": str(provenance_path),
            "validation_manifest": str(manifest_path),
        },
    }
    payload["attestation_hash"] = content_hash(payload)
    atomic_write_json(legacy_validation_attestation_path(run_root), payload)
    return payload


def verify_validation_code_provenance(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    provenance = _read_json(run_root / "development_provenance.json")
    current = validation_implementation_manifest()
    recorded = str(provenance.get("code_hash", ""))
    if recorded == current["implementation_hash"]:
        return {
            "status": "verified_scoped_implementation",
            "implementation_hash": current["implementation_hash"],
            "scope_version": current["scope_version"],
            "attestation_path": None,
            "legacy_broad_code_hash": None,
        }
    path = legacy_validation_attestation_path(run_root)
    attestation = _read_json(path)
    recorded_attestation_hash = str(attestation.get("attestation_hash", ""))
    hash_payload = dict(attestation)
    hash_payload.pop("attestation_hash", None)
    if (
        not recorded_attestation_hash
        or recorded_attestation_hash != content_hash(hash_payload)
        or attestation.get("status")
        != "verified_legacy_scientific_scope_unchanged"
        or attestation.get("legacy_broad_code_hash") != recorded
        or attestation.get("legacy_input_hash") != provenance.get("input_hash")
        or attestation.get("locked_test_labels_read") is not False
        or attestation.get("locked_test_outcomes_read") is not False
        or (
            attestation.get("validation_implementation") or {}
        ).get("implementation_hash")
        != current["implementation_hash"]
    ):
        raise RuntimeError(
            "Validation implementation differs from the run and no valid "
            "legacy scientific-scope attestation is available"
        )
    return {
        "status": "verified_legacy_scientific_scope_attestation",
        "implementation_hash": current["implementation_hash"],
        "scope_version": current["scope_version"],
        "attestation_path": str(path),
        "attestation_hash": recorded_attestation_hash,
        "legacy_broad_code_hash": recorded,
    }


def _scoped_files(package_root: Path) -> dict[str, Path]:
    files = {
        relative: package_root / Path(relative)
        for relative in _VALIDATION_SCOPE_PATHS
    }
    for relative_directory in _VALIDATION_SCOPE_DIRECTORIES:
        directory = package_root / Path(relative_directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".py", ".json"}:
                files[path.relative_to(package_root).as_posix()] = path
    missing = [
        relative
        for relative, path in files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Validation implementation scope is incomplete: {missing}"
        )
    return dict(sorted(files.items()))


def _normalized_validation_source_sha256(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    normalized = _ValidationProvenanceNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return content_hash(
        {
            "normalization": "validation-provenance-assignment-v1",
            "ast": ast.dump(
                normalized,
                annotate_fields=True,
                include_attributes=False,
            ),
        }
    )


class _ValidationProvenanceNormalizer(ast.NodeTransformer):
    """What counts as a change to the validation implementation.

    The hash is taken over the parsed tree rather than the file text, so
    comments have never moved it. Docstrings did, and that was the whole of
    a measured problem: ``mimic/storage.py`` is in the validation scope and
    holds the manifest, the hashing *and* the parquet read clauses, so
    rewording the docstring of a read clause shifted the validation code hash
    and with it the reuse key of the pilot validations -- 25,8 hours of
    computation (``validate-pilot-1000`` 23,9 h, ``validate-pilot-10000``
    1,9 h). On 2026-08-28 the stored hash was ``22c26fb6...`` against a current
    ``e127101f...``, and the run was rescued by a rule about spent pilot
    stages rather than by fixing the cause.

    The proposal on file was to split ``storage.py`` into a result-defining
    half and a pure-I/O half, and to scope only the first. Measured against
    the actual contents that split does not hold: ``development_feature_glob``
    decides *which* recordings enter an analysis and ``feature_parquet_clause``
    decides how they bind -- the defect where a recursive glob pulled in the
    pilot partitions came from exactly there. Both are result-defining despite
    looking like I/O, so the split would have had to put nearly everything on
    the scoped side and would not have fixed the docstring problem at all.

    Dropping docstrings does. It is also the more honest rule: if the only
    thing that changed is the prose, no number can have changed with it. What
    remains in scope is every name, constant, expression and control-flow
    decision -- which is what a result actually depends on.
    """

    def __init__(self) -> None:
        self._function_name = ""

    @staticmethod
    def _strip_docstring(node):
        """Remove a leading string expression, but never an empty body.

        A function whose entire body is its docstring would become a syntax
        error, so the docstring is replaced by ``pass`` rather than deleted.
        """

        body = getattr(node, "body", None)
        if not body:
            return node
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            remaining = body[1:]
            node.body = remaining if remaining else [ast.Pass()]
        return node

    def visit_Module(self, node: ast.Module):
        return self._strip_docstring(self.generic_visit(node))

    def visit_ClassDef(self, node: ast.ClassDef):
        return self._strip_docstring(self.generic_visit(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._strip_docstring(self.generic_visit(node))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "validation_provenance":
            return None
        if node.module == "storage":
            node.names = [
                alias
                for alias in node.names
                if alias.name != "source_tree_hash"
            ]
            if not node.names:
                return None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        previous = self._function_name
        self._function_name = node.name
        result = self.generic_visit(node)
        self._function_name = previous
        return self._strip_docstring(result)

    def visit_Assign(self, node: ast.Assign):
        if (
            self._function_name == "run_development_validation"
            and any(
                isinstance(target, ast.Name)
                and target.id == "code_hash"
                for target in node.targets
            )
        ):
            node.value = ast.Constant(
                value="VALIDATION_IMPLEMENTATION_HASH"
            )
            return node
        return self.generic_visit(node)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload
