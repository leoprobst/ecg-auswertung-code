from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from vcg_biomarker.mimic.validation_provenance import (
    LEGACY_NORMALIZED_VALIDATION_SOURCE_SHA256,
    _normalized_validation_source_sha256,
    locked_evaluation_implementation_hash,
    validation_implementation_manifest,
    verify_validation_code_provenance,
)


def test_validation_scope_is_deterministic_and_invalidates_legacy_anchor() -> None:
    first = validation_implementation_manifest()
    second = validation_implementation_manifest()

    assert first == second
    assert first["implementation_hash"]
    assert (
        first["files"]["mimic/validation.py"]["sha256"]
        != LEGACY_NORMALIZED_VALIDATION_SOURCE_SHA256
    )
    assert "core/preprocessing.py" in first["files"]
    assert "core/delineation/base.py" in first["files"]
    assert "mimic/feature_roles.py" in first["files"]
    for name in ("mechanism_endpoint", "mechanism_models", "mechanism_hypotheses", "broad_axis_adjustment"):
        assert f"mimic/{name}.py" in first["files"]
    assert locked_evaluation_implementation_hash()


def test_all_direct_validation_dependencies_are_bound_including_lazy_imports():
    from vcg_biomarker.mimic import validation
    manifest = validation_implementation_manifest()
    tree = ast.parse(Path(validation.__file__).read_text(encoding="utf-8"))
    dependencies = {f"mimic/{node.module.replace('.', '/')}.py"
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module}
    assert dependencies <= set(manifest["files"])


def test_validation_source_normalization_changes_only_provenance_assignment(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.py"
    scoped = tmp_path / "scoped.py"
    legacy.write_text(
        "from .storage import source_tree_hash\n"
        "def run_development_validation():\n"
        "    code_hash = source_tree_hash('all')\n"
        "    return code_hash\n",
        encoding="utf-8",
    )
    scoped.write_text(
        "from .validation_provenance import validation_implementation_hash\n"
        "def run_development_validation():\n"
        "    code_hash = validation_implementation_hash()\n"
        "    return code_hash\n",
        encoding="utf-8",
    )

    assert _normalized_validation_source_sha256(
        legacy
    ) == _normalized_validation_source_sha256(scoped)


def test_direct_scoped_validation_provenance_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    implementation_hash = validation_implementation_manifest()[
        "implementation_hash"
    ]
    (run_root / "development_provenance.json").write_text(
        json.dumps({"code_hash": implementation_hash}),
        encoding="utf-8",
    )

    verified = verify_validation_code_provenance(run_root)
    assert verified["status"] == "verified_scoped_implementation"
    assert verified["implementation_hash"] == implementation_hash

    (run_root / "development_provenance.json").write_text(
        json.dumps(
            {
                "code_hash": "tampered",
                "input_hash": "tampered",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "validation_code_scope_attestation.json").write_text(
        json.dumps({"status": "forged", "attestation_hash": "forged"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no valid"):
        verify_validation_code_provenance(run_root)
