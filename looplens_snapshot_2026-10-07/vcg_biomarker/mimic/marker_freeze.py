"""Outcome-independent marker inventory freeze and run preflight snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
from typing import Any

from vcg_biomarker.core.biomarkers import BiomarkerRegistry
from vcg_biomarker.core.projection_metadata import projection_metadata
from vcg_biomarker.core.biomarkers.catalog_biomarkers import (
    _feature_segment_dependencies,
)

from .config import MimicProjectConfig
from .feature_extraction import FEATURE_STAGE_VERSION, feature_implementation_hash
from .panels import PANEL_VERSION, default_feature_panels
from .projection_hypotheses import projection_hypothesis_rows
from .storage import (
    atomic_write_json,
    content_hash,
    file_sha256,
    operational_config_hash,
    scientific_config_hash,
)
from .code_scope import operational_source_hash, scientific_source_hash


MARKER_FREEZE_VERSION = "1.22.0"
DEPENDENCY_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "duckdb",
    "pyarrow",
    "wfdb",
    "neurokit2",
    "PyQt6",
    "pyqtgraph",
)
# Libraries whose version can move a computed number.  The GUI toolkits above
# stay recorded, but a PyQt update cannot change a marker or an estimate.
COMPUTATION_DISTRIBUTIONS: tuple[str, ...] = tuple(
    name for name in DEPENDENCY_DISTRIBUTIONS if name not in {"PyQt6", "pyqtgraph"}
)

# Bookkeeping inside small JSON artifacts that says when or where something
# was written, not what it contains.  A143 failed because a byte-identical
# inventory was rewritten with a new timestamp.
_VOLATILE_KEYS = frozenset(
    {"created_at", "finalized_at", "finished_at", "generated_at_utc",
     "mtime_ns", "path", "output_paths"}
)
_OPERATIONAL_NOTE_PREFIX = "operational_config_hash="


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_volatile(item) for key, item in value.items()
                if key not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value
                if not (isinstance(item, str) and item.startswith(_OPERATIONAL_NOTE_PREFIX))]
    return value


def _scientific_content_fingerprint(path: Path) -> dict[str, Any]:
    """What an upstream artifact says, without when or where it was written."""

    import json

    source = Path(path)
    if not source.is_file():
        return {"status": "missing"}
    if source.suffix.casefold() == ".json":
        try:
            parsed = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parsed = None
        if parsed is not None:
            return {"canonical_sha256": content_hash(_strip_volatile(parsed))}
    return {"sha256": file_sha256(source)}


def marker_freeze_payload() -> dict[str, Any]:
    from ..core.biomarkers.a108_robustness import measurement_contract
    from .pre_run_extensions import research_contract
    from ..core.measurement_capsule import VERSION as capsule_version
    from ..core.temporal_research import ANALYSES, VERSION as temporal_version
    from .decision_research import tracking_requirements
    from .mechanism_hypotheses import mechanism_hypothesis_rows
    from .research_coverage import research_coverage_rows, VERSION as coverage_version
    from .pr_calibration_contract import REQUIRED_APPROVAL_GATES, VERSION as pr_contract_version
    from .marker_plausibility import declared_range, plausibility_contract, MARKER_PLAUSIBILITY_VERSION
    panels = default_feature_panels()
    memberships: dict[str, list[str]] = {}
    for panel in panels:
        for marker_id in panel.feature_ids:
            memberships.setdefault(marker_id, []).append(panel.panel_id)
    markers = []
    for definition in sorted(BiomarkerRegistry.definitions(), key=lambda item: item.name):
        panel_ids = tuple(sorted(memberships.get(definition.name, ())))
        if not panel_ids:
            continue
        bounds = declared_range(definition.name, definition.unit)
        markers.append(
            {
                "marker_id": definition.name,
                "formula": definition.formula,
                "unit": definition.unit,
                "segment_dependencies": list(
                    _feature_segment_dependencies(
                        definition.name,
                        definition.family,
                    )
                ),
                "references": sorted(set(definition.references)),
                "reference_status": (
                    "references_declared_not_clinical_validation"
                    if definition.references else "no_reference_declared"
                ),
                "expected_direction": definition.pathological_direction,
                "implementation_status": definition.implementation_status,
                "source": definition.source,
                "panel_ids": list(panel_ids),
                "fixed_plane_projection": projection_metadata(definition.name),
                # Endpoint-specific promotion happens only after the run's
                # evidence gates. The inventory itself begins exploratory.
                "initial_evidence_tier": "exploratory",
                "plausibility_coverage": plausibility_contract(definition.name, definition.unit),
                "mathematical_plausibility_range": None if bounds is None else {
                    "lower": bounds.lower, "upper": bounds.upper,
                    "reason": bounds.reason,
                    "numerical_tolerance": bounds.numerical_tolerance,
                },
            }
        )
    return {
        "marker_freeze_version": MARKER_FREEZE_VERSION,
        "a108_exploratory_measurement_contract": measurement_contract(),
        "marker_plausibility_version": MARKER_PLAUSIBILITY_VERSION,
        "feature_stage_version": FEATURE_STAGE_VERSION,
        "panel_version": PANEL_VERSION,
        "feature_implementation_hash": feature_implementation_hash(),
        "panels": [panel.to_dict() for panel in panels],
        "markers": markers,
        "projection_hypotheses": projection_hypothesis_rows(markers),
        "mechanism_hypotheses": mechanism_hypothesis_rows(markers),
        "research_coverage": {"version": coverage_version,
                              "chapters": research_coverage_rows(markers)},
        "pr_geometry_calibration_contract": {
            "version": pr_contract_version, "required_approval_gates": list(REQUIRED_APPROVAL_GATES),
            "required_files": ["pr_geometry_calibration.json", "pr_calibration_review_report.json",
                               "pr_geometry_qualified.parquet"],
            "reference_discrepancy_is_not_automatically_measurement_error": True,
        },
        "temporal_research": {
            "version": temporal_version, "analyses": ANALYSES,
            "role": "sequence_or_pair_analysis_not_single_ecg_diagnostic_marker",
            "feature_column": "context_temporal_research_json",
            "clinical_claim_allowed": False,
        },
        "decision_research_tracking": tracking_requirements(),
        "pre_run_extensions": research_contract(),
        "measurement_capsule": {
            "version": capsule_version, "column": "context_measurement_capsule_json",
            "role": "non_scalar_foundation", "clinical_claim_allowed": False,
            "requires_new_extraction": True, "templates": "float32_cluster_exemplars",
        },
        "measurement_sensitivity": {
            "version": "1.0.0", "column": "context_measurement_sensitivity_json",
            "sampling_probability": .01, "seed": 20260914,
            "sampling": "source_record_hash_before_outcome_and_quality",
            "interval_kind": "sensitivity_not_confidence_interval", "clinical_claim_allowed": False,
        },
    }


def _dependency_versions(
    names: tuple[str, ...] = COMPUTATION_DISTRIBUTIONS,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _small_input_fingerprints(config: MimicProjectConfig) -> dict[str, dict]:
    candidates = {
        "inventory_manifest": config.work_root / "inventory.manifest.json",
        "data_inventory": config.work_root / "data_inventory.json",
        "ingest_manifest": config.standardized_root / "ingest.manifest.json",
        "clinical_events_manifest": config.standardized_root
        / "clinical_events"
        / "clinical_events.manifest.json",
        "master_index_manifest": config.master_root / "master_index.manifest.json",
        "evidence_manifest": config.evidence_root / "evidence.manifest.json",
        "labels_manifest": config.labels_root / "labels.manifest.json",
        "delineation_benchmark": config.resources_root
        / "delineation"
        / "benchmark_suite.json",
        # The biomarker audit is a derived gate output, not an upstream
        # scientific input.  It is written immediately before this snapshot
        # is finalized; hashing it here made the audit invalidate itself on
        # every retry because its generated timestamp changes.  Marker code,
        # formulas and panels remain protected by marker_freeze_payload(),
        # while the audit result is bound separately in the software gate.
        "gwas_feasibility_manifest": config.work_root
        / "genetics"
        / "gwas_feasibility.manifest.json",
        "gwas_feasibility_summary": config.work_root
        / "genetics"
        / "gwas_feasibility.json",
    }
    return {
        str(name): _scientific_content_fingerprint(path)
        for name, path in sorted(candidates.items(), key=lambda item: str(item[0]))
    }


def _snapshot_payload(
    config: MimicProjectConfig,
    *,
    run_id: str,
    freeze_hash: str,
    marker_payload: dict[str, Any],
) -> dict[str, Any]:
    """Everything that can change a number; nothing that only runs the study.

    Enforced from the freeze until the end of the full run.  Worker counts,
    the Windows build, file locations, GUI toolkits and the run machinery
    (``code_scope.OPERATIONAL_MODULES``) live in ``_observed_environment``:
    recorded, compared, reported as form drift, never a reason to stop.
    """

    mimic_root = Path(__file__).resolve().parent
    return {
        "snapshot_version": MARKER_FREEZE_VERSION,
        "run_id": run_id,
        "marker_freeze_hash": freeze_hash,
        "scientific_config_hash": scientific_config_hash(config),
        # The GUI has its own source-bound review gate and the run machinery
        # is excluded by code_scope; marker and signal code is covered
        # independently by feature_implementation_hash.
        "source_tree_scope": "vcg_biomarker/mimic scientific modules (code_scope)",
        "source_tree_hash": scientific_source_hash(mimic_root),
        "feature_implementation_hash": marker_payload[
            "feature_implementation_hash"
        ],
        "random_seed": int(config.random_seed),
        "python": platform.python_version(),
        "dependencies": _dependency_versions(),
        "inputs": _small_input_fingerprints(config),
    }


def _observed_environment(config: MimicProjectConfig) -> dict[str, Any]:
    """Recorded alongside the snapshot; drift here never refuses a stage."""

    package_root = Path(__file__).resolve().parents[1]
    mimic_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    lock_candidates = (
        project_root / "requirements-lock-win-py312.txt",
        project_root / "requirements-lock-win-py312-hashed.txt",
    )
    lock_path = next(
        (path for path in lock_candidates if path.is_file()),
        lock_candidates[0],
    )
    pyproject_path = project_root / "pyproject.toml"
    return {
        "operational_config_hash": operational_config_hash(config),
        "operational_source_hash": operational_source_hash(mimic_root),
        "platform": platform.platform(),
        "all_recorded_dependencies": _dependency_versions(DEPENDENCY_DISTRIBUTIONS),
        "dependency_lock_sha256": (
            file_sha256(lock_path) if lock_path.is_file() else "missing"
        ),
        "project_definition_sha256": (
            file_sha256(pyproject_path) if pyproject_path.is_file() else "missing"
        ),
        "runtime_root": str(package_root.parent.resolve()),
        "project_config_path": str(config.project_file.resolve()),
    }


def _record_form_drift(config: MimicProjectConfig, run_id: str, snapshot: dict) -> list[str]:
    """Append a line to ``form_drift.jsonl`` when the observed part moved."""

    import json

    stored = dict(snapshot.get("observed") or {})
    current = _observed_environment(config)
    changed = sorted(key for key in set(stored) | set(current)
                     if stored.get(key) != current.get(key))
    if not stored or not changed:
        return []
    log_path = config.runs_root / run_id / "form_drift.jsonl"
    entry = {"changed": changed, "frozen": {k: stored.get(k) for k in changed},
             "current": {k: current.get(k) for k in changed}}
    try:
        previous = log_path.read_text(encoding="utf-8").splitlines()[-1] if log_path.is_file() else ""
        last = json.loads(previous) if previous else {}
    except (OSError, ValueError):
        last = {}
    if {k: last.get(k) for k in ("changed", "current")} != {k: entry[k] for k in ("changed", "current")}:
        entry["at"] = datetime.now(timezone.utc).isoformat()
        try:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
    return changed


def freeze_marker_inventory(
    config: MimicProjectConfig,
    *,
    run_id: str,
    finalize_inputs: bool = False,
) -> dict[str, Any]:
    """Write an immutable marker freeze and matching preflight snapshot.

    A changed payload under the same run identifier is refused. This is the
    enforcement that turns post-pilot formula edits into a new run rather
    than a quiet redefinition of the hypotheses already inspected.
    """

    run_root = config.runs_root / run_id
    freeze_path = run_root / "marker_freeze.json"
    snapshot_path = run_root / "run_preflight_snapshot.json"
    payload = marker_freeze_payload()
    freeze_hash = content_hash(payload)
    if freeze_path.is_file():
        import json

        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous.get("freeze_hash") != freeze_hash:
            raise RuntimeError(
                "Marker definitions or panels changed after this run identifier "
                "was frozen. Use a new run_id and repeat both pilots."
            )
    else:
        atomic_write_json(
            freeze_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "freeze_hash": freeze_hash,
                "payload": payload,
            },
        )

    snapshot_payload = _snapshot_payload(
        config,
        run_id=run_id,
        freeze_hash=freeze_hash,
        marker_payload=payload,
    )
    snapshot_payload["input_snapshot_phase"] = (
        "prepared" if finalize_inputs else "preflight"
    )
    snapshot_hash = content_hash(snapshot_payload)
    if snapshot_path.is_file():
        import json

        previous_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        previous_payload = previous_snapshot.get("payload")
        if not isinstance(previous_payload, dict) or content_hash(
            previous_payload
        ) != previous_snapshot.get("snapshot_hash"):
            raise RuntimeError("Stored preflight snapshot is corrupt or was edited.")

        previous_phase = str(
            previous_payload.get("input_snapshot_phase", "prepared")
        )
        # Preparation stages intentionally regenerate inventory, evidence,
        # label and delineation manifests.  They therefore cannot be frozen
        # before preparation and compared byte-for-byte during the audit.
        # All non-input provenance remains immutable throughout.  At the
        # audit boundary the newly produced inputs are frozen exactly once;
        # pilots and the later full extraction then require an exact match.
        previous_non_inputs = {
            key: value
            for key, value in previous_payload.items()
            if key not in {"inputs", "input_snapshot_phase"}
        }
        current_non_inputs = {
            key: value
            for key, value in snapshot_payload.items()
            if key not in {"inputs", "input_snapshot_phase"}
        }
        may_finalize = previous_phase == "preflight" and finalize_inputs
        if previous_non_inputs != current_non_inputs or (
            previous_snapshot.get("snapshot_hash") != snapshot_hash
            and not may_finalize
        ):
            raise RuntimeError(
                "Source code, configuration, dependencies, or upstream inputs "
                "changed after this run identifier was frozen. Use a new run_id "
                "and repeat both pilots."
            )
        if may_finalize:
            atomic_write_json(
                snapshot_path,
                {
                    "created_at": previous_snapshot.get("created_at"),
                    "finalized_at": datetime.now(timezone.utc).isoformat(),
                    "snapshot_hash": snapshot_hash,
                    "payload": snapshot_payload,
                    "observed": previous_snapshot.get("observed")
                    or _observed_environment(config),
                },
            )
    else:
        atomic_write_json(
            snapshot_path,
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_hash": snapshot_hash,
                "payload": snapshot_payload,
                "observed": _observed_environment(config),
            },
        )
    return {
        "run_id": run_id,
        "freeze_hash": freeze_hash,
        "snapshot_hash": snapshot_hash,
        "marker_count": len(payload["markers"]),
        "panel_count": len(payload["panels"]),
        "freeze_path": str(freeze_path.resolve()),
        "snapshot_path": str(snapshot_path.resolve()),
    }


def assert_marker_freeze_current(
    config: MimicProjectConfig,
    *,
    run_id: str,
) -> dict[str, Any]:
    import json

    freeze_path = config.runs_root / run_id / "marker_freeze.json"
    snapshot_path = config.runs_root / run_id / "run_preflight_snapshot.json"
    if not freeze_path.is_file() or not snapshot_path.is_file():
        raise RuntimeError(
            "Pilot refused: marker freeze and preflight snapshot must exist."
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = content_hash(marker_freeze_payload())
    if (
        freeze.get("freeze_hash") != expected
        or content_hash(freeze.get("payload")) != expected
    ):
        raise RuntimeError(
            "Pilot refused: marker definitions changed after the freeze. "
            "Use a new run_id and repeat the marker audit."
        )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    stored_payload = snapshot.get("payload")
    if (
        not isinstance(stored_payload, dict)
        or content_hash(stored_payload) != snapshot.get("snapshot_hash")
        or stored_payload.get("marker_freeze_hash") != expected
    ):
        raise RuntimeError("Pilot refused: preflight snapshot and marker freeze differ.")
    current_payload = _snapshot_payload(
        config,
        run_id=run_id,
        freeze_hash=expected,
        marker_payload=freeze["payload"],
    )
    current_payload["input_snapshot_phase"] = "prepared"
    if stored_payload.get("input_snapshot_phase") != "prepared":
        raise RuntimeError(
            "Pilot refused: input provenance has not been finalized after "
            "the preparation stages."
        )
    current_hash = content_hash(current_payload)
    if current_hash != snapshot.get("snapshot_hash"):
        changed = sorted(
            key for key in set(stored_payload) | set(current_payload)
            if stored_payload.get(key) != current_payload.get(key)
        )
        if "inputs" in changed:
            stored_inputs = dict(stored_payload.get("inputs") or {})
            current_inputs = dict(current_payload.get("inputs") or {})
            changed.remove("inputs")
            changed += [f"inputs.{name}" for name in sorted(set(stored_inputs) | set(current_inputs))
                        if stored_inputs.get(name) != current_inputs.get(name)]
        raise RuntimeError(
            "Pilot refused: source code, configuration, dependencies, or "
            "upstream inputs changed after the preflight snapshot "
            f"({', '.join(changed)}). Use a new run_id and repeat both pilots."
        )
    return {
        "freeze_hash": expected,
        "snapshot_hash": current_hash,
        "form_drift": _record_form_drift(config, run_id, snapshot),
    }


__all__ = [
    "MARKER_FREEZE_VERSION",
    "assert_marker_freeze_current",
    "freeze_marker_inventory",
    "marker_freeze_payload",
]
