"""Explicit, isolated CLI for protocol-bound causal and mapping analyses.

python -m vcg_biomarker.mimic.research_extensions --manifest job.json --output NEW_DIR
No discovery of production artifacts, no implicit execution and no overwrites.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from .causal_treatment import CausalProtocol, estimate_treatment_effects
from .decision_research import TreatmentProfile
from .focus_localization import validate_mapping
from .mapping_comparison import (
    METHODS,
    beat_transform_features,
    compare_mapping_transforms,
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def research_readiness(work_root):
    """Capability inventory, not scientific authorization or data verification."""
    root = Path(work_root)
    return {
        "version": "1.0.0",
        "blocks_primary_development_run": False,
        "clinical_claim_allowed": False,
        "branches": {
            "causal_fixed_horizon": {
                "status": "requires_verified_protocol_and_record_adapter",
                "automatic_execution": False,
                "backend_available": True,
            },
            "mapping_localization": {
                "status": "requires_record_level_validation"
                if (root / "mapping_imports").is_dir()
                else "deferred_missing_mapping_reference",
                "automatic_execution": False,
                "backend_available": True,
            },
            "time_varying_causal": {
                "status": "deferred_unsupported_design",
                "automatic_execution": False,
                "backend_available": False,
            },
            "additional_continuous_survival_scores": {
                "status": "requires_typed_outcome_adapter",
                "automatic_execution": False,
                "backend_available": True,
            },
            "external_score_validation": {
                "status": "requires_frozen_score_and_compatible_external_endpoint",
                "automatic_execution": False,
                "backend_available": True,
            },
        },
    }


def import_mapping_beats(mapping, signal_root, *, task):
    """Verified rows + NPZ surface beats, never ablation/ICD-derived truth.

    NPZ keys: ecg_mv (samples x 12), baseline_mv (12), sampling_rate,
    onset, offset. Atrial imports must delineate a P wave, ventricular imports
    a QRS; segment column is mandatory. NPZ pickle is never enabled.
    """
    from ..core.vcg_geometry import VCG_AXIS_CONVENTION

    root = Path(signal_root).resolve(strict=True)
    if "signal_file" not in mapping or "segment" not in mapping:
        raise ValueError("mapping_requires_signal_file_and_segment")
    # Validate split and reference BEFORE touching any waveform.
    check = mapping.assign(transform="kors", axis_convention=VCG_AXIS_CONVENTION)
    validate_mapping(check, task)
    expected = "p" if task == "atrial_focus" else "qrs"
    if not mapping.segment.eq(expected).all():
        raise ValueError("wrong_delineated_segment_for_rhythm_task")
    rows: dict[str, list[dict]] = {method: [] for method in METHODS}
    provenance: dict[str, str] = {}
    owners: dict[str, tuple[str, str]] = {}
    for _, row in mapping.iterrows():
        path = (root / str(row.signal_file)).resolve(strict=True)
        if not path.is_relative_to(root) or path.suffix.lower() != ".npz":
            raise ValueError("signal_outside_import_root_or_not_npz")
        digest = file_hash(path)
        with np.load(path, allow_pickle=False) as signal:
            required = {"ecg_mv", "baseline_mv", "sampling_rate", "onset", "offset"}
            if not required <= set(signal.files):
                raise ValueError("incomplete_mapping_signal")
            # Content identity also catches duplicate archives with different compression.
            samples = np.asarray(signal["ecg_mv"], dtype="<f8")
            content_hash = hashlib.sha256(
                samples.tobytes() + str(samples.shape).encode()
            ).hexdigest()
            owner = (str(row.cohort_id), str(row.patient_id))
            if content_hash in owners and owners[content_hash] != owner:
                raise ValueError("duplicate_signal_across_patient_identities")
            owners[content_hash] = owner
            features = beat_transform_features(
                samples,
                signal["baseline_mv"],
                float(signal["sampling_rate"].item()),
                signal["onset"].item(),
                signal["offset"].item(),
                segment=expected,
            )
        if digest != file_hash(path):
            raise ValueError("signal_changed_during_import")
        provenance[str(path)] = digest
        for method in METHODS:
            axes = (
                "svd_invariant_no_anatomical_axes"
                if method == "svd"
                else "standard_12_lead_order"
                if method == "native_ecg"
                else VCG_AXIS_CONVENTION
            )
            rows[method].append(
                {
                    **row.to_dict(),
                    **features[method],
                    "transform": method,
                    "axis_convention": axes,
                }
            )
    frames = {method: pd.DataFrame(values) for method, values in rows.items()}
    columns = {method: list(features[method]) for method in METHODS}
    return frames, columns, provenance


def run_manifest(manifest_path, output_path):
    """Run only explicit verified imports. Failures are recorded, not suppressed."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    output = Path(output_path).resolve()
    manifest_digest = file_hash(manifest_path)
    job = json.loads(manifest_path.read_text(encoding="utf-8"))
    if job.get("schema_version") != "1.0.0":
        raise ValueError("unsupported_research_manifest")
    if output.exists():
        raise FileExistsError("research_output_must_be_new")
    root = manifest_path.parent
    data_path = (root / job["data"]).resolve(strict=True)
    if any(part.lower() in {"locked_test", "locked-test"} for part in data_path.parts):
        raise ValueError("locked_test_input_forbidden")
    if data_path.suffix.lower() != ".parquet":
        raise ValueError("parquet_required")
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        str(manifest_path): manifest_digest,
        str(data_path): file_hash(data_path),
    }
    source_names = (
        "research_extensions.py",
        "causal_treatment.py",
        "decision_research.py",
        "mapping_comparison.py",
        "focus_localization.py",
    )
    sources = {name: file_hash(Path(__file__).parent / name) for name in source_names}
    source_paths = {name: Path(__file__).parent / name for name in source_names}
    for name in ("vcg_transform.py", "vcg_geometry.py", "data_loader.py"):
        key = "core/" + name
        source_paths[key] = Path(__file__).parent.parent / "core" / name
        sources[key] = file_hash(source_paths[key])
    try:
        frame = pd.read_parquet(data_path)
        if job["kind"] == "causal":
            raw = dict(job["protocol"])
            raw["profile"] = TreatmentProfile(**raw["profile"])
            raw["confounders"] = tuple(raw["confounders"])
            if set(raw) - {f.name for f in fields(CausalProtocol)}:
                raise ValueError("unknown_causal_policy_fields")
            result = estimate_treatment_effects(
                frame, CausalProtocol(**raw), **job["verification"]
            )
        elif job["kind"] == "mapping":
            frames, columns, signal_provenance = import_mapping_beats(
                frame, root / job["signal_root"], task=job["task"]
            )
            provenance.update(signal_provenance)
            result = compare_mapping_transforms(
                frames, columns, task=job["task"], **job["comparison"]
            )
        else:
            raise ValueError("unknown_research_job_kind")
        if any(file_hash(path) != digest for path, digest in provenance.items()):
            raise ValueError("inputs_changed_during_analysis")
        if any(
            file_hash(source_paths[name]) != digest for name, digest in sources.items()
        ):
            raise ValueError("sources_changed_during_analysis")
        result["provenance"] = {
            "inputs_sha256": provenance,
            "sources_sha256": sources,
            "python": platform.python_version(),
            "dependencies": {
                name: version(name) for name in ("numpy", "pandas", "scikit-learn")
            },
        }
        # Validate the entire payload before creating a success artifact.
        serialized = json.dumps(result, indent=2, allow_nan=False)
        with (output / "result.json").open("x", encoding="utf-8") as handle:
            handle.write(serialized)
        return result
    except (ValueError, KeyError, TypeError, OSError) as exc:
        with (output / "failure.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                    "inputs_sha256": provenance,
                    "sources_sha256": sources,
                    "clinical_claim_allowed": False,
                },
                handle,
                indent=2,
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_manifest(args.manifest, args.output)


if __name__ == "__main__":
    main()
