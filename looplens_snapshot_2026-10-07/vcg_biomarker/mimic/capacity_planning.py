"""Conservative disk-capacity release evidence for a development full run.

This module reads manifests and file-size summaries only.  It never opens a
waveform, clinical label, or locked-test row.  The estimate deliberately keeps
record-scaled feature storage separate from marker/end-point result storage;
the latter must not be multiplied by the ECG count.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from .storage import atomic_write_json, file_sha256


CAPACITY_PLAN_VERSION = "1.2.0"
GIB = 1024**3


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not a positive integer")
    integer = int(value)
    if integer <= 0 or float(value) != integer:
        raise ValueError(f"{name} is not a positive integer")
    return integer


def _development_records(master: Mapping[str, Any]) -> int:
    splits = master.get("splits")
    if not isinstance(splits, list):
        raise ValueError("Master summary has no split counts")
    counts = {
        str(row.get("split")): _positive_int(row.get("ecg_count"), "ecg_count")
        for row in splits
        if isinstance(row, Mapping)
    }
    if "train" not in counts or "validation" not in counts:
        raise ValueError("Master summary lacks train/validation counts")
    return counts["train"] + counts["validation"]


def build_capacity_plan(
    master_summary: Mapping[str, Any],
    wide_pilot_profile: Mapping[str, Any],
    *,
    free_bytes: int,
    external_records: int,
    minimum_residual_gib: float = 20.0,
    uncertainty_multiplier: float = 1.10,
    concurrent_duckdb_spill_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a fail-closed capacity budget from a full-panel pilot.

    The fixed envelopes are intentionally generous on this workstation:
    result/checkpoint tables get 12 GiB because they scale chiefly with the
    frozen hypothesis inventory, and external cohorts get at least 20 GiB.
    Feature Parquet, its validation projection, and transient rewrite space
    are separately record-scaled from the wide-panel pilot.
    """

    if (
        not math.isfinite(float(uncertainty_multiplier))
        or float(uncertainty_multiplier) < 1
        or not math.isfinite(float(minimum_residual_gib))
        or float(minimum_residual_gib) < 0
        or isinstance(external_records, bool)
        or int(external_records) != external_records
        or external_records < 0
        or isinstance(free_bytes, bool)
        or int(free_bytes) != free_bytes
        or free_bytes < 0
    ):
        raise ValueError("Invalid capacity policy or inventory counts")
    if wide_pilot_profile.get("locked_test_read") is not False:
        raise ValueError("Pilot profile does not prove an unlocked scope")
    rows = _positive_int(
        wide_pilot_profile.get("feature_partition_rows_not_verified_unique_patients"),
        "pilot feature rows",
    )
    feature_bytes = _positive_int(
        wide_pilot_profile.get("feature_partition_logical_bytes"),
        "pilot feature bytes",
    )
    partitions = wide_pilot_profile.get("feature_partitions")
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("Pilot profile has no partition metadata")
    columns = {
        _positive_int(item.get("columns"), "pilot feature columns")
        for item in partitions
        if isinstance(item, Mapping)
    }
    if len(columns) != 1 or next(iter(columns)) < 1000:
        raise ValueError("Capacity planning requires a broad full-panel pilot")
    development_records = _development_records(master_summary)
    bytes_per_record = feature_bytes / rows
    projected_features = math.ceil(bytes_per_record * development_records)
    components = {
        "development_feature_partitions": projected_features,
        # Validation creates a bounded projected cache. Budgeting a second
        # complete feature store is conservative relative to the A023 ratio.
        "validation_projection_cache": projected_features,
        # Atomic Parquet/checkpoint rewrites and partition construction need
        # temporary coexistence with their source. 1.5 stores is deliberate.
        "transient_atomic_and_partition_workspace": math.ceil(1.5 * projected_features),
        "marker_endpoint_results_and_compressed_checkpoints_envelope": 12 * GIB,
        "external_code_sami_and_challenge_workspace_envelope": max(
            20 * GIB,
            math.ceil(bytes_per_record * max(0, int(external_records))),
        ),
    }
    spill_budget_known = concurrent_duckdb_spill_bytes is not None
    if spill_budget_known:
        components["concurrent_duckdb_spill_envelope"] = _positive_int(
            concurrent_duckdb_spill_bytes, "concurrent DuckDB spill bytes"
        )
    estimated_new = sum(components.values())
    uncertainty = math.ceil(estimated_new * float(uncertainty_multiplier))
    residual = math.ceil(float(minimum_residual_gib) * GIB)
    required_free = uncertainty + residual
    capacity_passed = spill_budget_known and int(free_bytes) >= required_free
    return {
        "version": CAPACITY_PLAN_VERSION,
        "status": (
            "missing_spill_budget"
            if not spill_budget_known
            else "capacity_passed"
            if capacity_passed
            else "insufficient_capacity"
        ),
        "concurrent_duckdb_spill_budget_known": spill_budget_known,
        "required_free_bytes_is_lower_bound": not spill_budget_known,
        "scope": "development_only_storage_plan_not_runtime_or_clinical_release",
        "locked_test_read": False,
        "pilot_scope": "historical_full_panel_size_baseline_not_formula_validation",
        "development_records": development_records,
        "external_records_budgeted": max(0, int(external_records)),
        "pilot_feature_rows": rows,
        "pilot_feature_columns": next(iter(columns)),
        "pilot_feature_bytes_per_record": bytes_per_record,
        "components_bytes": components,
        "estimated_new_bytes_before_uncertainty": estimated_new,
        "uncertainty_multiplier": float(uncertainty_multiplier),
        "estimated_new_bytes_with_uncertainty": uncertainty,
        "minimum_residual_bytes": residual,
        "required_free_bytes": required_free,
        "observed_free_bytes": int(free_bytes),
        "margin_bytes": int(free_bytes) - required_free,
        "capacity_gate_passed": capacity_passed,
        "assumptions": [
            "feature_bytes_scale_linearly_from_the_broad_1943_column_pilot",
            "result_rows_scale_with_frozen_marker_endpoint_inventory_not_ecg_count",
            "compressed_checkpoint_codec_is_required_for_the_new_run",
            "no_second_immutable_copy_of_the_full_run_is_included",
            "free_space_must_be_rechecked_immediately_before_launch",
        ],
        "not_proven": [
            "runtime_duration",
            "peak_ram",
            "future_disk_contention",
            "clinical_validity",
        ],
    }


def _written_development_feature_bytes(work_root: Path) -> int:
    """Bytes of development feature partitions already on the work volume."""

    total = 0
    for path in Path(work_root).glob("features/*/development/split=*/part-*.parquet"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def require_capacity_at_start(
    run_root: Path, work_root: Path, fingerprint: str
) -> None:
    """Recheck live free space against the reviewed source-bound capacity evidence.

    Called before full-run preparation; never deletes files or adjusts a budget.
    A pilot GO, a lower-bound estimate or a historic free-space value is insufficient.
    """
    try:
        release = json.loads(
            (run_root / "scientific_release.json").read_text(encoding="utf-8")
        )
        item = release["gate_evidence"]["resource_scaling_and_storage"]
        path = (run_root / item["relative_path"]).resolve()
        if (
            Path(item["relative_path"]).is_absolute()
            or not path.is_relative_to(run_root.resolve())
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError("capacity_evidence_not_bound_to_release")
        evidence = json.loads(path.read_text(encoding="utf-8"))
        # A scientific gate has status PASS, whereas its capacity calculation
        # has status capacity_passed. They cannot occupy the same JSON field.
        # Keep legacy standalone plans readable for diagnostic callers; the
        # workflow independently requires the complete scientific release.
        if "capacity_plan" in evidence:
            from .scientific_release import gate_content_passed
            if not gate_content_passed(evidence, "resource_scaling_and_storage", fingerprint):
                raise ValueError("resource_gate_missing_stale_or_incomplete")
            plan = evidence["capacity_plan"]
        else:
            plan = evidence
        if (
            plan.get("version") != CAPACITY_PLAN_VERSION
            or plan.get("run_fingerprint") != fingerprint
            or release.get("run_fingerprint") != fingerprint
            or Path(plan["work_root"]).resolve() != work_root.resolve()
            or plan.get("status") != "capacity_passed"
            or plan.get("capacity_gate_passed") is not True
            or plan.get("locked_test_read") is not False
            or plan.get("concurrent_duckdb_spill_budget_known") is not True
            or plan.get("required_free_bytes_is_lower_bound") is not False
        ):
            raise ValueError("capacity_plan_missing_stale_or_incomplete")
        required = _positive_int(plan["required_free_bytes"], "required_free_bytes")
        components = plan["components_bytes"]
        component_sum = sum(
            _positive_int(v, "component_bytes") for v in components.values()
        )
        _positive_int(components["concurrent_duckdb_spill_envelope"], "spill_bytes")
        multiplier = float(plan["uncertainty_multiplier"])
        residual = _positive_int(plan["minimum_residual_bytes"], "residual_bytes")
        if (
            not math.isfinite(multiplier)
            or multiplier < 1
            or required != math.ceil(component_sum * multiplier) + residual
        ):
            raise ValueError("inconsistent_capacity_arithmetic")
        free = shutil.disk_usage(work_root).free
        # A resumed run has already written part of what the plan budgets.
        # Without this credit the recheck counted the run's own feature store
        # against it: A147 was refused twice on resume (203.79 and 224.18 GiB
        # free against 233.65 required) although the space it lacked was its
        # own finished partitions. Only the feature partitions are credited,
        # and never beyond the component the plan reserved for them.
        credit = min(
            _written_development_feature_bytes(work_root),
            math.ceil(
                _positive_int(
                    components["development_feature_partitions"],
                    "component_bytes",
                )
                * multiplier
            ),
        )
        if free + credit < required:
            raise ValueError(
                f"insufficient_current_space: {free / GIB:.2f} GiB free "
                f"+ {credit / GIB:.2f} GiB already written; "
                f"{required / GIB:.2f} GiB required"
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as exc:
        raise RuntimeError(
            "Full development run refused by live capacity preflight: " + str(exc)
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-summary", type=Path, required=True)
    parser.add_argument("--wide-pilot-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        help="Actual analysis volume, not the report directory",
    )
    parser.add_argument(
        "--run-fingerprint",
        help="Required when this report is used for full-run release",
    )
    parser.add_argument("--external-records", type=int, default=347_410)
    parser.add_argument(
        "--concurrent-duckdb-spill-bytes",
        type=int,
        help="Combined peak spill allowance for concurrently active connections",
    )
    args = parser.parse_args()
    master = json.loads(args.master_summary.read_text(encoding="utf-8"))
    pilot = json.loads(args.wide_pilot_profile.read_text(encoding="utf-8"))
    work_root = (args.work_root or args.output.resolve().parent).resolve()
    free = shutil.disk_usage(work_root).free
    report = build_capacity_plan(
        master,
        pilot,
        free_bytes=free,
        external_records=args.external_records,
        concurrent_duckdb_spill_bytes=args.concurrent_duckdb_spill_bytes,
    )
    report["inputs"] = {
        "master_summary": str(args.master_summary.resolve()),
        "master_summary_sha256": file_sha256(args.master_summary),
        "wide_pilot_profile": str(args.wide_pilot_profile.resolve()),
        "wide_pilot_profile_sha256": file_sha256(args.wide_pilot_profile),
    }
    report["work_root"] = str(work_root)
    report["run_fingerprint"] = args.run_fingerprint
    atomic_write_json(args.output, report)
    print(
        f"capacity={report['status']} required="
        f"{report['required_free_bytes'] / GIB:.2f} GiB free="
        f"{report['observed_free_bytes'] / GIB:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
