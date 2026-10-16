"""Streaming personal follow-up and explicit protocol-bound extension jobs.

The full workflow reuses its own development-only feature relation. Standalone
jobs use explicit hashed parquet inputs, never discover or overwrite runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from .pre_run_extensions import (
    VERSION,
    activation_residuals,
    clinical_association,
    cohort_information,
    competing_diagnosis_frame,
    long_window_summary,
    personal_deviations,
    reference_timing_summary,
    research_contract,
    window_rank_stability,
    clinical_time_sensitivity,
)
from .storage import (
    atomic_write_json,
    atomic_write_parquet,
    file_sha256,
    source_tree_hash,
)

# The QRS-T coupling markers carried the LV-systolic endpoints in the
# pilots; their within-patient course is what a serial LVEF or heart-failure
# question needs. Descriptive history only, no new hypothesis family.
QRS_T_COUPLING_MARKERS = (
    "qrs_t_normalized_discrete_frechet",
    "qrs_t_dtw_distance",
    "qrs_t_morphological_discordance",
    "qrs_t_loop_mirror_symmetry",
)

PERSONAL_MARKERS = (
    "qrs_t_area_angle_direct_8lead_deg",
    "qrs_duration_ms",
    "qtc_fridericia_ms",
    *QRS_T_COUPLING_MARKERS,
)


def _input_columns(tasks):
    """Project only consumed columns; never load archived signals into model fits."""
    columns = {"source", "subject_id", "split"}
    for task in tasks:
        kind, options = task["kind"], task.get("options", {})
        if kind not in (
            "long_windows",
            "window_rank_stability",
            "extract_long_windows",
        ):
            columns.add("study_id")
        if kind == "personal_deviation":
            columns.update(
                ["ecg_time", "rhythm_context", "transform", *options["markers"]]
            )
        elif kind == "activation_residual":
            columns.update([*options["predictors"], *options["targets"]])
        elif kind in (
            "clinical_association",
            "competing_diagnosis",
            "clinical_time_sensitivity",
        ):
            columns.update(
                [
                    options["marker"],
                    *options.get(
                        "covariates",
                        (
                            "age",
                            "sex",
                            "heart_rate_bpm",
                            "signal_quality_score",
                            "care_setting",
                        ),
                    ),
                ]
            )
            if kind == "competing_diagnosis":
                for name in (options["target"], options["competitor"]):
                    columns.update([name, name + "_verified"])
            else:
                columns.update([options["outcome"], "reference_verified"])
            if kind == "clinical_time_sensitivity":
                columns.add("reference_delta_hours")
        elif kind == "reference_timing":
            columns.update(
                [
                    options["marker"],
                    "reference_kind",
                    "reference_verified",
                    "reference_value",
                    "reference_delta_hours",
                    "setting",
                ]
            )
        elif kind == "cohort_information":
            columns.update(options["markers"])
            if options.get("event"):
                columns.add(options["event"])
        elif kind in ("long_windows", "window_rank_stability"):
            columns.update(
                ["record_id", "start_s", "duration_s", "quality", *options["markers"]]
            )
        elif kind == "extract_long_windows":
            columns.update(["record_id", "waveform_path"])
        else:
            raise ValueError("unknown_extension_task:" + str(kind))
    return sorted(columns)


def stream_personal_deviations(connection, relation, available, output_root):
    """One projected scan, bounded Arrow batches with complete-patient carry.

    Outputs are separate partitions, not one multi-million-row Python table.
    SQL is supplied only by the existing trusted feature relation builder.
    """
    required = {
        "subject_id",
        "study_id",
        "split",
        "ecg_time",
        "rhythm_context",
        "extraction_status",
    }
    markers = [m for m in PERSONAL_MARKERS if m in available]
    if not required <= set(available) or not markers:
        return [], {
            "status": "non_estimable",
            "reason": "required_history_columns_missing",
            "missing_columns": sorted(required - set(available)),
            "clinical_claim_allowed": False,
        }
    root = Path(output_root) / "personal_deviations"
    root.mkdir(parents=True, exist_ok=True)
    selection = ",".join(
        "CASE WHEN extraction_status='ok' THEN \"" + name + '" END AS "' + name + '"'
        for name in markers
    )
    reader = connection.execute(f"""SELECT 'mimic' AS source, subject_id, study_id, split, ecg_time,
        rhythm_context, 'kors' AS transform, {selection}
        FROM {relation} WHERE split IN ('train','validation')
        ORDER BY subject_id, ecg_time, study_id""").to_arrow_reader(batch_size=8192)
    pending = pd.DataFrame()
    paths: list[Path] = []
    n_rows = 0

    def save(block):
        nonlocal n_rows
        if block.empty:
            return
        result = personal_deviations(block, markers)
        destination = root / f"part-{len(paths):05d}.parquet"
        atomic_write_parquet(result, destination)
        paths.append(destination)
        n_rows += len(result)

    for batch in reader:
        current = pd.concat([pending, batch.to_pandas()], ignore_index=True)
        last = current.subject_id.iloc[-1]
        save(current.loc[current.subject_id.ne(last)])
        pending = current.loc[current.subject_id.eq(last)].copy()
    save(pending)
    return paths, {
        "status": "complete",
        "n_results": n_rows,
        "partitions": len(paths),
        "markers": markers,
        "version": VERSION,
        "clinical_claim_allowed": False,
    }


def run_extension_job(manifest_path, output_root):
    """Execute specified adapters; a JSON task is not clinical authorization.

    Schema: version, input (parquet), input_sha256, tasks (list of kind/options).
    Input must carry source-namespaced identities and development split. Clinical
    tasks require record-level verified references, not a manifest assertion.
    """
    manifest_path, destination = (
        Path(manifest_path).resolve(),
        Path(output_root).resolve(),
    )
    manifest_hash = file_sha256(manifest_path)
    backend_hash = source_tree_hash(
        Path(__file__).resolve().parents[1], patterns=("*.py",)
    )
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        spec.get("version") != VERSION
        or not isinstance(spec.get("tasks"), list)
        or not spec["tasks"]
    ):
        raise ValueError("invalid_extension_manifest")
    source = (manifest_path.parent / spec["input"]).resolve()
    if any(
        "locked" in part.lower() or part.lower() in {"test", "test_split"}
        for part in source.parts
    ):
        raise ValueError("locked_test_input_forbidden_before_read")
    if source.suffix.lower() != ".parquet" or file_sha256(source) != spec.get(
        "input_sha256"
    ):
        raise ValueError("input_hash_or_format_mismatch")
    if destination.exists():
        raise FileExistsError("extension_output_must_be_new")
    implementation = {
        p.name: file_sha256(p)
        for p in (Path(__file__), Path(__file__).with_name("pre_run_extensions.py"))
    }
    data = pd.read_parquet(source, columns=_input_columns(spec["tasks"]))
    for name in ("long_window_extraction.py", "broad_axis_adjustment.py"):
        implementation[name] = file_sha256(Path(__file__).with_name(name))
    if "split" not in data or not data.split.isin(["train", "validation"]).all():
        raise ValueError("nondevelopment_input")
    destination.mkdir(parents=True)
    report = {
        "contract": research_contract(),
        "input_sha256": spec["input_sha256"],
        "manifest_sha256": manifest_hash,
        "source_hashes": implementation,
        "backend_source_hash": backend_hash,
        "tasks": [],
        "clinical_claim_allowed": False,
        "locked_test_read": False,
    }
    try:
        tables = []
        for index, task in enumerate(spec["tasks"]):
            kind, options = task["kind"], task.get("options", {})
            details = {}
            if kind == "personal_deviation":
                table = personal_deviations(data, **options)
            elif kind == "activation_residual":
                table, details["models"] = activation_residuals(data, **options)
            elif kind == "clinical_association":
                table = clinical_association(data, **options)
            elif kind == "competing_diagnosis":
                selected, details = competing_diagnosis_frame(
                    data, target=options["target"], competitor=options["competitor"]
                )
                table = clinical_association(
                    selected,
                    marker=options["marker"],
                    outcome="comparison_outcome",
                    covariates=options.get(
                        "covariates",
                        (
                            "age",
                            "sex",
                            "heart_rate_bpm",
                            "signal_quality_score",
                            "care_setting",
                        ),
                    ),
                    binary=True,
                )
            elif kind == "reference_timing":
                table = reference_timing_summary(data, **options)
            elif kind == "clinical_time_sensitivity":
                table = clinical_time_sensitivity(data, **options)
            elif kind == "cohort_information":
                table = cohort_information(data, **options)
            elif kind == "long_windows":
                table = long_window_summary(data, **options)
            elif kind == "window_rank_stability":
                table = window_rank_stability(data, **options)
            elif kind == "extract_long_windows":
                from .long_window_extraction import wfdb_windows

                required = {
                    "source",
                    "subject_id",
                    "record_id",
                    "split",
                    "waveform_path",
                }
                if (
                    not required <= set(data)
                    or data.groupby(["source", "subject_id"])
                    .split.nunique()
                    .gt(1)
                    .any()
                ):
                    raise ValueError("invalid_long_record_inventory")
                pieces, summaries = [], []
                for record in data.itertuples():
                    windows, summary, metadata = wfdb_windows(
                        manifest_path.parent / record.waveform_path,
                        source=record.source,
                        subject_id=record.subject_id,
                        record_id=record.record_id,
                        split=record.split,
                        **options,
                    )
                    pieces.append(windows)
                    summaries.append(summary)
                    details.setdefault("records", []).append(metadata)
                table = (
                    pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
                )
                summary_path = destination / f"task-{index:03d}-record-summary.parquet"
                atomic_write_parquet(
                    pd.concat(summaries, ignore_index=True)
                    if summaries
                    else pd.DataFrame(),
                    summary_path,
                )
                details["record_summary"] = {
                    "path": summary_path.name,
                    "sha256": file_sha256(summary_path),
                }
            else:
                raise ValueError("unknown_extension_task:" + str(kind))
            path = destination / f"task-{index:03d}.parquet"
            tables.append((path, table, task.get("domain", kind)))
            report["tasks"].append(
                {
                    "kind": kind,
                    "options": options,
                    "output": path.name,
                    "n_rows": len(table),
                    "details": details,
                }
            )
        # Freeze the complete family in the manifest; failed fits enter as p=1,
        # but retain missing displayed p/q rather than fabricated significance.
        from statsmodels.stats.multitest import multipletests

        tests = [
            (
                i,
                row,
                domain,
                float(table.loc[row, "p"])
                if "p" in table and pd.notna(table.loc[row, "p"])
                else 1.0,
                "p" in table and pd.notna(table.loc[row, "p"]),
            )
            for i, (_, table, domain) in enumerate(tables)
            if report["tasks"][i]["kind"]
            in (
                "clinical_association",
                "competing_diagnosis",
                "clinical_time_sensitivity",
            )
            for row in table.index
        ]
        if tests:
            values = np.array([item[3] for item in tests])
            global_q = multipletests(values, method="fdr_bh")[1]
            for position, (i, row, domain, _, valid) in enumerate(tests):
                if valid:
                    tables[i][1].loc[row, "q_global"] = global_q[position]
            for domain in {item[2] for item in tests}:
                positions = [k for k, item in enumerate(tests) if item[2] == domain]
                qs = multipletests(values[positions], method="fdr_bh")[1]
                for position, q in zip(positions, qs):
                    i, row, _, _, valid = tests[position]
                    if valid:
                        tables[i][1].loc[row, "q_domain"] = q
        for task_report, (path, table, _) in zip(report["tasks"], tables):
            atomic_write_parquet(table, path)
            task_report["sha256"] = file_sha256(path)
        report["multiplicity"] = (
            "BH_across_manifest_associations_and_within_domain_failed_fits_count_as_one_without_displayed_p"
        )
        if (
            file_sha256(source) != spec["input_sha256"]
            or file_sha256(manifest_path) != manifest_hash
        ):
            raise RuntimeError("upstream_changed_during_analysis")
        if any(
            file_sha256(Path(__file__).with_name(name)) != digest
            for name, digest in implementation.items()
        ):
            raise RuntimeError("implementation_changed_during_analysis")
        if (
            source_tree_hash(Path(__file__).resolve().parents[1], patterns=("*.py",))
            != backend_hash
        ):
            raise RuntimeError("backend_changed_during_analysis")
        report["status"] = "complete"
        atomic_write_json(destination / "report.json", report)
    except Exception as exc:
        atomic_write_json(
            destination / "failure.json",
            {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)},
        )
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run_extension_job(args.manifest, args.output)


if __name__ == "__main__":
    main()
