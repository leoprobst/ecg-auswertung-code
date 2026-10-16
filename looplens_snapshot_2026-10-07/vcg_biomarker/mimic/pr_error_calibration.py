"""External annotated beat audit; reference disagreement is not latent truth.

LUDB has already informed this delineator. These artifacts are calibration
candidates, never automatic permission to correct MIMIC measurements.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

VERSION = "1.0.0"
MEASUREMENT_VERSION = "beat_PR_QRS_energy_record_delineation_kors_500Hz_v1"
REASONS_NOT_APPROVED = (
    "LUDB_previously_used_in_delineator_development_not_untouched_validation",
    "manual_reference_error_not_separately_identified",
    "within_record_error_stationarity_and_transfer_to_MIMIC_unverified",
    "typical_10_second_records_do_not_supply_20_qualified_beats",
)


def geometry_value(xyz, onset, offset, fs):
    """Use the existing numerical marker kernel, in original beat coordinates."""
    from ..core.biomarkers.advanced_vcg_biomarkers import _qrs_initial_terminal_energy_balance
    if not 0 <= onset < offset < len(xyz) or offset - onset < round(.04 * fs):
        return None
    points = np.asarray(xyz[onset:offset + 1], dtype=float)
    if not np.isfinite(points).all():
        return None
    ctx = SimpleNamespace(fs=fs, vcg_segment=lambda _: points)
    return _qrs_initial_terminal_energy_balance(ctx).value


def match_reference_beats(references):
    """At most one preceding P per QRS; no pairing across an intervening QRS."""
    qrs = sorted(references["qrs"], key=lambda wave: wave.peak)
    result = []
    for i, wave in enumerate(qrs):
        previous_end = qrs[i-1].offset if i else -1
        p = [p for p in references["p"] if previous_end < p.onset < p.offset < wave.onset]
        if len(p) == 1:
            result.append((p[0], wave))
    return result


def measure_record(root, relative, *, role):
    from ..core.delineation.benchmark import _load_ludb_signal, load_ludb_global_references, LUDB_ANNOTATORS
    from ..core.delineation.fusion_delineator import FusionDelineator
    from ..core.preprocessing import prepare_signal_paths, detect_r_peaks_multilead
    from ..core.vcg_transform import VCGTransformRegistry
    from ..core.biomarkers.thesis_derived_biomarkers import _beat_sample
    from ..core.delineation.base import measurement_confidence_threshold
    path = Path(root) / relative
    signal, fs = _load_ludb_signal(path)
    if fs != 500:
        raise ValueError("Calibration protocol requires native 500 Hz")
    paths = prepare_signal_paths(signal, fs, powerline=50.)
    xyz = VCGTransformRegistry.transform(paths["delineation"], "kors").xyz
    peaks = detect_r_peaks_multilead(paths["delineation"], fs)
    predicted = FusionDelineator().delineate_record(paths["delineation"], peaks, fs,
                                                  ecg_id=int(path.name), vcg_xyz=xyz)
    reference, _ = load_ludb_global_references(path, fs)
    matches = match_reference_beats(reference)
    # Mutual uniqueness within a fixed 100 ms peak tolerance; no best-error match.
    candidate_indices = [[j for j, beat in enumerate(predicted.beats)
                          if abs(beat.r_peak.sample - qrs.peak) <= .1 * fs]
                         for _, qrs in matches]
    uses = {j: sum(j in ids for ids in candidate_indices) for ids in candidate_indices for j in ids}
    rows = []
    for (p, qrs), ids in zip(matches, candidate_indices):
        row = {"record_id": relative, "role": role, "reference_r_sample": qrs.peak,
               "reference_p_onset": p.onset, "reference_qrs_onset": qrs.onset,
               "reference_qrs_offset": qrs.offset, "p_annotation_leads": p.lead_count,
               "qrs_annotation_leads": qrs.lead_count, "fs": fs,
               "status": "non_estimable", "reason": "ambiguous_or_missing_beat_match"}
        if len(ids) == 1 and uses[ids[0]] == 1:
            beat = predicted.beats[ids[0]]
            po = _beat_sample(beat, "p_onset", "p")
            qo = _beat_sample(beat, "r_onset", "qrs")
            qe = _beat_sample(beat, "r_offset", "qrs")
            quality = all(np.isfinite(beat.segment_confidence(wave)) and
                measurement_confidence_threshold(wave) <= beat.segment_confidence(wave) <= 1
                for wave in ("p", "qrs"))
            row.update(predicted_p_onset=po, predicted_qrs_onset=qo, predicted_qrs_offset=qe)
            row["reason"] = "boundary_quality_or_reference_coverage"
            if (quality and None not in (po, qo, qe) and po < qo < qe
                    and min(p.lead_count, qrs.lead_count) >= 6
                    and min(p.onset, po) >= .25 * fs and max(qrs.offset, qe) < len(signal) - .25 * fs):
                observed_g = geometry_value(xyz, qo, qe, fs)
                reference_g = geometry_value(xyz, qrs.onset, qrs.offset, fs)
                if observed_g is not None and reference_g is not None:
                    observed_pr = (qo - po) / fs * 1000
                    reference_pr = (qrs.onset - p.onset) / fs * 1000
                    row.update(status="measured", reason="", pr_ms=observed_pr,
                        reference_pr_ms=reference_pr, geometry=observed_g, reference_geometry=reference_g,
                        pr_error_ms=observed_pr-reference_pr, geometry_error=observed_g-reference_g)
        rows.append(row)
    files = [path.with_suffix(ext) for ext in (".hea", ".dat", *("."+a for a in LUDB_ANNOTATORS))]
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files if file.is_file()}
    return rows, {"record_id": relative, "status": "processed", "role": role,
                  "n_reference_pairs": len(matches), "n_predicted_beats": len(predicted.beats), "files": hashes}


def estimate_error_moments(beats, *, bootstrap=500, seed=731):
    """Demeaned within-record discrepancy covariance, record-cluster bootstrap.

    Different leads annotate the SAME recording: never treat them as independent
    subjects or repeated noise measurements. Constant per-record bias is reported
    separately and excluded from the within-record variance used for a slope.
    """
    selected = beats.loc[beats.status.eq("measured")].copy()
    blocks, biases = [], []
    for _, group in selected.groupby("record_id"):
        values = group[["pr_error_ms", "geometry_error"]].to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all():
            continue
        centered = values - values.mean(axis=0)
        moment = centered.T @ centered
        blocks.append([len(values)-1, moment[0, 0], moment[0, 1], moment[1, 1]])
        biases.append(values.mean(axis=0))
    if len(blocks) < 30:
        return {"status": "non_estimable", "reason": "fewer_than_30_annotated_records",
                "n_records": len(blocks), "approved_for_correction": False}
    blocks = np.asarray(blocks)
    total = blocks.sum(axis=0)
    covariance = total[1:] / total[0]
    rng = np.random.default_rng(seed)
    resampled = blocks[rng.integers(len(blocks), size=(bootstrap, len(blocks)))].sum(axis=1)
    boot_values = resampled[:, 1:] / resampled[:, :1]
    limits = np.quantile(boot_values, [.025, .975], axis=0)
    return {"status": "candidate_reference_discrepancy", "n_records": len(blocks),
            "within_record_df": int(total[0]), "pr_error_variance_ms2": float(covariance[0]),
            "error_covariance": float(covariance[1]), "geometry_error_variance": float(covariance[2]),
            "record_cluster_bootstrap_95ci": limits.T.tolist(),
            "record_balanced_bias": np.mean(biases, axis=0).tolist(),
            "approved_for_correction": False, "bootstrap_iterations": bootstrap, "seed": seed,
            "limitation": "reference_disagreement_covariance_not_identified_latent_measurement_error"}


def run_calibration(database_root, output_root, *, max_records=None, workers=2):
    from concurrent.futures import ThreadPoolExecutor
    from ..core.delineation.benchmark import resolve_ludb_root, _load_ludb_metadata
    from .feature_extraction import feature_implementation_hash
    from .storage import atomic_write_json, atomic_write_parquet
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    root, output = resolve_ludb_root(database_root), Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    records = [line.strip() for line in (root / "RECORDS").read_text().splitlines() if line.strip()]
    metadata = _load_ludb_metadata(root / "ludb.csv")
    eligible, excluded = [], []
    for relative in records:
        info = metadata.get(Path(relative).name, {})
        # Explicit clinical sinus annotation, no documented pacing or extrasystoles.
        ok = (info.get("Rhythms", "").strip().lower() in {
                  "sinus rhythm", "sinus bradycardia", "sinus tachycardia", "sinus arrhythmia", "irregular sinus rhythm"} and
              not any(term in info.get("Conduction abnormalities", "").lower()
                      for term in ("iii degree", "ii degree", "dissociation")) and
              not info.get("Cardiac pacing", "").strip() and not info.get("Extrasystolies", "").strip())
        (eligible if ok else excluded).append(relative)
    if max_records is not None:
        eligible = eligible[:max_records]
    start_hash = feature_implementation_hash()
    def work(relative):
        role = "calibration" if int(hashlib.sha256(relative.encode()).hexdigest(), 16) % 2 == 0 else "check"
        try:
            return measure_record(root, relative, role=role)
        except Exception as exc:
            return [], {"record_id": relative, "role": role, "status": "failed",
                        "error_type": type(exc).__name__, "reason": str(exc)}
    all_beats, audits = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        for i, (rows, audit) in enumerate(pool.map(work, eligible), 1):
            all_beats.extend(rows)
            audits.append(audit)
            print(f"PR calibration {i}/{len(eligible)}: {audit['status']}", flush=True)
    beats = pd.DataFrame(all_beats)
    for name in ("status", "role", "record_id", "pr_error_ms", "geometry_error"):
        if name not in beats:
            beats[name] = pd.Series(dtype=str if name in ("status", "role", "record_id") else float)
    end_hash = feature_implementation_hash()
    if start_hash != end_hash:
        raise RuntimeError("Feature implementation changed during calibration")
    summary = {"version": VERSION, "measurement_version": MEASUREMENT_VERSION,
        "feature_implementation_hash": start_hash, "dataset": "LUDB-1.0.1",
        "dataset_root": str(root.resolve()), "transform": "kors", "fs": 500,
        "signal_path": "record_delineation_50Hz_not_median_template",
        "source": "https://physionet.org/content/ludb/1.0.1/",
        "metadata_sha256": hashlib.sha256((root / "ludb.csv").read_bytes()).hexdigest(),
        "n_source_records": len(records), "n_selected_records": len(eligible),
        "n_excluded_metadata": len(excluded), "max_records": max_records,
        "n_matched_qualified_beats": int(beats.status.eq("measured").sum()),
        "n_failed_records": sum(a["status"] == "failed" for a in audits),
        "approved_for_correction": False, "clinical_claim_allowed": False,
        "not_approved_reasons": list(REASONS_NOT_APPROVED),
        "roles_are_not_untouched_validation": True,
        "calibration": estimate_error_moments(beats.loc[beats.role.eq("calibration")]),
        "check": estimate_error_moments(beats.loc[beats.role.eq("check")])}
    atomic_write_parquet(beats, output / "annotated_beat_errors.parquet")
    atomic_write_json(output / "records.json", audits)
    atomic_write_json(output / "calibration_candidate.json", summary)
    # Blinded annotation worksheet, no predictions or errors exposed to reviewers.
    review = beats[["record_id", "reference_r_sample"]].drop_duplicates().copy() if len(beats) else pd.DataFrame()
    for name in ("reviewer_id", "p_onset", "qrs_onset", "qrs_offset", "rhythm", "pacing", "review_comment"):
        review[name] = None
    atomic_write_parquet(review, output / "independent_review_template.parquet")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run_calibration(args.database_root, args.output_root, max_records=args.max_records, workers=args.workers)
