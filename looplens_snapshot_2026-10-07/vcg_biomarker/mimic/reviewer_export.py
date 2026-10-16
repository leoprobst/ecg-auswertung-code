"""Local blinded signal packets; coordinator linkage must never be distributed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
from typing import Callable

import numpy as np
import pandas as pd

from ..core.data_loader import LEAD_NAMES_12
from .reference_tiers import blinded_reviewer_packet
from .storage import atomic_write_parquet

REVIEW_EXPORT_VERSION = "1.0.0"


def export_reviewer_packets(selection: pd.DataFrame, destination: Path, *,
                            signal_loader: Callable[[pd.Series], tuple[np.ndarray, float]]) -> dict:
    """Export numeric mV arrays only, never original signal headers/reports.

    The callback is a trusted local adapter receiving coordinator-only identity.
    It must return twelve channels in LEAD_NAMES_12 order and their actual rate.
    No automatic diagnosis or threshold is exported. A fresh destination is
    mandatory. Partial exports have no completion manifest and cannot be used.
    Only reviewer_packets/slot-N may be distributed, never the parent directory.
    Folder separation is not an OS permission boundary. Signals remain sensitive
    patient data, not anonymized public data.
    """
    required = {"review_case_id", "reviewer_slot", "endpoint_id", "selection_probability", "review_stratum"}
    if selection.empty or not required <= set(selection):
        raise ValueError("Nonempty coordinator selection with inclusion probabilities required")
    if selection.duplicated(["review_case_id", "reviewer_slot"]).any():
        raise ValueError("Duplicate case/reviewer assignment")
    slots = sorted(selection.reviewer_slot.unique())
    if len(slots) < 2 or any(not isinstance(s, (int, np.integer)) or isinstance(s, (bool, np.bool_)) or s < 1 for s in slots):
        raise ValueError("At least two positive integer reviewer slots required")
    if selection.review_case_id.isna().any() or selection.endpoint_id.isna().any():
        raise ValueError("Missing review case or endpoint identity")
    probabilities = pd.to_numeric(selection.selection_probability, errors="coerce")
    if not (np.isfinite(probabilities) & probabilities.gt(0) & probabilities.le(1)).all():
        raise ValueError("Invalid inclusion probabilities")
    for _, group in selection.groupby("review_case_id"):
        if set(group.reviewer_slot) != set(slots):
            raise ValueError("Every selected case must be assigned to every reviewer")
        constant_columns = [c for c in selection if c not in {"reviewer_slot"}]
        if any(group[c].nunique(dropna=False) != 1 for c in constant_columns):
            raise ValueError("Conflicting coordinator metadata for one review case")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    private = destination / "coordinator_private"
    private.mkdir()
    public = destination / "reviewer_packets"
    public.mkdir()
    # Random tokens prevent an unsalted source-ID hash from being brute-forced
    # by reviewers. Reproducibility resides in the private mapping/manifest.
    mapping = {case: secrets.token_hex(16) for case in selection.review_case_id.unique()}
    linked = selection.copy()
    linked["export_case_id"] = linked.review_case_id.map(mapping)
    atomic_write_parquet(linked, private / "selection_and_linkage.parquet")
    packet = blinded_reviewer_packet(selection)
    packet["review_case_id"] = packet.review_case_id.map(mapping)
    signal_hashes = {}
    for slot in slots:
        (public / f"slot-{slot}").mkdir()
    for case, group in selection.groupby("review_case_id", sort=True):
        values, sampling_rate = signal_loader(group.iloc[0])
        signal = np.asarray(values, dtype=float)
        if (signal.ndim != 2 or signal.shape[1] != 12 or signal.shape[0] < 2
                or not np.isfinite(signal).all() or isinstance(sampling_rate, (bool, np.bool_))
                or not np.isfinite(sampling_rate) or sampling_rate <= 0):
            raise ValueError("Invalid twelve-lead mV signal or sampling rate; export remains incomplete")
        token = mapping[case]
        # Serialize once and copy only these curated bytes to all reviewers.
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, signal_mv=signal, sampling_rate_hz=float(sampling_rate),
                            lead_names=np.asarray(LEAD_NAMES_12, dtype="U3"))
        data = buffer.getvalue()
        signal_hashes[token] = hashlib.sha256(data).hexdigest()
        for slot in slots:
            (public / f"slot-{slot}" / f"{token}.npz").write_bytes(data)
    reviewer_manifests = []
    for slot in slots:
        scoped = packet.loc[packet.reviewer_slot.eq(slot)].sort_values("review_case_id")
        scoped.to_csv(public / f"slot-{slot}" / "cases.csv", index=False)
        (public / f"slot-{slot}" / "instructions.json").write_text(json.dumps({
            "version": REVIEW_EXPORT_VERSION, "signal_unit": "mV", "time_unit": "second",
            "lead_order": list(LEAD_NAMES_12), "independent_review_required": True,
            "pipeline_results_withheld": True, "selection_strata_withheld": True,
            "clinical_use_allowed": False,
            "patient_data_remain_sensitive": True,
            "limitations": "Signal-only research review; no clinical context or adjudicated truth. Do not infer disease exclusion from missing context.",
        }, indent=2), encoding="utf-8")
        folder = public / f"slot-{slot}"
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}
        reviewer_manifests.append((folder, hashes))
    manifest = dict(version=REVIEW_EXPORT_VERSION, status="complete", n_cases=len(mapping),
                    n_reviewers=len(slots), signal_sha256=signal_hashes,
                    distribution_rule="Distribute only one reviewer_packets/slot-N directory; never coordinator_private or the export parent.",
                    clinical_claim_allowed=False)
    for folder, hashes in reviewer_manifests:
        (folder / "packet_complete.json").write_text(json.dumps({
            "version": REVIEW_EXPORT_VERSION, "status": "complete", "files_sha256": hashes,
            "clinical_use_allowed": False}, indent=2), encoding="utf-8")
    (destination / "export_complete.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
