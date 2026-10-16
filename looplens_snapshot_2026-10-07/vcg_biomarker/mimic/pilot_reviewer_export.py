"""Blinded exports from hash-verified archived pilot signals, overlay only."""
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .pilot_viewmodel import PilotViewModel
from .pilot_workspace import confined
from .reviewer_export import export_reviewer_packets
from .signals import _strict_lead_indices, _convert_to_millivolts


def export_pilot_review(workspace: Path, selection: pd.DataFrame, *, export_name: str) -> dict:
    model = PilotViewModel(workspace)
    destination = confined(model.root / "overlay" / "review_exports", export_name)
    if not {"study_id", "subject_id"} <= set(selection):
        raise ValueError("Pilot selection must include coordinator study/patient identity")
    cases = model.cases()
    if cases.study_id.duplicated().any():
        raise ValueError("Ambiguous archived study identity")
    selected = selection.copy()
    lookup = cases.set_index("study_id")
    paths = {}
    for _, row in selected.drop_duplicates(["subject_id", "study_id"]).iterrows():
        if (row.study_id not in lookup.index or pd.isna(row.subject_id)
                or lookup.loc[row.study_id, "subject_id"] != row.subject_id):
            raise ValueError("Selected case does not match archived patient/study identity")
        paths[row.study_id] = model.waveform(int(row.study_id))
    selected["signal_export_processing"] = "native_sampling_strict_lead_order_unit_conversion_only_no_filter_or_gap_repair"
    selected["source_pilot_run_id"] = model.run_id

    def signal_loader(row):
        path = paths[row.study_id]
        stem = path.with_suffix("") if path.suffix in {".hea", ".dat"} else path
        record = wfdb.rdrecord(str(stem))
        indices = _strict_lead_indices(record.sig_name)
        if record.p_signal is None:
            raise ValueError("Archived record has no physical signal")
        signal = np.asarray(record.p_signal, dtype=float)[:, indices]
        units = tuple(str(record.units[i] or "") for i in indices)
        converted = _convert_to_millivolts(signal, units)
        # Unlike the analytical loader this adapter does not resample, filter
        # or repair gaps: invalid signals abstain in the export contract.
        model.waveform(int(row.study_id))  # reverify after decoding
        return converted, float(record.fs)

    return export_reviewer_packets(selected, destination, signal_loader=signal_loader)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--selection", required=True,
                        help="Coordinator selection Parquet path relative to the pilot overlay")
    parser.add_argument("--export-name", required=True,
                        help="Fresh directory name inside overlay/review_exports")
    args = parser.parse_args()
    import os
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    selection_path = confined(args.workspace.resolve() / "overlay", args.selection)
    result = export_pilot_review(args.workspace, pd.read_parquet(selection_path), export_name=args.export_name)
    print(json.dumps({key: result[key] for key in ("version", "status", "n_cases", "n_reviewers")}, indent=2))
