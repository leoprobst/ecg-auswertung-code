"""One isolated low-priority laboratory computation; output is overlay-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path

from .pilot_viewmodel import PilotViewModel
from .pilot_workspace import confined
from .ecg_lab import ECGCaseLibrary
from .storage import atomic_write_json


@contextmanager
def single_lab_lock(overlay: Path):
    """OS-held lock is released even if a limited child process is killed."""
    overlay.mkdir(parents=True, exist_ok=True)
    with (overlay/"lab.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def analyze(workspace: Path, study_id: int, method: str, result_relative: str):
    model = PilotViewModel(workspace)
    result_path = confined(model.root / "overlay", result_relative)
    with single_lab_lock(model.root/"overlay"):
        library = ECGCaseLibrary(model.overlay_config())
        case = library.import_case(model.waveform(study_id), pseudonym=f"Pilot-{study_id}")
        payload = library.analyze_case(case.case_id, panel_ids=("primary_core", "sentinel"),
                                       vcg_method=method, acquisition_metadata=model.acquisition(study_id))
        arrays = library.root / case.case_id / "analyses" / payload["analysis_id"][:16] / "visualization_arrays.npz"
        atomic_write_json(result_path, {"payload": payload, "arrays": str(arrays), "study_id": study_id, "method": method})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--study-id", type=int, required=True)
    from ..core.vcg_transform import VCGTransformRegistry
    parser.add_argument("--method", choices=VCGTransformRegistry.available(), required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    analyze(args.workspace, args.study_id, args.method, args.result)
