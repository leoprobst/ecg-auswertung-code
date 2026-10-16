"""Hash-verified, read-only GUI demonstration cases.

These capsules are intentionally outside production-run discovery. Their
annotations are not clinical ground truth and no analysis output is written
back into the capsule directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CAPSULE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "demo_cases"
PUBLIC_PTBXL_CAPSULE = CAPSULE_ROOT / "public_ptbxl_hr00001"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def verify_demo_capsule(directory: Path) -> dict:
    """Reject changed, missing or traversing files before a waveform is read."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("role") != "gui_presentation_only":
        raise ValueError("Inkompatible Demokapsel oder falsche Datenrolle")
    if manifest.get("may_enter_production_analysis") is not False or \
            manifest.get("may_be_used_as_clinical_ground_truth") is not False:
        raise ValueError("Demokapsel ohne klare Produktions-/Referenzsperre")
    stem = manifest.get("record_stem")
    files = manifest.get("files_sha256")
    if not isinstance(stem, str) or not stem.isascii() or not stem.replace("_", "").isalnum() or \
            not isinstance(files, dict) or set(files) != {f"{stem}.hea", f"{stem}.mat"}:
        raise ValueError("Ungültiger oder unvollständiger Demokapsel-Dateivertrag")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Demodatei fehlt oder verlässt die Kapsel: {name}")
        if not isinstance(expected, str) or _digest(path) != expected.lower():
            raise ValueError(f"Hashprüfung der Demodatei fehlgeschlagen: {name}")
    return manifest


def load_public_demo_signal() -> tuple[object, float, tuple[str, ...], dict]:
    """Load only the bundled public ECG after its provenance checks pass."""
    from .vcglab.analysis import load_wfdb

    manifest = verify_demo_capsule(PUBLIC_PTBXL_CAPSULE)
    signal, fs, names, _metadata = load_wfdb(str(PUBLIC_PTBXL_CAPSULE / manifest["record_stem"]))
    if fs != manifest["expected_sampling_rate_hz"] or len(signal) != manifest["expected_samples"] or \
            tuple(names) != tuple(manifest["expected_leads"]):
        raise ValueError("Demodatei stimmt trotz Hash nicht mit dem Signalvertrag überein")
    return signal, fs, names, manifest
