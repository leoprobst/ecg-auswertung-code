"""Physical, verified pilot preservation; production is never moved or resumed.

Original files retain their bytes and provenance. New lab work belongs under
overlay/, never in original/. Absolute paths embedded in old provenance are
historical facts, not paths the workspace reader is permitted to follow.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import MimicProjectConfig
from .storage import atomic_write_json, atomic_write_parquet, file_sha256

WORKSPACE_VERSION = "1.0.0"


def confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Workspace path escapes its root")
    return path


def verify_workspace(root: Path) -> dict:
    payload = json.loads((root / "workspace.json").read_text(encoding="utf-8"))
    failures = []
    for item in payload["files"]:
        path = confined(root, item["path"])
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            failures.append(item["path"])
    return {"valid": not failures, "failed_files": failures,
            "file_count": len(payload["files"])}


def seal_workspace(root: Path) -> dict:
    """Apply read-only file attributes only to verified copied originals.

    This prevents accidental writes, not deliberate alteration by the owner;
    SHA-256 integrity verification remains the scientific protection.
    """
    verified = verify_workspace(root)
    if not verified["valid"]:
        raise ValueError("Cannot seal an altered workspace")
    payload = json.loads((root/"workspace.json").read_text(encoding="utf-8"))
    paths = [confined(root, item["path"]) for item in payload["files"]]
    original = (root/"original").resolve()
    if any(not path.is_relative_to(original) for path in paths):
        raise ValueError("Only copied original artifacts may be sealed")
    for path in paths:
        path.chmod(stat.S_IREAD)
    return {"sealed_files": len(paths), "byte_integrity_valid": True,
            "protection": "read_only_attribute_plus_sha256_not_owner_tamperproof"}


def create_pilot_workspace(
    config: MimicProjectConfig, run_id: str, *,
    source_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    if not re.fullmatch(r"A\d+-[A-Za-z0-9-]+", run_id):
        raise ValueError("Invalid pilot identifier")
    run = config.runs_root / run_id
    workflow = json.loads((run / "guided_workflow.manifest.json").read_text())
    if workflow.get("status") != "complete":
        raise ValueError("Only a completed pilot may be preserved")
    destination = config.work_root / "pilot_workspaces" / run_id
    if destination.exists():
        if (destination / "workspace.json").is_file() and verify_workspace(destination)["valid"]:
            return destination
        raise FileExistsError("Incomplete or altered workspace; never overwrite it")
    inputs: list[tuple[Path, str]] = [(run, f"runs/{run_id}")]
    selections = []
    for count in (1000, 10000):
        pilot = config.runs_root / f"{run_id}-pilot-{count}"
        provenance = json.loads((pilot / "development_provenance.json").read_text())
        if provenance.get("locked_test_labels_read") is not False:
            raise PermissionError("Pilot provenance does not exclude locked-test access")
        inputs.append((pilot, f"runs/{pilot.name}"))
        features = [p for p in config.features_root.glob(f"*/smoke-development-{count}")
                    if (p / "features.manifest.json").is_file()]
        if len(features) != 1:
            raise ValueError(f"Ambiguous feature root for pilot {count}")
        inputs.append((features[0], f"features/{count}"))
        for path in features[0].glob("split=*/*.parquet"):
            frame = pd.read_parquet(path, columns=["subject_id", "study_id", "split"])
            if not frame["split"].isin(["train", "validation"]).all():
                raise PermissionError("Non-development rows in pilot")
            selections.append(frame)
    selected = pd.concat(selections, ignore_index=True).drop_duplicates()
    files = [(p, str(Path(relative) / p.relative_to(directory)))
             for directory, relative in inputs for p in directory.rglob("*") if p.is_file()]
    if source_root is not None:
        files.extend((p, str(Path("source/vcg_biomarker") / p.relative_to(source_root / "vcg_biomarker")))
                     for p in (source_root / "vcg_biomarker").rglob("*.py"))
        files.extend((source_root / name, f"source/{name}") for name in
                     ("pyproject.toml", "requirements-lock-win-py312.txt")
                     if (source_root / name).is_file())
    files.append((config.project_file, "source/project.json"))
    estimated = sum(p.stat().st_size for p, _ in files) + len(selected) * 200_000
    if shutil.disk_usage(config.work_root).free < estimated * 1.2:
        raise OSError("Insufficient space for verified pilot copy plus safety margin")
    destination.mkdir(parents=True, exist_ok=False)
    original = destination / "original"
    inventory = []

    def record(path: Path) -> None:
        inventory.append({"path": path.relative_to(destination).as_posix(),
                          "bytes": path.stat().st_size, "sha256": file_sha256(path)})

    def copy(src: Path, relative: str) -> None:
        target = confined(original, relative)
        if src.is_symlink():
            raise ValueError(f"Symlink refused: {src}")
        target.parent.mkdir(parents=True, exist_ok=True)
        before = src.stat()
        digest = file_sha256(src)
        shutil.copy2(src, target)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_sha256(target) != digest:
            raise RuntimeError(f"Source changed or copy verification failed: {src}")
        inventory.append({"path": target.relative_to(destination).as_posix(),
                          "bytes": after.st_size, "sha256": digest})

    for index, (src, relative) in enumerate(files):
        copy(src, relative)
        if progress and index % 100 == 0:
            progress(f"Artifact copy {index + 1}/{len(files)}")
    from .storage import isolated_analysis_connection
    with isolated_analysis_connection(config) as connection:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.register("selected", selected)
        for relative in ("master/ecg_master_index.parquet", "evidence/clinical_context_features.parquet",
                         "labels/clinical_label_evidence.parquet", "labels/endpoint_index_cohorts.parquet"):
            source = config.work_root / relative
            scoped = connection.execute(
                "SELECT t.* FROM read_parquet(?) t SEMI JOIN selected s "
                "ON t.subject_id=s.subject_id AND t.study_id=s.study_id "
                "WHERE t.split IN ('train','validation')", [str(source)],
            ).fetchdf()
            target = confined(original, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(scoped, target)
            record(target)
        master = pd.read_parquet(original / "master/ecg_master_index.parquet")
        wave_rows = []
        for index, row in enumerate(master.itertuples(index=False)):
            raw = Path(row.waveform_path)
            if not raw.is_absolute():
                raw = config.mimic_ecg_root / raw
            raw = raw.resolve()
            if not raw.is_relative_to(config.mimic_ecg_root.resolve()):
                raise ValueError("Raw waveform outside configured ECG root")
            waveform_relative = raw.relative_to(config.mimic_ecg_root.resolve())
            for suffix in (".hea", ".dat"):
                copy(raw.with_suffix(suffix), str(Path("waveforms") / waveform_relative.with_suffix(suffix)))
            wave_rows.append({"subject_id": row.subject_id, "study_id": row.study_id,
                              "waveform": (Path("original/waveforms") / waveform_relative).as_posix()})
            if progress and index % 1000 == 0:
                progress(f"Waveform copy {index + 1}/{len(master)}")
        path = original / "case_index.parquet"
        atomic_write_parquet(pd.DataFrame(wave_rows), path)
        record(path)
    for name in ("cases", "exports", "scratch", "reviews"):
        (destination / "overlay" / name).mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "workspace.json", {
        "version": WORKSPACE_VERSION, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "historical_pilot_research_only", "clinical_claim_allowed": False,
        "production_resume_allowed": False, "locked_test_read": False,
        "original_read_only": True, "overlay_root": "overlay",
        "lab_workers": 1, "lab_memory_gb": 2, "selected_records": len(selected),
        "pilot_sizes": [1000, 10000], "files": inventory,
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify", "seal"))
    parser.add_argument("--project", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_workspace(args.workspace)
        print(json.dumps(result))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "seal":
        print(json.dumps(seal_workspace(args.workspace)))
    else:
        root = create_pilot_workspace(MimicProjectConfig.load(args.project), args.run_id,
                                     source_root=Path(__file__).resolve().parents[2],
                                     progress=lambda line: print(line, flush=True))
        print(root)


if __name__ == "__main__":
    main()
