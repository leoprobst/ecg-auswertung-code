from __future__ import annotations

from pathlib import Path


IMMUTABLE_DEVELOPMENT_MARKERS = (
    "freeze_manifest.json",
    "test_lock.json",
)


def immutable_development_run_markers(run_root: Path) -> tuple[Path, ...]:
    root = Path(run_root)
    return tuple(
        root / name
        for name in IMMUTABLE_DEVELOPMENT_MARKERS
        if (root / name).is_file()
    )


def assert_development_run_mutable(
    run_root: Path,
    *,
    operation: str,
) -> None:
    markers = immutable_development_run_markers(run_root)
    if not markers:
        return
    marker_names = ", ".join(path.name for path in markers)
    raise RuntimeError(
        f"Development run {Path(run_root).name!r} is immutable because "
        f"{marker_names} exists. {operation} must use a new run ID; frozen "
        "results remain read-only and must never be mixed with a new "
        "development analysis."
    )


__all__ = [
    "IMMUTABLE_DEVELOPMENT_MARKERS",
    "assert_development_run_mutable",
    "immutable_development_run_markers",
]
