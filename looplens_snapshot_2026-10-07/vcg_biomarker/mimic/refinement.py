"""One control surface for every open-ended stage.

Two stages have no natural end. The combination search can always go one level
deeper, and the marker bootstrap can always draw more resamples. Both are meant
to be started, watched, stopped when the user has seen enough, and continued
later without discarding what was already computed.

They were built separately, so they had separate stop files, separate live
status and separate buttons. From a user's seat that is one activity -- "keep
refining this run" -- and it should have one control. This module owns that
control; the stages call into it rather than each keeping their own.

Three things are deliberate:

* **The stop file keeps its old name.** ``STOP_DEEP_SEARCH`` predates this
  module. Renaming it would mean a stop written by a new GUI is invisible to a
  search already running from the old code -- the one moment when a stop button
  has to work. The name is historical; its meaning is now "stop refining".

* **Stopping is a request, not a kill.** The modes differ in what they promise
  to finish first, because that decides whether the result set stays comparable
  across endpoints. Ending mid-sweep leaves early endpoints deeply searched and
  later ones untouched, and "nothing found" then looks the same as "never
  looked".

* **Progress is reported as a position and a gain, never as a percentage.**
  There is no denominator: the work is unbounded by design. What a user needs
  in order to decide whether to stop is not "how far along" but "is it still
  getting better", and that is what the gain says.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .storage import atomic_write_json


REFINEMENT_CONTROL_VERSION = "1.0.0"

# Historical name, kept on purpose -- see the module docstring.
STOP_FILE_NAME = "STOP_DEEP_SEARCH"
LIVE_FILE_NAME = "refinement_live.json"
# The deep search published here before this module existed. Still read so a
# GUI showing live state does not go blank against an older running stage.
LEGACY_LIVE_FILE_NAME = "deep_search_live.json"

STOP_MODE_IMMEDIATE = "immediate"
STOP_MODE_AFTER_ENDPOINT = "after_endpoint"
STOP_MODE_AFTER_SWEEP = "after_sweep"
STOP_MODES: tuple[str, ...] = (
    STOP_MODE_IMMEDIATE,
    STOP_MODE_AFTER_ENDPOINT,
    STOP_MODE_AFTER_SWEEP,
)
DEFAULT_STOP_MODE = STOP_MODE_AFTER_SWEEP

# Which open-ended stages share this control.
ACTIVITY_COMBINATION_SEARCH = "deep-combination-search"
ACTIVITY_VALIDATION_REFINEMENT = "validate"


def stop_file_requested(run_root: Path) -> bool:
    return (Path(run_root) / STOP_FILE_NAME).is_file()


def requested_stop_mode(run_root: Path) -> Optional[str]:
    """Which kind of stop was asked for, if any."""

    path = Path(run_root) / STOP_FILE_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = str(payload.get("mode", DEFAULT_STOP_MODE))
    except (OSError, ValueError):
        # A hand-written or truncated stop file carries no mode. Reading it as
        # the safe one is better than reading it as the abrupt one.
        return DEFAULT_STOP_MODE
    return mode if mode in STOP_MODES else DEFAULT_STOP_MODE


def request_stop(
    run_root: Path,
    *,
    mode: str = DEFAULT_STOP_MODE,
    requested_by: str = "user",
) -> Path:
    """Ask the running refinement to end, in the given way."""

    if mode not in STOP_MODES:
        raise ValueError(
            f"Unknown stop mode: {mode!r}; expected one of {STOP_MODES}"
        )
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / STOP_FILE_NAME
    atomic_write_json(
        path,
        {
            "mode": mode,
            "requested_by": requested_by,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "control_version": REFINEMENT_CONTROL_VERSION,
        },
    )
    return path


def clear_stop(run_root: Path) -> None:
    """Remove a stop request so the run can be continued later.

    Continuing is the normal case for these stages -- run a month, add weeks --
    so the stop request must not outlive the stop it caused.
    """

    (Path(run_root) / STOP_FILE_NAME).unlink(missing_ok=True)


@dataclass
class RefinementActivity:
    """What an open-ended stage is doing right now, and whether it still pays.

    ``position``/``total`` describe where the sweep is, not how much work
    remains -- there is no such number. ``gain`` is the honest answer to "should
    I keep going": how much the last increment actually moved the result.
    """

    stage: str
    unit_label: str
    position: int = 0
    total: int = 0
    detail: str = ""
    level: Optional[int] = None
    gain: Optional[float] = None
    gain_label: str = ""
    estimated_next_seconds: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "control_version": REFINEMENT_CONTROL_VERSION,
            "stage": self.stage,
            "unit_label": self.unit_label,
            "position": int(self.position),
            "total": int(self.total),
            "detail": self.detail,
            "level": self.level,
            "gain": self.gain,
            "gain_label": self.gain_label,
            "estimated_next_seconds": self.estimated_next_seconds,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(self.extra)
        return payload


def publish_activity(run_root: Path, activity: RefinementActivity) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LIVE_FILE_NAME
    atomic_write_json(path, activity.to_dict())
    return path


def read_activity(run_root: Path) -> Optional[dict]:
    """Live state, falling back to the deep search's older file."""

    root = Path(run_root)
    for name in (LIVE_FILE_NAME, LEGACY_LIVE_FILE_NAME):
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unbekannt"
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} Tage"


def describe_activity(state: Optional[Mapping[str, Any]]) -> str:
    """One readable line for a status bar or a terminal board."""

    if not state:
        return "Keine offene Verfeinerung aktiv."

    stage = str(state.get("stage") or state.get("stage_id") or "")
    label = {
        ACTIVITY_COMBINATION_SEARCH: "Tiefensuche",
        ACTIVITY_VALIDATION_REFINEMENT: "Validierung",
    }.get(stage, stage or "Verfeinerung")

    unit = str(state.get("unit_label") or "Endpunkt")
    position = int(state.get("position") or state.get("endpoint_position") or 0)
    total = int(state.get("total") or state.get("endpoints_total") or 0)
    parts = [f"{label}: {unit} {position:,}/{total:,}" if total else f"{label}: laeuft"]

    detail = str(state.get("detail") or state.get("endpoint_id") or "")
    if detail:
        parts.append(detail)

    level = state.get("level", state.get("current_depth"))
    if level is not None:
        parts.append(f"Stufe {level}")

    gain = state.get("gain")
    if gain is not None:
        gain_label = str(state.get("gain_label") or "Gewinn")
        try:
            parts.append(f"{gain_label} {float(gain):.5f}")
        except (TypeError, ValueError):
            pass

    next_seconds = state.get(
        "estimated_next_seconds", state.get("estimated_next_level_seconds")
    )
    if next_seconds is not None:
        parts.append(f"naechste Stufe etwa {_format_duration(next_seconds)}")

    return " · ".join(parts)


def refinement_has_saturated(
    gain: Optional[float],
    *,
    threshold: float,
) -> Optional[bool]:
    """Whether the last increment moved the result less than ``threshold``.

    Returns None when there is nothing to compare yet. This is advisory: it
    tells a user that more time is unlikely to change anything, and never stops
    a run on its own. Deciding that a result is good enough is not a decision
    a progress reporter should be making.
    """

    if gain is None:
        return None
    try:
        return float(gain) < float(threshold)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ACTIVITY_COMBINATION_SEARCH",
    "ACTIVITY_VALIDATION_REFINEMENT",
    "DEFAULT_STOP_MODE",
    "LIVE_FILE_NAME",
    "REFINEMENT_CONTROL_VERSION",
    "RefinementActivity",
    "STOP_FILE_NAME",
    "STOP_MODES",
    "STOP_MODE_AFTER_ENDPOINT",
    "STOP_MODE_AFTER_SWEEP",
    "STOP_MODE_IMMEDIATE",
    "clear_stop",
    "describe_activity",
    "publish_activity",
    "read_activity",
    "refinement_has_saturated",
    "request_stop",
    "requested_stop_mode",
    "stop_file_requested",
]
