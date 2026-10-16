"""Throughput hill climbing below the resource grant (concept item 7).

On 8 cores with SMT and load-dependent clocking, more workers do not add
throughput linearly; past a knee they only add contention. The supervisor's
grant says how many workers the machine *may* give. This governor finds how
many actually *help*: it measures completed records per window, probes one
worker more, and keeps it only when the rate rises by at least ``gain``.
A rejected probe is retried after ``hold_seconds``.

It only ever lowers the admission below the grant, never raises it, so a
wrong decision in measurement noise costs at most one worker for a few
minutes. Windows are three minutes by default: at about four records per
second a one-minute count carries ~6 % Poisson noise, too much for a 2 %
threshold; three minutes brings it to ~3.7 %, and the gains that matter below
the knee (+1 of 8 workers, ~12 %) stand well clear of it.
"""
from __future__ import annotations

import time
from typing import Callable, Optional


class ThroughputGovernor:
    def __init__(
        self,
        *,
        window_seconds: float = 180.0,
        gain: float = 0.02,
        hold_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.gain = float(gain)
        self.hold_seconds = float(hold_seconds)
        self._clock = clock
        self.cap: Optional[int] = None
        self._baseline_rate: Optional[float] = None
        self._probing = False
        self._window_start: Optional[float] = None
        self._window_workers: Optional[int] = None
        self._window_count = 0
        self._hold_until = 0.0
        self.decisions: list[tuple[str, int]] = []

    def note_completed(self, count: int = 1) -> None:
        self._window_count += int(count)

    def admit(self, granted: int) -> int:
        """Workers to admit now, given the grant."""
        granted = max(1, int(granted))
        now = self._clock()
        effective = granted if self.cap is None else min(granted, self.cap + int(self._probing))
        if self._window_start is None or effective != self._window_workers:
            # The window restarts whenever the worker count changes for any
            # reason (grant, probe); a rate is only compared at one count.
            self._start_window(now, effective)
            return effective
        if now - self._window_start < self.window_seconds:
            return effective
        rate = self._window_count / max(now - self._window_start, 1e-9)
        if self.cap is None:
            self.cap, self._baseline_rate = effective, rate
            self.decisions.append(("baseline", effective))
        elif self._probing:
            self._probing = False
            if self._baseline_rate is not None and rate >= self._baseline_rate * (1.0 + self.gain):
                self.cap, self._baseline_rate = effective, rate
                self.decisions.append(("kept", effective))
            else:
                self._hold_until = now + self.hold_seconds
                self.decisions.append(("rejected", effective))
        elif effective == self.cap:
            self._baseline_rate = rate
        if (
            not self._probing
            and self.cap is not None
            and granted > self.cap
            and now >= self._hold_until
        ):
            self._probing = True
        effective = min(granted, self.cap + int(self._probing))
        self._start_window(now, effective)
        return effective

    def _start_window(self, now: float, workers: int) -> None:
        self._window_start = now
        self._window_workers = workers
        self._window_count = 0
