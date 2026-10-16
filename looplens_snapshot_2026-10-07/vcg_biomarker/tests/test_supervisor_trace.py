"""``jobs/supervisor.log`` must stop being an empty promise.

The file was created in July, opened for the child's output, and stayed at
zero bytes because the supervisor printed nothing. That is worse than having
no file: it answers "is this being recorded?" with a yes. The supervisor is
the component that decides when a month-long run pauses and when it resumes,
so those two lines are the ones it owes -- on 2026-08-26 a run stopped five
times before ten in the morning and nothing said so.
"""

from __future__ import annotations

from vcg_biomarker.mimic.supervisor import (
    MEMORY_PAUSE_MARKER,
    TRACE_INTERVAL_SECONDS,
)


class _Recorder:
    """A supervisor stripped to the tracing behaviour under test."""

    def __init__(self, clock):
        self.lines: list[str] = []
        self._clock = clock
        self._last_trace_at = 0.0

    _trace = None  # bound below


def _supervisor_with_captured_trace(monkeypatch, clock):
    from vcg_biomarker.mimic import supervisor as module

    recorder = _Recorder(clock)
    recorder._trace = lambda message: recorder.lines.append(message)
    recorder._trace_periodic = module.LocalJobSupervisor._trace_periodic.__get__(
        recorder
    )
    return recorder


class _Spec:
    def __init__(self, stage):
        self.stage = stage


class _Job:
    def __init__(self, stage):
        self.spec = _Spec(stage)


class _Memory:
    def __init__(self, available_gb):
        self.available_gb = available_gb


def test_the_routine_line_names_the_free_memory_and_the_running_stages(
    monkeypatch,
):
    now = [1000.0]
    sup = _supervisor_with_captured_trace(monkeypatch, lambda: now[0])
    sup._trace_periodic([_Job("extract-features")], _Memory(9.5))
    assert len(sup.lines) == 1
    assert "9.50 GB frei" in sup.lines[0]
    assert "extract-features" in sup.lines[0]


def test_the_routine_line_is_rate_limited(monkeypatch):
    now = [1000.0]
    sup = _supervisor_with_captured_trace(monkeypatch, lambda: now[0])
    sup._trace_periodic([_Job("validate")], _Memory(9.5))
    now[0] += TRACE_INTERVAL_SECONDS - 1
    sup._trace_periodic([_Job("validate")], _Memory(9.4))
    assert len(sup.lines) == 1, "eine Zeile alle zehn Minuten, nicht je Runde"

    now[0] += 2
    sup._trace_periodic([_Job("validate")], _Memory(9.3))
    assert len(sup.lines) == 2


def test_an_unknown_memory_state_is_said_not_swallowed(monkeypatch):
    now = [1000.0]
    sup = _supervisor_with_captured_trace(monkeypatch, lambda: now[0])
    sup._trace_periodic([], None)
    assert "Speicher unbekannt" in sup.lines[0]
    assert "keine" in sup.lines[0]


def test_a_broken_pipe_does_not_take_the_supervisor_down(monkeypatch, capsys):
    """It keeps a month-long study alive; it must not die for a log line."""

    from vcg_biomarker.mimic import supervisor as module

    def explode(*_args, **_kwargs):
        raise OSError("pipe closed")

    monkeypatch.setattr("builtins.print", explode)
    trace = module.LocalJobSupervisor._trace
    trace(object(), "eine Zeile")  # must not raise


def test_the_pause_marker_is_what_the_resume_pass_matches():
    """The two halves must agree, or a pause is never lifted."""

    import inspect

    from vcg_biomarker.mimic import supervisor as module

    source = inspect.getsource(module.LocalJobSupervisor)
    assert source.count("MEMORY_PAUSE_MARKER") >= 2
    assert MEMORY_PAUSE_MARKER.startswith("Automatische Pause")
