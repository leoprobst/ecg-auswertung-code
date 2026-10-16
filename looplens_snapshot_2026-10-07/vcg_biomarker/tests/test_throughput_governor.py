"""Concept item 7: hill climbing finds the throughput knee below the grant."""
from vcg_biomarker.mimic.throughput_governor import ThroughputGovernor


def _run(governor, grant_of, rate_of, seconds, step=5.0):
    history = []
    for _ in range(int(seconds / step)):
        workers = governor.admit(grant_of())
        governor.note_completed(int(round(rate_of(workers) * step)))
        _CLOCK[0] += step
        history.append(workers)
    return history


_CLOCK = [0.0]


def test_governor_climbs_to_the_knee_and_never_exceeds_the_grant():
    _CLOCK[0] = 0.0
    governor = ThroughputGovernor(window_seconds=60, gain=0.02, hold_seconds=120,
                                  clock=lambda: _CLOCK[0])
    grant = {"value": 4}
    knee = lambda workers: 10.0 * min(workers, 6)   # nothing gained beyond 6
    _run(governor, lambda: grant["value"], knee, 400)
    assert governor.cap == 4                          # held at the grant
    grant["value"] = 10
    history = _run(governor, lambda: grant["value"], knee, 3000)
    assert governor.cap == 6
    assert max(history) <= 10
    assert ("rejected", 7) in governor.decisions
    # A lower grant always wins.
    grant["value"] = 3
    assert governor.admit(3) == 3
