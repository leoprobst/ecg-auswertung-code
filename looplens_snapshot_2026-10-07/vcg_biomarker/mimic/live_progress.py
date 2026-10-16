"""Live terminal report: what the analysis is doing, right now.

A month-long run that only prints a spinner is unmonitorable. This renders a
compact, self-refreshing status board on stdout:

* which stage is running and, in plain English, what that stage actually does;
* progress inside that stage, and progress across the whole run;
* which stages are finished, how long each took, and what is still queued;
* an estimate of remaining time, weighted by how expensive each stage is
  rather than by counting stages equally.

Deliberately terminal-only. The GUI already has a job view; a second one would
be a second thing to keep in sync, and the point here is to watch a long batch
run from a console without opening anything.

All explanatory text is English by request. Numbers use plain digits so the
output stays readable when piped to a file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, TextIO


LIVE_PROGRESS_VERSION = "1.3.0"
UNIFORM_PROGRESS_STAGES = frozenset({"extract-features", "extract-features-pilot-1000", "extract-features-pilot-10000"})

# What each stage does, in words a reader who did not write it can follow.
STAGE_DESCRIPTIONS: Mapping[str, str] = {
    "inventory": "Cataloguing the raw source files and checking they are readable.",
    "benchmark": "Measuring this machine to pick sane worker and memory settings.",
    "ingest": "Converting the raw MIMIC tables into typed, validated columnar files.",
    "ingest-events": "Streaming the very large event tables (labs, medications, ICU) into a partitioned store.",
    "build-index": "Building the master ECG index: one row per recording, linked to its admission, patient and echo.",
    "parse-reports": "Reading the recorder's own text interpretation of each ECG.",
    "parse-notes": "Parsing discharge summaries and radiology reports for clinical mentions.",
    "build-evidence": "Joining every clinical source to each ECG: labs, drugs, echo, ICU events, care context.",
    "attest-event-evidence": "Proving the joined evidence matches the raw event store row for row.",
    "genetic-feasibility": "Inventorying spatial QRS-T angle GWAS files and the claims the available genetic data can and cannot support.",
    "build-labels": "Deriving the clinical endpoints each ECG will be tested against, and sealing the locked test.",
    "attest-labels": "Hashing the label artifacts so later stages cannot silently use different labels.",
    "validate-delineation": "Checking wave-boundary detection against externally annotated reference ECGs.",
    "audit-biomarkers": "Verifying every marker implementation computes and returns a plausible scalar.",
    "extract-features": "Computing all ECG/VCG markers for every recording. This is the longest stage.",
    "extract-features-pilot-1000": "Pilot on 1,000 recordings: proves the marker pipeline runs before the full sweep starts.",
    "validate-pilot-1000": "Pilot validation on 1,000 recordings: proves the analysis path produces readable tables.",
    "extract-features-pilot-10000": "Pilot on 10,000 recordings: checks that throughput and failure rate scale before committing days.",
    "validate-pilot-10000": "Pilot validation on 10,000 recordings: a last look at the analysis path at scale.",
    "pilot-go-no-go": "Writing the documented pilot gate and stopping before any full-cohort extraction can start.",
    "marker-redundancy": "Grouping markers that cannot be told apart, so one quantity is not counted under several names.",
    "control-panel": "Checking the findings against effects whose direction physiology already fixed, and measuring the false-positive rate on controls that must show nothing.",
    "segment-availability": "Asking whether a marker was measurable at all, and whether the answer differs between the patients who have the condition and those who do not.",
    "ext-icd-crosscheck": "Comparing this project's endpoint labels against an independently published derivation from the same source, to find where one of the two is wrong.",
    "two-stage-selection": "Choosing from the training split alone what deserves a confirmatory test, so a discovery can become a claim.",
    "within-subject-change": "Comparing each patient with themselves across dialysis and drug doses, so anatomy and chronic illness cancel instead of being modelled.",
    "voltage-mass-discordance": "Recovering the left ventricular mass from the echo and asking which hearts make too little voltage for their muscle.",
    "time-to-event": "Starting a clock at the recording and asking which markers predict the year that follows.",
    "occlusion-yield": "Measuring what the ST-elevation criterion misses, against a reference that never looks at the ECG.",
    "thesis-tests": "Computing each physiology and VCG thesis exactly as it was written down before the run.",
    "therapy-response": "Comparing ECGs recorded before and after documented drug administration.",
    "physiological-gap-markers": "Fitting normative models and measuring each patient's deviation from them.",
    "repeatability": "Estimating measurement error from repeat recordings, and per-patient marker trajectories.",
    "external-validation": "Recomputing the markers on external hospital cohorts to test transferability.",
    "population-cohorts": "Computing the prespecified ECG/VCG marker set in CODE-15 and SaMi-Trop without pooling cohorts.",
    "screen-broad-axes": "Screening every marker against every frequent laboratory value, procedure, organism and ICU measurement.",
    "deep-combination-search": "Growing marker combinations one component at a time, until extra depth stops paying.",
    "validate": "Testing every marker against every endpoint, with multiplicity control and bootstrap intervals.",
    "discover-phenome": "Screening the markers against thousands of diagnoses, labs and care-context variables.",
    "group-vcg": "Rendering group-level vector loops for visual inspection.",
    "report": "Assembling the result tables and the readable report.",
    "audit": "Running the acceptance checks that must pass before anything is frozen.",
    "freeze": "Locking the shortlist so the sealed test set can be opened exactly once.",
}

# Expected cost of each stage, in hours, for the full 800,035-record run on
# this machine. Hours rather than arbitrary units, because the board divides by
# the total to show a percentage and extrapolates an ETA from it -- both are
# only meaningful if the units are time.
#
# Where the numbers come from:
#
# * extract-features: the previous full run took 94 h for 640,351 records at
#   1,048 markers. Per record that is 1.93 s; the current panel measures
#   2.61 s (0.85 s preprocessing plus 1.76 s for 1,713 VCG markers), and there
#   are 800,035 records. That is 1.69x the work, so ~159 h.
# * validate (10.65 h), discover-phenome (5.29 h), attest-event-evidence
#   (1.55 h), ingest-events (1.05 h), report (0.70 h): measured durations of
#   the previous run, scaled where the marker count drives them.
# * screen-broad-axes and deep-combination-search: measured on real data while
#   building them.
# * The remainder are estimates; they are small enough that being wrong by a
#   factor of two moves the total by under two percent.
#
# The earlier table used invented units in which `inventory` and `benchmark`
# weighed as much as an hour of real work. Because the cheap preparation
# stages then carried ~40 % of the total, finishing them in two hours made the
# board read "40 % done" and extrapolate the whole run to under a day -- while
# extract-features, the actual four-to-seven-day stage, had not started.
# Estimated hours per stage. Values marked "gemessen" come from the August 2026
# pre-run on this machine under the real stage resource limits; the rest are
# still estimates and are wrong in unknown directions.
#
# The pre-run corrected several of these badly. validate-pilot-1000 carried
# 0.3 h and actually took 25.3 -- a factor of 84 -- which is exactly the kind
# of error that makes a board report "almost finished" while the longest stage
# so far runs for another day.
STAGE_WEIGHTS: Mapping[str, float] = {
    "inventory": 0.1,
    "genetic-feasibility": 0.05,
    "benchmark": 0.2,
    "ingest": 1.0,
    "ingest-events": 1.5,
    "build-index": 0.3,
    "parse-reports": 0.5,
    "parse-notes": 1.0,
    "build-evidence": 0.25,           # gemessen: 12,8 min
    # Direkt aus der Ereignisschicht gebaute Evidenz braucht keinen
    # Schattenbau mehr (Audit E9): Sekunden. Nur mit force 2,5-4,3 h.
    "attest-event-evidence": 0.02,
    "build-labels": 0.2,              # gemessen: 9,5 min
    "attest-labels": 0.05,            # gemessen: unter 1 min
    # A131: 15,1 h seriell, davon 13,9 h STAFF III. Aufnahmeparallel
    # (Audit E7) geschätzt bei 8 Workern; Messung steht aus.
    "validate-delineation": 2.5,
    "audit-biomarkers": 0.05,         # gemessen: 0,5 min
    "extract-features-pilot-1000": 0.2,   # gemessen: 12,1 min
    "validate-pilot-1000": 25.3,      # gemessen
    "extract-features-pilot-10000": 2.0,
    "validate-pilot-10000": 10.0,
    "pilot-go-no-go": 0.05,
    "extract-features": 129.0,        # aus gemessener Rate 0,726 s je EKG
    "therapy-response": 2.0,
    "physiological-gap-markers": 2.0,
    # A seeded 50,000-row sample and one correlation matrix: minutes, not
    # hours. Measured on 5,924 ECGs the matrix itself took seconds.
    "marker-redundancy": 0.3,
    # All four read a handful of markers over a cohort the endpoints already
    # define, so they cost a fraction of a screen over the full catalog.
    # Both read one finished table and fit nothing: seconds.
    "control-panel": 0.05,
    # Five columns over the cohort, then grouping in memory: about a minute.
    "segment-availability": 0.15,
    "ext-icd-crosscheck": 0.1,
    "two-stage-selection": 0.05,
    "within-subject-change": 0.4,
    "voltage-mass-discordance": 0.3,
    "time-to-event": 0.5,
    "occlusion-yield": 0.2,
    "thesis-tests": 0.5,
    "repeatability": 3.0,
    "external-validation": 3.0,
    "population-cohorts": 72.0,
    "validate": 250.0,                # untere Schranke, siehe Hinweis unten
    "discover-phenome": 8.0,
    "screen-broad-axes": 5.0,
    "deep-combination-search": 5.0,
    "group-vcg": 1.0,
    "report": 1.0,
    "audit": 1.0,
    "freeze": 0.2,
}
_DEFAULT_WEIGHT = 1.0

# Stages that do not end on their own: they refine until a user stops them.
# A share of a total is meaningless for these, and quoting one would invite
# the reader to believe in a finish time that does not exist. The board shows
# their position and their refinement gain instead.
OPEN_ENDED_STAGES: frozenset[str] = frozenset(
    {"deep-combination-search"}
)

# ``validate`` is the honest problem in this table. Its cost is dominated by
# the per-marker bootstrap, which scales linearly with the number of records:
# measured 0.01 s per marker at 1,000 rows and 3.95 s at 200,000. Extrapolated
# to 640,376 rows at the full 2,000 iterations that is roughly 61 hours per
# endpoint, and the scaling of the remaining 98.6 % of the pilot's endpoint
# time has never been measured at all. The number above is a lower bound kept
# so the board does not understate the stage; it is not a forecast.
VALIDATE_WEIGHT_IS_A_LOWER_BOUND = True


def stage_weight(stage: str) -> float:
    return float(STAGE_WEIGHTS.get(str(stage), _DEFAULT_WEIGHT))


def stage_description(stage: str) -> str:
    return STAGE_DESCRIPTIONS.get(
        str(stage),
        "Running a pipeline stage.",
    )


@dataclass
class StageProgress:
    stage: str
    state: str = "pending"
    fraction: float = 0.0
    message: str = ""
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed_seconds(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


@dataclass
class RunProgress:
    """Weighted progress across a whole guided run."""

    run_id: str
    stages: list[str]
    completed: set[str] = field(default_factory=set)
    current: Optional[str] = None
    current_fraction: float = 0.0
    current_message: str = ""
    started_at: float = field(default_factory=time.time)
    stage_history: dict[str, StageProgress] = field(default_factory=dict)
    job_id: str = ""
    attempt_id: str = ""
    feature_runtime_contract: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for stage in self.stages:
            self.stage_history.setdefault(stage, StageProgress(stage=stage))

    @property
    def total_weight(self) -> float:
        return sum(stage_weight(stage) for stage in self.stages) or 1.0

    @property
    def completed_weight(self) -> float:
        done = sum(
            stage_weight(stage)
            for stage in self.stages
            if stage in self.completed
        )
        if self.current is not None and self.current not in self.completed:
            done += stage_weight(self.current) * max(
                0.0,
                min(1.0, self.current_fraction),
            )
        return done

    @property
    def overall_fraction(self) -> float:
        return max(0.0, min(1.0, self.completed_weight / self.total_weight))

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def remaining_weight(self) -> float:
        """Measured hours still to come, excluding stages that never end.

        Open-ended stages are left out on purpose: they refine until someone
        stops them, so adding their weight would put a finish time on
        something that has none.  What this returns is therefore "time until
        the run reaches the open-ended part", which is a real quantity.
        """

        remaining = 0.0
        for stage in self.stages:
            if stage in self.completed or stage in OPEN_ENDED_STAGES:
                continue
            if stage == self.current:
                remaining += stage_weight(stage) * (
                    1.0 - max(0.0, min(1.0, self.current_fraction))
                )
            else:
                remaining += stage_weight(stage)
        return max(0.0, remaining)

    @property
    def estimate_is_lower_bound(self) -> bool:
        """Historical durations never establish a guaranteed time bound."""
        return False

    @property
    def estimate_has_unmodelled_work(self) -> bool:
        """A partial planning estimate is not a statistical lower bound."""

        if any(stage in OPEN_ENDED_STAGES and stage not in self.completed for stage in self.stages):
            return True
        if not VALIDATE_WEIGHT_IS_A_LOWER_BOUND:
            return False
        return any(
            stage == "validate" and stage not in self.completed
            for stage in self.stages
        )

    def estimated_remaining_seconds(self) -> Optional[float]:
        """Uncertain planning remainder from historical stage priors.

        These priors are not yet matched to the current panel, bootstrap
        scope and implementation. They cannot establish a prediction interval
        or guaranteed lower bound. Cheap preparation timings must not rescale
        expensive statistical stages. Missing work is disclosed separately.
        """

        remaining = self.remaining_weight * 3600.0
        return remaining if remaining > 0.0 else None

    def observed_weight_ratio(self) -> Optional[float]:
        """How the finished stages compared with their recorded weights.

        Reported next to the estimate rather than folded into it.  If it drifts
        far from 1.0 the weights need re-measuring -- which is a decision for a
        person, not something the board should quietly apply to a stage it has
        never seen run.
        """

        weight = 0.0
        elapsed = 0.0
        for stage in self.stages:
            if stage not in self.completed:
                continue
            record = self.stage_history.get(stage)
            if record is None or record.elapsed_seconds is None:
                continue
            weight += stage_weight(stage)
            elapsed += float(record.elapsed_seconds)
        if weight <= 0.0 or elapsed <= 0.0:
            return None
        return elapsed / (weight * 3600.0)

    def begin_stage(self, stage: str) -> None:
        self.current = stage
        self.current_fraction = 0.0
        self.current_message = ""
        record = self.stage_history.setdefault(stage, StageProgress(stage=stage))
        record.state = "running"
        if record.started_at is None:
            record.started_at = time.time()

    def update_stage(self, fraction: float, message: str = "") -> None:
        self.current_fraction = max(0.0, min(1.0, float(fraction)))
        if message:
            self.current_message = message
        if self.current is not None:
            record = self.stage_history[self.current]
            record.fraction = self.current_fraction
            record.message = self.current_message

    def finish_stage(self, stage: str) -> None:
        self.completed.add(stage)
        record = self.stage_history.setdefault(stage, StageProgress(stage=stage))
        record.state = "done"
        record.fraction = 1.0
        record.finished_at = time.time()
        if self.current == stage:
            self.current = None
            self.current_fraction = 0.0

    def stages_completed_text(self) -> str:
        return f"stage {len(self.completed)}/{len(self.stages)}"

    def to_dict(self) -> dict[str, object]:
        remaining = self.estimated_remaining_seconds()
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "overall_fraction": self.overall_fraction,
            "current_stage": self.current or "",
            "current_stage_fraction": self.current_fraction,
            "current_stage_message": self.current_message,
            "feature_runtime_contract": dict(self.feature_runtime_contract),
            "current_stage_description": (
                stage_description(self.current) if self.current else ""
            ),
            "stages_total": len(self.stages),
            "stages_completed": len(self.completed),
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": remaining,
            "estimate_is_lower_bound": self.estimate_is_lower_bound,
            "estimate_has_unmodelled_work": self.estimate_has_unmodelled_work,
            "estimate_status": "partial_planning_estimate" if self.estimate_has_unmodelled_work else "planning_estimate",
            "estimate_basis": "historical_stage_priors_not_configuration_matched",
            "remaining_low_seconds": None,
            "remaining_high_seconds": None,
            "remaining_weight_hours": self.remaining_weight,
            "observed_weight_ratio": self.observed_weight_ratio(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "live_progress_version": LIVE_PROGRESS_VERSION,
        }


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    seconds = int(max(0.0, seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _bar(fraction: float, width: int) -> str:
    width = max(4, width)
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "-" * (width - filled)


class LiveProgressReporter:
    """Renders the status board and keeps a machine-readable mirror on disk.

    The on-disk copy matters for a run that outlives its terminal: a detached
    job keeps writing it, so a fresh console can show the current state
    without the run having to be attached to anything.
    """

    def __init__(
        self,
        progress: RunProgress,
        *,
        stream: Optional[TextIO] = None,
        state_path: Optional[Path] = None,
        min_redraw_seconds: float = 1.0,
        enable_ansi: Optional[bool] = None,
    ) -> None:
        self.progress = progress
        self.stream = stream if stream is not None else sys.stdout
        self.state_path = Path(state_path) if state_path else None
        self.min_redraw_seconds = float(min_redraw_seconds)
        self._last_redraw = 0.0
        self._lines_drawn = 0
        if enable_ansi is None:
            enable_ansi = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enable_ansi = bool(enable_ansi)

    # -- lifecycle -----------------------------------------------------

    def begin_stage(self, stage: str) -> None:
        self.progress.begin_stage(stage)
        self.render(force=True)

    def update(self, fraction: float, message: str = "") -> None:
        self.progress.update_stage(fraction, message)
        self.render()

    def finish_stage(self, stage: str) -> None:
        self.progress.finish_stage(stage)
        self.render(force=True)

    def finish_run(self) -> None:
        self.render(force=True)
        self._write_state()
        self.stream.write("\nRun complete.\n")
        self.stream.flush()

    # -- rendering -----------------------------------------------------

    def render(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_redraw < self.min_redraw_seconds:
            return
        self._last_redraw = now
        lines = self.compose()
        if self.enable_ansi and self._lines_drawn:
            self.stream.write(f"\033[{self._lines_drawn}A")
            self.stream.write("\033[J")
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._lines_drawn = len(lines) + 1 if self.enable_ansi else 0
        self._write_state()

    def compose(self) -> list[str]:
        width = shutil.get_terminal_size((100, 30)).columns
        width = max(60, min(width, 120))
        bar_width = width - 34
        progress = self.progress
        lines: list[str] = []
        lines.append("=" * width)
        lines.append(
            f"ECG/VCG biomarker discovery - run {progress.run_id}"
        )
        lines.append("=" * width)

        overall = progress.overall_fraction
        lines.append(
            f"OVERALL   [{_bar(overall, bar_width)}] {overall * 100:5.1f}%"
        )
        remaining_text = format_duration(progress.estimated_remaining_seconds())
        if progress.estimate_has_unmodelled_work:
            remaining_text = f"planning estimate {remaining_text} + unknown work"
        else:
            remaining_text = f"planning estimate {remaining_text} (uncertain)"
        lines.append(
            f"          {progress.stages_completed_text()}   "
            f"elapsed {format_duration(progress.elapsed_seconds)}   "
            f"remaining {remaining_text}"
        )
        lines.append("")

        if progress.current:
            stage_fraction = progress.current_fraction
            lines.append(f"NOW       {progress.current}")
            lines.append(f"          {stage_description(progress.current)}")
            lines.append(
                f"STAGE     [{_bar(stage_fraction, bar_width)}] "
                f"{stage_fraction * 100:5.1f}%"
            )
            if progress.current_message:
                message = progress.current_message
                if len(message) > width - 10:
                    message = message[: width - 13] + "..."
                lines.append(f"          {message}")
        else:
            lines.append("NOW       waiting for the next stage")
        lines.append("")

        lines.append("STAGES")
        for stage in progress.stages:
            record = progress.stage_history.get(stage)
            if record is None:
                continue
            if stage in progress.completed:
                mark = "[x]"
                detail = f"done in {format_duration(record.elapsed_seconds)}"
            elif stage == progress.current:
                mark = "[>]"
                detail = f"{progress.current_fraction * 100:.0f}% of this stage"
            else:
                mark = "[ ]"
                detail = "queued"
            share = stage_weight(stage) / progress.total_weight * 100.0
            lines.append(
                f"  {mark} {stage:<26s} {share:4.1f}% of run   {detail}"
            )
        lines.append("=" * width)
        return lines

    def _write_state(self) -> None:
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.progress.to_dict()
            from .run_log import active_log
            log = active_log()
            if log is not None:
                payload["phase_profiles"] = log.phase_totals()
                payload["active_phase_names"] = [name for name, counts in payload["phase_profiles"].items() if counts.get("active", 0) > 0]
                payload["work_counter_semantics"] = "phase calls, not equal-cost units; nested times not additive"
            payload["stages"] = [
                {
                    "stage": stage,
                    "state": (
                        "done"
                        if stage in self.progress.completed
                        else "running"
                        if stage == self.progress.current
                        else "pending"
                    ),
                    "description": stage_description(stage),
                    "weight_share": stage_weight(stage)
                    / self.progress.total_weight,
                    "elapsed_seconds": (
                        self.progress.stage_history[stage].elapsed_seconds
                        if stage in self.progress.stage_history
                        else None
                    ),
                }
                for stage in self.progress.stages
            ]
            # Fold in whatever an open-ended stage is currently reporting, so
            # the CLI board and the GUI read one file and agree. Silent on
            # failure: a missing refinement line must never cost the run its
            # status file.
            try:
                from .refinement import read_activity

                run_root = self.state_path.parent
                activity = read_activity(run_root) or read_activity(
                    run_root / "deep_combinations"
                )
                if activity:
                    payload["refinement"] = activity
            except Exception:  # noqa: BLE001 - status is never load-bearing
                pass
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(temporary, self.state_path)
        except OSError:
            # A status file that cannot be written must never stop the run.
            pass


def parse_fraction(message: str) -> Optional[float]:
    """Extract a fraction from a progress message that embeds "current/total".

    Stage messages already carry counts such as "12,500/76,000"; reusing them
    means a stage does not have to be instrumented twice.
    """

    match = re.search(r"([\d.,]+)\s*/\s*([\d.,]+)", str(message))
    if not match:
        return None
    try:
        current = float(match.group(1).replace(".", "").replace(",", ""))
        total = float(match.group(2).replace(".", "").replace(",", ""))
    except ValueError:
        return None
    if total <= 0:
        return None
    return max(0.0, min(1.0, current / total))


def read_live_state(state_path: Path | str) -> Optional[dict]:
    """Read the status file written by a detached run."""

    path = Path(state_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_snapshot(
    state: Mapping[str, object],
    *,
    width: int = 100,
) -> str:
    """Render the full board from a status file, for a fresh console.

    Deliberately verbose: this is what someone watching a four-day run reads,
    and "45.7 %" alone answers none of the questions they actually have --
    which stage, how far into it, what is it doing, what is still to come, and
    how long each finished stage took.
    """

    width = max(72, min(int(width), 140))
    bar = width - 34
    rule = "=" * width
    thin = "-" * width
    lines = [
        rule,
        f"ECG/VCG-Biomarker-Analyse  |  Lauf: {state.get('run_id', '?')}",
        rule,
    ]

    overall = float(state.get("overall_fraction", 0.0) or 0.0)
    completed = int(state.get("stages_completed", 0) or 0)
    total = int(state.get("stages_total", 0) or 0)
    elapsed = float(state.get("elapsed_seconds", 0) or 0)
    remaining = state.get("estimated_remaining_seconds")
    lines.append(
        f"GESAMT    [{_bar(overall, bar)}] {overall * 100:5.1f} %"
    )
    partial = bool(state.get("estimate_has_unmodelled_work", state.get("estimate_is_lower_bound", False)))
    remaining_text = format_duration(remaining)
    if remaining is not None:
        remaining_text = f"Planungswert {remaining_text}" + (" + unbekannte Restarbeit" if partial else " (unsicher)")
    lines.append(
        f"          Stufe {completed}/{total}"
        f"   |  gelaufen {format_duration(elapsed)}"
        f"   |  verbleibend {remaining_text}"
    )
    if remaining is not None and not partial:
        finish = datetime.now(timezone.utc).astimezone() + timedelta(
            seconds=float(remaining)
        )
        prefix = "unverbindlicher Planungstermin"
        lines.append(f"          {prefix}: {finish:%d.%m. %H:%M}")
    ratio = state.get("observed_weight_ratio")
    if ratio is not None:
        try:
            value = float(ratio)
        except (TypeError, ValueError):
            value = None
        # Reported, never applied: how the finished stages compared with their
        # recorded cost. Far from 1.0 means the weights want re-measuring.
        if value is not None and (value < 0.5 or value > 2.0):
            lines.append(
                f"          Hinweis: erledigte Stufen liefen mit Faktor "
                f"{value:.2f} gegen ihre gemessenen Gewichte"
            )
    lines.append("")

    current = str(state.get("current_stage", "") or "")
    if current:
        fraction = float(state.get("current_stage_fraction", 0.0) or 0.0)
        lines.append(f"AKTUELL   {current}")
        description = str(state.get("current_stage_description", "") or "")
        for chunk in _wrap(description, width - 10):
            lines.append(f"          {chunk}")
        lines.append(
            f"STUFE     [{_bar(fraction, bar)}] {fraction * 100:5.1f} %"
        )
        message = str(state.get("current_stage_message", "") or "")
        for chunk in _wrap(message, width - 10):
            lines.append(f"          {chunk}")
    else:
        lines.append("AKTUELL   wartet auf die naechste Stufe")
    lines.append("")

    stages = state.get("stages")
    if isinstance(stages, list) and stages:
        # Column width from the longest name present, so a long stage id does
        # not shove every following column out of line.
        name_width = max(
            18,
            min(
                34,
                max(
                    len(str(entry.get("stage", "")))
                    for entry in stages
                    if isinstance(entry, dict)
                ),
            ),
        )
        lines.append(thin)
        lines.append(
            f"  {'':<3s} {'Stufe':<{name_width}s} {'Anteil':>7s}  "
            f"{'Status':<8s} {'Dauer':>12s}"
        )
        lines.append(thin)
        for entry in stages:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("stage", "?"))
            share = float(entry.get("weight_share", 0.0) or 0.0) * 100.0
            status = str(entry.get("state", "pending"))
            mark, label = {
                "done": ("[x]", "fertig"),
                "running": ("[>]", "laeuft"),
            }.get(status, ("[ ]", "wartet"))
            if name in OPEN_ENDED_STAGES:
                # This stage ends when a user says so. Printing a share of a
                # total would promise a finish that does not exist.
                label = "offen" if status == "running" else label
            seconds = entry.get("elapsed_seconds")
            duration = (
                format_duration(float(seconds))
                if seconds is not None
                else "--"
            )
            lines.append(
                f"  {mark} {name[:name_width]:<{name_width}s} {share:5.1f} %  "
                f"{label:<8s} {duration:>12s}"
            )
        lines.append(thin)

    refinement = state.get("refinement")
    if isinstance(refinement, Mapping) and refinement:
        from .refinement import describe_activity, refinement_has_saturated

        lines.append(thin)
        lines.append("  Offene Verfeinerung")
        lines.append(f"    {describe_activity(refinement)}")
        # No percentage here on purpose: the stage has no total to be a
        # percentage of. What a watcher needs is whether the last increment
        # still moved the result.
        saturated = refinement_has_saturated(
            refinement.get("gain"), threshold=0.0005
        )
        if saturated is True:
            lines.append(
                "    Der letzte Zuwachs bewegte das Ergebnis kaum noch -- "
                "laenger rechnen bringt hier voraussichtlich nichts."
            )
        elif saturated is False:
            lines.append(
                "    Das Ergebnis bewegt sich noch; laenger rechnen lohnt."
            )
        lines.append(thin)

    lines.append(f"letzte Aktualisierung: {state.get('updated_at', '?')}")
    return chr(10).join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Soft-wrap without importing textwrap for one call."""

    words = str(text).split()
    if not words:
        return []
    rows: list[str] = []
    line = words[0]
    for word in words[1:]:
        if len(line) + 1 + len(word) <= width:
            line = f"{line} {word}"
        else:
            rows.append(line)
            line = word
    rows.append(line)
    return rows


def render_live_board(
    state: Optional[Mapping[str, object]],
    *,
    run_id: str,
    state_path: object,
    width: int = 100,
) -> str:
    """The board, or an explanation of why there is not one yet."""

    if state is None:
        return chr(10).join(
            [
                "=" * 78,
                f"Noch kein Live-Status fuer den Lauf '{run_id}'.",
                "=" * 78,
                f"Erwartet unter: {state_path}",
                "",
                "Der Lauf schreibt diese Datei, sobald die erste Stufe "
                "beginnt.",
                "Laeuft ueberhaupt eine Aufgabe? -> mimic_cli jobs list",
            ]
        )
    return render_snapshot(state, width=width)


__all__ = [
    "LIVE_PROGRESS_VERSION",
    "LiveProgressReporter",
    "RunProgress",
    "STAGE_DESCRIPTIONS",
    "STAGE_WEIGHTS",
    "StageProgress",
    "format_duration",
    "parse_fraction",
    "read_live_state",
    "render_live_board",
    "render_snapshot",
    "stage_description",
    "stage_weight",
]
