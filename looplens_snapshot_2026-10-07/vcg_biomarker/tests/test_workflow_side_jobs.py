"""E8: population-cohorts runs as a supervised side job beside the study."""
from __future__ import annotations

from types import SimpleNamespace

from vcg_biomarker.mimic import workflow_side_jobs as side


class _Store:
    def __init__(self):
        self.jobs = {"study": SimpleNamespace(spec=SimpleNamespace(
            job_id="study", project_file="p.json"), state=SimpleNamespace(value="running"))}
        self.created = []
        self.states = []

    def create(self, *, stage, project_file, arguments, requested_by):
        job_id = f"{stage}-1"
        self.created.append((stage, project_file, dict(arguments), requested_by))
        self.jobs[job_id] = SimpleNamespace(spec=SimpleNamespace(job_id=job_id),
                                            state=SimpleNamespace(value="queued"))
        return self.jobs[job_id]

    def get(self, job_id):
        record = self.jobs[job_id]
        if job_id != "study" and self.states:
            record.state = SimpleNamespace(value=self.states.pop(0))
        return record


def test_side_job_is_queued_once_after_its_prerequisite_and_awaited(tmp_path):
    config = SimpleNamespace(runs_root=tmp_path)
    store = _Store()
    checkpoints = []
    control = SimpleNamespace(store=store, job_id="study",
                              checkpoint=lambda message="", **_: checkpoints.append(message))
    stages = ("inventory", "benchmark", "extract-features", "population-cohorts")

    def arguments(stage):
        return {"run_id": "R1", "panel_ids": ("primary_core", "sentinel")}

    # Before inventory: nothing is queued.
    assert side.launch_side_jobs(config, run_id="R1", stage_ids=stages, completed=(),
                                 job_control=control, arguments_for=arguments) == {}
    registry = side.launch_side_jobs(config, run_id="R1", stage_ids=stages,
                                     completed=("inventory",), job_control=control,
                                     arguments_for=arguments)
    assert registry == {"population-cohorts": "population-cohorts-1"}
    # Idempotent across stages and resumes.
    side.launch_side_jobs(config, run_id="R1", stage_ids=stages,
                          completed=("inventory", "benchmark"), job_control=control,
                          arguments_for=arguments)
    assert len(store.created) == 1
    assert store.created[0][1] == "p.json"
    assert store.created[0][3] == "study-run:study"

    store.states = ["queued", "running", "completed"]
    state = side.wait_for_side_job(config, "population-cohorts", run_id="R1",
                                   job_control=control, sleep=lambda _: None)
    assert state == "completed"
    assert len(checkpoints) == 2


def test_without_supervisor_the_order_stays_sequential(tmp_path):
    config = SimpleNamespace(runs_root=tmp_path)
    assert side.launch_side_jobs(config, run_id="R1", stage_ids=("population-cohorts",),
                                 completed=("inventory",), job_control=None,
                                 arguments_for=lambda stage: {}) == {}
    assert side.wait_for_side_job(config, "population-cohorts", run_id="R1",
                                  job_control=None) is None
