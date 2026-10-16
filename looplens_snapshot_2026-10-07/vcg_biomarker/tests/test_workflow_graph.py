import pytest
from vcg_biomarker.mimic.workflow import guided_study_spec
from vcg_biomarker.mimic.workflow_graph import dependency_graph, critical_path_seconds


@pytest.mark.parametrize("preset,pilot", [("fast_preview", False), ("production_path_preview", False), ("exhaustive_research", False), ("exhaustive_research", True)])
def test_generated_order_satisfies_every_declared_dependency(preset, pilot):
    stages = guided_study_spec(preset_id=preset, run_id="synthetic", stop_after_pilots=pilot).stage_ids
    graph = dependency_graph(stages)
    for stage, dependencies in graph.items():
        assert all(stages.index(d) < stages.index(stage) for d in dependencies)
    if not pilot and preset != "fast_preview":
        assert stages.index("physiological-gap-markers") < stages.index("marker-redundancy")


def test_critical_path_not_sum_parallel_effort_and_unknown_not_zero():
    stages = ("inventory", "genetic-feasibility", "benchmark")
    assert critical_path_seconds(stages, dict(zip(stages, (2., 5., 3.)))) == 7.
    assert critical_path_seconds(stages, {"inventory": 2.}) is None
    with pytest.raises(ValueError, match="Duplicate"):
        dependency_graph(("inventory", "inventory"))
    with pytest.raises(ValueError, match="Undeclared"):
        dependency_graph(("made-up-stage",))
