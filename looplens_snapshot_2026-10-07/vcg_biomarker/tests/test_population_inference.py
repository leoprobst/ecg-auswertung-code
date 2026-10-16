import copy

import numpy as np
import pytest

from vcg_biomarker.simulation.learning import PARAMETERS, OUTPUTS, design
from vcg_biomarker.simulation.population_inference import calibrate_joint, PopulationMatcher, predict_batch
from vcg_biomarker.simulation.identifiability import local_identifiability


def fixture_bank():
    rng = np.random.default_rng(81)
    factors = rng.uniform(.5, 1.5, (100, 6))
    models, rows = [], []
    for j, cycle in enumerate((500., 800., 1200.)):
        coef = rng.normal(size=(28, 4))
        model = {"schema": "ionic-forward-emulator/1", "clinical_use": False, "status": "research_surrogate",
                 "parameters": list(PARAMETERS), "outputs": list(OUTPUTS), "coefficients": coef.tolist(),
                 "calibration_radius_90": [.1]*4, "job_sha256": "synthetic_fixture",
                 "protocol": {"model": "test", "beats": 10, "cycle_ms": cycle, "output_ms": .25,
                              "cell_environment": {"source": "test"}, "precondition_max_beats": 100}}
        models.append(model)
        truth = design(factors)@coef+rng.normal(0, .05, (100, 4))
        for i in range(100):
            rows.append({"candidate_id": i, "cycle_ms": cycle, "job_sha256": "synthetic_fixture",
                         "partition": "fit" if i < 60 else "calibration" if i < 80 else "audit",
                         "status": "computed", "periodicity_check_passed": True,
                         "scales": dict(zip(PARAMETERS, factors[i])), "biomarkers": dict(zip(OUTPUTS, truth[i]))})
    return models, rows, factors


def test_joint_quantile_groups_protocols_and_never_uses_audit_for_calibration():
    models, rows, _ = fixture_bank()
    result, arrays = calibrate_joint(models, rows)
    assert result["counts"] == {"fit": 60, "calibration": 20, "audit": 20}
    scale = arrays["fit"][2].std(0)
    scores = np.max(np.abs(arrays["calibration"][3]-arrays["calibration"][2])/scale, axis=(1, 2))
    np.testing.assert_allclose(result["simultaneous_radius"], np.sort(scores)[18]*scale)
    modified = copy.deepcopy(rows)
    for row in modified:
        if row["partition"] == "audit":
            row["biomarkers"][OUTPUTS[0]] += 1000
    changed, _ = calibrate_joint(models, modified)
    assert changed["simultaneous_radius"] == result["simultaneous_radius"]
    assert changed["audit_joint_coverage"] == 0
    assert changed["calibration_data_sha256"] == result["calibration_data_sha256"]


@pytest.mark.parametrize("change", ["role", "scales", "duplicate", "job", "nonfinite"])
def test_joint_contract_rejects_leakage_and_invalid_rows(change):
    models, rows, _ = fixture_bank()
    if change == "role":
        rows[0]["partition"] = "audit"
    elif change == "scales":
        rows[0]["scales"][PARAMETERS[0]] = 1.49
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "job":
        rows[0]["job_sha256"] = "other"
    else:
        rows[0]["biomarkers"][OUTPUTS[0]] = float("nan")
    with pytest.raises(ValueError):
        calibrate_joint(models, rows)


def test_failed_protocol_excludes_whole_candidate_not_only_that_rate():
    models, rows, _ = fixture_bank()
    rows[0]["periodicity_check_passed"] = False
    result, _ = calibrate_joint(models, rows)
    assert result["excluded_incomplete_candidates"] == 1
    assert result["counts"]["fit"] == 59
    with pytest.raises(ValueError, match="Duplicate"):
        calibrate_joint([models[0], models[0]], rows)
    with pytest.raises(ValueError, match="quantile"):
        calibrate_joint(models, rows, coverage=.999)


def test_matcher_chunk_invariance_and_no_false_unique_claim():
    models, rows, factors = fixture_bank()
    calibration, _ = calibrate_joint(models, rows)
    observation = {"reference_kind": "synthetic_validation", "protocols": calibration["protocols"],
                   "outputs": list(OUTPUTS), "values": predict_batch(models, factors[:1])[0].tolist(),
                   "tolerances": np.full((3, 4), .01).tolist()}
    one = PopulationMatcher(models, calibration, factors, chunk_size=1).match(observation)
    many = PopulationMatcher(models, calibration, factors, chunk_size=40).match(observation)
    assert one == many
    assert one["accepted_candidates"] >= 1
    assert one["unique_parameters_identified"] is False
    assert one["posterior_probabilities"] is None
    observation["values"] = np.full((3, 4), 1e9).tolist()
    assert PopulationMatcher(models, calibration, factors).match(observation)["status"] == "no_candidate_on_finite_design"


def test_clinical_substitution_and_protocol_mismatch_rejected():
    models, rows, factors = fixture_bank()
    calibration, _ = calibrate_joint(models, rows)
    matcher = PopulationMatcher(models, calibration, factors)
    observation = {"reference_kind": "surface_ecg", "protocols": calibration["protocols"],
                   "outputs": list(OUTPUTS), "values": np.ones((3, 4)).tolist(), "tolerances": np.ones((3, 4)).tolist()}
    with pytest.raises(ValueError, match="Surface"):
        matcher.match(observation)
    observation["reference_kind"] = "independent_cellular_action_potential"
    with pytest.raises(ValueError, match="provenance"):
        matcher.match(observation)
    observation.update(reference_provenance="lab/example", measurement_method="ap_features_v1")
    assert matcher.match(observation)["clinical_use"] is False
    observation["protocols"] = observation["protocols"][:1]
    with pytest.raises(ValueError, match="protocols"):
        matcher.match(observation)
    altered = copy.deepcopy(models)
    altered[0]["coefficients"][0][0] += 1
    with pytest.raises(ValueError, match="belong"):
        PopulationMatcher(altered, calibration, factors)


def test_numerical_rank_not_practical_identifiability():
    models, _, _ = fixture_bank()
    report = local_identifiability(models, dict.fromkeys(PARAMETERS, 1.), [dict.fromkeys(OUTPUTS, 1e8)]*3)
    assert report["numerical_rank"] == 6
    assert report["unit_perturbation_below_tolerance_directions"] == 6
    assert report["unique_patient_parameters_identified"] is False


def test_atomic_checkpoint_retries_transient_windows_lock(tmp_path, monkeypatch):
    from pathlib import Path
    import json
    from vcg_biomarker.simulation.learning import atomic_json
    original = Path.replace
    calls = []

    def locked_then_replace(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("temporary indexer lock")
        return original(source, target)

    monkeypatch.setattr(Path, "replace", locked_then_replace)
    atomic_json(tmp_path/"status.json", {"phase": "complete"})
    assert len(calls) == 3
    assert json.loads((tmp_path/"status.json").read_text()) == {"phase": "complete"}


def test_exact_global_sensitivity_uniform_moments_and_interactions():
    from vcg_biomarker.simulation.identifiability import global_polynomial_sensitivity
    models, _, _ = fixture_bank()
    model = models[0]
    coef = np.zeros((28, 4))
    coef[1, 0] = 1  # x0
    coef[7, 0] = 1  # x0^2
    coef[8, 0] = 2  # 2*x0*x1
    model["coefficients"] = coef.tolist()
    report = global_polynomial_sensitivity(model)
    output = report["outputs"][OUTPUTS[0]]
    variance = 1/3+4/45+4/9
    assert output["variance"] == pytest.approx(variance)
    assert output["first_order"]["INa"] == pytest.approx((1/3+4/45)/variance)
    assert output["total_order"]["ICaL"] == pytest.approx((4/9)/variance)
    assert sum(output["first_order"].values())+sum(output["second_order"].values()) == pytest.approx(1)
    assert report["outputs"][OUTPUTS[1]]["first_order"] is None
