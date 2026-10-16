"""Grouped research-campaign, variant and real-data-prior regression gates."""
from dataclasses import replace
import json
import numpy as np
import pytest

from vcg_biomarker.simulation.ionic import CellExperiment, compile_model, simulate_cell
from vcg_biomarker.simulation.variants import SAN_VARIANT, gate_reference
from vcg_biomarker.simulation.learning import prepare, train_campaign, candidate_task
from vcg_biomarker.simulation.resources import capacity_target, PagingMonitor
from vcg_biomarker.simulation.calibration import train_priors, predict_priors
from vcg_biomarker.simulation.patient import PatientProfile, Observation
from vcg_biomarker.tests.test_patient_twin import training_data
from vcg_biomarker.tests.test_simulation_mechanisms import synthetic_bank


def test_san_variant_has_distinct_gates_and_conserves_current_routing():
    spec = CellExperiment(model="fabbri-2017", genetic_variant=SAN_VARIANT, beats=1)
    model, rhs, vi, factor = compile_model(spec)
    assert factor == .001
    states = list(model.states())
    names = [s.qname() for s in states]
    mi, hi = names.index("scn5a_1795insd.m"), names.index("scn5a_1795insd.h")
    y = np.array(model.initial_values(as_floats=True))
    for v in (-100., -40., -25., 0.):
        y[vi], y[mi], y[hi] = v, .2, .3
        m, h, tm, th = gate_reference(v, True)
        derived = rhs(0., y, 0.)
        assert np.isfinite(derived).all()
        assert derived[mi] == pytest.approx((m-.2)/tm)
        assert derived[hi] == pytest.approx((h-.3)/th)
    assert model.get("ina.gNa").eval() == pytest.approx(.01115)
    assert "scn5a_1795insd.I" in str(model.get("ina.INa").rhs())
    model.check_units()
    with pytest.raises(ValueError):
        replace(spec, model="ohara-2011").validate()
    with pytest.raises(ValueError):
        replace(spec, channel_scales={"INa": .5}).validate()


def test_san_ghk_zero_voltage_limits_are_finite_and_continuous():
    import warnings
    model, rhs, vi, _ = compile_model(CellExperiment(model="fabbri-2017"))
    _, vector, _, _ = compile_model(CellExperiment(model="fabbri-2017"), vectorized=True)
    state = np.array(model.initial_values(as_floats=True))
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for voltage in (-1e-5, 0., 1e-5):
            state[vi] = voltage
            scalar = np.array(rhs(0., state, 0.))
            values = vector(0., np.repeat(state[:, None], 2, axis=1), np.zeros(2))
            for a, b in zip(scalar, values):
                np.testing.assert_allclose(a, b, atol=1e-9)
            assert np.isfinite(scalar).all()
            results.append(scalar)
    np.testing.assert_allclose(results[1], (results[0]+results[2])/2, atol=1e-6, rtol=1e-7)


def test_external_inventory_boolean_events_are_not_missing(tmp_path):
    from audit_tools.simulation_research_status import csv_inventory
    path = tmp_path/"exams.csv"
    path.write_text("exam_id,death,timey\n1,True,1.2\n2,False,3\n3,unknown,1\n4,0,0\n", encoding="utf-8")
    result = csv_inventory(path)
    assert result["rows"] == 4
    assert result["rows_with_positive_finite_followup_and_event_indicator"] == 2


def test_mutant_voltage_clamp_matches_source_peak_and_persistent_fraction():
    # Pure WT vs pure mutant clamp (not the heterozygous sum) as in source Figure 1B.
    t = np.linspace(0., .05, 10001)
    traces = []
    for mutant in (False, True):
        m0, h0, _, _ = gate_reference(-100., mutant)
        m, h, tm, th = gate_reference(-25., mutant)
        mt, ht = m+(m0-m)*np.exp(-t/tm), h+(h0-h)*np.exp(-t/th)
        f = .006 if mutant else 0.
        traces.append((.5 if mutant else 1.) * mt**3 * ((1-f)*ht+f))
    assert .19 < traces[1].max()/traces[0].max() < .25
    assert .01 < traces[1][-1]/traces[1].max() < .018
    spec = CellExperiment(model="fabbri-2017", genetic_variant=SAN_VARIANT, beats=2)
    _, v, _, metadata = simulate_cell(spec)
    assert np.isfinite(v).all() and metadata["variant"]["clinical_use"] is False


def test_multirate_prepare_keeps_candidate_splits_and_freezes_modules(tmp_path):
    manifest = prepare(tmp_path/"job", samples=32, cycles_ms=(500., 800., 1200.), max_workers=8, precondition_max_beats=10)
    assert len(manifest["cases"]) == 96
    assert json.loads((tmp_path/"job"/"status.json").read_text())["total"] == 96
    assert manifest["max_workers"] == 8
    for i in range(32):
        cases = [r for r in manifest["cases"] if r["candidate_id"] == i]
        assert len({r["partition"] for r in cases}) == 1
        assert all(r["scales"] == cases[0]["scales"] for r in cases)
    for module in ("resources.py", "variants.py", "identifiability.py"):
        assert f"vcg_biomarker\\simulation\\{module}" in manifest["runtime_hashes"] or f"vcg_biomarker/simulation/{module}" in manifest["runtime_hashes"]


def test_multirate_fit_does_not_pool_targets_or_cross_audit_roles(tmp_path):
    rows = []
    for row in synthetic_bank():
        for j, cycle in enumerate((500., 800., 1200.)):
            copied = json.loads(json.dumps(row))
            copied.update(id=row["id"]*3+j, candidate_id=row["id"], cycle_ms=cycle)
            copied["biomarkers"]["apd90_ms"] += j*30
            rows.append(copied)
    manifest = {"cycles_ms": [500., 800., 1200.], "cycle_ms": 500., "job_sha256": "test",
                "model": "tentusscher-2006", "beats": 10, "output_ms": .25}
    result = train_campaign(rows, manifest, tmp_path)
    assert result["status"] == "research_surrogates"
    assert all(m["counts"]["audit"] == 20 for m in result["models"])
    assert len(list((tmp_path/"emulators").glob("*.json"))) == 3
    rows[1]["partition"] = "audit"
    with pytest.raises(ValueError, match="crossed"):
        train_campaign(rows, manifest, tmp_path)


def test_resource_pressure_and_yield_preserve_retryable_case(tmp_path):
    assert capacity_target(16, 20, 3, 10, .5, 15) > 4
    # Mapped-file reads with ample free memory must not throttle the campaign.
    assert capacity_target(16, 20, 3, 10, .5, 15, True) == capacity_target(16, 20, 3, 10, .5, 15)
    assert capacity_target(16, 20, 3, 2.9, .5, 15, True) == 1
    assert capacity_target(16, 20, 3, 1.9, .5, 15) == 0
    assert capacity_target(16, 50, 6, 8, .5, 15, active_workers=6, active_resident_gib=1.) == 13
    assert capacity_target(16, 50, 6, 2, .5, 15, active_workers=6, active_resident_gib=1.) == 2
    monitor = PagingMonitor()
    sample = monitor.sample()
    assert sample["hard_page_inputs_per_second"] is None or sample["hard_page_inputs_per_second"] >= 0
    monitor.close()
    (tmp_path/"cases").mkdir()
    (tmp_path/"yield").mkdir()
    (tmp_path/"yield"/"0").touch()
    case = {"id": 0, "candidate_id": 0, "partition": "fit", "scales": {}, "cycle_ms": 500.}
    manifest = {"job_sha256": "test", "model": "tentusscher-2006", "beats": 3, "cycle_ms": 800., "output_ms": 1.}
    assert candidate_task(str(tmp_path), case, manifest)["status"] == "interrupted"
    assert not (tmp_path/"cases/00000.json").exists()


def test_age_sex_prior_does_not_impute_body_measures_and_keeps_patient_roles():
    data = training_data()
    data[["height_cm", "weight_kg"]] = np.nan
    bundle = train_priors(data, feature_profile="age_sex")
    assert bundle["features"] == ["age", "female"]
    profile = PatientProfile({"age": Observation(50., "years", "measured")}, "male")
    value = predict_priors(bundle, profile)["lv_diameter_mm"]
    assert value["status"] == "population_prior" and value["required_covariates"] == ["age", "female"]
    assert train_priors(data)["models"]["lv_diameter_mm"]["status"] == "non_estimable"
    with pytest.raises(ValueError, match="train"):
        train_priors(data.assign(source_split="locked_test"), feature_profile="age_sex")


def test_bounded_kernel_cache_and_unreleased_native_guard():
    from vcg_biomarker.simulation.ionic import _compiled_cell
    spec = CellExperiment(beats=1)
    _, a, _, _ = simulate_cell(spec)
    with pytest.raises(ValueError, match="not released"):
        simulate_cell(replace(spec, rhs_backend="numba"))
    _, changed, _, _ = simulate_cell(replace(spec, channel_scales={"IKr": .5}))
    assert np.max(np.abs(changed-a)) > 1.
    assert _compiled_cell.cache_info().currsize <= 2
