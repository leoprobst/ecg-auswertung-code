import pytest

from vcg_biomarker.mimic import population_cohorts as population


@pytest.mark.parametrize("changed", population.POPULATION_ANALYSIS_SOURCES)
def test_every_analysis_dependency_changes_hash(tmp_path, monkeypatch, changed):
    for name in population.POPULATION_ANALYSIS_SOURCES:
        (tmp_path / name).write_text("# synthetic source\n", encoding="utf-8")
    monkeypatch.setattr(population, "__file__", str(tmp_path / "population_cohorts.py"))
    first = population._population_analysis_source_hash()
    assert population._population_analysis_source_hash() == first
    (tmp_path / changed).write_text("# changed source\n", encoding="utf-8")
    assert population._population_analysis_source_hash() != first


def test_missing_dependency_cannot_reuse_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(population, "__file__", str(tmp_path / "population_cohorts.py"))
    with pytest.raises(FileNotFoundError):
        population._population_analysis_source_hash()


def test_extraction_hash_ignores_statistics_but_tracks_extraction(tmp_path, monkeypatch):
    from pathlib import Path
    source = Path(population.__file__).read_text(encoding="utf-8")
    for name in population.POPULATION_ANALYSIS_SOURCES:
        (tmp_path / name).write_text(source if name == "population_cohorts.py" else "# fixture", encoding="utf-8")
    monkeypatch.setattr(population, "__file__", str(tmp_path / "population_cohorts.py"))
    monkeypatch.setattr(population, "feature_implementation_hash", lambda: "fixed_features")
    first = population._population_extraction_source_hash()
    analysis = population._population_analysis_source_hash()
    (tmp_path / "survival.py").write_text("# changed survival", encoding="utf-8")
    assert population._population_analysis_source_hash() != analysis
    assert population._population_extraction_source_hash() == first
    (tmp_path / "population_cohorts.py").write_text(source.replace(
        '"age+sex"', '"changed_analysis_adjustment"') + "\n# analysis documentation", encoding="utf-8")
    assert population._population_extraction_source_hash() == first
    (tmp_path / "population_cohorts.py").write_text(source.replace(
        "minimum_duration_seconds=6.5", "minimum_duration_seconds=7.5"), encoding="utf-8")
    assert population._population_extraction_source_hash() != first


def test_extraction_input_tracks_own_data_and_sampling_not_other_cohort(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(population, "_population_extraction_source_hash", lambda: "extract")
    inventory = pd.DataFrame(dict(cohort=["code15", "sami_trop"], path=["a", "b"],
        size=[1, 2], mtime_ns=[3, 4], sha256=["aa", "bb"]))
    def key(markers=("marker",), cap=None, quality=0.):
        return population._population_extraction_input_hash(inventory, "code15", markers, cap, quality)
    first = key()
    inventory.loc[1, "sha256"] = "changed_other_cohort"
    assert key() == first
    assert key(cap=1000) != first
    assert key(quality=.8) != first
    assert key(markers=("other",)) != first
    inventory.loc[0, "sha256"] = "changed_own_data"
    assert key() != first


def test_new_local_helper_and_nested_dependency_are_automatically_bound(tmp_path, monkeypatch):
    from pathlib import Path
    source = Path(population.__file__).read_text(encoding="utf-8")
    source = source.replace("signal = np.asarray(record.signal_millivolts, dtype=float)",
                            "signal = extraction_helper(record.signal_millivolts)")
    source += "\ndef extraction_helper(value):\n    return nested_helper(value)\n"
    source += "\ndef nested_helper(value):\n    return value * 1.0\n"
    path = tmp_path / "population_cohorts.py"
    path.write_text(source, encoding="utf-8")
    (tmp_path / "code15_cohort.py").write_text("# fixture", encoding="utf-8")
    monkeypatch.setattr(population, "__file__", str(path))
    monkeypatch.setattr(population, "feature_implementation_hash", lambda: "features")
    first = population._population_extraction_source_hash()
    path.write_text(source.replace("return value * 1.0", "return value * 2.0"), encoding="utf-8")
    assert population._population_extraction_source_hash() != first


def test_public_stage_invalidates_results_but_preserves_extraction_key(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import pandas as pd
    from vcg_biomarker.mimic.storage import capture_output_integrity
    roots = [tmp_path / name for name in ("code", "sami")]
    for root in roots:
        root.mkdir()
        (root / "exams.csv").write_text("fixture")
    config = SimpleNamespace(code15_root=roots[0], sami_trop_root=roots[1],
                             external_validation_root=tmp_path / "output", workers=1)
    monkeypatch.setattr(population, "audit_cohort_release", lambda *a: {"status": "ok"})
    monkeypatch.setattr(population, "_panel_markers", lambda *a: ())
    monkeypatch.setattr(population, "_transportable_validation_markers", lambda *a: ((), {}))
    monkeypatch.setattr(population, "feature_implementation_hash", lambda: "features")
    monkeypatch.setattr(population, "_population_extraction_source_hash", lambda: "extract")
    monkeypatch.setattr(population, "_population_analysis_source_hash", lambda: "analysis1")
    monkeypatch.setattr(population, "_source_inventory", lambda root, cohort, **kw:
        pd.DataFrame([dict(cohort=cohort, path=str(root), size=1, mtime_ns=2, sha256="source")]))
    real_hash = population.content_hash
    input_hashes, extraction_keys = [], []
    def record_hash(payload):
        value = real_hash(payload)
        if "analysis_implementation_hash" in payload:
            input_hashes.append(value)
        return value
    monkeypatch.setattr(population, "content_hash", record_hash)
    class StopAtExtraction(Exception):
        pass
    def stop(*args, **kwargs):
        extraction_keys.append(kwargs["extraction_input_hash"])
        raise StopAtExtraction
    monkeypatch.setattr(population, "_extract_cohort", stop)
    with pytest.raises(StopAtExtraction):
        population.run_population_cohort_analysis(config, run_id="fixture")
    output = config.external_validation_root / "fixture/population_cohorts"
    artifact = output / "source_inventory.parquet"
    (output / "population_cohorts.manifest.json").write_text(json.dumps(dict(
        stage_version=population.POPULATION_COHORT_STAGE_VERSION,
        input_hash=input_hashes[-1], output_paths=[str(artifact)],
        output_integrity=capture_output_integrity([artifact]))))
    result = population.run_population_cohort_analysis(config, run_id="fixture")
    assert result.notes == ("reused_exact_input_hash",)
    monkeypatch.setattr(population, "_population_analysis_source_hash", lambda: "analysis2")
    with pytest.raises(StopAtExtraction):
        population.run_population_cohort_analysis(config, run_id="fixture")
    assert input_hashes[0] != input_hashes[-1]
    assert extraction_keys[0] == extraction_keys[-1]
