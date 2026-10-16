"""Analyses identify themselves, and nothing computed becomes invisible.

The run identifier was a text field carrying the project's hardest question:
"change it when markers, evidence or configuration have changed". A person
cannot answer that reliably -- the software can, to the byte. These tests pin
the two directions in which the automation must not fail:

* the same configuration must **continue** its analysis, not fork a new one,
  or every restart throws away weeks of valid computation;
* a changed configuration must **not** continue, or two different studies
  silently merge into one set of results.

Plus the property that makes it safe to change the scheme at all: a directory
that predates it stays listed and readable.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.run_lineage import (
    GENERATED_RUN_ID,
    REGISTRY_FILENAME,
    AnalysisLineage,
    AnalysisRecord,
    RunFingerprint,
    format_run_id,
)


def _fingerprint(identity: str, **components) -> RunFingerprint:
    base = {
        "analysis": "a1",
        "markers": "m1",
        "delineation": "d1",
        "signal": "s1",
        "config": "c1",
        "panels": "p1",
    }
    base.update(components)
    return RunFingerprint(fingerprint=identity, components=base)


# ------------------------------------------------------------------ naming


def test_the_generated_name_is_sortable_recognisable_and_stable():
    name = format_run_id(7, "2026-08-20T09:15:00+00:00", "a3f9c2e1d4")
    assert name == "A007-2026-08-20-a3f9c2"
    assert GENERATED_RUN_ID.match(name)
    # Same inputs, same name: the identifier is derived, not allocated.
    assert name == format_run_id(7, "2026-08-20T23:59:59+00:00", "a3f9c2e1d4")


def test_generations_sort_in_order_as_plain_strings():
    names = [
        format_run_id(generation, "2026-08-20", "abcdef")
        for generation in (2, 10, 1)
    ]
    assert sorted(names) == [
        "A001-2026-08-20-abcdef",
        "A002-2026-08-20-abcdef",
        "A010-2026-08-20-abcdef",
    ]


def test_a_hand_written_directory_name_is_not_mistaken_for_a_generated_one():
    assert not GENERATED_RUN_ID.match("mimic-clinical-development-bigRun8")
    assert not GENERATED_RUN_ID.match("A7-2026-08-20-abcdef")


# ------------------------------------------------------------- resolution


def test_the_same_configuration_continues_its_analysis(tmp_path):
    """The expensive direction: a restart must not fork a new run."""

    lineage = AnalysisLineage(tmp_path)
    first = lineage.resolve(_fingerprint("identity-a"))
    again = AnalysisLineage(tmp_path).resolve(_fingerprint("identity-a"))
    assert again.run_id == first.run_id
    assert again.generation == first.generation
    assert len(AnalysisLineage(tmp_path).records) == 1


def test_a_changed_configuration_starts_a_new_generation(tmp_path):
    """The dangerous direction: two studies must not merge."""

    lineage = AnalysisLineage(tmp_path)
    first = lineage.resolve(_fingerprint("identity-a"))
    second = lineage.resolve(_fingerprint("identity-b", markers="m2"))
    assert second.run_id != first.run_id
    assert second.generation == first.generation + 1


def test_the_new_generation_records_what_changed(tmp_path):
    lineage = AnalysisLineage(tmp_path)
    lineage.resolve(_fingerprint("identity-a"))
    second = lineage.resolve(
        _fingerprint("identity-b", markers="m2", delineation="d2")
    )
    assert second.changed_from_previous == ("delineation", "markers")
    description = second.describe_change_de()
    assert "Delineation" in description
    assert "Marker-Code" in description


def test_analysis_backend_changes_are_named_explicitly(tmp_path):
    lineage = AnalysisLineage(tmp_path)
    lineage.resolve(_fingerprint("identity-a"))
    second = lineage.resolve(_fingerprint("identity-b", analysis="a2"))

    assert second.changed_from_previous == ("analysis",)
    assert "Analyse-Backend" in second.describe_change_de()


def test_the_first_analysis_says_so(tmp_path):
    record = AnalysisLineage(tmp_path).resolve(_fingerprint("identity-a"))
    assert record.changed_from_previous == ()
    assert "Erste Analyse" in record.describe_change_de()


def test_component_changes_never_decide_identity(tmp_path):
    """Components explain; the fingerprint decides.

    A decomposition that misses something produces a wrong sentence. An
    identity that misses something merges two studies. They are kept apart on
    purpose, and this test pins the direction of the dependency.
    """

    lineage = AnalysisLineage(tmp_path)
    first = lineage.resolve(_fingerprint("identity-a"))
    # Different components, same identity -> still the same analysis.
    same = lineage.resolve(_fingerprint("identity-a", markers="something-else"))
    assert same.run_id == first.run_id
    assert len(lineage.records) == 1


# ----------------------------------------------------------------- legacy


def test_directories_from_before_the_scheme_are_adopted(tmp_path):
    (tmp_path / "mimic-clinical-development-bigRun8").mkdir()
    (tmp_path / "mimic-development").mkdir()
    lineage = AnalysisLineage(tmp_path)
    adopted = lineage.adopt_existing_directories()
    assert {record.run_id for record in adopted} == {
        "mimic-clinical-development-bigRun8",
        "mimic-development",
    }
    for record in adopted:
        assert record.legacy
        # No fingerprint is invented: what produced them cannot be known.
        assert record.fingerprint == ""
        assert "vor der automatischen Benennung" in record.describe_change_de()


def test_adoption_is_idempotent(tmp_path):
    (tmp_path / "mimic-development").mkdir()
    lineage = AnalysisLineage(tmp_path)
    lineage.adopt_existing_directories()
    lineage.adopt_existing_directories()
    assert len(lineage.records) == 1


def test_a_legacy_run_never_becomes_the_thing_a_new_run_continues(tmp_path):
    """A legacy entry has no fingerprint and must not match anything."""

    (tmp_path / "mimic-development").mkdir()
    lineage = AnalysisLineage(tmp_path)
    lineage.adopt_existing_directories()
    assert lineage.by_fingerprint("") is None
    record = lineage.resolve(_fingerprint("identity-a"))
    assert record.run_id != "mimic-development"
    assert record.generation == 1


def test_a_file_in_the_runs_directory_is_not_adopted_as_a_run(tmp_path):
    (tmp_path / "feature_failure_report.parquet").write_bytes(b"")
    lineage = AnalysisLineage(tmp_path)
    assert lineage.adopt_existing_directories() == ()


# ------------------------------------------------------------------ notes


def test_notes_can_be_attached_and_survive_a_reload(tmp_path):
    lineage = AnalysisLineage(tmp_path)
    record = lineage.resolve(_fingerprint("identity-a"))
    lineage.set_notes(record.run_id, "Erster Lauf nach dem Delineations-Fix")
    reloaded = AnalysisLineage(tmp_path).by_run_id(record.run_id)
    assert reloaded is not None
    assert reloaded.notes == "Erster Lauf nach dem Delineations-Fix"


def test_notes_do_not_change_the_identity(tmp_path):
    """A note is not a result; annotating must not fork the analysis."""

    lineage = AnalysisLineage(tmp_path)
    record = lineage.resolve(_fingerprint("identity-a"))
    lineage.set_notes(record.run_id, "irgendetwas")
    again = lineage.resolve(_fingerprint("identity-a"))
    assert again.run_id == record.run_id
    assert again.fingerprint == record.fingerprint


def test_notes_can_be_written_for_a_legacy_run(tmp_path):
    """Frozen and legacy runs must stay annotatable.

    Notes live in the registry rather than in the run directory precisely so
    that a finished analysis can be described without touching its results.
    """

    (tmp_path / "mimic-development").mkdir()
    lineage = AnalysisLineage(tmp_path)
    lineage.adopt_existing_directories()
    assert lineage.set_notes("mimic-development", "Der alte Lauf") is not None
    assert AnalysisLineage(tmp_path).by_run_id("mimic-development").notes == (
        "Der alte Lauf"
    )


def test_notes_for_an_unknown_run_are_refused(tmp_path):
    assert AnalysisLineage(tmp_path).set_notes("nicht-da", "x") is None


# --------------------------------------------------------------- storage


def test_a_corrupt_registry_does_not_end_a_run(tmp_path):
    """Losing the registry costs the history, not the analysis."""

    (tmp_path / REGISTRY_FILENAME).write_text('{"analyses": [', encoding="utf-8")
    lineage = AnalysisLineage(tmp_path)
    assert lineage.records == []
    record = lineage.resolve(_fingerprint("identity-a"))
    assert record.generation == 1


def test_the_registry_round_trips(tmp_path):
    lineage = AnalysisLineage(tmp_path)
    lineage.resolve(_fingerprint("identity-a"))
    lineage.resolve(_fingerprint("identity-b", config="c2"))
    payload = json.loads((tmp_path / REGISTRY_FILENAME).read_text(encoding="utf-8"))
    assert len(payload["analyses"]) == 2
    restored = AnalysisLineage(tmp_path)
    assert [item.run_id for item in restored.records] == [
        item.run_id for item in lineage.records
    ]


def test_generation_numbers_do_not_repeat_after_a_reload(tmp_path):
    AnalysisLineage(tmp_path).resolve(_fingerprint("identity-a"))
    AnalysisLineage(tmp_path).resolve(_fingerprint("identity-b", config="c2"))
    third = AnalysisLineage(tmp_path).resolve(
        _fingerprint("identity-c", config="c3")
    )
    assert third.generation == 3
    assert len({item.run_id for item in AnalysisLineage(tmp_path).records}) == 3


@pytest.mark.parametrize("field_name", ["generation", "fingerprint", "run_id"])
def test_a_record_survives_a_missing_field(tmp_path, field_name):
    """A registry written by an older version must not be fatal."""

    payload = AnalysisRecord(
        run_id="A001-2026-08-20-abcdef",
        generation=1,
        fingerprint="abcdef",
        created_at="2026-08-20T00:00:00+00:00",
    ).to_dict()
    del payload[field_name]
    (tmp_path / REGISTRY_FILENAME).write_text(
        json.dumps({"analyses": [payload]}), encoding="utf-8"
    )
    assert len(AnalysisLineage(tmp_path).records) == 1
