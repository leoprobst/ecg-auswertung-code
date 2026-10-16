"""Rewording a docstring cost 25,8 hours of computation.

``mimic/storage.py`` is in the validation scope and holds the stage manifest,
the hashing, *and* the parquet read clauses. The provenance hash was already
taken over the parsed tree, so comments never mattered -- but docstrings did.
Rewording one shifted the validation code hash, and with it the reuse key of
the pilot validations: ``validate-pilot-1000`` at 23,9 h and
``validate-pilot-10000`` at 1,9 h. On 2026-08-28 the stored hash was
``22c26fb6...`` against a current ``e127101f...``, and the run was rescued by a
rule about spent pilot stages rather than by removing the cause.

The proposal on file was to split ``storage.py`` in two and scope only the
result-defining half. Measured against what the file actually contains, that
split does not hold: ``development_feature_glob`` decides which recordings
enter an analysis, and ``feature_parquet_clause`` decides how they bind -- the
defect where a recursive glob pulled the pilot partitions into a full-cohort
stage came from precisely there. Both look like I/O and are result-defining, so
the split would have put nearly everything on the scoped side and would not
have touched the docstring problem.

Dropping docstrings does, and it is the more honest rule: if the only thing
that changed is the prose, no number changed with it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from vcg_biomarker.mimic.validation_provenance import (
    _normalized_validation_source_sha256 as normalized_hash,
)


SOURCE = Path("vcg_biomarker/mimic/storage.py")


def _hash_variant(text: str) -> str:
    directory = Path(tempfile.mkdtemp())
    path = directory / "variant.py"
    path.write_text(text, encoding="utf-8")
    return normalized_hash(path)


@pytest.fixture(scope="module")
def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def baseline(source_text) -> str:
    return _hash_variant(source_text)


# --- Prose must not count ------------------------------------------------------


def test_rewording_a_docstring_does_not_move_the_hash(source_text, baseline):
    """The defect, on the file it actually happened to."""

    marker = '"""Column names of a parquet path or glob, without unifying schemas.'
    assert marker in source_text, "Ankerpunkt fuer den Test fehlt"
    reworded = source_text.replace(
        marker, '"""Spaltennamen, ohne Schemata zu vereinen.', 1
    )
    assert reworded != source_text
    assert _hash_variant(reworded) == baseline


def test_changing_a_comment_does_not_move_the_hash(source_text, baseline):
    """Already true before; pinned so the tree-based hashing is not lost."""

    marker = (
        "    # ``connection`` is accepted for call-site symmetry and is not used:"
    )
    assert marker in source_text, "Ankerpunkt fuer den Test fehlt"
    changed = source_text.replace(marker, "    # Anderer Wortlaut.", 1)
    assert _hash_variant(changed) == baseline


def test_a_module_docstring_can_be_replaced_entirely(source_text, baseline):
    start = source_text.index('"""')
    end = source_text.index('"""', start + 3) + 3
    replaced = source_text[:start] + '"""Kurz."""' + source_text[end:]
    assert replaced != source_text
    assert _hash_variant(replaced) == baseline


def test_a_function_that_is_only_a_docstring_stays_parseable():
    """Deleting the docstring would leave an empty body and a syntax error."""

    text = "def stub():\n    \"\"\"Nur ein Docstring.\"\"\"\n"
    other = "def stub():\n    \"\"\"Anderer Text.\"\"\"\n"
    assert _hash_variant(text) == _hash_variant(other)


def test_a_class_that_is_only_a_docstring_stays_parseable():
    text = "class Stub:\n    \"\"\"Nur ein Docstring.\"\"\"\n"
    other = "class Stub:\n    \"\"\"Anderer Text.\"\"\"\n"
    assert _hash_variant(text) == _hash_variant(other)


# --- Everything that decides a number must count -------------------------------


def test_changing_sql_inside_a_read_clause_moves_the_hash(source_text, baseline):
    """The read clause is prose-documented and result-defining at once."""

    marker = "union_by_name=true)\""
    assert marker in source_text
    changed = source_text.replace(marker, "union_by_name=false)\"", 1)
    assert _hash_variant(changed) != baseline


def test_changing_a_constant_moves_the_hash(source_text, baseline):
    marker = "_PROBE_TTL_SECONDS = 30"
    assert marker in source_text
    changed = source_text.replace(marker, "_PROBE_TTL_SECONDS = 60", 1)
    assert _hash_variant(changed) != baseline


def test_renaming_a_function_moves_the_hash(source_text, baseline):
    marker = "def development_feature_glob("
    assert marker in source_text
    changed = source_text.replace(marker, "def development_feature_pattern(", 1)
    assert _hash_variant(changed) != baseline


@pytest.mark.parametrize(
    "before, after",
    [
        ("if probe is None:", "if probe is not None:"),
        ("return None", "return {}"),
        ("chunk_size: int = 8 * 1024 * 1024", "chunk_size: int = 4 * 1024 * 1024"),
    ],
)
def test_control_flow_and_defaults_still_count(source_text, baseline, before, after):
    assert before in source_text, f"Stabiler Ankerpunkt fehlt: {before}"
    changed = source_text.replace(before, after, 1)
    assert _hash_variant(changed) != baseline


def test_a_string_that_is_not_a_docstring_still_counts():
    """Only the *leading* string of a body is prose; the rest are values."""

    text = 'def rule():\n    return "positive"\n'
    other = 'def rule():\n    return "negative"\n'
    assert _hash_variant(text) != _hash_variant(other)


def test_a_docstring_after_a_statement_is_a_value_not_prose():
    text = 'def f():\n    x = 1\n    "kommentarartig"\n    return x\n'
    other = 'def f():\n    x = 1\n    "anders"\n    return x\n'
    assert _hash_variant(text) != _hash_variant(other), (
        "eine Zeichenkette mitten im Koerper ist kein Docstring und muss "
        "zaehlen"
    )
