"""The four widest reads must keep their bound, and say so structurally.

Each was measured before it was fixed:

===============================================  ===========
Stelle                                           angefordert
===============================================  ===========
``screen-broad-axes``, Laborachse                    226 GB
``screen-broad-axes``, ICU-Achse                      74 GB
``trajectory.build_replicate_pairs`` (Self-Join)   2 x  9 GB
``trajectory.build_trajectory_frame``                  9 GB
===============================================  ===========

The preflight checker cannot protect them: it flags a wide column list built
from the *full* marker collection, and blocked code builds it from a block
variable instead -- so it reports "ok" precisely because the fix is in place.
That is correct behaviour for a risk checker and useless as a regression
guard, which is what this file is.

Behavioural equivalence lives in test_broad_axis_blocking.py,
test_trajectory_blocking.py and test_repeatability_blocking.py. Here we only
assert that the bound still exists at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


BLOCKED_FUNCTIONS = (
    ("broad_axis_stage", "_screen_one_axis", "_marker_blocks"),
    ("trajectory", "build_replicate_pairs", "_marker_blocks"),
    ("trajectory", "build_trajectory_frame", "_marker_blocks"),
    ("repeatability", "repeatability_table_from_parquet", "block_size"),
)


def _function(module: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(
        Path(f"vcg_biomarker/mimic/{module}.py").read_text(encoding="utf-8")
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{module}.{name} existiert nicht mehr")


@pytest.mark.parametrize(("module", "function", "token"), BLOCKED_FUNCTIONS)
def test_the_wide_read_still_goes_through_blocks(module, function, token):
    node = _function(module, function)
    names = {
        sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
    } | {
        sub.arg for sub in ast.walk(node) if isinstance(sub, ast.arg)
    }
    assert token in names, (
        f"{module}.{function} baut seinen Rahmen nicht mehr blockweise -- "
        "das war die Aenderung, die eine 226-GB-Anforderung verhindert hat"
    )


@pytest.mark.parametrize(("module", "function", "_token"), BLOCKED_FUNCTIONS)
def test_the_wide_read_still_loops(module, function, _token):
    """A block helper that is called once and ignored would pass the test above."""

    node = _function(module, function)
    assert any(
        isinstance(sub, (ast.For, ast.comprehension)) for sub in ast.walk(node)
    ), f"{module}.{function} hat keine Schleife mehr"


def test_the_budget_constants_are_still_finite():
    from vcg_biomarker.mimic.broad_axis_stage import BROAD_AXIS_CELL_BUDGET
    from vcg_biomarker.mimic.repeatability import REPEATABILITY_COLUMN_BLOCK
    from vcg_biomarker.mimic.trajectory import TRAJECTORY_MARKER_BLOCK

    for value in (
        BROAD_AXIS_CELL_BUDGET,
        REPEATABILITY_COLUMN_BLOCK,
        TRAJECTORY_MARKER_BLOCK,
    ):
        assert isinstance(value, int) and value > 0

    # 1,713 markers over 13,232,238 rows must not come back as one frame.
    assert BROAD_AXIS_CELL_BUDGET < 13_232_238 * 1_713
    assert TRAJECTORY_MARKER_BLOCK < 1_753
