"""What defines a result must be in the key that decides reuse.

Several stages build their reuse key from **declared version constants**
rather than from a source hash. That is a deliberate design -- a version
constant is an explicit statement that the computation changed, and it does
not invalidate a month of work because a comment moved. It only works if the
constant is raised when the computation changes.

On 2026-08-29 it did not. Three scientific corrections went into the
physiological gap markers -- a regression-to-the-mean correction, plausibility
bounds on the target, and a usability floor -- while ``GAP_MODEL_VERSION``
stayed at 1.1.0. Its reuse key carries no source hash, so the artifact
computed the previous evening, *before* the corrections, would have been
reused and all three would have been silently absent from the run.

The same shape appeared in the broad-axis screen: a prespecified effect floor
was added next to alphas that were already in the key, and was not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _input_hash_payload(module: str) -> str:
    """The source of every ``input_hash = content_hash(...)`` in a module.

    Read through the syntax tree rather than a text pattern: the first version
    of this test matched on indentation and missed a key that was indented
    differently, which is the same class of miss it exists to catch.
    """

    import ast

    tree = ast.parse(
        Path(f"vcg_biomarker/mimic/{module}.py").read_text(encoding="utf-8")
    )
    pieces: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "input_hash" not in targets:
            continue
        pieces.append(ast.unparse(node.value))
    assert pieces, f"{module}: kein input_hash gefunden"
    return chr(10).join(pieces)


@pytest.mark.parametrize(
    ("module", "constants"),
    [
        ("physiological_gap_models", ("GAP_MODEL_VERSION",)),
        (
            "broad_axis_stage",
            (
                "TRAIN_FDR_ALPHA",
                "VALIDATION_FDR_ALPHA",
                "MINIMUM_ABSOLUTE_RHO",
                "MINIMUM_ABSOLUTE_AUC_MARGIN",
            ),
        ),
        (
            "repeatability_stage",
            ("REPEATABILITY_VERSION", "TRAJECTORY_VERSION"),
        ),
    ],
)
def test_the_result_defining_constants_are_in_the_reuse_key(module, constants):
    payload = _input_hash_payload(module)
    missing = [name for name in constants if name not in payload]
    assert not missing, (
        f"{module}: {missing} bestimmen das Ergebnis, stehen aber nicht im "
        "Wiederverwendungsschluessel -- eine Aenderung daran wuerde ein "
        "Artefakt wiederverwenden, das unter anderen Werten entstand"
    )


def test_the_gap_version_was_raised_with_its_corrections():
    """The specific miss, pinned so it cannot recur unnoticed."""

    from vcg_biomarker.mimic.physiological_gap_models import (
        GAP_MODEL_VERSION,
        GAP_STAGE_VERSION,
    )

    assert GAP_MODEL_VERSION != "1.1.0", (
        "1.1.0 ist die Fassung ohne Bias-Korrektur, ohne Zielgrenzen und "
        "ohne Brauchbarkeitsschwelle"
    )
    assert GAP_STAGE_VERSION != "1.1.0"


def test_the_blocked_artifact_formats_carry_a_new_version():
    """The replicate-pair artifact became a directory of part files."""

    from vcg_biomarker.mimic.repeatability import REPEATABILITY_VERSION
    from vcg_biomarker.mimic.repeatability_stage import (
        REPEATABILITY_STAGE_VERSION,
    )
    from vcg_biomarker.mimic.trajectory import TRAJECTORY_VERSION

    assert TRAJECTORY_VERSION != "1.0.0"
    assert REPEATABILITY_VERSION != "1.0.0"
    assert REPEATABILITY_STAGE_VERSION != "1.0.0"


def test_every_stage_version_is_a_three_part_number():
    """A version that stops looking like one stops being raised."""

    from vcg_biomarker.mimic import (
        broad_axis_stage,
        physiological_gap_models,
        repeatability,
        repeatability_stage,
        trajectory,
    )

    versions = {
        "BROAD_AXIS_STAGE_VERSION": broad_axis_stage.BROAD_AXIS_STAGE_VERSION,
        "GAP_MODEL_VERSION": physiological_gap_models.GAP_MODEL_VERSION,
        "GAP_STAGE_VERSION": physiological_gap_models.GAP_STAGE_VERSION,
        "REPEATABILITY_VERSION": repeatability.REPEATABILITY_VERSION,
        "REPEATABILITY_STAGE_VERSION": (
            repeatability_stage.REPEATABILITY_STAGE_VERSION
        ),
        "TRAJECTORY_VERSION": trajectory.TRAJECTORY_VERSION,
    }
    for name, value in versions.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", str(value)), f"{name}={value!r}"
