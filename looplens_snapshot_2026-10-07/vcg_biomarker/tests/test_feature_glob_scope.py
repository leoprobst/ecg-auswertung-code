"""A stage must read the feature partitions and nothing else.

Three stages -- all of which had never run -- defaulted to
``features/<panel>/**/*.parquet``. On the real project that recursive pattern
matches **2,568** files where 2,563 are wanted:

* ``feature_failure_report.parquet``, a four-column diagnostic table
  (``extraction_status, error_type, error_message, record_count``) and
  alphabetically the **first** match, so it decides the bound schema;
* five partitions of ``smoke-development-1000`` -- the **pilot** cohort --
  pulled into a full-cohort analysis.

``union_by_name=true`` hid it completely: DuckDB merged the schemas, the query
bound, and the stage would have counted pilot recordings and failure-report
rows as ECGs. Removing that option for memory reasons turned the silent wrong
answer into a binder error, which is how it surfaced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from vcg_biomarker.mimic.storage import development_feature_glob


def test_the_glob_names_the_split_partitions_only():
    glob = development_feature_glob("/data/features", ("primary_core", "sentinel"))
    assert glob.replace("\\", "/").endswith(
        "primary_core_sentinel/development/split=*/part-*.parquet"
    )
    assert "**" not in glob


def test_a_pilot_scope_is_addressable_without_touching_the_full_cohort():
    full = development_feature_glob("/d", ("primary_core",))
    pilot = development_feature_glob(
        "/d", ("primary_core",), scope="smoke-development-1000"
    )
    assert full != pilot
    assert "development/split=" in full.replace("\\", "/")
    assert "smoke-development-1000/split=" in pilot.replace("\\", "/")


def test_the_pattern_excludes_the_files_that_broke_the_stage(tmp_path):
    """Reproduces the real directory, including the two intruders."""

    import glob as _glob

    root = tmp_path / "features" / "primary_core" / "development"
    for split in ("train", "validation"):
        folder = root / f"split={split}"
        folder.mkdir(parents=True)
        pd.DataFrame({"subject_id": [1], "study_id": [1]}).to_parquet(
            folder / "part-0.parquet", index=False
        )
    # The four-column diagnostic table that came first alphabetically.
    pd.DataFrame(
        {
            "extraction_status": ["signal_qc_failed"],
            "error_type": ["non_finite_samples"],
            "error_message": ["x"],
            "record_count": [3],
        }
    ).to_parquet(root / "feature_failure_report.parquet", index=False)
    # The pilot cohort.
    pilot = tmp_path / "features" / "primary_core" / "smoke-development-1000"
    (pilot / "split=train").mkdir(parents=True)
    pd.DataFrame({"subject_id": [9], "study_id": [9]}).to_parquet(
        pilot / "split=train" / "part-0.parquet", index=False
    )

    recursive = str(
        tmp_path / "features" / "primary_core" / "**" / "*.parquet"
    )
    loose = sorted(_glob.glob(recursive, recursive=True))
    assert len(loose) == 4
    assert any("feature_failure_report" in item for item in loose)
    assert any("smoke-development-1000" in item for item in loose)

    tight = sorted(
        _glob.glob(
            development_feature_glob(tmp_path / "features", ("primary_core",))
        )
    )
    assert len(tight) == 2
    assert not any("feature_failure_report" in item for item in tight)
    assert not any("smoke-development-1000" in item for item in tight)


def test_no_stage_still_defaults_to_a_recursive_feature_glob():
    """The pattern must not come back anywhere that reads features."""

    offenders = []
    pattern = re.compile(r'features_root[\s\S]{0,120}?"\*\*"')
    for path in sorted(Path("vcg_biomarker/mimic").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            offenders.append(f"{path.name}:{text[:match.start()].count(chr(10)) + 1}")
    assert not offenders, offenders


@pytest.mark.parametrize(
    "stage", ["repeatability", "screen-broad-axes", "deep-combination-search"]
)
def test_the_guided_run_passes_an_explicit_glob(stage, tmp_path):
    """A default is a fallback, not the path a month-long run relies on."""

    from vcg_biomarker.mimic.config import MimicProjectConfig
    from vcg_biomarker.mimic.presets import STUDY_PRESET_ID
    from vcg_biomarker.mimic.workflow import _stage_arguments

    config = MimicProjectConfig.discover(
        database_root=tmp_path / "database", work_root=tmp_path / "work"
    )
    arguments = _stage_arguments(
        config, stage, preset_id=STUDY_PRESET_ID, run_id="glob-test"
    )
    glob = arguments.get("feature_glob")
    assert glob, stage
    assert "**" not in str(glob)
    assert "split=" in str(glob).replace("\\", "/")
