"""Retiring superseded feature artifacts instead of proving evidence equality.

The evidence guard offers one escape: prove the event layer is unchanged by
rebuilding the whole thing into a shadow root. That is the right answer when
the evidence is what stayed the same. When the *panel* moved on, the artifacts
being protected are the obsolete ones, and an hour of shadow rebuild proves
something about files nobody will read again.
"""

from __future__ import annotations

import json

import pytest

from vcg_biomarker.mimic.feature_retirement import (
    bound_development_artifacts,
    retire_superseded_development_features,
)


class _Config:
    def __init__(self, root):
        self.work_root = root
        self.features_root = root / "features"

    def to_dict(self) -> dict:
        return {}


def _artifact(root, panel: str, scope: str, *, locked: bool = False) -> None:
    directory = root / "features" / panel / scope
    split = "locked_test" if locked else "train"
    (directory / f"split={split}").mkdir(parents=True, exist_ok=True)
    (directory / f"split={split}" / "part-000000.parquet").write_bytes(b"x" * 32)
    (directory / "features.manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "notes": [
                    f"panels={panel}",
                    f"locked_test_features_included={locked}",
                ],
            }
        ),
        encoding="utf-8",
    )


def test_it_lists_exactly_what_binds_the_evidence(tmp_path) -> None:
    _artifact(tmp_path, "primary_core_sentinel", "development")
    _artifact(tmp_path, "primary_core_sentinel", "smoke-development-1000")
    found = bound_development_artifacts(_Config(tmp_path))
    assert {(panel, scope) for panel, scope, _ in found} == {
        ("primary_core_sentinel", "development"),
        ("primary_core_sentinel", "smoke-development-1000"),
    }


def test_retiring_unbinds_the_evidence_and_keeps_the_files(tmp_path) -> None:
    from vcg_biomarker.mimic.evidence import _has_bound_development_features

    _artifact(tmp_path, "primary_core_sentinel", "development")
    config = _Config(tmp_path)
    assert _has_bound_development_features(config)

    summary = retire_superseded_development_features(
        config, reason="panel superseded"
    )
    # The guard stands down...
    assert not _has_bound_development_features(config)
    # ...and nothing was destroyed to achieve it.
    assert len(summary["retired"]) == 1
    archived = tmp_path / summary["retired"][0]["archive_path"]
    archive_root = tmp_path / "archives" / "superseded-features"
    surviving = list(archive_root.rglob("part-000000.parquet"))
    assert surviving, "artifacts must be moved, not deleted"
    manifest = json.loads(
        next(archive_root.rglob("retirement.manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["reason"] == "panel superseded"


def test_locked_test_artifacts_are_never_moved(tmp_path) -> None:
    """Convenience tooling does not get to touch the sealed split."""

    _artifact(tmp_path, "primary_core_sentinel", "development", locked=True)
    _artifact(tmp_path, "primary_core_sentinel", "other")
    summary = retire_superseded_development_features(
        _Config(tmp_path), reason="panel superseded"
    )
    retired = {item["scope"] for item in summary["retired"]}
    assert retired == {"other"}
    assert summary["skipped"][0]["scope"] == "development"
    assert (
        tmp_path / "features" / "primary_core_sentinel" / "development"
    ).is_dir()


def test_unreadable_provenance_is_treated_as_possibly_sealed(tmp_path) -> None:
    directory = tmp_path / "features" / "panel" / "development"
    (directory / "split=train").mkdir(parents=True)
    (directory / "split=train" / "part-000000.parquet").write_bytes(b"x")
    (directory / "features.manifest.json").write_text("{broken", encoding="utf-8")
    summary = retire_superseded_development_features(
        _Config(tmp_path), reason="panel superseded"
    )
    # Cannot read what it is, so it does not get moved.
    assert summary["retired"] == []
    assert summary["skipped"][0]["scope"] == "development"


def test_a_reason_is_required(tmp_path) -> None:
    _artifact(tmp_path, "panel", "development")
    with pytest.raises(ValueError, match="stated reason"):
        retire_superseded_development_features(_Config(tmp_path), reason="  ")


def test_scopes_can_be_restricted(tmp_path) -> None:
    _artifact(tmp_path, "panel", "development")
    _artifact(tmp_path, "panel", "smoke-development-20")
    summary = retire_superseded_development_features(
        _Config(tmp_path),
        reason="only the pilot",
        scopes=[("panel", "smoke-development-20")],
    )
    assert [item["scope"] for item in summary["retired"]] == [
        "smoke-development-20"
    ]
    assert (tmp_path / "features" / "panel" / "development").is_dir()
