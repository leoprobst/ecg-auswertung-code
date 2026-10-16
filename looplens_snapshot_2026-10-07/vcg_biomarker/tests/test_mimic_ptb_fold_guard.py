from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic.ptb_transfer import (
    PTB_ENDPOINT_SPECS,
    _assert_patient_fold_integrity,
    _load_ptb_frame,
    ptb_implementation_hash,
)
from vcg_biomarker.mimic.validation import (
    DevelopmentValidationOptions,
    _load_ptb_transfer_inputs,
)


def _ptb_feature_frame() -> pd.DataFrame:
    components = tuple(
        dict.fromkeys(
            component
            for endpoint in PTB_ENDPOINT_SPECS
            for component in endpoint.components
        )
    )
    labels = tuple(
        dict.fromkeys(
            label
            for endpoint in PTB_ENDPOINT_SPECS
            for label in endpoint.source_columns
        )
    )
    frame = pd.DataFrame({"ecg_id": [1, 2, 3]})
    for index, component in enumerate(components):
        frame[f"bm_{component}"] = [index, index + 1, 999_999]
    for label in labels:
        frame[label] = [0, 1, 1]
    return frame


def test_ptb_development_loader_never_reads_fold_10(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [10, 20, 30],
            "strat_fold": [1, 9, 10],
        }
    )
    metadata_path = tmp_path / "ptbxl_database.csv"
    cache_path = tmp_path / "features.parquet"
    metadata.to_csv(metadata_path, index=False)
    _ptb_feature_frame().to_parquet(cache_path, index=False)

    development = _load_ptb_frame(
        metadata_path,
        cache_path,
        folds=tuple(range(1, 10)),
    )
    assert set(development["ecg_id"]) == {1, 2}
    assert 10 not in set(development["strat_fold"])
    numeric = development.select_dtypes(include=[np.number])
    assert not numeric.eq(999_999).any().any()


def test_ptb_patient_overlap_across_folds_is_rejected() -> None:
    leaking = pd.DataFrame(
        {
            "subject_id": [100, 100],
            "strat_fold": [1, 9],
        }
    )
    with pytest.raises(RuntimeError, match="leakage"):
        _assert_patient_fold_integrity(leaking)


def test_ptb_hash_tracks_only_scientifically_relevant_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import ptb_transfer

    tracked = []

    def capture(path, *args, **kwargs):
        tracked.append(Path(path).name)
        return f"hash:{Path(path).name}"

    monkeypatch.setattr(ptb_transfer, "file_sha256", capture)
    value = ptb_implementation_hash()
    assert value
    assert tracked == [
        "ptb_transfer.py",
        "statistics.py",
        "models.py",
    ]
    assert not any("gui" in item for item in tracked)


def test_stale_ptb_freeze_does_not_abort_mimic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale/changed PTB code hash must degrade the supplementary
    PTB transfer comparison gracefully instead of raising and aborting
    the primary MIMIC train/validation run (regression test for the
    study-run failure at stage `validate`, job
    study-run-20260802T055447-38571168, step 18/21)."""
    from vcg_biomarker.mimic import validation as validation_module

    def _raise_stale(*_args, **_kwargs):
        raise RuntimeError(
            "PTB freeze refused: code changed after revalidation"
        )

    monkeypatch.setattr(
        validation_module,
        "load_ptb_frozen_transfer_scores",
        _raise_stale,
    )

    options = DevelopmentValidationOptions(include_ptb_frozen_scores=True)
    scores, provenance, available = _load_ptb_transfer_inputs(
        config=object(),
        options=options,
    )

    assert scores == []
    assert available is False
    assert provenance["status"] == "stale_freeze"
    assert "code changed after revalidation" in provenance["reason"]


def test_missing_ptb_revalidation_inputs_does_not_abort_mimic_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vcg_biomarker.mimic import validation as validation_module

    def _raise_missing(*_args, **_kwargs):
        raise FileNotFoundError("PTB-XL revalidation inputs are missing: []")

    monkeypatch.setattr(
        validation_module,
        "load_ptb_frozen_transfer_scores",
        _raise_missing,
    )

    options = DevelopmentValidationOptions(include_ptb_frozen_scores=True)
    scores, provenance, available = _load_ptb_transfer_inputs(
        config=object(),
        options=options,
    )

    assert scores == []
    assert available is False
    assert provenance["status"] == "stale_freeze"


def test_ptb_transfer_disabled_option_is_reported_as_disabled() -> None:
    options = DevelopmentValidationOptions(include_ptb_frozen_scores=False)
    scores, provenance, available = _load_ptb_transfer_inputs(
        config=object(),
        options=options,
    )
    assert scores == []
    assert available is False
    assert provenance == {
        "status": "disabled",
        "run_id": options.ptb_transfer_run_id,
    }
