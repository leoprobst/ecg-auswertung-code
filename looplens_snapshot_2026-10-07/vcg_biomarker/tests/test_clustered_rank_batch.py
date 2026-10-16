import numpy as np
import pytest

from vcg_biomarker.mimic.clustered_rank_association import (
    ClusteredRankBatch, clustered_rank_association,
)


@pytest.mark.parametrize("identity", ["valid", "few", "missing", "none"])
def test_batch_equals_scalar_for_ties_missingness_and_degeneracy(identity):
    rng = np.random.default_rng(197)
    x = rng.integers(0, 20, 400).astype(float)
    x[2] = np.nan
    subjects = np.repeat(np.arange(100)+1, 4).astype(object)
    if identity == "few": subjects[:] = 1
    if identity == "missing": subjects[3] = None
    if identity == "none": subjects = None
    batch = ClusteredRankBatch(x, subjects)
    values = [rng.normal(size=400), np.ones(400), rng.integers(0, 10, 400).astype(float)]
    values += [values[0].copy(), values[0].copy()]
    values[-2][10] = np.inf
    values[-1][11] = np.nan  # equal count but different patients is not a cache hit
    for y in values:
        actual, mask = batch.estimate(y)
        expected = clustered_rank_association(x[mask], y[mask],
            subjects[mask] if subjects is not None else None)
        assert actual.keys() == expected.keys()
        for key in expected:
            if isinstance(expected[key], float):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
            else:
                assert actual[key] == expected[key]
    assert batch.preparations == 3
    assert batch.cache_hits == 2


def test_batch_owns_inputs_and_does_not_prepare_underpowered_marker():
    x = np.arange(100.)
    subjects = np.arange(100)+1
    batch = ClusteredRankBatch(x, subjects)
    x[:] = 0
    subjects[:] = 1
    estimate, _ = batch.estimate(np.full(100, np.nan))
    assert estimate is None and batch.preparations == 0
    estimate, _ = batch.estimate(np.sin(np.arange(100.)))
    assert estimate["association_status"] == "estimable"
    assert estimate["n_patients"] == 100


def test_constant_exposure_preserves_abstention():
    batch = ClusteredRankBatch(np.ones(100), np.arange(100)+1)
    estimate, _ = batch.estimate(np.arange(100.))
    assert estimate["abstention_reason"] == "insufficient_or_constant_values"


def test_shape_and_identity_length_validation():
    with pytest.raises(ValueError): ClusteredRankBatch(np.ones((3, 2)), None)
    with pytest.raises(ValueError): ClusteredRankBatch(np.ones(100), [1])
    with pytest.raises(ValueError): ClusteredRankBatch(np.ones(100), None).estimate(np.ones(99))


@pytest.mark.parametrize("design", ["lab", "dose"])
def test_productive_paths_share_preparation_and_match_scalar(monkeypatch, design):
    import pandas as pd
    from vcg_biomarker.mimic import clustered_rank_association as core
    from vcg_biomarker.mimic.within_subject import within_subject_association, dose_response_association

    rng = np.random.default_rng(214)
    exposure = rng.normal(size=400)
    markers = {f"delta_m{i}": rng.normal(size=400) for i in range(4)}
    markers["delta_m2"][20] = np.nan
    markers["delta_m3"][20] = np.inf
    frame = pd.DataFrame(dict(subject_id=np.repeat(np.arange(100)+1, 4),
        delta_lab=exposure, dose_rank=exposure, **markers), index=[9]*400)
    calls = []
    prepare = core._prepare
    def counted(*args):
        calls.append(1)
        return prepare(*args)
    with monkeypatch.context() as patch:
        patch.setattr(core, "_prepare", counted)
        function = within_subject_association if design == "lab" else dose_response_association
        result = function(frame, [f"m{i}" for i in range(4)]).set_index("feature_id")
    assert len(calls) == 2  # actual masks, including duplicate DataFrame index labels
    for marker, y in markers.items():
        usable = np.isfinite(exposure) & np.isfinite(y)
        expected = clustered_rank_association(exposure[usable], y[usable], frame.subject_id.to_numpy()[usable])
        row = result.loc[marker.removeprefix("delta_")]
        for key, value in expected.items():
            assert row[key] == value
        assert row.n_pairs == usable.sum()
        assert row.median_delta_marker == np.median(y[usable])
