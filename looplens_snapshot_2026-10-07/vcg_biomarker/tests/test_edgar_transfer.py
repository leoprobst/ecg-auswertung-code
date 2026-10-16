import numpy as np
import pytest

from vcg_biomarker.simulation.edgar_transfer import (
    average, common_reference, loss, moments, nested_site_benchmark, ridge_candidates, site_groups,
)


def test_covariance_loss_matches_direct():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(5, 90)), rng.normal(size=(7, 90))
    stats = moments(x, y)
    model = ridge_candidates(stats, (.01,))[0]
    assert loss(model, stats) == pytest.approx(np.sum((model@x-y)**2)/np.sum(y**2))
    assert loss(model, stats, True) == pytest.approx(
        np.sum((model@x-np.roll(y, 30, axis=1))**2)/np.sum(y**2))
    assert all(np.allclose(a, b) for a, b in zip(average([stats, stats]), stats))


def test_reference_and_constant_channels():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(6, 30))
    out, valid = common_reference(np.vstack([x, np.zeros((1, 30))]))
    assert valid.sum() == 6
    assert np.allclose(out[:6].mean(0), 0)
    assert not out[6].any()
    shifted, _ = common_reference(x+np.arange(30))
    assert np.allclose(out[:6], shifted)
    with pytest.raises(ValueError):
        common_reference(np.zeros((6, 30)))


def test_parent_and_reference_union():
    pairs = [{'epicardial_path': str(i), 'episode_id': str(i)} for i in range(4)]
    rows = {str(i): {'parent_record': str(i), 'reference_native': [i, 0, 0]} for i in range(4)}
    rows['1']['parent_record'] = '0'
    rows['2']['reference_native'] = [1, 0, 0]
    groups = site_groups(pairs, rows)
    assert sorted(map(len, groups.values())) == [1, 3]


def test_nested_exact_transfer_and_held_out_independence():
    rng = np.random.default_rng(88)
    transform = rng.normal(size=(4, 3))
    groups = {}
    for k in range(5):
        x = rng.normal(size=(3, 100))
        groups[str(k)] = moments(x, transform@x)
    result = nested_site_benchmark(groups)
    assert result['mean_site_nmse'] < 1e-8
    assert result['mean_shift_control_nmse'] > 1
    changed = dict(groups)
    x = rng.normal(size=(3, 100))
    changed['0'] = moments(x, rng.normal(size=(4, 100))*100)
    other = nested_site_benchmark(changed)
    # Changing outer target cannot change its own selected hyperparameter.
    assert result['sites'][0]['alpha'] == other['sites'][0]['alpha']
    assert nested_site_benchmark(dict(list(groups.items())[:3]))['status'] == 'non_estimable'
    with pytest.raises(InterruptedError):
        nested_site_benchmark(groups, lambda: True)
