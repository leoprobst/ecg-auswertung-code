import numpy as np
import pytest
from scipy.ndimage import median_filter
from vcg_biomarker.core.running_median import centered_median


@pytest.mark.parametrize("samples", [1, 2, 7, 103, 2048])
@pytest.mark.parametrize("window", [1, 3, 13, 101, 1001])
def test_exact_order_statistic_including_reflected_edges(samples, window):
    rng = np.random.default_rng(108)
    values = rng.normal(size=(samples, 3))
    values[:, 1] = np.round(values[:, 1], 1)  # repeated values
    values[:, 2] = 0.
    if window > samples:
        # Explicit order-statistic oracle: the installed SciPy build returns
        # unstable non-input values for some windows much longer than a record.
        padded = np.pad(values, ((window // 2, window // 2), (0, 0)), mode="symmetric")
        expected = np.median(np.lib.stride_tricks.sliding_window_view(padded, window, axis=0), axis=-1)
    else:
        expected = median_filter(values, size=(window, 1), mode="reflect")
    actual = centered_median(values, window)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_other_dtypes_preserve_reference_semantics(dtype):
    values = np.arange(24, dtype=dtype).reshape(12, 2)
    actual = centered_median(values, 7)
    np.testing.assert_array_equal(actual, median_filter(values, size=(7, 1)))
    assert actual.dtype == dtype


def test_nonfinite_and_explicit_fallback():
    values = np.arange(36, dtype=float).reshape(18, 2)
    for replacement in (np.nan, np.inf):
        values[3, 1] = replacement
        np.testing.assert_array_equal(centered_median(values, 7), median_filter(values, size=(7, 1)))
    finite = np.arange(36, dtype=float).reshape(18, 2)
    np.testing.assert_array_equal(centered_median(finite, 7, accelerated=False), median_filter(finite, size=(7, 1)))


@pytest.mark.parametrize("window", [0, 2, -1, 2.5])
def test_invalid_windows_refused(window):
    with pytest.raises(ValueError):
        centered_median(np.ones((10, 2)), window)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_short_record_fallback_has_exact_reflected_order_statistic(dtype):
    values = np.array([[3, 7], [11, 2]], dtype=dtype)
    if dtype == np.int64:
        values += 2**60
    padded = np.pad(values, ((50, 50), (0, 0)), mode="symmetric")
    expected = np.stack([np.sort(padded[i:i + 101], axis=0)[50] for i in range(2)])
    actual = centered_median(values, 101, accelerated=False)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype
