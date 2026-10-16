"""Physical-time parameters shared by temporal VCG markers and their views."""

import math


SPEED_SMOOTHING_SIGMA_MS = 2.0  # preserves the previous 1 sample at 500 Hz


def speed_smoothing_sigma_samples(fs: float) -> float:
    """Gaussian standard deviation in samples, not a full window width."""
    rate = float(fs)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sampling rate must be finite and positive")
    return SPEED_SMOOTHING_SIGMA_MS * rate / 1000.0
