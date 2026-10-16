"""One declaration of the limits a marker value may not leave.

Amplitude ratios are formed in three different modules -- the classical
criteria, the third discovery generation and the loop catalog -- and each grew
its own guard against dividing by zero. All three guarded the *denominator*
(``max(abs(d), 1e-9)``, or abstaining below ``1e-12``) and none guarded the
*quotient*, which is the thing that is used. Measured on the current store:

===========================================  ================
marker                                       largest value
===========================================  ================
``classical_dominant_r_v1``                  1.260.002.535,94
``rvh_qrs_posterior_left_force_ratio``               39.469,8
``rvh_qrs_posterior_anterior_force_ratio``           20.468,9
``rvh_qrs_right_left_force_ratio``                   13.782,4
``rvh_p_anterior_posterior_force_ratio``              5.618,4
===========================================  ================

A guard on the denominator only moves the explosion: an S wave of 2e-9 mV is
not a measurement either, and the quotient it produces is nine orders of
magnitude outside anything physiological. Rank statistics survive it, which is
why it went unnoticed for so long -- ``classical_dominant_r_v1`` is the
second-best marker for right bundle branch block at AUC 0,889 with that value
in the column. Everything not rank based does not survive it.

The limit is deliberately far outside the clinical range rather than close to
it. An R/S of 10 is already a dominant R wave; at 50 the cap can only bind on
values that were never measurements, so it cannot quietly truncate a real
distribution.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


MARKER_BOUNDS_VERSION = "1.0.0"

# Largest magnitude an amplitude ratio may carry. See the module docstring for
# why it is 50 and not something tighter.
RATIO_LIMIT = 50.0


def bounded_ratio(
    numerator: float,
    denominator: float,
    *,
    limit: float = RATIO_LIMIT,
    minimum_denominator: float = 0.0,
) -> Optional[float]:
    """A ratio that cannot leave its declared magnitude.

    Returns ``None`` when either side is not finite, or when the denominator is
    below ``minimum_denominator`` -- an explicit abstention, which a consumer
    can tell apart from a real extreme. Otherwise the quotient is clipped to
    ``[-limit, +limit]``: saturating rather than abstaining keeps the ordering
    of the extreme cases, which is what the rank statistics read, while
    removing the magnitude that the linear models cannot survive.
    """

    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None
    if abs(denominator) <= max(0.0, float(minimum_denominator)):
        return None
    value = float(numerator) / float(denominator)
    if not np.isfinite(value):
        return None
    return float(np.clip(value, -limit, limit))


__all__ = ["MARKER_BOUNDS_VERSION", "RATIO_LIMIT", "bounded_ratio"]
