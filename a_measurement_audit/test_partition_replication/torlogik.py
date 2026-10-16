"""Pure gate functions of the pre-registered replication specification; no file access or fits."""
import math

def finite(*values):
    return all(v is not None and math.isfinite(v) for v in values)

def interval_inside(low, high, lower, upper, valid_resamples):
    return (valid_resamples >= 950 and finite(low, high, lower, upper)
            and lower <= low <= high <= upper)

def w_gate(coverage, median_ci, benchmark, rho, rho_low, rc_fraction,
           bias_fraction, median_valid, rho_valid):
    return (finite(coverage, rho, rho_low, rc_fraction, bias_fraction)
            and coverage >= .9
            and interval_inside(*median_ci, benchmark-.05, benchmark+.05, median_valid)
            and rho_valid >= 950 and rho >= .8 and rho_low >= .75
            and rc_fraction >= .8 and bias_fraction >= .8)

def m_gate(coverage, agreement, agreement_low, kappa, kappa_low,
           drift_ci, agreement_valid, kappa_valid, drift_valid):
    return (finite(coverage, agreement, agreement_low, kappa, kappa_low)
            and coverage >= .9 and agreement >= .85 and agreement_low >= .8
            and kappa >= .6 and kappa_low >= .5
            and agreement_valid >= 950 and kappa_valid >= 950
            and interval_inside(*drift_ci, -.05, .05, drift_valid))

def p_gate(n, drift_ci, valid):
    return n >= 1000 and interval_inside(*drift_ci, -.05, .05, valid)

def duration_gate(n, median_ci, p975_ci, median_valid, p975_valid):
    return (n >= 500 and interval_inside(*median_ci, 94., 114., median_valid)
            and interval_inside(*p975_ci, 160.9, 200.9, p975_valid))

def agreement_kappa(development, test):
    """Missing test classes remain in the original marker denominator."""
    if len(development) != len(test):
        raise ValueError('The same fixed ID list is required')
    n = len(development)
    if not n:
        return float('nan'), float('nan'), {}
    observed = sum(a == b for a, b in zip(development, test)) / n
    cats = set(development) | set(test)
    expected = sum(development.count(c)*test.count(c) for c in cats)/n**2
    table = {(a, b): sum(x == a and y == b for x, y in zip(development, test))
             for a in sorted(cats) for b in sorted(cats)}
    return observed, ((observed-expected)/(1-expected) if expected < 1 else float('nan')), table
