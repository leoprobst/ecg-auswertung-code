"""Prespecified native-coordinate equivalent-dipole comparator for EDGAR.

Homogeneous infinite conductor, NOT the finite FEM torso, not focal-origin proof.
No ground truth is given to the search; held-out electrode errors are reported
but never used for candidate selection. No assumed mV, ms or mm is needed here.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits


def fit_dipole(potentials, electrodes, candidates, sample_window=None, cancelled=None):
    """Channel x sample input; common-reference invariant variable dipole moment.

    Compress time once to channel covariance: identical LS ranking, independent
    of sample count in the candidate loop. Native scale is preserved in errors.
    """
    import psutil
    potentials, electrodes, candidates = [np.asarray(x, float) for x in (potentials, electrodes, candidates)]
    if (potentials.ndim != 2 or not 12 <= len(potentials) <= 512 or potentials.size > 8_000_000
            or electrodes.shape != (len(potentials), 3) or candidates.ndim != 2
            or candidates.shape[1] != 3 or not 2 <= len(candidates) <= 10000
            or not all(np.isfinite(x).all() for x in (potentials, electrodes, candidates))):
        raise ValueError('Finite bounded potential/geometry arrays required')
    start, end = (0, potentials.shape[1]) if sample_window is None else sample_window
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= potentials.shape[1]
            or end-start < 4):
        raise ValueError('Prespecified sample window required')
    # Constant channels can be missing electrodes or a reference channel. Do not
    # silently treat padded zero traces as genuine normal measurements.
    retained = np.ptp(potentials[:, start:end], axis=1) > 0
    excluded_indices = np.flatnonzero(~retained).tolist()
    indices = np.flatnonzero(retained)
    if len(indices) < 12:
        raise ValueError('Insufficient nonconstant measurement channels')
    potentials, electrodes = potentials[retained], electrodes[retained]
    train = indices % 5 != 0
    test = ~train
    if train.sum() < 8 or test.sum() < 2:
        raise ValueError('Insufficient prespecified train/held-out electrodes')
    observed = potentials[:, start:end]
    # Same reference for train/test, no held-out-channel influence on fit.
    reference = observed[train].mean(axis=0)
    y = observed[train]-reference
    with threadpool_limits(limits=1):
        covariance = y@y.T
    energy = float(np.trace(covariance))
    if energy <= 1e-20:
        raise ValueError('No measurable spatial signal')
    center = electrodes.mean(axis=0)
    scale = float(np.linalg.norm(np.ptp(electrodes, axis=0)))
    if scale <= 1e-12:
        raise ValueError('Degenerate electrode geometry')
    e, c = (electrodes-center)/scale, (candidates-center)/scale
    residuals = np.full(len(c), np.inf)

    def field(point):
        delta = e-point
        distances = np.linalg.norm(delta, axis=1)
        if distances.min() <= 1e-8:
            return None
        return delta/distances[:, None]**3

    with threadpool_limits(limits=1):
        for i, point in enumerate(c):
            if i % 64 == 0:
                if cancelled and cancelled():
                    raise InterruptedError('Cancelled dipole comparison')
                if psutil.virtual_memory().available < 2.5*1024**3:
                    raise MemoryError('Dipole comparison preserves RAM reserve')
            a = field(point)
            if a is None:
                continue
            a = a[train]-a[train].mean(axis=0)
            u, s, _ = np.linalg.svd(a, full_matrices=False)
            if s[-1] <= s[0]*1e-8:
                continue
            explained = float(np.sum(u*(covariance@u)))
            # Only round-off correction on an analytically nonnegative SSE.
            residuals[i] = max(0., energy-explained)/energy
        order = np.argsort(residuals, kind='stable')
        best = int(order[0])
        if not np.isfinite(residuals[best]):
            raise ValueError('No identifiable dipole design')
        a = field(c[best])
        a = a-a[train].mean(axis=0)
        moments = np.linalg.lstsq(a[train], y, rcond=1e-8)[0]
        predicted = a@moments
    errors = predicted[test]-(observed[test]-reference)
    baseline_energy = float(np.sum((observed[test]-reference)**2))
    return {'schema': 'edgar-native-dipole/1', 'candidate_index': best,
            'xyz_native': candidates[best].tolist(), 'sample_window': [start, end],
            'window_interpretation': 'whole supplied record' if sample_window is None else 'explicit sample indices',
            'training_normalized_sse': float(residuals[best]),
            'heldout_electrode_normalized_sse': float(np.sum(errors**2)/baseline_energy) if baseline_energy > 1e-20 else None,
            'heldout_electrode_rmse_native': float(np.sqrt(np.mean(errors**2))),
            'top_candidates': [{'index': int(i), 'xyz_native': candidates[i].tolist(),
                'normalized_sse': float(residuals[i])} for i in order[:5] if np.isfinite(residuals[i])],
            'candidate_count': len(candidates), 'training_channels': int(train.sum()),
            'heldout_channels': int(test.sum()), 'reference_used_in_search': False,
            'excluded_constant_channel_indices': excluded_indices,
            'constant_channel_policy': 'exclude and report; not classified as normal or proven bad electrode',
            'clinical_use': False, 'anatomical_focal_origin_claim': False,
            'model': 'equivalent moving-moment dipole at fixed position; homogeneous infinite conductor',
            'limitations': ['No finite torso/blood/lung conductivity boundaries.',
                'Whole-record equivalent dipole position need not equal earliest activation or ablation site.',
                'Native coordinates/errors only; no unverified mm/mV/ms claim.',
                'Electrode holdout is not patient-level validation or an uncertainty interval.']}


def reference_distance(result, reference_xyz, reference_type):
    if reference_type not in {'pacing_site', 'ablation_site'}:
        raise ValueError('Independent reference type required')
    xyz = np.atleast_2d(np.asarray(reference_xyz, float))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 1 <= len(xyz) <= 100 or not np.isfinite(xyz).all():
        raise ValueError('Registered finite reference coordinates required')
    # Multiple lesions are a reference set, not several independent origins.
    unique = np.unique(xyz, axis=0)
    distances = np.linalg.norm(np.asarray(result['xyz_native'])-unique, axis=1)
    return {'distance_native': float(distances[0]) if len(distances) == 1 else None,
            'reference_points': len(xyz), 'unique_reference_points': len(unique),
            'distances_to_reference_points_native': distances.tolist(),
            'min_distance_to_reference_set_native': float(distances.min()),
            'max_distance_to_reference_set_native': float(distances.max()),
            'reference_point_selected': False,
            'distance_unit': 'dataset_native_unverified', 'reference_type': reference_type,
            'localization_error_mm': None, 'clinical_use': False}
