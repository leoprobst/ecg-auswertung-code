"""Fit-only emulator comparison; calibrated simulator errors are not patient errors."""
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


def _model(kind):
    if kind == 'quadratic':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=ConstantKernel(1., constant_value_bounds='fixed')*Matern(length_scale=1., nu=2.5,
                  length_scale_bounds='fixed'), alpha=1e-6, normalize_y=True, optimizer=None))


def train_compare(x, y, roles, groups, proposal_x=None, seed=20260924):
    x, y = np.asarray(x, float), np.asarray(y, float)
    roles, groups = np.asarray(roles), np.asarray(groups)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(roles) != len(x)
            or len(groups) != len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()
            or set(roles)-{'fit', 'calibration', 'audit'}):
        raise ValueError('Finite, role-bound simulator bank required')
    for group in np.unique(groups):
        if len(set(roles[groups == group])) != 1:
            raise ValueError('Candidate/geometry family crosses roles')
    parts = {r: np.flatnonzero(roles == r) for r in ('fit', 'calibration', 'audit')}
    if any(len(v) < n for v, n in zip(parts.values(), (60, 30, 30))):
        return {'status': 'non_estimable', 'reason': 'insufficient_role_counts',
                'counts': {k: len(v) for k, v in parts.items()}}, None
    tr, cal, audit = parts.values()
    # GP cubic cost is bounded. Deterministic fit-only thinning, not audit selection.
    if len(tr) > 512:
        tr = np.sort(np.random.default_rng(seed).choice(tr, 512, replace=False))
    if len(np.unique(groups[tr])) < 3:
        return {'status': 'non_estimable', 'reason': 'insufficient_fit_groups'}, None
    scale = np.maximum(np.std(y[tr], axis=0), 1e-12)
    scores, models = {}, {}
    with threadpool_limits(limits=1):
        for kind in ('quadratic', 'gaussian_process'):
            losses = []
            for a, b in GroupKFold(3).split(x[tr], groups=groups[tr]):
                fit = _model(kind).fit(x[tr[a]], y[tr[a]])
                fold_scale = np.maximum(np.std(y[tr[a]], axis=0), 1e-12)
                losses.append(float(np.mean(((fit.predict(x[tr[b]])-y[tr[b]])/fold_scale)**2)))
            scores[kind] = float(np.mean(losses))
            models[kind] = _model(kind).fit(x[tr], y[tr])
        winner = min(scores, key=scores.get)  # No calibration/audit selection.
        reports = {}
        for kind, fit in models.items():
            errors = np.abs(fit.predict(x[cal])-y[cal])/scale
            maximum = errors.max(axis=1)
            rank = int(np.ceil((len(cal)+1)*.9))
            radius = float(np.sort(maximum)[rank-1])*scale
            began = time.perf_counter()
            predicted = fit.predict(x[audit])
            elapsed = time.perf_counter()-began
            residuals = np.abs(predicted-y[audit])
            reports[kind] = {'audit_mae': residuals.mean(0).tolist(),
                'audit_joint_coverage_90': float(np.mean(np.all(residuals <= radius, axis=1))),
                'simultaneous_error_radius': radius.tolist(),
                'audit_prediction_seconds': elapsed, 'audit_count': len(audit)}
        uncertainty = None
        if proposal_x is not None:
            proposal_x = np.asarray(proposal_x, float)
            if (proposal_x.ndim != 2 or proposal_x.shape[1] != x.shape[1] or len(proposal_x) > 8192
                    or not np.isfinite(proposal_x).all()):
                raise ValueError('Bounded finite proposal pool required')
            gp = models['gaussian_process']
            _, sd = gp[-1].predict(gp[0].transform(proposal_x), return_std=True)
            if sd.ndim == 1:
                sd = sd[:, None]
            uncertainty = np.mean(sd/scale, axis=1)
    return {'schema': 'twin-emulator-comparison/1', 'status': 'research_simulator_emulator',
        'selected_by_fit_cv': winner, 'fit_cv_standardized_mse': scores,
        'fit_count_used': len(tr), 'models': reports,
        'patient_calibration': False, 'clinical_use': False,
        'coverage_scope': 'finite simulator design, not external patient coverage',
        'coverage_guarantee': 'none: related protocols and only a few calibration geometry families',
        'training_seed': seed}, uncertainty


def select_active(uncertainty, candidate_ids, count=128):
    scores = np.asarray(uncertainty, float)
    if scores.shape != (len(candidate_ids),) or not np.isfinite(scores).all() or count < 1:
        raise ValueError('Finite fit-only proposal uncertainties required')
    # Stable tie-breaking independent of worker completion order.
    return [candidate_ids[i] for i in sorted(range(len(scores)), key=lambda i: (-scores[i], candidate_ids[i]))[:count]]
