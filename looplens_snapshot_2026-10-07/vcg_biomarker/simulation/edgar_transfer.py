"""Empirical native-sensor transfer benchmark, not anatomical ECGI or a twin.

Nested leave-stimulation-site-out validation within each preparation. Covariance
sufficient statistics avoid retaining all time series or repeating fits per beat.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from threadpoolctl import threadpool_limits

from .edgar import digest, signal_native

ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1., 10.)


def common_reference(values):
    """Exclude constant channels BEFORE common referencing; retain channel slots."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite channel-by-time signal required')
    valid = np.ptp(values, axis=1) > 0
    if valid.sum() < 3:
        raise ValueError('Insufficient nonconstant channels')
    out = np.zeros_like(values)
    out[valid] = values[valid] - values[valid].mean(axis=0)
    return out, valid


def moments(x, y):
    if x.shape[1] != y.shape[1] or x.shape[1] < 4:
        raise ValueError('Paired sample coordinates required')
    n = x.shape[1]
    return (x @ x.T/n, y @ y.T/n, y @ x.T/n,
            np.roll(y, n//3, axis=1) @ x.T/n)


def average(items):
    if not items:
        raise ValueError('Empty training set')
    return tuple(sum(item[k] for item in items)/len(items) for k in range(4))


def reverse(stats):
    xx, yy, yx, shifted = stats
    return yy, xx, yx.T, shifted.T


def ridge_candidates(stats, alphas=ALPHAS):
    xx, _, yx, _ = stats
    eig, vectors = np.linalg.eigh(xx)
    if eig[-1] <= 0 or np.min(eig) < -1e-9*eig[-1]:
        raise ValueError('Invalid or zero covariance')
    eig = np.maximum(eig, 0)
    scale = float(np.trace(xx)/len(xx))
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError('Positive frozen regularization required')
    projected = yx @ vectors
    return [(projected / (eig + alpha*scale)) @ vectors.T for alpha in alphas]


def loss(model, stats, shifted=False):
    xx, yy, yx, control = stats
    energy = float(np.trace(yy))
    if energy <= 0:
        raise ValueError('No target energy')
    error = float(np.sum((model @ xx)*model) -
                  2*np.sum(model*(control if shifted else yx)) + energy)
    if error < -1e-8*energy:
        raise ValueError('Invalid negative squared error')
    return max(0., error)/energy


def nested_site_benchmark(groups, cancelled=lambda: False):
    """Equal site weighting; outer observations never determine alpha or fit."""
    keys = sorted(groups)
    if len(keys) < 4:
        return {'status': 'non_estimable', 'reason': 'At least four distinct site groups required'}
    rows = []
    for held in keys:
        if cancelled():
            raise InterruptedError('Benchmark cancelled')
        train = [key for key in keys if key != held]
        inner = []
        for validation in train:
            fit = average([groups[k] for k in train if k != validation])
            inner.append([loss(model, groups[validation]) for model in ridge_candidates(fit)])
        selected = int(np.argmin(np.mean(inner, axis=0)))
        fit = average([groups[k] for k in train])
        model = ridge_candidates(fit, (ALPHAS[selected],))[0]
        eigenvalues = np.linalg.eigvalsh(fit[0])
        rows.append({'held_site': held, 'alpha': ALPHAS[selected],
                     'train_site_count': len(train),
                     'nmse': loss(model, groups[held]),
                     'time_shift_control_nmse': loss(model, groups[held], shifted=True),
                     'training_input_effective_rank': float(
                         eigenvalues.sum()**2 / np.sum(eigenvalues**2)),
                     'zero_prediction_nmse': 1.})
    return {'status': 'computed', 'sites': rows,
            'mean_site_nmse': float(np.mean([r['nmse'] for r in rows])),
            'median_site_nmse': float(np.median([r['nmse'] for r in rows])),
            'mean_shift_control_nmse': float(np.mean([r['time_shift_control_nmse'] for r in rows]))}


def site_groups(pairs, inventory):
    """Connect sites AND parent records so neither can cross the outer fold."""
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, pair in enumerate(pairs):
        row = inventory[pair['epicardial_path']]
        reference = np.asarray(row['reference_native'], float)
        if reference.size == 0 or not np.isfinite(reference).all():
            raise ValueError('Missing site reference')
        # Exact native reference sets, not spatial clustering tuned on performance.
        points = tuple(sorted(tuple(p) for p in np.atleast_2d(reference)))
        for key in (('site', points), ('parent', row['parent_record'])):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, pair in enumerate(pairs):
        groups.setdefault(find(i), []).append(pair)
    return {hashlib.sha256(json.dumps(sorted(p['episode_id'] for p in records)).encode()).hexdigest()[:16]:
            records for records in groups.values()}


def preparation_task(root, pairs, inventory, output):
    """Small isolated worker; hash-verified reads and no raw-data writes."""
    import psutil
    from .learning import atomic_json
    root, output = Path(root), Path(output)
    subject = pairs[0]['subject_id']
    if any(p['subject_id'] != subject or p['status'] != 'native_pair' for p in pairs):
        raise ValueError('Never pool incompatible preparations or overlapping records')
    cancel = lambda: (output.parent/'cancel').exists()
    groups, summary = {}, []
    with threadpool_limits(limits=1):
        for key, records in sorted(site_groups(pairs, inventory).items()):
            # Equal parent weight, then equal beat weight within parent.
            parent_stats = {}
            for pair in records:
                if cancel():
                    raise InterruptedError('Cancelled at record boundary')
                if psutil.virtual_memory().available < 2.25*2**30:
                    raise MemoryError('Yield for RAM reserve; completed preparation checkpoints retained')
                for side in ('torso', 'epicardial'):
                    if digest(root/pair[side+'_path']) != pair[side+'_sha256']:
                        raise ValueError('Changed EDGAR source')
                x, vx = common_reference(signal_native(root, pair['torso_path'])[0])
                y, vy = common_reference(signal_native(root, pair['epicardial_path'])[0])
                parent_key = inventory[pair['epicardial_path']]['parent_record']
                stats = moments(x, y)
                if parent_key not in parent_stats:
                    parent_stats[parent_key] = [list(stats), 1]
                else:
                    accumulated, count = parent_stats[parent_key]
                    for k in range(4):
                        accumulated[k] += stats[k]
                    parent_stats[parent_key][1] = count+1
                summary.append({'episode': pair['episode_id'], 'site_group': key,
                                'active_torso_channels': int(vx.sum()),
                                'active_epicardial_channels': int(vy.sum()),
                                'samples': x.shape[1]})
            groups[key] = average([tuple(a/n for a in sums) for sums, n in parent_stats.values()])
        result = {'schema': 'edgar-empirical-transfer/1', 'subject_id': subject,
                  'clinical_use': False, 'patient_generalization_tested': False,
                  'physical_units_verified': False, 'records': summary,
                  'independent_preparations': 1, 'site_groups': len(groups),
                  'inverse': nested_site_benchmark(groups, cancel),
                  'forward': nested_site_benchmark({k: reverse(v) for k, v in groups.items()}, cancel),
                  'method': 'native-sensor empirical ridge, not physical ECGI or anatomical origin',
                  'time_alignment': 'paired native sample indices; no resampling or best-lag selection',
                  'peak_rss_bytes': getattr(psutil.Process().memory_info(), 'peak_wset', psutil.Process().memory_info().rss)}
    atomic_json(output, result)
    return {'subject_id': subject, 'path': str(output), 'status': 'complete',
            'peak_rss_bytes': result['peak_rss_bytes']}


def safe_name(subject):
    return re.sub('[^a-zA-Z0-9_-]', '_', subject)
