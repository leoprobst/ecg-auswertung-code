"""Frozen organ-surrogate pilot, distinct from patient physiology calibration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from .edgar import digest as file_hash
from .learning import atomic_json, digest
from .model import Experiment, Tissue, Electrode
from .resources import capacity_target, PagingMonitor
from .twin_emulator import train_compare, select_active

LEADS = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')


def prepare(root, initial=512, maximum=4096, active_seconds=86400, seed=20260924):
    from .model import anatomical_tissue, coarsen_short_edges
    if initial < 128 or initial % 16 or not initial <= maximum <= 4096 or not 60 <= active_seconds <= 86400:
        raise ValueError('Invalid frozen campaign budget')
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    tissue = coarsen_short_edges(anatomical_tissue())
    np.savez_compressed(root/'tissue.npz', xyz=tissue.xyz, edges=tissue.edges, cells=tissue.cells,
                        regions=tissue.regions, fibers=tissue.fibers,
                        maps=np.asarray(json.dumps({'sites': {k: v.tolist() for k, v in tissue.sites.items()},
                                                    'branches': {k: v.tolist() for k, v in tissue.branches.items()}})))
    runtime = root/'runtime/vcg_biomarker/simulation'
    runtime.mkdir(parents=True)
    package = Path(__file__).parent
    shutil.copyfile(package.parent/'__init__.py', runtime.parent/'__init__.py')
    # Only backend dependencies; workers load captured geometry, not GUI anatomy.
    for name in ('__init__.py', 'twin_campaign.py', 'twin_emulator.py', 'model.py', 'conduction.py',
                 'solver.py', 'resources.py', 'learning.py', 'edgar.py', 'worker.py'):
        shutil.copyfile(package/name, runtime/name)
    base = qmc.LatinHypercube(3, seed=seed).random(maximum)
    family_order = np.random.default_rng(seed).permutation(16).tolist()
    roles = {g: 'fit' if i < 11 else 'calibration' if i < 14 else 'audit' for i, g in enumerate(family_order)}
    cases = {}
    for arm, presets in {'activation': ['reference', 'lbbb', 'rbbb', 'slow_zone'],
                         'repolarization': ['reference', 'short_ap', 'long_ap']}.items():
        cases[arm] = []
        for i, row in enumerate(base):
            group = i % 16
            p = presets[(i//16) % len(presets)]
            vector = [.6+.8*row[0], .7+.6*row[1], -10+20*row[2], .85+.02*group]
            cases[arm].append({'id': i, 'group': group, 'partition': roles[group], 'preset': p,
                               'parameters': vector, 'x': vector+[float(p == k) for k in presets]})
        (root/arm/'cases').mkdir(parents=True)
    manifest = {'schema': 'twin-organ-campaign/1', 'seed': seed, 'initial': initial, 'maximum': maximum,
        'batch': 128, 'max_active_seconds_per_arm': active_seconds, 'max_detailed_per_arm': 256,
        'detailed_status': 'blocked_no_converged_patient_mesh_or_matched_multifidelity_observation_operator',
        'geometry_scope': 'one generic topology, sixteen isotropic scale families; not sixteen independent anatomies',
        'outputs': ['activation_p95_minus_p05_ms', 'activated_fraction', 'late_signal_energy_fraction']+
                   ['rms_'+name+'_arbitrary_unit' for name in LEADS],
        'parameters': ['diffusion_scale', 'recovery_scale', 'chest_electrode_x_offset_mm', 'geometry_scale'],
        'initial_state': 'resting transient, not paced steady state', 'duration_ms': 600,
        'clinical_use': False, 'cases': cases,
        'max_workers': max(1, min(32, (os.cpu_count() or 2)-1)),
        'tissue_sha256': file_hash(root/'tissue.npz'),
        'runtime_hashes': {p.relative_to(root/'runtime').as_posix(): file_hash(p)
                           for p in (root/'runtime').rglob('*.py')}}
    from importlib.metadata import version
    manifest['dependencies'] = {name: version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'psutil', 'threadpoolctl')}
    manifest['campaign_sha256'] = digest(manifest)
    atomic_json(root/'manifest.json', manifest)
    return root


def _load_tissue(root, scale):
    with np.load(Path(root)/'tissue.npz', allow_pickle=False) as data:
        maps = json.loads(str(data['maps']))
        return Tissue(data['xyz']*scale, data['edges'], data['cells'], data['regions'], data['fibers'],
                      {k: np.array(v) for k, v in maps['sites'].items()},
                      {k: np.array(v) for k, v in maps['branches'].items()}, 'generic_scaled_graph')


def task(root, arm, case, campaign_hash):
    from .solver import simulate
    root = Path(root)
    start = time.monotonic()
    diffusion, recovery, offset, scale = case['parameters']
    tissue = _load_tissue(root, scale)
    spec = Experiment(diffusion_scale=diffusion, recovery_scale=recovery, preset=case['preset'])
    spec = replace(spec, electrodes=tuple(Electrode(e.name,
        (e.xyz_mm[0]+(offset if e.name.startswith('V') else 0), e.xyz_mm[1], e.xyz_mm[2]), e.kind)
        for e in spec.electrodes))
    def cancel():
        return ((root/'cancel').exists() or (root/'pause').exists()
                or (root/arm/f"yield-{case['id']}").exists())
    try:
        with threadpool_limits(limits=1):
            result = simulate(tissue, spec, cancelled=cancel)
        activation = result.activation_ms[np.isfinite(result.activation_ms)]
        if len(activation) < 4:
            raise ArithmeticError('Insufficient captured tissue')
        waveform = result.ecg[:, [result.lead_names.index(k) for k in LEADS]]
        energy = float(np.sum(waveform**2))
        if energy <= 0:
            raise ArithmeticError('No signal energy')
        y = [float(np.quantile(activation, .95)-np.quantile(activation, .05)),
             result.metadata['activated_fraction'], float(np.sum(waveform[result.times_ms >= 250]**2)/energy)]
        y.extend(np.sqrt(np.mean(waveform**2, axis=0)).tolist())
        row = {**case, 'status': 'computed', 'outputs': y, 'solver_metadata': result.metadata}
    except InterruptedError:
        return {**case, 'status': 'interrupted'}
    except (ArithmeticError, ValueError) as exc:
        row = {**case, 'status': 'non_estimable', 'reason': str(exc)}
    row.update(campaign_sha256=campaign_hash, elapsed_seconds=time.monotonic()-start)
    atomic_json(root/arm/'cases'/f"{case['id']:05d}.json", row)
    return row


def run(root):
    from .worker import acquire_worker_lock
    from importlib.metadata import version
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    unsigned = dict(manifest)
    bound = unsigned.pop('campaign_sha256')
    if digest(unsigned) != bound or file_hash(root/'tissue.npz') != manifest['tissue_sha256']:
        raise ValueError('Changed campaign inputs')
    for relative, expected in manifest['runtime_hashes'].items():
        if file_hash(root/'runtime'/relative) != expected:
            raise ValueError('Frozen runtime changed')
    expected_runtime = root/'runtime/vcg_biomarker/simulation/twin_campaign.py'
    if Path(__file__).resolve() != expected_runtime.resolve():
        raise ValueError('Launch from frozen runtime cwd/PYTHONPATH')
    if any(version(name) != expected for name, expected in manifest['dependencies'].items()):
        raise ValueError('Frozen dependency versions changed')
    lease = acquire_worker_lock(root/'campaign.lock')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    monitor = PagingMonitor()
    process = psutil.Process()
    known, limit, stable, peak = {}, 1, 0, .8
    psutil.cpu_percent(None)
    states = {}
    for arm, cases in manifest['cases'].items():
        completed = {}
        for path in (root/arm/'cases').glob('*.json'):
            row = json.loads(path.read_text())
            if row.get('campaign_sha256') != bound or row['parameters'] != cases[row['id']]['parameters']:
                raise ValueError('Incompatible checkpoint')
            completed[row['id']] = row
        queue_file = root/arm/'selection.json'
        selection = json.loads(queue_file.read_text()) if queue_file.exists() else {'selected': list(range(manifest['initial'])), 'elapsed': 0.}
        states[arm] = {'completed': completed, **selection, 'last_report_count': -1, 'finished': False}
    last, start = 0., time.monotonic()
    last_admitted_arm = None
    try:
        with ProcessPoolExecutor(max_workers=manifest['max_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            active = {}
            while not all(s['finished'] for s in states.values()) or active:
                now = time.monotonic()
                stop = (root/'cancel').exists() or (root/'pause').exists()
                if now-last >= 2:
                    own, resident = 0., []
                    for child in process.children(recursive=True):
                        try:
                            if child.pid not in known:
                                known[child.pid] = child
                                child.cpu_percent(None)
                            else:
                                cores = known[child.pid].cpu_percent(None)/100
                                own += cores
                                if cores > .01:
                                    resident.append(child.memory_info().rss/2**30)
                            m = child.memory_info()
                            peak = max(peak, getattr(m, 'peak_wset', m.rss)/2**30*1.5)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    free, cpu, paging = psutil.virtual_memory().available/2**30, psutil.cpu_percent(None), monitor.sample()
                    target = capacity_target(os.cpu_count() or 2, cpu, own, free, peak,
                        manifest['max_workers'], paging['paging_pressure'], len(active), sum(sorted(resident)[:len(active)]))
                    if target < limit:
                        limit, stable = target, 0
                    elif target > limit:
                        stable += 1
                        if stable >= 3:
                            limit, stable = min(target, limit+1), 0
                    else:
                        stable = 0
                    # One worker at a time yields at a solver-frame boundary.
                    if len(active) > limit and free < 2.+peak:
                        if not any((root/a/f'yield-{i}').exists() for a, i in active.values()):
                            a, i = list(active.values())[-1]
                            (root/a/f'yield-{i}').touch()
                    for arm, state in states.items():
                        if last and not state['finished'] and not stop:
                            state['elapsed'] += now-last
                        atomic_json(root/arm/'selection.json', {'selected': state['selected'], 'elapsed': state['elapsed']})
                    status = {'phase': 'stopping' if stop else 'simulation', 'pid': os.getpid(),
                              'active': len(active), 'admitted': limit, 'target': target, 'free_gib': free,
                              'cpu_percent': cpu, 'worker_peak_budget_gib': peak, **paging,
                              'arms': {a: {'done': len(s['completed']), 'selected': len(s['selected']),
                                          'elapsed_seconds': s['elapsed']} for a, s in states.items()}}
                    atomic_json(root/'status.json', status)
                    with (root/'resources.jsonl').open('a') as stream:
                        stream.write(json.dumps(status)+'\n')
                    last = now
                arm_order = sorted(states, key=lambda arm: arm == last_admitted_arm)
                for arm in arm_order:
                    state = states[arm]
                    inflight = {i for a, i in active.values() if a == arm}
                    pending = [i for i in state['selected'] if i not in state['completed'] and i not in inflight]
                    expired = state['elapsed'] >= manifest['max_active_seconds_per_arm']
                    if expired:
                        state['finished'] = True
                        continue
                    if not pending and not inflight and not state['finished'] and not stop:
                        rows = [r for _, r in sorted(state['completed'].items()) if r['status'] == 'computed']
                        proposals = [c for c in manifest['cases'][arm] if c['id'] not in state['selected'] and c['partition'] == 'fit']
                        if rows:
                            report, score = train_compare([r['x'] for r in rows], [r['outputs'] for r in rows],
                                [r['partition'] for r in rows], [r['group'] for r in rows],
                                [c['x'] for c in proposals] if proposals else None)
                            report['median_direct_seconds'] = float(np.median([r['elapsed_seconds'] for r in rows]))
                            report['detailed_stage'] = manifest['detailed_status']
                            report['computed'] = len(rows)
                            report['failed'] = len(state['completed'])-len(rows)
                            atomic_json(root/arm/f"emulator_{len(state['completed']):05d}.json", report)
                        else:
                            score = None
                        if proposals and score is not None and len(state['selected']) < manifest['maximum']:
                            state['selected'].extend(select_active(score, [c['id'] for c in proposals],
                                min(manifest['batch'], manifest['maximum']-len(state['selected']))))
                            pending = [i for i in state['selected'] if i not in state['completed']]
                        else:
                            state['finished'] = True
                    if pending and not stop and len(active) < limit:
                        i = pending[0]
                        active[pool.submit(task, str(root), arm, manifest['cases'][arm][i], bound)] = (arm, i)
                        last_admitted_arm = arm
                done, _ = wait(active, timeout=.5, return_when=FIRST_COMPLETED) if active else (set(), set())
                for future in done:
                    arm, i = active.pop(future)
                    row = future.result()
                    flag = root/arm/f'yield-{i}'
                    if flag.exists():
                        flag.unlink()
                    if row['status'] != 'interrupted':
                        states[arm]['completed'][i] = row
                if stop and not active:
                    break
                if not active:
                    time.sleep(.2)
        atomic_json(root/'status.json', {'phase': 'paused' if (root/'pause').exists() else 'cancelled' if (root/'cancel').exists()
                    else 'budget_exhausted' if any(s['elapsed'] >= manifest['max_active_seconds_per_arm'] for s in states.values()) else 'complete',
                    'elapsed_seconds': time.monotonic()-start, 'pid': os.getpid(),
                    'arms': {a: {'done': len(s['completed']), 'selected': len(s['selected'])} for a, s in states.items()},
                    'clinical_use': False, 'detailed_stage': manifest['detailed_status']})
    finally:
        monitor.close()
        lease.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('root')
    parser.add_argument('--initial', type=int, default=512)
    parser.add_argument('--maximum', type=int, default=4096)
    parser.add_argument('--seconds', type=int, default=86400)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(prepare(args.root, args.initial, args.maximum, args.seconds))
    else:
        run(args.root)
