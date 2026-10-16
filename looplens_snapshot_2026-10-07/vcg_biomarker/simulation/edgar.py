"""EDGAR MAT adapters: native data first, physical units only with attestation.

Challenge training data are NOT automatically an independent test cohort.
Files and MATLAB objects are never executed; originals are read-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np
from scipy.io import loadmat

SCHEMA = 'edgar-import/1'
COHORTS = {
    'Bratislava': ('human', 'ablation_site'),
    'Halifax': ('human', 'pacing_site'),
    'Auckland': ('porcine', 'pacing_site'),
    'Utah': ('canine', 'pacing_site_or_nonpaced'),
}
SOURCE = 'https://www.ecg-imaging.org/community/ecgi-data-challenge'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _safe_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != '.mat':
        raise ValueError('MAT path must remain inside EDGAR root')
    if any(re.search(r'(^|[_ -])(test|locked|holdout)([_ -]|$)', p, re.I)
           for p in path.relative_to(root).parts):
        raise ValueError('Test/locked input excluded from EDGAR import')
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError('MAT file exceeds per-file memory budget')
    return path


def load_native(root, relative):
    path = _safe_file(root, relative)
    # Selecting known data variables avoids deserialising MATLAB function workspaces.
    return loadmat(path, simplify_cells=True, variable_names=['ts', 'sock', 'torso',
        'EndoEpi', 'Epi', 'Heart7620', 'Torso7620', 'tank', 'geom', 'vest',
        'otherStructures', 'tank_closed'], verify_compressed_data_integrity=True)


def identity(relative):
    parts = Path(relative).parts
    cohort = parts[0]
    if cohort not in COHORTS:
        raise ValueError('Unknown EDGAR cohort')
    stem = Path(relative).stem
    if cohort == 'Bratislava':
        subject = re.match(r'P\d+', stem)
        if subject is None:
            raise ValueError('Missing Bratislava subject')
        subject = subject.group()
    elif cohort == 'Auckland':
        subject = re.match(r'Pig\d+', stem)
        if subject is None:
            raise ValueError('Missing Auckland subject')
        subject = subject.group()
    elif cohort == 'Utah':
        # Separate experimental preparations, not every beat as a new animal.
        subject = parts[1] if parts[1] in {'Signals1', 'Signals2'} else 'geometry_only'
    else:
        subject = '7620'
    episode = re.sub(r'-(ts|cs|ecg)$', '', stem)
    parent_record = re.sub(r'-b\d+$', '', episode)
    channel_type = ('epicardial' if stem.endswith('-cs') else
                    'named_ecg' if stem.endswith('-ecg') else 'body_surface')
    return {'cohort': cohort, 'subject_id': f'EDGAR:{cohort}:{subject}',
            'episode_id': episode, 'parent_record': parent_record,
            'species': COHORTS[cohort][0], 'reference_type': COHORTS[cohort][1],
            'signal_type': channel_type, 'data_role': 'unassigned',
            'clinical_use': False}


def _matrix(value, label):
    value = np.asarray(value)
    if value.dtype.kind not in 'fiu' or value.ndim != 2 or value.size > 16_000_000:
        raise ValueError(f'Invalid/excessive {label} matrix')
    return value


def signal_native(root, relative):
    data = load_native(root, relative).get('ts')
    if not isinstance(data, dict) or 'potvals' not in data:
        raise ValueError('Missing EDGAR ts.potvals')
    values = _matrix(data['potvals'], 'channel-by-sample potential')
    if not 1 <= values.shape[0] <= 1024 or not 4 <= values.shape[1] <= 200000:
        raise ValueError('Unsupported channel-by-sample layout')
    labels = data.get('leadlabels')
    if labels is not None:
        labels = [str(x) for x in np.atleast_1d(labels)]
        if len(labels) != len(values) or len(set(labels)) != len(labels):
            raise ValueError('Invalid lead labels')
    location = data.get('pacingLoc', data.get('pacingloc'))
    if location is not None:
        location = np.asarray(location, float)
        if not (location.shape == (3,) or (location.ndim == 2 and location.shape[1] == 3 and 1 <= len(location) <= 100)) or not np.isfinite(location).all():
            location = None  # absent/invalid reference, never fabricate a location
    return values, labels, location, data


@dataclass(frozen=True)
class Acquisition:
    """No defaults for unknown units, time origin or channel-to-node registration."""
    sampling_rate_hz: float
    voltage_unit: str
    geometry_unit: str
    coordinate_frame: str
    evidence: str
    reference_evidence: str
    registration_verified: bool
    channel_node_indices: tuple[int, ...] = ()  # explicit zero-based map

    def validate(self):
        if (not np.isfinite(self.sampling_rate_hz) or not 50 <= self.sampling_rate_hz <= 100000
                or self.voltage_unit not in {'mV', 'uV', 'V'}
                or self.geometry_unit not in {'mm', 'cm', 'm'}
                or not self.coordinate_frame.strip() or not self.evidence.strip()
                or not self.reference_evidence.strip()):
            raise ValueError('Verified acquisition conventions required; do not infer from amplitudes')
        if self.registration_verified is not True:
            raise ValueError('Coordinate registration not verified')


def physical_record(root, relative, acquisition: Acquisition):
    acquisition.validate()
    values, labels, location, _ = signal_native(root, relative)
    mv = values.T.astype(float) * {'mV': 1., 'uV': .001, 'V': 1000.}[acquisition.voltage_unit]
    if not np.isfinite(mv).all():
        raise ValueError('Missing/nonfinite channels require explicit exclusion; no imputation')
    return {'signal_mv': mv, 'times_ms': np.arange(len(mv)) * 1000/acquisition.sampling_rate_hz,
            'lead_names': labels, 'reference_xyz_mm': None if location is None else
            location * {'mm': 1., 'cm': 10., 'm': 1000.}[acquisition.geometry_unit],
            'provenance': {**identity(relative), 'path': str(relative),
                           'sha256': digest(_safe_file(root, relative)), 'acquisition': asdict(acquisition)}}


def _raw_surfaces(root, relative):
    """Preserve lung/vessel submeshes individually, never flatten their anatomy."""
    result = {}

    def visit(data, prefix=''):
        if not isinstance(data, dict):
            return
        if 'node' in data and 'face' in data:
            result[prefix] = data
        elif 'pts' in data and 'fac' in data:
            # Bratislava internal structures use a different explicit schema:
            # pts/fac are 3 x N, unlike the N x 3 node/face surfaces.
            points, faces = _matrix(data['pts'], 'pts'), _matrix(data['fac'], 'fac')
            if points.shape[0] != 3 or faces.shape[0] != 3:
                raise ValueError('Unverified pts/fac orientation')
            result[prefix] = {'node': points.T, 'face': faces.T}
        else:
            for key, value in data.items():
                if not key.startswith('__'):
                    visit(value, (prefix+'.' if prefix else '')+key)
    visit(load_native(root, relative))
    if not result:
        raise ValueError('No triangular EDGAR surfaces')
    return result


def surface_collection(root, relative):
    return {name: _validate_surface(data) for name, data in _raw_surfaces(root, relative).items()}


def geometry_qc(root, relative):
    result = {}
    for name, data in _raw_surfaces(root, relative).items():
        try:
            nodes, faces = _validate_surface(data)
            result[name] = {'status': 'validated_native_surface', 'nodes': len(nodes), 'faces': len(faces)}
        except ValueError as exc:
            result[name] = {'status': 'not_usable_for_solver', 'reason': str(exc)}
    return result


def surface_geometry(root, relative, part=None):
    collection = _raw_surfaces(root, relative)
    if part is not None:
        if part not in collection:
            raise ValueError('Unknown anatomical mesh part')
        return _validate_surface(collection[part])
    if len(collection) != 1:
        raise ValueError('Multiple anatomy parts; select a part explicitly')
    return _validate_surface(next(iter(collection.values())))


def _validate_surface(data):
    nodes, faces = (_matrix(data[k], k) for k in ('node', 'face'))
    # These challenge packages explicitly use one node/face per row (N x 3).
    if nodes.shape[1] != 3 or faces.shape[1] != 3:
        raise ValueError('Unverified surface orientation')
    nodes = nodes.astype(float)
    # MAT does not imply one-based: the supplied Bratislava surfaces are zero-based.
    # Boundary anchors establish the base unambiguously; otherwise abstain.
    if (not np.isfinite(nodes).all() or not np.isfinite(faces).all()
            or np.any(faces != np.floor(faces))):
        raise ValueError('Invalid surface indices')
    if faces.min() == 0 and faces.max() < len(nodes):
        base = 0
    elif faces.min() >= 1 and faces.max() == len(nodes):
        base = 1
    else:
        raise ValueError('Ambiguous surface index base')
    faces = faces.astype(np.int64)-base
    if np.any(np.linalg.norm(np.cross(nodes[faces[:, 1]]-nodes[faces[:, 0]],
                                     nodes[faces[:, 2]]-nodes[faces[:, 0]]), axis=1) <= 1e-12):
        raise ValueError('Degenerate surface triangles')
    return nodes, faces


def mapped_electrodes(root, geometry_path, acquisition: Acquisition, channels):
    acquisition.validate()
    nodes, _ = surface_geometry(root, geometry_path)
    indices = np.asarray(acquisition.channel_node_indices)
    if (indices.dtype.kind not in 'iu' or indices.shape != (channels,)
            or len(set(indices.tolist())) != channels or np.any(indices < 0) or np.any(indices >= len(nodes))):
        raise ValueError('Explicit one-to-one channel-to-node mapping required')
    return nodes[indices] * {'mm': 1., 'cm': 10., 'm': 1000.}[acquisition.geometry_unit]


def inventory(root, output, progress: Callable | None = None):
    """One-file-at-a-time QC/hash index; no huge duplicate signal cache or spill."""
    import psutil
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Keep derived index outside immutable EDGAR inputs')
    output.mkdir(parents=True, exist_ok=True)
    final = output/'inventory.jsonl'
    if final.exists():
        raise FileExistsError('Versioned inventory already exists')
    paths = sorted(root.rglob('*.mat'))
    counts, subjects, issues = {}, set(), {}
    with (output/'inventory.partial').open('x', encoding='utf-8') as stream:
        for i, path in enumerate(paths):
            if psutil.virtual_memory().available < 2.5*1024**3:
                raise MemoryError('EDGAR import suspended: keep 2 GiB reserve plus input margin')
            relative = path.relative_to(root)
            row = {'path': relative.as_posix(), 'schema': SCHEMA, 'clinical_use': False}
            try:
                checked = _safe_file(root, relative)
                row.update(sha256=digest(checked), bytes=checked.stat().st_size)
                cohort = relative.parts[0]
                counts[cohort] = counts.get(cohort, 0)+1
                if relative.parts[1] == 'Geometries':
                    try:
                        parts = surface_collection(root, relative)
                        row.update(kind='surface',
                                   parts={key: {'nodes': len(n), 'faces': len(f)} for key, (n, f) in parts.items()},
                                   status='native_geometry_only', requires_tetrahedral_mesh=True)
                    except ValueError as exc:
                        row.update(kind='geometry', status='geometry_review_required', reason=str(exc))
                else:
                    row.update(identity(relative))
                    values, labels, location, data = signal_native(root, relative)
                    subjects.add(row['subject_id'])
                    finite = np.isfinite(values)
                    row.update(kind='signal', channels=len(values), samples=values.shape[1], lead_names=labels,
                        nonfinite_values=int((~finite).sum()), constant_channels=int(np.sum(np.ptp(values, axis=1) == 0)),
                        reference_present=location is not None, reference_native=None if location is None else location.tolist(),
                        reference_points=0 if location is None else len(np.atleast_2d(location)),
                        native_min=float(values[finite].min()) if finite.any() else None,
                        native_max=float(values[finite].max()) if finite.any() else None,
                        opaque_matlab_note='note' in data and not isinstance(data['note'], str),
                        status='needs_acquisition_metadata', physical_units_verified=False,
                        missing=['sampling_rate_hz', 'voltage_unit', 'geometry_unit', 'coordinate_frame', 'channel_node_mapping'])
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row.update(status='rejected', reason=f'{type(exc).__name__}: {exc}')
            issues[row['status']] = issues.get(row['status'], 0)+1
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if progress and (i % 100 == 0 or i+1 == len(paths)):
                progress(i+1, len(paths))
    (output/'inventory.partial').replace(final)
    summary = {'schema': SCHEMA, 'root': str(root), 'files': len(paths), 'cohorts': counts,
               'subject_groups': sorted(subjects), 'statuses': issues, 'source': SOURCE,
               'inventory_sha256': digest(final), 'clinical_validation_complete': False,
               'warning': 'Native QC only. No unverified units, independent test claims or surface-to-volume inference.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def freeze_subject_roles(rows, validation_subjects):
    """Explicit preregistered split, shared by all beats and native/ECG views."""
    known = {r['subject_id'] for r in rows if r.get('kind') == 'signal'}
    validation = set(validation_subjects)
    if not validation or not validation < known:
        raise ValueError('Both development and validation subjects required')
    return {s: 'validation' if s in validation else 'development' for s in sorted(known)}


def paired_signal_manifest(rows):
    """Pair by preparation AND episode; do not double-count whole runs and cuts."""
    groups = {}
    for row in rows:
        if row.get('kind') == 'signal' and row.get('signal_type') in {'body_surface', 'epicardial'}:
            key = (row['subject_id'], row['episode_id'])
            group = groups.setdefault(key, {})
            if row['signal_type'] in group:
                raise ValueError('Ambiguous multiple recordings for one paired view')
            group[row['signal_type']] = row
    segmented = {(r['subject_id'], r['parent_record']) for r in rows
                 if r.get('kind') == 'signal' and r['episode_id'] != r['parent_record']}
    pairs = []
    for (subject, episode), group in sorted(groups.items()):
        if 'epicardial' not in group:
            continue
        epi, torso = group['epicardial'], group.get('body_surface')
        excluded_parent = episode == epi['parent_record'] and (subject, episode) in segmented
        matching_reference = bool(torso and epi.get('reference_native') is not None and
                                  epi.get('reference_native') == torso.get('reference_native'))
        matched_samples = bool(torso and torso['samples'] == epi['samples'])
        pairs.append({'subject_id': subject, 'episode_id': episode, 'cohort': epi['cohort'],
            'epicardial_path': epi['path'], 'torso_path': torso['path'] if torso else None,
            'epicardial_sha256': epi['sha256'], 'torso_sha256': torso['sha256'] if torso else None,
            'equal_sample_count': matched_samples, 'matching_reference': matching_reference,
            'exclude_overlapping_parent_record': excluded_parent,
            'status': 'native_pair' if torso and matched_samples and matching_reference and not excluded_parent
                      else 'review_or_overlapping_parent',
            'physical_comparison_ready': False, 'reason': 'Acquisition units/time mapping still require verification',
            'clinical_use': False})
    return pairs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root, args.output,
          lambda done, total: print(json.dumps({'done': done, 'total': total}), flush=True)), indent=2))
