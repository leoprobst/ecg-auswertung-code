"""Set-valued twin inference. Fits are not diagnoses or anatomical truth."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class TwinObservation:
    name: str
    value: float
    unit: str
    source: str
    measured_at: str | None
    role: str  # fit | additional | heldout
    representation: str  # signal | marker | clinical
    dependency_group: str  # same acquired ECG, specimen or echo


@dataclass(frozen=True)
class DiscriminationEvidence:
    observation: str
    source: str
    role: str
    compatible_groups: tuple[str, ...]
    excluded_groups: tuple[str, ...]
    new_measurement_recommended: bool = False


@dataclass(frozen=True)
class TwinHypothesisSet:
    schema: str
    status: str
    candidates: tuple[dict, ...]
    evidence: tuple[dict, ...]
    probabilities: dict | None
    prior_sensitive: bool | None
    clinical_use: bool = False


@dataclass(frozen=True)
class TwinValidationReport:
    schema: str
    n_heldout: int
    compatible_set_coverage: float | None
    point_prediction_rmse: float | None
    reference_kind: str
    clinical_use: bool = False


def _errors(components, n):
    required = {'measurement', 'geometry_electrodes', 'physiological_nuisance', 'model_discrepancy'}
    if set(components) != required:
        raise ValueError('Four separately documented error components required')
    total = np.zeros((n, n))
    for name, item in components.items():
        matrix = np.asarray(item['covariance'], float)
        if (not item.get('source') or matrix.shape != (n, n) or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.T, atol=1e-12)):
            raise ValueError('Invalid covariance/provenance: '+name)
        if np.linalg.eigvalsh(matrix).min() < -1e-10*max(1., np.linalg.norm(matrix)):
            raise ValueError('Covariance must be positive semidefinite')
        total += matrix
    if np.any(np.diag(total) <= 0):
        raise ValueError('Zero or missing uncertainty is not certainty')
    return total


def infer_hypotheses(observations, candidates, errors, multiplier=3., prior_sets=None,
                     probability_validation=None):
    """History matching with explicit marginal tolerances, not a confidence set.

    Held-out values cannot influence selection. Correlation is retained for the
    optional, separately validated synthetic Gaussian probability calculation.
    """
    if not observations or not candidates or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Bounded nonempty observations/candidates and tolerance required')
    if len(observations) > 4096 or len(candidates) > 131072:
        raise ValueError('Inference memory budget exceeded')
    names = [o.name for o in observations]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate observable')
    for o in observations:
        if (not o.source or not o.unit or not o.dependency_group or not np.isfinite(o.value)
                or o.role not in {'fit', 'additional', 'heldout'}
                or o.representation not in {'signal', 'marker', 'clinical'}):
            raise ValueError('Incomplete observation provenance')
    selected = np.array([o.role != 'heldout' for o in observations])
    if not selected.any():
        raise ValueError('No conditioning observations')
    for group in {o.dependency_group for o in observations}:
        kinds = {o.representation for o in observations if o.dependency_group == group and o.role != 'heldout'}
        if {'signal', 'marker'} <= kinds:
            raise ValueError('Same ECG signal and derived markers cannot be counted twice')
    covariance = _errors(errors, len(observations))
    limits = multiplier*np.sqrt(np.diag(covariance))
    ids = [c['id'] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate candidate identity')
    residuals, accepted, rows = [], [], []
    observed = np.array([o.value for o in observations])
    for candidate in candidates:
        if not candidate.get('group') or not candidate.get('model_version') or not candidate.get('provenance'):
            raise ValueError('Candidate mechanism, version and provenance required')
        prediction = np.array([candidate['predictions'][name] for name in names], float)
        if candidate.get('units') != dict(zip(names, [o.unit for o in observations])):
            raise ValueError('Prediction units must match each observation')
        physical = candidate.get('physiology_valid') is True and candidate.get('numerics_valid') is True
        in_domain = candidate.get('in_domain') is True
        residual = prediction-observed
        compatible = np.isfinite(prediction[selected]).all() and physical and in_domain
        compatible = bool(compatible and np.all(np.abs(residual[selected]) <= limits[selected]))
        accepted.append(compatible)
        residuals.append(residual)
        status = ('outside_checked_domain' if not in_domain else 'invalid_model' if not physical
                  else 'compatible' if compatible else 'contradicted')
        rows.append({'id': candidate['id'], 'group': candidate['group'], 'parameters': candidate.get('parameters', {}),
                     'status': status, 'max_standardized_discrepancy': float(
                         np.max(np.abs(residual[selected])/limits[selected])) if np.isfinite(residual[selected]).all() else None})
    accepted = np.array(accepted)
    residuals = np.array(residuals)
    evidence = []
    for i, o in enumerate(observations):
        if o.role == 'heldout':
            continue
        yes = {c['group'] for j, c in enumerate(candidates) if np.isfinite(residuals[j, i])
               and abs(residuals[j, i]) <= limits[i] and rows[j]['status'] not in {'invalid_model', 'outside_checked_domain'}}
        all_groups = {c['group'] for c in candidates}
        evidence.append(asdict(DiscriminationEvidence(o.name, o.source, o.role,
                              tuple(sorted(yes)), tuple(sorted(all_groups-yes)))))
    probabilities, sensitive = None, None
    # Clinical probability release is deliberately unavailable: synthetic tests
    # cannot attest the patient observation error distribution or physiology.
    if probability_validation is not None:
        if (probability_validation.get('reference_kind') != 'synthetic_known_gaussian'
                or probability_validation.get('candidate_contract_sha256') != candidate_contract(candidates)
                or probability_validation.get('calibration_passed') is not True
                or any(not c.get('synthetic', False) for c in candidates)):
            raise ValueError('Only bound, validated synthetic probability experiments are supported')
        sub = covariance[np.ix_(selected, selected)]
        if np.linalg.matrix_rank(sub) < len(sub):
            raise ValueError('Singular evidence covariance; remove redundant observations')
        r = residuals[:, selected]
        loglike = -.5*np.einsum('ij,ij->i', r, np.linalg.solve(sub, r.T).T)
        if not prior_sets or len(prior_sets) < 2:
            raise ValueError('At least two prespecified priors required for sensitivity')
        probabilities, winners = {}, set()
        for label, prior in prior_sets.items():
            p = np.array([prior[k] for k in ids], float)
            if not np.isfinite(p).all() or np.any(p <= 0):
                raise ValueError('Positive finite declared priors required')
            score = loglike+np.log(p/p.sum())
            score[~accepted] = -np.inf
            if not accepted.any():
                probabilities[label] = None
                continue
            weight = np.exp(score-np.max(score))
            weight /= weight.sum()
            grouped = {g: float(sum(w for c, w in zip(candidates, weight) if c['group'] == g))
                       for g in sorted({c['group'] for c in candidates})}
            probabilities[label] = grouped
            winners.add(max(grouped, key=grouped.get))
        sensitive = len(winners) > 1
    groups = {c['group'] for c, ok in zip(candidates, accepted) if ok}
    status = 'no_compatible_candidate_on_design' if not groups else 'not_distinguishable' if len(groups) > 1 else 'one_group_on_finite_design'
    return TwinHypothesisSet('twin-hypothesis-set/1', status, tuple(rows), tuple(evidence), probabilities, sensitive)


def candidate_contract(candidates):
    return hashlib.sha256(json.dumps(candidates, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_heldout(observations, candidates, result, errors, multiplier=3.):
    retained = {r['id'] for r in result.candidates if r['status'] == 'compatible'}
    selected = [i for i, o in enumerate(observations) if o.role == 'heldout']
    if not selected or not retained:
        return TwinValidationReport('twin-validation/1', len(selected), None, None, 'no_heldout_or_no_compatible_model')
    covariance = _errors(errors, len(observations))
    prediction = np.array([[c['predictions'][o.name] for o in observations] for c in candidates if c['id'] in retained])[:, selected]
    if not np.isfinite(prediction).all():
        return TwinValidationReport('twin-validation/1', len(selected), None, None,
                                    'incomplete_heldout_predictions_no_candidate_reselection')
    values = np.array([observations[i].value for i in selected])
    allowance = multiplier*np.sqrt(np.diag(covariance)[selected])
    covered = (values >= prediction.min(0)-allowance) & (values <= prediction.max(0)+allowance)
    return TwinValidationReport('twin-validation/1', len(selected), float(covered.mean()),
                                float(np.sqrt(np.mean((np.median(prediction, axis=0)-values)**2))),
                                'existing_heldout_observations_not_used_in_selection')


def compare_information_levels(levels, candidates, known_reference_id=None):
    """Paired retrospective comparison; contraction alone is not success.

    Each level supplies observations and the matching four-component error model.
    The same held-out values, sources and uncertainty must be used at every level.
    No level is selected as a winner and no new examination is recommended.
    """
    if len(levels) < 2:
        raise ValueError('At least two paired information levels required')
    reference = None
    rows = []
    for level in levels:
        observations, errors = level['observations'], level['errors']
        indices = [i for i, o in enumerate(observations) if o.role == 'heldout']
        signature = ([asdict(observations[i]) for i in indices],
                     _errors(errors, len(observations))[np.ix_(indices, indices)].tolist())
        if not indices or (reference is not None and signature != reference):
            raise ValueError('Information levels require identical held-out observations and uncertainty')
        reference = signature
        result = infer_hypotheses(observations, candidates, errors)
        validation = validate_heldout(observations, candidates, result, errors)
        retained = [r for r in result.candidates if r['status'] == 'compatible']
        rows.append({'level': level['name'], 'status': result.status,
                     'retained_candidates': len(retained),
                     'retained_groups': len({r['group'] for r in retained}),
                     'reference_retained': None if known_reference_id is None else
                         any(r['id'] == known_reference_id for r in retained),
                     'validation': asdict(validation)})
    return {'schema': 'twin-information-comparison/1', 'levels': rows,
            'clinical_use': False, 'contraction_is_not_validation': True,
            'new_examinations_recommended': False}
