"""Development-only observational constraints for twins, never ionic ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd

LABS = {
    'potassium': {'units': {'meq/l': 1., 'mmol/l': 1.}, 'range': (1., 10.), 'unit': 'mmol/l'},
    'ionized_calcium': {'units': {'mmol/l': 1.}, 'range': (.3, 3.), 'unit': 'mmol/l'},
    'blood_gas_ph': {'units': {'units': 1., 'ph': 1., '': 1.}, 'range': (6.5, 8.), 'unit': 'pH'},
}
SHIFTS_HOURS = (-24., -6., -1., -.25, 0., .25, 1., 6., 24.)


def exact_lab(frame, lab, suffix):
    spec = LABS[lab]
    value = pd.to_numeric(frame['value'+suffix], errors='coerce').to_numpy(float)
    units = frame['unit'+suffix].fillna('').astype(str).str.lower().str.strip()
    factors = units.map(spec['units']).to_numpy(float)
    value = value*factors
    valid = np.isfinite(value) & (value >= spec['range'][0]) & (value <= spec['range'][1])
    valid &= frame['censor'+suffix].isna().to_numpy()
    return value, valid


def eligible_pair(frame, lab, clock_shift_hours=0.):
    """Frozen lab links; stress scenario, not estimated clock error or rematching."""
    a, va = exact_lab(frame, lab, '0')
    b, vb = exact_lab(frame, lab, '1')
    delta0 = pd.to_numeric(frame.delta0, errors='coerce').to_numpy(float)-clock_shift_hours
    delta1 = pd.to_numeric(frame.delta1, errors='coerce').to_numpy(float)-clock_shift_hours
    valid = va & vb & (delta0 >= -2) & (delta0 <= 0) & (delta1 >= -2) & (delta1 <= 0)
    valid &= frame.event0.notna().to_numpy() & frame.event1.notna().to_numpy()
    valid &= frame.event0.ne(frame.event1).fillna(False).to_numpy(dtype=bool)
    valid &= frame.match0.eq('same_admission').to_numpy() & frame.match1.eq('same_admission').to_numpy()
    return b-a, valid


def fit_delta(exposure, outcome, covariates, eligible):
    """HC3 observational slope, one reference-blind pair per patient."""
    import statsmodels.api as sm
    from threadpoolctl import threadpool_limits
    x = np.column_stack((np.ones(len(exposure)), exposure, covariates))
    good = eligible & np.isfinite(x).all(1) & np.isfinite(outcome)
    x, y = x[good], outcome[good]
    if len(y) < 100 or np.unique(x[:, 1]).size < 10:
        return {'status': 'non_estimable', 'reason': 'insufficient_pairs_or_exposure_variation', 'n': len(y)}
    if np.linalg.matrix_rank(x) < x.shape[1]:
        return {'status': 'non_estimable', 'reason': 'rank_deficient', 'n': len(y)}
    scale = np.std(x[:, 1:], axis=0)
    standardized = np.column_stack((np.ones(len(x)), (x[:, 1:]-x[:, 1:].mean(0))/scale))
    condition = np.linalg.cond(standardized)
    if condition > 1e6:
        return {'status': 'non_estimable', 'reason': 'ill_conditioned', 'n': len(y)}
    with threadpool_limits(limits=1):
        fit = sm.OLS(y, standardized).fit(cov_type='HC3')
    effect, ci = float(fit.params[1]/scale[0]), (fit.conf_int()[1]/scale[0]).tolist()
    if not np.isfinite([effect, *ci, fit.pvalues[1]]).all():
        return {'status': 'non_estimable', 'reason': 'nonfinite_inference', 'n': len(y)}
    return {'status': 'computed', 'n': len(y), 'effect_ms_per_lab_unit': effect,
            'ci95': ci, 'p': float(fit.pvalues[1]), 'condition_number': float(condition),
            'inference': 'HC3 exploratory association, not treatment effect or cellular parameter'}


def evaluate_pairs(pairs, labs):
    if pairs.subject_id.duplicated().any():
        raise ValueError('At most one pair per patient')
    if not pairs.source_split.eq('train').all():
        raise ValueError('Existing development train membership required')
    if labs.duplicated(['subject_id', 'study_id', 'lab_group']).any():
        raise ValueError('Ambiguous lab linkage')
    rows, readiness = [], {}
    for lab in LABS:
        chosen = labs.loc[labs.lab_group.eq(lab)].drop(columns='lab_group')
        frame = pairs.copy()
        for side in ('0', '1'):
            renamed = chosen.rename(columns={c: c+side for c in chosen if c != 'subject_id'})
            frame = frame.merge(renamed, on=['subject_id', 'study_id'+side], how='left', validate='one_to_one')
        exposure, eligible = eligible_pair(frame, lab)
        readiness[lab] = {'unit': LABS[lab]['unit'], 'eligible_preceding_2h_pairs': int(eligible.sum()),
                         'fixed_link_clock_stress_counts': {
                             str(shift): int(eligible_pair(frame, lab, shift)[1].sum()) for shift in SHIFTS_HOURS}}
        hr0, hr1 = (pd.to_numeric(frame['hr'+s], errors='coerce').to_numpy(float) for s in ('0', '1'))
        eligible &= (hr0 >= 30) & (hr0 <= 200) & (hr1 >= 30) & (hr1 <= 200)
        eligible &= frame.usable0.fillna(False).to_numpy(bool) & frame.usable1.fillna(False).to_numpy(bool)
        # Machine report exclusions are imperfect; no independent rhythm adjudication implied.
        paced = frame.report0.fillna('').str.contains('PACED|PACEMAKER', case=False, regex=True)
        paced |= frame.report1.fillna('').str.contains('PACED|PACEMAKER', case=False, regex=True)
        eligible &= ~paced.to_numpy()
        covariates = np.column_stack((hr1-hr0, frame.age, frame.female, frame.interval_hours))
        for endpoint, field, bounds in [('qrs', 'qrs', (40, 250)), ('qtc_fridericia', 'qt', (200, 700))]:
            y0, y1 = (pd.to_numeric(frame[field+s], errors='coerce').to_numpy(float) for s in ('0', '1'))
            valid = eligible & (y0 >= bounds[0]) & (y0 <= bounds[1]) & (y1 >= bounds[0]) & (y1 <= bounds[1])
            if endpoint == 'qtc_fridericia':
                y0 = y0*np.cbrt(hr0/60)
                y1 = y1*np.cbrt(hr1/60)
            result = fit_delta(exposure, y1-y0, covariates, valid)
            rows.append({'lab': lab, 'endpoint': endpoint, **result})
    # Fixed family of six, including abstained hypotheses conservatively as p=1.
    from statsmodels.stats.multitest import multipletests
    qs = multipletests([r.get('p', 1.) for r in rows], method='fdr_bh')[1]
    for row, q in zip(rows, qs):
        row['q_global_six'] = float(q) if row['status'] == 'computed' else None
    return {'schema': 'mimic-twin-observational-constraints/1', 'clinical_use': False,
            'independent_cellular_calibration': False, 'causal_claim': False,
            'pairs_selected_before_lab_inspection': len(pairs), 'readiness': readiness, 'models': rows}
