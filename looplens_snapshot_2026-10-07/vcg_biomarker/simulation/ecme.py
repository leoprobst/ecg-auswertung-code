"""Pinned Cortassa 2006 ECME reference assay, NOT a validated human twin.

Retains the published guinea-pig/common-pool cell, contractile ATP demand,
mitochondrial TCA/respiration, membrane potential, calcium and CK shuttles.
It is deliberately separate from ORd, avoiding double-counted ATP/Ca fluxes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

SOURCE_SHA256 = '357eeb2c8fdfa108d400f1d1a796a51b5effeec4eb9c76e851a21b99d39e8afe'
SOURCE_URL = ('https://models.physiomeproject.org/w/andre/cortassa-ECME-2006/rawfile/'
              'c4bd63604cd1f826c37c44b9e2e2c7cd1fa6331a/cellml-1.0/S1.xml')


@dataclass(frozen=True)
class ECMEExperiment:
    duration_ms: float = 4000.
    cycle_ms: float = 4000.
    output_ms: float = 1.
    respiratory_capacity_scale: float = 1.
    proton_leak_scale: float = 1.
    cytosolic_atpase_scale: float = 1.
    rtol: float = 1e-7
    atol: float = 1e-10
    acknowledge_source_unit_limitations: bool = False

    def validate(self):
        values = [v for k, v in asdict(self).items() if k != 'acknowledge_source_unit_limitations']
        if not np.isfinite(values).all():
            raise ValueError('Finite ECME parameters required')
        if not self.acknowledge_source_unit_limitations:
            raise ValueError('ECME source has incomplete units: explicit research acknowledgement required')
        if not 1 <= self.duration_ms <= 60000 or not 250 <= self.cycle_ms <= 10000 or not .1 <= self.output_ms <= 10:
            raise ValueError('ECME time/output budget exceeded')
        if self.duration_ms/self.output_ms > 120000:
            raise ValueError('ECME output memory budget exceeded')
        if any(not .1 <= x <= 3 for x in (self.respiratory_capacity_scale, self.proton_leak_scale, self.cytosolic_atpase_scale)):
            raise ValueError('Unsupported mechanistic perturbation; not a disease calibration')
        if not 1e-9 <= self.rtol <= 1e-5 or not 1e-12 <= self.atol <= 1e-8:
            raise ValueError('Invalid ECME tolerances')


def load_reference():
    from .ionic import myokit_module
    myokit = myokit_module()
    from lxml import etree
    from myokit.formats.cellml.v1 import CellMLParser
    path = Path(__file__).resolve().parents[2]/'assets/simulation/cortassa-2006-S1.cellml'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Changed ECME reference asset')
    tree = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    ns = '{http://www.cellml.org/cellml/1.0#}'
    repairs = {'empty_dimensionless_unit_declarations': 0, 'removed_duplicate_metadata_ids': 0,
               'unannotated_numeric_literals': 0}
    # Source encoding compatibility, no numeric equations/initial states changed.
    for units in tree.iter(ns+'units'):
        if len(units) == 0 and units.get('base_units') == 'no':
            etree.SubElement(units, ns+'unit', units='dimensionless')
            repairs['empty_dimensionless_unit_declarations'] += 1
    seen = set()
    for element in tree.iter():
        key = '{http://www.cellml.org/metadata/1.0#}id'
        if key in element.attrib:
            value = element.get(key)
            if value in seen:
                element.attrib.pop(key)
                repairs['removed_duplicate_metadata_ids'] += 1
            seen.add(value)
        if element.tag == '{http://www.w3.org/1998/Math/MathML}cn' and ns+'units' not in element.attrib:
            element.set(ns+'units', 'dimensionless')
            repairs['unannotated_numeric_literals'] += 1
    model = CellMLParser().parse(tree).myokit_model()
    model.validate()
    try:
        model.check_units(myokit.UNIT_STRICT)
        unit_status = 'pass'
    except myokit.IncompatibleUnitError:
        unit_status = 'source_units_incomplete_not_clinically_released'
    return model, {**repairs, 'unit_status': unit_status}


@lru_cache(maxsize=2)
def compile_reference(respiratory_scale=1., leak_scale=1., atpase_scale=1.):
    from .ionic import compile_expression_rhs, myokit_module
    myokit = myokit_module()
    model, details = load_reference()
    # No modulus-dependent missed stimuli: solver supplies exact event intervals.
    stim = model.get('stimulus_current.Istim')
    stim.set_rhs(0)
    stim.set_binding('pace')
    for key, scale in (('rhoREN', respiratory_scale), ('gh', leak_scale), ('VATPase_cyto', atpase_scale)):
        matches = [v for v in model.variables(deep=True) if v.name() == key and v.is_literal()]
        if len(matches) != 1:
            raise ValueError('Unresolved ECME perturbation target: '+key)
        variable = matches[0]
        variable.set_rhs(myokit.Multiply(variable.rhs(), myokit.Number(scale)))
    model.validate()
    return model, compile_expression_rhs(model), details


def simulate_ecme(spec: ECMEExperiment, progress=None, cancelled=None):
    import psutil
    spec.validate()
    model, rhs, details = compile_reference(spec.respiratory_capacity_scale, spec.proton_leak_scale, spec.cytosolic_atpase_scale)
    names = [v.qname() for v in model.states()]
    initial = np.asarray(model.initial_values(as_floats=True), float)
    times = np.arange(0, spec.duration_ms, spec.output_ms)
    if not len(times) or times[-1] != spec.duration_ms:
        times = np.append(times, spec.duration_ms)
    values = np.empty((len(times), len(initial)))
    values[0] = initial
    boundaries = {0., spec.duration_ms}
    for onset in np.arange(0, spec.duration_ms, spec.cycle_ms):
        boundaries.update((float(onset), min(float(onset+.5), spec.duration_ms)))
    boundaries = sorted(boundaries)
    started = time.monotonic()
    evaluations = 0
    with threadpool_limits(limits=1):
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            if cancelled and cancelled():
                raise InterruptedError('Cancelled ECME experiment')
            if psutil.virtual_memory().available < 2.5*1024**3:
                raise MemoryError('ECME preserves RAM reserve')
            stimulus = -100. if ((lo+hi)/2) % spec.cycle_ms < .5 else 0.
            def rate(t, y):
                if cancelled and cancelled():
                    raise InterruptedError('Cancelled ECME experiment')
                return rhs(t, y, stimulus)
            result = solve_ivp(rate, (lo, hi), initial, method='BDF', rtol=spec.rtol, atol=spec.atol, dense_output=True)
            if not result.success or not np.isfinite(result.y).all():
                raise ArithmeticError('ECME integration failed: '+result.message)
            take = (times > lo) & (times <= hi)
            if np.any(take):
                values[take] = result.sol(times[take]).T
            initial = result.y[:, -1]
            evaluations += result.nfev
            if progress:
                progress(hi, spec.duration_ms, time.monotonic()-started)
    if not np.isfinite(values).all():
        raise ArithmeticError('Nonfinite ECME output')
    metrics = {}
    for name in ('ATPi', 'ATPi_cyto', 'ADPm', 'NADH', 'Dpsi', 'Cai', 'Cam', 'CrPi_mito', 'CrPi_cyto', 'V'):
        column = values[:, names.index('model_2.'+name)]
        if name not in {'V', 'Dpsi'} and column.min() < -1e-7:
            raise ArithmeticError('Negative ECME state '+name)
        metrics[name] = {'initial': float(column[0]), 'final': float(column[-1]),
                         'min': float(column.min()), 'max': float(column.max())}
    metadata = {'schema': 'published-ecme-assay/1', 'model': 'Cortassa-2006-S1',
                'species': 'guinea_pig', 'clinical_use': False, 'human_ORD_coupled': False,
                'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256,
                'source_encoding': details, 'experiment': asdict(spec), 'state_names': names,
                'metrics': metrics, 'nfev': evaluations, 'elapsed_seconds': time.monotonic()-started,
                'time_convention': 'ms per published S1 protocol; source CellML time is annotated dimensionless',
                'limitations': ['Published reference with incomplete dimensional annotations.',
                    'Guinea-pig common-pool ECC; no validated human parameter transfer.',
                    'Fixed mitochondrial delta-pH; no complete cytosolic acidosis/ROS/fatty-acid model.',
                    'Respiratory capacity perturbation is not an oxygen concentration or perfusion measurement.',
                    'No metabolite calibration from EDGAR electrograms alone.']}
    return times, values, metadata
