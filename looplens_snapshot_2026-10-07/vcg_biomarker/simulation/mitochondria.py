"""Beard 2005 mechanistic oxidative phosphorylation assay (isolated mitochondria).

Explicit matrix pH, membrane potential, Mg binding, electron transport, ATP
synthase and metabolite transport. No unsupported automatic ORd/ECME splice.
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

SHA256 = '4a4f931592df0e77a92c79447bf6d8947c7325f721a74b4d336d370abaaf3981'
SOURCE = ('https://models.cellml.org/workspace/beard_2005/rawfile/'
          'e8021d49ce9b86c3a3d7488ff1eb65b765a969ce/beard_2005.cellml')


@dataclass(frozen=True)
class MitochondrialExperiment:
    duration_s: float = 10.
    output_s: float = .02
    oxygen_mmol_l: float = .026
    bath_pH: float = 7.1
    bath_adp_mmol_l: float = 0.
    bath_phosphate_mmol_l: float = .125
    proton_leak_scale: float = 1.
    rtol: float = 1e-7
    atol: float = 1e-12
    parameter_set: str = 'cellml_source'

    def validate(self):
        if self.parameter_set not in {'cellml_source', 'paper_2006_corrected'}:
            raise ValueError('Unknown published parameter set')
        if not np.isfinite([v for k, v in asdict(self).items() if k != 'parameter_set']).all():
            raise ValueError('Finite mitochondrial protocol required')
        if not .01 <= self.duration_s <= 300 or not .001 <= self.output_s <= 1 or self.duration_s/self.output_s > 30000:
            raise ValueError('Mitochondrial resource budget exceeded')
        if not 1e-5 <= self.oxygen_mmol_l <= .2 or not 6.5 <= self.bath_pH <= 7.8:
            raise ValueError('Oxygen/pH outside assay domain')
        if not 0 <= self.bath_adp_mmol_l <= 5 or not .01 <= self.bath_phosphate_mmol_l <= 10 or not .1 <= self.proton_leak_scale <= 3:
            raise ValueError('Unsupported substrate/leak conditions')
        if not 1e-9 <= self.rtol <= 1e-5 or not 1e-14 <= self.atol <= 1e-10:
            raise ValueError('Invalid mitochondrial tolerance')


def load_beard():
    from .ionic import myokit_module
    myokit = myokit_module()
    from lxml import etree
    from myokit.formats.cellml.v1 import CellMLParser
    path = Path(__file__).resolve().parents[2]/'assets/simulation/beard-2005.cellml'
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Changed mitochondrial source')
    tree = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    ns = '{http://www.cellml.org/cellml/1.0#}'
    # CellML tags for compartment amount are bookkeeping dimensions, not SI units.
    # Keep all numeric water/volume factors; do not identify matrix and cytosol.
    for units in tree.iter(ns+'units'):
        if units.get('base_units') == 'yes' and units.get('name') in {'cell', 'cytoplasm', 'mitochondria'}:
            units.set('base_units', 'no')
            etree.SubElement(units, ns+'unit', units='dimensionless')
    model = CellMLParser().parse(tree).myokit_model()
    # CellML initial_value on algebraic constants lacks a separate literal unit.
    # Transfer the declared variable unit; numeric values remain unchanged.
    for variable in model.variables(deep=True):
        if variable.is_literal():
            variable.set_rhs(myokit.Number(variable.eval(), variable.unit()))
    model.validate()
    model.check_units(myokit.UNIT_STRICT)
    return model


@lru_cache(maxsize=3)
def compiled(spec: MitochondrialExperiment):
    from .ionic import compile_expression_rhs, myokit_module
    spec.validate()
    myokit = myokit_module()
    model = load_beard()
    if spec.parameter_set == 'paper_2006_corrected':
        # Table 1 correction: Wx=.6435, Wi=.0715; source uses Wm=.72376.
        # Source also disables AK; published corrected table gives 1e6.
        # Separate protocol, never a silent correction of the downloaded asset.
        for name, value in (('Fixed_parameters.W_m', .715), ('Adjustable_parameters.x_AK', 1e6)):
            variable = model.get(name)
            variable.set_rhs(myokit.Number(value, variable.unit()))
    for name, value in (('Fixed_parameters.pH_e', spec.bath_pH),
                        ('Environment.ADP_e', spec.bath_adp_mmol_l/1000),
                        ('Environment.Pi_e', spec.bath_phosphate_mmol_l/1000)):
        variable = model.get(name)
        variable.set_rhs(myokit.Number(value, variable.unit()))
    leak = model.get('Adjustable_parameters.x_Hle')
    leak.set_rhs(myokit.Multiply(leak.rhs(), myokit.Number(spec.proton_leak_scale)))
    model.get('dO2_dt.O2').set_initial_value(spec.oxygen_mmol_l/1000)
    model.validate()
    model.check_units(myokit.UNIT_STRICT)
    return model, compile_expression_rhs(model)


def simulate_mitochondria(spec: MitochondrialExperiment, progress=None, cancelled=None):
    import psutil
    model, rhs = compiled(spec)
    names = [v.qname() for v in model.states()]
    initial = np.asarray(model.initial_values(as_floats=True), float)
    times = np.arange(0, spec.duration_s, spec.output_s)
    times = np.append(times, spec.duration_s)
    states = np.empty((len(times), len(initial)))
    states[0] = initial
    started = time.monotonic()
    lo = 0.
    with threadpool_limits(limits=1):
        while lo < spec.duration_s:
            if cancelled and cancelled():
                raise InterruptedError('Cancelled mitochondrial assay')
            if psutil.virtual_memory().available < 2.5*1024**3:
                raise MemoryError('Mitochondrial assay preserves RAM reserve')
            hi = min(lo+1., spec.duration_s)
            def rate(t, y):
                if cancelled and cancelled():
                    raise InterruptedError('Cancelled mitochondrial assay')
                return rhs(t, y, 0.)
            result = solve_ivp(rate, (lo, hi), initial, method='BDF', rtol=spec.rtol, atol=spec.atol, dense_output=True)
            if not result.success or not np.isfinite(result.y).all():
                raise ArithmeticError('Mitochondrial integration failed')
            take = (times > lo) & (times <= hi)
            if take.any():
                states[take] = result.sol(times[take]).T
            initial = result.y[:, -1]
            lo = hi
            if progress:
                progress(hi, spec.duration_s, time.monotonic()-started)
    if not np.isfinite(states).all():
        raise ArithmeticError('Nonfinite mitochondrial output')
    for i, name in enumerate(names):
        if name != 'dPsi_dt.dPsi' and states[:, i].min() < -1e-10:
            raise ArithmeticError('Negative metabolite concentration: '+name)
    hydrogen = states[:, names.index('dH_x_dt.H_x')]
    if np.any(hydrogen <= 0):
        raise ArithmeticError('Nonpositive matrix proton concentration')
    ph = -np.log10(hydrogen)  # M relative to 1 mol/l, NOT mM
    # Total matrix adenylates are transported, but Mg binding cannot exceed totals.
    for compartment in ('x', 'i'):
        for nucleotide in ('ATP', 'ADP'):
            total = states[:, names.index(f'd{nucleotide}_{compartment}_dt.{nucleotide}_{compartment}')]
            bound = states[:, names.index(f'd{nucleotide}_m{compartment}_dt.{nucleotide}_m{compartment}')]
            if np.any(bound > total+1e-9):
                raise ArithmeticError('Bound nucleotide exceeds total')
    totals = {name: states[:, names.index(name)] for name in names}
    adenylates = totals['dATP_x_dt.ATP_x']+totals['dADP_x_dt.ADP_x']
    magnesium = totals['dMg_x_dt.Mg_x']+totals['dATP_mx_dt.ATP_mx']+totals['dADP_mx_dt.ADP_mx']
    balance_errors = {'matrix_adenylates_M': float(np.max(np.abs(adenylates-adenylates[0]))),
                      'matrix_magnesium_M': float(np.max(np.abs(magnesium-magnesium[0])))}
    if max(balance_errors.values()) > 1e-8:
        raise ArithmeticError('Mitochondrial conserved-pool error')
    metrics = {name: {'initial': float(states[0, i]), 'final': float(states[-1, i]),
                       'min': float(states[:, i].min()), 'max': float(states[:, i].max())}
               for i, name in enumerate(names)}
    return times, states, {'schema': 'published-mitochondrial-assay/1', 'source': SOURCE,
        'source_sha256': SHA256, 'experiment': asdict(spec), 'state_names': names,
        'state_units': [str(v.unit()) for v in model.states()], 'time_unit': 's',
        'matrix_pH_initial': float(ph[0]), 'matrix_pH_final': float(ph[-1]),
        'metrics': metrics, 'unit_check': 'pass_with_compartment_labels_normalized',
        'balance_errors': balance_errors,
        'reference_parameter_audit': {
            'source': 'https://doi.org/10.1371/journal.pcbi.0020008',
            'matrix_water_fraction': {'actual': model.get('Fixed_parameters.W_x').eval(), 'paper': .6435},
            'intermembrane_water_fraction': {'actual': model.get('Fixed_parameters.W_i').eval(), 'paper': .0715},
            'adenylate_kinase_activity': {'actual': model.get('Adjustable_parameters.x_AK').eval(), 'paper': 1e6},
            'note': 'Source protocol and corrected-paper parameters are explicit separate variants.'},
        'clinical_use': False, 'patient_calibrated': False, 'oxygen_protocol': 'clamped bath concentration',
        'elapsed_seconds': time.monotonic()-started,
        'limitations': ['Isolated mitochondrial assay, not human whole-cell perfusion/metabolism.',
            'No fatty-acid/glycolysis/TCA reaction network, ROS or whole-cell pH transport in this model.',
            'External bath pH and oxygen are imposed conditions, not blood-derived cell measurements.',
            'No automatic splice into ORd or ECME; compartment/flux matching requires independent validation.']}
