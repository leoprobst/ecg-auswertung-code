"""Explicit TP06 KATP extension, Kazbanov et al. 2014, equation 3.

DOI: 10.1371/journal.pcbi.1003891 (CC-BY). Opening fraction is prescribed,
NOT inferred from ATP, pH, a serum potassium value or a patient diagnosis.
The published base MMT file remains unchanged. Units: mV, mM, A/F.
"""
from __future__ import annotations

import numpy as np

SOURCE = "https://doi.org/10.1371/journal.pcbi.1003891"
VERSION = "kazbanov-katp-2014/1"
MAX_OPEN_FRACTION = .01  # Bounded research range, not all mathematically possible openings.


def katp_current(voltage_mv, reversal_mv, potassium_mmol_l, open_fraction):
    """Independent numeric equation reference; fraction .0055 means 0.55%."""
    v, ek, ko, f = np.broadcast_arrays(voltage_mv, reversal_mv, potassium_mmol_l, open_fraction)
    if (not all(np.isfinite(a).all() for a in (v, ek, ko, f)) or np.any(ko <= 0)
            or np.any((f < 0) | (f > MAX_OPEN_FRACTION))):
        raise ValueError("Invalid KATP scenario")
    return 155. * f * (ko/5.4)**.3 * (v-ek) / (40. + 3.5*np.exp(.025*v))


def add_katp(model, myokit, open_fraction):
    """Add current to BOTH membrane and intracellular potassium balances."""
    component = model.add_component("research_katp")
    fraction = component.add_variable("open_fraction")
    fraction.set_unit(myokit.units.dimensionless)
    fraction.set_rhs(open_fraction)
    current = component.add_variable("I")
    current.set_unit(myokit.parse_unit("A/F"))
    current.set_rhs(
        "155 [mS/uF] * open_fraction * (extra.Ko / 5.4 [mM])^0.3"
        " * (membrane.V - nernst.EK) / (40 + 3.5 * exp(0.025 [1/mV] * membrane.V))")
    for name in ("membrane.i_ion", "potassium.IK_total"):
        variable = model.get(name)
        variable.set_rhs(myokit.Plus(variable.rhs(), myokit.Name(current)))
    return fraction


def mechanism_metadata(open_fraction, regional=False):
    return {"schema": VERSION, "source": SOURCE, "open_fraction": float(open_fraction),
            "regional_override": bool(regional), "clinical_use": False,
            "potassium_balance_included": True,
            "limitations": ["Prescribed channel opening, no ATP/ADP/pH/oxygen metabolism solved.",
                            "Acidosis may be explored via explicit INa/ICaL factors; no inferred pH.",
                            "No ischemia duration or infarct diagnosis inferred."]}
