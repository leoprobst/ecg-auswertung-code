"""Explicit reduced research energy balance, coupled to the existing ORd Na/K pump.

NOT a published complete mitochondrial/pH model. ATP, ADP, Pi, oxygen and
substrate/acid balances are solved; coarse flux laws require explicit parameters.
The pump's ATP/ADP/H dependence is retained from the pinned O'Hara implementation.
No patient ATP/pH is inferred from serum values or diagnoses.
"""
from __future__ import annotations

import numpy as np

UNITS = {
    "atp0": "mM", "adp0": "mM", "pi0": "mM", "oxygen0": "mM", "glucose0": "mM",
    "lactate0": "mM", "co20": "mM", "oxygen_bath": "mM", "glucose_bath": "mM",
    "lactate_bath": "mM", "co2_bath": "mM", "km_oxygen": "mM", "km_adp": "mM",
    "km_pi": "mM", "km_glucose": "mM", "km_atp": "mM", "buffer_capacity": "mM",
    "oxidative_vmax": "mM/ms", "glycolytic_vmax": "mM/ms", "other_atpase_vmax": "mM/ms",
    "oxygen_exchange": "1/ms", "glucose_exchange": "1/ms", "lactate_exchange": "1/ms",
    "co2_exchange": "1/ms", "acid_exchange": "1/ms", "atp_per_oxygen": "1",
    "ph_reference": "1", "acid_per_hydrolysis": "1", "acid_per_glycolytic_glucose": "1",
    "mgatp_fraction": "1", "mgadp_fraction": "1",
}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {*UNITS, "source", "research_only"}:
        raise ValueError("Complete explicit metabolic parameter contract required")
    if config["research_only"] is not True or not isinstance(config["source"], str) or not config["source"].strip():
        raise ValueError("Metabolic research provenance required")
    values = np.array([config[k] for k in UNITS], float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Finite nonnegative metabolic parameters required")
    for key in ("atp0", "adp0", "pi0", "km_oxygen", "km_adp", "km_pi", "km_glucose", "km_atp", "buffer_capacity"):
        if config[key] <= 0:
            raise ValueError("Positive concentration/kinetic scale required: " + key)
    if (not 6 <= config["ph_reference"] <= 8 or not 1 <= config["atp_per_oxygen"] <= 6
            or not 0 < config["mgatp_fraction"] <= 1 or not 0 < config["mgadp_fraction"] <= 1):
        raise ValueError("Invalid pH, ATP/O2 stoichiometry or magnesium-bound fraction")
    if any(config[k] > 100 for k, unit in UNITS.items() if unit == "mM"):
        raise ValueError("Metabolic concentration budget exceeded")
    if any(config[k] > .01 for k, unit in UNITS.items() if unit in {"mM/ms", "1/ms"}):
        raise ValueError("Reduced-model flux budget exceeded")


def add_metabolism(model, myokit, config):
    validate_config(config)
    component = model.add_component("research_energy")

    def variable(name, rhs, unit):
        var = component.add_variable(name)
        var.set_unit(myokit.parse_unit(unit))
        var.set_rhs(rhs)
        return var

    for key, unit in UNITS.items():
        variable(key, f"{float(config[key]):.17g} [{unit}]", unit)
    # Define every state first, so later flux expressions can refer to all of them.
    states = {}
    for name, initial in (("ATP", config["atp0"]), ("ADP", config["adp0"]), ("Pi", config["pi0"]),
                          ("O2", config["oxygen0"]), ("glucose", config["glucose0"]),
                          ("lactate", config["lactate0"]), ("CO2", config["co20"]),
                          ("acid", 0.), ("net_carbon_exchange", 0.), ("net_oxygen_exchange", 0.),
                          ("oxygen_consumed", 0.)):
        var = variable(name, "0 [mM/ms]", "mM")
        var.promote(initial)
        states[name] = var
    variable("pH", "ph_reference - acid / buffer_capacity", "1")
    variable("Jox", "oxidative_vmax * O2/(km_oxygen+O2) * ADP/(km_adp+ADP) * Pi/(km_pi+Pi) * glucose/(km_glucose+glucose)", "mM/ms")
    variable("Jgly", "glycolytic_vmax * glucose/(km_glucose+glucose) * ADP/(km_adp+ADP) * Pi/(km_pi+Pi)", "mM/ms")
    variable("Jpump", "inak.INaK * cell.AFC / cell.vmyo", "mM/ms")
    variable("Jhyd", "Jpump + other_atpase_vmax * ATP/(km_atp+ATP)", "mM/ms")
    for name, expression in {
        "oxygen_supply": "oxygen_exchange*(oxygen_bath-O2)",
        "glucose_supply": "glucose_exchange*(glucose_bath-glucose)",
        "lactate_supply": "lactate_exchange*(lactate_bath-lactate)",
        "co2_supply": "co2_exchange*(co2_bath-CO2)",
    }.items():
        variable(name, expression, "mM/ms")
    expressions = {
        "ATP": "Jox + 2*Jgly - Jhyd", "ADP": "Jhyd - Jox - 2*Jgly", "Pi": "Jhyd-Jox-2*Jgly",
        "O2": "oxygen_supply-Jox/atp_per_oxygen",
        "glucose": "glucose_supply-Jgly-Jox/(6*atp_per_oxygen)",
        "lactate": "lactate_supply+2*Jgly", "CO2": "co2_supply+Jox/atp_per_oxygen",
        "acid": "acid_per_hydrolysis*Jhyd + acid_per_glycolytic_glucose*Jgly - acid_exchange*acid",
        "net_carbon_exchange": "6*glucose_supply+3*lactate_supply+co2_supply",
        "net_oxygen_exchange": "oxygen_supply", "oxygen_consumed": "Jox/atp_per_oxygen",
    }
    for name, expression in expressions.items():
        states[name].set_rhs(expression)
    for name, expression in (("inak.MgATP", "research_energy.ATP*research_energy.mgatp_fraction"),
                             ("inak.MgADP", "research_energy.ADP*research_energy.mgadp_fraction"),
                             ("inak.eP", "research_energy.Pi"),
                             ("inak.H", "1000 [mM] * 10^(-research_energy.pH)")):
        model.get(name).set_rhs(expression)


def balance_report(states, names, config):
    values = {name.split(".")[1]: states[:, i] for i, name in enumerate(names) if name.startswith("research_energy.")}
    for key in ("ATP", "ADP", "Pi", "O2", "glucose", "lactate", "CO2"):
        if values[key].min() < -1e-7:
            raise ArithmeticError("Metabolic pool became negative: " + key)
    ph = config["ph_reference"]-values["acid"]/config["buffer_capacity"]
    if ph.min() < 6 or ph.max() > 8:
        raise ArithmeticError("Outside reduced-model pH domain; no clipping")
    invariants = {
        "adenylates": values["ATP"]+values["ADP"], "phosphate_pool": values["ATP"]+values["Pi"],
        "carbon": 6*values["glucose"]+3*values["lactate"]+values["CO2"]-values["net_carbon_exchange"],
        "oxygen": values["O2"]+values["oxygen_consumed"]-values["net_oxygen_exchange"],
    }
    errors = {key: float(np.max(np.abs(v-v[0]))) for key, v in invariants.items()}
    if max(errors.values()) > 1e-5:
        raise ArithmeticError("Metabolic conservation check failed")
    return {"schema": "reduced-energy-coupling/1", "clinical_use": False, "source": config["source"],
            "balance_errors_mmol_l": errors, "pH_range": [float(ph.min()), float(ph.max())],
            "ATP_range_mmol_l": [float(values["ATP"].min()), float(values["ATP"].max())],
            "pump_reference": "https://doi.org/10.1371/journal.pcbi.1002061",
            "limitations": ["New reduced research coupling, not a validated full metabolism model.",
                            "Oxidative/glycolytic lumped flux laws and fixed buffer capacity are explicit assumptions.",
                            "Fixed magnesium-bound fractions; no full binding chemistry, mitochondrial potential, ROS or fatty-acid pathways.",
                            "Only Na/K pump ATP usage is mechanistically coupled; other ATPases are prescribed lumped demand.",
                            "Source ORd pump has documented parameter/erratum caveats; no replacement with a newer pump is implied.",
                            "No pH effect on other channels, KATP or extracellular acidosis inferred."]}


def synthetic_demo_config():
    """Explicit numerical fixture, NEVER patient defaults or published fitted rates."""
    return dict(source="synthetic numerical balance experiment; rates not patient-calibrated", research_only=True,
        atp0=9.8, adp0=.05, pi0=4.2, oxygen0=.03, glucose0=5., lactate0=1., co20=1.,
        oxygen_bath=.03, glucose_bath=5., lactate_bath=1., co2_bath=1.,
        km_oxygen=.005, km_adp=.05, km_pi=.5, km_glucose=1., km_atp=.1, buffer_capacity=30.,
        oxidative_vmax=.0001, glycolytic_vmax=.000002, other_atpase_vmax=.00003,
        oxygen_exchange=.001, glucose_exchange=.001, lactate_exchange=.001, co2_exchange=.001,
        acid_exchange=.0001, atp_per_oxygen=5., ph_reference=7., acid_per_hydrolysis=1.,
        acid_per_glycolytic_glucose=0., mgatp_fraction=1., mgadp_fraction=1.)
