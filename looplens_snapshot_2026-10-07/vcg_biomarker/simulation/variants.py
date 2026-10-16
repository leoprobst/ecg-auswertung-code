"""Source-bound variant mechanisms, not automatic genotype inference.

SCN5A 1795insD SAN heterozygous formulation: Wilders 2018,
https://pmc.ncbi.nlm.nih.gov/articles/PMC5855856/, equations 13–20.
No transfer of this SAN assay into ventricular Brugada/LQT3 diagnosis.
"""
import numpy as np

SAN_VARIANT = "SCN5A_1795insD_heterozygous_SAN_2018"
VARIANTS = {"reference": "Publizierte Referenz", SAN_VARIANT: "SCN5A 1795insD · heterozygoter SAN-Versuch (2018)"}


def gate_reference(v, mutant=False):
    """Independent numerical reference of activation/inactivation and time constants (s)."""
    a = float(v) - (8.1 if mutant else 0.)
    h = float(v) + (7.3 if mutant else 0.)
    e = a+41.
    am = 2000. if abs(e) < 1e-5 else 200.*e / (-np.expm1(-.1*e))
    bm = 8000.*np.exp(-.056*(a+66.))
    ah = 20.*np.exp(-.125*(h+75.))
    bh = 2000./(1+320.*np.exp(-.1*(h+75.)))
    return 1/(1+np.exp(-(a+42.0504)/8.3106)), 1/(1+np.exp((h+69.804)/4.4565)), 1/(am+bm), 1/(ah+bh)


def add_san_1795insd(model, myokit):
    """WT 50% + mutant 25% control conductance, distinct gates, 0.6% persistent mutant channels."""
    wt = model.get("ina.gNa")
    wt.set_rhs(myokit.Multiply(wt.rhs(), myokit.Number(.5)))
    component = model.add_component("scn5a_1795insd")

    def variable(name, expression, unit):
        result = component.add_variable(name)
        result.set_unit(myokit.parse_unit(unit))
        result.set_rhs(expression)
        return result

    variable("Va", "membrane.V - 8.1 [mV]", "mV")
    variable("Vh", "membrane.V + 7.3 [mV]", "mV")
    variable("m_inf", "1 / (1 + exp(-(Va + 42.0504 [mV]) / 8.3106 [mV]))", "1")
    variable("h_inf", "1 / (1 + exp((Vh + 69.804 [mV]) / 4.4565 [mV]))", "1")
    variable("E0", "Va + 41 [mV]", "mV")
    variable("alpha_m", "if(abs(E0) < 1e-5 [mV], 2000 [1/s], 200 [1/mV/s] * E0 / (1 - exp(-0.1 [1/mV] * E0)))", "1/s")
    variable("beta_m", "8000 [1/s] * exp(-0.056 [1/mV] * (Va + 66 [mV]))", "1/s")
    variable("alpha_h", "20 [1/s] * exp(-0.125 [1/mV] * (Vh + 75 [mV]))", "1/s")
    variable("beta_h", "2000 [1/s] / (1 + 320 * exp(-0.1 [1/mV] * (Vh + 75 [mV])))", "1/s")
    voltage = model.get("membrane.V").initial_value(as_float=True)
    m0, h0, _, _ = gate_reference(voltage, True)
    for name, initial in (("m", m0), ("h", h0)):
        state = variable(name, "0", "1")
        state.promote(float(initial))
        state.set_rhs(f"({name}_inf - {name}) * (alpha_{name} + beta_{name})")
    variable("I", "0.005575 [uS] * m^3 * ((1 - 0.006) * h + 0.006) * (membrane.V - rev.E_mh)", "nA")
    current = model.get("ina.INa")
    current.set_rhs(myokit.Plus(current.rhs(), myokit.Name(component.get("I"))))


def variant_metadata():
    return {"variant": SAN_VARIANT, "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5855856/",
            "wild_type_conductance_fraction": .5, "mutant_conductance_fraction": .25,
            "mutant_noninactivating_fraction": .006, "activation_shift_mv": 8.1, "inactivation_shift_mv": -7.3,
            "mutant_initial_gates": "steady-state gates at published initial voltage; initialization is not equilibration",
            "clinical_use": False, "scope": "heterozygous SAN single-cell mechanistic assay only",
            "limitations": ["No ventricular phenotype or ECG diagnosis inferred.",
                            "Experimental inactivation-kinetic changes omitted by the source formulation."]}
