"""Storage-disease research must not upgrade surface proxies into EPS claims."""
from vcg_biomarker.core.biomarkers.advanced_vcg_biomarkers import ADVANCED_SPECS


def test_surface_preexcitation_descriptions_preserve_pathway_uncertainty():
    specs = {spec.name: spec for spec in ADVANCED_SPECS}
    flag = specs["delta_wave_present_flag"].description
    quadrant = specs["accessory_pathway_quadrant_estimate"].description
    assert "cannot distinguish atrioventricular from fasciculoventricular" in flag
    assert "not a genotype or pathway-risk diagnosis" in flag
    assert "not age-normalized" in flag
    assert "do not interpret it as pediatric short PR" in flag
    assert "Not an ablation target or treatment indication" in quadrant
    assert "fasciculoventricular pathways can coexist" in quadrant
