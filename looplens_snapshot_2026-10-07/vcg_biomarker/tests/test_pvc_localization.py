from dataclasses import replace

import pytest

from vcg_biomarker.gui.pvc_localization import OutflowPVCInputs, localize_outflow_pvc


@pytest.fixture
def eligible():
    return OutflowPVCInputs(
        pvc_confirmed=True, inferior_axis_confirmed=True,
        lbbb_pattern_confirmed=True, structurally_normal=True,
        sinus_no_bundle_block=True, same_morphology=True,
        pvc_transition="v3", sinus_transition="same",
        pvc_v2_r_mv=.9, pvc_v2_s_mv=.6, pvc_v3_r_mv=.8,
        sinus_v2_r_mv=.5, sinus_v2_s_mv=.5,
    )


def test_outflow_indices_match_published_formulas_and_gate(eligible):
    result = localize_outflow_pvc(eligible)
    assert result.status == "candidate" and result.region == "LVOT"
    assert result.v2_transition_ratio == pytest.approx(1.2)
    assert result.v2s_v3r_index == pytest.approx(.75)
    assert result.probability is None
    assert localize_outflow_pvc(replace(eligible, structurally_normal=False)).status == "abstained"
    assert localize_outflow_pvc(replace(eligible, pvc_confirmed=False)).status == "abstained"


def test_outflow_disagreement_and_out_of_domain_are_not_forced(eligible):
    discordant = replace(eligible, pvc_v2_r_mv=.1, pvc_v2_s_mv=.9,
                         pvc_v3_r_mv=1.0, sinus_v2_r_mv=.5, sinus_v2_s_mv=.5)
    result = localize_outflow_pvc(discordant)
    assert result.status == "discordant" and result.region is None
    assert localize_outflow_pvc(replace(eligible, pvc_v3_r_mv=0)).status == "abstained"
    assert localize_outflow_pvc(replace(eligible, pvc_v2_s_mv=float("nan"))).status == "abstained"
    assert localize_outflow_pvc(replace(eligible, sinus_transition="earlier")).status == "discordant"


def test_v2_ratio_not_applied_outside_v3_transition(eligible):
    result = localize_outflow_pvc(replace(eligible, pvc_transition="other"))
    assert result.status == "candidate"
    assert result.v2_transition_ratio is None
    assert any("nicht angewandt" in text for text in result.criteria)
