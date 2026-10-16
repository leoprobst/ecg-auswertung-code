import pytest

from vcg_biomarker.mimic.user_guidance import biomarker_tooltip


@pytest.mark.parametrize("plane,axes", [
    ("frontal_xy", "X / Y"), ("horizontal_xz", "X / Z"), ("sagittal_yz", "Y / Z"),
])
def test_projection_tooltip_preserves_axis_order_and_research_limits(plane, axes):
    text = biomarker_tooltip(f"qrs_loop_{plane}_signed_area")
    assert plane in text and axes in text
    assert "Achsenumkehr" in text
    assert "keine validierte klinische Schwelle" in text
    assert "kein Nachweis einer Neuberechnung" in text


def test_unknown_marker_is_not_given_an_anatomical_claim():
    assert "2D-Projektion" not in biomarker_tooltip("unknown_svd_component")
