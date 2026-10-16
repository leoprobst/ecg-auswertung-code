"""Every classical criterion reaches the lab with a readable German label."""
from vcg_biomarker.gui.lab_marker_viewmodel import classical_criterion_label
from vcg_biomarker.mimic.panels import ALL_CLASSICAL_FEATURE_IDS


def test_every_classical_criterion_has_a_german_label():
    for feature_id in ALL_CLASSICAL_FEATURE_IDS:
        label = classical_criterion_label(feature_id)
        assert label and "_" not in label, feature_id
        # The generic fallback humanizes the English identifier.
        assert not label.lower().startswith("classical"), (feature_id, label)
    assert classical_criterion_label("romhilt_estes_points") == "Romhilt-Estes-Punkte"
    assert classical_criterion_label("classical_sokolow_lyon") == classical_criterion_label("sokolow_lyon")
