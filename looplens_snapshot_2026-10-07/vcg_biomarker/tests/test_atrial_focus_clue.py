import pytest

from vcg_biomarker.gui.atrial_focus_clue import focal_at_v1_clue


@pytest.mark.parametrize("pattern, expected", [
    ("negative", "rechtsatrial"),
    ("positive_negative", "rechtsatrial"),
    ("positive", "linksatrial"),
    ("negative_positive", "paraseptale"),
    ("isoelectric_positive", "paraseptale"),
    ("isoelectric", "paraseptale"),
])
def test_v1_focal_at_clue_is_broad_and_source_linked(pattern, expected):
    result = focal_at_v1_clue(focal_at_confirmed=True, p_wave_unobscured=True,
                              p_quality_accepted=True, v1_pattern=pattern)
    assert result.status == "clue" and expected in result.region_hint
    assert result.probability is None and len(result.sources) == 2


def test_v1_clue_abstains_without_focal_at_and_quality():
    arguments = dict(focal_at_confirmed=True, p_wave_unobscured=True,
                     p_quality_accepted=True, v1_pattern="positive")
    for key in ("focal_at_confirmed", "p_wave_unobscured", "p_quality_accepted"):
        assert focal_at_v1_clue(**(arguments | {key: False})).status == "abstained"
    assert focal_at_v1_clue(**(arguments | {"v1_pattern": "unclear"})).status == "abstained"
