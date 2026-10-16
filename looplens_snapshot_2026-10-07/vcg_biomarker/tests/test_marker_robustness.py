import pandas as pd
from vcg_biomarker.mimic.marker_robustness import summarize, PR, PRE, COUNT, FRECHET, MAD, FCOUNT


def test_matched_denominators_and_six_beat_sensitivity():
    data = pd.DataFrame({"rhythm_context": ["af"]*3, "signal_quality_tier": ["good"]*3,
        PR: [10., None, 30.], PRE: [15., 20., 35.], COUNT: [3, 6, 6],
        FRECHET: [1., 2., 3.], MAD: [0.1, 0.2, 0.3], FCOUNT: [3, 6, 6]})
    report = summarize(data)
    first, second = report["records"]
    assert first["direct8_qualified"] == 3 and first["paired_n"] == 2
    assert second["direct8_qualified"] == 2 and second["paired_n"] == 1
    assert second["paired_pre_minus_pr_median_deg"] == 5
    assert not report["mad_is_measurement_ci"]
    data.loc[0, FCOUNT] = float("nan")
    assert summarize(data)["inconsistent_frechet_counts"] == 1


def test_missing_columns_are_not_a_clean_audit():
    assert summarize(pd.DataFrame())["status"] == "unavailable"
