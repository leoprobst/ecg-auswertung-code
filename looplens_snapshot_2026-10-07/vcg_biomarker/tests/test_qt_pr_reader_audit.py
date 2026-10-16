from types import SimpleNamespace

from vcg_biomarker.mimic.qt_pr_reader_audit import annotated_pr


def test_qrs_peak_symbol_overrides_num_zero_and_t_is_not_p():
    annotation = SimpleNamespace(sample=[10, 20, 30, 50, 60, 70, 100, 120],
        symbol=["(", "p", ")", "(", "N", ")", "t", ")"],
        num=[0, 0, 0, 1, 0, 1, 0, 2])
    assert annotated_pr(annotation, 250) == [(60, 160.)]


def test_reader_without_p_waves_cannot_supply_pr_reference():
    annotation = SimpleNamespace(sample=[50, 60, 70, 100, 120],
        symbol=["(", "N", ")", "t", ")"], num=[1, 0, 1, 0, 2])
    assert annotated_pr(annotation, 250) == []
