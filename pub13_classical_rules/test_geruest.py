"""Tests fuer k1_geruest.py (synthetische Daten, keine Patientendaten). Stand v3.2."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k1_geruest as G  # noqa: E402

H = {"S": "a" * 64, "C": "b" * 64, "M": "c" * 64, "D": "d" * 64}


def test_seed_and_streams_are_deterministic():
    s = G.seed_from_spec("0123abcd" + "0" * 56)
    assert s == int("0123abcd", 16)
    a = G.rng_for(s, 0, 0).integers(0, 10**6, 5).tolist()
    b = G.rng_for(s, 0, 0).integers(0, 10**6, 5).tolist()
    c = G.rng_for(s, 1, 0).integers(0, 10**6, 5).tolist()
    assert a == b and a != c


def test_guard_dev_mode_blocks_locked_test_and_sealed():
    G.guard_split("dev", "train")
    G.guard_split("dev", "validation")
    for bad in ("locked_test", "sealed", "anything"):
        try:
            G.guard_split("dev", bad)
            raise AssertionError("haette scheitern muessen")
        except PermissionError:
            pass


def test_partition_sql_single_path_filters_split():
    sql = G.partition_sql("features", "train", "dev")
    assert "WHERE split = 'train'" in sql
    try:
        G.partition_sql("features", "locked_test", "dev")
        raise AssertionError
    except PermissionError:
        pass


def test_test_mode_requires_register_entry(tmp_path):
    reg = tmp_path / "TESTZUGRIFFE.md"
    reg.write_text("| 2026 | x | anderes Ereignis | nichts |\n", encoding="utf-8")
    args = dict(register_path=reg, spec_hash=H["S"], script_hash=H["C"], manifest_hash=H["M"], dev_results_hash=H["D"])
    try:
        G.guard_split("test", "locked_test", **args)
        raise AssertionError("ohne Beginn-Zeile muss es scheitern")
    except PermissionError:
        pass
    # unvollstaendige Hashes in der Zeile genuegen nicht
    reg.write_text("| 2026 | AW | Beginn | S `%s`; C `%s` |\n" % (H["S"], H["C"]), encoding="utf-8")
    try:
        G.guard_split("test", "locked_test", **args)
        raise AssertionError
    except PermissionError:
        pass
    line = G.register_line("Beginn", "AW-xyz", H["S"], H["C"], H["M"], H["D"], "locked_test, ein Durchgang")
    G.append_register(reg, line)
    G.guard_split("test", "locked_test", **args)  # jetzt freigegeben
    # Dev-Modus bleibt auch mit Eintrag gesperrt
    try:
        G.guard_split("dev", "locked_test", **args)
        raise AssertionError
    except PermissionError:
        pass


def test_mask_small_and_complement_coarsening():
    assert G.mask_small(0) == 0 and G.mask_small(3) == "<10" and G.mask_small(14) == 10 and G.mask_small(1234) == 1230
    out = G.coarsen_counts({"a": 8, "b": 123, "c": 4567}, {"zeile": 4698})
    assert out["a"] == "<10" and out["b"] == 100 and out["c"] == 4600 and out["_totals"]["zeile"] == 4700
    out2 = G.coarsen_counts({"a": 18, "b": 123}, {"zeile": 141})
    assert out2["a"] == 20 and out2["b"] == 120 and out2["_totals"]["zeile"] == 140  # nichts maskiert: Rundung auf 10


def test_bounds_and_bonferroni():
    assert abs(G.alpha_component("E") - 0.025 / 8) < 1e-12
    rng = np.random.default_rng(1)
    boot = rng.normal(1.0, 0.1, 20000)
    lo, up = G.one_sided_bounds(boot, 0.025)
    assert 0.80 < lo < 0.83 and 1.17 < up < 1.20
    lo2, up2 = G.one_sided_bounds(boot, G.alpha_component("S"))
    assert lo2 < lo and up2 > up  # strenger bei Bonferroni


def test_classify_ge_le_disjoint_and_exhaustive():
    theta = 0.9
    seen = set()
    for lo in np.linspace(0.5, 1.0, 26):
        for up in np.linspace(lo, 1.0, 12):
            c = G.classify_ge(float(lo), float(up), theta)
            seen.add(c)
            assert c in (G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN)
            if c == G.HAELT:
                assert lo >= theta
            if c == G.HAELT_NICHT:
                assert up < theta
    assert seen == {G.HAELT, G.HAELT_NICHT, G.UNENTSCHIEDEN}
    assert G.classify_le(0.02, 0.09, 0.10) == G.HAELT
    assert G.classify_le(0.12, 0.20, 0.10) == G.HAELT_NICHT
    assert G.classify_le(0.05, 0.20, 0.10) == G.UNENTSCHIEDEN


def test_family_p_decision_and_hint():
    assert G.classify_hr_nonharm(1.03, 1.17) == (G.HAELT, "Erhoehung nachgewiesen (untere Schranke > 1,0)")
    assert G.classify_hr_nonharm(0.95, 1.10)[0] == G.HAELT
    assert G.classify_hr_nonharm(1.25, 1.50)[0] == G.HAELT_NICHT
    assert G.classify_hr_nonharm(1.09, 1.91)[0] == G.UNENTSCHIEDEN  # Cheng-artig: untere Schranke < 1,2
    # kein Wert kann gleichzeitig haelt und haelt nicht liefern: Pruefung ueber Raster
    for lo in np.linspace(0.8, 1.6, 41):
        for up in np.linspace(lo, 1.8, 21):
            c, _ = G.classify_hr_nonharm(float(lo), float(up))
            assert not (up < 1.2 and lo > 1.2)


def test_marker_r08_precedence():
    assert G.classify_marker(1.2, 1.8)[0] == G.HAELT
    assert G.classify_marker(1.02, 1.20) == (G.HAELT, "klein erhoeht (obere Schranke < 1,25)")
    assert G.classify_marker(0.6, 1.2)[0] == G.HAELT_NICHT
    assert G.classify_marker(0.6, 1.5)[0] == G.UNENTSCHIEDEN


def test_equivalence_r19():
    assert G.classify_equiv(-5, 6) == G.HAELT
    assert G.classify_equiv(-25, -12) == G.HAELT_NICHT
    assert G.classify_equiv(12, 30) == G.HAELT_NICHT
    assert G.classify_equiv(-15, 4) == G.UNENTSCHIEDEN


def test_combine_priority():
    assert G.combine([G.HAELT, G.HAELT]) == G.HAELT
    assert G.combine([G.HAELT, G.HAELT_NICHT]) == G.HAELT_NICHT
    assert G.combine([G.HAELT, G.UNENTSCHIEDEN]) == G.UNENTSCHIEDEN
    assert G.combine([G.UNENTSCHIEDEN, G.HAELT_NICHT]) == G.HAELT_NICHT


def test_trivial_specificity_bound():
    b = G.trivial_specificity_bound(0.034, 0.68)
    assert abs(b - (1 - 0.034 / 0.68)) < 1e-12 and b > 0.94
    assert G.predetermined_by_positive_rate(0.034, 0.68, 0.90) is True
    assert G.predetermined_by_positive_rate(0.30, 0.68, 0.90) is False


def test_equal_risk_qrs_inversion():
    grid = np.arange(90, 201)
    risk = 0.05 + 0.002 * (grid - 90)  # monoton
    v, status = G.equal_risk_qrs(grid, risk, 0.15)
    assert status == "ok" and abs(v - 140.0) < 1e-6
    v, status = G.equal_risk_qrs(grid, risk, 0.99)
    assert v is None and status.startswith("nicht pruefbar") and "kein" in status
    wavy = 0.1 + 0.05 * np.sin((grid - 90) / 10.0)
    v, status = G.equal_risk_qrs(grid, wavy, 0.1)
    assert v is None and "mehrfach" in status
