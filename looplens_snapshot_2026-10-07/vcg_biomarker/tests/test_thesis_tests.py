"""The fixed thesis versions: known effects are found, every thesis runs."""
from __future__ import annotations

import functools
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd
import pytest

from vcg_biomarker.mimic import thesis_statistics as ts
from vcg_biomarker.mimic import thesis_tests as tt
from vcg_biomarker.mimic.thesis_data import ThesisData

RNG_SEED = 11


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    monkeypatch.setattr(tt, "directional_result", functools.partial(ts.directional_result, iterations=60))
    monkeypatch.setattr(tt, "cluster_bootstrap", functools.partial(ts.cluster_bootstrap, iterations=60))


class FakeData:
    """Stands in for ThesisData with frames built by each test."""

    def __init__(self, base, tables=None, **named):
        self.base_frame = base
        self.tables = {**(tables or {}), **named}
        self.connection = self.tables.get("connection")
        self.config = self.tables.get("config")

    def base(self, markers):
        return self.base_frame.copy()

    def context(self, columns):
        frame = self.tables.get("context", self.base_frame[["subject_id", "study_id"]]).copy()
        frame.attrs["missing_columns"] = [c for c in columns if c not in frame]
        return frame

    def labels(self, label_ids):
        return self.tables.get("labels", self.base_frame[["subject_id", "study_id"]]).copy()

    def nearest_lab(self, group, window, *, itemids=None, statuses=("usable",)):
        frame = self.tables.get(("lab", group), pd.DataFrame())
        if frame.empty:
            return frame
        return frame.loc[frame["value_status"].isin(statuses)].copy()

    def nearest_chart(self, itemids, window, low, high, *, fahrenheit_itemids=()):
        return self.tables.get(("chart", tuple(itemids)), pd.DataFrame()).copy()

    def infusion_running(self, groups):
        return self.tables.get("infusions", pd.DataFrame()).copy()

    def administrations(self, drugs):
        frame = self.tables.get("administrations", pd.DataFrame())
        return frame.loc[frame["drug"].isin(drugs)].copy() if not frame.empty else frame

    def hadm_with_codes(self, icd9, icd10, *, table="diagnoses_icd"):
        return self.tables.get(("codes", tuple(icd10)), set())

    def procedure_events(self, icd9, icd10):
        return self.tables.get(("procedures", tuple(icd10)), pd.DataFrame(columns=["subject_id", "hadm_id", "chartdate"]))

    def admissions(self):
        return self.tables.get("admissions", pd.DataFrame(columns=["subject_id", "hadm_id", "admittime", "dischtime"]))

    def reference_ranges(self, run_root):
        return self.tables.get("ranges", pd.DataFrame())


def _base(n, rng, **columns):
    subjects = np.arange(1, n + 1)
    frame = pd.DataFrame({
        "subject_id": subjects,
        "study_id": subjects * 10,
        "split": np.where(subjects % 2 == 0, "train", "validation"),
        "ecg_time": pd.Timestamp("2150-01-01 08:00") + pd.to_timedelta(rng.integers(0, 1000, n), unit="h"),
        "hadm_id": subjects * 100,
        "icustay_id": subjects * 1000,
        "age": rng.uniform(30, 90, n),
        "sex": rng.choice(["M", "F"], n),
        "heart_rate_bpm": rng.normal(75, 12, n),
        "machine_report": "Sinus rhythm",
        "report_af_state": "negative",
        "report_lbbb_state": "negative",
        "report_pacemaker_state": "negative",
    })
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_t10_finds_the_gradient_compensation():
    rng = np.random.default_rng(RNG_SEED)
    n = 400
    coupling = rng.uniform(0.5, 1.0, n)
    ratio = np.clip(0.9 - 0.8 * coupling + rng.normal(0, 0.05, n), 0.01, None)
    base = _base(n, rng, secondary_morphology_svg_to_qrs_area_delta_ratio=ratio,
                 secondary_morphology_t_qrs_area_delta_cosine=rng.normal(-0.8, 0.1, n),
                 secondary_morphology_mean_coupling_ratio=coupling)
    results = {r.component: r for r in tt.thesis_t10(FakeData(base))}
    assert results["median_R"].verdict == ts.CONFIRMED
    assert results["median_cosine"].verdict == ts.CONFIRMED
    assert results["R_vs_coupling"].verdict == ts.CONFIRMED


def test_t10_is_refuted_when_the_gradient_follows_the_qrs():
    rng = np.random.default_rng(RNG_SEED)
    n = 400
    base = _base(n, rng, secondary_morphology_svg_to_qrs_area_delta_ratio=rng.normal(1.4, 0.1, n),
                 secondary_morphology_t_qrs_area_delta_cosine=rng.normal(0.2, 0.1, n),
                 secondary_morphology_mean_coupling_ratio=rng.uniform(0.5, 1.0, n))
    results = {r.component: r for r in tt.thesis_t10(FakeData(base))}
    assert results["median_R"].verdict == ts.REFUTED


def _potassium_frame(rng):
    n = 900
    group = np.repeat(["nonhemolytic_high", "hemolytic_high", "normal"], n // 3)
    k = np.where(group == "normal", rng.uniform(3.6, 4.9, n), rng.uniform(5.6, 7.5, n))
    base = _base(n, rng)
    base["t_vector_magnitude"] = rng.normal(0.4, 0.1, n) + np.where(group == "nonhemolytic_high", 0.15, 0.0)
    base["tpeak_tend_ms"] = rng.normal(90, 10, n) - np.where(group == "nonhemolytic_high", 12, 0.0)
    labs = pd.DataFrame({
        "subject_id": base["subject_id"], "study_id": base["study_id"], "value_numeric": k,
        "value_status": np.where(group == "hemolytic_high", "hemolysis_reported", "usable"),
    })
    return base, labs


def test_t4_signature_only_in_non_haemolysed_potassium():
    base, labs = _potassium_frame(np.random.default_rng(RNG_SEED))
    results = {r.component: r for r in tt.thesis_t4(FakeData(base, {("lab", "potassium"): labs}))}
    for marker in ("t_vector_magnitude", "tpeak_tend_ms"):
        assert results[f"{marker}_nonhemolytic_auc"].verdict == ts.CONFIRMED
        assert results[f"{marker}_nonhemolytic_minus_hemolytic"].verdict == ts.CONFIRMED


def test_t5_breakpoint_order():
    rng = np.random.default_rng(RNG_SEED)
    n = 3000
    k = rng.uniform(3.8, 8.0, n)
    base = _base(n, rng)
    base["t_vector_magnitude"] = 0.4 + 0.3 * np.maximum(k - 5.5, 0) + rng.normal(0, 0.02, n)
    base["p_vector_magnitude"] = 0.1 - 0.05 * np.maximum(k - 6.5, 0) + rng.normal(0, 0.005, n)
    base["qrs_duration_ms"] = 90 + 25 * np.maximum(k - 7.3, 0) + rng.normal(0, 2, n)
    labs = pd.DataFrame({"subject_id": base["subject_id"], "study_id": base["study_id"],
                         "value_numeric": k, "value_status": "usable"})
    result = tt.thesis_t5(FakeData(base, {("lab", "potassium"): labs}), iterations=40)[0]
    assert result.extra["k_star_t"] < result.extra["k_star_p"] < result.extra["k_star_qrs"]
    assert abs(result.extra["k_star_t"] - 5.5) <= 0.2 and abs(result.extra["k_star_qrs"] - 7.3) <= 0.2
    assert result.verdict == ts.CONFIRMED


def test_t12_night_prolongation_at_equal_rate():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for stay in range(1, 301):
        for hour, shift in ((3, 12.0), (14, 0.0)):
            rows.append({"subject_id": stay, "study_id": stay * 10 + hour, "hadm_id": stay * 100,
                         "split": "train" if stay % 2 else "validation",
                         "ecg_time": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=hour),
                         "heart_rate_bpm": 70.0, "qt_interval_ms": 400 + shift + rng.normal(0, 8),
                         "report_pacemaker_state": "negative", "report_lbbb_state": "negative"})
    base = pd.DataFrame(rows)
    context = base[["subject_id", "study_id"]].assign(med_qt_prolonging_drug=False)
    result = tt.thesis_t12(FakeData(base, context=context))[0]
    assert result.verdict == ts.CONFIRMED and abs(result.estimate - 12.0) < 2.0


def test_t14_takotsubo_apex_and_sah_between():
    rng = np.random.default_rng(RNG_SEED)
    n = 600
    base = _base(n, rng)
    group = np.repeat(["takotsubo", "lad_stemi", "sah"], n // 3)
    base["ventricular_gradient_apicobasal_uv_ms"] = np.select(
        [group == "takotsubo", group == "lad_stemi"], [-50.0, 20.0], -40.0) + rng.normal(0, 10, n)
    codes = {("codes", ("I5181",)): set(base.loc[group == "takotsubo", "hadm_id"]),
             ("codes", ("I210",)): set(base.loc[group == "lad_stemi", "hadm_id"]),
             ("codes", ("I60",)): set(base.loc[group == "sah", "hadm_id"])}
    results = {r.component: r for r in tt.thesis_t14(FakeData(base, codes))}
    assert results["apicobasal_takotsubo_minus_lad"].verdict == ts.CONFIRMED
    assert results["sah_closer_to_takotsubo"].verdict == ts.CONFIRMED


def test_v1_criterion_at_the_standards_specificity():
    rng = np.random.default_rng(RNG_SEED)
    n = 2000
    base = _base(n, rng)
    case = rng.uniform(size=n) < 0.3
    base["qrs_initial_30ms_force_azimuth_deg"] = np.where(case, rng.normal(-60, 15, n), rng.normal(20, 15, n))
    base["qrs_initial_30ms_force_elevation_deg"] = rng.normal(30, 10, n)
    base["classical_pathological_q_waves_positive"] = np.where(case, rng.uniform(size=n) < 0.3,
                                                               rng.uniform(size=n) < 0.05)
    labels = base[["subject_id", "study_id"]].assign(
        old_mi_icd_state=np.where(case, "positive", "negative"),
        mi_or_revascularization_history_state=np.where(case, "positive", "negative"))
    result = tt.thesis_v1(FakeData(base, labels=labels))[0]
    assert result.verdict == ts.CONFIRMED
    assert result.estimate > 0.3
    # The VCG threshold was set to the Q-wave criterion's train specificity.
    assert abs(result.extra["train_specificity_target"] - 0.95) < 0.02


def test_v9_prognostic_increment_on_validation():
    rng = np.random.default_rng(RNG_SEED)
    n = 3000
    base = _base(n, rng)
    marker = rng.normal(0, 1, n)
    risk = 1 / (1 + np.exp(-(-2.0 + 1.5 * marker)))
    event = rng.uniform(size=n) < risk
    base["p_loop_terminal_force_vector"] = marker
    for name in ("p_loop_terminal_reversal_deg", "p_loop_terminal_area_fraction", "interatrial_block_vector_score"):
        base[name] = rng.normal(0, 1, n)
    base["classical_p_duration_ge_120"] = rng.normal(110, 10, n)
    labels = base[["subject_id", "study_id"]].assign(new_onset_af_1y_state=np.where(event, "positive", "negative"))
    context = base[["subject_id", "study_id"]].assign(dx_heart_failure=False)
    result = tt.thesis_v9(FakeData(base, labels=labels, context=context))[0]
    assert result.verdict == ts.CONFIRMED and result.estimate > 0.1


def test_every_thesis_runs_on_generic_data(tmp_path):
    rng = np.random.default_rng(RNG_SEED)
    n = 400
    base = _base(n, rng, **{name: rng.normal(1, 0.3, n) for name in tt.THESIS_MARKERS})
    base["report_pacemaker_state"] = np.where(np.arange(n) % 7 == 0, "positive", "negative")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    base[["subject_id", "study_id"]].assign(report_rbbb_state="positive").to_parquet(
        evidence / "machine_report_labels.parquet")
    labs = base[["subject_id", "study_id"]].assign(value_numeric=rng.uniform(3, 8, n), value_status="usable",
                                                    specimen_id=base["subject_id"])
    chart = base[["subject_id", "study_id"]].assign(value=rng.uniform(34, 38, n), delta_hours=0.0)
    tables = {("lab", group): labs for group in ("potassium", "ionized_calcium", "tsh", "hemoglobin")}
    tables.update({("chart", ids): chart for ids in ((224639,), (226512,), (223762, 226329), (224685,),
                                                    (220045,), (220050, 220179), (220277,))})
    data = FakeData(base, tables, connection=duckdb.connect(), config=SimpleNamespace(evidence_root=evidence))
    for thesis_id, function in tt.THESIS_FUNCTIONS.items():
        results = function(data, tmp_path) if thesis_id == "V4" else function(data)
        assert results, thesis_id
        assert all(isinstance(item, ts.ThesisResult) for item in results), thesis_id
        assert all(item.verdict in {ts.CONFIRMED, ts.REFUTED, ts.UNDECIDED, ts.NOT_EVALUABLE}
                   for item in results), thesis_id


def _sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def test_thesis_data_nearest_lab_chart_and_administrations(tmp_path, monkeypatch):
    t = pd.Timestamp("2150-01-01 12:00")
    master_root, evidence_root = tmp_path / "master", tmp_path / "evidence"
    master_root.mkdir(); evidence_root.mkdir()
    pd.DataFrame({"subject_id": [1, 2, 3], "study_id": [10, 20, 30], "split": ["train", "validation", "test"],
                  "ecg_time": [t, t, t], "hadm_id": [100, 200, 300], "icustay_id": [1000, None, 3000]}
                 ).to_parquet(master_root / "ecg_master_index.parquet")
    pd.DataFrame({"labevent_id": [1, 2, 3, 4], "subject_id": [1, 1, 2, 3], "itemid": [50971] * 4,
                  "lab_group": ["potassium"] * 4, "charttime": [t - pd.Timedelta(hours=2), t + pd.Timedelta(hours=1),
                                                             t + pd.Timedelta(hours=7), t],
                  "value_numeric": [4.0, 5.9, 6.0, 7.0], "value_status": ["usable"] * 4,
                  "specimen_id": [11, 12, 21, 31], "hemolysis_state": [None] * 4}
                 ).to_parquet(evidence_root / "labs.parquet")
    chart = tmp_path / "chart.parquet"
    pd.DataFrame({"subject_id": [1, 1, 2], "itemid": [223761, 223762, 223762],
                  "charttime": [t, t + pd.Timedelta(minutes=50), t], "valuenum": [98.6, 36.0, 60.0]}).to_parquet(chart)
    emar = tmp_path / "emar.parquet"
    pd.DataFrame({"subject_id": [1, 1, 3], "hadm_id": [100, 100, 300], "emar_id": ["a", "b", "c"],
                  "charttime": [t, t + pd.Timedelta(hours=12), t], "medication": ["Dofetilide", "Dofetilide", "Dofetilide"],
                  "event_txt": ["Administered", "Not Given", "Administered"]}).to_parquet(emar)
    relations = {"chartevents": f"read_parquet('{_sql(chart)}')", "emar": f"read_parquet('{_sql(emar)}')",
                 "emar_detail": None, "inputevents": None}
    config = SimpleNamespace(master_root=master_root, evidence_root=evidence_root)
    data = ThesisData(config, str(tmp_path / "none" / "*.parquet"), duckdb.connect())
    monkeypatch.setattr(data, "event_relation", lambda table: relations.get(table))
    labs = data.nearest_lab("potassium", 6).set_index("subject_id")
    assert labs.loc[1, "value_numeric"] == 5.9        # 1 h beats 2 h
    assert 2 not in labs.index and 3 not in labs.index  # 7 h away; locked test
    temps = data.nearest_chart((223762,), 1, 25, 45, fahrenheit_itemids=(223761,)).set_index("subject_id")
    assert abs(temps.loc[1, "value"] - 37.0) < 1e-9   # 98.6 F at 0 h beats 36 C at 50 min
    assert 2 not in temps.index                        # 60 C is implausible
    doses = data.administrations(("dofetilide",))
    assert list(doses["subject_id"]) == [1] and doses["drug"].iloc[0] == "dofetilide"
