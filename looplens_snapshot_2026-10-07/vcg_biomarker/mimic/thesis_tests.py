"""The fixed thesis versions T1-T18 and V1-V10 (``THESEN_FASSUNGEN_2026-09-24.md``).

One function per thesis; each returns one ``ThesisResult`` per component.
Every choice below -- unit, window, pairing, covariates, statistic, verdict
rule -- is the one written in the versions document before any value was
computed. Where a version needs a source the pipeline does not produce, the
component says so (``nicht_auswertbar`` with the reason) instead of
substituting another quantity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .thesis_data import ThesisData
from .thesis_statistics import (
    BOOTSTRAP_ITERATIONS,
    CONFIRMED,
    MINIMUM_UNITS,
    NOT_EVALUABLE,
    REFUTED,
    UNDECIDED,
    ThesisResult,
    angle_between,
    auc,
    cluster_bootstrap,
    deming_slope,
    directional_result,
    fold_angle,
    mcnemar_p,
    not_evaluable,
    ols_coefficient,
    paired_sensitivity_difference,
    spearman,
    threshold_at_specificity,
)

THESIS_TESTS_VERSION = "1.0.0"
FASSUNGEN_DOCUMENT = "THESEN_FASSUNGEN_2026-09-24.md"

# Every marker any version reads, loaded once.
THESIS_MARKERS: tuple[str, ...] = (
    "p_vector_magnitude", "qrs_vector_magnitude", "t_vector_magnitude",
    "spatial_qrs_t_angle_deg", "svg_azimuth", "svg_elevation", "svg_magnitude",
    "qrs_duration_ms", "qt_interval_ms", "pr_interval_ms", "qtc_fridericia_ms",
    "v6_r_peak_time_ms", "st_segment_duration_ms", "tpeak_tend_ms",
    "j_tpeak_ms", "j_tpeak_c_ms", "osborn_j_wave_amplitude_uv",
    "secondary_morphology_svg_to_qrs_area_delta_ratio",
    "secondary_morphology_t_qrs_area_delta_cosine",
    "secondary_morphology_mean_coupling_ratio",
    "t_area_vector_x_mv_ms", "t_area_vector_y_mv_ms", "t_area_vector_z_mv_ms",
    "qrs_area_vector_x_mv_ms", "qrs_area_vector_y_mv_ms", "qrs_area_vector_z_mv_ms",
    "qrs_area_vector_magnitude_mv_ms",
    "st_area_vector_x_mv_ms", "st_area_vector_y_mv_ms", "st_area_vector_z_mv_ms",
    "respiratory_axis_modulation_deg", "respiratory_cycles_per_record",
    "ventricular_gradient_apicobasal_uv_ms",
    "pr_segment_to_st_vector_angle_deg", "pr_segment_to_p_vector_ratio",
    "tq_atrial_rms",
    "classical_qtc_bazett", "classical_cornell_voltage",
    "classical_pathological_q_waves_positive",
    "classical_peaked_t_wave_heuristic_positive",
    "classical_st_depression_contiguous_positive",
    "classical_p_duration_ge_120", "classical_pr_prolongation_gt_200",
    "classical_left_axis_deviation_positive",
    "qrs_initial_30ms_force_azimuth_deg", "qrs_initial_30ms_force_elevation_deg",
    "t_loop_compactness", "t_loop_apex_sharpness", "t_loop_max_min_ratio",
    "t_loop_vector_reversal_index", "qrs_alternans_vector",
    "qrs_longest_slow_run_ms", "qrs_velocity_max",
    "p_loop_terminal_force_vector", "p_loop_terminal_reversal_deg",
    "p_loop_terminal_area_fraction", "interatrial_block_vector_score",
)

# Kors VCG axes: the project's aVR direction for demand ischaemia (T16),
# THESEN_AKUTE_ISCHAEMIE_2026-09-12.md.
AVR_DIRECTION = np.array([-0.30, -0.23, -0.25])

CATECHOLAMINE_ITEMIDS = {
    "epinephrine": (221289,),
    "norepinephrine": (221906,),
    "phenylephrine": (221749,),
    "vasopressin": (222315,),
    "dopamine": (221662,),
}


# --------------------------------------------------------------- helpers


def _positive(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().eq("positive").fillna(False)


def _without_paced_or_lbbb(frame: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    for column in ("report_pacemaker_state", "report_lbbb_state"):
        if column in frame:
            keep &= ~_positive(frame[column])
    return frame.loc[keep]


def _flag(frame: pd.DataFrame, column: str) -> pd.Series:
    """A boolean context column; absent or missing counts as False."""

    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].astype("boolean").fillna(False).astype(bool)


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].astype("string").str.lower().fillna("")


def _male(frame: pd.DataFrame) -> pd.Series:
    return frame["sex"].astype("string").str.upper().eq("M").astype(float)


def _first_per_subject(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    order = [column for column in ("subject_id", "ecg_time", "study_id") if column in frame]
    return frame.sort_values(order, kind="stable").drop_duplicates("subject_id", keep="first")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _missing(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column not in frame or _numeric(frame, column).notna().sum() == 0]


def _merge_value(frame: pd.DataFrame, values: pd.DataFrame, name: str) -> pd.DataFrame:
    if values is None or values.empty:
        return frame.assign(**{name: np.nan})
    renamed = values[["subject_id", "study_id", "value" if "value" in values else "value_numeric"]].rename(
        columns={"value": name, "value_numeric": name}
    )
    return frame.merge(renamed, on=["subject_id", "study_id"], how="left")


def _null_prediction(result: ThesisResult) -> ThesisResult:
    """A version that predicts no effect: refuted when the interval excludes
    the null; otherwise compatible, which without a prespecified
    equivalence margin is not a confirmation."""

    if result.verdict in (CONFIRMED, UNDECIDED, REFUTED) and np.isfinite(result.ci_low):
        if result.ci_low > 0 or result.ci_high < 0:
            result.verdict = REFUTED
            result.reason = "interval excludes zero although no effect was predicted"
        else:
            result.verdict = UNDECIDED
            result.reason = "compatible with no effect; no equivalence margin was fixed"
    return result


def _covariates_hr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rate = _numeric(result, "heart_rate_bpm")
    result["hr"] = rate
    result["hr2"] = rate ** 2
    result["male"] = _male(result) if "sex" in result else np.nan
    return result


def _pair_first_last(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """First and last ECG of each group (by ECG time) side by side."""

    ordered = frame.sort_values([group, "ecg_time", "study_id"], kind="stable")
    counts = ordered.groupby(group)["study_id"].transform("size")
    ordered = ordered.loc[counts >= 2]
    first = ordered.groupby(group, sort=False).head(1).set_index(group)
    last = ordered.groupby(group, sort=False).tail(1).set_index(group)
    paired = first.join(last, lsuffix="_1", rsuffix="_2")
    paired["subject_id"] = paired["subject_id_1"]
    paired["split"] = paired["split_1"]
    return paired.reset_index()


# --------------------------------------------------------------- Teil A


def thesis_t1(data: ThesisData) -> list[ThesisResult]:
    tid = "T1"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    daily = data.nearest_chart((224639,), 24, 20, 400)
    admission = data.nearest_chart((226512,), 24, 20, 400)
    potassium = data.nearest_lab("potassium", 6)
    if daily.empty and admission.empty:
        return [not_evaluable(tid, "all", "", "no weight in chartevents")]
    weight = pd.concat([daily.assign(priority=0), admission.assign(priority=1)])
    weight = weight.sort_values(["subject_id", "study_id", "priority"]).drop_duplicates(["subject_id", "study_id"])
    frame = _merge_value(base, weight, "weight_kg")
    frame = _merge_value(frame, potassium, "potassium")
    frame = frame.dropna(subset=["weight_kg", "potassium", "hadm_id"])
    pairs = _pair_first_last(frame, "hadm_id")
    if pairs.empty:
        return [not_evaluable(tid, "all", "", "no stay with two weighed ECGs")]
    pairs["delta_weight"] = pairs["weight_kg_2"] - pairs["weight_kg_1"]
    pairs = pairs.loc[(pairs["potassium_2"] - pairs["potassium_1"]).abs() < 0.3]
    for name in ("p_vector_magnitude", "qrs_vector_magnitude", "t_vector_magnitude"):
        with np.errstate(divide="ignore", invalid="ignore"):
            pairs[f"dlog_{name}"] = np.log(_numeric(pairs, f"{name}_2")) - np.log(_numeric(pairs, f"{name}_1"))
    pairs["dlog_qrs_minus_p"] = pairs["dlog_qrs_vector_magnitude"] - pairs["dlog_p_vector_magnitude"]
    for name in ("spatial_qrs_t_angle_deg", "svg_azimuth", "svg_elevation"):
        pairs[f"d_{name}"] = fold_angle(_numeric(pairs, f"{name}_2") - _numeric(pairs, f"{name}_1"))
    for name in ("qrs_duration_ms", "qt_interval_ms"):
        pairs[f"d_{name}"] = _numeric(pairs, f"{name}_2") - _numeric(pairs, f"{name}_1")
    pairs = pairs.replace([np.inf, -np.inf], np.nan)

    def slope(outcome: str, exposure: str = "delta_weight") -> Callable[[pd.DataFrame], float]:
        return lambda f: ols_coefficient(f, outcome, exposure)

    results = [
        directional_result(tid, "a_qrs_amplitude_vs_weight", "slope < 0",
                           pairs.dropna(subset=["dlog_qrs_vector_magnitude"]),
                           slope("dlog_qrs_vector_magnitude"), direction=-1),
        _null_prediction(directional_result(tid, "a_qrs_minus_p_vs_weight", "slope = 0 (common factor)",
                                            pairs.dropna(subset=["dlog_qrs_minus_p"]),
                                            slope("dlog_qrs_minus_p"), direction=1)),
    ]
    for name in ("spatial_qrs_t_angle_deg", "svg_azimuth", "svg_elevation"):
        results.append(_null_prediction(directional_result(
            tid, f"b_{name}_vs_weight", "slope = 0 (scale-free)",
            pairs.dropna(subset=[f"d_{name}"]), slope(f"d_{name}"), direction=1)))
    results.append(directional_result(
        tid, "c_qrs_duration_vs_qrs_amplitude", "slope > 0",
        pairs.dropna(subset=["d_qrs_duration_ms", "dlog_qrs_vector_magnitude"]),
        slope("d_qrs_duration_ms", "dlog_qrs_vector_magnitude"), direction=1))
    results.append(directional_result(
        tid, "c_qt_vs_qrs_amplitude", "slope > 0 (descriptive)",
        pairs.dropna(subset=["d_qt_interval_ms", "dlog_qrs_vector_magnitude"]),
        slope("d_qt_interval_ms", "dlog_qrs_vector_magnitude"), direction=1))
    return results


def thesis_t2(data: ThesisData) -> list[ThesisResult]:
    tid = "T2"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    context = data.context([
        "has_tte_30d", "posterior_wall_thickness_cm", "lv_mass_devereux_g",
        "lv_mass_index_devereux_g_m2", "lv_geometry_ase", "history_old_mi_before_ecg",
    ])
    frame = base.merge(context, on=["subject_id", "study_id"], how="inner")
    frame = frame.loc[
        _flag(frame, "has_tte_30d")
        & ~_flag(frame, "history_old_mi_before_ecg")
        & (_numeric(frame, "qrs_duration_ms") < 120)
    ]
    frame = _first_per_subject(frame)
    results: list[ThesisResult] = []

    wall = frame.assign(wall_mm=_numeric(frame, "posterior_wall_thickness_cm") * 10.0)
    wall = wall.dropna(subset=["wall_mm", "v6_r_peak_time_ms"]) if "v6_r_peak_time_ms" in wall else wall.iloc[0:0]

    def deming(f: pd.DataFrame) -> float:
        x, y = f["wall_mm"].to_numpy(float), _numeric(f, "v6_r_peak_time_ms").to_numpy(float)
        sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
        if not (sx > 0 and sy > 0):
            return float("nan")
        return deming_slope(x / sx, y / sy, 1.0) * sy / sx

    def mechanism(estimate: float, low: float, high: float) -> bool:
        return (np.isfinite(high) and high < 0) or (np.isfinite(low) and low > 4.0)

    result = directional_result(tid, "i_r_peak_time_vs_wall", "slope in 1.7-4 ms/mm (A) or ~0 (B)",
                                wall, deming, direction=1, refuted=mechanism)
    if result.verdict not in (REFUTED, NOT_EVALUABLE):
        if 1.7 <= result.estimate <= 4.0 and result.ci_low > 0 and result.train_estimate > 0 and result.validation_estimate > 0:
            result.verdict, result.reason = CONFIRMED, "mechanism A (constant transmural velocity)"
        elif result.ci_low <= 0 <= result.ci_high:
            result.verdict, result.reason = CONFIRMED, "mechanism B (no relation)"
        else:
            result.verdict, result.reason = UNDECIDED, "between the two mechanisms"
    results.append(result)

    mediation = frame.assign(male=_male(frame), mass=_numeric(frame, "lv_mass_devereux_g"),
                             qrs=_numeric(frame, "qrs_duration_ms")).dropna(subset=["male", "mass", "qrs"])

    def mediated_share(f: pd.DataFrame) -> float:
        raw = f.loc[f["male"] == 1, "qrs"].mean() - f.loc[f["male"] == 0, "qrs"].mean()
        adjusted = ols_coefficient(f, "qrs", "male", ("mass",))
        return float(1.0 - adjusted / raw) if raw and np.isfinite(raw) else float("nan")

    results.append(directional_result(tid, "ii_sex_difference_mediated_by_mass", "share >= 0.8",
                                      mediation, mediated_share, direction=1, null=0.8))

    geometry = frame.loc[_text(frame, "lv_geometry_ase").isin(["concentric_hypertrophy", "eccentric_hypertrophy"])].copy()
    geometry["lvmi"] = _numeric(geometry, "lv_mass_index_devereux_g_m2")
    geometry = geometry.dropna(subset=["lvmi", "classical_cornell_voltage"]) if "classical_cornell_voltage" in geometry else geometry.iloc[0:0]
    if len(geometry) >= 3:
        geometry["tertile"] = pd.qcut(geometry["lvmi"], 3, labels=False, duplicates="drop")

    def within_tertile(f: pd.DataFrame) -> float:
        differences, weights = [], []
        for _, part in f.groupby("tertile"):
            concentric = _numeric(part.loc[part["lv_geometry_ase"].eq("concentric_hypertrophy")], "classical_cornell_voltage")
            eccentric = _numeric(part.loc[part["lv_geometry_ase"].eq("eccentric_hypertrophy")], "classical_cornell_voltage")
            if concentric.notna().any() and eccentric.notna().any():
                differences.append(concentric.mean() - eccentric.mean())
                weights.append(len(part))
        return float(np.average(differences, weights=weights)) if differences else float("nan")

    results.append(directional_result(tid, "iii_voltage_concentric_minus_eccentric", "> 0",
                                      geometry, within_tertile, direction=1))
    return results


def thesis_t3(data: ThesisData) -> list[ThesisResult]:
    tid = "T3"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    calcium = data.nearest_lab("ionized_calcium", 1, itemids=(50808,))
    potassium = data.nearest_lab("potassium", 1, itemids=(50822,))
    if calcium.empty or potassium.empty or calcium["specimen_id"].isna().all():
        return [not_evaluable(tid, "all", "", "ionised calcium, whole-blood potassium or specimen_id missing")]
    paired = calcium[["subject_id", "study_id", "value_numeric", "specimen_id"]].rename(
        columns={"value_numeric": "ica"}).merge(
        potassium[["subject_id", "study_id", "value_numeric", "specimen_id"]].rename(columns={"value_numeric": "k"}),
        on=["subject_id", "study_id", "specimen_id"], how="inner")
    frame = _first_per_subject(_covariates_hr(base.merge(paired, on=["subject_id", "study_id"])))
    covariates = ("k", "hr", "hr2", "age", "male")
    results = []
    for component, marker, exposure, direction, others in (
        ("st_duration_vs_ica", "st_segment_duration_ms", "ica", -1, ("k", "hr", "hr2", "age", "male")),
        ("tpe_vs_k", "tpeak_tend_ms", "k", -1, ("ica", "hr", "hr2", "age", "male")),
        ("t_magnitude_vs_k", "t_vector_magnitude", "k", 1, ("ica", "hr", "hr2", "age", "male")),
    ):
        subset = frame.dropna(subset=[marker, exposure, *others]) if marker in frame else frame.iloc[0:0]
        results.append(directional_result(
            tid, component, "< 0" if direction < 0 else "> 0", subset,
            lambda f, m=marker, e=exposure, o=others: ols_coefficient(f, m, e, o), direction=direction))

    def standardized(f: pd.DataFrame, marker: str, exposure: str) -> float:
        others = tuple(c for c in ("ica", "k", "hr", "hr2", "age", "male") if c != exposure)
        sd_y, sd_x = _numeric(f, marker).std(), _numeric(f, exposure).std()
        return abs(ols_coefficient(f, marker, exposure, others)) * sd_x / sd_y if sd_y else float("nan")

    complete = frame.dropna(subset=["st_segment_duration_ms", "tpeak_tend_ms", "ica", *covariates]) \
        if {"st_segment_duration_ms", "tpeak_tend_ms"} <= set(frame) else frame.iloc[0:0]
    results.append(directional_result(
        tid, "calcium_specific_to_st", "|b_iCa,ST| - |b_iCa,Tpe| > 0", complete,
        lambda f: standardized(f, "st_segment_duration_ms", "ica") - standardized(f, "tpeak_tend_ms", "ica"),
        direction=1))
    results.append(directional_result(
        tid, "potassium_specific_to_tpe", "|b_K,Tpe| - |b_K,ST| > 0", complete,
        lambda f: standardized(f, "tpeak_tend_ms", "k") - standardized(f, "st_segment_duration_ms", "k"),
        direction=1))
    return results


def _potassium_groups(data: ThesisData, base: pd.DataFrame) -> pd.DataFrame:
    potassium = data.nearest_lab("potassium", 6, itemids=(50971,), statuses=("usable", "hemolysis_reported"))
    if potassium.empty:
        return pd.DataFrame()
    frame = base.merge(potassium[["subject_id", "study_id", "value_numeric", "value_status"]],
                       on=["subject_id", "study_id"])
    k = frame["value_numeric"]
    usable = frame["value_status"].eq("usable")
    hemolysed = frame["value_status"].eq("hemolysis_reported")
    frame["group"] = np.select(
        [usable & (k > 5.5), hemolysed & (k > 5.5), usable & k.between(3.5, 5.0)],
        ["nonhemolytic_high", "hemolytic_high", "normal"], default="other")
    return frame.loc[frame["group"].ne("other")]


def thesis_t4(data: ThesisData) -> list[ThesisResult]:
    tid = "T4"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    frame = _potassium_groups(data, base)
    if frame.empty:
        return [not_evaluable(tid, "all", "", "no potassium 50971 near the ECG")]
    frame = frame.sort_values(["subject_id", "ecg_time"]).drop_duplicates(["subject_id", "group"])
    results = []
    for marker, direction in (("t_vector_magnitude", 1), ("tpeak_tend_ms", -1)):
        if marker not in frame:
            results.append(not_evaluable(tid, marker, "", f"{marker} not extracted"))
            continue

        def aligned(f: pd.DataFrame, group: str, m=marker, d=direction) -> float:
            part = f.loc[f["group"].isin([group, "normal"])]
            return auc(d * _numeric(part, m).to_numpy(), part["group"].eq(group).astype(int).to_numpy())

        nonhemolytic = frame.loc[frame["group"].isin(["nonhemolytic_high", "normal"])]
        results.append(directional_result(
            tid, f"{marker}_nonhemolytic_auc", "aligned AUC > 0.5", nonhemolytic,
            lambda f: aligned(f, "nonhemolytic_high"), direction=1, null=0.5,
            unit_count=int(nonhemolytic["group"].eq("nonhemolytic_high").sum())))

        def difference(f: pd.DataFrame) -> float:
            return aligned(f, "nonhemolytic_high") - aligned(f, "hemolytic_high")

        results.append(directional_result(
            tid, f"{marker}_nonhemolytic_minus_hemolytic", "> 0", frame, difference, direction=1,
            refuted=lambda est, low, high: np.isfinite(low) and low <= 0 <= high and est <= 0.01,
            unit_count=int(frame["group"].eq("hemolytic_high").sum())))
    return results


def _hinge_breakpoints(
    k: np.ndarray, y: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> float:
    """Breakpoint of y = a + b*k + c*max(k - k*, 0) with minimal weighted SSE.

    Solved through the 3x3 normal equations per grid point, so a bootstrap
    over tens of thousands of rows stays linear in the number of rows.
    """

    w = np.asarray(weights, dtype=float)
    k = np.asarray(k, dtype=float)
    y = np.asarray(y, dtype=float)
    sw, swk, swkk = w.sum(), (w * k).sum(), (w * k * k).sum()
    swy, swky, swyy = (w * y).sum(), (w * k * y).sum(), (w * y * y).sum()
    best, best_sse = float("nan"), np.inf
    for knot in grid:
        h = np.maximum(k - knot, 0.0)
        swh, swkh, swhh, swhy = (w * h).sum(), (w * k * h).sum(), (w * h * h).sum(), (w * h * y).sum()
        gram = np.array([[sw, swk, swh], [swk, swkk, swkh], [swh, swkh, swhh]])
        right = np.array([swy, swky, swhy])
        if np.linalg.cond(gram) > 1e12:
            continue
        beta = np.linalg.solve(gram, right)
        sse = float(swyy - 2 * beta @ right + beta @ gram @ beta)
        if sse < best_sse:
            best, best_sse = float(knot), sse
    return best


def thesis_t5(data: ThesisData, *, iterations: int = BOOTSTRAP_ITERATIONS) -> list[ThesisResult]:
    tid = "T5"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    frame = _potassium_groups(data, base)
    frame = frame.loc[frame["value_status"].eq("usable")] if not frame.empty else frame
    if frame.empty:
        return [not_evaluable(tid, "all", "", "no non-hemolytic potassium near the ECG")]
    grid = np.round(np.arange(4.5, 8.0 + 1e-9, 0.1), 1)
    sinus = frame.loc[~_positive(frame.get("report_af_state", pd.Series("", index=frame.index)))]
    parts = {
        "t": frame.dropna(subset=["t_vector_magnitude"]) if "t_vector_magnitude" in frame else frame.iloc[0:0],
        "p": sinus.dropna(subset=["p_vector_magnitude"]) if "p_vector_magnitude" in sinus else sinus.iloc[0:0],
        "qrs": frame.dropna(subset=["qrs_duration_ms"]) if "qrs_duration_ms" in frame else frame.iloc[0:0],
    }
    columns = {"t": "t_vector_magnitude", "p": "p_vector_magnitude", "qrs": "qrs_duration_ms"}
    if any(part.loc[part["value_numeric"] >= 6.5, "subject_id"].nunique() < MINIMUM_UNITS for part in parts.values()):
        return [not_evaluable(tid, "order", "K*(T) < K*(P) < K*(QRS)", "fewer than 30 persons with K >= 6.5 in a marker")]
    points = {
        name: _hinge_breakpoints(part["value_numeric"].to_numpy(float),
                                 _numeric(part, columns[name]).to_numpy(float),
                                 np.ones(len(part)), grid)
        for name, part in parts.items()
    }
    subjects = pd.unique(frame["subject_id"])
    rng = np.random.default_rng(20260924)
    draws = []
    for _ in range(int(iterations)):
        counts = pd.Series(rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects))), index=subjects)
        values = {}
        for name, part in parts.items():
            weights = part["subject_id"].map(counts).fillna(0).to_numpy(float)
            keep = weights > 0
            values[name] = _hinge_breakpoints(part["value_numeric"].to_numpy(float)[keep],
                                              _numeric(part, columns[name]).to_numpy(float)[keep],
                                              weights[keep], grid)
        draws.append(values)
    draws = pd.DataFrame(draws).dropna()
    share_ordered = float(((draws["t"] < draws["p"]) & (draws["p"] < draws["qrs"])).mean()) if len(draws) else float("nan")
    share_p_first = float((draws["p"] < draws["t"]).mean()) if len(draws) else float("nan")
    share_qrs_first = float((draws["qrs"] < draws["t"]).mean()) if len(draws) else float("nan")
    result = ThesisResult(tid, "order", "K*(T) < K*(P) < K*(QRS)", estimate=share_ordered,
                          n_units=int(len(frame)), n_subjects=int(frame["subject_id"].nunique()),
                          extra={"k_star_t": points["t"], "k_star_p": points["p"], "k_star_qrs": points["qrs"],
                                 "share_p_before_t": share_p_first, "share_qrs_before_t": share_qrs_first})
    if share_p_first >= 0.95 or share_qrs_first >= 0.95:
        result.verdict = REFUTED
    elif points["t"] < points["p"] < points["qrs"] and share_ordered >= 0.95:
        result.verdict = CONFIRMED
    else:
        result.verdict = UNDECIDED
    result.reason = "estimate = bootstrap share of draws in the predicted order"
    return [result]


def _first_administration_pairs(
    data: ThesisData, base: pd.DataFrame, drugs: Sequence[str],
    *, before_hours: float = 48, after_min: float = 2, after_max: float = 72,
) -> pd.DataFrame:
    """Last ECG <= ``before_hours`` before the first eMAR administration of a
    drug and the first ECG ``after_min``-``after_max`` h after it; the marker
    values of both ECGs as ``<column>_before`` / ``<column>_after``."""

    administrations = data.administrations(drugs)
    if administrations.empty:
        return pd.DataFrame()
    first = (
        administrations.sort_values(["subject_id", "drug", "charttime"])
        .drop_duplicates(["subject_id", "drug"])[["subject_id", "drug", "charttime", "hadm_id", "dose_given"]]
    )
    merged = first.merge(base[["subject_id", "study_id", "ecg_time"]], on="subject_id")
    merged["hours"] = (merged["ecg_time"] - merged["charttime"]).dt.total_seconds() / 3600.0
    before = (
        merged.loc[(merged["hours"] < 0) & (merged["hours"] >= -before_hours)]
        .sort_values(["subject_id", "drug", "hours"]).groupby(["subject_id", "drug"]).tail(1)
    )
    after = (
        merged.loc[(merged["hours"] >= after_min) & (merged["hours"] <= after_max)]
        .sort_values(["subject_id", "drug", "hours"]).groupby(["subject_id", "drug"]).head(1)
    )
    pairs = before[["subject_id", "drug", "charttime", "hadm_id", "dose_given", "study_id"]].merge(
        after[["subject_id", "drug", "study_id"]], on=["subject_id", "drug"], suffixes=("_before", "_after"))
    if pairs.empty:
        return pd.DataFrame()
    markers = base.drop_duplicates("study_id").set_index("study_id")
    for side in ("before", "after"):
        joined = markers.reindex(pairs[f"study_id_{side}"].to_numpy())
        for column in markers.columns:
            if column != "subject_id":
                pairs[f"{column}_{side}"] = joined[column].to_numpy()
    pairs["split"] = pairs["split_before"]
    return pairs


def thesis_t6(data: ThesisData) -> list[ThesisResult]:
    tid = "T6"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    pairs = _first_administration_pairs(data, base, ("dofetilide", "sotalol", "ranolazine", "mexiletine"))
    if pairs.empty:
        return [not_evaluable(tid, "all", "", "no first administrations with ECGs before and after")]
    for marker in ("j_tpeak_c_ms", "tpeak_tend_ms"):
        pairs[f"d_{marker}"] = _numeric(pairs, f"{marker}_after") - _numeric(pairs, f"{marker}_before")
    results: list[ThesisResult] = []
    for drug in ("dofetilide", "sotalol"):
        part = pairs.loc[pairs["drug"].eq(drug)]
        drug_results = [
            directional_result(tid, f"{drug}_delta_{marker}", "> 0", part.dropna(subset=[f"d_{marker}"]),
                               lambda f, m=marker: float(_numeric(f, f"d_{m}").mean()), direction=1)
            for marker in ("j_tpeak_c_ms", "tpeak_tend_ms")
        ]
        a, b = drug_results
        if all(np.isfinite([a.ci_low, a.ci_high, b.ci_low, b.ci_high])):
            one_sided = (a.ci_high <= 0 and b.ci_low > 0) or (b.ci_high <= 0 and a.ci_low > 0)
            if one_sided and drug == "dofetilide":
                for item in drug_results:
                    item.verdict, item.reason = REFUTED, "only one repolarisation component prolonged"
        results.extend(drug_results)
    for drug in ("ranolazine", "mexiletine"):
        part = pairs.loc[pairs["drug"].eq(drug)].dropna(subset=["d_j_tpeak_c_ms"])
        results.append(directional_result(tid, f"{drug}_delta_j_tpeak_c_ms", "<= 0 (exploratory)", part,
                                          lambda f: float(f["d_j_tpeak_c_ms"].mean()), direction=-1))
    dofetilide = pairs.loc[pairs["drug"].eq("dofetilide")]
    potassium = data.nearest_lab("potassium", 12)
    if not potassium.empty and not dofetilide.empty:
        k = potassium[["study_id", "value_numeric"]].rename(columns={"study_id": "study_id_before", "value_numeric": "k_before"})
        interaction = dofetilide.merge(k, on="study_id_before", how="left").dropna(subset=["k_before", "d_j_tpeak_c_ms"])
        results.append(directional_result(tid, "dofetilide_delta_vs_potassium", "slope < 0", interaction,
                                          lambda f: ols_coefficient(f, "d_j_tpeak_c_ms", "k_before"), direction=-1))
    else:
        results.append(not_evaluable(tid, "dofetilide_delta_vs_potassium", "slope < 0", "no potassium before the dose"))
    sexed = dofetilide.dropna(subset=["d_j_tpeak_c_ms"]).assign(
        male=lambda f: f["sex_before"].astype("string").str.upper().eq("M")) if "sex_before" in dofetilide else dofetilide.iloc[0:0]
    results.append(directional_result(
        tid, "dofetilide_female_minus_male", "> 0", sexed,
        lambda f: float(f.loc[~f["male"], "d_j_tpeak_c_ms"].mean() - f.loc[f["male"], "d_j_tpeak_c_ms"].mean()),
        direction=1))
    return results


def thesis_t7(data: ThesisData) -> list[ThesisResult]:
    tid = "T7"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    temperature = data.nearest_chart((223762, 226329), 1, 25, 45, fahrenheit_itemids=(223761,))
    potassium = data.nearest_lab("potassium", 6)
    if temperature.empty:
        return [not_evaluable(tid, "all", "", "no temperature in chartevents")]
    frame = _merge_value(base, temperature, "temperature")
    frame = _merge_value(frame, potassium, "potassium").dropna(subset=["temperature", "hadm_id"])
    pairs = []
    for hadm, stay in frame.groupby("hadm_id"):
        coldest = stay.sort_values(["temperature", "ecg_time"]).iloc[0]
        if coldest["temperature"] >= 36.0:
            continue
        normal = stay.loc[stay["temperature"].between(36.0, 37.5)]
        if normal.empty:
            continue
        gap = (normal["ecg_time"] - coldest["ecg_time"]).abs()
        warm = normal.loc[gap.idxmin()]
        pairs.append((coldest, warm))
    if not pairs:
        return [not_evaluable(tid, "all", "", "no stay with an ECG below 36 C and a normothermic ECG")]
    rows = []
    for cold, warm in pairs:
        row = {"subject_id": cold["subject_id"], "split": cold["split"],
               "delta_temperature": cold["temperature"] - warm["temperature"],
               "delta_hr": cold["heart_rate_bpm"] - warm["heart_rate_bpm"],
               "delta_k": cold["potassium"] - warm["potassium"],
               "cold_below_35": cold["temperature"] < 35.0}
        for marker in ("qt_interval_ms", "j_tpeak_c_ms", "qrs_duration_ms", "pr_interval_ms", "osborn_j_wave_amplitude_uv"):
            a, b = pd.to_numeric(cold.get(marker), errors="coerce"), pd.to_numeric(warm.get(marker), errors="coerce")
            row[f"rel_{marker}"] = (a - b) / b if b else np.nan
            row[f"d_{marker}"] = a - b
        rows.append(row)
    pairs_frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    covariates = ("delta_hr", "delta_k")
    complete = pairs_frame.dropna(subset=["delta_temperature", *covariates])

    def per_degree(marker: str) -> Callable[[pd.DataFrame], float]:
        # Per degree of cooling: the exposure is warm minus cold.
        return lambda f: -ols_coefficient(f.dropna(subset=[f"rel_{marker}"]), f"rel_{marker}", "delta_temperature", covariates)

    results = [
        directional_result(tid, f"relative_{marker}_per_degree_cooling", "> 0", complete, per_degree(marker), direction=1)
        for marker in ("qt_interval_ms", "j_tpeak_c_ms", "qrs_duration_ms", "pr_interval_ms")
    ]

    def ordering(first: str, second: str) -> Callable[[pd.DataFrame], float]:
        return lambda f: abs(per_degree(first)(f)) - abs(per_degree(second)(f))

    qt_over_qrs = directional_result(tid, "qt_minus_qrs_sensitivity", "|QT| - |QRS| > 0", complete,
                                     ordering("qt_interval_ms", "qrs_duration_ms"), direction=1)
    _, _, draws = cluster_bootstrap(complete, ordering("qt_interval_ms", "qrs_duration_ms"))
    if len(draws) and float(np.mean(draws <= 0)) >= 0.95:
        qt_over_qrs.verdict, qt_over_qrs.reason = REFUTED, "QRS at least as temperature-sensitive as QT in >= 95 % of draws"
    results.append(qt_over_qrs)
    results.append(directional_result(tid, "qrs_minus_pr_sensitivity", "|QRS| - |PR| > 0", complete,
                                      ordering("qrs_duration_ms", "pr_interval_ms"), direction=1))
    cold = pairs_frame.loc[pairs_frame["cold_below_35"]].dropna(subset=["d_osborn_j_wave_amplitude_uv"])
    results.append(directional_result(tid, "j_wave_below_35", "> 0", cold,
                                      lambda f: float(f["d_osborn_j_wave_amplitude_uv"].mean()), direction=1))
    return results


def thesis_t8(data: ThesisData) -> list[ThesisResult]:
    tid = "T8"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    context = data.context(["c_reactive_protein_nearest", "potassium_nearest", "med_qt_prolonging_drug"])
    temperature = data.nearest_chart((223762, 226329), 1, 25, 45, fahrenheit_itemids=(223761,))
    frame = _covariates_hr(base.merge(context, on=["subject_id", "study_id"]))
    frame = _merge_value(frame, temperature, "temperature")
    frame["crp"] = _numeric(frame, "c_reactive_protein_nearest")
    frame = frame.loc[frame["crp"] > 0]
    frame = _first_per_subject(frame)
    frame["log_crp"] = np.log(frame["crp"])
    frame["k"] = _numeric(frame, "potassium_nearest")
    frame["qt_drug"] = frame.get("med_qt_prolonging_drug", pd.Series(False, index=frame.index)).fillna(False).astype(float)
    # "Temperature where available": the missing-indicator method keeps the
    # persons without a charted temperature.
    frame["temperature_missing"] = frame["temperature"].isna().astype(float)
    frame["temperature_filled"] = frame["temperature"].fillna(frame["temperature"].median())
    frame["log_crp_male"] = frame["log_crp"] * frame["male"]
    covariates = ("hr", "hr2", "age", "k", "temperature_filled", "temperature_missing", "qt_drug", "male")
    complete = frame.dropna(subset=["j_tpeak_c_ms", "log_crp", *covariates]) if "j_tpeak_c_ms" in frame else frame.iloc[0:0]
    main = directional_result(
        tid, "log_crp_coefficient", "> 0", complete,
        lambda f: ols_coefficient(f, "j_tpeak_c_ms", "log_crp", covariates), direction=1,
        refuted=lambda est, low, high: np.isfinite(low) and low <= 0 <= high and abs(est) < 1.0)
    interaction = directional_result(
        tid, "log_crp_by_male", "> 0", complete,
        lambda f: ols_coefficient(f, "j_tpeak_c_ms", "log_crp_male", ("log_crp", *covariates)), direction=1)
    return [main, interaction]


def thesis_t9(data: ThesisData) -> list[ThesisResult]:
    tid = "T9"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    tsh = data.nearest_lab("tsh", 24 * 7)
    if tsh.empty:
        return [not_evaluable(tid, "all", "", "no TSH within 7 days")]
    amiodarone = data.administrations(("amiodarone",))
    frame = _covariates_hr(base.merge(tsh[["subject_id", "study_id", "value_numeric"]], on=["subject_id", "study_id"]))
    if not amiodarone.empty:
        first_dose = amiodarone.groupby("subject_id")["charttime"].min()
        exposed = frame["subject_id"].map(first_dose)
        frame = frame.loc[exposed.isna() | (exposed > frame["ecg_time"])]
    frame = frame.loc[frame["value_numeric"] > 0]
    frame = _first_per_subject(frame)
    frame["log_tsh"] = np.log(frame["value_numeric"])
    frame["log_tsh2"] = frame["log_tsh"] ** 2
    rate_adjusted = ("log_tsh", "hr", "hr2", "age", "male")
    complete = frame.dropna(subset=["qt_interval_ms", *rate_adjusted]) if "qt_interval_ms" in frame else frame.iloc[0:0]
    results = [
        _null_prediction(directional_result(
            tid, "rate_adjusted_quadratic", "= 0 (no U)", complete,
            lambda f: ols_coefficient(f, "qt_interval_ms", "log_tsh2", rate_adjusted), direction=1)),
        directional_result(tid, "rate_adjusted_linear", "> 0", complete,
                           lambda f: ols_coefficient(f, "qt_interval_ms", "log_tsh", ("log_tsh2", "hr", "hr2", "age", "male")),
                           direction=1),
    ]
    quadratic = results[0]
    if quadratic.verdict == REFUTED and quadratic.estimate < 0:
        quadratic.verdict, quadratic.reason = UNDECIDED, "curvature away from a U"
    bazett = frame.dropna(subset=["classical_qtc_bazett", "log_tsh", "age", "male"]) if "classical_qtc_bazett" in frame else frame.iloc[0:0]
    results.append(directional_result(
        tid, "bazett_quadratic", "> 0 (U with Bazett)", bazett,
        lambda f: ols_coefficient(f, "classical_qtc_bazett", "log_tsh2", ("log_tsh", "age", "male")), direction=1))
    return results


def thesis_t10(data: ThesisData) -> list[ThesisResult]:
    tid = "T10"
    base = data.base(THESIS_MARKERS)
    column = "secondary_morphology_svg_to_qrs_area_delta_ratio"
    if column not in base:
        return [not_evaluable(tid, "all", "", "secondary-cluster gradient markers not extracted (extraction < 3.50.0)")]
    frame = _first_per_subject(base.dropna(subset=[column]))
    results = [
        directional_result(tid, "median_R", "< 1", frame, lambda f: float(_numeric(f, column).median()),
                           direction=-1, null=1.0,
                           refuted=lambda est, low, high: np.isfinite(low) and low >= 1.0),
        directional_result(tid, "median_cosine", "< 0", frame.dropna(subset=["secondary_morphology_t_qrs_area_delta_cosine"]),
                           lambda f: float(_numeric(f, "secondary_morphology_t_qrs_area_delta_cosine").median()), direction=-1),
        directional_result(tid, "R_vs_coupling", "Spearman < 0",
                           frame.dropna(subset=["secondary_morphology_mean_coupling_ratio"]),
                           lambda f: spearman(f, column, "secondary_morphology_mean_coupling_ratio"), direction=-1),
    ]
    return results


def _transition_pairs(
    frame: pd.DataFrame, state_column: str, *, max_days: float = 30.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cases: last ECG in a state followed by the first ECG out of it within
    ``max_days``. Controls: consecutive ECGs of persons never in the state."""

    ordered = frame.sort_values(["subject_id", "ecg_time", "study_id"], kind="stable").copy()
    ordered["in_state"] = _positive(ordered[state_column])
    nxt = ordered.groupby("subject_id").shift(-1)
    days = (nxt["ecg_time"] - ordered["ecg_time"]).dt.total_seconds() / 86400.0
    transition = ordered["in_state"] & ~nxt["in_state"].fillna(True).astype(bool) & (days <= max_days) & (days > 0)
    ever = ordered.groupby("subject_id")["in_state"].transform("any")
    control = ~ever & nxt["study_id"].notna() & (days <= max_days) & (days > 0)

    def build(mask: pd.Series) -> pd.DataFrame:
        first, second = ordered.loc[mask], nxt.loc[mask]
        result = pd.DataFrame({
            "subject_id": first["subject_id"].to_numpy(), "split": first["split"].to_numpy(),
            "days": days.loc[mask].to_numpy(),
        })
        qrs = first[["qrs_area_vector_x_mv_ms", "qrs_area_vector_y_mv_ms", "qrs_area_vector_z_mv_ms"]].to_numpy(float)
        t_next = second[["t_area_vector_x_mv_ms", "t_area_vector_y_mv_ms", "t_area_vector_z_mv_ms"]].to_numpy(float)
        result["angle"] = angle_between(t_next, qrs)
        return result.dropna(subset=["angle"])

    return build(transition), build(control)


def _gap_standardized_difference(cases: pd.DataFrame, controls: pd.DataFrame) -> Callable[[pd.DataFrame], float]:
    bins = [0, 1, 3, 7, 14, 30]

    def statistic(frame: pd.DataFrame) -> float:
        c = frame.loc[frame["role"].eq("case")].copy()
        k = frame.loc[frame["role"].eq("control")].copy()
        if c.empty or k.empty:
            return float("nan")
        c["bin"] = pd.cut(c["days"], bins, include_lowest=True).astype(str)
        k["bin"] = pd.cut(k["days"], bins, include_lowest=True).astype(str)
        target = c["bin"].value_counts(normalize=True)
        control_means = k.groupby("bin")["angle"].mean()
        common = target.index.intersection(control_means.index)
        if common.empty:
            return float("nan")
        weights = target.loc[common] / target.loc[common].sum()
        return float(c.loc[c["bin"].isin(common), "angle"].mean() - (control_means.loc[common] * weights).sum())

    return statistic


def thesis_t11(data: ThesisData) -> list[ThesisResult]:
    tid = "T11"
    base = data.base(THESIS_MARKERS)
    needed = ["qrs_area_vector_x_mv_ms", "t_area_vector_x_mv_ms"]
    if _missing(base, needed):
        return [not_evaluable(tid, "all", "", "area vector components not extracted")]
    results = []
    for arm, state, exclude in (("pacing", "report_pacemaker_state", None),
                                ("lbbb", "report_lbbb_state", "report_pacemaker_state")):
        frame = base if exclude is None else base.loc[~_positive(base[exclude])]
        if state not in frame:
            results.append(not_evaluable(tid, arm, "", f"{state} unavailable"))
            continue
        cases, controls = _transition_pairs(frame, state)
        combined = pd.concat([cases.assign(role="case"), controls.assign(role="control")], ignore_index=True)
        results.append(directional_result(
            tid, f"{arm}_angle_cases_minus_controls", "< 0", combined,
            _gap_standardized_difference(cases, controls), direction=-1, unit_count=len(cases)))
        if arm == "pacing":
            results.append(directional_result(
                tid, "pacing_angle_vs_days", "slope > 0", cases,
                lambda f: ols_coefficient(f, "angle", "days"), direction=1))
    return results


def thesis_t12(data: ThesisData) -> list[ThesisResult]:
    tid = "T12"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    context = data.context(["med_qt_prolonging_drug"])
    frame = base.merge(context, on=["subject_id", "study_id"], how="left").dropna(subset=["hadm_id", "qt_interval_ms", "heart_rate_bpm"])
    hour = frame["ecg_time"].dt.hour
    night = frame.loc[hour.between(0, 5)]
    day = frame.loc[hour.between(10, 17)]
    joined = night.merge(day, on="hadm_id", suffixes=("_night", "_day"))
    joined = joined.loc[(joined["heart_rate_bpm_night"] - joined["heart_rate_bpm_day"]).abs() <= 5]
    same_drug = _flag(joined, "med_qt_prolonging_drug_night") == _flag(joined, "med_qt_prolonging_drug_day")
    joined = joined.loc[same_drug].copy()
    joined["abs_hr"] = (joined["heart_rate_bpm_night"] - joined["heart_rate_bpm_day"]).abs()
    joined = joined.sort_values(["hadm_id", "abs_hr", "ecg_time_night", "ecg_time_day"]).drop_duplicates("hadm_id")
    joined["subject_id"] = joined["subject_id_night"]
    joined["split"] = joined["split_night"]
    joined["delta_qt"] = _numeric(joined, "qt_interval_ms_night") - _numeric(joined, "qt_interval_ms_day")
    return [directional_result(tid, "qt_night_minus_day", "> 0", joined, lambda f: float(f["delta_qt"].mean()), direction=1)]


def thesis_t13(data: ThesisData) -> list[ThesisResult]:
    tid = "T13"
    base = data.base(THESIS_MARKERS)
    tidal = data.nearest_chart((224685,), 1, 50, 2000)
    if tidal.empty or "respiratory_axis_modulation_deg" not in base:
        return [not_evaluable(tid, "all", "", "no tidal volume or no respiratory marker")]
    frame = _merge_value(base, tidal, "tidal_volume").dropna(subset=["tidal_volume", "respiratory_axis_modulation_deg"])
    frame = frame.loc[_numeric(frame, "respiratory_cycles_per_record") >= 2]
    frame = _first_per_subject(frame)
    return [directional_result(
        tid, "axis_modulation_vs_tidal_volume", "Spearman > 0", frame,
        lambda f: spearman(f, "respiratory_axis_modulation_deg", "tidal_volume"), direction=1,
        refuted=lambda est, low, high: np.isfinite(low) and low <= 0 <= high and abs(est) < 0.05)]


def thesis_t14(data: ThesisData) -> list[ThesisResult]:
    tid = "T14"
    base = data.base(THESIS_MARKERS)
    groups = {
        "takotsubo": data.hadm_with_codes(("42983",), ("I5181",)),
        "lad_stemi": data.hadm_with_codes(("41010", "41011"), ("I210",)),
        "sah": data.hadm_with_codes(("430",), ("I60",)),
    }
    membership = pd.Series(0, index=base.index)
    labels = pd.Series("", index=base.index)
    for name, hadms in groups.items():
        hit = base["hadm_id"].isin(hadms)
        membership += hit.astype(int)
        labels = labels.where(~hit, name)
    frame = base.loc[membership.eq(1)].assign(group=labels[membership.eq(1)])
    frame = frame.sort_values(["hadm_id", "ecg_time"]).drop_duplicates("hadm_id")
    marker = "ventricular_gradient_apicobasal_uv_ms"
    if marker not in frame:
        return [not_evaluable(tid, "all", "", f"{marker} not extracted")]
    frame = frame.dropna(subset=[marker])
    two = frame.loc[frame["group"].isin(["takotsubo", "lad_stemi"])]
    first = directional_result(
        tid, "apicobasal_takotsubo_minus_lad", "< 0", two,
        lambda f: float(_numeric(f.loc[f["group"].eq("takotsubo")], marker).median()
                        - _numeric(f.loc[f["group"].eq("lad_stemi")], marker).median()),
        direction=-1, unit_count=int(min(two["group"].value_counts().reindex(["takotsubo", "lad_stemi"]).fillna(0))))

    def closeness(f: pd.DataFrame) -> float:
        medians = f.groupby("group")[marker].median()
        if not {"takotsubo", "lad_stemi", "sah"} <= set(medians.index):
            return float("nan")
        return float(abs(medians["sah"] - medians["lad_stemi"]) - abs(medians["sah"] - medians["takotsubo"]))

    second = directional_result(tid, "sah_closer_to_takotsubo", "> 0", frame, closeness, direction=1,
                                unit_count=int(frame["group"].value_counts().reindex(list(groups)).fillna(0).min()))
    return [first, second]


def thesis_t15(data: ThesisData) -> list[ThesisResult]:
    tid = "T15"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    context = data.context(["dx_pericarditis"])
    labels = data.labels(["acute_mi_high_specificity"])
    frame = base.merge(context, on=["subject_id", "study_id"], how="left").merge(labels, on=["subject_id", "study_id"], how="left")
    pericarditis = _flag(frame, "dx_pericarditis")
    stemi = _positive(frame.get("acute_mi_high_specificity_state", pd.Series("", index=frame.index)))
    frame = frame.loc[pericarditis ^ stemi].assign(group=np.where(pericarditis[pericarditis ^ stemi], "pericarditis", "stemi"))
    frame = frame.sort_values(["subject_id", "ecg_time"]).drop_duplicates(["subject_id", "group"])
    results = []
    for marker in ("pr_segment_to_st_vector_angle_deg", "pr_segment_to_p_vector_ratio"):
        if marker not in frame:
            results.append(not_evaluable(tid, marker, "", f"{marker} not extracted (extraction < 3.50.0)"))
            continue
        part = frame.dropna(subset=[marker])
        results.append(directional_result(
            tid, f"{marker}_pericarditis_minus_stemi", "> 0", part,
            lambda f, m=marker: float(_numeric(f.loc[f["group"].eq("pericarditis")], m).mean()
                                      - _numeric(f.loc[f["group"].eq("stemi")], m).mean()),
            direction=1, unit_count=int(part["group"].value_counts().reindex(["pericarditis", "stemi"]).fillna(0).min())))
    return results


def thesis_t16(data: ThesisData) -> list[ThesisResult]:
    tid = "T16"
    base = data.base(THESIS_MARKERS)
    needed = ["st_area_vector_x_mv_ms", "st_area_vector_y_mv_ms", "st_area_vector_z_mv_ms"]
    if _missing(base, needed):
        return [not_evaluable(tid, "all", "", "ST area vector not extracted")]
    frame = _merge_value(base, data.nearest_lab("hemoglobin", 6), "hb")
    frame = _merge_value(frame, data.nearest_chart((220045,), 1, 20, 250), "hr_chart")
    frame = _merge_value(frame, data.nearest_chart((220050, 220179), 1, 40, 300), "sbp")
    frame = _merge_value(frame, data.nearest_chart((220277,), 1, 50, 100), "spo2")
    frame = frame.dropna(subset=["hb", "hr_chart", "sbp", "spo2", *needed])
    frame = frame.loc[frame["hb"] > 0]
    unit = AVR_DIRECTION / np.linalg.norm(AVR_DIRECTION)
    frame["st_avr_projection"] = frame[needed].to_numpy(float) @ unit
    frame["mismatch"] = np.log((frame["hr_chart"] * frame["sbp"]) / (frame["hb"] * frame["spo2"]))
    return [directional_result(
        tid, "avr_projection_vs_mismatch", "Spearman > 0", frame,
        lambda f: spearman(f, "st_avr_projection", "mismatch"), direction=1,
        refuted=lambda est, low, high: np.isfinite(low) and low <= 0 <= high and abs(est) < 0.05)]


def thesis_t17(data: ThesisData) -> list[ThesisResult]:
    tid = "T17"
    base = data.base(THESIS_MARKERS)
    context = data.context(["has_tte_30d", "la_volume_index"])
    frame = base.merge(context, on=["subject_id", "study_id"])
    frame = frame.loc[_positive(frame.get("report_af_state", pd.Series("", index=frame.index)))
                      & _flag(frame, "has_tte_30d")]
    if "tq_atrial_rms" not in frame:
        return [not_evaluable(tid, "all", "", "tq_atrial_rms not extracted")]
    frame = _first_per_subject(frame)
    frame["normalized_f_wave"] = _numeric(frame, "tq_atrial_rms") / _numeric(frame, "qrs_vector_magnitude")
    frame["la_volume_index"] = _numeric(frame, "la_volume_index")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["normalized_f_wave", "la_volume_index"])
    return [directional_result(
        tid, "f_wave_vs_la_volume_index", "Spearman < 0", frame,
        lambda f: spearman(f, "normalized_f_wave", "la_volume_index"), direction=-1,
        refuted=lambda est, low, high: np.isfinite(low) and low <= 0 <= high and abs(est) < 0.05)]


def thesis_t18(data: ThesisData) -> list[ThesisResult]:
    tid = "T18"
    base = data.base(THESIS_MARKERS)
    infusions = data.infusion_running(CATECHOLAMINE_ITEMIDS)
    if infusions.empty:
        return [not_evaluable(tid, "all", "", "no inputevents")]
    frame = base.merge(infusions, on=["subject_id", "study_id"])
    frame = _merge_value(frame, data.nearest_lab("potassium", 6), "k").dropna(subset=["icustay_id", "qt_interval_ms", "heart_rate_bpm"])
    none_running = ~frame[list(CATECHOLAMINE_ITEMIDS)].any(axis=1)
    results = []
    for drug in ("epinephrine", "norepinephrine", "phenylephrine"):
        exposed = frame.loc[frame[drug]]
        unexposed = frame.loc[none_running]
        joined = exposed.merge(unexposed, on="icustay_id", suffixes=("_on", "_off"))
        if joined.empty:
            results.append(not_evaluable(tid, f"{drug}_qt_effect", "> 0", "no stay with ECGs on and off the infusion"))
            continue
        joined["gap"] = (joined["ecg_time_on"] - joined["ecg_time_off"]).abs()
        joined = joined.sort_values(["icustay_id", "gap"]).drop_duplicates("icustay_id")
        joined["subject_id"] = joined["subject_id_on"]
        joined["split"] = joined["split_on"]
        joined["d_qt"] = _numeric(joined, "qt_interval_ms_on") - _numeric(joined, "qt_interval_ms_off")
        joined["d_hr"] = _numeric(joined, "heart_rate_bpm_on") - _numeric(joined, "heart_rate_bpm_off")
        joined["d_hr2"] = joined["d_hr"] ** 2
        joined["d_k"] = _numeric(joined, "k_on") - _numeric(joined, "k_off")

        def intercept(f: pd.DataFrame, covariates: Sequence[str]) -> float:
            data_ = f[["d_qt", *covariates]].dropna()
            if len(data_) < len(covariates) + 2:
                return float("nan")
            design = np.column_stack([np.ones(len(data_)), data_[list(covariates)].to_numpy(float)])
            return float(np.linalg.lstsq(design, data_["d_qt"].to_numpy(float), rcond=None)[0][0])

        complete = joined.dropna(subset=["d_qt", "d_hr", "d_k"])
        results.append(directional_result(tid, f"{drug}_qt_effect", "> 0", complete,
                                          lambda f: intercept(f, ("d_hr", "d_hr2")), direction=1))
        if drug == "epinephrine":
            def mediated(f: pd.DataFrame) -> float:
                total = intercept(f, ("d_hr", "d_hr2"))
                direct = intercept(f, ("d_hr", "d_hr2", "d_k"))
                return float((total - direct) / total) if total else float("nan")

            results.append(directional_result(tid, "epinephrine_share_via_potassium", ">= 0.5", complete,
                                              mediated, direction=1, null=0.5,
                                              refuted=lambda est, low, high: np.isfinite(high) and high <= 0))
    return results


# --------------------------------------------------------------- Teil B


def _criterion_result(
    tid: str,
    component: str,
    frame: pd.DataFrame,
    score: str,
    standard: Optional[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> ThesisResult:
    """Section B1: threshold on train at the standard's train specificity
    (95 % without a standard), paired comparison on validation."""

    prediction = ">= 0.05 sensitivity over the standard at equal specificity"
    if score not in frame or (standard is not None and standard not in frame):
        return not_evaluable(tid, component, prediction, "score or standard column missing")
    train = frame.loc[frame["split"].eq("train")].dropna(subset=[score])
    validation = frame.loc[frame["split"].eq("validation")].dropna(subset=[score]).copy()
    if standard is None:
        return not_evaluable(tid, component, prediction, "no clinical standard")
    train = train.dropna(subset=[standard])
    validation = validation.dropna(subset=[standard])
    counts = [int((part["label"] == value).sum()) for part in (train, validation) for value in (0, 1)]
    if min(counts) < MINIMUM_UNITS:
        return not_evaluable(tid, component, prediction, f"class counts train/validation {counts}")
    controls = train.loc[train["label"].eq(0)]
    specificity = 1.0 - float(controls[standard].astype(float).mean())
    cut = threshold_at_specificity(_numeric(controls, score).to_numpy(), specificity)
    validation["vcg_positive"] = (_numeric(validation, score) > cut).astype(float)
    validation["standard_positive"] = validation[standard].astype(float)
    estimate = paired_sensitivity_difference(validation, "vcg_positive", "standard_positive")
    low, high, _ = cluster_bootstrap(
        validation, lambda f: paired_sensitivity_difference(f, "vcg_positive", "standard_positive"),
        iterations=iterations)
    validation_controls = validation.loc[validation["label"].eq(0)]
    result = ThesisResult(
        tid, component, prediction, estimate, low, high,
        n_units=int(validation["label"].eq(1).sum()), n_subjects=int(validation["subject_id"].nunique()),
        extra={
            "train_specificity_target": specificity,
            "validation_specificity_vcg": 1.0 - float(validation_controls["vcg_positive"].mean()),
            "validation_specificity_standard": 1.0 - float(validation_controls["standard_positive"].mean()),
            "mcnemar_p": mcnemar_p(validation, "vcg_positive", "standard_positive"),
            "threshold": cut,
        },
    )
    if np.isfinite(high) and high < 0:
        result.verdict = REFUTED
    elif np.isfinite(low) and low > 0 and estimate >= 0.05:
        result.verdict = CONFIRMED
    else:
        result.verdict = UNDECIDED
    return result


def _train_z_sum(frame: pd.DataFrame, markers: Sequence[str], reference: pd.DataFrame) -> pd.Series:
    """Sum of markers standardised on ``reference`` (train rows with a label),
    each oriented by its train case-control direction."""

    total = pd.Series(0.0, index=frame.index)
    for marker in markers:
        values = _numeric(frame, marker)
        reference_values = _numeric(reference, marker)
        sd = float(reference_values.std())
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=frame.index)
        difference = (reference_values[reference["label"].eq(1)].mean()
                      - reference_values[reference["label"].eq(0)].mean())
        direction = 1.0 if not np.isfinite(difference) or difference >= 0 else -1.0
        total = total + direction * (values - reference_values.mean()) / sd
    return total


def thesis_v1(data: ThesisData) -> list[ThesisResult]:
    tid = "V1"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    labels = data.labels(["old_mi_icd", "mi_or_revascularization_history"])
    frame = _first_per_subject(base.merge(labels, on=["subject_id", "study_id"]))
    old = frame.get("old_mi_icd_state", pd.Series("", index=frame.index)).astype("string").str.lower()
    history = frame.get("mi_or_revascularization_history_state", pd.Series("", index=frame.index)).astype("string").str.lower()
    case = old.eq("positive").fillna(False)
    control = (old.eq("negative") & history.eq("negative")).fillna(False)
    frame = frame.loc[case | control].copy()
    frame["label"] = case[case | control].astype(int)
    azimuth = np.radians(_numeric(frame, "qrs_initial_30ms_force_azimuth_deg").to_numpy(float))
    elevation = np.radians(_numeric(frame, "qrs_initial_30ms_force_elevation_deg").to_numpy(float))
    vectors = np.column_stack([
        np.cos(elevation) * np.cos(azimuth), np.sin(elevation), np.cos(elevation) * np.sin(azimuth),
    ])
    usable = np.all(np.isfinite(vectors), axis=1)
    reference = vectors[(frame["split"].eq("train") & frame["label"].eq(0)).to_numpy() & usable]
    if len(reference) < MINIMUM_UNITS:
        return [not_evaluable(tid, "initial_force_direction_vs_q_waves", "", "too few train controls with an initial force")]
    mean = reference.mean(axis=0)
    inverse = np.linalg.pinv(np.cov(reference, rowvar=False))
    centred = vectors - mean
    distance = np.sqrt(np.einsum("ij,jk,ik->i", centred, inverse, centred))
    distance[~usable] = np.nan
    frame["initial_force_distance"] = distance
    return [_criterion_result(tid, "initial_force_direction_vs_q_waves", frame, "initial_force_distance",
                              "classical_pathological_q_waves_positive")]


def thesis_v2(data: ThesisData) -> list[ThesisResult]:
    tid = "V2"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    frame = _potassium_groups(data, base)
    if frame.empty:
        return [not_evaluable(tid, "all", "", "no potassium 50971")]
    k = frame["value_numeric"]
    frame = frame.loc[frame["group"].eq("normal") | (k >= 6.5)].copy()
    frame["label"] = np.where(frame["group"].eq("nonhemolytic_high"), 1,
                              np.where(frame["group"].eq("normal"), 0, -1))
    frame = frame.sort_values(["subject_id", "ecg_time"]).drop_duplicates(["subject_id", "label"])
    markers = ("t_loop_compactness", "t_loop_apex_sharpness", "t_loop_max_min_ratio")
    if _missing(frame, markers):
        return [not_evaluable(tid, "all", "", "T-loop shape markers not extracted")]
    labelled = frame.loc[frame["label"].isin([0, 1])].copy()
    reference = labelled.loc[labelled["split"].eq("train")]
    labelled["t_loop_score"] = _train_z_sum(labelled, markers, reference)
    first = _criterion_result(tid, "t_loop_shape_vs_peaked_t", labelled, "t_loop_score",
                              "classical_peaked_t_wave_heuristic_positive")
    results = [first]
    threshold = first.extra.get("threshold") if first.extra else None
    hemolysed = frame.loc[frame["label"].eq(-1)].copy()
    if threshold is not None and np.isfinite(threshold) and not hemolysed.empty:
        hemolysed["t_loop_score"] = _train_z_sum(hemolysed, markers, reference)
        combined = pd.concat([labelled.loc[labelled["label"].eq(1)], hemolysed], ignore_index=True)
        combined["fires"] = (combined["t_loop_score"] > threshold).astype(float)
        combined["kind"] = np.where(combined["label"].eq(1), "nonhemolytic", "hemolytic")
        results.append(directional_result(
            tid, "trigger_nonhemolytic_minus_hemolytic", "> 0", combined,
            lambda f: float(f.loc[f["kind"].eq("nonhemolytic"), "fires"].mean()
                            - f.loc[f["kind"].eq("hemolytic"), "fires"].mean()),
            direction=1, unit_count=int(combined["kind"].eq("hemolytic").sum())))
    else:
        results.append(not_evaluable(tid, "trigger_nonhemolytic_minus_hemolytic", "> 0",
                                     "no threshold or no haemolysed values"))
    return results


def thesis_v3(data: ThesisData) -> list[ThesisResult]:
    tid = "V3"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    procedures = data.procedure_events(("370",), ("0W9D",))
    if procedures.empty:
        return [not_evaluable(tid, "all", "", "no pericardiocentesis")]
    context = data.context(["dx_pericardial_effusion"])
    frame = base.merge(context, on=["subject_id", "study_id"], how="left")
    procedure_day = pd.to_datetime(procedures.groupby("hadm_id")["chartdate"].min())
    day = pd.to_datetime(frame["hadm_id"].map(procedure_day))
    days = (day - frame["ecg_time"].dt.normalize()).dt.days
    case = days.between(0, 1).fillna(False)
    effusion = _flag(frame, "dx_pericardial_effusion") & day.isna()
    frame = frame.loc[case | effusion].copy()
    frame["label"] = case[case | effusion].astype(int)
    frame = frame.sort_values(["subject_id", "ecg_time"]).drop_duplicates(["subject_id", "label"])
    frame["device_alternans"] = _text(frame, "machine_report").str.contains("alternans").astype(float)
    return [_criterion_result(tid, "vector_alternans_vs_device", frame, "qrs_alternans_vector", "device_alternans")]


def thesis_v4(data: ThesisData, run_root: Path) -> list[ThesisResult]:
    tid = "V4"
    from .reference_ranges import NormativeCohortPolicy, assign_strata

    ranges = data.reference_ranges(run_root)
    if ranges.empty:
        return [not_evaluable(tid, "all", "", "reference-range stage has not written this run's ranges")]
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    base = base.loc[_text(base, "machine_report").str.contains("normal ecg")]
    valve_columns = ["aortic_stenosis_severity", "aortic_regurgitation_severity", "mitral_stenosis_severity",
                     "mitral_regurgitation_severity", "tricuspid_regurgitation_severity"]
    context = data.context(["has_tte_90d", "preferred_lvef", *valve_columns])
    labels = data.labels(["echo_lvh_strict"])
    frame = base.merge(context, on=["subject_id", "study_id"]).merge(labels, on=["subject_id", "study_id"], how="left")
    frame = _first_per_subject(frame.loc[_flag(frame, "has_tte_90d")]).reset_index(drop=True)
    valves = [column for column in valve_columns if column in frame]
    abnormal = (
        (_numeric(frame, "preferred_lvef") < 50)
        | _positive(frame.get("echo_lvh_strict_state", pd.Series("", index=frame.index)))
        | (frame[valves].apply(pd.to_numeric, errors="coerce") >= 2).any(axis=1)
    )
    frame["abnormal_echo"] = abnormal.astype(float)
    frame = assign_strata(frame, NormativeCohortPolicy())
    outside = np.zeros(len(frame), dtype=bool)
    for marker in ("spatial_qrs_t_angle_deg", "svg_magnitude", "t_loop_compactness"):
        table = ranges.loc[ranges["marker_id"].eq(marker) & ranges["estimable"].fillna(False).astype(bool)]
        if table.empty or marker not in frame:
            continue
        limits = frame[["sex_band", "age_band", "rate_band"]].merge(
            table[["sex_band", "age_band", "rate_band", "percentile_2_5", "percentile_97_5"]].drop_duplicates(
                ["sex_band", "age_band", "rate_band"]),
            on=["sex_band", "age_band", "rate_band"], how="left")
        value = _numeric(frame, marker).to_numpy(float)
        low, high = limits["percentile_2_5"].to_numpy(float), limits["percentile_97_5"].to_numpy(float)
        with np.errstate(invalid="ignore"):
            outside |= np.nan_to_num((value < low) | (value > high), nan=0).astype(bool)
    frame["vcg_abnormal"] = outside
    return [directional_result(
        tid, "abnormal_echo_vcg_abnormal_minus_normal", "> 0", frame,
        lambda f: float(f.loc[f["vcg_abnormal"], "abnormal_echo"].mean() - f.loc[~f["vcg_abnormal"], "abnormal_echo"].mean()),
        direction=1, unit_count=int(min(frame["vcg_abnormal"].sum(), (~frame["vcg_abnormal"]).sum())))]


def thesis_v5(data: ThesisData) -> list[ThesisResult]:
    tid = "V5"
    base = data.base(THESIS_MARKERS)
    administrations = data.administrations(("doxorubicin", "daunorubicin", "idarubicin", "epirubicin"))
    if administrations.empty:
        return [not_evaluable(tid, "all", "", "no anthracycline administrations")]
    first = administrations.groupby("subject_id")["charttime"].min().rename("first_dose").reset_index()
    frame = base.merge(first, on="subject_id")
    frame["days"] = (frame["ecg_time"] - frame["first_dose"]).dt.total_seconds() / 86400.0
    before = frame.loc[frame["days"] < 0].sort_values("days").groupby("subject_id").tail(1)
    after = frame.loc[frame["days"].between(30, 365)].sort_values("days").groupby("subject_id").head(1)
    pairs = before.merge(after, on="subject_id", suffixes=("_before", "_after"))
    if pairs.empty:
        return [not_evaluable(tid, "all", "", "no ECG pair around the first dose")]
    context = data.context(["preferred_lvef", "has_tte_30d"])
    echo = frame.merge(context, on=["subject_id", "study_id"])
    later = echo.loc[echo["days"].between(30, 365) & _flag(echo, "has_tte_30d")]
    lvef_drop = _numeric(later, "preferred_lvef").groupby(later["subject_id"]).min().lt(50)
    pairs["lvef_drop"] = pairs["subject_id"].map(lvef_drop)
    pairs["split"] = pairs["split_before"]
    for marker, name in (("svg_magnitude", "svg"), ("qrs_area_vector_magnitude_mv_ms", "qrs_area")):
        pairs[f"d_{name}"] = (_numeric(pairs, f"{marker}_after") / _numeric(pairs, "p_vector_magnitude_after")
                              - _numeric(pairs, f"{marker}_before") / _numeric(pairs, "p_vector_magnitude_before"))
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=["lvef_drop"])
    pairs["lvef_drop"] = pairs["lvef_drop"].astype(float)
    return [
        directional_result(tid, f"delta_{name}_vs_lvef_drop", "Spearman < 0", pairs.dropna(subset=[f"d_{name}"]),
                           lambda f, n=name: spearman(f, f"d_{n}", "lvef_drop"), direction=-1)
        for name in ("svg", "qrs_area")
    ]


def _prognostic_result(
    tid: str, component: str, frame: pd.DataFrame, outcome: str, base_columns: Sequence[str],
    added: Sequence[str], *, iterations: int = BOOTSTRAP_ITERATIONS,
) -> ThesisResult:
    """Models fitted on train; delta AUC on validation; the interval
    resamples validation subjects with the models held fixed."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    prediction = "delta AUC > 0 on validation"
    columns = [*base_columns, *added]
    missing = [column for column in columns if column not in frame]
    if missing:
        return not_evaluable(tid, component, prediction, f"columns missing: {missing}")
    prepared = frame[["subject_id", "split", outcome, *columns]].copy()
    prepared[[outcome, *columns]] = prepared[[outcome, *columns]].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.dropna()
    train = prepared.loc[prepared["split"].eq("train")]
    validation = prepared.loc[prepared["split"].eq("validation")].copy()
    counts = [int((part[outcome] == value).sum()) for part in (train, validation) for value in (0, 1)]
    if min(counts) < MINIMUM_UNITS:
        return not_evaluable(tid, component, prediction, f"outcome counts train/validation {counts}")
    for name, features in (("base", list(base_columns)), ("full", columns)):
        scaler = StandardScaler().fit(train[features])
        model = LogisticRegression(max_iter=1000).fit(scaler.transform(train[features]), train[outcome].astype(int))
        validation[f"score_{name}"] = model.predict_proba(scaler.transform(validation[features]))[:, 1]

    def delta(f: pd.DataFrame) -> float:
        labels = f[outcome].astype(int).to_numpy()
        return auc(f["score_full"].to_numpy(), labels) - auc(f["score_base"].to_numpy(), labels)

    estimate = delta(validation)
    low, high, _ = cluster_bootstrap(validation, delta, iterations=iterations)
    result = ThesisResult(tid, component, prediction, estimate, low, high,
                          n_units=int(len(validation)), n_subjects=int(validation["subject_id"].nunique()),
                          extra={"validation_events": int(validation[outcome].sum()),
                                 "train_events": int(train[outcome].sum())})
    if np.isfinite(high) and high < 0:
        result.verdict = REFUTED
    elif np.isfinite(low) and low > 0:
        result.verdict = CONFIRMED
    else:
        result.verdict = UNDECIDED
    return result


def thesis_v6(data: ThesisData) -> list[ThesisResult]:
    tid = "V6"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    context = data.context(["has_tte_30d", "preferred_lvef", "vital_status_followup_hours"])
    if "vital_status_followup_hours" in context.attrs.get("missing_columns", []):
        return [not_evaluable(tid, "all", "", "vital_status_followup_hours missing (evidence before audit D4)")]
    frame = base.merge(context, on=["subject_id", "study_id"])
    frame = _first_per_subject(frame.loc[_flag(frame, "has_tte_30d")])
    frame = frame.loc[_numeric(frame, "preferred_lvef").between(36, 49)].copy()
    arrests = data.hadm_with_codes(("4275",), ("I46",))
    admissions = data.admissions()
    arrest_admissions = admissions.loc[admissions["hadm_id"].isin(arrests), ["subject_id", "admittime"]]
    joined = frame[["subject_id", "ecg_time"]].merge(arrest_admissions, on="subject_id", how="inner")
    days = (joined["admittime"] - joined["ecg_time"]).dt.total_seconds() / 86400.0
    event_subjects = set(joined.loc[(days > 0) & (days <= 365), "subject_id"])
    frame["event"] = frame["subject_id"].isin(event_subjects).astype(int)
    observed = _numeric(frame, "vital_status_followup_hours") >= 8760
    frame = frame.loc[frame["event"].eq(1) | observed].copy()
    frame["male"] = _male(frame)
    return [_prognostic_result(tid, "svg_and_qrs_t_angle_over_lvef", frame, "event",
                               ("preferred_lvef", "age", "male", "qrs_duration_ms"),
                               ("svg_magnitude", "spatial_qrs_t_angle_deg"))]


def thesis_v7(data: ThesisData) -> list[ThesisResult]:
    tid = "V7"
    base = _without_paced_or_lbbb(data.base(THESIS_MARKERS))
    labels = data.labels(["severe_aortic_stenosis_velocity"])
    context = data.context(["aortic_peak_velocity_m_per_s", "death_delta_days", "vital_status_followup_hours"])
    if "vital_status_followup_hours" in context.attrs.get("missing_columns", []):
        return [not_evaluable(tid, "all", "", "vital_status_followup_hours missing (evidence before audit D4)")]
    frame = base.merge(labels, on=["subject_id", "study_id"]).merge(context, on=["subject_id", "study_id"])
    severe = _positive(frame.get("severe_aortic_stenosis_velocity_state", pd.Series("", index=frame.index)))
    frame = _first_per_subject(frame.loc[severe]).copy()
    replacements = data.procedure_events(("3505", "3506", "3521", "3522"), ("02RF",))
    first_replacement = pd.to_datetime(replacements.groupby("subject_id")["chartdate"].min())
    death = _numeric(frame, "death_delta_days")
    replaced_at = (pd.to_datetime(frame["subject_id"].map(first_replacement)) - frame["ecg_time"].dt.normalize()).dt.days
    died_first = death.between(0, 365) & (replaced_at.isna() | (replaced_at > death))
    frame["event"] = died_first.astype(int)
    observed = _numeric(frame, "vital_status_followup_hours") >= 8760
    frame = frame.loc[frame["event"].eq(1) | observed].copy()
    frame["strain"] = frame.get("classical_st_depression_contiguous_positive", pd.Series(np.nan, index=frame.index)).astype(float)
    return [_prognostic_result(tid, "svg_over_strain_and_vmax", frame, "event",
                               ("strain", "aortic_peak_velocity_m_per_s"), ("svg_magnitude", "svg_azimuth"))]


def thesis_v8(data: ThesisData) -> list[ThesisResult]:
    tid = "V8"
    base = data.base(THESIS_MARKERS)
    labels = data.labels(["conduction_progression_1y"])
    frame = base.merge(labels, on=["subject_id", "study_id"])
    state = frame.get("conduction_progression_1y_state", pd.Series("", index=frame.index)).astype("string").str.lower()
    evaluable = state.isin(["positive", "negative"]).fillna(False)
    frame = frame.loc[evaluable].copy()
    frame["event"] = state[evaluable].eq("positive").astype(int)
    reports = data.connection.execute(
        "SELECT subject_id, study_id, report_rbbb_state FROM read_parquet(?)",
        [str(data.config.evidence_root / "machine_report_labels.parquet")],
    ).df()
    frame = frame.merge(reports, on=["subject_id", "study_id"], how="left")
    frame["rbbb"] = _text(frame, "report_rbbb_state").eq("positive").astype(float)
    frame["pr_long"] = (_numeric(frame, "pr_interval_ms") > 200).astype(float)
    frame["lad"] = _numeric(frame, "classical_left_axis_deviation_positive").fillna(0).astype(float)
    frame = _first_per_subject(frame.loc[(frame["rbbb"] == 1) | (frame["pr_long"] == 1) | (frame["lad"] == 1)])
    frame["slow_speed"] = _numeric(frame, "qrs_velocity_max") / _numeric(frame, "qrs_duration_ms")
    return [_prognostic_result(tid, "slow_runs_over_qrs_pr_block", frame, "event",
                               ("qrs_duration_ms", "pr_interval_ms", "rbbb", "pr_long", "lad"),
                               ("qrs_longest_slow_run_ms", "slow_speed"))]


def thesis_v9(data: ThesisData) -> list[ThesisResult]:
    tid = "V9"
    base = data.base(THESIS_MARKERS)
    labels = data.labels(["new_onset_af_1y"])
    context = data.context(["dx_heart_failure"])
    frame = base.merge(labels, on=["subject_id", "study_id"]).merge(context, on=["subject_id", "study_id"], how="left")
    state = frame.get("new_onset_af_1y_state", pd.Series("", index=frame.index)).astype("string").str.lower()
    evaluable = state.isin(["positive", "negative"]).fillna(False)
    frame = frame.loc[evaluable].copy()
    frame["event"] = state[evaluable].eq("positive").astype(int)
    frame = _first_per_subject(frame)
    frame["male"] = _male(frame)
    frame["heart_failure"] = frame.get("dx_heart_failure", pd.Series(False, index=frame.index)).fillna(False).astype(float)
    return [_prognostic_result(tid, "terminal_p_loop_over_clinical", frame, "event",
                               ("age", "male", "heart_failure", "classical_p_duration_ge_120"),
                               ("p_loop_terminal_force_vector", "p_loop_terminal_reversal_deg",
                                "p_loop_terminal_area_fraction", "interatrial_block_vector_score"))]


def thesis_v10(data: ThesisData) -> list[ThesisResult]:
    tid = "V10"
    base = data.base(THESIS_MARKERS)
    pairs = _first_administration_pairs(data, base, ("dofetilide",))
    if pairs.empty:
        return [not_evaluable(tid, "all", "", "no dofetilide starts with ECGs before and after")]
    administrations = data.administrations(("dofetilide",))
    discharge = data.admissions().drop_duplicates("hadm_id").set_index("hadm_id")["dischtime"]
    events = []
    for row in pairs.itertuples(index=False):
        doses = administrations.loc[administrations["subject_id"].eq(row.subject_id)]
        hours = (doses["charttime"] - row.charttime).dt.total_seconds() / 3600.0
        within = doses.loc[(hours > 0) & (hours <= 72)]
        start_dose = pd.to_numeric(pd.Series([row.dose_given]), errors="coerce").iloc[0]
        given = pd.to_numeric(within["dose_given"], errors="coerce")
        reduced = bool(np.isfinite(start_dose) and (given < start_dose).any())
        leave = discharge.get(row.hadm_id, pd.NaT)
        stayed = bool(pd.notna(leave) and (leave - row.charttime).total_seconds() / 3600.0 > 72)
        later = hours.loc[within.index]
        stopped = stayed and not bool((later > 24).any())
        events.append(int(reduced or stopped))
    pairs["event"] = events
    for marker in ("qtc_fridericia_ms", "t_loop_compactness", "t_loop_vector_reversal_index"):
        pairs[f"d_{marker}"] = _numeric(pairs, f"{marker}_after") - _numeric(pairs, f"{marker}_before")
    return [_prognostic_result(tid, "t_loop_change_over_qtc_change", pairs, "event",
                               ("d_qtc_fridericia_ms",), ("d_t_loop_compactness", "d_t_loop_vector_reversal_index"))]


# --------------------------------------------------------------- the stage

THESIS_FUNCTIONS: dict[str, Callable] = {
    "T1": thesis_t1, "T2": thesis_t2, "T3": thesis_t3, "T4": thesis_t4, "T5": thesis_t5,
    "T6": thesis_t6, "T7": thesis_t7, "T8": thesis_t8, "T9": thesis_t9, "T10": thesis_t10,
    "T11": thesis_t11, "T12": thesis_t12, "T13": thesis_t13, "T14": thesis_t14, "T15": thesis_t15,
    "T16": thesis_t16, "T17": thesis_t17, "T18": thesis_t18,
    "V1": thesis_v1, "V2": thesis_v2, "V3": thesis_v3, "V4": thesis_v4, "V5": thesis_v5,
    "V6": thesis_v6, "V7": thesis_v7, "V8": thesis_v8, "V9": thesis_v9, "V10": thesis_v10,
}


def _checkpointed_thesis_rows(
    root: Path,
    *,
    thesis_id: str,
    input_hash: str,
    calculate: Callable[[], list[ThesisResult]],
    force: bool = False,
) -> list[dict]:
    """Reuse only a complete, input-bound thesis block with verified bytes."""
    from .storage import (atomic_write_json, atomic_write_parquet,
                          capture_output_integrity, verify_output_integrity)

    block_root = root / ".thesis_blocks" / input_hash
    block = block_root / f"{thesis_id}.parquet"
    checkpoint = block_root / f"{thesis_id}.checkpoint.json"
    if not force and checkpoint.is_file():
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        integrity = record.get("output_integrity", [])
        if (record.get("input_hash") != input_hash
                or record.get("thesis_id") != thesis_id
                or len(integrity) != 1
                or integrity[0].get("path") != str(block.resolve())
                or not verify_output_integrity(integrity)):
            raise RuntimeError(f"Damaged or stale thesis checkpoint: {checkpoint}")
        rows = pd.read_parquet(block).to_dict(orient="records")
        if len(rows) != record.get("rows"):
            raise RuntimeError(f"Thesis checkpoint row count changed: {checkpoint}")
        return rows
    rows = [result.to_dict() for result in calculate()]
    atomic_write_parquet(pd.DataFrame(rows), block)
    atomic_write_json(checkpoint, {
        "version": 1, "input_hash": input_hash, "thesis_id": thesis_id,
        "rows": len(rows), "output_integrity": capture_output_integrity((block,)),
    })
    return rows


def run_thesis_tests(
    config,
    *,
    run_id: str,
    panel_ids: Sequence[str],
    feature_glob: Optional[str] = None,
    theses: Optional[Sequence[str]] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
):
    """Stage ``thesis-tests``: every fixed version, one row per component."""

    from .storage import (
        StageManifest,
        atomic_write_json,
        atomic_write_parquet,
        capture_output_integrity,
        content_hash,
        development_feature_glob,
        feature_manifest_path,
        file_sha256,
        input_artifact_fingerprints,
        isolated_analysis_connection,
        scientific_config_hash,
    )

    run_root = config.runs_root / str(run_id)
    output_root = run_root / "theses"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "thesis_tests.manifest.json"
    resolved_glob = feature_glob or development_feature_glob(config.features_root, tuple(panel_ids))
    selected = tuple(theses or THESIS_FUNCTIONS)
    unknown = [item for item in selected if item not in THESIS_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown theses: {unknown}")
    upstream = input_artifact_fingerprints({
        "feature_manifest": feature_manifest_path(resolved_glob),
        "evidence_manifest": config.evidence_root / "evidence.manifest.json",
        "labels_manifest": config.labels_root / "labels.manifest.json",
        "master_index_manifest": config.master_root / "master_index.manifest.json",
        "reference_ranges": run_root / "reference_ranges" / "marker_reference_ranges.parquet",
    })
    here = Path(__file__)
    input_hash = content_hash({
        "version": THESIS_TESTS_VERSION,
        "theses": list(selected),
        "run_id": str(run_id),
        "feature_glob": str(resolved_glob),
        "implementation": {
            name: file_sha256(here.with_name(name))
            for name in ("thesis_tests.py", "thesis_data.py", "thesis_statistics.py")
        },
        "upstream": upstream,
    })
    config_hash = scientific_config_hash(config)
    if not force and manifest_path.is_file():
        previous = StageManifest.load(manifest_path)
        if previous.matches(input_hash, config_hash, THESIS_TESTS_VERSION):
            return previous

    rows: list[dict] = []
    with isolated_analysis_connection(config) as connection:
        data = ThesisData(config, resolved_glob, connection)
        for thesis_id in selected:
            if progress:
                progress(f"Thesen: {thesis_id} rechnen")
            function = THESIS_FUNCTIONS[thesis_id]
            def calculate() -> list[ThesisResult]:
                try:
                    return function(data, run_root) if thesis_id == "V4" else function(data)
                except Exception as error:  # noqa: BLE001 - one thesis must not end the stage
                    return [not_evaluable(thesis_id, "all", "", f"{type(error).__name__}: {error}")]
            rows.extend(_checkpointed_thesis_rows(
                output_root, thesis_id=thesis_id, input_hash=input_hash,
                calculate=calculate, force=force))
    table = pd.DataFrame(rows)
    table["fassung_version"] = THESIS_TESTS_VERSION
    table["fassung_document"] = FASSUNGEN_DOCUMENT
    table_path = output_root / "thesis_test_results.parquet"
    summary_path = output_root / "thesis_test_summary.json"
    atomic_write_parquet(table, table_path)
    summary = {
        "version": THESIS_TESTS_VERSION,
        "run_id": str(run_id),
        "fassung_document": FASSUNGEN_DOCUMENT,
        "components": int(len(table)),
        "verdicts": {str(k): int(v) for k, v in table["verdict"].value_counts().items()} if len(table) else {},
        "by_thesis": {
            str(key): {str(k): int(v) for k, v in group["verdict"].value_counts().items()}
            for key, group in table.groupby("thesis_id")
        } if len(table) else {},
        "scope": "development_splits_only_locked_test_untouched",
    }
    atomic_write_json(summary_path, summary)
    manifest = StageManifest.complete(
        stage="thesis-tests",
        stage_version=THESIS_TESTS_VERSION,
        input_hash=input_hash,
        config_hash=config_hash,
        output_paths=(str(table_path.resolve()), str(summary_path.resolve())),
        row_counts={"components": int(len(table))},
        output_integrity=capture_output_integrity((table_path, summary_path)),
    )
    manifest.save(manifest_path)
    return manifest
