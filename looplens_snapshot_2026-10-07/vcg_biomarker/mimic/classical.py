from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from ..core.marker_bounds import RATIO_LIMIT
from ..core.data_loader import LEAD_MAP, LEAD_NAMES_12
from ..core.delineation.base import BeatDelineation, RecordDelineation
from ..core.delineation.base import measurement_confidence_threshold, ventricular_rr_seconds

# Audit S4. MIMIC-IV-ECG is quantised at 5 uV (200 adu/mV). A lead's own QRS
# onset is its first departure of four steps; a Q wave is extended backwards
# only over samples below two steps, so a slightly negative isoelectric
# residue before a late local onset no longer counts. In V2/V3, where UDMI
# gives only a duration, a Q wave must at least be visible.
Q_LOCAL_ONSET_MV = 0.02
Q_BACKWARD_EXTENSION_MV = 0.01
Q_MINIMUM_DEPTH_V2_V3_MV = 0.05


@dataclass(frozen=True)
class ClassicalCriterion:
    criterion_id: str
    value: Optional[float]
    positive: Optional[bool]
    evaluable: bool
    unit: str
    definition: str
    exclusion_reason: str = ""
    exploratory: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ClassicalECGCriteria:
    """Transparent signal-based ECG comparison criteria.

    These outputs are comparator features.  They are not a diagnostic engine
    and must be evaluated against the same assessable clinical cohort as VCG
    markers.
    """

    def compute(
        self,
        diagnostic_ecg: np.ndarray,
        delineation: RecordDelineation,
        *,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        machine_qrs_axis_deg: Optional[float] = None,
        machine_t_axis_deg: Optional[float] = None,
        paced_or_lbbb: bool = False,
        paced: Optional[bool] = None,
        ischemia_context: Optional[str] = None,
        measurement_sink: Optional[dict[str, float]] = None,
        measurements_only: bool = False,
        extended_measurements: bool = False,
    ) -> dict[str, ClassicalCriterion]:
        if measurements_only and measurement_sink is None:
            raise ValueError("measurements_only requires a measurement_sink")
        ecg = np.asarray(diagnostic_ecg, dtype=float)
        if measurement_sink is not None:
            from .lead_measurements import empty_lead_measurements
            measurement_sink.update(empty_lead_measurements(extended=extended_measurements))
        if ecg.ndim != 2 or ecg.shape[1] != 12:
            raise ValueError(f"Expected 12-lead ECG, got shape {ecg.shape}")
        beat = delineation.representative_beat
        if beat is None:
            return self._unavailable("no delineated beat")
        fs = delineation.fs
        p_evaluable = delineation.segment_confidence("p") >= measurement_confidence_threshold("p")
        qrs_evaluable = delineation.segment_confidence("qrs") >= measurement_confidence_threshold("qrs")
        st_evaluable = (
            qrs_evaluable
            and (
                beat.j_point is not None
                and beat.j_point.confidence >= 0.35
                or beat.r_offset is not None
                and beat.r_offset.confidence >= 0.35
            )
        )
        t_evaluable = delineation.segment_confidence("t") >= measurement_confidence_threshold("t")
        baseline = _baseline_by_lead(ecg, beat, fs)
        qrs = _qrs_amplitudes(ecg, beat, baseline, fs)
        if extended_measurements and qrs_evaluable and beat.r_onset is not None and beat.r_offset is not None:
            from .lab_qrs_timing import simple_qrs_timing
            a, b = beat.r_onset.sample, beat.r_offset.sample
            if 0 <= a < b < len(ecg):
                for lead in LEAD_NAMES_12:
                    index = LEAD_MAP[lead]
                    qrs[lead].update(simple_qrs_timing(ecg[a:b+1, index]-baseline[index], fs))
        p_wave = _p_wave_metrics(ecg, beat, baseline, fs)
        rr_seconds = ventricular_rr_seconds(
            (item.r_peak.sample for item in delineation.beats), fs
        )
        rr_ms = None if rr_seconds is None else rr_seconds * 1000.0
        st = _st_t_metrics(
            ecg,
            beat,
            baseline,
            fs,
            qrs=qrs,
            rr_ms=rr_ms,
        )
        if measurement_sink is not None:
            from .lead_measurements import store_lead_measurements

            def measured_window(start, end) -> bool:
                return (start is not None and end is not None
                        and 0 <= start.sample < end.sample < len(ecg))

            qrs_measured = qrs_evaluable and measured_window(beat.r_onset, beat.r_offset)
            p_measured = p_evaluable and measured_window(beat.p_onset, beat.p_offset) and beat.resolved_wave_status("p").value == "present"
            t_measured = t_evaluable and measured_window(beat.t_onset, beat.t_offset) and beat.resolved_wave_status("t").value == "present"
            j_marker = beat.j_point if beat.j_point is not None else beat.r_offset
            j_measured = (j_marker is not None and j_marker.confidence >= .35
                          and 0 <= j_marker.sample < len(ecg))
            store_lead_measurements(measurement_sink, ecg,
                {"qrs": qrs, "p": p_wave, "st": st, "t": st},
                {"qrs": qrs_measured, "p": p_measured, "st": qrs_measured and st_evaluable and j_measured, "t": t_measured},
                extended=extended_measurements)
            if extended_measurements:
                from .lead_measurements import measurement_column
                # Existing primitives clip out-of-record sample positions. Such
                # endpoints must not masquerade as measured J+20/60/80 values.
                for offset in (20, 60, 80):
                    if j_marker is None or not (0 <= j_marker.sample + offset * fs / 1000 < len(ecg)):
                        for lead in LEAD_NAMES_12:
                            measurement_sink[measurement_column(lead, "st", f"st{offset}", "mv")] = np.nan
        if measurements_only:
            return {}
        q_metrics = {
            lead: _initial_q_metric(ecg[:, LEAD_MAP[lead]], beat, baseline[LEAD_MAP[lead]], fs)
            for lead in LEAD_NAMES_12
        }

        results: dict[str, ClassicalCriterion] = {}
        # Audit 5.1: the diffuse ST pattern left the lab. It was computed only
        # with ``extended_measurements`` (the ECG lab), so no extracted row
        # ever carried it; it reads nothing the standard ST metrics lack.
        from .ischemia_patterns import diffuse_st_pattern
        diffuse = diffuse_st_pattern(st, quality_ok=bool(st_evaluable and qrs_evaluable
            and np.all(np.isfinite(baseline))), excluded=bool(paced_or_lbbb))
        results['lab_diffuse_st_depression_avr_v1_pattern'] = _criterion(
            'lab_diffuse_st_depression_avr_v1_pattern',
            float(len(diffuse['depressed_leads'])) if diffuse['evaluable'] else None,
            diffuse['positive'], 'leads',
            'Lab morphology screen: J depression >=0.1 mV in >=6 leads with J elevation >0 in aVR or V1; '
            'J-point operationalization, elevation has no source-defined minimum: inspect noise/baseline. '
            'Not proof of left-main obstruction or ACS. Source: ' + diffuse['source']
            + '; depressed=' + ','.join(diffuse['depressed_leads'])
            + '; elevated=' + ','.join(diffuse['elevated_leads']),
            diffuse['reason'], exploratory=True)
        from .ischemia_patterns import modified_sgarbossa
        smith = modified_sgarbossa(qrs, st, context=ischemia_context,
            quality_ok=bool(st_evaluable and qrs_evaluable
                and beat.r_onset is not None and beat.r_offset is not None
                and np.all(np.isfinite(baseline))))
        smith_id = 'sgarbossa_smith_modified_pacing' if ischemia_context == 'ventricular_pacing' else 'sgarbossa_smith_modified_lbbb'
        results[smith_id] = _criterion(
            smith_id,
            float(len(smith['triggered'])) if smith['evaluable'] else None,
            smith['positive'], 'triggered_components',
            'Modified Sgarbossa morphology; context=' + str(ischemia_context) + '; '
            'not an MI diagnosis or treatment recommendation. Source: ' + smith['source']
            + '; triggered: ' + ', '.join(f"{x['lead']}:{x['component']}" for x in smith['triggered']),
            smith['reason'])
        results.update(
            self._st_criteria(
                st,
                age=age,
                sex=sex,
                excluded=paced_or_lbbb or not st_evaluable,
                t_evaluable=t_evaluable,
                exclusion_reason=(
                    ""
                    if st_evaluable
                    else "ST/T boundaries below confidence threshold"
                ),
            )
        )
        results.update(
            self._lvh_criteria(
                qrs,
                beat,
                fs,
                sex,
                excluded=paced_or_lbbb or not qrs_evaluable,
                exclusion_reason=(
                    ""
                    if qrs_evaluable
                    else "QRS boundaries below confidence threshold"
                ),
            )
        )
        results.update(
            self._scar_criteria(
                qrs,
                q_metrics,
                excluded=paced_or_lbbb or not qrs_evaluable,
                exclusion_reason=(
                    ""
                    if qrs_evaluable
                    else "QRS boundaries below confidence threshold"
                ),
            )
        )
        results.update(
            self._rvh_criteria(
                qrs,
                q_metrics,
                p_wave,
                st,
                machine_qrs_axis_deg,
                excluded=paced_or_lbbb or not qrs_evaluable,
                exclusion_reason=(
                    ""
                    if qrs_evaluable
                    else "QRS boundaries below confidence threshold"
                ),
            )
        )
        results.update(self._atrial_criteria(p_wave))
        results.update(
            self._repolarization_criteria(delineation, beat, fs, st)
        )
        results.update(
            self._electrolyte_heuristics(
                qrs,
                p_wave,
                beat,
                fs,
                st,
                qrs_evaluable=qrs_evaluable,
                p_evaluable=p_evaluable,
                t_evaluable=t_evaluable,
            )
        )
        results.update(
            self._conduction_criteria(
                ecg,
                qrs,
                q_metrics,
                beat,
                baseline,
                fs,
                paced=paced_or_lbbb if paced is None else paced,
                qrs_evaluable=qrs_evaluable,
                p_evaluable=p_evaluable,
            )
        )
        results.update(
            self._reference_scores(
                ecg,
                qrs,
                q_metrics,
                p_wave,
                st,
                beat,
                baseline,
                fs,
                rr_ms=rr_ms,
                machine_qrs_axis_deg=machine_qrs_axis_deg,
                machine_t_axis_deg=machine_t_axis_deg,
                paced_or_lbbb=paced_or_lbbb,
                qrs_evaluable=qrs_evaluable,
                p_evaluable=p_evaluable,
                t_evaluable=t_evaluable,
                st_evaluable=st_evaluable,
            )
        )
        return results

    def _st_criteria(
        self,
        metrics: dict[str, dict[str, float]],
        *,
        age: Optional[int],
        sex: Optional[str],
        excluded: bool,
        t_evaluable: bool = True,
        exclusion_reason: str = "",
    ) -> dict[str, ClassicalCriterion]:
        if excluded:
            reason = exclusion_reason or (
                "paced rhythm or LBBB requires a separate ischemia framework"
            )
            criteria = {
                key: _criterion(
                    key, None, None, "mV", definition, reason
                )
                for key, definition in (
                    ("st_elevation_contiguous", "J-point elevation in two contiguous leads"),
                    ("st_depression_contiguous", "ST depression in two contiguous leads"),
                    ("t_inversion_contiguous", "T inversion in two contiguous leads"),
                    ("posterior_mi_pattern", "ST depression in V1-V3 posterior pattern"),
                    (
                        "wellens_ecg_pattern",
                        "Anterior Wellens morphology screen",
                    ),
                    (
                        "de_winter_ecg_pattern",
                        "Precordial de Winter morphology screen",
                    ),
                )
            }
            for key, definition in (
                (
                    "st_elevation_contiguous_mm",
                    "J-point elevation in two contiguous leads (mm, 10 mm/mV)",
                ),
                (
                    "st_depression_contiguous_mm",
                    "ST depression in two contiguous leads (mm, 10 mm/mV)",
                ),
            ):
                criteria[key] = _criterion(key, None, None, "mm", definition, reason)
            return criteria

        sex_normalized = str(sex or "").strip().upper()
        v2v3_threshold: Optional[float] = None
        if sex_normalized.startswith("M") and age is not None:
            v2v3_threshold = 0.25 if age < 40 else 0.20
        elif sex_normalized.startswith("F"):
            v2v3_threshold = 0.15
        thresholds: dict[str, Optional[float]] = {
            lead: 0.10 for lead in LEAD_NAMES_12
        }
        thresholds["V2"] = v2v3_threshold
        thresholds["V3"] = v2v3_threshold
        elevated = {
            lead: (
                metrics[lead]["j"] >= thresholds[lead]
                if thresholds[lead] is not None
                else None
            )
            for lead in LEAD_NAMES_12
        }
        depressed = {
            lead: (
                metrics[lead]["st60"] <= -0.05
                and metrics[lead]["st80"] <= metrics[lead]["st20"] + 0.02
            )
            for lead in LEAD_NAMES_12
        }
        inverted = {
            lead: _t_inversion_rs_component(metrics[lead])
            for lead in LEAD_NAMES_12
        }
        elevation_positive = _has_contiguous_pair_nullable(elevated)
        depression_positive = _has_contiguous_pair(depressed)
        inversion_positive = _has_contiguous_pair_nullable(
            inverted,
            groups=T_INVERSION_GROUPS,
        )
        posterior_depression = (
            sum(depressed[lead] for lead in ("V1", "V2", "V3")) >= 2
        )
        posterior_r = max(
            metrics[lead]["r_to_s_ratio"] for lead in ("V1", "V2")
        ) > 1.0
        posterior_upright_t = max(
            metrics[lead]["t_peak"] for lead in ("V1", "V2", "V3")
        ) > 0
        posterior = (
            posterior_depression and posterior_r and posterior_upright_t
        )
        wellens_leads = ("V2", "V3", "V4")
        wellens_flags = {
            lead: (
                abs(metrics[lead]["j"]) < 0.10
                and (
                    metrics[lead]["t_min"] <= -0.20
                    or bool(metrics[lead]["t_positive_then_negative"])
                )
            )
            for lead in wellens_leads
        }
        wellens = sum(wellens_flags.values()) >= 2
        de_winter_leads = ("V2", "V3", "V4", "V5", "V6")
        de_winter_flags = {
            lead: (
                metrics[lead]["j"] <= -0.10
                and metrics[lead]["st80"] - metrics[lead]["j"] >= 0.05
                and (
                    metrics[lead]["t_max"] >= 0.50
                    or metrics[lead]["t_to_r_ratio"] >= 0.75
                )
            )
            for lead in de_winter_leads
        }
        de_winter = sum(de_winter_flags.values()) >= 2
        results = {
            "st_elevation_contiguous": _criterion(
                "st_elevation_contiguous",
                max(metrics[lead]["j"] for lead in LEAD_NAMES_12),
                elevation_positive,
                "mV",
                "J-point elevation in >=2 contiguous leads; V2-V3 age/sex thresholds",
                (
                    "age/sex unavailable for the V2-V3 threshold"
                    if elevation_positive is None
                    else ""
                ),
            ),
            "st_depression_contiguous": _criterion(
                "st_depression_contiguous",
                min(metrics[lead]["st60"] for lead in LEAD_NAMES_12),
                depression_positive,
                "mV",
                "horizontal/downsloping J+60 ms depression <=-0.05 mV "
                "in >=2 contiguous leads",
            ),
            "t_inversion_contiguous": _criterion(
                "t_inversion_contiguous",
                min(metrics[lead]["t_peak"] for lead in LEAD_NAMES_12),
                inversion_positive,
                "mV",
                "UDMI-2018 R/S arm: T-wave amplitude <-0.10 mV and R/S >1 "
                "in >=2 contiguous leads; morphology component, not proof of acute ischemia",
                "T/QRS morphology unavailable" if inversion_positive is None else "",
            ),
            "posterior_mi_pattern": _criterion(
                "posterior_mi_pattern",
                min(metrics[lead]["st60"] for lead in ("V1", "V2", "V3")),
                posterior,
                "mV",
                "ST depression in >=2 of V1-V3 plus dominant anterior R "
                "and upright T; posterior-MI surrogate only",
            ),
            "wellens_ecg_pattern": _criterion(
                "wellens_ecg_pattern",
                float(sum(wellens_flags.values())),
                wellens,
                "lead_count",
                ">=2 of V2-V4 with <0.10 mV J deviation and either "
                "deep negative or positive-then-negative T morphology",
                exploratory=True,
            ),
            "de_winter_ecg_pattern": _criterion(
                "de_winter_ecg_pattern",
                float(sum(de_winter_flags.values())),
                de_winter,
                "lead_count",
                ">=2 of V2-V6 with >=0.10 mV J depression, >=0.05 mV "
                "upsloping recovery by J+80 ms and tall positive T morphology",
                exploratory=True,
            ),
        }
        elevation_mv = results["st_elevation_contiguous"]
        depression_mv = results["st_depression_contiguous"]
        results["st_elevation_contiguous_mm"] = _criterion(
            "st_elevation_contiguous_mm",
            elevation_mv.value * 10.0 if elevation_mv.value is not None else None,
            elevation_mv.positive,
            "mm",
            elevation_mv.definition
            + "; reported in mm at standard 10 mm/mV paper calibration",
            elevation_mv.exclusion_reason,
        )
        results["st_depression_contiguous_mm"] = _criterion(
            "st_depression_contiguous_mm",
            depression_mv.value * 10.0 if depression_mv.value is not None else None,
            depression_mv.positive,
            "mm",
            depression_mv.definition
            + "; reported in mm at standard 10 mm/mV paper calibration",
            depression_mv.exclusion_reason,
        )
        if not t_evaluable:
            for key in (
                "t_inversion_contiguous",
                "posterior_mi_pattern",
                "wellens_ecg_pattern",
                "de_winter_ecg_pattern",
            ):
                prior = results[key]
                results[key] = _criterion(
                    prior.criterion_id,
                    None,
                    None,
                    prior.unit,
                    prior.definition,
                    "T-wave boundaries below confidence threshold",
                    exploratory=prior.exploratory,
                )
        return results

    def _lvh_criteria(
        self,
        qrs: dict[str, dict[str, float]],
        beat: BeatDelineation,
        fs: int,
        sex: Optional[str],
        *,
        excluded: bool,
        exclusion_reason: str = "",
    ) -> dict[str, ClassicalCriterion]:
        if excluded:
            reason = exclusion_reason or (
                "paced rhythm or LBBB confounds voltage-based LVH rules"
            )
            return {
                key: _criterion(key, None, None, unit, definition, reason)
                for key, unit, definition in (
                    (
                        "sokolow_lyon",
                        "mV",
                        "S(V1)+max(R(V5),R(V6)) >=3.5 mV",
                    ),
                    (
                        "cornell_voltage",
                        "mV",
                        "sex-specific Cornell voltage",
                    ),
                    (
                        "cornell_product",
                        "mV*ms",
                        "sex-adjusted Cornell voltage-duration product",
                    ),
                    (
                        "peguero_lo_presti",
                        "mV",
                        "sex-specific Peguero-Lo Presti voltage",
                    ),
                )
            }
        male = str(sex or "").strip().upper().startswith("M")
        female = str(sex or "").strip().upper().startswith("F")
        # Audit S3: a missing lead makes a voltage sum unknown, not small.
        # ``max(1.0, nan)`` is 1.0 in Python, so a missing V6 used to vanish
        # from Sokolow-Lyon silently, and ``nan >= 3.5`` read as negative.
        sokolow = _strict_sum(
            qrs["V1"]["s_depth"],
            _strict_max(qrs["V5"]["r_height"], qrs["V6"]["r_height"]),
        )
        cornell = _strict_sum(qrs["V3"]["s_depth"], qrs["aVL"]["r_height"])
        deepest_s = _strict_max(*(item["s_depth"] for item in qrs.values()))
        peguero = _strict_sum(deepest_s, qrs["V4"]["s_depth"])
        qrs_ms = beat.qrs_duration_ms(fs)
        cornell_adjusted = cornell + (0.6 if female else 0.0)
        cornell_product = (
            cornell_adjusted * qrs_ms
            if qrs_ms is not None and np.isfinite(cornell_adjusted)
            else None
        )
        sex_evaluable = male or female

        def decided(value, rule):
            if value is None or not np.isfinite(value):
                return None
            return bool(rule(value))

        missing = "lead amplitude unavailable"
        return {
            "sokolow_lyon": _criterion(
                "sokolow_lyon", sokolow, decided(sokolow, lambda v: v >= 3.5), "mV",
                "S(V1)+max(R(V5),R(V6)) >=3.5 mV",
                "" if np.isfinite(sokolow) else missing,
            ),
            "cornell_voltage": _criterion(
                "cornell_voltage",
                cornell,
                decided(cornell, lambda v: v > (2.8 if male else 2.0))
                if sex_evaluable else None,
                "mV",
                "R(aVL)+S(V3) >2.8 mV men or >2.0 mV women",
                "sex unavailable" if not sex_evaluable
                else "" if np.isfinite(cornell) else missing,
            ),
            "cornell_product": _criterion(
                "cornell_product",
                cornell_product,
                decided(cornell_product, lambda v: v > 244.0)
                if sex_evaluable else None,
                "mV*ms",
                "(Cornell voltage +0.6 mV in women)*QRS >244 mV*ms",
                "sex unavailable" if not sex_evaluable
                else "" if cornell_product is not None
                else "QRS duration or lead amplitude unavailable",
            ),
            "peguero_lo_presti": _criterion(
                "peguero_lo_presti",
                peguero,
                decided(peguero, lambda v: v >= (2.8 if male else 2.3))
                if sex_evaluable else None,
                "mV",
                "deepest S in any lead + S(V4) >=2.8 mV men/2.3 mV women",
                "sex unavailable" if not sex_evaluable
                else "" if np.isfinite(peguero) else missing,
            ),
        }

    def _scar_criteria(
        self,
        qrs: dict[str, dict[str, float]],
        q_metrics: dict[str, dict[str, float]],
        *,
        excluded: bool,
        exclusion_reason: str = "",
    ) -> dict[str, ClassicalCriterion]:
        if any(not np.isfinite(metric.get(key, np.nan)) for metric in q_metrics.values()
               for key in ("duration_ms", "depth_mv")):
            excluded = True
            exclusion_reason = "Q-wave measurement unavailable: invalid QRS bounds or missing samples"
        if excluded:
            reason = exclusion_reason or (
                "paced rhythm or LBBB confounds Q/QS scar criteria"
            )
            return {
                key: _criterion(key, None, None, unit, definition, reason)
                for key, unit, definition in (
                    (
                        "pathological_q_waves",
                        "lead_count",
                        "UDMI-2018 Q-wave component; not a standalone prior-MI diagnosis",
                    ),
                    (
                        "qs_complexes",
                        "lead_count",
                        "Strict numerical QS screen in >=2 contiguous precordial leads; not a scar diagnosis",
                    ),
                    (
                        "poor_r_wave_progression",
                        "mV",
                        "R(V3)<=0.3 mV",
                    ),
                )
            }
        pathological = {
            lead: (
                (metric["duration_ms"] > 20
                 and metric["depth_mv"] >= Q_MINIMUM_DEPTH_V2_V3_MV)
                if lead in ("V2", "V3") else
                (metric["duration_ms"] >= 30 and metric["depth_mv"] >= 0.10)
                if lead in ("I", "II", "aVL", "aVF", "V4", "V5", "V6") else False
            )
            for lead, metric in q_metrics.items()
        }
        # UDMI 2018 Table 3 Q-wave arm, not the full prior-MI definition.
        # V2/V3 are a separate arm; Q/R alone is not a sufficient criterion.
        # QS morphology remains a separately reported technical screen.
        q_positive = any(pathological.get(lead, False) for lead in ("V2", "V3")) or any(
            sum(pathological.get(lead, False) for lead in group) >= 2
            for group in (("I", "aVL"), ("II", "III", "aVF"), ("V4", "V5", "V6"))
        )
        qs = {
            lead: _strict_qs_component(qrs[lead])
            for lead in ("V1", "V2", "V3", "V4", "V5", "V6")
        }
        qs_positive = _has_contiguous_pair_nullable(qs)
        r_heights = [qrs[f"V{index}"]["r_height"] for index in range(1, 7)]
        poor_progression = r_heights[2] <= 0.30
        return {
            "pathological_q_waves": _criterion(
                "pathological_q_waves",
                float(sum(pathological.values())),
                q_positive,
                "lead_count",
                "UDMI-2018 Q-wave arm: V2/V3 >20 ms and >=0.05 mV visible depth; otherwise >=30 ms AND >=0.10 mV "
                "in >=2 eligible leads of one group; QS and clinical confounders require separate review",
            ),
            "qs_complexes": _criterion(
                "qs_complexes",
                float(sum(value is True for value in qs.values())) if all(value is not None for value in qs.values()) else None,
                qs_positive,
                "lead_count",
                "Strict numerical QS screen: no positive QRS deflection (1e-12 mV numerical tolerance), "
                "negative depth >=0.10 mV in >=2 contiguous precordial leads; "
                "technical amplitude floor, not a clinical QS threshold or scar diagnosis; noise/baseline sensitive",
                "incomplete precordial morphology" if qs_positive is None else "",
            ),
            "poor_r_wave_progression": _criterion(
                "poor_r_wave_progression",
                r_heights[2],
                poor_progression,
                "mV",
                "R(V3)<=0.3 mV; isolated poor-R-progression surrogate",
            ),
        }

    def _rvh_criteria(
        self,
        qrs: dict[str, dict[str, float]],
        q_metrics: dict[str, dict[str, float]],
        p_wave: dict[str, dict[str, float]],
        st: dict[str, dict[str, float]],
        machine_axis: Optional[float],
        *,
        excluded: bool,
        exclusion_reason: str = "",
    ) -> dict[str, ClassicalCriterion]:
        p_fiducials = bool(p_wave["II"]["fiducial_evaluable"])
        p_amplitude = max(
            p_wave[lead]["positive_peak"] for lead in ("II", "III", "aVF")
        )
        p_pulmonale = p_amplitude > 0.25 if p_fiducials else None
        p_result = _criterion(
            "p_pulmonale",
            p_amplitude if p_fiducials else None,
            p_pulmonale,
            "mV",
            "inferior P-wave amplitude >0.25 mV",
            "" if p_fiducials else "P onset/offset unavailable",
        )
        if excluded:
            reason = exclusion_reason or (
                "paced rhythm or LBBB confounds QRS-based right-heart patterns"
            )
            output = {
                key: _criterion(key, None, None, unit, definition, reason)
                for key, unit, definition in (
                    (
                        "right_axis_deviation",
                        "degrees",
                        "QRS axis >90 degrees",
                    ),
                    (
                        "left_axis_deviation",
                        "degrees",
                        "QRS axis <-30 degrees",
                    ),
                    (
                        "dominant_r_v1",
                        "ratio",
                        "R/S ratio in V1 >1",
                    ),
                    (
                        "dominant_s_v5_v6",
                        "mV",
                        "dominant S in V5 or V6",
                    ),
                    (
                        "s1q3t3_pattern",
                        "component_count",
                        "S in I, initial Q in III and inverted T in III",
                    ),
                )
            }
            output["s1q3t3_pattern"] = _criterion(
                "s1q3t3_pattern",
                None,
                None,
                "component_count",
                "S in I, initial Q in III and inverted T in III; "
                "nonspecific right-heart-strain pattern",
                reason,
                exploratory=True,
            )
            output["p_pulmonale"] = p_result
            return output

        rs_v1 = _safe_ratio(qrs["V1"]["r_height"], qrs["V1"]["s_depth"])
        dominant_r_v1 = qrs["V1"]["r_height"] > qrs["V1"]["s_depth"]
        dominant_s_lateral = (
            qrs["V5"]["s_depth"] > qrs["V5"]["r_height"]
            or qrs["V6"]["s_depth"] > qrs["V6"]["r_height"]
        )
        qrs_axis = _frontal_axis_surrogate(qrs, machine_axis)
        axis_reason = "" if qrs_axis is not None else "QRS axis unavailable or net vector numerically indeterminate"
        right_axis = qrs_axis > 90 if qrs_axis is not None else None
        # -90 to -180 is conventionally "extreme axis deviation" / indeterminate,
        # not left axis deviation -- excluded by the lower bound.
        left_axis = -90.0 <= qrs_axis < -30.0 if qrs_axis is not None else None
        s1 = qrs["I"]["s_depth"] >= 0.10
        q3 = (
            q_metrics["III"]["duration_ms"] >= 30.0
            or q_metrics["III"]["depth_mv"] >= 0.10
        )
        t3 = st["III"]["t_min"] <= -0.10
        s1q3t3_components = (s1, q3, t3)
        return {
            "right_axis_deviation": _criterion(
                "right_axis_deviation",
                qrs_axis,
                right_axis,
                "degrees",
                (
                    "QRS axis >90 degrees; machine axis preferred, "
                    "otherwise lead-I/aVF net-amplitude surrogate with 2/sqrt(3) aVF scaling"
                ),
                exclusion_reason=axis_reason,
            ),
            "left_axis_deviation": _criterion(
                "left_axis_deviation",
                qrs_axis,
                left_axis,
                "degrees",
                (
                    "QRS axis -30 to -90 degrees (extreme/indeterminate "
                    "axis below -90 excluded); machine axis preferred, "
                    "otherwise lead-I/aVF net-amplitude surrogate with 2/sqrt(3) aVF scaling"
                ),
                exclusion_reason=axis_reason,
            ),
            "dominant_r_v1": _criterion(
                "dominant_r_v1",
                rs_v1,
                dominant_r_v1,
                "ratio",
                "R/S ratio in V1 >1",
            ),
            "dominant_s_v5_v6": _criterion(
                "dominant_s_v5_v6",
                max(qrs["V5"]["s_depth"], qrs["V6"]["s_depth"]),
                dominant_s_lateral,
                "mV",
                "dominant S in V5 or V6",
            ),
            "s1q3t3_pattern": _criterion(
                "s1q3t3_pattern",
                float(sum(s1q3t3_components)),
                all(s1q3t3_components),
                "component_count",
                "S(I)>=0.10 mV, initial Q(III)>=30 ms or >=0.10 mV, "
                "and negative T(III)<=-0.10 mV; nonspecific "
                "right-heart-strain pattern",
                exploratory=True,
            ),
            "p_pulmonale": p_result,
        }

    def _atrial_criteria(
        self,
        p_wave: dict[str, dict[str, float]],
    ) -> dict[str, ClassicalCriterion]:
        p_fiducials = bool(p_wave["II"]["fiducial_evaluable"])
        duration = p_wave["II"]["duration_ms"]
        ptf_v1 = p_wave["V1"]["terminal_negative_area_mvs"]
        bifid = (
            p_wave["II"]["peak_count"] >= 2 and duration >= 120
            if p_fiducials
            else None
        )
        exclusion = "" if p_fiducials else "P onset/offset unavailable"
        return {
            "p_duration_ge_120": _criterion(
                "p_duration_ge_120",
                duration,
                duration >= 120 if p_fiducials and np.isfinite(duration) else None,
                "ms",
                "P-wave duration >=120 ms",
                exclusion,
            ),
            "p_terminal_force_v1": _criterion(
                "p_terminal_force_v1",
                ptf_v1,
                ptf_v1 <= -0.004 if p_fiducials else None,
                "mV*s",
                "negative terminal P force V1 <=-0.004 mV*s",
                exclusion,
            ),
            "bifid_p_wave": _criterion(
                "bifid_p_wave",
                p_wave["II"]["peak_count"],
                bifid,
                "peak_count",
                "bifid P in lead II with duration >=120 ms",
                exclusion,
            ),
        }

    def _repolarization_criteria(
        self,
        delineation: RecordDelineation,
        beat: BeatDelineation,
        fs: int,
        st: dict[str, dict[str, float]],
    ) -> dict[str, ClassicalCriterion]:
        measurements = delineation.qt_rr_measurements()
        required_pairs = (
            5 if delineation.rhythm.requires_beatwise_rate_correction else 1
        )
        if len(measurements) < required_pairs:
            qtc_bazett = qtc_fridericia = qtc_framingham = jtc = None
            qtc_exclusion = (
                f"requires {required_pairs} valid beatwise QT-RR pairs; "
                f"{len(measurements)} available"
            )
        else:
            qt_values = np.asarray(
                [item.qt_ms for item in measurements],
                dtype=float,
            )
            rr_values = np.asarray(
                [item.preceding_rr_seconds for item in measurements],
                dtype=float,
            )
            bazett_values = qt_values / np.sqrt(rr_values)
            fridericia_values = qt_values / np.cbrt(rr_values)
            framingham_values = qt_values + 154.0 * (1.0 - rr_values)
            qtc_bazett = float(np.median(bazett_values))
            qtc_fridericia = float(np.median(fridericia_values))
            qtc_framingham = float(np.median(framingham_values))
            qrs_values = [item.qrs_ms for item in measurements]
            jtc = (
                float(
                    np.median(
                        fridericia_values
                        - np.asarray(qrs_values, dtype=float)
                    )
                )
                if all(value is not None for value in qrs_values)
                else None
            )
            qtc_exclusion = ""
        tpeak_tend = (
            (beat.t_offset.sample - beat.t_peak.sample) / fs * 1000.0
            if (
                beat.t_peak is not None
                and beat.t_offset is not None
                and beat.segment_confidence("t") >= measurement_confidence_threshold("t")
            )
            else None
        )
        values = {
            "qtc_bazett": qtc_bazett,
            "qtc_fridericia": qtc_fridericia,
            "qtc_framingham": qtc_framingham,
            "jtc_fridericia": jtc,
            "tpeak_tend": tpeak_tend,
        }
        results = {
            name: _criterion(
                name,
                value,
                None,
                "ms",
                {
                    "qtc_bazett": "QT corrected by Bazett",
                    "qtc_fridericia": "QT corrected by Fridericia",
                    "qtc_framingham": "QT corrected by Framingham",
                    "jtc_fridericia": "QTc Fridericia minus QRS duration",
                    "tpeak_tend": "T peak to T end interval",
                }[name],
                (
                    ""
                    if value is not None
                    else (
                        qtc_exclusion
                        if name.startswith(("qtc_", "jtc_"))
                        else "required fiducials unavailable"
                    )
                ),
            )
            for name, value in values.items()
        }
        results["qtc_gt_480"] = _criterion(
            "qtc_gt_480",
            qtc_fridericia,
            qtc_fridericia > 480 if qtc_fridericia is not None else None,
            "ms",
            "own QTc Fridericia >480 ms",
            qtc_exclusion,
        )
        results["qtc_gt_500"] = _criterion(
            "qtc_gt_500",
            qtc_fridericia,
            qtc_fridericia > 500 if qtc_fridericia is not None else None,
            "ms",
            "own QTc Fridericia >500 ms",
            qtc_exclusion,
        )
        return results

    def _electrolyte_heuristics(
        self,
        qrs: dict[str, dict[str, float]],
        p_wave: dict[str, dict[str, float]],
        beat: BeatDelineation,
        fs: int,
        st: dict[str, dict[str, float]],
        *,
        qrs_evaluable: bool,
        p_evaluable: bool,
        t_evaluable: bool,
    ) -> dict[str, ClassicalCriterion]:
        t_to_r = max(
            abs(st[lead]["t_peak"]) / max(qrs[lead]["r_height"], 0.05)
            for lead in ("V2", "V3", "V4")
        )
        qrs_ms = beat.qrs_duration_ms(fs)
        pr_ms = beat.pr_interval_ms(fs)
        p_amplitude = max(
            p_wave[lead]["positive_peak"] for lead in ("II", "V1")
        )
        components = [
            t_to_r >= 0.75,
            qrs_ms is not None and qrs_ms >= 120,
            pr_ms is not None and pr_ms > 200,
            p_amplitude < 0.05,
        ]
        score = float(sum(components))
        peaked_evaluable = qrs_evaluable and t_evaluable
        combined_evaluable = peaked_evaluable and p_evaluable
        return {
            "peaked_t_wave_heuristic": _criterion(
                "peaked_t_wave_heuristic",
                t_to_r if peaked_evaluable else None,
                t_to_r >= 0.75 if peaked_evaluable else None,
                "ratio",
                "precordial T/R ratio >=0.75; exploratory morphology proxy",
                (
                    ""
                    if peaked_evaluable
                    else "QRS or T boundaries below confidence threshold"
                ),
                exploratory=True,
            ),
            "qrs_widening_ge_120": _criterion(
                "qrs_widening_ge_120",
                qrs_ms if qrs_evaluable else None,
                (
                    qrs_ms >= 120
                    if qrs_evaluable and qrs_ms is not None
                    else None
                ),
                "ms",
                "QRS duration >=120 ms; nonspecific supportive pattern",
                "" if qrs_evaluable else "QRS boundaries below confidence threshold",
                exploratory=True,
            ),
            "pr_prolongation_gt_200": _criterion(
                "pr_prolongation_gt_200",
                pr_ms if p_evaluable and qrs_evaluable else None,
                (
                    pr_ms > 200
                    if p_evaluable and qrs_evaluable and pr_ms is not None
                    else None
                ),
                "ms",
                "P-onset to QRS-onset interval >200 ms; nonspecific measurement, "
                "not proof of first-degree AV block without verified 1:1 conduction",
                (
                    ""
                    if p_evaluable and qrs_evaluable
                    else "P or QRS boundaries below confidence threshold"
                ),
                exploratory=True,
            ),
            "p_wave_attenuation": _criterion(
                "p_wave_attenuation",
                p_amplitude if p_evaluable else None,
                p_amplitude < 0.05 if p_evaluable else None,
                "mV",
                "maximum P amplitude in II/V1 <0.05 mV; nonspecific supportive pattern",
                "" if p_evaluable else "P boundaries below confidence threshold",
                exploratory=True,
            ),
            "hyperkalemia_ecg_heuristic": _criterion(
                "hyperkalemia_ecg_heuristic",
                score if combined_evaluable else None,
                score >= 2 if combined_evaluable else None,
                "component_count",
                ">=2 of peaked T, QRS widening, PR prolongation, P attenuation",
                (
                    ""
                    if combined_evaluable
                    else "P, QRS or T boundaries below confidence threshold"
                ),
                exploratory=True,
            ),
        }

    def _conduction_criteria(
        self,
        ecg: np.ndarray,
        qrs: dict[str, dict[str, float]],
        q_metrics: dict[str, dict[str, float]],
        beat: BeatDelineation,
        baseline: np.ndarray,
        fs: int,
        *,
        paced: bool,
        qrs_evaluable: bool,
        p_evaluable: bool,
    ) -> dict[str, ClassicalCriterion]:
        qrs_ms = beat.qrs_duration_ms(fs)
        limb_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
        precordial_leads = ("V1", "V2", "V3", "V4", "V5", "V6")
        if qrs_evaluable:
            limb_low = all(
                qrs[lead]["peak_to_peak"] < 0.5 for lead in limb_leads
            )
            precordial_low = all(
                qrs[lead]["peak_to_peak"] < 1.0 for lead in precordial_leads
            )
            low_voltage: Optional[bool] = limb_low or precordial_low
            min_peak_to_peak: Optional[float] = min(
                qrs[lead]["peak_to_peak"] for lead in LEAD_NAMES_12
            )
        else:
            low_voltage = None
            min_peak_to_peak = None

        results: dict[str, ClassicalCriterion] = {
            "low_voltage_qrs": _criterion(
                "low_voltage_qrs",
                min_peak_to_peak,
                low_voltage,
                "mV",
                "peak-to-peak QRS <0.5 mV in all limb leads, or <1.0 mV "
                "in all precordial leads",
                (
                    ""
                    if qrs_evaluable
                    else "QRS boundaries below confidence threshold"
                ),
            ),
        }

        # Only pacing is excluded here. The recorder's LBBB statement used to
        # exclude these criteria too, so the LBBB pattern was never assessed
        # on the recordings the recorder called LBBB: on 9,873 pilot ECGs it
        # agreed with the recorder once in 51 positives and was blank on all
        # but 9 of the recorder's LBBB. A bundle-branch pattern is what these
        # criteria classify; the LBBB exclusion belongs to the voltage, ST, Q
        # and T rules, which keep it.
        bbb_excluded = paced or not qrs_evaluable
        if bbb_excluded:
            reason = (
                "paced rhythm confounds QRS morphology classification"
                if paced
                else "QRS boundaries below confidence threshold"
            )
            results["rbbb_pattern"] = _criterion(
                "rbbb_pattern",
                None,
                None,
                "component_count",
                "QRS >=120 ms with an rSR'-type terminal notch in V1 "
                "taller than the initial deflection, plus a wide "
                "terminal S in I or V6; morphology surrogate, not a "
                "full vector-loop BBB classification",
                reason,
                exploratory=True,
            )
            results["lbbb_pattern"] = _criterion(
                "lbbb_pattern",
                None,
                None,
                "component_count",
                "QRS >=120 ms with no initial septal q in I/V6, a "
                "predominantly negative QRS in V1, and no distinct "
                "terminal S in V6; morphology surrogate, not a full "
                "vector-loop BBB classification",
                reason,
                exploratory=True,
            )
            return results

        if qrs_ms is None:
            results["rbbb_pattern"] = _criterion(
                "rbbb_pattern", None, None, "component_count",
                "QRS >=120 ms with an rSR'-type terminal notch in V1 "
                "taller than the initial deflection, plus S depth >=0.10 mV "
                "in I or V6; S duration not assessed; morphology surrogate",
                "QRS duration unavailable", exploratory=True,
            )
            results["lbbb_pattern"] = _criterion(
                "lbbb_pattern", None, None, "component_count",
                "QRS >=120 ms with no initial septal q in I/V6, a "
                "predominantly negative QRS in V1, and no distinct "
                "terminal S in V6; morphology surrogate",
                "QRS duration unavailable", exploratory=True,
            )
            return results

        wide_qrs = qrs_ms >= 120
        v1_terminal_taller = _terminal_exceeds_initial_r(
            ecg, beat, "V1", baseline, fs
        )
        # Audit S5 (AHA/ACCF/HRS 2009): the lateral S of RBBB is wide --
        # longer than 40 ms or longer than the R wave -- not merely deep.
        timing = {
            lead: _lead_qrs_timing(ecg, beat, lead, baseline, fs)
            for lead in ("I", "V5", "V6")
        }
        wide_s_leads = [
            lead for lead in ("I", "V6")
            if qrs[lead]["s_depth"] >= 0.10
            and timing[lead] is not None
            and (
                timing[lead]["terminal_s_ms"] > 40.0
                or timing[lead]["terminal_s_ms"] > timing[lead]["r_duration_ms"]
            )
        ]
        s_timing_known = all(timing[lead] is not None for lead in ("I", "V6"))
        s_amplitude_component = bool(wide_s_leads)
        rbbb_components = (
            wide_qrs,
            bool(v1_terminal_taller),
            s_amplitude_component,
        )
        rbbb_decidable = v1_terminal_taller is not None and (
            s_timing_known or s_amplitude_component
        )
        results["rbbb_pattern"] = _criterion(
            "rbbb_pattern",
            float(sum(rbbb_components)),
            all(rbbb_components) if rbbb_decidable else None,
            "component_count",
            "QRS >=120 ms with an rSR'-type terminal notch in V1 "
            "taller than the initial deflection, plus a wide S in I or V6 "
            "(depth >=0.10 mV and duration >40 ms or longer than R); "
            "morphology surrogate, not a full vector-loop BBB classification",
            (
                ""
                if rbbb_decidable
                else "V1 QRS window too short to assess terminal notch"
                if v1_terminal_taller is None
                else "lateral S duration unavailable"
            ),
            exploratory=True,
        )

        no_initial_q = (
            q_metrics["I"]["duration_ms"] < 15.0
            and q_metrics["V6"]["duration_ms"] < 15.0
        )
        v1_predominantly_negative = (
            qrs["V1"]["s_depth"] > qrs["V1"]["r_height"]
        )
        no_terminal_s_v6 = qrs["V6"]["s_depth"] < 0.10
        # Audit S5 (AHA/ACCF/HRS 2009): delayed lateral activation, R-peak
        # time >60 ms in V5 and V6.
        late_lateral_r = (
            None
            if timing["V5"] is None or timing["V6"] is None
            else timing["V5"]["r_peak_time_ms"] > 60.0
            and timing["V6"]["r_peak_time_ms"] > 60.0
        )
        lbbb_components = (
            wide_qrs,
            no_initial_q,
            v1_predominantly_negative,
            no_terminal_s_v6,
            bool(late_lateral_r),
        )
        results["lbbb_pattern"] = _criterion(
            "lbbb_pattern",
            float(sum(lbbb_components)),
            all(lbbb_components) if late_lateral_r is not None else None,
            "component_count",
            "QRS >=120 ms with no initial septal q in I/V6, a "
            "predominantly negative QRS in V1, no distinct terminal S "
            "in V6 and R-peak time >60 ms in V5 and V6; morphology "
            "surrogate, not a full vector-loop BBB classification",
            "" if late_lateral_r is not None else "lateral R-peak time unavailable",
            exploratory=True,
        )
        return results

    def _reference_scores(
        self,
        ecg: np.ndarray,
        qrs: dict[str, dict[str, float]],
        q_metrics: dict[str, dict[str, float]],
        p_wave: dict[str, dict[str, float]],
        st: dict[str, dict[str, float]],
        beat: BeatDelineation,
        baseline: np.ndarray,
        fs: int,
        *,
        rr_ms: Optional[float],
        machine_qrs_axis_deg: Optional[float],
        machine_t_axis_deg: Optional[float],
        paced_or_lbbb: bool,
        qrs_evaluable: bool,
        p_evaluable: bool,
        t_evaluable: bool,
        st_evaluable: bool,
    ) -> dict[str, ClassicalCriterion]:
        """Published reference scores and indices (audit 5.1).

        Every cut-off below is the source's own; where a source gives none,
        the quantity is reported continuously and ``positive`` stays None.
        Point scores with an unmeasurable component are bounded rather than
        guessed: the verdict is given when it cannot change whatever the
        missing components score, and the point value only when every
        component is known.
        """
        results: dict[str, ClassicalCriterion] = {}
        qrs_ms = beat.qrs_duration_ms(fs) if qrs_evaluable else None
        qt_ms = (
            beat.qt_interval_ms(fs) if qrs_evaluable and t_evaluable else None
        )
        tpeak_tend = (
            (beat.t_offset.sample - beat.t_peak.sample) / fs * 1000.0
            if t_evaluable and beat.t_peak is not None and beat.t_offset is not None
            else None
        )

        # T amplitude in aVR. Tan et al. 2008 (PMID 18426436): an upright T
        # in aVR carried a relative risk of 5.0 for cardiovascular death; the
        # cohort excluded QRS >120 ms, atrial fibrillation, WPW and pacing.
        # Rhythm is not known here and stays a context for the analysis.
        tavr = st["aVR"]["t_peak"]
        tavr_reason = (
            "paced rhythm or LBBB (excluded in the source cohort)"
            if paced_or_lbbb
            else "T-wave boundaries below confidence threshold"
            if not t_evaluable
            else "QRS duration unavailable"
            if qrs_ms is None
            else "QRS >120 ms (excluded in the source cohort)"
            if qrs_ms > 120
            else ""
        )
        results["t_amplitude_avr"] = _criterion(
            "t_amplitude_avr",
            None if tavr_reason else tavr,
            None if tavr_reason or not np.isfinite(tavr) else bool(tavr > 0),
            "mV",
            "T amplitude in aVR at the T peak; positive = upright T in aVR "
            "(Tan et al. 2008, PMID 18426436; QRS >120 ms, pacing and LBBB "
            "excluded as in the source)",
            tavr_reason,
        )

        # iCEB (QT/QRS) and Tpeak-Tend/QT: continuous repolarisation indices
        # without a validated cut-off (review: Reynard et al. 2019, PMID
        # 30260143).
        iceb = (
            _safe_ratio(qt_ms, qrs_ms)
            if qt_ms is not None and qrs_ms is not None and qrs_ms > 0
            else float("nan")
        )
        results["iceb_qt_qrs_ratio"] = _criterion(
            "iceb_qt_qrs_ratio",
            iceb,
            None,
            "ratio",
            "index of cardio-electrophysiological balance, own QT / own QRS "
            "duration; continuous, no validated cut-off (Reynard et al. 2019, "
            "PMID 30260143)",
            "" if np.isfinite(iceb) else "QT or QRS boundaries unavailable",
        )
        tpe_qt = (
            _safe_ratio(tpeak_tend, qt_ms)
            if tpeak_tend is not None and qt_ms is not None and qt_ms > 0
            else float("nan")
        )
        results["tpeak_tend_qt_ratio"] = _criterion(
            "tpeak_tend_qt_ratio",
            tpe_qt,
            None,
            "ratio",
            "Tpeak-Tend / QT from the own fiducials; continuous, no validated "
            "cut-off (Reynard et al. 2019, PMID 30260143)",
            "" if np.isfinite(tpe_qt) else "T peak, T end or QT unavailable",
        )

        # Frontal QRS-T angle from the recorder's own axes. ARIC (Zhang et al.
        # 2007, PMID 17719331) found it a substitute for the spatial angle; it
        # used sex-specific percentiles, so no fixed cut-off is applied.
        device_angle = _frontal_axis_difference(machine_qrs_axis_deg, machine_t_axis_deg)
        results["frontal_qrs_t_angle_device"] = _criterion(
            "frontal_qrs_t_angle_device",
            device_angle,
            None,
            "degrees",
            "|recorder QRS axis - recorder T axis| folded to 0-180 degrees "
            "(Zhang et al. 2007, PMID 17719331); continuous",
            "" if device_angle is not None else "recorder QRS or T axis unavailable",
        )

        # 12-lead QRS voltage and its voltage-duration product (Molloy et al.
        # 1992, PMID 1401620). Molloy reported the partition at matched
        # specificity, not as a general cut-off; both stay continuous.
        voltage_sum = (
            _strict_sum(
                *(qrs[lead]["r_height"] + qrs[lead]["s_depth"] for lead in LEAD_NAMES_12)
            )
            if qrs_evaluable
            else float("nan")
        )
        results["qrs_voltage_sum_12lead"] = _criterion(
            "qrs_voltage_sum_12lead",
            voltage_sum,
            None,
            "mV",
            "sum over all 12 leads of the largest positive plus the largest "
            "negative QRS deflection (Molloy et al. 1992, PMID 1401620); continuous",
            "" if np.isfinite(voltage_sum) else "QRS boundaries or lead amplitude unavailable",
        )
        voltage_product = (
            voltage_sum * qrs_ms
            if qrs_ms is not None and np.isfinite(voltage_sum)
            else float("nan")
        )
        results["qrs_voltage_duration_product_12lead"] = _criterion(
            "qrs_voltage_duration_product_12lead",
            voltage_product,
            None,
            "mV*ms",
            "12-lead QRS voltage sum x QRS duration (Molloy et al. 1992, "
            "PMID 1401620); continuous",
            "" if np.isfinite(voltage_product) else "QRS duration or voltage sum unavailable",
        )

        results.update(
            self._romhilt_estes(
                ecg, qrs, p_wave, st, beat, baseline, fs,
                qrs_ms=qrs_ms,
                machine_qrs_axis_deg=machine_qrs_axis_deg,
                paced_or_lbbb=paced_or_lbbb,
                qrs_evaluable=qrs_evaluable,
                t_evaluable=t_evaluable,
                st_evaluable=st_evaluable,
            )
        )
        results.update(
            self._daniel_pe_score(
                ecg, q_metrics, st, beat, baseline, fs,
                qrs_ms=qrs_ms,
                rr_ms=rr_ms,
                paced_or_lbbb=paced_or_lbbb,
                qrs_evaluable=qrs_evaluable,
                t_evaluable=t_evaluable,
            )
        )

        # Macruz index: P duration / PR segment. The index is established as
        # a ratio (e.g. Lu et al. 2025, PMID 40994920); its historical cut-off
        # could not be verified from an accessible source, so it stays
        # continuous. Per-lead P dispersion is not computed: the P boundaries
        # are shared across leads, so every lead would report one duration.
        macruz = float("nan")
        if (
            p_evaluable
            and qrs_evaluable
            and bool(p_wave["II"]["fiducial_evaluable"])
            and beat.p_onset is not None
            and beat.p_offset is not None
            and beat.r_onset is not None
        ):
            p_duration = (beat.p_offset.sample - beat.p_onset.sample) / fs * 1000.0
            pr_segment = (beat.r_onset.sample - beat.p_offset.sample) / fs * 1000.0
            if p_duration > 0 and pr_segment > 0:
                macruz = _safe_ratio(p_duration, pr_segment)
        results["macruz_index"] = _criterion(
            "macruz_index",
            macruz,
            None,
            "ratio",
            "P-wave duration / PR segment (P offset to QRS onset) from the "
            "shared fiducials; continuous",
            "" if np.isfinite(macruz) else "P or QRS boundaries unavailable or PR segment not positive",
        )
        return results

    def _romhilt_estes(
        self,
        ecg: np.ndarray,
        qrs: dict[str, dict[str, float]],
        p_wave: dict[str, dict[str, float]],
        st: dict[str, dict[str, float]],
        beat: BeatDelineation,
        baseline: np.ndarray,
        fs: int,
        *,
        qrs_ms: Optional[float],
        machine_qrs_axis_deg: Optional[float],
        paced_or_lbbb: bool,
        qrs_evaluable: bool,
        t_evaluable: bool,
        st_evaluable: bool,
    ) -> dict[str, ClassicalCriterion]:
        """Romhilt-Estes point score (Romhilt & Estes 1968, PMID 4231231).

        Points as restated by Yang et al. 2022 (PMID 35112886): voltage 3,
        LV strain in V5/V6 3 (without digitalis), P-terminal negativity in V1
        >=0.10 mV and >=40 ms 3, QRS axis <=-30 degrees 2, QRS >=90 ms 1,
        intrinsicoid deflection in V5/V6 >=50 ms 1; LVH at >=5 points.
        Digitalis is not known at extraction; the strain arm scores the
        without-digitalis 3 points, which can only overstate the score.
        """
        definition = (
            "Romhilt-Estes points: voltage 3, V5/V6 strain 3 (digitalis "
            "unknown, scored as absent), P-terminal V1 >=0.10 mV and >=40 ms 3, "
            "axis -30 to -90 degrees 2, QRS >=90 ms 1, R-peak time V5/V6 "
            ">=50 ms 1; positive >=5 (PMID 4231231; PMID 35112886). Strain "
            "operationalised as J+60 ms deviation >=0.05 mV and T >=0.10 mV "
            "opposite to the net QRS"
        )
        if paced_or_lbbb or not qrs_evaluable:
            reason = (
                "paced rhythm or LBBB confounds voltage-based LVH rules"
                if paced_or_lbbb
                else "QRS boundaries below confidence threshold"
            )
            return {
                "romhilt_estes_points": _criterion(
                    "romhilt_estes_points", None, None, "points", definition, reason
                )
            }
        limb = ("I", "II", "III", "aVR", "aVL", "aVF")
        voltage = _any_known(
            [
                _threshold_flag(qrs[lead][key], 2.0)
                for lead in limb
                for key in ("r_height", "s_depth")
            ]
            + [_threshold_flag(qrs[lead]["s_depth"], 3.0) for lead in ("V1", "V2")]
            + [_threshold_flag(qrs[lead]["r_height"], 3.0) for lead in ("V5", "V6")]
        )
        if st_evaluable and t_evaluable:
            strain_flags = []
            for lead in ("V5", "V6"):
                r, s = qrs[lead]["r_height"], qrs[lead]["s_depth"]
                st60, t_value = st[lead]["st60"], st[lead]["t_peak"]
                if not all(np.isfinite(v) for v in (r, s, st60, t_value)) or r == s:
                    strain_flags.append(None)
                    continue
                sign = 1.0 if r > s else -1.0
                strain_flags.append(
                    bool(sign * st60 <= -0.05 and sign * t_value <= -0.10)
                )
            strain = _any_known(strain_flags)
        else:
            strain = None
        p_status = beat.resolved_wave_status("p").value
        if p_status == "absent":
            # No P wave (e.g. atrial fibrillation): the atrial arm cannot be met.
            atrial: Optional[bool] = False
        elif bool(p_wave["V1"]["fiducial_evaluable"]):
            depth, duration = _p_terminal_negativity(ecg, beat, "V1", baseline, fs)
            atrial = (
                None
                if not (np.isfinite(depth) and np.isfinite(duration))
                else bool(depth >= 0.10 and duration >= 40.0)
            )
        else:
            atrial = None
        axis = _frontal_axis_surrogate(qrs, machine_qrs_axis_deg)
        left_axis = None if axis is None else bool(-90.0 <= axis <= -30.0)
        wide = None if qrs_ms is None else bool(qrs_ms >= 90.0)
        timing = [
            _lead_qrs_timing(ecg, beat, lead, baseline, fs) for lead in ("V5", "V6")
        ]
        intrinsicoid = _any_known(
            [
                None if item is None else bool(item["r_peak_time_ms"] >= 50.0)
                for item in timing
            ]
        )
        value, positive = _bounded_points(
            [
                (3, voltage),
                (3, strain),
                (3, atrial),
                (2, left_axis),
                (1, wide),
                (1, intrinsicoid),
            ],
            threshold=5,
        )
        return {
            "romhilt_estes_points": _criterion(
                "romhilt_estes_points",
                value,
                positive,
                "points",
                definition,
                "" if value is not None or positive is not None
                else "a component is unmeasurable and could decide the score",
            )
        }

    def _daniel_pe_score(
        self,
        ecg: np.ndarray,
        q_metrics: dict[str, dict[str, float]],
        st: dict[str, dict[str, float]],
        beat: BeatDelineation,
        baseline: np.ndarray,
        fs: int,
        *,
        qrs_ms: Optional[float],
        rr_ms: Optional[float],
        paced_or_lbbb: bool,
        qrs_evaluable: bool,
        t_evaluable: bool,
    ) -> dict[str, ClassicalCriterion]:
        """Daniel ECG score for pulmonary embolism (Daniel et al. 2001,
        PMID 11502646; component table and footnotes as restated by Sekhri
        et al. 2012, PMC3542486, Table III).

        Tachycardia >100/min 2; incomplete RBBB (QRS 100-110 ms, S in I and
        terminal R in V1 >0.15 mV) 2 or complete RBBB (QRS >110 ms, same) 3;
        T inversion in all of V1-V4 4; V1 inversion depth <1 / 1-2 / >2 mm
        0 / 1 / 2; V2 and V3 1 / 2 / 3; S in I 0; Q in III (>0.15 mV) 1;
        inverted T in III 1; S1Q3T3 2. Score >=10 suggests severe pulmonary
        hypertension from PE (97.7 % specific in the source).
        """
        definition = (
            "Daniel ECG score, 21 points (PMID 11502646; component table "
            "PMC3542486): rate >100/min 2, incomplete/complete RBBB 2/3, "
            "T inversion V1-V4 4, graded V1/V2/V3 inversion depth, Q(III) 1, "
            "T(III) inverted 1, S1Q3T3 2; positive >=10. Rhythm is not "
            "verified as sinus; T inversion = dominant negative T deflection"
        )
        if paced_or_lbbb:
            return {
                "daniel_pe_score": _criterion(
                    "daniel_pe_score", None, None, "points", definition,
                    "paced rhythm or LBBB confounds QRS and T morphology",
                )
            }
        tachycardia = (
            None
            if rr_ms is None or not np.isfinite(rr_ms) or rr_ms <= 0
            else bool(60000.0 / rr_ms > 100.0)
        )
        if qrs_evaluable:
            s_lead_i = _flag(_s_after_r_depth(ecg, beat, "I", baseline), lambda v: v > 0.15)
            terminal_r_v1 = _flag(
                _terminal_r_amplitude(ecg, beat, "V1", baseline), lambda v: v > 0.15
            )
            q_depth = q_metrics["III"]["depth_mv"]
            q_lead_iii = _flag(q_depth, lambda v: v > 0.15)
        else:
            s_lead_i = terminal_r_v1 = q_lead_iii = None
        if s_lead_i is False or terminal_r_v1 is False:
            rbbb_points: Optional[int] = 0
        elif s_lead_i is None or terminal_r_v1 is None or qrs_ms is None:
            rbbb_points = None
        else:
            rbbb_points = 3 if qrs_ms > 110.0 else 2 if qrs_ms >= 100.0 else 0
        if t_evaluable:
            depth = {lead: _t_inversion_depth(st[lead]) for lead in ("V1", "V2", "V3", "V4", "III")}
        else:
            depth = {lead: None for lead in ("V1", "V2", "V3", "V4", "III")}

        def graded(lead: str, low: int) -> Optional[int]:
            value = depth[lead]
            if value is None:
                return None
            if value <= 0:
                return 0
            millimetres = value * 10.0
            return low if millimetres < 1.0 else low + 1 if millimetres <= 2.0 else low + 2

        all_anterior = _all_known(
            [None if depth[lead] is None else depth[lead] > 0 for lead in ("V1", "V2", "V3", "V4")]
        )
        t_lead_iii = None if depth["III"] is None else bool(depth["III"] > 0)
        s1q3t3 = _all_known([s_lead_i, q_lead_iii, t_lead_iii])
        components = [
            (_points(tachycardia, 2), 2),
            (rbbb_points, 3),
            (_points(all_anterior, 4), 4),
            (graded("V1", 0), 2),
            (graded("V2", 1), 3),
            (graded("V3", 1), 3),
            (_points(q_lead_iii, 1), 1),
            (_points(t_lead_iii, 1), 1),
            (_points(s1q3t3, 2), 2),
        ]
        value, positive = _bounded_graded_points(components, threshold=10)
        return {
            "daniel_pe_score": _criterion(
                "daniel_pe_score",
                value,
                positive,
                "points",
                definition,
                "" if value is not None or positive is not None
                else "a component is unmeasurable and could decide the score",
            )
        }

    @staticmethod
    def _unavailable(reason: str) -> dict[str, ClassicalCriterion]:
        return {
            "classical_ecg_unavailable": _criterion(
                "classical_ecg_unavailable",
                None,
                None,
                "",
                "No classical criteria evaluable",
                reason,
            )
        }


CONTIGUOUS_GROUPS = (
    ("II", "III"),
    ("II", "aVF"),
    ("III", "aVF"),
    ("I", "aVL"),
    ("I", "V5"),
    ("I", "V6"),
    ("aVL", "V5"),
    ("aVL", "V6"),
    ("V5", "V6"),
    ("V1", "V2"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "V5"),
    ("V5", "V6"),
)

T_INVERSION_GROUPS = (
    ("II", "III"),
    ("II", "aVF"),
    ("III", "aVF"),
    ("I", "aVL"),
    ("I", "V5"),
    ("I", "V6"),
    ("aVL", "V5"),
    ("aVL", "V6"),
    ("V5", "V6"),
    ("V2", "V3"),
    ("V3", "V4"),
    ("V4", "V5"),
    ("V5", "V6"),
)


def _baseline_by_lead(
    ecg: np.ndarray,
    beat: BeatDelineation,
    fs: int,
) -> np.ndarray:
    if (
        beat.p_offset is not None
        and beat.r_onset is not None
        and beat.resolved_wave_status("p").value == "present"
        and beat.marker_confidence("p_offset", "r_onset") >= 0.35
    ):
        start = beat.p_offset.sample + int(0.01 * fs)
        end = beat.r_onset.sample - int(0.01 * fs)
    else:
        qrs_onset = (
            beat.r_onset.sample
            if beat.r_onset is not None
            else beat.r_peak.sample - int(0.04 * fs)
        )
        start = qrs_onset - int(0.08 * fs)
        end = qrs_onset - int(0.02 * fs)
    start = max(0, start)
    end = min(ecg.shape[0], max(start + 1, end))
    return np.median(ecg[start:end], axis=0)


def _qrs_amplitudes(
    ecg: np.ndarray,
    beat: BeatDelineation,
    baseline: np.ndarray,
    fs: int,
) -> dict[str, dict[str, float]]:
    start = (
        beat.r_onset.sample
        if beat.r_onset is not None
        else beat.r_peak.sample - int(0.06 * fs)
    )
    end = (
        beat.r_offset.sample
        if beat.r_offset is not None
        else beat.r_peak.sample + int(0.08 * fs)
    )
    start = max(0, start)
    end = min(ecg.shape[0], max(start + 1, end))
    output = {}
    for lead in LEAD_NAMES_12:
        index = LEAD_MAP[lead]
        segment = ecg[start:end, index] - baseline[index]
        if segment.size == 0 or not np.all(np.isfinite(segment)):
            output[lead] = {key: float("nan") for key in ("r_height", "s_depth", "peak_to_peak")}
            continue
        output[lead] = {
            "r_height": float(max(0.0, np.max(segment))),
            "s_depth": float(max(0.0, -np.min(segment))),
            "peak_to_peak": float(np.ptp(segment)),
        }
    return output


def _p_wave_metrics(
    ecg: np.ndarray,
    beat: BeatDelineation,
    baseline: np.ndarray,
    fs: int,
) -> dict[str, dict[str, float]]:
    fiducial_evaluable = (
        beat.p_onset is not None
        and beat.p_offset is not None
        and beat.p_offset.sample > beat.p_onset.sample
        and beat.resolved_wave_status("p").value == "present"
        and beat.segment_confidence("p") >= measurement_confidence_threshold("p")
    )
    if fiducial_evaluable:
        start, end = beat.p_onset.sample, beat.p_offset.sample
    else:
        qrs_onset = (
            beat.r_onset.sample
            if beat.r_onset is not None
            else beat.r_peak.sample - int(0.05 * fs)
        )
        start, end = qrs_onset - int(0.16 * fs), qrs_onset - int(0.04 * fs)
    start = max(0, start)
    end = min(ecg.shape[0], max(start + 1, end))
    duration_ms = (
        (end - start) / fs * 1000.0
        if fiducial_evaluable
        else float("nan")
    )
    output = {}
    for lead in LEAD_NAMES_12:
        index = LEAD_MAP[lead]
        segment = ecg[start:end, index] - baseline[index]
        terminal = segment[-min(len(segment), int(0.04 * fs)):]
        peaks, _ = _find_prominent_peaks(segment)
        output[lead] = {
            "positive_peak": float(max(0.0, np.max(segment))),
            "negative_peak": float(min(0.0, np.min(segment))),
            "duration_ms": float(duration_ms),
            "terminal_negative_area_mvs": float(
                np.sum(np.minimum(terminal, 0.0)) / fs
            ),
            "peak_count": float(len(peaks)),
            "fiducial_evaluable": float(fiducial_evaluable),
        }
    return output


def _st_t_metrics(
    ecg: np.ndarray,
    beat: BeatDelineation,
    baseline: np.ndarray,
    fs: int,
    qrs: dict[str, dict[str, float]],
    rr_ms: Optional[float] = None,
) -> dict[str, dict[str, float]]:
    j = (
        beat.j_point.sample
        if beat.j_point is not None
        else beat.r_offset.sample
        if beat.r_offset is not None
        else beat.r_peak.sample + int(0.08 * fs)
    )
    j = int(np.clip(j, 0, ecg.shape[0] - 1))
    st20_index = int(np.clip(j + 0.02 * fs, 0, ecg.shape[0] - 1))
    st60_index = int(np.clip(j + 0.06 * fs, 0, ecg.shape[0] - 1))
    st80_index = int(np.clip(j + 0.08 * fs, 0, ecg.shape[0] - 1))
    t_peak = (
        beat.t_peak.sample
        if beat.t_peak is not None
        else int(np.clip(j + 0.20 * fs, 0, ecg.shape[0] - 1))
    )
    t_peak = int(np.clip(t_peak, 0, ecg.shape[0] - 1))
    t_start = (
        beat.t_onset.sample
        if beat.t_onset is not None
        else j + int(0.08 * fs)
    )
    t_end = (
        beat.t_offset.sample + 1
        if beat.t_offset is not None
        else t_peak + int(0.12 * fs) + 1
    )
    t_start = int(np.clip(t_start, 0, ecg.shape[0] - 1))
    t_end = int(np.clip(max(t_start + 1, t_end), 1, ecg.shape[0]))
    output = {}
    for lead in LEAD_NAMES_12:
        index = LEAD_MAP[lead]
        t_segment = ecg[t_start:t_end, index] - baseline[index]
        t_max = float(np.max(t_segment))
        t_min = float(np.min(t_segment))
        t_positive_then_negative = (
            t_max >= 0.05
            and t_min <= -0.05
            and int(np.argmax(t_segment)) < int(np.argmin(t_segment))
        )
        output[lead] = {
            "j": float(ecg[j, index] - baseline[index]),
            "st20": float(ecg[st20_index, index] - baseline[index]),
            "st60": float(ecg[st60_index, index] - baseline[index]),
            "st80": float(ecg[st80_index, index] - baseline[index]),
            "t_peak": float(ecg[t_peak, index] - baseline[index]),
            "t_max": t_max,
            "t_min": t_min,
            "t_positive_then_negative": float(t_positive_then_negative),
            "t_to_r_ratio": _safe_ratio(
                t_max,
                max(qrs[lead]["r_height"], 0.05),
            ),
            "r_to_s_ratio": _safe_ratio(
                qrs[lead]["r_height"],
                qrs[lead]["s_depth"],
            ),
            "rr_ms": rr_ms,
        }
    return output


def _initial_q_metric(
    lead: np.ndarray,
    beat: BeatDelineation,
    baseline: float,
    fs: int,
) -> dict[str, float]:
    # A Q is the initial negative deflection, not any negative run preceding
    # a multilead R trigger. A positive initial r followed by S is NOT a Q.
    if (beat.r_onset is None or beat.r_offset is None or not np.isfinite(fs) or fs <= 0
            or not 0 <= beat.r_onset.sample < beat.r_offset.sample < len(lead)):
        return {"duration_ms": float("nan"), "depth_mv": float("nan"), "q_to_r_ratio": float("nan")}
    segment = np.asarray(lead[beat.r_onset.sample:beat.r_offset.sample+1]) - baseline
    if not np.all(np.isfinite(segment)):
        return {"duration_ms": float("nan"), "depth_mv": float("nan"), "q_to_r_ratio": float("nan")}
    # Audit S4: the segment starts at the *global* QRS onset. In a lead that
    # depolarises later, isoelectric samples with a slightly negative
    # baseline residue became a "Q wave" at the 1e-12 tolerance (14.3 % of
    # ECGs Q-positive). The lead's own onset is its first departure of
    # Q_LOCAL_ONSET_MV (four 5-uV quantisation steps); from there the Q is
    # extended backwards only over clearly negative samples.
    departed = np.flatnonzero(np.abs(segment) >= Q_LOCAL_ONSET_MV)
    q_segment = np.empty(0)
    r_height = float(max(0.0, np.max(segment)))
    initial_r = len(departed) and bool(
        np.any(segment[: int(departed[0])] >= Q_BACKWARD_EXTENSION_MV)
    )
    if (
        len(departed)
        and segment[departed[0]] < 0
        and not initial_r
        and r_height > 1e-12
    ):
        # An initial positive deflection of two quantisation steps before the
        # negative departure is an r wave, and an r followed by S is no Q.
        start = int(departed[0])
        while start > 0 and segment[start - 1] < -Q_BACKWARD_EXTENSION_MV:
            start -= 1
        stops = np.flatnonzero(segment[start:] >= 0)
        end = start+int(stops[0]) if len(stops) else len(segment)
        q_segment = segment[start:end]
    duration_samples = len(q_segment)
    q_depth = float(-np.min(q_segment)) if duration_samples else 0.
    return {
        "duration_ms": duration_samples / fs * 1000.0,
        "depth_mv": q_depth,
        "q_to_r_ratio": _safe_ratio(q_depth, r_height),
    }


def _terminal_exceeds_initial_r(
    ecg: np.ndarray,
    beat: BeatDelineation,
    lead: str,
    baseline: np.ndarray,
    fs: int,
) -> Optional[bool]:
    """rSR'-type morphology proxy: does the terminal half of the QRS in
    `lead` show a higher positive peak than the initial half (a second,
    taller R' after the initial r), the classic RBBB V1 signature. A
    minimum absolute-amplitude floor avoids a noise-driven false positive
    on a flat/tiny signal. Returns None if the QRS window is too short
    to split into two halves."""
    if beat.r_onset is None or beat.r_offset is None:
        return None
    start = max(0, beat.r_onset.sample)
    end = min(ecg.shape[0], beat.r_offset.sample)
    index = LEAD_MAP[lead]
    segment = ecg[start:end, index] - baseline[index]
    if segment.size < 6:
        return None
    midpoint = segment.size // 2
    initial_peak = float(np.max(segment[:midpoint]))
    terminal_peak = float(np.max(segment[midpoint:]))
    return terminal_peak > initial_peak and terminal_peak >= 0.05


def _lead_qrs_timing(
    ecg: np.ndarray,
    beat: BeatDelineation,
    lead: str,
    baseline: np.ndarray,
    fs: int,
) -> Optional[dict[str, float]]:
    """R-peak time, R duration and terminal S duration of one lead (ms).

    R-peak time runs from the global QRS onset to the lead's largest
    positive deflection. The R wave is the positive run around that peak;
    the terminal S is the negative run that follows it up to the QRS
    offset. None when the window is invalid or the lead has no R.
    """
    if (
        beat.r_onset is None or beat.r_offset is None
        or not np.isfinite(fs) or fs <= 0
    ):
        return None
    start = max(0, beat.r_onset.sample)
    end = min(ecg.shape[0], beat.r_offset.sample + 1)
    index = LEAD_MAP[lead]
    segment = np.asarray(ecg[start:end, index], dtype=float) - baseline[index]
    if segment.size < 6 or not np.all(np.isfinite(segment)):
        return None
    peak = int(np.argmax(segment))
    if segment[peak] <= 0.05:
        return None
    left = peak
    while left > 0 and segment[left - 1] > 0:
        left -= 1
    right = peak
    while right + 1 < segment.size and segment[right + 1] > 0:
        right += 1
    tail = segment[right + 1:]
    negative = np.flatnonzero(tail < 0)
    if negative.size:
        s_start = int(negative[0])
        positive_after = np.flatnonzero(tail[s_start:] >= 0)
        s_end = s_start + (int(positive_after[0]) if positive_after.size else tail.size - s_start)
        terminal_s = s_end - s_start
    else:
        terminal_s = 0
    to_ms = 1000.0 / fs
    return {
        "r_peak_time_ms": peak * to_ms,
        "r_duration_ms": (right - left + 1) * to_ms,
        "terminal_s_ms": terminal_s * to_ms,
    }


def _strict_qs_component(metrics: dict[str, float]) -> Optional[bool]:
    """Conservative numerical screen, not a noise-validated clinical QS detector."""
    r, s = metrics.get("r_height"), metrics.get("s_depth")
    if any(not isinstance(value, (int, float, np.number)) or isinstance(value, (bool, np.bool_))
           or not np.isfinite(value) or value < 0 for value in (r, s)):
        return None
    return bool(r <= 1e-12 and s >= 0.10)


def _has_contiguous_pair(
    flags: dict[str, bool],
    *,
    groups: tuple[tuple[str, ...], ...] = CONTIGUOUS_GROUPS,
) -> bool:
    for group in groups:
        if len(group) >= 2 and all(bool(flags.get(lead, False)) for lead in group):
            return True
    return False


def _has_contiguous_pair_nullable(
    flags: dict[str, Optional[bool]],
    *,
    groups: tuple[tuple[str, ...], ...] = CONTIGUOUS_GROUPS,
) -> Optional[bool]:
    indeterminate = False
    for group in groups:
        if len(group) < 2:
            continue
        values = tuple(flags.get(lead, False) for lead in group)
        if all(value is True for value in values):
            return True
        if all(value is not False for value in values):
            indeterminate = True
    return None if indeterminate else False


def _t_inversion_rs_component(metrics):
    """Quantitative R/S arm of UDMI-2018 Table 2, not the unquantified prominent-R arm."""
    t_peak = metrics.get('t_peak', float('nan'))
    ratio = metrics.get('r_to_s_ratio', float('nan'))
    if not np.isfinite(t_peak):
        return None
    if t_peak >= -0.10:
        return False
    if not np.isfinite(ratio) or ratio < 0:
        return None
    return bool(ratio > 1)


def _frontal_axis_surrogate(qrs, machine_axis=None):
    """Corrected I/aVF amplitude surrogate, not an integrated QRS axis.

    Geometry: PMID 9933900. The numerical zero tolerance is not a
    validated clinical low-voltage or axis-confidence threshold.
    Invalid supplied machine measurements abstain rather than silently
    replacing the advertised preferred measurement with another method.
    """
    if machine_axis is not None:
        return float(machine_axis) if np.isfinite(machine_axis) and -180 <= machine_axis <= 180 else None
    x = qrs['I']['r_height'] - qrs['I']['s_depth']
    y = (2 / np.sqrt(3)) * (qrs['aVF']['r_height'] - qrs['aVF']['s_depth'])
    if not np.isfinite(x) or not np.isfinite(y) or np.hypot(x, y) <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(y, x)))


def _criterion(
    criterion_id: str,
    value: Optional[float],
    positive: Optional[bool],
    unit: str,
    definition: str,
    exclusion_reason: str = "",
    exploratory: bool = False,
) -> ClassicalCriterion:
    finite_value = (
        float(value)
        if value is not None and np.isfinite(value)
        else None
    )
    evaluable = (
        (positive is not None or finite_value is not None)
        and not exclusion_reason
    )
    return ClassicalCriterion(
        criterion_id=criterion_id,
        value=finite_value,
        positive=bool(positive) if positive is not None else None,
        evaluable=evaluable,
        unit=unit,
        definition=definition,
        exclusion_reason=exclusion_reason,
        exploratory=exploratory,
    )


def _strict_sum(*values: float) -> float:
    """Sum that is NaN when any component is not finite (audit S3)."""
    if any(value is None or not np.isfinite(value) for value in values):
        return float("nan")
    return float(sum(values))


def _strict_max(*values: float) -> float:
    """Maximum that is NaN when any component is not finite (audit S3)."""
    if any(value is None or not np.isfinite(value) for value in values):
        return float("nan")
    return float(max(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """An amplitude ratio that cannot leave its declared magnitude.

    The old guard divided by ``max(abs(d), 1e-9)``, which bounds the divisor
    and not the quotient: an S wave of 2e-9 mV in V1 produced
    ``classical_dominant_r_v1`` = 1.260.002.535,94 in the current store. See
    ``core.marker_bounds`` for why the cap is 50 and why capping is preferred
    to abstaining here -- a dominant R wave with no S wave is a finding, not a
    missing value, and saturating keeps it at the top of the ranking.

    The denominator's magnitude is used, as before, so the sign of the result
    is the numerator's. Only the magnitude behaviour changes.
    """

    # Audit S3: a missing amplitude is unknown, not a ratio of zero; the R/S
    # arm of the T-inversion rule turned that zero into a negative.
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    value = float(numerator) / max(abs(float(denominator)), 1e-9)
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, -RATIO_LIMIT, RATIO_LIMIT))


def _frontal_axis_difference(
    qrs_axis: Optional[float], t_axis: Optional[float]
) -> Optional[float]:
    """Unsigned angle between two frontal axes, folded to 0-180 degrees."""
    values = []
    for value in (qrs_axis, t_axis):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(number) or not -180.0 <= number <= 360.0:
            return None
        values.append(number)
    difference = abs(values[0] - values[1]) % 360.0
    return float(360.0 - difference if difference > 180.0 else difference)


def _threshold_flag(value: float, threshold: float) -> Optional[bool]:
    """``value >= threshold``, or None when the value is not finite."""
    if value is None or not np.isfinite(value):
        return None
    return bool(value >= threshold)


def _flag(value: Optional[float], rule) -> Optional[bool]:
    if value is None or not np.isfinite(value):
        return None
    return bool(rule(value))


def _any_known(flags: list[Optional[bool]]) -> Optional[bool]:
    """True if any flag is True, False if all are False, else None."""
    if any(flag is True for flag in flags):
        return True
    if flags and all(flag is False for flag in flags):
        return False
    return None


def _all_known(flags: list[Optional[bool]]) -> Optional[bool]:
    """False if any flag is False, True if all are True, else None."""
    if any(flag is False for flag in flags):
        return False
    if flags and all(flag is True for flag in flags):
        return True
    return None


def _points(flag: Optional[bool], points: int) -> Optional[int]:
    return None if flag is None else (points if flag else 0)


def _bounded_points(
    components: list[tuple[int, Optional[bool]]],
    *,
    threshold: float,
) -> tuple[Optional[float], Optional[bool]]:
    """Point sum and verdict of a score with possibly unknown components."""
    return _bounded_graded_points(
        [(_points(flag, points), points) for points, flag in components],
        threshold=threshold,
    )


def _bounded_graded_points(
    components: list[tuple[Optional[int], int]],
    *,
    threshold: float,
) -> tuple[Optional[float], Optional[bool]]:
    """(known points or None, maximum points) per component.

    The sum is reported only when every component is known. The verdict is
    reported when the unknown components cannot change it.
    """
    known = sum(points for points, _ in components if points is not None)
    unknown = sum(maximum for points, maximum in components if points is None)
    value = float(known) if unknown == 0 else None
    if known >= threshold:
        positive: Optional[bool] = True
    elif known + unknown < threshold:
        positive = False
    else:
        positive = None
    return value, positive


def _qrs_segment(
    ecg: np.ndarray, beat: BeatDelineation, lead: str, baseline: np.ndarray
) -> Optional[np.ndarray]:
    if beat.r_onset is None or beat.r_offset is None:
        return None
    start = max(0, beat.r_onset.sample)
    end = min(ecg.shape[0], beat.r_offset.sample + 1)
    index = LEAD_MAP[lead]
    segment = np.asarray(ecg[start:end, index], dtype=float) - baseline[index]
    if segment.size < 3 or not np.all(np.isfinite(segment)):
        return None
    return segment


def _s_after_r_depth(
    ecg: np.ndarray, beat: BeatDelineation, lead: str, baseline: np.ndarray
) -> float:
    """Depth (mV) of the negative deflection after the R peak; 0 without R."""
    segment = _qrs_segment(ecg, beat, lead, baseline)
    if segment is None:
        return float("nan")
    peak = int(np.argmax(segment))
    if segment[peak] <= 0:
        return 0.0
    return float(max(0.0, -np.min(segment[peak:])))


def _terminal_r_amplitude(
    ecg: np.ndarray, beat: BeatDelineation, lead: str, baseline: np.ndarray
) -> float:
    """Largest positive deflection (mV) after the QRS nadir of a lead."""
    segment = _qrs_segment(ecg, beat, lead, baseline)
    if segment is None:
        return float("nan")
    nadir = int(np.argmin(segment))
    return float(max(0.0, np.max(segment[nadir:])))


def _t_inversion_depth(metrics: dict[str, float]) -> Optional[float]:
    """Depth (mV) of a dominantly negative T wave; 0 when T is not inverted."""
    t_min, t_max = metrics.get("t_min", np.nan), metrics.get("t_max", np.nan)
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        return None
    if t_min < 0 and -t_min > t_max:
        return float(-t_min)
    return 0.0


def _p_terminal_negativity(
    ecg: np.ndarray,
    beat: BeatDelineation,
    lead: str,
    baseline: np.ndarray,
    fs: int,
) -> tuple[float, float]:
    """Depth (mV) and duration (ms) of the terminal negative P component.

    The component is the negative run around the lowest sample of the
    second half of the P wave; (0, 0) when that half never goes negative.
    """
    if (
        beat.p_onset is None
        or beat.p_offset is None
        or not 0 <= beat.p_onset.sample < beat.p_offset.sample < ecg.shape[0]
    ):
        return float("nan"), float("nan")
    index = LEAD_MAP[lead]
    segment = (
        np.asarray(ecg[beat.p_onset.sample:beat.p_offset.sample + 1, index], dtype=float)
        - baseline[index]
    )
    if segment.size < 4 or not np.all(np.isfinite(segment)):
        return float("nan"), float("nan")
    half = segment.size // 2
    lowest = half + int(np.argmin(segment[half:]))
    if segment[lowest] >= 0:
        return 0.0, 0.0
    left = right = lowest
    while left > 0 and segment[left - 1] < 0:
        left -= 1
    while right + 1 < segment.size and segment[right + 1] < 0:
        right += 1
    return float(-segment[lowest]), float((right - left + 1) / fs * 1000.0)


def _find_prominent_peaks(values: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.signal import find_peaks

    prominence = max(0.02, float(np.std(values)) * 0.3)
    return find_peaks(values, prominence=prominence)
