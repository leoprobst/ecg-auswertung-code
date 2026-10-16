"""Post-A108 exploratory measurements; no diagnostic or restitution claims.

All record summaries use explicit primary-cluster identities. Missing beats are
never removed before assessing adjacency. Shared measurements are record-cached.
"""
from functools import partial

import numpy as np

from .catalog_biomarkers import FeatureValue
from ..data_loader import LEAD_MAP
from ..delineation.base import RhythmClass, measurement_confidence_threshold
from ..vcg_geometry import _origin_candidate, _tp_candidate

VERSION = "post-A108-robustness-1.1.0"
REFERENCE = "https://doi.org/10.1038/s41598-026-37361-w"
LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
INDICES = tuple(LEAD_MAP[x] for x in LEADS)

# A baseline that needs only the QRS onset. The PR window needs a delineated
# P offset, which 37 % of the MIMIC recordings have, and it does not exist in
# atrial fibrillation. In the A117 10k pilot the PR-baseline Direct-8 angle
# was measurable in 56 % of recordings, in none with an AF or flutter
# pattern, and the beat median -- six eligible beats required -- lost a
# further quarter. The window below ends 4 ms before the QRS onset and is
# 24 ms long; f-waves in atrial fibrillation are accepted as part of the
# measurement, and the rhythm context beside the value says so. The PR
# definitions stay as they are; this is a second definition, not a fallback.
PREQRS_BASELINE_WINDOW_MS = (28.0, 4.0)
# Three beats give a median with one on either side; six was the point at
# which the PR summary lost more recordings to the count than to the baseline.
PREQRS_BEAT_SUMMARY_MINIMUM = 3


def preqrs_baseline(onset, fs):
    """Inclusive sample bounds of the pre-QRS window, or None if it leaves the record."""
    start = int(onset) - int(round(PREQRS_BASELINE_WINDOW_MS[0] * fs / 1000))
    end = int(onset) - int(round(PREQRS_BASELINE_WINDOW_MS[1] * fs / 1000))
    if start < 0 or end <= start:
        return None
    return start, end


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite(na + nb) or min(na, nb) <= 1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a / na, b / nb), -1, 1))))


def baseline_ok(signal, bounds, fs):
    """Use the existing 150-uV scatter gate, extended to native eight leads."""
    if bounds is None or fs <= 0:
        return False
    start, end = bounds
    if not 0 <= start < end < len(signal) or (end - start) / fs < .020:
        return False
    return _origin_candidate(
        signal[:, INDICES], start=start, end=end, fs=fs, method=VERSION,
        endpoint_confidence=1., minimum_duration_ms=20.,
    ) is not None


def native_metrics(signal, qrs, jt, baseline, fs, *, integral=True, detailed=True):
    """One common integration for area, ST-T half-vectors and lead sensitivity."""
    from .advanced_vcg_biomarkers import direct_qrs_t_angles_8lead
    if not baseline_ok(signal, baseline, fs):
        return {}, "baseline_short_or_unstable"
    q0, q1 = qrs
    j, end = jt
    if not 0 <= q0 < q1 == j < end < len(signal):
        return {}, "segment_boundaries"
    values = signal[:, INDICES]
    base = values[baseline[0]:baseline[1] + 1].mean(axis=0)
    q, t = values[q0:q1 + 1] - base, values[j:end + 1] - base
    if not np.isfinite(q).all() or not np.isfinite(t).all():
        return {}, "signal_nonfinite"
    qa = np.trapezoid(q, dx=1000 / fs, axis=0)
    ta = np.trapezoid(t, dx=1000 / fs, axis=0)
    area = angle(qa, ta)
    if area is None:
        return {}, "degenerate_area_vector"
    out = {"area": area, "stt_area": float(np.linalg.norm(ta))}
    if integral:
        _, out["integral"] = direct_qrs_t_angles_8lead(
            signal, qrs_bounds=qrs, t_bounds=jt, baseline_bounds=baseline)
    if not detailed:
        return out, "ok"
    # Exactly 50% of duration, including an interpolated midpoint for odd spans.
    midpoint = (len(t) - 1) / 2
    lo, hi = int(np.floor(midpoint)), int(np.ceil(midpoint))
    middle = (t[lo] + t[hi]) / 2
    left = np.vstack((t[:int(np.ceil(midpoint))], middle))
    right = np.vstack((middle, t[int(np.floor(midpoint)) + 1:]))
    lt = np.r_[np.arange(int(np.ceil(midpoint))), midpoint] * 1000 / fs
    rt = np.r_[midpoint, np.arange(int(np.floor(midpoint)) + 1, len(t))] * 1000 / fs
    early = np.trapezoid(left, x=lt, axis=0)
    late = np.trapezoid(right, x=rt, axis=0)
    out["early_late_angle"] = angle(early, late)
    out["late_early_ratio"] = (
        float(np.linalg.norm(late) / np.linalg.norm(early))
        if min(np.linalg.norm(early), np.linalg.norm(late)) > 1e-10 else None)
    for i, lead in enumerate(LEADS):
        dropped = angle(np.delete(qa, i), np.delete(ta, i))
        out[f"drop_{lead}"] = None if dropped is None else dropped - area
    deltas = [out[f"drop_{lead}"] for lead in LEADS]
    if all(value is not None for value in deltas):
        out["drop_max_abs"] = float(np.max(np.abs(deltas)))
        out["drop_sd"] = float(np.std(deltas, ddof=0))
    return out, "ok"


def _eligible(beat):
    return all(beat.segment_confidence(w) >= measurement_confidence_threshold(w)
               for w in ("qrs", "t", "pr"))


def _eligible_preqrs(beat):
    return all(beat.segment_confidence(w) >= measurement_confidence_threshold(w)
               for w in ("qrs", "t"))


def _representative(ctx):
    key = (VERSION, "representative")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    result, reason = {}, "missing_signal_or_segments"
    windows = {}
    signal = ctx.ecg_12lead
    beat = ctx.beat
    if (signal is not None and signal.ndim == 2 and signal.shape[1] == 12 and beat is not None
            and all(ctx.is_segment_evaluable(w) for w in ("qrs", "t"))):
        q, t, pr = ctx.bounds("qrs"), ctx.bounds("t"), ctx.bounds("pr")
        if q is not None and t is not None and pr is not None and ctx.is_segment_evaluable("pr"):
            result, reason = native_metrics(signal, q, (q[1], t[1]), pr, ctx.fs)
        if q is not None and t is not None:
            pre = preqrs_baseline(q[0], ctx.fs)
            if pre is not None:
                preqrs, preqrs_reason = native_metrics(
                    signal, q, (q[1], t[1]), pre, ctx.fs, detailed=False)
                for method in ("area", "integral"):
                    result[f"preqrs_{method}"] = preqrs.get(method)
                result["preqrs_status"] = preqrs_reason
                if preqrs_reason == 'ok':
                    windows['preqrs'] = {'baseline': pre, 'qrs': q, 'j_to_t_end': (q[1], t[1])}
        # TP is a separate variant, never a replacement for an absent PR window.
        if q is not None and t is not None and not ctx.delineation.rhythm.discrete_p_not_evaluable:
            beats = sorted(ctx.delineation.beats, key=lambda b: b.r_peak.sample)
            target = next((i for i, b in enumerate(beats)
                           if b.r_peak.sample == beat.r_peak.sample), None)
            candidates = []
            if target is not None:
                pairs = []
                if target > 0:
                    pairs.append((beats[target - 1], beat))
                if target + 1 < len(beats):
                    pairs.append((beat, beats[target + 1]))
                for left, right in pairs:
                    candidate = _tp_candidate(
                        signal[:, INDICES], left_beat=left, right_beat=right,
                        fs=ctx.fs, method="tp_native8", minimum_confidence=.35,
                        discrete_p_permitted=True)
                    if candidate is not None:
                        candidates.append(candidate)
            if candidates:
                best = max(candidates, key=lambda c: (c.confidence, c.duration_samples))
                tp, _ = native_metrics(signal, q, (q[1], t[1]),
                                       (best.start_sample, best.end_sample), ctx.fs)
                if tp:
                    windows['tp'] = {'baseline': (best.start_sample, best.end_sample),
                                     'qrs': q, 'j_to_t_end': (q[1], t[1])}
                from .advanced_vcg_biomarkers import _direct_qrs_t_angles
                original = _direct_qrs_t_angles(ctx)
                for i, method in enumerate(("area", "integral")):
                    result[f"tp_{method}"] = tp.get(method)
                    result[f"tp_pr_delta_{method}"] = (
                        tp[method] - original[i]
                        if tp.get(method) is not None and original[i] is not None else None)
    result['_measurement_windows'] = windows
    ctx._primitive_cache[key] = result, reason
    return result, reason


def _record(ctx):
    key = (VERSION, "record")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    signal = ctx._record_ecg_12lead
    beats = ctx.delineation.beats
    rows = {}
    if (signal is None or signal.ndim != 2 or signal.shape[1] != 12
            or len(signal) != ctx.delineation.n_samples):
        _capture_beat_reason(ctx, 'pr', None, 'record_signal_unavailable_or_incompatible')
        ctx._primitive_cache[key] = rows
        return rows
    if ctx.delineation.rhythm.discrete_p_not_evaluable:
        _capture_beat_reason(ctx, 'pr', None, 'rhythm_no_discrete_p')
        ctx._primitive_cache[key] = rows
        return rows
    peaks = [b.r_peak.sample for b in beats]
    if len(peaks) > 1 and np.any(np.diff(peaks) <= 0):
        _capture_beat_reason(ctx, 'pr', None, 'r_peak_order_invalid')
        ctx._primitive_cache[key] = rows
        return rows
    for i, beat in enumerate(beats):
        if getattr(beat, "morphology_cluster_id", None) != 0:
            _capture_beat_reason(ctx, 'pr', i, 'not_primary_morphology_cluster')
            continue
        if not _eligible(beat):
            _capture_beat_reason(ctx, 'pr', i, 'segment_quality_below_threshold')
            continue
        q = (beat.r_onset.sample, beat.r_offset.sample)
        baseline = (beat.p_offset.sample, beat.r_onset.sample)
        values, reason = native_metrics(signal, q, (q[1], beat.t_offset.sample), baseline, ctx.fs, detailed=False)
        _capture_beat_reason(ctx, 'pr', i, reason)
        if values:
            values["rr_ms"] = (peaks[i] - peaks[i - 1]) * 1000 / ctx.fs if i else None
            _capture_beat_window(ctx, values, i, beat, baseline, q)
            rows[i] = values
    ctx._primitive_cache[key] = rows
    return rows


def _record_preqrs(ctx):
    """Per-beat Direct-8 angles on the pre-QRS baseline; no P wave, no rhythm gate."""
    key = (VERSION, "record_preqrs")
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    signal = ctx._record_ecg_12lead
    beats = ctx.delineation.beats
    rows = {}
    if (signal is None or signal.ndim != 2 or signal.shape[1] != 12
            or len(signal) != ctx.delineation.n_samples):
        _capture_beat_reason(ctx, 'preqrs', None, 'record_signal_unavailable_or_incompatible')
        ctx._primitive_cache[key] = rows
        return rows
    peaks = [b.r_peak.sample for b in beats]
    if len(peaks) > 1 and np.any(np.diff(peaks) <= 0):
        _capture_beat_reason(ctx, 'preqrs', None, 'r_peak_order_invalid')
        ctx._primitive_cache[key] = rows
        return rows
    for i, beat in enumerate(beats):
        if getattr(beat, "morphology_cluster_id", None) != 0:
            _capture_beat_reason(ctx, 'preqrs', i, 'not_primary_morphology_cluster')
            continue
        if not _eligible_preqrs(beat):
            _capture_beat_reason(ctx, 'preqrs', i, 'segment_quality_below_threshold')
            continue
        q = (beat.r_onset.sample, beat.r_offset.sample)
        pre = preqrs_baseline(q[0], ctx.fs)
        if pre is None:
            _capture_beat_reason(ctx, 'preqrs', i, 'preqrs_window_outside_record')
            continue
        values, reason = native_metrics(signal, q, (q[1], beat.t_offset.sample), pre, ctx.fs, detailed=False)
        _capture_beat_reason(ctx, 'preqrs', i, reason)
        if values:
            _capture_beat_window(ctx, values, i, beat, pre, q)
            rows[i] = values
    ctx._primitive_cache[key] = rows
    return rows


def _capture_beat_reason(ctx, baseline, index, reason):
    if getattr(ctx, '_record_acquisition', {}).get('lab_capture_measurement_windows') is True:
        ctx._primitive_cache.setdefault((VERSION, 'beat_audit', baseline), {})[index] = reason


def _capture_beat_window(ctx, values, index, beat, baseline, qrs):
    if getattr(ctx, '_record_acquisition', {}).get('lab_capture_measurement_windows') is not True:
        return
    values['_measurement_window'] = {'beat_index': index, 'r_peak_sample': int(beat.r_peak.sample),
        'morphology_cluster_id': int(beat.morphology_cluster_id), 'baseline': baseline,
        'qrs': qrs, 'j_to_t_end': (qrs[1], beat.t_offset.sample)}


def _beat_summary_windows(ctx, rows, field, baseline, summary, minimum):
    if getattr(ctx, '_record_acquisition', {}).get('lab_capture_measurement_windows') is not True:
        return None
    eligible = [(i, row) for i, row in sorted(rows.items()) if row.get(field) is not None]
    items = [dict(row['_measurement_window'], value=float(row[field]))
             for _, row in eligible[:2000] if '_measurement_window' in row]
    recorded = ctx._primitive_cache.get((VERSION, 'beat_audit', baseline), {})
    audit, reason_counts = [], {}
    for index, beat in enumerate(ctx.delineation.beats):
        included = rows.get(index, {}).get(field) is not None
        reason = 'included' if included else recorded.get(index, recorded.get(None, 'reason_unavailable'))
        if not included and reason == 'ok':
            reason = field + '_unavailable_or_degenerate'
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if index < 2000:
            audit.append({'beat_index': index, 'r_peak_sample': int(beat.r_peak.sample),
                          'included': included, 'reason': reason})
    return {'schema_version': '1.1.0', 'scope': 'beat_summary', 'summary': summary,
        'minimum_beats': minimum, 'eligible_beats': len(eligible), 'record_beats': len(ctx.delineation.beats),
        'stored_beats': len(items), 'truncated': len(items) != len(eligible),
        'signal_path': 'record_feature_context_ecg_12lead', 'coordinate_system': 'analysis_samples_inclusive',
        'sampling_rate_hz': float(ctx.fs), 'leads': list(LEADS), 'baseline_kind': baseline,
        'baseline_estimator': 'arithmetic_mean', 'items': items,
        'beat_audit': audit, 'reason_counts': reason_counts, 'audit_truncated': len(audit) != len(ctx.delineation.beats),
        'reason_scope': 'first_failed_gate_in_calculation_order_not_all_possible_causes',
        'warning': 'Primary morphology cluster is not a verified beat diagnosis; MAD is not a confidence interval'}


def measure(ctx, *, kind, field):
    notes = f"{VERSION}; exploratory; post-pilot hypothesis; not a clinical rule"
    value = None
    measurement_windows = None
    if kind == "representative":
        data, reason = _representative(ctx)
        value = data.get(field)
        if value is None and reason == "ok":
            reason = "required_measurement_unavailable_or_degenerate"
        notes += f"; status={reason if value is None else 'ok'}"
        variant = field.split('_', 1)[0]
        if field in ('preqrs_area', 'preqrs_integral', 'tp_area', 'tp_integral') and value is not None:
            bounds = data.get('_measurement_windows', {}).get(variant)
            if bounds:
                measurement_windows = dict(bounds, schema_version='1.0.0',
                    coordinate_system='analysis_samples_inclusive', signal_path='feature_context_ecg_12lead',
                    sampling_rate_hz=float(ctx.fs), leads=list(LEADS), baseline_kind=variant,
                    baseline_estimator='arithmetic_mean', scope='representative_signal_not_beat_summary')
    elif kind in {"median_preqrs", "mad_preqrs", "count_preqrs"}:
        rows = _record_preqrs(ctx)
        values = [row[field] for row in rows.values() if row.get(field) is not None]
        if kind == "count_preqrs":
            value = float(len(values))
        elif len(values) >= PREQRS_BEAT_SUMMARY_MINIMUM:
            median = np.median(values)
            value = float(median if kind == "median_preqrs" else np.median(np.abs(values - median)))
        notes += (f"; baseline=pre-QRS {PREQRS_BASELINE_WINDOW_MS[0]:g}-{PREQRS_BASELINE_WINDOW_MS[1]:g} ms"
                  f"; eligible_beats={len(values)}; minimum={PREQRS_BEAT_SUMMARY_MINIMUM}"
                  "; MAD is not a confidence interval; f-waves not excluded")
        if kind != 'count_preqrs':
            measurement_windows = _beat_summary_windows(ctx, rows, field, 'preqrs',
                kind.split('_')[0], PREQRS_BEAT_SUMMARY_MINIMUM)
    else:
        rows = _record(ctx)
        beats = ctx.delineation.beats
        if kind == "count":
            value = float(len([row for row in rows.values() if row.get(field) is not None]))
            notes += "; PR-baseline primary-cluster beats; zero in rhythms without a discrete P"
        elif kind in {"median", "mad"}:
            values = [row[field] for row in rows.values() if row.get(field) is not None]
            if len(values) >= 6:
                median = np.median(values)
                value = float(median if kind == "median" else np.median(np.abs(values - median)))
            notes += f"; eligible_beats={len(values)}; minimum=6; MAD is not a confidence interval"
            measurement_windows = _beat_summary_windows(ctx, rows, field, 'pr', kind, 6)
        elif kind == "post":
            # Secondary morphology is NOT by itself proof of ectopy: require
            # explicitly adjudicated beat status, never relabel PVC from cluster.
            post = {i for i in rows if i > 0 and
                    str(getattr(beats[i - 1], "beat_type", "")).lower() in {"pvc", "pac"}
                    and bool(getattr(beats[i - 1], "beat_type_source", ""))}
            controls = {i for i in rows if i > 0 and i not in post and
                        getattr(beats[i - 1], "morphology_cluster_id", None) == 0 and
                        getattr(beats[i - 1], "beat_type", "unknown") == "sinus" and
                        bool(getattr(beats[i - 1], "beat_type_source", ""))}
            a = [rows[i][field] for i in sorted(post) if rows[i].get(field) is not None]
            b = [rows[i][field] for i in sorted(controls) if rows[i].get(field) is not None]
            if len(a) >= 3 and len(b) >= 6:
                value = float(np.median(a) - np.median(b))
            notes += f"; confirmed_postectopic={len(a)}; controls={len(b)}; minima=3/6"
        elif kind in {"long_rmssd", "long_rr_slope"}:
            duration = ctx.delineation.n_samples / ctx.fs
            stable = (ctx.delineation.rhythm.rhythm_class == RhythmClass.REGULAR
                      and ctx.delineation.rhythm.confidence >= .5
                      and all(getattr(b, "morphology_cluster_id", None) == 0 for b in beats))
            qualified = {i: row for i, row in rows.items() if row.get(field) is not None}
            if duration >= 120 and len(qualified) >= 100 and stable:
                if kind == "long_rmssd":
                    differences = [row[field] - qualified[i - 1][field]
                                   for i, row in qualified.items() if i - 1 in qualified]
                    if len(differences) >= 99:
                        value = float(np.sqrt(np.mean(np.square(differences))))
                else:
                    pairs = [(row["rr_ms"], row[field]) for i, row in qualified.items()
                             if i - 1 in qualified and row["rr_ms"] is not None]
                    if len(pairs) >= 99:
                        x, y = np.asarray(pairs).T
                        x = x - x.mean()
                        if np.dot(x, x) > 1e-10:
                            value = float(np.dot(x, y - y.mean()) / np.dot(x, x))
            notes += f"; duration_s={duration:g}; eligible_beats={len(qualified)}; minima=120s/100; not restitution"
    if value is None:
        notes = "not_evaluable: " + notes
    return FeatureValue(value, .8 if value is not None else 0., notes=notes, measurement_windows=measurement_windows)


def specifications(spec_type):
    rows = []
    def add(name, kind, field, unit, formula):
        rows.append(spec_type(name, "A108 explorative Messrobustheit", formula,
                              formula, unit, (), partial(measure, kind=kind, field=field),
                              (REFERENCE,)))
    for method in ("area", "integral"):
        add(f"qrs_t_{method}_angle_direct_8lead_tp_deg", "representative", f"tp_{method}",
            "deg", "Direct8 with measured stable TP baseline >=20ms; no PR fallback")
        add(f"qrs_t_{method}_angle_direct_8lead_tp_minus_pr_deg", "representative", f"tp_pr_delta_{method}",
            "deg", "TP variant minus unchanged original PR definition on same representative signal")
        for summary in ("median", "mad"):
            add(f"qrs_t_{method}_angle_direct_8lead_beat_{summary}_deg", summary, method,
                "deg", f"{summary} of >=6 qualified primary-cluster beat Direct8 {method} angles")
    for method in ("area", "integral"):
        add(f"qrs_t_{method}_angle_direct_8lead_preqrs_deg", "representative", f"preqrs_{method}",
            "deg", f"Direct8 {method} angle; baseline = mean of the 24 ms window ending 4 ms "
                   "before QRS onset; needs no P delineation; PR definition unchanged beside it")
        for summary in ("median", "mad"):
            add(f"qrs_t_{method}_angle_direct_8lead_beat_{summary}_preqrs_deg", f"{summary}_preqrs", method,
                "deg", f"{summary} of >={PREQRS_BEAT_SUMMARY_MINIMUM} qualified primary-cluster beat Direct8 "
                       f"{method} angles on the pre-QRS baseline")
    add("direct8_pr_eligible_beat_count", "count", "area", "count",
        "Primary-cluster beats with evaluable QRS, T and PR baseline entering the PR beat summaries")
    add("direct8_preqrs_eligible_beat_count", "count_preqrs", "area", "count",
        "Primary-cluster beats with evaluable QRS and T entering the pre-QRS beat summaries")
    add("stt_early_late_area_angle_direct_8lead_deg", "representative", "early_late_angle",
        "deg", "Angle between native8 signed ST-T half integrals split at exact J-Tend duration midpoint")
    add("stt_late_early_area_norm_ratio_direct_8lead", "representative", "late_early_ratio",
        "dimensionless", "norm(late ST-T half integral)/norm(early half); PR baseline; positive nondegenerate norms")
    for lead in LEADS:
        add(f"direct8_area_leave_{lead.lower()}_out_delta_deg", "representative", f"drop_{lead}",
            "deg", f"Area angle without {lead} minus eight-lead area angle; technical sensitivity, not localization")
    for summary in ("max_abs", "sd"):
        add(f"direct8_area_leave_one_out_{summary}_deg", "representative", f"drop_{summary}",
            "deg", f"{summary} of eight signed leave-one-lead-out changes; all eight must be evaluable")
    for field, unit in (("area", "deg"), ("stt_area", "mV*ms")):
        add(f"postectopic_direct8_{field}_delta", "post", field, unit,
            "Median confirmed postectopic primary-cluster value minus controls; minimum 3/6; no automatic ectopy label")
        for kind in ("long_rmssd", "long_rr_slope"):
            add(f"direct8_{field}_{kind}", kind, field, unit if kind == "long_rmssd" else f"{unit}/ms",
                f"{kind} within regular primary morphology; >=120s and >=100 eligible beats; no gap compression")
    return tuple(rows)


def measurement_contract():
    return {
        "version": VERSION, "hypothesis_origin": "after_A108_pilot_inspection",
        "clinical_claim_allowed": False, "baseline_minimum_duration_ms": 20,
        "baseline_scatter_gate_mv": .150, "baseline_scatter_space": "native_eight_leads",
        "baseline_method": "mean; measured TP variant does not replace original PR",
        "beat_summary_minimum": 6, "beat_summary_morphology_cluster": 0,
        "preqrs_baseline_window_ms": list(PREQRS_BASELINE_WINDOW_MS),
        "preqrs_beat_summary_minimum": PREQRS_BEAT_SUMMARY_MINIMUM,
        "preqrs_rhythm_gate": "none; atrial f-waves inside the window are part of the measurement",
        "postectopic_minimum": 3, "postectopic_control_minimum": 6,
        "postectopic_reference": "beat_type PVC/PAC with source; sinus controls with source",
        "long_record_minimum_seconds": 120, "long_record_minimum_beats": 100,
        "long_record_rhythm": "regular, confidence>=0.5, all beats primary morphology",
        "no_gap_compression": True, "no_restitution_claim": True,
        "stt_division": "exact 50% J-to-Tend duration, interpolated midpoint if necessary",
        "leave_one_out_claim": "technical sensitivity, not anatomical localization",
        "equivalence_review": {
            "early_late_t_angle_deg": "not equivalent: existing 40ms T-onset/T-end vectors in XYZ, new J-Tend half integrals in native8",
            "post_ectopic_t_axis_delta_deg": "not equivalent: existing T-peak XYZ axis, new QRS-to-STT native8 integral angle",
            "post_ectopic_t_amplitude_delta_mv": "not equivalent: peak amplitude versus integrated native8 STT magnitude",
            "vcg_origin_tp_pr_distance_uv": "not equivalent: baseline displacement versus induced QRS-T angle change",
        },
    }
