"""Discovery-only scalar adapters sharing one record-level NDAT computation."""
import json
import math

from .base import BiomarkerBase, BiomarkerDefinition, BiomarkerRegistry, BiomarkerResult
from .catalog_biomarkers import _context
from .ndat import negative_downslope_times, NDAT_LEADS, NDAT_VERSION

NDAT_MARKER_IDS = tuple(f"ndat_{lead.lower()}_ms" for lead in NDAT_LEADS) + ("ndat_precordial_spread_ms",)


def ndat_context_result(ctx):
    key = ("ndat", NDAT_VERSION)
    if key in ctx._primitive_cache:
        return ctx._primitive_cache[key]
    raw = ctx._record_raw_ecg_12lead
    acquisition = ctx._record_acquisition
    blocked = ""
    for field, minimum in (("original_sampling_rate_hz", 250), ("upper_bandwidth_hz", 100)):
        value = acquisition.get(field)
        if value is not None:
            try:
                number = float(value)
                if not math.isfinite(number) or number < minimum:
                    blocked = "insufficient_or_invalid_" + field
            except (TypeError, ValueError):
                blocked = "invalid_" + field
    if blocked:
        result = dict(status="not_evaluable", reason=blocked, leads={}, spread_ms=None)
    elif raw is None:
        result = dict(status="not_evaluable", reason="explicit_raw_record_required", leads={}, spread_ms=None)
    else:
        bounds = [(beat.r_onset.sample, beat.r_offset.sample) for beat in ctx.delineation.beats
                  if beat.r_onset is not None and beat.r_offset is not None
                  and beat.segment_confidence("qrs") >= .7]
        pacing = acquisition.get("pacing_present", getattr(ctx.delineation, "pacing_present", None))
        result = negative_downslope_times(raw, ctx.fs, bounds, pacing_present=pacing is True)
        result["pacing_status"] = "present" if pacing is True else "absent" if pacing is False else "unknown"
        result["acquisition_status"] = "original_analogue_bandwidth_not_verified_by_sampling_rate"
        result["acquisition_metadata"] = acquisition
        result["n_delineated_beats"] = len(ctx.delineation.beats)
        result["n_qrs_quality_eligible"] = len(bounds)
        # Do not conceal that most delineated beats were rejected upstream.
        if len(bounds) < .75 * len(ctx.delineation.beats):
            result.update(status="not_evaluable", reason="insufficient_quality_eligible_beats", spread_ms=None, order=[])
            for item in result.get("leads", {}).values():
                item.update(time_ms=None, status="not_evaluable")
                item["reasons"] = sorted(set(item.get("reasons", []) + ["insufficient_quality_eligible_beats"]))
    ctx._primitive_cache[key] = result
    return result


class NdatBiomarker(BiomarkerBase):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Native precordial negative-downslope timing; short-record discovery adaptation."

    @property
    def unit(self):
        return "ms"

    @property
    def category(self):
        return "ecg"

    def compute(self, delineation, ecg_12lead=None, vcg_xyz=None, fs=500):
        result = ndat_context_result(_context(delineation, ecg_12lead, vcg_xyz, fs))
        if self.name == "ndat_precordial_spread_ms":
            value = result.get("spread_ms")
        else:
            lead = self.name.split("_")[1].upper()
            value = result.get("leads", {}).get(lead, {}).get("time_ms")
        return BiomarkerResult(name=self.name, value=value, unit="ms", category="ecg",
            confidence=.7 if value is not None else 0., description=self.description,
            family="precordial_downslope_timing", references=["doi:10.1038/s41598-024-55789-w"],
            computation_notes=json.dumps(result, allow_nan=False),
            pathological_note="Post-outcome discovery; no clinical activation-time or pacing claim.")


def register_ndat_biomarkers():
    for name in NDAT_MARKER_IDS:
        if name in BiomarkerRegistry.available():
            if isinstance(BiomarkerRegistry.get(name), NdatBiomarker):
                continue
            raise RuntimeError(f"NDAT registry collision: {name}")
        BiomarkerRegistry.register_definition(BiomarkerDefinition(name=name,
            family="precordial_downslope_timing", description=NdatBiomarker(name).description,
            formula=("max(median NDAT_V1..V6)-min(median NDAT_V1..V6)" if "spread" in name else
                     "median over eligible beats of 1000*(argmin_QRS dV/dt-QRS_onset)/fs"),
            unit="ms", category="ecg", source="ndat_short_record_" + NDAT_VERSION,
            pathological_note="Post-outcome discovery; technical gates are not clinical cutoffs."))
        BiomarkerRegistry.register(NdatBiomarker(name))
