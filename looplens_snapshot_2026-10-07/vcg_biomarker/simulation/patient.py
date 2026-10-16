"""Patient observations are constraints with uncertainty, not hidden physics multipliers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import numpy as np

FIELDS = {
    "age": ("years", 18, 89), "height_cm": ("cm", 100, 230),
    "weight_kg": ("kg", 25, 300), "thorax_width_cm": ("cm", 15, 65),
    "thorax_depth_cm": ("cm", 10, 50), "heart_rate_bpm": ("bpm", 25, 220),
    "lv_diameter_mm": ("mm", 20, 100), "septum_mm": ("mm", 3, 40),
    "posterior_wall_mm": ("mm", 3, 40), "lv_edv_ml": ("ml", 20, 500),
    "potassium_mmol_l": ("mmol/l", 1.5, 9), "sodium_mmol_l": ("mmol/l", 100, 180),
    "ionized_calcium_mmol_l": ("mmol/l", .5, 2.5),
}


@dataclass(frozen=True)
class Observation:
    value: float
    unit: str
    source: str
    measured_at: str | None = None
    uncertainty: float | None = None  # measurement SD, in the same unit; no default precision
    censoring: str = "exact"

    def validate(self, name):
        if name not in FIELDS:
            raise ValueError("Unregistered patient measurement: " + name)
        unit, lo, hi = FIELDS[name]
        if self.unit != unit or not np.isfinite(self.value) or not lo <= self.value <= hi:
            raise ValueError(f"Invalid {name}: expected {unit}, range {lo}–{hi}")
        if not self.source.strip() or self.censoring != "exact":
            raise ValueError("A source and uncensored observation are required")
        if self.uncertainty is not None and (not np.isfinite(self.uncertainty) or self.uncertainty <= 0):
            raise ValueError("Measurement uncertainty must be positive, not an assumed zero")
        if self.measured_at is not None:
            datetime.fromisoformat(self.measured_at)


@dataclass(frozen=True)
class PatientProfile:
    observations: dict[str, Observation] = field(default_factory=dict)
    sex_recorded: str = "unknown"
    index_time: str | None = None
    clock_alignment_verified: bool = False
    geometry_source: str = "generic"

    def validate(self):
        if type(self.clock_alignment_verified) is not bool:
            raise ValueError("Clock verification must be an explicit boolean")
        if self.sex_recorded not in {"female", "male", "unknown"}:
            raise ValueError("Unknown sex coding; do not impute a half-male patient")
        if self.geometry_source not in {"generic", "surface_measured", "echo_constrained", "segmented_imaging"}:
            raise ValueError("Unknown geometry provenance")
        for name, observation in self.observations.items():
            observation.validate(name)
        if self.index_time:
            datetime.fromisoformat(self.index_time)

    def covariates(self):
        self.validate()
        out = {k: v.value for k, v in self.observations.items()}
        if self.sex_recorded != "unknown":
            out["female"] = float(self.sex_recorded == "female")
        return out

    def to_dict(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        payload["observations"] = {k: Observation(**v) for k, v in payload.get("observations", {}).items()}
        result = cls(**payload)
        result.validate()
        return result

    def ionic_constraints(self, max_age_hours=6.):
        """No clinical mmol/l-to-channel shortcut; only measured extracellular concentrations."""
        self.validate()
        if not np.isfinite(max_age_hours) or max_age_hours <= 0:
            raise ValueError("Positive temporal window required")
        constraints, unavailable = {}, {}
        for key in ("potassium_mmol_l", "sodium_mmol_l", "ionized_calcium_mmol_l"):
            obs = self.observations.get(key)
            if obs is None:
                unavailable[key] = "missing"
                continue
            obs.validate(key)
            if not self.clock_alignment_verified or not self.index_time or not obs.measured_at:
                unavailable[key] = "unverified_temporal_alignment"
                continue
            index, measured = datetime.fromisoformat(self.index_time), datetime.fromisoformat(obs.measured_at)
            if (index.tzinfo is None) != (measured.tzinfo is None):
                unavailable[key] = "incompatible_timezones"
                continue
            delta = (index - measured).total_seconds() / 3600
            if not 0 <= delta <= max_age_hours:
                unavailable[key] = "future_or_stale"
                continue
            constraints[key] = asdict(obs)
        return {"constraints": constraints, "unavailable": unavailable,
                "note": "Plasma concentration is not proven local myocardial interstitial concentration."}


def twin_readiness(profile: PatientProfile):
    profile.validate()
    gates = {
        "patient_anatomy_declared": profile.geometry_source == "segmented_imaging",
        "patient_mesh_verified": False,
        "surface_measurements": all(k in profile.observations for k in ("thorax_width_cm", "thorax_depth_cm")),
        "echo_constraints": any(k in profile.observations for k in ("lv_diameter_mm", "septum_mm", "lv_edv_ml")),
        "independent_mapping_reference": False,
        "mesh_convergence": False, "clinical_validation": False,
    }
    return {"stage": "patient_constrained_research" if gates["echo_constraints"] else "population_prior_research",
            "gates": gates, "claims": "No unique anatomy, ionic conductance or block location inferred from demographics."}


def pacing_suggestion(profile: PatientProfile):
    """An observed rate suggests a pacing protocol, never inferred conduction speed."""
    profile.validate()
    rate = profile.observations.get("heart_rate_bpm")
    if rate is None:
        return {"status": "unavailable", "reason": "missing_measured_rate"}
    return {"status": "protocol_suggestion", "cycle_ms": 60000 / rate.value,
            "source": rate.source, "note": "Mean rate does not reproduce AF, ectopy or autonomic physiology."}
