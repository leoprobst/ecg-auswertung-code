"""Faithful representations for the markers that live on a circle.

Twenty-four extracted markers are angular coordinates on a full turn. The
screening stage orders a marker and asks whether the positive class sits at
one end -- which is a category error for an angle: -179 degrees and +179
degrees are two degrees of arc apart and sit at opposite ends of the rank
list. No direction flip repairs that, because the defect is not the direction
but the seam.

Isaza et al. (Heart Rhythm 2024) regress the **cosine** of the spatial
ventricular gradient azimuth rather than the azimuth for exactly this reason,
and reach a 39-fold adjusted odds ratio for inducible ventricular arrhythmia
when it is combined with ejection fraction and gradient magnitude.

**Measured before this module was written.** Over the 1.168 marker-endpoint
pairs that the existing feature partitions and labels support for these
twenty-four markers, a faithful representation separates better by more than
0,05 AUC in **297 pairs (25,4 %)**, and 92 of those are invisible to the
current screen (raw |AUC-0,5| < 0,02). The median effect size roughly doubles,
from 0,0398 to 0,0672. Every one of the twenty-four has at least one such
pair, which is why all of them are covered rather than the sixteen that carry
93 % of the gains: a cut-off drawn on development results would be exactly the
kind of post-hoc selection this project refuses elsewhere.

For comparison, the folded-association diagnostic beside ``fit_marker_model``
found 2,67 % of *all* marker-endpoint pairs improved by folding. The circular
defect is an order of magnitude more damaging, and unlike the folded case it
has a fix that decides nothing: a projection is a deterministic per-record
transform, not a second inferential arm.

**What the two projections are.** The project's axis convention is
``frank-xyz-geh-v1.0.0`` -- +X left, +Y inferior, +Z posterior, with
``azimuth = atan2(Z, X)`` in the transverse plane. So for an azimuth theta:

    cos(theta) = X / hypot(X, Z)    how far left the vector points
    sin(theta) = Z / hypot(X, Z)    how far posterior it points

These are the transverse direction cosines. Both are continuous across the
seam, both are bounded in [-1, 1], and together they carry the whole
direction: no information is added and none is lost, only the scale is
repaired.

**One marker is an axis, not a direction.** ``loop_ellipse_orientation_deg``
is ``atan2`` over the first eigenvector of a segment covariance, taken in the
**frontal** X-Y plane rather than the transverse one. An eigenvector is
defined only up to sign, so an ellipse whose major axis lies along +X and one
along -X are the same ellipse reported 180 degrees apart. Measured on the
existing partitions, 12,8 % of recordings land on the opposite branch from the
other 87,2 %, and 17,2 % sit within 15 degrees of the seam. A single-angle
cosine cannot repair that; the **doubled** angle can, because cos(2*theta) and
sin(2*theta) are invariant under theta -> theta + 180. Those two are therefore
what this module registers for it, in the frontal plane.

**What this module does not do.** It does not retire, replace or alter any
source marker. All twenty-four keep their panels, their values and their
identifiers; the projections are additional columns beside them. They are
registered as post-outcome discovery hypotheses, so no curated panel changes
and none of them enters the combination search -- which is a real cost worth
stating, because a cosine and a sine only carry a full direction when
something can use them together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import (
    BiomarkerBase,
    BiomarkerDefinition,
    BiomarkerRegistry,
    BiomarkerResult,
)
from ..delineation.base import RecordDelineation


CIRCULAR_PROJECTION_VERSION = "1.0.0"
CIRCULAR_PROJECTION_SOURCE = "circular_projection_v1"

# The transverse azimuths. Every one of these routes through
# ``vcg_geometry.vcg_azimuth_deg`` -- or, for
# ``septal_activation_azimuth_deg``, through an inline copy of it -- so all
# twenty-three share the single convention ``atan2(Z, X)``: cosine is
# leftward, sine is posterior.
TRANSVERSE_AZIMUTH_SOURCES: tuple[str, ...] = (
    "peak_svg_azimuth_deg",
    "qrs_area_vector_azimuth_deg",
    "qrs_initial_10ms_force_azimuth_deg",
    "qrs_initial_20ms_force_azimuth_deg",
    "qrs_initial_30ms_force_azimuth_deg",
    "qrs_initial_40ms_force_azimuth_deg",
    "qrs_max_vector_azimuth",
    "qrs_terminal_10ms_force_azimuth_deg",
    "qrs_terminal_20ms_force_azimuth_deg",
    "qrs_terminal_30ms_force_azimuth_deg",
    "qrs_terminal_40ms_force_azimuth_deg",
    "septal_activation_azimuth_deg",
    "st_area_vector_azimuth_deg",
    "st_injury_vector_azimuth_deg",
    "st_j_plus_20ms_azimuth_deg",
    "st_j_plus_40ms_azimuth_deg",
    "st_j_plus_60ms_azimuth_deg",
    "st_j_plus_80ms_azimuth_deg",
    "svg_azimuth",
    "t_area_vector_azimuth_deg",
    "t_early_trajectory_azimuth_deg",
    "t_late_trajectory_azimuth_deg",
    "t_max_vector_azimuth",
)

# The one axis marker: frontal plane, doubled angle.
FRONTAL_ORIENTATION_SOURCES: tuple[str, ...] = ("loop_ellipse_orientation_deg",)

CIRCULAR_SOURCE_MARKERS: tuple[str, ...] = tuple(
    sorted(TRANSVERSE_AZIMUTH_SOURCES + FRONTAL_ORIENTATION_SOURCES)
)


def _stem(source: str) -> str:
    """The source name without its angular suffix.

    ``svg_azimuth`` becomes ``svg``, ``qrs_area_vector_azimuth_deg`` becomes
    ``qrs_area_vector``. The projection names are built from the stem, so a
    reader can see at a glance which marker a projection belongs to.
    """

    for suffix in ("_azimuth_deg", "_azimuth", "_orientation_deg"):
        if source.endswith(suffix):
            return source[: -len(suffix)]
    raise ValueError(f"not an angular marker name: {source}")


@dataclass(frozen=True)
class CircularProjectionSpec:
    name: str
    source: str
    component: str          # "cos" or "sin"
    multiple: int           # 1 for a direction, 2 for an axis
    axis_label: str         # what the projection points at
    plane: str
    family: str
    unit: str = "cosine"


def _projection_specs() -> tuple[CircularProjectionSpec, ...]:
    specs: list[CircularProjectionSpec] = []
    for source in TRANSVERSE_AZIMUTH_SOURCES:
        stem = _stem(source)
        specs.append(
            CircularProjectionSpec(
                f"{stem}_leftward", source, "cos", 1, "left (+X)",
                "transverse", "Richtungskosinus",
            )
        )
        specs.append(
            CircularProjectionSpec(
                f"{stem}_posterior", source, "sin", 1, "posterior (+Z)",
                "transverse", "Richtungskosinus",
            )
        )
    for source in FRONTAL_ORIENTATION_SOURCES:
        stem = _stem(source)
        specs.append(
            CircularProjectionSpec(
                f"{stem}_axis_leftward", source, "cos", 2, "left (+X)",
                "frontal", "Richtungskosinus",
            )
        )
        specs.append(
            CircularProjectionSpec(
                f"{stem}_axis_inferior", source, "sin", 2, "inferior (+Y)",
                "frontal", "Richtungskosinus",
            )
        )
    return tuple(specs)


CIRCULAR_PROJECTION_SPECS: tuple[CircularProjectionSpec, ...] = _projection_specs()


def project_angle(
    degrees: Optional[float],
    component: str,
    multiple: int = 1,
) -> Optional[float]:
    """The projection itself, isolated so it can be tested without a recording.

    ``multiple`` is 2 for an axis, which makes the result invariant under a
    180-degree turn -- the property that repairs an eigenvector arbitrary
    sign.
    """

    if degrees is None:
        return None
    value = float(degrees)
    if not np.isfinite(value):
        return None
    radians = np.radians(value * float(multiple))
    projected = float(np.cos(radians) if component == "cos" else np.sin(radians))
    if not np.isfinite(projected):
        return None
    return float(np.clip(projected, -1.0, 1.0))


def _projection_formula(spec: CircularProjectionSpec) -> str:
    trig = "cos" if spec.component == "cos" else "sin"
    angle = f"2*{spec.source}" if spec.multiple == 2 else spec.source
    return (
        f"{trig}({angle}), the {spec.plane}-plane projection onto "
        f"{spec.axis_label}"
    )


def _projection_description(spec: CircularProjectionSpec) -> str:
    return (
        f"Projection of {spec.source} onto {spec.axis_label} in the "
        f"{spec.plane} plane"
        + (" (doubled angle)" if spec.multiple == 2 else "")
        + "."
    )


class CircularProjectionBiomarker(BiomarkerBase):
    """A projection computed from the source marker own returned value.

    The source is called through the registry rather than recomputed from the
    signal. That is deliberate: it guarantees the projection is of exactly the
    number the screening stage sees in the source column, including every
    abstention the source makes, so the two can never drift apart. The segment
    and beat primitives underneath are memoised per recording, so the second
    call costs an ``atan2`` and a dictionary lookup.
    """

    def __init__(self, spec: CircularProjectionSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def unit(self) -> str:
        return self._spec.unit

    @property
    def category(self) -> str:
        return "vcg"

    @property
    def description(self) -> str:
        return _projection_description(self._spec)

    @property
    def references(self) -> list[str]:
        return [
            "Isaza N et al. The spatial ventricular gradient is associated "
            "with inducibility of ventricular arrhythmias during "
            "electrophysiology study. Heart Rhythm 2024;21(11):2160-2167."
        ]

    def compute(
        self,
        delineation: RecordDelineation,
        ecg_12lead: Optional[np.ndarray] = None,
        vcg_xyz: Optional[np.ndarray] = None,
        fs: int = 500,
    ) -> BiomarkerResult:
        notes = [f"circular_projection_version={CIRCULAR_PROJECTION_VERSION}"]
        value: Optional[float] = None
        confidence = 0.0
        try:
            source = BiomarkerRegistry.get(self._spec.source)
            result = source.compute(delineation, ecg_12lead, vcg_xyz, fs)
            value = project_angle(
                result.value, self._spec.component, self._spec.multiple
            )
            if value is None:
                reason = result.computation_notes or "source returned no angle"
                notes.insert(0, f"not_evaluable: {reason}")
            else:
                confidence = float(result.confidence)
        except Exception as error:  # noqa: BLE001 - one marker must not stop a panel
            notes.insert(0, f"not_evaluable: {type(error).__name__}: {error}")
        return BiomarkerResult(
            name=self._spec.name,
            value=value,
            unit=self._spec.unit,
            category="vcg",
            confidence=confidence,
            description=_projection_description(self._spec),
            references=self.references,
            computation_notes="; ".join(notes),
            family=self._spec.family,
            formula=_projection_formula(self._spec),
        )


def circular_projection_marker_ids() -> tuple[str, ...]:
    return tuple(spec.name for spec in CIRCULAR_PROJECTION_SPECS)


def register_circular_projection_biomarkers() -> None:
    existing = set(BiomarkerRegistry.available())
    for spec in CIRCULAR_PROJECTION_SPECS:
        if spec.name in existing:
            if isinstance(
                BiomarkerRegistry.get(spec.name), CircularProjectionBiomarker
            ):
                continue
            raise RuntimeError(
                "Circular projection collides with an existing biomarker: "
                f"{spec.name}"
            )
        BiomarkerRegistry.register_definition(
            BiomarkerDefinition(
                name=spec.name,
                family=spec.family,
                description=_projection_description(spec),
                formula=_projection_formula(spec),
                unit=spec.unit,
                category="vcg",
                source=CIRCULAR_PROJECTION_SOURCE,
                pathological_direction="data_driven",
                pathological_note=(
                    "A faithful scale for an angular marker, not a new "
                    "physiological quantity. The source marker is a "
                    "coordinate on a full turn, which a rank-based screen "
                    "cannot order: the seam at +/-180 degrees separates "
                    "values that are adjacent in the body. Measured over "
                    "1.168 marker-endpoint pairs, a faithful representation "
                    "separates better by more than 0,05 AUC in 25,4 % of "
                    "them, and 92 of those are invisible to the current "
                    "screen. No diagnostic cutoff is claimed."
                ),
            )
        )
        BiomarkerRegistry.register(CircularProjectionBiomarker(spec))
