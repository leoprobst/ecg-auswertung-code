"""Shared display and inventory contract for anatomical loop projections."""
from typing import Any

from .biomarkers.catalog_biomarkers import PLANES
from .vcg_geometry import VCG_AXIS_CONVENTION_VERSION


def projection_metadata(marker_id: str) -> dict[str, Any] | None:
    for segment in ("p", "qrs", "qrst", "st", "t"):
        for plane, indices in PLANES.items():
            prefix = f"{segment}_loop_{plane}_"
            if marker_id.startswith(prefix):
                return {
                    "plane": plane,
                    "axes": ["XYZ"[index] for index in indices],
                    "axis_convention": VCG_AXIS_CONVENTION_VERSION,
                    "segment": segment,
                    "geometry_measure": marker_id[len(prefix):],
                    "transform_dependent": True,
                    "clinical_threshold_validated": False,
                    "display_axis_order_must_be_preserved": True,
                }
    return None
