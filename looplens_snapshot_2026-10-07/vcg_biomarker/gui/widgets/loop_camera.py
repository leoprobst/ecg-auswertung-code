"""Perspective camera fit for a sphere, retaining all measured loop points."""
import math


def loop_camera_distance(radius: float, width: float, height: float,
                         horizontal_fov: float = 60., margin: float = 1.15) -> float:
    if (not all(math.isfinite(x) for x in (radius, width, height, horizontal_fov, margin))
            or radius <= 0 or width <= 0 or height <= 0
            or not 0 < horizontal_fov < 180 or margin < 1):
        raise ValueError("Finite positive loop and viewport geometry required")
    half_horizontal = math.radians(horizontal_fov)/2
    half_limiting = math.atan(math.tan(half_horizontal)*min(1., height/width))
    # Tangent to the full bounding sphere, not an orthographic r/tan shortcut.
    return margin * radius / math.sin(half_limiting)
