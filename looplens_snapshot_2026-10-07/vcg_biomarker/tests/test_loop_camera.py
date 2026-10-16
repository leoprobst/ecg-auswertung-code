import math
import pytest
from vcg_biomarker.gui.widgets.loop_camera import loop_camera_distance


@pytest.mark.parametrize("width,height", [(650,190),(990,295),(300,900),(600,600)])
def test_whole_bounding_sphere_fits_limiting_field_of_view(width,height):
    radius = 3.
    distance = loop_camera_distance(radius,width,height)
    half_horizontal = math.radians(60)/2
    half_vertical = math.atan(math.tan(half_horizontal)*height/width)
    angular_radius = math.asin(radius/distance)
    assert angular_radius < min(half_horizontal,half_vertical)


def test_flatter_viewport_needs_more_distance_and_scale_is_linear():
    assert loop_camera_distance(1,900,200) > loop_camera_distance(1,900,600)
    assert loop_camera_distance(2,900,200) == 2*loop_camera_distance(1,900,200)


@pytest.mark.parametrize("radius,width,height", [(0,600,200),(1,0,200),(1,600,float("nan"))])
def test_invalid_camera_geometry_rejected(radius,width,height):
    with pytest.raises(ValueError):
        loop_camera_distance(radius,width,height)
