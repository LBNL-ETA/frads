from pathlib import Path
from frads import geom, matrix, utils
import pyradiance as pr
import numpy as np
import pytest

@pytest.fixture
def window_polygon():
    return [
        geom.Polygon([np.array((0, 0.6667, 0)),
                      np.array((0, 0.6667, 3)),
                      np.array((2, 0.6667, 3)),
                      np.array((2, 0.6667, 0)),
                      ]),
        geom.Polygon([np.array((3, 0.6667, 0)),
                      np.array((3, 0.6667, 3)),
                      np.array((5, 0.6667, 3)),
                      np.array((5, 0.6667, 0)),
                      ])
    ]

@pytest.fixture
def window_primitives(window_polygon):
    return [
        pr.Primitive("void", "polygon", "window1", ("0"), window_polygon[0].coordinates),
        pr.Primitive("void", "polygon", "window2", ("0"), window_polygon[1].coordinates)
    ]

def test_surface_as_sender(window_primitives):
    basis = "kf"
    sender = matrix.SurfaceSender(
        window_primitives,
        basis,
    )
    assert sender.basis == "kf"
    assert sender.content is not None

def test_view_as_sender():
    view = pr.View(
        position=(0,  0, 0),
        direction=(0, -1, 0),
        horiz=180,
        vert=180,
        vtype='a',
    )
    ray_cnt = 5
    sender = matrix.ViewSender(view, ray_cnt, 4, 4)
    assert sender.xres == 4
    assert sender.yres == 4

def test_point_as_sender():
    pts_list = [[0,0,0,0,0,1], [0,0,3,0,0,1]]
    ray_cnt = 5
    sender = matrix.SensorSender(pts_list, ray_cnt)
    assert sender.yres == len(pts_list)

def test_surface_as_receiver(window_primitives):
    basis = "kf"
    out = None
    offset = 0.1
    receiver = matrix.SurfaceReceiver(
        window_primitives, basis, out=out, offset=offset)
    assert receiver.basis == 'kf'


def test_sky_as_receiver():
    basis = 'r1'
    out = Path("test.mtx")
    receiver = matrix.SkyReceiver(basis)
    assert receiver.basis == 'r1'


def test_sun_as_receiver():
    basis = 'r6'
    smx_path = None
    window_normals = None
    receiver = matrix.SunReceiver(basis, smx_path, window_normals)
    assert receiver.basis == "r6"

def test_surfaces_view_factor():
    mat = pr.Primitive(
        "void", "plastic", "mat", [], [0.5, 0.5, 0.5]
    )
    floor = pr.Primitive(
        "mat", "polygon", "floor", [], [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0]
    )
    walls = [
        pr.Primitive("mat", "polygon", "swall", [], [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0]),
        pr.Primitive("mat", "polygon", "ewall", [], [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]),
        pr.Primitive("mat", "polygon", "nwall", [], [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]),
        pr.Primitive("mat", "polygon", "wwall", [], [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]),
    ]
    ceiling = pr.Primitive(
        "mat", "polygon", "ceiling", [], [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1]
    )
    res = matrix.surfaces_view_factor([floor], [mat] + walls + [ceiling], ray_count=1000)
    assert sum(res["floor"].values()) == pytest.approx(1.0)
