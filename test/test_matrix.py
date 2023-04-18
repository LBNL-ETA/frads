from pathlib import Path
import sys
sys.path.append(".")

from frads import geom
from frads import matrix
import pyradiance as pr
from frads.types import View
from frads.types import ViewType

window_polygon = [
    geom.Polygon([geom.Vector(0, 0, 0),
                  geom.Vector(0, 0, 3),
                  geom.Vector(2, 0, 3),
                  geom.Vector(2, 0, 0),
                  ]),
    geom.Polygon([geom.Vector(3, 0, 0),
                  geom.Vector(3, 0, 3),
                  geom.Vector(5, 0, 3),
                  geom.Vector(5, 0, 0),
                  ])
]
window_primitives = [
    pr.Primitive("void", "polygon", "window1", ("0"), window_polygon[0].to_real()),
    pr.Primitive("void", "polygon", "window2", ("0"), window_polygon[1].to_real())
]

def test_surface_as_sender():
    basis = "kf"
    sender = matrix.surface_as_sender(
        window_primitives, basis, offset=None, left=None)
    assert sender.form == "s"
    assert sender.xres is None
    assert sender.yres is None

def test_view_as_sender():
    view = View(
        position=geom.Vector(0,  0, 0),
        direction=geom.Vector(0, -1, 0),
        hori=180,
        vert=180,
        vtype=ViewType.angular_fisheye,
        xres=4,
        yres=4,
    )
    ray_cnt = 5
    sender = matrix.view_as_sender(view, ray_cnt, view.xres, view.yres)
    assert sender.form == "v"
    assert sender.xres == view.xres
    assert sender.yres == view.yres

def test_point_as_sender():
    pts_list = [[0,0,0,0,0,1], [0,0,3,0,0,1]]
    ray_cnt = 5
    sender = matrix.points_as_sender(pts_list, ray_cnt)
    assert sender.form == "p"
    assert sender.xres is None
    assert sender.yres == len(pts_list) * ray_cnt

def test_surface_as_receiver():
    basis = "kf"
    out = None
    offset = 0.1
    receiver = matrix.surface_as_receiver(
        window_primitives, basis, out, offset=offset)
    assert receiver.receiver != ""
    assert receiver.basis == 'kf'
    assert receiver.modifier == ""


def test_sky_as_receiver():
    basis = 'r1'
    out = Path("test.mtx")
    receiver = matrix.sky_as_receiver(basis, out)
    assert receiver.receiver != ""
    assert receiver.basis == 'r1'
    assert receiver.modifier == ''


def test_sun_as_receiver():
    basis = 'r6'
    smx_path = None
    window_normals = None
    receiver = matrix.sun_as_receiver(basis, smx_path, window_normals)
    assert receiver.receiver != ""
    assert receiver.basis == "r6"
    assert receiver.modifier != ""


