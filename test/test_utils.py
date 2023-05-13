import math
import os
from pathlib import Path
import sys
sys.path.append(".")

from frads import geom
from frads import utils
from frads.types import PaneProperty
from frads.types import PaneRGB
from pyradiance import Primitive


test_dir_path = Path(os.path.dirname(__file__))
resource_dir = test_dir_path / "Resources"
check_decimal_to = 6
reinsrc6_path = resource_dir / "reinsrc6.rad"
grid_path = resource_dir / "grid.pts"
prim_path = resource_dir / "model" / "Objects" / "floor_openroom.rad"


pane_property_1 = PaneProperty(
    "VE1-2M_Low-E",  # name
    0.01,  # thickness
    "coated",  # gtype,
    "front",  # coated_side,
    [],  # wavelengths,
    [],  # transmittance,
    [],  # refl_front,
    [],  # refl_back,
)

pane_property_2 = PaneProperty(
    "PVB_laminated",  # name
    0.01,  # thickness
    "laminate",  # gtype,
    "front",  # coated_side,
    [],  # wavelengths,
    [],  # transmittance,
    [],  # refl_front,
    [],  # refl_back,
)

pane_rgb_1 = PaneRGB(
    pane_property_1,
    (0.042, 0.049, 0.043),
    (0.065, 0.058, 0.067),
    (0.756, 0.808, 0.744),
)

pane_rgb_2 = PaneRGB(
    pane_property_2,
    (0.11, 0.11, 0.11),
    (0.11, 0.11, 0.11),
    (0.63, 0.63, 0.63),
)

def test_polygon2prim():
    pass

def test_primitive_normal():
    pass

def test_samp_dir():
    pass

def test_up_vector():
    pass

def test_neutral_plastic_prim():
    pass

def test_neutral_trans_prim():
    pass

def test_color_plastic_prim():
    pass

def test_glass_prim():
    pass

def test_bsdf_prim():
    pass

def test_lambda_calc():
    pass

def test_angle_basis_coeff():
    pass

def test_opt2list():
    opt1 = {"ab": 2, "ad": 1024, "I": True}
    res1 = utils.opt2list(opt1)
    answer1 = ["-ab", "2", "-ad", "1024", "-I+"]
    assert res1 == answer1
    opt2 = {"vt": "a", "vf": "test/v1a.vf", "vp":[0, 0, 0]}
    res2 = utils.opt2list(opt2)
    answer2 = ["-vta", "-vf", "test/v1a.vf", "-vp", "0", "0", "0"]
    assert res2 == answer2
    opt3 = {"ab": 3, "ad": 64, "av": [.38, .38, .38], "u": True, "vt": "a"}
    res3 = utils.opt2list(opt3)
    answer3 = ["-ab", "3", "-ad", "64",
               "-av", "0.38", "0.38", "0.38", "-u+", "-vta"]
    assert res3 == answer3


def test_calc_reinsrc_dir():
    mf = 6
    vecs, omgs = utils.calc_reinsrc_dir(mf)
    with open(reinsrc6_path) as rdr:
        answer_lines = rdr.readlines()
    answer_vecs = []
    answer_omgs = []
    for vec, omg, line in zip(vecs, omgs, answer_lines):
        x, y, z, aomg = map(float, line.strip().split())
        assert round(vec.x, 3) == round(x, 3)
        assert round(vec.y, 3) == round(y, 3)
        assert round(vec.z, 3) == round(z, 3)
        assert round(omg, 3) == round(aomg, 3)

def test_pt_inclusion():
    pass

def test_gen_grid():
    polygon = geom.Polygon([
        geom.Vector(0, 0, 0),
        geom.Vector(0, 14, 0),
        geom.Vector(12, 14, 0),
        geom.Vector(12, 0, 0),
    ])
    height = 0.76
    spacing = 1
    result = utils.gen_grid(polygon, height, spacing)
    with open(grid_path) as rdr:
        lines = rdr.readlines()
    assert len(result) == len(lines)
    for res, ans in zip(result, lines):
        for r, a in zip(res, ans.split()):
            assert r == float(a)


def test_gen_blinds():
    pass

def test_analyze_vert_polygon():
    pass

def test_varays():
    pass

def test_get_glazing_primitive():
    prim = utils.get_glazing_primitive([pane_rgb_1, pane_rgb_2])
    res = str(prim).strip().replace("\n", " ")
    answer = ("void BRTDfunc VE1-2M_Low-E+PVB_laminated 10 if(Rdot,cr(fr("
              "0.110),ft(0.630),fr(0.065)),cr(fr(0.042),ft(0.756),fr(0.110)))"
              " if(Rdot,cr(fr(0.110),ft(0.630),fr(0.058)),cr(fr(0.049),ft("
              "0.808),fr(0.110))) if(Rdot,cr(fr(0.110),ft(0.630),fr(0.067)),"
              "cr(fr(0.043),ft(0.744),fr(0.110))) ft(0.630)*ft(0.756) ft("
              "0.630)*ft(0.808) ft(0.630)*ft(0.744) 0 0 0 glaze2.cal 0 9 0 "
              "0 0 0 0 0 0 0 0")
    assert res == answer
    prim2 = utils.get_glazing_primitive([pane_rgb_1])
    res2 = str(prim2).strip().replace("\n", " ")
    answer2 = ("void BRTDfunc VE1-2M_Low-E "
               "10 sr_clear_r sr_clear_g sr_clear_b "
               "st_clear_r st_clear_g st_clear_b 0 0 0 "
               "glaze1.cal 0 19 0 0 0 0 0 0 0 0 0 "
               "1 0.065 0.058 0.067 0.042 0.049 0.043 "
               "0.756 0.808 0.744")
    assert res2 == answer2

def test_unpack_primitives():
    prims = utils.unpack_primitives(prim_path)
    assert isinstance(prims[0], Primitive)

def test_nest_list():
    pass

def test_write_square_matrix():
    pass

def test_dhi2dni():
    pass

def test_is_number():
    pass

def test_silent_remove():
    pass

def test_square2disk():
    pass

def test_id_generator():
    pass

def get_nested_list_levels():
    pass
