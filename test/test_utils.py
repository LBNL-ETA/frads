import os
from pathlib import Path
import sys
sys.path.append(".")

from frads import utils
from pyradiance import Primitive


test_dir_path = Path(os.path.dirname(__file__))
resource_dir = test_dir_path / "Resources"
check_decimal_to = 6
reinsrc6_path = resource_dir / "reinsrc6.rad"
grid_path = resource_dir / "grid.pts"
prim_path = resource_dir / "model" / "Objects" / "floor_openroom.rad"


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
    answer3 = ["-ab", "3", "-ad", "64", "-av", "0.38", "0.38", "0.38", "-u+", "-vta"]
    assert res3 == answer3

def test_varays():
    pass


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
