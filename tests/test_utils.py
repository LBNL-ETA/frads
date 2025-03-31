import os
from pathlib import Path
import sys
sys.path.append(".")
import unittest

from frads import utils
from pyradiance import Primitive


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir_path = Path(os.path.dirname(__file__))
        resource_dir = test_dir_path / "Resources"
        cls.check_decimal_to = 6
        cls.reinsrc6_path = resource_dir / "reinsrc6.rad"
        cls.grid_path = resource_dir / "grid.pts"
        cls.prim_path = resource_dir / "model" / "Objects" / "floor_openroom.rad"


    def test_polygon2prim(self):
        pass

    def test_primitive_normal(self):
        pass

    def test_samp_dir(self):
        pass

    def test_up_vector(self):
        pass

    def test_neutral_plastic_prim(self):
        pass

    def test_neutral_trans_prim(self):
        pass

    def test_color_plastic_prim(self):
        pass

    def test_glass_prim(self):
        pass

    def test_bsdf_prim(self):
        pass

    def test_lambda_calc(self):
        pass

    def test_angle_basis_coeff(self):
        pass

    def test_opt2list(self):
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

    def test_varays(self):
        pass


    def test_unpack_primitives(self):
        prims = utils.unpack_primitives(self.prim_path)
        assert isinstance(prims[0], Primitive)

    def test_nest_list(self):
        pass

    def test_write_square_matrix(self):
        pass

    def test_dhi2dni(self):
        pass

    def test_is_number(self):
        pass

    def test_silent_remove(self):
        pass

    def test_square2disk(self):
        pass

    def test_id_generator(self):
        pass

    def get_nested_list_levels(self):
        pass
