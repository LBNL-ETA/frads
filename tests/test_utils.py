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
