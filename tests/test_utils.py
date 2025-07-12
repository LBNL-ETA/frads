import os
from pathlib import Path
import unittest

from frads import utils
from pyradiance import Primitive


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir_path = Path(os.path.dirname(__file__))
        resource_dir = test_dir_path / "Resources"
        cls.prim_path = resource_dir / "model" / "Objects" / "floor_openroom.rad"

    def test_unpack_primitives(self):
        prims = utils.unpack_primitives(self.prim_path)
        self.assertIsInstance(prims[0], Primitive)
