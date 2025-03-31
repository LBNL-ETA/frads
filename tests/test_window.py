import os
from pathlib import Path
import unittest
from frads.window import (
    create_glazing_system,
    Gas,
    Gap,
    GlazingSystem,
    AIR,
    ARGON,
    LayerInput,
)


class TestWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resources_dir = Path(__file__).parent / "Resources"

    def setUp(self):
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        shade_path = self.resources_dir / "2011-SA1.xml"
        blinds_path = self.resources_dir / "igsdb_product_19732.json"
        glass_layer = LayerInput(glass_path)
        shade_layer = LayerInput(shade_path)
        blinds_layer = LayerInput(blinds_path, slat_angle=45)
        self.double_glaze = [glass_layer, glass_layer]
        self.double_glaze_shade = [glass_layer, glass_layer, shade_layer]
        self.single_glaze_blinds = [glass_layer, blinds_layer]
        self.glazing_system = create_glazing_system(name="gs1", layer_inputs=self.double_glaze)

    def test_save_and_load(self):
        """
        Test the save method of the GlazingSystem class.
        """
        self.glazing_system.save("test.json")
        self.assertTrue(Path("test.json").exists())
        gs2 = GlazingSystem.from_json("test.json")
        os.remove("test.json")
        self.assertEqual(gs2.name, self.glazing_system.name)
        self.assertEqual(gs2.visible_back_reflectance, self.glazing_system.visible_back_reflectance)

    def test_simple_glazingsystem(self):
        """
        Test the GlazingSystem class.
        Build a GlazingSystem object consisting of two layer of clear glass.

        Check the thickness of the glazing system.
        Check the order and name of the layers.
        Check the composition of the default gap.
        """

        self.assertEqual(self.glazing_system.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(self.glazing_system.layers[1].product_name, "Generic Clear Glass")
        self.assertEqual(self.glazing_system.name, "gs1")
        self.assertEqual(self.glazing_system.gaps[0].gas[0].gas, "air")
        self.assertEqual(self.glazing_system.gaps[0].gas[0].ratio, 1)
        self.assertEqual(self.glazing_system.gaps[0].thickness, 0.0127)


    def test_customized_gap(self):
        """
        Test the building of a customized gap.
        A 0.03 m thick gap between the two glass layers. The gap is filled with 90% argon and 10% air.

        Check the thickness of the glazing system.
        Check the order and composition of the gap.
        """
        gs = create_glazing_system(
            name="gs2",
            layer_inputs=self.double_glaze,
            gaps=[Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03)],
        )

        self.assertEqual(gs.gaps[0].gas[0].gas, "air")
        self.assertEqual(gs.gaps[0].gas[0].ratio, 0.1)
        self.assertEqual(gs.gaps[0].gas[1].gas, "argon")
        self.assertEqual(gs.gaps[0].gas[1].ratio, 0.9)
        self.assertEqual(gs.gaps[0].thickness, 0.03)


    def test_multilayer_glazing_shading(self):
        """
        Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

        Check the thickness of the glazing system.
        Check the order of the layers.
        Check the order and composition of the gaps.
        """
        gs = create_glazing_system(
            name="gs3",
            layer_inputs=self.double_glaze_shade,
            gaps=[
                Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03),
                Gap([Gas("air", 1)], 0.01),
            ],
        )

        self.assertEqual(gs.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(gs.layers[1].product_name, "Generic Clear Glass")
        self.assertEqual(gs.layers[2].product_name, "Satine 5500 5%, White Pearl")

        self.assertEqual(gs.name, "gs3")
        self.assertEqual(gs.gaps[0].gas[0].gas, "air")
        self.assertEqual(gs.gaps[0].gas[0].ratio, 0.1)
        self.assertEqual(gs.gaps[0].gas[1].gas, "argon")
        self.assertEqual(gs.gaps[0].gas[1].ratio, 0.9)
        self.assertEqual(gs.gaps[0].thickness, 0.03)
        self.assertEqual(gs.gaps[1].gas[0].gas, "air")
        self.assertEqual(gs.gaps[1].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[1].thickness, 0.01)

        self.assertTrue(gs.visible_back_reflectance is not None)
        self.assertTrue(gs.solar_back_absorptance is not None)


    def test_venetian_blinds(self):
        """
        Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

        Check the thickness of the glazing system.
        Check the order of the layers.
        Check the order and composition of the gaps.
        """
        gs = create_glazing_system(
            name="gs3",
            layer_inputs=self.single_glaze_blinds,
        )

        self.assertEqual( gs.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual( gs.layers[0].thickness, 0.003048)
        self.assertEqual( gs.layers[1].product_name, "ODL Espresso Blind 14.8mm Slat")
        self.assertEqual( gs.layers[1].thickness, 0.010465180361560904)

        self.assertEqual( gs.gaps[0].gas[0].gas, "air")
        self.assertEqual( gs.gaps[0].gas[0].ratio, 1)
        self.assertEqual( gs.gaps[0].thickness, 0.0127)

        self.assertEqual( gs.thickness, 0.026213180361560902)
