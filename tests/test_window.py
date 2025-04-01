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
    Layer,
    get_glazing_layer_groups
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
        self.triple_glaze = [glass_layer, glass_layer, glass_layer]
        self.double_glaze_shade = [glass_layer, glass_layer, shade_layer]
        self.double_glaze_shade = [glass_layer, glass_layer, blinds_layer]
        self.single_glaze_blinds = [glass_layer, blinds_layer]
        self.dgu_glazing_system = create_glazing_system(name="dgu", layer_inputs=self.double_glaze)
        self.tgu_glazing_system = create_glazing_system(name="tgu", layer_inputs=self.triple_glaze)
        # self.dgu_shade_glazing_system = create_glazing_system(name="dgu_shade", layer_inputs=self.double_glaze_shade)
        # self.dgu_blinds_glazing_system = create_glazing_system(name="dgu_blinds", layer_inputs=self.double_glaze_blinds)

    def test_save_and_load(self):
        """
        Test the save method of the GlazingSystem class.
        """
        self.dgu_glazing_system.save("test.json")
        self.assertTrue(Path("test.json").exists())
        gs2 = GlazingSystem.from_json("test.json")
        os.remove("test.json")
        self.assertEqual(gs2.name, self.dgu_glazing_system.name)
        self.assertEqual(gs2.visible_back_reflectance, self.dgu_glazing_system.visible_back_reflectance)

    def test_simple_glazingsystem(self):
        """
        Test the GlazingSystem class.
        Build a GlazingSystem object consisting of two layer of clear glass.

        Check the thickness of the glazing system.
        Check the order and name of the layers.
        Check the composition of the default gap.
        """

        self.assertEqual(self.dgu_glazing_system.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(self.dgu_glazing_system.layers[1].product_name, "Generic Clear Glass")
        self.assertEqual(self.dgu_glazing_system.name, "dgu")
        self.assertEqual(self.dgu_glazing_system.gaps[0].gas[0].gas, "air")
        self.assertEqual(self.dgu_glazing_system.gaps[0].gas[0].ratio, 1)
        self.assertEqual(self.dgu_glazing_system.gaps[0].thickness, 0.0127)

    def test_get_glazing_layer_groups(self):
        glass_layer = Layer("glass", 0.03, "glazing", 1, 1, 1, 0 ,None,)
        shade_layer = Layer("shade", 0.03, "fabric", 1, 1, 1, 0 ,None,)
        blinds_layer = Layer("blinds", 0.03, "blinds", 1, 1, 1, 0 ,None,)
        group1 = get_glazing_layer_groups([glass_layer, glass_layer, shade_layer])
        group2 = get_glazing_layer_groups([shade_layer, glass_layer])
        group3 = get_glazing_layer_groups([glass_layer, glass_layer, glass_layer, blinds_layer])
        group4 = get_glazing_layer_groups([glass_layer, shade_layer, glass_layer, blinds_layer])
        group5 = get_glazing_layer_groups([glass_layer, glass_layer, glass_layer])
        self.assertEqual(group1, [('glazing', 2), ('fabric', 1)])
        self.assertEqual(group2, [('fabric', 1), ('glazing', 1)])
        self.assertEqual(group3, [('glazing', 3), ('blinds', 1)])
        self.assertEqual(group4, [('glazing', 1), ('fabric', 1), ('glazing', 1), ('blinds', 1)])
        self.assertEqual(group5, [('glazing', 3)])

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
            nsamp=1,
        )

        self.assertEqual(gs.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(gs.layers[0].thickness, 0.003048)
        self.assertEqual(gs.layers[1].product_name, "ODL Espresso Blind 14.8mm Slat")
        self.assertEqual(gs.layers[1].thickness, 0.010465180361560904)

        self.assertEqual(gs.gaps[0].gas[0].gas, "air")
        self.assertEqual(gs.gaps[0].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[0].thickness, 0.0127)

        self.assertEqual(gs.thickness, 0.026213180361560902)

if __name__ == "__main__":
    unittest.main()
