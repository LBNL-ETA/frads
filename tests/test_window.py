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
    GlazingLayerDefinition,
    FabricLayerDefinition,
    BlindsLayerDefinition,
    OpeningDefinitions,
    Layer,
    get_glazing_layer_groups,
    load_glazing_system,
)


class TestWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resources_dir = Path(__file__).parent / "Resources"

    def setUp(self):
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        shade_path = self.resources_dir / "2011-SA1.xml"
        blinds_path = self.resources_dir / "igsdb_product_19732.json"
        glass_layer = GlazingLayerDefinition(glass_path)
        openings = OpeningDefinitions(
            left_m=0.01,
            right_m=0.005,
            top_m=0.0025,
            bottom_m=0.005,
        )
        fabric_layer = FabricLayerDefinition(
            input_source=shade_path,
            openings=openings,
        )
        blinds_layer = BlindsLayerDefinition(
            input_source=blinds_path,
            slat_angle_deg=45,
        )
        self.double_glaze = [glass_layer, glass_layer]
        triple_glaze = [glass_layer, glass_layer, glass_layer]
        double_glaze_fabric = [glass_layer, glass_layer, fabric_layer]
        double_glaze_blinds = [glass_layer, glass_layer, blinds_layer]
        self.single_glaze_blinds = [glass_layer, blinds_layer]
        self.double_glaze_inner_fabric = [glass_layer, fabric_layer, glass_layer]
        self.double_glaze_system = create_glazing_system(
            name="dgu", layer_inputs=self.double_glaze
        )
        self.triple_glaze_system = create_glazing_system(
            name="tgu", layer_inputs=triple_glaze
        )

        self.double_glaze_fabric_system = create_glazing_system(
            name="dgu_shade", layer_inputs=double_glaze_fabric
        )
        self.double_glaze_blinds_system = create_glazing_system(
            name="dgu_blinds", layer_inputs=double_glaze_blinds, nsamp=1,
        )

    def test_save_and_load(self):
        """
        Test the save method of the GlazingSystem class.
        """
        self.double_glaze_fabric_system.save("test.json")
        self.assertTrue(Path("test.json").exists())
        gs2 = load_glazing_system("test.json")
        # os.remove("test.json")
        self.assertEqual(gs2.name, self.double_glaze_fabric_system.name)
        self.assertEqual(
            gs2.visible_back_reflectance,
            self.double_glaze_fabric_system.visible_back_reflectance,
        )

    def test_simple_glazingsystem(self):
        """
        Test the GlazingSystem class.
        Build a GlazingSystem object consisting of two layer of clear glass.

        Check the thickness of the glazing system.
        Check the order and name of the layers.
        Check the composition of the default gap.
        """

        self.assertEqual(
            self.double_glaze_system.layers[0].product_name, "Generic Clear Glass"
        )
        self.assertEqual(
            self.double_glaze_system.layers[1].product_name, "Generic Clear Glass"
        )
        self.assertEqual(self.double_glaze_system.name, "dgu")
        self.assertEqual(self.double_glaze_system.gaps[0].gas[0].gas, "air")
        self.assertEqual(self.double_glaze_system.gaps[0].gas[0].ratio, 1)
        self.assertEqual(self.double_glaze_system.gaps[0].thickness, 0.0127)

    def test_get_glazing_layer_groups(self):
        glass_layer = Layer(
            "glass",
            0.03,
            "glazing",
            1,
            1,
            1,
            0,
            None,
        )
        shade_layer = Layer(
            "shade",
            0.03,
            "fabric",
            1,
            1,
            1,
            0,
            None,
        )
        blinds_layer = Layer(
            "blinds",
            0.03,
            "blinds",
            1,
            1,
            1,
            0,
            None,
        )
        group1 = get_glazing_layer_groups([glass_layer, glass_layer, shade_layer])
        group2 = get_glazing_layer_groups([shade_layer, glass_layer])
        group3 = get_glazing_layer_groups(
            [glass_layer, glass_layer, glass_layer, blinds_layer]
        )
        group4 = get_glazing_layer_groups(
            [glass_layer, shade_layer, glass_layer, blinds_layer]
        )
        group5 = get_glazing_layer_groups([glass_layer, glass_layer, glass_layer])
        self.assertEqual(group1, [("glazing", 2), ("fabric", 1)])
        self.assertEqual(group2, [("fabric", 1), ("glazing", 1)])
        self.assertEqual(group3, [("glazing", 3), ("blinds", 1)])
        self.assertEqual(
            group4, [("glazing", 1), ("fabric", 1), ("glazing", 1), ("blinds", 1)]
        )
        self.assertEqual(group5, [("glazing", 3)])

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
        gs.save("temp.json")

        self.assertEqual(gs.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(gs.layers[0].thickness_m, 0.003048)
        self.assertEqual(gs.layers[1].product_name, "ODL Espresso Blind 14.8mm Slat")
        self.assertEqual(gs.layers[1].thickness_m, 0.010465180361560904)

        self.assertEqual(gs.gaps[0].gas[0].gas, "air")
        self.assertEqual(gs.gaps[0].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[0].thickness, 0.0127)

        self.assertEqual(gs.thickness, 0.026213180361560902)

    def test_inner_shade_gap(self):
        """
        Test left, right, top and bottom opening multipliers for inner shade with customized gap.

        Check gaps thickness
        Check the order of the gaps
        Check the thickness of the glazing system.
        Check the opening multipliers of the shading layer
        """

        gs = create_glazing_system(
            name="dgu_inner_shade",
            layer_inputs=self.double_glaze_inner_fabric,
            gaps=[Gap([Gas("air", 1)], 0.01), Gap([Gas("air", 1)], 0.005)],
        )

        self.assertEqual(gs.gaps[0].gas[0].gas, "air")
        self.assertEqual(gs.gaps[0].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[0].thickness, 0.01)

        self.assertEqual(gs.gaps[1].gas[0].gas, "air")
        self.assertEqual(gs.gaps[1].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[1].thickness, 0.005)

        self.assertEqual(gs.thickness, 0.022096)

        self.assertEqual(gs.layers[1].opening_multipliers.left, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.right, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.top, 0.5)
        self.assertEqual(gs.layers[1].opening_multipliers.bottom, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.front, 0.05)

if __name__ == "__main__":
    unittest.main()
