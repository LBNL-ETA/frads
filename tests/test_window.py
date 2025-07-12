from pathlib import Path
import unittest
import pywincalc as pwc
from frads.window import (
    create_glazing_system,
    Gas,
    Gap,
    LayerInput,
    OpeningDefinitions,
    Layer,
    get_layer_groups,
    load_glazing_system,
    generate_melanopic_bsdf,
)
import pyradiance as pr


class TestWindow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resources_dir = Path(__file__).parent / "Resources"
        glass_path = cls.resources_dir / "CLEAR_3.DAT"
        shade_path = cls.resources_dir / "2011-SA1.xml"
        blinds_path = cls.resources_dir / "igsdb_product_19732.json"
        glass_layer = LayerInput(glass_path)
        openings = OpeningDefinitions(
            left_m=0.01,
            right_m=0.005,
            top_m=0.0025,
            bottom_m=0.005,
        )
        fabric_layer = LayerInput(
            input_source=shade_path,
            openings=openings,
        )
        blinds_layer = LayerInput(
            input_source=blinds_path,
            slat_angle_deg=45,
        )
        cls.double_glaze = [glass_layer, glass_layer]
        triple_glaze = [glass_layer, glass_layer, glass_layer]
        cls.double_glaze_fabric = [glass_layer, glass_layer, fabric_layer]
        double_glaze_blinds = [glass_layer, glass_layer, blinds_layer]
        cls.single_glaze_blinds = [glass_layer, blinds_layer]
        cls.double_glaze_inner_fabric = [glass_layer, fabric_layer, glass_layer]
        cls.double_glaze_system = create_glazing_system(
            name="dgu", layer_inputs=cls.double_glaze,
            gaps=[Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03)],
        )
        cls.triple_glaze_system = create_glazing_system(
            name="tgu", layer_inputs=triple_glaze
        )

        cls.double_glaze_fabric_system = create_glazing_system(
            name="dgu_shade", layer_inputs=cls.double_glaze_fabric
        )
        cls.double_glaze_blinds_system = create_glazing_system(
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

    def test_get_layer_groups(self):
        glass_layer = Layer(
            product_name= "glass",
            thickness_m= 0.03,
            product_type= "glazing",
            conductivity= 1,
            emissivity_front= 1,
            emissivity_back= 1,
            ir_transmittance= 0,
            flipped= False,
        )
        shade_layer = Layer(
            product_name= "shade",
            thickness_m= 0.03,
            product_type= "fabric",
            conductivity= 1,
            emissivity_front= 1,
            emissivity_back= 1,
            ir_transmittance= 0,
            flipped= False,
        )
        blinds_layer = Layer(
            product_name= "blinds",
            thickness_m= 0.03,
            product_type= "blinds",
            conductivity= 1,
            emissivity_front= 1,
            emissivity_back= 1,
            ir_transmittance= 0,
            flipped= False,
        )
        group1 = get_layer_groups([glass_layer, glass_layer, shade_layer])
        group2 = get_layer_groups([shade_layer, glass_layer])
        group3 = get_layer_groups(
            [glass_layer, glass_layer, glass_layer, blinds_layer]
        )
        group4 = get_layer_groups(
            [glass_layer, shade_layer, glass_layer, blinds_layer]
        )
        group5 = get_layer_groups([glass_layer, glass_layer, glass_layer])
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
        self.assertEqual(self.double_glaze_system.gaps[0].gas[0].gas, "air")
        self.assertEqual(self.double_glaze_system.gaps[0].gas[0].ratio, 0.1)
        self.assertEqual(self.double_glaze_system.gaps[0].gas[1].gas, "argon")
        self.assertEqual(self.double_glaze_system.gaps[0].gas[1].ratio, 0.9)
        self.assertEqual(self.double_glaze_system.gaps[0].thickness_m, 0.03)

    def test_venetian_blinds(self):
        """
        Test GlazingSystem object with multiple layers of glazing and shading and more than one customized gap.

        Check the thickness of the glazing system.
        Check the order of the layers.
        Check the order and composition of the gaps.
        """

        self.assertEqual(self.double_glaze_blinds_system.layers[0].product_name, "Generic Clear Glass")
        self.assertEqual(self.double_glaze_blinds_system.layers[0].thickness_m, 0.003048)
        self.assertEqual(self.double_glaze_blinds_system.layers[2].product_name, "ODL Espresso Blind 14.8mm Slat")
        self.assertEqual(self.double_glaze_blinds_system.layers[2].thickness_m, 0.010465180361560904)

        self.assertEqual(self.double_glaze_blinds_system.gaps[0].gas[0].gas, "air")
        self.assertEqual(self.double_glaze_blinds_system.gaps[0].gas[0].ratio, 1)
        self.assertEqual(self.double_glaze_blinds_system.gaps[0].thickness_m, 0.0127)

        self.assertEqual(self.double_glaze_blinds_system.thickness, 0.04196118036156091)

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
        self.assertEqual(gs.gaps[0].thickness_m, 0.01)

        self.assertEqual(gs.gaps[1].gas[0].gas, "air")
        self.assertEqual(gs.gaps[1].gas[0].ratio, 1)
        self.assertEqual(gs.gaps[1].thickness_m, 0.005)

        self.assertEqual(gs.thickness, 0.022096)

        self.assertEqual(gs.layers[1].opening_multipliers.left, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.right, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.top, 0.5)
        self.assertEqual(gs.layers[1].opening_multipliers.bottom, 1)
        self.assertEqual(gs.layers[1].opening_multipliers.front, 0.05)

class TestBSDFGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resources_dir = Path(__file__).parent / "Resources"

    def test_melanopic_bsdf(self):
        product_data = pwc.parse_json_file(str(self.resources_dir / "igsdb_product_7406.json"))
        NM_PER_MM = 1e3
        layer = Layer(
            product_name= "glass",
            thickness_m= 0.03,
            product_type= "glazing",
            conductivity= 1,
            emissivity_front= 1,
            emissivity_back= 1,
            ir_transmittance= 0,
            flipped= False,
            spectral_data= {
                int(round(d.wavelength * NM_PER_MM)): (
                    d.direct_component.transmittance_front,
                    d.direct_component.reflectance_front,
                    d.direct_component.reflectance_back,
                )
                for d in product_data.measurements
            }
        )
        layer2 = Layer(
            product_name= "blinds",
            thickness_m= 0.03,
            product_type= "blinds",
            conductivity= 1,
            emissivity_front= 1,
            emissivity_back= 1,
            ir_transmittance= 0,
            flipped= False,
            shading_material=pr.ShadingMaterial(0.5, 0, 0),
            slat_width_m = 0.0160,
            slat_spacing_m = 0.0120,
            slat_thickness_m = 0.0006,
            slat_curve_m = 0.0,
            slat_angle_deg = 45.0,
            slat_conductivity = 160.00,
            nslats = 10,
        )
        result = generate_melanopic_bsdf(layers=[layer, layer2], gaps=[Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.03)],nsamp=1)

if __name__ == "__main__":
    unittest.main()
