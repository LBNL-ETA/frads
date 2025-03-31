from pathlib import Path
import tempfile

from frads.eplus import EnergyPlusSetup, load_energyplus_model, add_proxy_geometry_to_rmodels
from frads.window import create_glazing_system, LayerInput, GlazingSystem
from pyenergyplus.dataset import ref_models
import unittest

class TestGlazingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup the resources directory - replace with your actual resources path
        cls.resources_dir = Path(__file__).parent / "Resources"

    def setUp(self):
        # Create a new medium_office for each test - adjust based on how it's created in your tests
        self.medium_office = load_energyplus_model(ref_models["medium_office"])
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        blinds_path = self.resources_dir / "igsdb_product_19732.json"
        glass_layer = LayerInput(glass_path)
        blinds_layer = LayerInput(blinds_path, slat_angle=45)
        single_glaze_blinds = [glass_layer, blinds_layer]
        self.glazing_blinds_system = create_glazing_system(name="gs1", layer_inputs=single_glaze_blinds, nproc=8, nsamp=5)

    def test_add_glazingsystem(self):
        self.medium_office.add_glazing_system(self.glazing1_system)
        # Replace pytest assertions with unittest assertions
        self.assertNotEqual(self.medium_office.construction_complex_fenestration_state, {})
        self.assertIsInstance(self.medium_office.construction_complex_fenestration_state, dict)
        self.assertIsInstance(self.medium_office.matrix_two_dimension, dict)
        self.assertIsInstance(self.medium_office.window_material_glazing, dict)
        self.assertIsInstance(self.medium_office.window_thermal_model_params, dict)

    def test_add_glazingsystem_with_blinds(self):
        self.medium_office.add_glazing_system(self.glazing_blinds_system)

        epsetup = EnergyPlusSetup(self.medium_office)
        epsetup.run(design_day=True)
        # assert medium_office.construction_complex_fenestration_state != {}
        # assert isinstance(
        #     medium_office.construction_complex_fenestration_state, dict
        # )
        # assert isinstance(medium_office.matrix_two_dimension, dict)
        # assert isinstance(medium_office.window_material_glazing, dict)
        # assert isinstance(medium_office.window_thermal_model_params, dict)

    def test_add_blinds_geometry(self):
        self.medium_office.add_glazing_system(self.glazing_blinds_system)
        epsetup = EnergyPlusSetup(self.medium_office, enable_radiance=True, nproc=8)
        add_proxy_geometry_to_rmodels(epsetup, self.glazing_blinds_system)

if __name__ == "__main__":
    unittest.main()
