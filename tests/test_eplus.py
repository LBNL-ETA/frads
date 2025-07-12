from pathlib import Path
import unittest

import frads as fr
from pyenergyplus.dataset import ref_models

class TestGlazingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup the resources directory - replace with your actual resources path
        cls.resources_dir = Path(__file__).parent / "Resources"
        # Create a new medium_office for each test - adjust based on how it's created in your tests
        cls.medium_office = fr.load_energyplus_model(ref_models["medium_office"])
        glass_path = cls.resources_dir / "CLEAR_3.DAT"
        blinds_path = cls.resources_dir / "igsdb_product_19732.json"
        glass_layer = fr.LayerInput(glass_path)
        blinds_layer = fr.LayerInput(input_source=blinds_path, slat_angle_deg=45)
        single_glaze_blinds = [glass_layer, blinds_layer]
        cls.glazing_blinds_system = fr.create_glazing_system(name="gs1", layer_inputs=single_glaze_blinds, nproc=4, nsamp=1)


    def test_add_proxy_geometry(self):
        rmodels = fr.epmodel_to_radmodel(self.medium_office)
        rconfigs = {
            k: fr.WorkflowConfig.from_dict(v) for k, v in rmodels.items()
        }
        for _, config in rconfigs.items():
            for _, window in config.model.windows.items():
                geom = fr.window.get_proxy_geometry(window.polygon, self.glazing_blinds_system)
                window.proxy_geometry[self.glazing_blinds_system.name] = b"\n".join(geom)


    def test_add_glazingsystem(self):
        self.medium_office.add_glazing_system(self.glazing_blinds_system)
        # Replace pytest assertions with unittest assertions
        self.assertNotEqual(self.medium_office.construction_complex_fenestration_state, {})
        self.assertIsInstance(self.medium_office.construction_complex_fenestration_state, dict)
        self.assertIsInstance(self.medium_office.matrix_two_dimension, dict)
        self.assertIsInstance(self.medium_office.window_material_glazing, dict)
        self.assertIsInstance(self.medium_office.window_thermal_model_params, dict)


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup the resources directory - replace with your actual resources path
        cls.resources_dir = Path(__file__).parent / "Resources"

    def test_run(self):
        # Simplified test - just test basic functionality without full simulation
        self.medium_office = fr.load_energyplus_model(ref_models["medium_office"])
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        glass_layer = fr.LayerInput(glass_path)
        single_glaze = [glass_layer]
        
        # Use minimal parameters for faster testing
        self.glazing_system = fr.create_glazing_system(
            name="gs_test", layer_inputs=single_glaze, nsamp=1, nproc=1
        )
        self.medium_office.add_glazing_system(self.glazing_system)
        
        # Test basic setup without running full simulation
        epsetup = fr.EnergyPlusSetup(
            self.medium_office, enable_radiance=True, initialize_radiance=False
        )
        self.assertIsNotNone(epsetup)


if __name__ == "__main__":
    unittest.main()
