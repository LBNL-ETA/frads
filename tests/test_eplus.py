from pathlib import Path
import unittest

import frads as fr
from pyenergyplus.dataset import ref_models

class TestGlazingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup the resources directory - replace with your actual resources path
        cls.resources_dir = Path(__file__).parent / "Resources"

    def setUp(self):
        # Create a new medium_office for each test - adjust based on how it's created in your tests
        self.medium_office = fr.load_energyplus_model(ref_models["medium_office"])
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        blinds_path = self.resources_dir / "igsdb_product_19732.json"
        glass_layer = fr.window.LayerInput(glass_path)
        blinds_layer = fr.window.LayerInput(blinds_path, slat_angle=45)
        single_glaze_blinds = [glass_layer, blinds_layer]
        # double_glaze_blinds = [glass_layer, blinds_layer]
        self.glazing_blinds_system = fr.create_glazing_system(name="gs1", layer_inputs=single_glaze_blinds, nproc=4, nsamp=1)

    def test_add_proxy_geometry(self):
        rmodels = fr.epmodel_to_radmodel(self.medium_office)
        rconfigs = {
            k: fr.WorkflowConfig.from_dict(v) for k, v in rmodels.items()
        }
        for name, config in rconfigs.items():
            for window_name, window in config.model.windows.items():
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


if __name__ == "__main__":
    unittest.main()
