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
        # Create a new medium_office for each test - adjust based on how it's created in your tests

        self.medium_office = fr.load_energyplus_model(ref_models["medium_office"])
        glass_path = self.resources_dir / "CLEAR_3.DAT"
        blinds_path = self.resources_dir / "igsdb_product_19732.json"
        glass_layer = fr.LayerInput(glass_path)
        blinds_layer = fr.LayerInput(input_source=blinds_path, slat_angle_deg=45)
        single_glaze_blinds = [glass_layer, blinds_layer]
        single_glaze = [glass_layer]
        print("single blinds")
        # self.glazing_blinds_system = fr.create_glazing_system(
        #         name="gs1", layer_inputs=single_glaze_blinds, mbsdf=True, nproc=32, nsamp=2000)
        self.glazing_blinds_system = fr.window.load_glazing_system(self.resources_dir / "gs1.json")
        print("single ")
        # self.glazing_system = fr.create_glazing_system(name="gs2", layer_inputs=single_glaze, mbsdf=True, nproc=8, nsamp=2000)
        self.glazing_system = fr.window.load_glazing_system(self.resources_dir / "gs2.json")
        print("done")
        # self.glazing_blinds_system.save(self.resources_dir / "gs1.json")
        # self.glazing_system.save(self.resources_dir / "gs2.json")
        self.medium_office.add_glazing_system(self.glazing_blinds_system)
        self.medium_office.add_glazing_system(self.glazing_system)
        epsetup = fr.EnergyPlusSetup(self.medium_office, enable_radiance=True, initialize_radiance=False)
        epsetup.initialize_radiance(zones=["Perimeter_bot_ZN_1"], nproc=8)
        wpi_list = []
        def controller(state):
            if not epsetup.api.exchange.api_data_fully_ready(state):
                return
            # get the current time
            datetime = epsetup.get_datetime()
            # only calculate workplane illuminance during daylight hours
            if  datetime.hour >= 8 and datetime.hour < 18:
                wpi = epsetup.calculate_wpi(
                    zone="Perimeter_bot_ZN_1",
                    cfs_name={
                        "Perimeter_bot_ZN_1_Wall_South_Window": "gs2",
                    }, # {window: glazing system}
                ) # an array of illuminance for all sensors in the zone
                skycover = epsetup.get_total_sky_cover()
                edgps, ev = epsetup.calculate_edgps(zone="Perimeter_bot_ZN_1", cfs_name={"Perimeter_bot_ZN_1_Wall_South_Window": "gs2"})
                mev = epsetup.calculate_mev(zone="Perimeter_bot_ZN_1", cfs_name={
                    "Perimeter_bot_ZN_1_Wall_South_Window": "gs2",
                })
                print(wpi, ev, mev, skycover)
                wpi_list.append(wpi)
                wpi_list.append(mev)
        epsetup.add_melanopic_bsdf(self.glazing_blinds_system)
        epsetup.add_melanopic_bsdf(self.glazing_system)
        epsetup.set_callback("callback_begin_system_timestep_before_predictor",
                             controller)
        epsetup.run(design_day=True)


if __name__ == "__main__":
    unittest.main()
