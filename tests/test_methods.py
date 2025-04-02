from datetime import datetime
from pathlib import Path
import unittest

import pyradiance as pr
from pyenergyplus.dataset import ref_models, weather_files
import frads as fr


class TestMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resources_dir = Path(__file__).parent / "Resources"
        cls.objects_dir = Path(__file__).parent / "Objects"

    def setUp(self):
        self.cfg = {
            "settings": {
                "method": "2phase",
                "sky_basis": "r1",
                "epw_file": "",
                "wea_file": self.resources_dir / "oak.wea",
                "sensor_sky_matrix": ["-ab", "0"],
                "view_sky_matrix": ["-ab", "0"],
                "sensor_window_matrix": ["-ab", "0"],
                "view_window_matrix": ["-ab", "0"],
                "daylight_matrix": ["-ab", "0"],
            },
            "model": {
                "scene": {
                    "files": [
                        self.objects_dir / "walls.rad",
                        self.objects_dir / "ceiling.rad",
                        self.objects_dir / "floor.rad",
                        self.objects_dir / "ground.rad",
                    ]
                },
                "windows": {
                    "upper_glass": {
                        "file": self.objects_dir / "upper_glass.rad",
                        "matrix_name": "blinds30",
                    },
                    "lower_glass": {
                        "file": self.objects_dir / "lower_glass.rad",
                        "matrix_name": "blinds30",
                    },
                },
                "materials": {
                    "files": [self.objects_dir / "materials.mat"],
                    "matrices": {
                        "blinds30": {"matrix_file": self.resources_dir / "blinds30.xml"}
                    },
                },
                "sensors": {
                    "wpi": {"file": self.resources_dir / "grid.txt"},
                    "view1": {
                        "data": [[17, 5, 4, 1, 0, 0]],
                    },
                },
                "views": {
                    "view1": {
                        "file": self.resources_dir / "v1a.vf",
                        "xres": 16,
                        "yres": 16,
                    }
                },
                "surfaces": {},
            },
        }

        self.scene = fr.SceneConfig(
                files=[
                    self.objects_dir / "walls.rad",
                    self.objects_dir / "ceiling.rad",
                    self.objects_dir / "floor.rad",
                    self.objects_dir / "ground.rad",
                ]
            )


        self.window_1 = fr.WindowConfig(
                file=self.objects_dir / "upper_glass.rad",
                matrix_name="blinds30",
            )


        self.window_2 = fr.WindowConfig(
                file=self.objects_dir / "upper_glass.rad",
                # matrix_name="blinds30",
            )


        self.materials = fr.MaterialConfig(
                files=[self.objects_dir / "materials.mat"],
                matrices={"blinds30": {"matrix_file": self.resources_dir / "blinds30.xml"}},
            )


        self.wpi = fr.SensorConfig(file=self.resources_dir / "grid.txt")


        self.sensor_view_1 = fr.SensorConfig( data=[[17, 5, 4, 1, 0, 0]],)


        self.view_1 = fr.ViewConfig( file=self.resources_dir / "v1a.vf", xres=16, yres=16,)


    def test_model1(self):
        model = fr.Model(
            scene=self.scene,
            windows={"window_1": self.window_1},
            materials=self.materials,
            sensors={"wpi": self.wpi, "view1": self.sensor_view_1},
            views={"view_1": self.view_1},
        )
        self.assertEqual(model.scene.files, self.scene.files)
        self.assertEqual(model.windows["window_1"].file, self.window_1.file)
        self.assertEqual(model.windows["window_1"].matrix_name, self.window_1.matrix_name)
        self.assertEqual(model.materials.files, self.materials.files)
        self.assertIn(model.windows["window_1"].matrix_name, model.materials.matrices)
        self.assertEqual(model.sensors["wpi"].file, self.wpi.file)
        self.assertEqual(model.sensors["view_1"].data[0][0], self.sensor_view_1.data[0][0])
        self.assertEqual(model.views["view_1"].file, self.view_1.file)
        self.assertEqual(model.views["view_1"].xres, self.view_1.xres)
        self.assertEqual(model.views["view_1"].yres, self.view_1.yres)


    def test_model2(self):
        # auto-generate view_1 in sensors from view_1 in views
        model = fr.Model(
            materials=self.materials,
            sensors={"wpi": self.wpi},
            views={"view_1": self.view_1},
        )
        self.assertIn("view_1", model.sensors)
        self.assertEqual(model.sensors["view_1"].data[0][0], model.views["view_1"].view.vp[0])
        self.assertTrue(isinstance(model.scene, fr.SceneConfig))
        self.assertTrue(isinstance(model.windows, dict))
        self.assertEqual(model.scene.files, [])
        self.assertEqual(model.scene.bytes, b"")
        self.assertEqual(model.windows, {})


    def test_model3(self):
        # same name view and sensor but different position and direction
        sensor_view_2 = fr.SensorConfig(
            data=[[1, 5, 4, 1, 0, 0]],
        )

        with self.assertRaises(ValueError):
            fr.Model(
                scene=self.scene,
                windows={"window_1": self.window_1},
                materials=self.materials,
                sensors={"wpi": self.wpi, "view_1": sensor_view_2},
                views={"view_1": self.view_1},
            )


    def test_model4(self):
        # window matrix name not in materials
        materials = fr.MaterialConfig(files=[self.objects_dir / "materials.mat"])
        with self.assertRaises(ValueError):
            fr.Model(
                scene=self.scene,
                windows={"window_1": self.window_1},
                materials=materials,
                sensors={"wpi": self.wpi, "view_1": self.sensor_view_1},
                views={"view_1": self.view_1},
            )


    def test_no_sensors_views_surfaces_specified(self):
        settings = fr.Settings()
        model = fr.Model(
            scene=self.scene,
            windows={"window_1": self.window_1},
            materials=self.materials,
        )
        with self.assertRaises(ValueError):
            fr.WorkflowConfig(settings, model)


    def test_windows_not_specified_for_3phase_or_5phase_method(self):
        settings = fr.Settings()
        model = fr.Model(
            scene=self.scene,
            # windows={"window_1": window_1},
            materials=self.materials,
            sensors={"wpi": self.wpi},
            views={"view_1": self.view_1},
        )
        with self.assertRaises(ValueError):
            fr.WorkflowConfig(settings, model)


    def test_three_phase2(self):
        model = fr.Model(
            scene=self.scene,
            windows={"window_1": self.window_2},  # window_2 has no matrix_name
            materials=self.materials,
            sensors={"wpi": self.wpi, "view_1": self.sensor_view_1},
            views={"view_1": self.view_1},
        )
        settings = fr.Settings()
        settings.sensor_window_matrix = ['-ab', '1']
        settings.daylight_matrix = ['-ab', '1']

        cfg = fr.WorkflowConfig(settings, model)
        with fr.ThreePhaseMethod(cfg) as workflow:
            workflow.generate_matrices(view_matrices=False)
            a = workflow.calculate_sensor(
                "view_1",
                {"window_1": "blinds30"},  # blinds30 is the matrix_name
                datetime(2023, 1, 1, 12),
                800,
                100,
            )
        self.assertEqual(a.shape , (1, 1))


    def test_two_phase(self):
        time = datetime(2023, 1, 1, 12)
        dni = 800
        dhi = 100
        config = fr.WorkflowConfig.from_dict(self.cfg)
        with fr.TwoPhaseMethod(config) as workflow:
            workflow.generate_matrices()
            res = workflow.calculate_sensor("wpi", time, dni, dhi)
        self.assertEqual(res.shape , (195, 1))


    def test_three_phase(self):
        time = datetime(2023, 1, 1, 12)
        dni = 800
        dhi = 100
        config = fr.WorkflowConfig.from_dict(self.cfg)
        blind_prim = pr.Primitive(
            "void",
            "aBSDF",
            "blinds30",
            [str(self.resources_dir / "blinds30.xml"), "0", "0", "1", "."],
            [],
        )
        config.model.materials.glazing_materials = {"blinds30": blind_prim}
        with fr.ThreePhaseMethod(config) as workflow:
            workflow.generate_matrices(view_matrices=False)
            workflow.calculate_sensor(
                "wpi",
                {"upper_glass": "blinds30", "lower_glass": "blinds30"},
                time,
                dni,
                dhi,
            )
            res = workflow.calculate_edgps(
                "view1",
                {"upper_glass": "blinds30", "lower_glass": "blinds30"},
                time,
                dni,
                dhi,
            )
            res = workflow.calculate_sensor_from_wea("wpi")


    def test_eprad_threephase(self):
        """
        Integration test for ThreePhaseMethod using EnergyPlusModel and GlazingSystem
        """
        view_path = self.resources_dir / "view1.vf"
        clear_glass_path = self.resources_dir / "CLEAR_3.DAT"
        product_7406_path = self.resources_dir / "igsdb_product_7406.json"
        shade_bsdf_path = self.resources_dir / "ec60.xml"
        layers = [fr.window.LayerInput(product_7406_path), fr.window.LayerInput(clear_glass_path)]
        epmodel = fr.load_energyplus_model(ref_models["medium_office"])
        gs_ec60 = fr.create_glazing_system(
            name="ec60",
            layer_inputs=layers,
            gaps=[fr.Gap([fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127)],
        )
        epmodel.add_glazing_system(gs_ec60)
        rad_models = fr.epmodel_to_radmodel(
            epmodel, epw_file=weather_files["usa_ca_san_francisco"]
        )
        zone = "Perimeter_bot_ZN_1"
        zone_dict = rad_models[zone]
        zone_dict["model"]["views"]["view1"] = {
            "file": view_path,
            "xres": 16,
            "yres": 16,
        }
        zone_dict["model"]["sensors"]["view1"] = {
            "data": [[6.0, 7.0, 0.76, 0.0, -1.0, 0.0]]
        }
        zone_dict["model"]["materials"]["matrices"] = {
            "ec60": {"matrix_file": shade_bsdf_path}
        }
        zone_dict["model"]["surfaces"] = {}
        rad_cfg = fr.WorkflowConfig.from_dict(zone_dict)
        rad_cfg.settings.sensor_window_matrix = ["-ab", "0"]
        rad_cfg.settings.view_window_matrix = ["-ab", "0"]
        rad_cfg.settings.daylight_matrix = ["-ab", "0"]
        with fr.ThreePhaseMethod(rad_cfg) as rad_workflow:
            rad_workflow.generate_matrices(view_matrices=False)
            dni = 800
            dhi = 100
            dt = datetime(2023, 1, 1, 12)
            edgps, ev = rad_workflow.calculate_edgps(
                view="view1",
                bsdf={f"{zone}_Wall_South_Window": "ec60"},
                time=dt,
                dni=dni,
                dhi=dhi,
                ambient_bounce=1,
            )

        self.assertTrue("view1" in rad_workflow.view_senders)
        self.assertEqual(rad_workflow.view_senders["view1"].view.type, "a")
        self.assertEqual(rad_workflow.view_senders["view1"].view.vp, (6.0, 7.0, 0.76))
        self.assertEqual(rad_workflow.view_senders["view1"].view.vdir, (0.0, -1.0, 0.0))
        self.assertEqual(rad_workflow.view_senders["view1"].view.horiz, 180)
        self.assertEqual(rad_workflow.view_senders["view1"].view.vert, 180)
        self.assertEqual(rad_workflow.view_senders["view1"].xres, 16)

        self.assertEqual(list(rad_workflow.daylight_matrices.values())[0].array.shape, (
            145,
            146,
            3,
        ))
        self.assertTrue(
            list(rad_workflow.sensor_window_matrices.values())[0].ncols == [145]
            and list(rad_workflow.sensor_window_matrices.values())[0].ncomp == 3
        )
        self.assertTrue( edgps >= 0 and edgps <= 1)

if __name__=="__main__":
    unittest.main()
