from datetime import datetime

from frads.methods import TwoPhaseMethod, ThreePhaseMethod, WorkflowConfig
from frads.window import create_glazing_system, Gap, Gas
from frads.ep2rad import epmodel_to_radmodel
from frads.eplus import load_energyplus_model
import pytest
from pyenergyplus.dataset import ref_models, weather_files
import pyradiance as pr


@pytest.fixture
def cfg(resources_dir, objects_dir):
    return {
        "settings": {
            "method": "2phase",
            "sky_basis": "r1",
            "epw_file": "",
            "wea_file": resources_dir / "oak.wea",
            "sensor_sky_matrix": ["-ab", "0"],
            "view_sky_matrix": ["-ab", "0"],
            "sensor_window_matrix": ["-ab", "0"],
            "view_window_matrix": ["-ab", "0"],
            "daylight_matrix": ["-ab", "0"],
        },
        "model": {
            "scene": {
                "files": [
                    objects_dir / "walls.rad",
                    objects_dir / "ceiling.rad",
                    objects_dir / "floor.rad",
                    objects_dir / "ground.rad",
                ]
            },
            "windows": {
                "upper_glass": {
                    "file": objects_dir / "upper_glass.rad",
                    "matrix_file": "blinds30",
                },
                "lower_glass": {
                    "file": objects_dir / "lower_glass.rad",
                    "matrix_file": "blinds30",
                },
            },
            "materials": {
                "files": [objects_dir / "materials.mat"],
                "matrices": {"blinds30": {"matrix_file": resources_dir / "blinds30.xml"}},
            },
            "sensors": {
                "wpi": {"file": resources_dir / "grid.txt"},
                "view1": {
                    "data": [[17, 5, 4, 1, 0, 0]],
                },
            },
            "views": {
                "view1": {"file": resources_dir / "v1a.vf", "xres": 16, "yres": 16}
            },
        },
    }


def test_two_phase(cfg):
    time = datetime(2023, 1, 1, 12)
    dni = 800
    dhi = 100
    config = WorkflowConfig.from_dict(cfg)
    with TwoPhaseMethod(config) as workflow:
        workflow.generate_matrices()
        res = workflow.calculate_sensor("wpi", time, dni, dhi)
    assert res.shape == (195, 1)


def test_three_phase(cfg, resources_dir):
    time = datetime(2023, 1, 1, 12)
    dni = 800
    dhi = 100
    config = WorkflowConfig.from_dict(cfg)
    blind_prim = pr.Primitive(
        "void", "aBSDF", "blinds30", [str(resources_dir/"blinds30.xml"), "0", "0", "1", "."], [])
    config.model.materials.glazing_materials = {"blinds30": blind_prim}
    with ThreePhaseMethod(config) as workflow:
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


def test_eprad_threephase(resources_dir):
    """
    Integration test for ThreePhaseMethod using EnergyPlusModel and GlazingSystem
    """
    view_path = resources_dir / "view1.vf"
    clear_glass_path = resources_dir / "CLEAR_3.DAT"
    product_7406_path = resources_dir / "igsdb_product_7406.json"
    shade_bsdf_path = resources_dir / "ec60.xml"

    epmodel = load_energyplus_model(ref_models["medium_office"])
    gs_ec60 = create_glazing_system(
        name="ec60",
        layers=[product_7406_path, clear_glass_path],
        gaps=[Gap([Gas("air", 0.1), Gas("argon", 0.9)], 0.0127)],
    )
    epmodel.add_glazing_system(gs_ec60)
    rad_models = epmodel_to_radmodel(epmodel, epw_file=weather_files["usa_ca_san_francisco"])
    zone = "Perimeter_bot_ZN_1"
    zone_dict = rad_models[zone]
    zone_dict["model"]["views"]["view1"] = {"file": view_path, "xres": 16, "yres": 16}
    zone_dict["model"]["sensors"]["view1"] = {"data": [[17, 5, 4, 1, 0, 0]]}
    zone_dict["model"]["materials"]["matrices"] = {
        "ec60": {"matrix_file": shade_bsdf_path}
    }
    rad_cfg = WorkflowConfig.from_dict(zone_dict)
    rad_cfg.settings.sensor_window_matrix = ["-ab", "0"]
    rad_cfg.settings.view_window_matrix = ["-ab", "0"]
    rad_cfg.settings.daylight_matrix = ["-ab", "0"]
    with ThreePhaseMethod(rad_cfg) as rad_workflow:
        rad_workflow.generate_matrices(view_matrices=False)
        dni = 800
        dhi = 100
        dt = datetime(2023, 1, 1, 12)
        edgps = rad_workflow.calculate_edgps(
            view="view1",
            bsdf={f"{zone}_Wall_South_Window": "ec60"},
            date_time=dt,
            dni=dni,
            dhi=dhi,
            ambient_bounce=1,
        )

    assert "view1" in rad_workflow.view_senders
    assert rad_workflow.view_senders["view1"].view.vtype == "a"
    assert rad_workflow.view_senders["view1"].view.position == [6.0, 7.0, 0.76]
    assert rad_workflow.view_senders["view1"].view.direction == [0.0, -1.0, 0.0]
    assert rad_workflow.view_senders["view1"].view.horiz == 180
    assert rad_workflow.view_senders["view1"].view.vert == 180
    assert rad_workflow.view_senders["view1"].xres == 16

    assert list(rad_workflow.daylight_matrices.values())[0].array.shape == (145, 146, 3)
    assert (
        list(rad_workflow.sensor_window_matrices.values())[0].ncols == [145]
        and list(rad_workflow.sensor_window_matrices.values())[0].ncomp == 3
    )
    assert edgps >= 0 and edgps <= 1
