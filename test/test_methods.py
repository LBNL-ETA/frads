from pathlib import Path
from datetime import datetime
from frads.methods import TwoPhaseMethod, ThreePhaseMethod, WorkflowConfig

test_dir = Path(__file__).parent.resolve()
reinsrc6_path = Path("Resources", "reinsrc6.rad")
grid_path = Path("Resources", "grid.pts")
prim_path = Path("Resources/model/Objects/floor_openroom.rad")

cfg = {
    "settings": {
        "method": "2phase",
        "sky_basis": "r1",
        "epw_file": "",
        "wea_file": f"{test_dir}/Resources/oak.wea",
        "sensor_sky_matrix": ["-ab", "0"],
        "view_sky_matrix": ["-ab", "0"],
        "sensor_window_matrix": ["-ab", "0"],
        "view_window_matrix": ["-ab", "0"],
        "daylight_matrix": ["-ab", "0"],
    },
    "model": {
        "scene": {
            "files": [
                f"{test_dir}/Objects/walls.rad",
                f"{test_dir}/Objects/ceiling.rad",
                f"{test_dir}/Objects/floor.rad",
                f"{test_dir}/Objects/ground.rad",
            ]
        },
        "windows": {
            "upper_glass": {
                "file": f"{test_dir}/Objects/upper_glass.rad",
                "matrix_file": f"{test_dir}/Resources/blinds30.xml",
            },
            "lower_glass": {
                "file": f"{test_dir}/Objects/lower_glass.rad",
                "matrix_file": f"{test_dir}/Resources/blinds30.xml",
            },
        },
        "materials": {
            "files": [f"{test_dir}/Objects/materials.mat"],
        },
        "sensors": {
            'wpi': {
                "file": f"{test_dir}/Resources/grid.txt"
            },
            'view1': {
                'data': [[17, 5, 4, 1, 0, 0]],
            }
        },
        "views": {
            "view1": {"file": f"{test_dir}/Resources/v1a.vf", "xres": 16, "yres": 16}
        },
    },
}

def test_two_phase():
    time = datetime(2023, 1, 1, 12)
    dni = 800
    dhi = 100
    config = WorkflowConfig.from_dict(cfg)
    workflow = TwoPhaseMethod(config)
    workflow.generate_matrices()
    res = workflow.calculate_sensor('wpi', time, dni, dhi)
    assert res.shape == (195, 1)

def test_three_phase():
    time = datetime(2023, 1, 1, 12)
    dni = 800
    dhi = 100
    config = WorkflowConfig.from_dict(cfg)
    lower_glass = f"{test_dir}/Objects/lower_glass.rad"  
    upper_glass = f"{test_dir}/Objects/upper_glass.rad"
    workflow = ThreePhaseMethod(config)
    workflow.generate_matrices(view_matrices=False)
    workflow.calculate_sensor(
        'wpi', 
        [workflow.window_bsdfs['upper_glass'], workflow.window_bsdfs['lower_glass']], 
        time, 
        dni, 
        dhi
    )
    res = workflow.calculate_edgps(
        'view1', 
        [lower_glass, upper_glass],
        [workflow.window_bsdfs['upper_glass'], workflow.window_bsdfs['lower_glass']], 
        time, 
        dni, 
        dhi,
    )
    res = workflow.calculate_sensor_from_wea('wpi')


# def test_assemble_model():
#     # os.chdir("test")
#     config_path = Path("five_phase.cfg")
#     config = cli.parse_mrad_config(config_path)
#     breakpoint()
#     with methods.assemble_model(config) as model:
#         assert os.path.isfile(model.material_path)
#         assert len(model.window_groups) == 2
#         assert len(model.bsdf_xml) == 2
#         assert model.sender_grid['floor'] is not None
#     # os.removedirs("Matrices")
#     # os.chdir("..")


# def test_parse_mrad_config():
#     window_paths = [
#         Path("test", "Objects", "lower_glass.rad"),
#         Path("test", "Objects", "upper_glass.rad"),
#     ]
#     vdict = {"vf": "test/v1a.vf", "x": 4, "y": 4}
#     vmx_opt = {"ab": 2, "ad": 64, "lw": 1e-4}
#     epw_path = Path("Resources", "USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
#     cfg_path = Path("test", "Resources", "test.cfg")
#     config = cli.parse_mrad_config(cfg_path)
#     view = config["RaySender"].getview("view")
#     assert config["Model"].getpaths("windows") == window_paths
#     assert config["Model"].getpaths("ncps") == []
#     assert view.vtype == "a"
#     assert view.position == [17.0, 5.0, 4.0]
#     assert view.horiz == 180
#     assert config["SimControl"].getoptions("vmx_opt") == vmx_opt
#     assert config["Site"].getpaths("epw_path") == [epw_path]
#     assert config["Site"].getpaths("wea_path") is None


if __name__ == "__main__":
    test_assemble_model()
