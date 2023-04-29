import os
from pathlib import Path
import shutil
import sys
sys.path.append(".")

from frads import methods, cli

test_dir_path = Path(__file__).parent.resolve()
reinsrc6_path = Path("Resources", "reinsrc6.rad")
grid_path = Path("Resources", "grid.pts")
prim_path = Path("Resources/model/Objects/floor_openroom.rad")

def test_assemble_model():
    os.chdir("test")
    config_path = Path("five_phase.cfg")
    config = cli.parse_mrad_config(config_path)
    with methods.assemble_model(config) as model:
        assert os.path.isfile(model.material_path)
        assert len(model.window_groups) == 2
        assert len(model.bsdf_xml) == 2
        assert model.sender_grid['floor'] is not None
    # os.removedirs("Matrices")
    os.chdir("..")

def test_parse_mrad_config():
    window_paths = [
        Path("test", "Objects", "lower_glass.rad"),
        Path("test", "Objects", "upper_glass.rad"),
    ]
    vdict = {"vf": "test/v1a.vf", "x": 4, "y": 4}
    vmx_opt = {"ab": 2, "ad": 64, "lw": 1e-4}
    epw_path = Path("Resources", "USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
    cfg_path = Path("test", "Resources", "test.cfg")
    config = cli.parse_mrad_config(cfg_path)
    view = config["RaySender"].getview("view")
    assert config["Model"].getpaths("windows") == window_paths
    assert config["Model"].getpaths("ncps") == []
    assert view.vtype == "a"
    assert view.position == [17.0, 5.0, 4.0]
    assert view.horiz == 180
    assert config["SimControl"].getoptions("vmx_opt") == vmx_opt
    assert config["Site"].getpaths("epw_path") == [epw_path]
    assert config["Site"].getpaths("wea_path") is None
