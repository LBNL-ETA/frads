import os
from pathlib import Path
import shutil
import sys
sys.path.append(".")

from frads import methods
from frads import parsers

test_dir_path = Path(__file__).parent.resolve()
reinsrc6_path = Path("Resources", "reinsrc6.rad")
grid_path = Path("Resources", "grid.pts")
prim_path = Path("Resources/model/Objects/floor_openroom.rad")

def test_assemble_model():
    os.chdir("test")
    config_path = Path("five_phase.cfg")
    config = parsers.parse_mrad_config(config_path)
    with methods.assemble_model(config) as model:
        assert os.path.isfile(model.material_path)
        assert len(model.window_groups) == 2
        assert len(model.bsdf_xml) == 2
        assert model.sender_grid['floor'] is not None
    os.removedirs("Matrices")
    os.chdir("..")
