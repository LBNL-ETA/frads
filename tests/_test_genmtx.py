from pathlib import Path
import subprocess as sp
import os
import glob
import shutil


def test_help():
    cmd = ["gen", "-h"]
    sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)

def test_matrix_point_sky():
    grid_path = Path("test", "Resources", "grid.pts")
    sys_paths = [
        Path("test", "Objects/materials.mat"),
        Path("test", "Objects/walls.rad"),
        Path("test", "Objects/ceiling.rad"),
        Path("test", "Objects/floor.rad"),
    ]
    cmd = ["gen", "matrix", "point-sky", grid_path, *sys_paths]
    sp.run(cmd, check=True)
    assert os.path.isfile("grid_r4sky.mtx")
    os.remove("grid_r4sky.mtx")

