import os
from pathlib import Path
import sys
sys.path.append(".")

from frads import raycall
from frads import geom
from frads.types import View
from frads.types import Options


def test_get_rpict_cmd():
    vp1 = geom.Vector(0, 0, 0)
    vd1 = geom.Vector(0, -1, 0)
    vu1 = geom.Vector(0, 0, 1)
    view1 = View(vp1, vd1, vu1, "a", 180, 180)
    options1 = Options()
    options1.ab = 2
    options1.ad = 1024
    options1.av = (.38, .38, .34)
    options1.I = True
    cmd1 = raycall.get_rpict_command(view1, options1)
    cmd2 = raycall.get_rpict_command(view1, options1, "test.oct")
    answer = ["rpict", "-vta", "-vp", "0", "0", "0",
              "-vd", "0", "-1", "0", "-vu", "0", "0", "1",
              "-vv", "180", "-vh", "180", "-vo", "0", "-va", "0",
              "-vs", "0", "-vl", "0", "-x", "256", "-y", "256",
              "-ab", "2", "-ad", "1024",
              "-av", "0.38", "0.38", "0.34", "-I+"]
    answer2 = answer + ["test.oct"]
    assert cmd1 == answer
    assert cmd2 == answer2
