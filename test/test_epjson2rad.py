import json
from pathlib import Path
import os
import glob
import sys
sys.path.append(".")
from frads import epjson2rad



test_dir = Path(__file__).resolve().parent


def test_cfscase():
    os.chdir("test")
    epjson1 = test_dir / "Resources" / "CmplxGlz_SingleZone_DoubleClearAir.epJSON"
    with open(epjson1) as fp:
        json_data = json.load(fp)
    epjson2rad(json_data, epw="")
    assert os.path.isfile("Room 102.cfg")
    assert os.path.isfile("./Objects/Room102_South_Wall.rad")
    assert os.path.isfile("./Objects/Room102_South_Wall_window.rad")
    assert os.path.isfile("./Objects/Room102_Ceiling.rad")
    os.remove("Room 102.cfg")
    obj_files = glob.glob("./Objects/Room_102*.rad")
    for file in obj_files:
        os.remove(file)
    os.chdir("..")

def test_refcase():
    os.chdir("test")
    epjson2 = test_dir / "Resources" / "RefBldgMediumOfficeNew2004_Chicago.epJSON"
    with open(epjson2) as fp:
        json_data = json.load(fp)
    epjson2rad(json_data, epw="")
    assert os.path.isfile("./Objects/Perimeter_bot_ZN_1_Wall_South.rad")
    assert os.path.isfile("./Objects/Perimeter_mid_ZN_3_Ceiling.rad")
    assert os.path.isfile("Perimeter_bot_ZN_1.cfg")
    assert os.path.isfile("Perimeter_bot_ZN_2.cfg")
    assert os.path.isfile("Perimeter_bot_ZN_3.cfg")
    assert os.path.isfile("Perimeter_bot_ZN_4.cfg")
    assert os.path.isfile("Perimeter_mid_ZN_1.cfg")
    assert os.path.isfile("Perimeter_mid_ZN_2.cfg")
    assert os.path.isfile("Perimeter_mid_ZN_3.cfg")
    assert os.path.isfile("Perimeter_mid_ZN_4.cfg")
    assert os.path.isfile("Perimeter_top_ZN_1.cfg")
    assert os.path.isfile("Perimeter_top_ZN_2.cfg")
    assert os.path.isfile("Perimeter_top_ZN_3.cfg")
    assert os.path.isfile("Perimeter_top_ZN_4.cfg")
    cfg_files = glob.glob("./Perimeter*.cfg")
    obj_files = glob.glob("./Objects/Perimeter*.rad")
    for file in obj_files:
        os.remove(file)
    for file in cfg_files:
        os.remove(file)
    os.chdir("..")

if __name__ == "__main__":
    test_cfscase()
    test_refcase()
