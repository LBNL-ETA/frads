import json
from pathlib import Path
from frads import epjson2rad
from frads.eprad import EPModel

test_dir = Path(__file__).resolve().parent


def test_cfscase():
    epjson1 = test_dir / "Resources" / "CmplxGlz_SingleZone_DoubleClearAir.epJSON"
    epmodel = EPModel(epjson1)
    rad_model: dict = epjson2rad(epmodel, epw="")
    assert rad_model['Room 102']['model']['sensors']['Room 102_Room102_Floor'] is not None

def test_refcase():
    epjson2 = test_dir / "Resources" / "RefBldgMediumOfficeNew2004_Chicago.epJSON"
    epmodel = EPModel(epjson2)
    rad_model: dict = epjson2rad(epmodel, epw="")
    assert rad_model['Perimeter_bot_ZN_1']['model']['sensors']['Perimeter_bot_ZN_1_Perimeter_bot_ZN_1_Floor'] is not None

if __name__ == "__main__":
    test_cfscase()
    test_refcase()
