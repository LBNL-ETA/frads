import json
from pathlib import Path
from frads import epjson_to_rad
from frads.eprad import EnergyPlusModel

test_dir = Path(__file__).resolve().parent


def test_cfscase():
    epjson1 = test_dir / "Resources" / "CmplxGlz_SingleZone_DoubleClearAir.epJSON"
    epmodel = EnergyPlusModel(epjson1)
    rad_model: dict = epjson_to_rad(epmodel, epw="")
    assert (
        rad_model["Room 102"]["model"]["sensors"]["Room 102_Room102_Floor"] is not None
    )


def test_refcase():
    epjson2 = test_dir / "Resources" / "RefBldgMediumOfficeNew2004_Chicago.epJSON"
    epmodel = EnergyPlusModel(epjson2)
    rad_model: dict = epjson_to_rad(epmodel, epw="")
    assert (
        rad_model["Perimeter_bot_ZN_1"]["model"]["sensors"][
            "Perimeter_bot_ZN_1_Perimeter_bot_ZN_1_Floor"
        ]
        is not None
    )


if __name__ == "__main__":
    test_cfscase()
    test_refcase()
