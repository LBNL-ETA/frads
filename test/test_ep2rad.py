"""
"""
# from frads import epjson_to_rad
from frads import epmodel_to_radmodel
from frads.eplus import load_energyplus_model


def test_cfscase(resources_dir):
    epjson1 = resources_dir / "CmplxGlz_SingleZone_DoubleClearAir.epJSON"
    epmodel = load_energyplus_model(epjson1)
    rad_model = epmodel_to_radmodel(epmodel)
    assert (
        rad_model["Room 102"]["model"]["sensors"]["Room102 Floor"] is not None
    )


def test_refcase(resources_dir):
    epjson2 = resources_dir / "RefBldgMediumOfficeNew2004_Chicago_epJSON.epJSON"
    epmodel = load_energyplus_model(epjson2)
    rad_model = epmodel_to_radmodel(epmodel)
    assert (
        rad_model["Perimeter_bot_ZN_1"]["model"]["sensors"][
            "Perimeter_bot_ZN_1_Floor"
        ]
        is not None
    )
