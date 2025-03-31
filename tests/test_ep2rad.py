"""
"""
from pathlib import Path
from frads import epmodel_to_radmodel
from frads.eplus import load_energyplus_model
import unittest

class TestEp2Rad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup the resources directory - replace with your actual resources path
        cls.resources_dir = Path(__file__).parent / "Resources"

    def test_cfscase(self):
        epjson1 = self.resources_dir / "CmplxGlz_SingleZone_DoubleClearAir.epJSON"
        epmodel = load_energyplus_model(epjson1)
        rad_model = epmodel_to_radmodel(epmodel)
        assert (
            rad_model["Room 102"]["model"]["sensors"]["Room102 Floor"] is not None
        )

    def test_refcase(self):
        epjson2 = self.resources_dir / "RefBldgMediumOfficeNew2004_Chicago_epJSON.epJSON"
        epmodel = load_energyplus_model(epjson2)
        rad_model = epmodel_to_radmodel(epmodel)
        assert (
            rad_model["Perimeter_bot_ZN_1"]["model"]["sensors"][
                "Perimeter_bot_ZN_1_Floor"
            ]
            is not None
        )

if __name__ == "__main__":
    unittest.main()
