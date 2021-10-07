import unittest
import subprocess as sp
import os
import glob

class TestEpjson2Rad(unittest.TestCase):

    def test_help(self):
        cmd = ["ep2rad", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_cfscase(self):
        cmd = ["ep2rad", "./CmplxGlz_SingleZone_DoubleClearAir.idf"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertTrue(os.path.isfile("./CmplxGlz_SingleZone_DoubleClearAir.epJSON"))
        self.assertTrue(os.path.isfile("Room 102.cfg"))
        self.assertTrue(os.path.isfile("./Objects/Room102_South_Wall.rad"))
        self.assertTrue(os.path.isfile("./Objects/Room102_South_Wall_window.rad"))
        self.assertTrue(os.path.isfile("./Objects/Room102_Ceiling.rad"))
        os.remove("./CmplxGlz_SingleZone_DoubleClearAir.epJSON")
        os.remove("Room 102.cfg")
        obj_files = glob.glob("./Objects/Room_102*.rad")
        for file in obj_files:
            os.remove(file)

    def test_refcase(self):
        cmd = ["ep2rad", "./RefBldgMediumOfficeNew2004_Chicago.idf"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertTrue(os.path.isfile("./RefBldgMediumOfficeNew2004_Chicago.epJSON"))
        self.assertTrue(os.path.isfile("./Objects/Perimeter_bot_ZN_1_Wall_South.rad"))
        self.assertTrue(os.path.isfile("./Objects/Perimeter_mid_ZN_3_Ceiling.rad"))
        self.assertTrue(os.path.isfile("Perimeter_bot_ZN_1.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_bot_ZN_2.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_bot_ZN_3.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_bot_ZN_4.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_mid_ZN_1.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_mid_ZN_2.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_mid_ZN_3.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_mid_ZN_4.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_top_ZN_1.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_top_ZN_2.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_top_ZN_3.cfg"))
        self.assertTrue(os.path.isfile("Perimeter_top_ZN_4.cfg"))
        os.remove("./RefBldgMediumOfficeNew2004_Chicago.epJSON")
        cfg_files = glob.glob("./Perimeter*.cfg")
        obj_files = glob.glob("./Objects/Perimeter*.rad")
        for file in obj_files:
            os.remove(file)
        for file in cfg_files:
            os.remove(file)


if __name__ == "__main__":
    unittest.main()
