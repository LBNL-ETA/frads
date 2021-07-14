import unittest
import subprocess as sp
import os
import glob
import shutil
from frads import makesky, radutil

class TestGenmtx(unittest.TestCase):

    def test_help(self):
        cmd = ["genmtx", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_sun_mtx(self):
        window_path = "./Objects/upper_glass.rad"
        window_normals = radutil.primitive_normal([window_path])
        epw = makesky.getEPW("37", "-122")
        wea_metadata, wea_data = makesky.epw2wea(
            epw.fname, dhour=True, shour=6, ehour=20,
            remove_zero=True, window_normal=window_normals)
        cmd, wea_path = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        process = sp.run(cmd, stdout=sp.PIPE)
        with open('test.smx', 'wb') as wtr:
            wtr.write(process.stdout)
        cmd = ["genmtx", "-st", "v", "-s", "./v1a.vf", "-r", "sun", "-rs", "r6",
               "-env", "Objects/materials.mat", "./Objects/walls.rad", "./Objects/lower_glass.rad",
               "./Objects/upper_glass.rad", "./Objects/floor.rad", "./Objects/ceiling.rad",
               "-o", "test_genmtx_sun_mtx", "-wpths", "./Objects/upper_glass.rad", "-opt", "-ab 0",
               "-res", "16", "16", "-smx", 'test.smx']
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isdir("test_genmtx_sun_mtx"))
        self.assertEqual(len(glob.glob('test_genmtx_sun_mtx/*.hdr')), 555)
        shutil.rmtree("test_genmtx_sun_mtx")
        os.remove(wea_path)
        os.remove('test.smx')



if __name__ == "__main__":
    unittest.main()
