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
        cmd, _ = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        process = sp.run(cmd, stdout=sp.PIPE)
        with open('test.smx', 'wb') as wtr:
            wtr.write(process.stdout)
        cmd = ["genmtx", "-st", "v", "-s", "./v1a.vf", "-r", "sun", "-rs", "r6",
               "-o", "test_genmtx_sun_mtx", "-wpths", "./Objects/upper_glass.rad", "-opt", "-ab 0",
               "-res", "8", "8", "-smx", 'test.smx']
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isdir("test_genmtx_sun_mtx"))
        self.assertEqual(len(glob.glob('test_genmtx_sun_mtx/*.hdr')), 555)
        shutil.rmtree("test_genmtx_sun_mtx")
        os.remove('test.smx')

    def test_sun_mtx2(self):
        epw = makesky.getEPW("33.6","-112.4")
        wea_metadata, wea_data = makesky.epw2wea(
            epw.fname, dhour=True, shour=6, ehour=20,
            remove_zero=True)
        cmd, _ = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        process = sp.run(cmd, stdout=sp.PIPE)
        with open('test.smx', 'wb') as wtr:
            wtr.write(process.stdout)
        # cmd = ['rmtxop', '-ff', '-c', '.3', '.6', '.1', 'test.smx']
        # cmd2 = ['getinfo', '-']
        # cmd3 = ['total', '-if5186', '-t,']
        # ps1 = sp.Popen(cmd, stdout=sp.PIPE)
        # ps2 = sp.Popen(cmd2, stdin=ps1.stdout, stdout=sp.PIPE)
        # ps1.stdout.close()
        # ps3 = sp.Popen(cmd3, stdin=ps2.stdout, stdout=sp.PIPE)
        # ps2.stdout.close()
        # _out, _ = ps3.communicate()
        # ps3.stdout.close()
        # res = [float(i) for i in _out.split(b',') if float(i) > 0]
        # print(len(res))
        cmd = ["genmtx", '-vv', "-st", "v", "-s", "./v1a.vf", "-r", "sun", "-rs", "r6",
               "-env", "Objects/materials.mat", "./Objects/walls.rad", "./Objects/lower_glass.rad",
               "./Objects/upper_glass.rad", "./Objects/floor.rad", "./Objects/ceiling.rad",
               "-o", "test_genmtx_sun_mtx2", "-opt", "-ab 0",
               "-res", "8", "8", "-smx", 'test.smx']
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        # self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isdir("test_genmtx_sun_mtx2"))
        self.assertEqual(len(glob.glob('test_genmtx_sun_mtx2/*.hdr')), 647)
        # shutil.rmtree("test_genmtx_sun_mtx2")
        os.remove('test.smx')



if __name__ == "__main__":
    unittest.main()
