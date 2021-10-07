import unittest
import subprocess as sp
import os
import glob
import shutil
from frads import makesky, radutil, util


class TestGenmtx(unittest.TestCase):

    def test_help(self):
        cmd = ["genmtx", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_sun_mtx(self):
        window_path = os.path.join("Objects", "upper_glass.rad")
        window_normals = radutil.primitive_normal([window_path])
        _, url = util.get_epw_url(37, 122)
        epw = util.request(url, {})
        wea_metadata, wea_data = makesky.epw2wea(
            epw, dhour=True, shour=6, ehour=20,
            remove_zero=True, window_normal=window_normals)
        smx = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        with open('test.smx', 'wb') as wtr:
            wtr.write(smx)
        cmd = ["genmtx", "-st", "v", "-s", "v1a.vf", "-r", "sun", "-rs", "r6",
               "-o", "test_genmtx_sun_mtx", "-wpths", window_path, "-opt", "-ab 0",
               "-res", "8", "8", "-smx", 'test.smx']
        process = sp.run(cmd, stderr=sp.PIPE)
        print(process.stderr)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isdir("test_genmtx_sun_mtx"))
        self.assertEqual(len(glob.glob('test_genmtx_sun_mtx/*.hdr')), 555)
        shutil.rmtree("test_genmtx_sun_mtx")
        os.remove('test.smx')

    def test_sun_mtx2(self):
        _, url = util.get_epw_url(32.6056027, 114.712143)
        epw = util.request(url, {})
        wea_metadata, wea_data = makesky.epw2wea(
            epw, dhour=True, shour=6, ehour=20,
            remove_zero=True)
        smx = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        with open('test.smx', 'wb') as wtr:
            wtr.write(smx)
        cmd = ["genmtx", "-st", "v", "-s", "v1a.vf", "-r", "sun", "-rs", "r6",
               "-o", "test_genmtx_sun_mtx2", "-opt", "-ab 0",
               "-res", "8", "8", "-smx", 'test.smx']
        process = sp.run(cmd, stderr=sp.PIPE)
        print(process.stderr)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isdir("test_genmtx_sun_mtx2"))
        self.assertEqual(len(glob.glob('test_genmtx_sun_mtx2/*.hdr')), 657)
        shutil.rmtree("test_genmtx_sun_mtx2")
        os.remove('test.smx')


if __name__ == "__main__":
    unittest.main()
