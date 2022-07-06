import unittest
import subprocess as sp
import os

class TestGenfmtx(unittest.TestCase):

    def test_help(self):
        cmd = ["genfmtx", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_mtx(self):
        cmd = ["genfmtx", "-w", "Objects/upper_glass.rad", "-ncp", "Objects/overhang.rad",
               "-env", "Objects/materials.mat", "-o", "test_genfmtx", "-ss", "kf", "-rs", "kf"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isfile("test_genfmtx_tb0.mtx"))
        self.assertTrue(os.path.isfile("test_genfmtx_tb1.mtx"))
        self.assertTrue(os.path.isfile("test_genfmtx_tb2.mtx"))
        self.assertTrue(os.path.isfile("test_genfmtx_tb3.mtx"))
        os.remove("test_genfmtx_tb0.mtx")
        os.remove("test_genfmtx_tb1.mtx")
        os.remove("test_genfmtx_tb2.mtx")
        os.remove("test_genfmtx_tb3.mtx")

    def test_xml(self):
        cmd = ["genfmtx", "-wrap", "-w", "Objects/upper_glass.rad", "-ncp", "Objects/overhang.rad",
               "-env", "Objects/materials.mat", "-o", "test_genfmtx", "-ss", "kf", "-rs", "kf"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isfile("test_genfmtx_0.xml"))
        self.assertTrue(os.path.isfile("test_genfmtx_1.xml"))
        self.assertTrue(os.path.isfile("test_genfmtx_2.xml"))
        self.assertTrue(os.path.isfile("test_genfmtx_3.xml"))
        os.remove("test_genfmtx_0.xml")
        os.remove("test_genfmtx_1.xml")
        os.remove("test_genfmtx_2.xml")
        os.remove("test_genfmtx_3.xml")

if __name__ == "__main__":
    unittest.main()
