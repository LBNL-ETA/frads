import unittest
import subprocess as sp
import os
import glob
import shutil

class TestMrad(unittest.TestCase):

    def test_0_help(self):
        cmd = ["mrad", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_1_init(self):
        cmd = ["mrad", "init", "-W", "test.wea"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isfile("default.cfg"))
        os.remove("default.cfg")

    def test_1_init2(self):
        cmd = ["mrad", "init", "--latlon", "37.72", "-122.22"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(process.stderr, b'')
        self.assertTrue(os.path.isfile("default.cfg"))
        os.remove("default.cfg")

    def test_2_two_phase(self):
        cmd = ['mrad', '-vvvv', 'run', 'two_phase.cfg']
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(proc.stderr, b'')
        self.assertTrue(os.path.isfile("./Matrices/pdsmx_floor"))
        dsmx_size = os.stat("./Matrices/pdsmx_floor").st_size
        self.assertGreaterEqual(dsmx_size, 2.5e6)
        self.assertTrue(os.path.isdir("./Matrices/vdsmx_view_00"))
        vsmx_size = len(os.listdir("./Matrices/vdsmx_view_00"))
        self.assertEqual(vsmx_size, 146)
        view_results = glob.glob('Results/view_two_phase_view_00/*.hdr')
        self.assertEqual(len(view_results), 3)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")

    def test_3_three_phase(self):
        cmd = ['mrad', '-vvvv', 'run', 'three_phase.cfg']
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(proc.stderr, b'')
        view_results = glob.glob("Results/view_three_phase_view_00/*.hdr")
        self.assertEqual(len(view_results), 3)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")

    def test_4_five_phase(self):
        cmd = ['mrad', '-vvvv', 'run', 'five_phase.cfg']
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(proc.stderr, b'')
        view_results = glob.glob("Results/view_five_phase_view_00/*.hdr")
        self.assertEqual(len(view_results), 4385)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")


if __name__ == "__main__":
    unittest.main()
