import unittest
import subprocess as sp
import os
import glob
import shutil


class TestMrad(unittest.TestCase):
    def test_0_help(self):
        cmd = ["mrad", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")

    def test_1_init(self):
        cmd = ["mrad", "init", "-n", "test", "-W", "Resources/test.wea"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertTrue(os.path.isfile("test.cfg"))
        os.remove("test.cfg")

    def test_1_init2(self):
        cmd = [
            "mrad",
            "init",
            "--epw_path",
            "Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.epw",
        ]
        process = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertTrue(os.path.isfile("default.cfg"))
        os.remove("default.cfg")

    def test_2_two_phase(self):
        cmd = ["mrad", "-vvvv", "run", "two_phase.cfg"]
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(proc.stderr, b"")
        self.assertTrue(os.path.isfile("./Matrices/pdsmx_floor.mtx"))
        dsmx_size = os.stat("./Matrices/pdsmx_floor.mtx").st_size
        self.assertGreaterEqual(dsmx_size, 2.5e6)
        self.assertTrue(os.path.isdir("./Matrices/vdsmx_view_00"))
        vsmx_size = len(os.listdir("./Matrices/vdsmx_view_00"))
        self.assertEqual(vsmx_size, 146)
        view_results = glob.glob("Results/view_two_phase_view_00/*.hdr")
        self.assertEqual(len(view_results), 2)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")

    def test_3_three_phase(self):
        cmd = ["mrad", "-vvvv", "run", "three_phase.cfg"]
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        self.assertEqual(proc.stderr, b"")
        view_results = glob.glob("Results/view_three_phase_view_00/*.hdr")
        self.assertEqual(len(view_results), 1)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")

    def test_4_five_phase(self):
        cmd = ["mrad", "-vvvv", "run", "five_phase.cfg"]
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        view_results = glob.glob("Results/view_five_phase_view_00/*.hdr")
        self.assertEqual(len(view_results), 4387)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")

    def test_5_five_phase2(self):
        cmd = ["mrad", "-vvvv", "run", "five_phase2.cfg"]
        proc = sp.run(cmd, check=True, stderr=sp.PIPE)
        view_results = glob.glob("Results/view_five_phase2_view_00/*.hdr")
        self.assertEqual(len(view_results), 4387)
        shutil.rmtree("Matrices")
        shutil.rmtree("Results")


if __name__ == "__main__":
    unittest.main()
