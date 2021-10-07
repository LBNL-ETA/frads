import unittest
import subprocess as sp
import os
import shutil
from frads import makesky, radutil

class TestGetwea(unittest.TestCase):

    def test_help(self):
        cmd = ["getwea", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b'')

    def test_sun_mtx(self):
        window_path = "./Objects/upper_glass.rad"
        cmd = ['getwea', '-a', '37', '-o', '122', '-wpths', window_path, '-rz']
        process = sp.run(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        print(process.stderr)
        self.assertEqual(process.stderr, b'')
        with open('test.wea', 'wb') as wtr:
            wtr.write(process.stdout)


if __name__ == "__main__":
    unittest.main()
