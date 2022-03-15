"""
Unittests for getwea
"""

import unittest
import subprocess as sp


class TestGetwea(unittest.TestCase):
    """Test functionality associated with getwea command-line tool."""

    def test_help(self):
        """Test starting getwea."""
        cmd = ["getwea", "-h"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")

    def test_us_latlon(self):
        """Test downloading US EPW from Lat&Lon."""
        cmd = ["getwea", "-a", "37", "-o", "122"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")

    def test_non_us_latlon(self):
        """Test downloading non-US EPW from Lat&Lon."""
        cmd = ["getwea", "-a", "28", "-o", "-122"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")

    def test_us_zip(self):
        """Test downloading US EPW using zipcode."""
        cmd = ["getwea", "-z", "94720"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")

    def test_sun_mtx(self):
        """Test download and generating .wea based on window orientation
        and remove entries with zero solar radiance.
        """
        window_path = "./Objects/upper_glass.rad"
        cmd = ["getwea", "-a", "37", "-o", "122", "-wpths", window_path, "-rz"]
        process = sp.run(cmd, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
        self.assertEqual(process.stderr, b"")


if __name__ == "__main__":
    unittest.main()
