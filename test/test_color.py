import os
import unittest
from frads import color


class TestUtils(unittest.TestCase):

    test_dir_path = os.path.dirname(__file__)
    data_path = os.path.join(test_dir_path, "data")
    check_decimal_to = 6

    def test_get_tristi_paths(self):
        pass

    def test_load_cie_tristi(self):
        wvl = list(range(300, 1000, 5))
        observer = "2"
        trix, triy, triz, mlnp_i = color.load_cie_tristi(wvl, observer)
        answer_path = os.path.join(self.data_path, "sample_tri.dat")
        with open(answer_path, "r") as rdr:
            answer = rdr.readlines()
            tab = [row.split("\t") for row in answer]
        answer_trix = [float(row[2]) for row in tab]
        answer_triy = [float(row[3]) for row in tab]
        answer_triz = [float(row[4]) for row in tab]
        answer_mlnp = [float(row[5]) for row in tab]
        for r, a in zip(trix, answer_trix):
            self.assertAlmostEqual(r, a, self.check_decimal_to)
        for r, a in zip(triy, answer_triy):
            self.assertAlmostEqual(r, a, self.check_decimal_to)
        for r, a in zip(triz, answer_triz):
            self.assertAlmostEqual(r, a, self.check_decimal_to)

    def test_get_conversion_matrix(self):
        prim = "radiance"
        res = color.get_conversion_matrix(prim)
        answer = [
            2.56531284,
            -1.16684962,
            -0.398463227,
            -1.02210817,
            1.97828662,
            0.0438215555,
            0.0747243773,
            -0.251939567,
            1.17721519,
        ]
        for r, a in zip(res, answer):
            self.assertAlmostEqual(r, a, self.check_decimal_to)

    def test_rgb2xyz(self):
        r = 0.3
        g = 0.4
        b = 0.5
        coeffs = color.get_conversion_matrix("radiance", reverse=True)
        x, y, z = color.rgb2xyz(r, g, b, coeffs)
        answer_x = 0.364782628
        answer_y = 0.379968254
        answer_z = 0.482894621
        self.assertAlmostEqual(x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(y, answer_y, self.check_decimal_to)
        self.assertAlmostEqual(z, answer_z, self.check_decimal_to)

    def test_xyz2rgb(self):
        x = 0.3
        y = 0.4
        z = 0.5
        coeffs = color.get_conversion_matrix("radiance")
        r, g, b = color.xyz2rgb(x, y, z, coeffs)
        answer_r = 0.103622393
        answer_g = 0.506592973
        answer_b = 0.510249081
        self.assertAlmostEqual(r, answer_r, self.check_decimal_to)
        self.assertAlmostEqual(g, answer_g, self.check_decimal_to)
        self.assertAlmostEqual(b, answer_b, self.check_decimal_to)

    def test_spec2xyz(self):
        tri_path = os.path.join(self.data_path, "sample_tri.dat")
        with open(tri_path, "r") as rdr:
            tri = rdr.readlines()
            tab = [row.split("\t") for row in tri]
        sval = [float(row[1]) for row in tab]
        trix = [float(row[2]) for row in tab]
        triy = [float(row[3]) for row in tab]
        triz = [float(row[4]) for row in tab]
        mlnp = [float(row[5]) for row in tab]
        res_x, res_y, res_z = color.spec2xyz(trix, triy, triz, mlnp, sval)
        answer_x = 0.648757039
        answer_y = 0.676893567
        answer_z = 0.539558065
        self.assertAlmostEqual(res_x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(res_y, answer_y, self.check_decimal_to)
        self.assertAlmostEqual(res_z, answer_z, self.check_decimal_to)

    def test_xyz2xy(self):
        x, y = color.xyz2xy(0.648757039, 0.676893567, 0.539558066)
        answer_x = 0.34782
        answer_y = 0.362905
        self.assertAlmostEqual(x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(y, answer_y, self.check_decimal_to)


if __name__ == "__main__":
    unittest.main()
