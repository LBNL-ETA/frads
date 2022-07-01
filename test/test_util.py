import os
import unittest
from frads import util
from frads import radutil


class TestUtil(unittest.TestCase):

    test_dir_path = os.path.dirname(__file__)
    data_path = os.path.join(test_dir_path, 'data')
    check_decimal_to = 6


    def test_parse_vu(self):
        inp_str = "-vta -vv 180 -vh 180 -vp 0 0 0 -vd 0 -1 0"
        res = util.parse_vu(inp_str)
        answer = {"vt": "a", "vv": 180,
                "vh": 180, "vp": [0, 0, 0], "vd": [0, -1, 0]}
        self.assertEqual(res, answer)


    def test_parse_opt(self):
        inp_str = "-ab 8 -ad 1024 -I+ -u- -c 8 -aa .1 -lw 1e-8"
        res = util.parse_opt(inp_str)
        answer = {"ab": 8, "ad": 1024, "I": True, "u": False, "c": 8,
                "aa": .1, "lw": 1e-8}
        self.assertEqual(res, answer)


    def test_parse_idf(self):
        pass

    def test_parse_optics(self):
        pass

    def test_parse_igsdb_json(self):
        pass

    def test_igsdb_json(self):
        pass

    def test_get_tristi_paths(self):
        pass


    def test_load_cie_tristi(self):
        wvl = list(range(300, 1000, 5))
        observer = '2'
        trix, triy, triz, mlnp_i = util.load_cie_tristi(wvl, observer)
        answer_path = os.path.join(
                self.data_path, "sample_tri.dat")
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
        prim = 'radiance'
        res = util.get_conversion_matrix(prim)
        answer = [2.56531284, -1.16684962, -0.398463227,
                -1.02210817, 1.97828662, 0.0438215555,
                0.0747243773,-0.251939567,1.17721519]
        for r, a in zip(res, answer):
            self.assertAlmostEqual(r, a, self.check_decimal_to)
    

    def test_rgb2xyz(self):
        r = .3
        g = .4
        b = .5
        coeffs = util.get_conversion_matrix('radiance', reverse=True)
        x, y, z = util.rgb2xyz(r, g, b, coeffs)
        answer_x = 0.364782628
        answer_y = 0.379968254
        answer_z = 0.482894621
        self.assertAlmostEqual(x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(y, answer_y, self.check_decimal_to)
        self.assertAlmostEqual(z, answer_z, self.check_decimal_to)


    def test_xyz2rgb(self):
        x = .3
        y = .4
        z = .5
        coeffs = util.get_conversion_matrix('radiance')
        r, g, b = util.xyz2rgb(x, y, z, coeffs)
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
        res_x, res_y, res_z = util.spec2xyz(trix, triy, triz, mlnp, sval)
        answer_x = 0.648757039
        answer_y = 0.676893567
        answer_z = 0.539558065
        self.assertAlmostEqual(res_x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(res_y, answer_y, self.check_decimal_to)
        self.assertAlmostEqual(res_z, answer_z, self.check_decimal_to)


    def test_xyz2xy(self):
        x, y = util.xyz2xy(0.648757039, 0.676893567, 0.539558066)
        answer_x = 0.34782
        answer_y = 0.362905
        self.assertAlmostEqual(x, answer_x, self.check_decimal_to)
        self.assertAlmostEqual(y, answer_y, self.check_decimal_to)


    def test_unpack_idf(self):
        pass


    def test_nest_list(self):
        pass


    def test_write_square_matrix(self):
        pass


    def test_parse_bsdf_xml(self):
        pass


    def test_dhi2dni(self):
        pass


    def test_basename(self):
        pass


    def test_is_number(self):
        pass


    def test_silent_remove(self):
        pass


    def test_square2disk(self):
        pass


    def test_mkdir_p(self):
        pass


    def test_sprun(self):
        pass


    def test_spcheckout(self):
        pass


    def test_get_latlon_from_zipcode(self):
        pass


    def test_haversine(self):
        pass


    def test_get_epw_url(self):
        pass


    def test_request(self):
        pass


    def test_id_generator(self):
        pass


    def test_tokenize(self):
        pass

    def test_parse_branch(self):
        pass


    def test_parse_ttree(self):
        pass


    def get_nested_list_levels(self):
        pass


if __name__ == "__main__":
    unittest.main()
