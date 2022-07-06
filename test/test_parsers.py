import os
import unittest
from frads import parsers


class TestUtils(unittest.TestCase):

    test_dir_path = os.path.dirname(__file__)
    data_path = os.path.join(test_dir_path, "data")
    check_decimal_to = 6

    def test_parse_vu(self):
        inp_str = "-vta -vv 180 -vh 180 -vp 0 0 0 -vd 0 -1 0"
        res = parsers.parse_vu(inp_str)
        answer = {"vt": "a", "vv": 180, "vh": 180, "vp": [0, 0, 0], "vd": [0, -1, 0]}
        self.assertEqual(res, answer)

    def test_parse_opt(self):
        inp_str = "-ab 8 -ad 1024 -I+ -u- -c 8 -aa .1 -lw 1e-8"
        res = parsers.parse_opt(inp_str)
        answer = {
            "ab": 8,
            "ad": 1024,
            "I": True,
            "u": False,
            "c": 8,
            "aa": 0.1,
            "lw": 1e-8,
        }
        self.assertEqual(res, answer)

    def test_parse_idf(self):
        pass

    def test_parse_optics(self):
        pass

    def test_parse_igsdb_json(self):
        pass

    def test_parse_bsdf_xml(self):
        pass

    def test_parse_branch(self):
        pass

    def test_parse_ttree(self):
        pass

    def get_nested_list_levels(self):
        pass


if __name__ == "__main__":
    unittest.main()
