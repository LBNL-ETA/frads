import os
from pathlib import Path
import unittest
from frads import parsers


class TestUtils(unittest.TestCase):

    test_dir_path = os.path.dirname(__file__)
    data_path = os.path.join(test_dir_path, "Resources")
    check_decimal_to = 6
    epw_path = Path("Resources", "USA_CA_Oakland.Intl.AP.724930_TMY3.epw")
    wea_path = Path("Resources", "oak.wea")

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

    def test_parser_epw(self):
        with open(self.epw_path) as rdr:
            wea_metadata, wea_data = parsers.parse_epw(rdr.read())
        self.assertEqual(wea_data[0].month, 1)
        self.assertEqual(wea_data[-1].month, 12)
        self.assertEqual(wea_data[-1].day, 31)
        self.assertEqual(wea_metadata.latitude, 37.72)


    def test_parser_wea(self):
        with open(self.wea_path) as rdr:
            wea_metadata, wea_data = parsers.parse_wea(rdr.read())
        self.assertEqual(wea_data[0].month, 1)
        self.assertEqual(wea_data[-1].month, 12)
        self.assertEqual(wea_data[-1].day, 31)
        self.assertEqual(wea_data[-1].dni, 0)
        self.assertEqual(wea_metadata.latitude, 37.72)

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
