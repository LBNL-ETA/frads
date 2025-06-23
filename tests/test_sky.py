from datetime import datetime
from pathlib import Path
import sys
sys.path.append(".")
import unittest

from frads import geom
import pyradiance as pr
from frads import sky
from frads import utils
from frads.sky import WeaData
from frads.sky import WeaMetaData
import logging

logger = logging.getLogger()
logger.setLevel(50)


test_dir = Path(__file__).resolve().parent
epw_path = test_dir / "Resources" / "USA_CA_Oakland.Intl.AP.724930_TMY3.epw"
wea_path = test_dir / "Resources" / "oak.wea"

class TestSky(unittest.TestCase):

    # def test_basis_glow():
    #     basis = "r1"
    #     result = sky.basis_glow(basis)
    #     assert result == ("#@rfluxmtx h=u\n\n"
    #                       "void glow groundglow\n"
    #                       "0\n0\n4 1 1 1 0\n\n"
    #                       "groundglow source ground\n"
    #                       "0\n0\n4 0 0 -1 180\n\n"
    #                       "#@rfluxmtx u=+Y h=r1\n\n"
    #                       "void glow skyglow\n"
    #                       "0\n0\n4 1 1 1 0\n\n"
    #                       "skyglow source sky\n"
    #                       "0\n0\n4 0 0 1 180\n")


    # def test_gen_sun_source_full():
    #     """Generate sun sources for matrix generation."""
    #     reinsrc4_path = test_dir / "Resources" / "reinsrc4.rad"
    #     mf = 4
    #     prim_str, mod_str = sky.gen_sun_source_full(mf)
    #     prims = pr.parse_primitive(prim_str)
    #     answer_prims = utils.unpack_primitives(reinsrc4_path)
    #     for aprim, prim in zip(prims, answer_prims):
    #         assert str(aprim) == str(prim)

    def test_gendaymtx(self):
        """Generate a psuedo reinhart 6 sun matrix file given lat, lon, etc..."""
        # sky.gendaymtx(
        #     sun_mtx, 6, data=wea_data, meta=wea_metadata, direct=True, onesun=True
        # )
        pass

    def test_filter_wea(self):
        """."""
        # wea_data, _ = sky.filter_wea(
        #     wea_data, wea_metadata, start_hour=6, end_hour=20,
        #     remove_zero=True, daylight_hours_only=True)
        pass

    # def test_check_sun_above_horizon():
    #     sky.check_sun_above_horizon()
    #     pass


    def test_solar_angle(self):
        # sky.filter_data_by_direct_sun()
        pass

    def test_start_end_hour(self):
        """Remove wea data entries outside of the
        start and end hour."""
        # sh = None
        # eh = None
        # data = None
        # result = makesky.start_end_hour(sh, eh, data)
        pass

    def test_check_sun_above_horizon(self):
        """Remove non-daylight hour entries."""
        pass

    def test_filter_wea_zero_entry(self):
        """Remove wea data entries with zero solar luminance.
        If window normal supplied, eliminate entries not seen by window.
        Solar luminance determined using Perez sky model.
        Window field of view is 176 deg with 2 deg tolerance on each side.
        """
        pass

    def test_gen_sun_source_culled(self):
        # mf = 6
        # suns, mod, full_mod = sky.gen_sun_source_culled(mf, smx_path=self.sunmtx_path)
        # assert Equal(len(suns.splitlines()), 5186)
        # assert Equal(len(mod.splitlines()), 615)
        # sun_mtx.unlink()
        pass

    # def test_filter_data_by_direct_sun():
    #     meta = WeaMetaData("Oakland", "USA", 37.72, 122.22, 120, 2)
    #     data = [
    #         WeaData(datetime(2000, 1, 1, 6, 30), 0, 0),
    #         WeaData(datetime(2000, 1, 1, 7, 30), 71, 1),
    #         WeaData(datetime(2000, 1, 1, 8, 30), 463,  41),
    #         WeaData(datetime(2000, 1, 1, 13, 30), 717, 110),
    #         WeaData(datetime(2000, 7, 1, 15, 30), 473,  95),
    #         WeaData(datetime(2000, 7, 1, 16, 30), 244,  16),
    #         WeaData(datetime(2000, 1, 1, 17, 30),   0,   0),
    #     ]
    #     wnormal = geom.Vector(0, 1, 0)
    #     filtered = sky.filter_data_by_direct_sun(data, meta, window_normal=[wnormal])
    #     answer = [
    #         WeaData(datetime(2000, 1, 1,  7, 30),  71,   1),
    #         WeaData(datetime(2000, 1, 1,  8, 30), 463,  41),
    #         WeaData(datetime(2000, 1, 1, 13, 30), 717, 110),
    #         WeaData(datetime(2000, 7, 1, 15, 30), 473,  95),
    #     ]
    #     for res, ans in zip(filtered, answer):
    #         assert res == ans

    def test_parser_epw(self):
        with open(epw_path) as rdr:
            wea_metadata, wea_data = sky.parse_epw(rdr.read())
        self.assertEqual( wea_data[0].time.month, 1)
        self.assertEqual( wea_data[-1].time.month, 12)
        self.assertEqual( wea_data[-1].time.day, 31)
        self.assertEqual( wea_metadata.latitude, 37.72)

    def test_parser_wea(self):
        with open(wea_path) as rdr:
            wea_metadata, wea_data = sky.parse_wea(rdr.read())
        self.assertEqual(wea_data[0].time.month, 1)
        self.assertEqual(wea_data[-1].time.month, 12)
        self.assertEqual(wea_data[-1].time.day, 31)
        self.assertEqual(wea_data[-1].dni, 0)
        self.assertEqual(wea_metadata.latitude, 37.72)

if __name__ == "__main__":
    unittest.main()
