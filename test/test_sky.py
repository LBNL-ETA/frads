import unittest
import os
from frads import parsers
from frads import sky


class TestSky(unittest.TestCase):
    def test_basis_glow(self):
        basis = "r1"
        result = sky.basis_glow(basis)
        answer = "#@rfluxmtx h=u\n\n"
        answer += "void glow groundglow\n"
        answer += "0\n0\n4 1 1 1 0\n\n"
        answer += "groundglow source ground\n"
        answer += "0\n0\n4 0 0 -1 180\n\n"
        answer += "#@rfluxmtx u=+Y h=r1\n\n"
        answer += "void glow skyglow\n"
        answer += "0\n0\n4 1 1 1 0\n\n"
        answer += "skyglow source sky\n"
        answer += "0\n0\n4 0 0 1 180\n"
        self.assertEqual(result, answer)

    def test_gensun(self):
        """Generate sun sources for matrix generation."""
        pass

    def test_epw2sunmtx(self):
        """Generate reinhart 6 sun matrix file from a epw file."""
        pass

    def test_loc2sunmtx(self):
        """Generate a psuedo reinhart 6 sun matrix file given lat, lon, etc..."""
        pass

    def test_gendaymtx(self):
        """."""
        pass

    def test_gendaymtx_cmd(self):
        """."""
        pass

    def test_parse_csv(self):
        pass

    def test_sky_cont(self):
        pass

    def test_gendaylit_cmd(self):
        """Get a gendaylit command as a list."""
        result = sky.gendaylit_cmd(
            "1", "2", "13.5", "37", "122.2", "120", year="2022", dir_norm_ir="500", dif_hor_ir="300"
        )
        answer = "gendaylit 1 2 13.5 -a 37 -o 122.2 -m 120 -y 2022 -W 500 300"
        self.assertEqual(" ".join(result), answer)

    def test_solar_angle(self):
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

    def test_culled_sun(self):
        with open("Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.epw") as rdr:
            wea_metadata, wea_data = parsers.parse_epw(rdr.read())

        wea_data, _ = sky.filter_wea(wea_data, wea_metadata, start_hour=6, end_hour=20, remove_zero=True, daylight_hours_only=True)
        sky.gendaymtx(
            "sun.mtx", 6, data=wea_data, meta=wea_metadata, direct=True, onesun=True
        )
        gensun = sky.Gensun(6)
        suns, mod = gensun.gen_cull(smx_path="sun.mtx")
        self.assertEqual(len(suns.splitlines()), 5186)
        self.assertEqual(len(mod.splitlines()), 615)
        os.remove("sun.mtx")


if __name__ == "__main__":
    unittest.main()