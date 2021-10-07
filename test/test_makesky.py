import unittest
import subprocess as sp
import os
import glob
import shutil
from frads import makesky, radutil, radmtx, util

class TestMakesky(unittest.TestCase):


    def test_culled_sun(self):
        epw_fname, epw_url = util.get_epw_url(33.6, 112.4)
        epw = util.request(epw_url, {})
        wea_metadata, wea_data = makesky.epw2wea(
            epw, dhour=True, shour=6, ehour=20,
            remove_zero=True)
        smx = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        with open('sun.mtx', 'wb') as wtr:
            wtr.write(smx)
        gensun = makesky.Gensun(6)
        suns, mod = gensun.gen_cull(smx_path='sun.mtx')
        self.assertEqual(len(suns.splitlines()), 5186)
        self.assertEqual(len(mod.splitlines()), 647)
        os.remove('sun.mtx')



if __name__ == "__main__":
    unittest.main()
