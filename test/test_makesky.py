import unittest
import subprocess as sp
import os
import glob
import shutil
from frads import makesky, radutil, radmtx

class TestMakesky(unittest.TestCase):


    def test_culled_sun(self):
        epw = makesky.getEPW("33.6","-112.4")
        wea_metadata, wea_data = makesky.epw2wea(
            epw.fname, dhour=True, shour=6, ehour=20,
            remove_zero=True)
        cmd, _ = makesky.gendaymtx(
            wea_data, wea_metadata, mf=6, direct=True, onesun=True)
        process = sp.run(cmd, stdout=sp.PIPE)
        with open('sun.mtx', 'wb') as wtr:
            wtr.write(process.stdout)
        gensun = makesky.Gensun(6)
        suns, mod = gensun.gen_cull(smx_path='sun.mtx')
        self.assertEqual(len(suns.splitlines()), 5186)
        self.assertEqual(len(mod.splitlines()), 647)
        os.remove('sun.mtx')



if __name__ == "__main__":
    unittest.main()
