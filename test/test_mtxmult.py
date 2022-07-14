import os
from pathlib import Path
import unittest
from frads import mtxmult


class TestMtxmult(unittest.TestCase):

    test_dir_path = os.path.dirname(__file__)
    data_path = os.path.join(test_dir_path, "Resources")
    check_decimal_to = 6

    def test_pcomb(self):
        # inp =
        # ops =
        # out_dir =
        # nproc = 1
        # mtxmult.pcomb(inp, ops, out_dir, nproc=nproc)
        # self.assertEqual(res, answer)
        # nproc = 4
        # mtxmult.pcomb(inp, ops, out_dir, nproc=nproc)
        # self.assertEqual(res, answer)
        pass

    def test_mtxmult(self):
        pass


if __name__ == "__main__":
    unittest.main()
