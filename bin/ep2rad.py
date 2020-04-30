#!/usr/bin/env python3
from frads import epjson2rad as eprad
from frads import radutil as ru
import argparse
import pdb


def main(fpath):
    epjs = eprad.read_epjs(fpath)
    radobj = eprad.epJSON2Rad(epjs)
    pdb.set_trace()
    #with open('materials.rad', 'w') as wtr:
    #    [wtr.write(ru.put_primitive(self.mat_prims[p])) for p in self.mat_prims]
    #    [wtr.write(ru.put_primitive(self.wndw_mat_prims[p])) for p in self.wndw_mat_prims]

    #with open(f'{zn}_geom.rad', 'w') as wtr:
    #    [wtr.write(ru.put_primitive(prim)) for prim in srf_prims]

    #with open(f'{zn}_wndw.rad', 'w') as wtr:
    #    [wtr.write(ru.put_primitive(prim)) for prim in wsrf_prims]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath')
    args = parser.parse_args()
    main(args.fpath)



