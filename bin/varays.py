#!/usr/bin/env python3

import subprocess as sp
from frads import radmtx as rm
import argparse

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-x', required=True, help='square image resolution')
    aparser.add_argument('-c', default='1')
    aparser.add_argument('-vf', required=True)
    args = aparser.parse_args()
    cmd = "vwrays -ff -vf {} -x {} -y {} ".format(args.vf, args.x, args.x)
    cmd += '-c {} -pj 0.7 '.format(args.c)
    cmd += rm.Sender.crop2circle(args.c, args.x)
    sp.run(cmd, shell=True)
