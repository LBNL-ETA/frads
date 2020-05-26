#!/usr/bin/env python3
"""
Image operations with parallel processing.

Taoning Wang
"""

import argparse
import multiprocessing as mp
import glob
import os
import subprocess as sp
from frads import radutil


def main(op, inputs, out_dir, nproc=None):
    """Operate on input directories given a operation type.

    Parameters:
    op(str), operation type, choose either dcts or pcomb
    inputs(list), input directories/file.
                    For pcomb operations: include symbols like '+,'-','*' in-between;
                    For dcts, operation defaults matrix multiplications
    out_dir(str), path to store your output
    nproc(int), number of processors to use, default: total available
    trim(bool), Do you want to trim the square images to circluar ones

    """
    if op == "pcomb":
        radutil.pcombop(inputs, out_dir, nproc=nproc)
    elif op == 'dcts':
        radutil.dctsop(inputs, out_dir, nproc=nproc)


if __name__ == "__main__":
    program_scription = "Image operations with parallel processing"
    parser = argparse.ArgumentParser(prog='imgop', description=program_scription)
    parser.add_argument('-t', type=str, required=True, choices=['dcts','pcomb'],
                        help='operation types: {pcomb|dcts}')
    parser.add_argument('-i', type=str, required=True, nargs="+", help='list of inputs')
    parser.add_argument('-o', type=str, required=True, help="output directory")
    parser.add_argument('-n', type=int, help='number of processors to use')
    args = parser.parse_args()
    main(args.t, args.i, args.o, nproc=args.n)
