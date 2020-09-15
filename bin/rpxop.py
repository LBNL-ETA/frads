#!/usr/bin/env python3
"""
Multiprocessing image operations.

T.Wang
"""

import argparse
from frads import radutil


def main(opr, inputs, out_dir, nproc=None):
    """Operate on input directories given a operation type.

    Parameters:
    op(str), operation type, choose either dcts or pcomb
    inputs(list), input directories/file.
                    For pcomb operations: include symbols like '+,'-','*' in-between;
                    For dcts, operation defaults matrix multiplications
    out_dir(str), path to store your output
    nproc(int), number of processors to use, default: total available

    """
    if opr == "pcomb":
        radutil.pcombop(inputs, out_dir, nproc=nproc)
    elif opr == 'dcts':
        radutil.dctsop(inputs, out_dir, nproc=nproc)


if __name__ == "__main__":
    PROGRAM_SCRIPTION = "Image operations with parallel processing"
    parser = argparse.ArgumentParser(prog='imgop', description=PROGRAM_SCRIPTION)
    parser.add_argument('-t', type=str, required=True, choices=['dcts', 'pcomb'],
                        help='operation types: {pcomb|dcts}')
    parser.add_argument('-i', type=str, required=True, nargs="+", help='list of inputs')
    parser.add_argument('-o', type=str, required=True, help="output directory")
    parser.add_argument('-n', type=int, help='number of processors to use')
    args = parser.parse_args()
    main(args.t, args.i, args.o, nproc=args.n)
