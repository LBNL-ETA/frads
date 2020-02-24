#!/usr/bin/env python3

from frads import radutil

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('surface')
    parser.add_argument('spacing', type=float )
    parser.add_argument('height', type=float)
    parser.add_argument('-op', action='store_const', const='', default=True)
    args = parser.parse_args()
    with open(args.surface) as rdr:
        prim = radutil.parse_primitive(rdr.readlines())
    grid_list = radutil.gen_grid(prim[0]['polygon'], args.height, args.spacing, op=args.op)
    grid_str = '\n'.join([' '.join(map(str, row)) for row in grid_list])
    print(grid_str)
