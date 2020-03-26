#!/usr/bin/env python3
"""Commandline tool for generating facade matrix.
T.Wang"""

from frads import mfacade as fcd
from frads import radutil


def genfmtx_args(parser):
    parser.add_argument('-w', required=True, help='Window files')
    parser.add_argument('-ncp')
    parser.add_argument('-opt', type=str, default='-ab 1', help='Simulation parameters')
    parser.add_argument('-o', required=True, help='Output file path | directory')
    parser.add_argument('-rs', required=True, choices=['r1','r2','r4','r6','kf'])
    parser.add_argument('-ss', required=True, help='Sender sampling basis, kf|r1|r2|....')
    parser.add_argument('-forw', action='store_true', help='Crop to circle?')
    parser.add_argument('-refl', action='store_true', help='Crop to circle?')
    parser.add_argument('-wrap', action='store_true', help='Crop to circle?')
    parser.add_argument('-env', nargs='+', default=[], help='Environment file paths')
    return parser

def main(**kwargs):
    with open(kwargs['w']) as rdr:
        wndw_prims = radutil.parse_primitive(rdr.readlines())
    with open(kwargs['ncp']) as rdr:
        ncp_prims = radutil.parse_primitive(rdr.readlines())
    port_prims = fcd.genport(wpolys=wndw_prims, npolys=ncp_prims,
                             depth=None, scale=None)
    wndw_polygon = [p['polygon'] for p in wndw_prims if p['type']=='polygon']
    kwargs['env'].append(kwargs['ncp'])
    fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=kwargs['o'],
                env=kwargs['env'], sbasis=kwargs['ss'], rbasis=kwargs['rs'],
                opt=kwargs['opt'], refl=kwargs['refl'],
                forw=kwargs['forw'], wrap=kwargs['wrap'])


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    genfmtx_parser = genfmtx_args(parser)
    args = genfmtx_parser.parse_args()
    argmap = vars(args)
    main(**argmap)

