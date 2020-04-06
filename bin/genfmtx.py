#!/usr/bin/env python3
"""Commandline tool for generating facade matrix.
T.Wang"""

from frads import mfacade as fcd
from frads import radutil
import shutil
import os

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
    parser.add_argument('-s', action='store_true', help='Do solar calc')
    parser.add_argument('-env', nargs='+', default=[], help='Environment file paths')
    return parser

def klems_wrap(inp, out):
    """prepare wrapping for Klems basis."""
    cmd = f"rmtxop -fa -t -c .265 .67 .065 {inp} | getinfo - > {out}"
    os.system(cmd)

def main(**kwargs):
    with open(kwargs['w']) as rdr:
        wndw_prims = radutil.parse_primitive(rdr.readlines())
    with open(kwargs['ncp']) as rdr:
        ncp_prims = radutil.parse_primitive(rdr.readlines())
    port_prims = fcd.genport(wpolys=wndw_prims, npolys=ncp_prims,
                             depth=None, scale=None)
    wndw_polygon = [p['polygon'] for p in wndw_prims if p['type']=='polygon']
    kwargs['env'].append(kwargs['ncp'])
    all_prims = []
    for env in kwargs['env']:
        with open(env) as rdr:
            all_prims.extend(radutil.parse_primitive(rdr.readlines()))
    ncp_mod = [prim['modifier'] for prim in ncp_prims if prim['type']=='polygon'][0]
    for prim in all_prims:
        if prim['identifier'] == ncp_mod:
            ncp_mat = prim
            ncp_type = prim['type']
            break
    wrap2xml = kwargs['wrap']
    if kwargs['s'] and ncp_type=='BSDF':
        wrap2xml = False
        shutil.copyfile(ncp_mat['str_args'].split()[2], 'temp.xml')
        with open('temp.xml') as rdr:
            raw = rdr.read()
        raw = raw.replace('<Wavelength unit="Integral">Visible</Wavelength>',
                    '<Wavelength unit="Integral">Visible2</Wavelength>')
        raw = raw.replace('<Wavelength unit="Integral">Solar</Wavelength>',
                    '<Wavelength unit="Integral">Visible</Wavelength>')
        raw = raw.replace('<Wavelength unit="Integral">Visible2</Wavelength>',
                    '<Wavelength unit="Integral">Solar</Wavelength>')
        with open('solar.xml', 'w') as wtr:
            wtr.write(raw)
        _strarg = ncp_mat['str_args'].split()
        _strarg[2] = 'solar.xml'
        ncp_mat['str_args'] = ' '.join(_strarg)
        with open('env_solar.rad', 'w') as wtr:
            for prim in all_prims:
                wtr.write(radutil.put_primitive(prim))
        outsolar = '_solar_' + radutil.basename(kwargs['o'])
        fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=outsolar,
                    env=['env_solar.rad'], sbasis=kwargs['ss'], rbasis=kwargs['rs'],
                    opt=kwargs['opt'], refl=kwargs['refl'],
                    forw=kwargs['forw'], wrap=wrap2xml)
    fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=kwargs['o'],
                env=kwargs['env'], sbasis=kwargs['ss'], rbasis=kwargs['rs'],
                opt=kwargs['opt'], refl=kwargs['refl'],
                forw=kwargs['forw'], wrap=wrap2xml)
    if kwargs['s'] and ncp_type == 'BSDF':
        mtxs = [mtx for mtx in os.listdir() if mtx.endswith('.mtx')]
        vis_dict = {}
        sol_dict = {}
        oname = radutil.basename(kwargs['o'])
        for mtx in mtxs:
            if mtx.startswith(oname):
                _direc = radutil.basename(mtx).split('_')[-1][:2]
                vis_dict[_direc] = f"vis_{_direc}"
                klems_wrap(mtx, vis_dict[_direc])
            if mtx.startswith('_solar_'):
                _direc = radutil.basename(mtx).split('_')[-1][:2]
                sol_dict[_direc] = f"sol_{_direc}"
                klems_wrap(mtx, sol_dict[_direc])
        cmd = 'wrapBSDF -a kf -s Visible '
        cmd += ' '.join([f"-{key} {vis_dict[key]}" for key in vis_dict])
        cmd += ' -s Solar '
        cmd += ' '.join([f"-{key} {sol_dict[key]}" for key in sol_dict])
        cmd += f" > {oname}.xml"
        os.system(cmd)
        os.remove('temp.xml')
        os.remove('solar.xml')
        [os.remove(vis_dict[k]) for k in vis_dict]
        [os.remove(sol_dict[k]) for k in sol_dict]
        [os.remove(mtx) for mtx in mtxs]


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    genfmtx_parser = genfmtx_args(parser)
    args = genfmtx_parser.parse_args()
    argmap = vars(args)
    main(**argmap)

