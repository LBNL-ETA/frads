"""Commandline tool for generating facade matrix."""

from argparse import ArgumentParser
from threading import Thread
import logging
import os
import shutil
import subprocess as sp
import tempfile as tf
from frads import mfacade as fcd
from frads import radutil, util

def genfmtx_args(parser):
    parser.add_argument('-w', '--window', required=True, help='Window files')
    parser.add_argument('-ncp')
    parser.add_argument('-opt', type=str, default='-ab 1', help='Simulation parameters')
    parser.add_argument('-o', required=True, help='Output file path | directory')
    parser.add_argument('-rs', required=True, choices=['kf','r1','r2','r4','r6','sc*'])
    parser.add_argument('-ss', required=True, help='Sender sampling basis, kf|r1|r2|....')
    parser.add_argument('-forw', action='store_true', help='Doing front direction?')
    parser.add_argument('-refl', action='store_true', help='Doing reflection?')
    parser.add_argument('-wrap', action='store_true', help='Produce an xml file instead?')
    parser.add_argument('-s', '--solar', action='store_true', help='Do solar calc?')
    parser.add_argument('-env', nargs='+', default=[], help='Environment file paths')
    return parser

def klems_wrap(out, out2, inp, basis):
    """prepare wrapping for Klems basis."""
    cmd = f"rmtxop -fa -t -c .265 .67 .065 {inp} | getinfo - > {out}"
    sp.run(cmd, shell=True)
    basis_dict = {'kq':'Klems Quarter', 'kh':'Klems Half', 'kf':'Klems Full'}
    coeff = radutil.angle_basis_coeff(basis_dict[basis])
    with open(out, 'r') as rdr:
        rows = [map(float, l.split()) for l in rdr.readlines()]
    res = [[str(val/c) for val in row] for row, c in zip(rows, coeff)]
    with open(out2, 'w') as wtr:
        [wtr.write('\t'.join(row)+'\n') for row in res]

def main():
    parser = ArgumentParser()
    genfmtx_parser = genfmtx_args(parser)
    genfmtx_parser.add_argument('-v', '--verbose', action='count', default=0, help='verbose mode')
    args = genfmtx_parser.parse_args()
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    raw_wndw_prims = radutil.unpack_primitives(args.window)
    ncp_prims = radutil.unpack_primitives(args.ncp)
    wndw_prims = [p for p in raw_wndw_prims if p.ptype=='polygon']
    port_prims = fcd.genport(wpolys=wndw_prims, npolys=ncp_prims,
                             depth=None, scale=None)
    wndw_polygon = [radutil.parse_polygon(p.real_arg) for p in wndw_prims if p.ptype=='polygon']
    args.env.append(args.ncp)
    all_prims = []
    for env in args.env:
        all_prims.extend(radutil.unpack_primitives(env))
    ncp_mod = [prim.modifier for prim in ncp_prims if prim.ptype=='polygon'][0]
    ncp_mat: radutil.Primitive
    ncp_type: str = ''
    for prim in all_prims:
        if prim.identifier == ncp_mod:
            ncp_mat = prim
            ncp_type = prim.ptype
            break
    if ncp_type == '':
        raise ValueError("Unknown NCP material")
    wrap2xml = args.wrap
    dirname = os.path.dirname(args.o)
    dirname = '.' if dirname=='' else dirname
    if args.solar and ncp_type=='BSDF':
        logger.info('Computing for solar and visible spectrum...')
        wrap2xml = False
        xmlpath = ncp_mat.str_arg.split()[2]
        td = tf.mkdtemp()
        with open(xmlpath) as rdr:
            raw = rdr.read()
        raw = raw.replace('<Wavelength unit="Integral">Visible</Wavelength>',
                    '<Wavelength unit="Integral">Visible2</Wavelength>')
        raw = raw.replace('<Wavelength unit="Integral">Solar</Wavelength>',
                    '<Wavelength unit="Integral">Visible</Wavelength>')
        raw = raw.replace('<Wavelength unit="Integral">Visible2</Wavelength>',
                    '<Wavelength unit="Integral">Solar</Wavelength>')
        solar_xml_path = os.path.join(td, 'solar.xml')
        with open(solar_xml_path, 'w') as wtr:
            wtr.write(raw)
        _strarg = ncp_mat.str_arg.split()
        _strarg[2] = solar_xml_path
        solar_ncp_mat = radutil.Primitive(ncp_mat.modifier, ncp_mat.ptype, ncp_mat.identifier+".solar", ' '.join(_strarg), '0')

        _env_path = os.path.join(td, 'env_solar.rad')
        with open(_env_path, 'w') as wtr:
            for prim in all_prims:
                wtr.write(str(prim))
        outsolar = os.path.join(dirname, '_solar_' + util.basename(args.o))
        process_thread = Thread(target=fcd.Genfmtx,
                                kwargs={'win_polygons':wndw_polygon,
                                       'port_prim':port_prims, 'out':outsolar,
                                       'env':[_env_path], 'sbasis':args.ss,
                                       'rbasis':args.rs, 'opt':args.opt,
                                       'refl':args.refl, 'forw':args.forw,
                                       'wrap':wrap2xml})
        process_thread.start()
        #fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=outsolar,
        #            env=[_env_path], sbasis=args['ss'], rbasis=args['rs'],
        #            opt=args['opt'], refl=args['refl'], forw=args['forw'], wrap=wrap2xml)

    fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=args.o,
                env=args.env, sbasis=args.ss, rbasis=args.rs, opt=args.opt,
                refl=args.refl, forw=args.forw, wrap=wrap2xml)
    if args.solar and ncp_type == 'BSDF':
        process_thread.join()
        vis_dict = {}
        sol_dict = {}
        oname = util.basename(args['o'])
        mtxs = [os.path.join(dirname, mtx) for mtx in os.listdir(dirname) if mtx.endswith('.mtx')]
        for mtx in mtxs:
            _direc = util.basename(mtx).split('_')[-1][:2]
            mtxname = util.basename(mtx)
            if mtxname.startswith(oname):
                #vis_dict[_direc] = os.path.join(dirname, f"_vis_{_direc}")
                vis_dict[_direc] = os.path.join(td, f"vis_{_direc}")
                out2 = os.path.join(dirname, f"vis_{_direc}")
                klems_wrap(vis_dict[_direc], out2, mtx, args.ss)
            if mtxname.startswith('_solar_'):
                sol_dict[_direc] = os.path.join(td, f"sol_{_direc}")
                out2 = os.path.join(dirname, f"sol_{_direc}")
                klems_wrap(sol_dict[_direc], out2, mtx, args.ss)
        cmd = f"wrapBSDF -a {args.ss} -c -s Visible "
        cmd += ' '.join([f"-{key} {vis_dict[key]}" for key in vis_dict])
        cmd += ' -s Solar '
        cmd += ' '.join([f"-{key} {sol_dict[key]}" for key in sol_dict])
        cmd += f" > {os.path.join(dirname, oname)}.xml"
        os.system(cmd)
        shutil.rmtree(td)
        [os.remove(mtx) for mtx in mtxs]
