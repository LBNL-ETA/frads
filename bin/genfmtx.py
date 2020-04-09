#!/usr/bin/env python3
"""Commandline tool for generating facade matrix.
T.Wang"""

from frads import mfacade as fcd
from frads import radutil
from threading import Thread
from multiprocessing import Process
import tempfile as tf
import shutil
import os
import subprocess as sp

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
    sp.run(cmd, shell=True)

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
    dirname = os.path.dirname(kwargs['o'])
    dirname = '.' if dirname=='' else dirname
    if kwargs['s'] and ncp_type=='BSDF':
        logger.info('Computing for solar and visible spectrum...')
        wrap2xml = False
        xmlpath = ncp_mat['str_args'].split()[2]
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
        _strarg = ncp_mat['str_args'].split()
        _strarg[2] = solar_xml_path
        ncp_mat['str_args'] = ' '.join(_strarg)
        _env_path = os.path.join(td, 'env_solar.rad')
        with open(_env_path, 'w') as wtr:
            for prim in all_prims:
                wtr.write(radutil.put_primitive(prim))
        outsolar = os.path.join(dirname, '_solar_' + radutil.basename(kwargs['o']))
        process_thread = Thread(target=fcd.Genfmtx,
                                kwargs={'win_polygons':wndw_polygon,
                                       'port_prim':port_prims, 'out':outsolar,
                                       'env':[_env_path], 'sbasis':kwargs['ss'],
                                       'rbasis':kwargs['rs'], 'opt':kwargs['opt'],
                                       'refl':kwargs['refl'], 'forw':kwargs['forw'],
                                       'wrap':wrap2xml})
        process_thread.start()
        #sprocess = Process(target=fcd.Genfmtx,
        #                        kwargs={'win_polygons':wndw_polygon,
        #                               'port_prim':port_prims, 'out':outsolar,
        #                               'env':[_env_path], 'sbasis':kwargs['ss'],
        #                               'rbasis':kwargs['rs'], 'opt':kwargs['opt'],
        #                               'refl':kwargs['refl'], 'forw':kwargs['forw'],
        #                               'wrap':wrap2xml})
        #sprocess.start()
        #fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=outsolar,
        #            env=[_env_path], sbasis=kwargs['ss'], rbasis=kwargs['rs'],
        #            opt=kwargs['opt'], refl=kwargs['refl'], forw=kwargs['forw'], wrap=wrap2xml)

    fcd.Genfmtx(win_polygons=wndw_polygon, port_prim=port_prims, out=kwargs['o'],
                env=kwargs['env'], sbasis=kwargs['ss'], rbasis=kwargs['rs'],
                opt=kwargs['opt'], refl=kwargs['refl'],
                forw=kwargs['forw'], wrap=wrap2xml)
    if kwargs['s'] and ncp_type == 'BSDF':
        #sprocess.join()
        process_thread.join()
        vis_dict = {}
        sol_dict = {}
        oname = radutil.basename(kwargs['o'])
        mtxs = [os.path.join(dirname, mtx) for mtx in os.listdir(dirname) if mtx.endswith('.mtx')]
        for mtx in mtxs:
            _direc = radutil.basename(mtx).split('_')[-1][:2]
            mtxname = radutil.basename(mtx)
            if mtxname.startswith(oname):
                vis_dict[_direc] = os.path.join(dirname, f"vis_{_direc}")
                klems_wrap(mtx, vis_dict[_direc])
            if mtxname.startswith('_solar_'):
                sol_dict[_direc] = os.path.join(dirname, f"sol_{_direc}")
                klems_wrap(mtx, sol_dict[_direc])
        cmd = 'wrapBSDF -a kf -c -s Visible '
        cmd += ' '.join([f"-{key} {vis_dict[key]}" for key in vis_dict])
        cmd += ' -s Solar '
        cmd += ' '.join([f"-{key} {sol_dict[key]}" for key in sol_dict])
        cmd += f" > {os.path.join(dirname, oname)}.xml"
        os.system(cmd)
        shutil.rmtree(td)
        [os.remove(mtx) for mtx in mtxs]


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging
    parser = ArgumentParser()
    genfmtx_parser = genfmtx_args(parser)
    genfmtx_parser.add_argument('-vb', action='store_true', help='verbose mode')
    genfmtx_parser.add_argument('-db', action='store_true', help='debug mode')
    genfmtx_parser.add_argument('-si', action='store_true', help='silent mode')
    args = genfmtx_parser.parse_args()
    argmap = vars(args)
    logger = logging.getLogger('frads.mfacade')
    if argmap['db']:
        logger.setLevel(logging.DEBUG)
    elif argmap['vb']:
        logger.setLevel(logging.INFO)
    elif argmap['si']:
        logger.setLevel(logging.CRITICAL)
    main(**argmap)
