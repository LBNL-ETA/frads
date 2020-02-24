#!/usr/bin/env python3
"""
T.Wang

"""

import argparse
import logging
import os
import subprocess as sp
import tempfile as tf
from frads import genfmtx
from frads import makesky
from frads import radgeom
from frads import radutil
from frads import radmtx

log_level = 'info'  # set logging level
log_level_dict = {'info': 20, 'warning': 30, 'error': 40, 'critical': 50}
logger = logging.getLogger(__name__)
logger.setLevel(log_level_dict[log_level])
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_level + '-radmtx.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def parse_config(cfg_path):
    """Parse a configuration file into a dictionary."""
    cfg = {}
    _config = ConfigParser()
    _config.read(cfg_path)
    cfg = _config._sections
    cfg['lat'] = cfg['Site']['latitude']
    cfg['lon'] = cfg['Site']['longitude']
    cfg['timezone'] = cfg['Site']['timezone']
    cfg['elevation'] = cfg['Site']['elevation']
    cfg['orient'] = cfg['Site']['orientation']
    cfg['dimensions'] = cfg['Dimensions']
    filestrct = cfg['FileStructure']
    cfg['dmatrices'] = filestrct['matrices']
    simctrl = cfg['SimulationControl']
    cfg['parallel'] = True if simctrl['parallel'] == 'True' else False
    cfg['vmx_opt'] = simctrl['vmx']
    cfg['dmx_opt'] = simctrl['dmx']
    cfg['view'] = cfg['View']['view1']
    cfg['grid_height'] = float(cfg['Grid']['height'])
    cfg['grid_spacing'] = float(cfg['Grid']['spacing'])

    return cfg


def get_paths(config):
    """Where things are?"""
    bsdfd = os.path.join(root, config['dresources'], 'BSDFs')
    btdfd_vis = os.path.join(root, bsdfd, 'vis')
    btdfd_shgc = os.path.join(root, bsdfd, 'shgc')
    btdfs_vis = sorted([
        os.path.join(root, btdfd_vis, i) for i in os.listdir(btdfd_vis)
        if i.endswith('.mtx')
    ])
    btdfs_shgc = sorted([
        os.path.join(root, btdfd_shgc, i) for i in os.listdir(btdfd_shgc)
        if i.endswith('.mtx')
    ])


def assemble():
    """."""
    room = ""
    return setup


def make_room(dims):
    """Make a side-lit shoebox room."""
    theroom = room.Room(float(dims['width']), float(dims['depth']),
                        float(dims['height']))
    wndw_names = [i for i in dims if i.startswith('window')]
    for wd in wndw_names:
        wdim = map(float, dims[wd].split())
        theroom.swall.add_window(wd, theroom.swall.make_window(*wdim))
    theroom.swall.facadize(float(dims['facade_thickness']))
    theroom.surface_prim()
    theroom.window_prim()
    mlib = radutil.material_lib()
    sensor_grid = radutil.gen_grid(theroom.floor, grid_height, grid_spacing)
    nsensor = len(sensor_grid)
    return theroom, sensor_grid


def make_matrices(theroom, sensor_grid):
    vmxs = {}
    dmxs = {}
    for wname in theroom.swall.windows:
        ovmx = os.path.join(root, matricesd, 'vmx{}'.format(wname))
        odmx = os.path.join(root, matricesd, 'dmx{}'.format(wname))
        vmxs[wname] = ovmx
        dmxs[wname] = odmx
        if remake_matrices:
            wndw = theroom.wndw_prims[wname]
            mlib.extend(theroom.srf_prims)
            Genmtx(
                sender=Sender(sensor_grid),
                receiver=Receiver([wndw], 'kf'),
                out_path=ovmx,
                env=mlib,
                opt=vmx_opt,
            )
            Genmtx(
                sender=Sender(wndw, basis='kf'),
                receiver=Receiver('sky', 'r4'),
                out_path=odmx,
                env=mlib,
                opt=dmx_opt,
            )


def sender_vu(view, xres, yres, c2c, ray_cnt=None):
    pass


def sender_grid(grid):
    """prepare sender string as grid.
    Argument:
        grid: [[xp1, yp1, zp1, dx1, dy1, dz1]]
    Return:
        String representation of the sender.
    """
    gridstr = '\n'.join([' '.join(map(str, row)) for row in grid])
    return gridstr


def sender_srf(surfaces, basis, offset=None, ray_cnt=None):
    pass




def main(cfgpath):
    config = parse_config(cfgpath)
    get_paths()
    if config['dimensions'] != '':
        theroom, sensor_grid = make_room()
    else:
        theroom = assemble_room()
    make_matrices(theroom, sensor_grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    genmtx_parser = genmtx_args(parser)
    args = genmtx_parser.parse_args()
    main()
