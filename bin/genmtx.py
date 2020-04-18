#!/usr/bin/env python3
"""
Command-line tool for generating matrices for different scenarios.
TWang
"""

import argparse
import logging
import shutil
import tempfile as tf
from frads import radmtx as rm
from frads import radutil


def main(**kwargs):
    """Generate a matrix."""
    td = tf.mkdtemp()
    assert len(kwargs['r']) == len(kwargs['o'])
    # figure out environment
    env = ' '.join(kwargs['env'])
    if kwargs['i'] is not None:
        env = "{} -i {}".format(env, kwargs['i'])
    # figure out sender
    with open(kwargs['s']) as rdr:
        sndrlines = rdr.readlines()
    if kwargs['st'] == 's':
        prim_list = radutil.parse_primitive(sndrlines)
        sender = rm.Sender.as_surface(tmpdir=td,
            prim_list=prim_list, basis=kwargs['ss'], offset=kwargs['so'])
    elif kwargs['st'] == 'v':
        vudict = radutil.parse_vu(sndrlines[-1]) # use the last view from a view file
        sender = rm.Sender.as_view(tmpdir=td,
            vu_dict=vudict, ray_cnt=kwargs['rc'], xres=kwargs['xres'],
            yres=kwargs['yres'], c2c=kwargs['c2c'])
    elif kwargs['st'] == 'p':
        pts_list = [l.split() for l in sndrlines]
        sender = rm.Sender.as_pts(pts_list=pts_list, ray_cnt=kwargs['rc'], tmpdir=td)
    # figure out receiver
    if kwargs['r'][0] == 'sky':
        receiver = rm.Receiver.as_sky(kwargs['rs'])
    elif kwargs['r'][0] == 'sun':
        receiver = rm.Receiver.as_sun(tmpdir=td,
            basis=kwargs['rs'], smx_path=kwargs['smx'], window_paths=kwargs['wpths'])
    else: # assuming multiple receivers
        rcvr_prims = []
        for path in kwargs['r']:
            with open(path) as rdr:
                rlines = rdr.readlines()
            rcvr_prims.extend(radutil.parse_primitive(rlines))
        modifiers = set([prim['modifier'] for prim in rcvr_prims])
        receivers = []
        receiver = rm.Receiver(path=None, receiver='', basis=kwargs['rs'], modifier=None)
        for mod in modifiers:
            _receiver = [prim for prim in rcvr_prims
                         if prim['modifier'] == mod and prim['type'] in ('polygon', 'ring') ]
            if _receiver != []:
                receiver += rm.Receiver.as_surface(tmpdir=td,
                    prim_list=_receiver, basis=kwargs['rs'], offset=kwargs['ro'],
                    left=None, source='glow', out=kwargs['o'])
    # generate matrices
    if kwargs['r'][0] == 'sun':
        rm.rcontrib(sender=sender, receiver=receiver, env=kwargs['env'],
                   out=kwargs['o'], opt=kwargs['opt'])
    else:
        rm.rfluxmtx(sender=sender, receiver=receiver, env=kwargs['env'],
                   out=kwargs['o'], opt=kwargs['opt'])
    shutil.rmtree(td)


def genmtx_args(parser):
    parser.add_argument('-st', choices=['s','v','p'], required=True, help='Sender object type')
    parser.add_argument('-s', help='Sender object')
    parser.add_argument('-r', nargs='+', required=True, help='Receiver objects')
    parser.add_argument('-i', help='Scene octree file path')
    parser.add_argument('-o', nargs='+', required=True, help='Output file path | directory')
    parser.add_argument('-mod', help='modifier path for sun sources')
    parser.add_argument('-env', nargs='+', default='', help='Environment file paths')
    parser.add_argument('-rs', required=True, choices=['r1','r2','r4','r6','kf'],
                        help='Receiver sampling basis, kf|r1|r2|....')
    parser.add_argument('-ss', help='Sender sampling basis, kf|r1|r2|....')
    parser.add_argument('-ro', type=float,
                        help='Move receiver surface in normal direction')
    parser.add_argument('-so', type=float,
                        help='Move sender surface in normal direction')
    parser.add_argument('-opt', type=str, default='-ab 1', help='Simulation parameters')
    parser.add_argument('-rc', type=int, default=1, help='Ray count')
    parser.add_argument('-xres', type=int, help='X resolution')
    parser.add_argument('-yres', type=int, help='Y resolution')
    parser.add_argument('-c2c', action='store_true', help='Crop to circle?')
    parser.add_argument('-smx', help='Sky matrix file path')
    parser.add_argument('-wpths', nargs='+', help='Windows polygon paths')
    parser.add_argument('-vb', action='store_true', help='verbose mode')
    parser.add_argument('-db', action='store_true', help='debug mode')
    parser.add_argument('-si', action='store_true', help='silent mode')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    genmtx_parser = genmtx_args(parser)
    args = genmtx_parser.parse_args()
    argmap = vars(args)
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if argmap['db'] == True:
        logger.setLevel(logging.DEBUG)
    elif argmap['vb'] == True:
        logger.setLevel(logging.INFO)
    elif argmap['si'] == True:
        logger.setLevel(logging.ERROR)
    if argmap['rc'] > 1:
        argmap['opt'] += f" -c {argmap['rc']}"
    main(**argmap)
