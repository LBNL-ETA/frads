#!/usr/bin/env python3
"""
Command-line tool for generating matrices for different scenarios.
TWang
"""

from frads import radmtx as rm
from frads import radutil

def main(**kwargs):
    """Generate a matrix."""
    # figure out environment
    env = ' '.join(kwargs['env']
    if kwargs['i'] is not None:
        env = "{} -i {}".format(env, kwargs['i'])
    # figure out sender
    with open(kwargs['s']) as rdr:
        sndrlines = rdr.readlines()
    if kwargs['st'] == 's':
        prim_list = radutil.parse_primitive(sndrlines)
        sender = rm.Sender.as_surface(
            prim_list=prim_list, basis=kwargs['ss'], offset=kwargs['so'])
    elif kwargs['st'] == 'v':
        vudict = radutil.parse_vu(sndrlines[-1]) # use the last view from a view file
        sender = rm.Sender.as_view(
            vu_dict=vudict, ray_cnt=kwargs['rc'], xres=kwargs['xres'],
            yres=kwargs['yres'], c2c=kwargs['c2c'])
    elif kwargs['st'] == 'p':
        pts_list = [l.split() for l in sndrlines]
        sender = rm.Sender.as_pts(pts_list=pts_list, ray_cnt=kwargs['rc'])
    # figure out receiver
    if kwargs['r'][0] == 'sky':
        receiver = rm.Receiver.as_sky(kwargs['rs'])
    elif kwargs['r'][0] == 'sun':
        receiver = rm.Receiver.as_sun(
            basis=kwargs['rs'], smx_path=kwargs['smx'], window_paths=kwargs['wpths'])
    else: # assuming multiple receivers
        rcvr_prims = []
        for path in kwargs['r']:
            with open(path) as rdr:
                rlines = rdr.readlines()
            rcvr_prims.extend(radutil.parse_primitive(rlines))
        modifiers = set([prim['modifier'] for prim in rcvr_prims])
        receivers = []
        for mod in modifiers:
            _receiver = [prim for prim in rcvr_prims
                         if prim['modifier'] == mod and prim['type'] in ('polygon', 'ring') ]
            if _receiver != []:
                receivers.append(rm.Receiver.as_surface(
                    prim_list=_receiver, basis=kwargs['rs'], offset=kwargs['ro'],
                    left=kwargs['left'], source=kwargs['src'], out=kwargs['o']))
        receiver = receivers[0]
        for idx in range(1, len(receivers)):
            receiver += receivers[idx]
    # generate matrices
    if kwargs['r'][0] == 'sun':
        rm.rcontrib(sender=sender, receiver=receiver, env=kwargs['env'],
                   out=kwargs['o'], opt=kwargs['opt'])
    else:
        rm.rfluxmtx(sender=sender, receiver=receiver, env=kwargs['env'],
                   out=kwargs['o'], opt=kwargs['opt'])


def genmtx_args(parser):
    parser.add_argument('-st', choices=['s','v','p'], help='Sender object type')
    parser.add_argument('-s', help='Sender object')
    parser.add_argument('-r', nargs='+', required=True, help='Receiver objects')
    parser.add_argument('-i', help='Scene octree file path')
    parser.add_argument('-o', required=True, help='Output file path | directory')
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
    parser.add_argument('-vbose', action='store_true', help='verbose mode')
    parser.add_argument('-debug', action='store_true', help='debug mode')
    parser.add_argument('-silent', action='store_true', help='silent mode')
    return parser


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    genmtx_parser = genmtx_args(parser)
    args = genmtx_parser.parse_args()
    argmap = vars(args)
    if argmap['debug'] == True:
        rm.logger.setLevel(logging.DEBUG)
    elif argmap['v'] == True:
        rm.logger.setLevel(logging.INFO)
    elif argmap['silent'] == True:
        rm.logger.setLevel(logging.ERROR)
    if argmap['rc'] > 1:
        argmap['opt'] += f" -c {argmap['rc']}"
    main(**argmap)
