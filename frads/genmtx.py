"""
Command-line tool for generating matrices for different scenarios.
TWang
"""

import argparse
import logging
import os
import shutil
from frads import radmtx as rm
from frads import radutil


def genmtx_args(parser):
    parser.add_argument('-st', dest='sender_type', choices=['s','v','p'], required=True, help='Sender object type')
    parser.add_argument('-s', dest='sender', required=True, help='Sender object')
    parser.add_argument('-r', dest='receiver', nargs='+', required=True, help='Receiver objects')
    parser.add_argument('-i', dest='octree', help='Scene octree file path')
    parser.add_argument('-o', dest='outpath', nargs='+', required=True, help='Output file path | directory')
    # parser.add_argument('-mod', '--modpath', help='modifier path for sun sources')
    parser.add_argument('-env', nargs='+', default=[], help='Environment file paths')
    parser.add_argument('-rs', dest='receiver_basis', required=True, choices=('r1','r2','r4','r6','kf','sc25'),
                        help='Receiver sampling basis, ....')
    parser.add_argument('-ss', dest='sender_basis', help='Sender sampling basis if sender type is (s)urface, kf|r1|r2|....')
    parser.add_argument('-ro', dest='receiver_offset', type=float,
                        help='Move receiver surface in normal direction')
    parser.add_argument('-so', dest='sender_offset', type=float,
                        help='Move sender surface in normal direction')
    parser.add_argument('-opt', dest='option', type=str, default='-ab 1', help='Simulation parameters enclosed in double quotation marks, e.g. "-ab 1 -ad 64"')
    parser.add_argument('-rc', dest='ray_count', type=int, default=1, help='Ray count')
    parser.add_argument('-res', dest='resolu', nargs=2, default=[500, 500], type=int, help='X and Y resolution for the image')
    parser.add_argument('-smx', help='Sky matrix file path, used to cull redundant suns')
    parser.add_argument('-wpths', nargs='+', help='window primitive paths, used to cull redundant suns')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbose mode')
    return parser


def main():
    """Generate a matrix."""
    parser = argparse.ArgumentParser(
        prog='genmtx', description='Generate flux transport matrix given a ray sender and receiver(s)')
    genmtx_parser = genmtx_args(parser)
    args = genmtx_parser.parse_args()
    logger = logging.getLogger('frads')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    assert len(args.receiver) == len(args.outpath)
    env = args.env
    if args.octree is not None:
        env.extend(['-i', args.octree])
    # what's the sender
    with open(args.sender) as rdr:
        sndrlines = rdr.readlines()
    if args.sender_type == 's':
        prim_list = radutil.parse_primitive(sndrlines)
        sender = rm.Sender.as_surface(prim_list=prim_list, basis=args.sender_basis, offset=args.sender_offset)
    elif args.sender_type == 'v':
        vudict = radutil.parse_vu(sndrlines[-1]) # use the lasender_type view from a view file
        sender = rm.Sender.as_view(vu_dict=vudict, ray_cnt=args.ray_count,
                                   xres=args.resolu[0], yres=args.resolu[1])
    elif args.sender_type == 'p':
        pts_list = [l.split() for l in sndrlines]
        sender = rm.Sender.as_pts(pts_list=pts_list, ray_cnt=args.ray_count)
    # figure out receiver
    if args.receiver[0] == 'sky':
        logger.info('Sky is the receiver')
        receiver = rm.Receiver.as_sky(args.receiver_basis)
        outpath = args.outpath[0]
        if args.sender_type == 'v':
            radutil.mkdir_p(outpath)
    elif args.receiver[0] == 'sun':
        receiver = rm.Receiver.as_sun(basis=args.rs, smx_path=args.smx,
                                      window_paths=args.wpths)
    else: # assuming multiple receivers
        rcvr_prims = []
        for path in args.receiver:
            with open(path) as rdr:
                rlines = rdr.readlines()
            rcvr_prims.extend(radutil.parse_primitive(rlines))
        modifiers = set([prim['modifier'] for prim in rcvr_prims])
        receiver = rm.Receiver(receiver='', basis=args.receiver_basis, modifier=None)
        for mod, op in zip(modifiers, args.outpath):
            _receiver = [prim for prim in rcvr_prims
                         if prim['modifier'] == mod and prim['type'] in ('polygon', 'ring') ]
            if _receiver != []:
                if args.sender_type == 'v':
                    _outpath = os.path.join(op, '%04d.hdr')
                else:
                    _outpath = op
                receiver += rm.Receiver.as_surface(
                    prim_list=_receiver, basis=args.receiver_basis, offset=args.receiver_offset,
                    left=None, source='glow', out=_outpath)
        outpath = None
    # generate matrices
    if args.receiver[0] == 'sun':
        logger.info('Suns are the receivers.')
        sun_oct = 'sun.oct'
        rm.rcvr_oct(receiver, env, sun_oct)
        rm.rcontrib(sender=sender, modifier=receiver.modifier, octree=sun_oct,
                   out=outpath, opt=args.option)
    else:
        res = rm.rfluxmtx(sender=sender, receiver=receiver, env=env, opt=args.option, out=outpath)
        if (outpath is not None) and (args.sender_type != 'v'):
            with open(outpath, 'wb') as wtr:
                wtr.write(res)

