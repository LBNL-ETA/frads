"""
Command-line tool for generating matrices for different scenarios.
TWang
"""

import argparse
import logging
import os
from typing import Set, Union
import tempfile as tf
from frads import radmtx as rm
from frads import radgeom
from frads import radutil, util


def genmtx_args(parser):
    """Add arguments to r."""
    parser.add_argument(
        '-st', dest='sender_type', choices=['s', 'v', 'p'], required=True,
        help='Sender object type: (s)urface, (v)iew, (p)oint')
    parser.add_argument('-s', dest='sender', required=True,
                        help='Sender object: view | grid point | .rad file')
    parser.add_argument('-r', dest='receiver', nargs='+', required=True,
                        help='Receiver objects, sky | sun | *.rad files')
    parser.add_argument('-i', dest='octree', help='Scene octree file')
    parser.add_argument('-o', dest='outpath', nargs='+', required=True,
                        help='Output file path | directory')
    parser.add_argument('-env', nargs='+', default=[],
                        help='Environment files')
    parser.add_argument('-rs', dest='receiver_basis', required=True,
                        choices=('r1', 'r2', 'r4', 'r6', 'kf', 'sc25'),
                        help='Receiver sampling basis, ....')
    parser.add_argument('-ss', dest='sender_basis',
                        help='Surface sender sampling basis: kf|r1|r2|..')
    parser.add_argument('-ro', dest='receiver_offset', type=float,
                        help='Move receiver surface in normal direction')
    parser.add_argument('-so', dest='sender_offset', type=float,
                        help='Move sender surface in normal direction')
    parser.add_argument('-opt', dest='option', type=str, default='-ab 1',
                        help='Simulation parameters enclosed in quotes')
    parser.add_argument('-rc', dest='ray_count', type=int,
                        default=1, help='Ray count')
    parser.add_argument('-res', dest='resolu', nargs=2, default=[800, 800],
                        type=int, help='Image res., defeault=%(default)s')
    parser.add_argument('-smx', help='Sky matrix file')
    parser.add_argument('-wpths', nargs='+', help='window files paths')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='verbose mode')
    return parser


def main():
    """Generate a matrix."""
    description = 'Generate flux transport matrix'
    parser = argparse.ArgumentParser(prog='genmtx', description=description)
    genmtx_r = genmtx_args(parser)
    args = genmtx_r.parse_args()
    logger = logging.getLogger('frads')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    assert len(args.receiver) == len(args.outpath)
    env = args.env
    sender = None
    outpath = None
    if args.octree is not None:
        env.extend(['-i', args.octree])
    # what's the sender
    with open(args.sender) as rdr:
        sndrlines = rdr.readlines()
    if args.sender_type == 's':
        prim_list = radutil.parse_primitive(sndrlines)
        sender = rm.Sender.as_surface(prim_list=prim_list,
                                      basis=args.sender_basis,
                                      offset=args.sender_offset)
    elif args.sender_type == 'v':
        vudict = util.parse_vu(sndrlines[-1])  # use the last view
        sender = rm.Sender.as_view(vu_dict=vudict, ray_cnt=args.ray_count,
                                   xres=args.resolu[0], yres=args.resolu[1])
    elif args.sender_type == 'p':
        pts_list = [line.split() for line in sndrlines]
        sender = rm.Sender.as_pts(pts_list=pts_list, ray_cnt=args.ray_count)
    # figure out receiver
    if args.receiver[0] == 'sky':
        logger.info('Sky is the receiver')
        receiver = rm.Receiver.as_sky(args.receiver_basis)
        outpath = args.outpath[0]
        if args.sender_type == 'v':
            util.mkdir_p(outpath)
    elif args.receiver[0] == 'sun':
        full_modifier = False
        if args.sender_type != 'v':
            full_modifier = True
        window_normals: Union[None, Set[radgeom.Vector]] = None
        if args.wpths is not None:
            window_normals = radutil.primitive_normal(args.wpths)
        receiver = rm.Receiver.as_sun(
            basis=args.receiver_basis, smx_path=args.smx,
            window_normals=window_normals, full_mod=full_modifier)
    else:  # assuming multiple receivers
        rcvr_prims = []
        for path in args.receiver:
            rcvr_prims.extend(radutil.unpack_primitives(path))
        modifiers = set([prim.modifier for prim in rcvr_prims])
        receiver = rm.Receiver(
            receiver='', basis=args.receiver_basis, modifier=None)
        for mod, op in zip(modifiers, args.outpath):
            _receiver = [prim for prim in rcvr_prims
                         if prim.modifier == mod and
                         prim.ptype in ('polygon', 'ring')]
            if _receiver != []:
                if args.sender_type == 'v':
                    _outpath = os.path.join(op, '%04d.hdr')
                else:
                    _outpath = op
                receiver += rm.Receiver.as_surface(
                    prim_list=_receiver, basis=args.receiver_basis,
                    offset=args.receiver_offset, left=None,
                    source='glow', out=_outpath)
        outpath = None
    # generate matrices
    if args.receiver[0] == 'sun':
        logger.info('Suns are the receivers.')
        outpath = os.path.join(os.getcwd(), args.outpath[0])
        sun_oct = 'sun.oct'
        rm.rcvr_oct(receiver, env, sun_oct)
        with tf.TemporaryDirectory(dir=os.getcwd()) as tempd:
            mod_names = ["%04d" % (int(l[3:])-1)
                         for l in receiver.modifier.splitlines()]
            rm.rcontrib(sender=sender, modifier=receiver.modifier, octree=sun_oct,
                       out=tempd, opt=args.option)
            _files = [os.path.join(tempd, f) for f in sorted(os.listdir(tempd))
                      if f.endswith('.hdr')]
            util.mkdir_p(outpath)
            for idx, val in enumerate(_files):
                os.rename(val, os.path.join(outpath, mod_names[idx]+'.hdr'))

    else:
        res = rm.rfluxmtx(sender=sender, receiver=receiver, env=env, opt=args.option, out=outpath)
        if (outpath is not None) and (args.sender_type != 'v'):
            with open(outpath, 'wb') as wtr:
                wtr.write(res)

