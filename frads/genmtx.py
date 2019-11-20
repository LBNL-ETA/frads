#!/usr/bin/env python3
"""
Generate matrices.

Scenarios:
1. polygon to polygon (e.g facade matrix)
2. polygon to sky (e.g. daylight matrix)
3. view to polygon (e.g. image based view matrix)
4. grid to polygon (e.g. point based view matrix)
5. view to suns (e.g 5PM direct sun coefficient)
6. grid to suns

T.Wang

"""

import argparse
from frads import makesky
from frads import radgeom
import os
import subprocess as sp
import tempfile as tf
from frads import radutil
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('genmtx.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Sender(object):
    """Sender object for matrix generation."""

    def __init__(self,
                 sender,
                 basis=None,
                 offset=None,
                 xres=None,
                 yres=None,
                 c2c=True,
                 ray_cnt=None):
        """Instantiate the instance.

        Parameters:
            sender: path to view file/ pts grid/ other surface file;
            sampling: sender sampling basis, required when sender is a surface;
            offset: move the sender surface in its normal direction
            xres, yres: xy resolution of the image, required if sender is a viewfile;
            c2c (bool): Set to True to trim the rays that are sent to the corner of
            the image by setting the ray direction to 0 0 0;
        """
        self.sender = sender
        self.xres = xres
        self.yres = yres
        self.ray_cnt = ray_cnt
        self.c2c = c2c
        if isinstance(sender, str):
            if os.path.isfile(sender):
                with open(sender) as rdr:
                    lines = rdr.read().strip().splitlines()
                form = self.id_from_list(lines)
                if form == 'grid':
                    path = sender
                elif form == 'view':
                    path = self.prepare_view(lines[0])
                elif form == 'srf':
                    prim = radutil.parse_primitive(sender)
                    prim = [
                        p for p in prim if p['type'] in ['polygon', 'ring']
                    ]
                    prim_str = prepare_surface(prim, basis, offset=offset)
                    logger.info(prim_str)
                    path = self.write_srf(prim_str)
            else:
                raise OSError('path does not exist')
        elif isinstance(sender, dict):
            form = self.id_from_dict(sender)
            if form == 'view':
                path = self.prepare_view(sender)
            else:
                prim_str = prepare_surface([sender], basis, offset=offset)
                path = self.write_srf(prim_str)
        elif isinstance(sender, list):
            form = self.id_from_list(sender)
            if form == 'srf':
                prim_str = prepare_surface(sender, basis, offset=offset)
                path = self.write_srf(prim_str)
            elif form == 'grid':
                path = self.write_grid(sender)
            elif form == 'view':
                path = self.prepare_view(sender)
        self.form = form
        self.path = path
        logger.info("Sender form: {}".format(form))
        logger.info("Sender path: {}".format(path))

    def prepare_view(self, view):
        assert None not in [self.xres, self.yres], "Need to specify resolution"
        cmd = "vwrays -ff -x {} -y {} ".format(self.xres, self.yres)
        if self.ray_cnt is not None:
            if self.ray_cnt > 1:
                cmd += '-c {} -pj 0.7 '.format(self.ray_cnt)
        if isinstance(view, dict):
            vu_str = radutil.opt2str(view)
        elif os.path.isfile(view):
            with open(view) as rdr:
                vline = rdr.read().strip()
            view = radutil.parse_vu(vline)
            vu_str = radutil.opt2str(view)
        elif isinstance(view, str):
            view = radutil.parse_vu(view)
            vu_str = radutil.opt2str(view)
        cmd += vu_str
        fd, path = tf.mkstemp()
        if view['vt'] == 'a' and self.c2c:
            cmd += "| rcalc -if6 -of "
            cmd += '-e "DIM:{};CNT:{}" '.format(self.xres, self.ray_cnt)
            cmd += '-e "pn=(recno-1)/CNT+.5" '
            cmd += '-e "frac(x):x-floor(x)" -e "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM)"'
            cmd += ' -e "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0)"'
            cmd += ' -e "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir"'
        if os.name == "posix":
            cmd = cmd.replace('"', "'")
        cmd += "> {}".format(path)
        logger.info(cmd)
        sp.call(cmd, shell=True)
        return path

    def id_from_list(self, li):
        """ID sender type."""
        if isinstance(li[0], str) and radutil.is_number(li[0]):
            self.line_cnt = 1
            form = 'grid'
        elif isinstance(li[0], dict):
            form = 'srf'
        elif radutil.is_number(li[0].split()[0]):
            self.line_cnt = len(li)
            form = 'grid'
        elif '-vp ' in li[0]:
            form = 'view'
        else:
            form = 'srf'
        return form

    def id_from_dict(self, d):
        if 'vp' in d:
            form = 'view'
            self.prepare_view(d)
        else:
            form = 'srf'
        return form

    def write_srf(self, prim_str):
        fd, path = tf.mkstemp(prefix='sndr_srf')
        with open(path, 'w') as wtr:
            wtr.write(prim_str)
        return path

    def write_grid(self, li):
        if isinstance(li[0], list):
            grid_str = '\n'.join([' '.join(map(str, l)) for l in li])
        else:
            grid_str = '\n'.join(li)
        fd, path = tf.mkstemp(prefix='sndr_grid')
        with open(path, 'w') as wtr:
            wtr.write(grid_str)
        return path


class Receiver(object):
    """Receiver object for matrix generation."""

    def __init__(self,
                 rcvr,
                 basis,
                 rctype=None,
                 offset=None,
                 left=None,
                 smx_path=None,
                 window_paths=None,
                 out=None):
        """Instantiate the receiver object.

        Parameters:
            receiver (str): {sky | sun | file_path}
            sampling: receiver sampling basis {kf | r1 | sc25...}
            offset: move the receiver surface in its normal direction
        """
        self.rcvr = rcvr
        self.basis = basis
        if rcvr == 'sky':
            logger.info("receiver is sky with {}".format(basis))
            assert basis.startswith(
                'r'), 'Sky basis need to be Treganza/Reinhart'
            str_repr = makesky.basis_glow(basis)
            logger.info(str_repr)
        elif rcvr == 'sun':
            gensun = makesky.Gensun(int(basis[-1]))
            if (smx_path is None) and (window_paths is None):
                str_repr, self.mod_lines = gensun.gen_full()
            else:
                str_repr, self.mod_lines = gensun.gen_cull(
                    smx_path=smx_path, window_paths=window_paths)
        else:
            str_repr = prepare_surface(self.rcvr,
                                       basis,
                                       offset=offset,
                                       left=left,
                                       source='glow',
                                       out=out)
        self.str_repr = str_repr
        self.out = out

    def __add__(self, other):
        self.str_repr += other.str_repr
        return self


class Genmtx(object):
    """Generate Radiance matrix file."""

    def __init__(self,
                 sender=None,
                 out_path=None,
                 receiver=None,
                 env=None,
                 env_path=None,
                 opt_path=None,
                 **kwargs):
        """Initialize with inputs."""
        assert None not in [sender, receiver], "Miss sender and/or receiver"
        self.sender = sender
        self.receiver = receiver
        self.out_path = out_path
        self.tempd = tf.mkdtemp()
        env_str = [radutil.put_primitive(e) for e in env]
        self.env_path = os.path.join(self.tempd, 'env')
        with open(self.env_path, 'w') as wtr:
            [wtr.write(envs) for envs in env_str]
        #if type(env) is list:
        #    self.env_path = ' '.join(env)
        #else:
        #    self.env_path = env
        self.rfd, self.rcvr_path = tf.mkstemp(prefix='receiver')
        opt_dict = {}
        if opt_path is not None:
            with open(opt_path, 'r') as rd:
                line = rd.read()
            opt_dict.update(radutil.parse_opt(line))
        if kwargs['opt'] is not None:
            opt_dict.update(radutil.parse_opt(kwargs['opt']))
        self.opt_dict = opt_dict
        self.opt_str = radutil.opt2str(opt_dict)

        if self.receiver.rcvr == 'sun':  # if receiver is sun (for 5PM calc)
            with open(self.rcvr_path, 'w') as wtr:
                wtr.write(self.receiver.str_repr)
            self.mfd, self.mod_path = tf.mkstemp(prefix='sun_mod')
            with open(self.mod_path, 'w') as wtr:
                wtr.write(self.receiver.mod_lines)
            self.envfd, self._env_path = tf.mkstemp(prefix='with_suns')
            cmd = 'oconv {} {} > {}'.format(self.env_path, self.rcvr_path,
                                            self._env_path)
            logger.info(cmd)
            sp.call(cmd, shell=True)

        if self.sender.form == 'grid':
            self.pts_mtx()

        elif self.sender.form == 'view':
            digi_d = {
                'kq': '2',
                'kh': '2',
                'kf': '3',
                'r1': '3',
                'r2': '3',
                'r4': '4',
                'r6': '4'
            }
            digic = digi_d[self.receiver.basis]
            self.out_path = os.path.join(out_path, '%' + '0%sd.hdr' % (digic))
            self.img_mtx()

        elif self.sender.form == 'srf':
            with open(self.rcvr_path, 'w') as r:
                r.write(self.receiver.str_repr)
            cmd = 'rfluxmtx {} {} {} {} '.format(self.opt_str,
                                                 self.sender.path,
                                                 self.rcvr_path, self.env_path)
            if self.receiver.out is None:
                cmd += "> {}".format(self.out_path)
            logger.info(cmd)
            sp.call(cmd, shell=True)
        os.close(self.rfd)
        radutil.silent_remove(self.rcvr_path)

    def pts_mtx(self):
        """Generate point file based matrices."""
        if 'c' in self.opt_dict:
            assert int(
                self.opt_dict['c']) == 1, "ray count can't be greater than 1"
        with open(self.rcvr_path, 'w') as wtr:
            wtr.write(self.receiver.str_repr)

        if self.receiver.rcvr == 'sun':
            try:
                cmd = 'rcontrib < {} {} -fo+ -faf -M {} {} > {}'.format(
                    self.sender.path, self.opt_str, self.mod_path,
                    self._env_path, self.out_path)
                logger.info(cmd)
                sp.call(cmd, shell=True)
            finally:
                radutil.silent_remove(self._env_path)
                radutil.silent_remove(self.mod_path)
        else:
            with open(self.sender.path) as rdr:
                first_row = rdr.readlines()[0]
                first_pt = radgeom.Vector(*map(float, first_row.split()[:3]))
            if self.receiver.rcvr != 'sky':
                rcvr_cntr = self.receiver.rcvr[0]['polygon'].centroid()
                rcvr2sndr = first_pt - rcvr_cntr
                rcvr_normal = self.receiver.rcvr[0]['polygon'].normal()
                if rcvr2sndr * rcvr_normal < 0:
                    logger.warn('Suspicious receiver orientation')
            cmd = "rfluxmtx < {} {} -o {} -y {} - {} {}".format(
                self.sender.path, self.opt_str, self.out_path,
                self.sender.line_cnt, self.rcvr_path, self.env_path)
            logger.info(cmd)
            sp.call(cmd, shell=True)

    def img_mtx(self):
        """Generate image based matrices."""
        option = "{} -ffc `vwrays -vf {} -x {} -y {} -d` ".format(
            self.opt_str, self.sender.sender, self.sender.xres,
            self.sender.yres)
        option += "-o {}".format(
            self.out_path) if self.out_path is not None else ''
        with open(self.rcvr_path, 'w') as wtr:
            wtr.write(self.receiver.str_repr)
        if self.receiver.rcvr == "sun":
            try:
                cmd = "rcontrib < {} {} -fo+ -ffc -M {} {}".format(
                    self.sender.path, option, self.mod_path, self._env_path)
                logger.info(cmd)
                sp.call(cmd, shell=True)
            finally:
                radutil.silent_remove(self._env_path)
                radutil.silent_remove(self.mod_path)
        else:
            cmd = "rfluxmtx < {} {} - {} {}".format(self.sender.path, option,
                                                    self.rcvr_path,
                                                    self.env_path)
            logger.info(cmd)
            sp.call(cmd, shell=True)


def prepare_surface(prims,
                    basis,
                    left=False,
                    offset=None,
                    source=None,
                    out=None):
    """."""
    assert basis is not None, 'Sampling basis cannot be None'
    #[logger.debug(radutil.put_primitive(p)) for p in prims]
    upvector = radutil.up_vector(prims)
    basis = "-" + basis if left else basis
    modifier_set = set([p['modifier'] for p in prims])
    if len(modifier_set) != 1:
        logger.warn("Primitives don't share modifier")
    src_mod = "rfluxsrf{}".format(prims[0]['modifier'])
    header = '#@rfluxmtx h={} u={}\n'.format(basis, upvector)
    if out is not None:
        header += "#@rfluxmtx o={}\n\n".format(out)
    if source is not None:
        source_line = "void glow {}\n0\n0\n4 1 1 1 0\n\n".format(src_mod)
        header += source_line
    modifiers = [p['modifier'] for p in prims]
    identifiers = [p['identifier'] for p in prims]
    for p in prims:
        if p['identifier'] in modifiers:
            p['identifier'] = 'discarded'
    for p in prims:
        p['modifier'] = src_mod
    content = ''
    if offset is not None:
        for p in prims:
            pg = p['polygon']
            offset_vec = pg.normal().scale(offset)
            moved_pts = [pt + offset_vec for pt in pg.vertices]
            p['real_args'] = radgeom.Polygon(moved_pts).to_real()
            content += radutil.put_primitive(p)
    else:
        for p in prims:
            content += radutil.put_primitive(p)
    return header + content


def genmtx_args(parser):
    parser.add_argument('-s', required=True, help='Sender object')
    parser.add_argument('-r',
                        nargs='+',
                        required=True,
                        help='Receiver objects')
    parser.add_argument('-i', help='Scene octree file path')
    parser.add_argument('-o',
                        required=True,
                        help='Output file path | directory')
    parser.add_argument('-mod', help='modifier path for sun sources')
    parser.add_argument('-env',
                        nargs='+',
                        default='',
                        help='Environment file paths')
    parser.add_argument('-rs', help='Receiver sampling basis, kf|r1|r2|....')
    parser.add_argument('-ss', help='Sender sampling basis, kf|r1|r2|....')
    parser.add_argument('-ro',
                        type=float,
                        help='Move receiver surface in normal direction')
    parser.add_argument('-so',
                        type=float,
                        help='Move sender surface in normal direction')
    parser.add_argument('-opt', help='Simulation parameters')
    parser.add_argument('-xres', type=int, help='X resolution')
    parser.add_argument('-yres', type=int, help='Y resolution')
    parser.add_argument('-c', action='store_true', help='Crop to circle?')
    parser.add_argument('-smx', help='Sky matrix file path')
    parser.add_argument('-wpths', nargs='+', help='Windows polygon paths')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    genmtx_parser = genmtx_args(parser)
    args = genmtx_parser.parse_args()
    if args.i is not None:
        env = "{} -i {}".format(' '.join(args.env), args.i)
    else:
        env = ' '.join(args.env)

    _opt = radutil.parse_opt(args.opt)

    _sender = Sender(args.s,
                     basis=args.ss,
                     offset=args.so,
                     xres=args.xres,
                     yres=args.yres,
                     c2c=args.c,
                     ray_cnt=_opt['c'])

    if args.r[0] == 'sky':
        receiver = Receiver(args.r[0], args.rs)
    elif args.r[0] == 'sun':
        receiver = Receiver(args.r[0],
                            args.rs,
                            smx_path=args.smx,
                            window_paths=args.wpths)
    else:
        rcvr_prims = []
        for receiver in args.r:
            rcvr_prims.extend(radutil.parse_primitive(receiver))
        modifiers = set([prim['modifier'] for prim in rcvr_prims])
        receivers = []
        for mod in modifiers:
            _receiver = [
                prim for prim in rcvr_prims if prim['modifier'] == mod
                and prim['type'] in ['polygon', 'ring']
            ]
            if _receiver != []:
                receivers.append(Receiver(_receiver, args.rs, offset=args.ro))
        receiver = receivers[0]
        for idx in range(1, len(receivers)):
            receiver += receivers[idx]

    Genmtx(
        sender=_sender,
        out_path=args.o,
        receiver=receiver,
        env=env,
        opt=args.opt,
    )
