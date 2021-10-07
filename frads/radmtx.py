""" Support matrices generation.

radmtx module contains two class objects: sender and receiver, representing
the ray sender and receiver in the rfluxmtx operation. sender object is can
be instantiated as a surface, a list of points, or a view, and these are
typical forms of a sender. Similarly, a receiver object can be instantiated as
a surface, sky, or suns.
"""

from __future__ import annotations
import os
import copy
import subprocess as sp
import tempfile as tf
import logging
from frads import makesky
from frads import radgeom
from frads import radutil, util
from typing import Optional

logger = logging.getLogger('frads.radmtx')


class Sender:
    """Sender object for matrix generation with the following attributes:

    Attributes:
        form(str): types of sender, {surface(s)|view(v)|points(p)}
        sender(str): the sender object
        xres(int): sender x dimension
        yres(int): sender y dimension
    """

    def __init__(self, *, form: str, sender: bytes,
                 xres: Optional[int], yres: Optional[int]):
        """Instantiate the instance.

        Args:
            form(str): Sender as (s, v, p) for surface, view, and points;
            path(str): sender file path;
            sender(str):  content of the sender file;
            xres(int): x resolution of the image;
            yres(int): y resoluation or line count if form is pts;
        """
        self.form = form
        self.sender = sender
        self.xres = xres
        self.yres = yres
        logger.debug("Sender: %s", sender)

    @classmethod
    def as_surface(cls, *, prim_list: list, basis: str,
                   offset=None, left=None):
        """
        Construct a sender from a surface.

        Args:
            prim_list(list): a list of primitives
            basis(str): sender sampling basis
            offset(float): move the sender surface in its normal direction
            left(bool): Use left-hand rule instead for matrix generation

        Returns:
            A sender object (Sender)

        """
        prim_str = prepare_surface(prims=prim_list, basis=basis, offset=offset,
                                   left=left, source=None, out=None)
        return cls(form='s', sender=prim_str.encode(), xres=None, yres=None)

    @classmethod
    def as_view(cls, *, vu_dict: dict, ray_cnt: int, xres: int, yres: int) -> Sender:
        """
        Construct a sender from a view.

        Args:
            vu_dict: a dictionary containing view parameters;
            ray_cnt: ray count;
            xres, yres: image resolution
            c2c: Set to True to trim the fisheye corner rays.

        Returns:
            A sender object

        """
        if None in (xres, yres):
            raise ValueError("Need to specify resolution")
        vcmd = f"vwrays {radutil.opt2str(vu_dict)} -x {xres} -y {yres} -d"
        res_eval = util.spcheckout(vcmd.split()).decode().split()
        xres, yres = int(res_eval[1]), int(res_eval[3])
        logger.info("Changed resolution to %s %s", xres, yres)
        cmd = f"vwrays -ff -x {xres} -y {yres} "
        if ray_cnt > 1:
            vu_dict['c'] = ray_cnt
            vu_dict['pj'] = 0.7  # placeholder
        logger.debug("Ray count is %s", ray_cnt)
        cmd += radutil.opt2str(vu_dict)
        if vu_dict['vt'] == 'a':
            cmd += "|" + Sender.crop2circle(ray_cnt, xres)
        vrays = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE).stdout
        return cls(form='v', sender=vrays, xres=xres, yres=yres)

    @classmethod
    def as_pts(cls, *, pts_list: list, ray_cnt=1) -> Sender:
        """Construct a sender from a list of points.

        Args:
            pts_list(list): a list of list of float
            ray_cnt(int): sender ray count

        Returns:
            A sender object
        """
        if pts_list is None:
            raise ValueError("pts_list is None")
        if not all(isinstance(item, list) for item in pts_list):
            raise ValueError("All grid points has to be lists.")
        pts_list = [i for i in pts_list for _ in range(ray_cnt)]
        grid_str = os.linesep.join(
            [' '.join(map(str, li)) for li in pts_list]) + os.linesep
        return cls(form='p', sender=grid_str.encode(), xres=None, yres=len(pts_list))

    @staticmethod
    def crop2circle(ray_cnt: int, xres: int) -> str:
        """Flush the corner rays from a fisheye view

        Args:
            ray_cnt: ray count;
            xres: resolution of the square image;

        Returns:
            Command to generate cropped rays

        """
        cmd = "rcalc -if6 -of "
        cmd += f'-e "DIM:{xres};CNT:{ray_cnt}" '
        cmd += '-e "pn=(recno-1)/CNT+.5" '
        cmd += '-e "frac(x):x-floor(x)" '
        cmd += '-e "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM)" '
        cmd += '-e "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0)" '
        cmd += ' -e "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir"'
        if os.name == "posix":
            cmd = cmd.replace('"', "'")
        return cmd


class Receiver:
    """Receiver object for matrix generation."""

    def __init__(self, receiver: str, basis: str, modifier=None) -> None:
        """Instantiate the receiver object.

        Args:
            receiver(str): receiver string which can be appended to one another
            basis(str): receiver basis, usually kf, r4, r6;
            modifier(str): modifiers to the receiver objects;
        """
        self.receiver = receiver
        self.basis = basis
        self.modifier = modifier
        logger.debug("Receivers: %s", receiver)

    def __add__(self, other: Receiver) -> Receiver:
        self.receiver += '\n' + other.receiver
        return self

    @classmethod
    def as_sun(cls, *, basis, smx_path, window_normals, full_mod=False) -> Receiver:
        """Instantiate a sun receiver object.
        Args:
            basis: receiver sampling basis {kf | r1 | sc25...}
            smx_path: sky/sun matrix file path
            window_paths: window file paths
        Returns:
            A sun receiver object
        """

        gensun = makesky.Gensun(int(basis[-1]))
        if (smx_path is None) and (window_normals is None):
            str_repr = gensun.gen_full()
            return cls(receiver=str_repr, basis=basis, modifier=gensun.mod_str)
        str_repr, mod_str = gensun.gen_cull(smx_path=smx_path, window_normals=window_normals)
        if full_mod:
            return cls(receiver=str_repr, basis=basis, modifier=gensun.mod_str)
        return cls(receiver=str_repr, basis=basis, modifier=mod_str)

    @classmethod
    def as_sky(cls, basis) -> Receiver:
        """Instantiate a sky receiver object.
        Args:
            basis: receiver sampling basis {kf | r1 | sc25...}
        Returns:
            A sky receiver object
        """

        assert basis.startswith('r'), 'Sky basis need to be Treganza/Reinhart'
        sky_str = makesky.basis_glow(basis)
        logger.debug(sky_str)
        return cls(receiver=sky_str, basis=basis)

    @classmethod
    def as_surface(cls, prim_list: list, basis: str, out: str,
                   offset=None, left=False, source='glow') -> Receiver:
        """Instantiate a surface receiver object.
        Args:
            prim_list: list of primitives(dict)
            basis: receiver sampling basis {kf | r1 | sc25...}
            out: output path
            offset: offset the surface in its normal direction
            left: use instead left-hand rule for matrix generation
            source: light source for receiver object {glow|light}
        Returns:
            A surface receiver object
        """
        rcvr_str = prepare_surface(prims=prim_list, basis=basis, offset=offset,
                                   left=left, source=source, out=out)
        return cls(receiver=rcvr_str, basis=basis)


def prepare_surface(*, prims, basis, left, offset, source, out) -> str:
    """Prepare the sender or receiver surface, adding appropriate tags.
    Args:
        prims(list): list of primitives
        basis(str): sampling basis
        left(bool): use instead the left-hand rule
        offset(float): offset surface in its normal direction
        source(str): surface light source for receiver
        out: output path
    Returns:
        The receiver as string
    """

    if basis is None:
        raise ValueError('Sampling basis cannot be None')
    upvector = str(radutil.up_vector(prims)).replace(' ', ',')
    upvector = "-" + upvector if left else upvector
    modifier_set = {p.modifier for p in prims}
    if len(modifier_set) != 1:
        logger.warning("Primitives don't share modifier")
    src_mod = f"rflx{prims[0].modifier}"
    header = f'#@rfluxmtx h={basis} u={upvector}\n'
    if out is not None:
        header += f'#@rfluxmtx o="{out}"\n\n'
    if source is not None:
        source_line = f"void {source} {src_mod}\n0\n0\n4 1 1 1 0\n\n"
        header += source_line
    modifiers = [p.modifier for p in prims]
    content = ''
    for prim in prims:
        if prim.identifier in modifiers:
            _identifier = 'discarded'
        else:
            _identifier = prim.identifier
        _modifier = src_mod
        if offset is not None:
            poly = radutil.parse_polygon(prim.real_arg)
            offset_vec = poly.normal().scale(offset)
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            _real_args = radgeom.Polygon(moved_pts).to_real()
        else:
            _real_args = prim.real_arg
        new_prim = radutil.Primitive(
            _modifier, prim.ptype, _identifier, prim.str_arg, _real_args)
        content += str(new_prim) + '\n'
    return header + content


def rfluxmtx(*, sender, receiver, env, opt=None, out=None):
    """Calling rfluxmtx to generate the matrices.

    Args:
        sender: Sender object
        receiver: Receiver object
        env: model environment, basically anything that's not the
            sender or receiver
        opt: option string
        out: output path

    Returns:
        return the stdout of the command

    """
    if None in (sender, receiver):
        raise ValueError("Sender/Receiver object is None")
    opt = '' if opt is None else opt
    with tf.TemporaryDirectory() as tempd:
        receiver_path = os.path.join(tempd, 'receiver')
        with open(receiver_path, 'w') as wtr:
            wtr.write(receiver.receiver)
        if isinstance(env[0], dict):
            env_path = os.path.join(tempd, 'env')
            with open(env_path, 'w') as wtr:
                [wtr.write(str(prim)) for prim in env]
            env_paths = [env_path]
        else:
            env_paths = env
        cmd = ['rfluxmtx'] + opt.split()
        stdin = None
        if sender.form == 's':
            sender_path = os.path.join(tempd, 'sender')
            with open(sender_path, 'wb') as wtr:
                wtr.write(sender.sender)
            cmd.extend([sender_path, receiver_path])
        elif sender.form == 'p':
            cmd.extend(['-I+', '-faa', '-y', str(sender.yres), '-', receiver_path])
            stdin = sender.sender
        elif sender.form == 'v':
            cmd.extend(["-ffc", "-x", str(sender.xres), "-y", str(sender.yres), "-ld-"])
            if out is not None:
                util.mkdir_p(out)
                out = os.path.join(out, '%04d.hdr')
                cmd.extend(["-o", out])
            cmd.extend(['-', receiver_path])
            stdin = sender.sender
        cmd.extend(env_paths)
        return util.spcheckout(cmd, inp=stdin)


def rcvr_oct(receiver, env, oct_path):
    """Generate an octree of the environment and the receiver.
    Args:
        receiver: receiver object
        env: environment file paths
        oct_path: Path to write the octree to
    """

    with tf.TemporaryDirectory() as tempd:
        receiver_path = os.path.join(tempd, 'rcvr_path')
        with open(receiver_path, 'w') as wtr:
            wtr.write(receiver.receiver)
        ocmd = ['oconv', '-f'] + env + [receiver_path]
        octree = util.spcheckout(ocmd)
        with open(oct_path, 'wb') as wtr:
            wtr.write(octree)


def rcontrib(*, sender, modifier: str, octree, out, opt) -> None:
    """Calling rcontrib to generate the matrices.

    Args:
        sender: Sender object
        modifier: modifier str listing the receivers in octree
        octree: the octree that includes the environment and the receiver
        opt: option string
        out: output path

    Returns:
        None

    """
    lopt = opt.split()
    lopt.append('-fo+')
    with tf.TemporaryDirectory() as tempd:
        modifier_path = os.path.join(tempd, 'modifier')
        with open(modifier_path, 'w') as wtr:
            wtr.write(modifier)
        cmd = ['rcontrib'] + lopt
        stdin = sender.sender
        if sender.form == 'p':
            cmd += ['-I+', '-faf', '-y', str(sender.yres)]
        elif sender.form == 'v':
            util.mkdir_p(out)
            out = os.path.join(out, '%04d.hdr')
            cmd += ['-ffc', '-x', str(sender.xres), '-y', str(sender.yres)]
        cmd += ['-o', out, '-M', modifier_path, octree]
        util.spcheckout(cmd, inp=stdin)
