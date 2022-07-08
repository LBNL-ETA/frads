"""
This module contains routines to generate sender and receiver objects, generate
matrices by calling either rfluxmtx or rcontrib.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
import subprocess as sp
import tempfile as tf
from typing import Optional
from typing import Sequence
from typing import Union

from frads import sky
from frads import geom
from frads import utils
from frads import parsers
from frads.types import Primitive
from frads.types import Receiver
from frads.types import Sender


logger = logging.getLogger('frads.matrix')


def surface_as_sender(prim_list: list, basis: str, offset=None, left=None):
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
    prim_str = prepare_surface(prims=prim_list, basis=basis, offset=offset, left=left, source=None, out=None)
    return Sender('s', prim_str.encode(), None, None)


def view_as_sender(vu_dict: dict, ray_cnt: int, xres: int, yres: int) -> Sender:
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
    if (xres is None) or (yres is None):
        raise ValueError("Need to specify resolution")
    res_cmd = ["vwrays", *(utils.opt2list(vu_dict)), "-x", str(xres), "-y", str(yres), "-d"]
    res_proc = sp.run(res_cmd, check=True, stdout=sp.PIPE, encoding='ascii')
    res_eval = res_proc.stdout.split()
    new_xres, new_yres = int(res_eval[1]), int(res_eval[3])
    if (new_xres != xres) and (new_yres != yres):
        logger.info("Changed resolution to %s %s", new_xres, new_yres)
    vwrays_cmd = ["vwrays", "-ff", "-x", str(new_xres), "-y", str(new_yres)]
    if ray_cnt > 1:
        vu_dict['c'] = ray_cnt
        vu_dict['pj'] = 0.7  # placeholder
    logger.debug("Ray count is %s", ray_cnt)
    vwrays_cmd += utils.opt2list(vu_dict)
    vwrays_proc = sp.run(vwrays_cmd, check=True, stdout=sp.PIPE)
    if vu_dict['vt'] == 'a':
        flush_cmd = utils.flush_corner_rays_cmd(ray_cnt, xres)
        flush_proc = sp.run(flush_cmd, input=vwrays_proc.stdout, stdout=sp.PIPE)
        vrays = flush_proc.stdout
    else:
        vrays = vwrays_proc.stdout
    return Sender('v', vrays, xres, yres)


def points_as_sender(pts_list: list, ray_cnt: Optional[int]=None) -> Sender:
    """Construct a sender from a list of points.

    Args:
        pts_list(list): a list of list of float
        ray_cnt(int): sender ray count

    Returns:
        A sender object
    """
    ray_cnt = 1 if ray_cnt is None else ray_cnt
    if pts_list is None:
        raise ValueError("pts_list is None")
    if not all(isinstance(item, list) for item in pts_list):
        raise ValueError("All grid points has to be lists.")
    pts_list = [i for i in pts_list for _ in range(ray_cnt)]
    grid_str = os.linesep.join(
        [' '.join(map(str, li)) for li in pts_list]) + os.linesep
    return Sender('p', grid_str.encode(), None, len(pts_list))


def sun_as_receiver(basis, smx_path, window_normals, full_mod=False) -> Receiver:
    """Instantiate a sun receiver object.
    Args:
        basis: receiver sampling basis {kf | r1 | sc25...}
        smx_path: sky/sun matrix file path
        window_paths: window file paths
    Returns:
        A sun receiver object
    """

    gensun = sky.Gensun(int(basis[-1]))
    if (smx_path is None) and (window_normals is None):
        str_repr = gensun.gen_full()
        return Receiver(str_repr, basis, modifier=gensun.mod_str)
    str_repr, mod_str = gensun.gen_cull(smx_path=smx_path, window_normals=window_normals)
    if full_mod:
        return Receiver(receiver=str_repr, basis=basis, modifier=gensun.mod_str)
    return Receiver(receiver=str_repr, basis=basis, modifier=mod_str)


def sky_as_receiver(basis) -> Receiver:
    """Instantiate a sky receiver object.
    Args:
        basis: receiver sampling basis {kf | r1 | sc25...}
    Returns:
        A sky receiver object
    """

    if not basis.startswith('r'):
        raise ValueError(f'Sky basis need to be Treganza/Reinhart, found {basis}')
    sky_str = sky.basis_glow(basis)
    logger.debug(sky_str)
    return Receiver(sky_str, basis)


def surface_as_receiver(prim_list: Sequence[Primitive], basis: str, out: Union[None, str, Path],
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
    return Receiver(rcvr_str, basis)


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
    upvector = str(utils.up_vector(prims)).replace(' ', ',')
    upvector = "-" + upvector if left else upvector
    modifier_set = {p.modifier for p in prims}
    if len(modifier_set) != 1:
        logger.warning("Primitives don't share modifier")
    src_mod = f"rflx{prims[0].modifier}"
    src_mod += utils.id_generator()
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
            poly = parsers.parse_polygon(prim.real_arg)
            offset_vec = poly.normal().scale(offset)
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            _real_args = geom.Polygon(moved_pts).to_real()
        else:
            _real_args = prim.real_arg
        new_prim = Primitive(
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
                out = Path(out)
                out.mkdir(exist_ok=True)
                out = out / '%04d.hdr'
                cmd.extend(["-o", str(out)])
            cmd.extend(['-', receiver_path])
            stdin = sender.sender
        cmd.extend(env_paths)
        return utils.spcheckout(cmd, inp=stdin)


def rcvr_oct(receiver, env, oct_path: Union[str, Path]):
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
        octree = utils.spcheckout(ocmd)
        with open(oct_path, 'wb') as wtr:
            wtr.write(octree)


def rcontrib(*, sender, modifier: str, octree: Union[str, Path], out, opt) -> None:
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
            out = Path(out)
            out.mkdir(exist_ok=True)
            out = out / '%04d.hdr'
            cmd += ['-ffc', '-x', str(sender.xres), '-y', str(sender.yres)]
        cmd += ['-o', out, '-M', modifier_path, str(octree)]
        sp.run(cmd, check=True, input=stdin)
