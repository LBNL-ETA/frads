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
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from frads import sky
from frads import geom
from frads import parsers
from frads import raycall
from frads import utils
from frads.types import Primitive
from frads.types import Receiver
from frads.types import Sender
from frads.geom import Vector


logger: logging.Logger = logging.getLogger("frads.matrix")


def surface_as_sender(prim_list: list, basis: str, offset=None, left=None) -> Sender:
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
    prim_str = prepare_surface(
        prims=prim_list, basis=basis, offset=offset, left=left, source=None, out=None
    )
    logger.debug("Surface sender:\n%s", prim_str)
    return Sender("s", prim_str.encode(), None, None)


def view_as_sender(view: View, ray_cnt: int, xres: int, yres: int) -> Sender:
    """
    Construct a sender from a view.

    Args:
        view: a view object;
        ray_cnt: ray count;
        xres, yres: image resolution
        c2c: Set to True to trim the fisheye corner rays.

    Returns:
        A sender object

    """
    if (xres is None) or (yres is None):
        raise ValueError("Need to specify resolution")
    res_cmd = [
        "vwrays",
        *view.args(),
        "-x",
        str(xres),
        "-y",
        str(yres),
        "-d",
    ]
    logger.info("Check real image resolution: \n%s", " ".join(res_cmd))
    res_proc = sp.run(res_cmd, check=True, stdout=sp.PIPE, encoding="ascii")
    res_eval = res_proc.stdout.split()
    new_xres, new_yres = int(res_eval[1]), int(res_eval[3])
    if (new_xres != xres) and (new_yres != yres):
        logger.info("Changed resolution to %s %s", new_xres, new_yres)
    vwrays_cmd = ["vwrays", "-ff", "-x", str(new_xres), "-y", str(new_yres)]
    if ray_cnt > 1:
        vwrays_cmd.extend(["-c", str(ray_cnt), "-pj", "0.7"])
    logger.debug("Ray count is %s", ray_cnt)
    vwrays_cmd += view.args()
    logger.info("Generate view rays with: \n%s", " ".join(vwrays_cmd))
    vwrays_proc = sp.run(vwrays_cmd, check=True, stdout=sp.PIPE)
    if view.vtype == "a":
        flush_cmd = utils.get_flush_corner_rays_command(ray_cnt, xres)
        logger.info("Flushing -vta corner rays: \n%s", " ".join(flush_cmd))
        flush_proc = sp.run(
            flush_cmd, check=True, input=vwrays_proc.stdout, stdout=sp.PIPE
        )
        vrays = flush_proc.stdout
    else:
        vrays = vwrays_proc.stdout
    logger.debug("View sender:\n%s", vrays)
    return Sender("v", vrays, xres, yres)


def points_as_sender(pts_list: list, ray_cnt: Optional[int] = None) -> Sender:
    """
    Construct a sender from a list of points.

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
    grid_str = os.linesep.join([" ".join(map(str, li)) for li in pts_list]) + os.linesep
    logger.debug("Point sender:\n%s", grid_str)
    return Sender("p", grid_str.encode(), None, len(pts_list))


def sun_as_receiver(
    basis,
    smx_path: Path,
    window_normals: Optional[List[Vector]],
    full_mod: bool = False,
) -> Receiver:
    """
    Instantiate a sun receiver object.

    Args:
        basis: receiver sampling basis {kf | r1 | sc25...}
        smx_path: sky/sun matrix file path
        window_paths: window file paths
    Returns:
        A sun receiver object
    """

    # gensun = sky.Gensun(int(basis[-1]))
    if (smx_path is None) and (window_normals is None):
        str_repr, mod_str = sky.gen_sun_source_full(int(basis[-1]))
        return Receiver(str_repr, basis, modifier=mod_str)
    str_repr, mod_str, mod_str_full = sky.gen_sun_source_culled(
        int(basis[-1]), smx_path=smx_path, window_normals=window_normals
    )
    if full_mod:
        return Receiver(receiver=str_repr, basis=basis, modifier=mod_str_full)
    logger.debug("Sun receiver:\n%s", str_repr)
    logger.debug("Sun modifier:\n%s", mod_str)
    return Receiver(receiver=str_repr, basis=basis, modifier=mod_str)


def sky_as_receiver(basis: str, out) -> Receiver:
    """
    Instantiate a sky receiver object.

    Args:
        basis: receiver sampling basis {kf | r1 | sc25...}
    Returns:
        A sky receiver object
    """

    if not basis.startswith("r"):
        raise ValueError(f"Sky basis need to be Treganza/Reinhart, found {basis}")
    out.parent.mkdir(exist_ok=True)
    sky_str = f'#@rfluxmtx o="{str(out)}"\n'
    sky_str += sky.basis_glow(basis)
    logger.debug("Sky receiver:\n%s", sky_str)
    return Receiver(sky_str, basis)


def surface_as_receiver(
    prim_list: Sequence[Primitive],
    basis: str,
    out: Union[None, str, Path],
    offset=None,
    left: bool = False,
    source: str = "glow",
) -> Receiver:
    """
    Instantiate a surface receiver object.

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
    rcvr_str = prepare_surface(
        prims=prim_list, basis=basis, offset=offset, left=left, source=source, out=out
    )
    logger.debug("Surface receiver:\n%s", rcvr_str)
    return Receiver(rcvr_str, basis)


def prepare_surface(*, prims, basis, left, offset, source, out) -> str:
    """
    Prepare the sender or receiver surface, adding appropriate tags.

    Args:
        prims(list): list of primitives
        basis(str): sampling basis
        left(bool): use instead the left-hand rule
        offset(float): offset surface in its normal direction
        source(str): surface light source for receiver
        out: output path
    Returns:
        The surface sender/receiver primitive as string
    """

    if basis is None:
        raise ValueError("Sampling basis cannot be None")
    if (source is not None) and (source not in ("glow", "light")):
        raise ValueError(f"Unknown source type {source}")
    upvector = utils.up_vector(prims)
    upvector = upvector.scale(-1) if left else upvector
    upvector_str = str(upvector).replace(" ", ",")
    modifier_set = {p.modifier for p in prims}
    if len(modifier_set) != 1:
        logger.warning("Primitives don't share modifier")
    src_mod = f"rflx{prims[0].modifier}{utils.id_generator()}"
    header = f"#@rfluxmtx h={basis} u={upvector_str}\n"
    if out is not None:
        header += f'#@rfluxmtx o="{out}"\n\n'
    if source == "glow":
        source_prim = Primitive("void", source, src_mod, ("0"), (4, 1, 1, 1, 0))
        header += str(source_prim)
    elif source == "light":
        source_prim = Primitive("void", source, src_mod, ("0"), (3, 1, 1, 1))
        header += str(source_prim)
    modifiers = [p.modifier for p in prims]
    content = ""
    for prim in prims:
        if prim.identifier in modifiers:
            _identifier = "discarded"
        else:
            _identifier = prim.identifier
        _modifier = src_mod
        if offset is not None:
            poly = parsers.parse_polygon(prim.real_arg)
            offset_vec = poly.normal.scale(offset)
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            _real_args = geom.Polygon(moved_pts).to_real()
        else:
            _real_args = prim.real_arg
        new_prim = Primitive(
            _modifier, prim.ptype, _identifier, prim.str_arg, _real_args
        )
        content += str(new_prim) + "\n"
    return header + content


def rfluxmtx(
    sender: Sender,
    receiver: Receiver,
    env: Iterable[Path],
    opt: Optional[List[str]] = None,
) -> None:
    """
    Calling rfluxmtx to generate the matrices.

    Args:
        sender: Sender object
        receiver: Receiver object
        env: model environment, basically anything that's not the
            sender or receiver
        opt: option string

    Returns:
        return the stdout of the rfluxmtx run.
    """
    if None in (sender, receiver):
        raise ValueError("Sender/Receiver object is None")
    opt = [] if opt is None else opt
    _sender = None
    stdin: Optional[bytes] = sender.sender
    with tf.TemporaryDirectory() as tempd:
        receiver_path = Path(tempd, "receiver")
        with open(receiver_path, "w", encoding="ascii") as wtr:
            wtr.write(receiver.receiver)
        if sender.form == "s":
            sender_path = Path(tempd, "sender")
            with open(sender_path, "wb") as wtr:
                wtr.write(sender.sender)
            _sender = sender_path
            stdin = None
        elif sender.form == "p":
            opt.extend(["-I+", "-faa", "-y", str(sender.yres)])
        elif sender.form == "v":
            opt.extend(["-ffc", "-x", str(sender.xres), "-y", str(sender.yres), "-ld-"])
        cmd = raycall.get_rfluxmtx_command(
            receiver_path, option=opt, sender=_sender, sys_paths=env
        )
        logger.info("Running rfluxmtx with:\n%s", " ".join(cmd))
        proc = sp.run(cmd, check=True, input=stdin, stderr=sp.PIPE)
        if proc.stderr != b"":
            logger.warning(proc.stderr.decode())


def rcvr_oct(receiver, env, oct_path: Union[str, Path]) -> None:
    """
    Generate an octree of the environment and the receiver.

    Args:
        receiver: receiver object
        env: environment file paths
        oct_path: Path to write the octree to
    Returns:
        None
    """

    with tf.TemporaryDirectory() as tempd:
        receiver_path = os.path.join(tempd, "rcvr_path")
        with open(receiver_path, "w", encoding="utf-8") as wtr:
            wtr.write(receiver.receiver)
        ocmd = ["oconv", "-f", *map(str, env), receiver_path]
        logger.info("Generate octree with:\n%s", " ".join(ocmd))
        with open(oct_path, "wb") as wtr:
            sp.run(ocmd, check=True, stdout=wtr)


def rcontrib(
    sender,
    modifier: str,
    octree: Union[str, Path],
    out: Union[str, Path],
    opt: List[str],
) -> None:
    """
    Calling rcontrib to generate the matrices.

    Args:
        sender: Sender object
        modifier: modifier str listing the receivers in octree
        octree: the octree that includes the environment and the receiver
        opt: option string
        out: output path

    Returns:
        None

    """
    opt.append("-fo+")
    with tf.TemporaryDirectory() as tempd:
        modifier_path = os.path.join(tempd, "modifier")
        with open(modifier_path, "w", encoding="utf-8") as wtr:
            wtr.write(modifier)
        cmd = ["rcontrib", *opt]
        stdin = sender.sender
        if sender.form == "p":
            cmd += ["-I+", "-faf", "-y", str(sender.yres)]
        elif sender.form == "v":
            out = Path(out)
            out.mkdir(exist_ok=True)
            out = out / "%04d.hdr"
            cmd += ["-ffc", "-x", str(sender.xres), "-y", str(sender.yres)]
        cmd += ["-o", str(out), "-M", modifier_path, str(octree)]
        logger.info("Running rcontrib with:\n%s", " ".join(cmd))
        sp.run(cmd, check=True, input=stdin)
