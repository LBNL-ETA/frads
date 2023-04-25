"""
This module contains routines to generate sender and receiver objects, generate
matrices by calling either rfluxmtx or rcontrib.
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile as tf
from typing import List, Optional, Union, Sequence

from frads import geom, parsers, sky, utils
import numpy as np
import pyradiance as pr


logger: logging.Logger = logging.getLogger("frads.matrix")


@dataclass(frozen=True)
class Sender:
    """Sender object for matrix generation.

    Attributes:
        form: types of sender, {surface(s)|view(v)|points(p)}
        sender: the sender string
        xres: sender x dimension
        yres: sender y dimension
    """

    form: str
    sender: bytes
    xres: Optional[int]
    yres: Optional[int]


@dataclass
class Receiver:
    """Receiver object for matrix generation.

    Attributes:
        receiver: receiver string which can be appended to one another
        basis: receiver basis, usually kf, r4, r6;
        modifier: modifiers to the receiver objects;
    """

    receiver: str
    basis: str
    modifier: str = ""

    def __add__(self, other) -> "Receiver":
        return Receiver(
            self.receiver + "\n" + other.receiver, self.basis, self.modifier
        )


def surface_as_sender(prim_list: list, basis: str, offset=None, left=None) -> Sender:
    """
    Construct a sender from a surface.

    Args:
        prim_list: a list of primitives
        basis: sender sampling basis
        offset: move the sender surface in its normal direction
        left: Use left-hand rule instead for matrix generation

    Returns:
        A sender object (Sender)

    """
    prim_str = prepare_surface(
        prims=prim_list, basis=basis, offset=offset, left=left, source=None, out=None
    )
    logger.debug("Surface sender:\n%s", prim_str)
    return Sender("s", prim_str.encode(), None, None)


def view_as_sender(view, ray_cnt: int, xres: int, yres: int) -> Sender:
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
    res_eval = pr.vwrays(
        view=view.args(), xres=xres, yres=yres, dimensions=True
    ).split()
    new_xres, new_yres = int(res_eval[1]), int(res_eval[3])
    if (new_xres != xres) and (new_yres != yres):
        logger.info("Changed resolution to %s %s", new_xres, new_yres)
    vwrays_proc = pr.vwrays(
        view=view.args(),
        outform="f",
        xres=new_xres,
        yres=new_yres,
        ray_count=ray_cnt,
        pixel_jitter=0.7,
    )
    if view.vtype == "a":
        ray_flush_exp = (
            f"DIM:{xres};CNT:{ray_cnt};"
            "pn=(recno-1)/CNT+.5;"
            "frac(x):x-floor(x);"
            "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM);"
            "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0);"
            "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir;"
        )
        flushed_rays = pr.rcalc(vwrays_proc, inform='f', incount=6, outform="f", expr=ray_flush_exp)
        vrays = flushed_rays
    else:
        vrays = vwrays_proc
    logger.debug("View sender:\n%s", vrays)
    return Sender("v", vrays, xres, yres)


def load_matrix(file: Union[bytes, str, Path], dtype: str = "float"):
    """
    Load a Radiance matrix file into numpy array.

    Args:
        file: a file path

    Returns:
        A numpy array
    """
    npdtype = np.double if dtype.startswith("d") else np.single
    mtx = pr.rmtxop(file, outform=dtype[0].lower())
    nrows, ncols, ncomp, _ = parsers.parse_rad_header(pr.getinfo(mtx).decode())
    return np.frombuffer(pr.getinfo(mtx, strip_header=True), dtype=npdtype).reshape(
        nrows, ncols, ncomp
    )


def load_image_matrix(file, outform="d") -> np.ndarray:
    """
    Load a Radiance HDR image into numpy array.
    Args:
        file: a file path
    Returns:
        A numpy array
    """
    xres, yres = pr.get_image_dimensions(file)
    pix = pr.pvalue(file, outform=outform)
    return np.frombuffer(pix, np.double).reshape(xres, yres, 3)


def multiply_rgb(*mtx: np.ndarray, weights=None):
    """Multiply matrices as numpy ndarray."""
    resr = np.linalg.multi_dot([m[:, :, 0] for m in mtx])
    resg = np.linalg.multi_dot([m[:, :, 1] for m in mtx])
    resb = np.linalg.multi_dot([m[:, :, 2] for m in mtx])
    if weights:
        if len(weights) != 3:
            raise ValueError("Weights should have 3 values")
        return resr * weights[0] + resg * weights[1] + resb * weights[2]
    return np.array((resr, resg, resb))


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
    window_normals: Optional[List[geom.Vector]],
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
    prim_list: List[pr.Primitive],
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
        source_prim = pr.Primitive("void", source, src_mod, ("0"), (1, 1, 1, 0))
        header += str(source_prim)
    elif source == "light":
        source_prim = pr.Primitive("void", source, src_mod, ("0"), (1, 1, 1))
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
            poly = parsers.parse_polygon(prim.fargs)
            offset_vec = poly.normal.scale(offset)
            moved_pts = [pt + offset_vec for pt in poly.vertices]
            _real_args = geom.Polygon(moved_pts).to_real()
        else:
            _real_args = prim.fargs
        new_prim = pr.Primitive(
            _modifier, prim.ptype, _identifier, prim.sargs, _real_args
        )
        content += str(new_prim) + "\n"
    return header + content


def rfluxmtx(
    sender: Sender,
    receiver: Receiver,
    env: Sequence[Path],
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
    rays = None
    surface = None
    with tf.TemporaryDirectory() as tempd:
        receiver_path = Path(tempd, "receiver")
        with open(receiver_path, "w", encoding="ascii") as wtr:
            wtr.write(receiver.receiver)
        if sender.form == "s":
            sender_path = Path(tempd, "sender")
            with open(sender_path, "wb") as wtr:
                wtr.write(sender.sender)
            surface = sender_path
        elif sender.form == "p":
            opt.extend(["-I+", "-faa", "-y", str(sender.yres)])
            rays = sender.sender
        elif sender.form == "v":
            opt.extend(["-ffc", "-x", str(sender.xres), "-y", str(sender.yres), "-ld-"])
            rays = sender.sender
        pr.rfluxmtx(receiver_path, surface=surface, rays=rays, params=opt, scene=env)


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
        # ocmd = ["oconv", "-f", *map(str, env), receiver_path]
        # logger.info("Generate octree with:\n%s", " ".join(ocmd))
        with open(oct_path, "wb") as wtr:
            # sp.run(ocmd, check=True, stdout=wtr)
            wtr.write(pr.oconv(*map(str, env), receiver_path, frozen=True))


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
    xres = None
    yres = None
    inform = None
    outform = None
    with tf.TemporaryDirectory() as tempd:
        modifier_path = os.path.join(tempd, "modifier")
        with open(modifier_path, "w", encoding="utf-8") as wtr:
            wtr.write(modifier)
        if sender.form == "p":
            opt += ["-I+"]
            inform = "a"
            outform = "f"
            yres = sender.yres
        elif sender.form == "v":
            out = Path(out)
            out.mkdir(exist_ok=True)
            out = out / "%04d.hdr"
            inform = "f"
            outform = "c"
            xres = sender.xres
            yres = sender.yres
            # cmd += ["-ffc", "-x", str(sender.xres), "-y", str(sender.yres)]
        mod = pr.RcModifier()
        mod.modifier_path = modifier_path
        mod.xres = xres
        mod.yres = yres
        mod.output = str(out)
        pr.rcontrib(
            sender.sender,
            str(octree),
            [mod],
            inform=inform,
            outform=outform,
            params=opt,
        )
