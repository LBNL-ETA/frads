"""
This module contains all utility functions used throughout frads.
"""
from io import TextIOWrapper
import logging
import math
from pathlib import Path
import random
import string
import subprocess as sp
from typing import Any, Dict, Optional, List, Set, Tuple, Union
import sys
sys.path.insert(0, ".")

from frads import geom, parsers
from frads.types import PaneRGB
from pyradiance import Primitive, parse_primitive


logger: logging.Logger = logging.getLogger("frads.utils")


GEOM_TYPE = ["polygon", "ring", "tube", "cone"]

MATERIAL_TYPE = ["plastic", "glass", "trans", "dielectric", "BSDF"]


TREG_BASE = [
    (90.0, 0),
    (78.0, 30),
    (66.0, 30),
    (54.0, 24),
    (42.0, 24),
    (30.0, 18),
    (18.0, 12),
    (6.0, 6),
    (0.0, 1),
]


def polygon2prim(polygon: geom.Polygon, modifier: str, identifier: str) -> Primitive:
    """Generate a primitive from a polygon."""
    return Primitive(modifier, "polygon", identifier, [""], polygon.to_real())


def unpack_idf(path: str) -> dict:
    """Read and parse and idf files."""
    with open(path, "r") as rdr:
        return parsers.parse_idf(rdr.read())


def frange_inc(start, stop, step):
    """Generate increasing non-integer range."""
    r = start
    while r < stop:
        yield r
        r += step


def square2disk(in_square_a: float, in_square_b: float) -> tuple:
    """Shirley-Chiu square to disk mapping.
    Args:
        in_square_a: [-1, 1]
        in_square_b: [-1, 1]
    """
    if in_square_a + in_square_b > 0:
        if in_square_a > in_square_b:
            in_square_rgn = 0
        else:
            in_square_rgn = 1
    else:
        if in_square_b > in_square_a:
            in_square_rgn = 2
        else:
            in_square_rgn = 3
    out_disk_r = [in_square_a, in_square_b, -in_square_a, -in_square_b][in_square_rgn]
    if in_square_b * in_square_b > 0:
        phi_select_4 = 6 - in_square_a / in_square_b
    else:
        phi_select_4 = 0
    phi_select = [
        in_square_b / in_square_a,
        2 - in_square_a / in_square_b,
        4 + in_square_b / in_square_a,
        phi_select_4,
    ]
    out_disk_phi = math.pi / 4 * phi_select[in_square_rgn]
    out_disk_x = out_disk_r * math.cos(out_disk_phi)
    out_disk_y = out_disk_r * math.sin(out_disk_phi)
    return out_disk_x, out_disk_y, out_disk_r, out_disk_phi


def id_generator(size: int = 3, chars: Optional[str] = None) -> str:
    """Generate random characters."""
    if chars is None:
        chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(size))


def unpack_primitives(file: Union[str, Path, TextIOWrapper]) -> List[Primitive]:
    """Open a file a to parse primitive."""
    if isinstance(file, TextIOWrapper):
        lines = file.read()
    else:
        with open(file, "r", encoding="ascii") as rdr:
            lines = rdr.read()
    return parse_primitive(lines)


def primitive_normal(primitive_paths: List[str]) -> Set[geom.Vector]:
    """Return a set of normal vectors given a list of primitive paths."""
    _primitives: List[Primitive] = []
    _normals: List[geom.Vector]
    for path in primitive_paths:
        _primitives.extend(unpack_primitives(path))
    _normals = [parsers.parse_polygon(prim.fargs).normal for prim in _primitives]
    return set(_normals)


def samp_dir(primlist: list) -> geom.Vector:
    """Calculate the primitives' average sampling
    direction weighted by area."""
    primlist = [p for p in primlist if p.ptype in ("polygon", "ring")]
    normal_area = geom.Vector()
    for prim in primlist:
        polygon = parsers.parse_polygon(prim.fargs)
        normal_area += polygon.normal.scale(polygon.area)
    sdir = normal_area.scale(1.0 / len(primlist))
    sdir = sdir.normalize()
    return sdir


def up_vector(primitives: list) -> geom.Vector:
    """Define the up vector given primitives.

    Args:
        primitives: list of dictionary (primitives)

    Returns:
        returns a str as x,y,z

    """
    norm_dir = samp_dir(primitives)
    if norm_dir != geom.Vector(0, 0, 1):
        upvect = norm_dir.cross(geom.Vector(0, 0, 1).cross(norm_dir)).normalize()
    else:
        upvect = geom.Vector(0, 1, 0)
    return upvect


def neutral_plastic_prim(
    mod: str, ident: str, refl: float, spec: float, rough: float
) -> Primitive:
    """Generate a neutral color plastic material.
    Args:
        mod(str): modifier to the primitive
        ident(str): identifier to the primitive
        refl (float): measured reflectance (0.0 - 1.0)
        specu (float): material specularity (0.0 - 1.0)
        rough (float): material roughness (0.0 - 1.0)

    Returns:
        A material primtive
    """
    err_msg = "reflectance, speculariy, and roughness have to be 0-1"
    assert all(0 <= i <= 1 for i in [spec, refl, rough]), err_msg
    real_args = [refl, refl, refl, spec, rough]
    return Primitive(mod, "plastic", ident, [], real_args)


def neutral_trans_prim(
    mod: str, ident: str, trans: float, refl: float, spec: float, rough: float
) -> Primitive:
    """Generate a neutral color plastic material.
    Args:
        mod(str): modifier to the primitive
        ident(str): identifier to the primitive
        refl (float): measured reflectance (0.0 - 1.0)
        specu (float): material specularity (0.0 - 1.0)
        rough (float): material roughness (0.0 - 1.0)

    Returns:
        A material primtive
    """
    color = trans + refl
    t_diff = trans / color
    tspec = 0
    err_msg = "reflectance, speculariy, and roughness have to be 0-1"
    assert all(0 <= i <= 1 for i in [spec, refl, rough]), err_msg
    real_args = [color, color, color, spec, rough, t_diff, tspec]
    return Primitive(mod, "trans", ident, [], real_args)


def color_plastic_prim(mod, ident, refl, red, green, blue, specu, rough) -> Primitive:
    """Generate a colored plastic material.
    Args:
        mod(str): modifier to the primitive
        ident(str): identifier to the primitive
        refl (float): measured reflectance (0.0 - 1.0)
        red; green; blue (int): rgb values (0 - 255)
        specu (float): material specularity (0.0 - 1.0)
        rough (float): material roughness (0.0 - 1.0)

    Returns:
        A material primtive
    """
    err_msg = "reflectance, speculariy, and roughness have to be 0-1"
    assert all(0 <= i <= 1 for i in [specu, refl, rough]), err_msg
    red_eff = 0.3
    green_eff = 0.59
    blue_eff = 0.11
    weighted = red * red_eff + green * green_eff + blue * blue_eff
    matr = round(red / weighted * refl, 3)
    matg = round(green / weighted * refl, 3)
    matb = round(blue / weighted * refl, 3)
    real_args = [matr, matg, matb, specu, rough]
    return Primitive(mod, "plastic", ident, [], real_args)


def glass_prim(
    mod, ident, tr: float, tg: float, tb: float, refrac: float = 1.52
) -> Primitive:
    """Generate a glass material.

    Args:
        mod (str): modifier to the primitive
        ident (str): identifier to the primtive
        tr, tg, tb (float): transmmisivity in each channel (0.0 - 1.0)
        refrac (float): refraction index (default=1.52)
    Returns:
        material primtive (dict)

    """
    tmsv_red = tr * 1.08981 
    tmsv_green = tg * 1.08981
    tmsv_blue = tb * 1.08981
    real_args = [tmsv_red, tmsv_green, tmsv_blue, refrac]
    return Primitive(mod, "glass", ident, [], real_args)


def bsdf_prim(
    mod,
    ident,
    xmlpath,
    upvec,
    pe: bool = False,
    thickness: float = 0.0,
    xform=None,
    real_args: list = [""],
) -> Primitive:
    """Create a BSDF primtive."""
    str_args = [xmlpath, str(upvec)]
    str_args_count = 5
    if pe:
        _type = "aBSDF"
    else:
        str_args_count += 1
        str_args = [str(thickness), *str_args]
        _type = "BSDF"
    if xform is not None:
        str_args_count += len(xform.split())
        str_args.extend(*xform.split())
    else:
        str_args.append(".")
    return Primitive(mod, _type, ident, str_args, real_args)


def opt2list(opt: dict) -> List[str]:
    """Convert option dictionary to list.

    Key: str
    Value: str | float | int | bool | list

    Args:
        opt: option dictionary
    Returns:
        A list of strings
    """
    out = []
    for key, value in opt.items():
        if isinstance(value, str):
            if key == "vf":
                out.extend(["-" + key, value])
            else:
                out.append(f"-{key}{value}")
        elif isinstance(value, bool):
            if value:
                out.append(f"-{key}+")
            else:
                out.append(f"-{key}-")
        elif isinstance(value, (int, float)):
            out.extend(["-" + key, str(value)])
        elif isinstance(value, list):
            out.extend(["-" + key, *map(str, value)])
    return out


def calc_reinsrc_dir(
    mf: int, x1: float = 0.5, x2: float = 0.5
) -> Tuple[List[geom.Vector], List[float]]:
    """
    Calculate Reinhart/Treganza sampling directions.
    Direct translation of Radiance reinsrc.cal file.

    Args:
        mf(int): multiplication factor.
        x1(float, optional): bin position 1
        x2(float, optional): bin position 2
    Returns:
        A list of geom.Vector
        A list of solid angle associated with each vector
    """

    def rnaz(r):
        """."""
        if r > (mf * 7 - 0.5):
            return 1
        return mf * TNAZ[int(math.floor((r + 0.5) / mf))]

    def raccum(r):
        """."""
        if r > 0.5:
            return rnaz(r - 1) + raccum(r - 1)
        return 0

    def rfindrow(r, rem):
        """."""
        if (rem - rnaz(r)) > 0.5:
            return rfindrow(r + 1, rem - rnaz(r))
        return r

    TNAZ = [30, 30, 24, 24, 18, 12, 6]
    rowMax = 7 * mf + 1
    runlen = 144 * mf**2 + 3
    rmax = raccum(rowMax)
    alpha = 90 / (mf * 7 + 0.5)
    dvecs = []
    omegas = []
    for rbin in range(1, runlen):
        rrow = rowMax - 1 if rbin > (rmax - 0.5) else rfindrow(0, rbin)
        rcol = rbin - raccum(rrow) - 1
        razi_width = 2 * math.pi / rnaz(rrow)
        rah = alpha * math.pi / 180
        razi = (rcol + x2 - 0.5) * razi_width if rbin > 0.5 else 2 * math.pi * x2
        ralt = (rrow + x1) * rah if rbin > 0.5 else math.asin(-x1)
        cos_alt = math.cos(ralt)
        if rmax - 0.5 > rbin:
            romega = razi_width * (math.sin(rah * (rrow + 1)) - math.sin(rah * rrow))
        else:
            romega = 2 * math.pi * (1 - math.cos(rah / 2))
        dx = math.sin(razi) * cos_alt
        dy = math.cos(razi) * cos_alt
        dz = math.sin(ralt)
        dvecs.append(geom.Vector(dx, dy, dz))
        omegas.append(romega)
    return dvecs, omegas


def pt_inclusion(pt: geom.Vector, polygon_pts: List[geom.Vector]) -> int:
    """Test whether a point is inside a polygon
    using winding number algorithm."""

    def isLeft(pt0, pt1, pt2):
        """Test whether a point is left to a line."""
        return (pt1.x - pt0.x) * (pt2.y - pt0.y) - (pt2.x - pt0.x) * (pt1.y - pt0.y)

    # Close the polygon for looping
    # polygon_pts.append(polygon_pts[0])
    polygon_pts = [*polygon_pts, polygon_pts[0]]
    wn = 0
    for i in range(len(polygon_pts) - 1):
        if polygon_pts[i].y <= pt.y:
            if polygon_pts[i + 1].y > pt.y:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) > 0:
                    wn += 1
        else:
            if polygon_pts[i + 1].y <= pt.y:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) < 0:
                    wn -= 1
    return wn


def gen_grid(polygon: geom.Polygon, height: float, spacing: float) -> list:
    """Generate a grid of points for orthogonal planar surfaces.

    Args:
        polygon: a polygon object
        height: points' distance from the surface in its normal direction
        spacing: distance between the grid points
    Returns:
        List of the points as list
    """
    vertices = polygon.vertices
    plane_height = sum(i.z for i in vertices) / len(vertices)
    imin, imax, jmin, jmax, _, _ = polygon.extreme
    xlen_spc = (imax - imin) / spacing
    ylen_spc = (jmax - jmin) / spacing
    xstart = ((xlen_spc - int(xlen_spc) + 1)) * spacing / 2
    ystart = ((ylen_spc - int(ylen_spc) + 1)) * spacing / 2
    x0 = [x + xstart for x in frange_inc(imin, imax, spacing)]
    y0 = [x + ystart for x in frange_inc(jmin, jmax, spacing)]
    grid_dir = polygon.normal.reverse()
    grid_hgt = geom.Vector(0, 0, plane_height) + grid_dir.scale(height)
    raw_pts = [
        geom.Vector(round(i, 3), round(j, 3), round(grid_hgt.z, 3))
        for i in x0
        for j in y0
    ]
    scale_factor = 1 - 0.3 / (imax - imin)  # scale boundary down .3 meter
    _polygon = polygon.scale(
        geom.Vector(scale_factor, scale_factor, 0), polygon.centroid
    )
    _vertices = _polygon.vertices
    if polygon.normal == geom.Vector(0, 0, 1):
        # pt_incls = pt_inclusion(_vertices)
        _grid = [p for p in raw_pts if pt_inclusion(p, _vertices) > 0]
    else:
        # pt_incls = pt_inclusion(_vertices[::-1])
        _grid = [p for p in raw_pts if pt_inclusion(p, _vertices[::-1]) > 0]
    # _grid = [p for p in raw_pts if pt_incls.test_inside(p) > 0]
    grid = [p.to_list() + grid_dir.to_list() for p in _grid]
    return grid


def material_lib() -> Dict[str, Any]:
    """Generate a list of generic material primitives."""
    tmis = 0.6 * 1.08981
    return {
        "neutral_lambertian_0.2": neutral_plastic_prim(
            "void", "neutral_lambertian_0.2", 0.2, 0, 0
        ),
        "neutral_lambertian_0.5": neutral_plastic_prim(
            "void", "neutral_lambertian_0.5", 0.5, 0, 0
        ),
        "neutral_lambertian_0.7": neutral_plastic_prim(
            "void", "neutral_lambertian_0.7", 0.7, 0, 0
        ),
        "glass_60": glass_prim("void", "glass_60", tmis, tmis, tmis),
    }


def gen_blinds(depth, width, height, spacing, angle, curve, movedown) -> str:
    """Generate genblinds command for genBSDF."""
    nslats = int(round(height / spacing, 0))
    slat_cmd = "!genblinds blindmaterial blinds "
    slat_cmd += f"{depth} {width} {height} {nslats} {angle} {curve}"
    slat_cmd += "| xform -rz -90 -rx -90 -t "
    slat_cmd += f"{-width/2} {-height/2} {-movedown}\n"
    return slat_cmd


def get_glazing_primitive(panes: List[PaneRGB]) -> Primitive:
    """Generate a BRTDfunc to represent a glazing system."""
    if len(panes) > 2:
        raise ValueError("Only double pane supported")
    name = "+".join([pane.measured_data.name for pane in panes])
    if len(panes) == 1:
        str_arg = [
            "sr_clear_r",
            "sr_clear_g",
            "sr_clear_b",
            "st_clear_r",
            "st_clear_g",
            "st_clear_b",
            "0",
            "0",
            "0",
            "glaze1.cal",
        ]
        coated_real = 1 if panes[0].measured_data.coated_side == "front" else -1
        real_arg = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            coated_real,
            *[round(i, 3) for i in panes[0].glass_rgb],
            *[round(i, 3) for i in panes[0].coated_rgb],
            *[round(i, 3) for i in panes[0].trans_rgb],
        ]
    else:
        s12t_r, s12t_g, s12t_b = panes[0].trans_rgb
        s34t_r, s34t_g, s34t_b = panes[1].trans_rgb
        if panes[0].measured_data.coated_side == "back":
            s2r_r, s2r_g, s2r_b = panes[0].coated_rgb
            s1r_r, s1r_g, s1r_b = panes[0].glass_rgb
        else:  # front or neither side coated
            s2r_r, s2r_g, s2r_b = panes[0].glass_rgb
            s1r_r, s1r_g, s1r_b = panes[0].coated_rgb
        if panes[1].measured_data.coated_side == "back":
            s4r_r, s4r_g, s4r_b = panes[1].coated_rgb
            s3r_r, s3r_g, s3r_b = panes[1].glass_rgb
        else:  # front or neither side coated
            s4r_r, s4r_g, s4r_b = panes[1].glass_rgb
            s3r_r, s3r_g, s3r_b = panes[1].coated_rgb
        str_arg = [
            (
                f"if(Rdot,cr(fr({s4r_r:.3f}),ft({s34t_r:.3f}),fr({s2r_r:.3f})),"
                f"cr(fr({s1r_r:.3f}),ft({s12t_r:.3f}),fr({s3r_r:.3f})))"
            ),
            (
                f"if(Rdot,cr(fr({s4r_g:.3f}),ft({s34t_g:.3f}),fr({s2r_g:.3f})),"
                f"cr(fr({s1r_g:.3f}),ft({s12t_g:.3f}),fr({s3r_g:.3f})))"
            ),
            (
                f"if(Rdot,cr(fr({s4r_b:.3f}),ft({s34t_b:.3f}),fr({s2r_b:.3f})),"
                f"cr(fr({s1r_b:.3f}),ft({s12t_b:.3f}),fr({s3r_b:.3f})))"
            ),
            f"ft({s34t_r:.3f})*ft({s12t_r:.3f})",
            f"ft({s34t_g:.3f})*ft({s12t_g:.3f})",
            f"ft({s34t_b:.3f})*ft({s12t_b:.3f})",
            "0",
            "0",
            "0",
            "glaze2.cal",
        ]
        real_arg = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return Primitive("void", "BRTDfunc", name, str_arg, real_arg)


def batch_process(
    commands: List[List[str]],
    inputs: Optional[List[bytes]] = None,
    opaths: Optional[List[Path]] = None,
    nproc: Optional[int] = None,
) -> None:
    """Run commands in batches.

    Use subprocess.Popen to run commands.

    Args:
        commands: commands as a list of strings.
        inputs: list of standard input to the commands.
        opaths: list of paths to write standard output to.
        nproc: number of commands to run in parallel at a time.
    Returns:
        None
    """
    nproc = 1 if nproc is None else nproc
    command_groups = [commands[i : i + nproc] for i in range(0, len(commands), nproc)]
    stdin_groups: List[List[Any]] = [[None] * len(ele) for ele in command_groups]
    if inputs:
        if len(inputs) != len(commands):
            raise ValueError("Number of stdins not equal number of commands.")
        stdin_groups = [inputs[i : i + nproc] for i in range(0, len(inputs), nproc)]
    if opaths:
        if len(opaths) != len(commands):
            raise ValueError("Number of opaths not equal number of commands.")
        opath_groups = [opaths[i : i + nproc] for i in range(0, len(opaths), nproc)]
        for igrp, cgrp, ogrp in zip(stdin_groups, command_groups, opath_groups):
            processes = []
            fds = []
            for command, opath in zip(cgrp, ogrp):
                fd = open(opath, "wb")
                processes.append(sp.Popen(command, stdin=sp.PIPE, stdout=fd))
                fds.append(fd)
            for proc, sin in zip(processes, igrp):
                if (proc.stdin is not None) and (sin is not None):
                    proc.stdin.write(sin)
            for p, f in zip(processes, fds):
                p.wait()
                f.close()
    else:
        for igrp, cgrp in zip(stdin_groups, command_groups):
            processes = [sp.Popen(command, stdin=sp.PIPE) for command in cgrp]
            for proc, sin in zip(processes, igrp):
                if (proc.stdin is not None) and (sin is not None):
                    proc.stdin.write(sin)
            for proc in processes:
                proc.wait()
