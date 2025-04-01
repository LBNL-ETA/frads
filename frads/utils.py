"""
This module contains all utility functions used throughout frads.
"""

from io import TextIOWrapper
import logging
import re
from pathlib import Path
from random import choices
import string

import frads as fr
import numpy as np
from pyradiance import Primitive, parse_primitive, pvaluer


logger: logging.Logger = logging.getLogger("frads.utils")


def parse_rad_header(header_str: str) -> tuple:
    """Parse a Radiance matrix file header.

    Args:
        header_str: header as string
    Returns:
        A tuple contain nrow, ncol, ncomp, datatype
    Raises:
        ValueError: if any of NROWS NCOLS NCOMP FORMAT is not found.
    """
    compiled = re.compile(
        r" NROWS=(.*) | NCOLS=(.*) | NCOMP=(.*) | FORMAT=(.*) ", flags=re.X
    )
    matches = compiled.findall(header_str)
    if len(matches) != 4:
        raise ValueError("Can't find one of the header entries.")
    nrow = int([mat[0] for mat in matches if mat[0] != ""][0])
    ncol = int([mat[1] for mat in matches if mat[1] != ""][0])
    ncomp = int([mat[2] for mat in matches if mat[2] != ""][0])
    dtype = [mat[3] for mat in matches if mat[3] != ""][0].strip()
    return nrow, ncol, ncomp, dtype


def polygon_primitive(
    polygon: fr.geom.Polygon, modifier: str, identifier: str
) -> Primitive:
    """
    Generate a primitive from a polygon.
    Args:
        polygon: a Polygon object
        modifier: a Radiance primitive modifier
        identifier: a Radiance primitive identifier
    Returns:
        A Primitive object
    """
    return Primitive(modifier, "polygon", identifier, [], polygon.coordinates)


def parse_polygon(primitive: Primitive) -> fr.geom.Polygon:
    """
    Parse a primitive into a polygon.

    Args:
        primitive: a dictionary object containing a primitive
    Returns:
        A Polygon object
    """
    if primitive.ptype != "polygon":
        raise ValueError("Not a polygon: ", primitive.identifier)
    vertices = [
        np.array(primitive.fargs[i : i + 3]) for i in range(0, len(primitive.fargs), 3)
    ]
    return fr.geom.Polygon(vertices)


def array_hdr(array: np.ndarray, xres: int, yres: int, dtype: str = "d") -> bytes:
    """
    Call pvalue to generate a HDR image from a numpy array.
    Args:
        array: one-dimensional pixel values [[r1, g1, b1], [r2, g2, b2], ...]
        xres: x resolution
        yres: y resolution
        dtype: data type of the array. 'd' for double, 'f' for float
    Returns:
        HDR image in bytes
    """
    return pvaluer(array.tobytes(), inform=dtype, header=False, xres=xres, yres=yres)


def write_hdr(
    fname: str, array: np.ndarray, xres: int, yres: int, dtype: str = "d"
) -> None:
    """
    Write a array into a HDR image.
    Args:
        fname: output file name
        array: one-dimensional pixel values [[r1, g1, b1], [r2, g2, b2], ...]
        xres: x resolution
        yres: y resolution
        dtype: data type of the array. 'd' for double, 'f' for float
    Returns:
        None
    """
    with open(fname, "wb") as f:
        f.write(array_hdr(array, xres, yres, dtype=dtype))


def write_hdrs(
    array: np.ndarray, xres: int, yres: int, dtype: str = "d", outdir: str = "."
) -> None:
    """
    Write a series of HDR images to a file.
    Args:
        array: two-dimensional pixel values [[r1, g1, b1], [r2, g2, b2], ...]
            where each column of data represents a image.
        xres: x resolution
        yres: y resolution
        dtype: data type of the array. 'd' for double, 'f' for float
        outdir: output directory
    Returns:
        None
    """
    if not Path(outdir).exists():
        Path(outdir).mkdir(parents=True)
    # iterate through the columns of array
    for i in range(array.shape[1]):
        fname = f"{outdir}/hdr_{i}.hdr"
        write_hdr(fname, array[:, i], xres, yres, dtype=dtype)


def write_ep_rad_model(outpath: str, model: dict) -> None:
    """
    Write a epjson2rad model into a Radiance file.
    Args:
        outpath: output file path
        model: a model object
    """
    with open(outpath, "wb") as f:
        f.write(model["model"]["materials"]["bytes"])
        f.write(model["model"]["scene"]["bytes"])
        for window in model["model"]["windows"].values():
            f.write(window.bytes)


def unpack_primitives(file: str | Path | TextIOWrapper) -> list[Primitive]:
    """Open a file a to parse primitive."""
    if isinstance(file, TextIOWrapper):
        lines = file.read()
    else:
        with open(file, "r", encoding="ascii") as rdr:
            lines = rdr.read()
    return parse_primitive(lines)


def neutral_plastic_prim(
    mod: str, ident: str, refl: float, spec: float, rough: float
) -> Primitive:
    """
    Generate a neutral color plastic material.

    Args:
        mod: modifier to the primitive
        ident: identifier to the primitive
        refl: measured reflectance (0.0 - 1.0)
        spec: material specularity (0.0 - 1.0)
        rough: material roughness (0.0 - 1.0)

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
    """
    Generate a neutral color plastic material.

    Args:
        mod: modifier to the primitive
        ident: identifier to the primitive
        refl: measured reflectance (0.0 - 1.0)
        spec: material specularity (0.0 - 1.0)
        rough: material roughness (0.0 - 1.0)

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


def color_plastic_prim(
    mod: str,
    ident: str,
    refl: float,
    red: int,
    green: int,
    blue: int,
    specu: float,
    rough: float,
) -> Primitive:
    """Generate a colored plastic material.
    Args:
        mod: modifier to the primitive
        ident: identifier to the primitive
        refl : measured reflectance (0.0 - 1.0)
        red: green; blue (int): rgb values (0 - 255)
        specu: material specularity (0.0 - 1.0)
        rough: material roughness (0.0 - 1.0)

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


def add_manikin(
    manikin_file: str,
    manikin_name: str,
    zone: dict,
    position: list[float],
    rotation: float = 0,
) -> None:
    """Add a manikin to the scene.i
    Args:
        manikin_file: path to the manikin file
        manikin_name: name of the manikin
        zone: zone as a dictionary, must have 'model' key, we assume all scene
            data is inside the data key, and not files.
        position: position of the manikin (x, y), where x and y are 0-1
        rotation: rotation of the manikin in degree (0-360)
    Returns:
        A zone with added manikin
    Notes:
        Zone dictionary is modified in place.
    """
    zone["model"]["scene"]["bytes"] += b" "
    zone_primitives = parse_primitive(zone["model"]["scene"]["bytes"].decode())
    zone_polygons = [parse_polygon(p) for p in zone_primitives if p.ptype == "polygon"]
    xmin, xmax, ymin, ymax, zmin, _ = fr.geom.get_polygon_limits(zone_polygons)
    target = np.array(
        [xmin + (xmax - xmin) * position[0], ymin + (ymax - ymin) * position[1], zmin]
    )
    with open(manikin_file) as f:
        manikin_primitives = parse_primitive(f.read())
    non_polygon_primitives = [p for p in manikin_primitives if p.ptype != "polygon"]
    for primitive in non_polygon_primitives:
        zone["model"]["scene"]["bytes"] += primitive.bytes
    manikin_polygons = [
        parse_polygon(p) for p in manikin_primitives if p.ptype == "polygon"
    ]
    xminm, xmaxm, yminm, ymaxm, zminm, _ = fr.geom.get_polygon_limits(manikin_polygons)
    manikin_base_center = np.array([(xmaxm - xminm) / 2, (ymaxm - yminm) / 2, zminm])
    if rotation != 0:
        manikin_polygons = [
            p.rotate(manikin_base_center, np.array([0, 0, 1]), np.radians(rotation))
            for p in manikin_polygons
        ]
    move_vector = manikin_base_center - target
    moved_manikin_polygons = [polygon.move(move_vector) for polygon in manikin_polygons]
    moved_manikin = [
        polygon_primitive(polygon, primitive.modifier, primitive.identifier)
        for polygon, primitive in zip(moved_manikin_polygons, manikin_primitives)
    ]
    for primitive in moved_manikin:
        zone["model"]["scene"]["bytes"] += primitive.bytes
    manikin_rays = []
    for polygon in moved_manikin_polygons:
        manikin_rays.append([*polygon.centroid.tolist(), *polygon.normal.tolist()])
    zone["model"]["sensors"][manikin_name] = {"data": manikin_rays}


def glass_prim(
    mod: str, ident: str, tr: float, tg: float, tb: float, refrac: float = 1.52
) -> Primitive:
    """Generate a glass material.

    Args:
        mod: modifier to the primitive
        ident: identifier to the primtive
        tr: Transmissivity in red channel (0.0 - 1.0)
        tg: Transmissivity in green channel (0.0 - 1.0)
        tb: Transmissivity in blue channel (0.0 - 1.0)
        refrac: refraction index (default=1.52)
    Returns:
        material primtive (dict)

    """
    tmsv_red = tr * 1.08981
    tmsv_green = tg * 1.08981
    tmsv_blue = tb * 1.08981
    real_args = [tmsv_red, tmsv_green, tmsv_blue, refrac]
    return Primitive(mod, "glass", ident, [], real_args)


def pt_inclusion(pt: np.ndarray, polygon_pts: list[np.ndarray]) -> int:
    """Test whether a point is inside a polygon
    using winding number algorithm."""

    def isLeft(pt0, pt1, pt2):
        """Test whether a point is left to a line."""
        return (pt1[0] - pt0[0]) * (pt2[1] - pt0[1]) - (pt2[0] - pt0[0]) * (
            pt1[1] - pt0[1]
        )

    # Close the polygon for looping
    # polygon_pts.append(polygon_pts[0])
    polygon_pts = [*polygon_pts, polygon_pts[0]]
    wn = 0
    for i in range(len(polygon_pts) - 1):
        if polygon_pts[i][1] <= pt[1]:
            if polygon_pts[i + 1][1] > pt[1]:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) > 0:
                    wn += 1
        else:
            if polygon_pts[i + 1][1] <= pt[1]:
                if isLeft(polygon_pts[i], polygon_pts[i + 1], pt) < 0:
                    wn -= 1
    return wn


def gen_grid(polygon: fr.geom.Polygon, height: float, spacing: float) -> list[list[float]]:
    """Generate a grid of points for orthogonal planar surfaces.

    Args:
        polygon: a polygon object
        height: points' distance from the surface in its normal direction
        spacing: distance between the grid points
    Returns:
        List of the points as list
    """
    vertices = polygon.vertices
    plane_height = sum(i[2] for i in vertices) / len(vertices)
    imin, imax, jmin, jmax, _, _ = polygon.extreme
    xlen_spc = (imax - imin) / spacing
    ylen_spc = (jmax - jmin) / spacing
    xstart = (xlen_spc - int(xlen_spc) + 1) * spacing / 2
    ystart = (ylen_spc - int(ylen_spc) + 1) * spacing / 2
    x0 = np.arange(imin, imax, spacing) + xstart
    y0 = np.arange(jmin, jmax, spacing) + ystart
    grid_dir = polygon.normal * -1
    grid_hgt = np.array((0, 0, plane_height)) + grid_dir * height
    raw_pts = [
        np.array((round(i, 3), round(j, 3), round(grid_hgt[2], 3)))
        for i in x0
        for j in y0
    ]
    if np.array_equal(polygon.normal, np.array((0, 0, 1))):
        _grid = [p for p in raw_pts if pt_inclusion(p, vertices) > 0]
    else:
        _grid = [p for p in raw_pts if pt_inclusion(p, vertices[::-1]) > 0]
    grid = [p.tolist() + grid_dir.tolist() for p in _grid]
    return grid


def material_lib() -> dict[str, Primitive]:
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


def random_string(size: int) -> str:
    """Generate random characters."""
    chars = string.ascii_uppercase + string.digits
    return "".join(choices(chars, k=size))
