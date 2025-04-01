"""
This module contains all utility functions used throughout frads.
"""

from io import TextIOWrapper
import logging
from pathlib import Path
from random import choices
import string

import numpy as np
from pyradiance import Primitive, parse_primitive, pvaluer


logger: logging.Logger = logging.getLogger("frads.utils")


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


def random_string(size: int) -> str:
    """Generate random characters."""
    chars = string.ascii_uppercase + string.digits
    return "".join(choices(chars, k=size))


# def add_manikin(
#     manikin_file: str,
#     manikin_name: str,
#     zone: dict,
#     position: list[float],
#     rotation: float = 0,
# ) -> None:
#     """Add a manikin to the scene.i
#     Args:
#         manikin_file: path to the manikin file
#         manikin_name: name of the manikin
#         zone: zone as a dictionary, must have 'model' key, we assume all scene
#             data is inside the data key, and not files.
#         position: position of the manikin (x, y), where x and y are 0-1
#         rotation: rotation of the manikin in degree (0-360)
#     Returns:
#         A zone with added manikin
#     Notes:
#         Zone dictionary is modified in place.
#     """
#     zone["model"]["scene"]["bytes"] += b" "
#     zone_primitives = parse_primitive(zone["model"]["scene"]["bytes"].decode())
#     zone_polygons = [parse_polygon(p) for p in zone_primitives if p.ptype == "polygon"]
#     xmin, xmax, ymin, ymax, zmin, _ = fr.geom.get_polygon_limits(zone_polygons)
#     target = np.array(
#         [xmin + (xmax - xmin) * position[0], ymin + (ymax - ymin) * position[1], zmin]
#     )
#     with open(manikin_file) as f:
#         manikin_primitives = parse_primitive(f.read())
#     non_polygon_primitives = [p for p in manikin_primitives if p.ptype != "polygon"]
#     for primitive in non_polygon_primitives:
#         zone["model"]["scene"]["bytes"] += primitive.bytes
#     manikin_polygons = [
#         parse_polygon(p) for p in manikin_primitives if p.ptype == "polygon"
#     ]
#     xminm, xmaxm, yminm, ymaxm, zminm, _ = fr.geom.get_polygon_limits(manikin_polygons)
#     manikin_base_center = np.array([(xmaxm - xminm) / 2, (ymaxm - yminm) / 2, zminm])
#     if rotation != 0:
#         manikin_polygons = [
#             p.rotate(manikin_base_center, np.array([0, 0, 1]), np.radians(rotation))
#             for p in manikin_polygons
#         ]
#     move_vector = manikin_base_center - target
#     moved_manikin_polygons = [polygon.move(move_vector) for polygon in manikin_polygons]
#     moved_manikin = [
#         polygon_primitive(polygon, primitive.modifier, primitive.identifier)
#         for polygon, primitive in zip(moved_manikin_polygons, manikin_primitives)
#     ]
#     for primitive in moved_manikin:
#         zone["model"]["scene"]["bytes"] += primitive.bytes
#     manikin_rays = []
#     for polygon in moved_manikin_polygons:
#         manikin_rays.append([*polygon.centroid.tolist(), *polygon.normal.tolist()])
#     zone["model"]["sensors"][manikin_name] = {"data": manikin_rays}
