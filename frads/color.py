"""
This module contains all color and spectral-related functionalities.
"""
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from frads import color_data
from frads.types import ColorPrimaries


def get_interpolated_cie_xyz(
    inp_wvl: Sequence[Union[float, int]], observer: str
) -> List[tuple]:
    """Load CIE tristimulus data according to input wavelength.
    Also load melanopic action spectra data as well.

    Args:
        inp_wvl: a list of input wavelength in nm
        observer: 2° or 10° observer for the colar matching function.
    Returns:
        CIE-x
        CIE-y
        CIE-z
        Melanopic action spectra
        Index of input wavelength corresponding to available tristimulus data
    """

    if observer.startswith("2"):
        cie_xyz = color_data.CIE_XYZ_2
    elif observer.startswith("10"):
        cie_xyz = color_data.CIE_XYZ_10
    common_wvl = sorted(set(cie_xyz).intersection(inp_wvl))
    return [cie_xyz[wvl] for wvl in common_wvl]


def get_interpolated_mlnp(
    inp_wvl: Sequence[Union[float, int]],
) -> List[float]:
    """Load CIE tristimulus data according to input wavelength.
    Also load melanopic action spectra data as well.

    Args:
        inp_wvl: a list of input wavelength in nm
    Returns:
        Melanopic action spectra
    """
    mlnp = color_data.CIE_MLNP
    common_wvl = sorted(set(mlnp).intersection(inp_wvl))
    return [mlnp[wvl] for wvl in common_wvl]


def get_conversion_matrix(prims: ColorPrimaries, reverse: bool = False) -> tuple:
    """
    Get CIE conversion matrix based on color primaries.

    Args:
        prims: Color space color primaries
        reverse: get RGB to XYZ conversion matrix instead
    Returns:
        The conversion matrix coefficients in a 1-dimensional tuple.
    """

    yw_inv = 1 / prims.yw

    cie_d = (
        prims.xr * (prims.yg - prims.yb)
        + prims.xg * (prims.yb - prims.yr)
        + prims.xb * (prims.yr - prims.yg)
    )
    cie_c_rd = yw_inv * (
        prims.xw * (prims.yg - prims.yb)
        - prims.yw * (prims.xg - prims.xb)
        + prims.xg * prims.yb
        - prims.xb * prims.yg
    )
    cie_c_gd = yw_inv * (
        prims.xw * (prims.yb - prims.yr)
        - prims.yw * (prims.xb - prims.xr)
        - prims.xr * prims.yb
        + prims.xb * prims.yr
    )
    cie_c_bd = yw_inv * (
        prims.xw * (prims.yr - prims.yg)
        - prims.yw * (prims.xr - prims.xg)
        + prims.xr * prims.yg
        - prims.xg * prims.yr
    )

    coeff_00 = (
        prims.yg - prims.yb - prims.xb * prims.yg + prims.yb * prims.xg
    ) / cie_c_rd
    coeff_01 = (
        prims.xb - prims.xg - prims.xb * prims.yg + prims.xg * prims.yb
    ) / cie_c_rd
    coeff_02 = (prims.xg * prims.yb - prims.xb * prims.yg) / cie_c_rd
    coeff_10 = (
        prims.yb - prims.yr - prims.yb * prims.xr + prims.yr * prims.xb
    ) / cie_c_gd
    coeff_11 = (
        prims.xr - prims.xb - prims.xr * prims.yb + prims.xb * prims.yr
    ) / cie_c_gd
    coeff_12 = (prims.xb * prims.yr - prims.xr * prims.yb) / cie_c_gd
    coeff_20 = (
        prims.yr - prims.yg - prims.yr * prims.xg + prims.yg * prims.xr
    ) / cie_c_bd
    coeff_21 = (
        prims.xg - prims.xr - prims.xg * prims.yr + prims.xr * prims.yg
    ) / cie_c_bd
    coeff_22 = (prims.xr * prims.yg - prims.xg * prims.yr) / cie_c_bd

    if reverse:
        coeff_00 = prims.xr * cie_c_rd / cie_d
        coeff_01 = prims.xg * cie_c_gd / cie_d
        coeff_02 = prims.xb * cie_c_bd / cie_d
        coeff_10 = prims.yr * cie_c_rd / cie_d
        coeff_11 = prims.yg * cie_c_gd / cie_d
        coeff_12 = prims.yb * cie_c_bd / cie_d
        coeff_20 = (1 - prims.xr - prims.yr) * cie_c_rd / cie_d
        coeff_21 = (1 - prims.xg - prims.yg) * cie_c_gd / cie_d
        coeff_22 = (1 - prims.xb - prims.yb) * cie_c_bd / cie_d

    return (
        coeff_00,
        coeff_01,
        coeff_02,
        coeff_10,
        coeff_11,
        coeff_12,
        coeff_20,
        coeff_21,
        coeff_22,
    )


def rgb2xyz(
    red: float, green: float, blue: float, coeffs: tuple
) -> Tuple[float, float, float]:
    """Convert RGB to CIE XYZ.

    Args:
        red: red
        green: green
        blue: blue
        coeffs: coversion matrix.
    Returns:
        CIE X, Y, Z.

    Raise:
        ValueError with invalid coeffs.
    """
    if len(coeffs) != 9:
        raise ValueError(f"{len(coeffs)} coefficients found, expected 9")
    ciex = coeffs[0] * red + coeffs[1] * green + coeffs[2] * blue
    ciey = coeffs[3] * red + coeffs[4] * green + coeffs[5] * blue
    ciez = coeffs[6] * red + coeffs[7] * green + coeffs[8] * blue
    return ciex, ciey, ciez


def xyz2rgb(
    cie_x: float, cie_y: float, cie_z: float, coeffs: tuple
) -> Tuple[float, float, float]:
    """Convert CIE XYZ to RGB.

    Args:
        cie_x: cie_x
        cie_y: cie_y
        cie_z: cie_z
        coeffs: conversion matrix
    Returns:
        Red, Green, Blue

    Raise:
        ValueError for invalid coeffs.
    """
    if len(coeffs) != 9:
        raise ValueError(f"{len(coeffs)} coefficients found, expected 9")
    red = max(0, coeffs[0] * cie_x + coeffs[1] * cie_y + coeffs[2] * cie_z)
    green = max(0, coeffs[3] * cie_x + coeffs[4] * cie_y + coeffs[5] * cie_z)
    blue = max(0, coeffs[6] * cie_x + coeffs[7] * cie_y + coeffs[8] * cie_z)
    return red, green, blue


def spec2xyz(
    cie_xyz_bar,
    spec: list,
    wvl_range: float,
    emis: bool = False,
) -> tuple:
    """Convert spectral data to CIE XYZ.

    Args:
        cie_xyz_bar: CIE color matching function.
        spec: input spectral data as a dictionary sorted by wavelenth as key.
        emis: flag whether the input data is emissive in nature.
    Returns:
        CIE X, Y, Z

    Note:
        Assuming input wavelength is a subset of the CIE ones
    """
    sval_cie = [
        (sv * cxyz[0], sv * cxyz[1], sv * cxyz[2])
        for sv, cxyz in zip(spec, cie_xyz_bar)
    ]
    spec_len = len(spec)
    cie_x, cie_y, cie_z = [sum(s_c) / spec_len * wvl_range for s_c in zip(*sval_cie)]
    if not emis:
        avg_y = sum(i[1] for i in cie_xyz_bar) / spec_len * wvl_range
        cie_x /= avg_y
        cie_y /= avg_y
        cie_z /= avg_y
    return cie_x, cie_y, cie_z


def xyz2xy(cie_x: float, cie_y: float, cie_z: float) -> Tuple[float, float]:
    """Convert CIE XYZ to xy chromaticity.

    Args:
        cie_x: CIE X
        cie_y: CIE Y
        cie_z: CIE Z
    Returns:
        x chromoticity
        y chromoticity
    """
    _sum = cie_x + cie_y + cie_z
    if _sum == 0:
        return 0, 0
    chrom_x = cie_x / _sum
    chrom_y = cie_y / _sum
    return chrom_x, chrom_y
