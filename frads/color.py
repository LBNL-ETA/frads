"""
This module contains all color and spectral-related functionalities.
"""
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple


LEMAX = 683
# Melanopic luminous efficacy for D65
MLEMAX = 754


COLOR_PRIMARIES = {}

COLOR_PRIMARIES["radiance"] = {
    "cie_x_r": 0.640,
    "cie_y_r": 0.330,
    "cie_x_g": 0.290,
    "cie_y_g": 0.600,
    "cie_x_b": 0.150,
    "cie_y_b": 0.060,
    "cie_x_w": 1 / 3,
    "cie_y_w": 1 / 3,
}

COLOR_PRIMARIES["sharp"] = {
    "cie_x_r": 0.6898,
    "cie_y_r": 0.3206,
    "cie_x_w": 1 / 3,
    "cie_y_w": 1 / 3,
}

COLOR_PRIMARIES["adobe"] = {
    "cie_x_r": 0.640,
    "cie_y_r": 0.330,
    "cie_x_g": 0.210,
    "cie_y_g": 0.710,
    "cie_x_b": 0.150,
    "cie_y_b": 0.060,
    "cie_x_w": 0.3127,
    "cie_y_w": 0.3290,
}

COLOR_PRIMARIES["rimm"] = {
    "cie_x_r": 0.7347,
    "cie_y_r": 0.2653,
    "cie_x_g": 0.1596,
    "cie_y_g": 0.8404,
    "cie_x_b": 0.0366,
    "cie_y_b": 0.0001,
    "cie_x_w": 0.3457,
    "cie_y_w": 0.3585,
}

COLOR_PRIMARIES["709"] = {
    "cie_x_r": 0.640,
    "cie_y_r": 0.330,
    "cie_x_g": 0.300,
    "cie_y_g": 0.600,
    "cie_x_b": 0.150,
    "cie_y_b": 0.060,
    "cie_x_w": 0.3127,
    "cie_y_w": 0.3290,
}

COLOR_PRIMARIES["p3"] = {
    "cie_x_r": 0.680,
    "cie_y_r": 0.320,
    "cie_x_g": 0.265,
    "cie_y_g": 0.690,
    "cie_x_b": 0.150,
    "cie_y_b": 0.060,
    "cie_x_w": 0.314,
    "cie_y_w": 0.351,
}

COLOR_PRIMARIES["2020"] = {
    "cie_x_r": 0.708,
    "cie_y_r": 0.292,
    "cie_x_g": 0.170,
    "cie_y_g": 0.797,
    "cie_x_b": 0.131,
    "cie_y_b": 0.046,
    "cie_x_w": 0.3127,
    "cie_y_w": 0.3290,
}


def get_tristi_paths() -> Dict[str, Path]:
    """Get CIE tri-stimulus file paths.
    In addition, also getmelanopic action spetra path
    """
    standards_path = Path(__file__).parent / "data" / "standards"
    cie_path = {}
    # 2° observer
    cie_path["x2"] = standards_path / "CIE 1931 1nm X.dsp"
    cie_path["y2"] = standards_path / "CIE 1931 1nm Y.dsp"
    cie_path["z2"] = standards_path / "CIE 1931 1nm Z.dsp"
    # 10° observer
    cie_path["x10"] = standards_path / "CIE 1964 1nm X.dsp"
    cie_path["y10"] = standards_path / "CIE 1964 1nm Y.dsp"
    cie_path["z10"] = standards_path / "CIE 1964 1nm Z.dsp"
    # melanopic action spectra
    cie_path["mlnp"] = standards_path / "CIE S 026 1nm.dsp"
    return cie_path


def load_cie_tristi(inp_wvl: list, observer: str) -> tuple:
    """Load CIE tristimulus data according to input wavelength.
    Also load melanopic action spectra data as well.
    Args:
        inp_wvl: input wavelength in nm
        observer: 2° or 10° observer for the colar matching function.
    Returns:
        Tristimulus XYZ and melanopic action spectra
    """
    header_lines = 3
    cie_path = get_tristi_paths()
    with open(cie_path[f"x{observer}"]) as rdr:
        lines = rdr.readlines()[header_lines:]
        trix = {float(row.split()[0]): float(row.split()[1]) for row in lines}
    with open(cie_path[f"y{observer}"]) as rdr:
        lines = rdr.readlines()[header_lines:]
        triy = {float(row.split()[0]): float(row.split()[1]) for row in lines}
    with open(cie_path[f"z{observer}"]) as rdr:
        lines = rdr.readlines()[header_lines:]
        triz = {float(row.split()[0]): float(row.split()[1]) for row in lines}
    with open(cie_path["mlnp"]) as rdr:
        lines = rdr.readlines()[header_lines:]
        mlnp = {float(row.split()[0]): float(row.split()[1]) for row in lines}

    trix_i = [trix[wvl] for idx, wvl in enumerate(inp_wvl) if wvl in trix]
    triy_i = [triy[wvl] for wvl in inp_wvl if wvl in triy]
    triz_i = [triz[wvl] for wvl in inp_wvl if wvl in triz]
    mlnp_i = [mlnp[wvl] for wvl in inp_wvl if wvl in mlnp]
    oidx = [idx for idx, wvl in enumerate(inp_wvl) if wvl in triy]
    return trix_i, triy_i, triz_i, mlnp_i, oidx


def get_conversion_matrix(prims, reverse=False):
    # The whole calculation is based on the CIE (x,y) chromaticities below

    cie_x_r = COLOR_PRIMARIES[prims]["cie_x_r"]
    cie_y_r = COLOR_PRIMARIES[prims]["cie_y_r"]
    cie_x_g = COLOR_PRIMARIES[prims]["cie_x_g"]
    cie_y_g = COLOR_PRIMARIES[prims]["cie_y_g"]
    cie_x_b = COLOR_PRIMARIES[prims]["cie_x_b"]
    cie_y_b = COLOR_PRIMARIES[prims]["cie_y_b"]
    cie_x_w = COLOR_PRIMARIES[prims]["cie_x_w"]
    cie_y_w = COLOR_PRIMARIES[prims]["cie_y_w"]

    cie_y_w_inv = 1 / cie_y_w

    cie_d = (
        cie_x_r * (cie_y_g - cie_y_b)
        + cie_x_g * (cie_y_b - cie_y_r)
        + cie_x_b * (cie_y_r - cie_y_g)
    )
    cie_c_rd = cie_y_w_inv * (
        cie_x_w * (cie_y_g - cie_y_b)
        - cie_y_w * (cie_x_g - cie_x_b)
        + cie_x_g * cie_y_b
        - cie_x_b * cie_y_g
    )
    cie_c_gd = cie_y_w_inv * (
        cie_x_w * (cie_y_b - cie_y_r)
        - cie_y_w * (cie_x_b - cie_x_r)
        - cie_x_r * cie_y_b
        + cie_x_b * cie_y_r
    )
    cie_c_bd = cie_y_w_inv * (
        cie_x_w * (cie_y_r - cie_y_g)
        - cie_y_w * (cie_x_r - cie_x_g)
        + cie_x_r * cie_y_g
        - cie_x_g * cie_y_r
    )

    coeff_00 = (cie_y_g - cie_y_b - cie_x_b * cie_y_g + cie_y_b * cie_x_g) / cie_c_rd
    coeff_01 = (cie_x_b - cie_x_g - cie_x_b * cie_y_g + cie_x_g * cie_y_b) / cie_c_rd
    coeff_02 = (cie_x_g * cie_y_b - cie_x_b * cie_y_g) / cie_c_rd
    coeff_10 = (cie_y_b - cie_y_r - cie_y_b * cie_x_r + cie_y_r * cie_x_b) / cie_c_gd
    coeff_11 = (cie_x_r - cie_x_b - cie_x_r * cie_y_b + cie_x_b * cie_y_r) / cie_c_gd
    coeff_12 = (cie_x_b * cie_y_r - cie_x_r * cie_y_b) / cie_c_gd
    coeff_20 = (cie_y_r - cie_y_g - cie_y_r * cie_x_g + cie_y_g * cie_x_r) / cie_c_bd
    coeff_21 = (cie_x_g - cie_x_r - cie_x_g * cie_y_r + cie_x_r * cie_y_g) / cie_c_bd
    coeff_22 = (cie_x_r * cie_y_g - cie_x_g * cie_y_r) / cie_c_bd

    if reverse:
        coeff_00 = cie_x_r * cie_c_rd / cie_d
        coeff_01 = cie_x_g * cie_c_gd / cie_d
        coeff_02 = cie_x_b * cie_c_bd / cie_d
        coeff_10 = cie_y_r * cie_c_rd / cie_d
        coeff_11 = cie_y_g * cie_c_gd / cie_d
        coeff_12 = cie_y_b * cie_c_bd / cie_d
        coeff_20 = (1 - cie_x_r - cie_y_r) * cie_c_rd / cie_d
        coeff_21 = (1 - cie_x_g - cie_y_g) * cie_c_gd / cie_d
        coeff_22 = (1 - cie_x_b - cie_y_b) * cie_c_bd / cie_d

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


def rgb2xyz(r: float, g: float, b: float, coeffs: tuple) -> Tuple[float, float, float]:
    """Convert RGB to CIE XYZ.

    Args:
        r: red
        g: green
        b: blue
        coeffs: coversion matrix.
    Returns:
        CIE X, Y, Z.

    Raise:
        ValueError with invalid coeffs.
    """
    if len(coeffs) != 9:
        raise ValueError("%s coefficients found, expected 9", len(coeffs))
    x = coeffs[0] * r + coeffs[1] * g + coeffs[2] * b
    y = coeffs[3] * r + coeffs[4] * g + coeffs[5] * b
    z = coeffs[6] * r + coeffs[7] * g + coeffs[8] * b
    return x, y, z


def xyz2rgb(x: float, y: float, z: float, coeffs: tuple) -> Tuple[float, float, float]:
    """Convert CIE XYZ to RGB.

    Args:
        x: cie_x
        y: cie_y
        z: cie_z
        coeffs: conversion matrix
    Returns:
        Red, Green, Blue

    Raise:
        ValueError for invalid coeffs.
    """
    if len(coeffs) != 9:
        raise ValueError("%s coefficients found, expected 9", len(coeffs))
    red = max(0, coeffs[0] * x + coeffs[1] * y + coeffs[2] * z)
    green = max(0, coeffs[3] * x + coeffs[4] * y + coeffs[5] * z)
    blue = max(0, coeffs[6] * x + coeffs[7] * y + coeffs[8] * z)
    return red, green, blue


def spec2xyz(
    trix: List[float],
    triy: List[float],
    triz: List[float],
    mlnp: List[float],
    sval: List[float],
    emis=False,
) -> tuple:
    """Convert spectral data to CIE XYZ.

    Args:
        trix: tristimulus x function
        triy: tristimulus y function
        triz: tristimulus z function
        mlnp: melanopic activation function
        sval: input spectral data, either emissive or refl/trans.
        emis: flag whether the input data is emissive
    Returns:
        CIE X, Y, Z

    Raise:
        ValueError if input data are not of equal length.
    """
    if (
        (len(trix) != len(triy))
        or (len(trix) != len(triz))
        or (len(trix) != len(sval))
        or (len(mlnp) != len(sval))
    ):
        raise ValueError("Input data not of equal length")

    xs = [x * v for x, v in zip(trix, sval)]
    ys = [y * v for y, v in zip(triy, sval)]
    zs = [z * v for z, v in zip(triz, sval)]
    ms = [m * v for m, v in zip(mlnp, sval)]
    cie_X = sum(xs) / len(xs)
    cie_Y = sum(ys) / len(ys)
    cie_Z = sum(zs) / len(zs)
    if not emis:
        avg_y = sum(triy) / len(triy)
        cie_X /= avg_y
        cie_Y /= avg_y
        cie_Z /= avg_y
    return cie_X, cie_Y, cie_Z


def xyz2xy(cie_x: float, cie_y: float, cie_z: float) -> tuple:
    """Convert CIE XYZ to xy chromaticity."""
    _sum = cie_x + cie_y + cie_z
    if _sum == 0:
        return 0, 0
    chrom_x = cie_x / _sum
    chrom_y = cie_y / _sum
    return chrom_x, chrom_y
