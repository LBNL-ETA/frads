import argparse
import csv
from dataclasses import dataclass, field
import logging
import math
import os
from pathlib import Path
import random
import re
import string
import subprocess as sp
import time
from typing import Dict, List, Optional, NamedTuple, Tuple, Union, Generator, Any
import xml.etree.ElementTree as ET


logger = logging.getLogger("frads.util")

LEMAX = 683
# Melanopic luminous efficacy for D65
MLEMAX = 754

COLOR_PRIMARIES = {}

COLOR_PRIMARIES["radiance"] = {
    "cie_x_r": 0.640, "cie_y_r": 0.330,
    "cie_x_g": 0.290, "cie_y_g": 0.600,
    "cie_x_b": 0.150, "cie_y_b": 0.060,
    "cie_x_w": 1 / 3, "cie_y_w": 1 / 3
}

COLOR_PRIMARIES["sharp"] = {
    "cie_x_r": 0.6898, "cie_y_r": 0.3206,
    "cie_x_w": 1 / 3, "cie_y_w": 1 / 3
}

COLOR_PRIMARIES["adobe"] = {
    "cie_x_r": 0.640, "cie_y_r": 0.330,
    "cie_x_g": 0.210, "cie_y_g": 0.710,
    "cie_x_b": 0.150, "cie_y_b": 0.060,
    "cie_x_w": 0.3127, "cie_y_w": 0.3290
}

COLOR_PRIMARIES["rimm"] = {
    "cie_x_r": 0.7347, "cie_y_r": 0.2653,
    "cie_x_g": 0.1596, "cie_y_g": 0.8404,
    "cie_x_b": 0.0366, "cie_y_b": 0.0001,
    "cie_x_w": 0.3457, "cie_y_w": 0.3585
}

COLOR_PRIMARIES["709"] = {
    "cie_x_r": 0.640, "cie_y_r": 0.330,
    "cie_x_g": 0.300, "cie_y_g": 0.600,
    "cie_x_b": 0.150, "cie_y_b": 0.060,
    "cie_x_w": 0.3127, "cie_y_w": 0.3290
}

COLOR_PRIMARIES["p3"] = {
    "cie_x_r": 0.680, "cie_y_r": 0.320,
    "cie_x_g": 0.265, "cie_y_g": 0.690,
    "cie_x_b": 0.150, "cie_y_b": 0.060,
    "cie_x_w": 0.314, "cie_y_w": 0.351
}

COLOR_PRIMARIES["2020"] = {
    "cie_x_r": 0.708, "cie_y_r": 0.292,
    "cie_x_g": 0.170, "cie_y_g": 0.797,
    "cie_x_b": 0.131, "cie_y_b": 0.046,
    "cie_x_w": 0.3127, "cie_y_w": 0.3290
}


class PaneProperty(NamedTuple):
    name: str
    thickness: float
    gtype: str
    coated_side: str
    wavelength: List[float]
    transmittance: List[float]
    reflectance_front: List[float]
    reflectance_back: List[float]

    def get_tf_str(self):
        wavelength_tf = tuple(zip(self.wavelength, self.transmittance))
        return '\n'.join([' '.join(map(str, pair)) for pair in wavelength_tf])

    def get_rf_str(self):
        wavelength_rf = tuple(zip(self.wavelength, self.reflectance_front))
        return '\n'.join([' '.join(map(str, pair)) for pair in wavelength_rf])

    def get_rb_str(self):
        wavelength_rb = tuple(zip(self.wavelength, self.reflectance_back))
        return '\n'.join([' '.join(map(str, pair)) for pair in wavelength_rb])


class PaneRGB(NamedTuple):
    measured_data: PaneProperty
    coated_rgb: List[float]
    glass_rgb: List[float]
    trans_rgb: List[float]


def parse_vu(vu_str: str) -> dict:
    """Parse view string into a dictionary.

    Args:
        vu_str: view parameters as a string

    Returns:
        A view dictionary
    """

    args_list = vu_str.strip().split()
    vparser = argparse.ArgumentParser()
    vparser.add_argument('-v', action='store', dest='vt')
    vparser.add_argument('-vp', nargs=3, type=float)
    vparser.add_argument('-vd', nargs=3, type=float)
    vparser.add_argument('-vu', nargs=3, type=float)
    vparser.add_argument('-vv', type=float)
    vparser.add_argument('-vh', type=float)
    vparser.add_argument('-vo', type=float)
    vparser.add_argument('-va', type=float)
    vparser.add_argument('-vs', type=float)
    vparser.add_argument('-vl', type=float)
    vparser.add_argument('-x', type=int)
    vparser.add_argument('-y', type=int)
    vparser.add_argument('-vf', type=str)
    args, _ = vparser.parse_known_args(args_list)
    view_dict = vars(args)
    if view_dict['vt'] is not None:
        view_dict['vt'] = view_dict['vt'][-1]
    view_dict = {k: v for (k, v) in view_dict.items() if v is not None}
    return view_dict


def parse_opt(opt_str: str) -> dict:
    """Parsing option string into a dictionary.

    Args:
        opt_str: rtrace option parameters as a string

    Returns:
        An option dictionary
    """

    args_list = opt_str.strip().split()
    oparser = argparse.ArgumentParser()
    oparser.add_argument('-I', action='store_true', dest="I", default=None)
    oparser.add_argument('-I+', action='store_true', dest="I", default=None)
    oparser.add_argument('-I-', action='store_false', dest="I", default=None)
    oparser.add_argument('-i', action='store_true', dest='i', default=None)
    oparser.add_argument('-i+', action='store_true', dest='i', default=None)
    oparser.add_argument('-i-', action='store_false', dest='i', default=None)
    oparser.add_argument('-V', action='store_true', dest="V", default=None)
    oparser.add_argument('-V+', action='store_true', dest="V", default=None)
    oparser.add_argument('-V-', action='store_false', dest="V", default=None)
    oparser.add_argument('-u', action='store_true', dest='u', default=None)
    oparser.add_argument('-u+', action='store_true', dest='u', default=None)
    oparser.add_argument('-u-', action='store_false', dest='u', default=None)
    oparser.add_argument('-ld', action='store_true', dest='ld', default=None)
    oparser.add_argument('-ld+', action='store_true', dest='ld', default=None)
    oparser.add_argument('-ld-', action='store_false', dest='ld', default=None)
    oparser.add_argument('-w', action='store_true', dest='w', default=None)
    oparser.add_argument('-w+', action='store_true', dest='w', default=None)
    oparser.add_argument('-w-', action='store_false', dest='w', default=None)
    oparser.add_argument('-aa', type=float)
    oparser.add_argument('-ab', type=int)
    oparser.add_argument('-ad', type=int)
    oparser.add_argument('-ar', type=int)
    oparser.add_argument('-as', type=int)
    oparser.add_argument('-c', type=int, default=1)
    oparser.add_argument('-dc', type=int)
    oparser.add_argument('-dj', type=float)
    oparser.add_argument('-dp', type=int)
    oparser.add_argument('-dr', type=int)
    oparser.add_argument('-ds', type=int)
    oparser.add_argument('-dt', type=int)
    oparser.add_argument('-f', action='store')
    oparser.add_argument('-hd', action='store_const', const='', default=None)
    oparser.add_argument('-lr', type=int)
    oparser.add_argument('-lw', type=float)
    oparser.add_argument('-n', type=int)
    oparser.add_argument('-ss', type=int)
    oparser.add_argument('-st', type=int)
    args, _ = oparser.parse_known_args(args_list)
    opt_dict = vars(args)
    opt_dict = {k: v for (k, v) in opt_dict.items() if v is not None}
    return opt_dict


def tmit2tmis(tmit: float) -> float:
    """Convert from transmittance to transmissivity."""
    tmis = round((math.sqrt(0.8402528435 + 0.0072522239 * tmit**2)
                  - 0.9166530661) / 0.0036261119 / tmit, 3)
    return max(0, min(tmis, 1))


def parse_idf(content: str) -> dict:
    """Parse an IDF file into a dictionary."""
    sections = content.rstrip().split(';')
    sub_sections: List[List[str]] = []
    obj_dict: Dict[str, List[List[str]]] = {}
    for sec in sections:
        sec_lines = sec.splitlines()
        _lines = []
        for sl in sec_lines:
            content = sl.split('!')[0]
            if content != '':
                _lines.append(content)
        _lines = ' '.join(_lines).split(',')
        clean_lines = [i.strip() for i in _lines]
        sub_sections.append(clean_lines)

    for ssec in sub_sections:
        obj_dict[ssec[0].lower()] = []
    for ssec in sub_sections:
        obj_dict[ssec[0].lower()].append(ssec[1:])
    return obj_dict


def parse_optics(fpath):
    """Read and parse an optics file."""
    # enc = 'cp1250' #decoding needed to parse header
    with open(fpath, errors='ignore') as rdr:
        raw = rdr.read()
    header_lines = [i for i in raw.splitlines() if i.startswith('{')]
    if header_lines == []:
        raise Exception('No header in optics file')
    header = {}
    for line in header_lines:
        if line.strip().split("}")[-1] != "":
            key = re.search("{(.*?)}", line).group(1).strip()
            val = line.split("}")[-1].strip()
            header[key] = val
        elif line:
            content = re.search("{(.*?)}", line).group(1).strip()
            if content != "":
                key = content.split(":")[0].strip()
                val = content.split(":")[1].strip()
                header[key] = val
    name = header['Product Name'].replace(" ","_")
    thickness = float(header['Thickness'])
    gtype = header['Type']
    coated_side = header["Coated Side"].lower()
    data = [i.split() for i in raw.strip().splitlines() if not i.startswith('{')]
    wavelength = [float(row[0]) for row in data]
    transmittance = [float(row[1]) for row in data]
    reflectance_front = [float(row[2]) for row in data]
    reflectance_back = [float(row[3]) for row in data]
    if header['Units, Wavelength Units'] == 'SI Microns': # um to nm
        wavelength = [val * 1e3 for val in wavelength]
    return PaneProperty(
        name, thickness, gtype, coated_side, wavelength,
        transmittance, reflectance_front, reflectance_back)


def parse_igsdb_json(json_obj: dict):
    name = json_obj['name'].replace(" ", "_")
    gtype = json_obj['type']
    coated_side = json_obj['coated_side'].lower()
    thickness = json_obj['measured_data']['thickness']
    spectral_data = json_obj['spectral_data']['spectral_data']

    wavelength = []
    transmittance = []
    reflectance_front = []
    reflectance_back = []

    for row in spectral_data:
        wavelength.append(row['wavelength'] * 1e3) # um to nm
        transmittance.append(row['T'])
        reflectance_front.append(row['Rf'])
        reflectance_back.append(row['Rb'])
    return PaneProperty(
        name, thickness, gtype, coated_side, wavelength,
        transmittance, reflectance_front, reflectance_back)


def get_igsdb_json(igsdb_id, token, xml=False):
    """Get igsdb data by igsdb_id"""
    if token is None:
        raise ValueError("Need IGSDB token")
    url = "https://igsdb.lbl.gov/api/v1/products/{}"
    if xml:
        url += "/datafile"
    header = {"Authorization": "Token " + token}
    response = request(url.format(igsdb_id), header)
    if response == '{"detail":"Not found."}':
        raise ValueError("Unknown igsdb id: ", igsdb_id)
    return response


def get_tristi_paths():
    """Get CIE tri-stimulus file paths.
    In addition, also getmelanopic action spetra path
    """
    _file_path_ = os.path.dirname(__file__)
    standards_path = os.path.join(_file_path_, "data", "standards")
    cie_path = {}
    # 2° observer
    cie_path["x2"] = os.path.join(standards_path, "CIE 1931 1nm X.dsp")
    cie_path["y2"] = os.path.join(standards_path, "CIE 1931 1nm Y.dsp")
    cie_path["z2"] = os.path.join(standards_path, "CIE 1931 1nm Z.dsp")
    # 10° observer
    cie_path["x10"] = os.path.join(standards_path, "CIE 1964 1nm X.dsp")
    cie_path["y10"] = os.path.join(standards_path, "CIE 1964 1nm Y.dsp")
    cie_path["z10"] = os.path.join(standards_path, "CIE 1964 1nm Z.dsp")
    # melanopic action spectra
    cie_path["mlnp"] = os.path.join(standards_path, "CIE S 026 1nm.dsp")
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

    trix_i = [trix[wvl] for wvl in inp_wvl if wvl in trix]
    triy_i = [triy[wvl] for wvl in inp_wvl if wvl in triy]
    triz_i = [triz[wvl] for wvl in inp_wvl if wvl in triz]
    mlnp_i = [mlnp[wvl] for wvl in inp_wvl if wvl in mlnp]
    return trix_i, triy_i, triz_i, mlnp_i


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

    cie_d = (cie_x_r * (cie_y_g - cie_y_b) +
            cie_x_g * (cie_y_b - cie_y_r) + cie_x_b*(cie_y_r - cie_y_g))
    cie_c_rd = (cie_y_w_inv * (cie_x_w * (cie_y_g - cie_y_b) -
        cie_y_w * (cie_x_g - cie_x_b) + cie_x_g * cie_y_b - cie_x_b * cie_y_g))
    cie_c_gd = (cie_y_w_inv * (cie_x_w * (cie_y_b - cie_y_r) -
        cie_y_w * (cie_x_b - cie_x_r) - cie_x_r * cie_y_b + cie_x_b * cie_y_r))
    cie_c_bd = (cie_y_w_inv * (cie_x_w * (cie_y_r - cie_y_g) -
        cie_y_w * (cie_x_r - cie_x_g) + cie_x_r * cie_y_g - cie_x_g * cie_y_r))

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


    return (coeff_00, coeff_01, coeff_02, coeff_10,
            coeff_11, coeff_12, coeff_20, coeff_21, coeff_22)


def rgb2xyz(r, g, b, coeffs: tuple):
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


def xyz2rgb(x, y, z, coeffs: tuple):
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


def spec2xyz(trix: List[float], triy: List[float], triz: List[float],
        mlnp: List[float], sval: List[float], emis=False) -> tuple:
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
    if ((len(trix) != len(triy)) or (len(trix) != len(triz)) 
            or (len(trix) != len(sval)) or (len(mlnp) != len(sval))):
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


def unpack_idf(path: str) -> dict:
    """Read and parse and idf files."""
    with open(path, 'r') as rdr:
        return parse_idf(rdr.read())


def nest_list(inp: list, col_cnt: int) -> List[list]:
    """Make a list of list give the column count."""
    nested = []
    if len(inp) % col_cnt != 0:
        raise ValueError("Missing value in matrix data")
    for i in range(0, len(inp), col_cnt):
        sub_list = []
        for n in range(col_cnt):
            try:
                sub_list.append(inp[i+n])
            except IndexError:
                break
        nested.append(sub_list)
    return nested


def write_square_matrix(opath, sdata):
    nrow = len(sdata)
    ncol = len(sdata[0])
    with open(opath, 'w') as wt:
        header = '#?RADIANCE\nNCOMP=3\n'
        header += 'NROWS=%d\nNCOLS=%d\n' % (nrow, ncol)
        header += 'FORMAT=ascii\n\n'
        wt.write(header)
        for row in sdata:
            for val in row:
                string = '\t'.join(['%07.5f' % val] * 3)
                wt.write(string)
                wt.write('\t')
            wt.write('\n')


def parse_bsdf_xml(path: str) -> dict:
    """Parse BSDF file in XML format.
    TODO: validate xml first before parsing
    """
    error_msg = f"Error parsing {path}: "
    data_dict: dict = {"Def": "", "Solar": {}, "Visible": {}}
    tree = ET.parse(path)
    if (root := tree.getroot()) is None:
        raise ValueError(error_msg + "Root not found")
    tag = root.tag.rstrip('WindowElement')
    if (optical := root.find(tag + 'Optical')) is None:
        raise ValueError(error_msg + "Optical not found")
    if (layer := optical.find(tag + 'Layer')) is None:
        raise ValueError(error_msg + "Layer not found")
    if (data_def := layer.find(tag + 'DataDefinition')) is None:
        raise ValueError(error_msg + "data definition not found")
    if (data_struct_txt := data_def.findtext(tag + 'IncidentDataStructure')) is None:
        raise ValueError(error_msg + "data structure not found")
    data_dict["Def"] = data_struct_txt.strip()
    data_blocks = layer.findall(tag + 'WavelengthData')
    for block in data_blocks:
        if (wavelength_txt := block.findtext(tag + 'Wavelength')) is None:
            raise ValueError(error_msg + "wavelength not found")
        if wavelength_txt not in ("Solar", "Visible"):
            raise ValueError("Unknown %s" % wavelength_txt)
        if (dblock := block.find(tag + 'WavelengthDataBlock')) is None:
            raise ValueError(error_msg + "wavelength data block not found")
        if (direction := dblock.findtext(tag + 'WavelengthDataDirection')) is None:
            raise ValueError(error_msg + "wavelength direction not found")
        if (sdata_txt := dblock.findtext(tag + 'ScatteringData')) is None:
            raise ValueError(error_msg + "scattering data not found")
        sdata_txt = sdata_txt.strip()
        if sdata_txt.count('\n') == 21168:
            sdata_txt = sdata_txt.replace('\n\t', ' ')
        data_dict[wavelength_txt][direction] = sdata_txt
    return data_dict


def dhi2dni(GHI: float, DHI: float, alti: float) -> float:
    """Calculate direct normal from global horizontal
    and diffuse horizontal irradiance."""
    return (GHI - DHI) / math.cos(math.radians(90 - alti))


def frange_inc(start, stop, step):
    """Generate increasing non-integer range."""
    r = start
    while r < stop:
        yield r
        r += step


def frange_des(start, stop, step):
    """Generate descending non-integer range."""
    r = start
    while r > stop:
        yield r
        r -= step


def basename(fpath, keep_ext=False):
    """Get the basename from a file path."""
    name = os.path.basename(fpath)
    if not keep_ext:
        name = os.path.splitext(name)[0]
    return name


def is_number(string):
    """Test is string is a number."""
    try:
        float(string)
        return True
    except ValueError:
        return False


def silent_remove(path):
    """Remove a file, silent if file does not exist."""
    try:
        os.remove(path)
    except FileNotFoundError as e:
        logger.error(e)


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
    out_disk_r = [in_square_a, in_square_b,
                  -in_square_a, -in_square_b][in_square_rgn]
    if in_square_b * in_square_b > 0:
        phi_select_4 = 6 - in_square_a / in_square_b
    else:
        phi_select_4 = 0
    phi_select = [
        in_square_b/in_square_a,
        2 - in_square_a/in_square_b,
        4 + in_square_b/in_square_a,
        phi_select_4,
    ]
    out_disk_phi = math.pi / 4 * phi_select[in_square_rgn]
    out_disk_x = out_disk_r * math.cos(out_disk_phi)
    out_disk_y = out_disk_r * math.sin(out_disk_phi)
    return out_disk_x, out_disk_y, out_disk_r, out_disk_phi


def mkdir_p(path):
    """Make a directory, silent if exist."""
    try:
        os.makedirs(path)
    except OSError as e:
        logger.debug(e, exc_info=logger.getEffectiveLevel() == logging.DEBUG)
    except TypeError as e:
        logger.debug(e, exc_info=logger.getEffectiveLevel() == logging.DEBUG)


def sprun(cmd):
    """Call subprocess run"""
    logger.debug(cmd)
    proc = sp.run(cmd, check=True, stderr=sp.PIPE)
    if proc.stderr != b'':
        logger.warning(proc.stderr)


def spcheckout(cmd, inp=None):
    """Call subprocess run and return results."""
    logger.debug(cmd)
    proc = sp.run(cmd, input=inp, stderr=sp.PIPE, stdout=sp.PIPE)
    if proc.stderr != b'':
        logger.warning(proc.stderr)
    return proc.stdout


def id_generator(size=3, chars=None):
    """Generate random characters."""
    if chars is None:
        chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))


def tokenize(inp: str) -> Generator[str, None, None]:
    """Generator for tokenizing a string that
    is seperated by a space or a comma.
    Args:
       inp: input string
    Yields:
        next token
    """
    tokens = re.compile(
        ' +|[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)+|[\d*\.\d+]+|[{}]')
    for match in tokens.finditer(inp):
        if match.group(0)[0] in " ,":
            continue
        else:
            yield match.group(0)


def parse_branch(token: Generator[str, None, None]) -> Any:
    """Prase tensor tree branches recursively by opening and closing curly braces.
    Args:
        token: token generator object.
    Return:
        children: parsed branches as nexted list
    """
    children = []
    while True:
        value = next(token)
        if value == "{":
            children.append(parse_branch(token))
        elif value == "}":
            return children
        else:
            children.append(float(value))


def parse_ttree(data_str: str) -> list:
    """Parse a tensor tree.
    Args:
        data_str: input data string
    Returns:
        A nested list that is the tree
    """
    tokenized = tokenize(data_str)
    if next(tokenized) != "{":
        raise ValueError("Tensor tree data not starting with {")
    return parse_branch(tokenized)


def get_nested_list_levels(nested_list: list) -> int:
    """Calculate the number of levels given a nested list."""
    return isinstance(nested_list, list) and max(map(get_nested_list_levels, nested_list)) + 1


class TensorTree:
    """The tensor tree object.
    Anisotropic tensor tree has should have 16 lists
    Attributes:
        parsed: parsed tensor tree object)
        depth: number of tree levels
    """

    def __init__(self, parsed):
        self.parsed = parsed
        self.depth = get_nested_list_levels(parsed)

    def lookup(self, xp, yp) -> list:
        """Traverses a parsed tensor tree (a nexted list) given a input position."""
        branch_idx = self.get_branch_index(xp, yp)
        quads = [self.parsed[i] for i in branch_idx]
        return [self.traverse(quad, xp, yp) for quad in quads]

    def get_leaf_index(self, xp, yp):
        if xp < 0:
            if yp < 0:
                return range(0, 4)
            else:
                return range(4, 8)
        else:
            if yp < 0:
                return range(8, 12)
            else:
                return range(12, 16)

    def get_branch_index(self, xp, yp):
        """Gets a set of index."""
        if xp < 0:
            if yp < 0:
                return range(0, 16, 4)
            else:
                return range(2, 16, 4)
        else:
            if yp < 0:
                return range(1, 16, 4)
            else:
                return range(3, 16, 4)

    def traverse(self, quad, xp, yp, n=1) -> list:
        """Traverse a quadrant."""
        if len(quad) == 1:  # single leaf
            res = quad
        else:
            res = []
            # get x, y signage in relation to branches
            _x = xp + 1 / (2**n) if xp < 0 else xp - 1 / (2**n)
            _y = yp + 1 / (2**n) if yp < 0 else yp - 1 / (2**n)
            n += 1
            # which four branches to get? get index for them
            if n < self.depth:
                ochild = self.get_branch_index(_x, _y)
            else:
                ochild = self.get_leaf_index(_x, _y)
            sub_quad = [quad[i] for i in ochild]
            if all(isinstance(i, float) for i in sub_quad):
                res = sub_quad  # single leaf for each branch
            else:  # branches within this quadrant
                for sq in sub_quad:
                    if len(sq) > 4:  # there is another branch
                        res.append(self.traverse(sq, _x, _y, n=n))
                    else:  # just a leaf
                        res.append(sq)
        return res
