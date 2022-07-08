"""
This module contains all data parsing routines.
"""
import argparse
import re
import subprocess as sp
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
import xml.etree.ElementTree as ET

from frads import geom
from frads import utils
from frads.types import Primitive
from frads.types import PaneProperty
from frads.types import WeaMetaData
from frads.types import WeaDataRow


def parse_primitive(lines: list) -> List[Primitive]:
    """Parse Radiance primitives inside a file path into a list of dictionary.
    Args:
        lines: list of lines as strings

    Returns:
        list of primitives as dictionaries
    """
    # Expand in-line commands
    cmd_lines = [(idx, line) for idx, line in enumerate(lines) if line.startswith("!")]
    cmd_results = []
    for cmd in cmd_lines:
        cmd_results.append(
            sp.run(cmd[1][1:], shell=True, stdout=sp.PIPE).stdout.decode().splitlines()
        )
    counter = 0
    for idx, item in enumerate(cmd_lines):
        counter += item[0]
        lines[counter : counter + 1] = cmd_results[idx]
        counter += len(cmd_results[idx]) - 1 - item[0]

    content = " ".join(
        [i.strip() for i in lines if i.strip() != "" and i[0] != "#"]
    ).split()
    primitives: List[Primitive] = []
    idx = 0
    while idx < len(content):
        _modifier = content[idx]
        _type = content[idx + 1]
        if _type == "alias":
            _name_to = content[idx + 2]
            _name_from = content[idx + 3]
            primitives.append(
                Primitive(_modifier, _type, _name_to, _name_from, "", int_arg="")
            )
            idx += 4
            continue
        _identifier = content[idx + 2]
        str_arg_cnt = int(content[idx + 3])
        _str_args = " ".join(content[idx + 3 : idx + 4 + str_arg_cnt])
        idx += 5 + str_arg_cnt
        real_arg_cnt = int(content[idx])
        _real_args = " ".join(content[idx : idx + 1 + real_arg_cnt])
        idx += real_arg_cnt + 1
        primitives.append(
            Primitive(_modifier, _type, _identifier, _str_args, _real_args)
        )
    return primitives


def parse_polygon(real_arg: str) -> geom.Polygon:
    """Parse real arguments to polygon.
    Args:
        primitive: a dictionary object containing a primitive

    Returns:
        modified primitive
    """
    real_args = real_arg.split()
    coords = [float(i) for i in real_args[1:]]
    arg_cnt = int(real_args[0])
    vertices = [geom.Vector(*coords[i : i + 3]) for i in range(0, arg_cnt, 3)]
    return geom.Polygon(vertices)


def parse_vu(vu_str: str) -> dict:
    """Parse view string into a dictionary.

    Args:
        vu_str: view parameters as a string

    Returns:
        A view dictionary
    """

    args_list = vu_str.strip().split()
    vparser = argparse.ArgumentParser()
    vparser.add_argument("-v", action="store", dest="vt")
    vparser.add_argument("-vp", nargs=3, type=float)
    vparser.add_argument("-vd", nargs=3, type=float)
    vparser.add_argument("-vu", nargs=3, type=float)
    vparser.add_argument("-vv", type=float)
    vparser.add_argument("-vh", type=float)
    vparser.add_argument("-vo", type=float)
    vparser.add_argument("-va", type=float)
    vparser.add_argument("-vs", type=float)
    vparser.add_argument("-vl", type=float)
    vparser.add_argument("-x", type=int)
    vparser.add_argument("-y", type=int)
    vparser.add_argument("-vf", type=str)
    args, _ = vparser.parse_known_args(args_list)
    view_dict = vars(args)
    if view_dict["vt"] is not None:
        view_dict["vt"] = view_dict["vt"][-1]
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
    oparser.add_argument("-I", action="store_true", dest="I", default=None)
    oparser.add_argument("-I+", action="store_true", dest="I", default=None)
    oparser.add_argument("-I-", action="store_false", dest="I", default=None)
    oparser.add_argument("-i", action="store_true", dest="i", default=None)
    oparser.add_argument("-i+", action="store_true", dest="i", default=None)
    oparser.add_argument("-i-", action="store_false", dest="i", default=None)
    oparser.add_argument("-V", action="store_true", dest="V", default=None)
    oparser.add_argument("-V+", action="store_true", dest="V", default=None)
    oparser.add_argument("-V-", action="store_false", dest="V", default=None)
    oparser.add_argument("-u", action="store_true", dest="u", default=None)
    oparser.add_argument("-u+", action="store_true", dest="u", default=None)
    oparser.add_argument("-u-", action="store_false", dest="u", default=None)
    oparser.add_argument("-ld", action="store_true", dest="ld", default=None)
    oparser.add_argument("-ld+", action="store_true", dest="ld", default=None)
    oparser.add_argument("-ld-", action="store_false", dest="ld", default=None)
    oparser.add_argument("-w", action="store_true", dest="w", default=None)
    oparser.add_argument("-w+", action="store_true", dest="w", default=None)
    oparser.add_argument("-w-", action="store_false", dest="w", default=None)
    oparser.add_argument("-aa", type=float)
    oparser.add_argument("-ab", type=int)
    oparser.add_argument("-ad", type=int)
    oparser.add_argument("-ar", type=int)
    oparser.add_argument("-as", type=int)
    oparser.add_argument("-c", type=int, default=1)
    oparser.add_argument("-dc", type=int)
    oparser.add_argument("-dj", type=float)
    oparser.add_argument("-dp", type=int)
    oparser.add_argument("-dr", type=int)
    oparser.add_argument("-ds", type=int)
    oparser.add_argument("-dt", type=int)
    oparser.add_argument("-f", action="store")
    oparser.add_argument("-hd", action="store_const", const="", default=None)
    oparser.add_argument("-lr", type=int)
    oparser.add_argument("-lw", type=float)
    oparser.add_argument("-n", type=int)
    oparser.add_argument("-ss", type=int)
    oparser.add_argument("-st", type=int)
    args, _ = oparser.parse_known_args(args_list)
    opt_dict = vars(args)
    opt_dict = {k: v for (k, v) in opt_dict.items() if v is not None}
    return opt_dict


def parse_idf(content: str) -> dict:
    """Parse an IDF file into a dictionary."""
    sections = content.rstrip().split(";")
    sub_sections: List[List[str]] = []
    obj_dict: Dict[str, List[List[str]]] = {}
    for sec in sections:
        sec_lines = sec.splitlines()
        _lines = []
        for sl in sec_lines:
            content = sl.split("!")[0]
            if content != "":
                _lines.append(content)
        _lines = " ".join(_lines).split(",")
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
    with open(fpath, errors="ignore") as rdr:
        raw = rdr.read()
    header_lines = [i for i in raw.splitlines() if i.startswith("{")]
    if header_lines == []:
        raise Exception("No header in optics file")
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
    name = header["Product Name"].replace(" ", "_")
    thickness = float(header["Thickness"])
    gtype = header["Type"]
    coated_side = header["Coated Side"].lower()
    data = [i.split() for i in raw.strip().splitlines() if not i.startswith("{")]
    wavelength = [float(row[0]) for row in data]
    transmittance = [float(row[1]) for row in data]
    reflectance_front = [float(row[2]) for row in data]
    reflectance_back = [float(row[3]) for row in data]
    if header["Units, Wavelength Units"] == "SI Microns":  # um to nm
        wavelength = [val * 1e3 for val in wavelength]
    return PaneProperty(
        name,
        thickness,
        gtype,
        coated_side,
        wavelength,
        transmittance,
        reflectance_front,
        reflectance_back,
    )


def parse_igsdb_json(json_obj: dict):
    name = json_obj["name"].replace(" ", "_")
    gtype = json_obj["type"]
    coated_side = json_obj["coated_side"].lower()
    thickness = json_obj["measured_data"]["thickness"]
    spectral_data = json_obj["spectral_data"]["spectral_data"]

    wavelength = []
    transmittance = []
    reflectance_front = []
    reflectance_back = []

    for row in spectral_data:
        wavelength.append(row["wavelength"] * 1e3)  # um to nm
        transmittance.append(row["T"])
        reflectance_front.append(row["Rf"])
        reflectance_back.append(row["Rb"])
    return PaneProperty(
        name,
        thickness,
        gtype,
        coated_side,
        wavelength,
        transmittance,
        reflectance_front,
        reflectance_back,
    )


def parse_bsdf_xml(path: str) -> dict:
    """Parse BSDF file in XML format.
    TODO: validate xml first before parsing
    """
    error_msg = f"Error parsing {path}: "
    data_dict: dict = {"Def": "", "Solar": {}, "Visible": {}}
    tree = ET.parse(path)
    if (root := tree.getroot()) is None:
        raise ValueError(error_msg + "Root not found")
    tag = root.tag.rstrip("WindowElement")
    if (optical := root.find(tag + "Optical")) is None:
        raise ValueError(error_msg + "Optical not found")
    if (layer := optical.find(tag + "Layer")) is None:
        raise ValueError(error_msg + "Layer not found")
    if (data_def := layer.find(tag + "DataDefinition")) is None:
        raise ValueError(error_msg + "data definition not found")
    if (data_struct_txt := data_def.findtext(tag + "IncidentDataStructure")) is None:
        raise ValueError(error_msg + "data structure not found")
    data_dict["Def"] = data_struct_txt.strip()
    data_blocks = layer.findall(tag + "WavelengthData")
    for block in data_blocks:
        if (wavelength_txt := block.findtext(tag + "Wavelength")) is None:
            raise ValueError(error_msg + "wavelength not found")
        if wavelength_txt not in ("Solar", "Visible"):
            raise ValueError("Unknown %s" % wavelength_txt)
        if (dblock := block.find(tag + "WavelengthDataBlock")) is None:
            raise ValueError(error_msg + "wavelength data block not found")
        if (direction := dblock.findtext(tag + "WavelengthDataDirection")) is None:
            raise ValueError(error_msg + "wavelength direction not found")
        if (sdata_txt := dblock.findtext(tag + "ScatteringData")) is None:
            raise ValueError(error_msg + "scattering data not found")
        sdata_txt = sdata_txt.strip()
        if sdata_txt.count("\n") == 21168:
            sdata_txt = sdata_txt.replace("\n\t", " ")
        data_dict[wavelength_txt][direction] = sdata_txt
    return data_dict


def parse_rad_header(header_str: str) -> tuple:
    """Parse a Radiance matrix file header."""
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
    tokenized = utils.tokenize(data_str)
    if next(tokenized) != "{":
        raise ValueError("Tensor tree data not starting with {")
    return parse_branch(tokenized)


def get_nested_list_levels(nested_list: list) -> int:
    """Calculate the number of levels given a nested list."""
    return (
        isinstance(nested_list, list)
        and max(map(get_nested_list_levels, nested_list)) + 1
    )


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


def parse_wea(wea_str: str):
    """Parse a wea file in its entirety."""
    lines = wea_str.splitlines()
    place = lines[0].split(" ", 1)[1]
    lat = float(lines[1].split(" ", 1)[1])
    lon = float(lines[2].split(" ", 1)[1])
    tz = int(float(lines[3].split(" ", 1)[1]))
    ele = float(lines[4].split(" ", 1)[1])
    meta_data = WeaMetaData(place, "", lat, lon, tz, ele)
    data = []
    for li in lines[6:]:
        if li.strip() == "":
            continue
        line = li.split()
        month = int(line[0])
        day = int(line[1])
        hours = float(line[2])
        hour = int(hours)
        minute = int((hours - hour) * 60)
        dir_norm = float(line[3])
        dif_hor = float(line[4])
        data.append(WeaDataRow(month, day, hour, minute, 0, hours, dir_norm, dif_hor))
    return meta_data, data


def parse_epw(epw_str: str) -> tuple:
    """Parse epw file and return wea header and data."""
    raw = epw_str.splitlines()
    epw_header = raw[0].split(",")
    content = raw[8:]
    data = []
    for li in content:
        line = li.split(",")
        month = int(line[1])
        day = int(line[2])
        hour = int(line[3]) - 1
        hours = hour + 0.5
        dir_norm = float(line[14])
        dif_hor = float(line[15])
        data.append(WeaDataRow(month, day, hour, 30, 0, hours, dir_norm, dif_hor))
    city = epw_header[1]
    country = epw_header[3]
    latitude = float(epw_header[6])
    longitude = -1 * float(epw_header[7])
    tz = int(float(epw_header[8])) * (-15)
    elevation = float(epw_header[9].rstrip())
    meta_data = WeaMetaData(city, country, latitude, longitude, tz, elevation)
    return meta_data, data
