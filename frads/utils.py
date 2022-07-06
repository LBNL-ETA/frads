from io import TextIOWrapper
import logging
import math
import os
from pathlib import Path
import random
import re
import string
import subprocess as sp
from typing import Generator
from typing import List
from typing import Set

from frads import geom
from frads import parsers
from frads.types import Primitive
from frads.types import PaneRGB
from frads.types import ScatteringData
from frads.types import BSDFData


logger = logging.getLogger("frads.utils")


GEOM_TYPE = ['polygon', 'ring', 'tube', 'cone']

MATERIAL_TYPE = ['plastic', 'glass', 'trans', 'dielectric', 'BSDF']

BASIS_DICT = {
    '145': 'Klems Full',
    '73': 'Klems Half',
    '41': 'Klems Quarter',
}

TREG_BASE = [
    (90., 0),
    (78., 30),
    (66., 30),
    (54., 24),
    (42., 24),
    (30., 18),
    (18., 12),
    (6., 6),
    (0., 1),
]

ABASE_LIST = {
    "Klems Full": [(0., 1), (5., 8), (15., 16), (25., 20), (35., 24),
                   (45., 24), (55., 24), (65., 16), (75., 12), (90., 0)],
    "Klems Half": [(0., 1), (6.5, 8), (19.5, 12), (32.5, 16), (46.5, 20),
                   (61.5, 12), (76.5, 4), (90., 0)],
    "Klems Quarter": [(0., 1), (9., 8), (27., 12), (46., 12), (66., 8),
                      (90., 0)]
}


def tmit2tmis(tmit: float) -> float:
    """Convert from transmittance to transmissivity."""
    tmis = round((math.sqrt(0.8402528435 + 0.0072522239 * tmit**2)
                  - 0.9166530661) / 0.0036261119 / tmit, 3)
    return max(0, min(tmis, 1))


def polygon2prim(polygon: geom.Polygon,
                 modifier: str, identifier: str) -> Primitive:
    """Generate a primitive from a polygon."""
    return Primitive(modifier, 'polygon', identifier, '0', polygon.to_real())

# def get_igsdb_json(igsdb_id, token, xml=False):
#     """Get igsdb data by igsdb_id"""
#     if token is None:
#         raise ValueError("Need IGSDB token")
#     url = "https://igsdb.lbl.gov/api/v1/products/{}"
#     if xml:
#         url += "/datafile"
#     header = {"Authorization": "Token " + token}
#     response = request(url.format(igsdb_id), header)
#     if response == '{"detail":"Not found."}':
#         raise ValueError("Unknown igsdb id: ", igsdb_id)
#     return response


def unpack_idf(path: str) -> dict:
    """Read and parse and idf files."""
    with open(path, 'r') as rdr:
        return parsers.parse_idf(rdr.read())


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


def sdata2bsdf(sdata: ScatteringData) -> BSDFData:
    basis = BASIS_DICT[str(sdata.ncolumn)]
    lambdas = angle_basis_coeff(basis)
    bsdf = [list(map(lambda x, y: x/y, i, lambdas)) for i in sdata.sdata]
    return BSDFData(bsdf)


def bsdf2sdata(bsdf: BSDFData) -> ScatteringData:
    basis = BASIS_DICT[str(bsdf.ncolumn)]
    lambdas = angle_basis_coeff(basis)
    sdata = [list(map(lambda x, y: x/y, i, lambdas))
             for i in bsdf.bsdf]
    return ScatteringData(sdata)


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




def unpack_primitives(file: str | Path | TextIOWrapper) -> List[Primitive]:
    """Open a file a to parse primitive."""
    if isinstance(file, TextIOWrapper):
        lines = file.readlines()
    else:
        with open(file, 'r') as rdr:
            lines = rdr.readlines()
    return parsers.parse_primitive(lines)


def primitive_normal(primitive_paths: List[str]) -> Set[geom.Vector]:
    """Return a set of normal vectors given a list of primitive paths."""
    _primitives: List[Primitive] = []
    _normals: List[geom.Vector]
    for path in primitive_paths:
        _primitives.extend(unpack_primitives(path))
    _normals = [parsers.parse_polygon(prim.real_arg).normal() for prim in _primitives]
    return set(_normals)


def samp_dir(primlist: list) -> geom.Vector:
    """Calculate the primitives' average sampling
    direction weighted by area."""
    primlist = [p for p in primlist
                if p.ptype == 'polygon' or p.ptype == 'ring']
    normal_area = geom.Vector()
    for prim in primlist:
        polygon = parsers.parse_polygon(prim.real_arg)
        normal_area += polygon.normal().scale(polygon.area())
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
    xaxis = geom.Vector(1, 0, 0)
    yaxis = geom.Vector(0, 1, 0)
    norm_dir = samp_dir(primitives)
    if norm_dir not in (xaxis, xaxis.scale(-1)):
        upvect = xaxis.cross(norm_dir)
    else:
        upvect = yaxis.cross(norm_dir)
    return upvect


def neutral_plastic_prim(mod: str, ident: str, refl: float,
                         spec: float, rough: float) -> Primitive:
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
    err_msg = 'reflectance, speculariy, and roughness have to be 0-1'
    assert all(0 <= i <= 1 for i in [spec, refl, rough]), err_msg
    real_args = '5 {0} {0} {0} {1} {2} \n'.format(refl, spec, rough)
    return Primitive(mod, 'plastic', ident, '0', real_args)


def neutral_trans_prim(mod: str, ident: str, trans: float, refl: float,
                       spec: float, rough: float) -> Primitive:
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
    err_msg = 'reflectance, speculariy, and roughness have to be 0-1'
    assert all(0 <= i <= 1 for i in [spec, refl, rough]), err_msg
    real_args = "7 {0} {0} {0} {1} {2} {3} {4}".format(color, spec, rough, t_diff, tspec)
    return Primitive(mod, 'trans', ident, '0', real_args)


def color_plastic_prim(mod, ident, refl, red, green, blue, specu, rough):
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
    err_msg = 'reflectance, speculariy, and roughness have to be 0-1'
    assert all(0 <= i <= 1 for i in [specu, refl, rough]), err_msg
    red_eff = 0.3
    green_eff = 0.59
    blue_eff = 0.11
    weighted = red * red_eff + green * green_eff + blue * blue_eff
    matr = round(red / weighted * refl, 3)
    matg = round(green / weighted * refl, 3)
    matb = round(blue / weighted * refl, 3)
    real_args = '5 %s %s %s %s %s\n' % (matr, matg, matb, specu, rough)
    return Primitive(mod, 'plastic', ident, '0', real_args)


def glass_prim(mod, ident, tr, tg, tb, refrac=1.52):
    """Generate a glass material.

    Args:
        mod (str): modifier to the primitive
        ident (str): identifier to the primtive
        tr, tg, tb (float): transmmisivity in each channel (0.0 - 1.0)
        refrac (float): refraction index (default=1.52)
    Returns:
        material primtive (dict)

    """
    tmsv_red = tmit2tmis(tr)
    tmsv_green = tmit2tmis(tg)
    tmsv_blue = tmit2tmis(tb)
    real_args = '4 %s %s %s %s' % (tmsv_red, tmsv_green, tmsv_blue, refrac)
    return Primitive(mod, 'glass', ident, '0', real_args)


def bsdf_prim(mod, ident, xmlpath, upvec,
              pe=False, thickness=0.0, xform=None, real_args='0'):
    """Create a BSDF primtive."""
    str_args = '"{}" {} '.format(xmlpath, str(upvec))
    str_args_count = 5
    if pe:
        _type = 'aBSDF'
    else:
        str_args_count += 1
        str_args = '%s ' % thickness + str_args
        _type = 'BSDF'
    if xform is not None:
        str_args_count += len(xform.split())
        str_args += xform
    else:
        str_args += '.'
    str_args = '%s ' % str_args_count + str_args
    return Primitive(mod, _type, ident, str_args, real_args)


def lambda_calc(theta_lr, theta_up, nphi):
    """."""
    return ((math.cos(math.pi / 180 * theta_lr)**2 -
             math.cos(math.pi / 180 * theta_up)**2) * math.pi / nphi)


def angle_basis_coeff(basis: str) -> List[float]:
    '''Calculate klems basis coefficient'''
    ablist = ABASE_LIST[basis]
    lambdas = []
    for i in range(len(ablist) - 1):
        tu = ablist[i + 1][0]
        tl = ablist[i][0]
        np = ablist[i][1]
        lambdas.extend([lambda_calc(tl, tu, np) for _ in range(np)])
    return lambdas


def opt2str(opt: dict) -> str:
    out_str = ""
    for k, v in opt.items():
        if isinstance(v, list):
            val = ' '.join(map(str, v))
        else:
            val = v
        if k == 'vt' or k == 'f':
            out_str += "-{}{} ".format(k, val)
        elif k == 'hd':
            out_str += "-h "
        else:
            out_str += '-{} {} '.format(k, val)
    return out_str


def opt2list(opt: dict) -> list:
    out = []
    for k, v in opt.items():
        if isinstance(v, list):
            val = list(map(str, v))
        else:
            val = str(v)
        if k == 'vt' or k == 'f':
            out.append(f"-{k}{val}")
        elif k == 'hd':
            out.append("-h")
        else:
            if isinstance(val, list):
                out.extend(['-'+k, *val])
            else:
                out.extend(['-'+k, val])
    return out


class Reinsrc:
    """Calculate Reinhart/Treganza sampling directions.

    Direct translation of Radiance reinsrc.cal file.
    """

    TNAZ = [30, 30, 24, 24, 18, 12, 6]

    def __init__(self, mf: int):
        """Initialize with multiplication factor."""
        self.mf = mf
        self.rowMax = 7 * mf + 1
        self.rmax = self.raccum(self.rowMax)
        self.alpha = 90 / (mf * 7 + 0.5)

    def dir_calc(self, rbin: int, x1=0.5, x2=0.5) -> tuple:
        """Calculate the ray direction.

        Parameter:
            rbin: bin count
            x1, x2: sampling position (0.5, 0.5) is at the center
        Return:
            Sampling direction (tuple)
        """
        rrow = self.rowMax - \
            1 if rbin > (self.rmax - 0.5) else self.rfindrow(0, rbin)
        rcol = rbin - self.raccum(rrow) - 1
        razi_width = 2 * math.pi / self.rnaz(rrow)
        rah = self.alpha * math.pi / 180
        razi = (rcol + x2 - 0.5) * \
            razi_width if rbin > 0.5 else 2 * math.pi * x2
        ralt = (rrow + x1) * rah if rbin > 0.5 else math.asin(-x1)
        cos_alt = math.cos(ralt)
        if rbin < .5:
            romega = 2 * math.pi
        else:
            if self.rmax - .5 > rbin:
                romega = razi_width * (math.sin(rah * (rrow + 1)) - math.sin(rah * rrow))
            else:
                romega = 2 * math.pi * (1 - math.cos(rah / 2))
        dx = math.sin(razi) * cos_alt
        dy = math.cos(razi) * cos_alt
        dz = math.sin(ralt)
        return (dx, dy, dz, romega)

    def rnaz(self, r):
        """."""
        if r > (self.mf * 7 - .5):
            return 1
        else:
            return self.mf * self.TNAZ[int(math.floor((r + 0.5) / self.mf))]

    def raccum(self, r):
        """."""
        if r > 0.5:
            return self.rnaz(r - 1) + self.raccum(r - 1)
        else:
            return 0

    def rfindrow(self, r, rem):
        """."""
        if (rem - self.rnaz(r)) > 0.5:
            return self.rfindrow(r + 1, rem - self.rnaz(r))
        else:
            return r


class pt_inclusion(object):
    """Test whether a point is inside a polygon
    using winding number algorithm."""

    def __init__(self, polygon_pts):
        """Initialize the polygon."""
        self.pt_cnt = len(polygon_pts)
        polygon_pts.append(polygon_pts[0])
        self.polygon_pts = polygon_pts

    def isLeft(self, pt0, pt1, pt2):
        """Test whether a point is left to a line."""
        return (pt1.x - pt0.x) * (pt2.y - pt0.y) \
            - (pt2.x - pt0.x) * (pt1.y - pt0.y)

    def test_inside(self, pt):
        """Test if a point is inside the polygon."""
        wn = 0
        for i in range(self.pt_cnt):
            if self.polygon_pts[i].y <= pt.y:
                if self.polygon_pts[i + 1].y > pt.y:
                    if self.isLeft(self.polygon_pts[i],
                                   self.polygon_pts[i + 1], pt) > 0:
                        wn += 1
            else:
                if self.polygon_pts[i + 1].y <= pt.y:
                    if self.isLeft(self.polygon_pts[i],
                                   self.polygon_pts[i + 1], pt) < 0:
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
    plane_height = sum([i.z for i in vertices]) / len(vertices)
    imin, imax, jmin, jmax, _, _ = polygon.extreme()
    xlen_spc = ((imax - imin) / spacing)
    ylen_spc = ((jmax - jmin) / spacing)
    xstart = ((xlen_spc - int(xlen_spc) + 1)) * spacing / 2
    ystart = ((ylen_spc - int(ylen_spc) + 1)) * spacing / 2
    x0 = [x + xstart for x in frange_inc(imin, imax, spacing)]
    y0 = [x + ystart for x in frange_inc(jmin, jmax, spacing)]
    grid_dir = polygon.normal().reverse()
    grid_hgt = geom.Vector(0, 0, plane_height) + grid_dir.scale(height)
    raw_pts = [geom.Vector(round(i, 3), round(j, 3), round(grid_hgt.z, 3))
               for i in x0 for j in y0]
    scale_factor = 1 - 0.3/(imax - imin)  # scale boundary down .3 meter
    _polygon = polygon.scale(geom.Vector(
        scale_factor, scale_factor, 0), polygon.centroid())
    _vertices = _polygon.vertices
    if polygon.normal() == geom.Vector(0, 0, 1):
        pt_incls = pt_inclusion(_vertices)
    else:
        pt_incls = pt_inclusion(_vertices[::-1])
    _grid = [p for p in raw_pts if pt_incls.test_inside(p) > 0]
    grid = [p.to_list() + grid_dir.to_list() for p in _grid]
    return grid


def material_lib():
    mlib = []
    # carpet .2
    mlib.append(neutral_plastic_prim('void', 'carpet_20', .2, 0, 0))
    # Paint .5
    mlib.append(neutral_plastic_prim('void', 'white_paint_50', .5, 0, 0))
    # Paint .7
    mlib.append(neutral_plastic_prim('void', 'white_paint_70', .7, 0, 0))
    # Glass .6
    tmis = tmit2tmis(.6)
    mlib.append(glass_prim('void', 'glass_60', tmis, tmis, tmis))
    return mlib


def gen_blinds(depth, width, height, spacing, angle, curve, movedown):
    """Generate genblinds command for genBSDF."""
    nslats = int(round(height / spacing, 0))
    slat_cmd = "!genblinds blindmaterial blinds "
    slat_cmd += "{} {} {} {} {} {}".format(
        depth, width, height, nslats, angle, curve)
    slat_cmd += "| xform -rz -90 -rx -90 -t "
    slat_cmd += f"{-width/2} {-height/2} {-movedown}\n"
    return slat_cmd


def get_glazing_primitive(panes: List[PaneRGB]) -> Primitive:
    """Generate a BRTDfunc to represent a glazing system."""
    if len(panes) > 2:
        raise ValueError("Only double pane supported")
    name = "+".join([pane.measured_data.name for pane in panes])
    if len(panes) == 1:
        str_arg = "10 sr_clear_r sr_clear_g sr_clear_b "
        str_arg += "st_clear_r st_clear_g st_clear_b 0 0 0 glaze1.cal"
        coated_real = "1" if panes[0].measured_data.coated_side == 'front' else "-1"
        real_arg = f"19 0 0 0 0 0 0 0 0 0 {coated_real} "
        real_arg += " ".join(map(str, panes[0].glass_rgb)) + " "
        real_arg += " ".join(map(str, panes[0].coated_rgb)) + " "
        real_arg += " ".join(map(str, panes[0].trans_rgb))
    else:
        s12t_rgb = panes[0].trans_rgb
        s34t_rgb = panes[1].trans_rgb
        if panes[0].measured_data.coated_side == 'back':
            s2r_rgb = panes[0].coated_rgb
            s1r_rgb = panes[0].glass_rgb
        else:  # front or neither side coated
            s2r_rgb = panes[0].glass_rgb
            s1r_rgb = panes[0].coated_rgb
        if panes[1].measured_data.coated_side == 'back':
            s4r_rgb = panes[1].coated_rgb
            s3r_rgb = panes[1].glass_rgb
        else:  # front or neither side coated
            s4r_rgb = panes[1].glass_rgb
            s3r_rgb = panes[1].coated_rgb
        str_arg = "10\nif(Rdot,"
        str_arg += f"cr(fr({s4r_rgb[0]}),ft({s34t_rgb[0]}),fr({s2r_rgb[0]})),"
        str_arg += f"cr(fr({s1r_rgb[0]}),ft({s12t_rgb[0]}),fr({s3r_rgb[0]})))\n"
        str_arg += "if(Rdot,"
        str_arg += f"cr(fr({s4r_rgb[1]}),ft({s34t_rgb[1]}),fr({s2r_rgb[1]})),"
        str_arg += f"cr(fr({s1r_rgb[1]}),ft({s12t_rgb[1]}),fr({s3r_rgb[1]})))\n"
        str_arg += "if(Rdot,"
        str_arg += f"cr(fr({s4r_rgb[2]}),ft({s34t_rgb[2]}),fr({s2r_rgb[2]})),"
        str_arg += f"cr(fr({s1r_rgb[2]}),ft({s12t_rgb[2]}),fr({s3r_rgb[2]})))\n"
        str_arg += f"ft({s34t_rgb[0]})*ft({s12t_rgb[0]})\n"
        str_arg += f"ft({s34t_rgb[1]})*ft({s12t_rgb[1]})\n"
        str_arg += f"ft({s34t_rgb[2]})*ft({s12t_rgb[2]})\n"
        str_arg += "0 0 0 glaze2.cal"
        real_arg = "9 0 0 0 0 0 0 0 0 0"
    return Primitive("void", "BRTDfunc", name, str_arg, real_arg)


def flush_corner_rays_cmd(ray_cnt: int, xres: int) -> list:
    """Flush the corner rays from a fisheye view

    Args:
        ray_cnt: ray count;
        xres: resolution of the square image;

    Returns:
        Command to generate cropped rays

    """
    cmd = ["rcalc", "-if6", "-of", '-e', f"DIM:{xres};CNT:{ray_cnt}", '-e', "pn=(recno-1)/CNT+.5", '-e', "frac(x):x-floor(x)", '-e', "xpos=frac(pn/DIM);ypos=pn/(DIM*DIM)",'-e', "incir=if(.25-(xpos-.5)*(xpos-.5)-(ypos-.5)*(ypos-.5),1,0)", '-e', "$1=$1;$2=$2;$3=$3;$4=$4*incir;$5=$5*incir;$6=$6*incir"]
    return cmd
