"""Utility functions."""

import argparse
from typing import NamedTuple, List
from dataclasses import dataclass, field
import glob
import logging
import math
import os
import re
import subprocess as sp
import multiprocessing as mp
from frads import radgeom, radmtx, util
logger = logging.getLogger("frads.radutil")
try:
    import numpy as np
    numpy_installed = True
except ModuleNotFoundError:
    logger.info("Numpy not installed")
    numpy_installed = False


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


class Primitive(NamedTuple):
    modifier: str
    ptype: str
    identifier: str
    str_arg: str
    real_arg: str
    int_arg: str = '0'

    def __repr__(self) -> str:
        output = f"{self.modifier} {self.ptype} {self.identifier} "
        output += f"{self.str_arg} {self.int_arg} {self.real_arg} "
        return output

    def __str__(self) -> str:
        if '' in (self.modifier, self.ptype, self.identifier):
            return ''
        output = f"\n{self.modifier} {self.ptype} {self.identifier}\n"
        output += f"{self.str_arg}\n{self.int_arg}\n{self.real_arg}\n"
        return output

@dataclass
class ScatteringData:
    sdata: List[List[float]]
    ncolumn: int = field(init=False)
    nrow: int = field(init=False)
    basis: str = field(init=False)

    def __post_init__(self):
        self.ncolumn = len(self.sdata[0])
        self.nrow = len(self.sdata)
        self.basis = BASIS_DICT[str(self.ncolumn)]

    def to_bsdf(self):
        lambdas = angle_basis_coeff(self.basis)
        element_divi = lambda x,y: x/y
        bsdf = [list(map(element_divi, i, lambdas)) for i in self.sdata]
        return BSDFData(bsdf)


    def __repr__(self) -> str:
        out = ''
        for row in self.sdata:
            for val in row:
                string = '%07.5f' % val
                out += string + '\t'
            out += '\n'
        return out

    def __str__(self) -> str:
        out = '#?RADIANCE\nNCOMP=3\n'
        out += 'NROWS=%d\nNCOLS=%d\n' % (self.nrow, self.ncolumn)
        out += 'FORMAT=ascii\n\n'
        for row in self.sdata:
            for val in row:
                string = '\t'.join(['%07.5f' % val] * 3)
                out += string + '\t'
            out += '\n'
        return out


@dataclass
class BSDFData:
    bsdf: List[List[float]]
    ncolumn: int = field(init=False)
    nrow: int = field(init=False)
    basis: str = field(init=False)

    def __post_init__(self):
        self.ncolumn = len(self.bsdf[0])
        self.nrow = len(self.bsdf)
        self.basis = BASIS_DICT[str(self.ncolumn)]

    def to_sdata(self) -> ScatteringData:
        lambdas = angle_basis_coeff(self.basis)
        element_mult = lambda x,y: x/y
        sdata = [list(map(element_mult, i, lambdas))
                 for i in self.bsdf]
        return ScatteringData(sdata)


@dataclass(frozen=True)
class RadMatrix:
    tf: ScatteringData
    tb: ScatteringData


def parse_primitive(lines: list) -> list:
    """Parse Radiance primitives inside a file path into a list of dictionary.
    Args:
        lines: list of lines as strings

    Returns:
        list of primitives as dictionaries
    """
    # Expand in-line commands
    cmd_lines = [(idx, line) for idx, line in enumerate(lines)
                 if line.startswith('!')]
    cmd_results = []
    for cmd in cmd_lines:
        cmd_results.append(
            sp.run(cmd[1][1:], shell=True, stdout=sp.PIPE)
            .stdout.decode().splitlines())
    counter = 0
    for idx, item in enumerate(cmd_lines):
        counter += item[0]
        lines[counter:counter+1] = cmd_results[idx]
        counter += len(cmd_results[idx]) - 1 - item[0]

    content = ' '.join([i.strip() for i in lines
                        if i.strip() != '' and i[0] != '#']).split()
    primitives: List[Primitive] = []
    idx = 0
    while idx < len(content):
        _modifier = content[idx]
        _type = content[idx + 1]
        if _type == 'alias':
            _name_to = content[idx + 2]
            _name_from = content[idx + 3]
            primitives.append(Primitive(_modifier, _type, _name_to, _name_from, '', int_arg=''))
            idx += 4
            continue
        _identifier = content[idx + 2]
        str_arg_cnt = int(content[idx + 3])
        _str_args = ' '.join(content[idx + 3:idx + 4 + str_arg_cnt])
        idx += 5 + str_arg_cnt
        real_arg_cnt = int(content[idx])
        _real_args = ' '.join(content[idx:idx + 1 + real_arg_cnt])
        idx += real_arg_cnt + 1
        primitives.append(Primitive(
            _modifier, _type, _identifier, _str_args, _real_args))
    return primitives


def unpack_primitives(fpath: str) -> List[Primitive]:
    """Open a file a to parse primitive."""
    with open(fpath, 'r') as rdr:
        return parse_primitive(rdr.readlines())


def parse_polygon(real_arg: str) -> radgeom.Polygon:
    """Parse real arguments to polygon.
    Args:
        primitive: a dictionary object containing a primitive

    Returns:
        modified primitive
    """
    real_args = real_arg.split()
    coords = [float(i) for i in real_args[1:]]
    arg_cnt = int(real_args[0])
    vertices = [radgeom.Vector(*coords[i:i + 3]) for i in range(0, arg_cnt, 3)]
    return radgeom.Polygon(vertices)


def polygon2prim(polygon: radgeom.Polygon,
                 modifier: str, identifier: str) -> Primitive:
    """Generate a primitive from a polygon."""
    return Primitive(modifier, 'polygon', identifier, '0', polygon.to_real())


def samp_dir(primlist: list) -> radgeom.Vector:
    """Calculate the primitives' average sampling
    direction weighted by area."""
    primlist = [p for p in primlist
                if p.ptype == 'polygon' or p.ptype == 'ring']
    normal_area = radgeom.Vector()
    for prim in primlist:
        polygon = parse_polygon(prim.real_arg)
        normal_area += polygon.normal().scale(polygon.area())
    sdir = normal_area.scale(1.0 / len(primlist))
    sdir = sdir.normalize()
    return sdir


def up_vector(primitives: list) -> radgeom.Vector:
    """Define the up vector given primitives.

    Args:
        primitives: list of dictionary (primitives)

    Returns:
        returns a str as x,y,z

    """
    xaxis = radgeom.Vector(1, 0, 0)
    yaxis = radgeom.Vector(0, 1, 0)
    norm_dir = samp_dir(primitives)
    if norm_dir not in (xaxis, xaxis.scale(-1)):
        upvect = norm_dir.cross(xaxis)
    else:
        upvect = norm_dir.cross(yaxis)
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
    tmsv_red = util.tmit2tmis(tr)
    tmsv_green = util.tmit2tmis(tg)
    tmsv_blue = util.tmit2tmis(tb)
    real_args = '4 %s %s %s %s' % (tmsv_red, tmsv_green, tmsv_blue, refrac)
    return Primitive(mod, 'glass', ident, '0', real_args)


def bsdf_prim(mod, ident, xmlpath, upvec,
              pe=False, thickness=0.0, xform=None, real_args='0'):
    """Create a BSDF primtive."""
    str_args = '{} {} '.format(xmlpath, str(upvec))
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
        dx = math.sin(razi) * cos_alt
        dy = math.cos(razi) * cos_alt
        dz = math.sin(ralt)
        return (dx, dy, dz)

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


def gen_grid(polygon: radgeom.Polygon, height: float, spacing: float) -> list:
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
    x0 = [x + xstart for x in util.frange_inc(imin, imax, spacing)]
    y0 = [x + ystart for x in util.frange_inc(jmin, jmax, spacing)]
    grid_dir = polygon.normal().reverse()
    grid_hgt = radgeom.Vector(0, 0, plane_height) + grid_dir.scale(height)
    raw_pts = [radgeom.Vector(round(i, 3), round(j, 3), round(grid_hgt.z, 3))
               for i in x0 for j in y0]
    scale_factor = 1 - 0.3/(imax- imin) # scale boundary down .3 meter
    _polygon = polygon.scale(radgeom.Vector(scale_factor, scale_factor, 0), polygon.centroid())
    _vertices = _polygon.vertices
    if polygon.normal() == radgeom.Vector(0, 0, 1):
        pt_incls = pt_inclusion(_vertices)
    else:
        pt_incls = pt_inclusion(_vertices[::-1])
    _grid = [p for p in raw_pts if pt_incls.test_inside(p) > 0]
    grid = [p.to_list() + grid_dir.to_list() for p in _grid]
    return grid


def gengrid():
    """Commandline program for generating a grid of sensor points."""
    parser = argparse.ArgumentParser(
        prog='gengrid',
        description='Generate an equal-spaced sensor grid based on a surface.')
    parser.add_argument('surface', help='surface file path')
    parser.add_argument('spacing', type=float)
    parser.add_argument('height', type=float)
    parser.add_argument('-op', action='store_const', const='', default=True)
    args = parser.parse_args()
    prims = unpack_primitives(args.surface)
    polygon_prims = [prim for prim in prims if prim.ptype=='polygon']
    polygon = parse_polygon(polygon_prims[0].real_arg)
    if args.op:
        polygon = polygon.flip()
    grid_list = gen_grid(polygon, args.height, args.spacing)
    grid_str = '\n'.join([' '.join(map(str, row)) for row in grid_list])
    print(grid_str)


def material_lib():
    mlib = []
    # carpet .2
    mlib.append(neutral_plastic_prim('void', 'carpet_20', .2, 0, 0))
    # Paint .5
    mlib.append(neutral_plastic_prim('void', 'white_paint_50', .5, 0, 0))
    # Paint .7
    mlib.append(neutral_plastic_prim('void', 'white_paint_70', .7, 0, 0))
    # Glass .6
    tmis = util.tmit2tmis(.6)
    mlib.append(glass_prim('void', 'glass_60', tmis, tmis, tmis))
    return mlib


def pcomb(inputs):
    """Image operations with pcomb.
    Parameter: inputs,
        e.g: ['img1.hdr', '+', img2.hdr', '-', 'img3.hdr', 'output.hdr']
    """
    input_list = inputs[:-1]
    out_dir = inputs[-1]
    component_idx = range(0, len(input_list), 2)
    components = [input_list[i] for i in component_idx]
    color_op_list = []
    for c in 'rgb':
        color_op = input_list[:]
        for i in component_idx:
            color_op[i] = '%si(%d)' % (c, i/2+1)
        cstr = '%so=%s' % (c, ''.join(color_op))
        color_op_list.append(cstr)
    rgb_str = ';'.join(color_op_list)
    cmd = ['pcomb', '-e', '%s' % rgb_str]
    img_name = util.basename(input_list[0], keep_ext=True)
    for c in components:
        cmd.append('-o')
        cmd.append(c)
    res = util.spcheckout(cmd)
    with open(os.path.join(out_dir, img_name), 'wb') as wtr:
        wtr.write(res)


def dctsop(inputs, out_dir, nproc=1):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nproc = mp.cpu_count() if nproc is None else nproc
    process = mp.Pool(nproc)
    assert len(inputs) > 1
    mtx_inp = inputs[:-1]
    sky_dir = inputs[-1]
    sky_files = (os.path.join(sky_dir, i) for i in os.listdir(sky_dir))
    grouped = [mtx_inp+[skv] for skv in sky_files]
    [sub.append(out_dir) for sub in grouped]
    process.map(dctimestep, grouped)


def pcombop(inputs, out_dir, nproc=1):
    """
    inputs(list): e.g.[inpdir1,'+',inpdir2,'-',inpdir3]
    out_dir: output directory
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nproc = mp.cpu_count() if nproc is None else nproc
    process = mp.Pool(nproc)
    assert len(inputs) > 2
    assert len(inputs) % 2 == 1
    inp_dir = [inputs[i] for i in range(0, len(inputs), 2)]
    inp_cnt = len(inp_dir)
    ops = [inputs[i] for i in range(1, len(inputs), 2)]
    inp_lists = []
    for i in inp_dir:
        if os.path.isdir(i):
            inp_lists.append(glob.glob(os.path.join(i, '*.hdr')))
        else:
            inp_lists.append(i)
    inp_lists_full = []
    for i in range(inp_cnt):
        if os.path.isdir(inp_dir[i]):
            inp_lists_full.append(inp_lists[i])
        else:
            inp_lists_full.append(inp_dir[i])
    max_len = sorted([len(i) for i in inp_lists_full if type(i) == list])[0]
    for i in range(len(inp_lists_full)):
        if type(inp_lists_full[i]) is not list:
            inp_lists_full[i] = [inp_lists_full[i]] * max_len
    equal_len = all(len(i) == len(inp_lists_full[0]) for i in inp_lists_full)
    if not equal_len:
        logger.warning("Input directories don't the same number of files")
    grouped = [list(i) for i in zip(*inp_lists_full)]
    [sub.insert(i, ops[int((i-1)/2)])
     for sub in grouped for i in range(1, len(sub)+1, 2)]
    [sub.append(out_dir) for sub in grouped]
    process.map(pcomb, grouped)


def dctimestep(input_list):
    """Image operations in forms of Vs, VDs, VTDs, VDFs, VTDFs."""
    inputs = input_list[:-1]
    out_dir = input_list[-1]
    inp_dir_count = len(inputs)
    sky = input_list[-2]
    img_name = util.basename(sky)
    out_path = os.path.join(out_dir, img_name)

    if inputs[1].endswith('.xml') is False\
            and inp_dir_count > 2 and os.path.isdir(inputs[0]):
        combined = "'!rmtxop %s" % (' '.join(inputs[1:-1]))
        img = [i for i in os.listdir(inputs[0]) if i.endswith('.hdr')][0]
        str_count = len(img.split('.hdr')[0])  # figure out unix %0d string
        appendi = r"%0"+"%sd.hdr" % (str_count)
        new_inp_dir = [os.path.join(inputs[0], appendi), combined]
        cmd = "dctimestep -oc %s %s' > %s.hdr" \
            % (' '.join(new_inp_dir), sky, out_path)

    else:
        if not os.path.isdir(inputs[0]) and not inputs[1].endswith('.xml'):
            combined = os.path.join(os.path.dirname(inputs[0]), "tmp.vfdmtx")
            stderr = combine_mtx(inputs[:-1], combined)
            if stderr != "":
                print(stderr)
                return
            inputs_ = [combined]

        elif inp_dir_count == 5:
            combined = os.path.join(os.path.dirname(inputs[2]), "tmp.fdmtx")
            stderr = combine_mtx(inputs[2:4], combined)
            if stderr != "":
                print(stderr)
                return
            inputs_ = [inputs[0], inputs[1], combined]

        else:
            inputs_ = inputs[:-1]

        if os.path.isdir(inputs[0]):
            img = [i for i in os.listdir(inputs[0])
                   if i.endswith('.hdr')][0]
            str_count = len(img.split('.hdr')[0])
            appendi = r"%0"+"%sd.hdr" % (str_count)
            inputs_[0] = os.path.join(inputs[0], appendi)
            out_ext = ".hdr"
        else:
            out_ext = ".dat"

        input_string = ' '.join(inputs_)
        out_path = out_path + out_ext
        cmd = "dctimestep %s %s > %s" % (input_string, sky, out_path)
    sp.call(cmd, shell=True)


def rpxop():
    """Operate on input directories given a operation type."""
    PROGRAM_SCRIPTION = "Image operations with parallel processing"
    parser = argparse.ArgumentParser(
        prog='imgop', description=PROGRAM_SCRIPTION)
    parser.add_argument('-t', type=str, required=True,
                        choices=['dcts', 'pcomb'],
                        help='operation types: {pcomb|dcts}')
    parser.add_argument('-i', type=str, required=True,
                        nargs="+", help='list of inputs')
    parser.add_argument('-o', type=str, required=True, help="output directory")
    parser.add_argument('-n', type=int, help='number of processors to use')
    args = parser.parse_args()
    if args.t == "pcomb":
        pcombop(args.i, args.o, nproc=args.n)
    elif args.t == 'dcts':
        dctsop(args.i, args.o, nproc=args.n)


def combine_mtx(mtxs, out_dir):
    """."""
    cmd = "rmtxop"
    args = " -ff %s > %s" % (" ".join(mtxs), out_dir)
    process = sp.Popen(cmd + args, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    _, stderr = process.communicate()
    return stderr


def header_parser(header):
    """Parse for a matrix file header."""
    nrow = int(re.search('NROWS=(.*)\n', header).group(1))
    ncol = int(re.search('NCOLS=(.*)\n', header).group(1))
    ncomp = int(re.search('NCOMP=(.*)\n', header).group(1))
    return nrow, ncol, ncomp


def mtx2nparray(datastr):
    """Convert Radiance 3-channel matrix file to numpy array."""
    raw = datastr.split('\n\n')
    header = raw[0]
    nrow, ncol, ncomp = header_parser(header)
    data = raw[1].splitlines()
    rdata = np.array([i.split()[::ncomp] for i in data], dtype=float)
    gdata = np.array([i.split()[1::ncomp] for i in data], dtype=float)
    bdata = np.array([i.split()[2::ncomp] for i in data], dtype=float)
    assert len(bdata) == nrow
    assert len(bdata[0]) == ncol
    return rdata, gdata, bdata


def smx2nparray(datastr):
    raw = datastr.split('\n\n')
    header = raw[0]
    nrow, ncol, ncomp = header_parser(header)
    data = [i.splitlines() for i in raw[1:] if i != '']
    rdata = np.array([[i.split()[::ncomp][0] for i in row] for row in data],
                     dtype=float)
    gdata = np.array([[i.split()[1::ncomp][0] for i in row] for row in data],
                     dtype=float)
    bdata = np.array([[i.split()[2::ncomp][0] for i in row] for row in data],
                     dtype=float)
    if ncol == 1:
        assert np.size(bdata, 1) == nrow
        assert np.size(bdata, 0) == ncol
        rdata = rdata.flatten()
        gdata = gdata.flatten()
        bdata = bdata.flatten()
    else:
        assert np.size(bdata, 0) == nrow
        assert np.size(bdata, 1) == ncol
    return rdata, gdata, bdata


def mtxmult(mtxs):
    if numpy_installed:
        return numpy_mtxmult(mtxs)
    else:
        return rmtxop(mtxs)


def numpy_mtxmult(mtxs):
    """Matrix multiplication with Numpy."""
    resr = np.linalg.multi_dot([mat[0] for mat in mtxs]) * .265
    resg = np.linalg.multi_dot([mat[1] for mat in mtxs]) * .67
    resb = np.linalg.multi_dot([mat[2] for mat in mtxs]) * .065
    return resr + resg + resb


def rmtxop(*mtxs):
    cmd = ["rmtxop", *mtxs]
    return util.spcheckout(cmd)


def gen_blinds(depth, width, height, spacing, angle, curve, movedown):
    """Generate genblinds command for genBSDF."""
    nslats = int(round(height / spacing, 0))
    slat_cmd = "!genblinds blindmaterial blinds "
    slat_cmd += "{} {} {} {} {} {}".format(
        depth, width, height, nslats, angle, curve)
    slat_cmd += "| xform -rz -90 -rx -90 -t "
    slat_cmd += f"{-width/2} {-height/2} {-movedown}\n"
    return slat_cmd


def analyze_window(window_prim, movedown):
    """Parse window primitive and prepare for genBSDF."""
    window_polygon = window_prim['polygon']
    vertices = window_polygon.vertices
    assert len(vertices) == 4, "4-sided polygon only"
    window_center = window_polygon.centroid()
    window_normal = window_polygon.normal()
    window_normal
    dim1 = vertices[0] - vertices[1]
    dim2 = vertices[1] - vertices[2]
    if dim1.normalize() in (radgeom.Vector(0, 0, 1), radgeom.Vector(0, 0, -1)):
        height = dim1.length
        width = dim2.length
    else:
        height = dim2.length
        width = dim1.length
    _south = radgeom.Vector(0, -1, 0)
    angle2negY = window_normal.angle_from(_south)
    rotate_window = window_center.rotate_3d(
        radgeom.Vector(0, 0, 1), angle2negY).rotate_3d(
            radgeom.Vector(1, 0, 0), math.pi/2)
    translate = radgeom.Vector(0, 0, -movedown) - rotate_window
    return height, width, angle2negY, translate


def varays():
    """Commandline utility program for generating circular fisheye rays."""
    aparser = argparse.ArgumentParser(
        prog='varays',
        description='Generate a fisheye view rays with blackedout corners')
    aparser.add_argument(
        '-x', required=True, help='square image resolution')
    aparser.add_argument('-c', default='1', help='Ray count')
    aparser.add_argument('-vf', required=True, help='View file path')
    args = aparser.parse_args()
    cmd = "vwrays -ff -x {0} -y {0} ".format(args.x)
    if args.c != '1':
        cmd += '-c {} -pj 0.7 '.format(args.c)
    cmd += f"-vf {args.vf} | "
    cmd += radmtx.Sender.crop2circle(args.c, args.x)
    sp.run(cmd, shell=True)
