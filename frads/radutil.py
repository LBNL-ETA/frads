"""Utility functions."""

# Taoning.W

import argparse
from frads import radgeom
import math
import os

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


def test_environ(cmd):
    """Test if a list of programs are in the environment path."""
    return any(
        os.access(os.path.join(path, cmd), os.X_OK)
        for path in os.environ['PATH'].split(os.pathsep))


def parse_decor(fpath):
    with open(fpath, 'r') as rd:
        content = rd.readlines()
    decor = [l for l in content if l.startswith('#@')]
    cmd = [l for l in content if l.startswith('!')]
    return decor, cmd


def parse_primitive(content):
    """Parse Radiance primitives inside a file path into a list of dictionary."""
    content = ' '.join([
        i.strip() for i in content
        if i.strip() != '' and i[0] != '#' and i[0] != '!'
    ]).split()
    primitives = []
    idx = 0
    while idx < len(content):
        primitive = {}
        primitive['modifier'] = content[idx]
        primitive['type'] = content[idx + 1]
        primitive['identifier'] = content[idx + 2]
        str_arg_cnt = int(content[idx + 3])
        primitive['str_args'] = ' '.join(content[idx + 3:idx + 4 +
                                                 str_arg_cnt])
        primitive['int_arg'] = content[idx + 4 + str_arg_cnt]
        idx += 5 + str_arg_cnt
        real_arg_cnt = int(content[idx])
        primitive['real_args'] = ' '.join(content[idx:idx + 1 + real_arg_cnt])
        idx += real_arg_cnt + 1
        if primitive['type'] == 'polygon':
            primitive = parse_polygon(primitive)
        primitives.append(primitive)
    return primitives


def parse_polygon(primitive):
    assert primitive['type'] == 'polygon'
    real_args = primitive['real_args'].split()
    coords = [float(i) for i in real_args[1:]]
    arg_cnt = int(real_args[0])
    vertices = [radgeom.Vector(*coords[i:i + 3]) for i in range(0, arg_cnt, 3)]
    primitive['polygon'] = radgeom.Polygon(vertices)
    return primitive


def parse_vu(vu_str):
    """Parse view string into a dictionary."""
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
    args, _ = vparser.parse_known_args(args_list)
    view_dict = vars(args)
    view_dict['vt'] = view_dict['vt'][-1]
    view_dict = {k: v for (k, v) in view_dict.items() if v is not None}
    return view_dict


def parse_opt(opt_str):
    args_list = opt_str.strip().split()
    oparser = argparse.ArgumentParser()
    oparser.add_argument('-I', action='store_const', const='', default=None)
    oparser.add_argument('-V', action='store_const', const='', default=None)
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
    oparser.add_argument('-i', action='store_const', const='', default=None)
    oparser.add_argument('-lr', type=int)
    oparser.add_argument('-lw', type=float)
    oparser.add_argument('-n', type=int)
    oparser.add_argument('-ss', type=int)
    oparser.add_argument('-st', type=int)
    oparser.add_argument('-u', action='store_const', const='', default=None)
    args, _ = oparser.parse_known_args(args_list)
    opt_dict = vars(args)
    opt_dict = {k: v for (k, v) in opt_dict.items() if v is not None}
    return opt_dict


def put_primitive(prim):
    """Convert primitives from a dictionary into a string for writing."""
    if type(prim) is str:
        ostring = prim + os.linesep
    else:
        ostring = "{modifier} {type} {identifier}\
        \n{str_args}\n{int_arg}\n{real_args}\n\n".format(**prim)
    return ostring


def surface_normal(prim):
    """Get the surface normal from a polygon primitive."""
    if prim['type'] == 'polygon':
        return prim['polygon'].normal()
    elif prim['type'] == 'ring':
        real_args = prim['real_args'].split()
        return radgeom.Vector(*[float(i) for i in real_args[4:7]])


def surface_area(prim):
    """Get the surface area from a primitive."""
    if prim['type'] == 'polygon':
        return prim['polygon'].area()
    elif prim['type'] == 'ring':
        real_args = prim['real_args'].split()
        inner_radi = float(real_args[-2])
        outter_radi = float(real_args[-1])
        return math.pi * (outter_radi**2 - inner_radi**2)


def samp_dir(plist):
    """Calculate the primitives' average sampling direction weighted by area."""
    normal_areas = []
    plist = [p for p in plist if p['type'] == 'polygon' or p['type'] == 'ring']
    normal_area = radgeom.Vector()
    for p in plist:
        normal = surface_normal(p)
        area = surface_area(p)
        normal_area += normal.scale(area)
    samp_dir = normal_area.scale(1.0 / len(plist))
    samp_dir = samp_dir.unitize()
    return samp_dir


def up_vector(primitives):
    samp = [round(i, 1) for i in samp_dir(primitives).to_list()]
    abs_dir = [abs(i) for i in samp]
    if abs_dir == [0.0, 0.0, 1.0]:
        upvect = '+Y'
    else:
        upvect = '+Z'
    return upvect


def polygon_center(pts):
    """Calculate the center from a list of points."""
    pt_num = len(pts)
    xsum = 0
    ysum = 0
    zsum = 0
    for p in pts:
        xsum += p[0]
        ysum += p[1]
        zsum += p[2]
    xc = xsum / pt_num
    yc = ysum / pt_num
    zc = zsum / pt_num
    center = [xc, yc, zc]
    return center


def getbbox(polygon_list, offset=0.0):
    """Get boundary from a list of primitives."""
    extreme_list = [p.extreme() for p in polygon_list]
    lim = [i for i in zip(*extreme_list)]
    xmin = min(lim[0]) - offset
    xmax = max(lim[1]) + offset
    ymin = min(lim[2]) - offset
    ymax = max(lim[3]) + offset
    zmin = min(lim[4]) - offset
    zmax = max(lim[5]) + offset

    fp1 = radgeom.Point(xmin, ymin, zmin)
    fp2 = radgeom.Point(xmax, ymin, zmin)
    fp3 = radgeom.Point(xmax, ymax, zmin)
    fpg = radgeom.Rectangle3P(fp1, fp2, fp3)

    cp1 = radgeom.Point(xmin, ymin, zmax)
    cp2 = radgeom.Point(xmax, ymin, zmax)
    cp3 = radgeom.Point(xmax, ymax, zmax)
    cpg = radgeom.Rectangle3P(cp3, cp2, cp1)

    swpg = radgeom.Rectangle3P(cp2, fp2, fp1)

    ewpg = radgeom.Rectangle3P(fp3, fp2, cp2)

    s2n_vec = radgeom.Vector(0, ymax - ymin, 0)
    nwpg = radgeom.Polygon([v + s2n_vec for v in swpg.vertices]).flip()

    e2w_vec = radgeom.Vector(xmax - xmin, 0, 0)
    wwpg = radgeom.Polygon([v - e2w_vec for v in ewpg.vertices]).flip()

    return [fpg, cpg, ewpg, swpg, wwpg, nwpg]


def plastic_prim(mod, ident, refl, red, green, blue, specu, rough):
    """Generate a plastic material.

    Inputs:
        mod(str): modifier to the primitive
        ident(str): identifier to the primitive
        refl (float): measured reflectance (0.0 - 1.0)
        red; green; blue (int): rgb values (0 - 255)
        specu (float): material specularity (0.0 - 1.0)
        rough (float): material roughness (0.0 - 1.0)
    Return:
        material primtive (dict)

    """
    err_msg = 'reflectance, speculariy, and roughness have to be 0-1'
    assert all(0 <= i <= 1 for i in [specu, refl, rough]), err_msg
    prim = {'type': 'plastic', 'int_arg': '0', 'str_args': '0'}
    red_eff = 0.3
    green_eff = 0.59
    blue_eff = 0.11
    weighted = red * red_eff + green * green_eff + blue * blue_eff
    matr = round(red / weighted * refl, 3)
    matg = round(green / weighted * refl, 3)
    matb = round(blue / weighted * refl, 3)
    prim['modifier'] = mod
    prim['identifier'] = ident
    real_args = '5 %s %s %s %s %s\n' % (matr, matg, matb, specu, rough)
    prim['real_args'] = real_args

    return prim


def glass_prim(mod, ident, tr, tg, tb, refrac=1.52):
    """Generate a glass material.

    Inputs:
        mod (str): modifier to the primitive
        ident (str): identifier to the primtive
        tr, tg, tb (float): transmmisivity in each channel (0.0 - 1.0)
        refrac (float): refraction index (default=1.52)
    Return:
        material primtive (dict)

    """

    def convert(tmit):
        return round(
            (math.sqrt(0.8402528435 + 0.0072522239 * tmit**2) - 0.9166530661) /
            0.0036261119 / tmit, 3)

    prim = {'type': 'glass', 'int_arg': '0', 'str_args': '0'}
    tmsv_red = convert(tr)
    tmsv_green = convert(tg)
    tmsv_blue = convert(tb)
    prim['modifier'] = mod
    prim['identifier'] = ident
    real_args = '4 %s %s %s %s' % (tmsv_red, tmsv_green, tmsv_blue, refrac)
    prim['real_args'] = real_args
    return prim


def bsdf_prim(mod, ident, xml_fpath, up_vec, thickness=0.0, real_args='0'):
    """."""
    prim = {
        'modifier': mod,
        'identifier': ident,
        'type': 'BSDF',
        'int_arg': '0',
        'real_args': real_args
    }
    str_args = '6 %d %s %s .' \
        % (thickness, xml_fpath, ' '.join(map(str, up_vec)))
    prim['str_args'] = str_args

    return prim


def parse_idf(content):
    """Parse an IDF file into a dictionary."""
    sections = content.rstrip().split(';')
    sub_sections = []
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

    obj_dict = {}
    for sec in sub_sections:
        obj_dict[sec[0].lower()] = []
    for sec in sub_sections:
        obj_dict[sec[0].lower()].append(sec[1:])

    return obj_dict


def lambda_calc(theta_lr, theta_up, nphi):
    """."""
    return ((math.sin(math.pi / 180 * theta_up)**2 -
             math.sin(math.pi / 180 * theta_lr)**2) * math.pi / nphi)


def angle_basis_coeff(basis):
    '''Calculate klems basis coefficient'''
    ablist = ABASE_LIST[basis]
    lambdas = []
    for i in range(len(ablist) - 1):
        tu = ablist[i + 1][0]
        tl = ablist[i][0]
        np = ablist[i][1]
        lambdas.extend([lambda_calc(tl, tu, np) for n in range(np)])
    return lambdas


def dhi2dni(GHI, DHI, alti):
    """Calculate direct normal from global horizontal and diffuse horizontal."""
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


def basename(fpath):
    """."""
    return os.path.splitext(os.path.basename(fpath))[0]


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def silent_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError as e:
        print(e)
        pass


def opt2str(opt):
    assert isinstance(opt, dict)
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


class reinsrc(object):
    """Calculate Reinhart/Treganza sampling directions.

    Direct translation of Radiance reinsrc.cal file.
    """

    TNAZ = [30, 30, 24, 24, 18, 12, 6]

    def __init__(self, mf):
        """Initialize with multiplication factor."""
        self.mf = mf
        self.rowMax = 7 * mf + 1
        self.rmax = self.raccum(self.rowMax)
        self.alpha = 90 / (mf * 7 + 0.5)

    def dir_calc(self, rbin, x1=0.5, x2=0.5):
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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e, 'ignored')


def check_fresh(path1, path2):
    time1 = os.path.getmtime(path1)
    time2 = os.path.getmtime(path2)
    return time1 > time2


class pt_inclusion(object):
    """testing whether a point is inside a polygon using winding number algorithm."""

    def __init__(self, polygon_pts):
        """Initialize the polygon."""
        self.pt_cnt = len(polygon_pts)
        polygon_pts.append(polygon_pts[0])
        self.polygon_pts = polygon_pts

    def isLeft(self, pt0, pt1, pt2):
        """Test whether a point is left to a line."""
        return (pt1[0] - pt0[0]) * (pt2[1] - pt0[1]) \
            - (pt2[0] - pt0[0]) * (pt1[1] - pt0[1])

    def test_inside(self, pt):
        """Test if a point is inside the polygon."""
        wn = 0
        for i in range(self.pt_cnt):
            if self.polygon_pts[i][1] <= pt[1]:
                if self.polygon_pts[i + 1][1] > pt[1]:
                    if self.isLeft(self.polygon_pts[i],
                                   self.polygon_pts[i + 1], pt) > 0:
                        wn += 1
            else:
                if self.polygon_pts[i + 1][1] <= pt[1]:
                    if self.isLeft(self.polygon_pts[i],
                                   self.polygon_pts[i + 1], pt) < 0:
                        wn -= 1
        return wn


def gen_grid(polygon, height, spacing, op=False):
    """Generate a grid of points for orthogonal planar surfaces.

    Parameters:
            polygon: a polygon object
            height: points' distance from the surface in its normal direction
            spacing: distance between the grid points
            visualize: set to True to visualize the resulting grid points
    Output:
            write the point file to pts directory

    """
    #name = polygon['identifier']
    #modifier = polygon['modifier']
    #polygon = polygon['polygon']
    normal = polygon.normal()
    abs_norm = [abs(i) for i in normal.to_list()]
    drop_idx = abs_norm.index(max(abs_norm))
    pg_pts = [i.to_list() for i in polygon.vertices]
    pt_cnt = len(pg_pts)
    plane_height = sum([i[drop_idx] for i in pg_pts]) / pt_cnt
    [i.pop(drop_idx) for i in pg_pts]  # dimension reduction
    _ilist = [i[0] for i in pg_pts]
    _jlist = [i[1] for i in pg_pts]
    imax = max(_ilist)
    imin = min(_ilist)
    jmax = max(_jlist)
    jmin = min(_jlist)
    xlen_spc = ((imax - imin) / spacing)
    ylen_spc = ((jmax - jmin) / spacing)
    xstart = ((xlen_spc - int(xlen_spc) + 1)) * spacing / 2
    ystart = ((ylen_spc - int(ylen_spc) + 1)) * spacing / 2
    x0 = [float('%g' % x) + xstart for x in frange_inc(imin, imax, spacing)]
    y0 = [float('%g' % x) + ystart for x in frange_inc(jmin, jmax, spacing)]
    raw_pts = [[i, j] for i in x0 for j in y0]
    if polygon.normal() == radgeom.Vector(0, 0, 1):
        pt_incls = pt_inclusion(pg_pts)
    else:
        pt_incls = pt_inclusion(pg_pts[::-1])
    _grid = [p for p in raw_pts if pt_incls.test_inside(p) > 0]
    if op:
        grid_dir = normal.reverse()
    else:
        grid_dir = normal
    p_height = sum([height * i for i in grid_dir.to_list()]) + plane_height
    grid = []
    _idx = list(range(3))
    _idx.pop(drop_idx)
    for g in _grid:
        tup = [0.0] * 3 + grid_dir.to_list()
        tup[drop_idx] = p_height
        tup[_idx[0]] = g[0]
        tup[_idx[1]] = g[1]
        grid.append(tup)
    return [' '.join(map(str, row)) for row in grid]


def material_lib():
    mlib = []
    #carpet .25
    mlib.append(plastic_prim('void', 'carpet_25', .25, 128, 128, 128, 0, 0))
    # Paint .5
    mlib.append(plastic_prim('void', 'white_paint_50', .5, 128, 128, 128, 0,
                             0))
    # Paint .75
    mlib.append(plastic_prim('void', 'white_paint_80', .8, 128, 128, 128, 0,
                             0))
    # Glass .8
    mlib.append(glass_prim('void', 'glass_80', .8, .8, .8))
    return mlib
