import argparse
from dataclasses import dataclass, field
import logging
import math
import os
import subprocess as sp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from frads import radutil

logger = logging.getLogger("frads.util")


@dataclass
class MradConfig:
    name: str = ''
    vmx_basis: str = 'kf'
    vmx_opt: str = '-ab 5 -ad 262144 -lw 1e-6'
    fmx_basis: str = 'kf'
    fmx_opt: str = '-ab 3 -ad 65536 -lw 1e-4'
    smx_basis: str = 'r4'
    dmx_opt: str = '-ab 2 -ad 128 -c 5000'
    dsmx_opt: str = '-ab 6 -ad 262144 -lw 1e-6'
    cdsmx_opt: str = '-ab 1'
    cdsmx_basis: str = 'r6'
    ray_count: int = 1
    pixel_jitter: float = .7
    separate_direct: bool = False
    nprocess: int = 1
    overwrite: bool = False
    method: str = field(default_factory=str)
    base: str = os.getcwd()
    matrices: str = 'Matrices'
    results: str = 'Results'
    objects: str = 'Objects'
    resources: str = 'Resources'
    wea_path: str = field(default_factory=str)
    latitude: float = field(default_factory=float)
    longitude: float = field(default_factory=float)
    zipcode: str = field(default_factory=str)
    daylight_hours_only: bool = True
    start_hour: float = field(default_factory=float)
    end_hour: float = field(default_factory=float)
    orientation: int = 0
    material: str = "materials.mat"
    scene: str = field(default_factory=str)
    window_xml: str = ''
    window_paths: str = ''
    window_cfs: str = ''
    window_control: str = ''
    ncp_shade: str = ''
    grid_surface: str = ''
    grid_spacing: float = 0.67
    grid_height: float = .76
    view: str = ''
    objdir: str = field(init=False)
    mtxdir: str = field(init=False)
    resdir: str = field(init=False)
    rsodir: str = field(init=False)
    scene_paths: List[str] = field(init=False)
    grid_surface_paths: Dict[str, str] = field(init=False)
    windows: dict = field(init=False)
    klems_bsdfs: dict = field(init=False)
    sun_cfs: dict = field(init=False)

    def __post_init__(self):
        ""","""
        self.objdir = os.path.join(self.base, self.objects)
        self.mtxdir = os.path.join(self.base, self.matrices)
        self.resdir = os.path.join(self.base, self.results)
        self.rsodir = os.path.join(self.base, self.resources)
        self.scene_paths = [os.path.join(self.objdir, path)
                            for path in self.scene.split()]
        self.material_paths = [os.path.join(self.objdir, path)
                               for path in self.material.split()]
        self.windows = {basename(path): os.path.join(self.objdir, path)
                        for path in self.window_paths.split()}
        self.grid_surface_paths = {basename(path):os.path.join(self.objdir, path)
                                   for path in self.grid_surface.split()}
        window_xml = self.window_xml.split()
        if self.window_control.startswith('@'):
            # Load control schedule
            pass
        else:
            static_control = self.window_control.split()

            if len(window_xml) > 0 and (len(static_control) != len(self.windows)):
                raise ValueError("Need a control for each window")
            self.klems_bsdfs = {wname: os.path.join(self.rsodir, self.window_xml.split()[int(control)])
                                for wname, control in zip(self.windows.keys(), static_control)}
            if (self.window_cfs == '') or (self.window_cfs == self.window_xml):
                self.sun_cfs = self.klems_bsdfs
            else:
                cfs_paths = self.window_cfs.split()
                for wname, control in zip(self.windows.keys(), static_control):
                    _cfs = cfs_paths[int(control)]
                    if _cfs.endswith('.xml'):
                        self.sun_cfs[wname] = os.path.join(self.rsodir, _cfs)
                    elif _cfs.endswith('.rad'):
                        self.sun_cfs[wname] = os.path.join(self.objdir, _cfs)
                    else:
                        raise NameError("Unknow file type for dbsdf")



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
    vparser.add_argument('-x', type=int, default=500)
    vparser.add_argument('-y', type=int, default=500)
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


def tmit2tmis(tmit: float) -> float:
    """Convert from transmittance to transmissivity."""
    return round(
        (math.sqrt(0.8402528435 + 0.0072522239 * tmit**2) - 0.9166530661) /
        0.0036261119 / tmit, 3)


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


def unpack_idf(path: str) -> dict:
    """Read and parse and idf files."""
    with open(path, 'r') as rdr:
        return parse_idf(rdr.read())


def idf2mtx(fname:str, section: list, out_dir: str=None) -> None:
    """Converting bsdf data in idf format to Radiance format."""
    out_dir = os.getcwd() if out_dir is None else out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for sec in section:
        name = sec[0]
        output_path = os.path.join(out_dir, "%s_%s.mtx" % (fname, name))
        row_cnt = int(sec[1])
        col_cnt = int(sec[2])
        if sec[2] in radutil.BASIS_DICT:
            _btdf_data = nest_list(list(map(float, sec[3:])), col_cnt)
            lambdas = radutil.angle_basis_coeff(radutil.BASIS_DICT[sec[2]])
            sdata = [list(map(lambda x, y: x * y, i, lambdas)) for i in _btdf_data]
            with open(output_path, 'w') as wt:
                header = '#?RADIANCE\nNCOMP=3\n'
                header += 'NROWS=%d\nNCOLS=%d\n' % (row_cnt, col_cnt)
                header += 'FORMAT=ascii\n\n'
                wt.write(header)
                for row in sdata:
                    for val in row:
                        string = '\t'.join(['%07.5f' % val] * 3)
                        wt.write(string)
                        wt.write('\t')
                    wt.write('\n')


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



def parse_bsdf_xml(path: str) -> Optional[dict]:
    """Parse BSDF file in XML format.
    TODO: validate xml first before parsing
    """
    data_dict: Dict[str, Optional[Dict[str, str]]] = {}
    tree = ET.parse(path)
    root = tree.getroot()
    if root is None:
        return None
    tag = root.tag.rstrip('WindowElement')
    optical = root.find(tag+'Optical')
    layer = optical.find(tag+'Layer')
    datadef = layer.find(tag+'DataDefinition')\
        .find(tag+'IncidentDataStructure')
    dblocks = layer.findall(tag+'WavelengthData')
    data_dict = {'def': datadef.text.strip(), 'Solar': {}, 'Visible': {}}
    for block in dblocks:
        wavelength = block.find(tag+'Wavelength').text
        dblock = block.find(tag+'WavelengthDataBlock')
        direction = dblock.find(tag+'WavelengthDataDirection').text
        scattering_data = dblock.find(tag+'ScatteringData').text.strip()
        if scattering_data.count('\n') == 21168:
            data_string = scattering_data.replace('\n\t', ' ')
        else:
            data_string = scattering_data
        data_dict[wavelength][direction] = data_string
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
        pass


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
    logger.info(cmd)
    proc = sp.run(cmd, check=True, stderr=sp.PIPE)
    if proc.stderr != b'':
        logger.warning(proc.stderr)


def spcheckout(cmd, inp=None):
    """Call subprocess run and return results."""
    logger.info(cmd)
    proc = sp.run(cmd, input=inp, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
    if proc.stderr != b'':
        logger.warning(proc.stderr)
    return proc.stdout
