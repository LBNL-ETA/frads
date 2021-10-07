import argparse
import csv
from dataclasses import dataclass, field
import logging
import math
import os
import re
import ssl
import subprocess as sp
import time
from typing import Dict, List, Optional, NamedTuple, Tuple
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET


logger = logging.getLogger("frads.util")


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


@dataclass
class MradConfig:
    name: str = ''
    vmx_basis: str = 'kf'
    vmx_opt: str = '-ab 5 -ad 65536 -lw 1e-5'
    fmx_basis: str = 'kf'
    fmx_opt: str = '-ab 3 -ad 65536 -lw 5e-5'
    smx_basis: str = 'r4'
    dmx_opt: str = '-ab 2 -ad 128 -c 5000'
    dsmx_opt: str = '-ab 7 -ad 8196 -lw 5e-5'
    cdsmx_opt: str = '-ab 1'
    cdsmx_basis: str = 'r6'
    ray_count: int = 1
    pixel_jitter: float = .7
    separate_direct: bool = False
    nprocess: int = 1
    overwrite: bool = False
    method: str = field(default_factory=str)
    no_multiply: bool = False
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

    def to_dict(self):
        sim_ctrl = {
            "vmx_basis": self.vmx_basis,
            "vmx_opt": self.vmx_opt,
            "fmx_basis": self.fmx_basis,
            "smx_basis": self.smx_basis,
            "dmx_opt": self.dmx_opt,
            "dsmx_opt": self.dsmx_opt,
            "cdsmx_opt":  self.cdsmx_opt,
            "cdsmx_basis":  self.cdsmx_basis,
            "separate_direct":  self.separate_direct,
            "overwrite":  self.overwrite,
            "method":  self.method,
        }
        file_struct = {
            "base": self.base,
            "objects": self.objects,
            "matrices": self.matrices,
            "resources": self.resources,
            "results": self.results
        }
        model = {
            "material": self.material,
            "scene": self.scene,
            "window_paths": self.window_paths,
            "window_xml": self.window_xml,
            "window_cfs": self.window_cfs,
            "window_control": self.window_control,
        }
        raysender = {
            "grid_surface": self.grid_surface,
            "grid_spacing": self.grid_spacing,
            "grid_height": self.grid_height,
            "view": self.view
        }
        site = {
            "wea_path": self.wea_path,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "zipcode": self.zipcode,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "daylight_hours_only": self.daylight_hours_only
        }
        return {"Simulation Control": sim_ctrl,
                "File Structure": file_struct,
                "Site": site,
                "Model": model,
                "Ray Sender": raysender}



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


def spec2rgb(inp: str, cspace: str) -> list:
    """Convert spectral data to RGB.

    Convert wavelength and spectral data in visible
    spectrum to red, green, and blue given a color space.

    Args:
        inp: file path with wavelength spectral data
        cspace: Color space to transform the spectral data.
            { radiance | sharp | adobe | rimm | 709 | p3 | 2020 }
    Returns:
        Red, Green, Blue in a list.

    Raise:
        KeyError where cspace is not defined.
    """

    color_primaries = {
        'radiance': 'CIE_Radiance(i)', 'sharp': 'CIE_Sharp(i)',
        'adobe': 'CIE_Adobe(i)', 'rimm': 'CIE_RIMM(i)',
        '709': 'CIE_709(i)', 'p3': 'CIE_P3(i)', '2020': 'CIE_2020(i)'
    }
    try:
        primaries = color_primaries[cspace]
    except KeyError as key_error:
        print(color_primaries.keys())
        raise key_error
    cmd1 = [
        "rcalc", "-f", "cieresp.cal",
        "-e", "ty=triy($1);$1=ty",
        "-e", "$2=$2*trix($1);$3=$2*ty;$4=$2*triz($1)",
        "-e", "cond=if($1-359,831-$1,-1)"
    ]
    cmd2 = [
        "rcalc", "-f", "xyz_rgb.cal",
        "-e", f"CIE_pri(i)={primaries}",
        "-e", "$1=R($1,$2,$3);$2=G($1,$2,$3);$3=B($1,$2,$3)"
    ]
    proc = sp.run(cmd1, input=inp.encode(), check=True, stdout=sp.PIPE)
    res = [row.split('\t') for row in proc.stdout.decode().splitlines()]
    row_cnt = len(res)
    avg_0 = sum([float(row[0]) for row in res]) / row_cnt
    avg_1 = sum([float(row[1]) for row in res]) / row_cnt
    avg_2 = sum([float(row[2]) for row in res]) / row_cnt
    avg_3 = sum([float(row[3]) for row in res]) / row_cnt
    XYZ = f"{avg_1 / avg_0} {avg_2 / avg_0} {avg_3 / avg_0}"
    proc2 = sp.run(cmd2, input=XYZ.encode(), stdout=sp.PIPE)
    return proc2.stdout.decode().split()


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


def get_latlon_from_zipcode(zipcode: str) -> Tuple[float, float]:
    """Get latitude and longitude from US zipcode."""
    _file_path_ = os.path.dirname(__file__)
    zip2latlon = "zip_latlon.txt"
    zip2latlon_path = os.path.join(_file_path_, 'data', zip2latlon)
    with open(zip2latlon_path, 'r') as rdr:
        csvreader = csv.DictReader(rdr, delimiter='\t')
        for row in csvreader:
            if row['GEOID'] == zipcode:
                lat = float(row['INTPTLAT'])
                lon = float(row['INTPTLONG'])
                break
        else:
            raise ValueError('zipcode not found in US')
    return lat, lon


def haversine(lat1: float, lat2: float, lon1: float, lon2: float) -> float:
    """Calculate distance between two points on earth."""
    earth_radius = 6371e3
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lam = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2) * math.sin(delta_phi / 2)
         + (math.cos(phi1) * math.cos(phi2)
            * math.sin(delta_lam / 2) * math.sin(delta_lam / 2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c


def get_epw_url(lat: float, lon: float) -> Tuple[str, str]:
    """Get EPW name and url given latitude and longitude."""
    lon = lon * -1
    _file_path_ = os.path.dirname(__file__)
    epw_url = "epw_url.csv"
    epw_url_path = os.path.join(_file_path_, 'data', epw_url)
    distances = []
    urls = []
    with open(epw_url_path, 'r') as rdr:
        csvreader = csv.DictReader(rdr, delimiter=',')
        for row in csvreader:
            distances.append(haversine(
                float(row['Latitude']), float(lat),
                float(row['Longitude']), float(lon)))
            urls.append(row['URL'])
    min_idx = distances.index(min(distances))
    url = urls[min_idx]
    epw_fname = os.path.basename(url)
    return epw_fname, url


def request(url: str, header: dict) -> str:
    user_agents = 'Mozilla/5.0 (Windows NT 6.1) '
    user_agents += 'AppleWebKit/537.36 (KHTML, like Gecko) '
    user_agents += 'Chrome/41.0.2228.0 Safari/537.3'
    header['User-Agent'] = user_agents
    request = urllib.request.Request(
        url, headers=header
    )
    tmpctx = ssl.SSLContext()
    raw = ''
    for _ in range(3):  # try 3 times
        try:
            with urllib.request.urlopen(request, context=tmpctx) as resp:
                raw = resp.read().decode()
                break
        except urllib.error.HTTPError:
            time.sleep(1)
    if raw.startswith('404'):
        raise Exception(urllib.error.URLError)
    if raw == '':
        raise ValueError("Empty return from request")
    return raw
