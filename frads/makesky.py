"""
Routines for generating sky models
"""

import argparse
import csv
import datetime
import logging
import math
import os
import subprocess as sp
import tempfile as tf
from frads import radutil, util, radgeom
from typing import List, Union, Set, NamedTuple

LSEP = os.linesep

logger = logging.getLogger("frads.makesky")


class WeaMetaData(NamedTuple):
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: int
    elevation: float

    def wea_header(self) -> str:
        header = f"place {self.city}_{self.country}\n"
        header += f"latitude {self.latitude}\n"
        header += f"longitude {self.longitude}\n"
        header += f"time_zone {self.timezone}\n"
        header += f"site_elevation {self.elevation}\n"
        header += "weather_data_file_units 1\n"
        return header


class WeaDataRow(NamedTuple):
    month: int
    day: int
    hour: int
    minute: int
    second: int
    hours: float
    dni: float
    dhi: float

    def __str__(self):
        return f"{self.month} {self.day} {self.hours} {self.dni} {self.dhi}"

    def dt_string(self):
        return f"{self.month:02d}{self.day:02d}_{self.hour:02d}30"


def basis_glow(sky_basis):
    """Sky and ground glow primitives.
    Args:
        sky_basis: sky sampling basis, e.g. r1, r4
    Returns:
        ground and sky glow string
    """

    grnd_str = grndglow()
    sky_str = skyglow(sky_basis)
    return grnd_str + sky_str


def skyglow(basis: str, upvect='+Y') -> str:
    sky_string = f"#@rfluxmtx u={upvect} h={basis}{LSEP*2}"
    sky_string += f"void glow skyglow{LSEP}"
    sky_string += f"0{LSEP}0{LSEP}4 1 1 1 0{LSEP*2}"
    sky_string += f"skyglow source sky{LSEP}"
    sky_string += f"0{LSEP}0{LSEP}4 0 0 1 180{LSEP}"
    return sky_string


def grndglow(basis='u') -> str:
    ground_string = f"#@rfluxmtx h={basis}{LSEP*2}"
    ground_string += f"void glow groundglow{LSEP}"
    ground_string += f"0{LSEP}0{LSEP}4 1 1 1 0{LSEP*2}"
    ground_string += f"groundglow source ground{LSEP}"
    ground_string += f"0{LSEP}0{LSEP}4 0 0 -1 180{LSEP*2}"
    return ground_string


class Gensun(object):
    """Generate sun sources for matrix generation."""

    def __init__(self, mf: int):
        """."""
        self.runlen = 144 * mf**2 + 3
        self.rsrc = radutil.Reinsrc(mf)
        self.mod_str = LSEP.join([f'sol{i}' for i in range(1, self.runlen)])

    def gen_full(self):
        """Generate full treganza based sun sources."""
        out_lines = []
        for i in range(1, self.runlen):
            dirs = self.rsrc.dir_calc(i)
            line = f"void light sol{i} 0 0 3 1 1 1 sol{i} "
            line += "source sun 0 0 4 {:.6g} {:.6g} {:.6g} 0.533".format(*dirs)
            out_lines.append(line)
        return LSEP.join(out_lines)+LSEP

    def gen_cull(self, smx_path=None, window_normals=None):
        """Generate culled sun sources based on window orientation and
        climate based sky matrix. The reduced sun sources will significantly
        speed up the matrix generation.
        Args:
            smx_path: sky matrix path, usually the output of gendaymtx
            window_normals: window normals
        Returns:
            Sun receiver primitives strings
            Corresponding modifier strings
        """

        win_norm = []
        if smx_path is not None:
            cmd = f"rmtxop -ff -c .3 .6 .1 -t {smx_path} "
            cmd += "| getinfo - | total -if5186 -t,"
            dtot = [float(i)
                    for i in sp.check_output(cmd, shell=True).split(b',')]
        else:
            dtot = [1] * self.runlen
        out_lines = []
        mod_str = []
        if window_normals is not None:
            win_norm = window_normals
            for i in range(1, self.runlen):
                dirs = radgeom.Vector(*self.rsrc.dir_calc(i))
                _mod = 'sol'+str(i)
                v = 0
                if dtot[i - 1] > 0:
                    for norm in win_norm:
                        if norm * dirs < 0:
                            v = 1
                            mod_str.append(_mod)
                            break
                line = f"void light sol{i} 0 0 3 {v} {v} {v} sol{i} "
                line += f"source sun 0 0 4 {dirs.z:.6g} {dirs.x:.6g} {dirs.z:.6g} 0.533"
                out_lines.append(line)
        else:
            for i in range(1, self.runlen):
                dirs = radgeom.Vector(*self.rsrc.dir_calc(i))
                _mod = 'sol'+str(i)
                v = 0
                if dtot[i - 1] > 0:
                    v = 1
                    mod_str.append(_mod)
                line = f"void light sol{i} 0 0 3 {v} {v} {v} sol{i} "
                line += f"source sun 0 0 4 {dirs.z:.6g} {dirs.x:.6g} {dirs.z:.6g} 0.533"
                out_lines.append(line)
        logger.debug(out_lines)
        logger.debug(mod_str)
        return LSEP.join(out_lines), LSEP.join(mod_str)


def epw2sunmtx(epw_path: str) -> str:
    """Generate reinhart 6 sun matrix file from a epw file."""
    smx_path = util.basename(epw_path) + ".smx"
    with tf.NamedTemporaryFile() as wea:
        cmd = f"epw2wea {epw_path} {wea.name}"
        sp.call(cmd, shell=True)
        cmd = f"gendaymtx -od -u -m 6 -d -5 .533 {wea.name} > {smx_path}"
        sp.call(cmd, shell=True)
    return smx_path


def loc2sunmtx(basis, lat, lon, ele):
    """Generate a psuedo reinhart 6 sun matrix file given lat, lon, etc..."""
    tz = int(5 * round(lon / 5))
    metadata = WeaMetaData('city', 'country', lat, lon, tz, ele)
    header = metadata.wea_header()
    header = str.encode(header)
    string = ""
    mday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for mo in range(12):
        for da in range(mday[mo]):
            for ho in range(24):
                string += f"{mo+1} {da+1} {ho+.5} 100 100"
    string = str.encode(string)
    smx_path = f"sun_{lat}_{lon}_{basis}.smx"
    with tf.NamedTemporaryFile() as wea:
        wea.write(header)
        wea.write(string)
        cmd = f"gendaymtx -od -m 6 -u -d -5 .533 {wea.name} > {smx_path}"
        sp.call(cmd, shell=True)
    return smx_path


def gendaymtx(data_entry: List[WeaDataRow], metadata: WeaMetaData,
              mf=4, direct=False, solar=False, onesun=False,
              rotate=0, binary=False):
    """."""
    cmd = ['gendaymtx', '-r', str(rotate), '-m', str(mf)]
    if direct:
        cmd.append('-d')
    if solar:
        cmd.append('-O1')
    if onesun:
        cmd.append('-5')
        cmd.append('.533')
    if binary:
        cmd.append('-of')
    wea_input = metadata.wea_header() + "\n".join(map(str, data_entry))
    sky_matrix = util.spcheckout(cmd, inp=wea_input.encode())
    return sky_matrix


def gendaymtx_cmd(data_entry: List[str], metadata: WeaMetaData,
                  mf=4, direct=False, solar=False, onesun=False,
                  rotate=0, binary=False):
    """."""
    sun_only = ' -d' if direct else ''
    spect = ' -O1' if solar else ' -O0'
    _five = ' -5 .533' if onesun else ''
    bi = '' if binary is False or os.name == 'nt' else ' -o' + binary
    linesep = r'& echo' if os.name == 'nt' else LSEP
    wea_head = f"place test{linesep}latitude {metadata.latitude}{linesep}"
    wea_head += f"longitude {metadata.longitude}{linesep}"
    wea_head += f"time_zone {metadata.timezone}{linesep}site_elevation "
    wea_head += f"{metadata.elevation}{linesep}"
    wea_head += f"weather_data_file_units 1{linesep}"
    skv_cmd = f"gendaymtx -u -r {rotate} -m {mf}{sun_only}{_five}{spect}{bi}"
    wea_data = linesep.join(data_entry)
    if os.name == 'nt':
        wea_cmd = f'(echo {wea_head}{wea_data}) | '
    else:
        wea_cmd = f'echo "{wea_head}{wea_data}" | '
    cmd = wea_cmd + skv_cmd
    return cmd


def parse_csv(csv_path, ftype='csv', dt_col="date_time",
              dt_format="%Y%m%d %H:%M:%S", dni_col='DNI',
              dhi_col='DHI', stime=None, etime=None):
    """Parse a csv file containing direct normal
    and diffuse horizontal data, ignoring NULL
    and zero values.

    Parameters:
        csv_path: str
            Path to the csv/tsv file.
        ftype: str, default csv
            File type to choose between csv and tsv.
        dt_col: str, default date_time
            name of the datetime column.
        dt_format: str, default %Y%m%d %H:%M:%S
            Python datetime format.
        dni_col: str, default DNI
            Column name containing direct normal
            irradiation data.
        dhi_col: str, default DHI
            Column name containing diffuse horizontal
            irradiation data.
        stime: str, default None
            Exclude the data before this time. e.g.'09:00'
        etime: str, default None
            Exclude the data after this time. e.g.'17:00'
    Returns: list object
        a list of strings containing datetime
        and solar radiation values
    """

    data_entry = []
    if None not in (stime, etime):
        stime = datetime.datetime.strptime(stime, "%H:%M")
        etime = datetime.datetime.strptime(etime, "%H:%M")
        shours = stime.hour + stime.minute / 60.0
        ehours = etime.hour + etime.minute / 60.0
    else:
        shours = 0
        ehours = 24
    ftypes = {'csv': 'excel', 'tsv': 'excel-tab'}
    with open(csv_path, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile, dialect=ftypes[ftype])
        for row in reader:
            _dt = datetime.datetime.strptime(row[dt_col], dt_format)
            _year = str(_dt.year)
            _month = str(_dt.month)
            _day = str(_dt.day)
            _hour = _dt.hour
            _minute = _dt.minute
            _hours = round(_hour + _minute / 60.0, 2)
            if _hours < shours or _hours > ehours:
                continue
            try:
                _dni = float(row[dni_col])
                _dhi = float(row[dhi_col])
            except ValueError:
                continue
            if int(_dni) == 0 and int(_dhi) == 0:
                continue
            data_entry.append(
                [_year, _month, _day, _hour, _minute, _hours, _dni, _dhi])
    return data_entry


def sky_cont(mon, day, hrs, lat, lon, mer, dni, dhi,
             year=None, grefl=.2, spect='0', rotate=None):
    out_str = f'!gendaylit {mon} {day} {hrs} '
    out_str += f'-a {lat} -o {lon} -m {mer} '
    if year is not None:
        out_str += f'-y {year} '
    out_str += f'-W {dni} {dhi} -g {grefl} -O {spect}{LSEP*2}'
    if rotate is not None:
        out_str += f"| xform -rz {rotate}"
    out_str += f'skyfunc glow skyglow 0 0 4 1 1 1 0{LSEP*2}'
    out_str += f'skyglow source sky 0 0 4 0 0 1 180{LSEP*2}'
    out_str += f'skyfunc glow groundglow 0 0 4 1 1 1 0{LSEP*2}'
    out_str += f'groundglow source ground 0 0 4 0 0 -1 180{LSEP}'
    return out_str


def solar_angle(lat, lon, mer, month, day, hour):
    """Simplified translation from the Radiance sun.c and gensky.c code.

    This function test if the solar altitude is greater than zero
    """
    latitude_r = math.radians(lat)
    longitude_r = math.radians(lon)
    s_meridian = math.radians(mer)
    mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    julian_date = mo_da[month - 1] + day

    solar_decline = 0.4093 * math.sin((2 * math.pi / 365) * (julian_date - 81))

    solar_time = hour + (0.170 * math.sin((4 * math.pi / 373) * (julian_date - 80))
                         - 0.129 * math.sin((2 * math.pi / 355)
                                            * (julian_date - 8))
                         + (12/math.pi) * (s_meridian - longitude_r))

    altitude = math.asin(math.sin(latitude_r) * math.sin(solar_decline)
                         - math.cos(latitude_r) * math.cos(solar_decline)
                         * math.cos(solar_time * (math.pi / 12)))
    azimuth = -math.atan2(math.cos(solar_decline)*math.sin(solar_time*(math.pi/12.)),
                          - math.cos(latitude_r)*math.sin(solar_time)
                          - math.sin(latitude_r)*math.cos(solar_decline)
                          * math.cos(solar_time*(math.pi/12)))

    return altitude, azimuth


def start_end_hour(data: list, sh: float, eh: float):
    """Remove wea data entries outside of the
    start and end hour."""
    if sh == 0 and eh == 0:
        return data

    def filter_hour(dataline):
        return sh <= float(dataline[2]) <= eh
    return filter(filter_hour, data)


def check_sun_above_horizon(data, metadata):
    """Remove non-daylight hour entries."""
    def solar_altitude_check(row: WeaDataRow):
        alt, _ = solar_angle(metadata.latitude, metadata.longitude,
                             metadata.timezone, row.month,
                             row.day, row.hours)
        return alt > 0
    return filter(solar_altitude_check, data)


def remove_wea_zero_entry(data, metadata: WeaMetaData, window_normal=None):
    """Remove wea data entries with zero solar luminance.
    If window normal supplied, eliminate entries not seen by window.
    Solar luminance determined using Perez sky model.
    Window field of view is 176 deg with 2 deg tolerance on each side.
    """
    check_window_normal = True if window_normal is not None else False
    new_dataline = []
    data = filter(lambda row: row.dni != 0, data)
    for row in data:
        cmd = ['gendaylit', str(row.month), str(row.day), str(row.hours),
               '-a', str(metadata.latitude), '-o', str(metadata.longitude),
               '-m', str(metadata.timezone), '-W', str(row.dni), str(row.dhi)]
        process = sp.run(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        primitives = radutil.parse_primitive(
            process.stdout.decode().splitlines())
        if process.stderr == b'':
            light = float(primitives[0].real_arg.split()[2])
            dirs = radgeom.Vector(*list(map(float, primitives[1].real_arg.split()[1:4])))
            if light > 0:
                if check_window_normal:
                    for normal in window_normal:
                        if normal * dirs < -0.035: # 2deg tolerance
                            new_dataline.append(row)
                            break
                else:
                    new_dataline.append(row)
    return new_dataline


def parse_epw(epw_str: str) -> tuple:
    """Parse epw file and return wea header and data."""
    raw = epw_str.splitlines()
    epw_header = raw[0].split(',')
    content = raw[8:]
    data = []
    for li in content:
        line = li.split(',')
        month = int(line[1])
        day = int(line[2])
        hour = int(line[3]) - 1
        hours = hour + 0.5
        dir_norm = float(line[14])
        dif_hor = float(line[15])
        data.append(WeaDataRow(
            month, day, hour, 30, 0, hours, dir_norm, dif_hor))
    city = epw_header[1]
    country = epw_header[3]
    latitude = float(epw_header[6])
    longitude = -1 * float(epw_header[7])
    tz = int(float(epw_header[8])) * (-15)
    elevation = float(epw_header[9].rstrip())
    meta_data = WeaMetaData(city, country, latitude, longitude, tz, elevation)
    return meta_data, data


def epw2wea(epw_str,
            dhour=False, shour=None, ehour=None,
            remove_zero=False, window_normal=None):
    """epw2wea with added filter."""
    metadata, data = parse_epw(epw_str)
    if None not in (shour, ehour):
        data = start_end_hour(data, shour, ehour)
    if dhour:
        data = check_sun_above_horizon(data, metadata)
    if remove_zero:
        data = remove_wea_zero_entry(data, metadata,
                                     window_normal=window_normal)
    return metadata, list(data)


def getwea():
    """Commandline program for generating a .wea file
    from downloaded EPW data."""
    parser = argparse.ArgumentParser(
        prog='getwea',
        description="Download the EPW files and convert it to a wea file")
    parser.add_argument('-a', type=float, help='latitude')
    parser.add_argument('-o', type=float, help='longitude')
    parser.add_argument('-z', help='zipcode (U.S. only)')
    parser.add_argument('-dh', action="store_true",
                        help='output only for daylight hours')
    parser.add_argument('-sh', type=float, help='start hour (float)')
    parser.add_argument('-eh', type=float, help='end hour (float)')
    parser.add_argument('-rz', action='store_true', help='remove zero solar luminance entries')
    parser.add_argument('-wpths', nargs='+', help='window paths (.rad)')
    args = parser.parse_args()
    window_normals: Union[None, Set[radgeom.Vector]] = None
    if args.z is not None:
        lat, lon = util.get_latlon_from_zipcode(args.z)
    elif None not in (args.a, args.o):
        lat, lon = args.a, args.o
    else:
        print("Exit: need either latitude and longitude or U.S. postcode")
        exit()
    _, url = util.get_epw_url(lat, lon)
    epw = util.request(url, {})
    remove_zero = False
    if args.wpths is not None:
        window_normals = radutil.primitive_normal(args.wpths)
        remove_zero = True
    wea_metadata, wea_data = parse_epw(epw)
    if None not in (args.sh, args.eh):
        wea_data = start_end_hour(wea_data, args.sh, args.eh)
    if args.dh:
        wea_data = check_sun_above_horizon(wea_data, wea_metadata)
    if remove_zero:
        wea_data = remove_wea_zero_entry(
            wea_data, wea_metadata, window_normals)
    print(wea_metadata.wea_header())
    print('\n'.join(map(str, wea_data)))
