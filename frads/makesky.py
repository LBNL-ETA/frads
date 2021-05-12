"""
Routines for generating sky models
"""

import argparse
import csv
import datetime
import math
import os
import subprocess as sp
import tempfile as tf
import time
import urllib.request
import urllib.error
import ssl
from frads import radutil, util, radgeom
from typing import List

LSEP = os.linesep


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

    def __init__(self, mf):
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

        if window_normals is not None:
            win_norm = window_normals
        else:
            win_norm = [radgeom.Vector(0, -1, 0)]
        if smx_path is not None:
            cmd = f"rmtxop -ff -c .3 .6 .1 -t {smx_path} "
            cmd += "| getinfo - | total -if5186 -t,"
            dtot = [float(i)
                    for i in sp.check_output(cmd, shell=True).split(b',')]
        else:
            dtot = [1] * self.runlen

        out_lines = []
        mod_str = []
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
        # if mod_str[-1] != 'sol%s'%(self.runlen-1):
            # mod_str.append('sol%s'%(self.runlen-1))
        return LSEP.join(out_lines), LSEP.join(mod_str)


def epw2sunmtx(epw_path: str) -> str:
    """Generate reinhart 6 sun matrix file from a epw file."""
    smx_path = util.basename(epw_path) + ".smx"
    with tf.NamedTemporaryFile() as wea:
        cmd = f"epw2wea {epw_path} {wea.name}"
        sp.call(cmd, shell=True)
        # large file
        cmd = f"gendaymtx -od -u -m 6 -d -5 .533 {wea.name} > {smx_path}"
        sp.call(cmd, shell=True)
    return smx_path


def loc2sunmtx(basis, lat, lon, ele):
    """Generate a psuedo reinhart 6 sun matrix file given lat, lon, etc..."""
    tz = int(5 * round(lon / 5))
    header = f"place city_country{LSEP}"
    header += f"latitude {lat}{LSEP}"
    header += f"longitude {lon}{LSEP}"
    header += f"time_zone {tz}{LSEP}"
    header += f"site_elevation {ele}{LSEP}"
    header += "weather_data_file_units 1{LSEP}"
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


def gendaymtx(data_entry: List[str], lat: str, lon: str,
              timezone: str, ele: str, mf=4, direct=False,
              solar=False, onesun=False, rotate=0, binary=False):
    """."""
    sun_only = ' -d' if direct else ''
    spect = ' -O1' if solar else ' -O0'
    _five = ' -5 .533' if onesun else ''
    bi = '' if binary is False or os.name == 'nt' else ' -o' + binary
    linesep = r'& echo' if os.name == 'nt' else LSEP
    wea_head = f"place test{linesep}latitude {lat}{linesep}"
    wea_head += f"longitude {lon}{linesep}"
    wea_head += f"time_zone {timezone}{linesep}site_elevation {ele}{linesep}"
    wea_head += f"weather_data_file_units 1{linesep}"
    skv_cmd = f"gendaymtx -u -r {rotate} -m {mf}{sun_only}{_five}{spect}{bi}"
    if len(data_entry) > 1000:
        _, _path = tf.mkstemp()
        with open(_path, 'w') as wtr:
            wtr.write(wea_head.replace('\\n', '\n'))
            wtr.write('\n'.join(data_entry))
        cmd = skv_cmd + " " + _path
        return cmd, _path
    else:
        wea_data = linesep.join(data_entry)
        if os.name == 'nt':
            wea_cmd = f'(echo {wea_head}{wea_data}) | '
        else:
            wea_cmd = f'echo "{wea_head}{wea_data}" | '
        cmd = wea_cmd + skv_cmd
        return cmd, None


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
             year=None, grefl=.2, spect='0'):
    out_str = f'!gendaylit {mon} {day} {hrs} '
    out_str += f'-a {lat} -o {lon} -m {mer} '
    if year is not None:
        out_str += f'-y {year} '
    out_str += f'-W {dni} {dhi} -g {grefl} -O {spect}{LSEP*2}'
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
                          *math.cos(solar_time*(math.pi/12)))

    return altitude, azimuth


class epw2wea(object):
    """."""

    def __init__(self, *, epw: str, dh: bool, sh: float, eh: float,
                 remove_zero=False, window_normals=None):
        """."""
        self.epw = epw
        # self.wea = wea
        self.read_epw()  # read-in epw/tmy data

        if sh > 0 and eh > 0:
            self.start_end_hour(sh, eh)

        if dh:
            self.daylight()  # filter out non-daylight hours if asked

        if remove_zero:
            self.remove_entries(window_normals=window_normals)

        self.wea = self.header + '\n' + self.string
        self.dt_string = []
        for line in self.string.splitlines():
            entry = line.split()
            mo = int(entry[0])
            da = int(entry[1])
            hr = int(float(entry[2]))
            self.dt_string.append(f"{mo:02d}{da:02d}_{hr:02d}30")

    def remove_entries(self, window_normals=None):
        """Remove data entries with zero solar luminance."""
        check_window_normal = True if window_normals is not None else False
        new_string = []
        for line in self.string.splitlines():
            items = line.split()
            cmd = f'gendaylit {items[0]} {items[1]} {items[2]} '
            cmd += f'-a {self.latitude} -o {self.longitude} '
            cmd += f'-m {self.tz} -W {items[3]} {items[4]}'
            process = sp.run(cmd.split(), stderr=sp.PIPE, stdout=sp.PIPE)
            primitives = radutil.parse_primitive(
                process.stdout.decode().splitlines())
            light = 0
            if process.stderr == b'':
                light = float(primitives[0].real_arg.split()[2])
                dirs = radgeom.Vector(*list(map(float, primitives[1].real_arg.split()[1:4])))
                if light > 0:
                    if check_window_normal:
                        for normal in window_normals:
                            if normal * dirs < -0.035:
                                new_string.append(line)
                    else:
                        new_string.append(line)
        self.string = '\n'.join(new_string)

    def solar_altitude_check(self, string_line: str):
        mon, day, hours = string_line.split()[:3]
        alt, _ = solar_angle(self.latitude, self.longitude, self.tz,
                             int(mon), int(day), float(hours))
        return alt > 0

    def daylight(self):
        """."""
        string_line = self.string.splitlines()
        new_string = filter(self.solar_altitude_check, string_line)
        self.string = '\n'.join(new_string)

    def start_end_hour(self, sh, eh):
        string_line = self.string.splitlines()
        def filter_hour(string_line):
            hour = string_line.split()[2]
            return sh <= float(hour) <= eh
        new_string = filter(filter_hour, string_line)
        self.string = "\n".join(new_string)

    def read_epw(self):
        """."""
        with open(self.epw, 'r', newline=os.linesep) as epw:
            raw = epw.readlines()  # read-in epw content
        epw_header = raw[0].split(',')
        content = raw[8:]
        string = ""
        for li in content:
            line = li.split(',')
            month = int(line[1])
            day = int(line[2])
            hour = int(line[3]) - 1
            hours = hour + 0.5
            dir_norm = float(line[14])
            dif_hor = float(line[15])
            string += "%d %d %2.3f %.1f %.1f\n" \
                % (month, day, hours, dir_norm, dif_hor)
        self.string = string
        city = epw_header[1]
        country = epw_header[3]
        self.latitude = float(epw_header[6])
        self.longitude = -1 * float(epw_header[7])
        self.tz = int(float(epw_header[8])) * (-15)
        elevation = epw_header[9].rstrip()
        self.header = "place {}_{}\n".format(city, country)
        self.header += "latitude {}\n".format(self.latitude)
        self.header += "longitude {}\n".format(self.longitude)
        self.header += "time_zone {}\n".format(self.tz)
        self.header += "site_elevation {}\n".format(elevation)
        self.header += "weather_data_file_units 1\n"


class getEPW(object):
    """Download the closest EPW file from the given Lat and Lon."""
    _file_path_ = os.path.dirname(radutil.__file__)
    epw_url = "epw_url.csv"
    zip2latlon = "zip_latlon.txt"
    epw_url_path = os.path.join(_file_path_, 'data', epw_url)
    zip2latlon_path = os.path.join(_file_path_, 'data', zip2latlon)

    def __init__(self, lat: str, lon: str):
        self.lat = float(lat)
        self.lon = float(lon)
        distances = []
        urls = []
        with open(self.epw_url_path, 'r') as rdr:
            csvreader = csv.DictReader(rdr, delimiter=',')
            for row in csvreader:
                distances.append((float(row['Latitude']) - self.lat)**2
                                 + (float(row['Longitude']) - self.lon)**2)
                urls.append(row['URL'])
        min_idx = distances.index(min(distances))
        url = urls[min_idx]
        epw_fname = os.path.basename(url)
        user_agents = 'Mozilla/5.0 (Windows NT 6.1) '
        user_agents += 'AppleWebKit/537.36 (KHTML, like Gecko) '
        user_agents += 'Chrome/41.0.2228.0 Safari/537.3'
        request = urllib.request.Request(
            url, headers={'User-Agent': user_agents}
        )
        tmpctx = ssl.SSLContext()
        raw = ''
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, context=tmpctx) as resp:
                    raw = resp.read().decode()
                    break
            except urllib.error.HTTPError:
                time.sleep(1)
        assert not raw.startswith('404'), f'Bad URL:{url}'
        with open(epw_fname, 'w') as wtr:
            wtr.write(raw)
        self.fname = epw_fname

    @classmethod
    def from_zip(cls, zipcode):
        zipcode = str(zipcode)
        with open(cls.zip2latlon_path, 'r') as rdr:
            csvreader = csv.DictReader(rdr, delimiter='\t')
            for row in csvreader:
                if row['GEOID'] == zipcode:
                    lat = row['INTPTLAT']
                    lon = row['INTPTLONG']
                    break
            else:
                raise ValueError('zipcode not found in US')
        return cls(lat, lon)


def getwea():
    """Commandline program for generating a .wea file
    from downloaded EPW data."""
    parser = argparse.ArgumentParser(
        prog='getwea',
        description="Download the EPW files and convert it to a wea file")
    parser.add_argument('-a', help='latitude')
    parser.add_argument('-o', help='longitude')
    parser.add_argument('-z', help='zipcode (U.S. only)')
    parser.add_argument('-dh', action="store_true",
                        help='output only for daylight hours')
    parser.add_argument('-sh', type=float, default=0, help='start hour (float)')
    parser.add_argument('-eh', type=float, default=0, help='end hour (float)')
    args = parser.parse_args()
    if args.z is not None:
        epw = getEPW.from_zip(args.z)
    else:
        epw = getEPW(args.a, args.o)
    wea = epw2wea(epw=epw.fname, dh=args.dh, sh=args.sh, eh=args.eh)
    print(wea.wea)
