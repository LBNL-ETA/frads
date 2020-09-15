"""
Routines for generating sky models
"""

import csv
import datetime
import math
import os
import subprocess as sp
import tempfile as tf
import urllib.request
from frads import radutil
import pdb

LSEP = os.linesep

def basis_glow(sky_basis):
    grnd_str = grndglow()
    sky_str = skyglow(sky_basis)
    return grnd_str + sky_str


def skyglow(basis, upvect='+Y'):
    sky_string = f"#@rfluxmtx u={upvect} h={basis}{LSEP*2}"
    sky_string += f"void glow skyglow{LSEP}"
    sky_string += f"0{LSEP}0{LSEP}4 1 1 1 0{LSEP*2}"
    sky_string += f"skyglow source sky{LSEP}"
    sky_string += f"0{LSEP}0{LSEP}4 0 0 1 180{LSEP}"
    return sky_string


def grndglow(basis='u'):
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
        self.rsrc = radutil.reinsrc(mf)
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

    def gen_cull(self, smx_path=None, window_paths=None):
        """Generate culled sun sources based on window orientation and climate based
        sky matrix. The reduced sun sources will significantly speed up the matrix
        generation."""
        if window_paths is not None:
            wprims = []
            for wpath in window_paths:
                with open(wpath) as rdr:
                    wprims.append(radutil.parse_primitive(rdr.readlines()))
            wprims = [i for g in wprims for i in g]
            win_norm = [
                p['polygon'].normal().to_list() for p in wprims
                if p['type'] == 'polygon'
            ]
        else:
            win_norm = [[0, 0, -1]]
        if smx_path is not None:
            cmd = f"rmtxop -ff -c .3 .6 .1 -t {smx_path} "
            cmd += "| getinfo - | total -if5186 -t,"
            dtot = [float(i) for i in sp.check_output(cmd, shell=True).split(b',')]
        else:
            dtot = [1] * self.runlen
        out_lines = []
        for i in range(1, self.runlen):
            dirs = self.rsrc.dir_calc(i)
            if dtot[i - 1] > 0:
                for norm in win_norm:
                    v = 0
                    if sum([i * j for i, j in zip(norm, dirs)]) < 0:
                        v = 1
                        break
            else:
                v = 0
            line = f"void light sol{i} 0 0 3 {v} {v} {v} sol{i} "
            line += "source sun 0 0 4 {:.6g} {:.6g} {:.6g} 0.533".format(*dirs)
            out_lines.append(line)
        return LSEP.join(out_lines)


def epw2sunmtx(epw_path):
    """Generate reinhart 6 sun matrix file from a epw file."""
    smx_path = radutil.basename(epw_path) + ".smx"
    with tf.NamedTemporaryFile() as wea:
        cmd = f"epw2wea {epw_path} {wea.name}"
        sp.call(cmd, shell=True)
        cmd = f"gendaymtx -od -m 6 -d -5 .533 {wea.name} > {smx_path}"  # large file
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
        cmd = f"gendaymtx -od -m 6 -d -5 .533 {wea.name} > {smx_path}"
        sp.call(cmd, shell=True)
    return smx_path


def gendaymtx(data_entry, lat, lon, timezone, ele, mf=4, direct=False,
              solar=False, onesun=False, rotate=0, binary=False):
    """."""
    sun_only = ' -d' if direct else ''
    spect = ' -O1' if solar else ' -O0'
    _five = ' -5 .533' if onesun else ''
    bi = '' if binary == False or os.name == 'nt' else ' -o' + binary
    linesep = r'& echo' if os.name == 'nt' else LSEP
    wea_head = f"place test{linesep}latitude {lat}{linesep}longitude {lon}{linesep}"
    wea_head += f"time_zone {timezone}{linesep}site_elevation {ele}{linesep}"
    wea_head += f"weather_data_file_units 1{linesep}"
    skv_cmd = f"gendaymtx -r {rotate} -m {mf}{sun_only}{_five}{spect}{bi}"
    if len(data_entry) > 1000:
        _wea, _path = tf.mkstemp()
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


def parse_csv(csv_path, ftype='csv', dt_col="date_time", dt_format="%Y%m%d %H:%M:%S",
              dni_col='DNI', dhi_col='DHI', stime=None, etime=None):
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
            data_entry.append([_month, _day, _hour, _minute, _hours, _dni, _dhi])
    return data_entry


def sky_cont(mon, day, hrs, lat, lon, mer, dni, dhi, year=None, grefl=.2, spect='0'):
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

def solar_angle(*, lat, lon, mer, month, day, hour):
    """Simplified translation from the Radiance sun.c and gensky.c code.

    This function test if the solar altitude is greater than zero
    """
    latitude_r = math.radians(lat)
    longitude_r = math.radians(lon)
    s_meridian = math.radians(mer)
    mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    julian_date = mo_da[month - 1] + day

    solar_decline = 0.4093 * math.sin((2 * math.pi / 368) * (julian_date - 81))

    solar_time = hour + (0.170 * math.sin((4 * math.pi / 373) * (julian_date - 80))
                         - 0.129 * math.sin((2 * math.pi / 355) * (julian_date - 8))
                         + 12 * (s_meridian - longitude_r) / math.pi)

    altitude = math.asin(math.sin(latitude_r) * math.sin(solar_decline)
                    - math.cos(latitude_r) * math.cos(solar_decline)
                    * math.cos(solar_time * (math.pi / 12)))

    return altitude > 0

class epw2wea(object):
    """."""

    def __init__(self, *, epw, dh, sh, eh):
        """."""
        self.epw = epw
        #self.wea = wea
        self.read_epw()  # read-in epw/tmy data

        if sh is not None:
            self.s_hour(float(sh))

        if eh is not None:
            self.e_hour(float(eh))

        if dh:
            self.daylight()  # filter out non-daylight hours if asked

        self.wea = self.header + self.string
        self.dt_string = []
        for line in self.string.splitlines():
            entry = line.split()
            mo = int(entry[0])
            da = int(entry[1])
            hr = int(float(entry[2]))
            self.dt_string.append(f"{mo:02d}{da:02d}_{hr:02d}30")


    def daylight(self):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line
                      if (float(li.split()[3]) > 0) and (float(li.split()[4]) > 0)]
        self.string = "\n".join(new_string)

    def s_hour(self, sh):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line if float(li.split()[2]) >= sh]
        self.string = "\n".join(new_string)

    def e_hour(self, eh):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line if float(li.split()[2]) <= eh]
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
    #assert os.path.isfile(epw_url_path), 'File not found: {}'.format(epw_url_path)
    zip2latlon_path = os.path.join(_file_path_, 'data', zip2latlon)
    #assert os.path.isfile(zip2latlon_path),\
    #        'File not found: {}'.format(zip2latlon_path)

    def __init__(self, lat, lon):
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
        try:
            headers = ('User-Agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3')
            opener = urllib.request.build_opener()
            opener.addheaders = [headers]
            with opener.open(url) as resp:
                raw = resp.read().decode()
        except OSError as e:
            raise e
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
                raise 'zipcode not found in US'
        return cls(lat, lon)

