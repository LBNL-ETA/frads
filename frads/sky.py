"""
Routines for generating sky models
"""

import logging
import math
import os
from pathlib import Path
import subprocess as sp
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from frads import geom
from frads import parsers
from frads.types import WeaMetaData
from frads.types import WeaDataRow
from frads import utils

logger = logging.getLogger("frads.sky")


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
    sky_string = f"#@rfluxmtx u={upvect} h={basis}\n\n"
    sky_string += "void glow skyglow\n"
    sky_string += "0\n0\n4 1 1 1 0\n\n"
    sky_string += "skyglow source sky\n"
    sky_string += "0\n0\n4 0 0 1 180\n"
    return sky_string


def grndglow(basis='u') -> str:
    ground_string = f"#@rfluxmtx h={basis}\n\n"
    ground_string += "void glow groundglow\n"
    ground_string += "0\n0\n4 1 1 1 0\n\n"
    ground_string += "groundglow source ground\n"
    ground_string += "0\n0\n4 0 0 -1 180\n\n"
    return ground_string


class Gensun(object):
    """Generate sun sources for matrix generation."""

    def __init__(self, mf: int):
        """."""
        self.runlen = 144 * mf**2 + 3
        self.rsrc = utils.Reinsrc(mf)
        self.mod_str = os.linesep.join([f'sol{i}' for i in range(1, self.runlen)])

    def gen_full(self):
        """Generate full treganza based sun sources."""
        out_lines = []
        for i in range(1, self.runlen):
            dirs = self.rsrc.dir_calc(i)
            line = f"void light sol{i} 0 0 3 1 1 1 sol{i} "
            line += "source sun 0 0 4 {:.6g} {:.6g} {:.6g} 0.533".format(*dirs[:-1])
            out_lines.append(line)
        return os.linesep.join(out_lines)+os.linesep

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
                dirs = geom.Vector(*self.rsrc.dir_calc(i)[:-1])
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
                dirs = geom.Vector(*self.rsrc.dir_calc(i)[:-1])
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
        return os.linesep.join(out_lines), os.linesep.join(mod_str)


def gendaymtx(out: Union[str, Path],
              mf: int,
              data: Optional[Sequence[WeaDataRow]]=None,
              meta: Optional[WeaMetaData]=None,
              wpath: Optional[Path]=None,
              direct=False,
              solar=False,
              onesun=False,
              rotate: Optional[float]=None,
              binary=False) -> None:
    """
    Call gendaymtx to generate a sky/sun matrix and write results to out.
    """
    stdin = None
    cmd = ['gendaymtx', "-m", str(mf)]
    if binary:
        cmd.append('-of')
    if direct:
        cmd.append('-d')
    if onesun:
        cmd.extend(['-5', '.533'])
    if rotate is not None:
        cmd.extend(["-r", str(rotate)])
    if solar:
        cmd.append('-O1')
    if wpath is not None:
        cmd.append(str(wpath))
    elif (data is not None) and (meta is not None):
        wea_input = meta.wea_header() + "\n".join(map(str, data))
        stdin = wea_input.encode('utf-8')
    else:
        raise ValueError("Need to specify either .wea path or wea data.")
    with open(out, 'wb') as wtr:
        sp.run(cmd, input=stdin, stdout=wtr)


def gendaymtx_cmd(data_entry: List[str], metadata: WeaMetaData,
                  mf=4, direct=False, solar=False, onesun=False,
                  rotate=0, binary=False):
    """."""
    sun_only = ' -d' if direct else ''
    spect = ' -O1' if solar else ' -O0'
    _five = ' -5 .533' if onesun else ''
    bi = '' if binary is False or os.name == 'nt' else ' -o' + binary
    linesep = r'& echo' if os.name == 'nt' else os.linesep
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


def sky_cont(mon, day, hrs, lat, lon, mer, dni, dhi,
             year=None, grefl=.2, spect='0', rotate=None):
    out_str = f'!gendaylit {mon} {day} {hrs} '
    out_str += f'-a {lat} -o {lon} -m {mer} '
    if year is not None:
        out_str += f'-y {year} '
    out_str += f'-W {dni} {dhi} -g {grefl} -O {spect}{os.linesep*2}'
    if rotate is not None:
        out_str += f"| xform -rz {rotate}"
    out_str += f'skyfunc glow skyglow 0 0 4 1 1 1 0{os.linesep*2}'
    out_str += f'skyglow source sky 0 0 4 0 0 1 180{os.linesep*2}'
    out_str += f'skyfunc glow groundglow 0 0 4 1 1 1 0{os.linesep*2}'
    out_str += f'groundglow source ground 0 0 4 0 0 -1 180{os.linesep}'
    return out_str


def gendaylit_cmd(month: str, day: str, hours: str,
                  lat: str, lon: str, tzone: str,
                  year: str = None, dir_norm_ir: str = None,
                  dif_hor_ir: Optional[str] = None, dir_hor_ir: Optional[str] = None,
                  dir_norm_il: Optional[str] = None, dif_hor_il: str = None,
                  solar: bool = False) -> list:
    """Get a gendaylit command as a list."""
    cmd = ["gendaylit", month, day, hours]
    cmd += ["-a", lat, "-o", lon, "-m", tzone]
    if year is not None:
        cmd += ["-y", year]
    if None not in (dir_norm_ir, dif_hor_ir):
        cmd += ["-W", str(dir_norm_ir), str(dif_hor_ir)]
    if None not in (dir_hor_ir, dif_hor_ir):
        cmd += ["-G", str(dir_hor_ir), str(dif_hor_ir)]
    if None not in (dir_norm_il, dif_hor_il):
        cmd += ["-L", str(dir_norm_il), str(dif_hor_il)]
    if solar:
        cmd += ["-O", "1"]
    return cmd


def solar_angle(lat, lon, mer, month, day, hour):
    """
    Simplified translation from the Radiance sun.c and gensky.c code.
    """
    latitude_r = math.radians(lat)
    longitude_r = math.radians(lon)
    s_meridian = math.radians(mer)
    mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    julian_date = mo_da[month - 1] + day

    solar_decline = 0.4093 * math.sin((2 * math.pi / 365) * (julian_date - 81))

    solar_time = hour + (0.170 * math.sin((4 * math.pi / 373) * (julian_date - 80)) - 0.129 * math.sin((2 * math.pi / 355) * (julian_date - 8)) + (12/math.pi) * (s_meridian - longitude_r))
    altitude = math.asin(math.sin(latitude_r) * math.sin(solar_decline) - math.cos(latitude_r) * math.cos(solar_decline) * math.cos(solar_time * (math.pi / 12)))
    azimuth = -math.atan2(math.cos(solar_decline)*math.sin(solar_time*(math.pi/12.)), -math.cos(latitude_r)*math.sin(solar_time) - math.sin(latitude_r)*math.cos(solar_decline) * math.cos(solar_time*(math.pi/12)))

    return altitude, azimuth


def start_end_hour(data: Sequence[WeaDataRow], sh: float, eh: float):
    """Remove wea data entries outside of the
    start and end hour."""
    if sh == 0 and eh == 0:
        return data

    return [row for row in data if sh <= row.hours <= eh]


def check_sun_above_horizon(data, metadata):
    """Remove non-daylight hour entries."""
    def solar_altitude_check(row: WeaDataRow):
        alt, _ = solar_angle(metadata.latitude, metadata.longitude,
                             metadata.timezone, row.month,
                             row.day, row.hours)
        return alt > 0
    return [row for row in data if solar_altitude_check(row)]


def filter_data_with_zero_dni(data):
    """Filter out data entries with zero direct normal irradiance."""
    return [row for row in data if row.dni != 0]


def filter_data_by_direct_sun(
    data: Sequence[WeaDataRow],
    meta: WeaMetaData,
    window_normal: Optional[Sequence[geom.Vector]]=None,
) -> List[WeaDataRow]:
    """
    Remove wea data entries with zero solar luminance according to
    Perez All-Weather sky model. If window normal supplied,
    eliminate entries not seen by window. Window field of view
    is 176 deg with 2 deg tolerance on each side.
    """
    if window_normal is not None:
        logger.warning("Window normals detected:")
        for norm in window_normal:
            logger.warning(str(norm))
        if len(window_normal) == 0:
            window_normal = None
    new_dataline = []
    for row in data:
        cmd = gendaylit_cmd(str(row.month), str(row.day), str(row.hours),
                            str(meta.latitude), str(meta.longitude), str(meta.timezone),
                            dir_norm_ir=str(row.dni), dif_hor_ir=str(row.dhi))
        process = sp.run(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        if process.stderr == b'':
            primitives = parsers.parse_primitive(process.stdout.decode().splitlines())
            light = float(primitives[0].real_arg.split()[2])
            dirs = geom.Vector(*list(map(float, primitives[1].real_arg.split()[1:4])))
            if light > 0:
                if window_normal is not None:
                    for normal in window_normal:
                        if normal * dirs < -0.035:  # 2deg tolerance
                            logger.debug(f"{row.month} {row.day} {row.hours} inside of 176deg of {normal}")
                            new_dataline.append(row)
                            break
                else:
                    new_dataline.append(row)
        else:
            logger.warning(process.stderr.decode())
    return new_dataline


def filter_wea(
    wea_data: Sequence[WeaDataRow],
    meta_data: WeaMetaData,
    start_hour: Optional[float]=None,
    end_hour: Optional[float]=None,
    daylight_hours_only=False,
    remove_zero=False,
    window_normals=None,
) -> Tuple[List[WeaDataRow], List[str]]:
    """Obtain and prepare weather file data."""
    logger.info(f"Filtering wea data, starting with {len(wea_data)} rows")
    if (start_hour is not None) and (end_hour is not None):
        wea_data = start_end_hour(wea_data, start_hour, end_hour)
        logger.info(f"Filtering out hours outside of {start_hour} and {end_hour}: {len(wea_data)} rows remaining")
    if daylight_hours_only:
        wea_data = check_sun_above_horizon(wea_data, meta_data)
        logger.info(f"Filtering by daylight hours: {len(wea_data)} rows remaining")
    if remove_zero:
        wea_data = filter_data_with_zero_dni(wea_data)
    if window_normals is not None:
        wea_data = filter_data_by_direct_sun(wea_data, meta_data, window_normal=window_normals)
        logger.info(f"Filtering out zero DNI hours and suns not seen by window: {len(wea_data)} rows remaining")
    datetime_stamps = [row.dt_string() for row in wea_data]
    if len(wea_data) == 0:
        logger.warning("Empty wea file")
    return wea_data, datetime_stamps
