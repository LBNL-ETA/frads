"""
Routines for generating sky models
"""

import datetime
import logging
import math
import os
from pathlib import Path
import subprocess as sp
from typing import SupportsFloat, Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from frads import geom
from frads import parsers
from frads.types import WeaMetaData
from frads.types import WeaData
from frads import utils

logger: logging.Logger = logging.getLogger("frads.sky")

# Solar disk solid angle (sr)
SOLAR_SA = 6.7967e-5


def basis_glow(sky_basis: str) -> str:
    """
    Generate a set of regular sky and ground glow primitives string.

    Args:
        sky_basis(str): sky sampling basis, e.g. r1, r4
    Returns:
        ground and sky glow string, usually used for rfluxmtx calls.
    """
    grnd_str = grndglow()
    sky_str = skyglow(sky_basis)
    return grnd_str + sky_str


def skyglow(basis: str, upvect: str = "+Y") -> str:
    """
    Generate a set of skyglow string

    Args:
        basis(str): e.g., r1, r2, r4
        upvect(str): Optional, default=+Y
    Returns:
        A set of sky glow primitive string
    """
    sky_string = f"#@rfluxmtx u={upvect} h={basis}\n\n"
    sky_string += "void glow skyglow\n"
    sky_string += "0\n0\n4 1 1 1 0\n\n"
    sky_string += "skyglow source sky\n"
    sky_string += "0\n0\n4 0 0 1 180\n"
    return sky_string


def grndglow(basis: str = "u") -> str:
    """
    Generate a set of ground string
    Args:
        basis(str): Optional default=u
    Returns:
        A set of ground glow primitive string
    """
    ground_string = f"#@rfluxmtx h={basis}\n\n"
    ground_string += "void glow groundglow\n"
    ground_string += "0\n0\n4 1 1 1 0\n\n"
    ground_string += "groundglow source ground\n"
    ground_string += "0\n0\n4 0 0 -1 180\n\n"
    return ground_string


def gen_sun_source_full(mf: int) -> Tuple[str, str]:
    """
    Generate a full set of sun light sources according to Reinhart basis.

    Args:
        mf(int): multiplication factor, usually 1, 2, or 4.
    Returns:
        A tuple of full set of sun light and source primitive string
        and associated modifier string.
    """
    runlen = 144 * mf**2 + 3
    mod_str = os.linesep.join([f"sol{i}" for i in range(1, runlen)])
    dirs, omgs = utils.calc_reinsrc_dir(mf)
    lines = []
    for i, d in enumerate(dirs):
        lines.append(
            f"void light sol{i} 0 0 3 1 1 1 sol{i} source sun "
            f"0 0 4 {d.x:.6g} {d.y:.6g} {d.z:.6g} 0.533"
        )
    return os.linesep.join(lines) + os.linesep, mod_str


def gen_sun_source_culled(
    mf,
    smx_path: Optional[Path] = None,
    window_normals: Optional[List[geom.Vector]] = None,
) -> Tuple[str, str, str]:
    """
    Generate a culled set of sun sources based on either window orientation
    and/or climate-based sky matrix. The reduced set of sun sources will
    significantly speed up the direct-sun matrix generation.

    Args:
        mf(int): multiplication factor, usually 1, 2, or 4.
        smx_path(str): Optional, sky matrix path, usually the output of gendaymtx
        window_normals(str): Optional, window normals
    Returns:
        A tuple of culled set of sun light and source primitive string,
        corresponding modifier strings, and the full set of modifier string.
    """
    runlen = 144 * mf**2 + 3
    dirs, omgs = utils.calc_reinsrc_dir(mf)
    full_mod_str = os.linesep.join([f"sol{i}" for i in range(1, runlen)])
    win_norm = []
    if smx_path is not None:
        cmd1 = ["rmtxop", "-ff", "-c", ".3", ".6", ".1", "-t", str(smx_path)]
        cmd2 = ["getinfo", "-"]
        cmd3 = ["total", f"-if{runlen-1}", "-t,"]
        proc1 = sp.run(cmd1, check=True, stdout=sp.PIPE)
        proc2 = sp.run(cmd2, check=True, input=proc1.stdout, stdout=sp.PIPE)
        proc3 = sp.run(cmd3, check=True, input=proc2.stdout, stdout=sp.PIPE)
        dtot = [float(i) for i in proc3.stdout.split(b",")]
    else:
        dtot = [1] * runlen
    out_lines = []
    mod_str = []
    if window_normals is not None:
        win_norm = window_normals
        for i, d in enumerate(dirs):
            _mod = "sol" + str(i)
            v = 0
            if dtot[i] > 0:
                for norm in win_norm:
                    if norm * d < 0:
                        v = 1
                        mod_str.append(_mod)
                        break
            out_lines.append(
                f"void light sol{i} 0 0 3 {v} {v} {v} sol{i} source sun "
                f"0 0 4 {d.x:.6g} {d.y:.6g} {d.z:.6g} 0.533"
            )
    else:
        for i, d in enumerate(dirs):
            _mod = f"sol{i}"
            v = 0
            if dtot[i] > 0:
                v = 1
                mod_str.append(_mod)
            out_lines.append(
                f"void light sol{i} 0 0 3 {v} {v} {v} sol{i} source sun "
                f"0 0 4 {d.x:.6g} {d.y:.6g} {d.z:.6g} 0.533"
            )
    logger.debug(out_lines)
    logger.debug(mod_str)
    return os.linesep.join(out_lines), os.linesep.join(mod_str), full_mod_str


def gendaymtx(
    out: Union[str, Path],
    mf: int,
    data: Optional[Sequence[WeaData]] = None,
    meta: Optional[WeaMetaData] = None,
    wpath: Optional[Path] = None,
    direct: bool = False,
    solar: bool = False,
    onesun: bool = False,
    rotate: Optional[float] = None,
    binary: bool = False,
) -> List[str]:
    """
    Call gendaymtx to generate a sky/sun matrix
    and write results to out.  It takes either a .wea file path
    or wea data and metadata (defined in frads.types).
    If both are provided, .wea file path will be used.

    Args:
        out(str or pathlib.Path): outpath file path
        mf(int): multiplication factor
        data(Sequence[WeaData], optional): A sequence of WeaData.
        meta(WeaMetaData, optional): A instance of WeaMetaData object.
        wpath(Path, optional): .wea file path.
        direct(bool, optional): Whether to generate sun-only sky matrix.
        solar(bool, optional): Whether to generate sky matrix if solar spectrum.
        onesun(bool, optional): Whether to generate single sun matrix (five-phase).
        rotate(float, optional): rotate the sky counter-clock wise, looking down.
        binary(bool, optional): Whether to have outputs in single precision floats.
    Returns:
        cmd(List[str]): the gendaymtx command called.
    Raises:
        ValueError: An error occurs if neither a .wea path nor wea data is provided.
    """
    stdin = None
    cmd = ["gendaymtx", "-m", str(mf)]
    if binary:
        cmd.append("-of")
    if direct:
        cmd.append("-d")
    if onesun:
        cmd.extend(["-5", ".533"])
    if rotate is not None:
        cmd.extend(["-r", str(rotate)])
    if solar:
        cmd.append("-O1")
    if wpath is not None:
        cmd.append(str(wpath))
    elif (data is not None) and (meta is not None):
        wea_input = meta.wea_header() + "\n".join(map(str, data))
        stdin = wea_input.encode("utf-8")
    else:
        raise ValueError("Need to specify either .wea path or wea data.")
    with open(out, "wb") as wtr:
        logger.info("Calling gendaymtx with:\n%s", " ".join(cmd))
        sp.run(cmd, check=True, input=stdin, stdout=wtr)
    return cmd


def gen_perez_sky(row, meta, grefl: float = 0.2, spect: str = "0", rotate=None) -> str:
    gendaylit = gendaylit_cmd(
        str(row.month),
        str(row.day),
        str(row.hours),
        str(meta.latitude),
        str(meta.longitude),
        str(meta.timezone),
        dir_norm_ir=str(row.dni),
        dif_hor_ir=str(row.dhi),
    )
    out = []
    rot = f"| xform -rz {rotate}" if rotate is not None else ""
    out.append(f"!{' '.join(gendaylit)}{rot} \n")
    out.append("skyfunc glow sglow 0 0 4 1 1 1 0\n")
    out.append("sglow source sky 0 0 4 0 0 1 180\n")
    out.append("sglow source ground 0 0 4 0 0 -1 180\n")
    return " ".join(out)


def gendaylit_cmd(
    month: str,
    day: str,
    hours: str,
    lat: str,
    lon: str,
    tzone: str,
    year: str = None,
    dir_norm_ir: str = None,
    dif_hor_ir: Optional[str] = None,
    dir_hor_ir: Optional[str] = None,
    dir_norm_il: Optional[str] = None,
    dif_hor_il: str = None,
    solar: bool = False,
) -> list:
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


def solar_angle(
    lat: float,
    lon: float,
    mer: float,
    month,
    day: int,
    hour,
) -> Tuple[float, float]:
    """
    Simplified translation from the Radiance sun.c and gensky.c code.
    """
    latitude_r = math.radians(lat)
    longitude_r = math.radians(lon)
    s_meridian = math.radians(mer)
    mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    julian_date = mo_da[month - 1] + day

    solar_decline = 0.4093 * math.sin((2 * math.pi / 365) * (julian_date - 81))

    solar_time = hour + (
        0.170 * math.sin((4 * math.pi / 373) * (julian_date - 80))
        - 0.129 * math.sin((2 * math.pi / 355) * (julian_date - 8))
        + (12 / math.pi) * (s_meridian - longitude_r)
    )
    altitude = math.asin(
        math.sin(latitude_r) * math.sin(solar_decline)
        - math.cos(latitude_r)
        * math.cos(solar_decline)
        * math.cos(solar_time * (math.pi / 12))
    )
    azimuth = -math.atan2(
        math.cos(solar_decline) * math.sin(solar_time * (math.pi / 12.0)),
        -math.cos(latitude_r) * math.sin(solar_time)
        - math.sin(latitude_r)
        * math.cos(solar_decline)
        * math.cos(solar_time * (math.pi / 12)),
    )

    return altitude, azimuth


def start_end_hour(data: Sequence[WeaData], sh: float, eh: float) -> Sequence[WeaData]:
    """Remove wea data entries outside of the
    start and end hour."""
    if sh == 0 and eh == 0:
        return data
    return [row for row in data if sh <= (row.time.hour + row.time.minute / 60) <= eh]


def check_sun_above_horizon(data, metadata):
    """Remove non-daylight hour entries."""

    def solar_altitude_check(row: WeaData):
        alt, _ = solar_angle(
            metadata.latitude,
            metadata.longitude,
            metadata.timezone,
            row.time.month,
            row.time.day,
            row.time.hour + row.time.minute / 60,
        )
        return alt > 0

    return [row for row in data if solar_altitude_check(row)]


def filter_data_with_zero_dni(data):
    """Filter out data entries with zero direct normal irradiance."""
    return [row for row in data if row.dni != 0]


def solar_minute(data: WeaData) -> int:
    # assuming never leap year
    mo_da = [0,31,59,90,120,151,181,212,243,273,304,334]
    jd = mo_da[data.time.month - 1] + data.time.day
    return 24 * 60 * (jd - 1) + int(data.time.hour * 60.0 + data.time.minute + 0.5)


def filter_data_by_direct_sun(
    data: Sequence[WeaData],
    meta: WeaMetaData,
    window_normal: Optional[Sequence[geom.Vector]] = None,
) -> List[WeaData]:
    """
    Remove wea data entries with zero solar luminance according to
    Perez All-Weather sky model. If window normal supplied,
    eliminate entries not seen by window. Window field of view
    is 176 deg with 2 deg tolerance on each side.

    Args:
        data: Sequence[WeaData],
        meta: WeaMetaData,
        window_normal: Optional[Sequence[geom.Vector]] = None,
    Returns:
        data(List[WeaData]):
    """
    wea_input = meta.wea_header() + "\n".join(map(str, data))
    proc = sp.run(
        ["gendaymtx", "-u", "-D", "-"],
        check=True,
        input=wea_input,
        encoding="ascii",
        stderr=sp.PIPE,
        stdout=sp.PIPE,
    )
    prims = parsers.parse_primitive(proc.stdout.splitlines())
    light_prims = [prim for prim in prims if prim.ptype == "light"]
    keep_minutes = []
    if window_normal is not None:
        source_prims = [prim for prim in prims if prim.ptype == "source"]
        for lpr, spr in zip(light_prims, source_prims):
            if lpr.real_arg[1] > 0:
                sdir = geom.Vector(*spr.real_arg[1:4])
                for normal in window_normal:
                    if normal * sdir < -0.035:  # 2deg tolerance
                        keep_minutes.append(int(spr.modifier.lstrip("solar")))
                        break
    else:
        for lpr in light_prims:
            if lpr.real_arg[1] > 0:
                keep_minutes.append(int(lpr.identifier.lstrip("solar")))
    # inminutes = [solar_minute(d) for d in data]
    inminutes = []
    extra_day = 0
    for d in data:
        inm = solar_minute(d)
        if d.time.month == 2 and d.time.day == 29:
            extra_day = 1440
        inm += extra_day
        inminutes.append(inm)
    new_dataline = [data for data, minu in zip(data, inminutes) if minu in keep_minutes]
    return new_dataline


def filter_wea(
    wea_data: List[WeaData],
    meta_data: WeaMetaData,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
    daylight_hours_only: bool = False,
    remove_zero: bool = False,
    window_normals: Optional[List[geom.Vector]] = None,
) -> Tuple[List[WeaData], List[Any]]:
    """
    Obtain and prepare weather file data.

    Args:
        wea_data(List[WeaData]): A list of WeaData.
        meta_data(WeaMetaData): A instance of WeaMetaData object.
        start_hour(float, optional): Filter out wea data before this hour.
        end_hour(float, optional): Filter out wea data after this hour.
        daylight_hours_only(bool, optional): Filter out wea data below horizon.
        remove_zero(bool, optional): Filter out wea data with zero DNI.
        window_normals(List[geom.Vector], optional): Filter out wea data with direct
            sun not seen by these window normals.
    Returns:
        wea_data(List[WeaData]): Filterd list of wea data
        datetime_stamps(list): Remaining datetime stamps
    """
    logger.info("Filtering wea data, starting with %d rows", len(wea_data))
    if (start_hour is not None) and (end_hour is not None):
        wea_data = start_end_hour(wea_data, start_hour, end_hour)
        logger.info(
            "Filtering out hours outside of %f and %f: %d rows remaining",
            start_hour,
            end_hour,
            len(wea_data),
        )
    if daylight_hours_only:
        wea_data = check_sun_above_horizon(wea_data, meta_data)
        logger.info("Filtering by daylight hours: %d rows remaining", len(wea_data))
    if remove_zero:
        wea_data = filter_data_with_zero_dni(wea_data)
    if window_normals is not None:
        wea_data = filter_data_by_direct_sun(
            wea_data, meta_data, window_normal=window_normals
        )
        logger.info(
            "Filtering zero DNI hours and suns not seen by window: %d rows remaining",
            len(wea_data),
        )
    datetime_stamps = [str(row) for row in wea_data]
    if len(wea_data) == 0:
        logger.warning("Empty wea file")
    return wea_data, datetime_stamps
