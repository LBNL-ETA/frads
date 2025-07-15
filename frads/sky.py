"""
Routines for generating sky models
"""

import datetime
import logging
import math
import os
from pathlib import Path
from typing import NamedTuple, Sequence

import pyradiance as pr

logger: logging.Logger = logging.getLogger("frads.sky")


class WeaMetaData(NamedTuple):
    """Weather related meta data object.

    Attributes:
        city: City.
        country: Country.
        latitude: Latitude.
        longitude: Longitude.
        timezone: Timezone as standard meridian.
        elevation: Site elevation (m).
    """

    city: str
    country: str
    latitude: float
    longitude: float
    timezone: int
    elevation: float
    unit: int = 1

    def wea_header(self) -> str:
        """Return a .wea format header."""
        return (
            f"place {self.city}_{self.country}\n"
            f"latitude {self.latitude}\n"
            f"longitude {self.longitude}\n"
            f"time_zone {self.timezone}\n"
            f"site_elevation {self.elevation}\n"
            f"weather_data_file_units {self.unit}\n"
        )


class WeaData(NamedTuple):
    """Weather related data object.

    Attributes:
        month: Month.
        day: Day.
        hour: Hour.
        minute: Minutes.
        second: Seconds.
        hours: Times with minutes as fraction.
        dni: Direct normal irradiance (W/m2) or illuminance (lux).
        dhi: Diffuse horizontal irradiance (W/m2) or illuminance (lux).
        aod: Aeroal Optical Depth (default = 0).
        cc: Cloud cover (default = 0).
        year: default = 2000.
    """

    time: datetime.datetime
    dni: float
    dhi: float
    aod: float = 0
    cc: float = 0

    def __str__(self) -> str:
        return f"{self.time.month} {self.time.day} {self.time.hour + self.time.minute / 60} {self.dni} {self.dhi} {self.aod} {self.cc}"

    def dt_str(self) -> str:
        return f"{self.time.month:02d}{self.time.day:02d}_{self.time.hour:02d}{self.time.minute:02d}"


def parse_epw(epw_str: str) -> tuple:
    """Parse epw file and return wea header and data.

    Args:
        epw_str: String containing epw file.

    Returns:
        Tuple of meta data and wea data.
    """
    raw = epw_str.splitlines()
    epw_header = raw[0].split(",")
    content = raw[8:]
    data = []
    for li in content:
        line = li.split(",")
        year = int(line[0])
        month = int(line[1])
        day = int(line[2])
        hour = int(line[3]) - 1
        dir_norm = float(line[14])
        dif_hor = float(line[15])
        cc = float(line[19])
        aod = float(line[26])
        data.append(
            WeaData(
                datetime.datetime(year, month, day, hour, 30),
                dir_norm,
                dif_hor,
                cc,
                aod,
            )
        )
    city = epw_header[1]
    country = epw_header[3]
    latitude = float(epw_header[6])
    longitude = -1 * float(epw_header[7])
    tz = int(float(epw_header[8])) * (-15)
    elevation = float(epw_header[9].rstrip())
    meta_data = WeaMetaData(city, country, latitude, longitude, tz, elevation)
    return meta_data, data


def parse_wea(wea_str: str) -> tuple[WeaMetaData, list[WeaData]]:
    """Parse a WEA weather file.

    Args:
        wea_str: String containing the complete WEA file content.

    Returns:
        A tuple containing:
            - WeaMetaData: Location and site information
            - list[WeaData]: List of weather data entries for each time step

    Examples:
        >>> with open('weather.wea', 'r') as f:
        ...     content = f.read()
        >>> metadata, data = parse_wea(content)
        >>> print(f"Location: {metadata.city}")
        >>> print(f"Data points: {len(data)}")
    """
    lines = wea_str.splitlines()
    place = lines[0].split(" ", 1)[1]
    lat = float(lines[1].split(" ", 1)[1])
    lon = float(lines[2].split(" ", 1)[1])
    tz = int(float(lines[3].split(" ", 1)[1]))
    ele = float(lines[4].split(" ", 1)[1])
    meta_data = WeaMetaData(place, "", lat, lon, tz, ele)
    year = datetime.datetime.today().year
    data = []
    for li in lines[6:]:
        if li.strip() == "":
            continue
        line = li.split()
        month = int(line[0])
        day = int(line[1])
        hours = float(line[2])
        hour = int(hours)
        minute = int((hours - hour) * 60)
        dir_norm = float(line[3])
        dif_hor = float(line[4])
        data.append(
            WeaData(
                datetime.datetime(year, month, day, hour, minute), dir_norm, dif_hor
            )
        )
    return meta_data, data


def parse_epw_file(file: os.PathLike) -> tuple:
    """Parse an epw file using parse_epw.

    Args:
        file: Path to epw file.

    Returns:
        tuple of meta data and epw data.
    """
    with open(file, "r") as f:
        epw_str = f.read()
    return parse_epw(epw_str)


def parse_wea_file(file: os.PathLike) -> tuple:
    """Parse a wea file using parse_wea.

    Args:
        file: Path to wea file.

    Returns:
        tuple of meta data and wea data.
    """
    with open(file, "r") as f:
        wea_str = f.read()
    return parse_wea(wea_str)


def genskymtx(
    data: None | Sequence[WeaData] = None,
    meta: None | WeaMetaData = None,
    wpath: None | str | Path = None,
    onesun: bool = False,
    header: bool = True,
    average: bool = False,
    sun_only: bool = False,
    sky_only: bool = False,
    sun_file: None | str = None,
    sun_mods: None | str = None,
    daylight_hours_only: bool = False,
    sky_color: None | list[float] = None,
    ground_color: None | list[float] = None,
    rotate: None | float = None,
    outform: None | str = None,
    solar_radiance: bool = False,
    mfactor: int = 1,
) -> bytes:
    """Call gendaymtx to generate a sky/sun matrix
    Write results to out.  It takes either a .wea file path
    or wea data and metadata (defined in frads.types).
    If both are provided, .wea file path will be used.

    Args:
        data: A list of WeaData objects.
        meta: A WeaMetaData object.
        wpath: A .wea file path.
        onesun: If True, only one sun will be generated.
        header: If True, a header will be included in the output.
        average: If True, the output will be averaged.
        sun_only: If True, only sun will be generated.
        sky_only: If True, only sky will be generated.
        sun_file: A sun file path.
        sun_mods: A sun modifier.
        daylight_hours_only: If True, only daylight hours will be generated.
        sky_color: A list of sky color values.
        ground_color: A list of ground color values.
        rotate: A rotation value.
        outform: An output format.
        solar_radiance: If True, solar radiance will be generated.
        mfactor: An mfactor value.

    Returns:
        A bytes object containing the output.

    Raises:
        ValueError: An error occurs if neither a .wea path nor wea data is provided.
    """
    if wpath is None:
        if data is not None and meta is not None:
            inp = gen_wea(
                [d.time for d in data],
                [d.dni for d in data],
                [d.dhi for d in data],
                meta.latitude,
                meta.longitude,
                meta.timezone,
                elevation=meta.elevation,
                location=meta.city + meta.country,
            ).encode()
        else:
            raise ValueError("Either a .wea path or wea data is required.")
    else:
        inp = wpath
    _out = pr.gendaymtx(
        inp,
        header=header,
        average=average,
        sun_only=sun_only,
        sky_only=sky_only,
        solar_radiance=solar_radiance,
        sun_file=sun_file,
        sun_mods=sun_mods,
        daylight_hours_only=daylight_hours_only,
        sky_color=sky_color,
        ground_color=ground_color,
        rotate=rotate,
        outform=outform,
        onesun=onesun,
        mfactor=mfactor,
    )
    return _out


def gen_perez_sky(
    dt: datetime.datetime,
    latitude: float,
    longitude: float,
    timezone: int,
    year: None | int = None,
    dirnorm: None | float = None,
    diffhor: None | float = None,
    dirhor: None | float = None,
    dirnorm_illum: None | float = None,
    diffhor_illum: None | float = None,
    solar: bool = False,
    grefl: None | float = None,
    rotate: None | float = None,
) -> bytes:
    """Generate a Perez sky model using the gendaylit Radiance command.

    Creates a realistic sky luminance distribution based on the Perez all-weather
    sky model, which accounts for varying sky conditions from clear to overcast.

    Args:
        dt: Date and time for the sky calculation.
        latitude: Site latitude in degrees (positive North, negative South).
        longitude: Site longitude in degrees (positive East, negative West).
        timezone: Standard meridian for timezone in degrees.
        year: Year for the calculation (optional, uses dt.year if None).
        dirnorm: Direct normal irradiance in W/m² (optional).
        diffhor: Diffuse horizontal irradiance in W/m² (optional).
        dirhor: Direct horizontal irradiance in W/m² (optional).
        dirnorm_illum: Direct normal illuminance in lux (optional).
        diffhor_illum: Diffuse horizontal illuminance in lux (optional).
        solar: If True, include solar disk in the sky model.
        grefl: Ground reflectance (0.0-1.0, default varies by season).
        rotate: Sky rotation angle in degrees (optional).

    Returns:
        bytes: the sky primitive.
    """
    sun = pr.gendaylit(
        dt,
        latitude,
        longitude,
        timezone,
        year,
        dirnorm=dirnorm,
        diffhor=diffhor,
        dirhor=dirhor,
        dirnorm_illum=dirnorm_illum,
        diffhor_illum=diffhor_illum,
        solar=solar,
        grefl=grefl,
    )
    if rotate:
        sun = pr.xform(sun, rotatez=rotate)
    out = [pr.Primitive("skyfunc", "glow", "sglow", [], [1, 1, 1, 0]).bytes]
    out.append(pr.Primitive("sglow", "source", "sky", [], [0, 0, 1, 180]).bytes)
    out.append(pr.Primitive("sglow", "source", "ground", [], [0, 0, -1, 180]).bytes)
    return sun + b" ".join(out)


def gen_wea(
    datetimes: Sequence[datetime.datetime],
    dirnorm: Sequence[float],
    diffhor: Sequence[float],
    latitude: float,
    longitude: float,
    timezone: int,
    elevation: None | float = None,
    location: None | str = None,
) -> str:
    """Generate wea file from datetime, location, and sky."""
    if len(datetimes) != len(dirnorm) != len(diffhor):
        raise ValueError("datetimes, dirnorm, and diffhor must be the same length")
    rows = []
    if location is None or location == "":
        location = "_".join(
            [str(i) for i in [latitude, longitude, timezone, elevation]]
        )
    if elevation is None:
        elevation = 0
    rows.append(f"place {location}")
    rows.append(f"latitude {latitude}")
    rows.append(f"longitude {longitude}")
    rows.append(f"time_zone {timezone}")
    rows.append(f"site_elevation {elevation}")
    rows.append("weather_data_file_units 1")
    for dt, dni, dhi in zip(datetimes, dirnorm, diffhor):
        _hrs = dt.hour + dt.minute / 60  # middle of hour
        _row = f"{dt.month} {dt.day} {_hrs} {dni} {dhi}"
        rows.append(_row)
    return "\n".join(rows)


def solar_angle(
    lat: float,
    lon: float,
    mer: float,
    month,
    day: int,
    hour,
) -> tuple[float, float]:
    """Calculate solar altitude and azimuth angles for a given location and time.

    This is a simplified translation from the Radiance sun.c and gensky.c code.
    Computes the solar position using astronomical formulas.

    Args:
        lat: Latitude in degrees (positive for North, negative for South).
        lon: Longitude in degrees (positive for East, negative for West).
        mer: Standard meridian for the time zone in degrees.
        month: Month (1-12).
        day: Day of month (1-31).
        hour: Hour in decimal format (e.g., 13.5 for 1:30 PM).

    Returns:
        A tuple containing:
            - altitude: Solar altitude angle in radians (angle above horizon)
            - azimuth: Solar azimuth angle in radians (angle from North, clockwise)

    Examples:
        >>> # Calculate sun position for Oakland, CA on June 21st at noon
        >>> alt, az = solar_angle(37.8, -122.3, -120, 6, 21, 12.0)
        >>> print(f"Altitude: {math.degrees(alt):.1f}°")
        >>> print(f"Azimuth: {math.degrees(az):.1f}°")
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
    mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    jd = mo_da[data.time.month - 1] + data.time.day
    return 24 * 60 * (jd - 1) + int(data.time.hour * 60.0 + data.time.minute + 0.5)
