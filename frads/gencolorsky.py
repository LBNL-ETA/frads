"""
Call uvspec to generate spectrally resolved sky description.
This module contains only functionalities relating to calling uvspec.
Uvspec needs to be in the local environment.
This module is not merged into sky.rad because of this unique circumstance.
"""

import argparse
from datetime import datetime, timedelta
import logging
import math
import os
from pathlib import Path
import subprocess as sp
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from frads import sky
from frads import parsers
from frads import color
from frads import color_data
from frads.types import Primitive
from frads.types import WeaMetaData


# Wavelength range and step
START_WVL = 360
END_WVL = 800
WVL_STEP = 10

# Approximate spectrum from 3 samples at 440, 550, and 660nm
# These coefficients are generated using apm_1nm.dat solar spectrum
# with alpha = 0, spectral range 360 - 800 nm
KR0, KG0, KB0 = 90487.185526, 68805.727959, 70873.299179

# These coefficiens are derived from CIE Colorimetry S0, S1, and S2 (Table T.2)
# integrated with cie_y_bar over 360 - 800 nm
YS0, YS1, YS2 = 10700.685138588558, 193.0296802325152, 76.46615567113169


def samples2spec(l_r, l_g, l_b) -> dict:
    """Approximate full spectrum from three samples."""
    c_r, c_g, c_b = l_r * KR0, l_g * KG0, l_b * KB0
    cie_x, cie_y, cie_z = color.rgb2xyz(c_r, c_g, c_b, color_data.RGB2XYZ_SRGB)
    chrom_x, chrom_y = color.xyz2xy(cie_x, cie_y, cie_z)
    scale_y = cie_y / color_data.LEMAX
    m1 = (-1.3515 - 1.7703 * chrom_x + 5.9114 * chrom_y) / (
        0.0241 + 0.2562 * chrom_x - 0.7341 * chrom_y
    )
    m2 = (0.0300 - 31.4424 * chrom_x + 30.0717 * chrom_y) / (
        0.0241 + 0.2562 * chrom_x - 0.7341 * chrom_y
    )
    spec = {
        wvl: scale_y * (_s[0] + _s[1] * m1 + _s[2] * m2) / (YS0 + YS1 * m1 + YS2 * m2)
        for wvl, _s in color_data.CIE_S012.items()
        if START_WVL <= wvl <= END_WVL
    }
    return spec


def get_uniform_samples(step: int) -> tuple:
    """Get uniform sky sampling angles.
    args:
        step: uniform sampling spacing (deg), need to be divisive by 90.
    return:
        cos(theta)
        phi angles
    """
    if 90 % step != 0:
        raise Exception("Angluar resolution not divisive by 90")
    thetas = range(0, 90, step)
    cos_thetas = [-math.cos(math.radians(i)) for i in thetas]
    phis = list(range(0, 361, step))
    return cos_thetas, phis


def get_solar(dt, meta) -> Tuple[Primitive, Sequence[Union[float, int]], float, float]:
    # year: str, month: str, day: str, hours: str, lat: str, lon: str, tzone: str
    # )
    """Call gendaylit to get solar angles."""
    hours = dt.hour + dt.minute / 60.0
    cmd = sky.gendaylit_cmd(
        dt.month,
        dt.day,
        hours,
        meta.latitude,
        meta.longitude,
        meta.timezone,
        year=dt.year,
    )
    gdl_proc = sp.run(list(map(str, cmd)), check=True, stdout=sp.PIPE, stderr=sp.PIPE)
    gdl_prims = parsers.parse_primitive(gdl_proc.stdout.decode().splitlines())
    source_prim = [prim for prim in gdl_prims if prim.ptype == "source"][0]
    source_dir = source_prim.real_arg[1:4]
    zenith_angle = math.degrees(math.acos(source_dir[2]))
    azimuth_angle = math.degrees(math.atan2(-source_dir[0], -source_dir[1])) % 360
    return source_prim, source_dir, zenith_angle, azimuth_angle


def gen_rad_template() -> str:
    """Generate sky.rad file template."""
    sky_template = "void colordata skyfunc\n"
    sky_template += (
        "9 noop noop noop {path}/red.dat {path}/green.dat {path}/blue.dat . "
    )
    sky_template += '"Acos(Dz)/DEGREE" "mod(atan2(-Dx, -Dy)/DEGREE,360)"\n'
    sky_template += "0\n0\n\n"
    sky_template += "skyfunc glow skyglow 0 0 4 1 1 1 0\n"
    sky_template += "skyglow source skydome 0 0 4 0 0 1 180\n"
    return sky_template


def gen_header(anglestep: int) -> str:
    """Generate header for colordata data files."""
    theta_interval = 90 / anglestep
    phi_interval = 360 / anglestep + 1
    header = "# Theta and phi dimensions\n2\n0 90 "
    header += f"{theta_interval:.0f}\n0 360 {phi_interval:.0f}\n"
    return header


def get_logger(verbosity: int) -> logging.Logger:
    """Setup logger.
    args:
        verbosity: verbosity levels 0-5
    returns:
        logger object
    """
    logger = logging.getLogger("frads.gencolorsky")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    _level = verbosity * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_cli_args() -> argparse.Namespace:
    """Parse commandline interface arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("day", type=int)
    parser.add_argument("hour", type=int)
    parser.add_argument("minute", type=int)
    parser.add_argument("-a", "--latitude", type=float, required=True)
    parser.add_argument("-o", "--longitude", type=float, required=True)
    parser.add_argument("-m", "--tzone", type=int, required=True)
    parser.add_argument("-e", "--atm")
    parser.add_argument(
        "-f", "--rgb", action="store_true", help="RGB sampling to full spectrum approx."
    )
    parser.add_argument("-s", "--observer", choices=["2", "10"], default="2")
    parser.add_argument("-c", "--colorspace", default="radiance")
    parser.add_argument("-b", "--cloudcover", type=float)
    parser.add_argument("-t", "--total", action="store_true")
    parser.add_argument("-g", "--cloudprofile", type=int)
    parser.add_argument("-r", "--anglestep", type=int, default=3)
    parser.add_argument("-u", "--altitude", default=0)
    parser.add_argument("-d", "--aod", type=float)
    parser.add_argument(
        "-l",
        "--aerosol",
        choices=[
            "continental_clean",
            "continental_average",
            "continental_polluted",
            "urban",
            "maritime_clean",
            "maritime_polluted",
            "maritime_tropical",
            "desert",
            "antarctic",
        ],
    )
    parser.add_argument("-i", "--pmt", action="store_true")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose mode: \n"
        "\t-v=Debug\n"
        "\t-vv=Info\n"
        "\t-vvv=Warning\n"
        "\t-vvvv=Error\n"
        "\t-vvvvv=Critical\n"
        "default=Warning",
    )
    args = parser.parse_args()
    return args


class UvspecConfig:
    """Uvspec configuration.
    Attributes:
        aerosol: aerosol species file name (the file should reside in lib_path)
        albedo: surface reflectance
        altitude: altitude
        atmosphere: atmosphere composition profile
        azimuth: solar azimuth
        cloudcover: cloucover (0-1)
        latitude: latitude
        longitude: longitude
        output: user output setting (default: lambda ed edn uu)
        phis: Phi angles to sample
        solver: solver
        source: solar source file path
        time: date-time str (%Y %m %d %H %M %S)
        umu: cos(theta) angles to sample
        verbose: verbosity (verbose | quiet)
        wavelength: wavelength to sample
        zenith: solar zenith angle
    """

    def __init__(self):
        try:
            self.lib_path = Path(os.environ["LIBRADTRAN_DATA_FILES"])
        except KeyError as ke:
            raise "Can't find LIBRADTRAN_DATA_FILES in environment" from ke
        self._aerosol = ""
        self._albedo = 0.2
        self._altitude = 100
        self._aod = 0
        self._atmosphere = ""
        self._azimuth = 90
        self._band = ""
        self._cloudcover = 0
        self._latitude = "N 37"
        self._longitude = "W 122"
        self._output = "lambda edir edn uu"
        self._phis = ""
        self._post_process = ""
        self._solver = "pseudospherical"
        self._source = "apm_1nm"
        self._time = "2022 07 06 12 00 00"
        self._umu = ""
        self._verbose = "quiet"
        self._wavelength = 0
        self._zenith = 30
        self.wc_path = Path(__file__).parent / "data" / "WC.DAT"

    @property
    def config(self):
        """Get the uvpsec input string."""
        cfg = [f"data_files_path {str(self.lib_path)}"]
        cfg.append(f"source solar {str(self._source)}")
        if self._atmosphere:
            cfg.append(f"atmosphere_file {self._atmosphere}")
        cfg.append(self._solver)
        if self._band:
            cfg.append(f"mol_abs_param {self._band}")
        cfg.append(f"latitude {self._latitude}")
        cfg.append(f"longitude {self._longitude}")
        cfg.append(f"time {self._time}")
        cfg.append(f"altitude {self._altitude}")
        cfg.append(f"albedo {self._albedo}")
        cfg.append(f"sza {self._zenith}")
        cfg.append(f"phi0 {self._azimuth}")
        if self._cloudcover:
            cfg.append(f"wc_file 1D {self.wc_path}")
            cfg.append(f"cloudcover wc {self.cloudcover}")
            cfg.append("interpret_as_level wc")  # use independent pixel approximation
        if self._aerosol:
            cfg.append("aerosol_default")
            cfg.append(f"aerosol_species_file {self._aerosol}")
        if self._aod:
            cfg.append(f"aerosol_modify tau set {self._aod}")
        if self._umu:
            cfg.append(f"umu {self._umu}")
        if self._phis:
            cfg.append(f"phi {self._phis}")
        cfg.append(f"output_user {self._output}")
        if self._post_process:
            cfg.append(f"output_process {self._post_process}")
        cfg.append(self._verbose)
        if self._wavelength:
            cfg.append(f"wavelength {self._wavelength}")
        return "\n".join(cfg)

    @property
    def aerosol(self):
        """Get aerosol."""
        return self._azimuth

    @aerosol.setter
    def aerosol(self, path):
        self._aerosol = path

    @property
    def albedo(self):
        """Get ground albedo."""
        return self._albedo

    @albedo.setter
    def albedo(self, refl):
        self._albedo = refl

    @property
    def altitude(self):
        """Get site altitude."""
        return self._altitude

    @altitude.setter
    def altitude(self, alt):
        self._altitude = alt

    @property
    def aod(self):
        """Get aerosol optical depth."""
        return self._aod

    @aod.setter
    def aod(self, aod_):
        self._aod = aod_

    @property
    def atmosphere(self):
        """Get atmosphere file path."""
        return f"atmosphere_file {self._atmosphere}"

    @atmosphere.setter
    def atmosphere(self, atmosphere_file):
        self._atmosphere = atmosphere_file

    @property
    def azimuth(self):
        """Get solar azimuth angle."""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        self._azimuth = azimuth

    @property
    def band(self):
        """Get solar source band."""
        return self._band

    @band.setter
    def band(self, params):
        self._band = params

    @property
    def cloudcover(self):
        """Get cloudcover value."""
        return self._cloudcover

    @cloudcover.setter
    def cloudcover(self, cc):
        self._cloudcover = cc

    @property
    def latitude(self):
        """Get site latitude."""
        return self._latitude

    @latitude.setter
    def latitude(self, lat: float):
        if lat < 0:
            self._latitude = f"S {abs(lat):.2f}"
        else:
            self._latitude = f"N {lat:.2f}"

    @property
    def longitude(self):
        """Get site longitude."""
        return self._longitude

    @longitude.setter
    def longitude(self, lon: float):
        if lon < 0:
            self._longitude = f"E {abs(lon):.2f}"
        else:
            self._longitude = f"E {lon:.2f}"

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, ouput_user: str):
        self._output = ouput_user

    @property
    def post_process(self):
        return self._post_process

    @post_process.setter
    def post_process(self, process: str):
        self._post_process = process

    @property
    def phis(self):
        return self._phis

    @phis.setter
    def phis(self, phi: List[float]):
        self._phis = " ".join(map(str, phi))

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source: str):
        self._source = source

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, dt_: str):
        self._time = dt_

    @property
    def umu(self):
        return self._umu

    @umu.setter
    def umu(self, ct: List[float]):
        self._umu = " ".join(map(str, ct))

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v_q: str):
        if v_q.startswith("q"):
            self._verbose = "quiet"
        elif v_q.startswith("v"):
            self._verbose = "verbose"

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wvl: float):
        self._wavelength = wvl

    @property
    def zenith(self):
        return self._zenith

    @zenith.setter
    def zenith(self, zenith: float):
        self._zenith = zenith


def genskylrt_total(config):
    """."""
    config.band = "kato2"
    config.output = "edir edn"
    config.post_process = "sum"
    proc = sp.run(
        "uvspec",
        encoding="utf-8",
        input=config.config,
        stderr=sp.PIPE,
        stdout=sp.PIPE,
    )
    direct_hor, diff_hor = list(map(float, proc.stdout.strip().split()))
    return direct_hor, diff_hor


def genskylrt_solar(config):
    """."""
    config.band = "kato2"
    config.post_process = "sum"
    proc = sp.run(
        "uvspec",
        encoding="utf-8",
        input=config.config,
        stderr=sp.PIPE,
        stdout=sp.PIPE,
    )
    return [i / 1e3 for i in map(float, proc.stdout.strip().split())]


def genskylrt(config: UvspecConfig, wavelengths):
    """."""
    result = []
    for wvl in wavelengths:
        config.wavelength = wvl
        proc = sp.run(
            "uvspec",
            encoding="utf-8",
            input=config.config,
            stdout=sp.PIPE,
        )
        result.append(proc.stdout.strip().split())
    return list(zip(*result))


def main() -> None:
    """gencolorsky entry point."""
    args = parse_cli_args()
    logger = get_logger(args.verbose)
    verbose = args.verbose == 1
    if args.rgb:
        wavelengths = [440, 550, 680]
    else:
        wavelengths = list(range(START_WVL, END_WVL + 1, WVL_STEP))
    dt = datetime(args.year, args.month, args.day, args.hour, args.minute)
    meta = WeaMetaData(
        "country", "city", args.latitude, args.longitude, args.tzone, args.altitude
    )
    ct, phis = get_uniform_samples(args.anglestep)
    direct_sun = True
    source_prim, source_dir, zenith_angle, azimuth_angle = get_solar(dt, meta)
    uvspec = UvspecConfig()
    uvspec.source = uvspec.lib_path / "solar_flux" / "apm_1nm"
    uvspec.solver = "pseudospherical"
    uvspec.time = (dt + timedelta(hours=int(meta.timezone / (-15)))).strftime(
        "%Y %m %d %H %M %S"
    )
    uvspec.latitude = meta.latitude
    uvspec.longitude = meta.longitude
    uvspec.altitude = meta.elevation
    uvspec.zenith = zenith_angle
    uvspec.azimuth = azimuth_angle
    uvspec.verbose = "verbose" if verbose else "quiet"
    if args.atm:
        uvspec.atmosphere = args.atm
    if args.cloudcover:
        if args.cloudprofile:
            uvspec.wc_path = args.cloudprofile
        uvspec.cloudcover = args.cloudcover
        direct_sun = args.cloudcover != 1.0
    if args.aerosol:
        uvspec.aerosol = args.aerosol
    uvspec.aod = args.aod
    if args.total:
        logger.info(uvspec.config)
        direct_hor, diff_hor = genskylrt_total(uvspec)
        direct_normal = direct_hor / source_dir[2]
        print(f"DNI: {direct_normal:.2f} W/m2")
        print(f"DHI: {diff_hor:.2f} W/m2")
        print(f"GHI: {direct_hor + diff_hor:.2f} W/m2")
        exit()
    uvspec.umu = ct
    uvspec.phis = phis
    uvspec.output = "lambda edir edn uu"
    if not direct_sun:
        uvspec.output = "lambda uu"
    logger.info(uvspec.config)
    columns = genskylrt(uvspec, wavelengths)
    if args.rgb:
        wavelengths = range(360, 801, 5)
        new_columns = [list(range(360, 801, 5))]
        for col in columns[1:]:
            new_columns.append(list(samples2spec(*map(float, col[::-1])).values()))
        columns = new_columns
    wvl_range = END_WVL - START_WVL + WVL_STEP
    cie_xyz_bar = color.get_interpolated_cie_xyz(wavelengths, args.observer)
    if args.pmt:
        blue = genskylrt_solar(uvspec)
        pfact = color_data.LEMAX * wvl_range / wvl_length / 1e3
        mfact = color_data.MLEMAX * wvl_range / wvl_length / 1e3
        red = []
        green = []
        mlnp = color.get_interpolated_mlnp(wavelengths)
        for col in columns[1:]:
            col = list(map(float, col))
            cieys = [i * j[1] for i, j in zip(col, cie_xyz_bar)]
            edis = [i * j for i, j in zip(col, mlnp)]
            cie_y = pfact * sum(cieys)
            edi = mfact * sum(edis)
            red.append(cie_y)
            green.append(edi)
    else:
        coeffs = color_data.XYZ2RGB_RAD
        pfact = color_data.LEMAX / 1e3 / 179
        red = []
        green = []
        blue = []
        for col in columns[1:]:
            col = list(map(float, col))
            cie_x, cie_y, cie_z = color.spec2xyz(cie_xyz_bar, col, wvl_range, emis=True)
            cie_x *= pfact
            cie_y *= pfact
            cie_z *= pfact
            _r, _g, _b = color.xyz2rgb(cie_x, cie_y, cie_z, coeffs)
            red.append(_r)
            green.append(_g)
            blue.append(_b)
    out_dir = Path(
        f"lrt_{args.month:02d}{args.day:02d}_{args.hour:02d}"
        f"{args.minute:02d}_{args.latitude}_{args.longitude}"
    )
    out_dir.mkdir(exist_ok=True)
    if direct_sun:
        sidx = 2
    else:
        sidx = 0
    header = gen_header(args.anglestep)
    with open(out_dir / "red.dat", "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in red[sidx:]]))
    with open(out_dir / "green.dat", "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in green[sidx:]]))
    with open(out_dir / "blue.dat", "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in blue[sidx:]]))
    sky_template = gen_rad_template()
    with open(out_dir / "sky.rad", "w") as wtr:
        if direct_sun:
            wtr.write("void light solar\n0\n0\n3 ")
            wtr.write(f"{red[0]/sky.SOLAR_SA/source_dir[2]} ")
            wtr.write(f"{green[0]/sky.SOLAR_SA/source_dir[2]} ")
            wtr.write(f"{blue[0]/sky.SOLAR_SA/source_dir[2]}\n")
            wtr.write(str(source_prim) + "\n")
        wtr.write(sky_template.format(path=out_dir))
