"""Call uvspec to generate spectrally resolved sky description."""
import argparse
from datetime import datetime, timedelta
import logging
import math
import os
import subprocess as sp
import tempfile
from frads import makesky, util, radutil


def get_local_input(dt: str, latitude: float, longitude: float,
                    altitude: float, zenith: float,
                    azimuth: float) -> str:
    """Get geographical and solar geometry input to uvspec.
    args:
        dt: datetime string
        latitude: in deg
        longitude: in deg
        altitude: in km
        zenith: solar zenith angle (deg)
        azimuth: solar azimuth (deg) (0deg -> south)
        doy: day of year
    return
        uvspec input string
    """
    if longitude > 0:
        longitude_str = f"W {longitude:.2f}"
    else:
        longitude_str = f"E {abs(longitude):.2f}"
    if latitude > 0:
        latitude_str = f"N {latitude:.2f}"
    else:
        latitude_str = f"S {abs(latitude):.2f}"
    inp = f"latitude {latitude_str}\n"
    inp += f"longitude {longitude_str}\n"
    inp += f"time {dt}\n"
    inp += f"altitude {altitude}\n"
    inp += "albedo 0.2\n"
    inp += f"sza {zenith}\n"
    inp += f"phi0 {azimuth}\n"
    return inp


def get_output_input(umu: list, phis: list,
                     output=None, silent=True, verbose=False) -> str:
    """Get sampling angles and output format input for uvspec.
    args:
        umu:
        phis:
        output:
        silent:
    return:
        uvspec input string
    """
    if output is None:
        output = "lambda edir edn uu"
    inp = ""
    if umu is not None:
        inp += f"umu {' '.join(map(str, umu))}\n"
    if phis is not None:
        inp += f"phi {' '.join(map(str, phis))}\n"
    inp += f"output_user {output}\n"
    if silent:
        inp += "quiet\n"
    if verbose:
        inp += "verbose\n"
    return inp


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


def get_solar(year: str, month: str, day: str, hours: str,
              lat: str, lon: str, tzone: str):
    """Call gendaylit to get solar angles."""
    cmd = makesky.gendaylit_cmd(month, day, hours, lat, lon, tzone, year=year)
    gdl_proc = sp.run(list(map(str, cmd)), check=True,
                      stdout=sp.PIPE, stderr=sp.PIPE)
    gdl_prims = radutil.parse_primitive(gdl_proc.stdout.decode().splitlines())
    source_prim = [prim for prim in gdl_prims if prim.ptype == "source"][0]
    source_dir = list(map(float, source_prim.real_arg.split()[1:4]))
    zenith_angle = math.degrees(math.acos(source_dir[2]))
    azimuth_angle = math.degrees(math.atan2(-source_dir[0], -source_dir[1])) % 360
    return source_prim, source_dir, zenith_angle, azimuth_angle


def gen_rad_template() -> str:
    """Generate sky.rad file template.
    This function generates a static string for now.
    """
    sky_template = "void colordata skyfunc\n"
    sky_template += "9 noop noop noop red.dat green.dat blue.dat . "
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


def get_logger(verbosity: int):
    """Setup logger.
    args:
        verbosity: verbosity levels 0-5
    returns:
        logger object
    """
    logger = logging.getLogger("frads.gencolorsky")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    _level = verbosity * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def parse_cli_args():
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
    parser.add_argument("-s", "--observer", choices=["2", "10"], default="2")
    parser.add_argument("-c", "--colorspace", default="radiance")
    parser.add_argument("-b", "--cloudcover", type=float)
    parser.add_argument("-t", "--total", action="store_true")
    parser.add_argument("-g", "--cloudprofile", type=int)
    parser.add_argument("-r", "--anglestep", type=int, default=3)
    parser.add_argument("-u", "--altitude", default=0)
    parser.add_argument("-d", "--aod", type=float)
    parser.add_argument("-l", "--aerosol",
                        choices=["continental_clean",
                                 "continental_average",
                                 "continental_polluted",
                                 "urban",
                                 "maritime_clean",
                                 "maritime_polluted",
                                 "maritime_tropical",
                                 "desert",
                                 "antarctic"])
    parser.add_argument("-i", "--pmt", action="store_true")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Verbose mode: \n"
        "\t-v=Debug\n"
        "\t-vv=Info\n"
        "\t-vvv=Warning\n"
        "\t-vvvv=Error\n"
        "\t-vvvvv=Critical\n"
        "default=Warning")
    args = parser.parse_args()
    return args


def main():
    """gencolorsky entry point."""
    # Solar disk solid angle
    solar_sa = 6.7967e-5
    start_wvl = 360
    end_wvl = 800
    wvl_step = 10
    direct_sun = True
    try:
        lib_path = os.environ["LIBRADTRAN_DATA_FILES"]
    except KeyError as ke:
        raise KeyError("Can't find LIBRADTRAN_DATA_FILES in environment")
    args = parse_cli_args()
    logger = get_logger(args.verbose)
    verbose = True if args.verbose < 3 else False
    hours = args.hour + args.minute / 60.0
    wavelengths = range(start_wvl, end_wvl + 1, wvl_step)
    dt = datetime(args.year, args.month, args.day, args.hour, args.minute)
    dt_str = (dt + timedelta(hours=int(args.tzone / (-15)))).strftime(
        "%Y %m %d %H %M %S")
    ct, phis = get_uniform_samples(args.anglestep)
    source_prim, source_dir, zenith_angle, azimuth_angle = get_solar(
        args.year, args.month, args.day, hours,
        args.latitude, args.longitude, args.tzone)
    # Generate input to uvspec
    model = f"data_files_path {lib_path}\n"
    model += f"source solar {lib_path}solar_flux/apm_1nm\n"
    model += "pseudospherical\n"
    model += "aerosol_default\n"
    model += get_local_input(dt_str, args.latitude, args.longitude,
                             args.altitude, zenith_angle, azimuth_angle)
    if args.atm:
        model += f"atmosphere_file {atm_file}\n"
    if args.cloudcover:
        if args.cloudprofile:
            wc_path = args.cloudprofile
        else:
            _file_path_ = os.path.dirname(__file__)
            wc_path = os.path.join(_file_path_, 'data', "WC.DAT")
        model += f"wc_file 1D {wc_path}\n"
        model += f"cloudcover wc {args.cloudcover}\n"
        model += "interpret_as_level wc\n"  # use independent pixel approximation
        if args.cloudcover == 1:
            direct_sun = False
    if args.aerosol:
        model += f"aerosol_species_file {args.aerosol}\n"
    if args.aod:
        model += f"aerosol_modify tau set {args.aod}\n"
    if args.total:
        inp = model
        inp += "mol_abs_param kato2\n"
        inp += get_output_input(None, None, output="edir edn", verbose=verbose)
        inp += "output_process sum\n"
        logger.info(inp)
        proc = sp.run("uvspec", input=inp.encode(), stderr=sp.PIPE, stdout=sp.PIPE)
        direct_hor, diff_hor = list(map(float, proc.stdout.decode().strip().split()))
        direct_normal = direct_hor / source_dir[2]
        print(f"DNI: {direct_normal:.2f} W/m2")
        print(f"DHI: {diff_hor:.2f} W/m2")
        print(f"GHI: {direct_hor + diff_hor:.2f} W/m2")
        exit()
    elif not direct_sun:
        model += get_output_input(ct, phis, output="lambda uu")
    else:
        model += get_output_input(ct, phis)
    result = []
    for wvl in wavelengths:
        inp = model
        inp += f"wavelength {wvl}\n"
        logger.info(inp)
        proc = sp.run("uvspec", input=inp.encode(), stdout=sp.PIPE)
        result.append(proc.stdout.decode().strip().split())
    wvl_range = end_wvl - start_wvl + wvl_step
    wavelengths = list(wavelengths)
    wvl_length = len(wavelengths)
    trix, triy, triz, mlnp = util.load_cie_tristi(wavelengths, args.observer)
    columns = [col for col in zip(*result)]
    # Carry out additional full solar spectra run if pmt requested
    if args.pmt:
        inp = model
        inp += "mol_abs_param kato2\n"
        inp += get_output_input(ct, phis, output="lambda edir edn uu", verbose=verbose)
        inp += "output_process sum\n"
        logger.info(inp)
        proc = sp.run("uvspec", input=inp.encode(), stderr=sp.PIPE, stdout=sp.PIPE)
        blue = [i / 1e3 for i in map(float, proc.stdout.decode().strip().split())]
        pfact = util.LEMAX * wvl_range / wvl_length / 1e3
        mfact = util.MLEMAX * wvl_range / wvl_length / 1e3
        red = []
        green = []
        for col in columns[1:]:
            col = list(map(float, col))
            cieys = [i * j for i, j in zip(col, triy)]
            edis = [i * j for i,j in zip(col, mlnp)]
            cie_y = pfact * sum(cieys)
            edi = mfact * sum(edis)
            red.append(cie_y)
            green.append(edi)
    else:
        coeffs = util.get_conversion_matrix(args.colorspace)
        pfact = util.LEMAX * wvl_range / wvl_length / 1e3 / 179
        # Get RGB for each sampled point from sky
        red = []
        green = []
        blue = []
        for col in columns[1:]:
            col = list(map(float, col))
            ciexs = [i * j for i, j in zip(col, trix)]
            cieys = [i * j for i, j in zip(col, triy)]
            ciezs = [i * j for i, j in zip(col, triz)]
            cie_x = pfact * sum(ciexs)
            cie_y = pfact * sum(cieys)
            cie_z = pfact * sum(ciezs)
            _r, _g, _b = util.xyz2rgb(cie_x, cie_y, cie_z, coeffs)
            red.append(_r)
            green.append(_g)
            blue.append(_b)

    out_dir = f"cs_{args.month:02d}{args.day:02d}{args.hour:02d}"
    out_dir += f"{args.minute:02d}_{args.latitude}_{args.longitude}"
    util.mkdir_p(out_dir)
    if direct_sun:
        sidx = 2
    else:
        sidx = 0
    header = gen_header(args.anglestep)
    with open(os.path.join(out_dir, "red.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in red[sidx:]]))
    with open(os.path.join(out_dir, "green.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in green[sidx:]]))
    with open(os.path.join(out_dir, "blue.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in blue[sidx:]]))
    sky_template = gen_rad_template()
    with open(os.path.join(out_dir, "sky.rad"), "w") as wtr:
        if direct_sun:
            wtr.write("void light solar\n0\n0\n3 ")
            wtr.write(f"{red[0]/solar_sa/source_dir[2]} ")
            wtr.write(f"{green[0]/solar_sa/source_dir[2]} ")
            wtr.write(f"{blue[0]/solar_sa/source_dir[2]}\n")
            wtr.write(str(source_prim) + "\n")
        wtr.write(sky_template)
