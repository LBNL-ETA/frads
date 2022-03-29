"""Call uvspec to generate spectrally resolved sky description."""
import argparse
from datetime import datetime, timedelta
import logging
import math
import os
import subprocess as sp
import tempfile
from frads import makesky, util, radutil
try:
    import numpy as np
    NUMPY_FOUND = True
except ModuleNotFoundError:
    NUMPY_FOUND = False


def spec2xy(ip, triy_mean, trix, triy, triz) -> tuple:
    """Convert spectral data to CIE xy chromiticity.
    This is the numpy version of the spec2xy in frads.util.
    args:
        ip (numpy array): spectral power data
        trix (numpy array): tristimulus x
        triy (numpy array): tristimulus y
        triz (numpy array): tristimulus z
    return:
        chromiticity x
        chromiticity y
    """
    r_tx = (ip * trix).mean()
    r_ty = (ip * triy).mean()
    r_tz = (ip * triz).mean()
    cie_X = r_tx / triy_mean
    cie_Y = r_ty / triy_mean
    cie_Z = r_tz / triy_mean
    if cie_X + cie_Y + cie_Z == 0:
        return 0, 0
    chrom_x = cie_X / (cie_X + cie_Y + cie_Z)
    chrom_y = cie_Y / (cie_X + cie_Y + cie_Z)
    return chrom_x, chrom_y


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


def load_cie_tristi(inp, observer_deg) -> tuple:
    """Load CIE tristimulus data into numpy arrays, which is then interpolated
    with input spectral power data.
    args:
        inp: input spectral power data
        observer_deg: observer view anlges in deg, used to select the
                    corresponding CIE tristimulus data.
    return:
        interpolated CIE_x, CIE_y, and CIE_z
    """
    cie_path = util.get_tristi_paths()
    cie_x = np.loadtxt(cie_path[f"x{observer_deg}"], skiprows=3)
    cie_y = np.loadtxt(cie_path[f"y{observer_deg}"], skiprows=3)
    cie_z = np.loadtxt(cie_path[f"z{observer_deg}"], skiprows=3)
    cie_x_i = np.interp(inp[:, 0], cie_x[:, 0], cie_x[:, 1],
                        left=np.nan, right=np.nan)
    cie_y_i = np.interp(inp[:, 0], cie_y[:, 0], cie_y[:, 1],
                        left=np.nan, right=np.nan)
    cie_z_i = np.interp(inp[:, 0], cie_z[:, 0], cie_z[:, 1],
                        left=np.nan, right=np.nan)
    cie_x_i = cie_x_i[~np.isnan(cie_x_i)]
    cie_y_i = cie_y_i[~np.isnan(cie_y_i)]
    cie_z_i = cie_z_i[~np.isnan(cie_z_i)]
    return cie_x_i, cie_y_i, cie_z_i


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
    SOLAR_SA = 6.7967e-5
    args = parse_cli_args()
    logger = get_logger(args.verbose)
    verbose = True if args.verbose < 3 else False
    theta_interval = 90 / args.anglestep
    phi_interval = 360 / args.anglestep + 1
    header = "# Theta and phi dimensions\n2\n0 90 "
    header += f"{theta_interval:.0f}\n0 360 {phi_interval:.0f}\n"
    sky_template = "void colordata skyfunc\n"
    sky_template += "9 noop noop noop red.dat green.dat blue.dat . "
    sky_template += '"Acos(Dz)/DEGREE" "mod(atan2(-Dx, -Dy)/DEGREE,360)"\n'
    sky_template += "0\n0\n\n"
    sky_template += "skyfunc glow skyglow 0 0 4 1 1 1 0\n"
    sky_template += "skyglow source skydome 0 0 4 0 0 1 180\n"
    lib_path = os.environ["LIBRADTRAN_DATA_FILES"]
    hours = args.hour + args.minute / 60.0
    wavelengths = range(360, 801, 10)
    dt = datetime(args.year, args.month, args.day, args.hour, args.minute)
    dt_str = (dt + timedelta(hours=int(args.tzone / (-15)))).strftime(
        "%Y %m %d %H %M %S")
    direct_sun = True
    ct, phis = get_uniform_samples(args.anglestep)
    source_prim, source_dir, zenith_angle, azimuth_angle = get_solar(
        args.year, args.month, args.day, hours,
        args.latitude, args.longitude, args.tzone)
    # Input to uvspec
    model = f"data_files_path {lib_path}\n"
    model += f"source solar {lib_path}solar_flux/apm_1nm\n"
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
        model += "mol_abs_param kato2\n"
        model += get_output_input(None, None, output="edir edn", verbose=verbose)
        model += "pseudospherical\n"
        model += "output_process sum\n"
        model += "quiet\n"
        logger.info(model)
        proc = sp.run("uvspec", input=model.encode(), stderr=sp.PIPE, stdout=sp.PIPE)
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
    with tempfile.TemporaryDirectory() as td:
        tout = os.path.join(td, "temp.txt")
        with open(tout, "wb") as wtr:
            for wvl in wavelengths:
                inp = model
                inp += f"wavelength {wvl}\n"
                inp += "pseudospherical\n"
                logger.info(inp)
                proc = sp.run("uvspec", input=inp.encode(), stdout=sp.PIPE)
                wtr.write(proc.stdout)
                result.append(proc.stdout.decode().strip().split())
        int_proc = sp.run(["integrate", "-p", tout], stdout=sp.PIPE)
        integral = int_proc.stdout.decode().strip().split()
    coeffs = util.get_conversion_matrix(args.colorspace)
    if NUMPY_FOUND:
        result_array = np.array(result, dtype=float)
        cie_x_i, cie_y_i, cie_z_i = load_cie_tristi(result_array, args.observer)
        angular_xy = np.apply_along_axis(
            spec2xy, 0, result_array[:, 1:], cie_y_i.mean(),
            cie_x_i, cie_y_i, cie_z_i).T
        int_array = np.array(integral, dtype=float) / 1e3  # mW -> W
        xx = int_array * angular_xy[:, 0] / angular_xy[:, 1]
        zz = int_array * (1 - angular_xy[:, 0] - angular_xy[:, 1]) / angular_xy[:, 1]
        red = xx * coeffs[0] + int_array * coeffs[1] + zz * coeffs[2]
        green = xx * coeffs[3] + int_array * coeffs[4] + zz * coeffs[5]
        blue = xx * coeffs[6] + int_array * coeffs[7] + zz * coeffs[8]
    else:
        integral = [f / 1e3 for f in map(float, integral)]  # mW -> W
        wavelengths = list(wavelengths)
        wvl_length = len(wavelengths)
        trix, triy, triz = util.load_cie_tristi(wavelengths, args.observer)
        avg_0 = sum(triy) / wvl_length
        columns = [col for col in zip(*result)]
        chrom_xs = []
        chrom_ys = []
        # Get chromticity for each sampled point from sky
        for col in columns[1:]:
            col = list(map(float, col))
            ciexs = [i * j for i, j in zip(col, trix)]
            cieys = [i * j for i, j in zip(col, triy)]
            ciezs = [i * j for i, j in zip(col, triz)]
            avg_1 = sum(ciexs) / wvl_length
            avg_2 = sum(cieys) / wvl_length
            avg_3 = sum(ciezs) / wvl_length
            cie_x = avg_1 / avg_0
            cie_y = avg_2 / avg_0
            cie_z = avg_3 / avg_0
            chrom_x, chrom_y = util.xyz2xy(cie_x, cie_y, cie_z)
            chrom_xs.append(chrom_x)
            chrom_ys.append(chrom_y)
        xx = [i * x / y for i, x, y in zip(integral, chrom_xs, chrom_ys)]
        zz = [i * (1 - x - y) / y
              for i, x, y in zip(integral, chrom_xs, chrom_ys)]
        red = []
        green = []
        blue = []
        for x, y, z in zip(xx, integral, zz):
            _red, _green, _blue = util.xyz2rgb(x, y, z, coeffs)
            red.append(_red)
            green.append(_green)
            blue.append(_blue)

    out_dir = f"cs_{args.month:02d}{args.day:02d}{args.hour:02d}"
    out_dir += f"{args.minute:02d}_{args.latitude}_{args.longitude}"
    util.mkdir_p(out_dir)
    if direct_sun:
        sidx = 2
    else:
        sidx = 0
    with open(os.path.join(out_dir, "red.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in red[sidx:]]))
    with open(os.path.join(out_dir, "green.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in green[sidx:]]))
    with open(os.path.join(out_dir, "blue.dat"), "w") as wtr:
        wtr.write(header)
        wtr.write("\n".join([str(value) for value in blue[sidx:]]))
    with open(os.path.join(out_dir, "sky.rad"), "w") as wtr:
        if direct_sun:
            wtr.write("void light solar\n0\n0\n3 ")
            wtr.write(f"{red[0]/SOLAR_SA/source_dir[2]} ")
            wtr.write(f"{green[0]/SOLAR_SA/source_dir[2]} ")
            wtr.write(f"{blue[0]/SOLAR_SA/source_dir[2]}\n")
            wtr.write(str(source_prim) + "\n")
        wtr.write(sky_template)


if __name__ == "__main__":
    main()
