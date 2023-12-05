"""
mrad cli
"""
import argparse
import configparser
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List
import sys

from ..utils import unpack_primitives, parse_polygon, gen_grid, write_hdrs
from ..methods import WorkflowConfig, TwoPhaseMethod, ThreePhaseMethod, FivePhaseMethod
import numpy as np
from pyradiance import model as rmodel
from pyradiance import param as rparam

logger = logging.getLogger(__name__)

def parse_vu(vu_str: str) -> Optional[rmodel.View]:
    """Parse view string into a View object.

    Args:
        vu_str: view parameters as a string

    Returns:
        A view object
    """

    if vu_str.strip() == "":
        return
    args_list = vu_str.strip().split()
    vparser = argparse.ArgumentParser()
    vparser = rparam.add_view_args(vparser)
    vparser.add_argument("-x", type=int)
    vparser.add_argument("-y", type=int)
    args, _ = vparser.parse_known_args(args_list)
    if args.vf is not None:
        args, _ = vparser.parse_known_args(
            args.vf.readline().strip().split(), namespace=args
        )
        args.vf.close()
    if None in (args.vp, args.vd):
        raise ValueError("Invalid view")
    view = rmodel.View(
        position=args.vp,
        direction=args.vd,
        vtype=args.vt[-1],
        horiz=args.vh,
        vert=args.vv,
        vfore=args.vo,
        vaft=args.va,
        hoff=args.vs,
        voff=args.vl,
    )
    if args.x is not None:
        view.xres = args.x
    if args.y is not None:
        view.yres = args.y
    return view


def parse_mrad_config(cfg_path: Path) -> Dict[str, dict]:
    """
    Parse mrad configuration file.
    Args:
        cfg_path: path to the configuration file
    Returns:
        A dictionary of configuration in a format
        that can be used by methods.WorkflowConfig
    """
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    config = configparser.ConfigParser(
        allow_no_value=False,
        inline_comment_prefixes="#",
        interpolation=configparser.ExtendedInterpolation(),
        converters={
            "path": lambda x: Path(x.strip()),
            "paths": lambda x: [Path(i) for i in x.split()],
            "spaths": lambda x: x.split(),
            # "options": parse_opt,
            "options": rparam.parse_rtrace_args,
            "view": parse_vu,
        },
    )
    config.read(Path(__file__).parent / "data" / "mrad_default.cfg")
    config.read(cfg_path)
    # Convert config to dict
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, val in config.items(section):
            config_dict[section][key] = val
    config_dict["settings"] = {**config_dict["SimControl"], **config_dict["Site"]}
    for k, v in config_dict["settings"].items():
        # Convert sampling parameters string to list
        if k.endswith("_matrix"):
            config_dict["settings"][k] = v.split()
    config_dict["settings"]["separate_direct"] = config["SimControl"].getboolean(
        "separate_direct"
    )
    config_dict["settings"]["overwrite"] = config["SimControl"].getboolean(
        "overwrite", False
    )
    config_dict["settings"]["save_matrices"] = config["SimControl"].getboolean(
        "save_matrices", True
    )
    config_dict["model"] = {
        "scene": {},
        "materials": {},
        "windows": {},
        "views": {},
        "sensors": {},
    }
    config_dict["model"]["scene"]["files"] = config["Model"].getspaths("scene")
    config_dict["model"]["materials"]["files"] = config["Model"].getspaths("material")
    config_dict["model"]["materials"]["matrices"] = {
        k.stem: {"matrix_file": k} for k in config["Model"].getpaths("window_xmls")
    }
    for wpath, xpath in zip(
        config["Model"].getpaths("windows"), config["Model"].getpaths("window_xmls")
    ):
        config_dict["model"]["windows"][wpath.stem] = {
            "file": str(wpath),
            "matrix_file": xpath.stem,
        }
    if (grid_files := config["RaySender"].getspaths("grid_points")) is not None:
        for gfile in grid_files:
            name = gfile.stem
            with open(gfile) as f:
                config_dict["model"]["sensors"][name] = {
                    "data": [[float(v) for v in l.split()] for l in f.readlines()]
                }
    elif (grid_paths := config["RaySender"].getpaths("grid_surface")) is not None:
        for gpath in grid_paths:
            name: str = gpath.stem
            # Take the first polygon primitive
            gprimitives = unpack_primitives(gpath)
            surface_polygon = None
            for prim in gprimitives:
                if prim.ptype == "polygon":
                    surface_polygon = parse_polygon(prim)
                    break
            if surface_polygon is None:
                raise ValueError(f"No polygon found in {gpath}")
            config_dict["model"]["sensors"][name] = {
                "data": gen_grid(
                    surface_polygon,
                    config["RaySender"].getfloat("grid_height"),
                    config["RaySender"].getfloat("grid_spacing"),
                )
            }
    views = [i for i in config["RaySender"] if i.startswith("view")]
    for vname in views:
        if (view := config["RaySender"].getview("view")) is not None:
            config_dict["model"]["views"][vname] = {
                "view": " ".join(view.args()),
                "xres": view.xres,
                "yres": view.yres,
            }
    del (
        config_dict["SimControl"],
        config_dict["Site"],
        config_dict["Model"],
        config_dict["RaySender"],
    )
    return config_dict


def mrad_init(args: argparse.Namespace) -> None:
    """Initiate mrad operation.

    Args:
        args: argparse.Namespace
    Returns:
        None
    """

    def get_file_list(paths: List[Path], ext):
        file_list: List[str] = []
        for path in paths:
            if not path.is_dir():
                file_list.append(str(path))
            else:
                file_list.extend(sorted(path.glob(f"*.{ext}")))
        return file_list

    config = configparser.ConfigParser(allow_no_value=False)
    simcontrol = {
        "# vmx_opt": "",
        "# dmx_opt": "",
        "# smx_basis": "",
        "# overwrite": "",
        "# separate_direct": "",
    }
    site = {}
    model = {}
    raysender = {}
    if args.wea_path is not None:
        if not args.wea_path.is_file():
            raise FileNotFoundError(args.wea_path)
        site["wea_path"] = args.wea_path
    elif args.epw_path is not None:
        if not args.epw_path.is_file():
            raise FileNotFoundError(args.epw_path)
        site["epw_path"] = args.epw_path
    model["name"] = args.name
    if args.grid is not None:
        raysender["grid_surface"] = args.grid[0]
        raysender["grid_spacing"] = args.grid[1]
        raysender["grid_height"] = args.grid[2]
    material_list = get_file_list(args.material, "mat")
    object_list: List[str] = get_file_list(args.object, "rad")
    model["scene"] = "\n".join(object_list)
    model["material"] = "\n".join(material_list)
    if args.window is not None:
        window_list = get_file_list(args.window, "rad")
        model["windows"] = "\n".join(window_list)
    if args.bsdf is not None:
        xml_list = get_file_list(args.bsdf, "xml")
        if len(window_list) != len(xml_list):
            raise ValueError("Number of window and xml files not the same")
        model["window_xmls"] = "\n".join(xml_list)
    templ_config = {
        "SimControl": simcontrol,
        "Site": site,
        "Model": model,
        "RaySender": raysender,
    }
    config.read_dict(templ_config)
    with open(f"{args.name}.cfg", "w", encoding="utf-8") as rdr:
        config.write(rdr)


def mrad_run(args: argparse.Namespace) -> None:
    """Call mtxmethod to carry out the actual simulation."""
    config_dict = parse_mrad_config(args.cfg)
    config_dict["settings"]["name"] = config_dict["settings"].get("name", args.cfg.stem)
    wconfig = WorkflowConfig.from_dict(config_dict)
    workflow = None
    if method := wconfig.settings.method:
        if method.startswith(("2", "two")):
            logger.info("Using two-phase simulation")
            workflow = TwoPhaseMethod(wconfig)
        elif method.startswith(("3", "three")):
            logger.info("Using three-phase simulation")
            workflow = ThreePhaseMethod(wconfig)
        elif method.startswith(("5", "five")):
            logger.info("Using five-phase simulation")
            workflow = FivePhaseMethod(wconfig)
    else:
        # Use 3- or 5-phase methods if we have
        # window groups and bsdf defined
        if len(wconfig.model.windows) > 1:
            if wconfig.settings.separate_direct:
                logger.info("Using five-phase simulation")
                workflow = FivePhaseMethod(wconfig)
            else:
                logger.info("Using three-phase simulation")
                workflow = ThreePhaseMethod(wconfig)
        else:
            logger.info("Using two-phase method")
            workflow = TwoPhaseMethod(wconfig)
    if workflow is None:
        raise ValueError("No simulation method found")
    workflow.generate_matrices()
    for vname, view in wconfig.model.views.items():
        res = workflow.calculate_view_from_wea(vname)
        write_hdrs(res, view.xres, view.yres, outdir=os.path.join("Results", vname))
    for sensor in wconfig.model.sensors:
        res = workflow.calculate_sensor_from_wea(sensor)
        np.savetxt(os.path.join("Results", f"{sensor}.txt"), res)


def mrad() -> None:
    """mrad entry point: parse arugments for init and run subprograms."""
    parser = argparse.ArgumentParser(
        prog="mrad",
        description="Mrad is an executive program for Radiance matrix-based simulation methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparser = parser.add_subparsers()
    # Parse arguments for init subprogram
    parser_init = subparser.add_parser("init")
    parser_init.set_defaults(func=mrad_init)
    egroup = parser_init.add_mutually_exclusive_group(required=True)
    egroup.add_argument("-a", "--wea_path", metavar="wea_path", type=Path)
    egroup.add_argument("-e", "--epw_path", metavar="epw_path", type=Path)
    parser_init.add_argument("-n", "--name", metavar="model_name", default="default")
    parser_init.add_argument(
        "-o",
        "--object",
        required=True,
        metavar="object",
        nargs="+",
        type=Path,
        help="Objects to include, can include wildcards",
    )
    parser_init.add_argument(
        "-m",
        "--material",
        required=True,
        metavar="material",
        nargs="+",
        type=Path,
        help="Material files to include, can include wildcards",
    )
    parser_init.add_argument(
        "-w",
        "--window",
        nargs="+",
        metavar="window",
        type=Path,
        help="Window files to include, these are gonna get parsed into window groups",
    )
    parser_init.add_argument(
        "-x",
        "--bsdf",
        nargs="+",
        metavar="bsdf",
        type=Path,
        help="xml file paths to include for each window file",
    )
    parser_init.add_argument(
        "-g",
        "--grid",
        nargs=3,
        metavar=("surface path", "grid spacing", "grid height"),
        help="Grid file path, grid spacing and height",
    )
    parser_init.add_argument(
        "-u", "--view", metavar="view", help="Grid file path, grid spacing and height"
    )
    # Parse arguments for run subprogram
    parser_run = subparser.add_parser("run")
    parser_run.set_defaults(func=mrad_run)
    parser_run.add_argument("cfg", type=Path, help="configuration file path")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Verbose mode: -v=Debug, -vv=Info, -vvv=Warning, "
            "-vvvv=Error, -vvvvv=Critical"
        ),
    )
    args = parser.parse_args()
    # Setup logger
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s", "%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if args.verbose > 1:
        sys.tracebacklimit = 0
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args.func(args)
