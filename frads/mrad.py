"""
Executive command-line program for Radiance matrix-based simulation.
"""

from configparser import ConfigParser
import argparse
import glob
import logging
import os
from pathlib import Path
from frads import mtxmethod
from frads import util


logger = logging.getLogger("frads")



def initialize(args: argparse.Namespace) -> None:
    """Initiate mrad operation.
    Going through files in the standard file
    structure and generate a default.cfg file.
    Args:
        args: argparse.Namespace
    """
    if args.wea_path != Path(""):
        if not args.wea_path.is_file():
            raise FileNotFoundError(args.wea_path)
    elif args.epw_path != Path(""):
        if not args.epw_path.is_file():
            raise FileNotFoundError(args.epw_path)
    else:
        raise ValueError("Site not defined")

    config = ConfigParser(allow_no_value=False)
    config["SimControl"] = {
        "vmx_basis": "kf",
        "vmx_opt": "-ab 5 -ad 65536 -lw 1e-5",
        "fmx_basis": "kf",
        "smx_basis": "r4",
        "dmx_opt": "-ab 2 -ad 128 -c 5000",
        "dsmx_opt": "-ab 7 -ad 16384 -lw 5e-5",
        "cdsmx_opt": "-ab 1",
        "cdsmx_basis": "r6",
        "ray_count": 1,
        "nprocess": 1,
        "separate_direct": False,
        "overwrite": False,
        "method": "",
    }
    site = {
        "wea_path": args.wea_path, "epw_path": args.epw_path,
        "start_hour": "", "end_hour": "",
        "daylight_hours_only": ""
    }
    model = {
        "name": args.name, "material": "", "scene": "", "window_paths": "",
        "window_xml": "", "window_cfs": "", "window_control": "",
    }
    raysender = {
        "grid_surface": args.grid[0], "grid_spacing": args.grid[1],
        "grid_height": args.grid[2], "view": ""
    }
    material_list = []
    window_list = []
    object_list = []
    if args.object is not None:
        for obj in args.object:
            if obj.is_dir():
                object_list.extend(glob.glob(str(obj/"*")))
            else:
                object_list.extend(glob.glob(str(obj)))
    else:
        logger.warning("Object files not set")
    if args.material is not None:
        for mat in args.material:
            if mat.is_dir():
                material_list.extend(glob.glob(str(mat/"*")))
            else:
                material_list.extend(glob.glob(str(mat)))
    else:
        logger.warning("Material files not set")
    if args.window is not None:
        for win in args.window:
            if win.is_dir():
                window_list.extend(glob.glob(str(win/"*")))
            else:
                window_list.extend(glob.glob(str(win)))
    else:
        logger.warning("Window files not set")
    if args.xmls is not None:
        xml_list = []
        for xml in args.xmls:
            if xml.is_dir():
                xml_list.extend(glob.glob(str(xml/"*")))
            else:
                xml_list.extend(glob.glob(str(xml)))
        if (len(window_list) != len(xml_list)):
            raise ValueError("Number of window and xml files not the same")
        model["window_xmls"] = "\n".join(xml_list)
    model["scene"] = "\n".join(object_list)
    model["material"] = "\n".join(material_list)
    model["window_paths"] = "\n".join(window_list)
    templ_config = {"Site": site, "Model": model, "RaySender": raysender}
    config.read_dict(templ_config)
    with open(f"{args.name}.cfg", "w") as rdr:
        config.write(rdr)


def run(args: argparse.Namespace) -> None:
    """Call mtxmethod to carry out the actual simulation."""
    cfg = ConfigParser(allow_no_value=False, inline_comment_prefixes="#")
    with open(args.cfg) as rdr:
        cfg.read_string(rdr.read())
    try:
        name = cfg["Model"]["name"]
    except KeyError:
        name = args.cfg.stem
        cfg["Model"]["name"] = name
    util.mkdir_p("Matrices")
    util.mkdir_p("Results")
    model = mtxmethod.assemble_model(cfg)
    method = cfg["SimControl"]["method"]
    if method != "":
        _direct = False
        if method.startswith(("2", "two")):
            _method = mtxmethod.two_phase
        elif method.startswith(("3", "three")):
            _method = mtxmethod.three_phase
        elif method.startswith(("5", "five")):
            _method = mtxmethod.three_phase
        _method(model,
                cfg,
                direct=cfg["SimControl"]["separate_direct"])
    else:
        if "" in (cfg["Model"]["window_xml"], cfg["Model"]["window_paths"]):
            logger.info("Using two-phase method")
            mtxmethod.two_phase(model, cfg)
        else:
            if len(cfg["Model"]["ncp_shade"].split()) > 1:
                if cfg.getboolean("SimControl", "separate_direct"):
                    logger.info("Using six-phase simulation")
                    mtxmethod.four_phase(model, cfg, direct=True)
                else:
                    logger.info("Using four-phase simulation")
                    mtxmethod.four_phase(model, cfg)
            else:
                if cfg.getboolean("SimControl", "separate_direct"):
                    logger.info("Using five-phase simulation")
                    mtxmethod.three_phase(model, cfg, direct=True)
                else:
                    logger.info("Using three-phase simulation")
                    mtxmethod.three_phase(model, cfg)


def main() -> None:
    """mrad entry point: parse arugments for init and run subprograms."""
    prog_description = "Executive program for Radiance matrix-based simulation"
    parser = argparse.ArgumentParser(
        prog="mrad", description=prog_description,
        formatter_class=argparse.RawTextHelpFormatter)
    subparser = parser.add_subparsers()
    # Parse arguments for init subprogram
    parser_init = subparser.add_parser("init")
    parser_init.set_defaults(func=initialize)
    parser_init.add_argument("-W", "--wea_path", type=Path, default=Path(""))
    parser_init.add_argument("-E", "--epw_path", type=Path, default=Path(""))
    parser_init.add_argument("-n", "--name", default="default")
    # parser_init.add_argument("-Z", "--zipcode", default="")
    # parser_init.add_argument("-L", "--latlon", nargs=2, default=("", ""))
    parser_init.add_argument("-o", "--object", nargs="+", type=Path)
    parser_init.add_argument("-m", "--material", nargs="+", type=Path)
    parser_init.add_argument("-w", "--window", nargs="+", type=Path)
    parser_init.add_argument("-x", "--xmls", nargs="+", type=Path)
    parser_init.add_argument("-g", "--grid", nargs=3, default=("", 0, 0))
    # Parse arguments for run subprogram
    parser_run = subparser.add_parser("run")
    parser_run.add_argument("cfg", type=Path, help="configuration file path")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Verbose mode: \n"
        "\t-v=Debug\n"
        "\t-vv=Info\n"
        "\t-vvv=Warning\n"
        "\t-vvvv=Error\n"
        "\t-vvvvv=Critical\n"
        "default=Warning")
    parser_run.set_defaults(func=run)
    args = parser.parse_args()
    # Setup logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Call subprograms to do work
    args.func(args)
