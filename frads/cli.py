"""
Executive command-line program for Radiance matrix-based simulation.
"""

import argparse
import configparser
import logging
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pyradiance as pr
from pyradiance import SamplingParameters, lib
from pyradiance import param as rparam
from pyradiance import model as rmodel
import pywincalc as pwc

from frads import ncp
from frads.eplus import load_energyplus_model
from frads.ep2rad import epmodel_to_radmodel
from frads.room import create_south_facing_room
from frads.window import PaneRGB, get_glazing_primitive
from frads.matrix import (
    load_matrix,
    Matrix,
    SensorSender,
    SkyReceiver,
    SurfaceSender,
    SurfaceReceiver,
    SunMatrix,
    SunReceiver,
    ViewSender,
)
from frads.methods import (
    TwoPhaseMethod,
    ThreePhaseMethod,
    FivePhaseMethod,
    WorkflowConfig,
)
from frads.utils import (
    gen_grid,
    parse_polygon,
    primitive_normal,
    unpack_primitives,
    write_hdrs,
    write_ep_rad_model,
)


logger: logging.Logger = logging.getLogger("frads")


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
    if config_dict["settings"]["name"] == "":
        config_dict["settings"]["name"] = args.cfg.stem
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


def glaze(args) -> None:
    """Command-line program for generating BRTDfunc for glazing system."""
    if args.optics is not None:
        panes = [pwc.parse_optics_file(fpath) for fpath in args.optics]
    else:
        panes = []
        for item in args.igsdb:
            with open(item, "r", encoding="utf-8") as f:
                panes.append(pwc.parse_json(f.read()))
    pane_rgb = []
    photopic_wvl = range(380, 781, 10)
    for pane in panes:
        hemi = {
            d.wavelength
            * 1e3: (
                d.direct_component.transmittance_front,
                d.direct_component.transmittance_back,
                d.direct_component.reflectance_front,
                d.direct_component.reflectance_back,
            )
            for d in pane.measurements
        }
        tvf = [hemi[w][0] for w in photopic_wvl]
        rvf = [hemi[w][2] for w in photopic_wvl]
        rvb = [hemi[w][3] for w in photopic_wvl]
        tf_x, tf_y, tf_z = lib.spec_xyz(tvf, 380, 780)
        rf_x, rf_y, rf_z = lib.spec_xyz(rvf, 380, 780)
        rb_x, rb_y, rb_z = lib.spec_xyz(rvb, 380, 780)
        tf_rgb = lib.xyz_rgb(tf_x, tf_y, tf_z)
        rf_rgb = lib.xyz_rgb(rf_x, rf_y, rf_z)
        rb_rgb = lib.xyz_rgb(rb_x, rb_y, rb_z)
        if pane.coated_side == "front":
            coated_rgb = rf_rgb
            glass_rgb = rb_rgb
        else:
            coated_rgb = rb_rgb
            glass_rgb = rf_rgb
        pane_rgb.append(PaneRGB(pane, coated_rgb, glass_rgb, tf_rgb))
    print(get_glazing_primitive(pane_rgb))


def gengrid(args) -> None:
    """Commandline program for generating a grid of sensor points."""
    prims = unpack_primitives(args.surface)
    polygon_prims = [prim for prim in prims if prim.ptype == "polygon"]
    polygon = parse_polygon(polygon_prims[0]).flip()
    if args.op:
        polygon = polygon.flip()
    grid_list = gen_grid(polygon, args.height, args.spacing)
    cleanedup = []
    for row in grid_list:
        new_row = []
        for val in row:
            if val.is_integer():
                new_row.append(int(val))
            else:
                if (rounded := round(val, 1)).is_integer():
                    new_row.append(int(rounded))
                else:
                    new_row.append(rounded)
        cleanedup.append(new_row)
    grid_str = "\n".join([" ".join(map(str, row)) for row in cleanedup])
    print(grid_str)


def epjson2rad_cmd() -> None:
    """Command-line interface to converting epjson to rad."""
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=Path)
    parser.add_argument("-run", action="store_true", default=False)
    args = parser.parse_args()
    epmodel = load_energyplus_model(args.fpath)
    if "FenestrationSurface:Detailed" not in epmodel.epjs:
        raise ValueError("No windows found in this model")
    zones = epmodel_to_radmodel(epmodel)
    for zone in zones:
        write_ep_rad_model(f"{zone}.rad", zones[zone])


def genmtx_pts_sky(args) -> None:
    """Generate a point to sky matrix."""
    with open(args.pts, "r", encoding="ascii") as rdr:
        pts = [[float(i) for i in line.split()] for line in rdr.readlines()]
    sender = SensorSender(sensors=pts)
    out = Path(f"{args.pts.stem}_{args.basis}sky.mtx")
    receiver = SkyReceiver(args.basis, out=out)
    sys_paths = args.sys
    del args.pts, args.basis, args.sys, args.verbose, args.func
    mat = Matrix(sender, [receiver], octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args) if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_vu_sky(args) -> None:
    """Generate a view to sky matrix."""
    view = pr.load_views(args.vu)[0]
    sender = ViewSender(
        view,
        ray_count=1,
        xres=args.resolu[0],
        yres=args.resolu[1],
    )
    out = Path(f"{args.vu.stem}_{args.basis}sky")
    out.mkdir()
    receiver = SkyReceiver(args.basis, out=out / "%04d.hdr")
    sys_paths = args.sys
    del args.vu, args.basis, args.sys, args.resolu, args.verbose, args.func
    mat = Matrix(sender, [receiver], octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args) if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_srf_sky(args) -> None:
    """Generate a surface to sky matrix."""
    sender = SurfaceSender(
        surfaces=unpack_primitives(args.srf),
        basis=args.basis[0],
        offset=args.offset,
    )
    out = Path(f"{args.srf.stem}_{'_'.join(args.basis)}sky.mtx")
    receiver = SkyReceiver(args.basis[1], out=out)
    sys_paths = args.sys
    del args.srf, args.basis, args.sys, args.offset, args.verbose, args.func
    mat = Matrix(sender, [receiver], octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_pts_srf(args) -> None:
    """Generate a point to surface matrix."""
    with open(args.pts, "r", encoding="ascii") as rdr:
        pts = [[float(i) for i in line.split()] for line in rdr.readlines()]
    sender = SensorSender(sensors=pts, ray_count=1)
    rprims = unpack_primitives(args.srf)
    modifiers = {prim.modifier for prim in rprims if prim.ptype in ("polygon", "ring")}
    sys_paths = args.sys
    receivers = []
    for mod in modifiers:
        _receiver = [
            prim
            for prim in rprims
            if prim.modifier == mod and prim.ptype in ("polygon", "ring")
        ]
        if _receiver != []:
            outpath = Path(f"{args.pts.stem}_{args.srf.stem}.mtx")
            receivers.append(
                SurfaceReceiver(
                    surfaces=_receiver,
                    basis=args.basis,
                    offset=args.offset,
                    left_hand=False,
                    source="glow",
                    out=outpath,
                )
            )
    del args.pts, args.srf, args.sys, args.basis, args.offset, args.verbose, args.func
    mat = Matrix(sender, receivers, octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_vu_srf(args) -> None:
    """Generate a view to surface matrix."""
    view = pr.load_views(args.vu)[0]
    sender = ViewSender(
        view,
        ray_count=1,
        xres=args.resolu[0],
        yres=args.resolu[1],
    )
    rprims = unpack_primitives(args.srf)
    modifiers = {prim.modifier for prim in rprims if prim.ptype in ("polygon", "ring")}
    sys_paths = args.sys
    receivers = []
    for mod in modifiers:
        _receiver = [
            prim
            for prim in rprims
            if prim.modifier == mod and prim.ptype in ("polygon", "ring")
        ]
        if _receiver != []:
            outpath = Path(f"{args.vu.stem}_{args.srf.stem}")
            outpath.mkdir()
            receivers.append(
                SurfaceReceiver(
                    surfaces=_receiver,
                    basis=args.basis,
                    offset=args.offset,
                    left_hand=False,
                    source="glow",
                    out=outpath / "%04d.hdr",
                )
            )
    del (
        args.vu,
        args.srf,
        args.sys,
        args.basis,
        args.offset,
        args.resolu,
        args.verbose,
        args.func,
    )
    mat = Matrix(sender, receivers, octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_srf_srf(args) -> None:
    """Generate a surface to surface matrix."""
    sender = SurfaceSender(
        surfaces=unpack_primitives(args.ssrf),
        basis=args.basis[0],
        offset=args.offset[0],
    )
    rprims = unpack_primitives(args.rsrf)
    modifiers = {prim.modifier for prim in rprims if prim.ptype in ("polygon", "ring")}
    sys_paths = args.sys
    receivers = []
    for mod in modifiers:
        _receiver = [
            prim
            for prim in rprims
            if prim.modifier == mod and prim.ptype in ("polygon", "ring")
        ]
        if _receiver != []:
            outpath = Path(f"{args.ssrf.stem}_{args.rsrf.stem}.mtx")
            receivers.append(
                SurfaceReceiver(
                    surfaces=_receiver,
                    basis=args.basis[1],
                    offset=args.offset[1],
                    left_hand=False,
                    source="glow",
                    out=outpath,
                )
            )
    del args.ssrf, args.rsrf, args.sys, args.basis, args.offset, args.verbose, args.func
    mat = Matrix(sender, receivers, octree=None, surfaces=sys_paths)
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    mat.generate(params=sparams.args(), to_file=True)


def genmtx_pts_sun(args) -> None:
    """Generate a point to sun matrix."""
    with open(args.pts, "r", encoding="ascii") as rdr:
        sender = SensorSender(
            sensors=[[float(i) for i in line.split()] for line in rdr.readlines()],
            ray_count=1,
        )
    receiver = SunReceiver(
        basis=args.basis,
        sun_matrix=None,
        window_normals=None,
        full_mod=True,
    )
    sys_paths = args.sys
    out = f"{args.pts.stem}_sun.mtx"
    del args.pts, args.basis, args.sys, args.verbose, args.func
    mat = SunMatrix(
        sender,
        receiver,
        None,
        surfaces=sys_paths,
    )
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    with open(out, "wb") as f:
        f.write(mat.generate(parameters=sparams.args(), radmtx=True))


def genmtx_vu_sun(args) -> None:
    """Generate a view to sun matrix."""
    view = pr.load_views(args.vu)[0]
    xres, yres = args.resolu
    sender = ViewSender(
        view,
        ray_count=1,
        xres=args.resolu[0],
        yres=args.resolu[1],
    )
    wnormals = None
    sun_matrix = None
    if args.window is not None:
        wnormals = list(primitive_normal(args.window))
    if args.smx_path is not None:
        sun_matrix = load_matrix(args.smx_path)
    receiver = SunReceiver(
        basis=args.basis,
        sun_matrix=sun_matrix,
        window_normals=wnormals,
        full_mod=False,
    )
    outpath = Path(f"{args.vu.stem}_sun")
    sys_paths = args.sys
    del (
        args.vu,
        args.basis,
        args.sys,
        args.window,
        args.resolu,
        args.smx_path,
        args.verbose,
        args.func,
    )
    mtx = SunMatrix(
        sender,
        receiver,
        None,
        surfaces=sys_paths,
    )
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    mtx.generate(parameters=sparams.args())
    write_hdrs(mtx.array, xres=xres, yres=yres, outdir=str(outpath))


def genmtx_ncp(args: argparse.Namespace) -> None:
    """Generate a matrix/BSDF for a non-coplanar shading systems."""
    sys_paths = args.sys
    wrap = args.wrap
    wprims = unpack_primitives(args.window)
    nprims = unpack_primitives(args.ncp)
    sys_paths.append(args.ncp)
    ports = ncp.gen_port_prims_from_window_ncp(wprims, nprims)
    nmodel = ncp.NcpModel(wprims, ports, sys_paths, args.basis[0], args.basis[1])
    out = Path(f"{args.window.stem}_{args.ncp.stem}.mtx")
    del args.window, args.ncp, args.basis, args.sys, args.verbose, args.wrap
    sparams = SamplingParameters()
    sparams_dict = {k: v for k, v in vars(args).items() if v is not None}
    sparams.update_from_dict(sparams_dict)
    ncp.gen_ncp_mtx(nmodel, out, sparams.args(), wrap=wrap)


def gen() -> None:
    """Generate things."""
    parser = argparse.ArgumentParser(
        prog="gen",
        description="Generate things.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prog_subparser = parser.add_subparsers()

    # gen grid
    gengrid_descrp = "Generate an equal-spaced sensor grid based on a polygon."
    parser_grid = prog_subparser.add_parser(
        "grid",
        description=gengrid_descrp,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=gengrid_descrp,
    )
    parser_grid.set_defaults(func=gengrid)
    parser_grid.add_argument("surface", type=Path, help="surface file path.")
    parser_grid.add_argument("spacing", type=float, help="Grid spacing.")
    parser_grid.add_argument("height", type=float, help="Grid height from surface.")
    parser_grid.add_argument(
        "-op",
        action="store_true",
        help="Generate the grid to the opposite side of surface.",
    )

    # gen glaze
    genglaze_descrp = (
        "Generate a Radiance glazing material based on LBNL IGSDB or Optics file."
    )
    parser_glaze = prog_subparser.add_parser(
        "glaze",
        description=genglaze_descrp,
        help=genglaze_descrp,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_glaze.set_defaults(func=glaze)
    egroup = parser_glaze.add_mutually_exclusive_group(required=True)
    egroup.add_argument("-x", "--optics", nargs="+", type=Path, help="Optics file path")
    egroup.add_argument(
        "-d", "--igsdb", nargs="+", type=Path, help="IGSDB json file path"
    )
    parser_glaze.add_argument(
        "-c", "--cspace", default="radiance", help="Color space for color primaries"
    )
    parser_glaze.add_argument(
        "-s", "--observer", default="2", help="CIE Obvserver 2° or 10°"
    )

    # gen room
    genroom_descrp = "Generate a south-facing side-lit room"
    parser_room = prog_subparser.add_parser(
        "room",
        description=genroom_descrp,
        help=genroom_descrp,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_room.set_defaults(func=genradroom)
    parser_room.add_argument(
        "width", type=float, help="room width along X axis, starting from x=0"
    )
    parser_room.add_argument(
        "depth", type=float, help="room depth along Y axis, starting from y=0"
    )
    parser_room.add_argument(
        "flrflr",
        type=float,
        help="floor to floor height along Z axis, starting from z=0",
    )
    parser_room.add_argument(
        "flrclg",
        type=float,
        help="floor to ceiling height along Z axis, starting from z=0",
    )
    parser_room.add_argument(
        "-w",
        "--window",
        # dest="window",
        metavar=("start_x", "start_z", "width", "height"),
        nargs=4,
        action="append",
        type=float,
        help="Define a window from lower left corner",
    )
    parser_room.add_argument("-n", dest="name", help="Model name", default="model")
    parser_room.add_argument(
        "-t", dest="facade_thickness", metavar="Facade thickness", type=float
    )
    parser_room.add_argument("-r", "--rotate", type=float)

    # gen matrix
    genmatrix_descrp = "Generate various types of matrices."
    parser_matrix = prog_subparser.add_parser(
        "matrix",
        description=genmatrix_descrp,
        help=genmatrix_descrp,
    )
    mtx_subparser = parser_matrix.add_subparsers()

    # gen matrix point-sky
    parser_pts_sky = mtx_subparser.add_parser(
        "point-sky",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pts_sky.set_defaults(func=genmtx_pts_sky)
    parser_pts_sky.add_argument("pts", type=Path, help="Ray point file path")
    parser_pts_sky.add_argument("sys", nargs="+", type=Path, help="System file path[s]")
    parser_pts_sky.add_argument(
        "-b", "--basis", metavar="", default="r4", help="Sky basis."
    )
    parser_pts_sky = rparam.add_rcontrib_args(parser_pts_sky)
    # gen matrix view-sky
    parser_vu_sky = mtx_subparser.add_parser(
        "view-sky",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_vu_sky.set_defaults(func=genmtx_vu_sky)
    parser_vu_sky.add_argument("vu", type=Path, help="View file path.")
    parser_vu_sky.add_argument("sys", nargs="+", type=Path, help="System file path[s].")
    parser_vu_sky.add_argument(
        "-b", "--basis", metavar="", default="r4", help="Sky basis."
    )
    parser_vu_sky.add_argument(
        "-r",
        "--resolu",
        type=int,
        default=(800, 800),
        metavar="",
        help="Image resolution.",
    )
    parser_vu_sky = rparam.add_rcontrib_args(parser_vu_sky)
    # gen matrix surface-sky
    parser_srf_sky = mtx_subparser.add_parser(
        "surface-sky",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_srf_sky.set_defaults(func=genmtx_srf_sky)
    parser_srf_sky.add_argument("srf", type=Path, help="Surface file path.")
    parser_srf_sky.add_argument(
        "sys", nargs="+", type=Path, help="System file path[s]."
    )
    parser_srf_sky.add_argument(
        "-b",
        "--basis",
        metavar="",
        nargs=2,
        default=["kf", "r4"],
        help="Surface and sky basis",
    )
    parser_srf_sky.add_argument(
        "-f",
        "--offset",
        metavar="",
        type=float,
        default=0,
        help="Surface offset in its normal direction.",
    )
    parser_srf_sky = rparam.add_rcontrib_args(parser_srf_sky)
    # gen matrix point-surface
    parser_pts_srf = mtx_subparser.add_parser(
        "point-surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pts_srf.set_defaults(func=genmtx_pts_srf)
    parser_pts_srf.add_argument("pts", type=Path, help="Ray point file path.")
    parser_pts_srf.add_argument("srf", type=Path, help="Surface file path.")
    parser_pts_srf.add_argument(
        "sys", nargs="+", type=Path, help="System file path[s]."
    )
    parser_pts_srf.add_argument(
        "-b", "--basis", default="kf", metavar="", help="Surface basis."
    )
    parser_pts_srf.add_argument(
        "-f",
        "--offset",
        type=float,
        metavar="",
        default=0,
        help="Surface offset in its normal direction.",
    )
    parser_pts_srf = rparam.add_rcontrib_args(parser_pts_srf)
    # gen matrix view-surface
    parser_vu_srf = mtx_subparser.add_parser(
        "view-surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_vu_srf.set_defaults(func=genmtx_vu_srf)
    parser_vu_srf.add_argument("vu", type=Path, help="View file path.")
    parser_vu_srf.add_argument("srf", type=Path, help="Surface file path.")
    parser_vu_srf.add_argument("sys", nargs="+", type=Path, help="System file path[s].")
    parser_vu_srf.add_argument(
        "-b", "--basis", default="kf", metavar="", help="Surface basis."
    )
    parser_vu_srf.add_argument(
        "-r",
        "--resolu",
        nargs=2,
        type=int,
        metavar="",
        default=(800, 800),
        help="Image resolution.",
    )
    parser_vu_srf.add_argument(
        "-f",
        "--offset",
        type=float,
        metavar="",
        default=0,
        help="Surface offset in its normal direction.",
    )
    parser_vu_srf = rparam.add_rcontrib_args(parser_vu_srf)
    # gen matrix surface-surface
    parser_srf_srf = mtx_subparser.add_parser(
        "surface-surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_srf_srf.set_defaults(func=genmtx_srf_srf)
    parser_srf_srf.add_argument("ssrf", type=Path, help="Sender surface path.")
    parser_srf_srf.add_argument("rsrf", type=Path, help="Receiver surface path.")
    parser_srf_srf.add_argument("sys", nargs="+", type=Path, help="System path[s].")
    parser_srf_srf.add_argument(
        "-b",
        "--basis",
        nargs=2,
        default=["kf", "kf"],
        metavar="",
        help="Sender and receiver basis.",
    )
    parser_srf_srf.add_argument(
        "-f",
        "--offset",
        nargs=2,
        type=float,
        default=[0, 0],
        metavar="",
        help="Sender and receiver offsets in their normal direction.",
    )
    parser_srf_srf = rparam.add_rcontrib_args(parser_srf_srf)
    # gen matrix point-sun
    parser_pts_sun = mtx_subparser.add_parser(
        "point-sun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_pts_sun.set_defaults(func=genmtx_pts_sun)
    parser_pts_sun.add_argument("pts", type=Path, help="Point file path.")
    parser_pts_sun.add_argument(
        "sys", nargs="+", type=Path, help="System file path[s]."
    )
    parser_pts_sun.add_argument(
        "-b", "--basis", default="r6", metavar="", help="Sun basis."
    )
    parser_pts_sun = rparam.add_rcontrib_args(parser_pts_sun)
    # gen matrix view-sun
    parser_vu_sun = mtx_subparser.add_parser(
        "view-sun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_vu_sun.set_defaults(func=genmtx_vu_sun)
    parser_vu_sun.add_argument("vu", type=Path, help="View file path.")
    parser_vu_sun.add_argument("sys", nargs="+", type=Path, help="System file path[s].")
    parser_vu_sun.add_argument(
        "-b", "--basis", default="r6", metavar="", help="Sun basis."
    )
    parser_vu_sun.add_argument(
        "-f", "--window", nargs="+", type=Path, metavar="", help="Window file path[s]."
    )
    parser_vu_sun.add_argument(
        "-s", "--smx_path", type=Path, metavar="", help="Sky matrix file path."
    )
    parser_vu_sun.add_argument(
        "-r",
        "--resolu",
        nargs=2,
        type=int,
        default=(800, 800),
        metavar="",
        help="Image resolution.",
    )
    parser_vu_sun = rparam.add_rcontrib_args(parser_vu_sun)
    # gen matrix ncp
    parser_ncp = mtx_subparser.add_parser(
        "ncp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_ncp.set_defaults(func=genmtx_ncp)
    parser_ncp.add_argument("window", type=Path, help="Window file paths.")
    parser_ncp.add_argument("ncp", type=Path, help="Non-coplanar shading file paths.")
    parser_ncp.add_argument("sys", nargs="+", type=Path, help="System file path[s].")
    parser_ncp.add_argument(
        "-b",
        "--basis",
        nargs=2,
        default=["kf", "kf"],
        metavar="",
        help="Window and receiving basis.",
    )
    parser_ncp.add_argument(
        "-x",
        "--wrap",
        action="store_true",
        default=False,
        help="Generating a final xml file?",
    )
    parser_ncp = rparam.add_rcontrib_args(parser_ncp)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=(
            "Verbose mode: -v=Debug -vv=Info -vvv=Warning "
            "-vvvv=Error -vvvvv=Critical"
        ),
    )
    args = parser.parse_args()
    # Setup logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%y-%m-%d %H:%M:%S"
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
        parser.print_usage(sys.stderr)
        sys.exit(1)
    elif len(sys.argv) == 2 and (sys.argv[1] == "matrix"):
        parser_matrix.print_usage(sys.stderr)
        sys.exit(1)
    args.func(args)


# def rpxop() -> None:
#     """Operate on input directories given a operation type."""
#     parser = argparse.ArgumentParser(prog="rpxop", description="Batch image processing")
#     subparser = parser.add_subparsers()
#     parser_dcts = subparser.add_parser("dctimestep")
#     parser_dcts.set_defaults(func=mtxmult.batch_dctimestep)
#     parser_dcts.add_argument("mtx", nargs="+", type=Path, help="input matrices")
#     parser_dcts.add_argument("sky", type=Path, help="sky files directory")
#     parser_dcts.add_argument("out", type=Path, help="output directory")
#     parser_dcts.add_argument("-n", type=int, help="number of processors to use")
#     parser_pcomb = subparser.add_parser("pcomb")
#     parser_pcomb.set_defaults(func=mtxmult.batch_pcomb)
#     parser_pcomb.add_argument(
#         "inp", type=str, nargs="+", help="list of inputs, e.g., inp1 + inp2.hdr"
#     )
#     parser_pcomb.add_argument("out", type=Path, help="output directory")
#     parser_pcomb.add_argument("-n", type=int, help="number of processors to use")
#     args = parser.parse_args()
#     if args.func == mtxmult.batch_pcomb:
#         inp = [Path(i) for i in args.inp[::2]]
#         for i in inp:
#             if not i.exists():
#                 raise FileNotFoundError(i)
#         ops = args.inp[1::2]
#         args.func(inp, ops, args.out, nproc=args.n)
#     elif args.func == mtxmult.batch_dctimestep:
#         for i in args.mtx:
#             if not i.exists():
#                 raise FileNotFoundError(i)
#         args.func(args.mtx, args.sky, args.out, nproc=args.n)


def genradroom(args) -> None:
    """Commandline interface for generating a generic room.
    Resulting Radiance .rad files will be written to a local
    Objects directory, which will be created if not existed before.
    """
    aroom = create_south_facing_room(
        args.width,
        args.depth,
        args.flrflr,
        args.flrclg,
        wpd=args.window,
        swall_thickness=args.facade_thickness,
    )
    name = args.name
    objdir = Path("Objects")
    objdir.mkdir(exist_ok=True)
    with open(objdir / f"materials_{name}.mat", "w", encoding="ascii") as wtr:
        for prim in aroom.materials:
            wtr.write(str(prim) + "\n")
    with open(objdir / f"ceiling_{name}.rad", "w", encoding="ascii") as wtr:
        for prim in aroom.ceiling.primitives:
            wtr.write(str(prim) + "\n")
    with open(objdir / f"floor_{name}.rad", "w", encoding="ascii") as wtr:
        for prim in aroom.floor.primitives:
            wtr.write(str(prim) + "\n")
    with open(objdir / f"wall_{name}.rad", "w", encoding="ascii") as wtr:
        for prim in [
            *aroom.swall.primitives,
            *aroom.ewall.primitives,
            *aroom.nwall.primitives,
            *aroom.wwall.primitives,
        ]:
            wtr.write(str(prim) + "\n")
    for idx, srf in enumerate(aroom.swall.windows):
        with open(
            objdir / f"window_{idx:02d}_{name}.rad", "w", encoding="ascii"
        ) as wtr:
            wtr.write(str(srf.primitive) + "\n")
