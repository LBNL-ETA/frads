"""
Executive command-line program for Radiance matrix-based simulation.
"""

import argparse
from configparser import ConfigParser
import json
import glob
import logging
import os
from pathlib import Path
import subprocess as sp
import tempfile as tf
from typing import List
from typing import Optional
from typing import Sequence

from frads import color
from frads import epjson2rad
from frads import geom
from frads import matrix
from frads import methods
from frads import mtxmult
from frads import ncp
from frads import parsers
from frads import room
from frads import utils
from frads.types import PaneRGB
from frads.types import Receiver
from frads.types import Sender

try:
    import numpy as np

    NUMPY_FOUND = True
except ModuleNotFoundError:
    NUMPY_FOUND = False


logger = logging.getLogger("frads")


def mrad_init(args: argparse.Namespace) -> None:
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
    config["DEFAULT"] = {
        "start_hour": 0,
        "end_hour": 0,
        "daylight_hours_only": True,
        "overwrite": True,
    }
    config["SimControl"] = {
        "vmx_basis": "kf",
        "vmx_opt": "-ab 5 -ad 65536 -lw 1e-5",
        "fmx_basis": "kf",
        "smx_basis": "r4",
        "dmx_opt": "-ab 2 -ad 128 -c 5000",
        "dsmx_opt": "-ab 7 -ad 16384 -lw 5e-5",
        "cdsmx_opt": "-ab 1",
        "cdsmx_basis": "r6",
        "ray_count": "1",
        "nprocess": "1",
        "separate_direct": "False",
        "overwrite": "False",
        "method": "",
    }
    site = {
        "wea_path": args.wea_path,
        "epw_path": args.epw_path,
        "start_hour": "",
        "end_hour": "",
        "daylight_hours_only": "",
    }
    model = {
        "name": args.name,
        "material": "",
        "scene": "",
        "window_paths": "",
        "window_xml": "",
        "window_cfs": "",
        "window_control": "",
    }
    raysender = {
        "grid_surface": args.grid[0],
        "grid_spacing": args.grid[1],
        "grid_height": args.grid[2],
        "view": "",
    }
    material_list = []
    window_list = []
    object_list: List[str] = []
    if args.object is not None:
        for obj in args.object:
            if obj.is_dir():
                object_list.extend(obj.glob("*.rad"))
            else:
                object_list.append(str(obj))
    else:
        logger.warning("Object files not set")
    if args.material is not None:
        for mat in args.material:
            if mat.is_dir():
                material_list.extend(glob.glob(str(mat / "*")))
            else:
                material_list.extend(glob.glob(str(mat)))
    else:
        logger.warning("Material files not set")
    if args.window is not None:
        for win in args.window:
            if win.is_dir():
                window_list.extend(glob.glob(str(win / "*")))
            else:
                window_list.extend(glob.glob(str(win)))
    else:
        logger.warning("Window files not set")
    if args.xmls is not None:
        xml_list = []
        for xml in args.xmls:
            if xml.is_dir():
                xml_list.extend(glob.glob(str(xml / "*")))
            else:
                xml_list.extend(glob.glob(str(xml)))
        if len(window_list) != len(xml_list):
            raise ValueError("Number of window and xml files not the same")
        model["window_xmls"] = "\n".join(xml_list)
    model["scene"] = "\n".join(object_list)
    model["material"] = "\n".join(material_list)
    model["window_paths"] = "\n".join(window_list)
    templ_config = {"Site": site, "Model": model, "RaySender": raysender}
    config.read_dict(templ_config)
    with open(f"{args.name}.cfg", "w") as rdr:
        config.write(rdr)


def mrad_run(args: argparse.Namespace) -> None:
    """Call mtxmethod to carry out the actual simulation."""
    cfg = ConfigParser(allow_no_value=False, inline_comment_prefixes="#")
    cfg["DEFAULT"] = {
        "start_hour": 0,
        "end_hour": 0,
        "daylight_hours_only": True,
        "overwrite": True,
    }
    with open(args.cfg) as rdr:
        cfg.read_string(rdr.read())
    try:
        name = cfg["Model"]["name"]
    except KeyError:
        name = args.cfg.stem
        cfg["Model"]["name"] = name
    Path("Matrices").mkdir(exist_ok=True)
    Path("Results").mkdir(exist_ok=True)
    model = methods.assemble_model(cfg)
    method = cfg["SimControl"]["method"]
    if method != "":
        _direct = False
        if method.startswith(("2", "two")):
            methods.two_phase(model, cfg)
        elif method.startswith(("3", "three")):
            methods.three_phase(model, cfg)
        elif method.startswith(("5", "five")):
            methods.three_phase(model, cfg, direct=cfg["SimControl"]["separate_direct"])
    else:
        if "" in (cfg["Model"]["window_xml"], cfg["Model"]["window_paths"]):
            logger.info("Using two-phase method")
            methods.two_phase(model, cfg)
        else:
            if len(cfg["Model"]["ncp_shade"].split()) > 1:
                if cfg.getboolean("SimControl", "separate_direct"):
                    logger.info("Using six-phase simulation")
                    methods.four_phase(model, cfg, direct=True)
                else:
                    logger.info("Using four-phase simulation")
                    methods.four_phase(model, cfg)
            else:
                if cfg.getboolean("SimControl", "separate_direct"):
                    logger.info("Using five-phase simulation")
                    methods.three_phase(model, cfg, direct=True)
                else:
                    logger.info("Using three-phase simulation")
                    methods.three_phase(model, cfg)
    os.remove(model.material_path)
    os.remove(model.black_env_path)



def mrad() -> None:
    """mrad entry point: parse arugments for init and run subprograms."""
    prog_description = "Executive program for Radiance matrix-based simulation"
    parser = argparse.ArgumentParser(
        prog="mrad",
        description=prog_description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparser = parser.add_subparsers()
    # Parse arguments for init subprogram
    parser_init = subparser.add_parser("init")
    parser_init.set_defaults(func=mrad_init)
    parser_init.add_argument("-W", "--wea_path", type=Path, default=Path(""))
    parser_init.add_argument("-E", "--epw_path", type=Path, default=Path(""))
    parser_init.add_argument("-n", "--name", default="default")
    parser_init.add_argument("-o", "--object", nargs="+", type=Path)
    parser_init.add_argument("-m", "--material", nargs="+", type=Path)
    parser_init.add_argument("-w", "--window", nargs="+", type=Path)
    parser_init.add_argument("-x", "--xmls", nargs="+", type=Path)
    parser_init.add_argument("-g", "--grid", nargs=3, default=("", 0, 0))
    # Parse arguments for run subprogram
    parser_run = subparser.add_parser("run")
    parser_run.set_defaults(func=mrad_run)
    parser_run.add_argument("cfg", type=Path, help="configuration file path")
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
    # Setup logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Call subprograms to do work
    args.func(args)


def glazing():
    """Command-line program for generating BRTDfunc for glazing system."""
    aparser = argparse.ArgumentParser(
        prog="glazing", description="Generate BRTDfunc for a glazing system"
    )
    egroup = aparser.add_mutually_exclusive_group(required=True)
    egroup.add_argument("-x", "--optics", nargs="+", type=Path, help="Optics file path")
    egroup.add_argument("-d", "--igsdb", nargs="+", help="IGSDB json file path or ID")
    aparser.add_argument("-t", "--token", help="IGSDB token")
    aparser.add_argument("-c", "--cspace", default="radiance", help="Color space to determine primaries")
    aparser.add_argument("-s", "--observer", default="2", help="CIE Obvserver 2° or 10°")
    args = aparser.parse_args()
    if args.optics is not None:
        panes = [parsers.parse_optics(fpath) for fpath in args.optics]
    elif args.igsdb is not None:
        panes = []
        for item in args.igsdb:
            if Path(item).is_file():
                with open(item, "r") as rdr:
                    json_obj = json.load(rdr)
            elif item[0].isdigit():
                if args.token is None:
                    raise ValueError("Missing IGSDB token")
                json_string = utils.get_igsdb_json(item, args.token)
                json_obj = json.loads(json_string)
            else:
                raise ValueError("Unknown IGSDB entry format")
            panes.append(parsers.parse_igsdb_json(json_obj))
    else:
        raise ValueError("Need to specify either optics or igsdb file")
    pane_rgb = []
    coeffs = color.get_conversion_matrix(args.cspace)
    for pane in panes:
        trix, triy, triz, mlnp, oidx = color.load_cie_tristi(
            pane.wavelength, args.observer
        )
        trans = [pane.transmittance[idx] for idx in oidx]
        reflf = [pane.reflectance_front[idx] for idx in oidx]
        reflb = [pane.reflectance_back[idx] for idx in oidx]
        tf_x, tf_y, tf_z = color.spec2xyz(trix, triy, triz, mlnp, trans)
        rf_x, rf_y, rf_z = color.spec2xyz(trix, triy, triz, mlnp, reflf)
        rb_x, rb_y, rb_z = color.spec2xyz(trix, triy, triz, mlnp, reflb)
        tf_rgb = color.xyz2rgb(tf_x, tf_y, tf_z, coeffs)
        rf_rgb = color.xyz2rgb(rf_x, rf_y, rf_z, coeffs)
        rb_rgb = color.xyz2rgb(rb_x, rb_y, rb_z, coeffs)
        if pane.coated_side == "front":
            coated_rgb = rf_rgb
            glass_rgb = rb_rgb
        else:
            coated_rgb = rb_rgb
            glass_rgb = rf_rgb
        pane_rgb.append(PaneRGB(pane, coated_rgb, glass_rgb, tf_rgb))
    print(utils.get_glazing_primitive(pane_rgb))


def gengrid():
    """Commandline program for generating a grid of sensor points."""
    parser = argparse.ArgumentParser(
        prog="gengrid",
        description="Generate an equal-spaced sensor grid based on a surface.",
    )
    parser.add_argument("surface", help="surface file path")
    parser.add_argument("spacing", type=float)
    parser.add_argument("height", type=float)
    parser.add_argument("-op", action="store_const", const="", default=True)
    args = parser.parse_args()
    prims = utils.unpack_primitives(args.surface)
    polygon_prims = [prim for prim in prims if prim.ptype == "polygon"]
    polygon = parsers.parse_polygon(polygon_prims[0].real_arg)
    if args.op:
        polygon = polygon.flip()
    grid_list = utils.gen_grid(polygon, args.height, args.spacing)
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


def varays():
    """Commandline utility program for generating circular fisheye rays."""
    aparser = argparse.ArgumentParser(
        prog="varays",
        description="Generate a fisheye view rays with blackedout corners",
    )
    aparser.add_argument("-x", required=True, help="square image resolution")
    aparser.add_argument("-c", default="1", help="Ray count")
    aparser.add_argument("-vf", required=True, help="View file path")
    args = aparser.parse_args()
    cmd = ["vwrays", "-ff", "-x", args.x, "-y", args.x]
    if args.c != "1":
        cmd += ["-c", args.c, "-pj", "0.7"]
    cmd += ["-vf", args.vf]
    proc1 = sp.run(cmd, check=True, stdout=sp.PIPE)
    sp.run(utils.crop2circle(args.c, args.x), check=True, stdin=proc1.stdout)


def epjson2rad_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath")
    parser.add_argument("-run", action="store_true", default=False)
    args = parser.parse_args()
    epjs = epjson2rad.read_ep_input(args.fpath)
    if "FenestrationSurface:Detailed" not in epjs:
        raise ValueError("No windows found in this model")
    epjson2rad.epjson2rad(epjs)


def genmtx_sky(sender: Sender, args: argparse.Namespace):
    logger.info("Sky is the receiver")
    env = args.env
    if args.octree is not None:
        env.extend(["-i", str(args.octree)])
    receiver = matrix.sky_as_receiver(args.receiver_basis)
    outpath = args.output[0]
    if sender.form == "v":
        outpath.mkdir(exist_ok=True)
    res = matrix.rfluxmtx(
        sender=sender, receiver=receiver, env=env, opt=args.option, out=outpath
    )
    if (outpath is not None) and (sender.form != "v"):
        with open(outpath, "wb") as wtr:
            wtr.write(res)


def genmtx_sun(sender: Sender, env, args: argparse.Namespace) -> None:
    logger.info("Suns are the receivers.")
    full_modifier = False
    if sender.form != "v":
        full_modifier = True
    wnormals: Optional[Sequence[geom.Vector]] = None
    if args.window is not None:
        wnormals = utils.primitive_normal(args.window)
    receiver = matrix.sun_as_receiver(
        basis=args.receiver_basis,
        smx_path=args.smx,
        window_normals=wnormals,
        full_mod=full_modifier,
    )
    outpath = args.outpath[0]
    sun_oct = "sun.oct"
    matrix.rcvr_oct(receiver, env, sun_oct)
    with tf.TemporaryDirectory(dir=os.getcwd()) as tempd:
        mod_names = [
            "%04d" % (int(line[3:]) - 1) for line in receiver.modifier.splitlines()
        ]
        matrix.rcontrib(
            sender=sender,
            modifier=receiver.modifier,
            octree=sun_oct,
            out=tempd,
            opt=args.option,
        )
        _files = sorted(Path(tempd).glob("*.hdr"))
        outpath.mkdir(exist_ok=True)
        for idx, val in enumerate(_files):
            val.rename(outpath / (mod_names[idx] + ".hdr"))
        os.remove(sun_oct)


def genmtx_surface(sender: matrix.Sender, env, args: argparse.Namespace) -> None:
    rcvr_prims = []
    for path in args.rsurface:
        rcvr_prims.extend(utils.unpack_primitives(path))
    modifiers = set([prim.modifier for prim in rcvr_prims])
    receiver = Receiver(receiver="", basis=args.receiver_basis, modifier="")
    for mod, op in zip(modifiers, args.output):
        _receiver = [
            prim
            for prim in rcvr_prims
            if prim.modifier == mod and prim.ptype in ("polygon", "ring")
        ]
        if _receiver != []:
            if args.sender_type == "v":
                _outpath = os.path.join(op, "%04d.hdr")
            else:
                _outpath = op
            receiver += matrix.surface_as_receiver(
                prim_list=_receiver,
                basis=args.receiver_basis,
                offset=args.receiver_offset,
                left=None,
                source="glow",
                out=_outpath,
            )
    matrix.rfluxmtx(
        sender=sender, receiver=receiver, env=env, opt=args.option, out=None
    )


def genmtx_ncp(args: argparse.Namespace):
    wprims = utils.unpack_primitives(args.window)
    nprims = utils.unpack_primitives(args.ncp)
    env = args.env
    if args.octree is not None:
        env.extend(["-i", args.octree])
    env.append(args.ncp)
    if args.depth_scale is not None:
        ports = ncp.gen_port_prims_from_window(wprims, *args.depth_scale)
    else:
        ports = ncp.gen_port_prims_from_window_ncp(wprims, nprims)
    nmodel = ncp.Model(wprims, ports, env, args.sbasis, args.rbasis)
    ncp.gen_ncp_mtx(nmodel, args.output, args.opt, args.solar)


def genmtx():
    """Generate a matrix."""
    description = "Generate flux transport matrix"
    parser = argparse.ArgumentParser(prog="genmtx", description=description)
    subparser = parser.add_subparsers()

    parser_sky = subparser.add_parser("sky")
    parser_sky.set_defaults(func=genmtx_sky)
    egroup_sky = parser_sky.add_mutually_exclusive_group(required=True)
    egroup_sky.add_argument("-p", "--point", type=argparse.FileType("r"))
    egroup_sky.add_argument("-s", "--surface", type=argparse.FileType("r"))
    egroup_sky.add_argument("-u", "--view", type=argparse.FileType("r"))
    parser_sky.add_argument("-r", "--basis", default="r4")
    parser_sky.add_argument("-s", "--sbasis")
    parser_sky.add_argument("-so", "--soffset")

    parser_surface = subparser.add_parser("surface")
    parser_surface.set_defaulas(func=genmtx_surface)
    parser_surface.add_argument("rsurface", nargs="+", type=Path)
    egroup_surface = parser_surface.add_mutually_exclusive_group(required=True)
    egroup_surface.add_argument("-p", "--point", type=argparse.FileType("r"))
    egroup_surface.add_argument("-s", "--surface", type=argparse.FileType("r"))
    egroup_surface.add_argument("-u", "--view", type=argparse.FileType("r"))
    parser_surface.add_argument("-r", "--rbasis", default="kf")
    parser_surface.add_argument("-s", "--sbasis")
    parser_surface.add_argument("-so", "--soffset")
    parser_surface.add_argument("-ro", "--roffset")

    parser_sun = subparser.add_parser("sun")
    parser_sun.set_defaults(func=genmtx_sun)
    egroup_sun = parser_sun.add_mutually_exclusive_group(required=True)
    egroup_sun.add_argument("-p", "--point", type=argparse.FileType("r"))
    egroup_sun.add_argument("-u", "--view", type=argparse.FileType("r"))
    parser_sun.add_argument("-r", "--rbasis", default="r6")
    parser_sun.add_argument("-s", "--smx", help="Sky matrix file")
    parser_sun.add_argument("-w", "--window", nargs="+", help="window files paths")

    parser_ncp = subparser.add_parser("ncp")
    parser_ncp.set_defaults(func=genmtx_ncp)
    parser_ncp.add_argument("window")
    egroup_ncp = parser_ncp.add_mutually_exclusive_group(required=True)
    egroup_ncp.add_argument("-n", "--ncp", type=Path)
    egroup_ncp.add_argument("-d", "--depth_scale", nargs=2, type=float)
    parser_ncp.add_argument("-r", "--rbasis", default="kf")
    parser_ncp.add_argument("-s", "--sbasis", default="kf")
    parser_ncp.add_argument("-u", "--solar", action="store_true")

    parser.add_argument("-o", "--output", nargs="+", type=Path)
    parser.add_argument("-e", "--env", nargs="+", type=argparse.FileType("wb"))
    parser.add_argument("-i", "--oct", type=Path)
    parser.add_argument("-f", "--opt")
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
    logger = logging.getLogger("frads")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    _level = args.verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    assert len(args.receiver) == len(args.outpath)
    env = args.env
    if args.octree is not None:
        env.extend(["-i", args.octree])
    # what's the sender
    if args.surface:
        prims = utils.unpack_primitives(args.surface)
        sender = matrix.surface_as_sender(
            prim_list=prims, basis=args.sbasis, offset=args.soffset
        )
        args.surface.close()
    elif args.view:
        vudict = parsers.parse_vu(args.view.readlines()[-1])  # use the last view
        sender = matrix.view_as_sender(
            vu_dict=vudict,
            ray_cnt=args.ray_count,
            xres=args.resolu[0],
            yres=args.resolu[1],
        )
        args.view.close()
    elif args.point:
        pts = [line.split() for line in args.point.readlines()]
        sender = matrix.points_as_sender(pts_list=pts, ray_cnt=args.ray_count)
        args.point.close()
    else:
        sender = None
    if args.func is not genmtx_ncp:
        args.func(sender, env, args)
    else:
        args.func(args)
    # figure out receiver
    # if args.receiver[0] == 'sky':
    #     logger.info('Sky is the receiver')
    #     receiver = matrix.Receiver.as_sky(args.receiver_basis)
    #     outpath = args.outpath[0]
    #     if args.sender_type == 'v':
    #         util.mkdir_p(outpath)
    # elif args.receiver[0] == 'sun':
    #     full_modifier = False
    #     if args.sender_type != 'v':
    #         full_modifier = True
    #     window_normals: Union[None, Set[radgeom.Vector]] = None
    #     if args.wpths is not None:
    #         window_normals = radutil.primitive_normal(args.wpths)
    #     receiver = matrix.Receiver.as_sun(
    #         basis=args.receiver_basis, smx_path=args.smx,
    #         window_normals=window_normals, full_mod=full_modifier)
    # else:  # assuming multiple receivers
    #     rcvr_prims = []
    #     for path in args.receiver:
    #         rcvr_prims.extend(radutil.unpack_primitives(path))
    #     modifiers = set([prim.modifier for prim in rcvr_prims])
    #     receiver = matrix.Receiver(
    #         receiver='', basis=args.receiver_basis, modifier=None)
    #     for mod, op in zip(modifiers, args.outpath):
    #         _receiver = [prim for prim in rcvr_prims
    #                      if prim.modifier == mod and
    #                      prim.ptype in ('polygon', 'ring')]
    #         if _receiver != []:
    #             if args.sender_type == 'v':
    #                 _outpath = os.path.join(op, '%04d.hdr')
    #             else:
    #                 _outpath = op
    #             receiver += matrix.Receiver.as_surface(
    #                 prim_list=_receiver, basis=args.receiver_basis,
    #                 offset=args.receiver_offset, left=None,
    #                 source='glow', out=_outpath)
    #     outpath = None
    # generate matrices
    # if args.receiver[0] == 'sun':
    #     logger.info('Suns are the receivers.')
    #     outpath = os.path.join(os.getcwd(), args.outpath[0])
    #     sun_oct = 'sun.oct'
    #     matrix.rcvr_oct(receiver, env, sun_oct)
    #     with tf.TemporaryDirectory(dir=os.getcwd()) as tempd:
    #         mod_names = ["%04d" % (int(l[3:])-1)
    #                      for l in receiver.modifier.splitlines()]
    #         matrix.rcontrib(sender=sender, modifier=receiver.modifier, octree=sun_oct,
    #                    out=tempd, opt=args.option)
    #         _files = [os.path.join(tempd, f) for f in sorted(os.listdir(tempd))
    #                   if f.endswith('.hdr')]
    #         util.mkdir_p(outpath)
    #         for idx, val in enumerate(_files):
    #             os.rename(val, os.path.join(outpath, mod_names[idx]+'.hdr'))

    # else:
    #     res = matrix.rfluxmtx(sender=sender, receiver=receiver, env=env, opt=args.option, out=outpath)
    #     if (outpath is not None) and (args.sender_type != 'v'):
    #         with open(outpath, 'wb') as wtr:
    #             wtr.write(res)


def dctsnp():
    """Commandline program that performs matrix multiplication using numpy."""
    if not NUMPY_FOUND:
        print("Numpy not found")
        return
    aparser = argparse.ArgumentParser(
        prog="dctsnp", description="dctimestep but using numpy (non-image)"
    )
    aparser.add_argument("-m", "--mtx", required=True, nargs="+", help="scene matrix")
    aparser.add_argument("-s", "--smx", required=True, help="sky matrix")
    aparser.add_argument(
        "-w", "--weight", type=float, default=None, nargs=3, help="RGB weights"
    )
    aparser.add_argument("-o", "--output", required=True, help="output path")
    args = aparser.parse_args()

    def mtx_parser(fpath):
        if fpath.endswith(".xml"):
            raw = utils.spcheckout(["rmtxop", fpath])
        else:
            with open(fpath, "rb") as rdr:
                raw = rdr.read()
        return mtxmult.mtxstr2nparray(raw)

    npmtx = [mtx_parser(mtx) for mtx in args.mtx]
    with open(args.smx, "rb") as rdr:
        npmtx.append(mtxmult.smx2nparray(rdr.read()))
    result = mtxmult.numpy_mtxmult(npmtx, weight=args.weight)
    np.savetxt(args.output, result)


def rpxop():
    """Operate on input directories given a operation type."""
    PROGRAM_SCRIPTION = "Batch image processing."
    parser = argparse.ArgumentParser(prog="rpxop", description=PROGRAM_SCRIPTION)
    subparser = parser.add_subparsers()
    parser_dcts = subparser.add_parser("dctimestep")
    parser_dcts.set_defaults(func=mtxmult.batch_dctimestep)
    parser_dcts.add_argument("mtx", nargs="+", type=Path, help="input matrices")
    parser_dcts.add_argument("sky", type=Path, help="sky files directory")
    parser_dcts.add_argument("out", type=Path, help="output directory")
    parser_dcts.add_argument("-n", type=int, help="number of processors to use")
    parser_pcomb = subparser.add_parser("pcomb")
    parser_pcomb.set_defaults(func=mtxmult.batch_pcomb)
    parser_pcomb.add_argument(
        "inp", type=str, nargs="+", help="list of inputs, e.g., inp1 + inp2.hdr"
    )
    parser_pcomb.add_argument("out", type=Path, help="output directory")
    parser_pcomb.add_argument("-n", type=int, help="number of processors to use")
    args = parser.parse_args()
    if args.func == mtxmult.batch_pcomb:
        inp = [Path(i) for i in args.inp[::2]]
        for i in inp:
            if not i.exists():
                raise FileNotFoundError(i)
        ops = args.inp[1::2]
        args.func(inp, ops, args.out, nproc=args.n)
    elif args.func == mtxmult.batch_dctimestep:
        for i in args.mtx:
            if not i.exists():
                raise FileNotFoundError(i)
        args.func(args.mtx, args.sky, args.out, nproc=args.n)


def genradroom():
    """Commandline interface for generating a generic room.
    Resulting Radiance .rad files will be written to a local
    Objects directory, which will be created if not existed before.
    """

    parser = argparse.ArgumentParser(
        prog="genradroom", description="Generate a generic room"
    )
    parser.add_argument(
        "width", type=float, help="room width along X axis, starting from x=0"
    )
    parser.add_argument(
        "depth", type=float, help="room depth along Y axis, starting from y=0"
    )
    parser.add_argument(
        "height", type=float, help="room height along Z axis, starting from z=0"
    )
    parser.add_argument(
        "-w",
        dest="window",
        metavar=("start_x", "start_z", "width", "height"),
        nargs=4,
        action="append",
        type=float,
        help="Define a window from lower left corner",
    )
    parser.add_argument("-n", dest="name", help="Model name", default="model")
    parser.add_argument(
        "-t", dest="facade_thickness", metavar="Facade thickness", type=float
    )
    parser.add_argument("-r", "--rotate", type=float)
    args = parser.parse_args()
    dims = vars(args)
    for idx, window in enumerate(dims["window"]):
        dims["window_%s" % idx] = " ".join(map(str, window))
    dims.pop("window")
    aroom = room.make_room(dims)
    name = args.name
    material_primitives = utils.material_lib()
    objdir = Path("Objects")
    objdir.mkdir(exist_ok=True)
    with open(objdir / f"materials_{name}.mat", "w") as wtr:
        for prim in material_primitives:
            wtr.write(str(prim) + "\n")
    with open(objdir / f"ceiling_{name}.rad", "w") as wtr:
        for prim in aroom.srf_prims:
            if prim.identifier.startswith("ceiling"):
                wtr.write(str(prim) + "\n")
    with open(objdir / f"floor_{name}.rad", "w") as wtr:
        for prim in aroom.srf_prims:
            if prim.identifier.startswith("floor"):
                wtr.write(str(prim) + "\n")
    with open(objdir / f"wall_{name}.rad", "w") as wtr:
        for prim in aroom.srf_prims:
            if prim.identifier.startswith("wall"):
                wtr.write(str(prim) + "\n")
    for key, prim in aroom.wndw_prims.items():
        with open(objdir / f"{key}_{name}.rad", "w") as wtr:
            wtr.write(str(prim) + "\n")
