"""
gen cli
"""
import argparse
import logging
from pathlib import Path
import sys

from frads.matrix import load_matrix, Matrix, SensorSender, SkyReceiver, SurfaceSender, SurfaceReceiver, SunMatrix, SunReceiver, ViewSender
from frads.room import create_south_facing_room
from frads.utils import unpack_primitives, parse_polygon, gen_grid, primitive_normal, write_hdrs
from frads.window import PaneRGB, get_glazing_primitive
from frads import ncp
import pywincalc as pwc
from pyradiance import lib
import pyradiance as pr
from pyradiance.param import SamplingParameters
from pyradiance import param as rparam

logger = logging.getLogger(__name__)

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
        swall_thickness=args.facade_thickness,
        wpd=args.window,
    )
    name = args.name
    objdir = Path("Objects")
    objdir.mkdir(exist_ok=True)
    with open(objdir / f"materials_{name}.mat", "w", encoding="ascii") as wtr:
        for prim in aroom.materials.values():
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
            for prim in srf.primitives:
                wtr.write(str(prim) + "\n")
