"""
Call genBSDF to generate BSDF.

TODO:
    1. New feature for custom input period geometry
    2. New feature for custom section drawing
"""

import argparse
import logging
from math import cos, radians, degrees
import os
from pathlib import Path
import subprocess as sp
import tempfile as tf

from frads import radutil
from frads import radgeom


def setup_logger(verbose):
    """Set up a logger given verbosity."""
    logger = logging.getLogger("frads")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    _level = verbose * 10
    logger.setLevel(_level)
    console_handler.setLevel(_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_polygon_width_height(vertices):
    dim1 = vertices[0] - vertices[1]
    dim2 = vertices[1] - vertices[2]
    if dim1.normalize() in (radgeom.Vector(0, 0, 1), radgeom.Vector(0, 0, -1)):
        height = dim1.length()
        width = dim2.length()
    else:
        height = dim2.length()
        width = dim1.length()
    return width, height


def get_genblinds_command(name: str, mat_name: str, depth, width,
                          height, nslats, angle,
                          gap=None, curve=None, ext=False, gb_xform=True):
    if curve is None:
        curve = ""
    if gap is None:
        gap = 0
    else:
        if gap < 0:
            raise ValueError("gap can't be negative")
        gap = -gap if ext else 0
    cmd = f"!genblinds {mat_name} {name} {depth} "
    cmd += f"{width} {height} {nslats} {angle} {curve} "
    if gb_xform:
        cmd += f"| xform -rz -90 -rx 90 -t {-width/2} {height/2} {gap}"
    return cmd


def gen_blind_material(bmname, trans, refl, spec, rough) -> radutil.Primitive:
    if trans == 0:
        blind_material = radutil.neutral_plastic_prim(
            "void", bmname, refl, spec, rough)
    else:
        blind_material = radutil.neutral_trans_prim(
            "void", bmname, trans, refl, spec, rough)
    return blind_material


def gen_generic_blinds_bsdf(gblinds_cmd, bm_prim, opath, thickness,
                            spacing, logger, wm_prim=None, window_z=None,
                            ray_count=None, opt=None) -> None:
    window_str = ""
    wm_str = ""
    if None not in (wm_prim, window_z):
        wm_str = str(wm_prim)
        window_prim = radutil.Primitive(
            wm_prim.identifier,
            "polygon", "genbsdf_window", "0",
            f"12 -10 -10 {window_z} 10 -10 {window_z} 10 10 {window_z} -10 10 {window_z}")
        window_str = str(window_prim)
    with tf.TemporaryDirectory() as td:
        inp = os.path.join(td, "blinds.rad")
        with open(inp, 'w') as wtr:
            wtr.write(wm_str)
            wtr.write(window_str)
            wtr.write(str(bm_prim))
            wtr.write(gblinds_cmd)
        genbsdf_cmd = ["genBSDF", "-n", "4", "-f", "+b"]
        if opt is not None:
            genbsdf_cmd += ["-r", opt]
        if ray_count is not None:
            genbsdf_cmd += ["-c", str(ray_count)]
        genbsdf_cmd += ["-geom", "meter", "-dim"]
        genbsdf_cmd += ["-0.025", "0.025", "-0.012", str(-0.012+spacing)]
        genbsdf_cmd += [str(-thickness), "0", inp]
        logger.info(genbsdf_cmd)
        genbsdf_out = sp.run(genbsdf_cmd, check=True, stdout=sp.PIPE).stdout.decode()
        with open(opath, 'w') as wtr:
            wtr.write(genbsdf_out)


def gen_blinds_bsdf(args: argparse.Namespace) -> None:
    """Using genblinds and genBSDF to generate BSDF."""
    logger = setup_logger(args.verbose)
    logger.info("Creating blinds BSDF based on a specific window.")
    generic_width = 10
    generic_nslats = 10
    generic_name = "generic_blinds"
    generic_material_name = "generic_material"
    depth, spacing, angle = args.geom
    trans, refl, spec, rough = args.material
    blind_thickness = depth * cos(radians(angle))
    thickness = blind_thickness + args.gap
    height = generic_nslats * spacing
    window_z = 0 if args.ext else -thickness
    genblinds_cmd = get_genblinds_command(
        generic_name, generic_material_name, depth,
        generic_width, height, generic_nslats, angle,
        gap=args.gap, curve=args.slatcurve, ext=args.ext)
    logger.info(genblinds_cmd)
    blinds_material_primitive = gen_blind_material(
        generic_material_name, trans, refl, spec, rough)
    window_material_primitive = None
    if args.window_material is not None:
        window_material_primitives = radutil.unpack_primitives(
            args.window_material)
        # use only the first one
        window_material_primitive = window_material_primitives[0]
    gen_generic_blinds_bsdf(
        genblinds_cmd, blinds_material_primitive, args.outpath,
        thickness, spacing, logger,
        wm_prim=window_material_primitive, window_z=window_z,
        ray_count=args.c, opt=args.opt)


def blinds_proxy(args):
    generic_name = "generic_blinds"
    generic_material_name = "slat_material"
    depth, spacing, angle = args.blinds
    trans, refl, spec, rough = args.material
    blind_thickness = depth * cos(radians(angle))
    thickness = blind_thickness + args.gap
    blinds_material_primitive = gen_blind_material(
        generic_material_name, trans, refl, spec, rough)
    window_primitives = radutil.unpack_primitives(args.window)
    window_polygons = [radutil.parse_polygon(prim.real_arg)
                       for prim in window_primitives
                       if prim.ptype == "polygon"]
    # only use the first polygon
    window_polygon = window_polygons[0]
    window_normal = window_polygon.normal()
    if round(window_normal.z, 1) != 0:
        raise Exception("Can only analyze vertical polygons")
    window_vertices = window_polygon.vertices
    if len(window_vertices) != 4:
        raise Exception("4-sided polygon only")
    window_center = window_polygon.centroid()
    window_width, window_height = get_polygon_width_height(window_vertices)

    genblinds_cmd = get_genblinds_command(
        generic_name, generic_material_name, depth,
        window_width, window_height, window_height/spacing,
        angle, curve=args.slatcurve, gb_xform=False)
    blind_center = radgeom.Vector(blind_thickness / 2,
                                  window_width / 2,
                                  window_height / 2)
    blind_normal = radgeom.Vector(1, 0, 0)
    if args.ext:
        to_position = window_center - window_normal.scale(thickness/2)
    else:
        to_position = window_center + window_normal.scale(thickness/2)
    rot_angle = window_normal.angle_from(blind_normal)
    pos_z = radgeom.Vector(0, 0, 1)
    blind_center = blind_center.rotate_3d(pos_z, rot_angle)
    translate = to_position - blind_center
    genblinds_cmd += f"| xform -rz {-degrees(rot_angle)} "
    genblinds_cmd += f"-t {translate.x} {translate.y} {translate.z} "
    bsdf_prim = radutil.Primitive("void", "BSDF", "bsdfmaterial",
                                  f"6 {thickness} {args.xml} 0 0 1 .",
                                  "0")
    if args.ext:
        bsdf_polygon = window_polygon
    else:
        bsdf_polygon = window_polygon.move(window_normal.scale(thickness))
    bsdf_polygon_prim = radutil.polygon2prim(bsdf_polygon, "bsdfmaterial", "bsdf_polygon")
    with open(args.outpath, "w") as wtr:
        wtr.write(str(bsdf_prim))
        wtr.write(str(bsdf_polygon_prim))
        wtr.write(str(blinds_material_primitive))
        wtr.write(genblinds_cmd)


def get_parser():
    """Get commandline argument parser."""
    parser = argparse.ArgumentParser(prog="geombsdf")
    subparser = parser.add_subparsers()
    parser_blinds = subparser.add_parser("blinds")
    parser_blinds.set_defaults(func=gen_blinds_bsdf)
    parser_blinds.add_argument(
        'geom', nargs=3, type=float, metavar=('depth', 'spacing', 'angle'))
    parser_blinds.add_argument(
        'material', nargs=4, type=float,
        metavar=('trans', 'refl', 'spec', 'rough'))
    parser_blinds.add_argument('outpath', type=Path)
    parser_blinds.add_argument('-s', '--slatcurve', default="")
    parser_blinds.add_argument('-g', '--gap', type=float, default=0.0)
    parser_blinds.add_argument('-e', '--ext', action='store_true')
    # parser_blinds.add_argument('-w', '--window', type=Path)
    parser_blinds.add_argument('-m', '--window_material', type=Path)
    parser_proxy = subparser.add_parser("proxy")
    parser_proxy.set_defaults(func=blinds_proxy)
    parser_proxy.add_argument("xml")
    parser_proxy.add_argument("window")
    parser_proxy.add_argument("blinds", nargs=3, type=float)
    parser_proxy.add_argument("material", nargs=4, type=float)
    parser_proxy.add_argument('outpath', type=Path)
    parser_proxy.add_argument('-g', '--gap', type=float, default=0.0)
    parser_proxy.add_argument('-s', '--slatcurve', default="")
    parser_proxy.add_argument('-e', '--ext', action='store_true')
    parser_custom = subparser.add_parser("custom")
    parser_custom.add_argument('custom', nargs=3)
    parser_section = subparser.add_parser("section")
    parser_section.add_argument('section', nargs=2)
    parser.add_argument('-c', type=int, default=2000)
    parser.add_argument('-opt', help='Simulation parameters')
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Verbose mode: \n"
        "\t-v=Debug\n"
        "\t-vv=Info\n"
        "\t-vvv=Warning\n"
        "\t-vvvv=Error\n"
        "\t-vvvvv=Critical\n"
        "default=Warning")
    return parser


def main():
    """Generate a BSDF for macroscopic systems."""
    parser = get_parser()
    args = parser.parse_args()
    # logger = setup_logger(args.verbose)
    args.func(args)
    # if args.m[1] == 0:
    #     mat_prim = radutil.neutral_plastic_prim('void', 'blindmaterial', *args.m[1:])
    # else:
    #     mat_prim = radutil.neutral_trans_prim("void", "blindmaterial", *args.m)
    # mat_prim_str = str(mat_prim)
    # if args.custom:
    #     # Custom periodic geometry
    #     pass
    # elif args.section:
    #     section_primitives = radutil.unpack_primitives(args.section)
    #     if len(section_primitives) > 1:
    #         logger.warning(
    #             "Expecting only one polygon primitive, taking the first one.")
    #     section_polygon = radutil.parse_polygon(section_primitives[0].real_arg)
    #     if max([vert.z for vert in section_polygon.vertices]) > 0:
    #         logger.warning(
    #             "Polygon not on x-y plane")
    #     section_area = section_polygon.area()
    #     extrude_vector = radgeom.Vector(0, 0, section_area * 100)
    #     extrusion = section_polygon.extrude(extrude_vector)
    #     # Write to a temp file
    #     # !xform to arrays and rotate to -Z space
    #     # Call genBSDF

    # elif args.window:
    #     logger.info("Creating blinds BSDF based on a specific window.")
    #     window_primitives = radutil.unpack_primitives(args.window)
    #     window_polygons = [radutil.parse_polygon(prim.real_arg)
    #                        for prim in window_primitives
    #                        if prim.ptype == "polygon"]
    #     # Only take the first polygon
    #     window_polygon = window_polygons[0]
    #     window = window_primitives[0]  # only take the first window primitive
    #     env_primitives = [radutil.unpack_primitives(env) for env in args.env]
    #     env_primitives.append(window_primitives)
    #     env_identifier = [prim.identifier
    #                       for prims in env_primitives for prim in prims]
    #     env_paths = [str(path.resolve()) for path in args.env]
    #     depth, spacing, angle = args.blinds
    #     movedown = depth * math.cos(math.radians(float(angle)))
    #     window_move_negz = 0 if args.ext else movedown + args.gap
    #     height, width, angle2negY, translate = radutil.analyze_vert_polygon(
    #         window_polygons[0], window_move_negz)
    #     xform_window = f"!xform -rz {math.degrees(angle2negY)} "
    #     xform_window += f"-rx -90 -t {translate.x} {translate.y} "
    #     xform_window += f"{translate.z} {args.window.resolve()}\n"
    #     xform_env = ""
    #     if args.env != []:
    #         xform_env += f"!xform -rz {math.degrees(angle2negY)} "
    #         xform_env += f"-rx -90 -t {translate.x} {translate.y} "
    #         xform_env += f"{translate.z} {' '.join(env_paths)}\n"
    #     xform_cmd = xform_env + xform_window
    #     logger.info(xform_cmd)
    #     slat_cmd = radutil.gen_blinds(depth, width, height, spacing,
    #                                   angle, args.curve, movedown)
    #     logger.info(slat_cmd)
    #     lower_bound = max(movedown, window_move_negz)
    #     with tf.TemporaryDirectory() as td:
    #         inp = os.path.join(td, "blinds.rad")
    #         with open(inp, 'w') as wtr:
    #             wtr.write(mat_prim_str)
    #             wtr.write(xform_cmd)
    #             wtr.write(slat_cmd)
    #         cmd = ["genBSDF", "-n", "4", "-f", "+b",]
    #         if args.opt is not None: cmd += ["-r", args.opt]
    #         cmd += ["-c", str(args.c), "+geom", "meter", "-dim"]
    #         # cmd += [str(-width/2), str(width/2), str(-height/2), str(height/2)]
    #         cmd += ["-0.025", "0.025", "-0.012", str(-0.012+spacing)]
    #         cmd += [str(-lower_bound), "0", inp]
    #         logger.info(cmd)
    #         breakpoint()
    #         _stdout = sp.run(cmd, check=True, stdout=sp.PIPE).stdout.decode()
    #         xml_name = "{}_blinds_{}_{}_{}.xml".format(
    #             window.identifier, depth, spacing, angle)
    #         with open(xml_name, 'w') as wtr:
    #             wtr.write(_stdout)
    #     # move back
    #     pkgbsdf_cmd = ["pkgBSDF", "-s", xml_name]
    #     xform_back_cmd = ["xform", "-t", str(-translate.x),
    #                       str(-translate.y), str(-translate.z)]
    #     xform_back_cmd += ["-rx", "90", "-rz", str(-math.degrees(angle2negY))]
    #     ps1 = sp.Popen(pkgbsdf_cmd, stdout=sp.PIPE)
    #     ps2 = sp.Popen(xform_back_cmd, stdin=ps1.stdout, stdout=sp.PIPE)
    #     if ps1.stdout is not None:
    #         ps1.stdout.close()
    #     _stdout, _ = ps2.communicate()
    #     result_primitives = radutil.parse_primitive(_stdout.decode().splitlines())
    #     result_primitives = [prim for prim in result_primitives
    #                          if prim.identifier not in env_identifier]
    #     with open(args.o, 'w') as wtr:
    #         [wtr.write(str(prim)) for prim in result_primitives]
    # else:
    #     width = 10  # default blinds width
    #     # height = 0.096  # default blinds height
    #     nslats = 10
    #     depth, spacing, angle = args.blinds
    #     height = nslats * spacing
    #     if args.ext:
    #         glass_z = 0
    #         movedown = args.gap + depth * math.cos(math.radians(angle))
    #     else:
    #         glass_z = args.gap + depth * math.cos(math.radians(angle))
    #         movedown = depth * math.cos(math.radians(angle))
    #     lower_bound = min(-glass_z, -movedown)
    #     genblinds_cmd = radutil.gen_blinds(depth, width, height, spacing,
    #                                        angle, args.curve, movedown)
    #     # pt1 = radgeom.Vector(-width / 2, height / 2, -glass_z)
    #     # pt2 = radgeom.Vector(-width / 2, -height / 2, -glass_z)
    #     # pt3 = radgeom.Vector(width / 2, -height / 2, -glass_z)
    #     # tmis = util.tmit2tmis(.38)
    #     # glass_prim = radutil.glass_prim('void', 'glass1', tmis, tmis, tmis)
    #     # glazing_polygon = radgeom.Polygon.rectangle3pts(pt1, pt2, pt3)
    #     # glazing_prim_str = str(glass_prim)
    #     # glazing_prim_str += str(radutil.polygon2prim(
    #         # glazing_polygon, 'glass1', 'window'))
    #     with open("tmp_blinds.rad", 'w') as wtr:
    #         wtr.write(mat_prim_str)
    #         # wtr.write(glazing_prim_str)
    #         wtr.write(genblinds_cmd)
    #     genbsdf_cmd = "genBSDF -n 4 +f +b -geom meter -dim "
    #     genbsdf_cmd += f"-0.025 0.025 -0.012 {-0.012+spacing} {lower_bound} 0 tmp_blinds.rad"
    #     logger.info(genbsdf_cmd)
    #     genbsdf_process = sp.run(genbsdf_cmd.split(), check=True, stdout=sp.PIPE)
    #     xml_name = "blinds_{}_{}_{}.xml".format(depth, spacing, angle)
    #     if genbsdf_process.stdout is not None:
    #         with open(xml_name, 'wb') as wtr:
    #             wtr.write(genbsdf_process.stdout)
    #     else:
    #         raise Exception("genBSDF run failed.")
    # # os.remove("tmp_blinds.rad")
