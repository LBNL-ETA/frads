"""
Call genBSDF to generate BSDF.

TODO:
    1. New feature for custom input period geometry
    2. New feature for custom section drawing
"""

import argparse
import logging
import math
import subprocess as sp

from frads import radgeom, radutil, util


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


def get_parser():
    """Get commandline argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-blinds', nargs=3, type=float,
                        metavar=('depth', 'spacing', 'angle'))
    parser.add_argument('-curve', default='')
    parser.add_argument('-custom', nargs=3)
    parser.add_argument('-section', nargs=2)
    parser.add_argument('-gap', type=float, default=0.0)
    parser.add_argument('-ext', action='store_true')
    parser.add_argument('-m', nargs=4, type=float, required=True,
                        metavar=('t', 'refl', 'spec', 'rough'))
    parser.add_argument('-c', type=int, default=2000)
    parser.add_argument('-opt', type=str, default='-ab 1', help='Simulation parameters')
    parser.add_argument('-window')
    parser.add_argument('-env', nargs="+")
    parser.add_argument('-o', default='default_blinds.rad')
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
    logger = setup_logger(args.verbose)
    if args.m[1] == 0:
        mat_prim = radutil.neutral_plastic_prim('void', 'blindmaterial', *args.m[1:])
    else:
        mat_prim = radutil.neutral_trans_prim("void", "blindmaterial", *args.m)
    mat_prim_str = str(mat_prim)
    if args.custom:
        # Custom periodic geometry
        pass
    elif args.section:
        section_primitives = radutil.unpack_primitives(args.section)
        if len(section_primitives) > 1:
            logger.warning(
                "Expecting only one polygon primitive, taking the first one.")
        section_polygon = radutil.parse_polygon(section_primitives[0].real_arg)
        if max([vert.z for vert in section_polygon.vertices]) > 0:
            logger.warning(
                "Polygon not on x-y plane")
        section_area = section_polygon.area()
        extrude_vector = radgeom.Vector(0, 0, section_area * 100)
        extrusion = section_polygon.extrude(extrude_vector)
        # Write to a temp file
        # !xform to arrays and rotate to -Z space
        # Call genBSDF

    elif args.window:
        logger.info("Creating blinds BSDF based on a specific window.")
        window_primitives = radutil.unpack_primitives(args.window)
        window_polygons = [radutil.parse_polygon(prim.real_arg)
                           for prim in window_primitives
                           if prim.ptype == "polygon"]
        window = window_primitives[0]  # only take the first window primitive
        env_primitives = [radutil.unpack_primitives(env) for env in args.env]
        env_primitives.extend(window_primitives)
        env_identifier = [prim.identifier
                          for prims in env_primitives for prim in prims]
        depth, spacing, angle = args.blinds
        movedown = depth * math.cos(math.radians(float(angle)))
        window_move_negz = 0 if args.ext else movedown + args.gap
        height, width, angle2negY, translate = radutil.analyze_vert_polygon(
            window_polygons[0], window_move_negz)
        xform_cmd = f"!xform -rz {math.degrees(angle2negY)} "
        xform_cmd += f"-rx -90 -t {translate.x} {translate.y} "
        xform_cmd += f"{translate.z} {' '.join(args.env)}\n"
        xform_cmd += f"!xform -rz {math.degrees(angle2negY)} "
        xform_cmd += f"-rx -90 -t {translate.x} {translate.y} "
        xform_cmd += f"{translate.z} {args.window}\n"
        logger.info(xform_cmd)
        slat_cmd = radutil.gen_blinds(depth, width, height, spacing,
                                      angle, args.curve, movedown)
        logger.info(slat_cmd)
        lower_bound = max(movedown, window_move_negz)
        with open("tmp_blinds.rad", 'w') as wtr:
            wtr.write(mat_prim_str)
            wtr.write(xform_cmd)
            wtr.write(slat_cmd)
        cmd = ["genBSDF", "-n", "4", "-f", "+b",]
        cmd += ["-r", args.opt]
        cmd += ["-c", args.c, "+geom", "meter", "-dim"]
        cmd += [str(-width / 2), str(width / 2), str(-height / 2), str(height / 2)]
        cmd += [str(-lower_bound), "0", "tmp_blinds.rad"]
        logger.info(cmd)
        _stdout = sp.run(cmd, check=True, stdout=sp.PIPE).stdout.decode()
        xml_name = "{}_blinds_{}_{}_{}.xml".format(
            window.identifier, depth, spacing, angle)
        with open(xml_name, 'w') as wtr:
            wtr.write(_stdout)
        # move back
        pkgbsdf_cmd = ["pkgBSDF", "-s", xml_name]
        xform_back_cmd = ["xform", "-t", str(-translate.x),
                          str(-translate.y), str(-translate.z)]
        xform_back_cmd += ["-rx", "90", "-rz", str(-math.degrees(angle2negY))]
        ps1 = sp.Popen(pkgbsdf_cmd, stdout=sp.PIPE)
        ps2 = sp.Popen(xform_back_cmd, stdin=ps1.stdout, stdout=sp.PIPE)
        if ps1.stdout is not None:
            ps1.stdout.close()
        _stdout, _ = ps2.communicate()
        result_primitives = radutil.parse_primitive(_stdout.decode().splitlines())
        result_primitives = [prim for prim in result_primitives
                             if prim.identifier not in env_identifier]
        with open(args.o, 'w') as wtr:
            [wtr.write(str(prim)) for prim in result_primitives]
    else:
        width = 10  # default blinds width
        # height = 0.096  # default blinds height
        nslats = 10
        depth, spacing, angle = args.blinds
        height = nslats * spacing
        if args.ext:
            glass_z = 0
            movedown = args.gap + depth * math.cos(math.radians(angle))
        else:
            glass_z = args.gap + depth * math.cos(math.radians(angle))
            movedown = depth * math.cos(math.radians(angle))
        lower_bound = min(-glass_z, -movedown)
        genblinds_cmd = radutil.gen_blinds(depth, width, height, spacing,
                                           angle, args.curve, movedown)
        # pt1 = radgeom.Vector(-width / 2, height / 2, -glass_z)
        # pt2 = radgeom.Vector(-width / 2, -height / 2, -glass_z)
        # pt3 = radgeom.Vector(width / 2, -height / 2, -glass_z)
        # tmis = util.tmit2tmis(.38)
        # glass_prim = radutil.glass_prim('void', 'glass1', tmis, tmis, tmis)
        # glazing_polygon = radgeom.Polygon.rectangle3pts(pt1, pt2, pt3)
        # glazing_prim_str = str(glass_prim)
        # glazing_prim_str += str(radutil.polygon2prim(
            # glazing_polygon, 'glass1', 'window'))
        with open("tmp_blinds.rad", 'w') as wtr:
            wtr.write(mat_prim_str)
            # wtr.write(glazing_prim_str)
            wtr.write(genblinds_cmd)
        genbsdf_cmd = "genBSDF -n 4 +f +b -geom meter -dim "
        genbsdf_cmd += f"-0.025 0.025 -0.012 {-0.012+spacing} {lower_bound} 0 tmp_blinds.rad"
        logger.info(genbsdf_cmd)
        genbsdf_process = sp.run(genbsdf_cmd.split(), check=True, stdout=sp.PIPE)
        xml_name = "blinds_{}_{}_{}.xml".format(depth, spacing, angle)
        if genbsdf_process.stdout is not None:
            with open(xml_name, 'wb') as wtr:
                wtr.write(genbsdf_process.stdout)
        else:
            raise Exception("genBSDF run failed.")
    # os.remove("tmp_blinds.rad")
