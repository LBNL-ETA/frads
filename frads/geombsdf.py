"""Call genBSDF to generate BSDF."""

import argparse
import math
import logging
import os
import subprocess as sp
from frads import radutil, radgeom, util


"""
TODO:
    1. New feature for custom input period geometry
    2. New feature for custom section drawing
"""


logger = logging.getLogger("frads")


def get_parser():
    """Get commandline argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-blinds', nargs=3, type=float, metavar=('depth', 'spacing', 'angle'))
    parser.add_argument('-curve', default='')
    parser.add_argument('-custom', nargs=3)
    parser.add_argument('-section', nargs=2)
    parser.add_argument('-gap', type=float, default=0.0)
    parser.add_argument('-ext', action='store_true')
    parser.add_argument('-m', nargs=3, type=float, required=True, metavar=('refl', 'spec', 'rough'))
    parser.add_argument('-g', type=float)
    parser.add_argument('-window')
    parser.add_argument('-env')
    parser.add_argument('-o', default='default_blinds.rad')
    return parser


def main():
    """Generate a BSDF for macroscopic systems."""
    parser = get_parser()
    args = parser.parse_args()
    mat_prim = radutil.neutral_plastic_prim('void', 'blindmaterial', *args.m)
    mat_prim_str = str(mat_prim)
    if args.custom:
        # Custom periodic geometry
        pass
    elif args.section:
        # Custom section drawing
        pass
    elif args.window:
        primitves = radutil.unpack_primitives(args.window)
        env_primitives = radutil.unpack_primitives(args.env)
        env_identifier = [prim.identifier for prim in env_primitives]
        windows = [p for p in primitves if p.identifier.startswith('window')]
        window = windows[0] # only take the first window primitive
        depth, spacing, angle = args.blinds
        movedown = depth * math.cos(math.radians(float(angle)))
        window_movedown = 0 if args.ext else movedown + args.gap
        # window_polygon = radutil.parse_polygon(window.real_args)
        height, width, angle2negY, translate = radutil.analyze_window(window, window_movedown)
        xform_cmd = f'!xform -rz {math.degrees(angle2negY)} -rx -90 -t {translate.x} {translate.y} {translate.z} {args.env}\n'
        xform_cmd += f'!xform -rz {math.degrees(angle2negY)} -rx -90 -t {translate.x} {translate.y} {translate.z} {args.window}\n'
        logger.info(xform_cmd)
        slat_cmd = radutil.gen_blinds(depth, width, height, spacing, angle, args.curve, movedown)
        logger.info(slat_cmd)
        lower_bound = max(movedown, window_movedown)
        with open("tmp_blinds.rad", 'w') as wtr:
            wtr.write(mat_prim_str)
            wtr.write(xform_cmd)
            wtr.write(slat_cmd)
        cmd = ["genBSDF", "-n", "4", "-f", "+b", "-c", "500", "+geom", "meter", "-dim"]
        cmd += [str(-width/2), str(width/2), str(-height/2), str(height/2)]
        cmd += [str(-lower_bound), "0", "tmp_blinds.rad"]
        logger.info(cmd)
        _stdout = sp.run(cmd, check=True, stdout=sp.PIPE).stdout.decode()
        xml_name = "{}_blinds_{}_{}_{}.xml".format(window.identifier, depth, spacing, angle)
        with open(xml_name, 'w') as wtr:
            wtr.write(_stdout)
        #move back
        pkgbsdf_cmd = ["pkgBSDF", "-s", xml_name]
        xform_back_cmd = ["xform", "-t", str(-translate.x), str(-translate.y), str(-translate.z)]
        xform_back_cmd += ["-rx", "90", "-rz", str(-math.degrees(angle2negY))]
        ps1 = sp.Popen(pkgbsdf_cmd, stdout=sp.PIPE)
        ps2 = sp.Popen(xform_back_cmd, stdin=ps1.stdout, stdout=sp.PIPE)
        ps1.stdout.close()
        _stdout, _ = ps2.communicate()
        result_primitives = radutil.parse_primitive(_stdout.decode().splitlines())
        result_primitives = [prim for prim in result_primitives if prim['identifier'] not in env_identifier]
        with open(args.o, 'w') as wtr:
            [wtr.write(str(prim)) for prim in result_primitives]
    else:
        width = 10 # default blinds width
        height = 0.096 # default blinds height
        depth, spacing, angle = args.blinds
        if args.ext:
            glass_z = 0
            movedown = args.gap + depth * math.cos(math.radians(angle))
        else:
            glass_z = args.gap + depth * math.cos(math.radians(angle))
            movedown = depth * math.cos(math.radians(angle))
        lower_bound = min(-glass_z, -movedown)
        genblinds_cmd = radutil.gen_blinds(depth, width, height, spacing, angle, args.curve, movedown)
        pt1 = radgeom.Vector(-width/2, height/2, -glass_z)
        pt2 = radgeom.Vector(-width/2, -height/2, -glass_z)
        pt3 = radgeom.Vector(width/2, -height/2, -glass_z)
        tmis = util.tmit2tmis(.38)
        glass_prim = radutil.glass_prim('void', 'glass1', tmis, tmis, tmis)
        glazing_polygon = radgeom.Polygon.rectangle3pts(pt1, pt2, pt3)
        glazing_prim_str = str(glass_prim)
        glazing_prim_str += str(radutil.polygon2prim(glazing_polygon, 'glass1', 'window'))
        with open("tmp_blinds.rad", 'w') as wtr:
            wtr.write(mat_prim_str)
            wtr.write(glazing_prim_str)
            wtr.write(genblinds_cmd)
        cmd = f"genBSDF -n 4 +f +b -c 500 -geom meter -dim -0.025 0.025 -0.012 0.012 {lower_bound} 0 tmp_blinds.rad"
        _stdout = sp.run(cmd.split(), check=True, stdout=sp.PIPE).stdout.decode()
        xml_name = "blinds_{}_{}_{}.xml".format(depth, spacing, angle)
        with open(xml_name, 'w') as wtr:
            wtr.write(_stdout)
    os.remove("tmp_blinds.rad")
