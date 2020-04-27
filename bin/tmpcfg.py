#!/usr/bin/env python3

from frads import radutil
import configparser
import os


def setup_dir():
    dirs = ['Matrices', 'Objects',
            'Octrees', 'Resources', 'Results']
    [radutil.mkdir_p(i) for i in dirs]


def setup_cfg():
    config = configparser.RawConfigParser(allow_no_value=True)

    config['SimulationControl'] = {
        'vmx_opt': 'kf -ab 5 -ad 16000 -lw 1e-8',
        'fmx_opt': 'kf -ab 2',
        'dmx_opt': 'r4 -ab 2 -ad 1000 -lw 1e-4 -c 15000',
        'dsmx_opt': 'r4 -ab 5 -ad 32000 -lw 1e-10',
        'cdsmx_opt': 'r6 -ab 1',
        'ray_count': 5,
        'pixel_jitter': .7,
        'separate_direct': False,
        'nprocess': 1,
    }

    config['FileStructure'] = {
        'base': f"{os.path.abspath(os.getcwd())} # all directories fall under this one",
        'matrices': 'Matrices',
        'results': 'Results',
        'objects': 'Objects # where are the scene object files',
        'resources': 'Resources',
    }

    config['Site'] = {
        'wea': "# wea file path, assuming to be in the resources directory",
        'latitude': None,
        'longitude': None,
        'zipcode': "# US only",
        'daylight_hours_only': False,
        'start_hour': None,
        'end_hour': None,
    }

    config['Dimensions'] = {
        'depth': "# if all values filled in this section, a standard room will be created",
        'width': None,
        'height': None,
        'window1': "# format: x1, z1, width, heigh",
        'facade_thickness': "# window wall thickness",
        'orientation': "# south or west or ...",
    }

    config['Model'] = {
        'material': 'material.rad',
        'windows': None,
        'scene': None,
        'ncp_shade': None,
        'BSDF': None,
        'sunBSDF': "# CFS (Macroscopic systems) in addition to ones defined in windows",

    }

    config['Raysenders'] = {
        'view1': '-vf south.vf # view options, use -vf to specify view path',
        'view2': '# view options, use -vf to specify view path',
        'grid_surface': 'floor.rad # generate sensor grid from these surfaces',
        'distance': '.8 # sensor grid to be offseted in the surface normal direction',
        'spacing': '.6 # sensor grid spacing',
        'opposite': 'True # reverse the surface normal for grid generation',
    }

    with open('template.cfg', 'w') as cfg:
        config.write(cfg)

if __name__ == "__main__":
    setup_dir()
    setup_cfg()
