#!/usr/bin/env python3

from frads import radutil
import configparser
import os


def setup_dir():
    dirs = ['Matrices', 'Models', 'Objects',
            'Octrees', 'Resources', 'Results', 'Raysenders']
    [radutil.mkdir_p(i) for i in dirs]


def setup_cfg():
    config = configparser.RawConfigParser(allow_no_value=True)

    config.add_section('SimulationControl')
    config.set('SimulationControl', 'vmx_opt', 'kf -ab 8 -ad 65000 -lw 1e-8')
    config.set('SimulationControl', 'fmx_opt', 'kf -ab 5 -c 500 -lw 1e-4')
    config.set('SimulationControl', 'dmx_opt', 'r4 -ab 2 -c 5000')
    config.set('SimulationControl', 'dsmx_opt', 'r4 -ab 2 -c 5000')
    config.set('SimulationControl', 'view_ray_cnt', 1)
    config.set('SimulationControl', 'pixel_jitter', .7)
    config.set('SimulationControl', 'separate_direct', False)
    config.set('SimulationControl', 'nprocess', 1)

    config.add_section('FileStructure')
    config.set('FileStructure', 'base_dir', os.path.abspath(os.getcwd()))
    config.set('FileStructure', 'matrices', 'Matrices')
    config.set('FileStructure', 'results', 'Results')
    config.set('FileStructure', 'objects', 'Objects')
    config.set('FileStructure', 'raysenders', 'Raysenders')
    config.set('FileStructure', 'resources', 'Resources')

    config.add_section('Site')
    config.set('Site', 'wea')
    config.set('Site', 'lat')
    config.set('Site', 'lon')
    config.set('Site', 'zipcode')

    config.add_section('Dimensions')
    config.set('Dimensions','depth')
    config.set('Dimensions','width')
    config.set('Dimensions','height')
    config.set('Dimensions','window')
    config.set('Dimensions','facade_thickness')

    config.add_section('Model')
    config.set('Model', 'material', 'material.rad')
    config.set('Model', 'windows')
    config.set('Model', 'scene')
    config.set('Model', 'ncp_shade')
    config.set('Model', 'BSDF')

    config.add_section('Raysenders')
    config.set('Raysenders', 'views', 'south.vf')
    config.set('Raysenders', 'surfaces', 'floor.rad')
    config.set('Raysenders', 'distances', .8)
    config.set('Raysenders', 'spacings', .6)
    config.set('Raysenders', 'opposite', True)

    with open('template.cfg', 'w') as cfg:
        config.write(cfg)

if __name__ == "__main__":
    setup_dir()
    setup_cfg()
