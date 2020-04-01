#!/usr/bin/env python3

from frads import radutil
import configparser
import os


def setup_dir():
    dirs = [
        'Matrices', 'Models', 'Objects', 'Octrees', 'Resources', 'Results',
        'Sources', 'Raysenders'
    ]
    [radutil.mkdir_p(i) for i in dirs]


def setup_cfg():
    config = configparser.RawConfigParser(allow_no_value=True)

    config.add_section('SimulationControl')
    config.set('SimulationControl', 'vmx_opt', '-ab 8 -ad 65000 -lw 1e-8')
    config.set('SimulationControl', 'fmx_opt', '-ab 5 -c 500 -lw 1e-4')
    config.set('SimulationControl', 'dmx_opt', '-ab 2 -c 5000')
    config.set('SimulationControl', 'view_ray_cnt', 1)
    config.set('SimulationControl', 'pixel_jitter', .7)
    config.set('SimulationControl', 'separate_direct', False)
    config.set('SimulationControl', 'nproc', 1)

    config.add_section('FileStructure')
    config.set('FileStructure', 'base_dir', os.path.abspath(os.getcwd()))
    config.set('FileStructure', 'matrices', 'Matrices')
    config.set('FileStructure', 'results', 'Results')
    config.set('FileStructure', 'objects', 'Objects')
    config.set('FileStructure', 'raysenders', 'Raysenders')
    config.set('FileStructure', 'models', 'Models')
    config.set('FileStructure', 'resources', 'Resources')

    config.add_section('Site')
    config.set('Site', 'smx', None)
    config.set('Site', 'wea')
    config.set('Site', 'epw')
    config.set('Site', 'lat')
    config.set('Site', 'lon')
    config.set('Site', 'zipcode')

    config.add_section('Dimensions')
    config.set('Dimensions','depth')
    config.set('Dimensions','width')
    config.set('Dimensions','heigh')
    config.set('Dimensions','window')

    config.add_section('Model')
    config.set('Model', 'material', 'material.rad')
    config.set('Model', 'scene', ['obj1.rad', 'obj2.rad'])

    config.add_section('Raysenders')
    config.set('Raysenders', 'views', 'south.vf')
    config.set('Raysenders', 'surfaces', 'floor.rad')
    config.set('Raysenders', 'distances', .8)
    config.set('Raysenders', 'spacings', .6)

    config.add_section('Windows')
    config.set('Windows', 'window_group1', ['window1.rad', 'window2.rad'])
    config.set('Windows', 'window_group2', ['window3.rad'])

    config.add_section('Non-coplanar Shading')
    config.set('Non-coplanar Shading', 'material1', 'fabric.rad')
    config.set('Non-coplanar Shading', 'geometry1', 'awning.rad')
    config.set('Non-coplanar Shading', 'material2', 'concrete.rad')
    config.set('Non-coplanar Shading', 'geometry2', 'overhang.rad')

    config.add_section('BSDF')
    config.set('BSDF', 'vis', ['shade.xml', 'no_shade.xml'])
    config.set('BSDF', 'sol')
    config.set('BSDF', 'shgc')
    config.set('BSDF', 'mtx')

    with open('template.cfg', 'w') as cfg:
        config.write(cfg)

if __name__ == "__main__":
    setup_dir()
    setup_cfg()
