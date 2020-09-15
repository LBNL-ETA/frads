#!/usr/bin/env python3
import argparse
import configparser
import json
import os
import pdb
from frads import epjson2rad as eprad
from frads import radutil as ru
from frads import mtxmethod as mm


# Sensor grid dimension in meters
GRID_HEIGHT = 0.75
GRID_SPACING = 0.6

def read_epjs(fpath):
    with open(fpath) as rdr:
        epjs = json.load(rdr)
    return epjs

def main(kwargs):
    epjs = read_epjs(kwargs['fpath'])
    radobj = eprad.epJSON2Rad(epjs)
    for zn in radobj.zones:
        zone = radobj.zones[zn]
        ru.mkdir_p(zn)
        objdir = os.path.join(zn, 'Objects')
        ru.mkdir_p(objdir)
        with open(os.path.join(objdir, 'materials.rad'), 'w') as wtr:
            [wtr.write(ru.put_primitive(val)) for key, val in radobj.mat_prims.items()]
        scene_paths = []
        window_paths = []
        for stype in zone:
            if stype == 'Window':
                for key, val in zone['Window'].items():
                    _path = os.path.join(objdir, f"Window_{key}.rad")
                    window_paths.append(f"Window_{key}.rad")
                    with open(_path, 'w') as wtr:
                        wtr.write(ru.put_primitive(val))
            else:
                _path = os.path.join(objdir, f"{st}.rad")
                scene_paths.append(f"{st}.rad")
                with open(_path, 'w') as wtr:
                    [wtr.write(ru.put_primitive(val)) for key,val in zone[stype].items()]
        cfg = mm.cfg_template
        cfg['Model']['material'] = 'materials.rad'
        cfg['Model']['windows'] = ' '.join(window_paths)
        cfg['Model']['scene'] = ' '.join(scene_paths)
        cfg['Raysenders']['grid_surface'] = 'Floor.rad'
        cfg['Raysenders']['distance'] = GRID_HEIGHT
        cfg['Raysenders']['spacing'] = GRID_SPACING
        if kwargs['run']:
            mtxmtd = mm.MTXMethod(cfg)
        else:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read_dict(cfg)
            with open(os.path.join(zn, 'run.cfg'), 'w') as wtr:
                config.write(wtr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath')
    parser.add_argument('-run', action='store_true', default=False)
    args = parser.parse_args()
    main(vars(args))



