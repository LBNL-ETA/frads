"""EnergyPlus Radiance runtime interaction."""

import argparse
import configparser
import json
import logging
import os
import subprocess as sp
import sys
from frads import epjson2rad, util, radutil
from frads import mtxmethod as mm
### Import EnergyPlus Python Library ###
srcloc = {'win32': 'C:\\', 'darwin': '/Applications', 'linux': '/usr/local'}
dname = [os.path.join(srcloc[sys.platform], d)
         for d in os.listdir(srcloc[sys.platform]) if d.startswith('EnergyPlus')]
ephome = input('Where is EnergyPlus installed?') if len(
    dname) == 0 else dname[0]
if ephome not in sys.path:
    sys.path.append(ephome)
from pyenergyplus.api import EnergyPlusAPI
########################################

one_time = True

# Sensor grid dimension in meters
GRID_HEIGHT = 0.75
GRID_SPACING = 0.6


def get_sky_vector(mo, da, hr, mi, lat, lon, dni, dhi):
    hrs = hr + mi/60
    cmd1 = f"gendaylit {mo} {da} {hrs} -a {lat} -o {lon*-1} -W {dni} {dhi} "
    res1 = sp.run(cmd1.split(), check=True, capture_output=True).stdout
    cmd2 = "genskyvec -m 4"
    res2 = sp.run(cmd2.split(), input=res1,
                  capture_output=True).stdout.decode()
    return radutil.smx2nparray(res2)


def read_epjs(fpath):
    with open(fpath) as rdr:
        epjs = json.load(rdr)
    return epjs


def init_radiance(epjs_path, overwrite=False, nproc=1, run=False):
    global cfg, mtxmtd, pdsmx
    epjs = read_epjs(epjs_path)
    radobj = epjson2rad.epJSON2Rad(epjs)
    cwd = os.path.dirname(os.path.abspath(epjs_path))
    for zn in radobj.zones:
        site = radobj.site
        zone = radobj.zones[zn]
        util.mkdir_p(zn)
        objdir = 'Objects'
        rsodir = 'Resources'
        mtxdir = 'Matrices'
        resdir = 'Results'
        radutil.mkdir_p(objdir)
        radutil.mkdir_p(rsodir)
        radutil.mkdir_p(mtxdir)
        radutil.mkdir_p(resdir)
        with open(os.path.join(objdir, 'materials.rad'), 'w') as wtr:
            [wtr.write(radutil.put_primitive(val))
             for key, val in radobj.mat_prims.items()]
        scene_paths = []
        window_paths = []
        for st in zone:
            if st == 'Window':
                for key, val in zone['Window'].items():
                    _path = os.path.join(objdir, f"Window_{key}.rad")
                    window_paths.append(f"Window_{key}.rad")
                    with open(_path, 'w') as wtr:
                        wtr.write(radutil.put_primitive(val))
            else:
                _path = os.path.join(objdir, f"{st}.rad")
                scene_paths.append(f"{st}.rad")
                with open(_path, 'w') as wtr:
                    [wtr.write(radutil.put_primitive(val))
                     for key, val in zone[st].items()]
        cfg = mm.cfg_template
        cfg['nprocess'] = nproc
        cfg['overwrite'] = overwrite
        cfg['ray_count'] = 1
        cpath = os.path.join(cwd, zn)
        cfg['base'] = cpath
        cfg['latitude'] = site['latitude']
        cfg['longitude'] = site['longitude']
        cfg['material'] = 'materials.rad'
        cfg['windows'] = ' '.join(window_paths)
        cfg['scene'] = ' '.join(scene_paths)
        cfg['grid_surface'] = 'Floor.rad'
        cfg['grid_height'] = GRID_HEIGHT
        cfg['grid_spacing'] = GRID_SPACING
        cfg['objects'] = objdir
        cfg['results'] = resdir
        cfg['matrices'] = mtxdir
        cfg['resources'] = rsodir
        if radutiln:
            mtxmtd = mm.MTXMethod(cfg)
            pdsmx_path = mtxmtd.prep_2phase_pt()
            with open(pdsmx_path) as rdr:
                pdsmx = ru.mtx2nparray(rdr.read())
        else:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read_dict(cfg)
            with open(os.path.join(cpath, 'run.cfg'), 'w') as wtr:
                config.write(wtr)


def time_step_handler():
    global one_time, outdoor_dni_sensor, outdoor_dhi_sensor
    global west_zone_power_level, west_zone_light_design_level
    global north_zone_wall5_insol
    if one_time:
        if not api.exchange.api_data_fully_ready(state):
            return
        val = api.exchange.list_available_api_data_csv()
        with open('variable.csv', 'wb') as f:
            f.write(val)
        outdoor_dni_sensor = api.exchange.get_variable_handle(
            u"SITE DIRECT SOLAR RADIATION RATE PER AREA", u"Environment"
        )
        outdoor_dhi_sensor = api.exchange.get_variable_handle(
            u"SITE DIFFUSE SOLAR RADIATION RATE PER AREA", u"Environment"
        )
        west_zone_light_design_level = api.exchange.get_internal_variable_handle(
            u"Lighting Power Design Level", u"WEST ZONE LIGHTS 1"
        )
        west_zone_power_level = api.exchange.get_actuator_handle(
            "Lights", "Electric Power Level", "WEST ZONE LIGHTS 1"
        )
        north_zone_wall5_insol = api.exchange.get_actuator_handle(
            "Schedule:Compact", "Schedule Value", "INSOL WALL"
        )
        if outdoor_dhi_sensor == -1:
            sys.exit(1)
        one_time = False

    month = api.exchange.month()
    day = api.exchange.day_of_month()
    hour = api.exchange.hour()
    minute = api.exchange.minutes()
    dni = api.exchange.get_variable_value(outdoor_dni_sensor)
    dhi = api.exchange.get_variable_value(outdoor_dhi_sensor)
    if int(float(dni)) == 0 and int(float(dhi)) == 0:
        illum_val = 0
    else:
        skv = get_sky_vector(month, day, hour, minute,
                             cfg['latitude'],
                             cfg['longitude'],
                             dni, dhi)
        illum_val = radutil.mtxmult([pdsmx, skv]).mean()*179
    print(f"{month}/{day} {hour}:{minute} illum value is : {illum_val:.0f} lx")
    power_level = (1 - min(5000, illum_val)/5000) * \
        api.exchange.get_internal_variable_value(west_zone_light_design_level)
    print(f"\tlighting power: {power_level:.0f} W")
    api.exchange.set_actuator_value(north_zone_wall5_insol, 100)
    # api.exchange.reset_actuator(north_zone_wall5_insol)
    insol_wall = api.exchange.get_actuator_value(north_zone_wall5_insol)
    print("insol wall: ",  insol_wall)
    api.exchange.set_actuator_value(west_zone_power_level, power_level)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ipath')
    parser.add_argument('-w', required=True)
    parser.add_argument('-n', type=int)
    parser.add_argument('-f', action='store_true', default=False)
    args = parser.parse_args()
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    if args.ipath.endswith('.idf'):
        raise RuntimeError('.idf not supported')
    logger = logging.getLogger('frads')
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    init_radiance(args.ipath, run=True, nproc=args.n, overwrite=args.f)
    # api.runtime.callback_end_zone_timestep_after_zone_reporting(
    # time_step_handler)
    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(
        state, time_step_handler)
    #api.exchange.request_variable("SITE DIRECT SOLAR RADIATION RATE PER AREA", "ENVIRONMENT")
    #api.exchange.request_variable("SITE DIFFUSE SOLAR RADIATION RATE PER AREA", "ENVIRONMENT")
    eplus_cmd = ['-w', args.w, args.ipath]
    api.runtime.run_energyplus(eplus_cmd)
