#!/usr/bin/env python3
"""
T.Wang

"""

import argparse
import os
import subprocess as sp
import tempfile as tf
from configparser import ConfigParser
from frads import mfacade, makesky, radgeom, room
from frads import getepw, epw2wea, radutil, radmtx


class Mtxmethod(object):
    def __init__(self, config):
        self.config = config
        self.vmx_opt = config.simctrl['vmx_opt'][3:]
        self.vmx_basis = config.simctrl['vmx_opt'][:2]
        self.dmx_opt = config.simctrl['dmx_opt'][3:]
        self.dmx_basis = config.simctrl['dmx_opt'][:2]
        self.dsmx_opt = config.simctrl['dsmx_opt'][3:]
        self.dsmx_basis = config.simctrl['dsmx_opt'][:2]

    def two_phase(self):
        self.gen_dsmx()
        self.gen_smx()
        res = self.rmtxop(self.pdsmx, self.smxpath)

    def three_phase(self):
        self.gen_vmx()
        self.gen_dmx()
        self.gen_smx()

    def get_avgskv(self):
        sp.run(f"gendaymtx -m {self.mf_sky} -A {self.wea} > {avgskyv}", shell=True)

    def wndw_subdivide(self):
        pass

    def gen_smx(self, direct=False):
        mf = self.config.simctrl['dmx_opt'][1]
        self.smxpath = os.path.join(self.config.filestrct['matrices'],
                                    radutil.basename(self.config.wea_path)+'.smx')
        cmd = f"gendaymtx -m {mf} {self.config.wea_path} > {self.smxpath}"
        sp.run(cmd, shell=True)

    def gen_dmx(self, direct=False):
        self.dmxs = {}
        rcvr_sky = radmtx.Receiver.as_sky(self.dmx_basis)
        opt = self.dmx_opt
        append = ''
        if direct:
            append = '_d'
            opt += ' -ab 0'
        for wname in self.config.window_prims:
            radmtx.logger.info(f"Generating dmx for {wname}")
            self.dmxs[wname+append] = os.path.join(self.config.filestrct['matrices'],
                                                   f'dmx_{wname}{append}')
            sndr_wndw = radmtx.Sender.as_surface(
                prim_list=self.config.window_prims[wname], basis=self.vmx_basis,
                offset=None)
            radmtx.rfluxmtx(sender=sndr_wndw,receiver=rcvr_sky,
                       env=self.config.envpath, out=self.dmxs[wname+append], opt=opt)

    def gen_vmx(self, direct=False):
        opt = self.vmx_opt
        append = ''
        if direct:
            opt += ' -ab 1'
            append = '_d'
        if self.config.sensor_pts is not None:
            self.pvmxs = {}
            sndr_pts = radmtx.Sender.as_pts(pts_list=self.config.sensor_pts, ray_cnt=1)
            for wname in self.config.window_prims:
                logger.info(f"Generating pvmx for {wname}")
                self.pvmxs[wname+append] = os.path.join(self.config.filestrct['matrices'],
                                                        f'pvmx_{wname}{append}')
                rcvr_wndw = radmtx.Receiver.as_surface(
                    prim_list=self.config.window_prims[wname], basis=self.vmx_basis,
                    offset=None,left=None, source='glow', out=None)
                radmtx.rfluxmtx(sender=sndr_pts, receiver=rcvr_wndw,
                                env=self.config.envpath, opt=opt,
                                out=self.pvmxs[wname+append])
            sndr_pts.remove()
            rcvr_wndw.remove()
        if self.config.vu_dict is not None:
            self.vvmxs = {}
            sndr_v =  radmtx.Sender.as_view(
                vu_dict=self.config.vu_dict, ray_cnt=1, xres=1000, yres=1000, c2c=True)
            for wndw in self.config.window_prims:
                self.vvmxs[wname+append] = os.path.join(
                    self.config.filestrct['matrices'], f'vvmx_{wname}{append}')
                rcvr_wndw = radmtx.Receiver.as_surface(
                    prim_list=self.config.window_prims[wname], basis=self.vmx_basis,
                offset=None, left=None, source='glow', out=None)
                radmtx.rfluxmtx(sender=sndr_v,receiver=rcvr_wndw, env=self.config.envpath,
                           out=self.vvmxs[wname+append], opt=opt)
            sndr_v.remove()
            rcvr_wndw.remove()

    def gen_dsmx(self):
        env = self.config.envpath + self.config.windowpath
        sndr_pts = radmtx.Sender.as_pts(pts_list=self.config.sensor_pts, ray_cnt=1)
        rcvr_sky = radmtx.Receiver.as_sky(self.dsmx_basis)
        self.pdsmx = os.path.join(self.config.filestrct['matrices'], 'pdsmx')
        radmtx.rfluxmtx(sender=sndr_pts, receiver=rcvr_sky,
                        env=env, out=self.pdsmx, opt=self.dsmx_opt)


    def gen_fmx(self, direct=False):
        pass

    def compute_sensor(self, direct=False):
        append = ''
        if direct:
            append = '_d'
        presl = []
        for wname in self.config.windows_prims:
            cmd = f'rmtxop {self.pvmxs[wname+append]} {self.bsdf[wname+append]} '
            cmd += f'{self.dmxs[wname+append]} {self.smxpath} '
            cmd += f'| rmtxop -fa -c 47.4 119.9 11.6 - | getinfo -'
            _res = sp.run(cmd, check=True, stdout=sp.PIPE).stdout.decode()
            presl.append(map(float, _res.splitlines()))
        return [sum(l) for l in zip(*presl)]

    def rmtxop(self, *mtx):
        cmd = f"rmtxop {' '.join(*mtx)} "
        cmd += f'| rmtxop -fa -c 47.4 119.9 11.6 - | getinfo -'
        return sp.run(cmd, check=True, stdout=sp.PIPE).stdout.decode()

    def compute_vu(self):
        append = ''
        if direct:
            append = '_d'
        for wname in self.config['windows']:
            opath = os.path.join(self.resdir, wname)
            cmd = f'dctimstep {self.vvmxs[wname+append]} {self.bsdf[wname+append]} '
            cmd += f'{self.dmxs[wname+append]} {self.smx} '
            cmd += f'> {opath}'


    #def mtxmult(self, mtxs):
    #    """Matrix multiplication with Numpy."""
    #    resr = np.linalg.multi_dot([mat[0] for mat in mtxs]) * .265
    #    resg = np.linalg.multi_dot([mat[1] for mat in mtxs]) * .67
    #    resb = np.linalg.multi_dot([mat[2] for mat in mtxs]) * .065
    #    return resr + resg + resb


class Prepare(object):
    """."""
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.parse_config()
        self.get_paths()
        if self.dimensions != {} and None not in self.dimensions.values():
            self.make_room()
        else:
            self.assemble()
        self.get_wea()

    def parse_config(self):
        """Parse a configuration file into a dictionary."""
        _config = ConfigParser(allow_no_value=True)
        _config.read(self.cfg_path)
        cfg = _config._sections
        self.site = cfg['Site']
        self.filestrct = cfg['FileStructure']
        self.simctrl = cfg['SimulationControl']
        self.model = cfg['Model']
        self.dimensions = cfg['Dimensions']
        self.raysenders = cfg['Raysenders']


    def get_paths(self):
        """Where are ?"""
        objdir = self.filestrct['objects']
        raydir = self.filestrct['raysenders']
        mtxdir = self.filestrct['matrices']
        self.materialpath = os.path.join(objdir, self.model['material'])
        self.envpath = [self.materialpath] + [os.path.join(objdir, obj)
                        for obj in self.model['scene'].split()]
        self.windowpath = [os.path.join(objdir, obj)
                           for obj in self.model['windows'].split()]
        if self.raysenders['view'] is not None:
            self.viewpath = os.path.join(raydir, self.raysenders['view'])
        if self.model['bsdf'] is not None:
            self.bsdfpath = [os.path.join(mtxdir, bsdf)
                             for bsdf in self.model['BSDF'].split()]
        else:
            self.bsdfpath = None

    def assemble(self):
        self.window_prims = {}
        for path in self.windowpath:
            wname = radutil.basename(path)
            with open(path) as rdr:
                self.window_prims[wname] = radutil.parse_primitive(rdr.readlines())
        if self.raysenders['surface'] is None:
            self.sensor_pts = None
        else:
            surface_path = os.path.join(self.filestrct['objects'],self.raysenders['surface'])
            with open(surface_path) as rdr:
                prim = radutil.parse_primitive(rdr.readlines())
            self.sensor_pts = radutil.gen_grid(
                prim[0]['polygon'], float(self.raysenders['distance']),
                float(self.raysenders['spacing']), op=self.raysenders['op'])
        try:
            with open(self.viewpath) as rdr:
                self.vu_dict = radutil.parse_vu(rdr.readlines()[0])
        except AttributeError:
            self.vu_dict = None



    def make_room(self):
        """Make a side-lit shoebox room."""
        theroom = room.Room(float(self.dimensions['width']),
                            float(self.dimensions['depth']),
                            float(self.dimensions['height']))
        wndw_names = [i for i in self.dimensions if i.startswith('window')]
        for wd in wndw_names:
            wdim = map(float, self.dimensions[wd].split())
            theroom.swall.add_window(wd, theroom.swall.make_window(*wdim))
        theroom.swall.facadize(float(self.dimensions['facade_thickness']))
        theroom.surface_prim()
        theroom.window_prim()
        mlib = radutil.material_lib()
        sensor_grid = radutil.gen_grid(theroom.floor, grid_height, grid_spacing)
        nsensor = len(sensor_grid)
        return theroom, sensor_grid

    def get_wea(self):
        if self.site['wea'] is not None:
            self.wea_path = self.site['wea']
            with open(self.site['wea']) as rdr:
                raw = rdr.read()
            sec = raw.split('{os.linesep*2}')
            header = sec[0]
            lines = [l.split() for l in sec[1].splitines()]
            self.dts = [f"{int(l[0]):02d}{int(l[1]):02d}_{int(l[2])}30" for l in lines]
        else:
            if self.site['zipcode'] is not None:
                epw = getepw.getEPW.from_zip(self.site['zipcode'])
                self.site['lat'], self.site['lon'] = epw.lat, epw.lon
            elif None not in (self.site['lat'], self.site['lon']):
                epw = getepw.getEPW(self.site['lat'], self.site['lon'])
            else:
                raise NameError("Not site info defined")
            epw_path = os.path.join(self.filestrct['resources'], epw.fname)
            os.rename(epw.fname, epw_path)
            wea = epw2wea.epw2wea(epw=epw_path, dh=False, sh=None, eh=None)
            self.wea_path = os.path.join(self.filestrct['resources'], radutil.basename(epw.fname) + '.wea')
            with open(self.wea_path, 'w') as wtr:
                wtr.write(wea.wea)
            self.dts = wea.dt_string



def main(cfgpath):
    setup = Prepare(cfgpath)
    mrad = Mtxmethod(setup)
    if setup.model['bsdf'] is None:
        mrad.two_phase()
    else:
        if setup.simctrl['separate_direct']:
            mrad.five_phase()
        else:
            mrad.three_phase()


if __name__ == '__main__':
    import pdb
    import logging
    log_level = 'info'  # set logging level
    log_level_dict = {'info': 20, 'warning': 30, 'error': 40, 'critical': 50}
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level_dict[log_level])
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('-vb', action='store_true', help='verbose mode')
    parser.add_argument('-debug', action='store_true', help='debug mode')
    parser.add_argument('-silent', action='store_true', help='silent mode')
    args = parser.parse_args()
    argmap = vars(args)
    logger = logging.getLogger('frads.radmtx')
    if argmap['debug'] == True:
        logger.setLevel(logging.DEBUG)
    elif argmap['vb'] == True:
        logger.setLevel(logging.INFO)
    elif argmap['silent'] == True:
        logger.setLevel(logging.CRITICAL)
    main(args.cfg)
