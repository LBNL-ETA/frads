
"""
T.Wang

"""
import os
import copy
import tempfile as tf
import subprocess as sp
from frads import mfacade, radgeom, room
from frads import getepw, epw2wea, radutil, radmtx
from configparser import ConfigParser
import logging
import pdb

logger = logging.getLogger('frads.mtxmethod')

class MTXmethod(object):
    def __init__(self, config):
        self.logger = logging.getLogger("frads.mtxmethod.MTXmethod")
        self.config = config
        nproc = self.config.simctrl['nprocess']
        self.vmx_opt = config.simctrl['vmx_opt'][3:] + f' -n {nproc}'
        self.vmx_basis = config.simctrl['vmx_opt'][:2]
        self.dmx_opt = config.simctrl['dmx_opt'][3:] + f' -n {nproc}'
        self.dmx_basis = config.simctrl['dmx_opt'][:2]
        self.dsmx_opt = config.simctrl['dsmx_opt'][3:] + f' -n {nproc}'
        self.dsmx_basis = config.simctrl['dsmx_opt'][:2]
        self.cdsmx_opt = config.simctrl['cdsmx_opt'][3:] + f' -n {nproc}'
        self.cdsmx_basis = config.simctrl['cdsmx_opt'][:2]
        self.ray_cnt = int(config.simctrl['ray_cnt'])
        self.mtxdir = config.filestrct['matrices']
        self.resdir = config.filestrct['results']
        self.materialpath = config.materialpath
        self.windowpath = config.windowpath
        self.td = tf.mkdtemp()
        if self.config.sensor_pts is not None:
            self.sndr_pts = radmtx.Sender.as_pts(pts_list=self.config.sensor_pts,
                                            ray_cnt=self.ray_cnt, tmpdir=self.td)
        if self.config.vu_dict is not None:
            self.sndr_vu = radmtx.Sender.as_view(vu_dict=self.config.vu_dict,
                                            ray_cnt=self.ray_cnt, tmpdir=self.td,
                                           xres=self.config.xres, yres=self.config.yres)
        self.gen_smx(self.dmx_basis)
        self.rcvr_sky = radmtx.Receiver.as_sky(basis=self.dmx_basis, tmpdir=self.td)

    def prep_2phase(self):
        env = self.config.envpath + self.config.windowpath
        if self.config.sensor_pts is not None:
            self.pdsmx = os.path.join(self.config.filestrct['matrices'], 'pdsmx.mtx')
            radmtx.rfluxmtx(sender=sndr_pts, receiver=self.rcvr_sky,
                            env=env, out=self.pdsmx, opt=self.dsmx_opt)
        if self.config.vu_dict is not None:
            self.vdsmx = os.path.join(self.config.filestrct['matrices'], 'vdsmx')
            radmtx.rfluxmtx(sender=self.sndr_vu, receiver=self.rcvr_sky,
                            env=env, opt=self.dsmx_opt, out=self.vdsmx)

    def calc_2phase(self):
        if self.config.sensor_pts is not None:
            res = self.rmtxop(self.pdsmx, self.smxpath).splitlines()
            respath = os.path.join(self.config.filestrct['results'], 'pdsmx.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.config.dts[idx] + '\t')
                    wtr.write(res[idx] + os.linesep)
        if self.config.vu_dict is not None:
            opath = os.path.join(self.resdir, 'view2ph')
            self.dcts(self.vdsmx, self.smxpath, opath)
            ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath)) if
                      f.endswith('.hdr')]
            [os.rename(ofiles[idx], os.path.join(opath, self.config.dts[idx]+'.hdr'))
             for idx in range(len(ofiles))]

    def prep_3phase(self):
        self.td = tf.mkdtemp()
        self.pvmxs = {}
        self.dmxs = {}
        prcvr_wndws = radmtx.Receiver(
            path=None, receiver='', basis=self.vmx_basis, modifier=None)
        vrcvr_wndws = radmtx.Receiver(
            path=None, receiver='', basis=self.vmx_basis, modifier=None)
        for wname in self.config.window_prims:
            wndw_prim = self.config.window_prims[wname]
            self.logger.info(f"Generating dmx for {wname}")
            self.dmxs[wname] = os.path.join(self.mtxdir, f'dmx_{wname}.mtx')
            sndr_wndw = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                       env=self.config.envpath, out=self.dmxs[wname], opt=self.dmx_opt)
            if self.config.sensor_pts is not None:
                self.pvmxs[wname] = os.path.join(self.mtxdir, f'pvmx_{wname}.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname])
            if self.config.vu_dict is not None:
                self.vvmxs[wname] = os.path.join(self.mtxdir, f'vvmx_{wname}')
                vrcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.vvmxs[wname])
        if self.config.sensor_pts is not None:
            sndr_pts = radmtx.Sender.as_pts(
                pts_list=self.config.sensor_pts, ray_cnt=self.ray_cnt, tmpdir=self.td)
            self.logger.info("Generating point vmx")
            radmtx.rfluxmtx(sender=sndr_pts, receiver=prcvr_wndws,
                            env=self.config.envpath, opt=self.vmx_opt, out=None)
        if self.config.vu_dict is not None:
            self.logger.info("Generating view vmx")
            radmtx.rfluxmtx(sender=self.sndr_vu, receiver=vrcvr_wndws,
                            env=self.config.envpath, opt=self.vmx_opt, out=None)

    def calc_3phase(self):
        if self.config.sensor_pts is not None:
            self.logger.info("Computing for point grid results")
            presl = []
            for wname in self.config.window_prims:
                _res = self.rmtxop(self.pvmxs[wname], self.config.bsdf[wname],
                                   self.dmxs[wname], self.smxpath)
                presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
            res = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
            respath = os.path.join(self.resdir, 'points3ph.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.config.dts[idx] + ',')
                    wtr.write(','.join(map(str, res[idx])) + os.linesep)
        if self.config.vu_dict is not None:
            self.logger.info("Computing for rendering results")
            vresl = []
            for wname in self.config.window_prims:
                _vrespath = tf.mkdtemp()
                self.dcts(self.vvmxs[wname], self.config.bsdf[wname], self.dmxs[wname],
                          self.smxpath, _vrespath)
                vresl.append(_vrespath)
            [vresl.insert(i*2-1, '+') for i in range(1, len(vresl)+1)]
            opath = os.path.join(self.resdir, 'view3ph')
            radutil.pcombop(vresl, opath)
            ofiles = [os.path.join(opath, f) for f in sorted(os.listdir(opath)) if
                      f.endswith('.hdr')]
            [os.rename(ofiles[idx], os.path.join(opath, self.config.dts[idx]+'.hdr'))
             for idx in range(len(ofiles))]

    def blacken_env(self):
        blackened = copy.deepcopy(self.config.scene_prims)
        for prim in blackened:
            prim['modifier'] = 'black'
        _, self.blackenvpath = tf.mkstemp(prefix='blackenv')
        with open(self.blackenvpath, 'w') as wtr:
            for prim in blackened:
                wtr.write(radutil.put_primitive(prim))

    def prep_5phase(self):
        self.prep_3phase()
        self.blacken_env()
        self.gen_smx(self.dmx_basis, direct=True)
        self.gen_smx(self.cdsmx_basis, onesun=True, direct=True)
        rcvr_sun = radmtx.Receiver.as_sun(basis='r6', tmpdir=self.td,
                                          smx_path=self.smxpath,
                                          window_paths=self.config.windowpath)
        prcvr_wndws = radmtx.Receiver(path=None, receiver='', basis=self.vmx_basis,
                                      modifier=None)
        vrcvr_wndws = radmtx.Receiver(path=None, receiver='', basis=self.vmx_basis,
                                      modifier=None)
        for wname in self.config.window_prims:
            wndw_prim = self.config.window_prims[wname]
            sndr_wndw = radmtx.Sender.as_surface(tmpdir=self.td,
                prim_list=wndw_prim, basis=self.vmx_basis, offset=None)
            self.dmxs[wname+'_d'] = os.path.join(self.mtxdir, f'dmx_{wname}_d.mtx')
            self.logger.info(f"Generating direct daylight matrix for {wname}")
            radmtx.rfluxmtx(sender=sndr_wndw, receiver=self.rcvr_sky,
                            env=[self.config.materialpath, self.blackenvpath],
                            out=self.dmxs[wname+'_d'], opt=self.dmx_opt+' -ab 0')
            cdsenv = [self.materialpath, self.blackenvpath] + self.windowpath
            if self.config.sensor_pts is not None:
                self.pvmxs[wname+'_d'] = os.path.join(self.mtxdir, f'pvmx_{wname}_d.mtx')
                prcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.pvmxs[wname+'_d'])
            if self.config.vu_dict is not None:
                self.vvmxs[wname+'_d'] = os.path.join(self.mtxdir, f'vvmx_{wname}_d')
                vrcvr_wndws += radmtx.Receiver.as_surface(
                    prim_list=wndw_prim, basis=self.vmx_basis, tmpdir=self.td,
                    offset=None, left=None, source='glow', out=self.vvmxs[wname+'_d'])
        if self.config.sensor_pts is not None:
            self.logger.info(f"Generating direct view matrix for {wname}")
            radmtx.rfluxmtx(sender=self.sndr_pts, receiver=prcvr_wndws,
                            env=[self.config.materialpath, self.blackenvpath],
                            out=None, opt=self.vmx_opt+' -ab 1')
            self.logger.info(f"Generating direct sun matrix for {wname}")
            self.pcdsmx = os.path.join(self.mtxdir, 'pcdsmx.mtx')
            radmtx.rcontrib(sender=self.sndr_pts, receiver=rcvr_sun,
                            env=cdsenv, out=self.pcdsmx, opt=self.cdsmx_opt)
        if self.config.vu_dict is not None:
            radmtx.rfluxmtx(sender=self.sndr_vu, receiver=vrcvr_wndws,
                            env=[self.config.materialpath, self.blackenvpath],
                            out=None, opt=self.vmx_opt+' -ab 1')
            self.vcdsmx = os.path.join(self.mtxdir, 'vcdsmx')
            radmtx.rcontrib(sender=self.sndr_pts, receiver=rcvr_sun,
                            env=cdsenv, out=self.vcdsmx, opt=self.cdsmx_opt)

    def calc_5phase(self):
        if self.config.sensor_pts is not None:
            presl = []
            pdresl = []
            rescd = [list(map(float, l.strip().split('\t')))
                     for l in self.rmtxop(self.pcdsmx, self.smxd6path).splitlines()]
            for wname in self.config.window_prims:
                _res = self.rmtxop(self.pvmxs[wname], self.config.bsdf[wname],
                                   self.dmxs[wname], self.smxpath)
                _resd = self.rmtxop(self.pvmxs[wname+'_d'], self.config.bsdf[wname],
                                   self.dmxs[wname+'_d'], self.smxdpath)
                presl.append([map(float, l.strip().split('\t')) for l in _res.splitlines()])
                pdresl.append([map(float, l.strip().split('\t')) for l in _resd.splitlines()])
            res3 = [[sum(tup) for tup in zip(*line)]for line in zip(*presl)]
            res3d = [[sum(tup) for tup in zip(*line)]for line in zip(*pdresl)]
            res = [[x-y+z for x,y,z in zip(a,b,c)] for a,b,c in zip(res3, res3d, rescd)]
            respath = os.path.join(self.resdir, 'points5ph.txt')
            with open(respath, 'w') as wtr:
                for idx in range(len(res)):
                    wtr.write(self.config.dts[idx] + ',')
                    wtr.write(','.join(map(str, res[idx])) + os.linesep)
        if self.config.vu_dict is not None:
            vresl = []
            vdresl = []

    def gen_smx(self, mf, onesun=False, direct=False):
        sun_only = ' -d' if direct else ''
        _five = ' -5 .533' if onesun else ''
        oname = radutil.basename(self.config.wea_path)
        self.smxpath = os.path.join(self.mtxdir, oname+'.smx')
        self.smxdpath = os.path.join(self.mtxdir, oname+'_d.smx')
        self.smxd6path = os.path.join(self.mtxdir, oname+'_d6.smx')
        cmd = f"gendaymtx -ofd -m {mf[-1]}{sun_only}{_five} {self.config.wea_path}"
        if direct:
            if onesun:
                cmd += f" > {self.smxd6path}"
            else:
                cmd += f" > {self.smxdpath}"
        else:
            cmd += f" > {self.smxpath}"
        sp.run(cmd, shell=True)

    def rmtxop(self, *mtx):
        cmd = f"rmtxop {' '.join(mtx)} "
        cmd += f'| rmtxop -fa -c 47.4 119.9 11.6 - | rmtxop -fa -t - | getinfo -'
        self.logger.info(cmd)
        return sp.run(cmd, shell=True, check=True, stdout=sp.PIPE).stdout.decode()

    def dcts(self, *mtx, odir):
        cmd = f"dctimstep -o {os.path.join(odir, '%04d.hdr')} {' '.join(mtx)}"

    def get_avgskv(self):
        sp.run(f"gendaymtx -m {self.mf_sky} -A {self.wea} > {avgskyv}", shell=True)

############### Matrix multiplication using numpy ########################################


    #def mtxmult(self, mtxs):
    #    """Matrix multiplication with Numpy."""
    #    resr = np.linalg.multi_dot([mat[0] for mat in mtxs]) * .265
    #    resg = np.linalg.multi_dot([mat[1] for mat in mtxs]) * .67
    #    resb = np.linalg.multi_dot([mat[2] for mat in mtxs]) * .065
    #    return resr + resg + resb

##########################################################################################

class Prepare(object):
    """."""
    def __init__(self, cfg_path):
        self.logger = logging.getLogger('frads.mtxmethod.Prepare')
        self.cfg_path = cfg_path
        self.parse_config()
        self.get_paths()
        if self.dimensions != {} and None not in self.dimensions.values():
            self.logger.info("Generating a room based on defined dimensions")
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
        self.scenepath = [os.path.join(objdir, obj)
                          for obj in self.model['scene'].split()]
        self.envpath = [self.materialpath] + [os.path.join(objdir, obj)
                        for obj in self.model['scene'].split()]
        self.windowpath = [os.path.join(objdir, obj)
                           for obj in self.model['windows'].split()]
        if self.raysenders['view'] is not None:
            viewline = self.raysenders['view'].split()
            self.viewpath = os.path.join(raydir, viewline[0])
            self.xres = viewline[1]
            self.yres = viewline[2]
        if self.model['bsdf'] is not None:
            self.bsdfpath = [os.path.join(mtxdir, bsdf)
                             for bsdf in self.model['bsdf'].split()]
        else:
            self.bsdfpath = None

    def assemble(self):
        with open(self.materialpath) as rdr:
            self.mat_prims = radutil.parse_primitive(rdr.readlines())
        self.mat_prims.append({'modifier':'void', 'type':'plastic',
                          'identifier':'black', 'str_args':'0',
                          'int_arg':'0', 'real_args':'5 0 0 0 0 0'})
        self.scene_prims = []
        for spath in self.scenepath:
            with open(spath) as rdr:
                self.scene_prims.extend(radutil.parse_primitive(rdr.readlines()))
        self.window_prims = {}
        self.bsdf = {}
        for idx in range(len(self.windowpath)):
            wname = radutil.basename(self.windowpath[idx])
            with open(self.windowpath[idx]) as rdr:
                self.window_prims[wname] = radutil.parse_primitive(rdr.readlines())
            self.bsdf[wname] = self.bsdfpath[idx]
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
            self.logger.info(f"Downloaded {epw.fname}")
            epw_path = os.path.join(self.filestrct['resources'], epw.fname)
            os.rename(epw.fname, epw_path)
            wea = epw2wea.epw2wea(epw=epw_path, dh=True, sh=None, eh=None)
            self.wea_path = os.path.join(self.filestrct['resources'], radutil.basename(epw.fname) + '.wea')
            with open(self.wea_path, 'w') as wtr:
                wtr.write(wea.wea)
            self.dts = wea.dt_string
